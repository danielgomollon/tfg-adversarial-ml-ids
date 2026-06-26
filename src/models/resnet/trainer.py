"""
trainer.py 
Entrenamiento de TabularResNet para BigFlow-NIDS

==================================================================
Incluye:
  * CrossEntropyLoss + Class Weights + Label Smoothing=0.1
      - más estable que Focal Loss
      - class weights compensan el desbalance entre macro-clases
      - label smoothing suaviza etiquetas duras y mejora generalización
  * AdamW + CosineAnnealingWarmRestarts
      - AdamW corrige el weight decay de Adam original
      - Cosine reinicia el LR periódicamente para escapar mínimos locales
  * Early Stopping sobre val F1-macro (no sobre loss)
      - F1-macro captura mejor el rendimiento en clases desbalanceadas
      - la loss puede bajar mientras F1 de clases raras empeora
  * Gradient Clipping (max_norm=1.0)
      - evita explosiones de gradiente que ocurren con Spectral Norm
  * Corrección de prior bayesiana en inferencia
      - ajusta probabilidades de train (60/40) a producción (95/5)
      - sin reentrenar el modelo
  * Métricas completas: Acc, F1-macro, F1 por clase, AUC-ROC

  SISTEMA DE CHECKPOINTS (anti-crash Colab):
  ──────────────────────────────────────────
  - best_resnet.pt      : mejor modelo según val F1-macro (siempre)
  - checkpoint_last.pt  : estado completo al final de cada epoch
                          incluye epoch, optimizer, scheduler, history
                          permite reanudar exactamente donde se cortó

  Para reanudar tras un crash de Colab:
      trainer.run(resume=True)
  Buscará checkpoint_last.pt y continuará desde ese epoch.
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.distributions.beta import Beta   # para mixup tabular
from tqdm.notebook import tqdm as tqdm_notebook 
from sklearn.metrics import (f1_score, accuracy_score, 
                             roc_auc_score, classification_report)

from src.config                 import Config
from src.models.resnet.advanced_model  import TabularResNet
from src.helpers                  import create_batch_progress_bar, load_checkpoint
from src.helpers                  import save_checkpoint, create_progress_bar

# ------------------------------------------------------------------
# CLASS WEIGHTS
# ------------------------------------------------------------------
def compute_class_weights(y_train, use_weights=True, manual_weights=None):
    """
    Pesos inversamente proporcionales a la frecuencia de cada clase.
    Normalizados para que la media sea 1.0 -> la loss no cambia de escala.

    Ejemplo: si DDoS tiene 10x más muestras que ransomware,
    ransomware recibe weight=10 y DDoS weight=1.
    """
    if manual_weights is not None:
        weights = np.array(manual_weights, dtype=float)
        return torch.tensor(weights, dtype=torch.float32)    

    counts  = np.bincount(y_train, minlength=Config.NUM_CLASSES).astype(float)
    weights = 1.0 / (counts + 1e-8)
    weights = weights / weights.mean()
    # TEMPORAL: luego habrá que solucionarlo mejor
    # hemos visto que Exploits ha conseguido un F1 alto (82%) sin class weights siendo la macro-clase más rara, 
    # vamos entonces a suavizar los pesos para no penalizar tanto y solo corregir lo que falla
    #weights = np.clip(weights, 0.1, 5.0)
    weights = np.clip(weights, 0.5, 3.0)  

    return torch.tensor(weights, dtype=torch.float32)

# ------------------------------------------------------------------
# FOCAL LOSS
# ------------------------------------------------------------------
class FocalLoss(nn.Module):
    """
    Focal Loss — alternativa a CrossEntropy para clases muy desbalanceadas.
    Creada originalmente por Facebook AI para detección de objetos pequeños.

    La clave está en el factor (1 - p_t)^gamma:
      - Si el modelo clasifica un flujo Benigno con p=0.99 (muy seguro),
        ese ejemplo recibe un peso (1-0.99)^2 = 0.0001 → casi ignorado.
      - Si el modelo duda en un flujo Recon con p=0.55,
        ese ejemplo recibe un peso (1-0.55)^2 = 0.20 → gradiente completo.

    Resultado: el 100% del esfuerzo del gradiente se concentra en la
    frontera borrosa Recon/Benign, ignorando los Benign triviales.

    Parámetros:
        weight          : class weights (igual que CrossEntropy, siguen activos)
        gamma           : factor de focalización. 0.0 = CrossEntropy estándar.
                          2.0 es el valor canónico de la literatura (Lin et al. 2017)
        label_smoothing : igual que en CrossEntropy, suaviza etiquetas duras
    """
    def __init__(self, weight=None, gamma=3.0, label_smoothing=0.1):
        super().__init__()
        self.gamma           = gamma
        self.weight          = weight
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        # CrossEntropy base con label smoothing y class weights
        # reduction='none' para obtener la loss por muestra antes de aplicar gamma
        ce = F.cross_entropy(
            logits, targets,
            weight          = self.weight,
            label_smoothing = self.label_smoothing,
            reduction       = 'none',
        )
        # p_t = probabilidad asignada a la clase correcta
        # exp(-ce) es equivalente a p_t cuando no hay label smoothing
        # con label smoothing es una aproximación válida y estándar
        pt = torch.exp(-ce)

        # factor focal: penaliza menos los ejemplos fáciles (pt alto)
        focal_factor = (1.0 - pt) ** self.gamma

        return (focal_factor * ce).mean()

# ------------------------------------------------------------------
# EARLY STOPPING
# ------------------------------------------------------------------
class EarlyStopping:
    """
    Detiene el entrenamiento si val F1-macro no mejora en `patience` epochs.
    Guarda automáticamente el mejor checkpoint en disco.

    Se queda en trainer porque es lógica íntimamente ligada
    al bucle de entrenamiento, no una utilidad general reutilizable.
    """
    def __init__(self, patience, path):
        self.patience  = patience
        self.path      = path
        self.best      = -np.inf
        self.counter   = 0
        self.triggered = False

    def step(self, score, model, optimizer, scheduler, epoch):
        if score > self.best:
            self.best    = score
            self.counter = 0
            torch.save({
                'epoch'     : epoch,
                'model'     : model.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'scheduler' : scheduler.state_dict(),
                'val_f1'    : score,
            }, self.path)
            print(f"   [✓] Mejor modelo guardado — epoch {epoch} "
                  f"(val F1={score:.4f})", flush=True)
        else:
            self.counter += 1
            print(f"   [EarlyStopping] sin mejora {self.counter}/"
                  f"{self.patience} (mejor: {self.best:.4f})", flush=True)
            if self.counter >= self.patience:
                self.triggered = True


# ------------------------------------------------------------------
# EVALUACIÓN
# ------------------------------------------------------------------
def evaluate(model, loader, device, pi_train=None, pi_prod=0.05):
    """
    Evaluación completa del modelo con corrección bayesana a priori.

    Parámetros:
        pi_train : prior de ataque en train (guardado por el pipeline)
                   Si se proporciona, aplica corrección bayesiana de prior
        pi_prod  : prior estimado en producción (0.05 = distribución 95/5)

    Corrección bayesiana:
        El modelo aprende P(clase | x) con prior de train (60/40).
        En producción el prior real es 95/5.
        Corregimos las probabilidades de salida sin reentrenar:
            P_corr(ataque|x) ∝ P_train(ataque|x) * (π_prod / π_train)
        Esto reduce los falsos positivos en producción real.
    """
    model.eval()
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            logits, _ = model(X_batch.to(device))
            all_logits.append(logits.cpu())
            all_labels.append(y_batch)

    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels).numpy()
    probs  = F.softmax(logits, dim=1).numpy()

    # corrección de prior bayesiana (solo para evaluación en distribución real)
    if pi_train is not None:
        p_benign = probs[:, 0:1]
        p_attack = probs[:, 1:]

        factor_benign = (1.0 - pi_prod)  / (1.0 - pi_train + 1e-8)
        factor_attack = pi_prod           / (pi_train + 1e-8)

        probs_corr = np.concatenate(
            [p_benign * factor_benign, p_attack * factor_attack], axis=1
        )
        probs_corr /= probs_corr.sum(axis=1, keepdims=True) + 1e-8
        preds = np.argmax(probs_corr, axis=1)
    else:
        preds = np.argmax(probs, axis=1)

    acc      = accuracy_score(labels, preds)
    f1_macro = f1_score(labels, preds, average='macro',  zero_division=0)
    f1_per   = f1_score(labels, preds, average=None,     zero_division=0)

    # AUC-ROC multiclase One-vs-Rest
    try:
        auc = roc_auc_score(labels, probs, multi_class='ovr', average='macro')
    except ValueError:
        auc = 0.0   # ocurre si alguna clase no aparece en el batch

    return acc, f1_macro, f1_per, auc, preds, labels, probs


# ------------------------------------------------------------------
# TRAINER
# ------------------------------------------------------------------
class Trainer:
    def __init__(self):
        self.device    = Config.DEVICE
        self.processed = Config.DATA_PROCESSED_PATH
        self.models    = Config.MODELS_PATH
        self.logs      = Config.LOGS_PATH
        # Recuperamos MIXUP_ALPHA de Config (por defecto 0.2 si no existe)
        self.mixup_alpha = getattr(Config, 'MIXUP_ALPHA', Config.MIXUP_ALPHA)
        os.makedirs(self.models, exist_ok=True) 
        os.makedirs(self.logs,   exist_ok=True)

    def _load_data(self, suffix=""):
        """Carga los .npy generados por data_pipeline.py."""
        p = self.processed
        X_train = np.load(os.path.join(p, f"X_train{suffix}.npy"))
        y_train = np.load(os.path.join(p, f"y_train{suffix}.npy"))
        X_val   = np.load(os.path.join(p, f"X_val{suffix}.npy"))
        y_val   = np.load(os.path.join(p, f"y_val{suffix}.npy"))
        X_test  = np.load(os.path.join(p, f"X_test{suffix}.npy"))
        y_test  = np.load(os.path.join(p, f"y_test{suffix}.npy"))
        return X_train, y_train, X_val, y_val, X_test, y_test

    def _make_loader(self, X, y, shuffle=False):
        dataset = TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.long),
        )
        return DataLoader(
            dataset,
            batch_size  = Config.BATCH_SIZE,
            shuffle     = shuffle,
            num_workers = 0,        # 0 para compatibilidad Colab/Windows
            pin_memory  = self.device.type == 'cuda',
        )

    def run(self, suffix="", resume=False,
        epochs=None, lr=None, patience=None,
        dropout=None, inner_dropout=None,
        hidden_dim=None, n_blocks=None,
        mixup_alpha=None, use_class_weights=True):
        """
        Parámetros:
            suffix : sufijo de los .npy ("" para run completo, "_dry" para dry)
            resume : si True, busca checkpoint_last.pt y continúa desde ahí
                     útil cuando Colab corta la sesión a mitad del entrenamiento
        """
        
        """ IMPORTANTE CAMBIAR CUANDO EXPERIEMTNACIÓN ACABE
        Parametros opcionales (sobrescriben Config si se proporcionan desde Colab) 
        para mejor experimentación. Luego cambiarlo cuando tengamos el modelo
        """

        # Resolución de hiperparámetros: parámetro explícito > Config
        epochs        = epochs        if epochs        is not None else Config.EPOCHS
        lr            = lr            if lr            is not None else Config.LEARNING_RATE
        patience      = patience      if patience      is not None else Config.PATIENCE
        dropout       = dropout       if dropout       is not None else Config.DROPOUT
        inner_dropout = inner_dropout if inner_dropout is not None else Config.INNER_DROPOUT
        hidden_dim    = hidden_dim    if hidden_dim    is not None else Config.HIDDEN_DIM
        n_blocks      = n_blocks      if n_blocks      is not None else Config.N_BLOCKS
        mixup_alpha   = mixup_alpha   if mixup_alpha   is not None else Config.MIXUP_ALPHA

        # acordarme de quitarlo cuando finalicemos la experimentación para operar desde Config
        self.mixup_alpha = mixup_alpha

        print("\n" + "="*60)
        print(f"ENTRENAMIENTO — TabularResNet  [Fase {Config.EXPERIMENT_PHASE}]")
        print(f"Device : {self.device}")
        if resume:
            print("Modo   : RESUME (continuando desde último checkpoint)")
        print("="*60)

        # ---------------------------------------------------------
        # Calcular el siguiente índice para el archivo de log
        # ---------------------------------------------------------
        log_index = 1
        while os.path.exists(os.path.join(self.logs, f"training_log_{log_index}.txt")):
            log_index += 1
            
        self.current_log_file = f"training_log_{log_index}.txt"
        print(f"\n[-] Historial de épocas se guardará en: {self.current_log_file}")

        print("\n[-] 1. Cargando datos...")
        X_train, y_train, X_val, y_val, X_test, y_test = self._load_data(suffix)
        print(f"   Train : {X_train.shape}")
        print(f"   Val   : {X_val.shape}")
        print(f"   Test  : {X_test.shape}")

        pi_train = float(
            np.load(os.path.join(self.models, f"pi_train{suffix}.npy"))[0]
        )
        print(f"   π_train = {pi_train:.4f}")

        train_loader = self._make_loader(X_train, y_train, shuffle=True)
        val_loader   = self._make_loader(X_val,   y_val)
        test_loader  = self._make_loader(X_test,  y_test)

        print("\n[-] 2. Construyendo modelo...")
        model = TabularResNet(
            input_dim = X_train.shape[1],
            num_classes = Config.NUM_CLASSES,
            hidden_dim = hidden_dim,
            embed_dim = Config.EMBED_DIM,
            n_blocks = n_blocks,
            dropout = dropout,
            inner_dropout = inner_dropout,
        ).to(self.device)

        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"   Parámetros entrenables: {n_params:,}")

        # Loss: CrossEntropy + Class Weights + Label Smoothing
        # ─────────────────────────────────────────────────────────
        # CrossEntropy con weights: penaliza más los errores en clases raras
        # label_smoothing=0.1: suaviza etiquetas duras (0→0.1, 1→0.9)
        #   - mejora calibración y generalización
        #   - reduce sobreconfianza del modelo
        # Más estable que Focal Loss en clasificación multiclase con 8 clases
        # ─────────────────────────────────────────────────────────
        print("\n[-] 3. Calculando class weights...")
        weights = compute_class_weights(
            y_train, 
            manual_weights=Config.MANUAL_WEIGHTS if use_class_weights else None
        ).to(self.device)
        
        for i, (name, w) in enumerate(zip(Config.CLASS_NAMES, weights.cpu())):
            print(f"   Clase {i} ({name:<15}): weight = {w:.4f}")

        criterion = nn.CrossEntropyLoss(
            weight          = weights if use_class_weights else None,
            label_smoothing = 0.1,
        )

        #criterion = FocalLoss(
        #    weight          = weights if use_class_weights else None,
        #    gamma           = 2.0,    # subimos para forzar Recon
        #                              # subir a 3.0 si Recon sigue sin mejorar
        #    label_smoothing = 0.1,
        #)

        # Optimizador: AdamW
        # ─────────────────────
        # AdamW corrige el bug de weight decay de Adam original
        # (Adam aplica el decay sobre los gradientes adaptados, AdamW lo aplica
        # directamente sobre los pesos -> regularización más limpia)
        # cambiar luego en Config cuando termine experimentación
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr           = lr,
            weight_decay = Config.WEIGHT_DECAY,
            betas        = (0.9, 0.999),
        )

        # Scheduler: CosineAnnealingWarmRestarts
        # ──────────────────────────────────────────
        # Reinicia el LR a su valor máximo cada T_0 epochs
        # Permite escapar de mínimos locales que StepLR no puede
        # T_mult=2: cada reinicio duplica el período → explora menos con el tiempo
        #scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        #    optimizer, T_0=10, T_mult=2, eta_min=1e-6
        #)


        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=4, min_lr=1e-5
        )

        # paths de checkpoints
        path_best = os.path.join(self.models, "best_resnet.pt")
        path_last = os.path.join(self.models, "checkpoint_last.pt")

        # early Stopping sobre val F1-macro (cambiar luego en config)
        early_stop = EarlyStopping(patience=patience, path=path_best)

        # resume (restaurar estado desde checkpoint_last.pt)
        start_epoch = 1
        history = {
            'train_loss': [], 'val_loss': [],
            'val_acc'   : [], 'val_f1'  : [], 'val_auc': []
        }

        if resume and os.path.exists(path_last):
            print("\n[-] Restaurando checkpoint...")
            start_epoch, history, best_f1 = load_checkpoint(
                path_last, model, optimizer, scheduler, self.device
            )
            early_stop.best = best_f1
            start_epoch += 1   # continuamos desde el epoch siguiente
        elif resume:
            print("\n   [!] No se encontró checkpoint_last.pt — entrenando desde cero")

        # training loop
        print(f"\n[-] 4. Entrenando desde epoch {start_epoch} "
              f"hasta {epochs} (patience={patience})...")
        print("-"*60)

        # progreso visual global del entrenamiento
        pbar = create_progress_bar(epochs, start_epoch, Config.MODEL_NAME)

        for epoch in pbar:
            t0 = time.time()
            model.train()
            total_loss = 0.0
            
            # progreso visual por batch dentro de cada epoch
            batch_pbar = create_batch_progress_bar(train_loader, epoch, epochs)

            for X_batch, y_batch in batch_pbar:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                optimizer.zero_grad()

                # TABULAR MIXUP (Data Augmentation and Regularization)
                if self.mixup_alpha > 0:
                    # 1. Muestreo de lambda desde una distribución Beta
                    lam = Beta(self.mixup_alpha, self.mixup_alpha).sample().item()
                    lam = max(0.0, min(1.0, lam)) # Estabilidad numérica
                    
                    # 2. Barajado aleatorio de índices en el batch
                    index = torch.randperm(X_batch.size(0)).to(self.device)
                    
                    # 3. Interpolación lineal de características (Flujos sintéticos)
                    mixed_X = lam * X_batch + (1.0 - lam) * X_batch[index]
                    
                    # 4. Forward pass con los datos mezclados
                    logits, _ = model(mixed_X)
                    
                    # 5. Cálculo de la pérdida proporcional
                    loss = lam * criterion(logits, y_batch) + (1.0 - lam) * criterion(logits, y_batch[index])
                else:
                    # Flujo estándar si MIXUP_ALPHA = 0
                    logits, _ = model(X_batch)
                    loss = criterion(logits, y_batch)

                loss.backward()

                # Gradient clipping — evita explosiones con Spectral Norm
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()
                
                total_loss += loss.item()
                batch_pbar.set_postfix({'loss': f"{loss.item():.4f}"}, refresh=True)

            # permite varial el LR de CosineAnnealingWarmRestarts, si no, no varía nada durante entreno
            # QUITAR SI USANMOS REDUCELRONPLATEAU
            # scheduler.step()

            batch_pbar.close()

            avg_loss = total_loss / len(train_loader)
            
            # evaluación en validación
            val_acc, val_f1, _, val_auc, _, _, _ = evaluate(
                model, val_loader, self.device
            )

            # cálculo rápido del val loss
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X_b, y_b in val_loader:
                    lgt, _ = model(X_b.to(self.device))
                    val_loss += criterion(lgt, y_b.to(self.device)).item()
            val_loss /= len(val_loader)

            elapsed = time.time() - t0
            lr_now  = optimizer.param_groups[0]['lr']

            # al final del epoch, antes del early_stop.step:
            pbar.set_postfix({
                'T_Loss': f"{avg_loss:.4f}",
                'V_F1'  : f"{val_f1:.4f}",
                'AUC'   : f"{val_auc:.4f}",
                'LR'    : f"{lr_now:.2e}",
            })

            log_line = (
                f"Epoch {epoch:03d}/{epochs} | "
                f"T.Loss: {avg_loss:.4f} | V.Loss: {val_loss:.4f} | "
                f"F1: {val_f1:.4f} | AUC: {val_auc:.4f} | "
                f"LR: {lr_now:.2e} | t={elapsed:.0f}s"
            )
            
            print(log_line, flush=True)

            # guardamos en el txt con el índice auto-incremental
            with open(os.path.join(self.logs, self.current_log_file), "a") as f:
                f.write(log_line + "\n")

            history['train_loss'].append(avg_loss)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['val_f1'].append(val_f1)
            history['val_auc'].append(val_auc)

            # QUITAR SI USAMOS COSINEANNEALINGWARMRESTARTS, ya que no se basa en métrica de validación
            # PONER SI USAMOS REDUCELRONPLATEAU, PERO QUITANDO EL sheduler.step() de arriba
            scheduler.step(val_f1)

            # Checkpoint del mejor modelo (Early Stopping)
            early_stop.step(val_f1, model, optimizer, scheduler, epoch)

            # Checkpoint del último epoch — siempre, para resume tras crash
            save_checkpoint(
                path_last, epoch, model, optimizer, scheduler,
                history, early_stop.best
            )

            if early_stop.triggered:
                print(f"\n[!] Early Stopping activado en epoch {epoch}")
                break
        
        # quitar luego cuando terminemos experimentación
        print(f"\n   lr={lr} | epochs={epochs} | patience={patience}")
        print(f"   dropout={dropout} | hidden={hidden_dim} | blocks={n_blocks}")
        print(f"   mixup_alpha={mixup_alpha}")

        # evaluación final con el mejor checkpoint
        print("\n" + "="*60)
        print("EVALUACIÓN FINAL — Test set")
        print("="*60)

        checkpoint = torch.load(path_best, map_location=self.device, weights_only=False)
        model.load_state_dict(checkpoint['model'])
        print(f"   Cargado: epoch {checkpoint['epoch']} "
              f"(val F1={checkpoint['val_f1']:.4f})")

        # sin corrección — distribución laboratorio
        acc_lab, f1_lab, _, auc_lab, _, _, _ = evaluate(
            model, test_loader, self.device
        )

        # con corrección — distribución producción 95/5
        acc_prod, f1_prod, f1_per_prod, auc_prod, preds, labels, probs = evaluate(
            model, test_loader, self.device,
            pi_train=pi_train, pi_prod=0.05
        )

        print("\n--- Distribución laboratorio (sin corrección prior) ---")
        print(f"  Accuracy : {acc_lab:.4f}")
        print(f"  F1-macro : {f1_lab:.4f}")
        print(f"  AUC-ROC  : {auc_lab:.4f}")

        print("\n--- Distribución producción (corrección prior 95/5) ---")
        print(f"  Accuracy : {acc_prod:.4f}")
        print(f"  F1-macro : {f1_prod:.4f}")
        print(f"  AUC-ROC  : {auc_prod:.4f}")

        print("\n--- F1 por macro-clase (producción) ---")
        for i, (name, f1) in enumerate(zip(Config.CLASS_NAMES, f1_per_prod)):
            bar = "█" * int(f1 * 20)
            print(f"  {i} {name:<15} {f1:.4f} {bar}")

        print("\n--- Classification Report (producción) ---")
        print(classification_report(
            labels, preds,
            target_names  = Config.CLASS_NAMES,
            zero_division = 0,
        ))

        # Guardar artefactos para Fase 2 (ataques adversariales)
        # ─────────────────────────────────────────────────────────────
        # best_resnet.pt  → cargado por art_wrapper.py para FGSM/PGD/DeepFool
        # test_probs.npy  → curvas PR-AUC y comparativas pre/post ataque
        # train_history   → curvas de entrenamiento para la memoria del TFG
        # ─────────────────────────────────────────────────────────────
        np.save(os.path.join(self.models, "train_history.npy"), history)
        np.save(os.path.join(self.models, "test_preds.npy"),    preds)
        np.save(os.path.join(self.models, "test_labels.npy"),   labels)
        np.save(os.path.join(self.models, "test_probs.npy"),    probs)

        print("\n" + "="*60)
        print("ENTRENAMIENTO COMPLETADO")
        print(f"  Checkpoint  : {path_best}")
        print(f"  Mejor epoch : {checkpoint['epoch']}")
        print(f"  Val F1      : {checkpoint['val_f1']:.4f}")
        print(f"  Test F1     : {f1_prod:.4f}  (producción)")
        print(f"  Test AUC    : {auc_prod:.4f}  (producción)")
        print("="*60)

        return history, model


# ------------------------------------------------------------------
# ENTRY POINT
# ------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    from src.helpers import set_seed

    parser = argparse.ArgumentParser(description="Entrenar TabularResNet")
    parser.add_argument("--resume", action="store_true",
                        help="Continuar desde checkpoint_last.pt")
    parser.add_argument("--dry",    action="store_true",
                        help="Usar datos del dry run (_dry.npy)")
    args = parser.parse_args()

    set_seed(Config.SEED)
    suffix = "_dry" if args.dry else ""
    Trainer().run(suffix=suffix, resume=args.resume)