"""
src/models/resnet/refat_trainer.py
================================================================
RE-FAT: Randomized Entropy Fast Adversarial Training
Contribución Original (Fase 3 - Blue Team)

Entrenador optimizado para la TabularResNet.
Asume que los DataLoaders ya contienen una mezcla de tráfico limpio
y troyanos S3M generados offline. 

Hereda utilidades del trainer.py de Fase 1
pero inyecta la lógica de Defensa en Profundidad (Fast-AT + Entropic Loss).

Innovaciones:
1. Fast-AT (O(1)): Genera ruido adversarial en el espacio escalado 
   respetando las Domain Constraints.
2. Loss Combinada (Clean + Adv): Preserva el F1-Macro del tráfico original.
3. Entropic Boundary Loss: Redirige la incertidumbre (Zona Gris) hacia 
   la entropía máxima para activar el VAE en inferencia.

"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.distributions.beta import Beta

from src.config import Config
from src.models.resnet.advanced_model import TabularResNet
from src.utils.domain_constraints import DomainConstraints
from src.models.resnet.trainer import (
    compute_class_weights, EarlyStopping, evaluate
)
from src.helpers import (
    create_batch_progress_bar, load_checkpoint,
    save_checkpoint, create_progress_bar
)


# ===========================================================================
# ENTROPIC BOUNDARY LOSS
# ===========================================================================

class EntropicBoundaryLoss(nn.Module):
    """
    Loss combinada para RE-FAT.

    Filosofía:
    - Ejemplos con confianza ALTA (>upper): Cross-Entropy normal con class weights.
      El modelo debe ser seguro cuando tiene razón.
    - Ejemplos con confianza BAJA (frontera [lower, upper]): KL hacia uniforme.
      El modelo debe admitir incertidumbre en la zona gris.
    - kl_weight=0.3 (era 0.5): más conservador para no aplastar las probabilidades
      de ataque en clases raras.
    - Bounds asimétricos (0.25, 0.45): la zona gris empieza antes porque los
      ejemplos adversariales tienden a tener confianza moderada, no alta.

    Cambio clave respecto a v1:
    lower_bound=0.25, upper_bound=0.45 en lugar de (0.4, 0.6).
    Esto evita que ejemplos legítimos de ataque con confianza 0.5-0.6
    sean empujados hacia uniforme — esos NO son zona gris, son ataques
    que el modelo aún no clasifica bien y necesita aprender.
    """

    def __init__(
        self,
        weight       = None,
        num_classes  : int   = 8,
        lower_bound  : float = 0.25,
        upper_bound  : float = 0.45,
        kl_weight    : float = 0.3,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.kl_weight   = kl_weight
        self.weight      = weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs          = F.softmax(logits, dim=1)
        confidences, _ = torch.max(probs, dim=1)

        is_boundary = (
            (confidences >= self.lower_bound) &
            (confidences <= self.upper_bound)
        )
        is_certain = ~is_boundary

        loss = torch.tensor(0.0, device=logits.device)

        # Ejemplos claros — Cross-Entropy con class weights
        if is_certain.any():
            loss = loss + F.cross_entropy(
                logits[is_certain],
                targets[is_certain],
                weight=self.weight,
            )

        # Ejemplos en frontera — empujar a distribución uniforme
        if is_boundary.any():
            uniform = torch.full_like(
                probs[is_boundary], 1.0 / self.num_classes
            )
            kl_loss = F.kl_div(
                F.log_softmax(logits[is_boundary], dim=1),
                uniform,
                reduction='batchmean',
            )
            loss = loss + self.kl_weight * kl_loss

        return loss


# ===========================================================================
# RE-FAT TRAINER
# ===========================================================================

class REFAT_Trainer:
    """
    Entrenador RE-FAT v2.

    Cambios principales:
    - epsilon=0.05 por defecto (era 0.15)
    - clean_weight=0.6, adv_weight=0.4: el modelo prioriza tráfico limpio
    - Evaluación final guarda probabilidades crudas para prior sweep justo
    - Documentación de la comparativa justa en el output
    """

    def __init__(self):
        self.device    = Config.DEVICE
        self.processed = Config.DATA_PROCESSED_PATH
        self.models    = os.path.join(Config.MODELS_PATH, "fase3")
        self.logs      = os.path.join(Config.LOGS_PATH,   "fase3")
        os.makedirs(self.models, exist_ok=True)
        os.makedirs(self.logs,   exist_ok=True)

        self.dc            = DomainConstraints.from_artifacts()
        self.frozen_mask_t = torch.tensor(
            ~self.dc.forward_mask, device=self.device
        )

    def _load_data_hybrid(self):
        """Carga dataset combinado: Train original + Troyanos S3M."""
        p     = self.processed
        s3m_p = os.path.normpath(os.path.join(p, '..', 'ataques_s3m'))

        X_orig = np.load(os.path.join(p,     "X_train.npy"))
        y_orig = np.load(os.path.join(p,     "y_train.npy"))
        X_s3m  = np.load(os.path.join(s3m_p, "X_train_s3m.npy"))
        y_s3m  = np.load(os.path.join(s3m_p, "y_train_s3m.npy"))

        X_train = np.vstack([X_orig, X_s3m])
        y_train = np.concatenate([y_orig, y_s3m])

        X_val  = np.load(os.path.join(p, "X_val.npy"))
        y_val  = np.load(os.path.join(p, "y_val.npy"))
        X_test = np.load(os.path.join(p, "X_test.npy"))
        y_test = np.load(os.path.join(p, "y_test.npy"))

        print(f"   Train: {len(X_train):,} "
              f"({len(X_orig):,} original + {len(X_s3m):,} S3M troyanos)")
        print(f"   Val  : {len(X_val):,} | Test: {len(X_test):,}")

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
            num_workers = 2,
            pin_memory  = (self.device.type == 'cuda'),
        )

    def _generate_fast_adv(
        self,
        model   : nn.Module,
        X       : torch.Tensor,
        y       : torch.Tensor,
        epsilon : float,
    ) -> torch.Tensor:
        """
        Fast-AT con Random Start (Wong et al. 2020).

        Diferencia clave respecto a FGSM puro:
        el ruido inicial aleatorio evita el catastrophic overfitting
        porque el optimizador no puede memorizar un camino estático.

        epsilon=0.05: perturbación pequeña — suficiente para explorar
        la vecindad de la muestra sin cruzar la frontera de clases.
        """
        # Random Start — ruido uniforme inicial
        noise = torch.empty_like(X).uniform_(-epsilon, epsilon)
        noise[:, self.frozen_mask_t] = 0.0  # respetar física L4/L7

        X_adv = (X + noise).detach().requires_grad_(True)

        model.eval()  # no envenenar BatchNorm
        logits, _ = model(X_adv)
        loss      = F.cross_entropy(logits, y)
        loss.backward()

        grad = X_adv.grad.detach()
        grad[:, self.frozen_mask_t] = 0.0

        # Paso FGSM con alpha=1.25*epsilon
        alpha = epsilon * 1.25
        X_adv = X_adv.detach() + alpha * grad.sign()

        # Proyectar al ball de epsilon
        return torch.clamp(X_adv, X - epsilon, X + epsilon).detach()

    def run(
        self,
        resume              : bool  = False,
        epochs              : int   = 40,
        lr                  : float = 1e-3,
        patience            : int   = 10,
        dropout             : float = 0.1,
        inner_dropout       : float = 0.0,
        hidden_dim          : int   = 256,
        n_blocks            : int   = 4,
        mixup_alpha         : float = 0.2,
        use_class_weights   : bool = False,
        epsilon             : float = 0.05,    
        clean_weight        : float = 0.6,     
        adv_weight          : float = 0.4,     
    ):
        print("\n" + "="*60)
        print("ENTRENAMIENTO BLINDADO — RE-FAT v2 [Fase 3]")
        print(f"  epsilon={epsilon} | clean={clean_weight} | adv={adv_weight}")
        print("="*60)

        # Logs
        log_index = 1
        while os.path.exists(
            os.path.join(self.logs, f"refat_log_{log_index}.txt")
        ):
            log_index += 1
        self.current_log_file = f"refat_log_{log_index}.txt"

        # 1. Datos
        print("\n[-] 1. Cargando datos blindados...")
        X_train, y_train, X_val, y_val, X_test, y_test = \
            self._load_data_hybrid()

        pi_train     = float(
            np.load(os.path.join(Config.MODELS_PATH, "pi_train.npy"))[0]
        )
        train_loader = self._make_loader(X_train, y_train, shuffle=True)
        val_loader   = self._make_loader(X_val,   y_val)
        test_loader  = self._make_loader(X_test,  y_test)

        # 2. Modelo
        print("\n[-] 2. Construyendo modelo...")
        model = TabularResNet(
            input_dim     = X_train.shape[1],
            num_classes   = Config.NUM_CLASSES,
            hidden_dim    = hidden_dim,
            embed_dim     = Config.EMBED_DIM,
            n_blocks      = n_blocks,
            dropout       = dropout,
            inner_dropout = inner_dropout,
        ).to(self.device)

        # 3. Loss con class weights y criterios separados
        weights   = compute_class_weights(
            y_train,
            manual_weights = Config.MANUAL_WEIGHTS if use_class_weights else None,
        )
        weights_tensor = weights.to(self.device) if use_class_weights else None

        criterion_clean = nn.CrossEntropyLoss(
            weight=weights_tensor
        ).to(self.device)

        criterion_adv = EntropicBoundaryLoss(
            weight      = weights_tensor,
            num_classes = Config.NUM_CLASSES,
            lower_bound = 0.35,   # más alto — solo ejemplos realmente inciertos
            upper_bound = 0.55,
            kl_weight   = 0.2,    # más suave aún
        ).to(self.device)

        if use_class_weights:
            print("[-] Class weights aplicados.")

        # 4. Optimizador y scheduler
        optimizer  = torch.optim.AdamW(
            model.parameters(),
            lr           = lr,
            weight_decay = Config.WEIGHT_DECAY,
            betas        = (0.9, 0.999),
        )
        scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=4, min_lr=1e-5
        )

        path_best  = os.path.join(self.models, "resnet_refat_best.pt")
        path_last  = os.path.join(self.models, "resnet_refat_last.pt")
        early_stop = EarlyStopping(patience=patience, path=path_best)

        start_epoch = 1
        history     = {
            'train_loss': [], 'val_loss': [],
            'val_acc': [], 'val_f1': [], 'val_auc': [],
        }

        if resume and os.path.exists(path_last):
            print("\n[-] Restaurando checkpoint RE-FAT v2...")
            start_epoch, history, best_f1 = load_checkpoint(
                path_last, model, optimizer, scheduler, self.device
            )
            early_stop.best = best_f1
            start_epoch    += 1

        # 5. Bucle de entrenamiento
        print(
            f"\n[-] 3. Entrenando {start_epoch}→{epochs} | "
            f"ε={epsilon} | Mixup={mixup_alpha} | "
            f"clean/adv={clean_weight}/{adv_weight}"
        )
        print("-" * 60)
        pbar = create_progress_bar(epochs, start_epoch, "RE-FAT v2")

        for epoch in pbar:
            t0         = time.time()
            model.train()
            total_loss = 0.0
            batch_pbar = create_batch_progress_bar(
                train_loader, epoch, epochs
            )

            for X_batch, y_batch in batch_pbar:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                # A. Tabular Mixup sobre datos limpios
                lam = 1.0
                if mixup_alpha > 0:
                    lam    = Beta(mixup_alpha, mixup_alpha).sample().item()
                    lam    = max(0.0, min(1.0, lam))
                    index  = torch.randperm(X_batch.size(0)).to(self.device)
                    X_mix  = lam * X_batch + (1.0 - lam) * X_batch[index]
                    y_mix  = y_batch[index]
                else:
                    X_mix  = X_batch
                    y_mix  = y_batch

                # B. Fast-AT con Random Start
                X_adv = self._generate_fast_adv(
                    model, X_mix, y_batch, epsilon
                )

                # C. FORWARD & COMBINED LOSS — SEPARADO POR TIPO
                optimizer.zero_grad()
                model.train()

                X_combined         = torch.cat([X_mix, X_adv], dim=0)
                logits_combined, _ = model(X_combined)
                logits_clean, logits_adv = logits_combined.chunk(2, dim=0)

                # Loss limpia: Cross-Entropy PURA con class weights y Mixup
                # Los ejemplos limpios NUNCA van a EntropicBoundaryLoss
                loss_clean = lam * criterion_clean(logits_clean, y_batch)
                if mixup_alpha > 0:
                    loss_clean = loss_clean + (1.0 - lam) * criterion_clean(
                        logits_clean, y_mix
                    )

                # Loss adversarial: EntropicBoundaryLoss SOLO para ejemplos perturbados
                loss_adv = criterion_adv(logits_adv, y_batch)

                # Combinación ponderada — prioridad al tráfico limpio
                loss = clean_weight * loss_clean + adv_weight * loss_adv

                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=1.0
                )
                optimizer.step()

                total_loss += loss.item()
                batch_pbar.set_postfix(
                    {'loss': f"{loss.item():.4f}"}, refresh=True
                )

            batch_pbar.close()

            # Evaluación de época
            avg_loss = total_loss / len(train_loader)
            val_acc, val_f1, _, val_auc, _, _, _ = evaluate(
                model, val_loader, self.device
            )

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X_b, y_b in val_loader:
                    lgt, _ = model(X_b.to(self.device))
                    val_loss += criterion_clean(
                        lgt, y_b.to(self.device)
                    ).item()
            val_loss /= len(val_loader)

            elapsed = time.time() - t0
            lr_now  = optimizer.param_groups[0]['lr']

            log_line = (
                f"Epoch {epoch:03d}/{epochs} | "
                f"T.Loss: {avg_loss:.4f} | V.Loss: {val_loss:.4f} | "
                f"F1: {val_f1:.4f} | AUC: {val_auc:.4f} | "
                f"LR: {lr_now:.2e} | t={elapsed:.0f}s"
            )
            print(log_line, flush=True)
            with open(
                os.path.join(self.logs, self.current_log_file), "a"
            ) as f:
                f.write(log_line + "\n")

            history['train_loss'].append(avg_loss)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['val_f1'].append(val_f1)
            history['val_auc'].append(val_auc)

            scheduler.step(val_f1)
            early_stop.step(
                val_f1, model, optimizer, scheduler, epoch
            )
            save_checkpoint(
                path_last, epoch, model, optimizer,
                scheduler, history, early_stop.best,
            )

            if early_stop.triggered:
                print(f"\n[!] Early Stopping en epoch {epoch}")
                break

        # 6. Evaluación final
        print("\n" + "="*60)
        print("EVALUACIÓN FINAL RE-FAT v2")
        print("="*60)

        checkpoint = torch.load(
            path_best, map_location=self.device, weights_only=False
        )
        model.load_state_dict(checkpoint['model'])

        _, f1_lab, f1_per_lab, auc_lab, preds, labels, probs = evaluate(
            model, test_loader, self.device
        )

        # Guardar artefactos
        np.save(os.path.join(self.models, "train_history_refat.npy"), history)
        np.save(os.path.join(self.models, "test_preds_refat.npy"),    preds)
        np.save(os.path.join(self.models, "test_labels_refat.npy"),   labels)
        np.save(os.path.join(self.models, "test_probs_refat.npy"),    probs)

        print(f"\n  Test F1 (sin prior) : {f1_lab:.4f}")
        print(f"  Test AUC            : {auc_lab:.4f}")
        print(f"\n  IMPORTANTE: Comparativa justa requiere prior sweep.")
        print(f"  El modelo original usó π=0.05 como óptimo.")
        print(f"  Ejecutar prior sweep sobre test_probs_refat.npy")
        print(f"  antes de concluir sobre la mejora/degradación.")
        print(f"\n[-] Listo para acoplamiento con VAE + Mahalanobis.")

        return history, model