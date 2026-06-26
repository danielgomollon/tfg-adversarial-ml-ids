"""
src/models/resnet/sglat_trainer.py
================================================================
SGL-AT: Stochastic Geometric Latent Adversarial Training
Contribución TFG — Fase 3 (Blue Team)

Daniel Gomollón Embid — TFG 2025-2026

═══════════════════════════════════════════════════════════════
ARQUITECTURA: 5 Componentes Sinérgicos
═══════════════════════════════════════════════════════════════

1. Stochastic Fast-AT (RS-FGSM + Hybrid Dropout Mode)
   Genera ejemplos adversariales con Random Start.
   Modo híbrido: model.eval() protege BatchNorm,
   nn.Dropout forzado a train() mantiene la nube estocástica.

2. Centroid Atlas EMA (Warmup 5 épocas)
   Atlas de centroides por clase en espacio latente Z.
   Actualizado EXCLUSIVAMENTE con datos limpios.
   Nunca contaminado con troyanos S3M ni adversariales.

3. LGR — Latent Geometry Regularization
   Compactness: flujos cerca de su centroide de clase.
   Separation: centroides separados entre sí.
   S3M-Aware: troyanos empujados hacia clase verdadera.

4. ABH — Adaptive Boundary Hardening
   Zona boundary definida geométricamente (distancia entre
   centroides), no por confianza del softmax.
   Resuelve el colapso de clases raras de RE-FAT.

5. Asymmetric Gradient Clipping
   Grupos pre-computados en __init__ — costo O(1) total.
   Capas base: clip agresivo (0.1) en paso adversarial.
   SE blocks + final: clip normal (1.0) siempre.
   Sin ordenaciones masivas por batch.

═══════════════════════════════════════════════════════════════
CORRECCIONES SOBRE RE-FAT
═══════════════════════════════════════════════════════════════

RE-FAT v1/v2 fallaba por tres razones:
1. EntropicBoundaryLoss aplicada a datos limpios → aplastaba
   probabilidades de clases raras con confianza baja intrínseca
2. epsilon=0.15 demasiado grande → cruzaba fronteras de clase
3. Centroides no existían → zona boundary mal definida

SGL-AT resuelve los tres con arquitectura diferente:
- criterion_clean = CrossEntropy pura (datos limpios)
- criterion_adv   = ABHLoss geométrica (solo adversariales)
- epsilon=0.05 conservador
- Atlas EMA calibra la geometría antes de activar ABH
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
    save_checkpoint, create_progress_bar,
)


# ===========================================================================
# COMPONENTE 3+4: ABH LOSS — Adaptive Boundary Hardening
# ===========================================================================

class ABHLoss(nn.Module):
    """
    Adaptive Boundary Hardening Loss.

    Diferencia fundamental con EntropicBoundaryLoss:
    La zona boundary se define GEOMÉTRICAMENTE en el espacio
    latente Z (distancia a centroides), no por la confianza
    del softmax. Esto evita penalizar clases raras que
    tienen baja confianza intrínseca pero NO están en la frontera.

    Flujo de cálculo:
      1. Para cada muestra, calcular distancia a centroide propio
      2. Calcular distancia al centroide más cercano de otra clase
      3. Si ratio d_true/d_other > (1-margin): está en frontera
      4. En frontera → KL hacia uniforme
      5. Fuera de frontera → Cross-Entropy normal

    Parámetros
    ----------
    margin    : fracción de zona boundary [0, 1]
                0.3 → frontera cuando está al 70% del camino
                hacia otra clase
    kl_weight : peso de la penalización entrópica
    """

    def __init__(
        self,
        num_classes : int   = 8,
        margin      : float = 0.3,
        kl_weight   : float = 0.2,
        weight      : torch.Tensor = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.margin      = margin
        self.kl_weight   = kl_weight
        self.weight      = weight

    def forward(
        self,
        logits   : torch.Tensor,   # (B, C)
        targets  : torch.Tensor,   # (B,)
        Z        : torch.Tensor,   # (B, D) — espacio latente
        centroids: torch.Tensor,   # (C, D) — atlas EMA
    ) -> torch.Tensor:

        # ── Distancia al centroide propio ──────────────────────
        c_true = centroids[targets]                     # (B, D)
        d_true = torch.norm(Z - c_true, dim=1)         # (B,)

        # ── Distancia al centroide más cercano de otra clase ──
        # Vectorizado: calcular distancias a todos los centroides
        # Z: (B, D) → (B, 1, D)
        # centroids: (C, D) → (1, C, D)
        Z_exp   = Z.unsqueeze(1)                        # (B, 1, D)
        C_exp   = centroids.unsqueeze(0)                # (1, C, D)
        all_d   = torch.norm(Z_exp - C_exp, dim=2)     # (B, C)

        # Enmascarar centroide propio con inf para no elegirlo
        mask_own         = torch.zeros_like(all_d, dtype=torch.bool)
        mask_own.scatter_(1, targets.unsqueeze(1), True)
        all_d_masked     = all_d.masked_fill(mask_own, float('inf'))
        d_other, _       = all_d_masked.min(dim=1)     # (B,)

        # ── Zona boundary geométrica ───────────────────────────
        ratio       = d_true / (d_other + 1e-8)
        is_boundary = ratio > (1.0 - self.margin)      # (B,) bool

        # ── Loss base: Cross-Entropy con class weights ─────────
        loss = F.cross_entropy(logits, targets, weight=self.weight)

        # ── Penalización entrópica en frontera ─────────────────
        if is_boundary.any():
            n_boundary = is_boundary.sum()
            uniform    = torch.full(
                (n_boundary, self.num_classes),
                1.0 / self.num_classes,
                device=logits.device,
            )
            kl = F.kl_div(
                F.log_softmax(logits[is_boundary], dim=1),
                uniform,
                reduction='batchmean',
            )
            loss = loss + self.kl_weight * kl

        return loss


# ===========================================================================
# SGL-AT TRAINER
# ===========================================================================

class SGLAT_Trainer:
    """
    Entrenador SGL-AT con los 5 componentes integrados.

    Parámetros de entrenamiento recomendados para BigFlow-NIDS:
    - epsilon      = 0.05
    - warmup_epochs= 5    (construir atlas antes de activar LGR/ABH)
    - clean_weight = 0.55
    - adv_weight   = 0.30
    - lgr_weight   = 0.15
    - ema_momentum = 0.99 (atlas muy estable, actualización suave)
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

        # Atlas de centroides — inicializado en run() tras ver los datos
        self.centroids = None

        # Grupos de parámetros para Asymmetric Clipping
        # Pre-computados UNA VEZ — costo O(1) total
        # Se asignan en run() cuando el modelo está instanciado
        self._base_params     = None
        self._adaptive_params = None

    # ------------------------------------------------------------------
    # CARGA DE DATOS
    # ------------------------------------------------------------------
    def _load_data_hybrid(self):
        p     = self.processed
        s3m_p = os.path.normpath(os.path.join(p, '..', 'ataques_s3m_sgl'))

        X_orig = np.load(os.path.join(p,     "X_train.npy"))
        y_orig = np.load(os.path.join(p,     "y_train.npy"))
        X_s3m  = np.load(os.path.join(s3m_p, "X_train_s3m.npy"))
        y_s3m  = np.load(os.path.join(s3m_p, "y_train_s3m.npy"))

        # Marcador para identificar troyanos en el batch
        # 0 = original limpio, 1 = troyano S3M
        is_s3m_orig = np.zeros(len(X_orig), dtype=np.float32)
        is_s3m_s3m  = np.ones(len(X_s3m),  dtype=np.float32)

        X_train   = np.vstack([X_orig, X_s3m])
        y_train   = np.concatenate([y_orig, y_s3m])
        is_s3m    = np.concatenate([is_s3m_orig, is_s3m_s3m])

        X_val  = np.load(os.path.join(p, "X_val.npy"))
        y_val  = np.load(os.path.join(p, "y_val.npy"))
        X_test = np.load(os.path.join(p, "X_test.npy"))
        y_test = np.load(os.path.join(p, "y_test.npy"))

        print(f"   Train: {len(X_train):,} "
              f"({len(X_orig):,} limpio + {len(X_s3m):,} S3M)")
        print(f"   Val  : {len(X_val):,} | Test: {len(X_test):,}")

        return X_train, y_train, is_s3m, X_val, y_val, X_test, y_test

    def _make_loader(self, X, y, is_s3m=None, shuffle=False):
        if is_s3m is not None:
            dataset = TensorDataset(
                torch.tensor(X,      dtype=torch.float32),
                torch.tensor(y,      dtype=torch.long),
                torch.tensor(is_s3m, dtype=torch.float32),
            )
        else:
            # Val/Test sin marcador
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

    # ------------------------------------------------------------------
    # COMPONENTE 1: STOCHASTIC FAST-AT
    # Modo híbrido: BatchNorm protegido, Dropout activo
    # ------------------------------------------------------------------
    def _generate_fast_adv(
        self,
        model   : nn.Module,
        X       : torch.Tensor,
        y       : torch.Tensor,
        epsilon : float,
    ) -> torch.Tensor:
        """
        RS-FGSM con modo híbrido BatchNorm/Dropout.

        TRAMPA MORTAL 1 resuelta:
        - model.eval() → protege BatchNorm de contaminación adversarial
        - nn.Dropout forzado a train() → mantiene nube estocástica DAG
        """
        # 1. Modo eval global — proteger BatchNorm
        model.eval()

        # 2. Forzar SOLO Dropout a train — nube estocástica activa
        for m in model.modules():
            if isinstance(m, nn.Dropout):
                m.train()

        # 3. Random Start — nube adversarial inicial
        noise = torch.empty_like(X).uniform_(-epsilon, epsilon)
        noise[:, self.frozen_mask_t] = 0.0  # respetar física L4/L7

        X_adv = (X + noise).detach().requires_grad_(True)

        # 4. Forward + gradiente
        logits, _ = model(X_adv)
        loss      = F.cross_entropy(logits, y)
        loss.backward()

        grad = X_adv.grad.detach()
        grad[:, self.frozen_mask_t] = 0.0

        # 5. Paso FGSM
        alpha = epsilon * 1.25
        X_adv = X_adv.detach() + alpha * grad.sign()

        # 6. Proyectar al ball de epsilon y restaurar modo train
        X_adv  = torch.clamp(X_adv, X - epsilon, X + epsilon).detach()
        model.train()  # restaurar para el forward principal

        return X_adv

    # ------------------------------------------------------------------
    # COMPONENTE 2: CENTROID ATLAS EMA
    # Actualizado EXCLUSIVAMENTE con datos limpios
    # ------------------------------------------------------------------
    def _init_centroids(
        self, model: nn.Module, loader: DataLoader, num_classes: int
    ) -> torch.Tensor:
        """
        Inicializa el atlas de centroides haciendo un forward pass
        sobre datos limpios antes de empezar el entrenamiento.
        """
        print("   [Atlas] Inicializando centroides EMA...")
        model.eval()

        sums   = torch.zeros(num_classes, Config.EMBED_DIM,
                             device=self.device)
        counts = torch.zeros(num_classes, device=self.device)

        with torch.no_grad():
            for batch in loader:
                X_b = batch[0].to(self.device)
                y_b = batch[1].to(self.device)

                # Solo datos limpios para inicialización
                if len(batch) == 3:
                    is_s3m_b = batch[2].to(self.device)
                    clean    = is_s3m_b == 0
                    if clean.sum() == 0:
                        continue
                    X_b = X_b[clean]
                    y_b = y_b[clean]

                _, Z = model(X_b)
                for c in range(num_classes):
                    mask = y_b == c
                    if mask.sum() > 0:
                        sums[c]   += Z[mask].sum(0)
                        counts[c] += mask.sum()

        centroids = sums / (counts.unsqueeze(1) + 1e-8)
        model.train()
        print(f"   [Atlas] Centroides inicializados. "
              f"Shape: {centroids.shape}")
        return centroids

    @torch.no_grad()
    def _update_centroids(
        self,
        centroids    : torch.Tensor,
        Z_clean      : torch.Tensor,
        y_clean      : torch.Tensor,
        ema_momentum : float,
        num_classes  : int,
    ) -> torch.Tensor:
        """
        Actualización EMA del atlas.

        TRAMPA MORTAL 2 resuelta:
        Solo recibe Z_clean — nunca troyanos S3M ni adversariales.
        """
        for c in range(num_classes):
            mask = y_clean == c
            if mask.sum() > 0:
                batch_centroid  = Z_clean[mask].mean(0)
                centroids[c]    = (
                    ema_momentum * centroids[c]
                    + (1.0 - ema_momentum) * batch_centroid
                )
        return centroids

    # ------------------------------------------------------------------
    # COMPONENTE 3: LGR — Latent Geometry Regularization
    # ------------------------------------------------------------------
    def _compute_lgr_loss(
        self,
        Z         : torch.Tensor,
        y         : torch.Tensor,
        centroids : torch.Tensor,
        is_s3m    : torch.Tensor,
        num_classes: int,
    ) -> torch.Tensor:
        """
        LGR con tres términos:
        1. Compactness: flujos cerca de su centroide de clase
        2. Separation: centroides bien separados entre sí
        3. S3M-Aware: troyanos empujados hacia clase verdadera
                      y alejados del centroide benigno (clase 0)
        """
        # 1. Compactness — solo sobre datos limpios
        mask_clean = is_s3m == 0
        if mask_clean.sum() > 0:
            c_true        = centroids[y[mask_clean]]
            l_compact     = torch.norm(
                Z[mask_clean] - c_true, dim=1
            ).mean()
        else:
            l_compact = torch.tensor(0.0, device=Z.device)

        # 2. Separation — sobre centroides (independiente del batch)
        C_exp    = centroids.unsqueeze(0)           # (1, C, D)
        C_exp2   = centroids.unsqueeze(1)           # (C, 1, D)
        dist_mat = torch.norm(C_exp - C_exp2, dim=2)  # (C, C)

        # Excluir diagonal (distancia a sí mismo)
        eye      = torch.eye(num_classes, device=Z.device).bool()
        dist_off = dist_mat.masked_fill(eye, float('inf'))
        min_sep  = dist_off.min(dim=1).values       # (C,)
        l_sep    = -torch.log(min_sep + 1e-8).mean()

        # 3. S3M-Aware — solo sobre troyanos
        mask_s3m = is_s3m == 1
        if mask_s3m.sum() > 0:
            # Empujar hacia clase verdadera
            c_true_s3m  = centroids[y[mask_s3m]]
            d_true_s3m  = torch.norm(
                Z[mask_s3m] - c_true_s3m, dim=1
            )
            # Alejar de centroide benigno (clase 0)
            d_benign_s3m = torch.norm(
                Z[mask_s3m] - centroids[0].unsqueeze(0), dim=1
            )
            l_s3m = (d_true_s3m - d_benign_s3m).clamp(min=0).mean()
        else:
            l_s3m = torch.tensor(0.0, device=Z.device)

        return l_compact + 0.5 * l_sep + l_s3m

    # ------------------------------------------------------------------
    # COMPONENTE 5: ASYMMETRIC CLIPPING
    # Pre-computado — costo O(1) total, no por batch
    # ------------------------------------------------------------------
    def _precompute_param_groups(self, model: nn.Module):
        """
        Separa parámetros en base y adaptativos UNA SOLA VEZ.
        SE blocks y capa final reciben clip normal en paso adversarial.
        Capas base reciben clip agresivo para proteger conocimiento limpio.
        """
        self._base_params = [
            p for name, p in model.named_parameters()
            if 'se_block' not in name.lower()
            and 'fc_out' not in name.lower()
            and 'final' not in name.lower()
        ]
        self._adaptive_params = [
            p for name, p in model.named_parameters()
            if 'se_block' in name.lower()
            or 'fc_out' in name.lower()
            or 'final' in name.lower()
        ]
        print(f"   [Clipping] Base params     : "
              f"{sum(p.numel() for p in self._base_params):,}")
        print(f"   [Clipping] Adaptive params : "
              f"{sum(p.numel() for p in self._adaptive_params):,}")

    def _apply_asymmetric_clipping(
        self, is_adv_batch: bool, max_norm_base: float = 0.1
    ):
        """
        Clip asimétrico en dos llamadas nativas C++ — sin ordenaciones.
        """
        if not is_adv_batch:
            torch.nn.utils.clip_grad_norm_(
                self._base_params + self._adaptive_params,
                max_norm=1.0,
            )
        else:
            # Base: clip agresivo — protege conocimiento limpio
            torch.nn.utils.clip_grad_norm_(
                self._base_params, max_norm=max_norm_base
            )
            # Adaptive: clip normal — SE blocks aprenden defensa
            torch.nn.utils.clip_grad_norm_(
                self._adaptive_params, max_norm=1.0
            )

    # ------------------------------------------------------------------
    # BUCLE PRINCIPAL
    # ------------------------------------------------------------------
    def run(
        self,
        resume          : bool  = False,
        epochs          : int   = 45,
        lr              : float = 1e-3,
        patience        : int   = 12,
        dropout         : float = 0.1,
        inner_dropout   : float = 0.08,  # activo en inferencia adversarial
        hidden_dim      : int   = 256,
        n_blocks        : int   = 4,
        mixup_alpha     : float = 0.2,
        use_class_weights: bool = True,
        epsilon         : float = 0.05,
        warmup_epochs   : int   = 5,     # épocas sin LGR/ABH
        clean_weight    : float = 0.55,
        adv_weight      : float = 0.30,
        lgr_weight      : float = 0.25,
        ema_momentum    : float = 0.99,
    ):
        print("\n" + "=" * 60)
        print("ENTRENAMIENTO — SGL-AT [Fase 3]")
        print(f"  ε={epsilon} | warmup={warmup_epochs} | "
              f"clean/adv/lgr={clean_weight}/{adv_weight}/{lgr_weight}")
        print("=" * 60)

        # Logs
        log_idx = 1
        while os.path.exists(
            os.path.join(self.logs, f"sglat_log_{log_idx}.txt")
        ):
            log_idx += 1
        log_file = os.path.join(self.logs, f"sglat_log_{log_idx}.txt")

        # 1. Datos
        print("\n[-] 1. Cargando datos...")
        X_train, y_train, is_s3m, X_val, y_val, X_test, y_test = \
            self._load_data_hybrid()

        pi_train     = float(
            np.load(os.path.join(Config.MODELS_PATH, "pi_train.npy"))[0]
        )
        train_loader = self._make_loader(
            X_train, y_train, is_s3m, shuffle=True
        )
        val_loader   = self._make_loader(X_val, y_val)
        test_loader  = self._make_loader(X_test, y_test)

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

        # 3. Pre-computar grupos de clipping
        print("\n[-] 3. Pre-computando grupos de parámetros...")
        self._precompute_param_groups(model)

        # 4. Criterios separados — corrección raíz de RE-FAT
        weights_t = None
        if use_class_weights:
            weights_t = compute_class_weights(
                y_train,
                manual_weights=Config.MANUAL_WEIGHTS,
            ).to(self.device)
            print("[-] Class weights aplicados.")

        # criterion_clean: Cross-Entropy PURA — nunca modifica clases raras
        criterion_clean = nn.CrossEntropyLoss(
            weight=weights_t
        ).to(self.device)

        # criterion_adv: ABH geométrica — solo para adversariales
        criterion_adv = ABHLoss(
            num_classes = Config.NUM_CLASSES,
            margin      = 0.45,
            kl_weight   = 0.2,
            weight      = weights_t,
        ).to(self.device)

        # 5. Optimizador
        optimizer  = torch.optim.AdamW(
            model.parameters(), lr=lr,
            weight_decay=Config.WEIGHT_DECAY, betas=(0.9, 0.999),
        )
        scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=4, min_lr=1e-5
        )

        path_best  = os.path.join(self.models, "resnet_sglat_best.pt")
        path_last  = os.path.join(self.models, "resnet_sglat_last.pt")
        early_stop = EarlyStopping(patience=patience, path=path_best)

        start_epoch = 1
        history     = {
            'train_loss': [], 'val_loss': [],
            'val_acc': [], 'val_f1': [], 'val_auc': [],
            'lgr_active': [],
        }

        if resume and os.path.exists(path_last):
            print("\n[-] Restaurando checkpoint SGL-AT...")
            start_epoch, history, best_f1 = load_checkpoint(
                path_last, model, optimizer, scheduler, self.device
            )
            early_stop.best = best_f1
            start_epoch    += 1

        # 6. Inicializar atlas de centroides
        print("\n[-] 4. Inicializando atlas de centroides EMA...")
        self.centroids = self._init_centroids(
            model, train_loader, Config.NUM_CLASSES
        )

        # 7. Bucle de entrenamiento
        print(
            f"\n[-] 5. Entrenando {start_epoch}→{epochs} | "
            f"Warmup={warmup_epochs} épocas sin LGR/ABH"
        )
        print("-" * 60)
        pbar = create_progress_bar(epochs, start_epoch, "SGL-AT")

        for epoch in pbar:
            t0          = time.time()
            model.train()
            total_loss  = 0.0
            lgr_active  = epoch > warmup_epochs
            batch_pbar  = create_batch_progress_bar(
                train_loader, epoch, epochs
            )

            for batch in batch_pbar:
                X_batch    = batch[0].to(self.device)
                y_batch    = batch[1].to(self.device)
                is_s3m_b   = batch[2].to(self.device)  # (B,) float

                # Máscara datos limpios — NUNCA tocan el atlas
                mask_clean = is_s3m_b == 0

                # ── A. Tabular Mixup (solo sobre datos limpios) ──
                lam = 1.0
                if mixup_alpha > 0 and mask_clean.sum() > 1:
                    lam   = Beta(mixup_alpha, mixup_alpha).sample().item()
                    lam   = max(0.0, min(1.0, lam))
                    idx   = torch.randperm(
                        mask_clean.sum()
                    ).to(self.device)
                    X_clean = X_batch[mask_clean]
                    y_clean = y_batch[mask_clean]
                    X_mix   = lam * X_clean + (1.0 - lam) * X_clean[idx]
                    y_mix   = y_clean[idx]
                else:
                    X_clean = X_batch[mask_clean]
                    y_clean = y_batch[mask_clean]
                    X_mix   = X_clean
                    y_mix   = y_clean

                # ── B. Fast-AT con modo híbrido BatchNorm/Dropout ──
                if X_mix.shape[0] > 0:
                    X_adv = self._generate_fast_adv(
                        model, X_mix, y_clean, epsilon
                    )
                else:
                    X_adv = X_mix

                # ── C. Forward pass combinado ──────────────────────
                optimizer.zero_grad()
                model.train()

                # Pasar todo en un solo forward para eficiencia
                # Orden: [limpios/mix, adversariales, troyanos_s3m]
                X_s3m_b = X_batch[~mask_clean] if (~mask_clean).any() \
                          else None
                y_s3m_b = y_batch[~mask_clean] if (~mask_clean).any() \
                          else None

                parts_X = [X_mix, X_adv]
                parts_y = [y_clean, y_clean]
                if X_s3m_b is not None and X_s3m_b.shape[0] > 0:
                    parts_X.append(X_s3m_b)
                    parts_y.append(y_s3m_b)

                X_combined      = torch.cat(parts_X, dim=0)
                logits_all, Z_all = model(X_combined)

                n_clean = X_mix.shape[0]
                n_adv   = X_adv.shape[0]
                n_s3m   = X_s3m_b.shape[0] if X_s3m_b is not None else 0

                logits_clean = logits_all[:n_clean]
                Z_clean_fwd  = Z_all[:n_clean]
                logits_adv   = logits_all[n_clean:n_clean+n_adv]
                Z_adv        = Z_all[n_clean:n_clean+n_adv]

                if n_s3m > 0:
                    logits_s3m = logits_all[n_clean+n_adv:]
                    Z_s3m_fwd  = Z_all[n_clean+n_adv:]

                # ── D. Actualizar atlas EMA ────────────────────────
                # SOLO con Z_clean_fwd — jamás con adversariales o S3M
                with torch.no_grad():
                    self.centroids = self._update_centroids(
                        self.centroids, Z_clean_fwd.detach(),
                        y_clean, ema_momentum, Config.NUM_CLASSES,
                    )

                # ── E. Calcular pérdidas ───────────────────────────

                # E.1 Loss limpia — Cross-Entropy pura con Mixup
                loss_clean = lam * criterion_clean(logits_clean, y_clean)
                if mixup_alpha > 0 and lam < 1.0:
                    loss_clean = loss_clean + (1.0 - lam) * criterion_clean(
                        logits_clean, y_mix
                    )

                # E.2 Loss adversarial — ABH geométrica
                if lgr_active:
                    loss_adv = criterion_adv(
                        logits_adv, y_clean,
                        Z_adv, self.centroids,
                    )
                else:
                    # Warmup: solo Cross-Entropy en adversariales
                    loss_adv = criterion_clean(logits_adv, y_clean)

                # E.3 LGR — activa tras warmup
                if lgr_active:
                    # Construir Z y y completos para LGR
                    Z_full = Z_all
                    y_full = torch.cat(parts_y, dim=0)
                    is_s3m_full = torch.cat([
                        torch.zeros(n_clean + n_adv,
                                    device=self.device),
                        torch.ones(n_s3m, device=self.device),
                    ])
                    loss_lgr = self._compute_lgr_loss(
                        Z_full, y_full, self.centroids,
                        is_s3m_full, Config.NUM_CLASSES,
                    )
                else:
                    loss_lgr = torch.tensor(0.0, device=self.device)

                # E.4 Loss total ponderada
                loss = (
                    clean_weight * loss_clean
                    + adv_weight  * loss_adv
                    + lgr_weight  * loss_lgr
                )

                # ── F. Backward con Asymmetric Clipping ───────────
                loss.backward()
                self._apply_asymmetric_clipping(is_adv_batch=True)
                optimizer.step()

                total_loss += loss.item()
                batch_pbar.set_postfix(
                    {
                        'loss': f"{loss.item():.4f}",
                        'lgr' : 'ON' if lgr_active else 'warm',
                    },
                    refresh=True,
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
            status  = "LGR+ABH" if lgr_active else f"WARMUP {epoch}/{warmup_epochs}"

            log_line = (
                f"Epoch {epoch:03d}/{epochs} | [{status}] | "
                f"T.Loss: {avg_loss:.4f} | V.Loss: {val_loss:.4f} | "
                f"F1: {val_f1:.4f} | AUC: {val_auc:.4f} | "
                f"LR: {lr_now:.2e} | t={elapsed:.0f}s"
            )
            print(log_line, flush=True)
            with open(log_file, "a") as f:
                f.write(log_line + "\n")

            history['train_loss'].append(avg_loss)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['val_f1'].append(val_f1)
            history['val_auc'].append(val_auc)
            history['lgr_active'].append(lgr_active)

            scheduler.step(val_f1)
            early_stop.step(val_f1, model, optimizer, scheduler, epoch)
            save_checkpoint(
                path_last, epoch, model, optimizer,
                scheduler, history, early_stop.best,
            )

            if early_stop.triggered:
                print(f"\n[!] Early Stopping en epoch {epoch}")
                break

        # 8. Evaluación final
        print("\n" + "=" * 60)
        print("EVALUACIÓN FINAL SGL-AT")
        print("=" * 60)

        checkpoint = torch.load(
            path_best, map_location=self.device, weights_only=False
        )
        model.load_state_dict(checkpoint['model'])

        _, f1_lab, f1_per_lab, auc_lab, preds, labels, probs = evaluate(
            model, test_loader, self.device
        )

        np.save(os.path.join(self.models, "train_history_sglat.npy"), history)
        np.save(os.path.join(self.models, "test_preds_sglat.npy"),    preds)
        np.save(os.path.join(self.models, "test_labels_sglat.npy"),   labels)
        np.save(os.path.join(self.models, "test_probs_sglat.npy"),    probs)

        print(f"\n  Test F1 (sin prior) : {f1_lab:.4f}")
        print(f"  Test AUC            : {auc_lab:.4f}")
        print(f"\n  F1 por clase:")
        class_names = [
            'Benign', 'DoS', 'DDoS', 'Web/Injection',
            'Brute Force', 'Recon', 'Malware', 'Exploits'
        ]
        for i, (name, f1) in enumerate(zip(class_names, f1_per_lab)):
            bar = "█" * int(f1 * 20)
            print(f"    {i} {name:<15} {f1:.4f} {bar}")

        print(f"\n[-] Ejecutar prior sweep sobre test_probs_sglat.npy")
        print(f"[-] Listo para acoplamiento con VAE + Mahalanobis.")

        return history, model