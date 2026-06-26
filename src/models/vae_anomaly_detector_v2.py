"""
vae_anomaly_detector_v2.py
===========================================================================
VAE + Distancia de Mahalanobis — Detección de anomalías BigFlow-NIDS
Versión 2: Ablación de tres variantes de espacio de representación

VARIANTES IMPLEMENTADAS (MahalanobisMode):
  'z'          → Mahalanobis sobre espacio latente z  (baseline clásico)
  'recon'      → Mahalanobis sobre vector de errores de reconstrucción
                 por feature — shape (n, input_dim)
  'z+recon'    → Concatenación z ∥ recon_vector — inspirado en DAGMM
                 (Zong et al., 2018). Combina información estructural
                 del espacio latente con firma de error dimensional. 

Hipótesis:
  - 'z'      : sufre cuando ataques como Recon son superficialmente
                similares al tráfico benigno; z aprende esa similitud.
  - 'recon'  : captura qué features específicas el VAE falla en
                reconstruir — alta sensibilidad a desvíos puntuales.
  - 'z+recon': captura ambas fuentes de señal simultáneamente; en
                teoría el más potente (verificar con ablación).

Uso:
    # Entrenar y evaluar las tres variantes
    results = run_ablation(X_train_benign, X_val_benign, X_test, y_test)

    # O usar una sola variante
    detector = VAEAnomalyDetector(input_dim=66, mode='z+recon')
    detector.fit(X_train_benign, X_val_benign)
    results  = detector.evaluate(X_test, y_test)

Referencias:
  - Kingma & Welling (2013) — Auto-Encoding Variational Bayes
  - Zong et al. (2018)      — Deep Autoencoding Gaussian Mixture Model
  - Lee et al. (2018)       — Mahalanobis distance for OOD detection
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (roc_auc_score, f1_score, confusion_matrix,
                              precision_recall_curve, classification_report)
import joblib
from typing import Optional, Tuple, Literal
import time


# ===========================================================================
# CONFIGURACIÓN
# ===========================================================================

class VAEConfig:
    """Hiperparámetros del VAE."""

    # Arquitectura
    LATENT_DIM   = 32
    HIDDEN_DIMS  = [256, 128, 64]
    DROPOUT      = 0.1

    # Entrenamiento
    BATCH_SIZE   = 2048
    EPOCHS       = 100
    LR           = 1e-3
    WEIGHT_DECAY = 1e-5
    PATIENCE     = 15
    LR_PATIENCE  = 7
    LR_FACTOR    = 0.5
    BETA         = 0.1       # β-VAE: peso KL divergence

    # Mahalanobis
    THRESHOLD_PCT = 95.0     # percentil para umbral de anomalía

    # Sistema
    SEED   = 42
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


MahalanobisMode = Literal['z', 'recon', 'z+recon']


# ===========================================================================
# ARQUITECTURA VAE (sin cambios respecto a v1)
# ===========================================================================

class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim), nn.BatchNorm1d(dim),
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.net(x))


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim, dropout=0.1):
        super().__init__()
        layers, in_dim = [], input_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.BatchNorm1d(h), nn.GELU(),
                       nn.Dropout(dropout), ResidualBlock(h, dropout)]
            in_dim = h
        self.net    = nn.Sequential(*layers)
        self.fc_mu  = nn.Linear(in_dim, latent_dim)
        self.fc_var = nn.Linear(in_dim, latent_dim)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x):
        h = self.net(x)
        mu = self.fc_mu(h)
        log_var = torch.clamp(self.fc_var(h), -10.0, 4.0)
        return mu, log_var


class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dims, output_dim, dropout=0.1):
        super().__init__()
        layers, in_dim = [], latent_dim
        for h in reversed(hidden_dims):
            layers += [nn.Linear(in_dim, h), nn.BatchNorm1d(h), nn.GELU(),
                       nn.Dropout(dropout), ResidualBlock(h, dropout)]
            in_dim = h
        self.net    = nn.Sequential(*layers)
        self.fc_out = nn.Linear(in_dim, output_dim)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, z):
        return self.fc_out(self.net(z))


class VAE(nn.Module):
    def __init__(self, input_dim: int, config: VAEConfig = None):
        super().__init__()
        cfg = config or VAEConfig()
        self.input_dim  = input_dim
        self.latent_dim = cfg.LATENT_DIM
        self.beta       = cfg.BETA
        self.encoder = Encoder(input_dim, cfg.HIDDEN_DIMS, cfg.LATENT_DIM, cfg.DROPOUT)
        self.decoder = Decoder(cfg.LATENT_DIM, cfg.HIDDEN_DIMS, input_dim, cfg.DROPOUT)

    def reparameterize(self, mu, log_var):
        if self.training:
            return mu + torch.randn_like(mu) * torch.exp(0.5 * log_var)
        return mu

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        return self.decoder(z), mu, log_var, z

    def encode(self, x):
        return self.encoder(x)

    @staticmethod
    def loss(x, x_recon, mu, log_var, beta=1.0):
        recon = F.mse_loss(x_recon, x, reduction='mean')
        kl    = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        return recon + beta * kl, recon, kl


# ===========================================================================
# DETECTOR DE MAHALANOBIS (genérico — acepta cualquier espacio)
# ===========================================================================

class MahalanobisDetector:
    def __init__(self):
        self.scaler_mean : Optional[np.ndarray] = None
        self.scaler_std  : Optional[np.ndarray] = None
        self.mu_train    : Optional[np.ndarray] = None
        self.cov_inv     : Optional[np.ndarray] = None
        self.threshold   : Optional[float]      = None
        self.threshold_optimal : Optional[float] = None  # ← nuevo
        self.fitted      : bool                  = False

    def fit(self, vectors: np.ndarray, threshold_pct: float = 95.0):
        # 1. Estandarización interna — crítica para z+recon
        self.scaler_mean = vectors.mean(axis=0)
        self.scaler_std  = vectors.std(axis=0) + 1e-8  # evita div/0
        v_scaled = (vectors - self.scaler_mean) / self.scaler_std

        # 2. Mahalanobis sobre espacio escalado
        self.mu_train = v_scaled.mean(axis=0)
        cov = np.cov(v_scaled.T) + np.eye(v_scaled.shape[1]) * 1e-6
        self.cov_inv  = np.linalg.inv(cov)

        dists = self._batch_scaled(v_scaled)  # ← directo, sin re-escalar
        self.threshold = np.percentile(dists, threshold_pct)
        self.fitted    = True

        print(f"   [Mahalanobis] dim={vectors.shape[1]} | "
              f"μ_dist={dists.mean():.3f} | σ={dists.std():.3f} | "
              f"threshold(p{threshold_pct:.0f})={self.threshold:.3f}")

    def score(self, vectors: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("Llama fit() primero")
        v_scaled = (vectors - self.scaler_mean) / self.scaler_std
        return self._batch_scaled(v_scaled)

    def predict(self, vectors: np.ndarray) -> np.ndarray:
        return (self.score(vectors) > self.threshold).astype(int)

    def _batch_scaled(self, z_scaled: np.ndarray) -> np.ndarray:
        """Opera sobre vectores ya estandarizados."""
        diff = z_scaled - self.mu_train
        left = diff @ self.cov_inv
        return np.sqrt(np.maximum((left * diff).sum(axis=1), 0.0))

    # Mantén _batch como alias público por compatibilidad si lo necesitas
    def _batch(self, z: np.ndarray) -> np.ndarray:
        return self.score(z)


# ===========================================================================
# DETECTOR COMPLETO VAE + MAHALANOBIS (tres modos)
# ===========================================================================

class VAEAnomalyDetector:
    """
    VAE + Mahalanobis con tres modos de representación configurables.

    mode='z'       : espacio latente z  (baseline clásico)
    mode='recon'   : vector de errores por feature  (robusto a Recon)
    mode='z+recon' : z ∥ recon_vector  (inspirado en DAGMM, máxima info)
    """

    def __init__(self, input_dim: int, config: VAEConfig = None,
                 mode: MahalanobisMode = 'z+recon',
                 models_path: str = 'outputs/models'):
        self.cfg         = config or VAEConfig()
        self.input_dim   = input_dim
        self.mode        = mode
        self.models_path = models_path
        self.device      = torch.device(self.cfg.DEVICE)

        os.makedirs(models_path, exist_ok=True)
        torch.manual_seed(self.cfg.SEED)
        np.random.seed(self.cfg.SEED)

        self.vae      = VAE(input_dim, self.cfg).to(self.device)
        self.detector = MahalanobisDetector()

        # dimensión del espacio de representación según modo
        dim_map = {
            'z'      : self.cfg.LATENT_DIM,
            'recon'  : input_dim,
            'z+recon': self.cfg.LATENT_DIM + input_dim,
        }
        self.repr_dim = dim_map[mode]

        print(f"\n   [VAE] mode={mode} | repr_dim={self.repr_dim} | "
              f"device={self.device} | "
              f"params={sum(p.numel() for p in self.vae.parameters()):,}")

    # ------------------------------------------------------------------
    # REPRESENTACIÓN — corazón de las tres variantes
    # ------------------------------------------------------------------
    def _get_representation(self, X: np.ndarray) -> np.ndarray:
        """
        Obtiene el vector de representación según self.mode.

        'z'      → μ del encoder                         (n, latent_dim)
        'recon'  → (x_recon - x)²  por feature           (n, input_dim)
        'z+recon'→ concat(μ, (x_recon - x)²)             (n, latent_dim + input_dim)
        """
        self.vae.eval()
        loader = DataLoader(
            TensorDataset(torch.FloatTensor(X)),
            batch_size=self.cfg.BATCH_SIZE * 4,
            shuffle=False,
        )
        zs, recons = [], []
        with torch.no_grad():
            for (xb,) in loader:
                xb = xb.to(self.device)
                x_recon, mu, _, _ = self.vae(xb)

                if self.mode in ('z', 'z+recon'):
                    zs.append(mu.cpu().numpy())
                if self.mode in ('recon', 'z+recon'):
                    recons.append((x_recon - xb).pow(2).cpu().numpy())

        if self.mode == 'z':
            return np.concatenate(zs, axis=0)
        elif self.mode == 'recon':
            return np.concatenate(recons, axis=0)
        else:  # z+recon
            return np.concatenate(
                [np.concatenate(zs, axis=0),
                 np.concatenate(recons, axis=0)], axis=1
            )

    # ------------------------------------------------------------------
    # ENTRENAMIENTO
    # ------------------------------------------------------------------
    def fit(self, X_train_benign: np.ndarray,
            X_val_benign: Optional[np.ndarray] = None):
        """Entrena el VAE con tráfico benigno y ajusta el detector."""
        print(f"\n{'='*60}")
        print(f"VAE — ENTRENAMIENTO  [mode={self.mode}]")
        print(f"{'='*60}")
        print(f"   Train: {X_train_benign.shape}")
        if X_val_benign is not None:
            print(f"   Val:   {X_val_benign.shape}")

        train_loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_train_benign).to(self.device)),
            batch_size=self.cfg.BATCH_SIZE, shuffle=True, drop_last=True,
        )
        val_loader = None
        if X_val_benign is not None:
            val_loader = DataLoader(
                TensorDataset(torch.FloatTensor(X_val_benign).to(self.device)),
                batch_size=self.cfg.BATCH_SIZE * 2, shuffle=False,
            )

        optimizer = torch.optim.AdamW(
            self.vae.parameters(), lr=self.cfg.LR,
            weight_decay=self.cfg.WEIGHT_DECAY,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=self.cfg.LR_FACTOR,
            patience=self.cfg.LR_PATIENCE,
        )

        best_val_loss, patience_count, best_state = float('inf'), 0, None

        print(f"\n{'Epoch':>6} | {'T.Loss':>8} | {'T.Recon':>8} | "
              f"{'T.KL':>7} | {'V.Loss':>8} | {'LR':>10}")
        print("-" * 60)

        for epoch in range(1, self.cfg.EPOCHS + 1):
            t0 = time.time()
            self.vae.train()
            t_loss = t_recon = t_kl = 0.0

            for (xb,) in train_loader:
                optimizer.zero_grad()
                x_recon, mu, log_var, _ = self.vae(xb)
                loss, recon, kl = VAE.loss(xb, x_recon, mu, log_var, self.cfg.BETA)
                loss.backward()
                nn.utils.clip_grad_norm_(self.vae.parameters(), 1.0)
                optimizer.step()
                t_loss += loss.item(); t_recon += recon.item(); t_kl += kl.item()

            n = len(train_loader)
            t_loss /= n; t_recon /= n; t_kl /= n

            v_loss = float('nan')
            if val_loader is not None:
                self.vae.eval()
                v_loss = 0.0
                with torch.no_grad():
                    for (xb,) in val_loader:
                        x_recon, mu, log_var, _ = self.vae(xb)
                        loss, _, _ = VAE.loss(xb, x_recon, mu, log_var, self.cfg.BETA)
                        v_loss += loss.item()
                v_loss /= len(val_loader)
                scheduler.step(v_loss)

                if v_loss < best_val_loss - 1e-5:
                    best_val_loss  = v_loss
                    patience_count = 0
                    best_state = {k: v.cpu().clone()
                                  for k, v in self.vae.state_dict().items()}
                else:
                    patience_count += 1

            lr_now = optimizer.param_groups[0]['lr']
            dt     = time.time() - t0
            marker = " ✓" if patience_count == 0 and val_loader else ""
            print(f"{epoch:>6} | {t_loss:>8.4f} | {t_recon:>8.4f} | "
                  f"{t_kl:>7.4f} | {v_loss:>8.4f} | {lr_now:>10.2e}  "
                  f"[{dt:.0f}s]{marker}")

            if patience_count >= self.cfg.PATIENCE:
                print(f"\n   [EarlyStopping] epoch {epoch}")
                break

        if best_state is not None:
            self.vae.load_state_dict(best_state)
            print(f"   [✓] Mejor modelo restaurado (val_loss={best_val_loss:.4f})")

        # Ajustar Mahalanobis sobre el espacio de representación elegido
        print(f"\n[-] Ajustando Mahalanobis sobre espacio '{self.mode}'...")
        repr_benign = self._get_representation(X_train_benign)
        self.detector.fit(repr_benign, self.cfg.THRESHOLD_PCT)
        print(f"\n[✓] VAE+Mahalanobis[{self.mode}] listo")

    # ------------------------------------------------------------------
    # EVALUACIÓN
    # ------------------------------------------------------------------
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """
        Evalúa el detector. y_test: 0=benigno, !=0=ataque.
        
        Threshold de train (percentil p95 sobre benignos): usado en producción.
        Threshold óptimo (max F1 sobre PR curve del test): sólo para análisis
        académico — NO usar en producción para evitar data leakage conceptual.
        """
        print(f"\n{'='*60}")
        print(f"VAE — EVALUACIÓN  [mode={self.mode}]")
        print(f"{'='*60}")

        y_binary  = (y_test != 0).astype(int)
        repr_test = self._get_representation(X_test)
        scores    = self.detector.score(repr_test)

        # ── Threshold de train (el único válido para producción) ──────────
        auc   = roc_auc_score(y_binary, scores)
        preds = (scores > self.detector.threshold).astype(int)
        f1    = f1_score(y_binary, preds, zero_division=0)

        # ── Threshold óptimo por F1 (sólo análisis académico) ────────────
        prec, rec, thrs = precision_recall_curve(y_binary, scores)
        f1_curve = 2 * prec * rec / (prec + rec + 1e-8)
        best_idx = f1_curve.argmax()
        best_thr = thrs[best_idx] if best_idx < len(thrs) else self.detector.threshold

        # Guardar en el detector para que el pipeline híbrido pueda leerlo
        # si lo necesita, siendo explícito de que viene del test set
        self.detector.threshold_optimal = float(best_thr)

        print(f"\n   AUC-ROC:                         {auc:.4f}")
        print(f"   F1 (threshold train p{self.cfg.THRESHOLD_PCT:.0f}):    {f1:.4f}")
        print(f"   F1 óptimo* (thr={best_thr:.3f}):       {f1_curve[best_idx]:.4f}")
        print(f"   (* threshold calculado sobre test — sólo referencia académica)")

        # ── Scores agregados benigno/ataque ──────────────────────────────
        print(f"\n   Scores por clase (binario):")
        print(f"     Benigno — mean={scores[y_binary==0].mean():.3f} | "
            f"p95={np.percentile(scores[y_binary==0], 95):.3f}")
        print(f"     Ataque  — mean={scores[y_binary==1].mean():.3f} | "
            f"p95={np.percentile(scores[y_binary==1], 95):.3f}")

        # ── Scores por clase original — clave para diagnosticar recon/malware
        unique_classes = np.unique(y_test)
        if len(unique_classes) > 2:
            print(f"\n   Scores por clase original (diagnóstico):")
            print(f"   {'Clase':>5} | {'n':>7} | {'mean':>7} | {'p50':>7} | "
                f"{'p95':>7} | {'detected%':>10} | {'label'}")
            print(f"   {'-'*5}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}-+-{'-'*10}-+-{'-'*15}")
            for cls in unique_classes:
                mask       = (y_test == cls)
                cls_scores = scores[mask]
                detected   = (cls_scores > self.detector.threshold).sum()
                det_pct    = detected / mask.sum() * 100
                label      = "Benigno" if cls == 0 else f"Ataque-{cls}"
                print(f"   {cls:>5} | {mask.sum():>7,} | "
                    f"{cls_scores.mean():>7.3f} | "
                    f"{np.percentile(cls_scores, 50):>7.3f} | "
                    f"{np.percentile(cls_scores, 95):>7.3f} | "
                    f"{det_pct:>9.1f}% | {label}")

        # ── Reporte con threshold óptimo ─────────────────────────────────
        preds_opt = (scores > best_thr).astype(int)
        print(f"\n   Classification report (threshold óptimo={best_thr:.4f}):")
        print(classification_report(
            y_binary, preds_opt,
            target_names=['Benigno', 'Anomalía'], digits=4
        ))

        # ── Confusion matrix robusta ──────────────────────────────────────
        cm = confusion_matrix(y_binary, preds_opt, labels=[0, 1])
        tn_opt, fp_opt, fn_opt, tp_opt = cm.ravel()
        fpr_opt = fp_opt / (fp_opt + tn_opt + 1e-8)
        fnr_opt = fn_opt / (fn_opt + tp_opt + 1e-8)
        print(f"   FPR (threshold óptimo): {fpr_opt:.4f}  "
            f"FNR: {fnr_opt:.4f}")

        # Con threshold de train también
        cm_train = confusion_matrix(y_binary, preds, labels=[0, 1])
        tn_tr, fp_tr, fn_tr, tp_tr = cm_train.ravel()
        fpr_tr = fp_tr / (fp_tr + tn_tr + 1e-8)
        fnr_tr = fn_tr / (fn_tr + tp_tr + 1e-8)
        print(f"   FPR (threshold train):  {fpr_tr:.4f}  "
            f"FNR: {fnr_tr:.4f}")

        # ── Referencia: MSE escalar sin Mahalanobis ───────────────────────
        recon_scalar     = self._reconstruction_error_scalar(X_test)
        auc_recon_scalar = roc_auc_score(y_binary, recon_scalar)
        print(f"\n   AUC Recon escalar (sin Mahal): {auc_recon_scalar:.4f}")
        print(f"   Δ AUC (Mahal - escalar):       {auc - auc_recon_scalar:+.4f}")

        return {
            'mode'               : self.mode,
            'auc'                : auc,
            'f1_train_threshold' : f1,
            'f1_optimal'         : f1_curve[best_idx],
            'threshold_train'    : self.detector.threshold,
            'threshold_optimal'  : best_thr,      # sólo referencia académica
            'fpr_train'          : fpr_tr,
            'fnr_train'          : fnr_tr,
            'fpr_optimal'        : fpr_opt,
            'fnr_optimal'        : fnr_opt,
            'auc_recon_scalar'   : auc_recon_scalar,
            'scores'             : scores,
        }

    def evaluate_adversarial_interception(
        self, X_adv: np.ndarray, label: str = "Adversarial"
    ) -> dict:
        """Evalúa intercepción de ejemplos adversarios (Fase 2 → Fase 3)."""
        print(f"\n[VAE|{self.mode}] Intercepción de {label}...")
        repr_adv = self._get_representation(X_adv)
        scores   = self.detector.score(repr_adv)
        preds    = (scores > self.detector.threshold).astype(int)

        intercepted = preds.sum()
        rate        = intercepted / len(preds) * 100
        print(f"   Total: {len(preds):,} | "
              f"Interceptados: {intercepted:,} ({rate:.1f}%) | "
              f"No interceptados: {len(preds)-intercepted:,} ({100-rate:.1f}%)")

        return {
            'total': len(preds), 'intercepted': int(intercepted),
            'rate': rate, 'scores': scores,
        }

    # ------------------------------------------------------------------
    # INFERENCIA PÚBLICA
    # ------------------------------------------------------------------
    def anomaly_score(self, X: np.ndarray) -> np.ndarray:
        """Score de anomalía. Mayor = más anómalo."""
        return self.detector.score(self._get_representation(X))

    def predict(self, X: np.ndarray) -> np.ndarray:
        """1 = anomalía/Zero-Day, 0 = benigno."""
        return self.detector.predict(self._get_representation(X))

    # ------------------------------------------------------------------
    # PERSISTENCIA
    # ------------------------------------------------------------------
    def save(self, exact_vae_name: str = None, exact_mahal_name: str = None, suffix: str = ""):
        """Guarda el modelo. Permite nombres exactos o usa el sufijo por defecto."""
        s = f"_{suffix}" if suffix else f"_{self.mode}"
        
        name_vae   = exact_vae_name   or f"vae{s}.pt"
        name_mahal = exact_mahal_name or f"mahalanobis{s}.pkl"
        
        torch.save(self.vae.state_dict(), os.path.join(self.models_path, name_vae))
        joblib.dump(self.detector, os.path.join(self.models_path, name_mahal))
        print(f"   [✓] Guardado en {self.models_path}: {name_vae} + {name_mahal}")

    def load(self, exact_vae_name: str = None, exact_mahal_name: str = None, suffix: str = ""):
        """Carga el modelo. Permite nombres exactos para evitar errores de sufijos."""
        s = f"_{suffix}" if suffix else f"_{self.mode}"
        
        name_vae   = exact_vae_name   or f"vae{s}.pt"
        name_mahal = exact_mahal_name or f"mahalanobis{s}.pkl"
        
        ruta_vae   = os.path.join(self.models_path, name_vae)
        ruta_mahal = os.path.join(self.models_path, name_mahal)

        if not os.path.exists(ruta_vae) or not os.path.exists(ruta_mahal):
            raise FileNotFoundError(f"No se encontraron los archivos:\n- {ruta_vae}\n- {ruta_mahal}")

        # weights_only=False por si tu PyTorch es muy reciente y lanza warnings
        self.vae.load_state_dict(torch.load(ruta_vae, map_location=self.device, weights_only=False))
        self.detector = joblib.load(ruta_mahal)
        self.detector.fitted = True
        print(f"   [✓] Cargado desde {self.models_path}: {name_vae} + {name_mahal}")

    # ------------------------------------------------------------------
    # PRIVADOS
    # ------------------------------------------------------------------
    def _reconstruction_error_scalar(self, X: np.ndarray) -> np.ndarray:
        """MSE escalar por muestra — útil como referencia sin Mahalanobis."""
        self.vae.eval()
        loader = DataLoader(TensorDataset(torch.FloatTensor(X)),
                            batch_size=self.cfg.BATCH_SIZE * 4, shuffle=False)
        errors = []
        with torch.no_grad():
            for (xb,) in loader:
                xb = xb.to(self.device)
                x_recon, _, _, _ = self.vae(xb)
                errors.append(F.mse_loss(x_recon, xb, reduction='none')
                               .mean(dim=1).cpu().numpy())
        return np.concatenate(errors, axis=0)


# ===========================================================================
# ABLACIÓN — compara las tres variantes sobre el mismo VAE entrenado
# ===========================================================================

def run_ablation(
    X_train_benign : np.ndarray,
    X_val_benign   : np.ndarray,
    X_test         : np.ndarray,
    y_test         : np.ndarray,
    config         : VAEConfig = None,
    models_path    : str = 'outputs/models',
) -> dict:
    """
    Entrena UN solo VAE y evalúa las tres variantes de Mahalanobis.

    Esto garantiza que la comparación es justa: mismo VAE, mismos pesos,
    solo varía el espacio de representación usado para el detector.

    Devuelve un dict con las métricas de cada modo + tabla resumen.
    """
    print(f"\n{'='*60}")
    print("ABLACIÓN — VAE + MAHALANOBIS (3 variantes)")
    print(f"{'='*60}")
    print("Hipótesis: z+recon ≥ recon > z  (en datos con Recon mal separado)")

    cfg = config or VAEConfig()

    # 1. Entrenar el VAE en modo 'z+recon' (necesita acceso a ambas ramas)
    #    El VAE es idéntico para los tres modos — solo cambia _get_representation
    detector_base = VAEAnomalyDetector(
        input_dim   = X_train_benign.shape[1],
        config      = cfg,
        mode        = 'z+recon',     # el VAE se entrena igual para los tres
        models_path = models_path,
    )
    detector_base.fit(X_train_benign, X_val_benign)

    all_results = {}

    # 2. Evaluar las tres variantes reutilizando el VAE entrenado
    for mode in ('z', 'recon', 'z+recon'):
        print(f"\n{'─'*40}")
        print(f"  Evaluando modo: {mode}")
        print(f"{'─'*40}")

        # Crear detector con mismo VAE pero modo diferente
        d = VAEAnomalyDetector(
            input_dim   = X_train_benign.shape[1],
            config      = cfg,
            mode        = mode,
            models_path = models_path,
        )
        # Reutilizar pesos del VAE ya entrenado
        d.vae.load_state_dict(detector_base.vae.state_dict())
        d.vae.eval()

        # Ajustar Mahalanobis con este modo específico
        print(f"\n[-] Ajustando Mahalanobis [{mode}] sobre benignos de train...")
        repr_benign = d._get_representation(X_train_benign)
        d.detector.fit(repr_benign, cfg.THRESHOLD_PCT)

        # Evaluar
        results = d.evaluate(X_test, y_test)
        all_results[mode] = results

    # 3. Tabla resumen comparativa
    _print_ablation_summary(all_results)

    return all_results


def _print_ablation_summary(results: dict):
    """Imprime tabla resumen de la ablación."""
    print(f"\n{'='*60}")
    print("RESUMEN ABLACIÓN")
    print(f"{'='*60}")
    print(f"  {'Modo':<12} | {'AUC-ROC':>8} | {'F1 óptimo':>10} | "
          f"{'F1 train-thr':>13} | {'AUC Recon(ref)':>14}")
    print(f"  {'-'*12}-+-{'-'*8}-+-{'-'*10}-+-{'-'*13}-+-{'-'*14}")

    best_auc = max(r['auc'] for r in results.values())
    best_f1  = max(r['f1_optimal'] for r in results.values())

    for mode, r in results.items():
        auc_mark = " ◄" if r['auc']        == best_auc else ""
        f1_mark  = " ◄" if r['f1_optimal'] == best_f1  else ""
        print(f"  {mode:<12} | {r['auc']:>8.4f}{auc_mark:<3}| "
              f"{r['f1_optimal']:>10.4f}{f1_mark:<3}| "
              f"{r['f1_train_threshold']:>13.4f} | "
              f"{r['auc_recon_scalar']:>14.4f}")

    print(f"\n  ◄ = mejor valor en esa métrica")
    print(f"\n  Hipótesis confirmada: "
          + ("SÍ" if results.get('z+recon', {}).get('auc', 0) >=
             max(results.get('z', {}).get('auc', 0),
                 results.get('recon', {}).get('auc', 0))
             else "NO — resultado interesante para la memoria"))


# ===========================================================================
# SCRIPT PRINCIPAL
# ===========================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="VAE + Mahalanobis — Ablación 3 variantes BigFlow-NIDS"
    )
    parser.add_argument("--data-path",    default="outputs/data")
    parser.add_argument("--models-path",  default="outputs/models")
    parser.add_argument("--mode",         default="ablation",
                        choices=["z", "recon", "z+recon", "ablation"])
    parser.add_argument("--latent-dim",   type=int,   default=32)
    parser.add_argument("--epochs",       type=int,   default=100)
    parser.add_argument("--beta",         type=float, default=0.1)
    parser.add_argument("--threshold",    type=float, default=95.0)
    args = parser.parse_args()

    cfg = VAEConfig()
    cfg.LATENT_DIM    = args.latent_dim
    cfg.EPOCHS        = args.epochs
    cfg.BETA          = args.beta
    cfg.THRESHOLD_PCT = args.threshold

    print(f"\n{'='*60}")
    print("VAE ANOMALY DETECTOR v2 — BigFlow-NIDS")
    print(f"  mode={args.mode} | latent_dim={cfg.LATENT_DIM} | "
          f"β={cfg.BETA} | threshold=p{cfg.THRESHOLD_PCT:.0f}")
    print(f"{'='*60}")

    # Cargar datos
    p = args.data_path
    X_train_benign = np.load(os.path.join(p, "X_train_benign.npy"))
    X_val          = np.load(os.path.join(p, "X_val.npy"))
    y_val          = np.load(os.path.join(p, "y_val.npy"))
    X_test         = np.load(os.path.join(p, "X_test.npy"))
    y_test         = np.load(os.path.join(p, "y_test.npy"))
    X_val_benign   = X_val[y_val == 0]

    print(f"  X_train_benign: {X_train_benign.shape}")
    print(f"  X_val_benign:   {X_val_benign.shape}")
    print(f"  X_test:         {X_test.shape}")

    if args.mode == "ablation":
        # Comparación completa de las tres variantes
        results = run_ablation(
            X_train_benign, X_val_benign,
            X_test, y_test,
            config=cfg, models_path=args.models_path,
        )
    else:
        # Modo individual
        detector = VAEAnomalyDetector(
            input_dim   = X_train_benign.shape[1],
            config      = cfg,
            mode        = args.mode,
            models_path = args.models_path,
        )
        detector.fit(X_train_benign, X_val_benign)
        results = detector.evaluate(X_test, y_test)
        detector.save()

        print(f"\n{'='*60}")
        print(f"  AUC-ROC:    {results['auc']:.4f}")
        print(f"  F1 óptimo:  {results['f1_optimal']:.4f}")
        print(f"{'='*60}")