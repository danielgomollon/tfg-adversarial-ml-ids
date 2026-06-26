"""
vae_anomaly_detector.py
===========================================================================
VAE (Variational Autoencoder) + Distancia de Mahalanobis para detección
de anomalías y Zero-Days en BigFlow-NIDS.

Arquitectura:
  - Encoder: MLP con BatchNorm + residual connections → μ, log_σ²
  - Decoder: MLP simétrico con BatchNorm → reconstrucción
  - Espacio latente z: dimensión configurable (default=32)
  - Detector: Distancia de Mahalanobis sobre z de benignos de train

Principio de funcionamiento:
  El VAE se entrena EXCLUSIVAMENTE con tráfico benigno. Aprende a
  comprimir y reconstruir la distribución normal del tráfico de red.
  Ante un flujo anómalo (ataque, Zero-Day, ejemplo adversario), el
  error de reconstrucción y/o la distancia de Mahalanobis en el
  espacio latente z se disparan — señal directa de anomalía.

Detección en inferencia (sistema híbrido):
  1. VAE calcula distancia de Mahalanobis del flujo en espacio z
  2. Si dist > threshold → ALERTA (Zero-Day / anomalía)
  3. Si dist ≤ threshold → pasar a ResNet para clasificación fina

Uso:
    # Entrenamiento
    detector = VAEAnomalyDetector(input_dim=66)
    detector.fit(X_train_benign, X_val_benign)

    # Evaluación
    results = detector.evaluate(X_test, y_test)

    # Inferencia
    scores = detector.anomaly_score(X_new)
    alerts = detector.predict(X_new)

Referencias:
  - Kingma & Welling (2013) — Auto-Encoding Variational Bayes
  - Lee et al. (2018) — Mahalanobis distance for OOD detection
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve
from sklearn.metrics import classification_report
import joblib
from typing import Optional, Tuple
import time


# ===========================================================================
# CONFIGURACIÓN
# ===========================================================================

class VAEConfig:
    """Hiperparámetros del VAE — todos configurables en un solo lugar."""

    # Arquitectura
    LATENT_DIM     = 32       # dimensión espacio latente z
    HIDDEN_DIMS    = [256, 128, 64]  # capas encoder (decoder es simétrico)
    #HIDDEN_DIMS    = [512, 256, 128]  # hemos probado, pero es mismo resultado que 256-128-64 
    DROPOUT        = 0.1      # dropout en encoder/decoder

    # Entrenamiento
    BATCH_SIZE     = 2048     # batch grande → estabilidad en datos tabulares
    EPOCHS         = 100      # max epochs (early stopping activo)
    LR             = 1e-3     # learning rate inicial
    WEIGHT_DECAY   = 1e-5     # L2 regularization
    PATIENCE       = 15       # early stopping patience
    LR_PATIENCE    = 7        # reducir LR si no mejora
    LR_FACTOR      = 0.5      # factor reducción LR
    BETA           = 0.1      # peso KL divergence (β-VAE: >0.1 → más disentangled)

    # Mahalanobis
    N_COMPONENTS   = None     # None = usar todos los dims latentes
    THRESHOLD_PCT  = 95.0     # percentil para threshold de anomalía

    # Sistema
    SEED           = 42
    DEVICE         = 'cuda' if torch.cuda.is_available() else 'cpu'


# ===========================================================================
# ARQUITECTURA VAE
# ===========================================================================

class ResidualBlock(nn.Module):
    """
    Bloque residual para el encoder/decoder.
    Mejora el gradiente flow y estabiliza el entrenamiento.
    """
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


class Encoder(nn.Module):
    """
    Encoder q(z|x): x → μ, log_σ²
    
    Arquitectura:
      input → [Linear → BN → GELU → Residual] × n_layers → μ, log_σ²
    """
    def __init__(self, input_dim: int, hidden_dims: list, latent_dim: int,
                 dropout: float = 0.1):
        super().__init__()

        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers += [
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                ResidualBlock(h_dim, dropout),
            ]
            in_dim = h_dim

        self.net    = nn.Sequential(*layers)
        self.fc_mu  = nn.Linear(in_dim, latent_dim)
        self.fc_var = nn.Linear(in_dim, latent_dim)

        # inicialización de pesos — importante para estabilidad del VAE
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h      = self.net(x)
        mu     = self.fc_mu(h)
        log_var = self.fc_var(h)
        # clamping log_var para estabilidad numérica
        log_var = torch.clamp(log_var, -10.0, 4.0)
        return mu, log_var


class Decoder(nn.Module):
    """
    Decoder p(x|z): z → x_reconstruido

    Arquitectura simétrica al encoder.
    Sin activación final — la loss usa MSE sobre valores escalados.
    """
    def __init__(self, latent_dim: int, hidden_dims: list, output_dim: int,
                 dropout: float = 0.1):
        super().__init__()

        layers = []
        in_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            layers += [
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                ResidualBlock(h_dim, dropout),
            ]
            in_dim = h_dim

        self.net     = nn.Sequential(*layers)
        self.fc_out  = nn.Linear(in_dim, output_dim)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.net(z)
        return self.fc_out(h)


class VAE(nn.Module):
    """
    Variational Autoencoder para detección de anomalías en tráfico de red.

    El espacio latente z ~ N(μ, σ²) captura la distribución del tráfico
    benigno. Los flujos anómalos producen z fuera de esa distribución,
    detectable mediante distancia de Mahalanobis.
    """
    def __init__(self, input_dim: int, config: VAEConfig = None):
        super().__init__()
        cfg = config or VAEConfig()

        self.input_dim  = input_dim
        self.latent_dim = cfg.LATENT_DIM
        self.beta       = cfg.BETA

        self.encoder = Encoder(input_dim, cfg.HIDDEN_DIMS, cfg.LATENT_DIM,
                                cfg.DROPOUT)
        self.decoder = Decoder(cfg.LATENT_DIM, cfg.HIDDEN_DIMS, input_dim,
                                cfg.DROPOUT)

    def reparameterize(self, mu: torch.Tensor,
                       log_var: torch.Tensor) -> torch.Tensor:
        """
        Truco de reparametrización: z = μ + ε·σ, ε ~ N(0,1)
        Permite backprop a través del muestreo estocástico.
        """
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            # en inferencia usar la media directamente (más estable)
            return mu

    def forward(self, x: torch.Tensor):
        mu, log_var = self.encoder(x)
        z           = self.reparameterize(mu, log_var)
        x_recon     = self.decoder(z)
        return x_recon, mu, log_var, z

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Obtener μ, log_σ² sin reconstrucción."""
        return self.encoder(x)

    def get_latent(self, x: torch.Tensor) -> torch.Tensor:
        """Obtener z (media) para un batch — usado en Mahalanobis."""
        self.eval()
        with torch.no_grad():
            mu, _ = self.encoder(x)
        return mu

    @staticmethod
    def loss(x: torch.Tensor, x_recon: torch.Tensor,
             mu: torch.Tensor, log_var: torch.Tensor,
             beta: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        ELBO loss = Reconstrucción + β·KL

        Reconstrucción: MSE — apropiado para features continuas escaladas
        KL: -0.5 · Σ(1 + log_σ² - μ² - σ²)
        """
        recon_loss = F.mse_loss(x_recon, x, reduction='mean')
        kl_loss    = -0.5 * torch.mean(
            1 + log_var - mu.pow(2) - log_var.exp()
        )
        total_loss = recon_loss + beta * kl_loss
        return total_loss, recon_loss, kl_loss


# ===========================================================================
# DETECTOR DE MAHALANOBIS
# ===========================================================================

class MahalanobisDetector:
    """
    Detector de anomalías basado en distancia de Mahalanobis en el
    espacio latente z del VAE.

    Ventaja sobre error de reconstrucción:
      La distancia de Mahalanobis tiene en cuenta la covarianza del
      espacio latente — es invariante a la escala y correlación de
      las dimensiones latentes. Más robusto que MSE de reconstrucción.

    Cálculo:
      d(z) = sqrt((z - μ_train)ᵀ · Σ⁻¹ · (z - μ_train))
    """

    def __init__(self):
        self.mu_train    : Optional[np.ndarray] = None  # media z benignos
        self.cov_inv     : Optional[np.ndarray] = None  # Σ⁻¹ covarianza
        self.threshold   : Optional[float]      = None  # umbral anomalía
        self.fitted      : bool                 = False

    def fit(self, z_benign: np.ndarray, threshold_pct: float = 95.0):
        """
        Ajusta el detector con embeddings z de tráfico benigno.

        Parámetros
        ----------
        z_benign : array (n_samples, latent_dim) — embeddings benignos
        threshold_pct : percentil para el umbral de anomalía
        """
        self.mu_train = z_benign.mean(axis=0)
        cov           = np.cov(z_benign.T)

        # regularización para invertibilidad (evitar singular matrix)
        cov += np.eye(cov.shape[0]) * 1e-6
        self.cov_inv  = np.linalg.inv(cov)

        # calcular distancias sobre benignos para fijar threshold
        dists         = self._mahalanobis_batch(z_benign)
        self.threshold = np.percentile(dists, threshold_pct)
        self.fitted    = True

        print(f"   [Mahalanobis] μ_dist_benigno={dists.mean():.3f} | "
              f"σ={dists.std():.3f} | "
              f"threshold(p{threshold_pct:.0f})={self.threshold:.3f}")

    def score(self, z: np.ndarray) -> np.ndarray:
        """Calcular distancias de Mahalanobis para un batch de embeddings."""
        if not self.fitted:
            raise RuntimeError("MahalanobisDetector no ajustado — llama fit() primero")
        return self._mahalanobis_batch(z)

    def predict(self, z: np.ndarray) -> np.ndarray:
        """1 = anomalía, 0 = benigno."""
        return (self.score(z) > self.threshold).astype(int)

    def _mahalanobis_batch(self, z: np.ndarray) -> np.ndarray:
        """Distancia de Mahalanobis vectorizada sobre batch."""
        diff  = z - self.mu_train          # (n, d)
        left  = diff @ self.cov_inv        # (n, d)
        dists = np.sqrt(np.maximum(        # clamp negatives por float errors
            (left * diff).sum(axis=1), 0.0
        ))
        return dists


# ===========================================================================
# DETECTOR COMPLETO VAE + MAHALANOBIS
# ===========================================================================

class VAEAnomalyDetector:
    """
    Sistema completo de detección de anomalías: VAE + Mahalanobis.

    Integra entrenamiento, evaluación e inferencia en una única clase
    reutilizable. Diseñado para acoplarse con la TabularResNet en el
    sistema híbrido NIDS.

    Pipeline de inferencia:
      flujo → VAE.encode() → z → Mahalanobis.score() → alerta/normal
    """

    def __init__(self, input_dim: int, config: VAEConfig = None,
                 models_path: str = 'outputs/models'):
        self.cfg         = config or VAEConfig()
        self.input_dim   = input_dim
        self.models_path = models_path
        self.device      = torch.device(self.cfg.DEVICE)

        os.makedirs(models_path, exist_ok=True)

        torch.manual_seed(self.cfg.SEED)
        np.random.seed(self.cfg.SEED)

        self.vae       = VAE(input_dim, self.cfg).to(self.device)
        self.detector  = MahalanobisDetector()

        print(f"   [VAE] Device: {self.device}")
        print(f"   [VAE] Parámetros: "
              f"{sum(p.numel() for p in self.vae.parameters()):,}")

    # ------------------------------------------------------------------
    # ENTRENAMIENTO
    # ------------------------------------------------------------------
    def fit(self, X_train_benign: np.ndarray,
            X_val_benign: Optional[np.ndarray] = None):
        """
        Entrena el VAE exclusivamente con tráfico benigno.

        Parámetros
        ----------
        X_train_benign : array (n, features) — solo flujos benignos de train
        X_val_benign   : array (n, features) — benignos de val para early stopping
        """
        print("\n" + "="*60)
        print("VAE — ENTRENAMIENTO")
        print("="*60)
        print(f"   Train benign: {X_train_benign.shape}")
        if X_val_benign is not None:
            print(f"   Val benign:   {X_val_benign.shape}")

        # DataLoaders
        train_ds = TensorDataset(
            torch.FloatTensor(X_train_benign).to(self.device)
        )
        train_loader = DataLoader(
            train_ds, batch_size=self.cfg.BATCH_SIZE,
            shuffle=True, drop_last=True
        )

        val_loader = None
        if X_val_benign is not None:
            val_ds = TensorDataset(
                torch.FloatTensor(X_val_benign).to(self.device)
            )
            val_loader = DataLoader(
                val_ds, batch_size=self.cfg.BATCH_SIZE * 2,
                shuffle=False
            )

        # Optimizador
        optimizer = torch.optim.AdamW(
            self.vae.parameters(),
            lr=self.cfg.LR,
            weight_decay=self.cfg.WEIGHT_DECAY,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=self.cfg.LR_FACTOR,
            patience=self.cfg.LR_PATIENCE,
        )

        best_val_loss  = float('inf')
        patience_count = 0
        best_state     = None

        print(f"\n{'Epoch':>6} | {'T.Loss':>8} | {'T.Recon':>8} | "
              f"{'T.KL':>7} | {'V.Loss':>8} | {'LR':>10}")
        print("-" * 60)

        for epoch in range(1, self.cfg.EPOCHS + 1):
            t0 = time.time()

            # --- Train ---
            self.vae.train()
            t_loss = t_recon = t_kl = 0.0
            for (x_batch,) in train_loader:
                optimizer.zero_grad()
                x_recon, mu, log_var, _ = self.vae(x_batch)
                loss, recon, kl = VAE.loss(
                    x_batch, x_recon, mu, log_var, self.cfg.BETA
                )
                loss.backward()
                # gradient clipping para estabilidad
                nn.utils.clip_grad_norm_(self.vae.parameters(), 1.0)
                optimizer.step()

                t_loss  += loss.item()
                t_recon += recon.item()
                t_kl    += kl.item()

            n_batches = len(train_loader)
            t_loss  /= n_batches
            t_recon /= n_batches
            t_kl    /= n_batches

            # --- Val ---
            v_loss = float('nan')
            if val_loader is not None:
                self.vae.eval()
                v_loss = 0.0
                with torch.no_grad():
                    for (x_batch,) in val_loader:
                        x_recon, mu, log_var, _ = self.vae(x_batch)
                        loss, _, _ = VAE.loss(
                            x_batch, x_recon, mu, log_var, self.cfg.BETA
                        )
                        v_loss += loss.item()
                v_loss /= len(val_loader)
                scheduler.step(v_loss)

                # Early stopping
                if v_loss < best_val_loss - 1e-5:
                    best_val_loss  = v_loss
                    patience_count = 0
                    best_state     = {
                        k: v.cpu().clone()
                        for k, v in self.vae.state_dict().items()
                    }
                else:
                    patience_count += 1

            lr_now = optimizer.param_groups[0]['lr']
            dt     = time.time() - t0

            print(f"{epoch:>6} | {t_loss:>8.4f} | {t_recon:>8.4f} | "
                  f"{t_kl:>7.4f} | {v_loss:>8.4f} | {lr_now:>10.2e}  "
                  f"[{dt:.0f}s]"
                  + (" ✓" if patience_count == 0 and val_loader else ""))

            if patience_count >= self.cfg.PATIENCE:
                print(f"\n   [EarlyStopping] epoch {epoch}")
                break

        # Restaurar mejor estado
        if best_state is not None:
            self.vae.load_state_dict(best_state)
            print(f"   [✓] Mejor modelo restaurado (val_loss={best_val_loss:.4f})")

        # Ajustar detector de Mahalanobis sobre vector de errores de reconstrucción por muestra
        print("\n[-] Ajustando detector de Mahalanobis sobre vector de errores...")
        recon_errors_benign = self._recon_error_vector(X_train_benign)
        self.detector.fit(recon_errors_benign, self.cfg.THRESHOLD_PCT)

        # esto es sobre espacio latente z de benignos del train
        #z_benign = self._get_latents(X_train_benign)
        #self.detector.fit(z_benign, self.cfg.THRESHOLD_PCT)

        print("\n[✓] VAE entrenado y detector ajustado")

    def fit_mahalanobis_on_recon(self, X_train_benign: np.ndarray):
        """
        Ajusta Mahalanobis sobre el vector de errores de reconstrucción
        en lugar de sobre z. Cada dimensión del vector es el error MSE
        feature por feature — benignos tendrán errores ~0 en todas,
        ataques dispararán alguna dimensión específica.
        """
        self.vae.eval()
        loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_train_benign)),
            batch_size=self.cfg.BATCH_SIZE * 4,
            shuffle=False,
        )
        recon_errors = []
        with torch.no_grad():
            for (x_batch,) in loader:
                x_batch = x_batch.to(self.device)
                x_recon, _, _, _ = self.vae(x_batch)
                # error por feature, no agregado — shape (n, 66)
                err = (x_recon - x_batch).pow(2)
                recon_errors.append(err.cpu().numpy())
        
        recon_errors = np.concatenate(recon_errors, axis=0)
        self.detector.fit(recon_errors, self.cfg.THRESHOLD_PCT)
        print(f"   [Mahalanobis sobre Recon] ajustado sobre {recon_errors.shape}")

    # ------------------------------------------------------------------
    # EVALUACIÓN
    # ------------------------------------------------------------------
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray,
                 attack_label: int = 1) -> dict:
        """
        Evalúa el detector sobre el test set completo.

        y_test: 0 = benigno, cualquier otro valor = ataque
        attack_label: no usado directamente, la detección es binaria
        """
        print("\n" + "="*60)
        print("VAE — EVALUACIÓN")
        print("="*60)

        # scores de anomalía
        recon_errors_test = self._recon_error_vector(X_test)
        scores = self.detector.score(recon_errors_test)
        
        #z_test   = self._get_latents(X_test)
        #scores   = self.detector.score(z_test)

        # también calcular error de reconstrucción
        recon_errors = self._reconstruction_error(X_test)

        # etiquetas binarias: 0=benigno, 1=ataque
        y_binary = (y_test != 0).astype(int)

        # threshold óptimo por F1 en val (aquí usamos el de train)
        preds    = (scores > self.detector.threshold).astype(int)

        # métricas
        auc      = roc_auc_score(y_binary, scores)
        f1       = f1_score(y_binary, preds, zero_division=0)

        print(f"\n   AUC-ROC (Mahalanobis):     {auc:.4f}")
        print(f"   F1 binario (ataque/benigno): {f1:.4f}")
        print(f"   Threshold:                  {self.detector.threshold:.3f}")

        # distribución de scores por clase
        print(f"\n   Scores Mahalanobis:")
        print(f"     Benigno  — mean={scores[y_binary==0].mean():.3f} | "
              f"p95={np.percentile(scores[y_binary==0], 95):.3f}")
        print(f"     Ataque   — mean={scores[y_binary==1].mean():.3f} | "
              f"p95={np.percentile(scores[y_binary==1], 95):.3f}")

        # curva precision-recall para encontrar threshold óptimo
        prec, rec, thrs = precision_recall_curve(y_binary, scores)
        f1_curve = 2 * prec * rec / (prec + rec + 1e-8)
        best_idx = f1_curve.argmax()
        best_thr = thrs[best_idx] if best_idx < len(thrs) else self.detector.threshold

        print(f"\n   Threshold óptimo F1:        {best_thr:.3f}")
        print(f"   F1 con threshold óptimo:    {f1_curve[best_idx]:.4f}")

        # classification report con threshold óptimo
        preds_opt = (scores > best_thr).astype(int)
        print("\n" + classification_report(
            y_binary, preds_opt,
            target_names=['Benigno', 'Anomalía'],
            digits=4
        ))

        # Error de reconstrucción como métrica complementaria
        auc_recon = roc_auc_score(y_binary, recon_errors)
        print(f"   AUC-ROC (Recon Error):      {auc_recon:.4f}")

        return {
            'auc_mahalanobis' : auc,
            'auc_recon'       : auc_recon,
            'f1_binary'       : f1,
            'f1_optimal'      : f1_curve[best_idx],
            'threshold_train' : self.detector.threshold,
            'threshold_optimal': best_thr,
            'scores'          : scores,
            'recon_errors'    : recon_errors,
        }

    def evaluate_adversarial_interception(
        self, X_adv: np.ndarray, y_adv: np.ndarray,
        label: str = "Adversarial"
    ) -> dict:
        """
        Evalúa cuántos ejemplos adversarios que engañaron a la ResNet
        son interceptados por el VAE.

        Uso en Fase 2: pasar los ejemplos adversarios que la ResNet
        clasificó incorrectamente y ver si el VAE los detecta.
        """
        print(f"\n[VAE] Evaluando intercepción de ejemplos {label}...")
        #z      = self._get_latents(X_adv)
        #scores = self.detector.score(z)

        error_vector = self._recon_error_vector(X_adv)
        scores = self.detector.score(error_vector)

        preds  = (scores > self.detector.threshold).astype(int)

        intercepted = preds.sum()
        total       = len(preds)
        rate        = intercepted / total * 100

        print(f"   Ejemplos {label}: {total:,}")
        print(f"   Interceptados por VAE: {intercepted:,} ({rate:.1f}%)")
        print(f"   No interceptados:      {total-intercepted:,} ({100-rate:.1f}%)")

        return {
            'total'       : total,
            'intercepted' : int(intercepted),
            'rate'        : rate,
            'scores'      : scores,
        }

    # ------------------------------------------------------------------
    # INFERENCIA
    # ------------------------------------------------------------------
    def anomaly_score(self, X: np.ndarray) -> np.ndarray:
        """Distancia de Mahalanobis para cada flujo. Mayor = más anómalo."""
        #z = self._get_latents(X)
        vector_errores=self._recon_error_vector(X)
        return self.detector.score(vector_errores)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """1 = anomalía/Zero-Day, 0 = benigno."""
        #return self.detector.predict(self._get_latents(X))
        return self.detector.predict(self._recon_error_vector(X))


    # ------------------------------------------------------------------
    # PERSISTENCIA
    # ------------------------------------------------------------------
    def save(self, suffix: str = ""):
        """Guarda VAE y detector de Mahalanobis."""
        s = f"_{suffix}" if suffix else ""
        torch.save(
            self.vae.state_dict(),
            os.path.join(self.models_path, f"vae{s}.pt")
        )
        joblib.dump(
            self.detector,
            os.path.join(self.models_path, f"mahalanobis_detector{s}.pkl")
        )
        print(f"   [✓] VAE guardado: vae{s}.pt + mahalanobis_detector{s}.pkl")

    def load(self, suffix: str = ""):
        """Carga VAE y detector desde disco."""
        s = f"_{suffix}" if suffix else ""
        self.vae.load_state_dict(
            torch.load(
                os.path.join(self.models_path, f"vae{s}.pt"),
                map_location=self.device,
            )
        )
        self.detector = joblib.load(
            os.path.join(self.models_path, f"mahalanobis_detector{s}.pkl")
        )
        self.detector.fitted = True
        print(f"   [✓] VAE cargado: vae{s}.pt + mahalanobis_detector{s}.pkl")

    # ------------------------------------------------------------------
    # MÉTODOS PRIVADOS
    # ------------------------------------------------------------------
    def _get_latents(self, X: np.ndarray) -> np.ndarray:
        """Obtener embeddings z para un array numpy."""
        self.vae.eval()
        loader = DataLoader(
            TensorDataset(torch.FloatTensor(X)),
            batch_size=self.cfg.BATCH_SIZE * 4,
            shuffle=False,
        )
        zs = []
        with torch.no_grad():
            for (x_batch,) in loader:
                x_batch = x_batch.to(self.device)
                mu, _   = self.vae.encode(x_batch)
                zs.append(mu.cpu().numpy())
        return np.concatenate(zs, axis=0)

    def _reconstruction_error(self, X: np.ndarray) -> np.ndarray:
        """Error de reconstrucción MSE por muestra."""
        self.vae.eval()
        loader = DataLoader(
            TensorDataset(torch.FloatTensor(X)),
            batch_size=self.cfg.BATCH_SIZE * 4,
            shuffle=False,
        )
        errors = []
        with torch.no_grad():
            for (x_batch,) in loader:
                x_batch = x_batch.to(self.device)
                x_recon, _, _, _ = self.vae(x_batch)
                err = F.mse_loss(x_recon, x_batch, reduction='none')
                errors.append(err.mean(dim=1).cpu().numpy())
        return np.concatenate(errors, axis=0)

    def _recon_error_vector(self, X: np.ndarray) -> np.ndarray:
        """Error de reconstrucción por feature — shape (n, input_dim)."""
        self.vae.eval()
        loader = DataLoader(
            TensorDataset(torch.FloatTensor(X)),
            batch_size=self.cfg.BATCH_SIZE * 4,
            shuffle=False,
        )
        errors = []
        with torch.no_grad():
            for (x_batch,) in loader:
                x_batch = x_batch.to(self.device)
                x_recon, _, _, _ = self.vae(x_batch)
                err = (x_recon - x_batch).pow(2)
                errors.append(err.cpu().numpy())
        return np.concatenate(errors, axis=0)

# ===========================================================================
# SCRIPT PRINCIPAL
# ===========================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="VAE + Mahalanobis — Detección de anomalías BigFlow-NIDS"
    )
    parser.add_argument("--data-path",   default="outputs/data")
    parser.add_argument("--models-path", default="outputs/models")
    parser.add_argument("--latent-dim",  type=int,   default=32)
    parser.add_argument("--epochs",      type=int,   default=100)
    parser.add_argument("--beta",        type=float, default=1.0)
    parser.add_argument("--threshold",   type=float, default=95.0)
    args = parser.parse_args()

    # Configuración
    cfg = VAEConfig()
    cfg.LATENT_DIM   = args.latent_dim
    cfg.EPOCHS       = args.epochs
    cfg.BETA         = args.beta
    cfg.THRESHOLD_PCT = args.threshold

    print(f"\n{'='*60}")
    print(f"VAE ANOMALY DETECTOR — BigFlow-NIDS")
    print(f"{'='*60}")
    print(f"  Device:      {cfg.DEVICE}")
    print(f"  Latent dim:  {cfg.LATENT_DIM}")
    print(f"  Beta:        {cfg.BETA}")
    print(f"  Threshold:   p{cfg.THRESHOLD_PCT:.0f}")

    # Cargar datos
    print("\n[-] Cargando datos...")
    p = args.data_path
    X_train_benign = np.load(os.path.join(p, "X_train_benign.npy"))
    X_val          = np.load(os.path.join(p, "X_val.npy"))
    y_val          = np.load(os.path.join(p, "y_val.npy"))
    X_test         = np.load(os.path.join(p, "X_test.npy"))
    y_test         = np.load(os.path.join(p, "y_test.npy"))

    # Extraer benignos de val para early stopping del VAE
    X_val_benign = X_val[y_val == 0]
    print(f"  X_train_benign: {X_train_benign.shape}")
    print(f"  X_val_benign:   {X_val_benign.shape}")
    print(f"  X_test:         {X_test.shape}")

    # Entrenar
    detector = VAEAnomalyDetector(
        input_dim   = X_train_benign.shape[1],
        config      = cfg,
        models_path = args.models_path,
    )
    detector.fit(X_train_benign, X_val_benign)

    # Evaluar
    results = detector.evaluate(X_test, y_test)

    # Guardar
    detector.save()

    print(f"\n{'='*60}")
    print("COMPLETADO")
    print(f"  AUC-ROC Mahalanobis: {results['auc_mahalanobis']:.4f}")
    print(f"  AUC-ROC Recon Error: {results['auc_recon']:.4f}")
    print(f"  F1 binario:          {results['f1_binary']:.4f}")
    print(f"  F1 óptimo:           {results['f1_optimal']:.4f}")
    print(f"{'='*60}")