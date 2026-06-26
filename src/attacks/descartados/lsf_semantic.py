"""
src/attacks/lsf.py
================================================================
LSF v2 — Latent Semantic Fission (Semantic Projection)
Contribución 100% original — BigFlow-NIDS TFG (2025-2026)

Daniel Gomollón Embid

═══════════════════════════════════════════════════════════════
PARADIGMA: Fisión Semántica y Trasplante Temporal
═══════════════════════════════════════════════════════════════

LSF v2 destruye la IDENTIDAD estadística del flujo sin reducir
su volumen letal, evitando los artefactos de la fragmentación física.

CAPA 1 — Semantic Projection
  En lugar de partir el flujo a la mitad, se trasplanta el
  "esqueleto temporal" (IAT_AVG, IAT_STDDEV) de un flujo benigno real
  que tenga exactamente el mismo volumen (IN_BYTES) que el ataque.
  El volumen destructivo se mantiene, pero la firma temporal cambia.

CAPA 2 — S3M (Shadow Session Split-Merging)
  Camuflaje semántico: el flujo proyectado se mezcla con su gemelo
  benigno más cercano ponderando las features de contexto (BUF_*).

CAPA 3 — OSS (Omega Surgical Strike)
  Singularidad estadística de precisión: calibra el IAT final inyectando
  un micro-delay para que la varianza residual caiga exactamente en el
  percentil benigno objetivo.

═══════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional

from src.utils.domain_constraints import DomainConstraints, PHYSICAL_BOUNDS

# Features BUF_* conocidas por el atacante (OSINT razonable)
BUF_FEATURES_KNOWN = [
    'BUF_SCAN_RATE', 'BUF_FLOW_COUNT', 'BUF_HTTP_BYTES_AVG',
    'BUF_PORT_STD', 'BUF_NO_RESP_RATIO', 'BUF_SMALL_PKT_RATIO',
    'BUF_UNIQUE_DST_PORTS', 'BUF_UNIQUE_DST_IPS',
    'BUF_HTTP_RATIO', 'BUF_PORT_RANGE',
]

# ===========================================================================
# RESULTADO LSF v2
# ===========================================================================

@dataclass
class LSFResult:
    X_lsf            : np.ndarray   # Flujo final post Proj+S3M+OSS
    X_original       : np.ndarray
    y_true           : np.ndarray

    y_pred_orig      : np.ndarray
    y_pred_proj      : np.ndarray   # Post Capa 1
    y_pred_s3m       : np.ndarray   # Post Capa 2
    y_pred_lsf       : np.ndarray   # Post Capa 3 (Final)

    asr_proj         : float
    asr_s3m          : float
    asr_oss          : float        # ASR Final
    oss_iat_mean     : float
    attack_name      : str = "LSF v2 — Latent Semantic Fission"

    def summary(self) -> str:
        n = len(self.y_true)
        detected = (self.y_pred_orig != 0).sum()
        evaded = (self.y_pred_lsf == 0)[self.y_pred_orig != 0].sum()

        lines = [
            f"\n{'='*65}",
            f"ATAQUE: {self.attack_name}",
            f"{'='*65}",
            f"  Flujos originales detectados : {detected:,}/{n:,}",
            f"  IAT medio OSS inyectado      : {self.oss_iat_mean:.1f} ms",
            f"\n  Ablación por capas (contribución acumulada):",
            f"    Capa 1 — Semantic Proj     : {self.asr_proj*100:.1f}% ASR",
            f"    Capa 2 — Proj + S3M        : {self.asr_s3m*100:.1f}% ASR",
            f"    Capa 3 — Proj + S3M + OSS  : {self.asr_oss*100:.1f}% ASR",
            f"\n  Evasiones totales finales    : {evaded:,} ({self.asr_oss*100:.1f}%)",
        ]
        return "\n".join(lines)

    def evasion_by_class(self, class_names: Optional[dict] = None) -> str:
        lines = ["\n  Evasión por clase (Final LSF v2):"]
        mask_detected = self.y_pred_orig != 0
        evaded_mask = (self.y_pred_lsf == 0)

        for cls in np.unique(self.y_true):
            mask = (self.y_true == cls) & mask_detected
            if mask.sum() == 0:
                continue
            evaded = evaded_mask[mask].sum()
            total  = mask.sum()
            rate   = evaded / total * 100
            name   = (class_names.get(int(cls), f"Clase {cls}") if class_names else f"Clase {cls}")
            lines.append(f"    {name:<25} {evaded:>5}/{total:<5} ({rate:.1f}%)")
        return "\n".join(lines)


# ===========================================================================
# MOTOR LSF v2
# ===========================================================================

class LSFAttack:
    def __init__(
        self,
        constraints     : DomainConstraints,
        X_benign_scaled : np.ndarray,
        protocol        : str   = 'http',
        n_twins         : int   = 50,
        oss_target_pct  : int   = 50,
        oss_probe_sizes : list  = [40.0, 52.0, 498.0, 1500.0],
        verbose         : bool  = True,
    ):
        self.dc              = constraints
        self.protocol        = protocol
        self.n_twins         = n_twins
        self.oss_target_pct  = oss_target_pct
        self.oss_probe_sizes = np.array(oss_probe_sizes)
        self.verbose         = verbose

        self.X_benign_phys = self.dc.to_physical_space(X_benign_scaled)
        self.X_benign_sc   = X_benign_scaled

        # Grey-Box S3M weights
        self._BUF_KNOWN = BUF_FEATURES_KNOWN
        self._twin_weights = self._build_twin_weights()

        # Target OSS
        self._oss_sigma_target = self._compute_oss_target(X_benign_scaled, oss_target_pct)

        # Índices críticos
        self._idx_stddev  = self.dc._feat_idx('SRC_TO_DST_IAT_STDDEV')
        self._idx_avg     = self.dc._feat_idx('SRC_TO_DST_IAT_AVG')
        self._idx_pkts    = self.dc._feat_idx('IN_PKTS')
        self._idx_dur     = self.dc._feat_idx('FLOW_DURATION_MILLISECONDS')
        self._idx_max_iat = self.dc._feat_idx('SRC_TO_DST_IAT_MAX')
        self._idx_bytes   = self.dc._feat_idx('IN_BYTES')
        self._idx_max_pkt = self.dc._feat_idx('MAX_IP_PKT_LEN')
        self._idx_l7      = self.dc._feat_idx('L7_PROTO')

        if self.verbose:
            print(f"[LSF v2] Latent Semantic Fission inicializado:")
            print(f"  Protocolo objetivo : {protocol.upper()}")
            print(f"  Pool benigno       : {len(self.X_benign_phys):,} flujos")
            print(f"  OSS target σ       : {self._oss_sigma_target:.1f} ms")

    # ------------------------------------------------------------------
    # PIPELINE PRINCIPAL
    # ------------------------------------------------------------------
    def run(self, X: np.ndarray, y: np.ndarray, model: object, class_names: Optional[dict] = None) -> LSFResult:
        y_pred_orig = self._predict(model, X)
        mask_detected = y_pred_orig != 0

        # CAPA 1: Semantic Projection
        X_proj = self._apply_semantic_projection(X)
        y_pred_proj = self._predict(model, X_proj)
        asr_proj = self._compute_asr(y_pred_proj, mask_detected)

        # CAPA 2: S3M
        X_s3m = self._apply_s3m(X_proj)
        y_pred_s3m = self._predict(model, X_s3m)
        asr_s3m = self._compute_asr(y_pred_s3m, mask_detected)

        # CAPA 3: OSS
        X_oss, oss_iat = self._apply_oss(X_s3m)
        y_pred_lsf = self._predict(model, X_oss)
        asr_oss = self._compute_asr(y_pred_lsf, mask_detected)
        
        oss_iat_mean = float(oss_iat[mask_detected].mean()) if mask_detected.sum() > 0 else 0.0

        result = LSFResult(
            X_lsf=X_oss, X_original=X, y_true=y,
            y_pred_orig=y_pred_orig, y_pred_proj=y_pred_proj, y_pred_s3m=y_pred_s3m, y_pred_lsf=y_pred_lsf,
            asr_proj=asr_proj, asr_s3m=asr_s3m, asr_oss=asr_oss,
            oss_iat_mean=oss_iat_mean
        )

        if self.verbose:
            print(result.summary())
            if class_names:
                print(result.evasion_by_class(class_names))

        return result

    # ------------------------------------------------------------------
    # CAPA 1: SEMANTIC PROJECTION
    # ------------------------------------------------------------------
    def _apply_semantic_projection(self, X_scaled: np.ndarray) -> np.ndarray:
        X_phys = self.dc.to_physical_space(X_scaled)
        X_proj = X_phys.copy()
        
        l7_target = 7.0 if self.protocol == 'http' else 91.0
        
        if self._idx_l7 is not None:
            mask_proto = self.X_benign_phys[:, self._idx_l7] == l7_target
            pool_proto = self.X_benign_phys[mask_proto]
        else:
            pool_proto = self.X_benign_phys
            
        if len(pool_proto) == 0:
            return X_scaled # Fallback

        for i in range(len(X_phys)):
            if self._idx_bytes is not None:
                atk_bytes = X_phys[i, self._idx_bytes]
                diffs = np.abs(pool_proto[:, self._idx_bytes] - atk_bytes)
                best_proto = pool_proto[diffs.argmin()]
                
                # Proyectar solo el esqueleto temporal
                if self._idx_stddev is not None:
                    X_proj[i, self._idx_stddev] = best_proto[self._idx_stddev]
                if self._idx_avg is not None:
                    X_proj[i, self._idx_avg] = best_proto[self._idx_avg]

        X_proj = self.dc.apply_causal_graph(X_proj)
        return self.dc.to_scaled_space(X_proj)

    # ------------------------------------------------------------------
    # CAPA 2: S3M (Camuflaje Semántico Interno)
    # ------------------------------------------------------------------
    def _apply_s3m(self, X_scaled: np.ndarray) -> np.ndarray:
        n = len(X_scaled)
        X_phys = self.dc.to_physical_space(X_scaled)
        X_out = X_phys.copy()

        for i in range(n):
            candidates_idx = np.random.choice(len(self.X_benign_phys), size=self.n_twins, replace=False)
            candidates_sc = self.X_benign_sc[candidates_idx]

            diff = X_scaled[i] - candidates_sc
            weighted_dist = np.sqrt((diff ** 2 * self._twin_weights).sum(axis=1))
            best_idx = candidates_idx[weighted_dist.argmin()]
            twin_phys = self.X_benign_phys[best_idx]

            ratio = 0.5
            x_mixed = (X_phys[i] * (1 - ratio)) + (twin_phys * ratio)
            x_mixed = self.dc.apply_causal_graph(x_mixed[np.newaxis])[0]

            for feat, (vmin, vmax) in PHYSICAL_BOUNDS.items():
                idx = self.dc._feat_idx(feat)
                if idx is not None:
                    x_mixed[idx] = np.clip(x_mixed[idx], vmin, vmax)

            X_out[i] = x_mixed

        return self.dc.to_scaled_space(X_out)

    # ------------------------------------------------------------------
    # CAPA 3: OSS (Singularidad de Precisión)
    # ------------------------------------------------------------------
    def _apply_oss(self, X_scaled: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if any(idx is None for idx in [self._idx_stddev, self._idx_avg, self._idx_pkts, self._idx_dur]):
            return X_scaled, np.zeros(len(X_scaled))

        X_phys = self.dc.to_physical_space(X_scaled)
        X_out = X_phys.copy()
        iat_injected = np.zeros(len(X_phys))

        sigma_orig = X_phys[:, self._idx_stddev]
        mu_orig    = X_phys[:, self._idx_avg]
        N          = np.maximum(X_phys[:, self._idx_pkts], 2.0)

        for pkt_size in self.oss_probe_sizes:
            inner = ((self._oss_sigma_target**2 * (N + 1)) - (N * sigma_orig**2)) * (N + 1) / N
            valid = inner > 0
            iat   = np.where(valid, mu_orig + np.sqrt(np.maximum(inner, 0)), 0.0)
            iat   = np.maximum(iat, 0.0)
            mask  = valid & (iat > 0)

            if self._idx_dur is not None:
                X_out[:, self._idx_dur] = np.where(mask, X_phys[:, self._idx_dur] + iat, X_out[:, self._idx_dur])
            if self._idx_max_iat is not None:
                X_out[:, self._idx_max_iat] = np.where(mask, np.maximum(X_out[:, self._idx_max_iat], iat), X_out[:, self._idx_max_iat])
            if self._idx_bytes is not None:
                X_out[:, self._idx_bytes] = np.where(mask, X_phys[:, self._idx_bytes] + pkt_size, X_out[:, self._idx_bytes])
            if self._idx_max_pkt is not None:
                X_out[:, self._idx_max_pkt] = np.where(mask, np.maximum(X_out[:, self._idx_max_pkt], pkt_size), X_out[:, self._idx_max_pkt])

            sum_sq_new = (N * sigma_orig**2) + ((N / (N + 1)) * (iat - mu_orig)**2)
            sigma_new = np.sqrt(sum_sq_new / (N + 1))
            X_out[:, self._idx_stddev] = np.where(mask, sigma_new, X_out[:, self._idx_stddev])

            iat_injected = np.where(mask, iat, iat_injected)

        X_out = self.dc.apply_causal_graph(X_out)
        return self.dc.to_scaled_space(X_out), iat_injected

    # ------------------------------------------------------------------
    # UTILIDADES
    # ------------------------------------------------------------------
    def _predict(self, model, X: np.ndarray) -> np.ndarray:
        if hasattr(model, 'predict_proba'):
            return np.argmax(model.predict_proba(X), axis=1)
        return model.predict(X)

    def _compute_asr(self, y_pred: np.ndarray, mask_detected: np.ndarray) -> float:
        if mask_detected.sum() == 0: return 0.0
        return float((y_pred[mask_detected] == 0).mean())

    def _compute_oss_target(self, X_benign_scaled: np.ndarray, percentile: int) -> float:
        idx = self.dc._feat_idx('SRC_TO_DST_IAT_STDDEV')
        if idx is None: return 362.0
        X_benign_phys = self.dc.to_physical_space(X_benign_scaled)
        vals = X_benign_phys[:, idx]
        vals = vals[vals > 0]
        return float(np.percentile(vals, percentile)) if len(vals) > 0 else 362.0

    def _build_twin_weights(self) -> np.ndarray:
        weights = np.ones(len(self.dc.feature_names), dtype=np.float32)
        for feat in self._BUF_KNOWN:
            idx = self.dc._feat_idx(feat)
            if idx is not None: weights[idx] = 5.0
        return weights / weights.sum()