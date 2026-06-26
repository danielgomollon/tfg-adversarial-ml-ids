"""
src/attacks/lsf.py
=============================================================================
LSF — Latent Steganographic Forgery (Falsificación Esteganográfica Latente)
=============================================================================
Trabajo de Fin de Grado (2025-2026):
Análisis, Explotación y Mitigación de Vulnerabilidades de Sistemas de 
Detección de Intrusiones basados en Machine Learning.

Autor: Daniel Gomollón Embid

PARADIGMA: Esteganografía Tabular por Inyección Parásita.
A diferencia de la proyección semántica estricta que genera artefactos
secundarios indeseados, LSF v3 opera bajo un paradigma realista de 
"Piggybacking" (APT táctico). El ataque no intenta disfrazarse por sí mismo; 
infecta un flujo benigno masivo (>100KB) ya existente, ocultando la carga 
letal volumétrica dentro de la varianza natural del huésped.

ARQUITECTURA DE CAPAS:
  - Capa 1 | PPE (Parasitic Payload Embedding): Selección de un "huésped"
           masivo y adición de la carga letal bruta (Bytes y Paquetes).
  - Capa 2 | TCC (Throughput Consistency Calibration): Estiramiento
           quirúrgico del FLOW_DURATION_MILLISECONDS para igualar el
           Throughput (ancho de banda) original del huésped, logrando
           invisibilidad ante detectores de velocidad y volumen.
  - Capa 3 | S3M (Shadow Session Split-Merging): Camuflaje del contexto
           histórico de la IP (features BUF_*) para evadir la memoria
           temporal del NIDS.

USO:
  lsf = LSFAttack(constraints=dc, X_benign_scaled=X_benign_sc, n_twins=50)
  resultado = lsf.run(X_ataques_sc, y_ataques, modelo, class_names)

Referencias:
Ptacek & Newsham (1998) - "Insertion, Evasion, and Denial of Service: Eluding Network Intrusion Detection". 
(La Biblia de la evasión de IDS. Citarlo para justificar que LSF v3 es la versión de la era de la IA de las técnicas de inserción de Ptacek).

Zander et al. (2007) - "Covert Channels and Steganography in IPv4 and IPv6". 
(Para justificar la Capa 1 y 2: esconder datos dentro de un flujo huésped gigante).
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Optional
from src.utils.domain_constraints import DomainConstraints, PHYSICAL_BOUNDS

BUF_FEATURES_KNOWN = [
    'BUF_SCAN_RATE', 'BUF_FLOW_COUNT', 'BUF_HTTP_BYTES_AVG',
    'BUF_PORT_STD', 'BUF_NO_RESP_RATIO', 'BUF_SMALL_PKT_RATIO',
    'BUF_UNIQUE_DST_PORTS', 'BUF_UNIQUE_DST_IPS',
    'BUF_HTTP_RATIO', 'BUF_PORT_RANGE',
]

@dataclass
class LSFResult:
    X_lsf            : np.ndarray   # Flujo final post PPE+TCC+S3M
    X_original       : np.ndarray
    y_true           : np.ndarray

    y_pred_orig      : np.ndarray
    y_pred_ppe_tcc   : np.ndarray   # Post Capa 1 y 2
    y_pred_lsf       : np.ndarray   # Post Capa 3 (Final)

    asr_ppe_tcc      : float
    asr_lsf          : float        # ASR Final
    attack_name      : str = "LSF v3 — Latent Steganographic Forgery"

    def summary(self) -> str:
        n = len(self.y_true)
        detected = (self.y_pred_orig != 0).sum()
        evaded = (self.y_pred_lsf == 0)[self.y_pred_orig != 0].sum()

        lines = [
            f"\n{'='*65}",
            f"ATAQUE: {self.attack_name}",
            f"{'='*65}",
            f"  Flujos originales detectados : {detected:,}/{n:,}",
            f"\n  Ablación por capas (contribución acumulada):",
            f"    Capa 1+2 (Parasitic Embedding + TCC) : {self.asr_ppe_tcc*100:.1f}% ASR",
            f"    Capa 3   (Capa 1+2 + S3M Context)    : {self.asr_lsf*100:.1f}% ASR",
            f"\n  Evasiones totales finales            : {evaded:,} ({self.asr_lsf*100:.1f}%)",
        ]
        return "\n".join(lines)

    def evasion_by_class(self, class_names: Optional[dict] = None) -> str:
        lines = ["\n  Evasión por clase (Final LSF v3):"]
        mask_detected = self.y_pred_orig != 0
        evaded_mask = (self.y_pred_lsf == 0)

        for cls in np.unique(self.y_true):
            mask = (self.y_true == cls) & mask_detected
            if mask.sum() == 0: continue
            evaded = evaded_mask[mask].sum()
            total  = mask.sum()
            rate   = evaded / total * 100
            name   = (class_names.get(int(cls), f"Clase {cls}") if class_names else f"Clase {cls}")
            lines.append(f"    {name:<25} {evaded:>5}/{total:<5} ({rate:.1f}%)")
        return "\n".join(lines)


class LSFAttack:
    def __init__(
        self,
        constraints     : DomainConstraints,
        X_benign_scaled : np.ndarray,
        n_twins         : int   = 50,
        verbose         : bool  = True,
    ):
        self.dc              = constraints
        self.n_twins         = n_twins
        self.verbose         = verbose

        self.X_benign_phys = self.dc.to_physical_space(X_benign_scaled)
        self.X_benign_sc   = X_benign_scaled

        # Grey-Box S3M weights
        self._BUF_KNOWN = BUF_FEATURES_KNOWN
        self._twin_weights = self._build_twin_weights()

        # Pre-computar índices para inyección parásita
        self._idx_bytes_in  = self.dc._feat_idx('IN_BYTES')
        self._idx_pkts_in   = self.dc._feat_idx('IN_PKTS')
        self._idx_dur       = self.dc._feat_idx('FLOW_DURATION_MILLISECONDS')
        self._idx_throughput= self.dc._feat_idx('SRC_TO_DST_AVG_THROUGHPUT')

    def run(self, X: np.ndarray, y: np.ndarray, model: object, class_names: Optional[dict] = None) -> LSFResult:
        y_pred_orig = self._predict(model, X)
        mask_detected = y_pred_orig != 0

        if self.verbose:
            print("\n[LSF v3] Iniciando Parasitic Payload Embedding...")

        # CAPA 1 y 2: PPE + TCC (Esteganografía Volumétrica y Calibración Temporal)
        X_ppe = self._apply_parasitic_embedding_and_tcc(X)
        y_pred_ppe = self._predict(model, X_ppe)
        asr_ppe = self._compute_asr(y_pred_ppe, mask_detected)

        # CAPA 3: S3M (Camuflaje de Contexto IP)
        if self.verbose:
            print("[LSF v3] Aplicando Camuflaje de Contexto (S3M)...")
            
        X_lsf = self._apply_s3m(X_ppe)
        y_pred_lsf = self._predict(model, X_lsf)
        asr_lsf = self._compute_asr(y_pred_lsf, mask_detected)

        result = LSFResult(
            X_lsf=X_lsf, X_original=X, y_true=y,
            y_pred_orig=y_pred_orig, y_pred_ppe_tcc=y_pred_ppe, y_pred_lsf=y_pred_lsf,
            asr_ppe_tcc=asr_ppe, asr_lsf=asr_lsf
        )

        if self.verbose:
            print(result.summary())
            if class_names:
                print(result.evasion_by_class(class_names))

        return result

    def _apply_parasitic_embedding_and_tcc(self, X_scaled: np.ndarray) -> np.ndarray:
        """
        CAPA 1 (PPE): Busca un host masivo y le suma el volumen del ataque.
        CAPA 2 (TCC): Estira el tiempo para que el throughput sea idéntico al host original.
        """
        X_phys = self.dc.to_physical_space(X_scaled)
        X_out = np.copy(X_phys)

        if self._idx_bytes_in is None or self._idx_dur is None:
            return X_scaled

        # Filtrar posibles "huéspedes masivos" (> 100 KB para absorber ataques fácilmente)
        host_mask = self.X_benign_phys[:, self._idx_bytes_in] > 100000
        pool_hosts = self.X_benign_phys[host_mask]
        
        if len(pool_hosts) == 0:
            pool_hosts = self.X_benign_phys # Fallback si no hay flujos masivos

        for i in range(len(X_phys)):
            atk_bytes = X_phys[i, self._idx_bytes_in]
            atk_pkts  = X_phys[i, self._idx_pkts_in] if self._idx_pkts_in else 0
            
            # Seleccionar un huésped aleatorio del pool masivo
            host_idx = np.random.randint(0, len(pool_hosts))
            host = pool_hosts[host_idx].copy()
            
            host_bytes_orig = max(host[self._idx_bytes_in], 1.0)
            host_dur_orig = max(host[self._idx_dur], 1.0)
            target_throughput = host_bytes_orig / (host_dur_orig / 1000.0) # Bytes per second
            
            # CAPA 1: Inyección Parásita
            new_bytes = host_bytes_orig + atk_bytes
            new_pkts  = host[self._idx_pkts_in] + atk_pkts if self._idx_pkts_in else host[self._idx_pkts_in]
            
            # Clonamos el esqueleto del huésped...
            X_out[i] = host
            # ...y le inyectamos nuestra carga
            X_out[i, self._idx_bytes_in] = new_bytes
            if self._idx_pkts_in is not None:
                X_out[i, self._idx_pkts_in] = new_pkts

            # CAPA 2: Throughput Consistency Calibration (TCC)
            # Despejamos Duration: Duration(s) = Total Bytes / Target Throughput
            if target_throughput > 0:
                new_duration_ms = (new_bytes / target_throughput) * 1000.0
                X_out[i, self._idx_dur] = new_duration_ms

        X_out = self.dc.apply_causal_graph(X_out)
        return self.dc.to_scaled_space(X_out)

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

    def _predict(self, model, X: np.ndarray) -> np.ndarray:
        if hasattr(model, 'predict_proba'):
            return np.argmax(model.predict_proba(X), axis=1)
        return model.predict(X)

    def _compute_asr(self, y_pred: np.ndarray, mask_detected: np.ndarray) -> float:
        if mask_detected.sum() == 0: return 0.0
        return float((y_pred[mask_detected] == 0).mean())

    def _build_twin_weights(self) -> np.ndarray:
        weights = np.ones(len(self.dc.feature_names), dtype=np.float32)
        for feat in self._BUF_KNOWN:
            idx = self.dc._feat_idx(feat)
            if idx is not None: weights[idx] = 5.0
        return weights / weights.sum()