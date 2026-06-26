"""
src/attacks/omega.py
================================================================
OMEGA — Outlier-Masked Evasion via Global Aggregation
Contribución original del TFG (2025-2026). 

Paradigma: Un único paquete de singularidad estadística
           colapsa las métricas de segundo orden del flujo.
           
Innovación (SSMB + OPSEC): 
Encuentra el MOT (Minimum Outlier Threshold) mediante búsqueda 
logarítmica y bisección. Contiene capacidades de despliegue en 
producción (Malleable C2 y Ghost Teardown), desactivables para 
evaluación matemática pura en entornos offline (Zero-Margin).
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass

from src.utils.domain_constraints import DomainConstraints


@dataclass
class OMEGAResult:
    X_omega_sc     : np.ndarray
    X_orig_sc      : np.ndarray
    y_true         : np.ndarray
    y_pred_orig    : np.ndarray
    y_pred_omega   : np.ndarray
    asr            : float
    stats_shift    : dict
    avg_mot_ms     : float
    avg_pkt_size   : float
    attack_name    : str = "OMEGA (SSMB Algorithmic Singularity)"

    def summary(self) -> str:
        n        = len(self.y_true)
        evaded   = (self.y_pred_omega == 0).sum()
        detected = (self.y_pred_orig != 0).sum()
        lines    = [
            f"\n{'='*85}",
            f" 💥 INFORME DE IMPACTO TÁCTICO: {self.attack_name}",
            f"{'='*85}",
            f"  Flujos base detectados   : {detected:,}/{n:,}",
            f"  Evasiones con 1 paquete  : {evaded:,} ({self.asr*100:.1f}%)",
            f"  MOT (Singularidad Mínima): {self.avg_mot_ms / 1000:.3f} s de retraso exacto",
            f"  Tamaño de paquete óptimo : {self.avg_pkt_size:.0f} bytes",
            f"\n  Colapso Estadístico Promedio (Orig → OMEGA):",
        ]
        for feat, info in self.stats_shift.items():
            delta_pct = ((info['after'] - info['before']) / (info['before'] + 1e-8)) * 100
            lines.append(
                f"    {feat:<30} | {info['before']:>10.2f} → {info['after']:>10.2f} [{delta_pct:>+8.1f}%]"
            )
        lines.append(f"{'='*85}")
        return "\n".join(lines)


class OMEGAAttack:
    def __init__(
        self,
        constraints         : DomainConstraints,
        opsec_margin_ms     : float = 0.0,   # 0.0 para evaluación offline/matemática exacta
        use_ghost_teardown  : bool  = False, # Apagado por defecto para evitar anomalías no controladas
        verbose             : bool  = True,
    ):
        self.dc = constraints
        self.opsec_margin_ms = opsec_margin_ms
        self.use_ghost_teardown = use_ghost_teardown
        self.verbose = verbose
        
        # Sondas SSMB: LAN, Wi-Fi lag, Mobile lag, Timeout TCP
        self.sonar_probes_ms = np.array([50.0, 500.0, 2000.0, 5000.0, 15000.0, 30000.0, 119900.0])
        # Tamaños exactos de protocolo TCP: RST(40), Keep-Alive(52), MTU(1500)
        self.protocol_sizes  = np.array([40.0, 52.0, 1500.0])
        
    def _inject_singularity(self, X_phys: np.ndarray, iat_targets: np.ndarray, pkt_sizes: np.ndarray) -> np.ndarray:
        """Motor físico base: Matemáticas exactas Welford para SSMB."""
        X_mod = X_phys.copy()
        
        idx_dur_ms  = self.dc._feat_idx('FLOW_DURATION_MILLISECONDS')
        idx_max_iat = self.dc._feat_idx('SRC_TO_DST_IAT_MAX')
        idx_max_pkt = self.dc._feat_idx('MAX_IP_PKT_LEN')
        idx_lon_pkt = self.dc._feat_idx('LONGEST_FLOW_PKT')
        idx_in_byte = self.dc._feat_idx('IN_BYTES')
        idx_in_pkts = self.dc._feat_idx('IN_PKTS')
        idx_stddev  = self.dc._feat_idx('SRC_TO_DST_IAT_STDDEV')
        idx_avg     = self.dc._feat_idx('SRC_TO_DST_IAT_AVG')

        if idx_max_pkt is not None: X_mod[:, idx_max_pkt] = np.maximum(X_phys[:, idx_max_pkt], pkt_sizes)
        if idx_lon_pkt is not None: X_mod[:, idx_lon_pkt] = np.maximum(X_phys[:, idx_lon_pkt], pkt_sizes)
        if idx_max_iat is not None: X_mod[:, idx_max_iat] = np.maximum(X_phys[:, idx_max_iat], iat_targets)
        if idx_dur_ms  is not None: X_mod[:, idx_dur_ms]  = X_phys[:, idx_dur_ms] + iat_targets
        if idx_in_byte is not None: X_mod[:, idx_in_byte] = X_phys[:, idx_in_byte] + pkt_sizes
        if idx_in_pkts is not None: X_mod[:, idx_in_pkts] = X_phys[:, idx_in_pkts] + 1
        
        if all(i is not None for i in [idx_stddev, idx_avg, idx_in_pkts]):
            N          = np.maximum(X_phys[:, idx_in_pkts], 1)
            mu_orig    = X_phys[:, idx_avg]
            sigma_orig = X_phys[:, idx_stddev]
            
            sum_sq_orig = N * sigma_orig**2
            new_outlier = (N / (N + 1)) * (iat_targets - mu_orig)**2
            
            X_mod[:, idx_stddev] = np.sqrt((sum_sq_orig + new_outlier) / (N + 1))
            X_mod[:, idx_avg]    = (N * mu_orig + iat_targets) / (N + 1)
            
        return X_mod

    def _apply_ghost_teardown(self, X_phys: np.ndarray, pkt_sizes: np.ndarray) -> np.ndarray:
        """OPSEC: Falsifica banderas TCP de cierre para evadir heurísticas."""
        X_mod = X_phys.copy()
        idx_rst = self.dc._feat_idx('RST_FLAG_CNT') or self.dc._feat_idx('RST_FLAG_COUNT')
        idx_fin = self.dc._feat_idx('FIN_FLAG_CNT') or self.dc._feat_idx('FIN_FLAG_COUNT')
        
        if idx_rst is not None:
            mask_rst = (pkt_sizes == 40.0)
            X_mod[mask_rst, idx_rst] += 1
        if idx_fin is not None:
            mask_fin = (pkt_sizes == 52.0)
            X_mod[mask_fin, idx_fin] += 1
            
        return X_mod

    def run(self, X_attacks_raw: np.ndarray, y: np.ndarray, model: object) -> OMEGAResult:
        X_phys    = X_attacks_raw.copy()
        X_orig_sc = self.dc.to_scaled_space(X_phys)
        
        def get_preds(X_scaled):
            if hasattr(model, 'predict_proba'):
                return np.argmax(model.predict_proba(X_scaled), axis=1)
            return model.predict(X_scaled)

        y_pred_orig = get_preds(X_orig_sc)
        mask_detected = y_pred_orig != 0
        n_samples = len(X_phys)
        
        if self.verbose:
            print(f"\n[OMEGA] Ejecutando SSMB en {mask_detected.sum()} flujos...")
            if self.opsec_margin_ms == 0.0 and not self.use_ghost_teardown:
                print(f"        Modo: Matemática Pura (Evaluación Offline / Zero-Margin)")

        best_iat  = np.full(n_samples, self.sonar_probes_ms[-1])
        best_size = np.full(n_samples, self.protocol_sizes[-1])
        evaded    = np.zeros(n_samples, dtype=bool)

        for pkt_size in self.protocol_sizes:
            base_pkt_array = np.full(n_samples, pkt_size)
            lower_bound = np.zeros(n_samples)
            upper_bound = np.full(n_samples, self.sonar_probes_ms[-1])
            found_bracket = np.zeros(n_samples, dtype=bool)

            # Fase 1: Sonar de Espectro
            for probe_ms in self.sonar_probes_ms:
                mask_to_test = (~found_bracket) & mask_detected
                if not np.any(mask_to_test): break
                
                probe_array = np.where(mask_to_test, probe_ms, 0)
                X_test_phys = self._inject_singularity(X_phys, probe_array, base_pkt_array)
                X_test_phys = self.dc.apply_causal_graph(X_test_phys)
                preds = get_preds(self.dc.to_scaled_space(X_test_phys))
                
                success = (preds == 0) & mask_to_test
                upper_bound[success] = probe_ms
                found_bracket[success] = True
                lower_bound[(~success) & mask_to_test] = probe_ms

            # Fase 2: Micro-Bisección
            mask_to_bisect = found_bracket & mask_detected
            for _ in range(10):
                if not np.any(mask_to_bisect): break
                mid_iat = (lower_bound + upper_bound) / 2.0
                X_test_phys = self._inject_singularity(X_phys, mid_iat, base_pkt_array)
                X_test_phys = self.dc.apply_causal_graph(X_test_phys)
                preds = get_preds(self.dc.to_scaled_space(X_test_phys))
                
                success = (preds == 0) & mask_to_bisect
                upper_bound[success] = mid_iat[success]
                lower_bound[~success] = mid_iat[~success]

            update_mask = found_bracket & (~evaded | (upper_bound < best_iat))
            best_iat[update_mask]  = upper_bound[update_mask]
            best_size[update_mask] = pkt_size
            evaded[update_mask]    = True

        # --- FASE 3: ENSAMBLAJE FINAL DEL PAYLOAD ---
        final_iat_payload = best_iat + self.opsec_margin_ms
        X_omega_phys = self._inject_singularity(X_phys, final_iat_payload, best_size)
        
        if self.use_ghost_teardown:
            X_omega_phys = self._apply_ghost_teardown(X_omega_phys, best_size)
            
        X_omega_phys = self.dc.apply_causal_graph(X_omega_phys)
        X_omega_sc   = self.dc.to_scaled_space(X_omega_phys)
        
        y_pred_omega = get_preds(X_omega_sc)
        asr = (y_pred_omega[mask_detected] == 0).mean() if mask_detected.sum() > 0 else 0.0
        
        stats_shift = {}
        for feat in ['SRC_TO_DST_IAT_STDDEV', 'SRC_TO_DST_IAT_MAX', 'FLOW_DURATION_MILLISECONDS', 'IN_BYTES']:
            idx = self.dc._feat_idx(feat)
            if idx is not None:
                stats_shift[feat] = {
                    'before': float(X_phys[mask_detected, idx].mean()) if mask_detected.sum() > 0 else 0.0,
                    'after' : float(X_omega_phys[mask_detected, idx].mean()) if mask_detected.sum() > 0 else 0.0
                }

        evaded_final_mask = (y_pred_omega == 0) & mask_detected
        avg_mot = float(best_iat[evaded_final_mask].mean()) if evaded_final_mask.sum() > 0 else 0.0
        avg_size = float(best_size[evaded_final_mask].mean()) if evaded_final_mask.sum() > 0 else 0.0

        result = OMEGAResult(
            X_omega_sc=X_omega_sc, X_orig_sc=X_orig_sc, y_true=y,
            y_pred_orig=y_pred_orig, y_pred_omega=y_pred_omega,
            asr=asr, stats_shift=stats_shift, avg_mot_ms=avg_mot, avg_pkt_size=avg_size
        )
        
        if self.verbose:
            print(result.summary())
        
        return result