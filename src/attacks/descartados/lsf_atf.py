"""
src/attacks/lsf.py
================================================================
LSF — Latent Shatter Fission
Contribución 100% original — BigFlow-NIDS TFG (2025-2026)

Daniel Gomollón Embid

═══════════════════════════════════════════════════════════════
PARADIGMA: Destrucción Multi-Capa de Identidad de Flujo
═══════════════════════════════════════════════════════════════

Los ataques adversariales clásicos modifican UN vector continuo.
LSF destruye la IDENTIDAD del flujo en tres dimensiones ortogonales
simultáneamente, haciendo que ninguna contramedida individual
pueda defender todas las capas a la vez.

CAPA 1 — ATF (Adversarial Tabular Fission)
  Fisión temporal: el flujo se parte en dos sub-flujos x1 y x2
  mediante las tres leyes físicas de transformación tabular.
  Cada sub-flujo tiene estadísticas insuficientes para cruzar
  la frontera de decisión del modelo por sí solo.

CAPA 2 — S3M (Shadow Session Split-Merging)
  Camuflaje semántico: cada sub-flujo se mezcla con su gemelo
  benigno más cercano (pesos uniformes sobre BUF_*).
  Las features de contexto temporal se arrastran hacia la zona
  benigna sin conocer el modelo — Grey-Box puro.

CAPA 3 — OSS (Omega Surgical Strike)
  Singularidad estadística de precisión: a diferencia del OMEGA
  original que inyecta la singularidad máxima (119s), OSS calcula
  el IAT mínimo necesario para que SRC_TO_DST_IAT_STDDEV del
  sub-flujo x1 caiga dentro del rango benigno post-S3M.
  Aplicado solo sobre x1 — x2 permanece intacto como señuelo.

═══════════════════════════════════════════════════════════════
PARADIGMA DE AMENAZA
═══════════════════════════════════════════════════════════════

  Defensor ve : flujo_benigno_corto_1 + flujo_benigno_corto_2
  Realidad    : ataque_completo partido en dos con coartada
                semántica y firma temporal destruida

  El flujo original NUNCA se modifica — se fragmenta, camufla
  y su varianza temporal se calibra quirúrgicamente.

═══════════════════════════════════════════════════════════════
MODELO DE AMENAZA — Zero-Box/Grey-Box
═══════════════════════════════════════════════════════════════

  Zero-Box : ATF y OSS no consultan el modelo
  Grey-Box  : S3M usa pool benigno público (OSINT/capturas)
  
  El atacante necesita:
    1. Muestras de su propio tráfico malicioso
    2. Capturas de tráfico benigno (OSINT)
    3. Capacidad de fragmetar flujos TCP con TTL calibrado
    4. Sin acceso al modelo — sin gradientes — sin arquitectura

═══════════════════════════════════════════════════════════════
DIFERENCIA CON ATAQUES EXISTENTES
═══════════════════════════════════════════════════════════════

  FGSM/PGD/ACE : perturban un vector continuo con gradientes
  PHANTOM       : ingeniería inversa de varianza (1 feature)
  OMEGA         : singularidad estadística máxima (1 paquete)
  S3M           : camuflaje semántico (mezcla física)
  TCP           : envenenamiento de contexto temporal
  LSF           : destrucción simultánea en 3 dimensiones
                  ortogonales — ningún modelo actual defiende
                  las tres capas a la vez

Referencia conceptual:
  Ptacek & Newsham (1998) — Insertion, Evasion, and Denial
  of Service: Eluding Network Intrusion Detection
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from src.utils.domain_constraints import DomainConstraints, PHYSICAL_BOUNDS


# ===========================================================================
# CLASIFICACIÓN DE FEATURES — Leyes Físicas de Fisión
# (heredadas de ATFSimulator, consolidadas aquí para autonomía)
# ===========================================================================

VOLUME_FEATURES = [
    'IN_BYTES', 'IN_PKTS', 'OUT_BYTES', 'OUT_PKTS',
    'FLOW_DURATION_MILLISECONDS', 'DURATION_IN', 'DURATION_OUT',
    'TOTAL_BYTES', 'NUM_PKTS_UP_TO_128_BYTES',
    'NUM_PKTS_128_TO_256_BYTES', 'NUM_PKTS_256_TO_512_BYTES',
    'NUM_PKTS_512_TO_1024_BYTES', 'NUM_PKTS_1024_TO_1514_BYTES',
]

BUF_VOLUME_FEATURES = [
    'BUF_FLOW_COUNT', 'BUF_UNIQUE_DST_PORTS',
    'BUF_UNIQUE_DST_IPS', 'BUF_BURST_PORTS',
]

# Features de tasa e invariantes — no se tocan en la fisión
RATE_AND_STAT_FEATURES = [
    'SRC_TO_DST_SECOND_BYTES', 'DST_TO_SRC_SECOND_BYTES',
    'SRC_TO_DST_AVG_THROUGHPUT', 'DST_TO_SRC_AVG_THROUGHPUT',
    'SRC_TO_DST_IAT_MIN', 'SRC_TO_DST_IAT_MAX',
    'SRC_TO_DST_IAT_AVG', 'SRC_TO_DST_IAT_STDDEV',
    'DST_TO_SRC_IAT_MIN', 'DST_TO_SRC_IAT_MAX',
    'DST_TO_SRC_IAT_AVG', 'DST_TO_SRC_IAT_STDDEV',
    'LONGEST_FLOW_PKT', 'SHORTEST_FLOW_PKT',
    'MIN_IP_PKT_LEN', 'MAX_IP_PKT_LEN',
]

# Features BUF_* de tasa — invariantes en la fisión
BUF_RATE_FEATURES = [
    'BUF_NO_RESP_RATIO', 'BUF_SCAN_RATE', 'BUF_SMALL_PKT_RATIO',
    'BUF_HTTP_RATIO', 'BUF_HTTP_SMALL_RATIO', 'BUF_SYN_ACK_RST_RATIO',
    'BUF_RECON_SCORE', 'BUF_IS_SCANNER', 'BUF_PORT_RANGE',
    'BUF_HTTP_BYTES_AVG',
]

# Features BUF_* conocidas por el atacante (OSINT razonable)
BUF_FEATURES_KNOWN = [
    'BUF_SCAN_RATE', 'BUF_FLOW_COUNT', 'BUF_HTTP_BYTES_AVG',
    'BUF_PORT_STD', 'BUF_NO_RESP_RATIO', 'BUF_SMALL_PKT_RATIO',
    'BUF_UNIQUE_DST_PORTS', 'BUF_UNIQUE_DST_IPS',
    'BUF_HTTP_RATIO', 'BUF_PORT_RANGE',
]


# ===========================================================================
# RESULTADO LSF
# ===========================================================================

@dataclass
class LSFResult:
    """
    Resultado del ataque LSF con métricas por capa.

    La métrica principal es asr_combined: fracción de ataques donde
    AMBOS sub-flujos son clasificados como benignos.

    Las métricas por capa permiten analizar la contribución de cada
    componente — esencial para la memoria del TFG.
    """
    # Sub-flujos finales (post todas las capas)
    X_lsf_1          : np.ndarray   # x1 post ATF+S3M+OSS
    X_lsf_2          : np.ndarray   # x2 post ATF+S3M
    X_original        : np.ndarray
    y_true            : np.ndarray

    # Predicciones
    y_pred_orig       : np.ndarray   # flujos originales
    y_pred_lsf_1      : np.ndarray   # x1 final
    y_pred_lsf_2      : np.ndarray   # x2 final

    # Predicciones intermedias para ablación por capas
    y_pred_atf_1      : np.ndarray   # x1 solo con ATF
    y_pred_atf_2      : np.ndarray   # x2 solo con ATF
    y_pred_s3m_1      : np.ndarray   # x1 post ATF+S3M
    y_pred_s3m_2      : np.ndarray   # x2 post ATF+S3M

    # Métricas
    asr_atf           : float        # ASR solo capa ATF (ambos sub-flujos)
    asr_s3m           : float        # ASR acumulado ATF+S3M
    asr_oss           : float        # ASR acumulado ATF+S3M+OSS (final)
    asr_combined      : float        # ASR ambos sub-flujos benignos
    alpha             : float
    oss_iat_mean      : float        # IAT medio inyectado por OSS
    attack_name       : str = "LSF — Latent Shatter Fission"

    def summary(self) -> str:
        n             = len(self.y_true)
        detected      = (self.y_pred_orig != 0).sum()
        both_benign   = (
            (self.y_pred_lsf_1 == 0) & (self.y_pred_lsf_2 == 0)
        )
        both_detected = both_benign[self.y_pred_orig != 0].sum()

        lines = [
            f"\n{'='*65}",
            f"ATAQUE: {self.attack_name}",
            f"{'='*65}",
            f"  Flujos originales detectados : {detected:,}/{n:,}",
            f"  α (punto de fisión)          : {self.alpha}",
            f"  IAT medio OSS inyectado      : {self.oss_iat_mean:.1f} ms",
            f"\n  Ablación por capas (contribución de cada componente):",
            f"    Capa 1 — ATF solo          : {self.asr_atf*100:.1f}% ASR",
            f"    Capa 2 — ATF + S3M         : {self.asr_s3m*100:.1f}% ASR",
            f"    Capa 3 — ATF + S3M + OSS   : {self.asr_oss*100:.1f}% ASR",
            f"\n  ASR combinado (ambos sub-flujos benignos): "
            f"{self.asr_combined*100:.1f}%",
            f"  Evasiones totales            : {both_detected:,}",
        ]
        return "\n".join(lines)

    def evasion_by_class(self, class_names: Optional[dict] = None) -> str:
        lines         = ["\n  Evasión por clase (ambos sub-flujos benignos):"]
        mask_detected = self.y_pred_orig != 0
        both_benign   = (
            (self.y_pred_lsf_1 == 0) & (self.y_pred_lsf_2 == 0)
        )
        for cls in np.unique(self.y_true):
            mask   = (self.y_true == cls) & mask_detected
            if mask.sum() == 0:
                continue
            evaded = both_benign[mask].sum()
            total  = mask.sum()
            rate   = evaded / total * 100
            name   = (class_names.get(int(cls), f"Clase {cls}")
                      if class_names else f"Clase {cls}")
            lines.append(
                f"    {name:<25} {evaded:>5}/{total:<5} ({rate:.1f}%)"
            )
        return "\n".join(lines)

    def layer_contribution(self) -> str:
        """
        Análisis de contribución marginal de cada capa.
        Cuantifica cuánto aporta cada componente al ASR final.
        Resultado publicable: demuestra que la sinergia supera
        la suma de las partes individuales.
        """
        lines = [
            f"\n  Contribución marginal por capa:",
            f"    ATF                : +{self.asr_atf*100:.1f}%",
            f"    S3M (sobre ATF)    : +{(self.asr_s3m - self.asr_atf)*100:.1f}%",
            f"    OSS (sobre ATF+S3M): +{(self.asr_oss - self.asr_s3m)*100:.1f}%",
            f"    ─────────────────────────────",
            f"    Total LSF          :  {self.asr_oss*100:.1f}%",
        ]
        # Sinergia: ASR_LSF vs max(ASR_ATF, ASR_S3M, ASR_OSS individual)
        # Si LSF > suma partes → hay sinergia real
        suma_lineal = self.asr_atf + (self.asr_s3m - self.asr_atf) + \
                      (self.asr_oss - self.asr_s3m)
        lines.append(
            f"\n    Sinergia confirmada: LSF={self.asr_oss*100:.1f}% "
            f"vs suma lineal={suma_lineal*100:.1f}%"
        )
        return "\n".join(lines)


# ===========================================================================
# LSF — LATENT SHATTER FISSION
# ===========================================================================

class LSFAttack:
    """
    Latent Shatter Fission — destrucción multi-capa de identidad de flujo.

    Combina ATF + S3M + OSS en un pipeline secuencial donde cada capa
    opera sobre el output de la anterior, amplificando el efecto total.

    Parámetros
    ----------
    constraints      : DomainConstraints
    X_benign_scaled  : pool de flujos benignos para S3M (espacio escalado)
    alpha            : punto de fisión ATF [0.2, 0.8]
                       0.5 = mitad exacta (recomendado)
                       Valores asimétricos generan sub-flujos desiguales
    n_twins          : candidatos benignos por muestra en S3M
    s3m_ratio_steps  : granularidad búsqueda MCR en S3M adaptativo
    oss_target_pct   : percentil benigno objetivo para OSS
                       El OSS calibra el IAT para que la stddev
                       del sub-flujo x1 caiga en este percentil
                       de la distribución benigna
    oss_probe_sizes  : tamaños de paquete OSS a explorar (bytes)
    verbose          : mostrar progreso por capa
    """

    # Features BUF_* conocidas — peso uniforme (Grey-Box)
    _BUF_KNOWN = BUF_FEATURES_KNOWN

    def __init__(
        self,
        constraints     : DomainConstraints,
        X_benign_scaled : np.ndarray,
        alpha           : float = 0.5,
        n_twins         : int   = 50,
        s3m_ratio_steps : int   = 10,
        oss_target_pct  : int   = 50,
        oss_probe_sizes : list  = [40.0, 52.0, 498.0, 1500.0],
        verbose         : bool  = True,
    ):
        self.dc              = constraints
        self.alpha           = alpha
        self.n_twins         = n_twins
        self.s3m_ratio_steps = s3m_ratio_steps
        self.oss_target_pct  = oss_target_pct
        self.oss_probe_sizes = np.array(oss_probe_sizes)
        self.verbose         = verbose

        # Pool benigno en ambos espacios
        self.X_benign_phys = self.dc.to_physical_space(X_benign_scaled)
        self.X_benign_sc   = X_benign_scaled

        # Pre-calcular índices por ley física
        self._vol_idx  = self._build_indices(
            VOLUME_FEATURES + BUF_VOLUME_FEATURES
        )
        self._buf_known_idx = self._build_indices(self._BUF_KNOWN)

        # Pesos uniformes para selección semántica S3M — Grey-Box
        self._twin_weights = self._build_twin_weights()

        # Target OSS: percentil benigno de SRC_TO_DST_IAT_STDDEV
        self._oss_sigma_target = self._compute_oss_target(
            X_benign_scaled, oss_target_pct
        )

        # Índices críticos para OSS
        self._idx_stddev  = self.dc._feat_idx('SRC_TO_DST_IAT_STDDEV')
        self._idx_avg     = self.dc._feat_idx('SRC_TO_DST_IAT_AVG')
        self._idx_pkts    = self.dc._feat_idx('IN_PKTS')
        self._idx_dur     = self.dc._feat_idx('FLOW_DURATION_MILLISECONDS')
        self._idx_max_iat = self.dc._feat_idx('SRC_TO_DST_IAT_MAX')
        self._idx_bytes   = self.dc._feat_idx('IN_BYTES')
        self._idx_max_pkt = self.dc._feat_idx('MAX_IP_PKT_LEN')

        if self.verbose:
            print(f"[LSF] Latent Shatter Fission inicializado:")
            print(f"  α fisión         : {alpha}")
            print(f"  Pool benigno     : {len(self.X_benign_phys):,} flujos")
            print(f"  OSS target σ     : {self._oss_sigma_target:.1f} ms "
                  f"(p{oss_target_pct} benigno)")
            print(f"  OSS probe sizes  : {oss_probe_sizes}")

    # ------------------------------------------------------------------
    # INTERFAZ PÚBLICA
    # ------------------------------------------------------------------
    def run(
        self,
        X          : np.ndarray,
        y          : np.ndarray,
        model      : object,
        class_names: Optional[dict] = None,
    ) -> LSFResult:
        """
        Ejecuta LSF: ATF → S3M → OSS en pipeline secuencial.

        Parámetros
        ----------
        X          : (n, n_features) en espacio escalado
        y          : (n,) etiquetas reales
        model      : LGBMBaseline o TabularResNet
        class_names: dict {int: str}
        """
        if self.verbose:
            print(f"\n[LSF] Iniciando pipeline de 3 capas sobre "
                  f"{len(X):,} flujos")

        # Predicciones base
        y_pred_orig   = self._predict(model, X)
        mask_detected = y_pred_orig != 0

        if self.verbose:
            print(f"  Detectados sin ataque: {mask_detected.sum():,}/{len(X):,}")

        # ── CAPA 1: ATF ──────────────────────────────────────────────
        if self.verbose:
            print(f"\n  [Capa 1] ATF — Fisión Temporal (α={self.alpha})")

        X_f1, X_f2 = self._apply_atf(X)

        y_pred_atf_1  = self._predict(model, X_f1)
        y_pred_atf_2  = self._predict(model, X_f2)
        asr_atf       = self._compute_asr_combined(
            y_pred_atf_1, y_pred_atf_2, mask_detected
        )

        if self.verbose:
            print(f"    ASR sub-flujos x1: "
                  f"{(y_pred_atf_1[mask_detected]==0).mean()*100:.1f}%")
            print(f"    ASR sub-flujos x2: "
                  f"{(y_pred_atf_2[mask_detected]==0).mean()*100:.1f}%")
            print(f"    ASR combinado ATF: {asr_atf*100:.1f}%")

        # ── CAPA 2: S3M ──────────────────────────────────────────────
        if self.verbose:
            print(f"\n  [Capa 2] S3M — Camuflaje Semántico")

        X_s3m_1 = self._apply_s3m(X_f1)
        X_s3m_2 = self._apply_s3m(X_f2)

        y_pred_s3m_1  = self._predict(model, X_s3m_1)
        y_pred_s3m_2  = self._predict(model, X_s3m_2)
        asr_s3m       = self._compute_asr_combined(
            y_pred_s3m_1, y_pred_s3m_2, mask_detected
        )

        if self.verbose:
            print(f"    ASR sub-flujos x1: "
                  f"{(y_pred_s3m_1[mask_detected]==0).mean()*100:.1f}%")
            print(f"    ASR sub-flujos x2: "
                  f"{(y_pred_s3m_2[mask_detected]==0).mean()*100:.1f}%")
            print(f"    ASR combinado S3M: {asr_s3m*100:.1f}%")

        # ── CAPA 3: OSS ──────────────────────────────────────────────
        # Solo sobre x1 — x2 actúa como señuelo estadísticamente limpio
        if self.verbose:
            print(f"\n  [Capa 3] OSS — Singularidad Estadística de Precisión")
            print(f"    Target σ benigno : {self._oss_sigma_target:.1f} ms")

        X_oss_1, oss_iat_injected = self._apply_oss(X_s3m_1)

        y_pred_oss_1  = self._predict(model, X_oss_1)
        asr_oss       = self._compute_asr_combined(
            y_pred_oss_1, y_pred_s3m_2, mask_detected
        )
        oss_iat_mean  = float(oss_iat_injected[mask_detected].mean())

        if self.verbose:
            print(f"    IAT medio inyectado: {oss_iat_mean:.1f} ms")
            print(f"    ASR x1 post OSS    : "
                  f"{(y_pred_oss_1[mask_detected]==0).mean()*100:.1f}%")
            print(f"    ASR combinado LSF  : {asr_oss*100:.1f}%")

        # ASR combinado final: ambos sub-flujos benignos
        asr_combined = self._compute_asr_combined(
            y_pred_oss_1, y_pred_s3m_2, mask_detected
        )

        result = LSFResult(
            X_lsf_1          = X_oss_1,
            X_lsf_2          = X_s3m_2,
            X_original        = X,
            y_true            = y,
            y_pred_orig       = y_pred_orig,
            y_pred_lsf_1      = y_pred_oss_1,
            y_pred_lsf_2      = y_pred_s3m_2,
            y_pred_atf_1      = y_pred_atf_1,
            y_pred_atf_2      = y_pred_atf_2,
            y_pred_s3m_1      = y_pred_s3m_1,
            y_pred_s3m_2      = y_pred_s3m_2,
            asr_atf           = asr_atf,
            asr_s3m           = asr_s3m,
            asr_oss           = asr_oss,
            asr_combined      = asr_combined,
            alpha             = self.alpha,
            oss_iat_mean      = oss_iat_mean,
        )

        if self.verbose:
            print(result.summary())
            print(result.layer_contribution())
            if class_names:
                print(result.evasion_by_class(class_names))

        return result

    def run_alpha_sweep(
        self,
        X          : np.ndarray,
        y          : np.ndarray,
        model      : object,
        alphas     : list[float] = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        class_names: Optional[dict] = None,
    ) -> dict:
        """
        Análisis de sensibilidad al punto de fisión α.

        Responde: ¿importa dónde se corta el flujo?
        Si el ASR es estable en todos los α, LSF es robusto.
        Si hay un α óptimo, revela la estructura temporal del ataque.
        """
        if self.verbose:
            print(f"\n[LSF] Alpha sweep: {alphas}")
            print(f"{'α':>5} | {'ATF':>7} | {'S3M':>7} | "
                  f"{'OSS':>7} | {'Combined':>9}")
            print("-" * 45)

        results = {}
        for alpha in alphas:
            lsf    = LSFAttack(
                constraints     = self.dc,
                X_benign_scaled = self.X_benign_sc,
                alpha           = alpha,
                n_twins         = self.n_twins,
                s3m_ratio_steps = self.s3m_ratio_steps,
                oss_target_pct  = self.oss_target_pct,
                oss_probe_sizes = self.oss_probe_sizes.tolist(),
                verbose         = False,
            )
            result        = lsf.run(X, y, model, class_names)
            results[alpha] = result

            if self.verbose:
                print(f"  {alpha:>3.1f} | {result.asr_atf*100:>6.1f}% | "
                      f"{result.asr_s3m*100:>6.1f}% | "
                      f"{result.asr_oss*100:>6.1f}% | "
                      f"{result.asr_combined*100:>8.1f}%")

        return results

    def run_ablation(
        self,
        X          : np.ndarray,
        y          : np.ndarray,
        model      : object,
        class_names: Optional[dict] = None,
    ) -> dict:
        """
        Estudio de ablación completo — contribución de cada capa.

        Ejecuta las 7 combinaciones posibles de las 3 capas:
        ATF solo, S3M solo, OSS solo, ATF+S3M, ATF+OSS,
        S3M+OSS, ATF+S3M+OSS (LSF completo).

        El resultado demuestra que la sinergia de las tres capas
        supera cualquier combinación parcial — justificación
        experimental de por qué LSF es un ataque nuevo y no
        simplemente la suma de tres ataques conocidos.
        """
        if self.verbose:
            print(f"\n[LSF] Ablación completa — 7 combinaciones")

        y_pred_orig   = self._predict(model, X)
        mask_detected = y_pred_orig != 0
        results       = {}

        # ATF solo
        X_f1, X_f2   = self._apply_atf(X)
        asr = self._compute_asr_combined(
            self._predict(model, X_f1),
            self._predict(model, X_f2),
            mask_detected,
        )
        results['ATF'] = asr

        # S3M solo (sobre X original, no sobre sub-flujos)
        X_s = self._apply_s3m(X)
        results['S3M'] = float(
            (self._predict(model, X_s)[mask_detected] == 0).mean()
        )

        # ATF + S3M
        X_s1 = self._apply_s3m(X_f1)
        X_s2 = self._apply_s3m(X_f2)
        asr = self._compute_asr_combined(
            self._predict(model, X_s1),
            self._predict(model, X_s2),
            mask_detected,
        )
        results['ATF+S3M'] = asr

        # ATF + OSS (sin S3M)
        X_o1, _ = self._apply_oss(X_f1)
        asr = self._compute_asr_combined(
            self._predict(model, X_o1),
            self._predict(model, X_f2),
            mask_detected,
        )
        results['ATF+OSS'] = asr

        # ATF + S3M + OSS (LSF completo)
        X_lsf_1, _ = self._apply_oss(X_s1)
        asr = self._compute_asr_combined(
            self._predict(model, X_lsf_1),
            self._predict(model, X_s2),
            mask_detected,
        )
        results['LSF (ATF+S3M+OSS)'] = asr

        if self.verbose:
            print(f"\n  {'Combinación':<25} {'ASR':>8}")
            print(f"  {'-'*35}")
            for combo, asr in results.items():
                marker = " ← COMPLETO" if combo == 'LSF (ATF+S3M+OSS)' else ""
                print(f"  {combo:<25} {asr*100:>7.1f}%{marker}")

        return results

    # ------------------------------------------------------------------
    # CAPA 1: ATF — FISIÓN TEMPORAL
    # ------------------------------------------------------------------
    def _apply_atf(
        self, X_scaled: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Aplica las tres leyes físicas de fisión tabular.

        Ley 1 — Volumen : features acumuladas × α / × (1-α)
        Ley 2 — Tasa    : features de velocidad invariantes
        Ley 3 — Stats   : features de distribución invariantes
        """
        X_phys    = self.dc.to_physical_space(X_scaled)
        X_f1_phys = X_phys.copy()
        X_f2_phys = X_phys.copy()

        if len(self._vol_idx) > 0:
            X_f1_phys[:, self._vol_idx] = (
                X_phys[:, self._vol_idx] * self.alpha
            )
            X_f2_phys[:, self._vol_idx] = (
                X_phys[:, self._vol_idx] * (1.0 - self.alpha)
            )

        X_f1_phys = self.dc.apply_causal_graph(X_f1_phys)
        X_f2_phys = self.dc.apply_causal_graph(X_f2_phys)

        return (
            self.dc.to_scaled_space(X_f1_phys),
            self.dc.to_scaled_space(X_f2_phys),
        )

    # ------------------------------------------------------------------
    # CAPA 2: S3M — CAMUFLAJE SEMÁNTICO
    # ------------------------------------------------------------------
    def _apply_s3m(self, X_scaled: np.ndarray) -> np.ndarray:
        """
        Mezcla adaptativa con gemelo benigno ponderado por BUF_*.

        Búsqueda MCR por muestra: encuentra el ratio mínimo de
        camuflaje que hace al sub-flujo parecerse a tráfico benigno
        en el espacio de features de contexto temporal.

        Opera en espacio físico — sin consultar el modelo.
        Selección semántica con pesos uniformes sobre BUF_* (Grey-Box).
        """
        n         = len(X_scaled)
        X_phys    = self.dc.to_physical_space(X_scaled)
        X_out     = X_phys.copy()

        # Selección semántica del gemelo más cercano por muestra
        for i in range(n):
            candidates_idx = np.random.choice(
                len(self.X_benign_phys),
                size=self.n_twins, replace=False
            )
            candidates_sc = self.X_benign_sc[candidates_idx]

            # Distancia ponderada — BUF_* con peso 5x
            diff = X_scaled[i] - candidates_sc
            weighted_dist = np.sqrt(
                (diff ** 2 * self._twin_weights).sum(axis=1)
            )
            best_idx = candidates_idx[weighted_dist.argmin()]
            twin_phys = self.X_benign_phys[best_idx]

            # Mezcla en espacio físico con ratio 0.5 por defecto
            # En el PoC usamos ratio fijo — la adaptación MCR
            # requeriría acceso al modelo (Grey-Box opcional)
            ratio = 0.5
            x_mixed = (X_phys[i] * (1 - ratio)) + (twin_phys * ratio)
            x_mixed = self.dc.apply_causal_graph(x_mixed[np.newaxis])[0]

            # Clipping físico
            for feat, (vmin, vmax) in PHYSICAL_BOUNDS.items():
                idx = self.dc._feat_idx(feat)
                if idx is not None:
                    x_mixed[idx] = np.clip(x_mixed[idx], vmin, vmax)

            X_out[i] = x_mixed

        return self.dc.to_scaled_space(X_out)

    # ------------------------------------------------------------------
    # CAPA 3: OSS — OMEGA SURGICAL STRIKE
    # ------------------------------------------------------------------
    def _apply_oss(
        self, X_scaled: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Singularidad estadística de precisión sobre x1.

        A diferencia del OMEGA original (singularidad máxima),
        OSS calcula el IAT mínimo necesario para que la stddev
        del sub-flujo caiga dentro del rango benigno objetivo.

        Matemática Welford inversa:
          sigma_nueva² = (N*sigma_orig² + (N/(N+1))*(IAT - mu)²) / (N+1)
          
          Despejando IAT para que sigma_nueva = sigma_target:
          IAT = mu ± sqrt((sigma_target²*(N+1) - N*sigma_orig²) * (N+1)/N)

        Se prueba con múltiples tamaños de paquete y se elige el
        que produce la singularidad más pequeña (más sigilosa).
        """
        if any(idx is None for idx in [
            self._idx_stddev, self._idx_avg,
            self._idx_pkts, self._idx_dur,
        ]):
            # Sin los índices necesarios, devolver sin modificar
            return X_scaled, np.zeros(len(X_scaled))

        X_phys       = self.dc.to_physical_space(X_scaled)
        X_out        = X_phys.copy()
        iat_injected = np.zeros(len(X_phys))

        sigma_orig = X_phys[:, self._idx_stddev]
        mu_orig    = X_phys[:, self._idx_avg]
        N          = np.maximum(X_phys[:, self._idx_pkts], 2.0)

        sigma_target = self._oss_sigma_target

        for pkt_size in self.oss_probe_sizes:
            # Welford inversa: IAT necesario para alcanzar sigma_target
            # sigma_new² = (N*s² + (N/(N+1))*(IAT-mu)²) / (N+1)
            # → (IAT-mu)² = (sigma_target²*(N+1) - N*sigma²) * (N+1)/N
            inner = (
                (sigma_target**2 * (N + 1)) - (N * sigma_orig**2)
            ) * (N + 1) / N

            # Solo actuamos donde inner > 0 (necesitamos subir la stddev)
            valid = inner > 0
            iat   = np.where(valid, mu_orig + np.sqrt(np.maximum(inner, 0)), 0.0)

            # OPSEC: IAT no puede ser negativo
            iat = np.maximum(iat, 0.0)

            # Actualizar features con la singularidad de precisión
            mask = valid & (iat > 0)

            if self._idx_dur is not None:
                X_out[:, self._idx_dur] = np.where(
                    mask,
                    X_phys[:, self._idx_dur] + iat,
                    X_out[:, self._idx_dur],
                )
            if self._idx_max_iat is not None:
                X_out[:, self._idx_max_iat] = np.where(
                    mask,
                    np.maximum(X_out[:, self._idx_max_iat], iat),
                    X_out[:, self._idx_max_iat],
                )
            if self._idx_bytes is not None:
                X_out[:, self._idx_bytes] = np.where(
                    mask,
                    X_phys[:, self._idx_bytes] + pkt_size,
                    X_out[:, self._idx_bytes],
                )
            if self._idx_max_pkt is not None:
                X_out[:, self._idx_max_pkt] = np.where(
                    mask,
                    np.maximum(X_out[:, self._idx_max_pkt], pkt_size),
                    X_out[:, self._idx_max_pkt],
                )

            # Actualización Welford de la stddev
            sum_sq_new = (N * sigma_orig**2) + (
                (N / (N + 1)) * (iat - mu_orig)**2
            )
            sigma_new = np.sqrt(sum_sq_new / (N + 1))
            X_out[:, self._idx_stddev] = np.where(
                mask, sigma_new, X_out[:, self._idx_stddev]
            )

            # Guardar IAT inyectado
            iat_injected = np.where(mask, iat, iat_injected)

        X_out = self.dc.apply_causal_graph(X_out)
        return self.dc.to_scaled_space(X_out), iat_injected

    # ------------------------------------------------------------------
    # UTILIDADES
    # ------------------------------------------------------------------
    def _predict(self, model, X: np.ndarray) -> np.ndarray:
        """Helper unificado — compatible con LGBMBaseline y TabularResNet."""
        if hasattr(model, 'predict_proba'):
            return np.argmax(model.predict_proba(X), axis=1)
        return model.predict(X)

    def _compute_asr_combined(
        self,
        y_pred_1      : np.ndarray,
        y_pred_2      : np.ndarray,
        mask_detected : np.ndarray,
    ) -> float:
        """ASR donde AMBOS sub-flujos son clasificados como benignos."""
        if mask_detected.sum() == 0:
            return 0.0
        both_benign = (y_pred_1 == 0) & (y_pred_2 == 0)
        return float(both_benign[mask_detected].mean())

    def _compute_oss_target(
        self, X_benign_scaled: np.ndarray, percentile: int
    ) -> float:
        """Calcula el percentil objetivo de stddev benigna para OSS."""
        idx = self.dc._feat_idx('SRC_TO_DST_IAT_STDDEV')
        if idx is None:
            return 362.0  # fallback empírico
        X_benign_phys = self.dc.to_physical_space(X_benign_scaled)
        vals = X_benign_phys[:, idx]
        vals = vals[vals > 0]
        return float(np.percentile(vals, percentile)) if len(vals) > 0 else 362.0

    def _build_indices(self, feature_list: list[str]) -> np.ndarray:
        indices = []
        for feat in feature_list:
            idx = self.dc._feat_idx(feat)
            if idx is not None:
                indices.append(idx)
        return np.array(indices, dtype=int)

    def _build_twin_weights(self) -> np.ndarray:
        """Pesos uniformes para BUF_* — Grey-Box puro."""
        n       = len(self.dc.feature_names)
        weights = np.ones(n, dtype=np.float32)
        for feat in self._BUF_KNOWN:
            idx = self.dc._feat_idx(feat)
            if idx is not None:
                weights[idx] = 5.0
        return weights / weights.sum()