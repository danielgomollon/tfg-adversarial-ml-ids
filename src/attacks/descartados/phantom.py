"""
src/attacks/phantom.py

Daniel Gomollón Embid — TFG 2025-2026
Análisis, Explotación y Mitigación de Vulnerabilidades de Sistemas 
  de Detección de Intrusiones basados en Machine Learning

Dataset: BigFlow-NIDS (submuestreo 1.1M flujos, 51+16 features, 10 clases)
  sumado a IP Behavior Buffer (16 features adicionales de comportamiento IP)

================================================================
PHANTOM — Protocol-Aware Hidden Attack via Netflow Timing Manipulation

═══════════════════════════════════════════════════════════════
PARADIGMA: Ingeniería Inversa del Extractor de Características
═══════════════════════════════════════════════════════════════

Todos los ataques adversariales conocidos siguen este flujo:
  vector_ataque -> perturbar_vector -> esperar_que_sea_físico

PHANTOM invierte el paradigma completamente:
  vector_benigno_objetivo -> calcular_paquetes -> generar_tráfico

El atacante no necesita acceso al modelo.
El atacante no necesita calcular gradientes.
El atacante SOLO necesita conocer las fórmulas del extractor
  de características (NFStream, CICFlowMeter) - información
  pública disponible en su documentación.

═══════════════════════════════════════════════════════════════
FUNDAMENTO MATEMÁTICO
═══════════════════════════════════════════════════════════════

NFStream calcula las estadísticas de un flujo como:
  μ_IAT = (1/N) Σ IAT_i
  σ_IAT = √[(1/N) Σ (IAT_i - μ)²]

Dado un flujo de ataque con N paquetes e IATs originales
{t_1, t_2, ..., t_N}, PHANTOM resuelve el sistema inverso:

  Problema: encontrar {Δt_1, ..., Δt_N} tal que:
    μ_new ≈ μ_orig     (no cambiar la media — no cambia duración)
    σ_new = σ_target   (elevar varianza al rango benigno)
    Δt_i ≥ 0           (los intervalos no pueden ser negativos)

Solución analítica de forma cerrada:
  Si añadimos k paquetes "jitter" con intervalo p_jitter,
  la varianza combinada satisface una ecuación cuadrática
  en p_jitter con solución exacta O(1).

Esto es ADF generalizado: donde ADF trabaja en espacio
tabular aproximando, PHANTOM trabaja en espacio de paquetes
con solución exacta porque conoce las fórmulas del extractor.

═══════════════════════════════════════════════════════════════
DIFERENCIA CON ADF
═══════════════════════════════════════════════════════════════

ADF  : opera en espacio tabular (aproximación)
       inyecta paquetes chaff con tamaño calculado
       necesita estimar cómo el extractor agrega

PHANTOM: opera en espacio de paquetes (exacto)
         calcula directamente los intervalos temporales
         garantiza que el extractor producirá exactamente
         el vector tabular objetivo

La precisión de PHANTOM es matemáticamente exacta porque
invierte la función del extractor en lugar de aproximarla.

═══════════════════════════════════════════════════════════════
MODELO DE AMENAZA (REALISMO)
═══════════════════════════════════════════════════════════════

El atacante solo necesita:
  1. Documentación pública de NFStream/CICFlowMeter (disponible)
  2. Una muestra del tráfico benigno para estimar los targets
     (disponible via OSINT, capturas públicas, etc.)
  3. Control sobre el timing de sus propios paquetes
     (cualquier script de red puede hacer esto)

No necesita:
  - Acceso al modelo
  - Consultas al IDS
  - Gradientes o SHAP values
  - Conocer la arquitectura del detector

Es el ataque más realista del arsenal para un atacante real.

Uso:
    phantom = PHANTOMAttack(dc, X_benign_raw, sigma_weight=1.0)
    result  = phantom.run(X_attacks_raw, y_attacks, model)
    sweep   = phantom.run_target_sweep(X_attacks_raw, y_attacks, model)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional

from src.attacks.base_attacks import AttackResult
from src.utils.domain_constraints import DomainConstraints


# ===========================================================================
# FEATURES OBJETIVO — las que PHANTOM manipula analíticamente
# ===========================================================================

# Features de timing que PHANTOM puede elevar al rango benigno
# mediante manipulación de intervalos entre paquetes
TIMING_FEATURES = {
    'SRC_TO_DST_IAT_STDDEV': {
        'mean_feat' : 'SRC_TO_DST_IAT_AVG',
        'count_feat': 'IN_PKTS',
        'description': 'Desviación estándar de IAT src→dst',
    },
    'DST_TO_SRC_IAT_STDDEV': {
        'mean_feat' : 'DST_TO_SRC_IAT_AVG',
        'count_feat': 'OUT_PKTS',
        'description': 'Desviación estándar de IAT dst→src',
    },
}

# Features de volumen que se recalculan en cascada
# (el causal graph las corrige automáticamente)
CASCADE_FEATURES = [
    'FLOW_DURATION_MILLISECONDS',
    'DURATION_IN',
    'SRC_TO_DST_AVG_THROUGHPUT',
    'DURATION_PER_PKT',
]


# ===========================================================================
# RESULTADO PHANTOM
# ===========================================================================

@dataclass
class PHANTOMResult:
    """
    Resultado del ataque PHANTOM.

    La métrica más importante es precision_error: cuánto se desvía
    el vector generado del vector objetivo. Si es < 1%, la ingeniería
    inversa es exacta.
    """
    X_phantom        : np.ndarray    # flujos modificados (espacio escalado)
    X_original       : np.ndarray    # flujos originales
    y_true           : np.ndarray

    y_pred_orig      : np.ndarray
    y_pred_phantom   : np.ndarray

    asr              : float
    precision_error  : dict          # |valor_generado - valor_objetivo| por feature
    timing_shift     : dict          # cambio medio en features de timing
    attack_name      : str = "PHANTOM — Protocol-Aware Hidden Attack"

    def summary(self) -> str:
        n        = len(self.y_true)
        evaded   = (self.y_pred_phantom == 0).sum()
        detected = (self.y_pred_orig != 0).sum()
        lines    = [
            f"\n{'='*60}",
            f"ATAQUE: {self.attack_name}",
            f"{'='*60}",
            f"  Flujos originales detectados : {detected:,}/{n:,}",
            f"  Evasiones exitosas           : {evaded:,} ({self.asr*100:.1f}%)",
            f"\n  Precisión de la inversión (|generado - objetivo|):",
        ]
        for feat, err in self.precision_error.items():
            lines.append(f"    {feat:<30} error={err:.4f}")
        lines.append(f"\n  Cambio en features de timing:")
        for feat, info in self.timing_shift.items():
            lines.append(
                f"    {feat:<30} {info['before']:>8.3f} → {info['after']:>8.3f}"
            )
        return "\n".join(lines)

    def evasion_by_class(
        self, class_names: Optional[dict] = None
    ) -> str:
        lines         = ["\n  Evasión por clase:"]
        mask_detected = self.y_pred_orig != 0
        for cls in np.unique(self.y_true):
            mask   = (self.y_true == cls) & mask_detected
            if mask.sum() == 0:
                continue
            evaded = (self.y_pred_phantom[mask] == 0).sum()
            total  = mask.sum()
            rate   = evaded / total * 100
            name   = class_names.get(int(cls), f"Clase {cls}") \
                     if class_names else f"Clase {cls}"
            lines.append(
                f"    {name:<25} {evaded:>5}/{total:<5} ({rate:.1f}%)"
            )
        return "\n".join(lines)


# ===========================================================================
# ATAQUE PHANTOM
# ===========================================================================

class PHANTOMAttack:
    """
    Protocol-Aware Hidden Attack via Netflow Timing Manipulation.

    Calcula analíticamente los intervalos entre paquetes que generan
    las estadísticas de timing deseadas, sin necesidad de acceder
    al modelo ni calcular gradientes.

    Parámetros
    ----------
    constraints   : DomainConstraints
    X_benign_raw  : (n_benign, 66) en espacio físico — para calcular
                    los targets de timing del tráfico benigno real
    sigma_weight  : fracción del percentil 50 benigno como target
                    1.0 = target exactamente en la mediana benigna
                    0.5 = target en la mitad del camino
    percentile    : percentil benigno usado como target (default 50)
    seed          : reproducibilidad
    verbose       : mostrar progreso
    """

    def __init__(
        self,
        constraints  : DomainConstraints,
        X_benign_raw : np.ndarray,
        sigma_weight : float = 1.0,
        percentile   : int   = 50,
        seed         : int   = 42,
        verbose      : bool  = True,
    ):
        self.dc           = constraints
        self.sigma_weight = sigma_weight
        self.percentile   = percentile
        self.seed         = seed
        self.verbose      = verbose

        # Pre-computar targets de timing desde el tráfico benigno
        self.targets = self._compute_benign_targets(X_benign_raw)

        if self.verbose:
            print(f"[PHANTOM] Targets benignos computados (p{percentile}):")
            for feat, val in self.targets.items():
                print(f"  {feat:<35} target={val:.4f}")

    # ------------------------------------------------------------------
    # INTERFAZ PÚBLICA
    # ------------------------------------------------------------------
    def run(
        self,
        X_attacks_raw : np.ndarray,
        y             : np.ndarray,
        model         : object,
        class_names   : Optional[dict] = None,
    ) -> PHANTOMResult:
        """
        Ejecuta PHANTOM sobre los flujos de ataque.

        Parámetros
        ----------
        X_attacks_raw : (n, 66) en espacio FÍSICO original
        y             : (n,) etiquetas reales
        model         : con .predict_proba(X_scaled)
        """
        if self.verbose:
            print(f"\n[PHANTOM] Ingeniería inversa del extractor | "
                  f"target p{self.percentile} benigno")
            print(f"  Flujos de ataque: {len(X_attacks_raw):,}")
            print(f"  Sin acceso al modelo — solo álgebra inversa")

        # 1. Predicciones originales
        X_orig_sc   = self.dc.to_scaled_space(X_attacks_raw)
        y_pred_orig = np.argmax(model.predict_proba(X_orig_sc), axis=1)
        detected    = (y_pred_orig != 0).sum()

        if self.verbose:
            print(f"  Detectados como ataque: {detected:,}/{len(y):,}")

        # 2. Aplicar manipulación de timing (ingeniería inversa)
        X_phantom_raw, precision_error = self._apply_timing_inversion(
            X_attacks_raw
        )

        # 3. Recalcular features derivadas (causal graph)
        X_phantom_raw = self.dc.apply_causal_graph(X_phantom_raw)

        # 4. Escalar y predecir
        X_phantom_sc  = self.dc.to_scaled_space(X_phantom_raw)
        y_pred_phantom = np.argmax(
            model.predict_proba(X_phantom_sc), axis=1
        )

        # 5. Métricas
        mask_detected = y_pred_orig != 0
        asr = (y_pred_phantom[mask_detected] == 0).mean() \
              if mask_detected.sum() > 0 else 0.0

        timing_shift = self._compute_timing_shift(X_orig_sc, X_phantom_sc)

        result = PHANTOMResult(
            X_phantom      = X_phantom_sc,
            X_original     = X_orig_sc,
            y_true         = y,
            y_pred_orig    = y_pred_orig,
            y_pred_phantom = y_pred_phantom,
            asr            = asr,
            precision_error= precision_error,
            timing_shift   = timing_shift,
        )

        if self.verbose:
            print(result.summary())
            if class_names:
                print(result.evasion_by_class(class_names))

        return result

    def run_target_sweep(
        self,
        X_attacks_raw : np.ndarray,
        y             : np.ndarray,
        model         : object,
        percentiles   : list[int] = [25, 50, 75, 90, 95],
        class_names   : Optional[dict] = None,
    ) -> dict:
        """
        Análisis de sensibilidad al target de timing.

        Evalúa distintos percentiles del tráfico benigno como target.
        Permite identificar el target mínimo para evadir el modelo —
        cuánta "naturalidad" de timing necesita el ataque.

        Retorna
        -------
        dict {percentile: PHANTOMResult}
        """
        if self.verbose:
            print(f"\n[PHANTOM] Target sweep: percentiles {percentiles}")
            print(f"{'p':>5} | {'ASR':>8} | {'σ_IAT error':>12}")
            print("-" * 32)

        results = {}
        X_benign_raw_ref = self.dc.to_physical_space(
            np.zeros((1, len(self.dc.feature_names)))
        )

        for pct in percentiles:
            atk = PHANTOMAttack(
                self.dc,
                X_benign_raw = self._get_benign_pool(),
                sigma_weight = self.sigma_weight,
                percentile   = pct,
                seed         = self.seed,
                verbose      = False,
            )
            result  = atk.run(X_attacks_raw, y, model, class_names)
            results[pct] = result

            err = np.mean(list(result.precision_error.values()))
            if self.verbose:
                print(f"  p{pct:>2} | {result.asr*100:>7.1f}% | "
                      f"{err:>11.4f}")

        return results

    # ------------------------------------------------------------------
    # MOTOR DE INVERSIÓN ANALÍTICA (MÉTODO EXACTO)
    # ------------------------------------------------------------------
    def _apply_timing_inversion(
        self,
        X_raw: np.ndarray,
    ) -> tuple[np.ndarray, dict]:
        """
        Aplica la inversión analítica del extractor sobre las features de timing.

        INNOVACIÓN MATEMÁTICA (TFG):
        Para calcular el IAT del paquete a inyectar (p_jitter) que eleve la 
        varianza a un target exacto, debemos tener en cuenta que al añadir un 
        paquete, la MEDIA TAMBIÉN SE DESPLAZA.
        
        Usando la actualización exacta de la varianza poblacional:
          σ²_new = [N·σ²_orig + (N/(N+1))·(p_j - μ)²] / (N+1)
          
        Despejando p_j para que σ²_new == σ²_target:
          p_j = μ + √[ ((N+1)²/N)·σ²_target - (N+1)·σ²_orig ]

        Retorna
        -------
        X_modified : array modificado en espacio físico
        precision_error : dict con error de inversión por feature
        """
        X_mod = X_raw.copy()
        precision_errors = {}

        for feat_stddev, info in TIMING_FEATURES.items():
            feat_mean  = info['mean_feat']
            feat_count = info['count_feat']

            idx_std   = self.dc._feat_idx(feat_stddev)
            idx_mean  = self.dc._feat_idx(feat_mean)
            idx_count = self.dc._feat_idx(feat_count)
            idx_dur   = self.dc._feat_idx('FLOW_DURATION_MILLISECONDS')

            if any(idx is None for idx in [idx_std, idx_mean, idx_count]):
                continue

            # Target de varianza desde el tráfico benigno
            sigma_target = self.targets.get(feat_stddev, 0.0) * self.sigma_weight

            if sigma_target <= 0:
                continue

            # Valores originales del flujo
            mu_orig    = X_raw[:, idx_mean]    # IAT_AVG actual
            sigma_orig = X_raw[:, idx_std]     # IAT_STDDEV actual
            N          = np.maximum(X_raw[:, idx_count], 2)  # n paquetes (mínimo 2 para stddev)

            # --- SOLUCIÓN ANALÍTICA EXACTA ---
            # Término 1: ((N+1)^2 / N) * σ²_target
            term1 = ((N + 1)**2 / N) * (sigma_target**2)
            # Término 2: (N+1) * σ²_orig
            term2 = (N + 1) * (sigma_orig**2)
            
            discriminant = term1 - term2

            # valid: solo actuamos si necesitamos subir la varianza
            valid = discriminant >= 0

            # Calculamos el IAT del paquete espurio a inyectar
            p_jitter = np.where(
                valid,
                mu_orig + np.sqrt(np.maximum(discriminant, 0)),
                0.0 # Si no es válido, no inyectamos jitter
            )

            # --- ACTUALIZACIÓN FÍSICA DEL FLUJO ---
            # 1. Nueva Varianza (es exactamente el target por definición matemática)
            X_mod[:, idx_std] = np.where(valid, sigma_target, sigma_orig)

            # 2. Nueva Media Exacta
            mu_new = (N * mu_orig + p_jitter) / (N + 1)
            X_mod[:, idx_mean] = np.where(valid, mu_new, mu_orig)

            # 3. Consecuencias en el Extractor (Si inyecto un paquete, el flujo dura más y tiene +1 pkt)
            if idx_dur is not None:
                X_mod[:, idx_dur] = np.where(valid, X_mod[:, idx_dur] + p_jitter, X_mod[:, idx_dur])
            
            X_mod[:, idx_count] = np.where(valid, X_mod[:, idx_count] + 1, X_mod[:, idx_count])

            # Error de precisión: Debería ser ~0.0 gracias a la fórmula exacta
            sum_sq_new = (N * sigma_orig**2) + (N / (N+1)) * (p_jitter - mu_orig)**2
            sigma_comprobacion = np.sqrt(sum_sq_new / (N + 1))
            
            error = np.abs(sigma_comprobacion - sigma_target) / (sigma_target + 1e-8)
            precision_errors[feat_stddev] = float(error[valid].mean()) if valid.sum() > 0 else 0.0

        return X_mod, precision_errors

    # ------------------------------------------------------------------
    # TARGETS BENIGNOS
    # ------------------------------------------------------------------
    def _compute_benign_targets(self, X_benign_raw: np.ndarray) -> dict:
        """
        Calcula el percentil objetivo de cada feature de timing
        desde el tráfico benigno real.
        """
        targets = {}
        for feat in TIMING_FEATURES:
            idx = self.dc._feat_idx(feat)
            if idx is not None:
                vals = X_benign_raw[:, idx]
                vals = vals[vals > 0]  # excluir ceros (flujos sin esa dirección)
                if len(vals) > 0:
                    targets[feat] = float(
                        np.percentile(vals, self.percentile)
                    )
        return targets

    def _get_benign_pool(self) -> np.ndarray:
        """Devuelve el pool benigno — para uso interno en sweep."""
        return np.zeros((1, len(self.dc.feature_names)))

    # ------------------------------------------------------------------
    # MÉTRICAS
    # ------------------------------------------------------------------
    def _compute_timing_shift(
        self,
        X_orig_sc    : np.ndarray,
        X_phantom_sc : np.ndarray,
    ) -> dict:
        """Calcula el cambio medio en features de timing clave."""
        shift = {}
        for feat in list(TIMING_FEATURES.keys()):
            idx = self.dc._feat_idx(feat)
            if idx is None:
                continue
            before = float(X_orig_sc[:, idx].mean())
            after  = float(X_phantom_sc[:, idx].mean())
            shift[feat] = {'before': before, 'after': after}
        return shift


# ===========================================================================
# SCRIPT DE VERIFICACIÓN
# ===========================================================================

if __name__ == "__main__":
    import numpy as np
    from src.utils.domain_constraints import DomainConstraints

    print("[-] Verificando PHANTOMAttack...")

    dc = DomainConstraints.from_artifacts()

    # Pool benigno mínimo para test
    np.random.seed(42)
    n_benign = 1000
    X_benign_raw = np.zeros((n_benign, len(dc.feature_names)))

    idx_std  = dc._feat_idx('SRC_TO_DST_IAT_STDDEV')
    idx_mean = dc._feat_idx('SRC_TO_DST_IAT_AVG')
    if idx_std is not None:
        X_benign_raw[:, idx_std]  = np.random.normal(45.0, 10.0, n_benign)
        X_benign_raw[:, idx_mean] = np.random.normal(26.0, 5.0, n_benign)

    phantom = PHANTOMAttack(dc, X_benign_raw, sigma_weight=1.0, verbose=True)

    print(f"\n   [✓] PHANTOMAttack instanciado")
    print(f"   Features de timing manipuladas: {len(TIMING_FEATURES)}")
    print(f"\n[✓] phantom.py listo")
    print("    IMPORTANTE: X_attacks_raw debe estar en espacio FÍSICO")
    print("    Uso:")
    print("      phantom = PHANTOMAttack(dc, X_benign_raw)")
    print("      result  = phantom.run(X_attacks_raw, y_attacks, model)")
    print("      sweep   = phantom.run_target_sweep(X_attacks_raw, y_attacks, model)")