"""
src/attacks/atf_simulator.py
================================================================
ATF — Adversarial Tabular Fission (Fisión Tabular Adversaria)
Contribución conceptual original del TFG.

Motivación:
  Los ataques de gradiente (FGSM, PGD, ACE) atacan el MODELO.
  El ATF ataca el PIPELINE DE DATOS — específicamente el extractor
  de características de red (CICFlowMeter, NFStream, nflow).

  Mecanismo físico real (no implementado aquí, requiere entorno de red):
    Un atacante forja un paquete TCP FIN con TTL calibrado para que
    llegue al NIDS pero muera antes de llegar al servidor víctima.
    El extractor del NIDS ve el FIN, cierra el flujo y genera una
    fila tabular con las estadísticas acumuladas hasta ese momento.
    El servidor nunca ve el FIN y sigue recibiendo paquetes del
    atacante como si fuera una nueva conexión — generando una
    segunda fila. El ataque se divide en dos sub-flujos estadísticamente
    insuficientes para cruzar la frontera de decisión del modelo.

  Esta clase implementa una SIMULACIÓN DETERMINISTA del efecto tabular
  de esa fisión, aplicando las tres leyes físicas de transformación:

  Ley 1 — Volumen     : features de conteo y duración se multiplican por α
  Ley 2 — Tasa        : features de velocidad y throughput se mantienen
  Ley 3 — Estadísticas: features de distribución y tamaño se mantienen

  Después de la fisión, las features derivadas (ratios, flags binarios)
  se recalculan mediante el grafo causal de DomainConstraints para
  garantizar coherencia física de los sub-flujos generados.

Diferencia con los ataques algorítmicos:
  FGSM/PGD/ACE: perturbación continua en el espacio de features
  ATF          : partición discreta del flujo en el dominio temporal
  El ATF no necesita conocer el modelo — es Zero-Box.

Uso:
    from src.attacks.atf_simulator import ATFSimulator
    atf = ATFSimulator(dc, alpha=0.5)
    result = atf.run(X_attacks, y_attacks, model)
    print(result.summary())

Referencias:
  Ptacek & Newsham (1998) — Insertion, Evasion, and Denial of Service:
    Eluding Network Intrusion Detection
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional

from src.attacks.base_attacks import AttackResult
from src.utils.domain_constraints import DomainConstraints


# ===========================================================================
# CLASIFICACIÓN DE FEATURES SEGÚN LAS TRES LEYES FÍSICAS
# ===========================================================================

# LEY 1 — VOLUMEN: features que miden cantidad acumulada a lo largo del flujo.
# Al dividir el flujo por α, estas features se escalan proporcionalmente.
# Fundamento: si un ataque envía 1000 paquetes en 10s, la primera mitad
# (α=0.5) envía ~500 paquetes en ~5s.
VOLUME_FEATURES = [
    'IN_BYTES',                     # bytes enviados — se divide
    'IN_PKTS',                      # paquetes enviados — se divide
    'OUT_BYTES',                    # bytes respuesta — se divide
    'OUT_PKTS',                     # paquetes respuesta — se divide
    'FLOW_DURATION_MILLISECONDS',   # duración total — se divide
    'DURATION_IN',                  # duración entrada — se divide
    'DURATION_OUT',                 # duración salida — se divide
    'TOTAL_BYTES',                  # IN+OUT bytes — se divide (o recalcula)
    'NUM_PKTS_UP_TO_128_BYTES',     # conteos de distribución — se dividen
    'NUM_PKTS_128_TO_256_BYTES',
    'NUM_PKTS_256_TO_512_BYTES',
    'NUM_PKTS_512_TO_1024_BYTES',
    'NUM_PKTS_1024_TO_1514_BYTES',
]

# LEY 2 — TASA: features que miden velocidad/throughput instantáneo.
# La tasa de envío no cambia al dividir el flujo — si atacas a 100pps,
# ambas mitades también van a 100pps.
RATE_FEATURES = [
    'SRC_TO_DST_SECOND_BYTES',      # bytes/segundo src→dst — invariante
    'DST_TO_SRC_SECOND_BYTES',      # bytes/segundo dst→src — invariante
    'SRC_TO_DST_AVG_THROUGHPUT',    # throughput medio — invariante
    'DST_TO_SRC_AVG_THROUGHPUT',    # throughput medio — invariante
    'SRC_TO_DST_IAT_MIN',           # inter-arrival times — invariantes
    'SRC_TO_DST_IAT_MAX',           # (el espaciado entre paquetes no
    'SRC_TO_DST_IAT_AVG',           #  cambia al cortar el flujo)
    'SRC_TO_DST_IAT_STDDEV',
    'DST_TO_SRC_IAT_MIN',
    'DST_TO_SRC_IAT_MAX',
    'DST_TO_SRC_IAT_AVG',
    'DST_TO_SRC_IAT_STDDEV',
]

# LEY 3 — ESTADÍSTICAS: features de distribución y tamaño de paquete.
# Asumimos distribución uniforme del payload — el tamaño max/min/medio
# de paquete de un ataque DDoS es constante a lo largo del flujo.
STAT_FEATURES = [
    'LONGEST_FLOW_PKT',             # tamaño máximo paquete — invariante
    'SHORTEST_FLOW_PKT',            # tamaño mínimo paquete — invariante
    'MIN_IP_PKT_LEN',               # longitud mínima IP — invariante
    'MAX_IP_PKT_LEN',               # longitud máxima IP — invariante
    'TCP_WIN_MAX_IN',               # ventana TCP atacante — invariante
    'TCP_WIN_MAX_OUT',              # ventana TCP servidor — invariante
    'MIN_TTL',                      # TTL del atacante — invariante
    'MAX_TTL',                      # TTL del servidor — invariante
    'TCP_FLAGS',                    # flags TCP combinados — invariante
    'CLIENT_TCP_FLAGS',             # flags TCP cliente — invariante
    'SERVER_TCP_FLAGS',             # flags TCP servidor — invariante
    'PROTOCOL',                     # protocolo de red — invariante
    'L7_PROTO',                     # protocolo L7 — invariante
    'L4_DST_PORT',                  # puerto destino — invariante
    'IS_HTTP',                      # flag HTTP — invariante
    'IS_HTTPS',                     # flag HTTPS — invariante
    'FLOW_RANK_IN_IP',              # rank temporal — caso especial (ver nota)
    'IS_BLIND_SQLI_CANDIDATE',      # heurística — invariante
    'IS_PROBE',                     # heurística — invariante
    'IS_RECON_HTTP',                # heurística — invariante
]

# NOTA FLOW_RANK_IN_IP: en la fisión real, x2 tendría un rank mayor que x1
# porque es temporalmente posterior. Aquí lo mantenemos como aproximación
# conservadora — documentar esta limitación en la memoria.

# BUF_* — caso especial: se recalculan mediante el causal graph después
# de aplicar las leyes 1-3. En la simulación, x1 tiene el historial
# hasta la fisión y x2 empieza con buffer frío (reset). Esto se modela
# dividiendo las features BUF_* de volumen por α y manteniendo las de tasa.
BUF_VOLUME_FEATURES = [
    'BUF_FLOW_COUNT',               # flujos acumulados — se divide
    'BUF_UNIQUE_DST_PORTS',         # puertos únicos — se divide (aprox)
    'BUF_UNIQUE_DST_IPS',           # IPs únicas — se divide (aprox)
    'BUF_BURST_PORTS',              # puertos en ráfaga — se divide
]

BUF_RATE_FEATURES = [
    'BUF_NO_RESP_RATIO',            # ratio — invariante
    'BUF_SCAN_RATE',                # flujos/segundo — invariante
    'BUF_SMALL_PKT_RATIO',          # ratio — invariante
    'BUF_HTTP_RATIO',               # ratio — invariante
    'BUF_HTTP_SMALL_RATIO',         # ratio — invariante
    'BUF_SYN_ACK_RST_RATIO',        # ratio — invariante
    'BUF_RECON_SCORE',              # score compuesto — invariante
    'BUF_IS_SCANNER',               # flag binario — invariante
]

BUF_STAT_FEATURES = [
    'BUF_PORT_STD',                 # std de puertos — se divide (aprox)
    'BUF_PORT_RANGE',               # rango de puertos — invariante
    'BUF_HTTP_BYTES_AVG',           # bytes medios HTTP — invariante
]


# ===========================================================================
# RESULTADO ATF — compatible con AttackResult para el evaluador
# ===========================================================================

@dataclass
class ATFResult:
    """
    Resultado de la simulación ATF.

    Extiende la información de AttackResult con métricas específicas
    de la fisión: cuántos sub-flujos x1 y x2 evaden por separado.
    """
    # Sub-flujos generados
    X_fission_1      : np.ndarray   # primera mitad del flujo (α)
    X_fission_2      : np.ndarray   # segunda mitad (1-α)
    X_original       : np.ndarray
    y_true           : np.ndarray

    # Predicciones
    y_pred_orig      : np.ndarray   # modelo sobre flujos completos
    y_pred_fission_1 : np.ndarray   # modelo sobre x1
    y_pred_fission_2 : np.ndarray   # modelo sobre x2

    # Métricas
    asr_x1           : float        # ASR de sub-flujos x1
    asr_x2           : float        # ASR de sub-flujos x2
    asr_both         : float        # ASR ambos sub-flujos benigno (evasión total)
    alpha            : float
    attack_name      : str = "ATF — Adversarial Tabular Fission"

    def summary(self) -> str:
        n = len(self.y_true)
        both_evaded = (
            (self.y_pred_fission_1 == 0) & (self.y_pred_fission_2 == 0)
        ).sum()
        return (
            f"\n{'='*60}\n"
            f"ATAQUE: {self.attack_name}\n"
            f"{'='*60}\n"
            f"  Flujos originales   : {n:,}\n"
            f"  α (punto de fisión) : {self.alpha}\n"
            f"  Sub-flujos x1 → Benigno : {(self.y_pred_fission_1==0).sum():,} "
            f"({self.asr_x1*100:.1f}%)\n"
            f"  Sub-flujos x2 → Benigno : {(self.y_pred_fission_2==0).sum():,} "
            f"({self.asr_x2*100:.1f}%)\n"
            f"  Evasión total (ambos)   : {both_evaded:,} "
            f"({self.asr_both*100:.1f}%)\n"
            f"  [Referencia] Flujos completos detectados: "
            f"{(self.y_pred_orig!=0).sum():,}/{n:,}\n"
        )

    def evasion_by_class(self, class_names: Optional[dict] = None) -> str:
        lines = ["\n  Evasión total (ambos benigno) por clase:"]
        both_benigno = (
            (self.y_pred_fission_1 == 0) & (self.y_pred_fission_2 == 0)
        )
        for cls in np.unique(self.y_true):
            mask    = self.y_true == cls
            evaded  = both_benigno[mask].sum()
            total   = mask.sum()
            rate    = evaded / total * 100 if total > 0 else 0
            name    = class_names.get(int(cls), f"Clase {cls}") \
                      if class_names else f"Clase {cls}"
            lines.append(f"    {name:<25} {evaded:>5}/{total:<5} ({rate:.1f}%)")
        return "\n".join(lines)


# ===========================================================================
# SIMULADOR ATF
# ===========================================================================

class ATFSimulator:
    """
    Simulación determinista de la Fisión Tabular Adversaria.

    No hereda de BaseAttack porque no perturba un vector continuo —
    realiza una partición discreta en el dominio temporal del flujo.

    Parámetros
    ----------
    constraints : DomainConstraints — para inversión de escalado y causal graph
    alpha       : punto de fisión (0.5 = mitad exacta del flujo)
                  x1 recibe las primeras α·N estadísticas
                  x2 recibe las últimas (1-α)·N estadísticas
    alpha_sweep : si True, run() evalúa múltiples valores de α
                  para análisis de sensibilidad al punto de corte
    verbose     : mostrar progreso
    """

    def __init__(
        self,
        constraints  : DomainConstraints,
        alpha        : float = 0.5,
        alpha_sweep  : bool  = False,
        verbose      : bool  = True,
    ):
        self.dc          = constraints
        self.alpha       = alpha
        self.alpha_sweep = alpha_sweep
        self.verbose     = verbose

        # Pre-calcular índices por ley física
        self._vol_idx  = self._build_indices(VOLUME_FEATURES + BUF_VOLUME_FEATURES)
        self._rate_idx = self._build_indices(RATE_FEATURES + BUF_RATE_FEATURES)
        self._stat_idx = self._build_indices(STAT_FEATURES + BUF_STAT_FEATURES)

    # ------------------------------------------------------------------
    # INTERFAZ PÚBLICA
    # ------------------------------------------------------------------
    def run(
        self,
        X          : np.ndarray,
        y          : np.ndarray,
        model      : object,
        class_names: Optional[dict] = None,
    ) -> ATFResult:
        """
        Ejecuta la simulación ATF sobre el conjunto de ataques X.

        Parámetros
        ----------
        X          : (n, 66) en espacio escalado — muestras de ataque
        y          : (n,) etiquetas reales
        model      : modelo con .predict_proba(X) → (n, n_classes)
        class_names: dict {int: str} para el resumen por clase

        Retorna
        -------
        ATFResult con sub-flujos, predicciones y métricas de evasión
        """
        if self.verbose:
            print(f"\n[ATF] Simulando fisión tabular sobre {len(X):,} flujos | "
                  f"α={self.alpha}")

        # Predicciones sobre flujos originales (referencia)
        y_pred_orig = np.argmax(model.predict_proba(X), axis=1)
        detected    = (y_pred_orig != 0).sum()
        if self.verbose:
            print(f"  Flujos detectados como ataque: {detected:,}/{len(X):,}")

        # Aplicar fisión
        X_f1, X_f2 = self._apply_fission(X, self.alpha)

        # Predicciones sobre sub-flujos
        y_pred_f1 = np.argmax(model.predict_proba(X_f1), axis=1)
        y_pred_f2 = np.argmax(model.predict_proba(X_f2), axis=1)

        # Métricas
        # Solo contamos como evasión los flujos que el modelo detectaba
        mask_detected = y_pred_orig != 0
        asr_x1 = (y_pred_f1[mask_detected] == 0).mean() \
                 if mask_detected.sum() > 0 else 0.0
        asr_x2 = (y_pred_f2[mask_detected] == 0).mean() \
                 if mask_detected.sum() > 0 else 0.0
        both_benigno = (y_pred_f1 == 0) & (y_pred_f2 == 0)
        asr_both = both_benigno[mask_detected].mean() \
                   if mask_detected.sum() > 0 else 0.0

        result = ATFResult(
            X_fission_1      = X_f1,
            X_fission_2      = X_f2,
            X_original       = X,
            y_true           = y,
            y_pred_orig      = y_pred_orig,
            y_pred_fission_1 = y_pred_f1,
            y_pred_fission_2 = y_pred_f2,
            asr_x1           = asr_x1,
            asr_x2           = asr_x2,
            asr_both         = asr_both,
            alpha            = self.alpha,
        )

        if self.verbose:
            print(result.summary())
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

        Evalúa múltiples valores de α y devuelve las métricas para
        cada uno — permite identificar el α más efectivo y ver si
        la vulnerabilidad es robusta o depende del punto de corte.

        Retorna
        -------
        dict {alpha: ATFResult} — resultados por valor de α
        """
        if self.verbose:
            print(f"\n[ATF] Alpha sweep: {alphas}")
            print(f"{'α':>6} | {'ASR x1':>8} | {'ASR x2':>8} | "
                  f"{'ASR ambos':>10} | {'Evasión total':>14}")
            print("-" * 55)

        results = {}
        for alpha in alphas:
            sim     = ATFSimulator(self.dc, alpha=alpha, verbose=False)
            result  = sim.run(X, y, model, class_names=class_names)
            results[alpha] = result

            if self.verbose:
                print(f"  {alpha:>4.1f} | {result.asr_x1*100:>7.1f}% | "
                      f"{result.asr_x2*100:>7.1f}% | "
                      f"{result.asr_both*100:>9.1f}% | "
                      f"{(result.y_pred_fission_1==0).sum() + (result.y_pred_fission_2==0).sum():>8,} "
                      f"sub-flujos")

        return results

    # ------------------------------------------------------------------
    # MOTOR DE FISIÓN
    # ------------------------------------------------------------------
    def _apply_fission(
        self,
        X_scaled : np.ndarray,
        alpha    : float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Aplica las tres leyes físicas de fisión sobre X en espacio escalado.

        Proceso:
          1. Invertir escalado → espacio físico real
          2. Aplicar Ley 1 (Volumen × α / × (1-α))
          3. Ley 2 y 3 se mantienen (no se tocan)
          4. Recalcular features derivadas (causal graph)
          5. Volver a escalar → espacio del modelo

        Parámetros
        ----------
        X_scaled : (n, 66) en espacio escalado
        alpha    : punto de fisión [0, 1]

        Retorna
        -------
        X_f1 : (n, 66) primer sub-flujo en espacio escalado
        X_f2 : (n, 66) segundo sub-flujo en espacio escalado
        """
        # 1. Invertir al espacio físico
        X_phys = self.dc.to_physical_space(X_scaled)

        # 2. Crear los dos sub-flujos como copias del original
        X_phys_f1 = X_phys.copy()
        X_phys_f2 = X_phys.copy()

        # 3. Ley 1 — Volumen: escalar proporcionalmente
        if len(self._vol_idx) > 0:
            X_phys_f1[:, self._vol_idx] = X_phys[:, self._vol_idx] * alpha
            X_phys_f2[:, self._vol_idx] = X_phys[:, self._vol_idx] * (1.0 - alpha)

        # Leyes 2 y 3 — Tasa y Estadísticas: sin cambios (ya son copias)

        # 4. Recalcular features derivadas para garantizar coherencia física
        X_phys_f1 = self.dc.apply_causal_graph(X_phys_f1)
        X_phys_f2 = self.dc.apply_causal_graph(X_phys_f2)

        # 5. Volver al espacio escalado
        X_f1 = self.dc.to_scaled_space(X_phys_f1)
        X_f2 = self.dc.to_scaled_space(X_phys_f2)

        return X_f1, X_f2

    # ------------------------------------------------------------------
    # PRIVADOS
    # ------------------------------------------------------------------
    def _build_indices(self, feature_list: list[str]) -> np.ndarray:
        """Construye array de índices para una lista de features."""
        indices = []
        for feat in feature_list:
            idx = self.dc._feat_idx(feat)
            if idx is not None:
                indices.append(idx)
        return np.array(indices, dtype=int)


# ===========================================================================
# SCRIPT DE VERIFICACIÓN
# ===========================================================================

if __name__ == "__main__":
    from src.utils.domain_constraints import DomainConstraints
    import numpy as np

    print("[-] Verificando ATFSimulator...")
    dc = DomainConstraints.from_artifacts()

    atf = ATFSimulator(dc, alpha=0.5, verbose=True)
    print(f"   [✓] Instanciado: alpha={atf.alpha}")
    print(f"   Features Volumen  : {len(atf._vol_idx)}")
    print(f"   Features Tasa     : {len(atf._rate_idx)}")
    print(f"   Features Estadíst.: {len(atf._stat_idx)}")

    # Verificar que las tres categorías cubren todas las features
    all_covered = set(atf._vol_idx) | set(atf._rate_idx) | set(atf._stat_idx)
    n_features  = len(dc.feature_names)
    uncovered   = [dc.feature_names[i] for i in range(n_features)
                   if i not in all_covered]
    if uncovered:
        print(f"   [WARN] Features sin clasificar en ATF: {uncovered}")
    else:
        print(f"   [✓] Todas las {n_features} features clasificadas")

    print("\n[✓] atf_simulator.py listo para run() y run_alpha_sweep()")