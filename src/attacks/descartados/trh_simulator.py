"""
src/attacks/trh_simulator.py
================================================================
TRH — Temporal Resonance Hijacking
Contribución conceptual original del TFG.

Motivación:
  Los ataques de gradiente (FGSM, PGD, ACE) atacan el MODELO.
  El ATF ataca el EXTRACTOR de características.
  El TRH ataca la RESOLUCIÓN TEMPORAL del buffer deslizante.

  Los modelos de ML para NIDS tienen una vulnerabilidad de aliasing:
  procesan vectores de estadísticas promediadas sobre una ventana
  temporal. Si el atacante envía tráfico en ráfagas sincronizadas
  con la ventana del buffer (patrón on/off), las features BUF_*
  resultantes son estadísticamente indistinguibles del tráfico benigno
  esporádico, aunque el payload malicioso se haya entregado completo.

  A diferencia del ATF (que requiere simular un extractor externo),
  el TRH se ejecuta directamente sobre el IPBehaviorBuffer real del
  pipeline — los resultados son experimentales, no aproximaciones.

Mecanismo (patrón strobe):
  flujo_ataque = [e1, e2, e3, e4, e5, e6, e7, e8, e9, e10]
                  ↓  duty_cycle=0.3 (30% activo, 70% silencio)
  flujo_strobe = [e1, -, -, e4, -, -, e7, -, -, e10]

  El buffer ve un flujo de baja densidad temporal → features BUF_*
  con valores similares a tráfico benigno esporádico → modelo predice
  benigno aunque el atacante haya enviado el payload completo.

Diferencia con ATF:
  ATF  : divide el flujo en dos sub-flujos (fisión temporal)
  TRH  : mantiene un único flujo pero reduce su densidad temporal
          mediante patrón on/off sincronizado con la ventana del buffer

Parámetros del experimento:
  duty_cycle   : fracción de eventos que sobreviven (0.3 = 30% activo)
  pattern      : 'uniform' | 'burst' | 'periodic'
    - uniform  : eliminación aleatoria uniforme
    - burst    : ráfagas cortas separadas por silencios largos
    - periodic : patrón regular de N_on / N_off eventos

Uso:
    from src.attacks.trh_simulator import TRHSimulator
    trh = TRHSimulator(buffer, dc, duty_cycle=0.3)
    result = trh.run(flows_attack, y_attack, model)

Referencias:
  Shannon-Nyquist (1949) — teorema de muestreo (aliasing temporal)
  Contribución original — BigFlow-NIDS TFG (2025-2026)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional

from src.utils.domain_constraints import DomainConstraints


# ===========================================================================
# RESULTADO TRH
# ===========================================================================

@dataclass
class TRHResult:
    """
    Resultado de la simulación TRH.

    Incluye comparativa entre el flujo original y el flujo strobe,
    y el análisis de degradación de las features BUF_* clave.
    """
    X_strobe         : np.ndarray    # flujos stroboscopados en espacio escalado
    X_original       : np.ndarray    # flujos originales
    y_true           : np.ndarray

    y_pred_orig      : np.ndarray    # predicciones sobre flujos completos
    y_pred_strobe    : np.ndarray    # predicciones sobre flujos strobe

    asr              : float         # Attack Success Rate (evasiones / detectados)
    duty_cycle       : float
    pattern          : str
    buf_degradation  : dict          # degradación media de features BUF_* clave
    attack_name      : str = "TRH — Temporal Resonance Hijacking"

    def summary(self) -> str:
        n          = len(self.y_true)
        evaded     = (self.y_pred_strobe == 0).sum()
        detected   = (self.y_pred_orig != 0).sum()
        return (
            f"\n{'='*60}\n"
            f"ATAQUE: {self.attack_name}\n"
            f"{'='*60}\n"
            f"  Flujos originales detectados : {detected:,}/{n:,}\n"
            f"  Duty cycle (activo/silencio) : {self.duty_cycle:.0%}\n"
            f"  Patrón strobe               : {self.pattern}\n"
            f"  Evasiones exitosas          : {evaded:,} ({self.asr*100:.1f}%)\n"
            f"\n  Degradación features BUF_* (original → strobe):\n"
            + "\n".join(
                f"    {k:<25} {v['original']:>8.3f} → {v['strobe']:>8.3f} "
                f"(-{v['reduction']*100:.1f}%)"
                for k, v in self.buf_degradation.items()
            )
        )

    def evasion_by_class(self, class_names: Optional[dict] = None) -> str:
        lines = ["\n  Evasión por clase:"]
        mask_detected = self.y_pred_orig != 0
        for cls in np.unique(self.y_true):
            mask  = (self.y_true == cls) & mask_detected
            if mask.sum() == 0:
                continue
            evaded = (self.y_pred_strobe[mask] == 0).sum()
            total  = mask.sum()
            rate   = evaded / total * 100
            name   = class_names.get(int(cls), f"Clase {cls}") \
                     if class_names else f"Clase {cls}"
            lines.append(f"    {name:<25} {evaded:>5}/{total:<5} ({rate:.1f}%)")
        return "\n".join(lines)


# ===========================================================================
# SIMULADOR TRH
# ===========================================================================

class TRHSimulator:
    """
    Temporal Resonance Hijacking — aliasing del IPBehaviorBuffer.

    Parámetros
    ----------
    buffer       : IPBehaviorBuffer — el buffer real del pipeline
    constraints  : DomainConstraints — para recalcular features derivadas
    duty_cycle   : fracción de eventos del flujo que sobreviven [0.1, 0.9]
    pattern      : 'uniform' | 'burst' | 'periodic'
    n_on         : eventos activos por ciclo (solo para pattern='periodic')
    n_off        : eventos silenciados por ciclo (solo para pattern='periodic')
    seed         : semilla para reproducibilidad
    verbose      : mostrar progreso
    """

    def __init__(
        self,
        buffer,
        constraints  : DomainConstraints,
        duty_cycle   : float = 0.3,
        pattern      : str   = 'periodic',
        n_on         : int   = 3,
        n_off        : int   = 7,
        seed         : int   = 42,
        verbose      : bool  = True,
    ):
        self.buffer     = buffer
        self.dc         = constraints
        self.duty_cycle = duty_cycle
        self.pattern    = pattern
        self.n_on       = n_on
        self.n_off      = n_off
        self.seed       = seed
        self.verbose    = verbose

        # Features BUF_* a monitorizar para medir degradación
        self.buf_monitor = [
            'BUF_SCAN_RATE',
            'BUF_FLOW_COUNT',
            'BUF_UNIQUE_DST_PORTS',
            'BUF_NO_RESP_RATIO',
            'BUF_SMALL_PKT_RATIO',
        ]

    # ------------------------------------------------------------------
    # INTERFAZ PÚBLICA
    # ------------------------------------------------------------------
    def run(
        self,
        X_flows_raw  : np.ndarray,
        y            : np.ndarray,
        model        : object,
        class_names  : Optional[dict] = None,
    ) -> TRHResult:
        """
        Ejecuta la simulación TRH sobre los flujos de ataque.

        A diferencia de los ataques de gradiente, el TRH opera sobre
        los flujos en espacio FÍSICO ORIGINAL (pre-escalado), los pasa
        por el buffer real, y obtiene features BUF_* reales — no
        aproximaciones.

        Parámetros
        ----------
        X_flows_raw : (n, 66) en espacio ORIGINAL sin escalar
                      Usar X_train_raw_reconstructed.npy o equivalente
        y           : (n,) etiquetas reales
        model       : modelo con .predict_proba(X_scaled) → (n, n_classes)
        class_names : dict {int: str} para el resumen por clase

        Retorna
        -------
        TRHResult con flujos strobe, predicciones y métricas de degradación
        """
        if self.verbose:
            print(f"\n[TRH] Temporal Resonance Hijacking | "
                  f"duty_cycle={self.duty_cycle:.0%} | pattern={self.pattern}")
            print(f"  Flujos a procesar: {len(X_flows_raw):,}")

        # 1. Escalar flujos originales y obtener predicciones de referencia
        X_orig_sc   = self.dc.to_scaled_space(X_flows_raw)
        y_pred_orig = np.argmax(model.predict_proba(X_orig_sc), axis=1)
        detected    = (y_pred_orig != 0).sum()

        if self.verbose:
            print(f"  Detectados como ataque: {detected:,}/{len(y):,}")

        # 2. Aplicar patrón strobe sobre los flujos en espacio físico
        X_strobe_raw = self._apply_strobe(X_flows_raw)

        # 3. Recalcular features derivadas para coherencia física
        X_strobe_raw = self.dc.apply_causal_graph(X_strobe_raw)

        # 4. Escalar flujos strobe
        X_strobe_sc = self.dc.to_scaled_space(X_strobe_raw)

        # 5. Predicciones sobre flujos strobe
        y_pred_strobe = np.argmax(model.predict_proba(X_strobe_sc), axis=1)

        # 6. Métricas
        mask_detected = y_pred_orig != 0
        asr = (y_pred_strobe[mask_detected] == 0).mean() \
              if mask_detected.sum() > 0 else 0.0

        # 7. Degradación de features BUF_*
        buf_degradation = self._compute_buf_degradation(
            X_orig_sc, X_strobe_sc
        )

        result = TRHResult(
            X_strobe        = X_strobe_sc,
            X_original      = X_orig_sc,
            y_true          = y,
            y_pred_orig     = y_pred_orig,
            y_pred_strobe   = y_pred_strobe,
            asr             = asr,
            duty_cycle      = self.duty_cycle,
            pattern         = self.pattern,
            buf_degradation = buf_degradation,
        )

        if self.verbose:
            print(result.summary())
            if class_names:
                print(result.evasion_by_class(class_names))

        return result

    def run_duty_sweep(
        self,
        X_flows_raw  : np.ndarray,
        y            : np.ndarray,
        model        : object,
        duty_cycles  : list[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
        class_names  : Optional[dict] = None,
    ) -> dict:
        """
        Análisis de sensibilidad al duty_cycle.

        Evalúa múltiples valores de duty_cycle y devuelve las métricas
        para cada uno. Permite identificar el umbral crítico donde el
        modelo deja de detectar el ataque — la "frecuencia de resonancia"
        del buffer.

        El plot resultante (ASR vs duty_cycle) es la curva de Nyquist
        del sistema NIDS — contenido directo para la memoria.

        Retorna
        -------
        dict {duty_cycle: TRHResult}
        """
        if self.verbose:
            print(f"\n[TRH] Duty cycle sweep: {duty_cycles}")
            print(f"{'DC':>6} | {'ASR':>8} | "
                  f"{'BUF_SCAN_RATE↓':>16} | {'BUF_FLOW_COUNT↓':>16}")
            print("-" * 55)

        results = {}
        for dc_val in duty_cycles:
            sim = TRHSimulator(
                self.buffer, self.dc,
                duty_cycle=dc_val,
                pattern=self.pattern,
                n_on=self.n_on,
                n_off=self.n_off,
                seed=self.seed,
                verbose=False,
            )
            result = sim.run(X_flows_raw, y, model, class_names)
            results[dc_val] = result

            scan_rate_red = result.buf_degradation.get(
                'BUF_SCAN_RATE', {}).get('reduction', 0)
            flow_count_red = result.buf_degradation.get(
                'BUF_FLOW_COUNT', {}).get('reduction', 0)

            if self.verbose:
                print(f"  {dc_val:>4.0%} | {result.asr*100:>7.1f}% | "
                      f"{scan_rate_red*100:>14.1f}% | "
                      f"{flow_count_red*100:>14.1f}%")

        return results

    # ------------------------------------------------------------------
    # MOTOR STROBE
    # ------------------------------------------------------------------
    def _apply_strobe(self, X_raw: np.ndarray) -> np.ndarray:
        """
        Aplica el patrón stroboscópico sobre los flujos en espacio físico.

        El patrón simula la visión del buffer cuando el atacante envía
        tráfico en ráfagas sincronizadas con la ventana temporal.

        Para cada flujo, las features de VOLUMEN (conteos acumulados)
        se escalan por el duty_cycle — equivale a que el buffer solo
        "vio" ese porcentaje del tráfico real.

        Las features de TASA y ESTADÍSTICAS se mantienen porque la
        velocidad y distribución de los paquetes activos no cambia.

        Parámetros
        ----------
        X_raw : (n, 66) en espacio físico original

        Retorna
        -------
        X_strobe : (n, 66) con features de volumen escaladas por duty_cycle
        """
        np.random.seed(self.seed)
        X_strobe = X_raw.copy()

        if self.pattern == 'uniform':
            # Eliminación aleatoria uniforme
            # Cada flujo pierde (1-duty_cycle) de sus eventos aleatoriamente
            scale = self._uniform_scale(len(X_raw))

        elif self.pattern == 'burst':
            # Ráfagas cortas separadas por silencios largos
            # El atacante concentra el tráfico en ventanas cortas
            scale = self._burst_scale(len(X_raw))

        elif self.pattern == 'periodic':
            # Patrón regular N_on activo / N_off silencio
            # Más realista para un atacante con script de temporización
            scale = self._periodic_scale(len(X_raw))

        else:
            raise ValueError(f"pattern debe ser 'uniform', 'burst' o 'periodic'")

        # Aplicar escala a features de volumen
        # (conteos acumulados que el buffer habría observado)
        vol_features = [
            'IN_BYTES', 'IN_PKTS', 'OUT_BYTES', 'OUT_PKTS',
            'FLOW_DURATION_MILLISECONDS', 'DURATION_IN', 'DURATION_OUT',
            'NUM_PKTS_UP_TO_128_BYTES', 'NUM_PKTS_128_TO_256_BYTES',
            'NUM_PKTS_256_TO_512_BYTES', 'NUM_PKTS_512_TO_1024_BYTES',
            'NUM_PKTS_1024_TO_1514_BYTES',
            'BUF_FLOW_COUNT', 'BUF_UNIQUE_DST_PORTS', 'BUF_UNIQUE_DST_IPS',
            'BUF_BURST_PORTS',
        ]

        for feat in vol_features:
            idx = self.dc._feat_idx(feat)
            if idx is not None:
                X_strobe[:, idx] = X_raw[:, idx] * scale

        # BUF_SCAN_RATE — caso especial: flujos/segundo
        # Si el atacante reduce su tasa durante el silencio, la scan rate baja
        idx_sr = self.dc._feat_idx('BUF_SCAN_RATE')
        if idx_sr is not None:
            X_strobe[:, idx_sr] = X_raw[:, idx_sr] * scale

        # Garantizar mínimos físicos (no puede haber 0 paquetes si hay bytes)
        idx_pkts = self.dc._feat_idx('IN_PKTS')
        if idx_pkts is not None:
            X_strobe[:, idx_pkts] = np.maximum(X_strobe[:, idx_pkts], 1.0)

        return X_strobe

    def _uniform_scale(self, n: int) -> np.ndarray:
        """Escala uniforme — cada flujo tiene varianza alrededor del duty_cycle."""
        # Pequeña varianza para simular que no todos los flujos
        # son atacados exactamente igual
        scale = np.random.normal(
            loc=self.duty_cycle,
            scale=self.duty_cycle * 0.1,
            size=n,
        ).clip(0.05, 1.0)
        return scale.astype(np.float32)

    def _burst_scale(self, n: int) -> np.ndarray:
        """
        Ráfagas cortas: el atacante concentra todo en ventanas breves.
        Algunos flujos tienen duty_cycle alto (cuando coinciden con la ráfaga)
        y otros muy bajo (cuando caen en el silencio).
        """
        scale = np.where(
            np.random.random(n) < self.duty_cycle,
            np.random.uniform(0.8, 1.0, n),    # ráfaga activa
            np.random.uniform(0.02, 0.1, n),   # silencio casi total
        )
        return scale.astype(np.float32)

    def _periodic_scale(self, n: int) -> np.ndarray:
        """
        Patrón regular N_on / N_off — el más realista para un script de ataque.
        El duty_cycle efectivo = n_on / (n_on + n_off).
        """
        period = self.n_on + self.n_off
        # Posición de cada flujo dentro del ciclo periódico
        position = np.arange(n) % period
        scale = np.where(
            position < self.n_on,
            np.ones(n),          # activo — escala completa
            np.zeros(n) + 0.02,  # silencio — casi cero
        )
        # Suavizar ligeramente para simular transiciones reales
        scale = scale + np.random.normal(0, 0.02, n)
        scale = scale.clip(0.02, 1.0)
        return scale.astype(np.float32)

    # ------------------------------------------------------------------
    # MÉTRICAS DE DEGRADACIÓN BUF_*
    # ------------------------------------------------------------------
    def _compute_buf_degradation(
        self,
        X_orig_sc   : np.ndarray,
        X_strobe_sc : np.ndarray,
    ) -> dict:
        """
        Calcula la degradación media de las features BUF_* clave.

        Retorna un dict con la reducción porcentual de cada feature
        monitoreada — permite cuantificar cuánto "cegó" el strobe
        al buffer del IDS.
        """
        degradation = {}
        for feat in self.buf_monitor:
            idx = self.dc._feat_idx(feat)
            if idx is None:
                continue
            orig_mean   = float(X_orig_sc[:, idx].mean())
            strobe_mean = float(X_strobe_sc[:, idx].mean())
            reduction   = max(0.0, (orig_mean - strobe_mean) /
                              (abs(orig_mean) + 1e-8))
            degradation[feat] = {
                'original'  : orig_mean,
                'strobe'    : strobe_mean,
                'reduction' : reduction,
            }
        return degradation


# ===========================================================================
# SCRIPT DE VERIFICACIÓN
# ===========================================================================

if __name__ == "__main__":
    from src.utils.domain_constraints import DomainConstraints
    from src.ip_behavior_buffer import IPBehaviorBuffer
    import numpy as np

    print("[-] Verificando TRHSimulator...")

    dc     = DomainConstraints.from_artifacts()
    buffer = IPBehaviorBuffer(window_seconds=120, max_ips=200_000)

    for pattern in ('uniform', 'burst', 'periodic'):
        trh = TRHSimulator(
            buffer, dc,
            duty_cycle=0.3,
            pattern=pattern,
            verbose=False,
        )
        print(f"   [✓] pattern='{pattern}' instanciado")

    print(f"\n   Features BUF_* monitoreadas: {len(trh.buf_monitor)}")
    print("\n[✓] trh_simulator.py listo — usar con X_train_raw_reconstructed.npy")
    print("    Recuerda: X_flows_raw debe estar en espacio FÍSICO, no escalado")