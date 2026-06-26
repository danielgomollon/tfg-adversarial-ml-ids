"""
src/attacks/tcp_poisoning.py
================================================================
TCP — Temporal Context Poisoning
(No confundir con el protocolo de red TCP)

Contribución 100% original — BigFlow-NIDS TFG (2025-2026)

═══════════════════════════════════════════════════════════════
PARADIGMA: Envenenamiento de Contexto Temporal
═══════════════════════════════════════════════════════════════

Todos los ataques adversariales conocidos modifican el flujo
malicioso en el momento del ataque (disfraz activo).

El TCP invierte este paradigma:
  El flujo malicioso NO se modifica.
  El atacante construye un historial benigno deliberado en el
  IPBehaviorBuffer ANTES de lanzar el ataque real.

Cuando el ataque llega, las features BUF_* reflejan semanas
de comportamiento benigno — el flujo malicioso es un punto
aislado en un historial masivamente benigno.

═══════════════════════════════════════════════════════════════
ANALOGÍA: La Coartada del APT
═══════════════════════════════════════════════════════════════

Todos los ataques actuales:
  Día del robo → el ladrón se disfraza

TCP:
  Semanas antes  → el ladrón construye una coartada
  Día del robo   → entra sin disfraz, pero con reputación limpia

Esto es exactamente lo que hacen los APT reales:
  - Semanas de tráfico benigno desde la IP comprometida
  - Luego exfiltración en un único flujo
  - El IDS ve: historial impecable + un flujo raro
                → decisión: benigno (el historial domina)

═══════════════════════════════════════════════════════════════
MATEMÁTICA DEL DILUTION RATIO
═══════════════════════════════════════════════════════════════

Sea H(IP, t) el historial del buffer en el instante t.
Las features BUF_* son funciones de H:

  BUF_SCAN_RATE(t)    = flow_count(H) / elapsed_time(H)
  BUF_NO_RESP_RATIO(t)= no_resp(H)    / flow_count(H)
  BUF_FLOW_COUNT(t)   = flow_count(H)

Con N_warmup flujos benignos antes del ataque:

  BUF_FLOW_COUNT(t_ataque) = N_warmup + 1
  BUF_NO_RESP_RATIO(t_ataque) ≈ 0 / (N_warmup + 1) → 0

El único flujo malicioso no puede mover significativamente
ninguna feature BUF_* si N_warmup >> 1.

Dilution Ratio (DR) = N_warmup / 1 (flujos benignos / malicioso)
  → Cuantifica el "coste de reputación" del ataque
  → DR mínimo para evasión = contribución medible del TFG

═══════════════════════════════════════════════════════════════
DIFERENCIA CON TRH Y ATF
═══════════════════════════════════════════════════════════════

TRH: manipula la densidad temporal del tráfico malicioso
     (el ataque llega, pero el buffer lo ve esporádico)

ATF: divide el flujo malicioso en dos sub-flujos
     (el extractor genera dos filas incompletas)

TCP: construye historial benigno ANTES del ataque
     (el flujo malicioso llega sin modificar, pero el
      buffer está "inundado" de reputación benigna)

Son tres vectores de ataque ortogonales sobre el pipeline.

═══════════════════════════════════════════════════════════════
IMPLEMENTACIÓN SOBRE EL DATASET
═══════════════════════════════════════════════════════════════

Simulamos el warmup benigno modificando el estado del buffer
ANTES de procesar el flujo malicioso:

  1. Para cada flujo de ataque X_attack[i]:
     a. Sintetizar N_warmup flujos benignos representativos
     b. Pre-procesar esos flujos por el buffer (warmup)
     c. Extraer las features BUF_* del flujo malicioso
        con el buffer ya "caliente" de historial benigno
     d. Resetear el buffer para el siguiente ataque

  Las features non-BUF del flujo malicioso no se modifican —
  el ataque real llega intacto. Solo el contexto temporal cambia.

Uso:
    tcp = TCPoisoning(buffer, dc, n_warmup=50)
    result = tcp.run(X_attacks_raw, y_attacks, model)

    # Análisis del Dilution Ratio
    sweep = tcp.run_dr_sweep(X_attacks_raw, y_attacks, model)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional

from src.utils.domain_constraints import DomainConstraints


# ===========================================================================
# RESULTADO TCP
# ===========================================================================

@dataclass
class TCPResult:
    """
    Resultado del Temporal Context Poisoning.

    La métrica más importante es buf_contamination:
    cuánto han cambiado las features BUF_* respecto al flujo original.
    Si cambian mucho con poco warmup, el buffer es vulnerable.
    """
    X_poisoned       : np.ndarray    # flujos con contexto envenenado (escalado)
    X_original       : np.ndarray    # flujos originales sin modificar
    y_true           : np.ndarray

    y_pred_orig      : np.ndarray    # predicciones sobre flujos sin contexto
    y_pred_poisoned  : np.ndarray    # predicciones tras envenenar el contexto

    asr              : float
    n_warmup         : int           # flujos benignos inyectados por ataque
    dilution_ratio   : float         # n_warmup / 1
    buf_shift        : dict          # cambio medio en features BUF_* clave
    attack_name      : str = "TCP — Temporal Context Poisoning"

    def summary(self) -> str:
        n        = len(self.y_true)
        evaded   = (self.y_pred_poisoned == 0).sum()
        detected = (self.y_pred_orig != 0).sum()
        lines    = [
            f"\n{'='*60}",
            f"ATAQUE: {self.attack_name}",
            f"{'='*60}",
            f"  Flujos originales detectados : {detected:,}/{n:,}",
            f"  Dilution Ratio (DR)          : {self.dilution_ratio:.0f}x",
            f"  Warmup benignos por ataque   : {self.n_warmup:,}",
            f"  Evasiones exitosas           : {evaded:,} ({self.asr*100:.1f}%)",
            f"\n  Desplazamiento features BUF_* (original → envenenado):",
        ]
        for feat, info in self.buf_shift.items():
            lines.append(
                f"    {feat:<25} {info['before']:>8.3f} → {info['after']:>8.3f} "
                f"  Δ={info['delta']:>+8.3f}"
            )
        return "\n".join(lines)

    def evasion_by_class(self, class_names: Optional[dict] = None) -> str:
        lines        = ["\n  Evasión por clase:"]
        mask_detected = self.y_pred_orig != 0
        for cls in np.unique(self.y_true):
            mask   = (self.y_true == cls) & mask_detected
            if mask.sum() == 0:
                continue
            evaded = (self.y_pred_poisoned[mask] == 0).sum()
            total  = mask.sum()
            rate   = evaded / total * 100
            name   = class_names.get(int(cls), f"Clase {cls}") \
                     if class_names else f"Clase {cls}"
            lines.append(f"    {name:<25} {evaded:>5}/{total:<5} ({rate:.1f}%)")
        return "\n".join(lines)


# ===========================================================================
# SIMULADOR TCP
# ===========================================================================

class TCPoisoning:
    """
    Temporal Context Poisoning — envenenamiento del historial del buffer.

    Parámetros
    ----------
    buffer       : IPBehaviorBuffer — el buffer real del pipeline
    constraints  : DomainConstraints
    n_warmup     : flujos benignos sintéticos a inyectar antes de cada ataque
    benign_proto : prototipo de flujo benigno en espacio FÍSICO
                   Si None, se sintetiza desde X_benign_raw
    seed         : reproducibilidad
    verbose      : mostrar progreso
    """

    # Features BUF_* que monitorizamos para medir el envenenamiento
    BUF_MONITOR = [
        'BUF_SCAN_RATE',
        'BUF_FLOW_COUNT',
        'BUF_UNIQUE_DST_PORTS',
        'BUF_NO_RESP_RATIO',
        'BUF_SMALL_PKT_RATIO',
        'BUF_RECON_SCORE',
    ]

    def __init__(
        self,
        buffer,
        constraints  : DomainConstraints,
        n_warmup     : int            = 50,
        X_benign_raw : Optional[np.ndarray] = None,
        seed         : int            = 42,
        verbose      : bool           = True,
    ):
        self.buffer      = buffer
        self.dc          = constraints
        self.n_warmup    = n_warmup
        self.seed        = seed
        self.verbose     = verbose

        # Pre-computar prototipos benignos para el warmup
        # Usamos múltiples prototipos para variar el warmup y no
        # generar patrones repetitivos que el modelo pueda detectar
        if X_benign_raw is not None:
            self._benign_pool = X_benign_raw
        else:
            self._benign_pool = None

        np.random.seed(seed)

    # ------------------------------------------------------------------
    # INTERFAZ PÚBLICA
    # ------------------------------------------------------------------
    def run(
        self,
        X_attacks_raw : np.ndarray,
        y             : np.ndarray,
        model         : object,
        class_names   : Optional[dict] = None,
    ) -> TCPResult:
        """
        Ejecuta el TCP sobre los flujos de ataque.

        El flujo malicioso NO se modifica — solo se envenena el
        contexto temporal del buffer antes de cada clasificación.

        Parámetros
        ----------
        X_attacks_raw : (n, 66) en espacio FÍSICO original
        y             : (n,) etiquetas reales
        model         : con .predict_proba(X_scaled)
        class_names   : dict {int: str}
        """
        if self.verbose:
            print(f"\n[TCP] Temporal Context Poisoning | "
                  f"n_warmup={self.n_warmup} | DR={self.n_warmup:.0f}x")
            print(f"  Flujos de ataque: {len(X_attacks_raw):,}")
            print(f"  El flujo malicioso NO se modifica — "
                  f"solo el contexto temporal")

        # 1. Predicciones originales (sin contexto envenenado)
        X_orig_sc   = self.dc.to_scaled_space(X_attacks_raw)
        y_pred_orig = np.argmax(model.predict_proba(X_orig_sc), axis=1)
        detected    = (y_pred_orig != 0).sum()

        if self.verbose:
            print(f"  Detectados como ataque: {detected:,}/{len(y):,}")

        # 2. Generar flujos con contexto envenenado
        if self.verbose:
            print(f"\n  Envenenando contexto temporal...")

        X_poisoned_sc = self._poison_context(X_attacks_raw)

        # 3. Predicciones tras envenenar
        y_pred_poisoned = np.argmax(
            model.predict_proba(X_poisoned_sc), axis=1
        )

        # 4. Métricas
        mask_detected   = y_pred_orig != 0
        asr = (y_pred_poisoned[mask_detected] == 0).mean() \
              if mask_detected.sum() > 0 else 0.0

        # 5. Desplazamiento de features BUF_*
        buf_shift = self._compute_buf_shift(X_orig_sc, X_poisoned_sc)

        result = TCPResult(
            X_poisoned      = X_poisoned_sc,
            X_original      = X_orig_sc,
            y_true          = y,
            y_pred_orig     = y_pred_orig,
            y_pred_poisoned = y_pred_poisoned,
            asr             = asr,
            n_warmup        = self.n_warmup,
            dilution_ratio  = float(self.n_warmup),
            buf_shift       = buf_shift,
        )

        if self.verbose:
            print(result.summary())
            if class_names:
                print(result.evasion_by_class(class_names))

        return result

    def run_dr_sweep(
        self,
        X_attacks_raw : np.ndarray,
        y             : np.ndarray,
        model         : object,
        warmup_values : list[int]  = [1, 5, 10, 25, 50, 100, 200, 500],
        class_names   : Optional[dict] = None,
    ) -> dict:
        """
        Análisis del Dilution Ratio — curva ASR vs n_warmup.

        Esta es la contribución medible central del TCP:
        encuentra el DR mínimo para evasión, que cuantifica el
        "coste de reputación" que debe pagar el atacante.

        Un DR bajo (ej. 10) significa que bastan 10 flujos benignos
        previos para engañar al IDS — vulnerabilidad crítica.
        Un DR alto (ej. 500) significa que el atacante necesita
        semanas de preparación — el buffer es robusto.

        Retorna
        -------
        dict {n_warmup: TCPResult}
        """
        if self.verbose:
            print(f"\n[TCP] Dilution Ratio sweep: {warmup_values}")
            print(f"{'DR':>6} | {'ASR':>8} | "
                  f"{'BUF_SCAN_RATE Δ':>16} | {'BUF_RECON_SCORE Δ':>18}")
            print("-" * 55)

        results = {}
        for n_warmup in warmup_values:
            sim = TCPoisoning(
                self.buffer, self.dc,
                n_warmup    = n_warmup,
                X_benign_raw= self._benign_pool,
                seed        = self.seed,
                verbose     = False,
            )
            result = sim.run(X_attacks_raw, y, model, class_names)
            results[n_warmup] = result

            sr_delta    = result.buf_shift.get('BUF_SCAN_RATE', {}).get('delta', 0)
            recon_delta = result.buf_shift.get('BUF_RECON_SCORE', {}).get('delta', 0)

            if self.verbose:
                print(f"  {n_warmup:>4}x | {result.asr*100:>7.1f}% | "
                      f"{sr_delta:>+15.3f} | {recon_delta:>+17.3f}")

        # Encontrar DR mínimo para ASR > 50%
        dr_min = None
        for nw, res in sorted(results.items()):
            if res.asr > 0.5:
                dr_min = nw
                break

        if self.verbose and dr_min is not None:
            print(f"\n  [✓] DR mínimo para ASR>50%: {dr_min}x")
            print(f"      Interpretación: el atacante necesita enviar "
                  f"{dr_min} flujos benignos antes de cada ataque")
        elif self.verbose:
            print(f"\n  [!] ASR < 50% en todo el rango — buffer robusto")

        return results

    # ------------------------------------------------------------------
    # MOTOR DE ENVENENAMIENTO
    # ------------------------------------------------------------------
    def _poison_context(self, X_attacks_raw: np.ndarray) -> np.ndarray:
        """
        Para cada flujo de ataque:
          1. Resetear el buffer (simulación de IP nueva)
          2. Inyectar n_warmup flujos benignos (construir reputación)
          3. Extraer las features BUF_* del flujo malicioso
             con el buffer ya caliente de historial benigno
          4. Combinar features BUF_* envenenadas con features
             non-BUF originales del flujo malicioso

        El resultado es un flujo donde:
          - Features de red (IN_BYTES, IN_PKTS, etc.) = ataque real
          - Features BUF_* = reflejo de historial benigno previo
        """
        n = len(X_attacks_raw)
        X_poisoned_raw = X_attacks_raw.copy()

        # Pool de flujos benignos para el warmup
        if self._benign_pool is not None:
            benign_pool = self._benign_pool
        else:
            # Sintetizar benignos básicos si no hay pool
            benign_pool = self._synthesize_benign_pool(
                n_samples=max(1000, self.n_warmup * 10),
                feature_dim=X_attacks_raw.shape[1],
            )

        # Índices de features BUF_* en el array de features
        buf_indices = self.dc.buf_indices

        if self.verbose:
            print(f"    Procesando {n:,} flujos con DR={self.n_warmup}x...",
                  end="", flush=True)

        for i in range(n):
            # 1. Buffer limpio para esta IP (simula IP nueva)
            self.buffer.reset()

            # 2. Inyectar warmup benigno
            warmup_idx    = np.random.choice(
                len(benign_pool), size=self.n_warmup, replace=True
            )
            warmup_flows  = benign_pool[warmup_idx]

            # Convertir a DataFrame para el buffer
            warmup_df = self._array_to_buffer_df(warmup_flows)
            warmup_result = self.buffer.update_and_extract(warmup_df)

            # 3. Procesar el flujo malicioso real a través del buffer caliente
            attack_df     = self._array_to_buffer_df(
                X_attacks_raw[i:i+1]
            )
            attack_result = self.buffer.update_and_extract(attack_df)

            # 4. Extraer solo las features BUF_* del resultado
            # (las otras features vienen del flujo malicioso original)
            buf_feature_names = [
                'BUF_UNIQUE_DST_PORTS', 'BUF_UNIQUE_DST_IPS',
                'BUF_FLOW_COUNT', 'BUF_NO_RESP_RATIO', 'BUF_SCAN_RATE',
                'BUF_SMALL_PKT_RATIO', 'BUF_PORT_STD', 'BUF_PORT_RANGE',
                'BUF_IS_SCANNER', 'BUF_HTTP_RATIO', 'BUF_HTTP_BYTES_AVG',
                'BUF_HTTP_SMALL_RATIO', 'BUF_BURST_PORTS',
                'BUF_SYN_ACK_RST_RATIO', 'BUF_RECON_SCORE',
            ]

            for j, feat in enumerate(buf_feature_names):
                if feat in attack_result.columns:
                    feat_idx = self.dc._feat_idx(feat)
                    if feat_idx is not None:
                        # Actualizar solo la feature BUF_* en el vector de ataque
                        X_poisoned_raw[i, feat_idx] = \
                            float(attack_result[feat].iloc[0])

        self.buffer.reset()  # limpiar al terminar

        if self.verbose:
            print(" ✓")

        # Recalcular derivadas y escalar
        X_poisoned_raw = self.dc.apply_causal_graph(X_poisoned_raw)
        return self.dc.to_scaled_space(X_poisoned_raw)

    def _array_to_buffer_df(self, X_raw: np.ndarray):
        """
        Convierte un array numpy (n, 66) al DataFrame que espera
        IPBehaviorBuffer.update_and_extract().

        Mapea las columnas usando feature_names del DomainConstraints.
        """
        import polars as pl

        feature_names = self.dc.feature_names

        # Columnas que necesita el buffer
        required_buf_cols = [
            'IPV4_SRC_ADDR', 'IPV4_DST_ADDR', 'L4_DST_PORT',
            'OUT_PKTS', 'IN_BYTES', 'FLOW_START_MILLISECONDS',
            'L7_PROTO', 'TCP_FLAGS',
        ]

        # Construir dict de columnas disponibles en X_raw
        data = {}
        for i, name in enumerate(feature_names):
            data[name] = X_raw[:, i].astype(np.float64)

        # Añadir columnas del buffer que no están en features
        # (IP addresses y timestamps — sintéticos para la simulación)
        n = len(X_raw)
        data['IPV4_SRC_ADDR']           = np.full(n, 167772161.0)  # 10.0.0.1
        data['IPV4_DST_ADDR']           = np.full(n, 167772162.0)  # 10.0.0.2
        data['FLOW_START_MILLISECONDS'] = (
            np.arange(n, dtype=np.float64) * 1000.0
        )  # timestamps sintéticos 0, 1s, 2s...

        return pl.DataFrame(data)

    def _synthesize_benign_pool(
        self,
        n_samples  : int,
        feature_dim: int,
    ) -> np.ndarray:
        """
        Genera flujos benignos sintéticos básicos cuando no hay pool real.
        Usa valores típicos de tráfico HTTP benigno.
        """
        pool = np.zeros((n_samples, feature_dim), dtype=np.float64)

        # Valores físicos típicos de tráfico benigno HTTP
        defaults = {
            'IN_BYTES'                   : 1500.0,
            'IN_PKTS'                    : 5.0,
            'OUT_BYTES'                  : 800.0,
            'OUT_PKTS'                   : 4.0,
            'FLOW_DURATION_MILLISECONDS' : 200.0,
            'L4_DST_PORT'               : 80.0,
            'PROTOCOL'                  : 6.0,
            'L7_PROTO'                  : 7.0,
            'TCP_FLAGS'                 : 24.0,
        }

        for feat, val in defaults.items():
            idx = self.dc._feat_idx(feat)
            if idx is not None:
                noise = np.random.normal(1.0, 0.1, n_samples)
                pool[:, idx] = val * np.clip(noise, 0.5, 2.0)

        return pool

    # ------------------------------------------------------------------
    # MÉTRICAS
    # ------------------------------------------------------------------
    def _compute_buf_shift(
        self,
        X_orig_sc    : np.ndarray,
        X_poisoned_sc: np.ndarray,
    ) -> dict:
        """Calcula el desplazamiento medio de features BUF_* clave."""
        shift = {}
        for feat in self.BUF_MONITOR:
            idx = self.dc._feat_idx(feat)
            if idx is None:
                continue
            before = float(X_orig_sc[:, idx].mean())
            after  = float(X_poisoned_sc[:, idx].mean())
            shift[feat] = {
                'before' : before,
                'after'  : after,
                'delta'  : after - before,
            }
        return shift


# ===========================================================================
# SCRIPT DE VERIFICACIÓN
# ===========================================================================

if __name__ == "__main__":
    from src.utils.domain_constraints import DomainConstraints
    from src.ip_behavior_buffer import IPBehaviorBuffer
    import numpy as np

    print("[-] Verificando TCPoisoning...")

    dc     = DomainConstraints.from_artifacts()
    buffer = IPBehaviorBuffer(window_seconds=120, max_ips=200_000)

    tcp = TCPoisoning(buffer, dc, n_warmup=50, verbose=True)
    print(f"   [✓] Instanciado: DR={tcp.n_warmup}x")
    print(f"   BUF features monitorizadas: {len(tcp.BUF_MONITOR)}")

    print("\n[✓] tcp_poisoning.py listo")
    print("    IMPORTANTE: X_attacks_raw debe estar en espacio FÍSICO")
    print("    Uso:")
    print("      tcp    = TCPoisoning(buffer, dc, n_warmup=50)")
    print("      result = tcp.run(X_attacks_raw, y_attacks, model)")
    print("      sweep  = tcp.run_dr_sweep(X_attacks_raw, y_attacks, model)")