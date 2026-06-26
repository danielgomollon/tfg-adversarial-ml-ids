"""
src/attacks/tcp_poisoning.py
================================================================
TCP — Temporal Context Poisoning
Contribución 100% original — BigFlow-NIDS TFG (2025-2026)

MODELO DE AMENAZA (Grey-Box Realista):
  El atacante SABE:
    - Que los IDS modernos usan contexto temporal de IP
    - Que construir reputación previa ayuda a evadir
    - El protocolo y puerto de su propio ataque
    - Si un flujo fue detectado o no (oráculo binario)
    - Que puede rotar IPs para evitar que el historial
      acumulado lo delate tras reintentos fallidos

  El atacante NO SABE:
    - Que existe un IPBehaviorBuffer con ventana de 120s
    - Las features BUF_* ni su fórmula interna
    - El DR óptimo necesario
    - Los parámetros internos del modelo ni sus scalers

Referencias:
- Biggio et al. (2012) - "Poisoning Attacks against Support Vector Machines". 
(Paper clásico sobre envenenamiento de IA).
"""

from __future__ import annotations

import numpy as np
import polars as pl
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum

from src.utils.domain_constraints import DomainConstraints

# ===========================================================================
# POOL DE IPs SINTÉTICAS PARA ROTACIÓN
# Rango 10.0.0.0/8 — IPs privadas, operacionalmente realistas
# El atacante controla una botnet o usa proxies en este rango
# ===========================================================================

def _generate_ip_pool(n: int, rng: np.random.Generator) -> np.ndarray:
    """
    Genera un pool de IPs sintéticas en el rango 10.x.x.x.
    Cada IP tiene su propio historial independiente en el buffer.
    """
    # Rango: 10.0.1.1 — 10.0.255.254 (evitamos .0 y .255)
    octets_b = rng.integers(0, 256, size=n)
    octets_c = rng.integers(1, 255, size=n)
    octets_d = rng.integers(1, 255, size=n)
    # Convertir a float (formato que espera el buffer)
    ips = (10 * 256**3
           + octets_b * 256**2
           + octets_c * 256
           + octets_d).astype(np.float64)
    return ips


# ===========================================================================
# ENUMS
# ===========================================================================

class WarmupStrategy(Enum):
    CONSERVATIVE = "conservative"   # pocos flujos, muy espaciados
    MODERATE     = "moderate"       # balance velocidad/seguridad
    AGGRESSIVE   = "aggressive"     # muchos flujos, más rápido
    ADAPTIVE     = "adaptive"       # aprende del feedback del oráculo


class OracleResponse(Enum):
    BENIGN = 0
    ATTACK = 1


# ===========================================================================
# TCP ADVERSARY — PERSPECTIVA DEL ATACANTE (Grey-Box)
# ===========================================================================

class TCPAdversary:
    """
    Motor de decisión del atacante.

    Opera en grey-box puro: solo observa si sus flujos son detectados.
    No conoce el buffer, sus features, ni sus parámetros internos.

    Innovación: IP Rotation con Reputation Ledger.
    El atacante mantiene un registro interno de qué IPs tienen
    reputación "limpia" (no detectadas) y cuáles están "quemadas"
    (detectadas repetidamente). Rota IPs cuando una IP acumula
    demasiadas detecciones — exactamente como hace una botnet real.

    Parámetros
    ----------
    strategy           : estrategia inicial de warmup
    max_oracle_budget  : máximo de consultas totales al oráculo
    ip_pool_size       : IPs disponibles para rotación
    burn_threshold     : detecciones antes de quemar una IP
    backoff_factor     : multiplica el warmup al detectar fallo
    patience           : intentos por flujo antes de rotar IP
    """

    _PROFILES = {
        'http'   : {'port': 80.0,  'l7': 7.0,  'iat_mean': 8.0,
                    'bytes_mean': 2400.0, 'pkts_mean': 8.0},
        'https'  : {'port': 443.0, 'l7': 91.0, 'iat_mean': 10.0,
                    'bytes_mean': 3200.0, 'pkts_mean': 10.0},
        'dns'    : {'port': 53.0,  'l7': 5.0,  'iat_mean': 30.0,
                    'bytes_mean': 120.0,  'pkts_mean': 2.0},
        'generic': {'port': 80.0,  'l7': 7.0,  'iat_mean': 12.0,
                    'bytes_mean': 1500.0, 'pkts_mean': 5.0},
    }

    _STRATEGY_INIT = {
        WarmupStrategy.CONSERVATIVE : 10,
        WarmupStrategy.MODERATE     : 30,
        WarmupStrategy.AGGRESSIVE   : 100,
        WarmupStrategy.ADAPTIVE     : 15,
    }

    def __init__(
        self,
        strategy          : WarmupStrategy = WarmupStrategy.ADAPTIVE,
        max_oracle_budget : int   = 300,
        ip_pool_size      : int   = 50,
        burn_threshold    : int   = 2,
        backoff_factor    : float = 2.0,
        patience          : int   = 3,
        seed              : int   = 42,
    ):
        self.strategy          = strategy
        self.max_oracle_budget = max_oracle_budget
        self.ip_pool_size      = ip_pool_size
        self.burn_threshold    = burn_threshold
        self.backoff_factor    = backoff_factor
        self.patience          = patience
        self.rng               = np.random.default_rng(seed)

        # Estado interno del atacante
        self._oracle_calls     = 0
        self._detections       = 0
        self._evasions         = 0
        self._current_warmup_n = self._STRATEGY_INIT[strategy]
        self._history          : list[dict] = []

        # Reputation Ledger — registro de IPs del atacante
        # {ip_float: {'detections': int, 'evasions': int, 'burned': bool}}
        self._ip_pool    = _generate_ip_pool(ip_pool_size, self.rng)
        self._ip_ledger  = {
            ip: {'detections': 0, 'evasions': 0, 'burned': False}
            for ip in self._ip_pool
        }
        self._current_ip_idx = 0   # índice en el pool activo

    # ------------------------------------------------------------------
    # GESTIÓN DEL POOL DE IPs
    # ------------------------------------------------------------------
    @property
    def current_ip(self) -> float:
        return self._ip_pool[self._current_ip_idx]

    @property
    def active_ips(self) -> list[float]:
        """IPs no quemadas disponibles."""
        return [ip for ip in self._ip_pool
                if not self._ip_ledger[ip]['burned']]

    def rotate_ip(self, force: bool = False):
        """
        Rota a la siguiente IP disponible del pool.
        Si force=True, quema la IP actual primero.
        El atacante rota cuando una IP acumula demasiadas detecciones.
        """
        if force:
            self._ip_ledger[self.current_ip]['burned'] = True

        # Buscar la siguiente IP no quemada
        available = self.active_ips
        if not available:
            # Pool agotado — regenerar (simula nueva botnet/proxy)
            self._ip_pool   = _generate_ip_pool(self.ip_pool_size, self.rng)
            self._ip_ledger = {
                ip: {'detections': 0, 'evasions': 0, 'burned': False}
                for ip in self._ip_pool
            }
            available = list(self._ip_pool)

        # Elegir la IP con menos detecciones (la más "limpia")
        best_ip = min(
            available,
            key=lambda ip: self._ip_ledger[ip]['detections']
        )
        self._current_ip_idx = int(
            np.where(self._ip_pool == best_ip)[0][0]
        )

    def should_rotate(self) -> bool:
        """
        El atacante decide si rotar IP basándose en su historial.
        No conoce la ventana de evicción — usa heurística de detecciones.
        """
        ledger = self._ip_ledger[self.current_ip]
        return ledger['detections'] >= self.burn_threshold

    # ------------------------------------------------------------------
    # ORÁCULO Y APRENDIZAJE
    # ------------------------------------------------------------------
    def consult_oracle(self, response: OracleResponse):
        """Recibe feedback del IDS y actualiza el modelo mental."""
        self._oracle_calls += 1
        ledger = self._ip_ledger[self.current_ip]

        if response == OracleResponse.ATTACK:
            self._detections         += 1
            ledger['detections']     += 1

            if self.strategy == WarmupStrategy.ADAPTIVE:
                # Más warmup en el próximo intento
                self._current_warmup_n = min(
                    int(self._current_warmup_n * self.backoff_factor),
                    500
                )
            # Si la IP está quemada, rotar antes del próximo intento
            if self.should_rotate():
                self.rotate_ip(force=True)
        else:
            self._evasions       += 1
            ledger['evasions']   += 1

            if (self.strategy == WarmupStrategy.ADAPTIVE
                    and self._evasions > 3
                    and self._evasions % 5 == 0):
                # Reducir warmup si lleva rachas de éxito — más eficiente
                self._current_warmup_n = max(
                    int(self._current_warmup_n * 0.85), 5
                )

    @property
    def oracle_budget_exhausted(self) -> bool:
        return self._oracle_calls >= self.max_oracle_budget

    @property
    def detection_rate_observed(self) -> float:
        total = self._detections + self._evasions
        return self._detections / total if total > 0 else 1.0

    # ------------------------------------------------------------------
    # DECISIONES TÁCTICAS
    # ------------------------------------------------------------------
    def decide_warmup_n(self) -> int:
        if self.strategy == WarmupStrategy.ADAPTIVE:
            return self._current_warmup_n
        return self._STRATEGY_INIT[self.strategy]

    def decide_iat(self, profile: str) -> float:
        """
        IAT entre flujos de warmup.
        El atacante no sabe la ventana de evicción — usa comportamiento
        humano realista para el protocolo.
        """
        base = self._PROFILES.get(profile, self._PROFILES['generic'])['iat_mean']

        if self.strategy == WarmupStrategy.CONSERVATIVE:
            return base * self.rng.uniform(1.5, 3.0)
        elif self.strategy == WarmupStrategy.AGGRESSIVE:
            return base * self.rng.uniform(0.3, 0.7)
        elif self.strategy == WarmupStrategy.ADAPTIVE:
            # Si detección alta → ser más conservador
            if self.detection_rate_observed > 0.3:
                return base * self.rng.uniform(2.0, 4.0)
            return base * self.rng.uniform(0.8, 1.5)
        # return base * self.rng.uniform(0.9, 1.1)  # MODERATE (modo evaluación laboratorio)
        return 80.0 + self.rng.uniform(-10.0, 10.0) # Entre 70s y 90s (modo evaluación realista)

    def select_profile(self, x_attack_raw: np.ndarray, dc) -> str:
        idx_l7   = dc._feat_idx('L7_PROTO')
        idx_port = dc._feat_idx('L4_DST_PORT')
        if idx_l7 is not None:
            l7 = float(x_attack_raw[idx_l7])
            if l7 == 7.0:   return 'http'
            if l7 == 91.0:  return 'https'
            if l7 == 5.0:   return 'dns'
        if idx_port is not None:
            port = float(x_attack_raw[idx_port])
            if port == 443.0: return 'https'
            if port == 53.0:  return 'dns'
        return 'generic'

    def log_operation(self, flow_idx: int, warmup_n: int,
                      iat_mean: float, evaded: bool,
                      attempts: int, ip_used: float,
                      ip_rotations: int):
        self._history.append({
            'flow_idx'     : flow_idx,
            'warmup_n'     : warmup_n,
            'iat_mean'     : iat_mean,
            'evaded'       : evaded,
            'attempts'     : attempts,
            'ip_used'      : ip_used,
            'ip_rotations' : ip_rotations,
        })

    def summary(self) -> str:
        burned   = sum(1 for l in self._ip_ledger.values() if l['burned'])
        active   = len(self.active_ips)
        rotations = sum(h['ip_rotations'] for h in self._history) \
                    if self._history else 0
        warmups  = [h['warmup_n'] for h in self._history]
        lines = [
            f"\n  [TCPAdversary] Resumen operacional:",
            f"    Estrategia           : {self.strategy.value}",
            f"    Consultas oráculo    : {self._oracle_calls}/{self.max_oracle_budget}",
            f"    Tasa detección obs.  : {self.detection_rate_observed*100:.1f}%",
            f"    IPs del pool         : {self.ip_pool_size}",
            f"    IPs quemadas         : {burned}",
            f"    IPs activas restantes: {active}",
            f"    Rotaciones totales   : {rotations}",
            f"    Warmup final adj.    : {self._current_warmup_n} flujos",
        ]
        if warmups:
            lines.append(
                f"    Warmup medio usado   : {np.mean(warmups):.1f} flujos"
            )
        return "\n".join(lines)


# ===========================================================================
# RESULTADO TCP
# ===========================================================================

@dataclass
class TCPResult:
    X_poisoned       : np.ndarray
    X_original       : np.ndarray
    y_true           : np.ndarray
    y_pred_orig      : np.ndarray
    y_pred_poisoned  : np.ndarray
    asr              : float
    adversary_summary: str
    buf_shift        : dict
    warmup_cost      : dict
    attack_name      : str = "TCP — Temporal Context Poisoning"

    def summary(self) -> str:
        n        = len(self.y_true)
        evaded   = (self.y_pred_poisoned == 0).sum()
        detected = (self.y_pred_orig != 0).sum()
        lines = [
            f"\n{'='*65}",
            f"ATAQUE: {self.attack_name}",
            f"{'='*65}",
            f"  Flujos originales detectados : {detected:,}/{n:,}",
            f"  Evasiones exitosas           : {evaded:,} ({self.asr*100:.1f}%)",
            f"\n  Coste Operacional del Atacante:",
            f"    Warmup total enviado       : {self.warmup_cost['total_warmup_flows']:,} flujos",
            f"    Warmup medio por ataque    : {self.warmup_cost['mean_warmup_per_attack']:.1f}",
            f"    Consultas al oráculo       : {self.warmup_cost['oracle_calls']}",
            f"    Intentos fallidos          : {self.warmup_cost['failed_attempts']}",
            f"    Rotaciones de IP           : {self.warmup_cost['ip_rotations']}",
            f"    IPs quemadas               : {self.warmup_cost['ips_burned']}",
            f"    Coste por evasión          : {self.warmup_cost['cost_per_evasion']:.1f} flujos/evasión",
            f"\n  Desplazamiento BUF_* (original → envenenado):",
        ]
        for feat, info in self.buf_shift.items():
            lines.append(
                f"    {feat:<25} {info['before']:>8.3f} → "
                f"{info['after']:>8.3f}  Δ={info['delta']:>+8.3f}"
            )
        lines.append(self.adversary_summary)
        return "\n".join(lines)

    def evasion_by_class(self, class_names: Optional[dict] = None) -> str:
        lines = ["\n  Evasión por clase:"]
        mask_detected = self.y_pred_orig != 0
        for cls in np.unique(self.y_true):
            mask = (self.y_true == cls) & mask_detected
            if mask.sum() == 0:
                continue
            evaded = (self.y_pred_poisoned[mask] == 0).sum()
            total  = mask.sum()
            rate   = evaded / total * 100
            name   = (class_names.get(int(cls), f"Clase {cls}")
                      if class_names else f"Clase {cls}")
            lines.append(
                f"    {name:<25} {evaded:>5}/{total:<5} ({rate:.1f}%)"
            )
        return "\n".join(lines)


# ===========================================================================
# TCP POISONING — FRAMEWORK DE EVALUACIÓN DEL DEFENSOR
# ===========================================================================

class TCPoisoning:
    """
    Framework de evaluación del Temporal Context Poisoning.

    Separa claramente lo que sabe el defensor (todo) de lo que
    sabe el atacante (solo feedback binario del oráculo).
    """

    BUF_MONITOR = [
        'BUF_SCAN_RATE', 'BUF_FLOW_COUNT', 'BUF_UNIQUE_DST_PORTS',
        'BUF_NO_RESP_RATIO', 'BUF_SMALL_PKT_RATIO', 'BUF_RECON_SCORE',
    ]

    def __init__(
        self,
        buffer,
        constraints  : object,           # DomainConstraints
        adversary    : Optional[TCPAdversary] = None,
        X_benign_raw : Optional[np.ndarray]   = None,
        verbose      : bool = True,
    ):
        self.buffer       = buffer
        self.dc           = constraints
        self.adversary    = adversary or TCPAdversary(
            strategy=WarmupStrategy.ADAPTIVE
        )
        self._benign_pool = X_benign_raw
        self.verbose      = verbose

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

        if self.verbose:
            print(f"\n[TCP] Temporal Context Poisoning")
            print(f"  Estrategia atacante  : {self.adversary.strategy.value}")
            print(f"  Pool IPs             : {self.adversary.ip_pool_size}")
            print(f"  Burn threshold       : {self.adversary.burn_threshold} detecciones/IP")
            print(f"  Budget oráculo       : {self.adversary.max_oracle_budget}")
            print(f"  Flujos de ataque     : {len(X_attacks_raw):,}")

        X_orig_sc   = self.dc.to_scaled_space(X_attacks_raw)
        y_pred_orig = np.argmax(model.predict_proba(X_orig_sc), axis=1)

        if self.verbose:
            detected = (y_pred_orig != 0).sum()
            print(f"  Detectados sin warmup: {detected:,}/{len(y):,}\n")

        X_poisoned_sc, warmup_cost = self._poison_context_adversarial(
            X_attacks_raw, y_pred_orig, model
        )
        y_pred_poisoned = np.argmax(
            model.predict_proba(X_poisoned_sc), axis=1
        )

        mask_detected = y_pred_orig != 0
        asr = (y_pred_poisoned[mask_detected] == 0).mean() \
              if mask_detected.sum() > 0 else 0.0

        result = TCPResult(
            X_poisoned        = X_poisoned_sc,
            X_original        = X_orig_sc,
            y_true            = y,
            y_pred_orig       = y_pred_orig,
            y_pred_poisoned   = y_pred_poisoned,
            asr               = asr,
            adversary_summary = self.adversary.summary(),
            buf_shift         = self._compute_buf_shift(X_orig_sc, X_poisoned_sc),
            warmup_cost       = warmup_cost,
        )

        if self.verbose:
            print(result.summary())
            if class_names:
                print(result.evasion_by_class(class_names))

        return result

    def run_strategy_sweep(
        self,
        X_attacks_raw : np.ndarray,
        y             : np.ndarray,
        model         : object,
        class_names   : Optional[dict] = None,
    ) -> dict:
        """
        Compara las 4 estrategias del atacante.
        Resultado clave: ADAPTIVE converge al óptimo sin conocer el buffer.
        """
        if self.verbose:
            print(f"\n[TCP] Strategy sweep")
            print(f"{'Estrategia':<15} | {'ASR':>7} | "
                  f"{'Warmup medio':>13} | {'Rotaciones':>11} | "
                  f"{'IPs quemadas':>13}")
            print("-" * 68)

        results = {}
        for strategy in WarmupStrategy:
            adv = TCPAdversary(
                strategy          = strategy,
                max_oracle_budget = self.adversary.max_oracle_budget,
                ip_pool_size      = self.adversary.ip_pool_size,
                burn_threshold    = self.adversary.burn_threshold,
                backoff_factor    = self.adversary.backoff_factor,
                patience          = self.adversary.patience,
            )
            tcp    = TCPoisoning(self.buffer, self.dc,
                                 adversary=adv,
                                 X_benign_raw=self._benign_pool,
                                 verbose=False)
            result = tcp.run(X_attacks_raw, y, model, class_names)
            results[strategy.value] = result

            if self.verbose:
                wc = result.warmup_cost
                print(
                    f"  {strategy.value:<13} | "
                    f"{result.asr*100:>6.1f}% | "
                    f"{wc['mean_warmup_per_attack']:>12.1f} | "
                    f"{wc['ip_rotations']:>10} | "
                    f"{wc['ips_burned']:>12}"
                )

        return results

    def run_budget_sweep(
        self,
        X_attacks_raw : np.ndarray,
        y             : np.ndarray,
        model         : object,
        budgets       : list[int] = [10, 25, 50, 100, 200, 500],
    ) -> dict:
        """
        ASR vs presupuesto de oráculo.
        Cuantifica el coste mínimo de reconocimiento del ataque.
        """
        if self.verbose:
            print(f"\n[TCP] Budget sweep: {budgets}")
            print(f"{'Budget':>8} | {'ASR':>7} | "
                  f"{'Warmup medio':>13} | {'Rotaciones':>11}")
            print("-" * 46)

        results = {}
        for budget in budgets:
            adv = TCPAdversary(
                strategy          = WarmupStrategy.ADAPTIVE,
                max_oracle_budget = budget,
                ip_pool_size      = self.adversary.ip_pool_size,
                burn_threshold    = self.adversary.burn_threshold,
            )
            tcp    = TCPoisoning(self.buffer, self.dc,
                                 adversary=adv,
                                 X_benign_raw=self._benign_pool,
                                 verbose=False)
            result = tcp.run(X_attacks_raw, y, model)
            results[budget] = result

            if self.verbose:
                wc = result.warmup_cost
                print(
                    f"  {budget:>6} | "
                    f"{result.asr*100:>6.1f}% | "
                    f"{wc['mean_warmup_per_attack']:>12.1f} | "
                    f"{wc['ip_rotations']:>10}"
                )

        return results

    def run_pool_size_sweep(
        self,
        X_attacks_raw : np.ndarray,
        y             : np.ndarray,
        model         : object,
        pool_sizes    : list[int] = [5, 10, 25, 50, 100],
    ) -> dict:
        """
        ASR vs tamaño del pool de IPs del atacante.

        Responde: ¿cuántas IPs necesita la botnet para ser efectiva?
        Un pool pequeño (5 IPs) se quema rápido. Un pool grande
        mantiene reputación limpia más tiempo.
        Resultado original y cuantificable para la memoria.
        """
        if self.verbose:
            print(f"\n[TCP] Pool size sweep: {pool_sizes}")
            print(f"{'Pool IPs':>9} | {'ASR':>7} | "
                  f"{'IPs quemadas':>13} | {'Rotaciones':>11}")
            print("-" * 46)

        results = {}
        for pool_size in pool_sizes:
            adv = TCPAdversary(
                strategy          = WarmupStrategy.ADAPTIVE,
                max_oracle_budget = self.adversary.max_oracle_budget,
                ip_pool_size      = pool_size,
                burn_threshold    = self.adversary.burn_threshold,
            )
            tcp    = TCPoisoning(self.buffer, self.dc,
                                 adversary=adv,
                                 X_benign_raw=self._benign_pool,
                                 verbose=False)
            result = tcp.run(X_attacks_raw, y, model)
            results[pool_size] = result

            if self.verbose:
                wc = result.warmup_cost
                print(
                    f"  {pool_size:>7} | "
                    f"{result.asr*100:>6.1f}% | "
                    f"{wc['ips_burned']:>12} | "
                    f"{wc['ip_rotations']:>10}"
                )

        return results

    # ------------------------------------------------------------------
    # MOTOR DE ENVENENAMIENTO
    # ------------------------------------------------------------------
    def _poison_context_adversarial(
        self,
        X_attacks_raw : np.ndarray,
        y_pred_orig   : np.ndarray,
        model         : object,
    ) -> tuple[np.ndarray, dict]:

        n              = len(X_attacks_raw)
        X_poisoned_raw = X_attacks_raw.copy()
        mask_detected  = y_pred_orig != 0

        benign_pool = (
            self._benign_pool
            if self._benign_pool is not None
            else self._synthesize_benign_pool(1000, X_attacks_raw.shape[1])
        )

        buf_feature_names = [
            'BUF_UNIQUE_DST_PORTS', 'BUF_UNIQUE_DST_IPS',
            'BUF_FLOW_COUNT', 'BUF_NO_RESP_RATIO', 'BUF_SCAN_RATE',
            'BUF_SMALL_PKT_RATIO', 'BUF_PORT_STD', 'BUF_PORT_RANGE',
            'BUF_IS_SCANNER', 'BUF_HTTP_RATIO', 'BUF_HTTP_BYTES_AVG',
            'BUF_HTTP_SMALL_RATIO', 'BUF_BURST_PORTS',
            'BUF_SYN_ACK_RST_RATIO', 'BUF_RECON_SCORE',
        ]

        total_warmup   = 0
        failed_att     = 0
        ip_rotations   = 0
        warmup_list    = []

        if self.verbose:
            print(f"  Ejecutando envenenamiento adversarial grey-box...")

        for i in range(n):
            if not mask_detected[i]:
                continue

            if self.adversary.oracle_budget_exhausted:
                # Sin budget: usar último warmup conocido sin feedback
                x_candidate = self._inject_warmup(
                    X_attacks_raw[i], benign_pool, buf_feature_names,
                    warmup_n = self.adversary.decide_warmup_n(),
                    profile  = self.adversary.select_profile(
                        X_attacks_raw[i], self.dc),
                    src_ip   = self.adversary.current_ip,
                )
                X_poisoned_raw[i] = x_candidate
                continue

            evaded        = False
            attempts      = 0
            rotations_i   = 0
            last_candidate = X_attacks_raw[i].copy()

            for attempt in range(self.adversary.patience):
                warmup_n = self.adversary.decide_warmup_n()
                profile  = self.adversary.select_profile(
                    X_attacks_raw[i], self.dc)
                iat_mean = self.adversary.decide_iat(profile)
                src_ip   = self.adversary.current_ip

                x_candidate = self._inject_warmup(
                    X_attacks_raw[i], benign_pool, buf_feature_names,
                    warmup_n=warmup_n, profile=profile,
                    iat_mean=iat_mean, src_ip=src_ip,
                )
                total_warmup += warmup_n
                warmup_list.append(warmup_n)
                attempts     += 1

                # Consulta al oráculo — el atacante solo ve 0/1
                x_sc   = self.dc.to_scaled_space(x_candidate[np.newaxis])
                pred   = np.argmax(model.predict_proba(x_sc), axis=1)[0]
                oracle = (OracleResponse.BENIGN if pred == 0
                          else OracleResponse.ATTACK)

                # Antes de notificar al adversario, registrar si va a rotar
                ip_before = self.adversary.current_ip
                self.adversary.consult_oracle(oracle)
                if self.adversary.current_ip != ip_before:
                    rotations_i  += 1
                    ip_rotations += 1

                if oracle == OracleResponse.BENIGN:
                    evaded         = True
                    last_candidate = x_candidate
                    break
                else:
                    failed_att    += 1
                    last_candidate = x_candidate

            X_poisoned_raw[i] = last_candidate

            self.adversary.log_operation(
                flow_idx=i, warmup_n=warmup_n, iat_mean=iat_mean,
                evaded=evaded, attempts=attempts,
                ip_used=src_ip, ip_rotations=rotations_i,
            )

        # Limpiar buffer al terminar — no contaminar evaluaciones futuras
        self.buffer.reset()

        if self.verbose:
            print(f"  ✓ Completado")

        evaded_total = sum(1 for h in self.adversary._history if h['evaded'])
        ips_burned   = sum(
            1 for l in self.adversary._ip_ledger.values() if l['burned']
        )

        warmup_cost = {
            'total_warmup_flows'      : total_warmup,
            'mean_warmup_per_attack'  : float(np.mean(warmup_list))
                                        if warmup_list else 0.0,
            'oracle_calls'            : self.adversary._oracle_calls,
            'failed_attempts'         : failed_att,
            'ip_rotations'            : ip_rotations,
            'ips_burned'              : ips_burned,
            'cost_per_evasion'        : (total_warmup / evaded_total
                                         if evaded_total > 0
                                         else float('inf')),
        }

        X_poisoned_raw = self.dc.apply_causal_graph(X_poisoned_raw)
        return self.dc.to_scaled_space(X_poisoned_raw), warmup_cost

    def _inject_warmup(
        self,
        x_attack_raw      : np.ndarray,
        benign_pool       : np.ndarray,
        buf_feature_names : list[str],
        warmup_n          : int,
        profile           : str,
        iat_mean          : float = 8.0,
        src_ip            : float = 167772161.0,
    ) -> np.ndarray:
        """
        Inyecta warmup benigno en el buffer desde una IP específica
        y extrae las BUF_* resultantes para el flujo de ataque.

        Cada llamada usa la IP que el adversario ha seleccionado —
        IPs distintas tienen historiales distintos en el buffer.
        """
        # NO reseteamos el buffer aquí — las IPs distintas
        # tienen historiales independientes, igual que en producción.
        # Solo reseteamos el historial de la IP actual si es una
        # IP nueva (buffer la verá como primera vez).

        prof     = TCPAdversary._PROFILES.get(
            profile, TCPAdversary._PROFILES['generic']
        )
        idx_pool = self.adversary.rng.choice(
            len(benign_pool), size=warmup_n, replace=True
        )
        warmup_flows = benign_pool[idx_pool].copy()

        # Ajustar protocolo del warmup al del ataque
        idx_port = self.dc._feat_idx('L4_DST_PORT')
        idx_l7   = self.dc._feat_idx('L7_PROTO')
        if idx_port is not None:
            warmup_flows[:, idx_port] = prof['port']
        if idx_l7 is not None:
            warmup_flows[:, idx_l7] = prof['l7']

        # Timestamps realistas — Poisson process, IAT ~ Exp(mean)
        iats         = self.adversary.rng.exponential(iat_mean, size=warmup_n)
        timestamps_s = np.cumsum(iats)

        warmup_df = self._array_to_buffer_df(
            warmup_flows, timestamps_s, src_ip=src_ip
        )
        self.buffer.update_and_extract(warmup_df)

        # Flujo de ataque justo tras el último warmup
        # El atacante no introduce gap deliberado — no sabe de la evicción
        t_attack  = timestamps_s[-1] + self.adversary.rng.exponential(iat_mean)
        attack_df = self._array_to_buffer_df(
            x_attack_raw[np.newaxis],
            np.array([t_attack]),
            src_ip=src_ip,
        )
        attack_result = self.buffer.update_and_extract(attack_df)

        # Sustituir BUF_* en el vector de ataque
        x_poisoned = x_attack_raw.copy()
        for feat in buf_feature_names:
            if feat in attack_result.columns:
                feat_idx = self.dc._feat_idx(feat)
                if feat_idx is not None:
                    x_poisoned[feat_idx] = float(attack_result[feat][0])

        return x_poisoned

    def _array_to_buffer_df(
        self,
        X_raw        : np.ndarray,
        timestamps_s : Optional[np.ndarray] = None,
        src_ip       : float = 167772161.0,
    ) -> pl.DataFrame:
        n    = len(X_raw)
        data = {}

        # 1. EXCLUSIÓN DE BUF_* (Mantenemos nuestra protección contra DuplicateError)
        for i, name in enumerate(self.dc.feature_names):
            if not name.startswith('BUF_'):
                data[name] = X_raw[:, i].astype(np.float64)

        # 2. VECTORIZACIÓN DE CLAUDE (Máxima eficiencia en dispersión)
        data['IPV4_SRC_ADDR'] = np.full(n, src_ip, dtype=np.float64)
        
        dst_base = 167772160
        data['IPV4_DST_ADDR'] = (
            dst_base + self.adversary.rng.integers(1, 65535, size=n)
        ).astype(np.float64)

        port_choices = np.array([80.0, 443.0, 8080.0, 8443.0, 53.0, 22.0])
        port_probs   = np.array([0.40, 0.40,  0.05,   0.05,   0.08, 0.02])
        ports = self.adversary.rng.choice(port_choices, size=n, p=port_probs)

        l7_protos = np.zeros(n, dtype=np.float64)
        l7_protos[np.isin(ports, [80.0, 8080.0])]  = 7.0   # HTTP
        l7_protos[np.isin(ports, [443.0, 8443.0])] = 91.0  # HTTPS
        l7_protos[ports == 53.0]                   = 5.0   # DNS

        idx_port = self.dc._feat_idx('L4_DST_PORT')
        idx_l7   = self.dc._feat_idx('L7_PROTO')
        
        if idx_port is not None:
            data[self.dc.feature_names[idx_port]] = ports
        if idx_l7 is not None:
            data[self.dc.feature_names[idx_l7]] = l7_protos

        data['FLOW_START_MILLISECONDS'] = (
            timestamps_s * 1000.0 if timestamps_s is not None
            else np.arange(n, dtype=np.float64) * 1000.0
        )
        
        return pl.DataFrame(data)

    def _synthesize_benign_pool(
        self, n_samples: int, feature_dim: int
    ) -> np.ndarray:
        pool = np.zeros((n_samples, feature_dim), dtype=np.float64)
        prof = TCPAdversary._PROFILES['http']
        rng  = self.adversary.rng
        for feat, key in [('IN_BYTES', 'bytes_mean'), ('IN_PKTS', 'pkts_mean')]:
            idx = self.dc._feat_idx(feat)
            if idx is not None:
                noise = rng.normal(1.0, 0.15, n_samples)
                pool[:, idx] = prof[key] * np.clip(noise, 0.3, 2.5)
        
        return pool

    def _compute_buf_shift(
        self, X_orig_sc: np.ndarray, X_poisoned_sc: np.ndarray
    ) -> dict:
        shift = {}
        for feat in self.BUF_MONITOR:
            idx = self.dc._feat_idx(feat)
            if idx is None:
                continue
            before = float(X_orig_sc[:, idx].mean())
            after  = float(X_poisoned_sc[:, idx].mean())
            shift[feat] = {
                'before': before, 'after': after, 'delta': after - before
            }
        
        return shift