"""
src/attacks/s3m.py
================================================================
S3M — Shadow Session Split-Merging
Contribución original del TFG. Ataque de camuflaje semántico tabular.

Daniel Gomollón Embid — TFG 2025-2026

═══════════════════════════════════════════════════════════════
PARADIGMA: Caballo de Troya Tabular Adaptativo
═══════════════════════════════════════════════════════════════

Los modelos tabulares de detección de intrusiones clasifican flujos
completos — no pueden distinguir si los estadísticos de un flujo
provienen de comportamiento benigno o malicioso mezclado.

S3M explota esta limitación con tres innovaciones sobre el mixup clásico:

1. MCR — Minimum Camouflage Ratio (Búsqueda Adaptativa)
   En lugar de un ratio global fijo, S3M encuentra el ratio mínimo
   de camuflaje necesario para que CADA muestra evada el modelo.
   Resultado: el Perfil de Camuflaje Mínimo del IDS — cuánto tráfico
   benigno necesita inyectar un atacante para pasar desapercibido.

2. Selección Semántica del Gemelo Benigno con Pesos de Importancia
   No elige un flujo benigno al azar sino el más similar al ataque
   ponderando por importancia de feature (gain LightGBM).
   Las features BUF_* con alto gain reciben mayor peso — el gemelo
   seleccionado arrastra las features más discriminativas hacia
   valores benignos, maximizando la efectividad de la mezcla.

3. Preservación de Firma de Ataque (Attack Signature Preservation)
   Identifica las features Forward que son funcionales para el ataque
   (sin ellas, el ataque no llega al objetivo) y las protege de la
   mezcla. Solo camufla las features Forward que son detectables
   pero no funcionales.

═══════════════════════════════════════════════════════════════
DOBLE USO EN EL TFG
═══════════════════════════════════════════════════════════════

Ataque    : evadir el IDS en producción
Defensa   : generador de datos para Tabular Mixup AT en Fase 3
            Los troyanos generados son exactamente el tipo de ejemplos
            difíciles que el Adversarial Training necesita ver

═══════════════════════════════════════════════════════════════
MODELO DE AMENAZA
═══════════════════════════════════════════════════════════════

El atacante necesita:
  1. Muestras de su propio tráfico malicioso
  2. Capturas de tráfico benigno (OSINT, capturas públicas)
  3. Capacidad de generar flujos TCP/UDP con estadísticas controladas
  4. Acceso de consulta al IDS (grey-box) — para búsqueda MCR

Es el ataque más realista para un atacante con reconocimiento previo
de la red objetivo.

Referencias:
- Wagner & Soto (2002) - "Mimicry Attacks on Host-Based Intrusion Detection Systems". 
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional

from src.attacks.base_attacks import BaseAttack, AttackResult
from src.utils.domain_constraints import DomainConstraints, PHYSICAL_BOUNDS


# ===========================================================================
# RESULTADO S3M
# ===========================================================================

@dataclass
class S3MResult:
    """
    Resultado del ataque S3M con métricas de camuflaje.

    La métrica más importante es mcr_distribution: la distribución
    del Minimum Camouflage Ratio entre muestras — cuánto camuflaje
    necesita este IDS específico para ser engañado.
    """
    X_trojan         : np.ndarray    # flujos troyanizados (espacio escalado)
    X_original       : np.ndarray    # flujos originales
    y_true           : np.ndarray

    y_pred_orig      : np.ndarray
    y_pred_trojan    : np.ndarray

    asr              : float
    ratio_used       : float         # ratio global si mode='fixed'
    mcr_per_sample   : np.ndarray    # MCR por muestra (NaN si no evadió)
    semantic_distance: np.ndarray    # distancia L2 troyano vs original
    twin_similarity  : np.ndarray    # similitud con el gemelo benigno elegido
    attack_name      : str = "S3M — Shadow Session Split-Merging"

    def summary(self) -> str:
        n         = len(self.y_true)
        evaded    = (self.y_pred_trojan == 0).sum()
        detected  = (self.y_pred_orig != 0).sum()
        mcr_valid = self.mcr_per_sample[~np.isnan(self.mcr_per_sample)]

        lines = [
            f"\n{'='*60}",
            f"ATAQUE: {self.attack_name}",
            f"{'='*60}",
            f"  Flujos detectados originalmente : {detected:,}/{n:,}",
            f"  Evasiones exitosas              : {evaded:,} ({self.asr*100:.1f}%)",
            f"\n  Perfil de Camuflaje Mínimo (MCR):",
        ]
        if len(mcr_valid) > 0:
            lines += [
                f"    MCR medio    : {mcr_valid.mean():.3f}",
                f"    MCR mínimo   : {mcr_valid.min():.3f}",
                f"    MCR máximo   : {mcr_valid.max():.3f}",
            ]
        else:
            lines.append("    MCR: sin evasiones")

        lines += [
            f"\n  Distancia semántica (L2 troyano vs original):",
            f"    Media : {self.semantic_distance.mean():.4f}",
            f"    Máxima: {self.semantic_distance.max():.4f}",
        ]
        return "\n".join(lines)

    def evasion_by_class(self, class_names: Optional[dict] = None) -> str:
        lines         = ["\n  Evasión por clase:"]
        mask_detected = self.y_pred_orig != 0
        for cls in np.unique(self.y_true):
            mask   = (self.y_true == cls) & mask_detected
            if mask.sum() == 0:
                continue
            evaded = (self.y_pred_trojan[mask] == 0).sum()
            total  = mask.sum()
            rate   = evaded / total * 100
            name   = (class_names.get(int(cls), f"Clase {cls}")
                      if class_names else f"Clase {cls}")
            lines.append(f"    {name:<25} {evaded:>5}/{total:<5} ({rate:.1f}%)")
        return "\n".join(lines)

    def mcr_profile(self) -> str:
        """Distribución del MCR — resultado central de S3M para la memoria."""
        lines = ["\n  Distribución MCR (fracción de tráfico benigno necesaria):"]
        bins  = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        valid = self.mcr_per_sample[~np.isnan(self.mcr_per_sample)]
        if len(valid) == 0:
            return "    Sin evasiones — MCR no calculable"
        for i in range(len(bins) - 1):
            count = ((valid >= bins[i]) & (valid < bins[i+1])).sum()
            bar   = '█' * int(count / len(valid) * 40)
            lines.append(f"    [{bins[i]:.1f}-{bins[i+1]:.1f}] {bar} {count}")
        return "\n".join(lines)


# ===========================================================================
# ATAQUE S3M
# ===========================================================================

class S3MAttack(BaseAttack):
    """
    Shadow Session Split-Merging — camuflaje semántico tabular adaptativo.

    Parámetros
    ----------
    constraints      : DomainConstraints
    X_benign_scaled  : pool de flujos benignos en espacio escalado.
                       Cuanto más grande, mejor la selección semántica.
    mode             : 'adaptive' → búsqueda MCR por muestra (recomendado)
                       'fixed'    → ratio fijo, equivalente a mixup clásico
    ratio            : ratio fijo para mode='fixed' [0, 1]
                       1.0 = 100% benigno | 0.0 = sin camuflaje
    ratio_steps      : granularidad búsqueda MCR en mode='adaptive'
                       10 → busca en [0.1, 0.2, ..., 1.0]
    twin_selection   : 'semantic' → gemelo más similar ponderado por gain
                       'random'   → selección aleatoria (baseline)
    preserve_payload : si True, protege features Forward funcionales
                       del ataque de la mezcla
    payload_features : índices de features Forward a preservar.
                       Si None, se infiere automáticamente.
    n_twins          : candidatos benignos evaluados por muestra.
                       Mayor = mejor gemelo, más lento.
    """

    # nos aseguramos de ser rigurosos con grey-box y no importar 
    # ganancia real de features de los modelos víctima
    _BUF_FEATURES_KNOWN = [
        'BUF_SCAN_RATE', 'BUF_FLOW_COUNT', 'BUF_HTTP_BYTES_AVG',
        'BUF_PORT_STD', 'BUF_NO_RESP_RATIO', 'BUF_SMALL_PKT_RATIO',
        'BUF_UNIQUE_DST_PORTS', 'BUF_UNIQUE_DST_IPS',
        'BUF_HTTP_RATIO', 'BUF_PORT_RANGE',
    ]

    def _build_feature_weights(self) -> np.ndarray:
        """
        Pesos uniformes para selección del gemelo benigno.

        El atacante sabe que los IDS modernos usan contexto temporal
        de IP (conocimiento OSINT razonable), pero NO conoce la
        importancia relativa de cada feature — eso requeriría acceso
        white-box al modelo.

        Todas las BUF_* reciben el mismo peso elevado (5x) respecto
        a las features de flujo individual.
        """
        n       = len(self.dc.feature_names)
        weights = np.ones(n, dtype=np.float32)

        for feat in self._BUF_FEATURES_KNOWN:
            idx = self.dc._feat_idx(feat)
            if idx is not None:
                weights[idx] = 5.0  # peso uniforme — grey-box puro

        return weights / weights.sum()

    def __init__(
        self,
        constraints      : DomainConstraints,
        X_benign_scaled  : np.ndarray,
        mode             : str   = 'adaptive',
        ratio            : float = 0.7,
        ratio_steps      : int   = 10,
        twin_selection   : str   = 'semantic',
        preserve_payload : bool  = True,
        payload_features : Optional[np.ndarray] = None,
        n_twins          : int   = 50,
        **kwargs,
    ):
        kwargs['device'] = 'cpu'
        # epsilon=1.0 — no aplica en S3M (opera en espacio físico con mixup)
        super().__init__(constraints, epsilon=1.0, **kwargs)

        if mode not in ('adaptive', 'fixed'):
            raise ValueError("mode debe ser 'adaptive' o 'fixed'")
        if twin_selection not in ('semantic', 'random'):
            raise ValueError("twin_selection debe ser 'semantic' o 'random'")

        self.mode             = mode
        self.ratio            = ratio
        self.ratio_steps      = ratio_steps
        self.twin_selection   = twin_selection
        self.preserve_payload = preserve_payload
        self.n_twins          = n_twins

        # Pool benigno en ambos espacios — la selección semántica
        # opera en escalado, la mezcla en físico
        self.X_benign_phys = self.dc.to_physical_space(X_benign_scaled)
        self.X_benign_sc   = X_benign_scaled

        # forward_mask viene de BaseAttack → self.dc.perturbable_mask
        self.forward_idx = np.where(self.forward_mask)[0]

        # Features de payload a preservar
        self.payload_idx = (payload_features if payload_features is not None
                            else self._infer_payload_features())

        # Pesos de importancia para selección semántica ponderada
        self._twin_weights = self._build_feature_weights()

        if self.verbose:
            print(f"[S3M] Inicializado:")
            print(f"  Pool benigno     : {len(self.X_benign_phys):,} flujos")
            print(f"  Features Forward : {len(self.forward_idx)}")
            print(f"  Features payload : {len(self.payload_idx)} (preservadas)")
            print(f"  Modo             : {mode}")
            print(f"  Twin selection   : {twin_selection}")

    @property
    def name(self) -> str:
        return (f"S3M — Shadow Session Split-Merging "
                f"({self.mode}, twin={self.twin_selection}, "
                f"payload={'ON' if self.preserve_payload else 'OFF'})")

    # ------------------------------------------------------------------
    # HELPER UNIFICADO DE PREDICCIÓN
    # Compatible con LGBMBaseline y TabularResNet sin prior correction —
    # la corrección bayesiana es para producción, no para evaluación adversarial
    # ------------------------------------------------------------------
    def _predict(self, model, X: np.ndarray) -> np.ndarray:
        if hasattr(model, 'predict_proba'):
            return np.argmax(model.predict_proba(X), axis=1)
        return model.predict(X)

    # ------------------------------------------------------------------
    # INTERFAZ PRINCIPAL — override de BaseAttack.run()
    # ------------------------------------------------------------------
    def run(
        self,
        X          : np.ndarray,
        y          : np.ndarray,
        model      : object,
        class_names: Optional[dict] = None,
    ) -> S3MResult:
        """
        Ejecuta S3M sobre flujos de ataque.

        Override completo de BaseAttack.run() para devolver S3MResult
        con métricas de camuflaje en lugar de AttackResult genérico.

        Parámetros
        ----------
        X          : (n, n_features) en espacio escalado
        y          : (n,) etiquetas reales
        model      : LGBMBaseline o TabularResNet
        class_names: dict {int: str}
        """
        y_pred_orig   = self._predict(model, X)
        mask_detected = y_pred_orig != 0

        if self.verbose:
            print(f"\n[S3M] {self.name}")
            print(f"  Flujos de ataque         : {len(X):,}")
            print(f"  Detectados originalmente : {mask_detected.sum():,}")

        X_trojan_sc, n_queries = self._generate_perturbation(X, y, model)
        y_pred_trojan = self._predict(model, X_trojan_sc)

        asr = (y_pred_trojan[mask_detected] == 0).mean() \
              if mask_detected.sum() > 0 else 0.0

        result = S3MResult(
            X_trojan          = X_trojan_sc,
            X_original        = X,
            y_true            = y,
            y_pred_orig       = y_pred_orig,
            y_pred_trojan     = y_pred_trojan,
            asr               = asr,
            ratio_used        = self.ratio,
            mcr_per_sample    = self._last_mcr,
            semantic_distance = self._last_semantic_dist,
            twin_similarity   = self._last_twin_sim,
        )

        if self.verbose:
            print(result.summary())
            print(result.mcr_profile())
            if class_names:
                print(result.evasion_by_class(class_names))

        return result

    # ------------------------------------------------------------------
    # GENERACIÓN DE PERTURBACIONES
    # ------------------------------------------------------------------
    def _generate_perturbation(
        self,
        X     : np.ndarray,
        y     : np.ndarray,
        model : object,
    ) -> tuple[np.ndarray, int]:
        """
        Genera flujos troyanizados para cada muestra.

        Proceso por muestra:
          1. Seleccionar gemelo benigno (semántico ponderado o aleatorio)
          2. Mezclar en espacio físico con el ratio apropiado
          3. Aplicar causal graph para coherencia física
          4. Si mode='adaptive': buscar MCR mínimo que evade el modelo
        """
        n_samples     = len(X)
        X_phys        = self.dc.to_physical_space(X)
        n_queries     = 0

        X_trojan_phys  = X_phys.copy()
        mcr_per_sample = np.full(n_samples, np.nan)
        semantic_dist  = np.zeros(n_samples)
        twin_similarity= np.zeros(n_samples)

        # Selección de gemelos benignos — una sola vez para todos
        twins_phys, twin_sim = self._select_twins(X, X_phys, n_samples)
        twin_similarity      = twin_sim

        if self.mode == 'fixed':
            # Mezcla vectorizada con ratio fijo — rápido
            X_trojan_phys     = self._mix(X_phys, twins_phys, self.ratio)
            mcr_per_sample[:] = self.ratio
            n_queries         = 0  # sin consultas al modelo en modo fixed

        else:
            # Búsqueda adaptativa del MCR mínimo por muestra
            # Granularidad: ratio_steps intervalos en [0, 1]
            ratios = np.linspace(0, 1, self.ratio_steps + 1)[1:]  # [0.1, ..., 1.0]

            for i in range(n_samples):
                evaded = False
                for ratio_candidate in ratios:
                    x_candidate = self._mix_single(
                        X_phys[i], twins_phys[i], ratio_candidate
                    )
                    x_sc      = self.dc.to_scaled_space(x_candidate[np.newaxis])
                    pred      = self._predict(model, x_sc)[0]
                    n_queries += 1

                    if pred == 0:  # evasión conseguida
                        X_trojan_phys[i]  = x_candidate
                        mcr_per_sample[i] = ratio_candidate
                        evaded            = True
                        break

                # Sin evasión con ningún ratio — usar el máximo disponible
                # como mejor intento (no dejar el flujo sin modificar)
                if not evaded:
                    X_trojan_phys[i] = self._mix_single(
                        X_phys[i], twins_phys[i], ratios[-1]
                    )

        # Calcular distancia semántica en espacio escalado
        X_trojan_sc   = self.dc.to_scaled_space(X_trojan_phys)
        semantic_dist = np.linalg.norm(X_trojan_sc - X, axis=1)

        # Guardar métricas para S3MResult
        self._last_mcr           = mcr_per_sample
        self._last_semantic_dist = semantic_dist
        self._last_twin_sim      = twin_similarity

        return X_trojan_sc, n_queries

    def _select_twins(
        self,
        X_sc   : np.ndarray,
        X_phys : np.ndarray,
        n      : int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Selecciona el gemelo benigno más adecuado para cada muestra.

        Modo 'semantic' (recomendado):
            Distancia ponderada por importancia de feature.
            El gemelo seleccionado es el que, al mezclarse, más reduce
            las features BUF_* discriminativas hacia valores benignos.

        Modo 'random':
            Selección aleatoria — equivalente al mixup clásico.
            Útil como baseline de comparación.
        """
        twins_phys = np.zeros_like(X_phys)
        twin_sim   = np.zeros(n)

        if self.twin_selection == 'random':
            idx          = np.random.choice(
                len(self.X_benign_phys), size=n, replace=True
            )
            twins_phys   = self.X_benign_phys[idx]
            twin_sim[:]  = -1.0  # sin métrica en modo aleatorio
            return twins_phys, twin_sim

        # Selección semántica ponderada por importancia
        for i in range(n):
            candidates_idx = np.random.choice(
                len(self.X_benign_phys), size=self.n_twins, replace=False
            )
            candidates_sc = self.X_benign_sc[candidates_idx]

            # Distancia ponderada global — no solo subespacio Forward
            # Las BUF_* tienen peso alto → gemelo con BUF_* similares
            # a los valores benignos arrastra la mezcla en la dirección correcta
            diff          = X_sc[i] - candidates_sc           # (n_twins, n_feat)
            weighted_dist = np.sqrt(
                (diff ** 2 * self._twin_weights).sum(axis=1)
            )

            best_idx       = candidates_idx[weighted_dist.argmin()]
            twins_phys[i]  = self.X_benign_phys[best_idx]
            twin_sim[i]    = 1.0 / (1.0 + weighted_dist.min())

        return twins_phys, twin_sim

    # ------------------------------------------------------------------
    # MEZCLA SEMÁNTICA
    # ------------------------------------------------------------------
    def _mix(
        self,
        X_atk_phys : np.ndarray,
        X_ben_phys : np.ndarray,
        ratio      : float,
    ) -> np.ndarray:
        """
        Mezcla vectorizada para mode='fixed'.
        ratio=0.7 → 70% benigno + 30% ataque.
        """
        X_trojan = (X_atk_phys * (1 - ratio)) + (X_ben_phys * ratio)

        if self.preserve_payload and len(self.payload_idx) > 0:
            X_trojan[:, self.payload_idx] = X_atk_phys[:, self.payload_idx]

        X_trojan = self.dc.apply_causal_graph(X_trojan)
        X_trojan = self._apply_physical_bounds(X_trojan)
        return X_trojan

    def _mix_single(
        self,
        x_atk  : np.ndarray,
        x_ben  : np.ndarray,
        ratio  : float,
    ) -> np.ndarray:
        """Mezcla de una sola muestra para búsqueda MCR adaptativa."""
        x_trojan = (x_atk * (1 - ratio)) + (x_ben * ratio)

        if self.preserve_payload and len(self.payload_idx) > 0:
            x_trojan[self.payload_idx] = x_atk[self.payload_idx]

        x_trojan = self.dc.apply_causal_graph(x_trojan[np.newaxis])[0]
        x_trojan = self._apply_physical_bounds(x_trojan[np.newaxis])[0]
        return x_trojan

    def _apply_physical_bounds(self, X: np.ndarray) -> np.ndarray:
        """
        Clipping a los límites físicos del dataset.
        Importa PHYSICAL_BOUNDS directamente del módulo — es una
        constante de módulo, no un atributo de instancia de DomainConstraints.
        """
        X_clipped = X.copy()
        for feat, (vmin, vmax) in PHYSICAL_BOUNDS.items():
            idx = self.dc._feat_idx(feat)
            if idx is not None:
                X_clipped[:, idx] = np.clip(X_clipped[:, idx], vmin, vmax)
        return X_clipped

    # ------------------------------------------------------------------
    # SWEEP DE RATIO — Perfil de Vulnerabilidad Semántica del IDS
    # ------------------------------------------------------------------
    def run_ratio_sweep(
        self,
        X          : np.ndarray,
        y          : np.ndarray,
        model      : object,
        ratios     : list[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        class_names: Optional[dict] = None,
    ) -> dict:
        """
        Curva ASR vs ratio de camuflaje sobre flujos detectados.

        La curva resultante es el Perfil de Vulnerabilidad Semántica
        del IDS — a partir de qué fracción de tráfico benigno el modelo
        deja de detectar la amenaza.

        Análogo al epsilon sweep del ablation study pero en el espacio
        semántico de camuflaje en lugar del espacio de perturbación L-inf.

        Opera solo sobre flujos detectados — el ASR es significativo.
        """
        # Filtrar a flujos detectados una sola vez fuera del bucle
        y_pred_orig   = self._predict(model, X)
        mask_detected = y_pred_orig != 0
        X_det         = X[mask_detected]
        y_det         = y[mask_detected]

        if self.verbose:
            print(f"\n[S3M] Ratio sweep sobre {mask_detected.sum():,} "
                  f"flujos detectados")
            print(f"{'Ratio':>7} | {'ASR':>8} | {'L2 medio':>10} | "
                  f"{'Twin sim':>10}")
            print("-" * 45)

        results       = {}
        original_mode = self.mode
        self.mode     = 'fixed'

        for r in ratios:
            self.ratio = r
            result     = self.run(X_det, y_det, model, class_names)
            results[r] = result

            if self.verbose:
                print(f"  {r:>5.2f} | {result.asr*100:>7.1f}% | "
                      f"{result.semantic_distance.mean():>9.4f} | "
                      f"{result.twin_similarity.mean():>9.4f}")

        self.mode  = original_mode
        self.ratio = 0.7
        return results

    # ------------------------------------------------------------------
    # INFERENCIA DE FEATURES DE PAYLOAD
    # ------------------------------------------------------------------
    def _infer_payload_features(self) -> np.ndarray:
        """
        Infiere features Forward funcionales para el ataque.

        Por defecto no preserva ninguna — el usuario puede especificar
        payload_features explícitamente si conoce su ataque.

        En BigFlow-NIDS las features de timing y conteo de paquetes
        son las más funcionales para el payload de Recon y DoS.
        """
        return np.array([], dtype=int)


# ===========================================================================
# GENERADOR PARA ADVERSARIAL TRAINING (Fase 3)
# ===========================================================================

def generate_s3m_augmentation(
    X_train_atk_sc : np.ndarray,
    X_train_ben_sc : np.ndarray,
    y_train_atk    : np.ndarray,
    dc             : DomainConstraints,
    n_augmented    : int   = 5000,
    ratio_range    : tuple = (0.3, 0.8),
    seed           : int   = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Genera ejemplos de augmentación para Tabular Mixup AT en Fase 3.

    En lugar de un ratio fijo, muestrea ratios del rango [ratio_range]
    para crear diversidad en los ejemplos de entrenamiento adversarial.
    Cuanto más diverso el rango, más robusto el modelo re-entrenado.

    Las etiquetas preservan la clase del ataque original — el modelo
    debe aprender que estos flujos mezclados siguen siendo ataques,
    no flujos benignos.

    Uso en Fase 3:
        X_aug, y_aug = generate_s3m_augmentation(
            X_train[idx_atk], X_train[idx_ben], y_train[idx_atk], dc
        )
        X_combined = np.vstack([X_train, X_aug])
        y_combined = np.concatenate([y_train, y_aug])
        lgbm.fit(X_combined, y_combined, ...)

    Parámetros
    ----------
    X_train_atk_sc : flujos de ataque de train en espacio escalado
    X_train_ben_sc : flujos benignos de train en espacio escalado
    y_train_atk    : etiquetas de los flujos de ataque
    dc             : DomainConstraints para transformaciones espaciales
    n_augmented    : número de ejemplos sintéticos a generar
    ratio_range    : rango de ratios de camuflaje [min, max]
    seed           : reproducibilidad

    Retorna
    -------
    X_augmented : (n_augmented, n_features) — troyanos para AT
    y_augmented : (n_augmented,) — etiquetas originales de ataque
    """
    np.random.seed(seed)

    idx_atk = np.random.choice(
        len(X_train_atk_sc), size=n_augmented, replace=True
    )
    idx_ben = np.random.choice(
        len(X_train_ben_sc), size=n_augmented, replace=True
    )

    X_atk_phys = dc.to_physical_space(X_train_atk_sc[idx_atk])
    X_ben_phys = dc.to_physical_space(X_train_ben_sc[idx_ben])

    # Ratios aleatorios del rango — diversidad máxima para AT
    ratios = np.random.uniform(ratio_range[0], ratio_range[1], n_augmented)

    X_aug_phys = (
        X_atk_phys * (1 - ratios[:, np.newaxis])
        + X_ben_phys *  ratios[:, np.newaxis]
    )

    X_aug_phys = dc.apply_causal_graph(X_aug_phys)
    X_aug_sc   = dc.to_scaled_space(X_aug_phys)

    # Etiquetas: clase del ataque original — no benigno
    y_augmented = y_train_atk[idx_atk]

    return X_aug_sc, y_augmented


# ===========================================================================
# SCRIPT DE VERIFICACIÓN
# ===========================================================================

if __name__ == "__main__":
    import numpy as np
    from src.utils.domain_constraints import DomainConstraints

    print("[-] Verificando S3MAttack...")
    dc = DomainConstraints.from_artifacts()

    np.random.seed(42)
    n_features  = len(dc.feature_names)
    X_benign_sc = np.random.normal(0, 0.3, (1000, n_features)).clip(-1, 1)

    for mode in ('fixed', 'adaptive'):
        attack = S3MAttack(
            dc,
            X_benign_scaled = X_benign_sc,
            mode            = mode,
            verbose         = False,
        )
        print(f"   [✓] mode='{mode}': {attack.name}")

    print(f"\n   Features Forward disponibles : {len(attack.forward_idx)}")
    print(f"   Features payload preservadas : {len(attack.payload_idx)}")
    print(f"\n[✓] s3m.py listo")
    print("    Uso ataque  : attack.run(X_ataques, y_ataques, lgbm_wrapper)")
    print("    Uso AT      : generate_s3m_augmentation(X_train, X_benign, y, dc)")
    print("    Ratio sweep : attack.run_ratio_sweep(X_ataques, y_ataques, model)")