"""
src/attacks/sgfp_attack.py
================================================================
SGFP — SHAP-Guided Feature Perturbation
Baseline de caja blanca sobre LightGBM

Motivación:
  LightGBM no tiene gradientes diferenciables. Los valores SHAP
  actúan como proxy direccional: indican qué features contribuyen
  más a clasificar el flujo como ataque y en qué dirección.

  El SGFP es el equivalente del FGSM para modelos de árboles:
  un único paso en la dirección opuesta al vector SHAP de la
  clase de ataque actual.

  Es el baseline de la Fase 1 del arsenal LightGBM — establece
  el ASR mínimo esperado antes de técnicas más sofisticadas
  (IRT-SHAP, EDM).

Diferencia con IRT-SHAP:
  SGFP    : un paso  → X_adv = X - ε·sign(shap)
  IRT-SHAP: N pasos  → iterativo, 
        # recalcula SHAP en cada paso (aún en desarrollo y cambios de paradigma)

Gradient masking (herencia de ACE):
  Solo perturba features Forward — las Backward, Derived e
  Immutable quedan congeladas. El atacante no puede controlar
  la respuesta del servidor ni el historial del buffer.

Modo targeted:
  direction = -λ·sign(shap_clase_real) + (1-λ)·sign(shap_benigno)
  Combina alejarse de la clase real con acercarse a benigno.

Uso:
    explainer = joblib.load('outputs/models/lgbm/lgbm_shap_explainer.pkl')
    attack    = SGFPAttack(dc, explainer, epsilon=0.1, mode='targeted')
    result    = attack.run(X_attacks, y_attacks, lgbm_wrapper)

    # Comparativa untargeted vs targeted
    for mode in ('untargeted', 'targeted'):
        attack = SGFPAttack(dc, explainer, epsilon=0.1, mode=mode)
        result = attack.run(X_attacks, y_attacks, lgbm_wrapper)
        print(result.summary())
"""

from __future__ import annotations

import numpy as np
import joblib
from typing import Optional

from src.attacks.base_attacks import BaseAttack
from src.utils.domain_constraints import DomainConstraints


class SGFPAttack(BaseAttack):
    """
    SHAP-Guided Feature Perturbation — baseline de un paso para LightGBM.

    Parámetros
    ----------
    constraints   : DomainConstraints — restricciones físicas y causales
    explainer     : TreeExplainer pre-computado sobre el modelo TARGET REAL
                    Cargar con: joblib.load('lgbm_shap_explainer.pkl')
    epsilon       : radio máximo de perturbación en espacio escalado
    mode          : 'targeted'   → empuja hacia clase 0 (Benigno)
                    'untargeted' → aleja de la clase real
    lambda_target : peso del empuje hacia benigno en modo targeted [0, 1]
                    0.0 = solo aleja de la clase real
                    1.0 = solo empuja hacia benigno
                    0.5 = balance (recomendado)
    device        : forzado a 'cpu' — LightGBM no usa GPU en inferencia
    batch_size    : heredado de BaseAttack (controla el procesado por lotes)
    verbose       : mostrar progreso
    """

    def __init__(
        self,
        constraints   : DomainConstraints,
        explainer,
        epsilon       : float = 0.1,
        mode          : str   = 'targeted',
        lambda_target : float = 0.5,
        feature_ranges: Optional[np.ndarray] = None,
        **kwargs,
    ):
        # LightGBM siempre en CPU
        kwargs['device'] = 'cpu'
        super().__init__(constraints, epsilon=epsilon, **kwargs)

        if mode not in ('targeted', 'untargeted'):
            raise ValueError(
                f"mode debe ser 'targeted' o 'untargeted', no '{mode}'"
            )

        self.explainer     = explainer
        self.mode          = mode
        self.lambda_target = lambda_target
        self.feature_ranges = feature_ranges  

    @property
    def name(self) -> str:
        return (f"SGFP — SHAP-Guided Feature Perturbation "
                f"({self.mode}, ε={self.epsilon}, λ={self.lambda_target})")

    def _generate_perturbation(
        self,
        X     : np.ndarray,
        y     : np.ndarray,
        model : object,
    ) -> tuple[np.ndarray, int]:
        """
        Un único paso de perturbación guiada por SHAP.

        1. Calcular SHAP sobre X (estado original)
        2. Construir dirección según modo (ver _compute_direction)
        3. Top-K adaptado al espacio Forward disponible
        4. Paso único L-inf: X_adv = X + ε·sign(direction)

        BaseAttack.run() aplica la proyección física final
        (to_physical_space → apply_causal_graph → to_scaled_space).
        """
        shap_vals = self.explainer.shap_values(X, check_additivity=False)
        n_queries = len(X)

        n_samples, n_features = X.shape
        direction_raw   = self._compute_direction(shap_vals, y, n_samples, n_features)
        shap_magnitudes = np.abs(direction_raw)

        # Top-K adaptado: si hay física, el ranking solo considera features Forward
        # Evita desperdiciar el presupuesto en features que el gradient masking eliminará
        f_mask = getattr(self, 'forward_mask', None)
        if f_mask is None and self.dc is not None:
            f_mask = self.dc.forward_mask

        if f_mask is not None:
            shap_forward = shap_magnitudes.copy()
            shap_forward[:, ~f_mask] = 0.0
            K_FEATURES      = min(10, int(f_mask.sum()))
            top_k_threshold = np.sort(shap_forward, axis=1)[:, -K_FEATURES][:, np.newaxis]
            top_k_mask      = shap_forward >= top_k_threshold
        else:
            K_FEATURES      = 10
            top_k_threshold = np.sort(shap_magnitudes, axis=1)[:, -K_FEATURES][:, np.newaxis]
            top_k_mask      = shap_magnitudes >= top_k_threshold

        # Gradient masking implícito en shap_forward — no hace falta línea adicional
        direction_final = np.where(top_k_mask, np.sign(direction_raw), 0.0)

        # Paso único L-inf
        perturbation = self.epsilon * direction_final
        X_adv_raw    = X + perturbation

        return X_adv_raw, n_queries

    # ------------------------------------------------------------------
    # DIRECCIÓN DE PERTURBACIÓN
    # ------------------------------------------------------------------
    def _compute_direction(
        self,
        shap_vals  : list | np.ndarray,
        y          : np.ndarray,
        n          : int,
        n_features : int,
    ) -> np.ndarray:
        """
        Dirección de perturbación basada en SHAP puro de la clase objetivo.

        En SHAP multiclase los valores suman cero entre clases por construcción
        (propiedad de eficiencia de Shapley), por lo que sample_shap[0] ya
        contiene la información contrastiva completa. Combinar con la repulsión
        de otras clases añadiría ruido redundante.

        Targeted   : sigue el gradiente SHAP de clase 0 (Benigno) directamente.
        Untargeted : invierte el gradiente SHAP de la clase real del ejemplo.
        """
        direction = np.zeros((n, n_features), dtype=np.float32)

        # Detectar estructura real del explainer:
        # - list  → (n_clases,) de arrays (n_samples, n_features)  [estructura clásica]
        # - 3D    → ndarray (n_samples, n_features, n_clases)       [LightGBM TreeExplainer]
        is_list = isinstance(shap_vals, list)
        is_3d   = isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 3

        for i in range(n):
            if is_list:
                # shape resultante → (n_clases, n_features)
                sample_shap = np.array([shap_vals[c][i] for c in range(len(shap_vals))])
            elif is_3d:
                # shap_vals[i] → (n_features, n_clases) — transponemos
                sample_shap = shap_vals[i].T  # → (n_clases, n_features)
            else:
                # Fallback binario
                if self.mode == 'targeted':
                    direction[i] = shap_vals[i].flatten()[:n_features].astype(np.float32)
                else:
                    direction[i] = -shap_vals[i].flatten()[:n_features].astype(np.float32)
                continue

            if self.mode == 'targeted':
                combined = sample_shap[0]           # Gradiente directo hacia Benigno (clase 0)
            else:
                clase_real = int(y[i])
                combined = -sample_shap[clase_real]  # Alejarse de la clase real

            direction[i] = combined[:n_features].astype(np.float32)

        return direction

    # ------------------------------------------------------------------
    # COMPARATIVA UNTARGETED VS TARGETED
    # ------------------------------------------------------------------
    def run_mode_comparison(
        self,
        X          : np.ndarray,
        y          : np.ndarray,
        model      : object,
        class_names: Optional[dict] = None,
    ) -> dict:
        """
        Compara ASR entre modo untargeted y targeted para distintos λ.

        Análogo a la comparativa FGSM untargeted vs targeted, pero
        para LightGBM con SHAP como proxy del gradiente.

        Retorna
        -------
        dict con resultados de ambos modos y el sweep de λ
        """
        results = {}

        if self.verbose:
            print(f"\n[SGFP] Comparativa untargeted vs targeted")
            print(f"{'Modo':<20} | {'λ':>5} | {'ASR':>8} | {'L2':>8}")
            print("-" * 48)

        # Untargeted baseline
        attack_un = SGFPAttack(
            self.dc, self.explainer,
            epsilon=self.epsilon, mode='untargeted',
            batch_size=self.batch_size, verbose=False,
        )
        result_un = attack_un.run(X, y, model, class_names)
        results['untargeted'] = result_un

        if self.verbose:
            print(f"  {'untargeted':<18} | {'—':>5} | "
                  f"{result_un.asr*100:>7.1f}% | {result_un.l2_mean:>7.4f}")

        # Targeted con sweep de λ
        for lam in [0.0, 0.25, 0.5, 0.75, 1.0]:
            attack_t = SGFPAttack(
                self.dc, self.explainer,
                epsilon=self.epsilon, mode='targeted',
                lambda_target=lam,
                batch_size=self.batch_size, verbose=False,
            )
            result_t = attack_t.run(X, y, model, class_names)
            results[f'targeted_lambda_{lam}'] = result_t

            if self.verbose:
                print(f"  {'targeted':<18} | {lam:>5.2f} | "
                      f"{result_t.asr*100:>7.1f}% | {result_t.l2_mean:>7.4f}")

        return results


# ===========================================================================
# SCRIPT DE VERIFICACIÓN
# ===========================================================================

if __name__ == "__main__":
    import numpy as np
    import joblib
    from src.utils.domain_constraints import DomainConstraints

    print("[-] Verificando SGFPAttack...")

    dc        = DomainConstraints.from_artifacts()
    explainer = joblib.load('outputs/models/lgbm/lgbm_shap_explainer.pkl')

    for mode in ('untargeted', 'targeted'):
        attack = SGFPAttack(
            dc, explainer,
            epsilon=0.1, mode=mode, verbose=False,
        )
        print(f"   [✓] mode='{mode}': {attack.name}")

    print(f"\n   Forward perturbables: {attack.forward_mask.sum()}")
    print("\n[✓] sgfp_attack.py listo")
    print("    Uso:")
    print("      explainer = joblib.load('lgbm_shap_explainer.pkl')")
    print("      attack    = SGFPAttack(dc, explainer, epsilon=0.1)")
    print("      result    = attack.run(X_attacks, y_attacks, lgbm_wrapper)")
    print("      sweep     = attack.run_mode_comparison(X_attacks, y_attacks, lgbm_wrapper)")