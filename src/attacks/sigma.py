"""
src/attacks/sigma.py
================================================================
SIGMA — Shapley Iterative Gradient-Mapped Ascent
El equivalente a PGD (Projected Gradient Descent) para árboles de decisión.
"""

from __future__ import annotations

import numpy as np
from typing import Optional

from src.attacks.base_attacks import BaseAttack
from src.utils.domain_constraints import DomainConstraints


class SIGMAAttack(BaseAttack):
    def __init__(
        self,
        constraints   : Optional[DomainConstraints],
        explainer,
        epsilon       : float = 0.1,
        alpha         : float = 0.02,
        steps         : int   = 10,
        mode          : str   = 'targeted',
        lambda_target : float = 0.5,
        feature_ranges: Optional[np.ndarray] = None,
        **kwargs,
    ):
        kwargs['device'] = 'cpu'
        super().__init__(constraints, epsilon=epsilon, **kwargs)

        if mode not in ('targeted', 'untargeted'):
            raise ValueError(f"mode debe ser 'targeted' o 'untargeted'")

        self.explainer      = explainer
        self.alpha          = alpha
        self.steps          = steps
        self.mode           = mode
        self.lambda_target  = lambda_target
        self.feature_ranges = feature_ranges

    @property
    def name(self) -> str:
        return (f"SIGMA — Shapley Iterative Gradient-Mapped Ascent "
                f"({self.mode}, ε={self.epsilon}, α={self.alpha}, steps={self.steps})")

    def _generate_perturbation(
        self,
        X     : np.ndarray,
        y     : np.ndarray,
        model : object,
    ) -> tuple[np.ndarray, int]:

        X_adv      = X.copy()
        X_adv_best = X.copy()
        asr_best   = 0.0
        n_queries  = 0
        n_samples, n_features = X.shape

        # Caja L-inf — el ataque nunca sale de este rango respecto al original
        X_min_eps = X - self.epsilon
        X_max_eps = X + self.epsilon

        # Gradient masking estático — calculado una sola vez
        f_mask = getattr(self, 'forward_mask', None)
        if f_mask is None and self.dc is not None:
            f_mask = self.dc.forward_mask

        already_evaded = np.zeros(n_samples, dtype=bool)

        for step in range(self.steps):
            active_mask = ~already_evaded
            if not active_mask.any():
                break

            shap_vals = self.explainer.shap_values(X_adv, check_additivity=False)
            n_queries += n_samples

            direction_raw   = self._compute_direction(shap_vals, y, n_samples, n_features)
            shap_magnitudes = np.abs(direction_raw)

            # Top-K adaptado: ranking solo entre features Forward si hay física
            if f_mask is not None:
                shap_forward    = shap_magnitudes.copy()
                shap_forward[:, ~f_mask] = 0.0
                K_FEATURES      = min(10, int(f_mask.sum()))
                top_k_threshold = np.sort(shap_forward, axis=1)[:, -K_FEATURES][:, np.newaxis]
                top_k_mask = (shap_forward >= top_k_threshold) & (shap_forward > 0.0)
            else:
                K_FEATURES      = 10
                top_k_threshold = np.sort(shap_magnitudes, axis=1)[:, -K_FEATURES][:, np.newaxis]
                top_k_mask      = shap_magnitudes >= top_k_threshold

            direction_final = np.where(top_k_mask, np.sign(direction_raw), 0.0)
            # Gradient masking implícito en shap_forward — no hace falta línea adicional

            # Paso alpha solo en muestras activas
            perturbation = self.alpha * direction_final
            X_adv[active_mask] = X_adv[active_mask] + perturbation[active_mask]

            # Proyección L-inf
            X_adv = np.clip(X_adv, X_min_eps, X_max_eps)

            # Proyección causal — aplicar al final de cada paso y re-clipar
            if self.dc is not None and hasattr(self.dc, 'apply_causal_graph'):
                X_adv = self.dc.apply_causal_graph(X_adv)
                X_adv = np.clip(X_adv, X_min_eps, X_max_eps)

            # Early stopping por muestra + guardar mejor estado global
            preds = model.predict(X_adv)
            n_queries += n_samples
            newly_evaded    = (preds == 0) & active_mask
            already_evaded |= newly_evaded

            asr_current = already_evaded.mean()
            if asr_current > asr_best:
                asr_best   = asr_current
                X_adv_best = X_adv.copy()

            if self.verbose:
                print(f"  step {step+1:>2}/{self.steps} | ASR: {asr_current*100:.1f}% "
                      f"| Activas: {active_mask.sum()} | Nuevas: {newly_evaded.sum()}")

        return X_adv_best, n_queries

    def _compute_direction(self, shap_vals, y, n, n_features) -> np.ndarray:
        """
        Dirección basada en SHAP puro de la clase objetivo.

        En SHAP multiclase los valores suman cero entre clases por construcción
        (propiedad de eficiencia de Shapley). Por tanto, sample_shap[0] ya
        contiene la información contrastiva completa — no es necesario combinar
        con la repulsión de otras clases, que añadiría ruido redundante.
        """
        direction = np.zeros((n, n_features), dtype=np.float32)

        is_list = isinstance(shap_vals, list)
        is_3d   = isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 3

        for i in range(n):
            if is_list:
                sample_shap = np.array([shap_vals[c][i] for c in range(len(shap_vals))])
            elif is_3d:
                # shap_vals[i] → (n_features, n_clases) — transponemos
                sample_shap = shap_vals[i].T  # → (n_clases, n_features)
            else:
                if self.mode == 'targeted':
                    direction[i] = shap_vals[i].flatten()[:n_features].astype(np.float32)
                else:
                    direction[i] = -shap_vals[i].flatten()[:n_features].astype(np.float32)
                continue

            if self.mode == 'targeted':
                combined = sample_shap[0]           # Gradiente directo hacia Benigno
            else:
                clase_real = int(y[i])
                combined = -sample_shap[clase_real]  # Alejarse de la clase real

            direction[i] = combined[:n_features].astype(np.float32)

        return direction