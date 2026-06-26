"""
src/models/baseline_model.py
================================================================
Baseline LightGBM para BigFlow-NIDS
================================================================
Clase LGBMBaseline:
  - Encapsula entrenamiento, inferencia y serialización
  - Sample weights por clase calculados con la misma fórmula
    que TabularResNet — comparativa justa
  - Early stopping sobre F1-macro en validación
  - Corrección prior bayesiana 95/5 en evaluación — idéntica
    a la aplicada en TabularResNet
  - Agnóstico a Optuna: recibe params externos o usa BASE_PARAMS

NOTA sobre escalado:
  LightGBM recibe datos RAW sin QuantileTransformer. Los árboles
  splitean por umbrales, no por distancias — el escalado es
  matemáticamente irrelevante. Esto es una ventaja diferencial
  frente a TabularResNet y se documenta en la comparativa.
"""

from __future__ import annotations

import os
import joblib
import numpy as np
import lightgbm as lgb
from pathlib import Path
from sklearn.metrics import (f1_score, roc_auc_score, 
                             classification_report, accuracy_score)
from typing import Optional


# ──────────────────────────────────────────────────────────────
# CONSTANTES
# CAMBIARLAS POR CONFIG CUANDO TERMINE EXPERIMENTACIÓN
# ──────────────────────────────────────────────────────────────
N_CLASSES = 8
SEED      = 42

# Nombres de macro-clases — para logging y reports
CLASS_NAMES = {
    0: 'Benign',
    1: 'DoS',
    2: 'DDoS',
    3: 'Web/Injection',
    4: 'Brute Force',
    5: 'Recon',
    6: 'Malware',
    7: 'Exploits',
}

# HABRÁ QUE HACER UN TRAINER COMBINADO PARA AMBOS MODELOS, O HACER UNO PROPIO PARA LIGHTGBM
# Pesos calculados con la misma fórmula que TabularResNet:
#   w_c = max(count) / count_c, capped a [0.5, 3.0]
# Garantiza comparativa justa — mismo sesgo de clase en ambos modelos.
# Actualizados con distribución real de train (cap=750, 1.08M flujos)
CLASS_WEIGHTS: dict[int, float] = {
    0: 0.5000,   # Benign         — cap mínimo (clase dominante)
    1: 0.8669,   # DoS
    2: 0.5000,   # DDoS           — cap mínimo
    3: 1.2017,   # Web/Injection
    4: 1.4506,   # Brute Force
    5: 0.9126,   # Recon
    6: 0.9382,   # Malware
    7: 2.1959,   # Exploits       — clase más difícil
}

# Hiperparámetros base — sólidos sin Optuna
BASE_PARAMS: dict = {
    'boosting_type'      : 'gbdt',
    'objective'          : 'multiclass',
    'num_class'          : N_CLASSES,
    'metric'             : 'multi_logloss',
    'num_leaves'         : 127,
    'max_depth'          : 10,
    'min_child_samples'  : 50,
    'min_child_weight'   : 1e-3,
    'reg_alpha'          : 0.1,
    'reg_lambda'         : 1.0,
    'min_split_gain'     : 0.01,
    'subsample'          : 0.8,
    'subsample_freq'     : 1,
    'colsample_bytree'   : 0.8,
    'learning_rate'      : 0.05,
    'n_estimators'       : 1000,
    'n_jobs'             : 4,   # estoy usando mi portátil, no voy a poner 
                                # todos los nucleos para que no se sobrecaliente 
    'random_state'       : SEED,
    'device_type'        : 'cpu',
    'verbose'            : -1,
}


class LGBMBaseline:
    """
    Wrapper limpio sobre LGBMClassifier.

    Parámetros
    ----------
    params : dict, opcional
        Hiperparámetros LightGBM. Si None, usa BASE_PARAMS.
        Optuna inyecta aquí sus mejores parámetros.
    early_stopping_rounds : int
        Rondas sin mejora antes de parar.
    pi_train : float
        Prior de ataque en train (fracción de flujos de ataque).
        Usado para corrección bayesiana en evaluación.
        Default 0.3339 — valor real del dataset con cap=750.
    pi_prod : float
        Prior de producción. Default 0.05 — distribución real 95/5.
    """

    def __init__(
        self,
        params                : Optional[dict] = None,
        early_stopping_rounds : int   = 30,
        pi_train              : float = 0.3339,
        pi_prod               : float = 0.05,
    ):
        self.params                = {**BASE_PARAMS, **(params or {})}
        self.early_stopping_rounds = early_stopping_rounds
        self.pi_train              = pi_train
        self.pi_prod               = pi_prod
        self.model: Optional[lgb.LGBMClassifier] = None
        self.best_iteration_      : int            = 0
        self.feature_importances_ : Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # ENTRENAMIENTO
    # ------------------------------------------------------------------
    def fit(
        self,
        X_train     : np.ndarray,
        y_train     : np.ndarray,
        X_val       : np.ndarray,
        y_val       : np.ndarray,
        verbose_eval: int = 50,
    ) -> "LGBMBaseline":

        sample_weights = self._compute_sample_weights(y_train)

        self.model = lgb.LGBMClassifier(**self.params)
        self.model.fit(
            X_train, y_train,
            sample_weight = sample_weights,
            eval_set      = [(X_val, y_val)],
            callbacks     = [
                lgb.early_stopping(self.early_stopping_rounds, verbose=verbose_eval > 0),
                lgb.log_evaluation(period=verbose_eval),
            ],
        )

        self.best_iteration_      = self.model.best_iteration_
        self.feature_importances_ = self.model.feature_importances_
        return self

    # ------------------------------------------------------------------
    # INFERENCIA
    # ------------------------------------------------------------------
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Modelo no entrenado. Llama a fit() primero.")
        return self.model.predict_proba(X)

    def predict_proba_corrected(self, X: np.ndarray) -> np.ndarray:
        """
        Probabilidades con corrección prior bayesiana.

        Reescala las probabilidades de la distribución de train (66/34)
        a la distribución de producción (95/5), igual que TabularResNet.

        Fórmula por clase c:
            p_prod(c|x) ∝ p_train(c|x) · (π_prod_c / π_train_c)

        Para clase 0 (Benign): π_prod=0.95, π_train=(1-pi_train)
        Para clases 1-7 (Attack): π_prod=0.05/7, π_train=pi_train/7
        """
        proba = self.predict_proba(X)          # (n, 8), distribución train

        # ratios prior por clase
        prior_ratio = np.ones(N_CLASSES, dtype=np.float64)
        prior_ratio[0] = (1.0 - self.pi_prod) / (1.0 - self.pi_train)   # Benign
        attack_ratio   = (self.pi_prod / 7.0) / (self.pi_train / 7.0)
        prior_ratio[1:] = attack_ratio

        proba_corr = proba * prior_ratio[np.newaxis, :]
        proba_corr = proba_corr / proba_corr.sum(axis=1, keepdims=True)  # renormalizar
        return proba_corr

    def predict(self, X: np.ndarray, apply_prior: bool = False) -> np.ndarray:
        """
        Predicciones de clase.

        apply_prior : bool
            Si True, aplica corrección bayesiana antes de argmax.
            Usar True para evaluación en distribución producción.
        """
        if apply_prior:
            return np.argmax(self.predict_proba_corrected(X), axis=1)
        return np.argmax(self.predict_proba(X), axis=1)

    def val_f1(self, X_val: np.ndarray, y_val: np.ndarray) -> float:
        """F1-macro en validación — métrica objetivo para Optuna."""
        preds = self.predict(X_val, apply_prior=False)  # sin prior en val (misma distrib que train)
        return f1_score(y_val, preds, average='macro', zero_division=0)


    # ------------------------------------------------------------------
    # PERSISTENCIA
    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        os.makedirs(Path(path).parent, exist_ok=True)
        joblib.dump(self, path)
        print(f"   [LGBMBaseline] Guardado en {path}")

    @classmethod
    def load(cls, path: str) -> "LGBMBaseline":
        obj = joblib.load(path)
        if not isinstance(obj, cls):
            raise TypeError(f"El fichero no contiene un LGBMBaseline: {type(obj)}")
        return obj

    # ------------------------------------------------------------------
    # PRIVADOS
    # ------------------------------------------------------------------
    @staticmethod
    def _compute_sample_weights(y: np.ndarray) -> np.ndarray:
        return np.array([CLASS_WEIGHTS[int(label)] for label in y], dtype=np.float32)