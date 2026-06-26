"""
src/models/lgbm_tuner.py
================================================================
Búsqueda de hiperparámetros con Optuna para LGBMBaseline
================================================================
Diseño:
  - LGBMTuner recibe un LGBMBaseline y lo optimiza
  - Separación de responsabilidades: el modelo no sabe de Optuna,
    Optuna no sabe de evaluación ni de métricas del proyecto
  - best_params se inyectan de vuelta en LGBMBaseline
  - Estudio persistido en SQLite — si el kernel muere, los trials
    se recuperan automáticamente con load_if_exists=True
  - best_model se guarda por separado 

Uso típico (desde notebook 03 o CLI):
    tuner  = LGBMTuner(n_trials=40, timeout=3600)
    result = tuner.run(X_train, y_train, X_val, y_val)
    model  = result.best_model
    print(result.summary())
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import optuna

from src.models.baseline.baseline_model import LGBMBaseline, BASE_PARAMS, N_CLASSES, SEED


# ──────────────────────────────────────────────────────────────
# RESULTADO DE TUNING — dataclass limpia, serializable
# ──────────────────────────────────────────────────────────────
@dataclass
class TuningResult:
    best_params  : dict
    best_f1      : float
    best_model   : LGBMBaseline
    study        : optuna.Study
    elapsed_s    : float
    n_trials_run : int = field(init=False)

    def __post_init__(self):
        self.n_trials_run = len(self.study.trials)

    def summary(self) -> str:
        lines = [
            "=" * 50,
            "OPTUNA TUNING — RESUMEN",
            "=" * 50,
            f"  Trials completados : {self.n_trials_run}",
            f"  Mejor val F1-macro : {self.best_f1:.4f}",
            f"  Tiempo total       : {self.elapsed_s:.0f}s",
            "",
            "  Mejores parámetros:",
        ]
        for k, v in self.best_params.items():
            lines.append(f"    {k:<25} = {v}")
        lines.append("=" * 50)
        return "\n".join(lines)


# ──────────────────────────────────────────────────────────────
# ESPACIO DE BÚSQUEDA
# ──────────────────────────────────────────────────────────────
def _suggest_params(trial: optuna.Trial) -> dict:
    """
    Espacio de búsqueda TPE para LightGBM multiclase.
    Rangos calibrados para datasets tabulares de ~1M filas.

    max_depth acotado junto a num_leaves para evitar árboles
    demasiado profundos con num_leaves alto (sobreajuste).
    """
    num_leaves = trial.suggest_int('num_leaves', 31, 255)
    # max_depth coherente con num_leaves: log2(num_leaves) + margen
    max_depth  = trial.suggest_int('max_depth', 6, 12)
    
    return {
        'num_leaves'        : trial.suggest_int('num_leaves', 31, 255),
        'max_depth'         : trial.suggest_int('max_depth', 6, 12),
        'min_child_samples' : trial.suggest_int('min_child_samples', 20, 200),
        'learning_rate'     : trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'subsample'         : trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree'  : trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha'         : trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda'        : trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'min_split_gain'    : trial.suggest_float('min_split_gain', 0.0, 1.0),
        'min_child_weight'  : trial.suggest_float('min_child_weight', 1e-4, 1e-1, log=True),
    }


# ──────────────────────────────────────────────────────────────
# TUNER
# ──────────────────────────────────────────────────────────────
class LGBMTuner:
    """
    Optimizador de hiperparámetros para LGBMBaseline.

    Parámetros
    ----------
    n_trials : int
        Número máximo de trials Optuna.
    timeout : int
        Tiempo máximo en segundos (el que llegue primero para).
    early_stopping_rounds : int
        Early stopping dentro de cada trial.
    study_name : str
        Nombre del estudio Optuna (útil para resumir/continuar).
    storage : str, opcional
        URI SQLite para persistir el estudio en disco.
        Si None, usa f"sqlite:///{study_name}.db" por defecto.
        Con load_if_exists=True el estudio se reanuda si existe.
    """

    def __init__(
        self,
        n_trials             : int = 40,        # 30-40
        timeout              : int = 3600,      # 1 hora
        early_stopping_rounds: int = 20,
        study_name           : str = "lgbm_bigflow_nids",
        storage              : Optional[str] = None,
    ):
        self.n_trials              = n_trials
        self.timeout               = timeout
        self.early_stopping_rounds = early_stopping_rounds
        self.study_name            = study_name
        self.storage               = storage or f"sqlite:///{study_name}.db"

    def run(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val  : np.ndarray,
        y_val  : np.ndarray,
    ) -> TuningResult:

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        # Estudio persistido en SQLite — se reanuda si el kernel muere
        study = optuna.create_study(
            direction  = "maximize",
            study_name = self.study_name,
            sampler    = optuna.samplers.TPESampler(seed=SEED),
            storage        = self.storage,
            load_if_exists = True,       # reanuda trials previos
        )

        already_done = len(study.trials)
        remaining    = max(0, self.n_trials - already_done)

        if already_done > 0:
            print(f"   [Optuna] Reanudando estudio — "
                  f"{already_done} trials previos, {remaining} restantes")

        def objective(trial: optuna.Trial) -> float:
            params   = _suggest_params(trial)
            baseline = LGBMBaseline(
                params                = params,
                early_stopping_rounds = self.early_stopping_rounds,
            )
            baseline.fit(X_train, y_train, X_val, y_val, verbose_eval=0)
            return baseline.val_f1(X_val, y_val)

        t0 = time.time()
        study.optimize(
            objective,
            n_trials          = self.n_trials,
            timeout           = self.timeout,
            show_progress_bar = True,
        )
        elapsed = time.time() - t0

        print(f"\n   [Optuna] Completado — mejor val F1: {study.best_value:.4f}")
        print(f"   Mejores params: {study.best_params}")

        # reentrenar modelo final con mejores params y más estimators
        # n_estimators=2000 + early_stopping=50 para convergencia óptima
        best_params = {**study.best_params, 'n_estimators': 2000}
        print("\n   [Final] Entrenando modelo final con mejores parámetros...")
        
        best_model  = LGBMBaseline(
            params                = best_params,
            early_stopping_rounds = 50,
        )
        best_model.fit(X_train, y_train, X_val, y_val, verbose_eval=100)

        return TuningResult(
            best_params = best_params,
            best_f1     = study.best_value,
            best_model  = best_model,
            study       = study,
            elapsed_s   = elapsed,
        )