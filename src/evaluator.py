"""
src/evaluation/evaluator.py
================================================================
Evaluador unificado — ResNet y LightGBM comparten esta clase
================================================================
Diseño:
  - ModelEvaluator es agnóstico al tipo de modelo
  - Recibe probabilidades (np.ndarray) no el modelo directamente
  - Corrección prior bayesiana 95/5 idéntica para ambos modelos
  - Mismo formato de output → tabla comparativa directa
  - Reutilizable para ataques adversariales (Fase 2):
    evaluator.evaluate(probs_adversarial, y_test)

Métricas IDS completas:
  - F1-macro, Accuracy, AUC-ROC  (comparativa entre modelos)
  - PR-AUC                        (más informativa en datasets desbalanceados)
  - FPR — False Positive Rate     (crítico en producción IDS)
  - FNR — False Negative Rate     (ataques no detectados)
  - Recall ataque                 (sensibilidad global a ataques)

Protocolo de modelo compatible:
    Cualquier objeto con .predict_proba(X) -> np.ndarray (n, n_classes)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol

import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import (
    f1_score, accuracy_score, classification_report,
    roc_auc_score, average_precision_score,
    confusion_matrix,
)
from sklearn.preprocessing import label_binarize


# ──────────────────────────────────────────────────────────────
# PROTOCOLO — duck typing para ResNet y LightGBM
# ──────────────────────────────────────────────────────────────
class ClassifierProtocol(Protocol):
    def predict_proba(self, X: np.ndarray) -> np.ndarray: ...


# ──────────────────────────────────────────────────────────────
# RESULTADO DE EVALUACIÓN — inmutable y serializable
# ──────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class EvalResult:
    label        : str
    # métricas generales
    f1_macro     : float
    accuracy     : float
    auc_roc      : float
    pr_auc       : float          # área bajo curva Precision-Recall
    f1_per_class : np.ndarray
    # métricas específicas IDS
    fpr          : float          # False Positive Rate — benignos clasificados como ataque
    fnr          : float          # False Negative Rate — ataques no detectados
    attack_recall: float          # recall global sobre todas las clases de ataque
    # predicciones
    preds        : np.ndarray
    probs        : np.ndarray     # probs corregidas con prior
    distribution : str = "production"

    def __str__(self) -> str:
        return (
            f"[{self.label}] F1={self.f1_macro:.4f} | "
            f"AUC={self.auc_roc:.4f} | "
            f"FPR={self.fpr:.4f} | FNR={self.fnr:.4f}"
        )


# ──────────────────────────────────────────────────────────────
# RESULTADO DE CALIBRACIÓN DE UMBRALES
# ──────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class ThresholdResult:
    thresholds  : np.ndarray   # vector de umbrales óptimos por clase
    f1_val      : float        # F1-macro en val con umbrales optimizados
    f1_val_base : float        # F1-macro en val sin optimizar (baseline)
    delta_f1    : float        # ganancia neta

    def __str__(self) -> str:
        return (
            f"Threshold calibration: "
            f"val F1 {self.f1_val_base:.4f} → {self.f1_val:.4f} "
            f"(+{self.delta_f1:.4f})"
        )


# ──────────────────────────────────────────────────────────────
# OPTIMIZADOR DE UMBRALES
# Calibración post-entrenamiento — no requiere GPU ni reentrenar
# ──────────────────────────────────────────────────────────────
class ThresholdOptimizer:
    """
    Calibración óptima de umbrales de decisión por clase.

    Problema: argmax asume umbral 0.5 implícito para todas las clases.
    Con datasets desbalanceados el umbral óptimo por clase es diferente.

    Ejemplo con los resultados actuales de ResNet:
      Malware:  precision=0.99, recall=0.72 → umbral demasiado alto
      Recon:    precision=0.67, recall=0.64 → umbral desajustado
      Exploits: precision=0.78, recall=0.91 → umbral posiblemente bajo

    Método: escalar las probabilidades por clase con un vector de pesos
    y optimizar ese vector con Nelder-Mead sobre F1-macro en validación.

    IMPORTANTE: los umbrales se calibran en VAL y se aplican en TEST.
    Nunca calibrar sobre test — sería data leakage.

    Parámetros
    ----------
    n_classes  : int
    max_iter   : int
        Iteraciones máximas del optimizador Nelder-Mead.
    """

    def __init__(self, n_classes: int = 8, max_iter: int = 10000):
        self.n_classes = n_classes
        self.max_iter  = max_iter

    def fit(
        self,
        probs_val: np.ndarray,
        y_val    : np.ndarray,
    ) -> ThresholdResult:
        """
        Encuentra el vector de umbrales óptimo sobre el set de validación.

        Parámetros
        ----------
        probs_val : np.ndarray, shape (n, n_classes)
            Probabilidades ya corregidas con prior (salida de _apply_prior).
        y_val     : np.ndarray, shape (n,)
            Etiquetas reales de validación.

        Retorna
        -------
        ThresholdResult con thresholds óptimos y métricas de mejora.
        """
        # F1 baseline sin calibración
        preds_base = np.argmax(probs_val, axis=1)
        f1_base    = f1_score(y_val, preds_base, average='macro', zero_division=0)

        def neg_f1(thresholds: np.ndarray) -> float:
            thresholds_clipped = np.clip(np.abs(thresholds), 0.5, 2.0)
            # escalar probabilidades por clase y predecir
            scaled = probs_val * np.abs(thresholds)  # abs para evitar negativos
            preds  = np.argmax(scaled, axis=1)
            return -f1_score(y_val, preds, average='macro', zero_division=0)

        # punto de partida: umbrales uniformes (equivale a argmax estándar)
        x0 = np.ones(self.n_classes)

        result = minimize(
            neg_f1, x0,
            method  = 'Nelder-Mead',
            options = {'maxiter': self.max_iter, 'xatol': 1e-4, 'fatol': 1e-4},
        )

        best_thresholds = np.abs(result.x)
        f1_optimized    = -result.fun

        return ThresholdResult(
            thresholds  = best_thresholds,
            f1_val      = f1_optimized,
            f1_val_base = f1_base,
            delta_f1    = f1_optimized - f1_base,
        )

    def apply(self, probs, thresholds):
        """
        Aplica el vector de umbrales a probabilidades y devuelve predicciones.

        Parámetros
        ----------
        probs      : np.ndarray, shape (n, n_classes)
            Probabilidades corregidas con prior.
        thresholds : np.ndarray, shape (n_classes,)
            Vector de umbrales obtenido de fit().

        Retorna
        -------
        np.ndarray, shape (n,) — predicciones calibradas.
        """
        thresholds_clipped = np.clip(np.abs(thresholds), 0.5, 2.0)

        return np.argmax(probs * np.abs(thresholds), axis=1)

    def print_thresholds(
        self,
        thresholds : np.ndarray,
        class_names: Optional[list[str]] = None,
    ) -> None:
        """Imprime el vector de umbrales con nombres de clase."""
        names = class_names or [f"class_{i}" for i in range(len(thresholds))]
        print(f"\n--- Umbrales calibrados por clase ---")
        baseline = np.ones(len(thresholds))
        for i, (name, t) in enumerate(zip(names, thresholds)):
            delta = t - baseline[i]
            arrow = "↑ más sensible" if delta > 0.05 else ("↓ más estricto" if delta < -0.05 else "≈ sin cambio")
            print(f"  {i} {name:<15} {t:.4f}  {arrow}")


# ──────────────────────────────────────────────────────────────
# EVALUADOR PRINCIPAL
# ──────────────────────────────────────────────────────────────
class ModelEvaluator:
    """
    Evaluador unificado para ResNet y LightGBM.

    Parámetros
    ----------
    n_classes   : int
        Número de clases del problema.
    class_names : list[str]
        Nombres de las clases para el classification report.
    pi_prod     : float
        Prior de ataque en producción (típicamente 0.05 para 95/5).
    """

    CLASS_NAMES_DEFAULT = [
        'Benign', 'DoS', 'DDoS', 'Web/Injection',
        'Brute Force', 'Recon', 'Malware', 'Exploits',
    ]

    def __init__(
        self,
        n_classes  : int = 8,
        class_names: Optional[list[str]] = None,
        pi_prod    : float = 0.05,
    ):
        self.n_classes   = n_classes
        self.class_names = class_names or self.CLASS_NAMES_DEFAULT
        self.pi_prod     = pi_prod

    # ------------------------------------------------------------------
    def evaluate(
        self,
        probs_raw : np.ndarray,
        y_true    : np.ndarray,
        pi_train  : float,
        label     : str = "Eval",
        verbose   : bool = True,
    ) -> EvalResult:
        """
        Evalúa probabilidades crudas del modelo aplicando corrección prior.

        Parámetros
        ----------
        probs_raw : np.ndarray, shape (n, n_classes)
            Probabilidades directas del modelo (sin corrección).
        y_true    : np.ndarray, shape (n,)
            Etiquetas reales.
        pi_train  : float
            Proporción de ataques en el conjunto de entrenamiento.
        label     : str
            Nombre descriptivo para el log.
        verbose   : bool
            Si True, imprime el informe completo.
        """
        # distribución laboratorio
        preds_lab = np.argmax(probs_raw, axis=1)
        f1_lab    = f1_score(y_true, preds_lab, average='macro', zero_division=0)
        acc_lab   = accuracy_score(y_true, preds_lab)

        # corrección prior → producción
        probs_prod = self._apply_prior(probs_raw, pi_train)
        preds_prod = np.argmax(probs_prod, axis=1)
        f1_prod    = f1_score(y_true, preds_prod, average='macro', zero_division=0)
        acc_prod   = accuracy_score(y_true, preds_prod)
        f1_cls     = f1_score(
            y_true, preds_prod, average=None,
            zero_division=0, labels=list(range(self.n_classes))
        )

        # AUC-ROC multiclase OvR
        y_bin = label_binarize(y_true, classes=list(range(self.n_classes)))
        try:
            auc_roc = roc_auc_score(y_bin, probs_prod, average='macro', multi_class='ovr')
        except ValueError:
            auc_roc = float('nan')

        # PR-AUC — más informativa que ROC en datasets muy desbalanceados
        try:
            pr_auc = average_precision_score(y_bin, probs_prod, average='macro')
        except ValueError:
            pr_auc = float('nan')

        # métricas IDS desde matriz de confusión binaria (Benign vs Ataque)
        fpr, fnr, attack_recall = self._ids_metrics(y_true, preds_prod)

        result = EvalResult(
            label         = label,
            f1_macro      = f1_prod,
            accuracy      = acc_prod,
            auc_roc       = auc_roc,
            pr_auc        = pr_auc,
            f1_per_class  = f1_cls,
            fpr           = fpr,
            fnr           = fnr,
            attack_recall = attack_recall,
            preds         = preds_prod,
            probs         = probs_prod,
        )

        if verbose:
            self._print_report(result, f1_lab, acc_lab, y_true, preds_prod)

        return result

    # ------------------------------------------------------------------
    def evaluate_with_thresholds(
        self,
        probs_val : np.ndarray,
        y_val     : np.ndarray,
        probs_test: np.ndarray,
        y_test    : np.ndarray,
        pi_train  : float,
        label     : str = "Eval+Thresholds",
        verbose   : bool = True,
    ) -> tuple[EvalResult, ThresholdResult]:
        """
        Calibra umbrales en val y evalúa en test con umbrales optimizados.

        Flujo correcto:
          1. Aplica corrección prior a val y test
          2. Optimiza umbrales sobre val  (nunca sobre test)
          3. Aplica umbrales a test
          4. Evalúa y devuelve EvalResult + ThresholdResult

        Parámetros
        ----------
        probs_val  : probabilidades crudas del modelo en validación
        y_val      : etiquetas reales de validación
        probs_test : probabilidades crudas del modelo en test
        y_test     : etiquetas reales de test
        pi_train   : prior de ataque en train
        label      : nombre descriptivo para el log
        verbose    : si True imprime informe completo

        Retorna
        -------
        (EvalResult, ThresholdResult)
        """
        # corrección prior en ambos splits
        probs_val_corr  = self._apply_prior(probs_val,  pi_train)
        probs_test_corr = self._apply_prior(probs_test, pi_train)

        # calibrar umbrales en VAL
        optimizer        = ThresholdOptimizer(n_classes=self.n_classes)
        threshold_result = optimizer.fit(probs_val_corr, y_val)

        if verbose:
            print(f"\n{threshold_result}")
            optimizer.print_thresholds(threshold_result.thresholds, self.class_names)

        # aplicar umbrales calibrados a TEST
        preds_test_cal = optimizer.apply(probs_test_corr, threshold_result.thresholds)

        # calcular métricas completas sobre test calibrado
        f1_prod  = f1_score(y_test, preds_test_cal, average='macro', zero_division=0)
        acc_prod = accuracy_score(y_test, preds_test_cal)
        f1_cls   = f1_score(
            y_test, preds_test_cal, average=None,
            zero_division=0, labels=list(range(self.n_classes))
        )

        y_bin = label_binarize(y_test, classes=list(range(self.n_classes)))
        try:
            auc_roc = roc_auc_score(y_bin, probs_test_corr, average='macro', multi_class='ovr')
        except ValueError:
            auc_roc = float('nan')
        try:
            pr_auc = average_precision_score(y_bin, probs_test_corr, average='macro')
        except ValueError:
            pr_auc = float('nan')

        fpr, fnr, attack_recall = self._ids_metrics(y_test, preds_test_cal)

        # métricas laboratorio para el print
        preds_lab = np.argmax(probs_test_corr, axis=1)
        f1_lab    = f1_score(y_test, preds_lab, average='macro', zero_division=0)
        acc_lab   = accuracy_score(y_test, preds_lab)

        result = EvalResult(
            label         = label,
            f1_macro      = f1_prod,
            accuracy      = acc_prod,
            auc_roc       = auc_roc,
            pr_auc        = pr_auc,
            f1_per_class  = f1_cls,
            fpr           = fpr,
            fnr           = fnr,
            attack_recall = attack_recall,
            preds         = preds_test_cal,
            probs         = probs_test_corr,
        )

        if verbose:
            self._print_report(result, f1_lab, acc_lab, y_test, preds_test_cal)

        return result, threshold_result

    # ------------------------------------------------------------------
    def compare(self, results: dict[str, EvalResult]) -> None:
        """
        Tabla comparativa entre múltiples modelos/experimentos.
        Incluye métricas generales + métricas IDS críticas.

        Uso:
            evaluator.compare({
                'ResNet'    : result_resnet,
                'LightGBM'  : result_lgbm,
                'ResNet+ADV': result_adv,
            })
        """
        col = 14
        pad = 22

        print(f"\n{'='*74}")
        print("TABLA COMPARATIVA — MÉTRICAS GENERALES")
        print(f"{'='*74}")
        print(f"{'Modelo':<{pad}} {'F1-macro':>{col}} {'PR-AUC':>{col}} {'AUC-ROC':>{col}} {'Accuracy':>{col}}")
        print("-" * (pad + col * 4 + 3))
        for name, r in results.items():
            print(f"{name:<{pad}} {r.f1_macro:>{col}.4f} {r.pr_auc:>{col}.4f} {r.auc_roc:>{col}.4f} {r.accuracy:>{col}.4f}")

        print(f"\n{'='*74}")
        print("TABLA COMPARATIVA — MÉTRICAS IDS (CRÍTICAS EN PRODUCCIÓN)")
        print(f"{'='*74}")
        print(f"  FPR = benignos clasificados como ataque  (↓ mejor)")
        print(f"  FNR = ataques no detectados              (↓ mejor)")
        print(f"{'Modelo':<{pad}} {'FPR':>{col}} {'FNR':>{col}} {'Attack Recall':>{col}}")
        print("-" * (pad + col * 3 + 2))
        for name, r in results.items():
            print(f"{name:<{pad}} {r.fpr:>{col}.4f} {r.fnr:>{col}.4f} {r.attack_recall:>{col}.4f}")

        print(f"\n{'='*74}")
        print("F1 POR CLASE")
        print(f"{'='*74}")
        print(f"{'Clase':<{pad}}", end="")
        for name in results:
            print(f"{name:>{col}}", end="")
        print()
        print("-" * (pad + col * len(results)))
        for i, cls in enumerate(self.class_names):
            print(f"{cls:<{pad}}", end="")
            for r in results.values():
                val = r.f1_per_class[i] if i < len(r.f1_per_class) else float('nan')
                print(f"{val:>{col}.4f}", end="")
            print()
        print("=" * 74)

    # ------------------------------------------------------------------
    def _ids_metrics(
        self,
        y_true: np.ndarray,
        preds : np.ndarray,
    ) -> tuple[float, float, float]:
        """
        Métricas binarias IDS: colapsa multiclase a Benign(0) vs Ataque(1+).

        Retorna
        -------
        fpr          : FP / (FP + TN)  — benignos clasificados como ataque
        fnr          : FN / (FN + TP)  — ataques no detectados
        attack_recall: TP / (TP + FN)  — sensibilidad global a ataques
        """
        y_bin    = (y_true != 0).astype(int)
        pred_bin = (preds  != 0).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_bin, pred_bin, labels=[0, 1]).ravel()

        fpr           = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr           = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        attack_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        return float(fpr), float(fnr), float(attack_recall)

    # ------------------------------------------------------------------
    def _apply_prior(self, probs: np.ndarray, pi_train: float) -> np.ndarray:
        """Corrección bayesiana del prior de laboratorio al prior de producción."""
        corrected      = probs.copy().astype(np.float64)
        prior_train    = np.zeros(self.n_classes)
        prior_train[0] = 1.0 - pi_train
        prior_train[1:] = pi_train / (self.n_classes - 1)

        prior_prod     = np.zeros(self.n_classes)
        prior_prod[0]  = 1.0 - self.pi_prod
        prior_prod[1:] = self.pi_prod / (self.n_classes - 1)

        for c in range(self.n_classes):
            if prior_train[c] > 0:
                corrected[:, c] *= prior_prod[c] / prior_train[c]

        row_sums = corrected.sum(axis=1, keepdims=True)
        corrected /= np.where(row_sums > 0, row_sums, 1.0)
        return corrected.astype(np.float32)

    # ------------------------------------------------------------------
    def _print_report(
        self,
        result  : EvalResult,
        f1_lab  : float,
        acc_lab : float,
        y_true  : np.ndarray,
        preds   : np.ndarray,
    ) -> None:
        print(f"\n{'='*60}")
        print(f"EVALUACIÓN — {result.label}")
        print(f"{'='*60}")

        print(f"\n--- Distribución laboratorio (sin corrección prior) ---")
        print(f"  Accuracy : {acc_lab:.4f}")
        print(f"  F1-macro : {f1_lab:.4f}")

        print(f"\n--- Distribución producción (prior {self.pi_prod*100:.0f}/{(1-self.pi_prod)*100:.0f}) ---")
        print(f"  Accuracy      : {result.accuracy:.4f}")
        print(f"  F1-macro      : {result.f1_macro:.4f}")
        print(f"  AUC-ROC       : {result.auc_roc:.4f}")
        print(f"  PR-AUC        : {result.pr_auc:.4f}")

        print(f"\n--- Métricas IDS (producción) ---")
        print(f"  FPR           : {result.fpr:.4f}  ← benignos clasificados como ataque")
        print(f"  FNR           : {result.fnr:.4f}  ← ataques no detectados")
        print(f"  Attack Recall : {result.attack_recall:.4f}  ← ataques detectados")

        print(f"\n--- F1 por macro-clase (producción) ---")
        for i, (name, f1) in enumerate(zip(self.class_names, result.f1_per_class)):
            bar = "█" * int(f1 * 20)
            print(f"  {i} {name:<15} {f1:.4f} {bar}")

        print(f"\n--- Classification Report (producción) ---")
        print(classification_report(
            y_true, preds,
            target_names  = self.class_names,
            zero_division = 0,
        ))