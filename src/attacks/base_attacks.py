"""
src/attacks/base_attack.py
================================================================
Clase abstracta padre para todos los ataques adversariales.

Responsabilidades:
  - Cargar y aplicar restricciones físicas (DomainConstraints).
  - Proyección (clipping) en el espacio real (físico) tras cada perturbación.
  - Saneamiento Semántico (Grafo Causal) para variables derivadas.
  - Soporte híbrido: Tensores en GPU (Colab) y Numpy en CPU (Local).
  - Manejo iterativo por lotes (Batches) para evitar OOM (Out of Memory).
  - Métricas de ataque avanzadas: ASR general, ASR por clase, L2, Linf.

Subclases deben implementar:
  _generate_perturbation(X, y, model) → X_adv_raw, n_queries

Uso:
    class FGSM(BaseAttack):
        def _generate_perturbation(self, X, y, model):
            ...

    attack = FGSM(constraints, epsilon=0.1)
    result = attack.run(X_attacks, y_attacks, model, class_names)
"""

from __future__ import annotations

import numpy as np
import torch
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from src.utils.domain_constraints import DomainConstraints, PHYSICAL_BOUNDS


# ===========================================================================
# RESULTADO DE ATAQUE
# ===========================================================================

@dataclass
class AttackResult:
    """
    Resultado de un ataque adversarial. Encapsula datos y métricas.

    Campos
    ------
    X_adv        : ejemplos adversariales (n, 66) en espacio escalado.
    X_original   : ejemplos originales sin perturbar.
    y_true       : etiquetas reales.
    y_pred_orig  : predicciones del modelo sobre originales (pre-ataque).
    y_pred_adv   : predicciones del modelo sobre adversariales (post-ataque).
    asr          : Attack Success Rate — fracción de verdaderos positivos evadidos.
    l2_mean      : perturbación L2 media por muestra.
    linf_mean    : perturbación Linf (máxima alteración) media por muestra.
    n_queries    : número total de queries al modelo.
    attack_name  : nombre del algoritmo de ataque.
    """
    X_adv        : np.ndarray
    X_original   : np.ndarray
    y_true       : np.ndarray
    y_pred_orig  : np.ndarray
    y_pred_adv   : np.ndarray
    asr          : float
    l2_mean      : float
    linf_mean    : float
    n_queries    : int
    attack_name  : str

    def summary(self) -> str:
        """Genera un resumen global del ataque para impresión por consola."""
        evaded = (self.y_pred_adv == 0).sum()
        total  = len(self.y_true)
        return (
            f"\n{'='*55}\n"
            f"ATAQUE: {self.attack_name}\n"
            f"{'='*55}\n"
            f"  Muestras atacadas   : {total:,}\n"
            f"  Evasiones exitosas  : {evaded:,} ({self.asr*100:.1f}%)\n"
            f"  ASR                 : {self.asr:.4f}\n"
            f"  Perturbación L2     : {self.l2_mean:.4f}\n"
            f"  Perturbación L∞     : {self.linf_mean:.4f}\n"
            f"  Queries al modelo   : {self.n_queries:,}\n"
        )

    def evasion_by_class(self, class_names: Optional[dict] = None) -> str:
        """Genera el desglose del Attack Success Rate por cada clase de ataque."""
        lines = ["\n  Evasión por clase (Transferibilidad a Benigno):"]
        for cls in np.unique(self.y_true):
            mask    = self.y_true == cls
            evaded  = (self.y_pred_adv[mask] == 0).sum()
            total   = mask.sum()
            rate    = evaded / total * 100 if total > 0 else 0
            name    = class_names.get(int(cls), f"Clase {cls}") if class_names else f"Clase {cls}"
            lines.append(f"    {name:<25} {evaded:>5}/{total:<5} ({rate:.1f}%)")
        return "\n".join(lines)


# ===========================================================================
# CLASE BASE ABSTRACTA
# ===========================================================================

class BaseAttack(ABC):
    """
    Clase abstracta para ataques adversariales sobre NIDS tabular.

    Parámetros
    ----------
    constraints  : DomainConstraints — motor físico y causal del dataset.
    epsilon      : radio de perturbación máxima en espacio escalado.
    device       : dispositivo PyTorch ('cuda' o 'cpu').
    batch_size   : tamaño de batch para ataques iterativos (evita OOM en GPU).
    verbose      : muestra el progreso y resultados por consola.
    """

    def __init__(
        self,
        constraints : DomainConstraints,
        epsilon     : float = 0.1,
        device      : str   = 'cuda' if torch.cuda.is_available() else 'cpu',
        batch_size  : int   = 512,
        verbose     : bool  = True,
    ):
        self.dc = constraints
        self.epsilon = epsilon
        self.device = device
        self.batch_size = batch_size
        self.verbose = verbose

        # --- INTERRUPTOR DE ABLACIÓN ---
        if self.dc is not None:
            # FÍSICA ON: Cargamos las máscaras del motor físico
            self.forward_mask = self.dc.perturbable_mask
            self.frozen_mask = ~self.dc.perturbable_mask
            
            # Tensores para cálculo rápido en GPU
            self.forward_mask_t = torch.BoolTensor(self.forward_mask).to(self.device)
            self.frozen_mask_t = torch.BoolTensor(self.frozen_mask).to(self.device)
        else:
            # FÍSICA OFF: Matemáticas sin reglas. Todas las features son atacables.
            self.forward_mask = None
            self.frozen_mask = None
            self.forward_mask_t = None
            self.frozen_mask_t = None

    # ------------------------------------------------------------------
    # INTERFAZ PÚBLICA
    # ------------------------------------------------------------------
    def run(self, X: np.ndarray, y: np.ndarray, model: object, class_names: Optional[dict] = None) -> AttackResult:
        """
        Ejecuta el ataque sobre el conjunto de muestras X.

        Parámetros
        ----------
        X           : (n, 66) en espacio escalado — muestras de ataque.
        y           : (n,) etiquetas reales.
        model       : modelo objetivo (TabularResNet o LightGBM).
        class_names : dict {int: str} para el resumen de evasión por clase.

        Retorna
        -------
        AttackResult con X_adv, métricas, predicciones y coste de evasión.
        """
        if self.verbose:
            print(f"\n[{self.name}] Atacando {len(X):,} muestras | ε={self.epsilon} | device={self.device}")

        # 1. Predicciones originales (baseline de detección)
        y_pred_orig = np.argmax(model.predict_proba(X), axis=1) if hasattr(model, 'predict_proba') else model.predict(X)
        n_correctly_detected = (y_pred_orig != 0).sum()
        
        if self.verbose:
            print(f"  Detectados previamente como ataque: {n_correctly_detected:,}/{len(X):,}")

        # 2. Generar perturbaciones (Delega la matemática al algoritmo específico)
        X_adv_raw, n_queries = self._generate_perturbation(X, y, model)

        # 3. PROYECCIÓN FÍSICA Y CAUSAL (El paso crítico final)
        X_adv = self._project_physics(X_adv_raw, X)

        # 4. Predicciones finales sobre el paquete adversarial saneado
        y_pred_adv = np.argmax(model.predict_proba(X_adv), axis=1) if hasattr(model, 'predict_proba') else model.predict(X_adv)

        # 5. Cálculo de métricas
        # ASR: de los que el modelo detectaba correctamente, cuántos evade
        mask_detected = y_pred_orig != 0
        asr = (y_pred_adv[mask_detected] == 0).mean() if mask_detected.sum() > 0 else 0.0

        delta = X_adv - X
        l2_mean = float(np.linalg.norm(delta, axis=1).mean())
        linf_mean = float(np.abs(delta).max(axis=1).mean())

        result = AttackResult(
            X_adv=X_adv, X_original=X, y_true=y,
            y_pred_orig=y_pred_orig, y_pred_adv=y_pred_adv,
            asr=asr, l2_mean=l2_mean, linf_mean=linf_mean,
            n_queries=n_queries, attack_name=self.name
        )

        if self.verbose:
            print(result.summary())
            if class_names:
                print(result.evasion_by_class(class_names))

        return result

    # ------------------------------------------------------------------
    # MÉTODOS ABSTRACTOS — cada ataque implementa su propia lógica
    # ------------------------------------------------------------------
    @abstractmethod
    def _generate_perturbation(self, X: np.ndarray, y: np.ndarray, model: object) -> tuple[np.ndarray, int]:
        """
        Genera perturbaciones matemáticas adversariales.

        Retorna
        -------
        X_adv_raw : array (n, 66) perturbado matemáticamente (sin sanear).
        n_queries : número total de queries/iteraciones al modelo.
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Nombre del ataque para logging."""
        ...

    # ------------------------------------------------------------------
    # MOTOR FÍSICO HÍBRIDO (El Saneamiento Semántico)
    # ------------------------------------------------------------------
    def _project_physics(self, X_adv_sc: np.ndarray, X_orig_sc: np.ndarray) -> np.ndarray:
        """
        Garantiza la viabilidad física del ataque operando en el mundo real.
        Aplica las restricciones físicas si están activadas, 
        o un simple recorte matemático si estamos en modo ablación.
        """
        # 1. Truncar el esfuerzo del ataque a Epsilon (Matemática base)
        delta = np.clip(X_adv_sc - X_orig_sc, -self.epsilon, self.epsilon)

        # --- MODO ABLACIÓN (FÍSICA OFF) ---
        if self.dc is None:
            X_math = X_orig_sc + delta
            # El clip a -5.2 / 5.2 es por los límites teóricos del escalador Quantile/Standard
            return np.clip(X_math, -5.199, 5.199).astype(np.float32)

        # --- MODO REALIDAD (FÍSICA ON) ---
        X_proj = X_orig_sc.copy()
        
        # 2. Aplicar Delta SOLO en variables Forward
        X_proj[:, self.forward_mask] = X_orig_sc[:, self.forward_mask] + delta[:, self.forward_mask]
        
        # 3. Restaurar variables inmutables y Backward
        X_proj[:, self.frozen_mask] = X_orig_sc[:, self.frozen_mask]

        # 4. Bajar al mundo físico
        X_phys = self.dc.to_physical_space(X_proj)

        # 5. Aplicar Clipping Absoluto de seguridad
        for feat_name, (min_val, max_val) in PHYSICAL_BOUNDS.items():
            idx = self.dc._feat_idx(feat_name)
            if idx is not None:
                X_phys[:, idx] = np.clip(X_phys[:, idx], min_val, max_val)

        # 6. Grafo Causal y vuelta a escalar
        X_phys = self.dc.apply_causal_graph(X_phys)
        return self.dc.to_scaled_space(X_phys)

    def _project_tensor(self, X_adv_t: torch.Tensor, X_orig_t: torch.Tensor) -> torch.Tensor:
        """
        Puente de proyección GPU -> CPU -> GPU.
        Vital para ataques iterativos (PGD) donde PyTorch calcula gradientes en GPU,
        pero la física y el QuantileTransformer de Scikit-Learn requieren CPU.
        """
        X_adv_np = X_adv_t.detach().cpu().numpy()
        X_orig_np = X_orig_t.detach().cpu().numpy()
        
        X_proj_np = self._project_physics(X_adv_np, X_orig_np)
        
        return torch.FloatTensor(X_proj_np).to(X_orig_t.device)

    # ------------------------------------------------------------------
    # HELPERS PYTORCH Y BATCHING
    # ------------------------------------------------------------------
    def _batch_iterator(self, X: np.ndarray, y: np.ndarray):
        """Iterador para procesar ataques en lotes y prevenir OOM en Colab."""
        n = len(X)
        for start in range(0, n, self.batch_size):
            end = min(start + self.batch_size, n)
            yield X[start:end], y[start:end], start, end

    def _to_tensor(self, X: np.ndarray) -> torch.Tensor:
        """Convierte Numpy array a Tensor en el dispositivo actual."""
        return torch.FloatTensor(X).to(self.device)

    def _to_numpy(self, t: torch.Tensor) -> np.ndarray:
        """Devuelve un Tensor desde la GPU a un Numpy array en CPU."""
        return t.detach().cpu().numpy()

    def _get_gradients(self, X_t: torch.Tensor, y_t: torch.Tensor, model_fn: callable) -> torch.Tensor:
        """
        Calcula el gradiente de la función de pérdida respecto a las entradas X.
        Usado exclusivamente por ataques de caja blanca (FGSM, PGD).
        """
        X_t = X_t.clone().requires_grad_(True)
        logits = model_fn(X_t)
        loss = torch.nn.functional.cross_entropy(logits, y_t)
        loss.backward()
        return X_t.grad.detach()