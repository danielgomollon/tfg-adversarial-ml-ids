"""
src/attacks/sah_attack.py
================================================================
SAH — Semantic Anchor Hijacking
Contribución original del TFG. Ataque de norma L0 sobre TabularResNet.

Daniel Gomollón Embid — TFG 2025-2026

═══════════════════════════════════════════════════════════════
PARADIGMA: Secuestro Quirúrgico de Anclas Semánticas
═══════════════════════════════════════════════════════════════

ACE (norma L∞) : perturba todas las features Forward un poco
SAH (norma L0) : perturba k features Forward exactamente,
                 dejando el 95%+ del flujo intacto

El atacante no añade ruido — extirpa quirúrgicamente las k features
que más delatan el ataque y las reemplaza por tejido benigno,
navegando hacia el centroide benigno más cercano sin romper
las dependencias causales del flujo.

═══════════════════════════════════════════════════════════════
INNOVACIONES SOBRE SAH BÁSICO
═══════════════════════════════════════════════════════════════

1. SmoothGrad para identificación de anclas
   Un solo gradiente en redes profundas está lleno de ruido
   ("shattered gradients"). SAH promedia n_smooth gradientes
   con perturbaciones gaussianas para encontrar las k features
   verdaderamente críticas — estables bajo ruido.

2. Proyección subespacial iterativa (sin reemplazo directo)
   En lugar de "pegar" el valor del centroide benigno (Efecto
   Frankenstein), SAH avanza iterativamente en la dirección
   del centroide y se detiene en el primer paso que evasiona.
   Perturbación mínima garantizada + coherencia causal en cada paso.

3. Nearest cluster en subespacio k
   El centroide benigno se busca en el subespacio de las k
   anclas — no en las 66 dimensiones. Esto garantiza que el
   "tejido benigno" es el más compatible con el flujo de ataque
   en las dimensiones que importan.

═══════════════════════════════════════════════════════════════
COMPARATIVA EN LA MEMORIA
═══════════════════════════════════════════════════════════════

              Norma    Features tocadas   Mecanismo
  FGSM        L∞       todas              gradiente one-shot
  ACE         L∞       todas Forward      gradiente + momentum
  SAH         L0       k Forward          anclas semánticas + proyección
  
  Hipótesis: SAH ≥ ACE en sigilo (L2 menor) con ASR comparable
             porque ataca las features más discriminativas,
             no todas por igual.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional
from sklearn.cluster import KMeans

from src.attacks.base_attacks import BaseAttack
from src.utils.domain_constraints import DomainConstraints


# ===========================================================================
# CENTROIDES BENIGNOS
# ===========================================================================

@dataclass
class BenignAnchors:
    """
    Centroides benignos pre-computados para SAH.

    KMeans sobre X_train_benign en espacio escalado.
    Cuanto mayor n_clusters, más fino el nearest-cluster — pero
    con 8-12 es suficiente para capturar la diversidad del tráfico benigno.
    """
    n_clusters : int  = 10
    seed       : int  = 42
    centroids_ : Optional[np.ndarray] = field(default=None, init=False)
    fitted_    : bool = field(default=False, init=False)

    def fit(self, X_benign_scaled: np.ndarray) -> 'BenignAnchors':
        print(f"   [SAH] Computando {self.n_clusters} centroides benignos "
              f"sobre {len(X_benign_scaled):,} muestras...")
        km = KMeans(n_clusters=self.n_clusters, random_state=self.seed, n_init=10)
        km.fit(X_benign_scaled)
        self.centroids_ = km.cluster_centers_.astype(np.float32)
        self.fitted_    = True
        print(f"   [SAH] Centroides listos: shape={self.centroids_.shape}")
        return self

    def nearest_in_subspace(
        self,
        x_attack  : np.ndarray,  # (n, 66)
        k_indices : np.ndarray,  # (k,)
    ) -> np.ndarray:
        """
        Encuentra el centroide más cercano en el subespacio de k features.

        La búsqueda ocurre solo en las k dimensiones — garantiza que
        el centroide elegido es el más compatible con el flujo de ataque
        en las features que el SAH va a modificar.
        """
        if not self.fitted_:
            raise RuntimeError("Llama fit() antes de usar BenignAnchors")

        x_sub         = x_attack[:, k_indices]           # (n, k)
        centroids_sub = self.centroids_[:, k_indices]     # (n_clusters, k)

        diff    = x_sub[:, np.newaxis, :] - centroids_sub[np.newaxis, :, :]
        dist_sq = (diff ** 2).sum(axis=2)                 # (n, n_clusters)
        best_idx = dist_sq.argmin(axis=1)                 # (n,)

        return self.centroids_[best_idx]                  # (n, 66)

    def global_centroid(self) -> np.ndarray:
        """Centroide global — fallback si fitted_ es False."""
        return self.centroids_.mean(axis=0)


# ===========================================================================
# SAH ATTACK
# ===========================================================================

class SAHAttack(BaseAttack):
    """
    Semantic Anchor Hijacking — secuestro quirúrgico de anclas semánticas.

    Parámetros
    ----------
    constraints    : DomainConstraints
    benign_anchors : BenignAnchors pre-computado con X_train_benign escalado
    k_anchors      : features ancla a secuestrar (3-5 recomendado)
    n_smooth       : muestras SmoothGrad para identificar anclas estables
                     1  = gradiente puntual (ruidoso, equivalente al básico)
                     10 = SmoothGrad robusto (recomendado)
    smooth_std     : desviación del ruido gaussiano en SmoothGrad
                     relativa al rango de cada feature (0.05 recomendado)
    proj_steps     : pasos de proyección subespacial iterativa
                     más pasos = perturbación más mínima, más lento
    proj_alpha     : tamaño de cada paso de proyección [0, 1]
                     fracción del vector dirección hacia el centroide
    epsilon        : bound máximo de perturbación (compatibilidad BaseAttack)
    """

    def __init__(
        self,
        constraints    : DomainConstraints,
        benign_anchors : BenignAnchors,
        k_anchors      : int   = 4,
        n_smooth       : int   = 10,
        smooth_std     : float = 0.05,
        proj_steps     : int   = 10,
        proj_alpha     : float = 0.1,
        **kwargs,
    ):
        super().__init__(constraints, **kwargs)
        self.anchors    = benign_anchors
        self.k_anchors  = k_anchors
        self.n_smooth   = n_smooth
        self.smooth_std = smooth_std
        self.proj_steps = proj_steps
        self.proj_alpha = proj_alpha

        self.frozen_mask_t = ~self.forward_mask_t

    @property
    def name(self) -> str:
        return (f"SAH — Semantic Anchor Hijacking "
                f"(k={self.k_anchors}, smooth={self.n_smooth}, "
                f"proj_steps={self.proj_steps})")

    # ------------------------------------------------------------------
    # GENERACIÓN DE PERTURBACIÓN
    # ------------------------------------------------------------------
    def _generate_perturbation(
        self,
        X     : np.ndarray,
        y     : np.ndarray,
        model : object,
    ) -> tuple[np.ndarray, int]:

        model.eval()
        X_adv_raw = X.copy()
        n_queries = 0

        for X_batch, y_batch, start, end in self._batch_iterator(X, y):
            X_t  = self._to_tensor(X_batch)
            y_t  = torch.LongTensor(y_batch).to(self.device)
            y_target = torch.zeros_like(y_t)

            # ── Paso 1: SmoothGrad — identificar anclas estables ──────────
            # Promediamos n_smooth gradientes con ruido gaussiano.
            # Las features con gradiente alto y estable bajo ruido son
            # las verdaderas anclas semánticas del modelo.
            grad_accum = torch.zeros_like(X_t)

            for _ in range(self.n_smooth):
                # Ruido gaussiano en el espacio escalado
                noise  = torch.randn_like(X_t) * self.smooth_std
                X_noisy = (X_t + noise).requires_grad_(True)

                logits = model(X_noisy)
                loss   = F.cross_entropy(logits, y_target)
                loss.backward()
                n_queries += len(X_batch)

                grad = X_noisy.grad.detach()

                # Gradient masking asimétrico — solo Forward
                grad[:, self.frozen_mask_t] = 0.0
                grad_accum += grad.abs()

            # Media de gradientes suavizados
            smooth_grad = (grad_accum / self.n_smooth).cpu().numpy()  # (batch, 66)

            # ── Paso 2: Selección de k anclas por muestra ─────────────────
            X_batch_np  = X_batch.copy()
            X_adv_batch = X_batch.copy()

            for i in range(len(X_batch)):
                # Top-k features Forward con mayor SmoothGrad
                k_indices = np.argsort(smooth_grad[i])[-self.k_anchors:]

                # ── Paso 3: Nearest centroide benigno en subespacio k ──────
                x_single      = X_batch_np[i:i+1]
                best_centroid = self.anchors.nearest_in_subspace(
                    x_single, k_indices
                )[0]  # (66,)

                # ── Paso 4: Proyección subespacial iterativa ───────────────
                # Vector dirección hacia el centroide en las k dimensiones
                # NO reemplazamos directamente — avanzamos en pasos pequeños
                # y nos detenemos cuando el modelo cambia de opinión.
                # Esto garantiza perturbación mínima + coherencia causal.
                x_current = X_batch_np[i].copy()
                evaded    = False

                for step in range(self.proj_steps):
                    # Dirección hacia el centroide en las k anclas
                    direction_k = best_centroid[k_indices] - x_current[k_indices]

                    # Paso proporcional a la distancia restante
                    x_candidate = x_current.copy()
                    x_candidate[k_indices] += self.proj_alpha * direction_k

                    # Proyección causal — coherencia física en cada micro-paso
                    # Evita el Efecto Frankenstein: si cambiamos IAT_AVG,
                    # el causal graph recalcula THROUGHPUT automáticamente
                    x_candidate_2d = x_candidate[np.newaxis]
                    x_candidate_2d = self._apply_causal_projection(x_candidate_2d)
                    x_candidate    = x_candidate_2d[0]

                    # Evaluar si ya evasionó
                    x_t_eval = self._to_tensor(x_candidate[np.newaxis])
                    with torch.no_grad():
                        pred = model(x_t_eval).argmax(dim=1).item()
                    n_queries += 1

                    x_current = x_candidate  # avanzar siempre

                    if pred == 0:  # Evasión conseguida — parar aquí
                        evaded = True
                        if self.verbose:
                            print(f"    [SAH] muestra {i} evadida en paso {step+1}"
                                  f"/{self.proj_steps}")
                        break

                X_adv_batch[i] = x_current

                if self.verbose and not evaded:
                    print(f"    [SAH] muestra {i} no evadida tras "
                          f"{self.proj_steps} pasos")

            X_adv_raw[start:end] = X_adv_batch

        return X_adv_raw, n_queries

    def _apply_causal_projection(self, X: np.ndarray) -> np.ndarray:
        """Aplica causal graph si está disponible — coherencia física."""
        if self.dc is not None and hasattr(self.dc, 'apply_causal_graph'):
            return self.dc.apply_causal_graph(X)
        return X

    # ------------------------------------------------------------------
    # SWEEPS DE ANÁLISIS
    # ------------------------------------------------------------------
    def run_k_sweep(
        self,
        X          : np.ndarray,
        y          : np.ndarray,
        model      : object,
        k_values   : list[int] = [1, 2, 3, 4, 5, 8, 10],
        class_names: Optional[dict] = None,
    ) -> dict:
        """
        Curva ASR vs k — cuántas anclas necesita SAH para evadir.

        El resultado central de SAH para la memoria: demuestra que
        con k=3 o k=4 el ASR satura — no hace falta tocar más features.
        Eso es la firma del ataque L0 quirúrgico.
        """
        if self.verbose:
            print(f"\n[SAH] k-sweep: {k_values}")
            print(f"{'k':>4} | {'ASR':>8} | {'L2 medio':>10} | {'Linf medio':>10}")
            print("-" * 42)

        results = {}
        for k in k_values:
            atk = SAHAttack(
                self.dc, self.anchors,
                k_anchors  = k,
                n_smooth   = self.n_smooth,
                smooth_std = self.smooth_std,
                proj_steps = self.proj_steps,
                proj_alpha = self.proj_alpha,
                epsilon    = self.epsilon,
                device     = self.device,
                batch_size = self.batch_size,
                verbose    = False,
            )
            result    = atk.run(X, y, model, class_names)
            results[k] = result

            if self.verbose:
                print(f"  {k:>2} | {result.asr*100:>7.1f}% | "
                      f"{result.l2_mean:>10.4f} | {result.linf_mean:>10.4f}")

        return results

    def run_smooth_sweep(
        self,
        X          : np.ndarray,
        y          : np.ndarray,
        model      : object,
        n_values   : list[int] = [1, 3, 5, 10, 20],
        class_names: Optional[dict] = None,
    ) -> dict:
        """
        Curva ASR vs n_smooth — impacto del suavizado de gradiente.

        n_smooth=1 equivale al SAH básico sin SmoothGrad.
        Si el ASR sube con n_smooth, confirma que los gradientes
        locales eran ruidosos y SmoothGrad mejora la selección de anclas.
        """
        if self.verbose:
            print(f"\n[SAH] SmoothGrad sweep: n_smooth={n_values}")
            print(f"{'n_smooth':>10} | {'ASR':>8} | {'L2 medio':>10}")
            print("-" * 35)

        results = {}
        for n in n_values:
            atk = SAHAttack(
                self.dc, self.anchors,
                k_anchors  = self.k_anchors,
                n_smooth   = n,
                smooth_std = self.smooth_std,
                proj_steps = self.proj_steps,
                proj_alpha = self.proj_alpha,
                epsilon    = self.epsilon,
                device     = self.device,
                batch_size = self.batch_size,
                verbose    = False,
            )
            result       = atk.run(X, y, model, class_names)
            results[n]   = result

            if self.verbose:
                print(f"  {n:>8} | {result.asr*100:>7.1f}% | "
                      f"{result.l2_mean:>10.4f}")

        return results


# ===========================================================================
# SCRIPT DE VERIFICACIÓN
# ===========================================================================

if __name__ == "__main__":
    import numpy as np
    from src.utils.domain_constraints import DomainConstraints

    print("[-] Verificando SAHAttack...")
    dc = DomainConstraints.from_artifacts()

    np.random.seed(42)
    X_benign_dummy = np.random.randn(1000, 66).astype(np.float32)

    anchors = BenignAnchors(n_clusters=10).fit(X_benign_dummy)

    for k in [2, 4]:
        attack = SAHAttack(
            dc, anchors,
            k_anchors  = k,
            n_smooth   = 10,
            proj_steps = 10,
            verbose    = False,
        )
        print(f"   [✓] k={k}: {attack.name}")

    print(f"\n   Forward perturbables : {attack.forward_mask_t.sum().item()}")
    print(f"   Frozen (gradient=0)  : {attack.frozen_mask_t.sum().item()}")
    print(f"   Centroides benignos  : {anchors.centroids_.shape}")
    print(f"\n[✓] sah_attack.py listo")
    print("    Uso:")
    print("      anchors = BenignAnchors(n_clusters=10).fit(X_train_benign_sc)")
    print("      attack  = SAHAttack(dc, anchors, k_anchors=4, n_smooth=10)")
    print("      result  = attack.run(X_attacks, y_attacks, model_wrapped)")
    print("      k_sweep = attack.run_k_sweep(X_attacks, y_attacks, model_wrapped)")
    print("      s_sweep = attack.run_smooth_sweep(X_attacks, y_attacks, model_wrapped)")
