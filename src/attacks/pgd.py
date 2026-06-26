"""
src/attacks/pgd.py
================================================================
Implementación de Projected Gradient Descent (PGD) con
restricciones físicas (PGD-Constrained).

Combina modo Targeted/Untargeted con optimizaciones de rendimiento
(Smart Random Start y Early Stopping por batch).
"""

import numpy as np
import torch
import torch.nn.functional as F
from src.attacks.base_attacks import BaseAttack 

class PGDAttack(BaseAttack):
    def __init__(
        self, 
        constraints, 
        epsilon=0.1, 
        alpha=0.01, 
        steps=10, 
        random_start=True, 
        mode='targeted', 
        **kwargs
    ):
        super().__init__(constraints, epsilon=epsilon, **kwargs)
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        
        if mode not in ('untargeted', 'targeted'):
            raise ValueError("mode debe ser 'untargeted' o 'targeted'")
        self.mode = mode

    @property
    def name(self) -> str:
        return f"PGD ({self.mode}, steps={self.steps}, α={self.alpha})"

    def _generate_perturbation(self, X: np.ndarray, y: np.ndarray, model: object) -> tuple[np.ndarray, int]:
        model.eval()
        X_adv_raw = np.zeros_like(X)
        n_queries = 0

        for X_batch, y_batch, start, end in self._batch_iterator(X, y):
            X_t = self._to_tensor(X_batch)
            y_t = torch.LongTensor(y_batch).to(self.device)

            X_adv_t = X_t.clone()

            # Smart Random Start (Añade ruido solo a las features manipulables)
            if self.random_start:
                if self.forward_mask_t is not None:
                    # FÍSICA ON: Ruido inicial solo en variables controlables (Forward)
                    noise = torch.zeros_like(X_adv_t)
                    noise[:, self.forward_mask_t] = torch.empty(
                        X_adv_t.shape[0], self.forward_mask_t.sum()
                    ).to(self.device).normal_(0, self.epsilon / 2).clamp(-self.epsilon, self.epsilon)
                    # proyectar el ruido inicial
                    X_adv_t = X_adv_t + noise
                else:
                    # FÍSICA OFF: Ruido inicial en TODAS las variables (Matemática pura)
                    noise = torch.empty_like(X_adv_t).normal_(
                        0, self.epsilon / 2
                    ).clamp(-self.epsilon, self.epsilon).to(self.device)
                    X_adv_t = X_adv_t + noise

            # Bucle Iterativo del PGD
            for step in range(self.steps):
                X_adv_t.requires_grad_(True)
                logits = model(X_adv_t)
                
                if self.mode == 'untargeted':
                    # Nos alejamos de la clase real
                    loss = F.cross_entropy(logits, y_t)
                    loss.backward()
                    grad = X_adv_t.grad.detach()
                    X_adv_t = X_adv_t.detach() + self.alpha * grad.sign()
                else:
                    # TARGETED: Nos acercamos a la clase Benigno (0)
                    y_target = torch.zeros_like(y_t)
                    loss = F.cross_entropy(logits, y_target)
                    loss.backward()
                    grad = X_adv_t.grad.detach()

                    # --- GRADIENT MASKING (Solo si hay motor físico) ---
                    if self.frozen_mask_t is not None:
                        grad[:, self.frozen_mask_t] = 0.0
                    
                    # Paso iterativo, nos acercamos a la clase objetivo
                    X_adv_t = X_adv_t.detach() - self.alpha * grad.sign()

                n_queries += len(X_batch)

                # Proyección física + causal graph tras el salto
                X_adv_t = self._project_tensor(X_adv_t, X_t)

                # Early stopping — evaluar si el batch ya es todo benigno
                with torch.no_grad():
                    preds = model(X_adv_t).argmax(dim=1)
                    n_queries += len(X_batch) # Contabilizamos esta query extra
                    all_evaded = (preds == 0).all()

                if self.verbose and step % max(1, self.steps // 5) == 0:
                    asr_step = (preds == 0).float().mean().item()
                    print(f"    step {step+1:>3}/{self.steps} | ASR batch: {asr_step*100:.1f}%")

                if all_evaded:
                    if self.verbose:
                        print(f"    [early stop] step {step+1} — batch completo evadido")
                    break

            X_adv_raw[start:end] = self._to_numpy(X_adv_t)

        return X_adv_raw, n_queries
    
    