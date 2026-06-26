"""
src/attacks/ace_pro.py
================================================================
ACE Pro — Asymmetric Control Evasion (with Causal-Guided Momentum)
Contribución avanzada y original del TFG.

Optimización evolutiva de ACE para TabularResNet bajo Grafos Causales.
Utiliza "Causal-Guided Momentum" (CGM): anticipa y rectifica el gradiente
antes de acumular inercia para alinearlo con las barreras físicas,
evitando el overshooting identificado empíricamente.
"""

import numpy as np
import torch
import torch.nn.functional as F
from src.attacks.base_attacks import BaseAttack

class ACEProAttack(BaseAttack):
    def __init__(
        self,
        constraints,
        epsilon       : float = 0.1,
        alpha         : float = 0.01,
        steps         : int   = 40,
        momentum      : float = 0.5,
        random_start  : bool  = True,
        adaptive_alpha: bool  = True,
        cgm_rectify   : bool  = True,
        **kwargs,
    ):
        super().__init__(constraints, epsilon=epsilon, **kwargs)
        self.alpha          = alpha
        self.steps          = steps
        self.momentum       = momentum
        self.random_start   = random_start
        self.adaptive_alpha = adaptive_alpha
        self.cgm_rectify   = cgm_rectify
        self.frozen_mask_t = ~self.forward_mask_t

    @property
    def name(self) -> str:
        cgm_status = " + CGM" if self.cgm_rectify else ""
        return (f"ACE Pro (NI-FGSM{cgm_status} | μ={self.momentum}, steps={self.steps}, α={self.alpha})")

    def _generate_perturbation(self, X: np.ndarray, y: np.ndarray, model: object) -> tuple[np.ndarray, int]:
        
        pytorch_model = model.model if hasattr(model, 'model') else model
        pytorch_model.eval()
        
        X_adv_raw = np.zeros_like(X)
        n_queries = 0

        for X_batch, y_batch, start, end in self._batch_iterator(X, y):
            X_t      = self._to_tensor(X_batch)
            y_target = torch.zeros_like(torch.LongTensor(y_batch).to(self.device)) 

            X_adv_t = X_t.clone()

            if self.random_start:
                noise = torch.zeros_like(X_adv_t)
                noise[:, self.forward_mask_t] = torch.empty(
                    X_adv_t.shape[0], int(self.forward_mask_t.sum())
                ).to(self.device).normal_(0, self.epsilon / 2).clamp(-self.epsilon, self.epsilon)
                X_adv_t = self._project_tensor(X_adv_t + noise, X_t)

            velocity     = torch.zeros_like(X_adv_t)
            loss_prev    = float('inf')
            alpha_actual = self.alpha
            patience     = 0
            MAX_PATIENCE = 3

            X_adv_best = X_t.clone()
            asr_best   = 0.0

            for step in range(self.steps):
                X_nesterov = X_adv_t.detach() + alpha_actual * self.momentum * velocity
                X_nesterov = self._project_tensor(X_nesterov, X_t)
                X_nesterov.requires_grad_(True)

                out = pytorch_model(X_nesterov)
                logits = out[0] if isinstance(out, tuple) else out
                n_queries += len(X_batch)

                logit_target = logits[:, 0]
                logit_other  = logits[:, 1:].max(dim=1).values
                loss         = torch.clamp(logit_other - logit_target, min=0.0).mean()

                pytorch_model.zero_grad()
                loss.backward()

                grad = X_nesterov.grad.detach()
                grad[:, self.frozen_mask_t] = 0.0

                # === Causal-Guided Momentum (CGM) Corregido ===
                if self.cgm_rectify:
                    # Usamos self.epsilon (float) directamente
                    X_min_sc_t = torch.clamp(X_t - self.epsilon, 0.0, 1.0)
                    X_max_sc_t = torch.clamp(X_t + self.epsilon, 0.0, 1.0)

                    mask_wall_min = (grad < 0) & (X_nesterov <= X_min_sc_t)
                    mask_wall_max = (grad > 0) & (X_nesterov >= X_max_sc_t)

                    grad[mask_wall_min] = 0.0
                    grad[mask_wall_max] = 0.0
                # ===============================================

                grad_norm       = grad.abs().mean(dim=1, keepdim=True).clamp(min=1e-8)
                grad_normalized = grad / grad_norm

                velocity[:, self.forward_mask_t] = (
                    self.momentum * velocity[:, self.forward_mask_t]
                    + (1 - self.momentum) * grad_normalized[:, self.forward_mask_t]
                )

                X_adv_t = X_adv_t.detach() - alpha_actual * velocity.sign()
                X_adv_t = self._project_tensor(X_adv_t, X_t)

                if self.adaptive_alpha:
                    loss_val = loss.item()
                    if loss_val >= loss_prev:
                        patience += 1
                        if patience >= MAX_PATIENCE:
                            alpha_actual *= 0.5
                            patience      = 0
                    else:
                        patience  = 0
                    loss_prev = loss_val

                with torch.no_grad():
                    out_preds = pytorch_model(X_adv_t)
                    logits_preds = out_preds[0] if isinstance(out_preds, tuple) else out_preds
                    preds = logits_preds.argmax(dim=1)
                    n_queries += len(X_batch)
                    asr_batch  = (preds == 0).float().mean().item()

                if asr_batch > asr_best:
                    asr_best   = asr_batch
                    X_adv_best = X_adv_t.clone().detach()

                if (preds == 0).all():
                    break

            X_adv_raw[start:end] = self._to_numpy(X_adv_best)  

        return X_adv_raw, n_queries