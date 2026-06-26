"""
src/attacks/oea.py
================================================================
OEA — Out-of-Distribution Extrapolation Attack
(Proyecto TESSERACT: Colapso por Vacío Topológico)

Fuerza a la muestra a alejarse lo máximo posible del centro
de la distribución de entrenamiento, empujándola hacia los
límites del hiperespacio donde el modelo sufre de "Fail-Open".

Referencias:
- Hendrycks & Gimpel (2016) - "A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks". 
(Servirá para justificar por qué las redes fallan en el vacío).
"""

import numpy as np
import torch
import torch.nn.functional as F
from src.attacks.base_attacks import BaseAttack

class OEAAttack(BaseAttack):
    def __init__(
        self,
        constraints,
        epsilon: float = 0.5,
        alpha: float = 0.05,
        steps: int = 50,
        **kwargs,
    ):
        super().__init__(constraints, epsilon=epsilon, **kwargs)
        self.alpha = alpha
        self.steps = steps

    @property
    def name(self) -> str:
        return f"OEA (Extrapolación OOD | steps={self.steps}, ε={self.epsilon})"

    def _generate_perturbation(self, X: np.ndarray, y: np.ndarray, model: object) -> tuple[np.ndarray, int]:
        pytorch_model = model.model if hasattr(model, 'model') else model
        pytorch_model.eval()
        
        X_adv_raw = np.zeros_like(X)
        n_queries = 0

        for X_batch, y_batch, start, end in self._batch_iterator(X, y):
            X_t = self._to_tensor(X_batch)
            X_adv_t = X_t.clone().detach()

            X_adv_best = X_t.clone()
            best_ood_score = torch.zeros(len(X_batch), device=self.device)

            for step in range(self.steps):
                X_adv_t.requires_grad_(True)
                
                # ==============================================================
                # EXTRACCIÓN SEGURA (Soporte para TabularResNet que devuelve tuplas)
                # ==============================================================
                out = pytorch_model(X_adv_t)
                logits = out[0] if isinstance(out, tuple) else out
                n_queries += len(X_batch)
                
                prob_benign = torch.softmax(logits, dim=1)[:, 0]

                # ==============================================================
                # LA FUNCIÓN DE PÉRDIDA TESSERACT (OEA)
                # ==============================================================
                evasion_loss = F.mse_loss(prob_benign, torch.ones_like(prob_benign))
                
                # Maximizamos la distancia L2 desde el origen (El Vacío)
                ood_loss = -torch.norm(X_adv_t, p=2, dim=1).mean() 
                
                total_loss = evasion_loss + 0.1 * ood_loss
                
                pytorch_model.zero_grad()
                total_loss.backward()

                grad = X_adv_t.grad.detach()
                
                # Gradient Masking físico
                if hasattr(self, 'forward_mask_t') and self.forward_mask_t is not None:
                    grad[:, ~self.forward_mask_t] = 0.0 
                
                X_adv_t = X_adv_t.detach() - self.alpha * grad.sign()
                X_adv_t = self._project_tensor(X_adv_t, X_t)

                # Registro del mejor ataque OOD
                with torch.no_grad():
                    out_preds = pytorch_model(X_adv_t)
                    logits_preds = out_preds[0] if isinstance(out_preds, tuple) else out_preds
                    preds = logits_preds.argmax(dim=1)
                    
                    current_ood_score = torch.norm(X_adv_t, p=2, dim=1)
                    
                    mask_better = (preds == 0) & (current_ood_score > best_ood_score)
                    
                    if mask_better.any():
                        X_adv_best[mask_better] = X_adv_t[mask_better].clone()
                        best_ood_score[mask_better] = current_ood_score[mask_better]

            X_adv_raw[start:end] = self._to_numpy(X_adv_best)

        return X_adv_raw, n_queries