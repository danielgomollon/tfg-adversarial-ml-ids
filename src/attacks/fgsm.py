"""
src/attacks/fgsm.py
================================================================
Fast Gradient Sign Method (FGSM) — Caja Blanca sobre TabularResNet.
Ataque de un solo paso. Baseline del arsenal adversarial.

Dos variantes:
  - Untargeted : maximiza el error sobre la clase real
  - Targeted   : minimiza el error hacia clase 0 (Benigno)
                 Más efectivo para evasión en IDS.
"""

import numpy as np
import torch
import torch.nn.functional as F
from src.attacks.base_attacks import BaseAttack


class FGSMAttack(BaseAttack):
    def __init__(
        self,
        constraints,
        epsilon    : float = 0.1,
        mode       : str   = 'untargeted',  # default clásico para reproducibilidad
        **kwargs,
    ):
        super().__init__(constraints, epsilon=epsilon, **kwargs)

        if mode not in ('untargeted', 'targeted'):
            raise ValueError(f"mode debe ser 'untargeted' o 'targeted', no '{mode}'")
        self.mode = mode

    @property
    def name(self) -> str:
        return f"FGSM ({self.mode})"

    def _generate_perturbation(
        self,
        X     : np.ndarray,
        y     : np.ndarray,
        model : object,
    ) -> tuple[np.ndarray, int]:
        
        model.eval()
        X_adv_raw = np.zeros_like(X)
        n_queries = 0

        for X_batch, y_batch, start, end in self._batch_iterator(X, y):
            X_t = self._to_tensor(X_batch)
            y_t = torch.LongTensor(y_batch).to(self.device)

            # Forward pass con seguimiento de gradientes
            X_t_grad = X_t.clone().requires_grad_(True)
            logits   = model(X_t_grad)

            if self.mode == 'untargeted':
                # Maximizar loss sobre la clase real
                loss = F.cross_entropy(logits, y_t)
                loss.backward()
                grad = X_t_grad.grad.detach()
                
                # Ecuación clásica: Nos alejamos de la clase real (+)
                X_adv_raw_t = X_t + self.epsilon * grad.sign()

            else:  
                # TARGETED: Empujar hacia Benigno (Clase 0)
                y_target = torch.zeros_like(y_t)
                loss = F.cross_entropy(logits, y_target)
                loss.backward()
                grad = X_t_grad.grad.detach()

                # --- GRADIENT MASKING (Solo si hay motor físico) ---
                if self.frozen_mask_t is not None:
                    grad[:, self.frozen_mask_t] = 0.0
                
                # Ecuación dirigida: Nos acercamos a la clase objetivo (-)
                X_adv_raw_t = X_t - self.epsilon * grad.sign()

            n_queries += len(X_batch)

            # Devolvemos el tensor crudo (BaseAttack.run() hará la magia física)
            X_adv_raw[start:end] = self._to_numpy(X_adv_raw_t)

        return X_adv_raw, n_queries


if __name__ == "__main__":
    from src.utils.domain_constraints import DomainConstraints
    print("[-] Verificando FGSMAttack...")
    dc = DomainConstraints.from_artifacts()
    attack = FGSMAttack(dc, epsilon=0.1, mode='targeted', verbose=False)
    print(f"   [✓] Instanciado correctamente: {attack.name}")