"""
src/attacks/ace_attack.py
================================================================
ACE — Asymmetric Control Evasion Attack
Contribución original del TFG. Ataque de caja gris sobre TabularResNet.

Motivación:
    El PGD clásico trata todas las features como perturbables y empuja
    el gradiente en todas las dimensiones por igual. En tráfico de red,
    esto genera ejemplos físicamente imposibles.

    ACE introduce cuatro innovaciones sobre PGD:

    1.  Gradient Masking Asimétrico:
        Solo optimiza en el subespacio Forward — perturbaciones
        físicamente realizables. El presupuesto epsilon es uniforme
        por diseño: en redes, las variables de alto gradiente
        suelen coincidir con las más vigiladas. Expandir epsilon rompería
        el sigilo operativo del atacante real.

    2.  Causal Momentum Reset (CMR):
        La inercia acumulada en variables restringidas físicamente actúa
        como "Gradiente Fantasma", atascando el ataque contra muros físicos.
        CMR detecta cuándo la Proyección Física bloquea un movimiento y
        resetea la inercia de esa dimensión a 0, redirigiendo la "energía"
        del ataque hacia variables libres. Contribución 100% original.

    3.  Nesterov Look-Ahead (NI-ACE):
        Evalúa el gradiente en el punto futuro proyectado por la inercia,
        no en el punto actual. Mitiga el overshooting en el espacio físico
        restringido — demostrado empíricamente en el sweep de momentum.

    4.  Modo Dual (Evasión vs. ECHO):
        Soporta 'targeted' para cruzar la frontera de decisión (Etapa 2 y 3)
        y 'echo' para maximizar la entropía/incertidumbre (Etapa 4), 
        atacando la confianza del modelo y la fatiga del analista del SOC.

Referencias:
    Contribución original — BigFlow-NIDS TFG (2025-2026)
    
    Momentum iterativo: Dong et al. (2018) — MI-FGSM
    
    Goodfellow et al. (2014) - "Explaining and Harnessing Adversarial Examples". 
    (El paper fundacional que inventó FGSM).

    Madry et al. (2017) - "Towards Deep Learning Models Resistant to Adversarial Attacks". 
    (El paper que inventó PGD, la base matemática de ACE).

    Nguyen et al. (2015) - "Deep Neural Networks are Easily Fooled: High Confidence Predictions for Unrecognizable Images". 
    [ECHO](Aunque trata de visión, habla de cómo las redes se confunden estrepitosamente, sirviendo como base teórica para manipular la confianza).
    
    Inspirado en: Constrained Adversarial Examples (Chernikova & Oprea, 2019)
"""

import numpy as np
import torch
import torch.nn.functional as F
from src.attacks.base_attacks import BaseAttack

class ACEAttack(BaseAttack):
    """
    Asymmetric Control Evasion — PGD restringido al subespacio Forward
    con momentum asimétrico y scheduler de alpha adaptativo.

    Parámetros
    ----------
    constraints    : DomainConstraints — motor físico y causal
    epsilon        : radio de perturbación uniforme en espacio escalado
    alpha          : tamaño del paso base
    steps          : número de iteraciones
    momentum       : factor de acumulación de inercia [0, 1]
                     0.0 = ACE clásico, 0.9 = MI-ACE (recomendado)
    random_start   : ruido gaussiano asimétrico en Forward para warm-up
    adaptive_alpha : scheduler interno — reduce alpha si loss no mejora
    loss_mode      : 'targeted' (evasión) o 'echo' (máxima entropía)
    """

    def __init__(
        self,
        constraints,
        epsilon       : float = 0.1,
        alpha         : float = 0.01,
        steps         : int   = 20,
        momentum      : float = 0.0,
        random_start  : bool  = True,
        adaptive_alpha: bool  = True,
        loss_mode     : str   = 'targeted',
        **kwargs,
    ):
        super().__init__(constraints, epsilon=epsilon, **kwargs)
        self.alpha          = alpha
        self.steps          = steps
        self.momentum       = momentum
        self.random_start   = random_start
        self.adaptive_alpha = adaptive_alpha
        self.loss_mode      = loss_mode.lower()
        self.frozen_mask_t  = ~self.forward_mask_t

        if self.loss_mode not in ('targeted', 'echo'):
            raise ValueError("loss_mode debe ser 'targeted' o 'echo'")

    @property
    def name(self) -> str:
        prefijo = "ECHO" if self.loss_mode == 'echo' else "ACE"
        return f"{prefijo} (NI-FGSM + CMR | μ={self.momentum}, steps={self.steps}, α={self.alpha})"

    def _generate_perturbation(self, X: np.ndarray, y: np.ndarray, model: object) -> tuple[np.ndarray, int]:
        pytorch_model = model.model if hasattr(model, 'model') else model
        pytorch_model.eval()
        
        X_adv_raw = np.zeros_like(X)
        n_queries = 0

        for X_batch, y_batch, start, end in self._batch_iterator(X, y):
            X_t      = self._to_tensor(X_batch)
            y_t      = torch.LongTensor(y_batch).to(self.device)
            y_target = torch.zeros_like(y_t)

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

            X_adv_best  = X_t.clone()
            best_metric = -float('inf')

            for step in range(self.steps):
                # 1. Nesterov Look-Ahead
                X_nesterov = X_adv_t.detach() + alpha_actual * self.momentum * velocity
                X_nesterov = self._project_tensor(X_nesterov, X_t)
                X_nesterov.requires_grad_(True)

                # 2. Forward pass y selección de Loss
                out = pytorch_model(X_nesterov)
                logits = out[0] if isinstance(out, tuple) else out
                probs = torch.softmax(logits, dim=1)
                n_queries += len(X_batch)

                quarantine_mask = torch.zeros(len(X_batch), dtype=torch.bool, device=self.device)
                near_cliff_mask = torch.zeros(len(X_batch), dtype=torch.bool, device=self.device)

                if self.loss_mode == 'targeted':
                    # Margen Logit: Cruza la frontera hacia benigno
                    logit_target = logits[:, 0]
                    logit_other  = logits[:, 1:].max(dim=1).values
                    loss         = torch.clamp(logit_other - logit_target, min=0.0).mean()
                else:
                    # MODO ECHO: Logit-Lock + Freno Magnético
                    # Target P(Benigno) = 0.5
                    logit_benign = logits[:, 0]
                    logit_other  = logits[:, 1:].max(dim=1).values

                    # LOGIT-LOCK: En lugar de MSE en Softmax, minimizamos la diferencia de logits
                    loss = torch.abs(logit_benign - logit_other).mean()
                    
                    prob_benign = probs[:, 0]
                    
                    # ZONA DE CUARENTENA: Objetivo final (40% - 60%)
                    quarantine_mask = (prob_benign >= 0.40) & (prob_benign <= 0.60)
                    
                    # CAMPO DE FRENADO: Zona de peligro (20% a 80%)
                    near_cliff_mask = (prob_benign > 0.20) & (prob_benign < 0.80) & ~quarantine_mask

                # 3. Backward pass con Gradient Masking
                pytorch_model.zero_grad()
                loss.backward()

                grad = X_nesterov.grad.detach()
                grad[:, self.frozen_mask_t] = 0.0

                # MEJORA DE REALISMO EN MODO ECHO
                if self.loss_mode == 'echo':
                    # 1. Congelación absoluta si ya estamos en la zona perfecta
                    grad[quarantine_mask] = 0.0
                    velocity[quarantine_mask] = 0.0 # matamos inercia también
                    
                    # 2. Freno Magnético: Reducimos la velocidad al 10% si estamos cerca del abismo
                    grad[near_cliff_mask] *= 0.1

                grad_norm       = grad.abs().mean(dim=1, keepdim=True).clamp(min=1e-8)
                grad_normalized = grad / grad_norm

                # 4. Actualización de Momentum
                velocity[:, self.forward_mask_t] = (
                    self.momentum * velocity[:, self.forward_mask_t]
                    + (1 - self.momentum) * grad_normalized[:, self.forward_mask_t]
                )

                # 5. Paso adversarial y Proyección Física
                X_adv_pre = X_adv_t.detach() - alpha_actual * velocity.sign()
                X_adv_post = self._project_tensor(X_adv_pre, X_t)

                # 6. Causal Momentum Reset (CMR)
                actual_step = X_adv_post - X_adv_t.detach()
                wall_mask   = (actual_step == 0) & (velocity != 0)
                velocity[wall_mask] = 0.0 

                X_adv_t = X_adv_post

                # 7. Adaptive Alpha Scheduler
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

                # 8. Registro del mejor éxito (ASR o Entropía) y Early Stopping
                with torch.no_grad():
                    out_preds = pytorch_model(X_adv_t)
                    logits_preds = out_preds[0] if isinstance(out_preds, tuple) else out_preds
                    
                    probs_preds = torch.softmax(logits_preds, dim=1)
                    preds = logits_preds.argmax(dim=1)
                    n_queries += len(X_batch)

                    if self.loss_mode == 'targeted':
                        metric_batch = (preds == 0).float().mean().item()
                        if metric_batch > best_metric:
                            best_metric = metric_batch
                            X_adv_best  = X_adv_t.clone().detach()
                        
                        # Early stop clásico: paramos si el batch entero evade
                        if (preds == 0).all(): 
                            break
                    else:
                        # MODO ECHO: Evaluamos basándonos puramente en cuántos flujos atrapamos
                        prob_benign_preds = probs_preds[:, 0]
                        current_quarantine = ((prob_benign_preds >= 0.40) & (prob_benign_preds <= 0.60)).sum().item()
                        
                        if current_quarantine > best_metric:
                            best_metric = current_quarantine
                            X_adv_best  = X_adv_t.clone().detach()
                            
                        # Early stop si todos están en cuarentena
                        if current_quarantine == len(X_batch):
                            break

            X_adv_raw[start:end] = self._to_numpy(X_adv_best)  

        return X_adv_raw, n_queries