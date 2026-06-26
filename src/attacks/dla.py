"""
src/attacks/dla_attack.py
================================================================
DLA — Deep Latent Anchoring
Contribución original del TFG. Ataque de secuestro de espacio latente
sobre TabularResNet mediante Forward Hooks de PyTorch.

Daniel Gomollón Embid — TFG 2025-2026

═══════════════════════════════════════════════════════════════
PARADIGMA: Colisión de Representación + Empuje Logit (Híbrido)
═══════════════════════════════════════════════════════════════

DLA asigna a cada muestra el ancla latente benigna más cercana.
En lugar de depender exclusivamente del MSE (que puede ser lento en
cruzar la frontera si epsilon es pequeño), esta evolución combina:
1. MSE Loss: Atrae la muestra hacia el "corazón" de la distribución
   benigna en el espacio latente (máximo realismo y sigilo).
2. Cross-Entropy Loss: Fuerza a los logits finales a clasificar como
   benigno (máxima letalidad de evasión).

El parámetro `latent_weight` controla la sinergia entre ambas fuerzas.

═══════════════════════════════════════════════════════════════
NEAREST ANCHOR ASSIGNMENT
═══════════════════════════════════════════════════════════════

Usar el mismo ancla para todas las muestras genera gradientes
contradictorios — DoS y Malware tienen representaciones latentes
muy distintas y el MSE promedio no converge para ninguna.

DLA asigna a cada muestra la ancla latente más cercana del pool,
garantizando que cada muestra tiene el objetivo de convergencia
más fácil de alcanzar desde su representación actual.

Soporta dos Modelos de Amenaza (Threat Models):
1. White-Box : Asume acceso al código y que la red devuelve (logits, latents).
               Es más rápido y no usa memoria extra.
2. Grey-Box  : Asume desconocimiento del código exacto. Usa Forward Hooks
               para infiltrarse en la GPU e interceptar cualquier capa
               objetivo, haciéndolo universal y agnóstico a la implementación.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
from typing import Optional

from src.attacks.base_attacks import BaseAttack
from src.utils.domain_constraints import DomainConstraints

class DLAAttack(BaseAttack):
    def __init__(
        self,
        constraints   : DomainConstraints,
        target_layer  : nn.Module,
        X_anchors     : np.ndarray,
        epsilon       : float = 0.1,
        alpha         : Optional[float] = None,
        extraction_mode : str   = 'grey_box',
        steps         : int   = 40,
        momentum      : float = 0.4,
        latent_weight : float = 0.5,  # 50% MSE Latente / 50% CE Clasificación
        adaptive_alpha: bool  = True, # Aterrizaje suave
        **kwargs,
    ):
        super().__init__(constraints, epsilon=epsilon, **kwargs)

        if extraction_mode not in ('white_box', 'grey_box'):
            raise ValueError("extraction_mode debe ser 'white_box' o 'grey_box'")
        
        if extraction_mode == 'grey_box' and target_layer is None:
            raise ValueError("El modo 'grey_box' requiere especificar una target_layer para el Hook.")

        self.extraction_mode = extraction_mode
        self.target_layer   = target_layer
        self.steps          = steps
        self.momentum       = momentum
        self.latent_weight  = latent_weight
        self.adaptive_alpha = adaptive_alpha
        self.alpha_start    = alpha if alpha is not None else (epsilon * 2.5 / steps)
        
        self.X_anchors_np   = X_anchors          
        self.frozen_mask_t  = ~self.forward_mask_t

        self._current_latent: Optional[torch.Tensor] = None

    @property
    def name(self) -> str:
        return f"DLA (Hybrid | steps={self.steps}, λ={self.latent_weight}, ε={self.epsilon})"

    def _hook_fn(self, module, input, output):
        """Intercepta el output de target_layer en cada forward pass."""
        self._current_latent = output

    def _extract_logits_and_latent(self, pytorch_model, X_tensor):
        """El motor dual: extrae la información según el Modelo de Amenaza."""
        out = pytorch_model(X_tensor)
        
        # Siempre aseguramos los logits (solución anti-crasheo de tuplas)
        logits = out[0] if isinstance(out, tuple) else out
        
        if self.extraction_mode == 'white_box':
            if not isinstance(out, tuple):
                raise ValueError("[!] Modo White-Box fallido: El modelo no devuelve una tupla con latentes.")
            latents = out[1]
        else:
            # Modo Grey-Box: El Hook ya ha guardado el latente en self._current_latent
            latents = self._current_latent
            
        return logits, latents

    def _generate_perturbation(
        self,
        X     : np.ndarray,
        y     : np.ndarray,
        model : object,
    ) -> tuple[np.ndarray, int]:

        device        = self.device
        pytorch_model = model.model if hasattr(model, 'model') else model
        pytorch_model.eval()

        X_adv_raw   = np.zeros_like(X)
        n_queries   = 0
        
        # Funciones de coste (Queremos MINIMIZAR ambas)
        mse_loss_fn = nn.MSELoss(reduction='none') # None para ponderar por muestra si hace falta
        ce_loss_fn  = nn.CrossEntropyLoss(reduction='mean')

        X_anchors_t = torch.tensor(self.X_anchors_np, dtype=torch.float32, device=device)

        # Plantamos el Hook solo si somos Grey-Box
        hook_handle = None
        if self.extraction_mode == 'grey_box':
            hook_handle = self.target_layer.register_forward_hook(self._hook_fn)

        try:
            # 1. Pre-computar representaciones latentes del pool de anclas
            with torch.no_grad():
                _, latents_anchors = self._extract_logits_and_latent(pytorch_model, X_anchors_t)
                latents_anchors = latents_anchors.clone().detach()
            n_queries += len(X_anchors_t)

            # 2. Bucle por lotes
            for X_batch, _, start, end in self._batch_iterator(X, y):
                X_t     = self._to_tensor(X_batch)
                X_adv_t = X_t.clone()

                with torch.no_grad():
                    _, latents_batch = self._extract_logits_and_latent(pytorch_model, X_adv_t)
                    latents_batch = latents_batch.clone().detach()
                n_queries += len(X_batch)

                diff    = latents_batch.unsqueeze(1) - latents_anchors.unsqueeze(0)
                dist_sq = (diff ** 2).sum(dim=2)       

                best_idx       = dist_sq.argmin(dim=1) 
                latent_targets = latents_anchors[best_idx]

                velocity   = torch.zeros_like(X_adv_t)
                X_adv_best = X_t.clone()
                asr_best   = 0.0
                target_labels = torch.zeros(len(X_batch), dtype=torch.long, device=device)

                # Inicialización correcta del Scheduler ANTES del bucle de pasos 
                current_alpha = self.alpha_start
                loss_prev = float('inf')
                patience = 0
                MAX_PATIENCE = 3

                # Bucle de Optimización Latente 
                for step in range(self.steps):
                    X_adv_t.requires_grad_(True)

                    logits, latent_adv = self._extract_logits_and_latent(pytorch_model, X_adv_t)
                    n_queries += len(X_batch)

                    loss_mse = mse_loss_fn(latent_adv, latent_targets).mean()
                    loss_ce  = ce_loss_fn(logits, target_labels)
                    loss = (self.latent_weight * loss_mse) + ((1.0 - self.latent_weight) * loss_ce)

                    pytorch_model.zero_grad()
                    loss.backward()

                    grad = X_adv_t.grad.detach()
                    grad[:, self.frozen_mask_t] = 0.0

                    grad_norm = grad.abs().mean(dim=1, keepdim=True).clamp(min=1e-8)
                    grad_normalized = grad / grad_norm

                    velocity[:, self.forward_mask_t] = (
                        self.momentum * velocity[:, self.forward_mask_t]
                        + (1 - self.momentum) * grad_normalized[:, self.forward_mask_t]
                    )

                    if self.adaptive_alpha:
                        loss_val = loss.item()
                        if loss_val >= loss_prev:
                            patience += 1
                            if patience >= MAX_PATIENCE:
                                current_alpha *= 0.5
                                patience = 0
                        else:
                            patience = 0
                        loss_prev = loss_val

                    X_adv_t = X_adv_t.detach() - current_alpha * velocity.sign()
                    
                    # Proyección tensorial al hipercubo epsilon
                    X_adv_t = self._project_tensor(X_adv_t, X_t)

                    if (step + 1) % 5 == 0 or step == self.steps - 1:
                        with torch.no_grad():
                            logits_preds, _ = self._extract_logits_and_latent(pytorch_model, X_adv_t)
                            preds = logits_preds.argmax(dim=1)
                            
                            n_queries += len(X_batch)
                            asr_step  = (preds == 0).float().mean().item()

                            if asr_step > asr_best:
                                asr_best   = asr_step
                                X_adv_best = X_adv_t.clone().detach()

                            if (preds == 0).all():
                                break

                X_adv_raw[start:end] = self._to_numpy(X_adv_best)

        finally:
            if hook_handle is not None:
                hook_handle.remove()
            self._current_latent = None

        return X_adv_raw, n_queries

