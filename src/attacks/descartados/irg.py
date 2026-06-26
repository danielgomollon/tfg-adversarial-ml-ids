"""
src/attacks/irg_attack.py
================================================================
IRG — Invariant Residual Ghosting
Contribución 100% original — BigFlow-NIDS TFG (2025-2026)

Daniel Gomollón Embid

═══════════════════════════════════════════════════════════════
PARADIGMA: Silenciamiento de Activaciones Inter-Capa
═══════════════════════════════════════════════════════════════

FGSM/PGD/ACE atacan la SALIDA del modelo (logits, loss final).
DLA          ataca el ESPACIO LATENTE (penúltima capa).
PGA          ataca los GRADIENTES DE PESOS (meta-optimización).

IRG ataca la DINÁMICA INTERNA de la red:
  minimiza el cambio de activación entre capas consecutivas.

Cuando un flujo benigno trivial pasa por una red profunda,
cada capa tiene "poco que añadir" — el cambio de activación
ΔA_l = A_l(x) - A_{l-1}(x) es pequeño en magnitud.

Cuando un flujo de ataque pasa, las capas trabajan activamente
para detectar la anomalía — ΔA_l es grande.

IRG invierte esto: perturba x para que ΔA_l → 0 en todas las
capas, obligando a la red a procesar el ataque con la misma
"indiferencia" con que procesaría tráfico completamente benigno.

═══════════════════════════════════════════════════════════════
UNIVERSALIDAD
═══════════════════════════════════════════════════════════════

A diferencia de DLA (requiere espacio latente identificable)
o PGA (requiere diferenciación de segundo orden costosa),
IRG funciona sobre cualquier red profunda:

  ResNet      → ΔA_l captura los residuos explícitos
  Transformer → ΔA_l captura el delta de atención entre capas
  MLP densa   → ΔA_l captura el cambio de representación
  TabNet      → ΔA_l captura el cambio entre pasos de atención

Solo requiere:
  1. La red tenga capas secuenciales (cualquier red profunda)
  2. Acceso a las activaciones intermedias (hooks de PyTorch)
  3. Diferenciación automática estándar (sin segundo orden)

═══════════════════════════════════════════════════════════════
INVISIBILIDAD AL VAE + MAHALANOBIS
═══════════════════════════════════════════════════════════════

El VAE + Mahalanobis de la Fase 3 detecta anomalías en el
espacio de features originales (espacio de reconstrucción).

IRG opera en el espacio de activaciones intermedias — ortogonal
al espacio de features. Un flujo puede tener features que el
VAE clasifica como anómalas, pero si sus ΔA_l son pequeños,
la ResNet lo procesa como benigno igualmente.

Esta ortogonalidad es la propiedad más importante del IRG:
  - DLA es DETECTABLE por el VAE (anomalía en features)
  - IRG es INVISIBLE al VAE (anomalía en dinámica interna)

═══════════════════════════════════════════════════════════════
LOSS FUNCTION
═══════════════════════════════════════════════════════════════

L_IRG(x) = α · L_CE(f(x), y_benigno)        [clasificación]
           + β · Σ_l ||ΔA_l(x)||²_F          [silenciar capas]
           + γ · ||x_adv - x_orig||²_2       [perturbación mínima]

donde:
  ΔA_l(x) = A_l(x) - A_{l-1}(x)   cambio entre capas l-1 y l
  ||·||²_F                          norma de Frobenius al cuadrado
  α, β, γ                           hiperparámetros de balance

Interpretación de los términos:
  α · L_CE  → empuja hacia clase benigno (objetivo principal)
  β · ΔA_l  → silencia la dinámica interna (invisibilidad)
  γ · ||Δx||² → minimiza la perturbación física (realismo)

═══════════════════════════════════════════════════════════════
DIFERENCIAS CON DLA Y PGA
═══════════════════════════════════════════════════════════════

              Espacio    Orden grad.  Universal  Costoso
  DLA         latente    1            No (1 capa) No
  PGA         parámetros 2            Sí          Sí (VRAM x2)
  IRG         activac.   1            Sí          No

IRG tiene la universalidad del PGA sin su coste computacional,
y la eficiencia del DLA sin su limitación a una sola capa.

Uso:
    attack = IRGAttack(dc, epsilon=0.1, beta=0.5, gamma=0.01)
    result = attack.run(X_attacks, y_attacks, model_wrapped)
    sweep  = attack.run_beta_sweep(X_attacks, y_attacks, model_wrapped)
"""

from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional, List
from tqdm.auto import tqdm
from src.attacks.base_attacks import BaseAttack
from src.utils.domain_constraints import DomainConstraints

class ActivationHookManager:
    def __init__(self, model: torch.nn.Module, min_size: int = 16):
        self.model = model
        self.min_size = min_size
        self._handles = []
        self._activations: List[torch.Tensor] = []

    def register(self) -> 'ActivationHookManager':
        self._handles = []
        self._activations = []
        for name, module in self.model.named_modules():
            if isinstance(module, (torch.nn.Linear, torch.nn.LayerNorm, torch.nn.BatchNorm1d, torch.nn.ReLU, torch.nn.GELU, torch.nn.SiLU)):
                handle = module.register_forward_hook(self._hook_fn)
                self._handles.append(handle)
        return self

    def _hook_fn(self, module, input, output):
        if isinstance(output, torch.Tensor) and output.numel() >= self.min_size:
            self._activations.append(output)

    def get_activations(self) -> List[torch.Tensor]:
        return self._activations

    def clear(self) -> None:
        self._activations = []

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []

class IRGAttack(BaseAttack):
    def __init__(self, constraints, epsilon=0.1, alpha=1.0, beta=0.5, gamma=0.01, steps=40, momentum=0.8, n_layers=None, **kwargs):
        super().__init__(constraints, epsilon=epsilon, **kwargs)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.steps = steps
        self.momentum = momentum
        self.alpha_step = (epsilon * 2.5) / steps # Paso óptimo dinámico
        self.n_layers = n_layers

    @property
    def name(self) -> str:
        return f"IRG Pro (α={self.alpha}, β={self.beta}, steps={self.steps})"

    def _compute_activation_delta_loss(self, activations: List[torch.Tensor]) -> torch.Tensor:
        if len(activations) < 2:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        acts = activations[-self.n_layers:] if self.n_layers is not None else activations
        loss_delta = torch.tensor(0.0, device=self.device)

        for l in range(1, len(acts)):
            A_prev_flat = acts[l - 1].reshape(acts[l - 1].shape[0], -1)
            A_curr_flat = acts[l].reshape(acts[l].shape[0], -1)

            if A_prev_flat.shape == A_curr_flat.shape:
                # Usamos L1 Loss (Absolute Error) en lugar de L2 (MSE)
                # L1 es mucho menos susceptible a la explosión de gradientes
                delta = (A_curr_flat - A_prev_flat).abs().mean(dim=1)
                loss_delta = loss_delta + delta.mean()
            else:
                loss_delta = loss_delta + A_curr_flat.abs().mean(dim=1).mean()

        return loss_delta / max(len(acts) - 1, 1)

    def _extract_inner_model(self, model: object) -> torch.nn.Module:
        if hasattr(model, 'model') and isinstance(model.model, torch.nn.Module):
            return model.model
        if isinstance(model, torch.nn.Module):
            return model
        return model

    def _generate_perturbation(self, X, y, model) -> tuple[np.ndarray, int]:
        inner_model = self._extract_inner_model(model)
        inner_model.eval()
        X_adv_raw = np.zeros_like(X)
        n_queries = 0

        hook_manager = ActivationHookManager(inner_model).register()

        try:
            pbar = tqdm(total=len(X), desc=f"👻 IRG GHOSTING (β={self.beta})", unit="flow")

            for X_batch, y_batch, start, end in self._batch_iterator(X, y):
                X_t = self._to_tensor(X_batch)
                X_orig_t = X_t.clone().detach()
                y_target = torch.zeros(len(y_batch), dtype=torch.long, device=self.device)
                
                X_adv_best = X_t.clone()
                asr_best = 0.0

                X_adv_t = X_t.clone()
                # Random Start
                noise = torch.zeros_like(X_adv_t)
                noise[:, self.forward_mask_t] = torch.empty(X_adv_t.shape[0], int(self.forward_mask_t.sum())).to(self.device).normal_(0, self.epsilon / 2).clamp(-self.epsilon, self.epsilon)
                X_adv_t = self._project_tensor(X_adv_t + noise, X_t)

                velocity = torch.zeros_like(X_adv_t)

                for step in range(self.steps):
                    X_nesterov = X_adv_t.detach() + self.alpha_step * self.momentum * velocity
                    X_nesterov = self._project_tensor(X_nesterov, X_t)
                    X_nesterov.requires_grad_(True)

                    hook_manager.clear()
                    logits = inner_model(X_nesterov)
                    activations = hook_manager.get_activations()
                    n_queries += len(X_batch)

                    # --- MULTI-OBJECTIVE GRADIENT BALANCING ---
                    # 1. Gradiente de Evasión (C&W adaptado)
                    logit_0 = logits[:, 0]
                    logit_1 = logits[:, 1]
                    loss_ce = torch.clamp(logit_1 - logit_0 + 0.1, min=0.0).mean()
                    
                    inner_model.zero_grad()
                    loss_ce.backward(retain_graph=True)
                    grad_ce = X_nesterov.grad.detach().clone()
                    grad_ce = grad_ce / (grad_ce.norm(p=2, dim=1, keepdim=True) + 1e-8)

                    # 2. Gradiente de Silenciamiento
                    X_nesterov.grad.zero_()
                    loss_delta = self._compute_activation_delta_loss(activations)
                    
                    inner_model.zero_grad()
                    loss_delta.backward()
                    grad_delta = X_nesterov.grad.detach().clone()
                    grad_delta = grad_delta / (grad_delta.norm(p=2, dim=1, keepdim=True) + 1e-8)

                    # 3. Fusión Normalizada
                    grad_total = (self.alpha * grad_ce) + (self.beta * grad_delta)
                    grad_total[:, ~self.forward_mask_t] = 0.0 # Bloqueo causal

                    # Actualización Nesterov
                    velocity = self.momentum * velocity + grad_total
                    X_adv_t = self._project_tensor(X_adv_t.detach() - self.alpha_step * velocity.sign(), X_t)

                    # --- VERIFICACIÓN ---
                    with torch.no_grad():
                        hook_manager.clear()
                        preds = inner_model(X_adv_t).argmax(dim=1)
                        asr_batch = (preds == 0).float().mean().item()
                    n_queries += len(X_batch)

                    if asr_batch > asr_best:
                        asr_best = asr_batch
                        X_adv_best = X_adv_t.clone()

                    if (preds == 0).all(): 
                        break

                X_adv_raw[start:end] = self._to_numpy(X_adv_best)
                pbar.update(len(X_batch))

            pbar.close()

        finally:
            hook_manager.remove()

        return X_adv_raw, n_queries

    def run_beta_sweep(self, X, y, model, betas=[0.0, 0.1, 0.5, 1.0, 2.0], class_names=None):
        results = {}
        for b in betas:
            attack = IRGAttack(self.dc, epsilon=self.epsilon, alpha=self.alpha, beta=b, gamma=self.gamma, steps=self.steps, momentum=self.momentum, n_layers=self.n_layers, device=self.device, verbose=False)
            results[b] = attack.run(X, y, model, class_names)
        return results


# ===========================================================================
# SCRIPT DE VERIFICACIÓN
# ===========================================================================

if __name__ == "__main__":
    import numpy as np
    from src.utils.domain_constraints import DomainConstraints

    print("[-] Verificando IRGAttack...")

    dc = DomainConstraints.from_artifacts()

    attack = IRGAttack(
        dc,
        epsilon    = 0.1,
        alpha      = 1.0,
        beta       = 0.5,
        gamma      = 0.01,
        steps      = 20,
        verbose    = True,
    )
    print(f"   [✓] Instanciado: {attack.name}")
    print(f"   Forward perturbables : {attack.forward_mask_t.sum().item()}")
    print(f"   Frozen (gradient=0)  : {(~attack.forward_mask_t).sum().item()}")

    print("\n[✓] irg_attack.py listo")
    print("    Uso:")
    print("      attack = IRGAttack(dc, epsilon=0.1, beta=0.5)")
    print("      result = attack.run(X_attacks, y_attacks, model_wrapped)")
    print("      sweep  = attack.run_beta_sweep(X_attacks, y_attacks, model_wrapped)")
    print("      layers = attack.run_layers_sweep(X_attacks, y_attacks, model_wrapped)")
    print("\n    El sweep de β es el experimento central:")
    print("      β=0.0 → baseline ACE/PGD")
    print("      β>0   → IRG activo")
    print("      Si ASR(β>0) > ASR(β=0) → silenciar capas mejora la evasión")