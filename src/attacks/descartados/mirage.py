"""
src/attacks/mirage.py
================================================================
MIRAGE — Manifold-Interpolated Residual Attack via Geometric Evasion
Contribución original del TFG. Ataque de trayectoria residual completa.

Daniel Gomollón Embid — TFG 2025-2026

═══════════════════════════════════════════════════════════════
PARADIGMA: Secuestro de Trayectoria Residual
═══════════════════════════════════════════════════════════════

DLA  : minimiza MSE en UNA capa latente
MIRAGE: minimiza MSE ponderado en TODAS las capas residuales

Fuerza al flujo de ataque a seguir la misma trayectoria de
transformación interna que el tráfico benigno — no solo llegar
al mismo punto final.

Fundamento:
  Loss = Σ_l λ_l · ||R_l(x_adv) - R_l(x_benigno)||²
  
  donde R_l(x) = salida_bloque_l(x) - entrada_bloque_l(x)
  
  λ_l : peso por capa — capas más profundas más peso porque
        están más cerca de la decisión final
"""

from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, List

from src.attacks.base_attacks import BaseAttack
from src.utils.domain_constraints import DomainConstraints


class MIRAGEAttack(BaseAttack):
    """
    MIRAGE — Manifold-Interpolated Residual Attack via Geometric Evasion.

    Parámetros
    ----------
    constraints    : DomainConstraints
    residual_blocks: lista de nn.Module — los bloques residuales de la ResNet
                     Obtener con: model_wrapped.model.get_residual_blocks()
                     o manualmente: [model.block1, model.block2, ...]
    X_anchors      : pool de flujos benignos de alta confianza (n, 66)
    epsilon        : radio de perturbación
    alpha          : tamaño de paso
    steps          : iteraciones
    momentum       : factor Nesterov (hereda de NI-ACE)
    layer_weights  : pesos por capa [λ_1, ..., λ_L]
                     None = pesos crecientes automáticos (capas profundas
                     tienen más peso — más cercanas a la decisión)
    adaptive_alpha : scheduler interno
    n_restarts     : reinicios aleatorios
    """

    def __init__(
        self,
        constraints     : DomainConstraints,
        residual_blocks : List[nn.Module],
        X_anchors       : np.ndarray,
        epsilon         : float = 0.1,
        alpha           : Optional[float] = None,
        steps           : int   = 40,
        momentum        : float = 0.4,
        layer_weights   : Optional[List[float]] = None,
        adaptive_alpha  : bool  = True,
        n_restarts      : int   = 1,
        **kwargs,
    ):
        super().__init__(constraints, epsilon=epsilon, **kwargs)

        self.residual_blocks = residual_blocks
        self.X_anchors_np    = X_anchors
        self.steps           = steps
        self.momentum        = momentum
        self.adaptive_alpha  = adaptive_alpha
        self.n_restarts      = n_restarts
        self.alpha           = alpha if alpha is not None else (epsilon * 2.5 / steps)
        if self.forward_mask_t is not None:
            self.frozen_mask_t = ~self.forward_mask_t

        n_blocks = len(residual_blocks)
        if layer_weights is not None:
            self.layer_weights = layer_weights
        else:
            # Pesos crecientes: capas más profundas tienen más influencia
            # λ_l = l / Σl — normalizado para que sumen 1
            weights = [float(i + 1) for i in range(n_blocks)]
            total   = sum(weights)
            self.layer_weights = [w / total for w in weights]

        # Hooks para capturar entradas y salidas de cada bloque
        self._block_inputs  : List[Optional[torch.Tensor]] = [None] * n_blocks
        self._block_outputs : List[Optional[torch.Tensor]] = [None] * n_blocks
        self._hook_handles  = []

    @property
    def name(self) -> str:
        return (f"MIRAGE — Residual Trajectory Attack "
                f"(L={len(self.residual_blocks)}, "
                f"μ={self.momentum}, ε={self.epsilon})")

    def _register_hooks(self) -> None:
        """Registra hooks de entrada y salida en cada bloque residual."""
        self._hook_handles = []
        for l, block in enumerate(self.residual_blocks):
            # Captura la entrada al bloque (para calcular el residuo)
            def make_input_hook(idx):
                def hook(module, input, output):
                    self._block_inputs[idx]  = input[0]
                    self._block_outputs[idx] = output
                return hook
            handle = block.register_forward_hook(make_input_hook(l))
            self._hook_handles.append(handle)

    def _remove_hooks(self) -> None:
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles = []

    def _compute_residuals(self) -> List[torch.Tensor]:
        """
        Calcula R_l = output_l - input_l para cada bloque.
        Requiere haber hecho un forward pass con los hooks activos.
        """
        residuals = []
        for l in range(len(self.residual_blocks)):
            inp = self._block_inputs[l]
            out = self._block_outputs[l]
            if inp is not None and out is not None:
                # Si las dimensiones no coinciden (proyección skip),
                # usar solo el output directamente
                if inp.shape == out.shape:
                    residuals.append(out - inp)
                else:
                    residuals.append(out)
        return residuals

    def _residual_trajectory_loss(
        self,
        residuals_adv    : List[torch.Tensor],
        residuals_target : List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Loss = Σ_l λ_l · MSE(R_l(x_adv), R_l(x_benigno))

        Fuerza que la trayectoria residual del ataque se acerque
        a la trayectoria del flujo benigno asignado.
        """
        total_loss = torch.tensor(0.0, device=self.device)
        mse        = nn.MSELoss(reduction='mean')

        for l, (r_adv, r_target, lw) in enumerate(
            zip(residuals_adv, residuals_target, self.layer_weights)
        ):
            total_loss = total_loss + lw * mse(r_adv, r_target.detach())

        return total_loss

    def _generate_perturbation(
        self,
        X     : np.ndarray,
        y     : np.ndarray,
        model : object,
    ) -> tuple[np.ndarray, int]:

        pytorch_model = model.model
        pytorch_model.eval()

        X_adv_raw = np.zeros_like(X)
        n_queries = 0

        X_anchors_t = torch.tensor(
            self.X_anchors_np, dtype=torch.float32, device=self.device
        )

        self._register_hooks()

        try:
            # Pre-computar trayectorias residuales del pool de anclas
            with torch.no_grad():
                _, _ = pytorch_model(X_anchors_t)
                residuals_anchors = [
                    r.clone().detach() for r in self._compute_residuals()
                ]
            n_queries += len(X_anchors_t)
            # residuals_anchors[l] shape: (n_anchors, hidden_dim)

            for X_batch, y_batch, start, end in self._batch_iterator(X, y):
                X_t = self._to_tensor(X_batch)

                X_adv_best = X_t.clone()
                asr_best   = 0.0

                for restart in range(self.n_restarts):
                    X_adv_t = X_t.clone()

                    # Random start asimétrico en Forward
                    noise = torch.zeros_like(X_adv_t)
                    noise[:, self.forward_mask_t] = torch.empty(
                        X_adv_t.shape[0],
                        int(self.forward_mask_t.sum()),
                    ).to(self.device).normal_(0, self.epsilon / 2).clamp(
                        -self.epsilon, self.epsilon
                    )
                    X_adv_t = self._project_tensor(X_adv_t + noise, X_t)

                    # ── Nearest Anchor en espacio residual ────────────────
                    # Asignar a cada muestra la ancla cuya trayectoria
                    # residual es más cercana — más fácil de alcanzar
                    with torch.no_grad():
                        _, _ = pytorch_model(X_adv_t)           # ← X_adv_t es lo correcto
                        residuals_batch = self._compute_residuals()     # ← nombre correcto
                    n_queries += len(X_batch)

                    # Distancia entre trayectorias: suma de MSE por capa
                    # (batch, n_anchors)
                    traj_dist = torch.zeros(
                        len(X_batch), len(X_anchors_t), device=self.device
                    )
                    for l, (r_b, r_a) in enumerate(
                        zip(residuals_batch, residuals_anchors)
                    ):
                        # r_b: (batch, dim), r_a: (n_anchors, dim)
                        diff     = r_b.unsqueeze(1) - r_a.unsqueeze(0)
                        traj_dist += self.layer_weights[l] * (diff**2).mean(dim=2)

                    best_idx         = traj_dist.argmin(dim=1)
                    # Trayectoria objetivo por muestra
                    residuals_target = [
                        r_a[best_idx] for r_a in residuals_anchors
                    ]

                    velocity      = torch.zeros_like(X_adv_t)
                    loss_prev     = float('inf')
                    alpha_actual  = self.alpha
                    patience      = 0
                    MAX_PATIENCE  = 3
                    X_adv_restart = X_t.clone()
                    asr_restart   = 0.0

                    # ── Bucle Nesterov sobre trayectoria residual ─────────
                    for step in range(self.steps):

                        # Nesterov look-ahead
                        X_nesterov = (
                            X_adv_t.detach()
                            + alpha_actual * self.momentum * velocity
                        )
                        X_nesterov = self._project_tensor(X_nesterov, X_t)
                        X_nesterov.requires_grad_(True)

                        # Forward para capturar residuos en punto futuro
                        _, _ = pytorch_model(X_nesterov)
                        n_queries += len(X_batch)
                        residuals_nesterov = self._compute_residuals()

                        # Loss de trayectoria residual
                        loss = self._residual_trajectory_loss(
                            residuals_nesterov, residuals_target
                        )

                        pytorch_model.zero_grad()
                        loss.backward()

                        grad = X_nesterov.grad.detach()
                        grad[:, self.frozen_mask_t] = 0.0

                        grad_norm       = grad.abs().mean(
                            dim=1, keepdim=True
                        ).clamp(min=1e-8)
                        grad_normalized = grad / grad_norm

                        velocity[:, self.forward_mask_t] = (
                            self.momentum * velocity[:, self.forward_mask_t]
                            + (1 - self.momentum)
                            * grad_normalized[:, self.forward_mask_t]
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
                                patience = 0
                            loss_prev = loss_val

                        with torch.no_grad():
                            logits, _  = pytorch_model(X_adv_t)
                            preds      = logits.argmax(dim=1)

                            n_queries += len(X_batch)
                            asr_batch  = (preds == 0).float().mean().item()

                        if asr_batch > asr_restart:
                            asr_restart   = asr_batch
                            X_adv_restart = X_adv_t.clone().detach()

                        if self.verbose and step % max(1, self.steps // 5) == 0:
                            print(f"    [MIRAGE r{restart+1}] "
                                  f"step {step+1:>3}/{self.steps} | "
                                  f"ASR: {asr_batch*100:.1f}% | "
                                  f"traj_loss: {loss.item():.6f}")

                        if (preds == 0).all():
                            if self.verbose:
                                print(f"    [MIRAGE early stop] "
                                      f"step {step+1} — batch evadido")
                            break

                    if asr_restart > asr_best:
                        asr_best   = asr_restart
                        X_adv_best = X_adv_restart.clone()

                X_adv_raw[start:end] = self._to_numpy(X_adv_best)

        finally:
            self._remove_hooks()

        return X_adv_raw, n_queries


# ===========================================================================
# SCRIPT DE VERIFICACIÓN
# ===========================================================================

if __name__ == "__main__":
    from src.utils.domain_constraints import DomainConstraints

    print("[-] Verificando MIRAGEAttack...")
    dc = DomainConstraints.from_artifacts()

    print("\n[✓] MIRAGEAttack verificado estructuralmente")
    print("    Para ejecutar:")
    print("      # Extraer bloques residuales")
    print("      blocks = model_wrapped.model.get_residual_blocks()")
    print("      # o manualmente según arquitectura:")
    print("      print(model_wrapped.model)  # ver estructura")
    print("      # Seleccionar anclas benignos")
    print("      probs    = model_wrapped.predict_proba(X_test_benign)")
    print("      top_idx  = np.argsort(probs[:, 0])[-50:]")
    print("      X_anchors = X_test_benign[top_idx]")
    print("      # Ejecutar")
    print("      attack = MIRAGEAttack(dc, blocks, X_anchors, epsilon=2.0)")
    print("      result = attack.run(X_ataques, y_ataques, model_wrapped)")
