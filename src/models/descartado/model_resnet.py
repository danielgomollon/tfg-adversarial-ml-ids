""" 
model.py
Single Tabular ResNet + SE Block + Spectral Normalization

=============================================================
Arquitectura diseñada para detección de intrusiones (NIDS) sobre BigFlow-NIDS.

Componentes:
  - TabularResNet    : backbone con bloques residuales utilizando arquitectura Pre-Norm (flujo de gradiente óptimo)
  - SE Block         : Squeeze-and-Excitation para introducir mecanismo de atención por feature
  - Spectral Norm    : estabiliza gradientes limitando la constante Lipschitz, mejora robustez ante AT
  - Embedding z      : espacio latente compartido con el VAE 

Justificación matemática:
  ResNet  -> resuelve vanishing gradient, aprende representaciones profundas
  SE      -> recalibra features aprendiendo qué canales importan por contexto
  SN      -> limita constante de Lipschitz -> gradientes estables -> mejor AT
"""

import torch
import torch.nn as nn

from torch.nn.utils import spectral_norm

# ------------------------------------------------------------------
# BLOQUE RESIDUAL CON SE + SPECTRAL NORM
# ------------------------------------------------------------------
class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block.
    Aprende un peso por feature (canal) de forma adaptativa.
    ratio=4 es el estándar: comprime a dim//4 y luego expande.
    """
    def __init__(self, dim, ratio=4):
        super().__init__()
        hidden = max(dim // ratio, 8)  # mínimo 8 neuronas
        self.se = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (batch, dim)
        scale = self.se(x)   # aprende importancia de cada feature
        return x * scale     # recalibra multiplicativamente


class ResBlock(nn.Module):
    """
    Bloque residual tabular con:
      - Spectral Norm en ambas lineales (estabilidad de gradiente)
      - BatchNorm + GELU (mejor que ReLU en tabular data)
      - SE Block para atención por feature
      - Dropout para regularización
      - Skip connection
    """
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.block = nn.Sequential(
            spectral_norm(nn.Linear(dim, dim)),
            nn.BatchNorm1d(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            spectral_norm(nn.Linear(dim, dim)),
            nn.BatchNorm1d(dim),
        )
        self.se      = SEBlock(dim)
        self.act     = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        out = self.block(x)
        out = self.se(out)          # atención por feature
        out = out + residual        # skip connection
        out = self.act(out)
        out = self.dropout(out)
        return out


# ------------------------------------------------------------------
# MODELO PRINCIPAL
# ------------------------------------------------------------------
class TabularResNet(nn.Module):
    """
    Tabular ResNet + SE Block + Spectral Normalization.

    Parámetros:
        input_dim   : número de features (47 en BigFlow-NIDS)
        num_classes : número de macro-clases (8)
        hidden_dim  : dimensión oculta de los bloques (256)
        embed_dim   : dimensión del espacio latente z (128)
                      Usado por el VAE en Fase 2
        n_blocks    : número de bloques residuales (4)
        dropout     : tasa de dropout (0.15)

    Forward devuelve:
        logits  : (batch, num_classes) — para la loss de clasificación
        z       : (batch, embed_dim)   — embedding latente para el VAE
    """
    def __init__(
        self,
        input_dim   = 47,
        num_classes = 8,
        hidden_dim  = 256,
        embed_dim   = 128,
        n_blocks    = 4,
        dropout     = 0.15,
    ):
        super().__init__()

        # proyección inicial: input -> hidden_dim
        self.input_proj = nn.Sequential(
            spectral_norm(nn.Linear(input_dim, hidden_dim)),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Bloques residuales
        self.res_blocks = nn.ModuleList([
            ResBlock(hidden_dim, dropout) for _ in range(n_blocks)
        ])

        # Proyección al espacio latente z (compartido con VAE)
        self.embed_proj = nn.Sequential(
            spectral_norm(nn.Linear(hidden_dim, embed_dim)),
            nn.BatchNorm1d(embed_dim),
            nn.GELU(),
        )

        # Cabeza de clasificación
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
        )

        # Inicialización de pesos
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # x: (batch, input_dim)
        out = self.input_proj(x)

        for block in self.res_blocks:
            out = block(out)

        z      = self.embed_proj(out)     # embedding latente
        logits = self.classifier(z)       # predicción de clase

        return logits, z

    def predict(self, x):
        """Devuelve solo logits. Útil para inferencia."""
        logits, _ = self.forward(x)
        return logits