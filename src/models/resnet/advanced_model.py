""" 
advanced_model.py
Tabular ResNet + SE Block + SwiGLU + Learnable Skip Connections + Spectral Normalization

=============================================================
Arquitectura diseñada para detección de intrusiones (NIDS) sobre BigFlow-NIDS.

Componentes:
  - TabularResNet    : backbone con bloques residuales utilizando arquitectura Pre-Norm (flujo de gradiente óptimo)
  - SwiGLU           : Gated Linear Unit con activación Mish mejorar enrutamiento dinámico.
  - SE Block         : Squeeze-and-Excitation para introducir mecanismo de atención por feature
  - Spectral Norm    : estabiliza gradientes limitando la constante Lipschitz, mejora robustez ante AT
  - Embedding z      : espacio latente z para el VAE 
  - SkipInit         : Learnable Skip Connections para apagar ramas ruidosas bajo ataque.

Justificación matemática:
  ResNet    -> resuelve vanishing gradient, aprende representaciones profundas
  SE        -> recalibra features aprendiendo qué canales importan por contexto
  SN        -> limita constante de Lipschitz -> gradientes estables -> mejor AT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils import spectral_norm

# ------------------------------------------------------------------
# BLOQUE RESIDUAL CON SE + SwiGLU + SPECTRAL NORM 
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
            nn.Mish(),
            nn.Linear(hidden, dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (batch, dim)
        scale = self.se(x)   # aprende importancia de cada feature
        return x * scale     # recalibra multiplicativamente


class SwiGLU(nn.Module):
    """
    Gated Linear Unit (variante con Mish).
    Divide la proyección en dos, usando una mitad como "puerta" (gate) para la otra.
    Multiplica el rendimiento al permitir a la red ignorar features irrelevantes dinámicamente.
    """
    def __init__(self, in_dim, out_dim):
        super().__init__()
        # proyectamos al doble de tamaño para partirlo en dos
        self.proj = spectral_norm(nn.Linear(in_dim, out_dim * 2))

    def forward(self, x):
        x = self.proj(x)
        x1, x2 = x.chunk(2, dim=-1)
        return F.mish(x1) * x2  # puerta multiplicativa


class ResBlock(nn.Module):
    """
    Bloque residual tabular.
    - Arquitectura Pre-Norm con SwiGLU.
    - Spectral Normalization en la proyección principal.
    - Un único Dropout final para preservar la identidad del gradiente cuando hagamos AT.
    """
    def __init__(self, dim, dropout=0.1, inner_dropout=0.0):
        super().__init__()
        
        # Parámetro entrenable para la Skip Connection (iniciado a 1.0)
        # Permite a la red decidir cuánto caso hacerle a este bloque
        self.skip_gain = nn.Parameter(torch.ones(1))

        # Arquitectura Pre-Norm (Normalizar ANTES de transformar)
        #1. Transformación no lineal con Gating (Sin dropout interno)
        self.norm1 = nn.LayerNorm(dim)
        self.swiglu = SwiGLU(dim, dim)
        
        # Dropout interno configurable
        self.use_inner_dropout = inner_dropout > 0
        if self.use_inner_dropout:
            self.dropout_inner1 = nn.Dropout(inner_dropout)

        # 2. Proyección lineal estabilizada con Spectral Norm (Sin dropout interno)
        self.norm2 = nn.LayerNorm(dim)
        self.lin2 = spectral_norm(nn.Linear(dim, dim))
        
        if self.use_inner_dropout:
            self.dropout_inner2 = nn.Dropout(inner_dropout)

        # 3. metemos el SE Block al final del bloque para recalibrar features antres de SkipConnections
        self.se = SEBlock(dim)

        # 4. único dropout al final (siempre activo para regularizar salida del bloque)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x

        # Brazo de transformación (ideal para fgsm y pgd)
        out = self.norm1(x)
        out = self.swiglu(out)
        
        if self.use_inner_dropout:
            out = self.dropout_inner1(out)

        out = self.norm2(out)
        out = self.lin2(out)
        
        if self.use_inner_dropout:
            out = self.dropout_inner2(out)

        out = self.se(out)

        #importante este dropout antes dela skip que regulariza solo la rama transformada
        out = self.dropout(out) 
        
        # learnable skip connection + dropout final
        # el gradiente puede viajar por "residual" sin interrupciones
        return residual + (self.skip_gain * out)


# ------------------------------------------------------------------
# MODELO PRINCIPAL
# ------------------------------------------------------------------
class TabularResNet(nn.Module):
    """
    Tabular ResNet + SE Block + Spectral Normalization.

    Parámetros:
        input_dim     : número de features (51 en BigFlow-NIDS + 15 del IP Buffer = 66)
        num_classes   : número de macro-clases (8)
        hidden_dim    : dimensión oculta de los bloques (256)
        embed_dim     : dimensión del espacio latente z (128) Usado por VAE en Fase 2
        n_blocks      : número de bloques residuales (4)
        dropout       : tasa de dropout (0.1)
        inner_dropout : tasa de dropout interno en bloques (0.0 para proteger contra AT)

    Forward devuelve:
        logits  : (batch, num_classes) — para la loss de clasificación
        z       : (batch, embed_dim)   — embedding latente para el VAE
    """
    def __init__(
        self,
        input_dim   = 66,
        num_classes = 8,
        hidden_dim  = 256,
        embed_dim   = 128,
        n_blocks    = 4,
        dropout     = 0.1,
        inner_dropout = 0.0,   # <-- Lo dejamos a 0.0 para proteger los ataques
    ):
        super().__init__()

        # proyección inicial: input -> hidden_dim
        self.input_proj = nn.Sequential(
            spectral_norm(nn.Linear(input_dim, hidden_dim)),
            nn.LayerNorm(hidden_dim),
            nn.Mish(),
            nn.Dropout(dropout),
        )

        # Bloques residuales profundos
        self.res_blocks = nn.ModuleList([
            ResBlock(hidden_dim, dropout, inner_dropout=inner_dropout) for _ in range(n_blocks)
        ])

        # Proyección al espacio latente z (compartido con VAE)
        self.embed_proj = nn.Sequential(
            spectral_norm(nn.Linear(hidden_dim, embed_dim)),
            nn.LayerNorm(embed_dim),
            nn.Mish(),
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

        z = self.embed_proj(out)          # embedding latente
        logits = self.classifier(z)       # predicción de clase

        return logits, z

    def predict(self, x):
        """Devuelve solo logits. Útil para inferencia."""
        logits, _ = self.forward(x)
        return logits