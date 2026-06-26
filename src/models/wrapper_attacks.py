"""
src/models/wrapper_attacks.py
================================================================
Wrapper para adaptar los modelos (TabularResNet y LightGBM) a los ataques adversarios.

Los ataques esperan un modelo que exponga métodos estándar como 'predict_proba' 
(estilo Scikit-Learn) para calcular el ASR y 'eval()'.
Este wrapper unifica la API para que el pipeline de ataques sea agnóstico
a la arquitectura subyacente (Red Neuronal vs Árboles) ocultando la complejidad 
del espacio latente 'z' y la inferencia de PyTorch.
"""

import os
import torch
import joblib
import numpy as np

from src.config import Config
from src.models.resnet.advanced_model import TabularResNet

class ResNetWrapper:
    def __init__(self, model: torch.nn.Module, device: str):
        self.model = model
        self.device = device
        self.model.eval() # Aseguramos que siempre esté en modo evaluación
    
    def eval(self):
        """Permite a los ataques llamar a model.eval() de forma segura."""
        self.model.eval()
        return self
        
    def __call__(self, x_tensor: torch.Tensor) -> torch.Tensor:
        """Llamada directa para el cálculo de gradientes en PyTorch (Caja Blanca)."""
        logits, _ = self.model(x_tensor)
        return logits
        
    def predict_proba(self, X_np: np.ndarray) -> np.ndarray:
        """Inferencia desde Numpy para compatibilidad con la clase BaseAttack."""
        self.model.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor(X_np).to(self.device)
            logits, _ = self.model(X_t)
            probs = torch.nn.functional.softmax(logits, dim=1)
            return probs.cpu().numpy()

    def predict(self, X_np: np.ndarray) -> np.ndarray:
        """Devuelve directamente las clases predichas."""
        probs = self.predict_proba(X_np)
        return np.argmax(probs, axis=1)


class LightGBMWrapper:
    """Wrapper para la arquitectura de árboles (LightGBM)"""
    def __init__(self, model):
        self.model = model

    def eval(self):
        """Dummy method. LightGBM no necesita modo eval, pero el pipeline lo llamará."""
        return self
        
    def __call__(self, x_tensor):
        """Bloqueo de seguridad: LightGBM no tiene gradientes diferenciables."""
        raise NotImplementedError("LightGBM no soporta llamadas forward con tensores para cálculo de gradientes. Usa SGFPAttack con SHAP.")
        
    def predict_proba(self, X_np: np.ndarray) -> np.ndarray:
        """Inferencia directa desde Numpy devolviendo (N, C)."""
        return self.model.predict_proba(X_np)

    def predict(self, X_np: np.ndarray) -> np.ndarray:
        """Devuelve directamente las clases predichas."""
        return self.model.predict(X_np)


def load_resnet_for_attack(device: str, input_dim: int, models_path: str = Config.MODELS_PATH) -> ResNetWrapper:
    """
    Instancia la arquitectura, carga los pesos de la Fase 1 y devuelve el Wrapper.
    """
    print("[-] Instanciando TabularResNet...")
    model = TabularResNet(
        input_dim = input_dim,
        num_classes = Config.NUM_CLASSES,
        hidden_dim = Config.HIDDEN_DIM,
        embed_dim = Config.EMBED_DIM,
        n_blocks = Config.N_BLOCKS,
        dropout = Config.DROPOUT,
        inner_dropout = 0.0 # CRÍTICO: 0.0 para inferencia/ataques
    ).to(device)

    ruta_pesos = os.path.join(models_path, "best_resnet.pt")
    
    if not os.path.exists(ruta_pesos):
        raise FileNotFoundError(f"No se encontró el modelo en {ruta_pesos}")

    print(f"[-] Cargando pesos desde {ruta_pesos}...")
    checkpoint = torch.load(ruta_pesos, map_location=device, weights_only=False)
    
    # Manejar el diccionario de pesos del checkpoint
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)
        
    return ResNetWrapper(model, device)

def load_resnet_from_checkpoint(
    device      : str,
    input_dim   : int,
    checkpoint_path : str,
) -> ResNetWrapper:
    """
    Carga cualquier checkpoint de ResNet desde un path explícito.
    Útil para cargar modelos de Fase 3 (RE-FAT) sin modificar el wrapper original.
    """
    print(f"[-] Instanciando TabularResNet...")
    model = TabularResNet(
        input_dim   = input_dim,
        num_classes = Config.NUM_CLASSES,
        hidden_dim  = Config.HIDDEN_DIM,
        embed_dim   = Config.EMBED_DIM,
        n_blocks    = Config.N_BLOCKS,
        dropout     = Config.DROPOUT,
        inner_dropout = 0.0,
    ).to(device)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint no encontrado: {checkpoint_path}")

    print(f"[-] Cargando pesos desde {checkpoint_path}...")
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    model.load_state_dict(
        checkpoint['model'] if 'model' in checkpoint else checkpoint
    )
    print(f"    Epoch guardada: {checkpoint.get('epoch', '?')}")
    return ResNetWrapper(model, device)

def load_lgbm_for_attack(models_path: str = Config.MODELS_PATH) -> LightGBMWrapper:
    """
    Carga el modelo LightGBM preentrenado desde disco y devuelve su Wrapper.
    """
    ruta_modelo = os.path.join(models_path, "lgbm", "lgbm_baseline.pkl") # Ajusta el nombre si tu pkl se llama diferente
    
    if not os.path.exists(ruta_modelo):
        raise FileNotFoundError(f"No se encontró el modelo LightGBM en {ruta_modelo}")

    print(f"[-] Cargando modelo LightGBM desde {ruta_modelo}...")
    lgbm_model = joblib.load(ruta_modelo)
    
    return LightGBMWrapper(lgbm_model)