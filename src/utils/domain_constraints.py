"""
src/utils/domain_constraints.py
================================================================
ADN del dataset BigFlow-NIDS para ataques adversariales (Fase 2).

Arquitectura híbrida: 
Combina el diseño OOP (Factory Pattern & Dataclasses) con un motor 
físico estricto. Evita la generación de ejemplos adversarios 
"Frankenstein" matemáticamente imposibles.

Categorías de features:
  - FORWARD   : Controladas por el atacante origen (perturbables por PGD/SHAP).
  - BACKWARD  : Respuesta del servidor víctima (congeladas).
  - DERIVED   : Relaciones matemáticas entre Forward y Backward (recalculadas dinámicamente).
  - IMMUTABLE : Metadatos, IPs, Puertos, Flags L4/L7 y Buffer del NIDS (congeladas).

Principio de causalidad física:
  Un atacante real solo puede modificar lo que genera él mismo.
  Perturbar OUT_BYTES o BUF_SCAN_RATE directamente es físicamente imposible:
  el servidor decide su respuesta y el IDS calcula el buffer en base al pasado.

Uso:
    from src.utils.domain_constraints import DomainConstraints
    dc = DomainConstraints.from_artifacts('outputs/models', 'data/processed')
    mask = dc.forward_mask          # array booleano (66,)
    X_phys_corregido = dc.apply_causal_graph(X_phys_mutado)
"""

from __future__ import annotations

import numpy as np
import joblib
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


# ===========================================================================
# CLASIFICACIÓN DE FEATURES — Decisiones de Dominio y Física Estricta
# ===========================================================================

# FORWARD — El atacante origen controla estas features directamente
FORWARD_FEATURES = [
    'IN_BYTES',                     # bytes enviados por el atacante
    'IN_PKTS',                      # paquetes enviados por el atacante
    'MIN_TTL',                      # TTL del atacante
    'LONGEST_FLOW_PKT',             # tamaño mayor paquete del atacante
    'SHORTEST_FLOW_PKT',            # tamaño menor paquete del atacante
    'MIN_IP_PKT_LEN',               # longitud mínima IP del atacante
    'MAX_IP_PKT_LEN',               # longitud máxima IP del atacante
    'SRC_TO_DST_SECOND_BYTES',      # throughput atacante→destino
    'SRC_TO_DST_AVG_THROUGHPUT',    # throughput medio atacante→destino
    'NUM_PKTS_UP_TO_128_BYTES',     # distribución tamaños paquete
    'NUM_PKTS_128_TO_256_BYTES',
    'NUM_PKTS_256_TO_512_BYTES',
    'NUM_PKTS_512_TO_1024_BYTES',
    'NUM_PKTS_1024_TO_1514_BYTES',
    'TCP_WIN_MAX_IN',               # ventana TCP del atacante
    'SRC_TO_DST_IAT_MIN',           # inter-arrival times atacante→destino
    'SRC_TO_DST_IAT_MAX',
    'SRC_TO_DST_IAT_AVG',
    'SRC_TO_DST_IAT_STDDEV',
    'FLOW_DURATION_MILLISECONDS',   # el atacante controla cuánto dura el flujo
    'DURATION_IN',                  # duración flujo entrada
]

# BACKWARD — Respuesta del servidor. El atacante NO controla esto (Asimetría)
BACKWARD_FEATURES = [
    'OUT_BYTES',                    # bytes de respuesta del servidor
    'OUT_PKTS',                     # paquetes de respuesta del servidor
    'SERVER_TCP_FLAGS',             # flags TCP del servidor
    'MAX_TTL',                      # TTL del servidor
    'DST_TO_SRC_SECOND_BYTES',      # throughput destino→atacante
    'DST_TO_SRC_AVG_THROUGHPUT',    # throughput medio destino→atacante
    'TCP_WIN_MAX_OUT',              # ventana TCP del servidor
    'DST_TO_SRC_IAT_MIN',           # inter-arrival times destino→atacante
    'DST_TO_SRC_IAT_MAX',
    'DST_TO_SRC_IAT_AVG',
    'DST_TO_SRC_IAT_STDDEV',
    'DURATION_OUT',                 # duración flujo salida
]

# DERIVED — Grafo Causal. 
# No se optimizan con gradientes, se RECALCULAN matemáticamente
# a partir de las bases Forward y Backward en cada iteración del ataque.
DERIVED_FEATURES = [
    'SRC_BYTES_PER_PKT',            # IN_BYTES / IN_PKTS
    'PKT_RATIO',                    # IN_PKTS / OUT_PKTS
    'RESPONSE_RATIO',               # IN_BYTES / OUT_BYTES
    'TCP_WIN_RATIO',                # TCP_WIN_MAX_IN / TCP_WIN_MAX_OUT
    'TOTAL_BYTES',                  # IN_BYTES + OUT_BYTES
    'DURATION_PER_PKT',             # FLOW_DURATION / (IN_PKTS + OUT_PKTS)
    'IS_UNIDIRECTIONAL',            # 1.0 si OUT_PKTS == 0 else 0.0
]

# IMMUTABLE — Estructura de la conexión y motor del IDS.
# Cambiar un puerto o un flag rompe la semántica L4/L7 de la conexión real.
IMMUTABLE_FEATURES = [
    'L4_DST_PORT',                  # Cambiar el puerto destruye el ataque (ej. 80 -> 83)
    'PROTOCOL',                     # protocolo de red (TCP=6, UDP=17)
    'L7_PROTO',                     # detectado por DPI L7
    'TCP_FLAGS',                    # Flags combinados
    'CLIENT_TCP_FLAGS',             # Flags del atacante (congelados para mantener semántica ej. SYN)
    'IS_HTTP',                      # Derivada de L7
    'IS_HTTPS',                     # Derivada de L7
    'IS_BLIND_SQLI_CANDIDATE',      # Heurística del pipeline
    'IS_PROBE',                     # Heurística del pipeline
    'IS_RECON_HTTP',                # Heurística del pipeline
    'FLOW_RANK_IN_IP',              # Estado temporal asignado por el pipeline
    
    # BUF_* — El historial pasado de la IP no se puede reescribir desde el presente
    'BUF_UNIQUE_DST_PORTS',
    'BUF_UNIQUE_DST_IPS',
    'BUF_FLOW_COUNT',
    'BUF_NO_RESP_RATIO',
    'BUF_SCAN_RATE',
    'BUF_SMALL_PKT_RATIO',
    'BUF_PORT_STD',
    'BUF_PORT_RANGE',
    'BUF_IS_SCANNER',
    'BUF_HTTP_RATIO',
    'BUF_HTTP_BYTES_AVG',
    'BUF_HTTP_SMALL_RATIO',
    'BUF_BURST_PORTS',
    'BUF_SYN_ACK_RST_RATIO',
    'BUF_RECON_SCORE',
]


# ===========================================================================
# CONSTRAINTS FÍSICOS ABSOLUTOS (Clipping en Espacio Real)
# ===========================================================================

# Límites duros del protocolo/hardware en espacio ORIGINAL (pre-escalado).
# Solo limitamos las bases. Las variables derivadas (ratios) se auto-regulan
# gracias al Grafo Causal (apply_causal_graph).
# los límites específicos son empíricos de máximos observados en train 
PHYSICAL_BOUNDS = {
    'IN_BYTES'                   : (40.0,   np.inf),        # Headers mínimos de IPv4 + TCP (20+20 bytes)
    'IN_PKTS'                    : (1.0,    140_225.0),     # Al menos 1 paquete para existir
    'FLOW_DURATION_MILLISECONDS' : (0.0,    120_830.0),     # Max empírico de train (2 min ventana)
    'DURATION_IN'                : (0.0,    120_830.0),  
    'MIN_TTL'                    : (0.0,    255.0),      
    'LONGEST_FLOW_PKT'           : (28.0,   1500.0),        # MTU Estándar aproximado
    'SHORTEST_FLOW_PKT'          : (28.0,   1500.0),
    'MIN_IP_PKT_LEN'             : (0.0,    1500.0),        # Puede ser 0 si no hay payload
    'MAX_IP_PKT_LEN'             : (28.0,   1500.0),        # siempre al menos header IP
    'TCP_WIN_MAX_IN'             : (0.0,    65535.0),       # Límite ventana TCP estándar (16 bits)
    'SRC_TO_DST_IAT_MIN'         : (0.0,    59_995.0), 
    'SRC_TO_DST_IAT_MAX'         : (0.0,    60_636.0),
    'SRC_TO_DST_IAT_AVG'         : (0.0,    59_996.0),
    'SRC_TO_DST_IAT_STDDEV'      : (0.0,    29_325.0),
    'SRC_TO_DST_SECOND_BYTES'    : (0.0,    2_278.0),
    'SRC_TO_DST_AVG_THROUGHPUT'  : (6.0,    14_654_532.0),  # Min=6 (al menos 1 byte en >0s)
    'NUM_PKTS_UP_TO_128_BYTES'   : (0.0,    140_225.0),     # Acotado por IN_PKTS total
    'NUM_PKTS_128_TO_256_BYTES'  : (0.0,    174.0),
    'NUM_PKTS_256_TO_512_BYTES'  : (0.0,    603.0),
    'NUM_PKTS_512_TO_1024_BYTES' : (0.0,    2_047.0),
    'NUM_PKTS_1024_TO_1514_BYTES': (0.0,    6_155.0),
}


# ===========================================================================
# CLASE PRINCIPAL OOP
# ===========================================================================

@dataclass
class DomainConstraints:
    """
    Encapsula las restricciones de dominio del dataset BigFlow-NIDS.
    Se inyecta en la clase BaseAttack para garantizar la viabilidad del ataque.

    Proporciona:
      - Máscaras booleanas Forward/Backward/Immutable/Derived
      - Inversión de escalado para aplicar el saneamiento físico
      - Grafo causal vectorizado para corrección de variables derivadas

    Parámetros
    ----------
    feature_names  : array de nombres de features (66,)
    scaler_global  : QuantileTransformer para features non-BUF
    scaler_benign  : QuantileTransformer para features BUF_* calibrado en benignos
    buf_indices    : índices de features BUF_* en el array de features
    other_indices  : índices de features non-BUF
    X_train_raw    : array (n, 66) en espacio original para referencias estadísticas
    """
    feature_names  : np.ndarray
    scaler_global  : object
    scaler_benign  : object
    buf_indices    : np.ndarray
    other_indices  : np.ndarray
    X_train_raw    : np.ndarray

    # Máscaras booleanas (Calculadas en __post_init__)
    forward_mask   : np.ndarray = field(init=False)
    backward_mask  : np.ndarray = field(init=False)
    immutable_mask : np.ndarray = field(init=False)
    derived_mask   : np.ndarray = field(init=False)
    
    # La máscara definitiva para los gradientes (Solo se ataca el Forward)
    perturbable_mask: np.ndarray = field(init=False) 

    def __post_init__(self):
        n = len(self.feature_names)

        self.forward_mask   = self._build_mask(FORWARD_FEATURES,   n)
        self.backward_mask  = self._build_mask(BACKWARD_FEATURES,  n)
        self.immutable_mask = self._build_mask(IMMUTABLE_FEATURES, n)
        self.derived_mask   = self._build_mask(DERIVED_FEATURES,   n)

        # Para el PGD/SHAP, solo se consideran "optimizables" las Forward
        self.perturbable_mask = self.forward_mask.copy()

        self._validate_coverage()

    # ------------------------------------------------------------------
    # MOTOR FÍSICO Y CAUSAL (El "Saneamiento Semántico")
    # ------------------------------------------------------------------
    def apply_causal_graph(self, X_phys: np.ndarray) -> np.ndarray:
        """
        Saneamiento Semántico Vectorizado.
        Recalcula las características derivadas a partir de las bases mutadas.
        Se ejecuta en ESPACIO REAL (físico) antes de volver a escalar.
        
        Parámetros
        ----------
        X_phys : np.ndarray de forma (n_samples, n_features) en magnitud real.
        
        Retorna
        -------
        X_phys corregido causalmente.
        """
        # Helper interno para operaciones seguras (evitar inf/nan por /0)
        def safe_div(num, den):
            return num / (den + 1e-8)

        # Obtener índices rápidos
        i_in_bytes  = self._feat_idx("IN_BYTES")
        i_in_pkts   = self._feat_idx("IN_PKTS")
        i_out_bytes = self._feat_idx("OUT_BYTES")
        i_out_pkts  = self._feat_idx("OUT_PKTS")
        i_dur       = self._feat_idx("FLOW_DURATION_MILLISECONDS")
        i_win_in    = self._feat_idx("TCP_WIN_MAX_IN")
        i_win_out   = self._feat_idx("TCP_WIN_MAX_OUT")

        # 1. SRC_BYTES_PER_PKT
        idx = self._feat_idx("SRC_BYTES_PER_PKT")
        if idx is not None:
            X_phys[:, idx] = safe_div(X_phys[:, i_in_bytes], X_phys[:, i_in_pkts])

        # 2. PKT_RATIO
        idx = self._feat_idx("PKT_RATIO")
        if idx is not None:
            X_phys[:, idx] = safe_div(X_phys[:, i_in_pkts], X_phys[:, i_out_pkts])

        # 3. RESPONSE_RATIO
        idx = self._feat_idx("RESPONSE_RATIO")
        if idx is not None:
            X_phys[:, idx] = safe_div(X_phys[:, i_in_bytes], X_phys[:, i_out_bytes])

        # 4. TOTAL_BYTES
        idx = self._feat_idx("TOTAL_BYTES")
        if idx is not None:
            X_phys[:, idx] = X_phys[:, i_in_bytes] + X_phys[:, i_out_bytes]

        # 5. DURATION_PER_PKT
        idx = self._feat_idx("DURATION_PER_PKT")
        if idx is not None:
            total_pkts = X_phys[:, i_in_pkts] + X_phys[:, i_out_pkts]
            X_phys[:, idx] = safe_div(X_phys[:, i_dur], total_pkts)

        # 6. TCP_WIN_RATIO
        idx = self._feat_idx("TCP_WIN_RATIO")
        if idx is not None:
            X_phys[:, idx] = safe_div(X_phys[:, i_win_in], X_phys[:, i_win_out])

        # 7. IS_UNIDIRECTIONAL (Hard Rule)
        idx = self._feat_idx("IS_UNIDIRECTIONAL")
        if idx is not None:
            X_phys[:, idx] = np.where(X_phys[:, i_out_pkts] == 0, 1.0, 0.0)

        return X_phys

    # ------------------------------------------------------------------
    # TRANSFORMACIONES ESPACIALES
    # ------------------------------------------------------------------
    def to_physical_space(self, X_scaled: np.ndarray) -> np.ndarray:
        """Invierte el QuantileTransformer al espacio original físico."""
        X_raw = np.empty_like(X_scaled, dtype=np.float64)
        X_raw[:, self.other_indices] = self.scaler_global.inverse_transform(X_scaled[:, self.other_indices])
        X_raw[:, self.buf_indices] = self.scaler_benign.inverse_transform(X_scaled[:, self.buf_indices])
        return X_raw

    def to_scaled_space(self, X_raw: np.ndarray) -> np.ndarray:
        """Escala un array físico al espacio de tensores del modelo."""
        X_sc = np.empty(X_raw.shape, dtype=np.float32)
        X_sc[:, self.other_indices] = self.scaler_global.transform(X_raw[:, self.other_indices]).astype(np.float32)
        X_sc[:, self.buf_indices] = self.scaler_benign.transform(X_raw[:, self.buf_indices]).astype(np.float32)
        return X_sc

    def get_benign_centroids(self) -> np.ndarray:
        """
        Calcula el valor medio de cada feature para el tráfico benigno.
        Retorna un array (66,) en ESPACIO ESCALADO listo para el modelo.
        """
        # Extraemos las muestras benignas del X_train_raw (donde label == 0)
        benign_mean_phys = np.mean(self.X_train_raw, axis=0).reshape(1, -1)
        
        # Convertimos la media física al espacio escalado que entiende la red
        return self.to_scaled_space(benign_mean_phys).flatten()

    # ------------------------------------------------------------------
    # FACTORY — Carga desde artefactos
    # ------------------------------------------------------------------
    @classmethod
    def from_artifacts(
        cls,
        models_path : str = 'outputs/models',
        data_path   : str = 'data/processed',
    ) -> 'DomainConstraints':
        """
        Carga todos los artefactos necesarios y construye DomainConstraints.

        Uso:
            dc = DomainConstraints.from_artifacts()
            dc.summary()
        """
        mp = Path(models_path)

        feature_names  = np.load(mp / 'feature_names.npy',  allow_pickle=True)
        buf_indices    = np.load(mp / 'buf_indices.npy')
        other_indices  = np.load(mp / 'other_indices.npy')
        scaler_global  = joblib.load(mp / 'quantile_scaler_global.pkl')
        scaler_benign  = joblib.load(mp / 'quantile_scaler_benign.pkl')

        # X_train_raw se carga si existe, para referencias estadísticas futuras
        raw_path = mp / 'X_train_raw_reconstructed.npy'
        if raw_path.exists():
            X_train_raw = np.load(raw_path)
        else:
            X_train_raw = np.zeros((1, len(feature_names)))
            
        return cls(
            feature_names = feature_names,
            scaler_global = scaler_global,
            scaler_benign = scaler_benign,
            buf_indices   = buf_indices,
            other_indices = other_indices,
            X_train_raw   = X_train_raw,
        )

    # ------------------------------------------------------------------
    # UTILIDADES PRIVADAS Y LOGGING
    # ------------------------------------------------------------------
    def _build_mask(self, feature_list: list[str], n: int) -> np.ndarray:
        mask = np.zeros(n, dtype=bool)
        for feat in feature_list:
            idx = self._feat_idx(feat)
            if idx is not None:
                mask[idx] = True
            else:
                print(f"   [WARN] Feature '{feat}' no encontrada en feature_names")
        return mask

    def _feat_idx(self, name: str) -> Optional[int]:
        matches = np.where(self.feature_names == name)[0]
        return int(matches[0]) if len(matches) > 0 else None

    def _validate_coverage(self) -> None:
        """Verifica que todas las features estén clasificadas correctamente."""
        all_classified = (
            self.forward_mask | self.backward_mask | 
            self.immutable_mask | self.derived_mask
        )
        unclassified = np.where(~all_classified)[0]
        if len(unclassified) > 0:
            names = [self.feature_names[i] for i in unclassified]
            print(f"   [WARN] Features sin clasificar: {names}")
            print(f"          Se congelan (IMMUTABLE) por seguridad física.")
            self.immutable_mask[unclassified] = True

    def summary(self) -> None:
        """Imprime un resumen estructural del dominio físico del dataset."""
        print("\n" + "=" * 60)
        print("DOMAIN CONSTRAINTS — Motor Físico y Causal (BigFlow-NIDS)")
        print("=" * 60)
        print(f"  Features totales   : {len(self.feature_names)}")
        print(f"  Forward (Attack)   : {self.forward_mask.sum()} (Gradientes activos)")
        print(f"  Backward (Server)  : {self.backward_mask.sum()} (Congeladas)")
        print(f"  Derived (Causal)   : {self.derived_mask.sum()} (Auto-calculadas)")
        print(f"  Immutable (IDS/L4) : {self.immutable_mask.sum()} (Congeladas)")

        print("\n  Features FORWARD (Perturbables por PGD/SHAP):")
        for i, name in enumerate(self.feature_names):
            if self.forward_mask[i]:
                print(f"    [{i:>2}] {name}")

# ===========================================================================
# SCRIPT DE VERIFICACIÓN
# ===========================================================================
if __name__ == "__main__":
    dc = DomainConstraints.from_artifacts()
    dc.summary()
    print("\n[✓] DomainConstraints híbrido listo para BaseAttack")