import os
import torch

class Config:
    """
    Centro de configuración global del proyecto TFG.
    Toda constante que cambie entre experimentos vive aquí.
    Nunca hardcodear valores en los scripts.
    """

    # MODEL_NAME = "LightGBM" o "TabularResNet"
    # DATASET = "BigFlow-NIDS" o "NF-UQ-NIDS-V2"
    # EXPERIMENT_PHASE = 1, 2 o 3 (ver README para detalles de cada fase)
    # EXPERIMENT_SAVE = "resultados_2_buffer" o "resultados_3_surrogate" (carpeta dentro de processed)
    MODEL_NAME = "TabularResNet"
    
    DATASET = "BigFlow-NIDS"
    EXPERIMENT_PHASE = 1
    EXPERIMENT_SAVE = "resultados_2_buffer"

    _PHASE_TRAIN_SIZES = {
        1: 1_200_000,
        2: 100_000,
        #3: 2_000_000,
    }

    # ------------------------------------------------------------------
    # RUTAS
    # ------------------------------------------------------------------
    DATA_RAW_PATH       = os.path.join("data", "raw", DATASET)
    DATA_PROCESSED_PATH = os.path.join("data", "processed", EXPERIMENT_SAVE)
    MODELS_PATH         = os.path.join("outputs", "models")
    LOGS_PATH           = os.path.join("outputs", "logs")

    # ------------------------------------------------------------------
    # RUTAS SURROGATE
    # ------------------------------------------------------------------
    DATA_RAW_PATH_SURROGATE       = os.path.join("data", "raw", "NF-UQ-NIDS-V2")
    DATA_PROCESSED_PATH_SURROGATE = os.path.join("data", "processed", "resultados_3_surrogate")
    MODELS_PATH_SURROGATE         = os.path.join("outputs", "models", "surrogate")
    LOGS_PATH_SURROGATE           = os.path.join("outputs", "logs", "surrogate")

    # ------------------------------------------------------------------
    # HARDWARE Y REPRODUCIBILIDAD
    # ------------------------------------------------------------------
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SEED   = 42

    # ------------------------------------------------------------------
    # SPLITS CRONOLÓGICOS (porcentaje de archivos parquet)
    # ------------------------------------------------------------------
    TRAIN_SIZE = 0.70
    VAL_SIZE   = 0.15
    TEST_SIZE  = 0.15

    
    #class ResNet:
    # ------------------------------------------------------------------
    # HIPERPARÁMETROS DE ARQUITECTURA TABULAR RESNET
    # ------------------------------------------------------------------
    HIDDEN_DIM  = 256
    EMBED_DIM  = 128      # ratio 2:1 para el VAE, proporcional a hidden_dim (cambiar si hidden_dim y n_blocks cambia mucho)
    N_BLOCKS    = 4
    DROPOUT     = 0.10
    INNER_DROPOUT = 0.0   # para proteger contra ataques adversariales
    MIXUP_ALPHA = 0.2     # data augmentation para regularizar y mejorar generalización

    # ------------------------------------------------------------------
    # HIPERPARÁMETROS DE ENTRENAMIENTO
    # ------------------------------------------------------------------
    BATCH_SIZE    = 8192    # En vez de 256 que estaba antes (usaba solo 0.2GB RAM) 4096 o 8192 es un buen punto 
                            # de partida para aprovechar la GPU de Colab de 15GB VRAM sin OOM
                            # AUNQUE HABRÁ QUE AUMENTAR ÉPOCAS PARA COMPENSAR EL AUMENTO DE BATCH SIZE 
    EPOCHS        = 50
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY  = 1e-4
    PATIENCE      = 10

    # Weights manuales calibrados para tus problemas específicos
    # Basados en los F1 actuales — más bajo el F1, más alto el weight
    MANUAL_WEIGHTS = [
        0.5,   # 0 Benign        F1=0.99 → penalizar poco
        0.7,   # 1 DoS           F1=0.94 → casi perfecto
        0.7,   # 2 DDoS          F1=0.95 → casi perfecto
        3.0,   # 3 Web/Injection F1=0.69 → necesita empuje
        1.0,   # 4 Brute Force   F1=0.85 → aceptable
        2.0,   # 5 Recon/Scan    F1=0.51 → máxima prioridad
        1.5,   # 6 Malware       F1=0.81 → leve empuje
        2.5,   # 7 Exploits      F1=0.86 → proteger con muestras bajas
    ]
    

    class LightGBM:
        # ------------------------------------------------------------------
        # HIPERPARÁMETROS PARA BASELINE LIGHTGBM 
        # ------------------------------------------------------------------
        OPTUNA_TRIALS = 30
        OPTUNA_TIMEOUT = 3600
        PI_PROD = 0.05



    # ------------------------------------------------------------------
    # DISTRIBUCIONES DE CADA SPLIT
    # ------------------------------------------------------------------
    TRAIN_PCT_BENIGN = 0.60   # 60/40 — maximiza aprendizaje de ataques
    VAL_PCT_BENIGN   = 0.75   # 75/25 — calibración de umbrales
    TEST_PCT_BENIGN  = 0.95   # 95/5  — distribución real de producción


    # ------------------------------------------------------------------
    # CONSTANTES DE DISEÑO IP BEHAVIOR BUFFER
    # ------------------------------------------------------------------
    _MAX_PORT_HISTORY  = 100    # últimos N puertos para std/range (memoria acotada)
    # cualquier usuario en la actualidad con más de 150 puertos únicos y ratio > 70% sin respuesta, es scanner
    # porque 50 puertos, con tener chrome abierto, con spotify y varias apps, ya lo supera fácil
    _SCANNER_PORT_THR  = 150    # puertos únicos mínimos para flag IS_SCANNER
    _SCANNER_RESP_THR  = 0.70   # ratio mínimo sin respuesta para IS_SCANNER
    _SMALL_PKT_BYTES   = 100    # umbral bytes para considerar paquete "probe"
    _BURST_WINDOW = 20          # últimos 20 flujos para detectar bursts (puertos consecutivos en poco tiempo)

    # ------------------------------------------------------------------
    # MAPEO ETIQUETAS -> MACRO-CLASES
    # ------------------------------------------------------------------
    NUM_CLASSES = 8

    ATTACK_MAPPING = {
        'Benign': 0,
        # Grupo 1 — DoS
        'DoS': 1, 'DoS_attacks-SlowHTTPTest': 1, 'DoS_attacks-Hulk': 1,
        'DoS_attacks-GoldenEye': 1, 'DoS_attacks-Slowloris': 1,
        # Grupo 2 — DDoS
        'DDoS': 2, 'DDOS_attack-HOIC': 2, 'DDoS_attacks-LOIC-HTTP': 2,
        'DDOS_attack-LOIC-UDP': 2,
        # Grupo 3 — Web Attacks / Injection
        'xss': 3, 'injection': 3, 'SQL_Injection': 3,
        # Grupo 4 — Brute Force
        'password': 4, 'FTP-BruteForce': 4, 'SSH-Bruteforce': 4,
        'Brute_Force_-XSS': 4, 'Brute_Force_-Web': 4,
        # Grupo 5 — Recon + Scanning 
        # indistinguibles con estas features, se agrupan para no confundir al modelo
        'Reconnaissance': 5, 'scanning': 5,
        # Grupo 6 — Malware / Intrusión avanzada
        'Bot': 6, 'Backdoor': 6, 'ransomware': 6, 'Shellcode': 6,
        'mitm': 6, 'Infilteration': 6,
        # Grupo 7 — Exploits / Fuzzing
        'Exploits': 7, 'Fuzzers': 7, 'Generic': 7, 'Analysis': 7, 'Theft': 7,
    }

    CLASS_NAMES = [
        'Benign',         # 0
        'DoS',            # 1
        'DDoS',           # 2
        'Web/Injection',  # 3
        'Brute Force',    # 4
        'Recon',          # 5
        'Malware',        # 6
        'Exploits',       # 7
    ]

    # ------------------------------------------------------------------
    # COLUMNAS A ELIMINAR
    # ------------------------------------------------------------------
    COLS_TO_DROP = [
        'IPV4_SRC_ADDR', 'IPV4_DST_ADDR', 'L4_SRC_PORT',
        'FLOW_START_MILLISECONDS', 'FLOW_END_MILLISECONDS',
        'Label', 'DNS_QUERY_ID',
        # nuevas — features de protocolo específico estructuralmente sparse
        # después del análisis de los npy en notebook 01_eda_visualizacion.ipynb
        'FTP_COMMAND_RET_CODE',
        'ICMP_TYPE',
        'ICMP_IPV4_TYPE',
        'DNS_TTL_ANSWER',
        'DNS_QUERY_TYPE',
        'RETRANSMITTED_OUT_BYTES',
        'RETRANSMITTED_OUT_PKTS',
        'RETRANSMITTED_IN_BYTES',
        'RETRANSMITTED_IN_PKTS',
    ]

    # Schema inconsistente entre parquets → cast obligatorio a Float64
    COLS_FORCE_FLOAT64 = [
        'SRC_TO_DST_SECOND_BYTES',
        'DST_TO_SRC_SECOND_BYTES',
    ]



    @classmethod
    def n_train(cls) -> int:
        return cls._PHASE_TRAIN_SIZES[cls.EXPERIMENT_PHASE]

    @classmethod
    def n_val(cls) -> int:
        return max(200_000, cls.n_train() // 6)

    @classmethod
    def n_test(cls) -> int:
        return cls.n_val()
    

    # ------------------------------------------------------------------
    # TAMAÑOS DRY RUN (Para pruebas rápidas de pipeline)
    # ------------------------------------------------------------------
    N_TRAIN_DRY = 10_000
    N_VAL_DRY   =  2_000
    N_TEST_DRY  =  2_000
    N_FILES_DRY =      5