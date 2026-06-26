"""
prepare_zerodays.py
====================================================================
Script local para extraer la clase excluida "Worms" (Zero-Day), 
aplicar ingeniería de características, pasar por el buffer de comportamiento,
escalar con los artefactos de la Fase 1 y exportar X_worms_sc.npy listo para inferencia.
"""

import os
import glob
import numpy as np
import polars as pl
import joblib

from src.config import Config
from src.ip_behavior_buffer import IPBehaviorBuffer

def extraer_zero_days():
    print("\n" + "=" * 70)
    print("ETL LOCAL: GENERANDO DATASET ZERO-DAY (WORMS)")
    print("=" * 70)

    raw_files = glob.glob(os.path.join(Config.DATA_RAW_PATH, "*.parquet"))
    if not raw_files:
        raise FileNotFoundError(f"[!] No se encontraron parquets en {Config.DATA_RAW_PATH}")

    frames = []
    print("[-] 1. Escaneando parquets crudos en busca de Gusanos...")
    for f in raw_files:
        df_worms = (
            pl.scan_parquet(f)
            .filter(pl.col("Attack") == "Worms")
            .collect()
        )
        if df_worms.height > 0:
            frames.append(df_worms)

    if not frames:
        raise ValueError("[!] No hay Gusanos en los datos crudos. Revisa el dataset base.")

    df_raw = pl.concat(frames)
    print(f"  [✓] Extraídas {df_raw.height:,} muestras de Worms.")

    print("\n[-] 2. Aplicando Ingeniería de Características (Feature Engineering)...")
    df_fe = df_raw.with_columns([
        pl.col(c).cast(pl.Float64) for c in Config.COLS_FORCE_FLOAT64 if c in df_raw.columns
    ]).with_columns([
        pl.when(pl.col('OUT_PKTS') == 0).then(1.0).otherwise(0.0).alias('IS_UNIDIRECTIONAL'),
        (pl.col('IN_BYTES') / (pl.col('IN_PKTS') + 1e-8)).alias('SRC_BYTES_PER_PKT'),
        (pl.col('IN_PKTS') / (pl.col('OUT_PKTS') + 1e-8)).alias('PKT_RATIO'),
        pl.when(pl.col('L7_PROTO') == 7.0).then(1.0).otherwise(0.0).alias('IS_HTTP'),
        pl.when(pl.col('L7_PROTO') == 91.0).then(1.0).otherwise(0.0).alias('IS_HTTPS'),
        (pl.col('IN_BYTES') / (pl.col('OUT_BYTES') + 1e-8)).alias('RESPONSE_RATIO'),
        (pl.col('TCP_WIN_MAX_IN') / (pl.col('TCP_WIN_MAX_OUT') + 1e-8)).alias('TCP_WIN_RATIO'),
        (pl.col('IN_BYTES') + pl.col('OUT_BYTES')).alias('TOTAL_BYTES'),
        (pl.col('FLOW_DURATION_MILLISECONDS') / (pl.col('IN_PKTS') + pl.col('OUT_PKTS') + 1e-8)).alias('DURATION_PER_PKT'),
        pl.when((pl.col('L7_PROTO').is_in([7.0, 91.0])) & (pl.col('FLOW_DURATION_MILLISECONDS') > 5000) & (pl.col('IN_BYTES') < 500)).then(1.0).otherwise(0.0).alias('IS_BLIND_SQLI_CANDIDATE'),
        pl.when((pl.col('OUT_PKTS') == 0) & (pl.col('IN_BYTES') < 100)).then(1.0).otherwise(0.0).alias('IS_PROBE'),
        pl.when((pl.col('L7_PROTO').is_in([7.0, 91.0])) & (pl.col('FLOW_DURATION_MILLISECONDS') > 5000) & (pl.col('IN_BYTES') < 200)).then(1.0).otherwise(0.0).alias('IS_RECON_HTTP'),
    ])

    # Rank del flujo en la IP
    df_fe = df_fe.with_columns(
        pl.col("FLOW_START_MILLISECONDS").rank(method="ordinal").over("IPV4_SRC_ADDR").alias("_rank_raw"),
        pl.col("FLOW_START_MILLISECONDS").count().over("IPV4_SRC_ADDR").alias("_count_ip"),
    ).with_columns(
        pl.when(pl.col("_count_ip") > 1).then((pl.col("_rank_raw") - 1) / (pl.col("_count_ip") - 1)).otherwise(0.0).cast(pl.Float32).alias("FLOW_RANK_IN_IP")
    ).drop(["_rank_raw", "_count_ip"])

    print("\n[-] 3. Pasando flujos por el IP Behavior Buffer...")
    # Creamos un buffer fresco solo para estos datos
    buffer_zd = IPBehaviorBuffer(window_seconds=120, max_ips=10000)
    df_fe = buffer_zd.update_and_extract(df_fe)

    print("\n[-] 4. Alineando y limpiando columnas...")
    cols_to_drop = [c for c in Config.COLS_TO_DROP if c in df_fe.columns]
    df_fe = df_fe.drop(cols_to_drop)

    # Cargar el orden exacto de las columnas que espera el modelo
    feat_path = os.path.join(Config.MODELS_PATH, "feature_names.npy")
    if not os.path.exists(feat_path):
        raise FileNotFoundError(f"[!] Falta {feat_path}. Necesitas los artefactos de la Fase 1 en local.")
    
    feature_names = np.load(feat_path)
    
    # Extraer Numpy array puro
    X_worms_raw = df_fe.select(feature_names).to_numpy().astype(np.float64)

    print("\n[-] 5. Limpieza (Inf/NaN) y Escalado dual (QuantileTransformers)...")
    train_medians = np.load(os.path.join(Config.MODELS_PATH, "train_medians.npy"))
    buf_indices = np.load(os.path.join(Config.MODELS_PATH, "buf_indices.npy"))
    other_indices = np.load(os.path.join(Config.MODELS_PATH, "other_indices.npy"))

    scaler_global = joblib.load(os.path.join(Config.MODELS_PATH, "quantile_scaler_global.pkl"))
    scaler_benign = joblib.load(os.path.join(Config.MODELS_PATH, "quantile_scaler_benign.pkl"))

    # Aplicar medianas de train
    X_worms_raw = np.where(np.isinf(X_worms_raw), np.nan, X_worms_raw)
    mask = np.isnan(X_worms_raw)
    X_worms_raw[mask] = np.take(train_medians, np.where(mask)[1])

    # Escalar
    X_worms_sc = np.empty_like(X_worms_raw, dtype=np.float32)
    X_worms_sc[:, other_indices] = scaler_global.transform(X_worms_raw[:, other_indices])
    X_worms_sc[:, buf_indices]   = scaler_benign.transform(X_worms_raw[:, buf_indices])

    print("\n[-] 6. Exportando matriz final...")
    out_path = os.path.join(Config.DATA_PROCESSED_PATH, "..", "ataques_zero_days", "X_worms_sc.npy")
    np.save(out_path, X_worms_sc)

    print(f"  [✓] ¡Listo! Archivo guardado en: data\\processed")
    print(f"  [✓] Forma (Shape) final: {X_worms_sc.shape}")
    print("\n>>> Ya puedes subir 'X_worms_sc.npy' a tu entorno de Colab. <<<")

if __name__ == "__main__":
    extraer_zero_days()