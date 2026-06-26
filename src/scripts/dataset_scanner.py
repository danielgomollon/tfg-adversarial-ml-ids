import polars as pl
import os
from src.config import Config

def audit_dataset_features():
    print("[-] Iniciando Auditoría del Dataset...")
    path_pattern = os.path.join(Config.DATA_RAW_PATH, "*.parquet")

    # Schema heterogéneo entre parquets: usamos ignore_errors + schema inference
    # low_memory=False para que intente unificar tipos antes de fallar
    lf = pl.scan_parquet(path_pattern, allow_missing_columns=True)
    
    schema = lf.collect_schema()
    features = list(schema.names())

    print("\n" + "="*60)
    print(f"AUDITORÍA DE FEATURES ({len(schema)} columnas detectadas)")
    print("="*60)

    for i in range(0, len(features), 5):
        print(" | ".join(features[i:i+5]))

    # vista previa amplia y con todos los tipos
    print("\n" + "="*60)
    print("VISTA PREVIA DE LOS DATOS (3 Filas, todas las columnas)")
    print("="*60)
    first_file = sorted(os.listdir(Config.DATA_RAW_PATH))[0]  # sorted para reproducibilidad
    df_sample = pl.read_parquet(
        os.path.join(Config.DATA_RAW_PATH, first_file), n_rows=10
    )
    with pl.Config(tbl_cols=-1, tbl_width_chars=250):
        print(df_sample)

    # distribución de etiquetas en todo el dataset 
    print("\n" + "="*60)
    print("DISTRIBUCIÓN DE ETIQUETAS (dataset completo, lazy scan)")
    print("="*60)

    LABEL_COL = "Attack"

    if LABEL_COL and LABEL_COL in features:
        print("\n" + "="*60)
        print(f"DISTRIBUCIÓN DE ETIQUETAS — columna: '{LABEL_COL}'")
        print("="*60)

        label_dist = (
            lf.select(LABEL_COL)
            .group_by(LABEL_COL)
            .agg(pl.len().alias("count"))
            .sort("count", descending=True)
            .collect()
        )

        total = label_dist["count"].sum()
        print(f"{'Etiqueta':<45} {'Count':>10} {'%':>8}")
        print("-" * 65)
        for row in label_dist.iter_rows():
            label, count = row
            pct = count / total * 100
            flag = "  <-- MINORITARIA" if pct < 0.5 else ""
            print(f"{str(label):<45} {count:>10,} {pct:>7.3f}%{flag}")

        total_attacks = sum(r[1] for r in label_dist.iter_rows()
                           if "benign" not in str(r[0]).lower())
        print(f"\nFlujos benignos:  {total - total_attacks:,}  "
              f"({(total - total_attacks)/total*100:.2f}%)")
        print(f"Flujos de ataque: {total_attacks:,}  "
              f"({total_attacks/total*100:.2f}%)")
        print(f"Clases detectadas: {len(label_dist)}")
    else:
        print("\nRAELLENA LABEL_COL con la columna correcta y vuelve a ejecutar.")

    # --- COLUMNAS TEMPORALES ---
    print("\n" + "="*60)
    print("REVISIÓN DE COLUMNAS TEMPORALES / LEAKAGE POTENCIAL")
    print("="*60)
    time_suspects = [c for c in features if any(
        kw in c.upper() for kw in ["TIME", "START", "END", "DURATION", "STAMP", "EPOCH", "MILLI"]
    )]
    for col in time_suspects:
        print(f"  {col:<45} tipo: {schema[col]}")
    print("\nRECOMENDACION:")
    print("  - START / END absolutos    -> ELIMINAR (leakage temporal)")
    print("  - DURATION / IAT / STATS   -> CONSERVAR (disponibles post-flow)")

    # --- SCHEMA HETEROGÉNEO: columnas con conflicto de tipos entre parquets ---
    print("\n" + "="*60)
    print("CONFLICTOS DE SCHEMA ENTRE PARQUETS")
    print("="*60)
    print("Escaneando tipos por archivo (puede tardar 1-2 min)...")

    type_map = {}  # col -> set de tipos encontrados
    files = sorted(os.listdir(Config.DATA_RAW_PATH))
    files = [f for f in files if f.endswith(".parquet")]

    for fname in files:
        fpath = os.path.join(Config.DATA_RAW_PATH, fname)
        s = pl.read_parquet(fpath, n_rows=1).schema
        for col, dtype in s.items():
            if col not in type_map:
                type_map[col] = set()
            type_map[col].add(str(dtype))

    conflicting = {col: types for col, types in type_map.items() if len(types) > 1}
    if conflicting:
        print(f"\nColumnas con tipos inconsistentes entre archivos ({len(conflicting)}):")
        for col, types in conflicting.items():
            print(f"  {col:<45} tipos encontrados: {types}")
        print("\nESTAS COLUMNAS necesitan cast explícito en data_pipeline.py")
        print("Recomendacion general: castear todo a Float32 salvo etiqueta.")
    else:
        print("Sin conflictos de schema detectados.")

    # --- NULOS: archivo por archivo para evitar el SchemaError ---
    print("\n" + "="*60)
    print("CALIDAD DE DATOS: NULOS (muestra primer parquet)")
    print("="*60)
    df_first = pl.read_parquet(
        os.path.join(Config.DATA_RAW_PATH, files[0])
    )
    null_summary = df_first.null_count().transpose(
        include_header=True, column_names=["null_count"]
    ).filter(pl.col("null_count") > 0)

    if len(null_summary) == 0:
        print("Sin valores nulos en el primer parquet.")
    else:
        print(null_summary)

    # Columnas constantes
    print("\nColumnas constantes en primer parquet:")
    constant_cols = [c for c in df_first.columns if df_first[c].n_unique() == 1]
    print(constant_cols if constant_cols else "Ninguna.")


if __name__ == "__main__":
    audit_dataset_features()