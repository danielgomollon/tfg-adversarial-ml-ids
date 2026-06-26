import polars as pl

PARQUET_PATH = "data/raw/BigFlow-NIDS/merged_part_10.parquet"

def verify_new_features(parquet_path):
    print(f"[-] Cargando {parquet_path}...")
    df = pl.read_parquet(parquet_path)
    
    print("\n" + "="*50)
    print("TEST 1: OUT_BYTES en HTTP por clase de ataque")
    print("="*50)
    http_df = df.filter(pl.col("L7_PROTO") == 7.0)
    if http_df.height > 0:
        print(http_df.group_by("Attack").agg([
            pl.len().alias("N_Flujos_HTTP"),
            pl.col("OUT_BYTES").median().alias("Mediana_OUT_BYTES"),
            (pl.col("OUT_BYTES") < 400).mean().alias("Ratio_menor_400B")
        ]).sort("Ratio_menor_400B", descending=True))
    else:
        print("[!] No hay tráfico HTTP en este parquet.")

    print("\n" + "="*50)
    print("TEST 2: RST flag por clase de ataque")
    print("="*50)
    print(df.group_by("Attack").agg([
        pl.len().alias("N_Flujos"),
        ((pl.col("TCP_FLAGS").cast(pl.Int32) & 4) > 0).mean().alias("Ratio_RST"),
        ((pl.col("CLIENT_TCP_FLAGS").cast(pl.Int32) & 4) > 0).mean().alias("Ratio_RST_Client"),
        ((pl.col("SERVER_TCP_FLAGS").cast(pl.Int32) & 4) > 0).mean().alias("Ratio_RST_Server"),
    ]).sort("Ratio_RST", descending=True))

    # TEST 3: BUF_BURST_PORTS — verificar que Recon tiene puertos muy variados
    print("\n" + "="*50)
    print("TEST 3: Diversidad de puertos destino por clase")
    print("="*50)
    print(df.group_by("Attack").agg([
        pl.len().alias("N_Flujos"),
        pl.col("L4_DST_PORT").n_unique().alias("Puertos_Unicos"),
        pl.col("L4_DST_PORT").std().alias("Port_STD"),
    ]).sort("Puertos_Unicos", descending=True))

verify_new_features(PARQUET_PATH)