import polars as pl
import os

# Ajusta esta ruta a donde tengas los parquets crudos de BigFlow-NIDS
raw_path = "data/raw/BigFlow-NIDS" 

print("[*] Buscando flujos Recon para autopsia de features...")

# Buscar el primer parquet que tenga Recon
files = [f for f in os.listdir(raw_path) if f.endswith('.parquet')]

for fname in files:
    df = pl.read_parquet(os.path.join(raw_path, fname))
    
    # Filtramos por palabras clave típicas de escaneo/reconocimiento
    recon_flows = df.filter(pl.col("Attack").str.contains("(?i)(scan|recon|nmap|probe)"))
    
    if recon_flows.height > 15:
        print(f"[*] ¡Recon encontrado en {fname}! Procediendo a la extracción...\n")
        
        # Filtramos Benignos que probablemente estén causando Falsos Positivos (0 paquetes de respuesta)
        benign_unidirectional = df.filter(
            (pl.col("Attack") == "Benign") & 
            (pl.col("OUT_PKTS") == 0)
        )
        
        cols_to_inspect = [
            "L4_DST_PORT", "PROTOCOL", "IN_PKTS", "OUT_PKTS", "IN_BYTES", 
            "OUT_BYTES", "TCP_FLAGS", "FLOW_DURATION_MILLISECONDS", "Attack"
        ]
        
        print("--- 15 Flujos RECON Reales ---")
        print(recon_flows.select(cols_to_inspect).head(15).to_pandas())
        
        print("\n--- 15 Flujos BENIGNOS Unidireccionales (Candidatos a Falso Positivo) ---")
        print(benign_unidirectional.select(cols_to_inspect).head(15).to_pandas())
        
        print("\n--- Estadísticas Comparativas ---")
        print("RECON Medias:")
        print(recon_flows.select(["IN_PKTS", "IN_BYTES", "FLOW_DURATION_MILLISECONDS"]).mean().to_pandas())
        print("\nBENIGN (Unidireccional) Medias:")
        print(benign_unidirectional.select(["IN_PKTS", "IN_BYTES", "FLOW_DURATION_MILLISECONDS"]).mean().to_pandas())
        break