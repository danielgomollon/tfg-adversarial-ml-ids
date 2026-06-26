import os
import numpy as np
import joblib
import polars as pl
from sklearn.preprocessing import QuantileTransformer

from src.config import Config

ATTACK_MAPPING = {
    'Benign'        : 0,
    'DoS'           : 1,
    'DDoS'          : 2,
    'injection'     : 3,  # Web/Injection
    'xss'           : 3,  # Web/Injection
    'Brute Force'   : 4,  # nativo de NF-UQ-NIDS-v2
    'password'      : 4,  # Brute Force
    'scanning'      : 5,  # Recon
    'Reconnaissance': 5,  # Recon
    'Bot'           : 6,  # Malware
    'Backdoor'      : 6,  # Malware
    'ransomware'    : 6,  # Malware
    'Theft'         : 6,  # Malware
    'mitm'          : 6,  # Malware
    'Infilteration' : 6,  # Malware
    'Exploits'      : 7,
    'Fuzzers'       : 7,  # Exploits
    'Shellcode'     : 7,  # Exploits
    'Generic'       : 7,  # Exploits
    'Analysis'      : 5,  # Recon (análisis de red)
}

COLS_TO_DROP = [
    'IPV4_SRC_ADDR', 'IPV4_DST_ADDR', 'L4_SRC_PORT',
    'FLOW_START_MILLISECONDS', 'FLOW_END_MILLISECONDS',
    'Label', 'DNS_QUERY_ID', 'Dataset',  
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

class DataPipeline:
    """
    Pipeline ETL para BigFlow-NIDS.

    Responsabilidades:
      - Split temporal estricto por archivo parquet
      - Muestreo estratificado proporcional (benignos fijos + ataques
        proporcionales a su peso natural en el dataset)
      - Las clases raras del dataset (ransomware, mitm, Shellcode, Theft,
        Analysis, Brute_Force_-XSS, SQL_Injection) son tan escasas (<6K filas
        cada una sobre 66M totales) que el muestreo proporcional las incluye
        completas de forma natural. No se necesita lógica de mínimos.
      - Limpieza de inf/NaN con medianas de TRAIN propagadas a val/test
        para evitar data leakage implícito y reproducibilidad en producción
      - Normalización QuantileTransformer fiteada SOLO en benignos de train
        (el scaler aprende la distribución del tráfico normal; los ataques
        quedarán fuera de esa distribución, amplificando la señal de anomalía
        para el VAE + Mahalanobis en Fase 3)
      - Exportación de artefactos .npy listos para PyTorch y XGBoost

    Uso:
        dp = DataPipeline()
        dp.run(dry_run=True)   # verificación rápida (~segundos)
        dp.run(dry_run=False)  # run completo (5-10 min)
    """

    def __init__(self):
        self.raw_path       = Config.DATA_RAW_PATH_SURROGATE
        self.processed_path = Config.DATA_PROCESSED_PATH_SURROGATE
        self.models_path    = Config.MODELS_PATH_SURROGATE
        os.makedirs(self.processed_path, exist_ok=True)
        os.makedirs(self.models_path,    exist_ok=True)

    # ------------------------------------------------------------------
    # SPLIT TEMPORAL
    # ------------------------------------------------------------------
    def _get_ordered_files(self, dry_run): # lista 
        files = sorted(f for f in os.listdir(self.raw_path) if f.endswith(".parquet"))
        if not files:
            raise FileNotFoundError(f"No se encontraron parquets en {self.raw_path}")
        if dry_run:
            files = files[:Config.N_FILES_DRY]
            print(f"   [DRY RUN] {len(files)} parquets")
        else:
            print(f"   Parquets detectados: {len(files)}")
        return files

    def _split_files_temporal(self, files): # tupla
        n         = len(files)
        train_end = max(1, int(n * Config.TRAIN_SIZE))
        val_end   = max(train_end + 1, train_end + int(n * Config.VAL_SIZE))

        train_files = files[:train_end]
        val_files   = files[train_end:val_end] or files[:1]
        test_files  = files[val_end:]          or files[:1]

        print(f"   Split: {len(train_files)} train | "
              f"{len(val_files)} val | {len(test_files)} test parquets")
        return train_files, val_files, test_files

    def _safe_drop(self, lf: pl.LazyFrame, path: str) -> pl.LazyFrame:
        """Drop solo las columnas que existen en el schema del parquet."""
        schema = pl.scan_parquet(path).schema
        cols_existentes = [c for c in COLS_TO_DROP if c in schema]
        cols_force = [c for c in Config.COLS_FORCE_FLOAT64 if c in schema]
        
        return (
            lf.with_columns([pl.col(c).cast(pl.Float64) for c in cols_force])
            .filter(pl.col("Attack") != "Worms")
            .drop(cols_existentes)
            .drop_nulls(subset=["Attack"])
        )

    # ------------------------------------------------------------------
    # carga pl.LazyFrame — un parquet a la vez para evitar SchemaError
    # ------------------------------------------------------------------
    def _load_files(self, file_list):
        frames = []
        for fname in file_list:
            path = os.path.join(self.raw_path, fname)
            lf = pl.scan_parquet(path)
            lf = self._safe_drop(lf, path)
            frames.append(lf)
        return pl.concat(frames, how="diagonal_relaxed")

    # ------------------------------------------------------------------
    # MUESTREO ESTRATIFICADO CON MÍNIMOS GARANTIZADOS
    # Estrategia:
    #   - Benignos: muestra fija de n_benign filas (submuestreo de la clase
    #     mayoritaria para respetar el ratio configurado en pct_benign).
    #   - Ataques: muestreo proporcional al peso natural de cada subclase.
    #     Las clases minoritarias (<6K filas totales) son tan escasas que
    #     su peso natural hace que se cojan prácticamente completas dentro
    #     del presupuesto de n_attack. No se necesita lógica especial.
    #
    # Memoria: carga en lotes de BATCH_FILES parquets para no superar ~4 GB.

    def _stratified_sample(self, file_list, n_total, pct_benign, name, dry_run): # pl.LazyFrame
        n_benign = int(n_total * pct_benign)
        n_attack = n_total - n_benign

        BATCH_FILES = 8
        chunks_benign, chunks_attack = [], []

        for i in range(0, len(file_list), BATCH_FILES):
            batch = file_list[i:i + BATCH_FILES]
            frames = []
            for fname in batch:
                path = os.path.join(self.raw_path, fname)
                lf = pl.scan_parquet(path)
                lf = self._safe_drop(lf, path)
                frames.append(lf)

            df_batch = pl.concat(frames, how="diagonal_relaxed").collect()
            chunks_benign.append(df_batch.filter(pl.col("Attack") == "Benign"))
            chunks_attack.append(df_batch.filter(pl.col("Attack") != "Benign"))
            del df_batch, frames # liberar mem

        df_benign = pl.concat(chunks_benign)
        df_attack = pl.concat(chunks_attack)
        del chunks_benign, chunks_attack # liberar mem

        # benignos: submuestreo fijo 
        samp_benign = df_benign.sample(n=min(df_benign.height, n_benign), seed=Config.SEED)
        del df_benign # liberar

        # ataques: cost-sensitive en dos pasadas 
        counts = (
            df_attack.group_by("Attack")
            .agg(pl.len().alias("n"))
            .sort("n")          # ascendente: procesar raras primero
        )
        total_attack = counts["n"].sum()

        sampled:        list[pl.DataFrame] = []
        abundant_rows:  list[tuple]        = []
        n_rare_total = 0

        # pasada 1: raras completas
        for cls, cnt in counts.iter_rows():
            cuota = int(n_attack * cnt / total_attack) if total_attack > 0 else 0
            if cnt <= cuota:
                # clase rara: NO se muestrea, se coge completa
                sampled.append(df_attack.filter(pl.col("Attack") == cls))
                n_rare_total += cnt
            else:
                abundant_rows.append((cls, cnt))

        # pasada 2: redistribución proporcional entre abundantes
        n_remain       = max(0, n_attack - n_rare_total)
        total_abundant = sum(cnt for _, cnt in abundant_rows)

        for cls, cnt in abundant_rows:
            take = min(cnt, max(1, int(n_remain * cnt / total_abundant))) \
                   if total_abundant > 0 else 0
            if take > 0:
                sampled.append(
                    df_attack.filter(pl.col("Attack") == cls)
                    .sample(n=take, seed=Config.SEED)
                )
        del df_attack # liberar mem

        result = pl.concat([samp_benign] + sampled)
        result = result.sample(fraction=1.0, seed=Config.SEED)   # shuffle global

        # log detallado por macro-clase (post-mapeo) 
        n_atk    = result.filter(pl.col("Attack") != "Benign").height
        cls_dist = (
            result.group_by("Attack")
            .agg(pl.len().alias("n"))
            .sort("n", descending=True)
        )
        print(f"   [{name}] Total: {result.height:,} | "
              f"Benignos: {samp_benign.height:,} ({samp_benign.height/result.height*100:.1f}%) | "
              f"Ataques: {n_atk:,} ({n_atk/result.height*100:.1f}%)")
        for cls, cnt in cls_dist.iter_rows():
            marker = " ← RARA (completa)" if cnt < 1_000 else ""
            print(f"      {cls:<35} {cnt:>7,}  ({cnt/result.height*100:.3f}%){marker}")

        return result

    # ------------------------------------------------------------------
    # LIMPIEZA inf / NaN
    # acepta medianas externas para propagar las de train a val/test
    # y evitar data leakage implícito en los splits de evaluación.
    # ------------------------------------------------------------------
    @staticmethod
    def _clean(arr, medians=None):
        """
        Reemplaza inf -> NaN -> mediana.

        Si `medians` es None, calcula la mediana del propio array (uso en train).
        Si `medians` se proporciona, usa esas medianas fijas (uso en val/test
        y en inferencia en producción).
        """
        arr = np.where(np.isinf(arr), np.nan, arr)
        if medians is None:
            medians = np.nanmedian(arr, axis=0)
        mask = np.isnan(arr)
        arr[mask] = np.take(medians, np.where(mask)[1])
        return arr, medians

    # ------------------------------------------------------------------
    # MAPEO A MACRO-CLASES
    # ------------------------------------------------------------------
    @staticmethod
    def _apply_mapping(df):
        clases_conocidas = set(ATTACK_MAPPING.keys())
        clases_en_df     = set(df["Attack"].unique().to_list())
        sin_mapeo        = clases_en_df - clases_conocidas
        
        if sin_mapeo:
            print(f"   [INFO] Eliminando clases sin mapeo: {sin_mapeo}")
            df = df.filter(pl.col("Attack").is_in(list(clases_conocidas)))
        
        return (
            df.with_columns(
                pl.col("Attack")
                .replace(ATTACK_MAPPING)
                .cast(pl.Int32)
                .alias("Target")
            ).drop("Attack")
        )

    # ------------------------------------------------------------------
    # PIPELINE PRINCIPAL
    # ------------------------------------------------------------------
    def run(self, dry_run = False):
        mode = "DRY RUN" if dry_run else f"FULL RUN — Fase {Config.EXPERIMENT_PHASE}"
        print("\n" + "="*60)
        print(f"DATA PIPELINE — BigFlow-NIDS  [{mode}]")
        print("="*60)

        n_train = Config.N_TRAIN_DRY if dry_run else Config.n_train()
        n_val   = Config.N_VAL_DRY   if dry_run else Config.n_val()
        n_test  = Config.N_TEST_DRY  if dry_run else Config.n_test()

        # 1. Split temporal
        print("\n[-] 1. Split temporal...")
        files = self._get_ordered_files(dry_run)
        train_files, val_files, test_files = self._split_files_temporal(files)

        # 2. Muestreo
        # representación estadísticamente válida de clases raras en métricas.
        print(f"\n[-] 2. Muestreo ({n_train:,} / {n_val:,} / {n_test:,})...")
        df_train = self._stratified_sample(
            # train_files, n_train, Config.TRAIN_PCT_BENIGN, "TRAIN", dry_run)
            train_files, n_train, 0.5, "TRAIN", dry_run)
        
        df_val = self._stratified_sample(
            # val_files, n_val, Config.VAL_PCT_BENIGN, "VAL", dry_run)
            val_files, n_val, 0.8, "VAL", dry_run)
        
        df_test = self._stratified_sample(
            # test_files, n_test, Config.TEST_PCT_BENIGN, "TEST", dry_run)
            test_files, n_test, 0.8, "TEST", dry_run)

        # 3. Mapeo a macro-clases
        print("\n[-] 3. Mapeando a macro-clases (0–8)...")
        
        # guardar etiquetas originales de test ANTES del mapeo
        y_test_raw = df_test["Attack"].to_numpy()

        df_train = self._apply_mapping(df_train)
        df_val   = self._apply_mapping(df_val)
        df_test  = self._apply_mapping(df_test)

        # 5. Arrays numpy (float64 para precisión en el scaler)
        print("\n[-] 5. Extrayendo arrays numpy...")
        feature_cols = [c for c in df_train.columns if c != "Target"]

        X_train  = df_train.select(feature_cols).to_numpy().astype(np.float64)
        y_train  = df_train["Target"].to_numpy()
        X_val    = df_val.select(feature_cols).to_numpy().astype(np.float64)
        y_val    = df_val["Target"].to_numpy()
        X_test   = df_test.select(feature_cols).to_numpy().astype(np.float64)
        y_test   = df_test["Target"].to_numpy()

        # 6. Limpieza inf/NaN
        # medianas calculadas en train y propagadas a val/test/benign
        # para garantizar consistencia en producción.
        print("\n[-] 6. Limpiando inf/NaN (medianas fijadas en train)...")
        
        # capturamos medianas de train
        X_train, train_medians  = self._clean(X_train, medians=None)
        
        # aquí las aplicamos al resto (usamos '_' para ignorar el segundo valor devuelto)
        X_val, _    = self._clean(X_val, medians=train_medians)
        X_test, _   = self._clean(X_test, medians=train_medians)
        
        for label, arr in [("X_train", X_train), ("X_val", X_val),
                            ("X_test",  X_test)]:
            print(f"   {label}: inf={np.isinf(arr).sum()}, nan={np.isnan(arr).sum()}")

        # 7. QuantileTransformer — fit X_train COMPLETO, no solo benignos, 
        # no liar con VAE (filtrar benignos de X_train solo)
        print("\n[-] 7. Ajustando QuantileTransformer global sobre todo Train...")
        scaler = QuantileTransformer(
            output_distribution='normal',
            n_quantiles=min(X_train.shape[0], 10_000),
            random_state=Config.SEED,
        )
        
        scaler.fit(X_train)

        X_train_sc  = scaler.transform(X_train).astype(np.float32)
        X_val_sc    = scaler.transform(X_val).astype(np.float32)
        X_test_sc   = scaler.transform(X_test).astype(np.float32)

        # 8. Prior de ataque para corrección bayesiana en inferencia
        pi_train = float((y_train != 0).mean())
        print(f"\n[-] 8. Prior de ataque en train: π = {pi_train:.4f}")

        # 9. Diagnóstico de distribución post-escalado
        print("\n[-] 9. Diagnóstico post-escalado...")
        for label, arr, y in [
            ("X_train_sc",  X_train_sc,  y_train),
            ("X_val_sc",    X_val_sc,    y_val),
            ("X_test_sc",   X_test_sc,   y_test),
        ]:
            benign_mask  = (y == 0)
            attack_mask  = ~benign_mask
            unique, cnts = np.unique(y, return_counts=True)
            dist_str = " | ".join(f"c{c}:{n:,}" for c, n in zip(unique, cnts))
            print(f"   {label}: shape={arr.shape} "
                  f"mean_b={arr[benign_mask].mean():.3f} "
                  f"mean_a={arr[attack_mask].mean():.3f} "
                  f"[{dist_str}]")
        
        # 10. Guardado
        print("\n[-] 10. Guardando artefactos...")
        s = "_dry" if dry_run else ""
        p, m = self.processed_path, self.models_path

        # Datos procesados
        np.save(os.path.join(p, f"X_train{s}.npy"),        X_train_sc)
        np.save(os.path.join(p, f"y_train{s}.npy"),        y_train)
        np.save(os.path.join(p, f"X_val{s}.npy"),          X_val_sc)
        np.save(os.path.join(p, f"y_val{s}.npy"),          y_val)
        np.save(os.path.join(p, f"X_test{s}.npy"),         X_test_sc)
        np.save(os.path.join(p, f"y_test{s}.npy"),         y_test)

        # Artefactos del modelo / inferencia
        joblib.dump(scaler, os.path.join(m, f"quantile_scaler{s}.pkl"))
        np.save(os.path.join(m, f"pi_train{s}.npy"),       np.array([pi_train]))
        np.save(os.path.join(m,  "feature_names.npy"),     np.array(feature_cols))
        
        # medianas de train necesarias para limpiar nuevos flujos en
        # producción de forma consistente con lo visto durante el entrenamiento
        np.save(os.path.join(m, f"train_medians{s}.npy"),  train_medians)

        # etiquetas originales de test para análisis por sub-clase post-entrenamiento
        np.save(os.path.join(p, f"y_test_raw{s}.npy"), y_test_raw)

        print("\n" + "="*60)
        print(f"PIPELINE COMPLETADO  [{mode}]")
        print(f"  Features      : {len(feature_cols)}")
        print(f"  Train         : {X_train_sc.shape}")
        print(f"  Val           : {X_val_sc.shape}")
        print(f"  Test          : {X_test_sc.shape}")
        print(f"  π_train       : {pi_train:.4f}")
        print(f"  Scaler fit en : Todo X_train (Global)")
        if dry_run:
            print("\n  DRY RUN OK — ejecuta sin --dry-run para el run completo")
        print("="*60)


if __name__ == "__main__":
    import argparse
    from src.helpers import set_seed

    parser = argparse.ArgumentParser(description="NF-UQ-NIDS-v2 Data Pipeline")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Ejecuta con pocos datos para verificar el pipeline"
    )
    args = parser.parse_args()

    set_seed(Config.SEED)
    DataPipeline().run(dry_run=args.dry_run)