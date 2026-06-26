import os
import numpy as np
import joblib
import polars as pl
from sklearn.preprocessing import QuantileTransformer

from src.config import Config
from src.ip_behavior_buffer  import IPBehaviorBuffer

class DataPipeline:
    """
    Pipeline ETL para BigFlow-NIDS.

    Responsabilidades:
      - Muestreo por IP completa (T1) con cap de flujos por IP (T3):
          Los flujos de cada IP atacante entran JUNTOS y en orden temporal,
          garantizando que el buffer llegue caliente cuando el modelo ve
          los flujos críticos de Recon. Cap de MAX_FLOWS_PER_IP evita que
          una sola IP con 50K flujos domine el dataset.
      - FLOW_RANK_IN_IP (T6): posicion relativa del flujo dentro del
          historial de su IP (0.0 = primer flujo, 1.0 = ultimo). El modelo
          aprende que los primeros flujos de un scanner aun parecen benignos.
      - Buffer deslizante de comportamiento por IP (IPBehaviorBuffer)
          que añade 15 features temporales sin necesidad de ventanas explicitas
      - Limpieza de inf/NaN con medianas de TRAIN propagadas a val/test
      - Escalado dual: QuantileTransformer global para features no-BUF,
          QuantileTransformer benigno para features BUF_* (calibrado sobre
          trafico normal — ataques quedan fuera de esa distribucion)
      - Exportacion de artefactos .npy listos para PyTorch y LightGBM

    Uso:
        dp = DataPipeline()
        dp.run(dry_run=True)   # verificación rápida (segundos)
        dp.run(dry_run=False)  # run completo (5-10 min)
    """

    # Cap de flujos por IP — evita que una IP con 50K flujos domine el dataset.
    # Justificacion: queremos aprender el patron general de Recon, no el
    # comportamiento especifico de un unico atacante concreto.
    # 500 flujos/IP es suficiente para que el buffer llegue bien caliente
    # y el modelo vea el patron completo sin sobreajuste por IP, pero limitamos los ataques
    # Por lo que subimos a 750 para tener al menos 1-1.1M de flujos para entrenar la resnet 
    MAX_FLOWS_PER_IP = 750

    def __init__(self):
        self.raw_path       = Config.DATA_RAW_PATH
        self.processed_path = Config.DATA_PROCESSED_PATH
        self.models_path    = Config.MODELS_PATH
        os.makedirs(self.processed_path, exist_ok=True)
        os.makedirs(self.models_path,    exist_ok=True)

        # Buffer deslizante — instancia única que persiste entre parquets
        # window=120s captura scanners lentos, max_ips=200K ≈ 40MB RAM
        self._ip_buffer = IPBehaviorBuffer(window_seconds=120, max_ips=200_000)

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


    # ------------------------------------------------------------------
    # FLOW_RANK_IN_IP
    # Posicion relativa del flujo dentro del historial de su IP (0.0-1.0).
    # Requiere que IPV4_SRC_ADDR aun este disponible (antes del drop).
    # El modelo aprende que los primeros flujos de un scanner todavia
    # parecen benignos — el rank bajo es señal de buffer frio.
    # NOTA: se calcula ANTES del drop(COLS_TO_DROP), por eso recibe el
    # df crudo. Se añade como columna y se elimina la IP despues.
    # ------------------------------------------------------------------
    @staticmethod
    def _add_flow_rank(df: pl.DataFrame) -> pl.DataFrame:
        """
        Añade FLOW_RANK_IN_IP: rango normalizado [0, 1] de cada flujo
        dentro del grupo de su IP origen, ordenado temporalmente.

        rank=0.0 → primer flujo visto de esa IP en este split
        rank=1.0 → ultimo flujo visto de esa IP en este split

        Implementacion: rank entero por IP / (n_flujos_ip - 1).
        IPs con un solo flujo reciben rank=0.0.
        """
        return df.with_columns(
            # rank() dentro de cada IP, ordenado por timestamp
            pl.col("FLOW_START_MILLISECONDS")
              .rank(method="ordinal")
              .over("IPV4_SRC_ADDR")
              .alias("_rank_raw"),
            # conteo de flujos por IP para normalizar
            pl.col("FLOW_START_MILLISECONDS")
              .count()
              .over("IPV4_SRC_ADDR")
              .alias("_count_ip"),
        ).with_columns(
            # normalizar a [0, 1] — IPs con 1 flujo quedan en 0.0
            pl.when(pl.col("_count_ip") > 1)
              .then((pl.col("_rank_raw") - 1) / (pl.col("_count_ip") - 1))
              .otherwise(0.0)
              .cast(pl.Float32)
              .alias("FLOW_RANK_IN_IP")
        ).drop(["_rank_raw", "_count_ip"])

    # ------------------------------------------------------------------    
    # MUESTREO POR IP COMPLETA CON CAP
    # Estrategia:
    #   1. Cargar todos los flujos de los parquets del split con buffer
    #   2. Calcular FLOW_RANK_IN_IP antes del drop de columnas de ID
    #   3. Aplicar cap MAX_FLOWS_PER_IP por IP: tomar los primeros
    #      N flujos ordenados temporalmente — el buffer llega mas caliente
    #      a los flujos finales que son los mas discriminativos
    #   4. Separar benignos y ataques
    #   5. Benignos: submuestreo por IP completa hasta n_benign flujos
    #   6. Ataques: muestreo proporcional por clase, IPs completas (con cap)
    #      Las clases raras se incluyen completas de forma natural
    #
    # El cap se aplica ANTES del muestreo para que el presupuesto de flujos
    # sea predecible. Sin cap, una IP con 50K flujos podria consumir el
    # presupuesto entero de su clase.
    # ------------------------------------------------------------------
    def _stratified_sample(self, file_list, n_total, pct_benign, name, dry_run):
        n_benign = int(n_total * pct_benign)
        n_attack = n_total - n_benign

        BATCH_FILES = 8
        chunks_benign, chunks_attack = [], []

        for i in range(0, len(file_list), BATCH_FILES):
            batch  = file_list[i:i + BATCH_FILES]
            frames = []
            for fname in batch:
                # Carga con buffer — IPV4_SRC_ADDR todavia disponible aqui
                # para calcular FLOW_RANK_IN_IP antes del drop
                raw_df = (
                    pl.scan_parquet(os.path.join(self.raw_path, fname))
                    .with_columns([
                        pl.col(c).cast(pl.Float64)
                        for c in Config.COLS_FORCE_FLOAT64
                    ])
                    .filter(pl.col("Attack") != "Worms")
                    .drop_nulls(subset=["Attack"])
                    .sort("FLOW_START_MILLISECONDS")
                    .collect()
                )
                raw_df = raw_df.with_columns([
                    # Flujo sin respuesta del destino — señal directa de scanning
                    pl.when(pl.col('OUT_PKTS') == 0)
                    .then(1.0).otherwise(0.0)
                    .alias('IS_UNIDIRECTIONAL'),

                    # Bytes por paquete — probes de Recon son paquetes minimos
                    (pl.col('IN_BYTES') / (pl.col('IN_PKTS') + 1e-8))
                    .alias('SRC_BYTES_PER_PKT'),

                    # Ratio paquetes entrada/salida
                    (pl.col('IN_PKTS') / (pl.col('OUT_PKTS') + 1e-8))
                    .alias('PKT_RATIO'),

                    # Flag HTTP puro — L7_PROTO=7
                    pl.when(pl.col('L7_PROTO') == 7.0)
                    .then(1.0).otherwise(0.0)
                    .alias('IS_HTTP'),

                    # Flag HTTPS/TLS — L7_PROTO=91
                    pl.when(pl.col('L7_PROTO') == 91.0)
                    .then(1.0).otherwise(0.0)
                    .alias('IS_HTTPS'),

                    # Ratio bytes respuesta/peticion — Web/Injection: ~6.4, Benign: ~0.15
                    (pl.col('IN_BYTES') / (pl.col('OUT_BYTES') + 1e-8))
                    .alias('RESPONSE_RATIO'),

                    # Ratio ventanas TCP — señal de sesion HTTP activa vs conexion rapida
                    (pl.col('TCP_WIN_MAX_IN') / (pl.col('TCP_WIN_MAX_OUT') + 1e-8))
                    .alias('TCP_WIN_RATIO'),

                    # Bytes totales de la sesion — ataques HTTP generan sesiones grandes
                    (pl.col('IN_BYTES') + pl.col('OUT_BYTES'))
                    .alias('TOTAL_BYTES'),

                    # Duracion por paquete — ataques web tienen IAT caracteristico
                    (pl.col('FLOW_DURATION_MILLISECONDS') /
                    (pl.col('IN_PKTS') + pl.col('OUT_PKTS') + 1e-8))
                    .alias('DURATION_PER_PKT'),

                    # Duracion alta + protocolo web + pocos bytes = candidato Blind SQLi
                    pl.when(
                        (pl.col('L7_PROTO').is_in([7.0, 91.0])) &
                        (pl.col('FLOW_DURATION_MILLISECONDS') > 5000) &
                        (pl.col('IN_BYTES') < 500)
                    ).then(1.0).otherwise(0.0)
                    .alias('IS_BLIND_SQLI_CANDIDATE'),

                    # Probe puro — sin respuesta y paquete pequeño
                    pl.when(
                        (pl.col('OUT_PKTS') == 0) & (pl.col('IN_BYTES') < 100)
                    ).then(1.0).otherwise(0.0)
                    .alias('IS_PROBE'),

                    # Flujo HTTP sospechoso — duracion larga + pocos bytes enviados
                    pl.when(
                        (pl.col('L7_PROTO').is_in([7.0, 91.0])) &
                        (pl.col('FLOW_DURATION_MILLISECONDS') > 5000) &
                        (pl.col('IN_BYTES') < 200)
                    ).then(1.0).otherwise(0.0)
                    .alias('IS_RECON_HTTP'),
                ])

                # FLOW_RANK_IN_IP — calculado aqui cuando IPV4_SRC_ADDR
                # todavia existe, antes del drop
                raw_df = self._add_flow_rank(raw_df)

                # Cap solo sobre ataques — los benignos no tienen el problema del buffer
                # frio y su volumen alto es realista (servidores con trafico legitimo continuo)
                # Ordenamos por timestamp y tomamos los primeros
                # MAX_FLOWS_PER_IP flujos de cada IP. Los primeros flujos son los
                # mas valiosos: el buffer los ve en orden y llega caliente a los
                # flujos finales, que son los mas discriminativos para Recon.
                raw_df = (
                    raw_df
                    .sort("FLOW_START_MILLISECONDS")
                    .with_columns(
                        pl.col("FLOW_START_MILLISECONDS")
                        .rank(method="ordinal")
                        .over("IPV4_SRC_ADDR")
                        .alias("_ip_flow_rank")
                    )
                    .filter(pl.col("_ip_flow_rank") <= self.MAX_FLOWS_PER_IP)
                    .drop("_ip_flow_rank")
                )

                # Buffer temporal — requiere IP/puerto, debe ir ANTES del drop
                raw_df = self._ip_buffer.update_and_extract(raw_df)

                # Drop de columnas de identificacion (IP, puerto, timestamps, etc.)
                cols_to_drop = [c for c in Config.COLS_TO_DROP if c in raw_df.columns]
                raw_df = raw_df.drop(cols_to_drop)

                frames.append(raw_df)

            df_batch = pl.concat(frames, how="diagonal_relaxed")
            
            chunks_benign.append(df_batch.filter(pl.col("Attack") == "Benign"))
            chunks_attack.append(df_batch.filter(pl.col("Attack") != "Benign"))
            del df_batch, frames # liberar mem

            # Log del estado del buffer cada batch
            buf_stats = self._ip_buffer.stats_summary()
            print(f"      [Buffer] IPs activas: {buf_stats['n_ips']:,} | "
                  f"RAM: {buf_stats['memory_mb']:.1f}MB | "
                  f"Parquets: {buf_stats['n_parquets']}")

        df_benign = pl.concat(chunks_benign)
        df_attack = pl.concat(chunks_attack)
        del chunks_benign, chunks_attack # liberar mem

        # Muestreo benignos 
        # Submuestreo aleatorio simple hasta n_benign.
        # Los benignos no tienen el problema del buffer frio — su señal
        # no depende del contexto temporal acumulado.
        samp_benign = df_benign.sample(n=min(df_benign.height, n_benign), seed=Config.SEED)
        del df_benign # liberar

        # Muestreo ataques por clase con dos pasadas
        counts = (
            df_attack.group_by("Attack")
            .agg(pl.len().alias("n"))
            .sort("n")          # ascendente: procesar raras primero
        )
        total_attack = counts["n"].sum()

        sampled = []
        abundant_rows = []
        n_rare_total = 0

        # pasada 1: calses raras completas
        for cls, cnt in counts.iter_rows():
            cuota = int(n_attack * cnt / total_attack) if total_attack > 0 else 0
            if cnt <= cuota:
                # clase rara: NO se muestrea, se coge completa
                sampled.append(df_attack.filter(pl.col("Attack") == cls))
                n_rare_total += cnt
            else:
                abundant_rows.append((cls, cnt))

        # pasada 2: clases abundantes — muestreo proporcional redistribuido
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

        # ESTO QUITAR MÁS ADELANTE
        # Log FLOW_RANK_IN_IP si existe en el resultado
        if "FLOW_RANK_IN_IP" in result.columns:
            rank_vals = result["FLOW_RANK_IN_IP"]
            print(f"      [T6] FLOW_RANK_IN_IP: "
                  f"mean={rank_vals.mean():.3f} | "
                  f"rank<0.1: {(rank_vals < 0.1).sum():,} flujos "
                  f"(buffer frio)")

        return result

    # ------------------------------------------------------------------
    # LIMPIEZA inf / NaN
    # Medianas calculadas en train y propagadas a val/test para evitar
    # data leakage y garantizar consistencia en produccion. 
    # ------------------------------------------------------------------
    @staticmethod
    def _clean(arr, medians=None):
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
        unmapped = set(df["Attack"].unique().to_list()) - set(Config.ATTACK_MAPPING.keys())
        if unmapped:
            print(f"   AVISO: etiquetas sin mapear → {unmapped}")
        return (
            df.with_columns(
                pl.col("Attack")
                .replace(Config.ATTACK_MAPPING)
                .cast(pl.Int32)
                .alias("Target")    # nombre etiqueta Attack -> Target
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

        # Muestreo por IP completa: flujos de cada IP entran juntos y ordenados
        # temporalmente para que el buffer llegue caliente a los flujos de ataque.
        # Cap por IP evita sobreajuste a comportamientos individuales.
        print(f"\n[-] 2. Muestreo por IP completa con cap y rank "
              f"({n_train:,} / {n_val:,} / {n_test:,})...")
        print(f"  Cap por IP: {self.MAX_FLOWS_PER_IP} flujos max")
        print("   [Buffer] TRAIN - buffer empieza en frio")
        df_train = self._stratified_sample(
            train_files, n_train, Config.TRAIN_PCT_BENIGN, "TRAIN", dry_run)

        # Reset entre splits — val/test simulan arranques independientes.
        # En produccion el buffer NO se resetea.
        print("\n   [Buffer] Reset - VAL empieza en frio")
        self._ip_buffer.reset()
        df_val = self._stratified_sample(
            val_files, n_val, Config.VAL_PCT_BENIGN, "VAL", dry_run)

        print("\n   [Buffer] Reset - TEST empieza en frio")
        self._ip_buffer.reset()
        df_test = self._stratified_sample(
            test_files, n_test, Config.TEST_PCT_BENIGN, "TEST", dry_run)

        # 3. Mapeo a macro-clases
        print("\n[-] 3. Mapeando a macro-clases (0–8)...")
        
        # guardar etiquetas originales de test antes del mapeo
        y_test_raw = df_test["Attack"].to_numpy()

        df_train = self._apply_mapping(df_train)
        df_val   = self._apply_mapping(df_val)
        df_test  = self._apply_mapping(df_test)

        # 4. Benignos puros para VAE (Enfoque resnet -> latent z space <- VAE)
        print("\n[-] 4. Extrayendo benignos puros para VAE...")
        df_benign_vae = df_train.filter(pl.col("Target") == 0).drop("Target")
        print(f"   Flujos benignos para VAE: {df_benign_vae.height:,}")

        # 5. Arrays numpy (float64 para precisión en el scaler)
        print("\n[-] 5. Extrayendo arrays numpy...")
        feature_cols = [c for c in df_train.columns if c != "Target"]
        buf_cols      = [c for c in feature_cols if c.startswith("BUF_")]
        other_cols    = [c for c in feature_cols if not c.startswith("BUF_")]
        buf_indices   = [feature_cols.index(c) for c in buf_cols]
        other_indices = [feature_cols.index(c) for c in other_cols]

        print(f"   Features totales: {len(feature_cols)} "
              f"({len(other_cols)} globales + {len(buf_cols)} BUF_*)")

        X_train  = df_train.select(feature_cols).to_numpy().astype(np.float64)
        y_train  = df_train["Target"].to_numpy()
        X_val    = df_val.select(feature_cols).to_numpy().astype(np.float64)
        y_val    = df_val["Target"].to_numpy()
        X_test   = df_test.select(feature_cols).to_numpy().astype(np.float64)
        y_test   = df_test["Target"].to_numpy()
        X_benign = df_benign_vae.to_numpy().astype(np.float64)

        # 6. Limpieza inf/NaN
        # medianas calculadas en train y propagadas a val/test/benign
        # para garantizar consistencia en producción.
        print("\n[-] 6. Limpiando inf/NaN (medianas fijadas en train)...")
        
        # capturamos medianas de train
        X_train, train_medians  = self._clean(X_train, medians=None)
        
        # aquí las aplicamos al resto (usamos '_' para ignorar el segundo valor devuelto)
        X_val, _    = self._clean(X_val, medians=train_medians)
        X_test, _   = self._clean(X_test, medians=train_medians)
        X_benign, _ = self._clean(X_benign, medians=train_medians)
        
        for label, arr in [("X_train", X_train), ("X_val", X_val),
                            ("X_test",  X_test),  ("X_benign", X_benign)]:
            print(f"   {label}: inf={np.isinf(arr).sum()}, nan={np.isnan(arr).sum()}")

        # 7. QuantileTransformer - Escalado dual
        # scaler_global: QuantileTransformer sobre X_train COMPLETO para features no-BUF
        # scaler_benign: QuantileTransformer ajustado SOLO sobre benignos de train para 
        # features BUF_* — calibra las features de buffer respecto a la distribucion del 
        # trafico normal. Una IP con BUF_SCAN_RATE en el percentil 99.9 benigno queda 
        # en la cola de la distribucion normal, señal directa de anomalia.
        print("\n[-] 7. Ajustando scalers (global + benigno para BUF_*)...")
        scaler_global = QuantileTransformer(
            output_distribution='normal',
            n_quantiles=min(X_train.shape[0], 10_000),
            random_state=Config.SEED,
        )
        scaler_global.fit(X_train[:, other_indices])

        benign_mask   = (y_train == 0)
        scaler_benign = QuantileTransformer(
            output_distribution='normal',
            n_quantiles=min(int(benign_mask.sum()), 10_000),
            random_state=Config.SEED,
        )
        scaler_benign.fit(X_train[benign_mask][:, buf_indices])
        print(f"   Scaler global  -> {len(other_cols)} features no-BUF")
        print(f"   Scaler benigno -> {len(buf_cols)} features BUF_* "
              f"({benign_mask.sum():,} flujos benignos)")

        def scale_split(X):
            X_sc = np.empty(X.shape, dtype=np.float32)
            X_sc[:, other_indices] = scaler_global.transform(
                X[:, other_indices]).astype(np.float32)
            X_sc[:, buf_indices]   = scaler_benign.transform(
                X[:, buf_indices]).astype(np.float32)
            return X_sc

        X_train_sc  = scale_split(X_train)
        X_val_sc    = scale_split(X_val)
        X_test_sc   = scale_split(X_test)
        X_benign_sc = scale_split(X_benign)

        # 8. Prior de ataque para corrección bayesiana en inferencia
        pi_train = float((y_train != 0).mean())
        print(f"\n[-] 8. Prior de ataque en train: pi = {pi_train:.4f}")

        # 9. Diagnóstico de distribución post-escalado (bm -> benign mask)
        print("\n[-] 9. Diagnóstico post-escalado...")
        for label, arr, y in [
            ("X_train_sc", X_train_sc, y_train),
            ("X_val_sc",   X_val_sc,   y_val),
            ("X_test_sc",  X_test_sc,  y_test),
        ]:
            bm           = (y == 0)
            unique, cnts = np.unique(y, return_counts=True)
            dist_str     = " | ".join(f"c{c}:{n:,}" for c, n in zip(unique, cnts))
            print(f"   {label}: shape={arr.shape} "
                  f"mean_b={arr[bm].mean():.3f} "
                  f"mean_a={arr[~bm].mean():.3f} [{dist_str}]")
        
        # Diagnóstico features buffer antes de escalar
        buf_indices_diag = [i for i, c in enumerate(feature_cols) if c.startswith("BUF_")]
        if buf_indices_diag:
            print("\n   Diagnóstico features buffer (X_train, sin escalar):")
            for idx in buf_indices_diag:
                col      = feature_cols[idx]
                vals     = X_train[:, idx]
                nz_pct   = (vals > 0).mean() * 100
                print(f"     {col:<25} mean={vals.mean():8.2f} "
                      f"max={vals.max():8.0f} nonzero={nz_pct:.1f}%")

        # 10. Guardado de artefactos
        print("\n[-] 10. Guardando artefactos...")
        s = "_dry" if dry_run else ""
        p, m = self.processed_path, self.models_path

        np.save(os.path.join(p, f"X_train{s}.npy"),        X_train_sc)
        np.save(os.path.join(p, f"y_train{s}.npy"),        y_train)
        np.save(os.path.join(p, f"X_val{s}.npy"),          X_val_sc)
        np.save(os.path.join(p, f"y_val{s}.npy"),          y_val)
        np.save(os.path.join(p, f"X_test{s}.npy"),         X_test_sc)
        np.save(os.path.join(p, f"y_test{s}.npy"),         y_test)
        np.save(os.path.join(p, f"X_train_benign{s}.npy"), X_benign_sc)

        # Ambos scalers necesarios para inferencia en produccion
        joblib.dump(scaler_global, os.path.join(m, f"quantile_scaler{s}.pkl"))
        joblib.dump(scaler_global, os.path.join(m, f"quantile_scaler_global{s}.pkl"))
        joblib.dump(scaler_benign, os.path.join(m, f"quantile_scaler_benign{s}.pkl"))

        np.save(os.path.join(m, f"pi_train{s}.npy"),      np.array([pi_train]))
        np.save(os.path.join(m,  "feature_names.npy"),    np.array(feature_cols))
        np.save(os.path.join(m, f"train_medians{s}.npy"), train_medians)
        np.save(os.path.join(p, f"y_test_raw{s}.npy"),    y_test_raw)

        # Indices de columnas por tipo de scaler — necesarios en inferencia
        np.save(os.path.join(m, f"buf_indices{s}.npy"),   np.array(buf_indices))
        np.save(os.path.join(m, f"other_indices{s}.npy"), np.array(other_indices))

        # Buffer serializado para cold-start en produccion (IDS/IPS online)
        # soluciona el cold start del buffer en entornos de industria 
        import pickle
        buf_path = os.path.join(m, f"ip_buffer{s}.pkl")
        with open(buf_path, 'wb') as f:
            pickle.dump(self._ip_buffer, f)
        print(f"    Buffer serializado: {buf_path}." + "\n") 
        print(f"    Scalers: quantile_scaler.pkl (global) + quantile_scaler_benign.pkl (BUF_*)" +      
              f"\n\nEl buffer ya tiene el historial de IPs activas")

        print("\n" + "="*60)
        print(f"PIPELINE COMPLETADO  [{mode}]")
        print(f"  Tecnicas activas: IP sampling + cap/IP + flow rank")
        print(f"  Cap por IP      : {self.MAX_FLOWS_PER_IP} flujos")
        print(f"  Features total  : {len(feature_cols)}")
        print(f"  Features BUF_*  : {len(buf_cols)} (scaler benigno)")
        print(f"  Features otras  : {len(other_cols)} (scaler global)")
        print(f"  Train           : {X_train_sc.shape}")
        print(f"  Val             : {X_val_sc.shape}")
        print(f"  Test            : {X_test_sc.shape}")
        print(f"  Benign (VAE)    : {X_benign_sc.shape}")
        print(f"  pi_train        : {pi_train:.4f}")
        if dry_run:
            print("\n  DRY RUN OK - ejecuta sin --dry-run para el run completo")
        print("="*60)


if __name__ == "__main__":
    import argparse
    from src.helpers import set_seed

    parser = argparse.ArgumentParser(description="BigFlow-NIDS Data Pipeline con buffer caliente")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Ejecuta con pocos datos para verificar el pipeline"
    )
    args = parser.parse_args()

    set_seed(Config.SEED)
    DataPipeline().run(dry_run=args.dry_run)