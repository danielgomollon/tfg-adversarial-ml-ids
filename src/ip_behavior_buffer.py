"""
ip_behavior_buffer.py 
Buffer deslizante de comportamiento por IP
====================================================================
Captura contexto temporal entre parquets consecutivos sin necesidad
de ventanas de tiempo explícitas ni reescritura del pipeline.

Diseño:
  - Solo guarda estadísticas agregadas por IP, no flujos completos
  - Memoria: ~200 bytes/IP activa → 200K IPs ≈ 40MB RAM
  - Persiste entre parquets para capturar scanners que cruzan archivos
  - Evicción automática de IPs inactivas (ventana configurable)
  - Causalidad estricta: features calculadas con comportamiento PASADO
    (antes de actualizar con el flujo actual)

Features generadas (15 nuevas columnas):
  BUF_UNIQUE_DST_PORTS  → puertos destino únicos visitados
  BUF_UNIQUE_DST_IPS    → IPs destino únicas contactadas
  BUF_FLOW_COUNT        → total flujos de esta IP
  BUF_NO_RESP_RATIO     → ratio flujos sin respuesta (puertos cerrados)
  BUF_SCAN_RATE         → flujos por segundo desde primera aparición
  BUF_SMALL_PKT_RATIO   → ratio paquetes < 100 bytes (probes de Recon)
  BUF_PORT_STD          → desviación estándar de puertos destino
  BUF_PORT_RANGE        → rango max-min de puertos destino
  BUF_IS_SCANNER        → flag duro: 1.0 si es casi certeza de scanner
  BUF_HTTP_RATIO        → ratio flujos HTTP sobre total flujos de la IP
  BUF_HTTP_BYTES_AVG    → bytes promedio enviados en flujos HTTP
  BUF_HTTP_SMALL_RATIO  → ratio peticiones HTTP desnudas (< 500 bytes, firma SQLMap)
  BUF_BURST_PORTS         → puertos únicos en ventana de ráfaga
  BUF_SYN_ACK_RST_RATIO  ratio flujos TCP_FLAGS==22
                          Verificado: IPs Recon>0.40: 20.7% | Benign>0.40: 0.0%
  BUF_RECON_SCORE        score compuesto [0,1] umbrales hard verificados:
                            +0.5 si SYN_ACK_RST_RATIO > 0.40
                            +0.3 si unique_ports > 50 AND no_resp > 30%
                            +0.2 si scan_rate > 10 flujos/segundo

Integración en DataPipeline:
  Instanciar una vez en __init__ y llamar a update_and_extract()
  en cada batch de parquets ANTES de drop(COLS_TO_DROP).
"""

import numpy as np
from collections import defaultdict
from typing import Optional
import polars as pl

from src.config import Config

class IPBehaviorBuffer:
    """
    Buffer deslizante de comportamiento temporal por IP origen.

    Parámetros
    ----------
    window_seconds : int
        Ventana de inactividad tras la cual una IP se considera "fría"
        y sus estadísticas se eliminan. 120s captura scanners lentos
        (stealth scanning) que 60s perdería.
    max_ips : int
        Límite de IPs activas simultáneas. 200_000 IPs × ~200 bytes
        ≈ 40MB RAM. Si se supera, se evictan las IPs más antiguas.
    """

    def __init__(self, window_seconds: int = 120, max_ips: int = 200_000):
        self.window   = window_seconds
        self.max_ips  = max_ips
        self._stats   = defaultdict(self._empty_entry)
        self._n_calls = 0   # contador de parquets procesados (debug)

    # ------------------------------------------------------------------
    # INTERFAZ PÚBLICA
    # ------------------------------------------------------------------
    def update_and_extract(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Procesa un DataFrame (un parquet o batch), añade las 15 features
        de comportamiento temporal y actualiza el buffer interno.

        El orden es crítico:
          1. Extraer features con estado ANTERIOR al flujo actual
          2. Actualizar estado con el flujo actual
        Esto garantiza causalidad — el modelo nunca ve el futuro.

        Parámetros
        ----------
        df: pl.DataFrame con columnas:
            IPV4_SRC_ADDR, IPV4_DST_ADDR, L4_DST_PORT,
            OUT_PKTS, IN_BYTES, FLOW_START_MILLISECONDS,
            L7_PROTO, 

        Retorna
        -------
        pl.DataFrame con 14 columnas adicionales BUF_*.
        """
        self._n_calls += 1

        # Columnas mínimas necesarias — evitar cargar el DataFrame completo
        # índices: 0=SRC_IP, 1=DST_IP, 2=DST_PORT, 3=OUT_PKTS, 4=IN_BYTES, 5=TS_MS, 6=L7_PROTO
        required = [
            'IPV4_SRC_ADDR', 'IPV4_DST_ADDR', 'L4_DST_PORT',
            'OUT_PKTS', 'IN_BYTES',
            'FLOW_START_MILLISECONDS',
            'L7_PROTO', 'TCP_FLAGS', # para contar flujos HTTP (L7_PROTO=7 o 91)
        ]

        # verificar que existen (defensivo)
        available = set(df.columns)
        missing   = [c for c in required if c not in available]
        if missing:
            raise ValueError(f"IPBehaviorBuffer: columnas faltantes -> {missing}")

        # extraer solo lo necesario como numpy para iterar bien
        mini = df.select(required).to_numpy()
        # cols: 0=SRC_IP, 1=DST_IP, 2=DST_PORT, 3=OUT_PKTS, 4=IN_BYTES,
        #       5=TS_MS, 6=L7_PROTO, 7=TCP_FLAGS

        # evictar IPs inactivas antes de procesar el parquet
        # Usamos el timestamp máximo del parquet como referencia
        ts_max_s = float(mini[:, 5].max()) / 1000.0
        self._evict_stale(ts_max_s)

        # calcular features y actualizar buffer en un solo pase
        # mejor hacer un loop aparte para no complicar tanto el código
        n_rows  = mini.shape[0]
        features = np.zeros((n_rows, 15), dtype=np.float32)

        for i in range(n_rows):
            src_ip   = mini[i, 0]
            dst_ip   = mini[i, 1]
            dst_port = int(mini[i, 2])
            dst_pkts = int(mini[i, 3])
            src_bytes= float(mini[i, 4])
            ts_s     = float(mini[i, 5]) / 1000.0
            l7       = float(mini[i, 6])
            tcp_flag  = int(mini[i, 7])

            s = self._stats[src_ip]
            fc = s['flow_count']

            # 1. extraer features con estadoprevio (causalidad)
            features[i, 0] = len(s['dst_ports'])            # BUF_UNIQUE_DST_PORTS
            features[i, 1] = len(s['dst_ips'])              # BUF_UNIQUE_DST_IPS
            features[i, 2] = fc                             # BUF_FLOW_COUNT
            features[i, 3] = s['no_resp'] / max(fc, 1)      # BUF_NO_RESP_RATIO
            features[i, 4] = (                              # BUF_SCAN_RATE
                fc / max(ts_s - s['first_seen'], 1.0)
                if s['first_seen'] > 0 else 0.0
            )
            features[i, 5] = s['small_pkts'] / max(fc, 1)   # BUF_SMALL_PKT_RATIO

            ph = s['port_history']
            if len(ph) > 1:
                ph_arr         = np.array(ph, dtype=np.float32)
                features[i, 6] = float(ph_arr.std())                # BUF_PORT_STD
                features[i, 7] = float(ph_arr.max() - ph_arr.min()) # BUF_PORT_RANGE
            # else: ya son 0.0 por np.zeros

            # BUF_IS_SCANNER — flag duro heurístico
            features[i, 8] = 1.0 if (                               # BUF_IS_SCANNER
                len(s['dst_ports']) >= Config._SCANNER_PORT_THR and
                s['no_resp'] / max(fc, 1) >= Config._SCANNER_RESP_THR
            ) else 0.0

            features[i, 9]  = s['http_flows']     / max(fc, 1)   # BUF_HTTP_RATIO
            features[i, 10] = s['http_bytes_out'] / max(fc, 1)   # BUF_HTTP_BYTES_AVG
            features[i, 11] = s['http_small_flows'] / max(fc, 1) # BUF_HTTP_SMALL_FLOW_RATIO
            
            # la feature es el número de puertos únicos en la ventana de ráfaga
            features[i, 12] = len(set(s['burst_ports_list']))   

            # BUF_SYN_ACK_RST_RATIO
            syn_rst_ratio   = s['syn_ack_rst'] / max(fc, 1)
            features[i, 13] = syn_rst_ratio

            # BUF_RECON_SCORE — umbrales hard verificados empiricamente
            recon_score = 0.0
            if syn_rst_ratio > 0.40:
                recon_score += 0.5
            if len(s['dst_ports']) > 50 and s['no_resp'] / max(fc, 1) > 0.30:
                recon_score += 0.3
            scan_rate = (
                fc / max(ts_s - s['first_seen'], 1.0)
                if s['first_seen'] > 0 else 0.0
            )
            if scan_rate > 10.0:
                recon_score += 0.2
            features[i, 14] = min(1.0, recon_score)

            # 2. actualizar estado con flujo actual 

            s['dst_ports'].add(dst_port)
            s['dst_ips'].add(dst_ip)
            s['flow_count']     += 1
            s['no_resp']        += 1 if dst_pkts == 0 else 0    # OUT_PKTS=0 -> sin respuesta -> OK
            s['small_pkts']     += 1 if src_bytes < Config._SMALL_PKT_BYTES else 0
            s['last_seen']      = ts_s
            if s['first_seen']  == 0:
                s['first_seen'] = ts_s
            
            # actualización web

            # is_http se define localmente en el Paso 2 por localidad y rendimiento:
            # 1. Evita el overhead de llamar a funciones externas en un bucle de millones de iteraciones.
            # 2. Al estar aislado en la fase de actualización, previene por diseño que el 
            #    protocolo del paquete actual se filtre accidentalmente en el Paso 1 (Extracción).
            is_http = l7 in (7.0, 91.0)
            s['http_flows']     += 1 if l7 in (7.0, 91.0) else 0
            s['http_bytes_out'] += src_bytes if l7 in (7.0, 91.0) else 0 # IN_BYTES del atacante -> OK
            s['http_small_flows'] += 1 if (is_http and src_bytes < 500) else 0

            s['syn_ack_rst']      += 1 if tcp_flag == 22 else 0

            s['burst_ports_list'].append(dst_port)
            if len(s['burst_ports_list']) > Config._BURST_WINDOW:
                s['burst_ports_list'].pop(0)
            
            # Historial de puertos acotado — evitar memory leak por IP
            ph.append(dst_port)
            if len(ph) > Config._MAX_PORT_HISTORY:
                ph.pop(0)

        # Evictar si superamos el límite de IPs
        self._evict_overflow()

        # Construir DataFrame de features y concatenar
        buf_df = pl.DataFrame({
            'BUF_UNIQUE_DST_PORTS'  : features[:, 0],
            'BUF_UNIQUE_DST_IPS'    : features[:, 1],
            'BUF_FLOW_COUNT'        : features[:, 2],
            'BUF_NO_RESP_RATIO'     : features[:, 3],
            'BUF_SCAN_RATE'         : features[:, 4],
            'BUF_SMALL_PKT_RATIO'   : features[:, 5],
            'BUF_PORT_STD'          : features[:, 6],
            'BUF_PORT_RANGE'        : features[:, 7],
            'BUF_IS_SCANNER'        : features[:, 8],
            'BUF_HTTP_RATIO'        : features[:, 9],
            'BUF_HTTP_BYTES_AVG'    : features[:, 10],
            'BUF_HTTP_SMALL_RATIO'  : features[:, 11],
            'BUF_BURST_PORTS'       : features[:, 12],
            'BUF_SYN_ACK_RST_RATIO' : features[:, 13],
            'BUF_RECON_SCORE'       : features[:, 14],
        })

        return pl.concat([df, buf_df], how='horizontal')

    def reset(self):
        """Limpia el buffer completamente. Útil entre splits train/val/test."""
        self._stats.clear()
        self._n_calls = 0

    def memory_mb(self) -> float:
        """Estimación de RAM usada por el buffer en MB."""
        # ~200 bytes por IP: set dst_ports + set dst_ips + list port_history + scalars
        return len(self._stats) * 200 / (1024 ** 2)

    def stats_summary(self) -> dict:
        """Resumen del estado del buffer para logging."""
        return {
            'n_ips'      : len(self._stats),
            'memory_mb'  : round(self.memory_mb(), 2),
            'n_parquets' : self._n_calls,
        }

    # ------------------------------------------------------------------
    # MÉTODOS PRIVADOS
    # ------------------------------------------------------------------
    @staticmethod
    def _empty_entry() -> dict:
        return {
            'dst_ports'         : set(),
            'dst_ips'           : set(),
            'port_history'      : [],
            'burst_ports_list'  : [],
            'flow_count'        : 0,
            'no_resp'           : 0,
            'small_pkts'        : 0,
            'first_seen'        : 0.0,
            'last_seen'         : 0.0,
            'http_flows'        : 0,        # flujos HTTP (L7_PROTO=7 o 91)
            'http_bytes_out'    : 0.0,      # bytes enviados en flujos HTTP
            'http_small_flows'  : 0,
            'syn_ack_rst'       : 0,        # flujos con TCP_FLAGS==22 (SYN+ACK+RST)    
        }

    def _evict_stale(self, current_ts: float):
        """Elimina IPs sin actividad en los últimos window_seconds."""
        stale = [
            ip for ip, s in self._stats.items()
            if s['last_seen'] > 0 and (current_ts - s['last_seen']) > self.window
        ]
        for ip in stale:
            del self._stats[ip]

    def _evict_overflow(self):
        """Si hay más IPs que max_ips, elimina las más antiguas."""
        # Solo limpiamos si superamos el límite por un margen (ej. 10.000)
        # Esto evita hacer un 'sorted' costoso todo el rato
        if len(self._stats) < self.max_ips + 10_000:
            return

        overflow = len(self._stats) - self.max_ips
        
        if overflow <= 0:
            return
        # Ordenar por last_seen y eliminar las más antiguas
        oldest = sorted(
            self._stats.items(),
            key=lambda x: x[1]['last_seen']
        )[:overflow]
        for ip, _ in oldest:
            del self._stats[ip]