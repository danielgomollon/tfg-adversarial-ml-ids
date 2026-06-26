"""
src/attacks/leaf.py
================================================================
LEAF — Latent Evasion via Activation Frontiers

Daniel Gomollón Embid — TFG 2025-2026
Análisis, Explotación y Mitigación de Vulnerabilidades de Sistemas
de Detección de Intrusiones basados en Machine Learning

═══════════════════════════════════════════════════════════════
PARADIGMA: Ingeniería Inversa de la Geometría del Árbol
═══════════════════════════════════════════════════════════════

SHAP responde a: ¿qué features contribuyen más a la predicción actual?
LEAF responde a: ¿qué features necesito mover menos para cambiar la predicción?

A diferencia de los ataques basados en SHAP (como SGFP), LEAF no depende de
la importancia global, sino de la arquitectura intrínseca del ensemble. 
Identifica umbrales (frontiers) que, aunque tengan un peso SHAP bajo por su
poca frecuencia en el entrenamiento, se encuentran a una distancia mínima 
del flujo actual en el subespacio Forward.

═══════════════════════════════════════════════════════════════
SCORE DE RENTABILIDAD Y GREEDY PURSUIT
═══════════════════════════════════════════════════════════════

El ataque opera bajo un esquema de "Greedy Pursuit" (Búsqueda Codiciosa) 
para garantizar el sigilo (L0 mínimo). El proceso se divide en:

1. Mapeo de Fronteras: Se extraen todos los nodos de decisión del modelo.
2. Cálculo de Score: Para cada feature atacable, se evalúa:
   score(f, n) = Δ_ganancia_benigna(n) / (|x_f - τ_n| + ε)
3. Optimización Quirúrgica: Las perturbaciones se aplican de forma iterativa, 
   una variable a la vez, priorizando los saltos de mayor rentabilidad. 
   La perturbación de una muestra se congela en el instante exacto en que 
   se logra la evasión, minimizando el impacto en el DomainConstraints.

═══════════════════════════════════════════════════════════════
DIFERENCIA CON SGFP (Proxy vs. Realidad)
═══════════════════════════════════════════════════════════════

SGFP  : Usa valores SHAP como proxy del gradiente.
        → Infravalora features con umbral cercano pero baja importancia global.
        → Aplica perturbaciones aproximadas (ε·sign).

LEAF  : Accede a los umbrales exactos mediante dump_model().
        → Encuentra el salto mínimo real hacia la hoja benigna más rentable.
        → Ejecución quirúrgica: solo altera lo necesario para evadir.

En escenarios con "Física ON", LEAF maximiza la eficiencia del presupuesto 
al evitar movimientos innecesarios que activen dependencias causales 
imprevistas en el motor de restricciones.
"""

from __future__ import annotations

import numpy as np
from typing import Optional

from src.attacks.base_attacks import BaseAttack
from src.utils.domain_constraints import DomainConstraints


class LEAFAttack(BaseAttack):
    """
    LEAF — Latent Evasion via Activation Frontiers.
    Ataque iterativo y quirúrgico para modelos de árboles de decisión.
    """

    def __init__(
        self,
        constraints : Optional[DomainConstraints],
        model_trees,
        epsilon     : float = 0.1,
        mode        : str   = 'targeted',
        top_k       : int   = 10, # Máximo de features a intentar mover
        min_gain    : float = 1e-4,
        **kwargs,
    ):
        kwargs['device'] = 'cpu'
        super().__init__(constraints, epsilon=epsilon, **kwargs)

        if mode not in ('targeted', 'untargeted'):
            raise ValueError(f"mode debe ser 'targeted' o 'untargeted'")

        self.mode     = mode
        self.top_k    = top_k
        self.min_gain = min_gain

        # Cachear el mapa de fronteras una sola vez
        self.frontier_map = self._build_frontier_map(model_trees)

    @property
    def name(self) -> str:
        return f"LEAF (Greedy Pursuit | {self.mode}, ε={self.epsilon})"

    def _build_frontier_map(self, model_trees) -> dict:
        """Parsea el ensemble completo y extrae nodos de decisión."""
        if not hasattr(model_trees, 'dump_model'):
            raise ValueError("[LEAF] model_trees debe ser el Booster nativo.")
            
        dump = model_trees.dump_model()
        frontier_map = {}
        lr = dump.get('parameters', {}).get('learning_rate', 0.1)

        for tree_idx, tree_info in enumerate(dump.get('tree_info', [])):
            n_iter     = dump.get('num_tree_per_iteration', 1)
            clase_arbol = tree_idx % n_iter  

            self._recorrer_nodo(
                nodo        = tree_info['tree_structure'],
                clase_arbol = clase_arbol,
                frontier_map= frontier_map,
                lr          = lr,
            )
        return frontier_map

    def _recorrer_nodo(self, nodo: dict, clase_arbol: int, frontier_map: dict, lr: float) -> None:
        """Recursión para extraer splits y calcular delta de ganancia benigna."""
        if 'split_feature' not in nodo:
            return

        feat_idx  = nodo['split_feature']
        threshold = nodo['threshold']

        val_izq = self._get_leaf_value(nodo['left_child'])
        val_der = self._get_leaf_value(nodo['right_child'])

        if clase_arbol == 0:
            delta = (val_izq - val_der) * lr
        else:
            delta = -(val_izq - val_der) * lr

        if feat_idx not in frontier_map:
            frontier_map[feat_idx] = []

        frontier_map[feat_idx].append((threshold, delta))

        self._recorrer_nodo(nodo['left_child'],  clase_arbol, frontier_map, lr)
        self._recorrer_nodo(nodo['right_child'], clase_arbol, frontier_map, lr)

    def _get_leaf_value(self, nodo: dict) -> float:
        if 'leaf_value' in nodo:
            return float(nodo['leaf_value'])
        val_izq = nodo.get('left_child',  {}).get('leaf_value', 0.0)
        val_der = nodo.get('right_child', {}).get('leaf_value', 0.0)
        return (float(val_izq) + float(val_der)) / 2.0

    def _generate_perturbation(self, X, y, model):
        n_samples, n_features = X.shape
        n_queries = 0

        f_mask = getattr(self, 'forward_mask', None)
        if f_mask is None and self.dc is not None:
            f_mask = self.dc.forward_mask

        integer_mask = None
        if self.dc is not None and hasattr(self.dc, 'integer_mask'):
            integer_mask = self.dc.integer_mask

        # 1. Calcular el ranking de movimientos (acciones) PARA CADA MUESTRA
        sample_actions = []
        for i in range(n_samples):
            scores = {}
            for feat_idx, splits in self.frontier_map.items():
                if f_mask is not None and not f_mask[feat_idx]:
                    continue
                if feat_idx >= n_features:
                    continue

                valor_actual = X[i, feat_idx]
                es_entera    = (integer_mask is not None and integer_mask[feat_idx])
                mejor_score  = -np.inf
                mejor_delta  = 0.0
                mejor_dir    = 0.0

                for threshold, delta_benigno in splits:
                    if abs(delta_benigno) < self.min_gain:
                        continue

                    distancia = abs(valor_actual - threshold)
                    score     = delta_benigno / (distancia + 1e-6)

                    if self.mode == 'targeted':
                        if score > mejor_score:
                            mejor_score = score
                            if delta_benigno > 0:
                                # Izquierda
                                if valor_actual > threshold:
                                    mejor_dir = -1.0
                                    if es_entera: mejor_delta = np.floor(threshold) - valor_actual
                                    else:         mejor_delta = threshold - valor_actual - 1e-5
                            else:
                                # Derecha
                                if valor_actual <= threshold:
                                    mejor_dir = 1.0
                                    if es_entera: mejor_delta = np.ceil(threshold) + 1 - valor_actual
                                    else:         mejor_delta = threshold - valor_actual + 1e-5
                    
                if mejor_score > -np.inf:
                    # Aplicar restricción de epsilon aquí mismo
                    if abs(mejor_delta) <= self.epsilon:
                        delta_aplicable = mejor_delta
                    else:
                        delta_aplicable = self.epsilon * mejor_dir
                    
                    scores[feat_idx] = (mejor_score, delta_aplicable)

            # Ordenar las acciones por rentabilidad
            top_features = sorted(scores.items(), key=lambda x: -x[1][0])[:self.top_k]
            # Guardamos solo la tupla (feat_idx, delta_aplicable)
            sample_actions.append([(f, d) for f, (s, d) in top_features])

        # 2. Bucle iterativo de inyección (Greedy Pursuit)
        X_adv_raw = X.copy()
        X_adv_best = X.copy()
        evaded_mask = np.zeros(n_samples, dtype=bool)

        for step in range(self.top_k):
            cambios_aplicados = False
            
            # Inyectamos el siguiente mejor movimiento a las muestras que aún no han caído
            for i in range(n_samples):
                if not evaded_mask[i] and step < len(sample_actions[i]):
                    feat_idx, delta_aplicable = sample_actions[i][step]
                    X_adv_raw[i, feat_idx] += delta_aplicable
                    cambios_aplicados = True

            if not cambios_aplicados:
                break # Ya no hay más movimientos posibles en el top_k

            # Aplicar física y grafo causal al batch entero
            if self.dc is not None:
                X_adv_phys = self.dc.to_physical_space(X_adv_raw)
                X_adv_phys = self.dc.apply_causal_graph(X_adv_phys)
                X_adv_valid = self.dc.to_scaled_space(X_adv_phys)
            else:
                X_adv_valid = X_adv_raw.copy()

            # Comprobar evasión
            preds = model.predict(X_adv_valid)
            n_queries += n_samples
            
            nuevos_evadidos = (preds == 0) & (~evaded_mask)
            
            # Guardar la versión física válida de los recién evadidos
            if nuevos_evadidos.any():
                X_adv_best[nuevos_evadidos] = X_adv_valid[nuevos_evadidos]
            
            evaded_mask = evaded_mask | (preds == 0)

            if evaded_mask.all():
                break # Evasión total conseguida

        # Para las muestras que nunca lograron evadir, devolvemos su última mutación
        no_evadidos = ~evaded_mask
        if no_evadidos.any():
            X_adv_best[no_evadidos] = X_adv_valid[no_evadidos]

        return X_adv_best, n_queries


    # ------------------------------------------------------------------
    # ANÁLISIS DE FRONTERAS (Útil para memoria del TFG)
    # ------------------------------------------------------------------
    def frontier_analysis(
        self,
        X      : np.ndarray,
        f_mask : Optional[np.ndarray] = None,
    ) -> dict:
        """
        Analiza las fronteras disponibles para el atacante.

        Retorna estadísticas sobre distancias medias, features más
        atacables y distribución de scores — contenido directo para
        la sección de análisis de vulnerabilidad de la memoria.
        """
        if f_mask is None and self.dc is not None:
            f_mask = self.dc.forward_mask

        analisis = {
            'distancia_media_por_feature' : {},
            'score_medio_por_feature'     : {},
            'n_umbrales_por_feature'      : {},
            'features_mas_cercanas'       : [],
        }

        for feat_idx, splits in self.frontier_map.items():
            if f_mask is not None and not f_mask[feat_idx]:
                continue
            if feat_idx >= X.shape[1]:
                continue

            valores    = X[:, feat_idx]
            distancias = []
            scores     = []

            for threshold, delta_benigno in splits:
                if abs(delta_benigno) < self.min_gain:
                    continue
                dist  = np.abs(valores - threshold).mean()
                score = abs(delta_benigno) / (dist + 1e-6)
                distancias.append(dist)
                scores.append(score)

            if distancias:
                analisis['distancia_media_por_feature'][feat_idx] = np.mean(distancias)
                analisis['score_medio_por_feature'][feat_idx]     = np.mean(scores)
                analisis['n_umbrales_por_feature'][feat_idx]      = len(splits)

        # Ordenar por score medio descendente
        analisis['features_mas_cercanas'] = sorted(
            analisis['score_medio_por_feature'].items(),
            key=lambda x: -x[1]
        )[:20]

        return analisis


# ===========================================================================
# SCRIPT DE VERIFICACIÓN
# ===========================================================================

if __name__ == "__main__":
    import joblib
    import numpy as np
    from src.utils.domain_constraints import DomainConstraints

    print("[-] Verificando LEAFAttack...")

    dc         = DomainConstraints.from_artifacts()
    lgbm_model = joblib.load('outputs/models/lgbm/lgbm_model.pkl')

    # LEAF necesita el modelo crudo, no el wrapper
    raw_model = lgbm_model.model if hasattr(lgbm_model, 'model') else lgbm_model

    for mode in ('untargeted', 'targeted'):
        attack = LEAFAttack(
            constraints = dc,
            model_trees = raw_model,
            epsilon     = 0.1,
            mode        = mode,
            verbose     = False,
        )
        print(f"   [✓] mode='{mode}': {attack.name}")

    print(f"\n   Features con fronteras: {len(attack.frontier_map)}")
    print(f"   Nodos totales         : {sum(len(v) for v in attack.frontier_map.values()):,}")
    print(f"\n[✓] leaf.py listo")
    print("    Uso:")
    print("      raw_model = lgbm_wrapper.model")
    print("      attack    = LEAFAttack(dc, raw_model, epsilon=0.1)")
    print("      result    = attack.run(X_attacks, y_attacks, lgbm_wrapper)")
    print("      analisis  = attack.frontier_analysis(X_attacks)")
