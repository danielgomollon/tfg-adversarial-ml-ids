"""
src/attacks/leaf.py
================================================================
LEAF — Latent Evasion via Activation Frontiers (Momentum Rollback)

Daniel Gomollón Embid — TFG 2025-2026

═══════════════════════════════════════════════════════════════
PARADIGMA: Supervivencia Física y Momentum Tolerante
═══════════════════════════════════════════════════════════════
La evaluación de los árboles requiere que las variables crucen un umbral
para cambiar la probabilidad. Si el presupuesto Epsilon no llega al umbral
en un solo paso, un Rollback estricto (P_nueva > P_actual) descartaría el
movimiento, paralizando el ataque. 

LEAF-Momentum acepta saltos neutros (P_nueva >= P_actual) permitiendo
aprovechar el máximo presupuesto Epsilon para acercarse a la frontera y
detonar el Grafo Causal. Además, los umbrales se mapean desde el espacio
físico mediante proyecciones seguras (floor/ceil) para garantizar la
supervivencia del salto frente a las restricciones de dominio.
"""

from __future__ import annotations
import numpy as np
from typing import Optional
from src.attacks.base_attacks import BaseAttack
from src.utils.domain_constraints import DomainConstraints

class LEAFAttack(BaseAttack):
    def __init__(
        self,
        constraints : Optional[DomainConstraints],
        model_trees,
        epsilon     : float = 0.1,
        mode        : str   = 'targeted',
        top_k       : int   = 5,
        min_gain    : float = 1e-4,
        **kwargs,
    ):
        kwargs['device'] = 'cpu'
        super().__init__(constraints, epsilon=epsilon, **kwargs)

        if mode not in ('targeted', 'untargeted'):
            raise ValueError("mode debe ser 'targeted' o 'untargeted'")

        self.mode           = mode
        self.top_k          = top_k
        self.max_candidates = top_k * 3
        self.min_gain       = min_gain

        # 1. Mapeo estructural pre-calculado para velocidad extrema
        self.frontier_map_scaled = self._build_frontier_map(model_trees)

    @property
    def name(self) -> str:
        return f"LEAF (Momentum Rollback | {self.mode}, ε={self.epsilon})"

    def _build_frontier_map(self, model_trees) -> dict:
        if not hasattr(model_trees, 'dump_model'): return {}
        dump = model_trees.dump_model()
        splits_raw = []
        lr = dump.get('parameters', {}).get('learning_rate', 0.1)

        def _traverse(node, c_arbol):
            if 'split_feature' not in node: return
            f_idx   = node['split_feature']
            thr_raw = node['threshold']
            v_izq   = node['left_child'].get('leaf_value', 0.0)
            v_der   = node['right_child'].get('leaf_value', 0.0)
            delta   = (v_izq - v_der) * lr if c_arbol == 0 else -(v_izq - v_der) * lr
            
            splits_raw.append((f_idx, thr_raw, delta))
            
            _traverse(node['left_child'],  c_arbol)
            _traverse(node['right_child'], c_arbol)

        for i, tree in enumerate(dump.get('tree_info', [])):
            _traverse(tree['tree_structure'], i % dump.get('num_tree_per_iteration', 1))

        frontier_map_sc = {}
        if self.dc is not None and splits_raw:
            n_feat = len(self.dc.perturbable_mask)
            base_phys = self.dc.to_physical_space(np.zeros((1, n_feat)))[0]
            is_int = getattr(self.dc, 'integer_mask', np.zeros(n_feat, dtype=bool))
            
            mat_thr   = np.tile(base_phys, (len(splits_raw), 1))
            mat_left  = np.tile(base_phys, (len(splits_raw), 1))
            mat_right = np.tile(base_phys, (len(splits_raw), 1))
            
            for i, (f_idx, thr_raw, _) in enumerate(splits_raw):
                mat_thr[i, f_idx] = thr_raw
                # Supervivencia física estricta
                mat_left[i, f_idx]  = np.floor(thr_raw) if is_int[f_idx] else (thr_raw - 1e-3)
                mat_right[i, f_idx] = np.ceil(thr_raw) + 1.0 if is_int[f_idx] else (thr_raw + 1e-3)
                
            sc_thr   = self.dc.to_scaled_space(mat_thr)
            sc_left  = self.dc.to_scaled_space(mat_left)
            sc_right = self.dc.to_scaled_space(mat_right)
            
            for i, (f_idx, _, delta) in enumerate(splits_raw):
                if f_idx not in frontier_map_sc: frontier_map_sc[f_idx] = []
                frontier_map_sc[f_idx].append((sc_thr[i, f_idx], sc_left[i, f_idx], sc_right[i, f_idx], delta))
        
        return frontier_map_sc

    def _generate_perturbation(self, X, y, model):
        n_samples, n_features = X.shape
        n_queries = 0

        f_mask = getattr(self, 'forward_mask', None)
        if f_mask is None and self.dc is not None: f_mask = self.dc.forward_mask

        # ── 1. Ranking de Oportunidades ───────────────────────────────────────
        sample_actions = []
        for i in range(n_samples):
            scores = {}
            for feat_idx, splits in self.frontier_map_scaled.items():
                if f_mask is not None and not f_mask[feat_idx]: continue
                if feat_idx >= n_features: continue

                val_sc = X[i, feat_idx]
                best_score = -np.inf
                best_delta = 0.0

                for thr_sc, target_left_sc, target_right_sc, delta in splits:
                    if abs(delta) < self.min_gain: continue

                    target_sc = None
                    dir_ = 0.0

                    if self.mode == 'targeted':
                        if delta > 0 and val_sc > thr_sc:
                            target_sc = target_left_sc
                            dir_ = -1.0
                        elif delta <= 0 and val_sc <= thr_sc:
                            target_sc = target_right_sc
                            dir_ = 1.0
                    else:
                        if val_sc > thr_sc:
                            target_sc = target_left_sc
                            dir_ = -1.0
                        else:
                            target_sc = target_right_sc
                            dir_ = 1.0
                            
                    if target_sc is not None:
                        dist = abs(target_sc - val_sc)
                        if dist <= 0: continue # Ya hemos cruzado este umbral
                        
                        score = abs(delta) / (dist + 1e-6)
                        if score > best_score:
                            best_score = score
                            # Si podemos llegar al umbral seguro, saltamos allí. Si no, empujamos al máximo.
                            if dist <= self.epsilon:
                                best_delta = target_sc - val_sc
                            else:
                                best_delta = self.epsilon * dir_

                if best_score > -np.inf and best_delta != 0.0:
                    scores[feat_idx] = (best_score, best_delta)

            candidatos = sorted(scores.items(), key=lambda x: -x[1][0])
            sample_actions.append([(f, d) for f, (s, d) in candidatos[:self.max_candidates]])

        # ── 2. Vectorized Greedy Pursuit (Tolerancia >=) ──────────────────────
        X_curr_sc   = X.copy()
        X_adv_best  = X.copy()
        
        prob_actual = model.predict_proba(X_curr_sc)[:, 0]
        evaded_mask = prob_actual >= 0.5
        n_queries  += n_samples

        for step in range(self.max_candidates):
            if evaded_mask.all(): break

            X_cand_sc = X_curr_sc.copy()
            active_mask = np.zeros(n_samples, dtype=bool)

            for i in range(n_samples):
                if evaded_mask[i] or step >= len(sample_actions[i]): continue
                feat_idx, delta_sc = sample_actions[i][step]
                X_cand_sc[i, feat_idx] = np.clip(X_curr_sc[i, feat_idx] + delta_sc, 0.0, 1.0)
                active_mask[i] = True

            if not active_mask.any(): break

            # Física y Causal Graph
            if self.dc is not None and hasattr(self.dc, 'apply_causal_graph'):
                X_eval_phys = self.dc.to_physical_space(X_cand_sc)
                X_eval_phys = self.dc.apply_causal_graph(X_eval_phys)
                X_eval_sc   = self.dc.to_scaled_space(X_eval_phys)
            else:
                X_eval_sc   = X_cand_sc

            prob_nueva = model.predict_proba(X_eval_sc)[:, 0]
            n_queries += n_samples

            # MAGIA: Aceptamos saltos que sumen inercia y no perjudiquen (>=)
            improved = active_mask & (prob_nueva >= prob_actual)
            
            if improved.any():
                X_curr_sc[improved]   = X_eval_sc[improved]
                prob_actual[improved] = prob_nueva[improved]

                newly_evaded = improved & (prob_actual > 0.5)
                evaded_mask |= newly_evaded
                X_adv_best[newly_evaded] = X_curr_sc[newly_evaded]

        # Consolidar esfuerzos de las que no evadieron
        not_evaded = ~evaded_mask
        if not_evaded.any():
            X_adv_best[not_evaded] = X_curr_sc[not_evaded]

        return X_adv_best, n_queries

    def frontier_analysis(self, X: np.ndarray, f_mask: Optional[np.ndarray] = None) -> dict:
        if f_mask is None and self.dc is not None: f_mask = self.dc.forward_mask
        analisis = { 'distancia_media_por_feature' : {}, 'score_medio_por_feature' : {},
                     'n_umbrales_por_feature'      : {}, 'features_mas_cercanas'   : [] }
        X_phys = self.dc.to_physical_space(X) if self.dc is not None else X.copy()
        for feat_idx, splits in self.frontier_map.items():
            if f_mask is not None and not f_mask[feat_idx]: continue
            valores = X_phys[:, feat_idx]
            distancias, scores = [], []
            for threshold_raw, delta_benigno in splits:
                if abs(delta_benigno) < self.min_gain: continue
                dist  = np.abs(valores - threshold_raw).mean()
                distancias.append(dist)
                scores.append(abs(delta_benigno) / (dist + 1e-6))
            if distancias:
                analisis['distancia_media_por_feature'][feat_idx] = np.mean(distancias)
                analisis['score_medio_por_feature'][feat_idx]     = np.mean(scores)
                analisis['n_umbrales_por_feature'][feat_idx]      = len(splits)
        analisis['features_mas_cercanas'] = sorted(analisis['score_medio_por_feature'].items(), key=lambda x: -x[1])[:20]
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
