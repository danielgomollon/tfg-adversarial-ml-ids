"""
src/attacks/thorn_attack.py
================================================================
THORN — Threshold-Oriented Root Navigation (Scaled Synergy Edition)
Contribución 100% original del TFG. El depredador apex de LightGBM.

Daniel Gomollón Embid — TFG 2025-2026

═══════════════════════════════════════════════════════════════
PARADIGMA: Sinergia de Umbrales Pre-Escalada
═══════════════════════════════════════════════════════════════

1. Cascading Threshold Synergy: Agrupa umbrales idénticos que aparecen 
   en múltiples árboles y calcula su impacto destructivo global.
2. Scaled Pre-Mapping: Traduce los umbrales de la sinergia global al 
   espacio escalado durante la inicialización, evitando el cuello de 
   botella computacional en tiempo de inferencia.
3. Vectorized Evaluation: Evalúa el impacto de cruzar los umbrales a 
   nivel de lote (batch), consolidando únicamente aquellos saltos que 
   mejoran la probabilidad de evasión hacia la clase benigna.
"""

from __future__ import annotations
import numpy as np
from typing import Optional, List, Dict
from src.attacks.base_attacks import BaseAttack
from src.utils.domain_constraints import DomainConstraints

class THORNAttack(BaseAttack):
    def __init__(
        self,
        constraints: Optional[DomainConstraints],
        model_trees,
        epsilon: float = 0.1,  
        num_trees: int = 30,
        eps_tiny: float = 1e-3,
        verbose: bool = True,
        **kwargs,
    ):
        super().__init__(constraints, epsilon=epsilon, **kwargs)
        self.num_trees = num_trees
        self.eps_tiny = eps_tiny
        self.verbose = verbose
        self.synergy_ranking_raw = self._build_synergy_ranking(model_trees)

    @property
    def name(self) -> str:
        return f"THORN (Physical Synergy | N={self.num_trees}, ε={self.epsilon})"

    def _get_expected_value(self, node: dict) -> tuple[float, int]:
        if "leaf_value" in node: return node["leaf_value"], node.get("leaf_count", 1)
        lv, lc = self._get_expected_value(node["left_child"])
        rv, rc = self._get_expected_value(node["right_child"])
        t = lc + rc
        return ((lv * lc) + (rv * rc)) / t if t > 0 else 0.0, t

    def _extract_nodes_recursive(self, node: dict, tree_idx: int, nodes_list: list):
        if "split_feature" in node:
            f_idx = node["split_feature"]
            if self.forward_mask[f_idx]:
                lv, _ = self._get_expected_value(node["left_child"])
                rv, _ = self._get_expected_value(node["right_child"])
                nodes_list.append({
                    "tree_idx": tree_idx, "feat_idx": f_idx, "threshold_raw": node["threshold"],
                    "left_val": lv, "right_val": rv, "impact": abs(lv - rv)
                })
            self._extract_nodes_recursive(node["left_child"], tree_idx, nodes_list)
            self._extract_nodes_recursive(node["right_child"], tree_idx, nodes_list)

    def _build_synergy_ranking(self, booster) -> List[Dict]:
        if not hasattr(booster, 'dump_model'): return []
        raw_nodes = []
        for i, tree in enumerate(booster.dump_model()["tree_info"][:self.num_trees]):
            self._extract_nodes_recursive(tree["tree_structure"], i, raw_nodes)

        synergy = {}
        for n in raw_nodes:
            key = (n["feat_idx"], round(n["threshold_raw"], 4))
            if key not in synergy:
                synergy[key] = {"feat_idx": n["feat_idx"], "threshold_raw": n["threshold_raw"], 
                                "left_val": n["left_val"], "right_val": n["right_val"], "impact": 0.0}
            synergy[key]["impact"] += n["impact"]
            
        return sorted(synergy.values(), key=lambda x: x["impact"], reverse=True)

    def _generate_perturbation(self, X: np.ndarray, y: np.ndarray, model: object) -> tuple[np.ndarray, int]:
        if not self.synergy_ranking_raw: return X.copy(), 0

        X_adv_sc = X.copy()
        X_phys_curr = self.dc.to_physical_space(X) if self.dc else X.copy()
        
        # Uso seguro de getattr para evitar AttributeError
        integer_mask = getattr(self.dc, 'integer_mask', np.zeros(X.shape[1], dtype=bool)) if self.dc else np.zeros(X.shape[1], dtype=bool)

        current_probs = model.predict_proba(X_adv_sc)[:, 0]
        evaded_mask = current_probs > 0.5
        n_queries = len(X)

        for step, root in enumerate(self.synergy_ranking_raw):
            if evaded_mask.all(): break

            feat_idx = root["feat_idx"]
            thr_raw = root["threshold_raw"]
            target_left = root["left_val"] < root["right_val"]
            is_int = integer_mask[feat_idx]

            X_temp_phys = X_phys_curr.copy()

            # Salto en ESPACIO FÍSICO EXACTO
            if target_left:
                to_move = (~evaded_mask) & (X_temp_phys[:, feat_idx] > thr_raw)
                if not to_move.any(): continue
                X_temp_phys[to_move, feat_idx] = np.floor(thr_raw) if is_int else (thr_raw - self.eps_tiny)
            else:
                to_move = (~evaded_mask) & (X_temp_phys[:, feat_idx] <= thr_raw)
                if not to_move.any(): continue
                X_temp_phys[to_move, feat_idx] = np.ceil(thr_raw) + 1.0 if is_int else (thr_raw + self.eps_tiny)

            # Traducir a escalado para aplicar TU epsilon
            X_temp_sc = self.dc.to_scaled_space(X_temp_phys) if self.dc else X_temp_phys
            delta_sc = np.clip(X_temp_sc - X, -self.epsilon, self.epsilon)
            X_temp_sc_clipped = X + delta_sc

            # Física sobre el tensor ya recortado (Lookahead real)
            if self.dc:
                X_eval_phys = self.dc.to_physical_space(X_temp_sc_clipped)
                X_eval_phys = self.dc.apply_causal_graph(X_eval_phys)
                X_eval_sc = self.dc.to_scaled_space(X_eval_phys)
            else:
                X_eval_sc = X_temp_sc_clipped
                X_eval_phys = X_temp_phys

            new_probs = model.predict_proba(X_eval_sc)[:, 0]
            n_queries += len(X)

            improved = to_move & (new_probs > current_probs)
            
            if improved.any():
                X_adv_sc[improved] = X_temp_sc_clipped[improved]
                X_phys_curr[improved] = X_eval_phys[improved]
                current_probs[improved] = new_probs[improved]
                evaded_mask = current_probs > 0.5
                
        return X_adv_sc, n_queries