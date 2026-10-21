"""
两目标 NSGA-II（自包含，Deb et al. 2002 三件套）
================================================
目标（均最大化）:
  Obj1 = 收益  = IC 均值主导复合收益 − 因子相关性去冗余惩罚   ∈ [0,1]
  Obj2 = 换手  = 1 − 周换手率（低换手=高分）               ∈ [0,1]

与生产 factor_alchemy/ga 的区别（本实验版）:
  - 砍掉 regime 鲁棒目标（原 Obj2）
  - 换手率从 Obj1 惩罚项「拎出来」独立成目标
  - 收益用「IC 均值主导 + 毛Sharpe 辅助」复合口径（data_loader.ic_strength_return_score）:
    |ic_mean| 占 0.7、毛Sharpe 占 0.3，修正纯毛Sharpe 只奖励 Top30 尾部、全截面秩相关弱/负的隐患
    （🔴 不用 icir=ic_mean/ic_std，避免线性加权构造 ic_std≈0 的伪高 ICIR）
  - 染色体 = 连续权重向量 → softmax → 取 top-k(≤4) 因子做 rank 加权组合
  - 评估口径统一走 data_loader.evaluate_composite（与 GFlowNet 共用）
"""
from __future__ import annotations

import numpy as np

try:
    from .data_loader import (
        evaluate_composite,
        linear_combine,
        cross_sectional_rank,
        _cross_sectional_ic,
        ic_strength_return_score,
    )
except ImportError:  # 兼容平铺脚本运行（python nsga2_biobjective.py）
    from data_loader import (
        evaluate_composite,
        linear_combine,
        cross_sectional_rank,
        _cross_sectional_ic,
        ic_strength_return_score,
    )

MAX_FACTORS = 4
CORR_THRESHOLD = 0.30      # 平均 |因子相关| 超过此值开始惩罚
CORR_PENALTY_MAX = 0.25    # 去冗余惩罚上限


def _softmax_topk(chromo: np.ndarray, k: int = MAX_FACTORS) -> dict[int, float]:
    """染色体 → 因子权重 dict（softmax 后取 top-k 归一化）。返回 {因子索引: 权重}。"""
    w = np.exp(np.clip(chromo, -10.0, 10.0))
    w = w / w.sum()
    top = np.argsort(-w)[:k]
    sel = {int(i): float(w[i]) for i in top if w[i] > 1e-6}
    if not sel:
        return {}
    s = sum(sel.values())
    return {i: v / s for i, v in sel.items()}


def _factor_ic_matrix(factor_dict, forward_returns):
    """每个因子的截面 IC 序列 → (n_factors, n_weeks)。用于因子间去冗余相关。"""
    names = list(factor_dict.keys())
    F = forward_returns.astype(np.float64).values
    mat = []
    for n in names:
        C = factor_dict[n].astype(np.float64).values
        mat.append(_cross_sectional_ic(C, F))
    return names, np.vstack(mat)  # (n_factors, n_weeks)


class NSGA2Biobjective:
    def __init__(self, factor_dict, forward_returns, max_factors=MAX_FACTORS, seed=42):
        self.factor_dict = factor_dict
        self.forward_returns = forward_returns
        self.names = list(factor_dict.keys())
        self.n_factors = len(self.names)
        self.max_factors = max_factors
        self.rng = np.random.default_rng(seed)

        # 预计算因子间相关（用 IC 序列的 Pearson 相关作为「预测力模式」冗余代理）
        names, ic_mat = _factor_ic_matrix(factor_dict, forward_returns)
        ic_mat = np.nan_to_num(ic_mat, nan=0.0)
        corr = np.corrcoef(ic_mat)  # (n_factors, n_factors)
        corr = np.nan_to_num(corr, nan=0.0)
        self.corr_matrix = np.abs(corr)

    # ---- 染色体解码与评估 ----
    def _decode(self, chromo):
        idx_w = _softmax_topk(chromo, self.max_factors)
        return {self.names[i]: w for i, w in idx_w.items()}

    def _corr_penalty(self, selected_idx):
        if len(selected_idx) <= 1:
            return 0.0
        vals = []
        for a in range(len(selected_idx)):
            for b in range(a + 1, len(selected_idx)):
                vals.append(self.corr_matrix[selected_idx[a], selected_idx[b]])
        avg = float(np.mean(vals))
        return float(np.clip(avg / CORR_THRESHOLD * CORR_PENALTY_MAX, 0.0, CORR_PENALTY_MAX))

    def evaluate(self, chromo):
        """返回两目标 (obj1_ret, obj2_turnover)，均最大化。"""
        idx_w = _softmax_topk(chromo, self.max_factors)
        if not idx_w:
            return (0.0, 0.0)
        weights = {self.names[i]: w for i, w in idx_w.items()}
        comp = linear_combine(self.factor_dict, weights)
        r = evaluate_composite(comp, self.forward_returns)
        if r is None:
            return (0.0, 0.0)
        corr_pen = self._corr_penalty(list(idx_w.keys()))
        obj1 = float(np.clip(ic_strength_return_score(r) - corr_pen, 0.0, 1.0))
        obj2 = r["score_turnover"]
        return (obj1, obj2)

    # ---- NSGA-II 核心 ----
    @staticmethod
    def fast_non_dominated_sort(objectives):
        n = len(objectives)
        dom_count = np.zeros(n, dtype=int)
        dom_set = [[] for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                oi, oj = objectives[i], objectives[j]
                if np.all(oi >= oj) and np.any(oi > oj):
                    dom_set[i].append(j); dom_count[j] += 1
                elif np.all(oj >= oi) and np.any(oj > oi):
                    dom_set[j].append(i); dom_count[i] += 1
        fronts = []
        cur = [i for i in range(n) if dom_count[i] == 0]
        fronts.append(cur)
        fi = 0
        while fronts[fi]:
            nxt = []
            for i in fronts[fi]:
                for j in dom_set[i]:
                    dom_count[j] -= 1
                    if dom_count[j] == 0:
                        nxt.append(j)
            fi += 1
            if nxt:
                fronts.append(nxt)
            else:
                break
        return fronts

    @staticmethod
    def crowding_distance(objectives, front_idx):
        n = len(objectives)
        dist = np.zeros(n)
        if len(front_idx) <= 2:
            for i in front_idx:
                dist[i] = np.inf
            return dist
        fobj = objectives[front_idx]
        for m in range(objectives.shape[1]):
            order = np.argsort(fobj[:, m])
            rng = fobj[order[-1], m] - fobj[order[0], m]
            if rng == 0:
                continue
            dist[front_idx[order[0]]] = np.inf
            dist[front_idx[order[-1]]] = np.inf
            for k in range(1, len(front_idx) - 1):
                dist[front_idx[order[k]]] += (
                    fobj[order[k + 1], m] - fobj[order[k - 1], m]
                ) / rng
        return dist

    def _tournament(self, objectives, fronts, crowding, k=3):
        rank = np.full(len(objectives), np.inf)
        for r, f in enumerate(fronts):
            for i in f:
                rank[i] = r
        cand = self.rng.choice(len(objectives), k, replace=False)
        best = cand[0]
        for c in cand[1:]:
            if rank[c] < rank[best] or (rank[c] == rank[best] and crowding[c] > crowding[best]):
                best = c
        return best

    def _uniform_crossover(self, p1, p2):
        mask = self.rng.random(self.n_factors) < 0.5
        c1 = np.where(mask, p1, p2)
        c2 = np.where(mask, p2, p1)
        return c1, c2

    def _gaussian_mutation(self, chromo, sigma=0.3):
        return chromo + self.rng.normal(0, sigma, self.n_factors)

    def _evaluate_population(self, pop):
        return np.array([self.evaluate(c) for c in pop])

    def run(self, pop_size=120, n_generations=40, crossover_prob=0.7,
            mutation_prob=0.3, verbose=True):
        pop = [self.rng.uniform(-3, 3, self.n_factors) for _ in range(pop_size)]
        objectives = self._evaluate_population(pop)

        for gen in range(n_generations):
            objectives = np.nan_to_num(objectives, nan=0.0)
            fronts = self.fast_non_dominated_sort(objectives)
            crowding = self.crowding_distance(objectives, list(range(pop_size)))

            if verbose and fronts and fronts[0]:
                f1 = objectives[fronts[0]]
                print(f"  Gen {gen+1}/{n_generations} | Pareto={len(fronts[0])} "
                      f"ret[{f1[:,0].min():.3f},{f1[:,0].max():.3f}] "
                      f"turn[{f1[:,1].min():.3f},{f1[:,1].max():.3f}]")

            offspring = []
            while len(offspring) < pop_size:
                p1 = pop[self._tournament(objectives, fronts, crowding)]
                p2 = pop[self._tournament(objectives, fronts, crowding)]
                if self.rng.random() < crossover_prob:
                    c1, c2 = self._uniform_crossover(p1, p2)
                else:
                    c1, c2 = p1.copy(), p2.copy()
                if self.rng.random() < mutation_prob:
                    c1 = self._gaussian_mutation(c1)
                if self.rng.random() < mutation_prob:
                    c2 = self._gaussian_mutation(c2)
                offspring.append(c1)
                if len(offspring) < pop_size:
                    offspring.append(c2)

            offspring_obj = self._evaluate_population(offspring[:pop_size])
            combined_pop = pop + offspring[:pop_size]
            combined_obj = np.nan_to_num(np.vstack([objectives, offspring_obj]), nan=0.0)

            # 精英选择：逐 Front 填充，末 Front 按拥挤距离截断
            fronts = self.fast_non_dominated_sort(combined_obj)
            selected = []
            for f in fronts:
                if len(selected) + len(f) <= pop_size:
                    selected.extend(f)
                else:
                    remain = pop_size - len(selected)
                    cd = self.crowding_distance(combined_obj, f)
                    order = sorted(f, key=lambda x: -cd[x])
                    selected.extend(order[:remain])
                    break
            pop = [combined_pop[i] for i in selected]
            objectives = combined_obj[selected]

        # 最终 Pareto 前沿
        fronts = self.fast_non_dominated_sort(objectives)
        front1 = fronts[0] if fronts else []
        pareto = []
        for i in front1:
            chromo = pop[i]
            weights = self._decode(chromo)
            comp = linear_combine(self.factor_dict, weights)
            r = evaluate_composite(comp, self.forward_returns)
            pareto.append({
                "chromosome": chromo,
                "weights": weights,
                "obj_ret": float(objectives[i, 0]),
                "obj_turnover": float(objectives[i, 1]),
                "gross_sharpe": (r or {}).get("gross_sharpe", 0.0),
                "icir": (r or {}).get("icir", 0.0),
                "turnover": (r or {}).get("turnover", 1.0),
                "n_factors": len(weights),
            })
        return pareto
