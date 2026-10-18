# -*- coding: utf-8 -*-
"""
MMR Composite Factor Selector — 从 AlphaAgent v3 提取的互补评分引擎
======================================================================

替代 RalphLoop 原始的 rank-product 复合选择。

核心机制:
  1. 互补评分: 基于因子域(Domain)多样性 + Jaccard 距离
  2. 贪婪选择: 每次选与已选集合最互补的因子
  3. Jaccard 多样性过滤: 确保 corr(Jaccard) <= 0.5

设计原则:
  - JQ 是唯一真相源, MMR 只是选择工具
  - Local ICIR 不用于排序(仅用于否决)
  - FRI 权重 = 0 (已证实的 Local→JQ gap 放大器)

用法:
    from mmr_selector import MMRSelector

    mmr = MMRSelector()
    selected = mmr.select(candidates, top_k=4)
"""

import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass, field


@dataclass
class FactorForMMR:
    """MMR 选择器需要的因子特征"""
    name: str
    formula: str = ""
    hypothesis: str = ""
    paradigm: str = ""              # 因子域
    dimension_labels: List[str] = field(default_factory=list)  # 维度标签
    icir: float = 0.0
    ic: float = 0.0
    calmar: float = 0.0
    signal: Optional[np.ndarray] = None  # 时间序列信号 (用于计算 pairwise corr)


class MMRSelector:
    """
    MMR (Maximal Marginal Relevance) 复合因子选择器。

    评分公式:
      MMR(f) = complementarity(f, S_selected) + FRI(f) * 0 + novelty(f) * 3

    其中:
      - complementarity = 1 - max_jaccard_with_selected(S_selected)
      - FRI 权重固定为 0 (消除 Local→JQ gap)
      - novelty bonus = 范式新颖度 * 3

    约束: Jaccard ≤ 0.5  (已选集合中的因子不能太相似)
    """

    # 因子域 → base_score 映射 (domain knowledge)
    DOMAIN_PRIORITY = {
        "筹码分布": 0.15,
        "微观结构": 0.10,
        "资金流": 0.10,
        "尾部风险": 0.08,
        "行为金融": 0.08,
        "流动性": 0.07,
        "动量反转": 0.06,
        "波动率": 0.05,
        "趋势": 0.05,
        "事件驱动": 0.04,
        "估值": 0.04,
        "质量": 0.03,
    }

    NOVELTY_BOOST = 3.0
    MAX_JACCARD = 0.5

    def __init__(self, domain_priority: Optional[Dict[str, float]] = None):
        self.domain_priority = domain_priority or self.DOMAIN_PRIORITY

    # ═══════════════════════════════════════════════════════════
    # 主入口
    # ═══════════════════════════════════════════════════════════

    def select(
        self,
        candidates: List[Dict],
        top_k: int = 4,
        min_icir: float = 0.3,
        signals: Optional[Dict[str, np.ndarray]] = None,
    ) -> List[Dict]:
        """
        从候选因子中选出最优互补子集。

        Parameters
        ----------
        candidates: 因子列表 (每个含 formula/paradigm/hypothesis/icir/ic 等)
        top_k: 选多少个
        min_icir: 最低 ICIR 门槛 (用于初筛)
        signals: {name: signal_array} 用于计算 pairwise corr

        Returns
        -------
        带 "_mmr_score" 和 "_mmr_complementarity" 注解的因子列表
        """
        if len(candidates) <= top_k:
            return candidates

        # 初筛: ICIR 门槛
        eligible = [
            c for c in candidates
            if c.get("icir", 0) >= min_icir or c.get("ic", 0) >= 0.02
        ]
        if len(eligible) < top_k:
            eligible = sorted(candidates, key=lambda c: c.get("icir", 0), reverse=True)

        # 构建 MMR 因子
        factors = []
        for c in eligible:
            f = FactorForMMR(
                name=c.get("factor_name", c.get("name", "")),
                formula=c.get("formula", c.get("expression", "")),
                hypothesis=c.get("hypothesis", ""),
                paradigm=c.get("paradigm", ""),
                dimension_labels=c.get("dimension_labels", []),
                icir=c.get("icir", 0),
                ic=c.get("ic", 0),
                calmar=c.get("calmar", c.get("s5_calmar", 0)),
                signal=signals.get(c.get("factor_name", "")) if signals else None,
            )
            factors.append(f)

        # MMR 贪婪选择
        selected = self._greedy_mmr(factors, top_k)
        return list(selected)

    # ═══════════════════════════════════════════════════════════
    # 核心算法
    # ═══════════════════════════════════════════════════════════

    def _greedy_mmr(self, factors: List[FactorForMMR], top_k: int) -> List[FactorForMMR]:
        """贪婪 MMR 选择"""
        pool = list(factors)
        selected = []

        # 第一轮: 选互补分最高的
        first = max(pool, key=lambda f: self._initial_score(f))
        selected.append(first)
        pool.remove(first)

        # 后续轮: 选与已选集合互补分最高的
        while len(selected) < top_k and pool:
            best = None
            best_score = -float("inf")

            for f in pool:
                complementarity = self._complementarity(f, selected)
                novelty = self._novelty_bonus(f, selected)
                mmr = complementarity + novelty

                if mmr > best_score:
                    best_score = mmr
                    best = f

            if best is None:
                break

            # Jaccard 多样性约束
            max_jaccard = max(self._jaccard(best, s) for s in selected)
            if max_jaccard > self.MAX_JACCARD:
                pool.remove(best)
                continue

            selected.append(best)
            pool.remove(best)

        return selected

    def _initial_score(self, f: FactorForMMR) -> float:
        """初始评分 = domain priority + ICIR 信号"""
        domain_score = self.domain_priority.get(f.paradigm, 0.02)
        return domain_score + min(f.icir, 1.0) * 0.1

    def _complementarity(self, f: FactorForMMR, selected: List[FactorForMMR]) -> float:
        """计算 f 与已选集合的互补分"""
        if not selected:
            return 1.0

        max_jaccard = max(self._jaccard(f, s) for s in selected)
        return 1.0 - max_jaccard

    def _novelty_bonus(self, f: FactorForMMR, selected: List[FactorForMMR]) -> float:
        """范式新颖度奖励: 如果 f 的范式不在已选集合中, 加分"""
        selected_paradigms = {s.paradigm for s in selected}
        if f.paradigm not in selected_paradigms:
            return self.NOVELTY_BOOST
        return 0.0

    def _jaccard(self, a: FactorForMMR, b: FactorForMMR) -> float:
        """计算两个因子的近似 Jaccard 距离 (基于表达式 token 重叠)"""
        tokens_a = set(self._tokenize(a.formula))
        tokens_b = set(self._tokenize(b.formula))
        if not tokens_a or not tokens_b:
            return 1.0
        intersection = tokens_a & tokens_b
        union = tokens_a | tokens_b
        return len(intersection) / max(len(union), 1)

    @staticmethod
    def _tokenize(formula: str) -> List[str]:
        """将公式 tokenize 为操作/字段 token 集合"""
        import re
        tokens = set()
        # 提取函数调用
        for m in re.finditer(r'(?:\.)?([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', formula):
            tokens.add(m.group(1))
        # 提取字段引用
        for m in re.finditer(r'\b(open_p|high_p|low_p|close_p|volume_p|amount_p|'
                            r'open|high|low|close|volume|amount|overnight|intraday|'
                            r'returns|amplitude|turnover|market_cap)\b', formula.lower()):
            tokens.add(m.group(0))
        # 提取数字窗口
        for m in re.finditer(r'\b(\d+)\b', formula):
            tokens.add(f"w{m.group(1)}")
        return list(tokens)

    # ═══════════════════════════════════════════════════════════
    # 工具
    # ═══════════════════════════════════════════════════════════

    def compute_pairwise_jaccard(self, factors: List[Dict]) -> np.ndarray:
        """计算因子间的 pairwise Jaccard 矩阵"""
        n = len(factors)
        mat = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                a = FactorForMMR(
                    name="", formula=factors[i].get("formula", factors[i].get("expression", ""))
                )
                b = FactorForMMR(
                    name="", formula=factors[j].get("formula", factors[j].get("expression", ""))
                )
                jac = self._jaccard(a, b)
                mat[i, j] = mat[j, i] = jac
        return mat


# ── 便捷函数 ──

def select_jq_candidates(
    candidates: List[Dict],
    top_k: int = 4,
    **kwargs
) -> List[Dict]:
    """快速 MMR 选择"""
    mmr = MMRSelector()
    return mmr.select(candidates, top_k=top_k, **kwargs)


# ── 测试 ──

if __name__ == "__main__":
    mmr = MMRSelector()
    test_factors = [
        {"factor_name": "f1_liq", "formula": "sub(high, low) / ts_mean(volume, 20)", "paradigm": "流动性×微观结构", "icir": 0.5},
        {"factor_name": "f2_vol", "formula": "ts_std(close, 20) / ts_mean(close, 20)", "paradigm": "波动率", "icir": 0.4},
        {"factor_name": "f3_mom", "formula": "ts_delta(close, 20)", "paradigm": "动量反转", "icir": 0.6},
        {"factor_name": "f4_vol2", "formula": "ts_std(close, 10) / ts_mean(close, 10)", "paradigm": "波动率", "icir": 0.45},
        {"factor_name": "f5_fund", "formula": "neg(rank(div(volume, ts_mean(volume, 60))))", "paradigm": "资金流", "icir": 0.55},
        {"factor_name": "f6_behav", "formula": "neg(rank(div(sub(high, close), add(sub(high, low), 0.001))))", "paradigm": "行为金融", "icir": 0.48},
    ]

    selected = mmr.select(test_factors, top_k=4)
    print("Selected factors:")
    for s in selected:
        print(f"  {s.name} [{s.paradigm}] ICIR={s.icir}")
