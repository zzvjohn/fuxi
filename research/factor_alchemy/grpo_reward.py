"""
GRPO Reward — 在线训练快速 FRI 评估

由于 GRPO 训练每轮需评估 K=4 个候选因子，完整 bootstrap FRI 太慢（~15秒/因子）。
使用简化版 FRI：跳过高开销的 bootstrap precision，保留 ICIR + persistence + novelty。
目标：~1-3秒/因子，适用于在线 RL 训练闭环。

简化 FRI = 0.35 × ICIR_norm + 0.35 × novelty + 0.30 × persistence

用法:
    from research.factor_alchemy.grpo_reward import FastFRIEvaluator
    evaluator = FastFRIEvaluator(close_df, open_df, high_df, low_df, vol_df)
    reward = evaluator.evaluate("ts_delta(close_p, 5) / close_p.shift(5)")
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sys
import time

# ── 复用 FRI 的核心函数 ──────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from research.factor_alchemy.fri import (
    _exec_formula, _cross_section_zscore,
    _compute_persistence,
)
from research.factor_alchemy.fri import (
    MIN_IC_POINTS, PERSISTENCE_WEEKS,
)


class FastFRIEvaluator:
    """快速因子评估器，用于 GRPO 在线训练。

    缓存已评估因子以加速重复评估。
    """

    def __init__(
        self,
        close_df: pd.DataFrame,
        open_df: pd.DataFrame,
        high_df: pd.DataFrame,
        low_df: pd.DataFrame,
        vol_df: pd.DataFrame,
        forward_returns: Optional[pd.DataFrame] = None,
        existing_names: Optional[List[str]] = None,
    ):
        self.close_df = close_df
        self.open_df = open_df
        self.high_df = high_df
        self.low_df = low_df
        self.vol_df = vol_df

        # 前向收益 (默认 T+1 收益)
        if forward_returns is not None:
            self.forward_returns = forward_returns
        else:
            self.forward_returns = close_df.pct_change().shift(-1)

        # 已有因子名称 (用于 novelty)
        self.existing_names = set(existing_names) if existing_names else set()

        # 缓存: {formula_hash → (reward, icir, novelty, persistence)}
        self._cache: Dict[str, Tuple[float, float, float, float]] = {}

        # 已有因子池 (lazy build)
        self._existing_pool: Dict[str, pd.DataFrame] = {}

    def evaluate(self, formula: str, verbose: bool = False) -> float:
        """评估单个因子公式，返回奖励 [0, 1]。

        如果因子已缓存，直接返回缓存值。
        """
        formula_hash = str(hash(formula.strip()))
        if formula_hash in self._cache:
            if verbose:
                r, i, n, p = self._cache[formula_hash]
                print(f"  [cache hit] reward={r:.3f} (ICIR={i:.3f} nov={n:.3f} pers={p:.3f})")
            return self._cache[formula_hash][0]

        t0 = time.time()

        # 1. 执行公式
        fv_raw = _exec_formula(formula, self.close_df, self.open_df,
                               self.high_df, self.low_df, self.vol_df)
        if fv_raw is None or fv_raw.empty:
            if verbose:
                print(f"  [FAIL] 公式执行失败")
            self._cache[formula_hash] = (0.0, 0.0, 0.0, 0.0)
            return 0.0

        # 2. 横截面标准化
        fv = _cross_section_zscore(fv_raw)

        # 3. ICIR (点估计，不做 bootstrap)
        icir = self._compute_icir_fast(fv)

        # 4. Persistence
        persistence, _ = _compute_persistence(fv)

        # 5. Novelty (与已有因子比较)
        novelty = self._compute_novelty_fast(fv)

        # 6. 简化 FRI → 奖励
        reward = 0.35 * icir + 0.35 * novelty + 0.30 * persistence

        # 复杂度惩罚: 公式越长惩罚越大
        formula_len = len(formula.strip())
        if formula_len > 200:
            reward -= 0.05
        if formula_len > 400:
            reward -= 0.10

        reward = max(0.0, min(1.0, reward))

        self._cache[formula_hash] = (reward, icir, novelty, persistence)

        if verbose:
            elapsed = time.time() - t0
            print(f"  reward={reward:.3f} (ICIR={icir:.3f} nov={novelty:.3f} "
                  f"pers={persistence:.3f}) [{elapsed:.1f}s]")

        return reward

    def evaluate_batch(
        self, formulas: List[str], verbose: bool = False
    ) -> List[float]:
        """批量评估多个因子公式"""
        return [self.evaluate(f, verbose=verbose) for f in formulas]

    def add_to_existing(self, name: str, formula: str):
        """将评估后的因子加入已有池 (更新 novelty 基准)"""
        fv_raw = _exec_formula(formula, self.close_df, self.open_df,
                               self.high_df, self.low_df, self.vol_df)
        if fv_raw is not None:
            fv = _cross_section_zscore(fv_raw)
            self._existing_pool[name] = fv
            self.existing_names.add(name)

    # ── 私有方法 ──────────────────────────────────────────

    def _compute_icir_fast(self, factor_values: pd.DataFrame) -> float:
        """快速 ICIR (点估计，不做 bootstrap)"""
        ic_series = self._compute_ic_series(factor_values)
        if ic_series is None or len(ic_series) < MIN_IC_POINTS:
            return 0.0

        mu = ic_series.mean()
        sigma = ic_series.std()
        if sigma < 1e-10:
            return 0.0

        icir = abs(mu / sigma)
        # 归一化到 [0, 1]: 假设 ICIR=1.0 为满分
        return min(icir / 1.0, 1.0)

    def _compute_ic_series(
        self, factor_values: pd.DataFrame, min_stocks: int = 30
    ) -> Optional[pd.Series]:
        """日频 Rank IC 序列"""
        ic_vals = {}
        common_dates = factor_values.index.intersection(self.forward_returns.index)

        for dt in sorted(common_dates):
            fv = factor_values.loc[dt].dropna()
            fr = self.forward_returns.loc[dt].dropna()
            common = fv.index.intersection(fr.index)
            if len(common) < min_stocks:
                continue
            ic = pd.Series(fv[common]).rank().corr(pd.Series(fr[common]).rank())
            if not np.isnan(ic):
                ic_vals[dt] = ic

        if len(ic_vals) < MIN_IC_POINTS:
            return None
        return pd.Series(ic_vals).sort_index()

    def _compute_novelty_fast(self, factor_values: pd.DataFrame) -> float:
        """快速新颖性 (与已有因子的最大 Spearman 相关)"""
        if not self._existing_pool:
            return 1.0  # 无已有因子 → 完全新颖

        max_corr = 0.0
        # 取最新一期截面计算相关
        latest = factor_values.iloc[-1].dropna()
        if len(latest) < 20:
            return 1.0

        for name, existing_fv in self._existing_pool.items():
            existing_latest = existing_fv.iloc[-1].dropna()
            common = latest.index.intersection(existing_latest.index)
            if len(common) < 20:
                continue
            corr = abs(latest[common].corr(existing_latest[common], method='spearman'))
            max_corr = max(max_corr, corr)

        return 1.0 - max_corr

    @property
    def cache_size(self) -> int:
        return len(self._cache)


# ── 便捷函数 ──────────────────────────────────────────────

def build_existing_pool_from_formulas(
    evaluator: FastFRIEvaluator,
    formulas: Dict[str, str],  # {name: formula}
) -> Dict[str, pd.DataFrame]:
    """从公式字典构建已有因子池"""
    pool = {}
    for name, formula in formulas.items():
        fv_raw = _exec_formula(formula, evaluator.close_df, evaluator.open_df,
                               evaluator.high_df, evaluator.low_df, evaluator.vol_df)
        if fv_raw is not None:
            fv = _cross_section_zscore(fv_raw)
            pool[name] = fv
    return pool
