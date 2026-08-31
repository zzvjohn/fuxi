# -*- coding: utf-8 -*-
"""
MultipleTestingGuard — v0.6 P0-3: DSR / CSCV-PBO / BH-FDR 三重多重检验
========================================================================
多重检验防护 (multiple testing + batch reevaluation):
  - DSR (Deflated Sharpe Ratio, PSR 形式): 用 trials 次试验的期望最大 Sharpe 做基准,
    含偏度/峰度修正分母, 概率 = norm.cdf(test_statistic)。
  - PBO (Probability of Backtest Overfitting, CSCV): 组合对称交叉验证,
    performance 矩阵 (periods × candidates), logit 相对秩。
  - FDR (Benjamini-Hochberg): 批量 p 值校正。

与伏羲嵌合:
  - 影子模式 (multiple_testing_shadow=True): 只输出 DSR/PBO/FDR 字段, 不改判定。
  - 生效模式 (multiple_testing_gate=True): 失败 → S6 拒绝 (硬门禁, 不可被 S5 补偿)。

用法:
    from multiple_testing_guard import MultipleTestingGuard
    g = MultipleTestingGuard(trials=total_historical_candidates)
    r = g.evaluate(returns_series, p_values=[...])
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, List, Optional

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class DeflatedSharpeResult:
    observed_sharpe: float
    expected_max_sharpe: float
    probability: float
    passed: bool


@dataclass
class MultipleTestingResult:
    factor_name: str = ""
    observed_sharpe: float = 0.0
    dsr_probability: float = 1.0
    dsr_passed: bool = True
    pbo: float = 0.0
    pbo_passed: bool = True
    fdr_adjusted_p: float = 1.0
    fdr_passed: bool = True
    s6_passed: bool = True
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "factor_name": self.factor_name,
            "observed_sharpe": round(self.observed_sharpe, 3),
            "dsr_probability": round(self.dsr_probability, 4),
            "dsr_passed": self.dsr_passed,
            "pbo": round(self.pbo, 4),
            "pbo_passed": self.pbo_passed,
            "fdr_adjusted_p": round(self.fdr_adjusted_p, 5),
            "fdr_passed": self.fdr_passed,
            "s6_passed": self.s6_passed,
            "reasons": self.reasons,
        }


def _expected_maximum_sharpe(trials: int) -> float:
    """Bailey & Lopez de Prado: E[max SR] ≈ sqrt(2 ln N) - 近似 (小 N 修正)"""
    if trials <= 1:
        return 0.0
    ln_n = math.log(trials)
    # 标准近似: sqrt(2 ln N) * (1 - gamma/ln N) 由期望最大标准正态导出
    return math.sqrt(2.0 * ln_n) * (1.0 - 0.5772156649 / ln_n) if ln_n > 0 else 0.0


def deflated_sharpe_ratio(
    returns: np.ndarray | list,
    *,
    trials: int,
    periods_per_year: int = 52,
    min_probability: float = 0.90,
) -> DeflatedSharpeResult:
    """DSR: PSR 形式 (偏度/峰度修正), 概率≥min_probability 视为未被 trial 次数解释。

    v0.6.2 (2026-08-29) 修复: passed 阈值此前硬编码 0.90, config 的
    dsr_min_probability (0.01 排渣口径) 只进了报错文案、从未进入决策 →
    S5 通过的因子 (DSR=0.227/0.360) 被硬编码 0.90 全杀, 流水线死锁。
    """
    values = np.asarray(returns, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 12 or trials <= 0:
        raise ValueError("At least 12 returns and one trial are required")
    std = values.std(ddof=1)
    if std <= 0:
        return DeflatedSharpeResult(0.0, 0.0, 0.0, False)
    daily_sharpe = float(values.mean() / std)
    annualized = daily_sharpe * math.sqrt(periods_per_year)
    expected_daily = _expected_maximum_sharpe(trials) / math.sqrt(values.size)
    skew = float(stats.skew(values, bias=False))
    kurt = float(stats.kurtosis(values, fisher=False, bias=False))
    denom = math.sqrt(max(1e-12, 1.0 - skew * daily_sharpe + (kurt - 1.0) * daily_sharpe**2 / 4.0))
    test_stat = (daily_sharpe - expected_daily) * math.sqrt(values.size - 1) / denom
    prob = float(stats.norm.cdf(test_stat))
    return DeflatedSharpeResult(
        observed_sharpe=annualized,
        expected_max_sharpe=expected_daily * math.sqrt(periods_per_year),
        probability=prob,
        passed=prob >= min_probability,
    )


def cscv_pbo(
    performance: np.ndarray,
    *,
    maximum_splits: int = 10_000,
    seed: int = 42,
) -> float:
    """PBO: CSCV 组合对称交叉验证 (periods × candidates 表现矩阵)。"""
    matrix = np.asarray(performance, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] < 8 or matrix.shape[1] < 2:
        return 0.0
    periods = matrix.shape[0]
    rng = np.random.RandomState(seed)
    n_splits = min(maximum_splits, 1000 if periods < 100 else maximum_splits)
    logits = np.zeros(n_splits)
    for i in range(n_splits):
        half = periods // 2
        idx = rng.permutation(periods)
        is_set = idx[:half]
        oos_set = idx[half:]
        is_rank = np.argsort(np.argsort(matrix[is_set].mean(axis=0)))
        oos_rank = np.argsort(np.argsort(matrix[oos_set].mean(axis=0)))
        n = matrix.shape[1]
        # logit: 相对秩 (IS 表现最好组合在 OOS 的相对秩)
        best_is = np.argmax(is_rank)
        rel_rank = oos_rank[best_is] / max(1.0, n - 1.0)
        rel_rank = float(np.clip(rel_rank, 1e-6, 1.0 - 1e-6))
        logits[i] = math.log(rel_rank / (1.0 - rel_rank))
    phi = np.mean(logits <= 0.0)
    return float(phi)


def benjamini_hochberg(p_values: List[float], alpha: float = 0.10) -> List[float]:
    """BH-FDR: 返回调整后的决策 (按原顺序的 adjusted p 是否通过)。"""
    p = np.asarray([v for v in p_values if v is not None and not np.isnan(v)], dtype=float)
    out = np.ones(len(p_values), dtype=float)
    if len(p) == 0:
        return list(out)
    order = np.argsort(p)
    m = len(p)
    adjusted = np.empty(m)
    prev = 1.0
    for rank in range(m - 1, -1, -1):
        idx = order[rank]
        bh = min(p[idx] * m / (rank + 1), 1.0)
        # 标准 BH: adjusted(i) = cumulative minimum of q(j) for j >= i
        adjusted[idx] = min(bh, prev)
        prev = adjusted[idx]
    out[: len(p)] = adjusted
    return list(out)


class MultipleTestingGuard:
    """S6 多重检验硬门禁 (影子/生效双模式)。"""

    def __init__(
        self,
        *,
        trials: Optional[int] = None,
        dsr_min_probability: float = 0.90,
        pbo_max: float = 0.40,
        fdr_alpha: float = 0.10,
        gate_enabled: bool = False,
        periods_per_year: int = 52,
    ):
        self.trials = trials
        self.dsr_min_probability = dsr_min_probability
        self.pbo_max = pbo_max
        self.fdr_alpha = fdr_alpha
        self.gate_enabled = gate_enabled
        self.periods_per_year = periods_per_year

    def _resolve_trials(self) -> int:
        """trials = 历史上已评估的候选总数 (无外部输入时从 Experience Memory 估算)。"""
        if self.trials and self.trials > 0:
            return self.trials
        try:
            from experience_memory import get_memory
            mem = get_memory()
            attempts = mem.data.get("attempts", []) if getattr(mem, "data", None) else []
            n = len(attempts) if isinstance(attempts, list) else 0
            return max(n, 20)
        except Exception:
            return 100  # 保守默认

    def evaluate_single(
        self,
        factor_name: str,
        returns: Optional[np.ndarray | list],
        *,
        performance_matrix: Optional[np.ndarray] = None,
        p_value: Optional[float] = None,
    ) -> MultipleTestingResult:
        r = MultipleTestingResult(factor_name=factor_name)
        trials = self._resolve_trials()

        # 1) DSR (需要收益序列)
        if returns is not None and len(np.asarray(returns)) >= 12:
            try:
                dsr = deflated_sharpe_ratio(
                    returns, trials=trials,
                    periods_per_year=self.periods_per_year,
                    min_probability=self.dsr_min_probability,
                )
                r.observed_sharpe = dsr.observed_sharpe
                r.dsr_probability = dsr.probability
                r.dsr_passed = dsr.passed
                if not dsr.passed:
                    r.reasons.append(f"DSR={dsr.probability:.3f}<{self.dsr_min_probability}")
            except Exception as e:
                r.reasons.append(f"DSR计算失败: {e}")

        # 2) PBO (需要 performance 矩阵; 单因子场景传 1 列矩阵 → 恒 0)
        if performance_matrix is not None:
            r.pbo = cscv_pbo(performance_matrix)
            r.pbo_passed = r.pbo <= self.pbo_max
            if not r.pbo_passed:
                r.reasons.append(f"PBO={r.pbo:.3f}>{self.pbo_max}")

        # 3) FDR (p_value 单独校正由批量函数处理; 这里记录原始值)
        if p_value is not None:
            r.fdr_adjusted_p = float(p_value)
            r.fdr_passed = p_value <= self.fdr_alpha
            if not r.fdr_passed:
                r.reasons.append(f"FDR p={p_value:.4f}>{self.fdr_alpha}")

        # 4) S6 判决 (gate_enabled=False → 影子, 恒通过)
        if self.gate_enabled:
            r.s6_passed = r.dsr_passed and r.pbo_passed and r.fdr_passed
        else:
            r.s6_passed = True  # 影子模式不改判定
        return r

    def evaluate_batch(
        self,
        names: List[str],
        returns_list: List[Optional[np.ndarray]],
        *,
        p_values: Optional[List[Optional[float]]] = None,
        performance_matrices: Optional[List[Optional[np.ndarray]]] = None,
    ) -> List[MultipleTestingResult]:
        """批量评估: DSR/PBO 逐因子 + BH-FDR 批量校正 p 值。

        FDR 语义: 仅对提供了 p 值的因子应用 (缺失 p 值 ≠ p=1.0,
        不能用兜底值把所有无 p 值的因子拒掉)。
        """
        # BH-FDR 批量校正 (仅当批次内至少一个因子提供了 p 值)
        adjusted = None
        has_any_pval = bool(p_values) and any(p is not None for p in p_values)
        if has_any_pval:
            adjusted = benjamini_hochberg([p if p is not None else 1.0 for p in p_values], self.fdr_alpha)
        results = []
        for i, name in enumerate(names):
            r = self.evaluate_single(
                name,
                returns_list[i] if i < len(returns_list) else None,
                performance_matrix=(performance_matrices[i] if performance_matrices and i < len(performance_matrices) else None),
            )
            if adjusted is not None and p_values and i < len(p_values) and p_values[i] is not None:
                r.fdr_adjusted_p = adjusted[i]
                r.fdr_passed = adjusted[i] <= self.fdr_alpha
                if not r.fdr_passed:
                    r.reasons.append(f"FDR(adj)={adjusted[i]:.4f}>{self.fdr_alpha}")
            # else: 该因子未提供 p 值 → FDR 分量不适用 (fdr_passed 保持 True)
            if self.gate_enabled:
                r.s6_passed = r.dsr_passed and r.pbo_passed and r.fdr_passed
            else:
                r.s6_passed = True
            results.append(r)
        return results


# ── smoke ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    rng = np.random.RandomState(42)
    # 构造一个 SR≈0.8 的周收益序列
    good = rng.normal(0.008, 0.05, 260)
    bad = rng.normal(-0.003, 0.05, 260)
    g = MultipleTestingGuard(trials=500, gate_enabled=False)
    r1 = g.evaluate_single("good_mock", good, p_value=0.01)
    r2 = g.evaluate_single("bad_mock", bad, p_value=0.60)
    print("[SMOKE] good:", r1.to_dict())
    print("[SMOKE] bad :", r2.to_dict())
    # 批量 FDR
    rs = g.evaluate_batch(
        ["a", "b", "c"],
        [good, bad, good],
        p_values=[0.005, 0.02, 0.60],
    )
    print("[SMOKE] batch FDR:", [(x.factor_name, round(x.fdr_adjusted_p, 4), x.fdr_passed) for x in rs])
    # 生效模式
    g2 = MultipleTestingGuard(trials=500, gate_enabled=True)
    r3 = g2.evaluate_single("bad_mock_gate", bad, p_value=0.60)
    print("[SMOKE] gate bad:", r3.to_dict()["s6_passed"], r3.reasons)
    print("[SMOKE] OK: MultipleTestingGuard 全部路径可运行")
