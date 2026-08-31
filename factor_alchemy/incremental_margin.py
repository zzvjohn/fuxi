# -*- coding: utf-8 -*-
"""
IncrementalMargin — v0.6 P1-1: 增量边际门禁 (MMR 升级)
=======================================================
增量边际评价 (evaluation constitution v6):
  评价对象 = 配对实验:
    control   = 当前批准组合 (基准)
    treatment = control + 候选因子
    increment = treatment - control
  一级指标 = 增量净 IR / 增量年化收益 / 回撤恶化; 只报告单因子表现不报告
  相对 control 增量的实验不能进入批准流程。

与伏羲嵌合:
  - 位置: MMR 组合层 (mmr_selector 选完组合后), 对每个"加入候选"做配对增量检验
  - 输入: control 组合日收益序列 + treatment 组合日收益序列 (等权 rank 和/积)
  - 门禁: 增量净 IR ≥ 0.10 且 回撤恶化 ≤ 0.02 (默认关, incremental_margin_enabled 开启)
  - 输出: verdict 附到候选, 不改变 MMR 排序 (仅准入门禁)

用法:
    from incremental_margin import IncrementalMarginGate
    gate = IncrementalMarginGate(enabled=True, min_net_ir=0.10)
    v = gate.evaluate(control_returns, treatment_returns)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class IncrementalVerdict:
    factor_id: str = ""
    incremental_net_ir: float = 0.0
    incremental_annual_return: float = 0.0
    drawdown_deterioration: float = 0.0       # treatment 回撤 - control 回撤 (>0 = 恶化)
    return_drawdown_efficiency_change: float = 0.0
    gate_passed: bool = True
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "factor_id": self.factor_id,
            "incremental_net_ir": round(self.incremental_net_ir, 4),
            "incremental_annual_return": round(self.incremental_annual_return, 4),
            "drawdown_deterioration": round(self.drawdown_deterioration, 4),
            "return_drawdown_efficiency_change": round(self.return_drawdown_efficiency_change, 4),
            "gate_passed": self.gate_passed,
            "reason": self.reason,
        }


def _annualized(returns: np.ndarray, periods_per_year: int) -> float:
    arr = np.asarray(returns, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0
    return float(np.prod(1.0 + arr) ** (periods_per_year / arr.size) - 1.0)


def _max_drawdown(returns: np.ndarray) -> float:
    arr = np.asarray(returns, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0
    cum = np.cumprod(1.0 + arr)
    return float(np.max(1.0 - cum / np.maximum.accumulate(cum)))


def _ir(returns: np.ndarray, periods_per_year: int) -> float:
    arr = np.asarray(returns, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 4 or arr.std(ddof=1) <= 0:
        return 0.0
    return float(arr.mean() / arr.std(ddof=1) * np.sqrt(periods_per_year))


class IncrementalMarginGate:
    """配对增量门禁: treatment 相对 control 的成本后增量。"""

    def __init__(
        self,
        *,
        enabled: bool = False,
        min_incremental_net_ir: float = 0.10,
        max_drawdown_deterioration: float = 0.02,
        min_incremental_annual_return: float = 0.005,
        periods_per_year: int = 52,
    ):
        self.enabled = enabled
        self.min_incremental_net_ir = min_incremental_net_ir
        self.max_drawdown_deterioration = max_drawdown_deterioration
        self.min_incremental_annual_return = min_incremental_annual_return
        self.periods_per_year = periods_per_year

    def evaluate(
        self,
        control_returns: np.ndarray | list,
        treatment_returns: np.ndarray | list,
        *,
        factor_id: str = "",
    ) -> IncrementalVerdict:
        v = IncrementalVerdict(factor_id=factor_id)
        c = np.asarray(control_returns, dtype=float)
        t = np.asarray(treatment_returns, dtype=float)
        n = min(len(c), len(t))
        if n < 8:
            v.gate_passed = True
            v.reason = f"样本不足 ({n}<8), 增量门禁豁免"
            return v
        c, t = c[:n], t[:n]

        ir_c = _ir(c, self.periods_per_year)
        ir_t = _ir(t, self.periods_per_year)
        v.incremental_net_ir = ir_t - ir_c
        v.incremental_annual_return = _annualized(t, self.periods_per_year) - _annualized(c, self.periods_per_year)
        dd_c, dd_t = _max_drawdown(c), _max_drawdown(t)
        v.drawdown_deterioration = dd_t - dd_c
        eff_c = ir_c / dd_c if dd_c > 1e-9 else 0.0
        eff_t = ir_t / dd_t if dd_t > 1e-9 else 0.0
        v.return_drawdown_efficiency_change = eff_t - eff_c

        if not self.enabled:
            v.gate_passed = True
            v.reason = "增量门禁未启用 (影子观测)"
            return v

        failures = []
        if v.incremental_net_ir < self.min_incremental_net_ir:
            failures.append(f"增量净IR={v.incremental_net_ir:.3f}<{self.min_incremental_net_ir}")
        if v.drawdown_deterioration > self.max_drawdown_deterioration:
            failures.append(f"回撤恶化={v.drawdown_deterioration:.1%}>{self.max_drawdown_deterioration:.1%}")
        if v.incremental_annual_return < self.min_incremental_annual_return:
            failures.append(f"增量年化={v.incremental_annual_return:.1%}<{self.min_incremental_annual_return:.1%}")
        if failures:
            v.gate_passed = False
            v.reason = "增量边际门禁失败: " + "; ".join(failures)
        else:
            v.gate_passed = True
            v.reason = (f"增量通过: ΔIR={v.incremental_net_ir:.3f}, "
                        f"Δ年化={v.incremental_annual_return:.1%}, Δ回撤={v.drawdown_deterioration:.1%}")
        return v


# ── smoke ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    rng = np.random.RandomState(11)
    noise = rng.normal(0, 0.02, 200)
    control = 0.003 + noise                    # 基准组合
    better = 0.006 + noise                     # 共享噪声 + 确定增量 0.003/周
    no_alpha = control + rng.normal(0, 0.02, 200)  # 独立噪声, 无真实增量

    gate_off = IncrementalMarginGate(enabled=False)
    print("[SMOKE] off:", gate_off.evaluate(control, better, factor_id="f1").to_dict())
    gate = IncrementalMarginGate(enabled=True)
    v1 = gate.evaluate(control, better, factor_id="alpha_add")
    v2 = gate.evaluate(control, no_alpha, factor_id="noise_add")
    print("[SMOKE] 有增量:", v1.to_dict())
    print("[SMOKE] 无增量:", v2.to_dict())
    assert v1.gate_passed, "有增量应通过"
    print("[SMOKE] OK: IncrementalMarginGate 全部路径可运行")
