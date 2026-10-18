# -*- coding: utf-8 -*-
"""
IncrementalMargin — 增量边际门禁 (MMR 升级)
========================================================
借鉴 LLM_QUANT_FACTORY evaluation.md (评价宪法 v6):
  评价对象 = 配对实验:
    control   = 当前批准组合 (基准, M13 起为**固定版本快照**, 见 control_snapshot.py)
    treatment = control + 候选因子
    increment = treatment - control
  一级指标 = 增量净 IR / 增量年化收益 / 回撤恶化; 只报告单因子表现不报告
  相对 control 增量的实验不能进入批准流程。

与伏羲嵌合:
  - 位置: MMR 组合层 (mmr_selector 选完组合后), 对每个"加入候选"做配对增量检验
  - 输入: control 组合日收益序列 + treatment 组合日收益序列 (等权 rank 和/积)
  - 门禁: 增量净 IR ≥ 0.10 且 回撤恶化 ≤ 0.02 (由 config V06_EXPERIMENTAL 控制)
  - 输出: verdict 附到候选, 不改变 MMR 排序 (仅准入门禁)

🔴 M13 (2026-09-27) 语义修复 — 五态不可压布尔:
  | evaluation_status | decision    | gate_passed | 是否剥夺 eligible | 含义                     |
  |-------------------|-------------|-------------|------------------|--------------------------|
  | valid             | pass/reject | True/False  | 仅 reject 剥夺   | 有读数, 门禁生效         |
  | data_insufficient | deferred    | **False**   | **不剥夺**       | 期数不足/零方差/空序列   |
  | not_evaluated     | deferred    | **False**   | **不剥夺**       | 门禁未启用 (影子观测)    |
  | error             | deferred    | **False**   | **不剥夺**       | 计算异常 (由调用方登记)  |
  | precheck_rejected | deferred    | **False**   | **不剥夺**       | 前置拒绝                 |
  · 旧实现在期数不足时 `gate_passed=True` + "豁免" ⇒ 措辞上"通过"、结构上放行,
    违反"缺数永不奖惩"; 现在一律 `deferred` 且 **不得**据此放行或拒绝,
    只登记进补数队列 (由 incremental_pairing 负责)。
  · 分母可见: 每个 verdict 携带 n_control / n_treatment / n_paired /
    n_dropped_nonfinite / n_dropped_to_min_len。
  · 配对口径: control/treatment 按 **共同有限样本** 对齐后再计算 (旧实现各自过滤,
    两侧有效样本可能不同 ⇒ 增量不是同口径之差)。

用法:
    from incremental_margin import IncrementalMarginGate
    gate = IncrementalMarginGate(enabled=True, min_incremental_net_ir=0.10)
    v = gate.evaluate(control_returns, treatment_returns, factor_id="cand")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

#: 五态 (全系统一致口径)
STATUS_VALID = "valid"
STATUS_DATA_INSUFFICIENT = "data_insufficient"
STATUS_NOT_EVALUATED = "not_evaluated"
STATUS_ERROR = "error"
STATUS_PRECHECK_REJECTED = "precheck_rejected"

#: 三值决策 (只有 valid 才可能 pass/reject)
DECISION_PASS = "pass"
DECISION_REJECT = "reject"
DECISION_DEFERRED = "deferred"


@dataclass
class IncrementalVerdict:
    factor_id: str = ""
    incremental_net_ir: float = 0.0
    incremental_annual_return: float = 0.0
    drawdown_deterioration: float = 0.0       # treatment 回撤 - control 回撤 (>0 = 恶化)
    return_drawdown_efficiency_change: float = 0.0
    #: 仅当 evaluation_status == "valid" 时有意义; 其余状态恒 False
    gate_passed: bool = False
    #: pass | reject | deferred  (deferred = 不奖惩, 进补数队列)
    decision: str = DECISION_DEFERRED
    evaluation_status: str = STATUS_NOT_EVALUATED
    reason: str = ""
    # ── 分母 (可见性铁律) ────────────────────────────────────────────
    n_control: int = 0
    n_treatment: int = 0
    n_paired: int = 0
    n_dropped_nonfinite: int = 0
    n_dropped_to_min_len: int = 0
    min_periods: int = 8

    def to_dict(self) -> dict:
        return {
            "factor_id": self.factor_id,
            "incremental_net_ir": round(self.incremental_net_ir, 4),
            "incremental_annual_return": round(self.incremental_annual_return, 4),
            "drawdown_deterioration": round(self.drawdown_deterioration, 4),
            "return_drawdown_efficiency_change": round(self.return_drawdown_efficiency_change, 4),
            "gate_passed": self.gate_passed,
            "decision": self.decision,
            "evaluation_status": self.evaluation_status,
            "reason": self.reason,
            "n_control": self.n_control,
            "n_treatment": self.n_treatment,
            "n_paired": self.n_paired,
            "n_dropped_nonfinite": self.n_dropped_nonfinite,
            "n_dropped_to_min_len": self.n_dropped_to_min_len,
            "min_periods": self.min_periods,
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
    """配对增量门禁: treatment 相对 control 的成本后增量 (五态输出)。"""

    def __init__(
        self,
        *,
        enabled: bool = False,
        min_incremental_net_ir: float = 0.10,
        max_drawdown_deterioration: float = 0.02,
        min_incremental_annual_return: float = 0.005,
        periods_per_year: int = 52,
        min_incremental_periods: int = 8,
    ):
        self.enabled = enabled
        self.min_incremental_net_ir = min_incremental_net_ir
        self.max_drawdown_deterioration = max_drawdown_deterioration
        self.min_incremental_annual_return = min_incremental_annual_return
        self.periods_per_year = periods_per_year
        #: 期数下限; 低于它 = data_insufficient (不豁免通过)。下限与 _ir 的 4 期
        #: 最小值取 max, 免得 min_periods<4 时静默落进 _ir 的 0.0 降级分支。
        self.min_incremental_periods = max(4, int(min_incremental_periods))

    # ── 配对准备 ──────────────────────────────────────────────────────
    def _pair(self, control_returns, treatment_returns, min_periods: int):
        """⇒ (c, t, verdict_skeleton)；不做任何放行/拒绝判定。"""
        c_all = np.asarray(control_returns, dtype=float).ravel()
        t_all = np.asarray(treatment_returns, dtype=float).ravel()
        n_c, n_t = int(c_all.size), int(t_all.size)
        n = min(n_c, n_t)
        dropped_len = int(max(n_c, n_t) - n)
        c, t = c_all[:n], t_all[:n]
        mask = np.isfinite(c) & np.isfinite(t)
        dropped_nf = int(n - int(mask.sum()))
        c, t = c[mask], t[mask]
        c = c[~np.isnan(c)]
        t = t[~np.isnan(t)]
        return c, t, {
            "n_control": n_c, "n_treatment": n_t,
            "n_paired": int(c.size),
            "n_dropped_nonfinite": dropped_nf,
            "n_dropped_to_min_len": dropped_len,
            "min_periods": int(min_periods),
        }

    def evaluate(
        self,
        control_returns: np.ndarray | list,
        treatment_returns: np.ndarray | list,
        *,
        factor_id: str = "",
        min_periods: Optional[int] = None,
    ) -> IncrementalVerdict:
        mp = int(min_periods if min_periods is not None else self.min_incremental_periods)
        mp = max(4, mp)
        c, t, meta = self._pair(control_returns, treatment_returns, mp)
        v = IncrementalVerdict(factor_id=factor_id, **meta)

        # ① 数据不足 (含空序列)：**不豁免、不放行、不拒绝**
        if v.n_paired == 0:
            v.evaluation_status = STATUS_DATA_INSUFFICIENT
            v.decision = DECISION_DEFERRED
            v.reason = (f"无可配对样本 (control={v.n_control}, treatment={v.n_treatment}) "
                        f"⇒ 不豁免, 进补数队列")
            return v
        if v.n_paired < mp:
            v.evaluation_status = STATUS_DATA_INSUFFICIENT
            v.decision = DECISION_DEFERRED
            v.reason = (f"期数不足 (n_paired={v.n_paired}<{mp}) ⇒ 不豁免, 进补数队列")
            return v

        # ② 零方差 (退化)：同属数据不足，不得当作"失败"
        if float(c.std(ddof=1)) <= 0 or float(t.std(ddof=1)) <= 0:
            v.evaluation_status = STATUS_DATA_INSUFFICIENT
            v.decision = DECISION_DEFERRED
            v.reason = "零方差序列 (不可估 IR) ⇒ 不豁免, 进补数队列"
            return v

        # ③ 读数 (无论开关是否启用都算，供影子观测)
        ir_c = _ir(c, self.periods_per_year)
        ir_t = _ir(t, self.periods_per_year)
        v.incremental_net_ir = ir_t - ir_c
        v.incremental_annual_return = (_annualized(t, self.periods_per_year)
                                       - _annualized(c, self.periods_per_year))
        dd_c, dd_t = _max_drawdown(c), _max_drawdown(t)
        v.drawdown_deterioration = dd_t - dd_c
        eff_c = ir_c / dd_c if dd_c > 1e-9 else 0.0
        eff_t = ir_t / dd_t if dd_t > 1e-9 else 0.0
        v.return_drawdown_efficiency_change = eff_t - eff_c

        # ④ 未启用 = 未评估 (不是"通过")
        if not self.enabled:
            v.evaluation_status = STATUS_NOT_EVALUATED
            v.decision = DECISION_DEFERRED
            v.gate_passed = False
            v.reason = "增量门禁未启用 (影子观测) ⇒ 不记为通过"
            return v

        # ⑤ 生效态
        failures = []
        if v.incremental_net_ir < self.min_incremental_net_ir:
            failures.append(f"增量净IR={v.incremental_net_ir:.3f}<{self.min_incremental_net_ir}")
        if v.drawdown_deterioration > self.max_drawdown_deterioration:
            failures.append(f"回撤恶化={v.drawdown_deterioration:.1%}>{self.max_drawdown_deterioration:.1%}")
        if v.incremental_annual_return < self.min_incremental_annual_return:
            failures.append(f"增量年化={v.incremental_annual_return:.1%}<{self.min_incremental_annual_return:.1%}")
        v.evaluation_status = STATUS_VALID
        if failures:
            v.gate_passed = False
            v.decision = DECISION_REJECT
            v.reason = "增量边际门禁失败: " + "; ".join(failures)
        else:
            v.gate_passed = True
            v.decision = DECISION_PASS
            v.reason = (f"增量通过: ΔIR={v.incremental_net_ir:.3f}, "
                        f"Δ年化={v.incremental_annual_return:.1%}, "
                        f"Δ回撤={v.drawdown_deterioration:.1%} "
                        f"(n_paired={v.n_paired})")
        return v


# ── smoke ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    rng = np.random.RandomState(11)
    noise = rng.normal(0, 0.02, 200)
    control = 0.003 + noise                    # 基准组合
    better = 0.006 + noise                     # 共享噪声 + 确定增量 0.003/周
    no_alpha = control + rng.normal(0, 0.02, 200)  # 独立噪声, 无真实增量

    gate_off = IncrementalMarginGate(enabled=False)
    v_off = gate_off.evaluate(control, better, factor_id="f1")
    print("[SMOKE] off:", v_off.evaluation_status, v_off.decision, v_off.gate_passed)
    assert v_off.evaluation_status == "not_evaluated" and v_off.decision == "deferred"
    assert not v_off.gate_passed, "未启用不得记为通过"

    v_short = gate_off.evaluate(control[:5], better[:5], factor_id="short")
    print("[SMOKE] short:", v_short.evaluation_status, v_short.decision, v_short.reason)
    assert v_short.evaluation_status == "data_insufficient"
    assert v_short.decision == "deferred" and not v_short.gate_passed, "期数不足不得豁免通过"

    v_zero = gate_off.evaluate(np.zeros(50), np.zeros(50), factor_id="zero_var")
    print("[SMOKE] zero-var:", v_zero.evaluation_status, v_zero.decision)
    assert v_zero.evaluation_status == "data_insufficient"

    gate = IncrementalMarginGate(enabled=True)
    v1 = gate.evaluate(control, better, factor_id="alpha_add")
    v2 = gate.evaluate(control, no_alpha, factor_id="noise_add")
    print("[SMOKE] 有增量:", v1.evaluation_status, v1.decision, v1.to_dict()["incremental_net_ir"])
    print("[SMOKE] 无增量:", v2.evaluation_status, v2.decision, v2.to_dict()["incremental_net_ir"])
    assert v1.gate_passed and v1.decision == "pass", "有增量应通过"
    assert v2.gate_passed is False and v2.decision == "reject"

    # 分母可见性: 含 NaN / 长度不等
    c_nan = list(control) + [np.nan] * 3
    t_nan = list(better)[:190]
    v3 = gate.evaluate(c_nan, t_nan, factor_id="denominator")
    print("[SMOKE] 分母:", {k: v3.to_dict()[k] for k in
                          ("n_control", "n_treatment", "n_paired",
                           "n_dropped_nonfinite", "n_dropped_to_min_len")})
    assert v3.n_paired == 190 and v3.n_dropped_to_min_len == 13
    print("[SMOKE] OK: IncrementalMarginGate 五态/分母/配对口径全部可运行")
