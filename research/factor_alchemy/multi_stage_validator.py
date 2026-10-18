# -*- coding: utf-8 -*-
"""
Multi-Stage Validator — 对标 FactorMiner Section 3.2 的多阶段因子验证管线
================================================================================

对标 FactorMiner 的四阶段验证:
  Stage 1: Fast IC Screening       — 快速排渣 (<1min/batch)
  Stage 2: Correlation Check       — 库内去重 (vs 现有因子池)
  Stage 3: Batch Deduplication     — 批次内去重
  Stage 4: Full OOS + Replacement  — 完整验证 + 替换检查

设计原则:
  - 只在 Stage 1-3 通过后才提交 JQ（节省配额）
  - Stage 3+ 的候选才值得跑完整 JQ 回测
  - 可独立运行，也可集成到 Ralph Loop

用法:
    from multi_stage_validator import MultiStageValidator

    validator = MultiStageValidator(data_dir=Path("data"))
    results = validator.validate(batch_candidates)

    for stage_name, passed in results.items():
        print(f"{stage_name}: {len(passed)} passed")
"""

import json
import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set, Any
from dataclasses import dataclass, field
from datetime import datetime


# ── 默认路径 ──────────────────────────────────────────────
DATA_DIR = Path(__file__).parent.parent.parent / "data"


@dataclass
class ValidationResult:
    """单个候选因子的验证结果"""
    factor_name: str
    formula: str = ""
    paradigm: str = ""
    hypothesis: str = ""

    # Stage 1
    s1_ic: float = 0.0           # Fast IC
    s1_icir: float = 0.0
    s1_passed: bool = False
    s1_reason: str = ""
    # R2 (2026-09-22): measurement state is independent from pass/fail.
    # valid | not_evaluated | data_insufficient | error
    s1_eval_status: str = "not_evaluated"
    # v0.7 频率对称: S1 裁决口径标注 ("daily" | "weekly"), 供归因/审计
    s1_freq: str = "daily"
    # 2026-09-12 (W2): S1 IC 有效天数 (样本量守卫证据) — -1 = 上游未提供
    s1_n_days: int = -1
    s1_low_sample: bool = False   # 是否因样本不足被判 reserve/reject
    # W3 (2026-09-12): 已证伪方向硬屏蔽命中 (shadow 只标 / enforce 硬拒)
    s1_forbidden_hit: bool = False
    s1_forbidden_reason: str = ""
    # M10 (2026-09-27): AST 结构重复命中 (shadow 只标 / enforce 硬拒)
    s1_struct_dup_hit: bool = False
    s1_struct_dup_reason: str = ""
    # W6 (2026-09-12): S2–S4 失效归因 (只标不裁决, 用于诚实归因)
    s2_corr_source: str = ""          # library_recompute | signal_recompute | precomputed | default_zero
    s3_signal_missing: bool = False   # 批次内相关性因 signal 未接线而无法计算
    s3_skeleton_dup: bool = False     # 同批内骨架重复 (shadow, 免费去重信号)
    s4_lib_meta_missing: bool = False # 库元数据缺失导致替换检查空转

    # Stage 2
    s2_max_corr: float = 0.0     # 与库的最大相关系数
    s2_max_corr_factor: str = ""
    s2_passed: bool = False
    s2_reason: str = ""
    # M8 (2026-09-26): S2 五态状态 (与 s1/s4/s5/s6 对齐)。
    # valid = 相关性已计算; data_insufficient = max_corr 缺省 (上游无法求值)。
    # 🔴 passed=True + eval_status=data_insufficient = "放行但未评估", 不得读作"相关低"。
    s2_eval_status: str = "not_evaluated"

    # Stage 3
    s3_max_batch_corr: float = 0.0  # 批次内最大相关系数
    s3_passed: bool = False
    s3_reason: str = ""
    # M8 (2026-09-26): S3 五态状态 + 配对分母。
    # valid = 至少一对真实相关性已计算; not_evaluated = 全部未计算 (signal 不可用);
    # data_insufficient = 部分对未计算 (分母含缺数, 不得声称"无冗余")。
    s3_eval_status: str = "not_evaluated"
    s3_pairs_computed: int = 0      # 批内已计算的相关性对数 (逐候选)
    s3_pairs_missing: int = 0       # 批内因 signal 缺失/不可比而未能计算的对数

    # Stage 4
    s4_oos_ic: float = 0.0
    s4_oos_icir: float = 0.0
    s4_replacement: bool = False    # 是否替换了库中已有因子
    s4_replaced_factor: str = ""
    s4_passed: bool = False
    s4_reason: str = ""
    s4_eval_status: str = "not_evaluated"
    s4_decision_basis: str = "not_evaluated"
    # S4 伪 OOS 影子标记 (v0.6): candidate 无 oos_ic 字段时回退到 S1 IC
    # 该回退意味着 S4 与 S1 判定重复 — 只标记, 不改判定 (rollback 安全)
    s4_fallback_to_s1ic: bool = False
    # M8 (2026-09-26): S4 替换检查状态 (与 OOS 状态分离计分母)。
    # valid = 与库元数据真实比较过; not_evaluated = 无需替换/无对应因子;
    # data_insufficient = S2 相关性未计算 ⇒ 无从判断是否需要替换。
    s4_replacement_eval_status: str = "not_evaluated"

    # Stage 5 (v0.3 新增): 联合正向过滤 + Calmar
    s5_year1_excess: float = 0.0    # 年度1超额收益
    s5_year2_excess: float = 0.0    # 年度2超额收益
    s5_calmar: float = 0.0          # Calmar比率
    s5_passed: bool = False
    s5_reason: str = ""
    s5_eval_status: str = "not_evaluated"
    s5_marking_diagnostics: Any = None
    s5_economic_readings_status: str = "not_evaluated"
    # ── W5 (2026-09-12): S5 口径留痕 (7 字段, 诊断用, 不影响裁决) ──
    # 背景: 换口径会换通过集合 (实测 ex25 均值变动 −0.1421), 但 208 个
    #   ralph_loop_result_*.json 中这些字段 0 个出现 ⇒ "不可比"只能靠人肉考古。
    s5_calmar_mode: str = ""            # legacy | common_window | vol_scaled
    s5_calmar_2025: float = 0.0
    s5_calmar_2026: float = 0.0
    s5_calmar_2025_full: float = 0.0    # 全长对照 (量化截断效应)
    s5_calmar_2026_full: float = 0.0
    s5_window_days_2025: int = 0
    s5_window_days_2026: int = 0
    s5_window_matched: Any = None       # false ⇒ 窗口不可比, 须告警
    s5_calmar_threshold_effective: float = 1.0

    # Stage 6 (v0.6 实验): 多重检验门禁 DSR/PBO/BH-FDR
    # s6_passed 仅表示有效计算通过；影子模式的策略放行另记。
    s6_computed: bool = False
    s6_eval_status: str = "not_evaluated"
    s6_policy_allows: bool = True
    s6_dsr_prob: float = 0.0        # Deflated Sharpe 概率
    s6_pbo: float = 0.0             # CSCV-PBO
    s6_fdr_adj_p: float = 0.0       # BH-FDR 校正 p 值
    s6_passed: bool = False
    s6_reason: str = ""

    # Overall
    final_grade: str = ""          # A/B/C/D
    total_stages_passed: int = 0
    eligible_for_jq: bool = False  # 是否提交 JQ
    # v0.6: 序列门语义 (S1→S2→... 连续通过的最深阶段; 1=只过S1, 5=过S1-S5)
    # 与 total_stages_passed(计数语义)并存, 供"评价宪法"方向迁移时对照
    gate_level: int = 0

    # P-20260828-004: S5 IC bootstrap 置信区间 (影子模式: 只输出不改变判定)
    ic_boot_mean: float = 0.0
    ic_ci95_low: float = 0.0
    ic_ci95_high: float = 0.0
    ic_boot_n: int = 0

    # P-20260828-001: Δlag 滞后衰减 (影子模式: 只输出不改变判定)
    s1_ic_lag1: float = 0.0      # 信号右移一天后的 RankIC
    delta_lag: float = 0.0       # 当期IC - lag1IC; >0 = 延迟一天后预测力衰减
    high_decay_risk: bool = False

    def to_dict(self) -> dict:
        return {
            "factor_name": self.factor_name,
            "formula": self.formula[:200],
            "s1": {"ic": self.s1_ic, "icir": self.s1_icir, "passed": self.s1_passed, "reason": self.s1_reason,
                   "eval_status": self.s1_eval_status,
                   "freq": self.s1_freq, "n_days": self.s1_n_days, "low_sample": self.s1_low_sample,
                   "forbidden_hit": self.s1_forbidden_hit,
                   "forbidden_reason": self.s1_forbidden_reason,
                   "struct_dup_hit": self.s1_struct_dup_hit,
                   "struct_dup_reason": self.s1_struct_dup_reason},
            "s2": {"max_corr": self.s2_max_corr, "max_corr_factor": self.s2_max_corr_factor, "passed": self.s2_passed, "reason": self.s2_reason,
                   "corr_source": self.s2_corr_source, "eval_status": self.s2_eval_status},
            "s3": {"max_batch_corr": self.s3_max_batch_corr, "passed": self.s3_passed, "reason": self.s3_reason,
                   "signal_missing": self.s3_signal_missing, "skeleton_dup": self.s3_skeleton_dup,
                   "eval_status": self.s3_eval_status,
                   "pairs_computed": self.s3_pairs_computed,
                   "pairs_missing": self.s3_pairs_missing},
            "s4": {"oos_ic": self.s4_oos_ic, "oos_icir": self.s4_oos_icir, "replacement": self.s4_replacement, "passed": self.s4_passed, "reason": self.s4_reason, "fallback_to_s1ic": self.s4_fallback_to_s1ic,
                   "eval_status": self.s4_eval_status, "decision_basis": self.s4_decision_basis,
                   "lib_meta_missing": self.s4_lib_meta_missing,
                   "replacement_eval_status": self.s4_replacement_eval_status},
            "s5": {"year1_excess": self.s5_year1_excess, "year2_excess": self.s5_year2_excess, "calmar": self.s5_calmar, "passed": self.s5_passed, "reason": self.s5_reason,
                   "eval_status": self.s5_eval_status,
                   "marking_diagnostics": self.s5_marking_diagnostics,
                   "economic_readings_status": self.s5_economic_readings_status,
                   # W5 (2026-09-12): 口径留痕 — 不影响裁决, 仅诊断
                   "calmar_mode": self.s5_calmar_mode,
                   "calmar_2025": self.s5_calmar_2025, "calmar_2026": self.s5_calmar_2026,
                   "calmar_2025_full": self.s5_calmar_2025_full,
                   "calmar_2026_full": self.s5_calmar_2026_full,
                   "window_days_2025": self.s5_window_days_2025,
                   "window_days_2026": self.s5_window_days_2026,
                   "window_matched": self.s5_window_matched,
                   "calmar_threshold_effective": self.s5_calmar_threshold_effective},
            "s6": {"computed": self.s6_computed, "eval_status": self.s6_eval_status,
                   "policy_allows": self.s6_policy_allows,
                   "dsr_prob": round(self.s6_dsr_prob, 4), "pbo": round(self.s6_pbo, 4), "fdr_adj_p": round(self.s6_fdr_adj_p, 4), "passed": self.s6_passed, "reason": self.s6_reason},
            "ic_boot": {"mean": round(self.ic_boot_mean, 5), "ci95_low": round(self.ic_ci95_low, 5), "ci95_high": round(self.ic_ci95_high, 5), "n_obs": self.ic_boot_n},
            "delta_lag": {"ic_lag1": round(self.s1_ic_lag1, 5), "delta_lag": round(self.delta_lag, 5), "high_decay_risk": self.high_decay_risk},
            "final_grade": self.final_grade,
            "gate_level": self.gate_level,
            "eligible_for_jq": self.eligible_for_jq,
        }


def block_bootstrap_ic_ci(
    ic_series,
    n_boot: int = 200,
    block_days: int = 21,
    seed: int = 42,
) -> dict:
    """
    P-20260828-004: 月度块 bootstrap 95% CI (影子模式, 只输出不改判定)。

    对标 arXiv 2026-08-03 'fitness reliability' 组件: IC 单点估计的
    不确定性量化。月度块 (block_days≈21交易日) 重抽样保留 IC 序列的
    月度相关性结构, 避免 i.i.d. bootstrap 低估方差。

    Returns:
        {"mean": float, "ci95_low": float, "ci95_high": float, "n_obs": int}
    """
    vals = np.asarray([v for v in ic_series if v is not None and not np.isnan(v)], dtype=float)
    out = {"mean": 0.0, "ci95_low": 0.0, "ci95_high": 0.0, "n_obs": 0}
    if len(vals) < 40:
        return out
    out["n_obs"] = len(vals)

    n = len(vals)
    block = max(1, min(block_days, n))
    n_blocks = int(np.ceil(n / block))
    # 起始位置列表: 0, block, 2*block, ...
    starts = np.arange(0, n, block)

    rng = np.random.RandomState(seed)
    boot_means = np.empty(n_boot)
    for b in range(n_boot):
        # 抽取 n_blocks 个块 (可重复)
        idx = rng.randint(0, len(starts), size=n_blocks)
        sample = np.concatenate([
            vals[s : min(s + block, n)] for s in starts[idx]
        ])
        boot_means[b] = np.mean(sample)

    out["mean"] = float(np.mean(boot_means))
    out["ci95_low"] = float(np.percentile(boot_means, 2.5))
    out["ci95_high"] = float(np.percentile(boot_means, 97.5))
    return out


class MultiStageValidator:
    """多阶段因子验证器 — 对标 FactorMiner 的 4-stage 管线 (v0.3: +S5 联合过滤)"""

    def __init__(
        self,
        data_dir: Path = DATA_DIR,
        ic_threshold: float = 0.02,
        # v0.6.1 (2026-08-29): icir 门槛 0.3 → 0.1 口径校准。
        # 0.3 是周频/小池口径标准; S1 现场计算为日频全市场口径 (fwd20),
        # 实测 JQ 验证过的王者因子现场 ICIR 最高仅 0.189 → 0.3 门槛系统性堵死
        # (8-26/27/28 连续三天 s1_passed=0, 流水线零产出)。0.1 与 Forge 自身
        # 合格线 (icir_threshold=0.1) 对齐, S1 恢复"排渣预筛"定位, 最终裁决仍在 S5/S6。
        icir_threshold: float = 0.1,
        corr_threshold: float = 0.5,
        batch_corr_threshold: float = 0.7,
        replacement_k: float = 1.1,   # 新因子ICIR ≥ k × 旧因子ICIR才替换
        # S5 参数 (v0.3 新增)
        calmar_threshold: float = 1.0,       # Calmar比率最低阈值
        year1_label: str = "2025",           # 年度1标签
        year2_label: str = "2026",           # 年度2标签
        # v0.7.12 (W2, 2026-09-12): S1 样本量守卫 — IC/ICIR 的统计底线。
        # 依据: 实测 120 条候选公式 n_days 中位 158 / p10 118, 仅 1.7% <60,
        #   故 60 是"只挡真短窗"的安全阈; 但 n=27 这类窗口的 ICIR 无统计意义
        #   (原报告的 n=5 案例已证伪, 真漏损是"求值失败 n_days=0 被误标为无IC")。
        # 开关: env FUXI_S1_MIN_IC_DAYS (0 = 关闭守卫, 仅打印归因)。
        min_ic_days: Optional[int] = None,
    ):
        self.data_dir = data_dir
        self.ic_threshold = ic_threshold
        self.icir_threshold = icir_threshold
        self.corr_threshold = corr_threshold
        self.batch_corr_threshold = batch_corr_threshold
        self.replacement_k = replacement_k
        self.calmar_threshold = calmar_threshold
        self.year1_label = year1_label
        self.year2_label = year2_label
        # W2: 样本量守卫阈值 (None → 读 env, 缺省 60)
        if min_ic_days is None:
            try:
                import os as _os
                min_ic_days = int(_os.environ.get("FUXI_S1_MIN_IC_DAYS", "60"))
            except Exception:
                min_ic_days = 60
        self.min_ic_days = int(min_ic_days)
        self._library_signals: Optional[Dict[str, np.ndarray]] = None
        # v0.6: S6 门禁生效状态 (stage6 内从 config 懒加载后回写, 供 summary 用)
        self._s6_gate_enabled = False

    # ── Stage 1: Fast IC Screening ────────────────────────

    def stage1_fast_ic(
        self,
        candidates: List[Dict],
        results: List[ValidationResult],
    ) -> List[ValidationResult]:
        """
        Stage 1: 快速 IC 筛选。

        对标 FactorMiner: Stage 1 Fast IC Screening (IC ≥ τ_ic)

        使用候选因子携带的 IC/ICIR 值做快速过滤。
        如果候选没有预计算的 IC，则标记为需要通过（跳到 Stage 2 计算）。

        v0.7 频率对称 (2026-08-29): natural_freq=="weekly" 的候选在
        V07_DUAL_LANE 开启时用周频字段 (weekly_ic/weekly_icir) 对 τ_w 门槛裁决,
        一因子一通道 (XOR, 不重复日频裁决); V07 关闭时全部走现状日频口径。
        """
        # W2: 每次调用重置计数 (避免跨轮累加)
        self._s1_low_sample = 0
        self._s1_eval_failed = 0
        self._s1_n_days_seen = 0
        # W3: 同上
        self._s1_forbidden_hits = 0
        self._s1_forbidden_rejected = 0
        # M10 (2026-09-27): 同上
        self._s1_struct_dup_hits = 0
        self._s1_struct_dup_rejected = 0
        # W6: S2–S4 空转归因计数
        self._s2_default_zero = 0
        self._s3_signal_missing = 0
        self._s3_skel_dup = 0
        self._s3_skel_rejected = 0
        self._s4_lib_meta_missing = 0
        self._s5_window_unmatched = 0
        # v0.7: 周频门槛懒加载 (V07 开关, 默认关闭 = 零回归)
        _v07_on = False
        _w_ic_min, _w_icir_min = 0.02, 0.1
        try:
            from config import V07_DUAL_LANE
            _v07_on = bool(V07_DUAL_LANE.get("enabled", False))
            if _v07_on:
                _w_ic_min = float(V07_DUAL_LANE.get("weekly_ic_min", 0.005))
                # v0.7 P4 (2026-08-29): τ_w 动态读取 (校准集 effective > config 基线)
                try:
                    from lane_calibration import get_effective_tau_w
                    _w_icir_min = get_effective_tau_w()
                except Exception:
                    _w_icir_min = float(V07_DUAL_LANE.get("weekly_icir_threshold", 0.15))
        except Exception:
            pass

        for i, cand in enumerate(candidates):
            # ── v0.7: 频率分流 ──
            _freq = ""
            if _v07_on:
                try:
                    from weekly_lane import infer_natural_freq
                    _freq = infer_natural_freq(cand)
                except Exception:
                    _freq = ""
            if _freq == "weekly":
                # ── P-20260907-006 (2026-09-11 转正): 符号感知 ──
                # 原 abs() 无方向意识, 负向强信号被当正向通过 → 负向因子按原方向
                # 回测必灭 (gp_evo_001/R3 事故)。改为保留符号, 阈值为正 → 负向自然落榜。
                ic = float(cand.get("weekly_ic", 0) or 0)
                icir = float(cand.get("weekly_icir", 0) or 0)
                results[i].s1_freq = "weekly"
                # W2: 周频样本量透传 (weekly_n = 周频有效周数)
                _wn = cand.get("weekly_n", None)
                if isinstance(_wn, (int, float)):
                    results[i].s1_n_days = int(_wn)
                if abs(ic) < 1e-8 and abs(icir) < 1e-8:
                    results[i].s1_passed = False
                    results[i].s1_eval_status = "data_insufficient"
                    results[i].s1_reason = "❌ 无周频 IC 数据 — Stage1 拒绝 (weekly lane)"
                    continue
                results[i].s1_ic = ic
                results[i].s1_icir = icir
                results[i].s1_eval_status = "valid"
                if ic >= _w_ic_min and icir >= _w_icir_min:
                    results[i].s1_passed = True
                    results[i].s1_reason = f"[周频] IC={ic:.4f}, ICIR={icir:.3f} ✅"
                else:
                    fails = []
                    if ic < _w_ic_min:
                        fails.append(f"周频 IC={ic:.4f}<{_w_ic_min}"
                                     + (" (负向, 符号守卫)" if ic < 0 else ""))
                    if icir < _w_icir_min:
                        fails.append(f"周频 ICIR={icir:.3f}<{_w_icir_min}"
                                     + (" (负向, 符号守卫)" if icir < 0 else ""))
                    results[i].s1_passed = False
                    results[i].s1_reason = f"❌ {'; '.join(fails)}"
                continue

            # ── W3 (2026-09-12): 已证伪方向硬屏蔽 (S1 前置门, 由 ralph_loop 打标) ──
            # shadow 模式下只标注不裁决 (零回归); enforce 才真拒。
            _fh = cand.get("_forbidden_hit")
            if _fh and _fh.get("blocked"):
                results[i].s1_forbidden_hit = True
                results[i].s1_forbidden_reason = _fh.get("reason", "")
                self._s1_forbidden_hits = getattr(self, "_s1_forbidden_hits", 0) + 1
                if _fh.get("enforced"):
                    results[i].s1_passed = False
                    results[i].s1_eval_status = "not_evaluated"
                    self._s1_forbidden_rejected = getattr(
                        self, "_s1_forbidden_rejected", 0) + 1
                    results[i].s1_reason = f"❌ 已证伪方向硬屏蔽 — {_fh.get('reason','')}"
                    continue

            # ── M10 (2026-09-27): AST 结构重复硬拒 (S1 前置门, 由 ralph_loop 打标) ──
            # shadow 只标注不裁决 (零回归); enforce (would_reject) 才真拒。
            # 与 W3 forbidden 同模式: 打标在 ralph_loop _phase_evaluate (符号守卫后)。
            _sdh = cand.get("_struct_dup_hit")
            if _sdh:
                results[i].s1_struct_dup_hit = True
                results[i].s1_struct_dup_reason = _sdh.get("reason", "")
                self._s1_struct_dup_hits = getattr(self, "_s1_struct_dup_hits", 0) + 1
                if _sdh.get("would_reject"):
                    results[i].s1_passed = False
                    results[i].s1_eval_status = "not_evaluated"
                    self._s1_struct_dup_rejected = getattr(
                        self, "_s1_struct_dup_rejected", 0) + 1
                    results[i].s1_reason = f"❌ AST 结构重复硬拒 — {_sdh.get('reason','')}"
                    continue

            # ── daily lane (现状路径, 零改动) ──
            results[i].s1_freq = "daily"
            # ── P-20260907-006 (2026-09-11 转正): 符号感知 (原 abs 同缺陷) ──
            ic = float(cand.get("ic", cand.get("daily_ic", 0)) or 0)
            icir = float(cand.get("icir", cand.get("daily_icir", 0)) or 0)

            # ── W2 (2026-09-12): 样本量透传 + 求值失败归因区分 ──
            _nd = cand.get("n_days", None)
            _nd = int(_nd) if isinstance(_nd, (int, float)) and _nd is not None else None
            _eval_failed = bool(cand.get("_ic_eval_failed", False))
            if _nd is not None:
                results[i].s1_n_days = _nd
            self._s1_n_days_seen = getattr(self, "_s1_n_days_seen", 0) + 1

            if abs(ic) < 1e-8 and abs(icir) < 1e-8:
                # 无预计算 IC — 拒绝（没有real数据不能信任）
                results[i].s1_passed = False
                if _eval_failed or _nd == 0:
                    results[i].s1_eval_status = "error"
                    # W2 归因: 公式**求值失败**(语法/字段/rolling 不支持),
                    # 不等于"上游没算 IC"。原实现两者共用同一句 reason,
                    # 污染了 S1 拒绝归因 (fd 审计 C 类"流程故障"即此坑)。
                    results[i].s1_low_sample = True
                    self._s1_eval_failed = getattr(self, "_s1_eval_failed", 0) + 1
                    results[i].s1_reason = (
                        "❌ 公式求值失败（IC 无法计算, n_days=0）— Stage1拒绝 "
                        "[W2: 归因=流程故障, 非因子无效]")
                else:
                    results[i].s1_eval_status = "not_evaluated"
                    results[i].s1_reason = "❌ 无IC数据（status=reserve/未评估）— Stage1拒绝"
                continue

            # ── W2: 样本量守卫 (n_days 已知且低于底线 ⇒ ICIR 不可信) ──
            if (self.min_ic_days > 0 and _nd is not None
                    and 0 < _nd < self.min_ic_days):
                results[i].s1_passed = False
                results[i].s1_low_sample = True
                results[i].s1_eval_status = "data_insufficient"
                self._s1_low_sample = getattr(self, "_s1_low_sample", 0) + 1
                results[i].s1_reason = (
                    f"❌ 样本不足 n_days={_nd}<{self.min_ic_days} "
                    f"(IC={ic:.4f}, ICIR={icir:.3f}) — Stage1拒绝 [W2: ICIR 无统计意义]")
                continue

            results[i].s1_ic = ic
            results[i].s1_icir = icir
            results[i].s1_eval_status = "valid"

            # P-20260828-001: Δlag 滞后衰减 (影子: 只标记不改判定)
            lag1 = float(cand.get("ic_lag1", 0) or 0)
            dl = float(cand.get("delta_lag", 0) or 0)
            results[i].s1_ic_lag1 = lag1
            results[i].delta_lag = dl
            # 判据 (提案阈值): Δlag > 0.015 或 Δlag/|IC| > 50% → high_decay_risk
            if dl > 0.015 or (ic > 1e-8 and dl / ic > 0.5):
                results[i].high_decay_risk = True

            if ic >= self.ic_threshold and icir >= self.icir_threshold:
                results[i].s1_passed = True
                results[i].s1_reason = f"IC={ic:.4f}, ICIR={icir:.3f} ✅"
            else:
                fails = []
                if ic < self.ic_threshold:
                    fails.append(f"IC={ic:.4f}<{self.ic_threshold}"
                                 + (" (负向, 符号守卫)" if ic < 0 else ""))
                if icir < self.icir_threshold:
                    fails.append(f"ICIR={icir:.3f}<{self.icir_threshold}"
                                 + (" (负向, 符号守卫)" if icir < 0 else ""))
                results[i].s1_passed = False
                results[i].s1_reason = f"❌ {'; '.join(fails)}"

        return results

    # ── Stage 2: Correlation Check ────────────────────────

    def stage2_correlation(
        self,
        candidates: List[Dict],
        results: List[ValidationResult],
        library_signals: Optional[Dict[str, np.ndarray]] = None,
    ) -> List[ValidationResult]:
        """
        Stage 2: 与因子库的相关性检查。

        对标 FactorMiner: Stage 2 Correlation Check (ρ ≤ θ)

        检查候选因子与已有因子库的相关性。如果库信号不可用，
        则使用候选携带的 max_corr 信息。
        """
        for i, cand in enumerate(candidates):
            if not results[i].s1_passed:
                continue

            # v0.6.1: 种子重检豁免 — 种子原式本就在库中 (周期性裁决, 非新因子发现),
            # 与库模板 rank 相关=1.0 是自比较, 不应被 S2 去重误伤
            if cand.get("_seed_recheck"):
                results[i].s2_passed = True
                results[i].s2_reason = "种子重检豁免 (周期性裁决, 非新因子发现)"
                continue

            # 优先使用候选自带的 max_corr
            max_corr = float(cand.get("max_corr", cand.get("max_correlation", 0)) or 0)
            max_corr_factor = cand.get("max_corr_factor", cand.get("max_correlation_factor", ""))
            _src = "precomputed" if abs(max_corr) > 1e-8 else "default_zero"

            # 如果库信号可用，重新计算
            # W6 审计 (2026-09-12): 此分支双重失效 —
            #   (1) `library_signals` 全流水线从未传入 (run_v4_pipeline 无此参数) ⇒ 恒 None;
            #   (2) `cand["signal"]` 全代码库无任何写入点 (仅 S2/S3/behavior_homogeneity 读) ⇒ 恒 None。
            # 但 S2 **并未失效**: 上游 `_phase_evaluate` 已用
            #   `ic_comp.compute_max_corr_vs_library_batch` 把真实 max_corr 写进候选
            #   ⇒ S2 走 precomputed 路径正常工作 (实测 152→94, 拒 38%)。
            # P-20260917-025 F1: `cand["signal"]` 是 **ndarray** ⇒ 不能直接取真值
            #   (`if ... and cand.get("signal")` 会抛
            #    "truth value of an array with more than one element is ambiguous")。
            #   🔴 该分支在 P-024 接线修复**之前从未执行过** (signal 恒 None ⇒ 短路),
            #      接线一修好立即崩 ⇒ 潜伏消费点缺陷, 非接线本身的问题。
            #   同时补形状防御 (跨长度直接广播爆炸) 与有限性防御 (常数列 corr=nan)。
            _cand_sig = cand.get("signal")
            if library_signals and _cand_sig is not None:
                cand_signal = np.asarray(_cand_sig, dtype=float).ravel()
                if cand_signal.size:
                    for lib_name, lib_signal in library_signals.items():
                        _ls = np.asarray(lib_signal, dtype=float).ravel()
                        if _ls.shape != cand_signal.shape:
                            continue  # 形状不一致 ⇒ 不可比, 跳过 (不猜对齐)
                        valid = ~(np.isnan(cand_signal) | np.isnan(_ls))
                        if valid.sum() < 20:
                            continue
                        if np.unique(cand_signal[valid]).size < 2 or np.unique(_ls[valid]).size < 2:
                            continue  # 常数列 ⇒ corr 无定义
                        _c = np.corrcoef(cand_signal[valid], _ls[valid])[0, 1]
                        if not np.isfinite(_c):
                            continue
                        corr = abs(_c)
                        if corr > max_corr:
                            max_corr = corr
                            max_corr_factor = lib_name
                    _src = "signal_recompute"

            results[i].s2_max_corr = max_corr
            results[i].s2_max_corr_factor = max_corr_factor
            results[i].s2_corr_source = _src
            # M8 (2026-09-26): 五态状态与判定解耦 — 缺省 0 是"未计算", 不是"相关低"。
            results[i].s2_eval_status = (
                "data_insufficient" if _src == "default_zero" else "valid")
            if _src == "default_zero":
                # W6 归因: max_corr 为缺省 0 时, "≤0.5 ✅" 是**空通过**, 不是"相关低"
                self._s2_default_zero = getattr(self, "_s2_default_zero", 0) + 1

            if max_corr <= self.corr_threshold:
                results[i].s2_passed = True
                if _src == "default_zero":
                    results[i].s2_reason = (f"max_corr 缺省 {max_corr:.3f} (≤{self.corr_threshold}) ✅ "
                                            f"[W6/M8: 空通过 — 无相关性数据 "
                                            f"(status=data_insufficient), 非'相关低']")
                else:
                    results[i].s2_reason = (f"max_corr={max_corr:.3f} (≤{self.corr_threshold}, "
                                            f"来源={_src}) ✅")
            else:
                results[i].s2_passed = False
                results[i].s2_reason = f"max_corr={max_corr:.3f} vs {max_corr_factor} (> {self.corr_threshold}) ❌"

        return results

    # ── Stage 3: Batch Deduplication ───────────────────────

    def stage3_batch_dedup(
        self,
        candidates: List[Dict],
        results: List[ValidationResult],
    ) -> List[ValidationResult]:
        """
        Stage 3: 批次内去重。

        对标 FactorMiner: Stage 3 Batch Deduplication

        检查同一批次中候选因子之间的冗余度。
        优先保留 ICIR 更高的因子。
        """
        # 收集 Stage 1+2 都通过的候选
        passed_idx = [i for i in range(len(results)) if results[i].s1_passed and results[i].s2_passed]

        # v0.6.1: 种子重检候选豁免批次去重 (同理 S2 豁免)
        for i in passed_idx[:]:
            if candidates[i].get("_seed_recheck"):
                results[i].s3_passed = True
                results[i].s3_reason = "种子重检豁免 (周期性裁决)"
                passed_idx.remove(i)

        # M8 (2026-09-26): 逐候选配对分母 (computed/missing) — 支撑"未计算"诚实归因。
        # 缺口: 原实现只有 idx_b 被标注, 从未进入比较的 idx_a 会命中末段兜底,
        # 被写成"批次内无高相关冗余 ✅" (把没检查说成检查过且干净)。
        _pair_stats: Dict[int, List[int]] = {}   # idx -> [computed, missing]

        def _mark_pair(idx: int, computed: bool) -> None:
            _st = _pair_stats.setdefault(idx, [0, 0])
            _st[0 if computed else 1] += 1

        if len(passed_idx) <= 1:
            for i in passed_idx:
                results[i].s3_passed = True
                results[i].s3_reason = "批次内无需去重 (≤1个候选)"
            return results

        # 按 ICIR 排序 (高优先)
        passed_idx.sort(key=lambda i: results[i].s1_icir, reverse=True)

        # ── W6 (2026-09-12): 骨架级批次去重 (免费, 默认 shadow) ──
        # 审计: `cand["signal"]` 全代码库无写入点 ⇒ 下方 corr 分支恒不执行 ⇒ S3
        #   以 "批次内无高相关冗余 ✅" 100% 通过 (实测 94→94)。这既是失效, 也是
        #   误报——它把"没检查"说成了"检查过且干净"。
        # 本块用零成本的结构指纹做真实批次去重 (同批同骨架 = 结构冗余):
        #   默认只标 `s3_skeleton_dup` 不裁决; env FUXI_S3_SKELETON_DEDUP=enforce 才拒。
        _skel_mode = "shadow"
        try:
            _skel_mode = os.environ.get("FUXI_S3_SKELETON_DEDUP", "shadow").strip().lower()
        except Exception:
            pass
        if _skel_mode in ("shadow", "enforce"):
            try:
                from subtree_fingerprinter import SubtreeFingerprinter as _SFP
                _sfp = _SFP()
                _seen_skel: Dict[str, int] = {}
                for idx_a in passed_idx:          # 已按 ICIR 降序 → 先到者优先保留
                    _f = candidates[idx_a].get("formula", candidates[idx_a].get("expression", ""))
                    if not _f:
                        continue
                    try:
                        _sk = _sfp.parse_expression(_f)
                    except Exception:
                        continue
                    if not _sk:
                        continue
                    if _sk in _seen_skel:
                        results[idx_a].s3_skeleton_dup = True
                        self._s3_skel_dup = getattr(self, "_s3_skel_dup", 0) + 1
                        if _skel_mode == "enforce":
                            results[idx_a].s3_passed = False
                            results[idx_a].s3_reason = (
                                f"批次内结构冗余: 骨架与 "
                                f"{candidates[_seen_skel[_sk]].get('factor_name','?')} 相同 "
                                f"({_sk[:60]}) ❌ [W6 骨架去重]")
                            self._s3_skel_rejected = getattr(self, "_s3_skel_rejected", 0) + 1
                    else:
                        _seen_skel[_sk] = idx_a
            except Exception as _ske:
                pass

        kept = set()
        for idx_a in passed_idx:
            if idx_a in kept:
                continue
            kept.add(idx_a)

            for idx_b in passed_idx:
                if idx_b in kept:
                    continue
                # 比较两个候选 (需要信号数据)
                sig_a = candidates[idx_a].get("signal")
                sig_b = candidates[idx_b].get("signal")
                if sig_a is None or sig_b is None:
                    # W6 归因: signal 未接线 → 该对无法比较 (非"无冗余")
                    results[idx_b].s3_signal_missing = True
                    _mark_pair(idx_a, False)   # M8: 双方都记缺数 (分母可见)
                    _mark_pair(idx_b, False)
                if sig_a is not None and sig_b is not None:
                    # P-20260917-025 F4: 与 stage2 (F1) 同族防御 —— 本分支在接线修复前
                    #   **从未真正执行过** (signal 恒 None ⇒ 恒短路); 修好后首次执行。
                    #   不同因子的信号来自各自面板, 股票集合不保证一致 ⇒ 长度不等时
                    #   `np.isnan(a) | np.isnan(b)` 布尔广播直接抛 ValueError 并拖死整轮。
                    #   归一化为 1-D float; 形状不等 ⇒ 记 W6 归因 (不可比), 不崩。
                    _sa = np.asarray(sig_a, dtype=float).ravel()
                    _sb = np.asarray(sig_b, dtype=float).ravel()
                    if _sa.shape != _sb.shape:
                        results[idx_b].s3_signal_missing = True
                        _mark_pair(idx_a, False)   # M8: 形状不可比 = 该对未计算
                        _mark_pair(idx_b, False)
                        _sa = None
                    valid = (~(np.isnan(_sa) | np.isnan(_sb))
                             if _sa is not None else np.zeros(0, dtype=bool))
                    if valid.sum() >= 20:
                        corr = abs(np.corrcoef(_sa[valid], _sb[valid])[0, 1])
                        _mark_pair(idx_a, True)    # M8: 真实计算的分母
                        _mark_pair(idx_b, True)
                        results[idx_b].s3_eval_status = "valid"
                        if corr > self.batch_corr_threshold:
                            # idx_b 被 idx_a "吃掉了"
                            results[idx_b].s3_max_batch_corr = corr
                            results[idx_b].s3_passed = False
                            results[idx_b].s3_reason = (
                                f"批次内与 {candidates[idx_a].get('factor_name', '?')} "
                                f"高相关 ({corr:.3f} > {self.batch_corr_threshold}) ❌"
                            )
                        else:
                            results[idx_b].s3_passed = True
                            results[idx_b].s3_reason = f"批次内最大corr={corr:.3f} ✅"
                    elif _sa is not None:
                        # M8: 有效样本 <20 ⇒ 该对未能计算 (非"无冗余")
                        _mark_pair(idx_a, False)
                        _mark_pair(idx_b, False)

        # M8 (2026-09-26): 逐候选配对分母回写 (完整呈现"检查了多少/缺了多少")。
        # 状态 = 配对分母的纯函数 (与判定/措辞解耦):
        #   valid = 全部配对已计算; data_insufficient = 有缺数但算过至少一对;
        #   not_evaluated = 一对都没算过。
        for _idx, (_pc, _pm) in _pair_stats.items():
            results[_idx].s3_pairs_computed = int(_pc)
            results[_idx].s3_pairs_missing = int(_pm)
            if _pc == 0 and _pm > 0:
                results[_idx].s3_eval_status = "not_evaluated"
            elif _pm > 0:
                results[_idx].s3_eval_status = "data_insufficient"
            else:
                results[_idx].s3_eval_status = "valid"

        for i in passed_idx:
            if results[i].s3_reason == "":
                results[i].s3_passed = True
                _pc, _pm = _pair_stats.get(i, [0, 0])
                if _pc == 0 and (results[i].s3_signal_missing or _pm > 0):
                    # M8: 全部未计算 — 状态=not_evaluated, 禁止"无冗余 ✅"
                    results[i].s3_eval_status = "not_evaluated"
                    results[i].s3_signal_missing = True
                    results[i].s3_reason = ("⚠️ 批次内相关性未计算 (signal 不可用) "
                                            "→ 空通过 [W6/M8: status=not_evaluated]")
                    self._s3_signal_missing = getattr(self, "_s3_signal_missing", 0) + 1
                elif _pc > 0 and _pm > 0:
                    # M8: 部分未计算 — 分母含缺数, 禁止"无冗余 ✅"
                    results[i].s3_eval_status = "data_insufficient"
                    results[i].s3_reason = (f"⚠️ 批次内相关性部分未计算 "
                                            f"(已算 {_pc} 对/缺 {_pm} 对) "
                                            f"→ 部分空通过 [M8 归因]")
                elif results[i].s3_skeleton_dup:
                    results[i].s3_eval_status = "valid"
                    results[i].s3_reason = ("⚠️ 批次内骨架重复 (shadow, 未裁决) "
                                            "[W6 骨架去重]")
                elif _pc > 0:
                    # M8: 只有真实算过才允许写"无冗余"
                    results[i].s3_eval_status = "valid"
                    results[i].s3_reason = "批次内无高相关冗余 ✅"
                else:
                    # 未进入任何比较 — 不假装检查过
                    results[i].s3_eval_status = "not_evaluated"
                    results[i].s3_reason = ("⚠️ 批次内未进入相关性比较 "
                                            "→ 空通过 [M8: status=not_evaluated]")

        return results

    # ── Stage 4: Full OOS + Replacement ───────────────────

    def stage4_oos_replacement(
        self,
        candidates: List[Dict],
        results: List[ValidationResult],
        library_factors: Optional[Dict] = None,  # factor_name → {icir, ...}
    ) -> List[ValidationResult]:
        """
        Stage 4: OOS 验证 + 替换检查。

        对标 FactorMiner: Stage 4 Full Validation (Robustness & OOS Test)
                         + Replacement Check (IC ≥ τ_ic_high & IC ≥ k·IC_i)

        检查候选因子是否值得替换库中高相关但 ICIR 更低的因子。
        """
        mode = str(os.environ.get("FUXI_S4_OOS_MODE", "audit") or "audit").strip().lower()
        if mode not in ("legacy", "audit", "strict"):
            raise ValueError("FUXI_S4_OOS_MODE must be legacy, audit, or strict")

        for i, cand in enumerate(candidates):
            if not all([results[i].s1_passed, results[i].s2_passed, results[i].s3_passed]):
                continue

            # OOS 验证 (使用候选自带的 OOS 指标)
            # v0.6 影子标记: 无独立 oos_ic 字段时回退 S1 IC → S4 与 S1 判定重复
            # (伪 OOS)。只标记不改判定, 供"评价宪法"删减方案对照统计。
            has_real_oos = (
                cand.get("oos_ic") is not None or cand.get("oos_icir") is not None
            )
            if not has_real_oos:
                results[i].s4_fallback_to_s1ic = True
                results[i].s4_eval_status = "not_evaluated"
                results[i].s4_decision_basis = "legacy_s1_proxy"
                if mode == "strict":
                    results[i].s4_passed = False
                    results[i].s4_reason = (
                        "❌ 独立 OOS 未执行 (status=not_evaluated); strict 模式禁止用 S1 代理"
                    )
                    continue
            else:
                results[i].s4_eval_status = "valid"
                results[i].s4_decision_basis = "independent_oos"
            oos_ic = abs(float(cand.get("oos_ic", cand.get("ic", 0)) or 0))
            oos_icir = abs(float(cand.get("oos_icir", cand.get("icir", 0)) or 0))
            results[i].s4_oos_ic = oos_ic
            results[i].s4_oos_icir = oos_icir

            # 替换检查
            max_corr_factor = results[i].s2_max_corr_factor
            if max_corr_factor and library_factors and max_corr_factor in library_factors:
                # M8 (2026-09-26): 与库元数据真实比较过 ⇒ 替换检查已评估
                results[i].s4_replacement_eval_status = "valid"
                old_icir = abs(library_factors[max_corr_factor].get("icir", 0))
                if oos_icir >= self.replacement_k * old_icir:
                    results[i].s4_replacement = True
                    results[i].s4_replaced_factor = max_corr_factor
                    results[i].s4_passed = True
                    results[i].s4_reason = (
                        f"替换 {max_corr_factor}: 新ICIR={oos_icir:.3f} ≥ "
                        f"{self.replacement_k}×旧ICIR={old_icir:.3f} ✅"
                    )
                else:
                    results[i].s4_passed = False
                    results[i].s4_reason = (
                        f"不替换 {max_corr_factor}: 新ICIR={oos_icir:.3f} < "
                        f"{self.replacement_k}×旧ICIR={old_icir:.3f} ❌"
                    )
            else:
                results[i].s4_passed = True
                # M8 (2026-09-26): 替换检查空转 — 状态与判定分离 (缺数≠通过)
                results[i].s4_replacement_eval_status = "not_evaluated"
                if max_corr_factor and not library_factors:
                    # W6 归因: S2 已找到最相关库因子, 但库元数据缺失 ⇒ 替换检查空转
                    results[i].s4_lib_meta_missing = True
                    self._s4_lib_meta_missing = getattr(self, "_s4_lib_meta_missing", 0) + 1
                    results[i].s4_reason = (
                        f"⚠️ 库元数据缺失 (max_corr_factor={max_corr_factor}), "
                        f"替换检查空转 → 空通过 [W6 归因]")
                elif not max_corr_factor and results[i].s2_eval_status == "data_insufficient":
                    # M8: S2 相关性未计算 ⇒ 无从判断是否需要替换 (不得写"无对应因子")
                    results[i].s4_replacement_eval_status = "data_insufficient"
                    results[i].s4_reason = ("⚠️ 替换检查未执行: S2 相关性未计算 "
                                            "(status=data_insufficient) "
                                            "→ 空通过 [M8 归因]")
                elif not library_factors:
                    results[i].s4_reason = ("⚠️ 无库元数据 → 替换检查空转 "
                                            "[W6 归因]")
                else:
                    results[i].s4_reason = "无替换需求或库中无对应因子 ✅"

            if not has_real_oos and mode == "audit":
                results[i].s4_reason = (
                    "⚠️ 独立 OOS 未执行 (status=not_evaluated); "
                    "沿用旧 S1 代理仅维持门链兼容，不得表述为 OOS 通过"
                )

        return results

    # ── Stage 5: 联合正向过滤 + Calmar (v0.3 新增) ────────

    def stage5_joint_filter(
        self,
        candidates: List[Dict],
        results: List[ValidationResult],
        s5_filter: Any = None,
    ) -> List[ValidationResult]:
        """
        Stage 5: 联合正向过滤 + Calmar 比率约束。

        对标中金 CICC Loop Engineering 报告的 11 项联合过滤中的:
          - 两个独立年度的同时正向过滤（主要瓶颈，~60%拒绝）
          - Calmar 比率 > 1.0

        Parameters
        ----------
        candidates: 候选因子列表，每个需包含:
            - factor_name, formula (用于 S5 实时计算)
            - 或预计算: year1_excess/excess_2025, year2_excess/excess_2026, calmar
        results: 验证结果列表
        s5_filter: S5JointFilter 实例 (可选)
            - None: 使用预计算数据
            - 提供: 对通过 S1-S4 的因子进行实时 S5 验证

        Returns
        -------
        更新后的 results
        """
        # 收集需要通过 S5 实时计算的因子
        real_compute_candidates = []
        real_compute_indices = []

        for i, cand in enumerate(candidates):
            if not all([results[i].s1_passed, results[i].s2_passed,
                        results[i].s3_passed, results[i].s4_passed]):
                continue

            # 检查是否有预计算数据
            has_precomputed = (
                (cand.get("excess_2025") is not None and cand.get("excess_2026") is not None) or
                (cand.get("year1_excess") is not None and cand.get("year2_excess") is not None)
            )

            if s5_filter is not None and not has_precomputed and s5_filter.is_ready:
                real_compute_candidates.append(cand)
                real_compute_indices.append(i)
            elif has_precomputed:
                # 使用结构化预计算数据
                self._apply_stage5_from_data(cand, results, i)
            else:
                results[i].s5_eval_status = "not_evaluated"
                results[i].s5_passed = False
                results[i].s5_reason = (
                    "S5 未执行: 无实时过滤器且无结构化预计算读数 "
                    "(status=not_evaluated, 不计作因子失败)"
                )

        # 批量 S5 实时计算
        if real_compute_candidates and s5_filter is not None:
            print(f"  [S5] Computing real joint filter for {len(real_compute_candidates)} candidates...")
            s5_results = s5_filter.validate_batch(real_compute_candidates, verbose=True)
            for j, s5r in enumerate(s5_results):
                idx = real_compute_indices[j]
                self._apply_stage5_from_s5result(s5r, results, idx)
                # v0.6.1: S5 组合收益序列回传 → S6 DSR / 增量门禁数据通道
                _pr = getattr(s5r, "portfolio_returns", None)
                if _pr is not None and len(_pr) > 0:
                    candidates[idx]["_combo_returns"] = _pr

        return results

    def _apply_stage5_from_data(
        self, cand: Dict, results: List[ValidationResult], i: int
    ):
        """从预计算数据应用 S5 过滤（兼容旧格式）"""
        results[i].s5_eval_status = "valid"
        failures = []

        # 年度超额
        yr1 = float(cand.get("excess_2025", cand.get("year1_excess",
                     cand.get("excess_year1", 0))) or 0)
        results[i].s5_year1_excess = yr1
        if yr1 <= 0:
            failures.append(f"{self.year1_label}超额={yr1:.1%} ≤ 0")

        yr2 = float(cand.get("excess_2026", cand.get("year2_excess",
                     cand.get("excess_year2", 0))) or 0)
        results[i].s5_year2_excess = yr2
        if yr2 <= 0:
            failures.append(f"{self.year2_label}超额={yr2:.1%} ≤ 0")

        # Calmar
        calmar = float(cand.get("calmar", cand.get("calmar_ratio",
                     cand.get("calmar_ratio_annual", 0))) or 0)
        if calmar == 0:
            sharpe = float(cand.get("sharpe", cand.get("annual_sharpe", 0)) or 0)
            maxdd = abs(float(cand.get("maxdd", cand.get("max_drawdown", 1)) or 1))
            if sharpe > 0 and maxdd > 0 and maxdd < 1:
                calmar = sharpe / maxdd

        results[i].s5_calmar = calmar
        if calmar < self.calmar_threshold:
            if calmar > 0:
                failures.append(f"Calmar={calmar:.2f} < {self.calmar_threshold}")

        if failures:
            results[i].s5_passed = False
            results[i].s5_reason = "❌ 联合正向过滤失败: " + "; ".join(failures)
        else:
            results[i].s5_passed = True
            results[i].s5_reason = (
                f"✅ 两年正向过滤通过: {self.year1_label}超额={yr1:.1%}, "
                f"{self.year2_label}超额={yr2:.1%}, Calmar={calmar:.2f}"
            )

    def _apply_stage5_from_s5result(
        self, s5r: Any, results: List[ValidationResult], i: int
    ):
        """从 S5JointFilter 结果应用到 ValidationResult"""
        _elapsed = float(getattr(s5r, "elapsed_sec", 0.0) or 0.0)
        _w25 = int(getattr(s5r, "window_days_2025", 0) or 0)
        _w26 = int(getattr(s5r, "window_days_2026", 0) or 0)
        _reason = str(getattr(s5r, "reason", "") or "")
        _error = str(getattr(s5r, "error", "") or "")
        if _error in {"held_price_observations_missing", "benchmark_dates_missing",
                      "weekly_data_unavailable", "Price data not available"}:
            results[i].s5_eval_status = "data_insufficient"
        elif _error:
            results[i].s5_eval_status = "error"
        elif _elapsed > 0 or _w25 > 0 or _w26 > 0:
            results[i].s5_eval_status = "valid"
        elif _reason.startswith("🧹 公式合理性拦截"):
            results[i].s5_eval_status = "precheck_rejected"
        else:
            results[i].s5_eval_status = "not_evaluated"
        results[i].s5_year1_excess = s5r.excess_2025
        results[i].s5_year2_excess = s5r.excess_2026
        results[i].s5_calmar = max(s5r.calmar_2025, s5r.calmar_2026)
        results[i].s5_passed = bool(s5r.passed and not _error)
        results[i].s5_reason = s5r.reason
        results[i].s5_marking_diagnostics = getattr(s5r, "marking_diagnostics", None)
        results[i].s5_economic_readings_status = str(
            getattr(s5r, "economic_readings_status", "not_evaluated") or "not_evaluated")
        # ── W5 (2026-09-12): 口径留痕透传 (getattr 兜底, 不做硬依赖) ──
        results[i].s5_calmar_mode = str(getattr(s5r, "calmar_mode", "") or "")
        results[i].s5_calmar_2025 = float(getattr(s5r, "calmar_2025", 0) or 0)
        results[i].s5_calmar_2026 = float(getattr(s5r, "calmar_2026", 0) or 0)
        results[i].s5_calmar_2025_full = float(getattr(s5r, "calmar_2025_full", 0) or 0)
        results[i].s5_calmar_2026_full = float(getattr(s5r, "calmar_2026_full", 0) or 0)
        results[i].s5_window_days_2025 = int(getattr(s5r, "window_days_2025", 0) or 0)
        results[i].s5_window_days_2026 = int(getattr(s5r, "window_days_2026", 0) or 0)
        results[i].s5_window_matched = getattr(s5r, "calmar_window_matched", None)
        results[i].s5_calmar_threshold_effective = float(
            getattr(s5r, "calmar_threshold_effective", 1.0) or 1.0)
        # 不可比告警计数 (诊断, 不裁决)
        if results[i].s5_window_matched is False:
            self._s5_window_unmatched = getattr(self, "_s5_window_unmatched", 0) + 1

    # ── Stage 6: 多重检验门禁 DSR/PBO/BH-FDR (v0.6 实验) ──

    def stage6_multiple_testing(
        self,
        candidates: List[Dict],
        results: List[ValidationResult],
        guard: Any = None,
    ) -> List[ValidationResult]:
        """
        Stage 6: 多重检验门禁 (LLM_QUANT_FACTORY 评价宪法借鉴)。

        - DSR (Deflated Sharpe Ratio): 收益序列 SR 经多次试验次数折损后的概率
        - PBO (CSCV): 回测过拟合概率 (单因子场景无性能矩阵 → 不计算)
        - BH-FDR: 批次内 p 值多重校正

        影子模式 (multiple_testing_gate=False, 默认): 只计算输出，policy_allows 保持 True。
        生效模式: DSR<阈值 或 FDR 不过 → s6_passed=False → eligible_for_jq 被剥夺。

        candidate 需携带 return_series (日/周收益数组) 才能计算 DSR;
        无收益序列的候选 s6_computed=False (不参与判定)。
        """
        if guard is None:
            try:
                from config import V06_EXPERIMENTAL
                from multiple_testing_guard import MultipleTestingGuard
                gate_on = bool(V06_EXPERIMENTAL.get("multiple_testing_gate", False))
                self._s6_gate_enabled = gate_on
                # v0.6.1 (2026-08-29) 口径修复:
                # 1) trials 批次级 (原 _resolve_trials 用全库 248 attempts → 过度折损,
                #    王者因子 DSR 只有 0.1 级, 0.90 门槛全杀 → 流水线死锁)
                # 2) periods_per_year=252 (S5 回传的 _combo_returns 是日频组合收益,
                #    原 52 按周频年化低估 SR 2.2 倍)
                if V06_EXPERIMENTAL.get("dsr_trials_batch_level", True):
                    trials = max(5, len(candidates))
                else:
                    trials = None
                guard = MultipleTestingGuard(
                    trials=trials,
                    dsr_min_probability=float(V06_EXPERIMENTAL.get("dsr_min_probability", 0.01)),
                    pbo_max=float(V06_EXPERIMENTAL.get("pbo_max", 0.40)),
                    fdr_alpha=float(V06_EXPERIMENTAL.get("fdr_alpha", 0.10)),
                    gate_enabled=gate_on,
                    periods_per_year=252,
                )
            except Exception:
                from multiple_testing_guard import MultipleTestingGuard
                guard = MultipleTestingGuard(gate_enabled=False)
        else:
            self._s6_gate_enabled = bool(getattr(guard, "gate_enabled", False))

        # 收集批次 p 值做 BH-FDR (candidate 携带的 ic_p_value)
        names, rets, pvals = [], [], []
        for i, cand in enumerate(candidates):
            names.append(results[i].factor_name)
            # v0.6.1: 收益序列优先级 return_series > S5 回传的组合收益 (_combo_returns)
            rs = cand.get("return_series")
            if rs is None:
                rs = cand.get("_combo_returns")
            rets.append(np.asarray(rs, dtype=float) if rs is not None else None)
            pvals.append(cand.get("ic_p_value"))

        batch = guard.evaluate_batch(names, rets, p_values=pvals)
        n_computed = 0
        for i, r in enumerate(batch):
            results[i].s6_computed = r.evaluation_status == "valid"
            results[i].s6_eval_status = r.evaluation_status
            results[i].s6_policy_allows = r.policy_allows
            results[i].s6_dsr_prob = r.dsr_probability
            results[i].s6_pbo = r.pbo
            results[i].s6_fdr_adj_p = r.fdr_adjusted_p
            results[i].s6_passed = r.s6_passed
            results[i].s6_reason = "; ".join(r.reasons) if r.reasons else "✅"
            if results[i].s6_computed:
                n_computed += 1

        mode = "生效(硬门禁)" if guard.gate_enabled else "影子(只输出)"
        n_reject = sum(1 for r in results if r.s6_computed and not r.s6_passed)
        n_unresolved = sum(1 for r in results if not r.s6_computed)
        if n_computed > 0 or n_reject > 0 or n_unresolved > 0:
            print(f"  [S6-多重检验/{mode}] 可计算: {n_computed}/{len(results)} | "
                  f"有效拒绝: {n_reject} | 未完成: {n_unresolved}")
        # v0.6.1: 拒绝原因明细 (原只有一行汇总, DSR 全杀时无法定位)
        for r in results:
            if r.s6_computed and not r.s6_passed:
                print(f"    [S6拒绝] {r.factor_name[:40]}: DSR={r.s6_dsr_prob:.3f} | {r.s6_reason}")
        return results

    # ── 全流程验证 (v0.3 增强: 包含 S5) ───────────────────

    def validate(
        self,
        candidates: List[Dict],
        library_signals: Optional[Dict[str, np.ndarray]] = None,
        library_factors: Optional[Dict] = None,
        s5_filter: Any = None,
        s5_lightweight: bool = False,  # v0.5.2
    ) -> Tuple[List[ValidationResult], Dict]:
        """
        完整五阶段验证 (v0.5.2: S2 真实相关性 + S5 轻量回退)。

        Parameters
        ----------
        candidates: 候选因子列表, 每个包含:
            factor_name, formula, ic/icir/daily_ic/daily_icir,
            max_corr, max_corr_factor, signal (可选), oos_ic/oos_icir (可选)
        library_signals: {factor_name: signal_array} — 库中因子的信号序列
        library_factors: {factor_name: {icir, ...}} — 库中因子的元数据
        s5_filter: S5JointFilter 实例 (可选)
            - None: 使用 candidate 中预计算的 excess_2025/excess_2026/calmar
            - 提供: 使用真实行情数据计算
        s5_lightweight: v0.5.2 — S5 数据来自 IC 年拆分代理 (非真实回测)
            - True 时降低 S5 阈值 (适配 IC-scale 数据)
            - True 时 eligible_for_jq 必须 S5 通过 (禁止绕过)

        Returns
        -------
        (results, summary)
        """
        # 初始化结果
        results = [
            ValidationResult(
                factor_name=c.get("factor_name", f"candidate_{i}"),
                # 2026-09-01 防御: expression 兜底 (与 _phase_generate/_phase_evaluate
                # 的双键读取口径一致), 防止上游候选缺 formula 键时结果公式丢失
                formula=c.get("formula", c.get("expression", "")),
                paradigm=c.get("paradigm", ""),
                hypothesis=c.get("hypothesis", ""),
            )
            for i, c in enumerate(candidates)
        ]

        # Stage 1
        results = self.stage1_fast_ic(candidates, results)

        # ── W2 (2026-09-12): S1 样本量 / 求值失败归因读数 ──
        _n_low = int(getattr(self, "_s1_low_sample", 0))
        _n_ef = int(getattr(self, "_s1_eval_failed", 0))
        _nds = [r.s1_n_days for r in results if getattr(r, "s1_n_days", -1) > 0]
        _med = sorted(_nds)[len(_nds) // 2] if _nds else -1
        if _nds or _n_ef:
            print(f"  [S1-W2] min_ic_days={self.min_ic_days} | 低样本拒 {_n_low} | "
                  f"公式求值失败 {_n_ef} | n_days 中位={_med} (归因读数)")
        self._s1_w2_stats = {"low_sample": _n_low, "eval_failed": _n_ef,
                             "n_days_median": _med, "min_ic_days": self.min_ic_days}

        # ── W3 (2026-09-12): 已证伪方向命中读数 ──
        _n_fh = int(getattr(self, "_s1_forbidden_hits", 0))
        _n_fr = int(getattr(self, "_s1_forbidden_rejected", 0))
        if _n_fh:
            print(f"  [S1-W3] 命中已证伪方向 {_n_fh} 候选"
                  + (f" → 硬拒 {_n_fr}" if _n_fr else " (仅标记, shadow)"))

        # ── M10 (2026-09-27): AST 结构重复命中读数 ──
        _n_sdh = int(getattr(self, "_s1_struct_dup_hits", 0))
        _n_sdr = int(getattr(self, "_s1_struct_dup_rejected", 0))
        if _n_sdh:
            print(f"  [S1-M10] 命中 AST 结构重复 {_n_sdh} 候选"
                  + (f" → 硬拒 {_n_sdr}" if _n_sdr else " (仅标记, shadow)"))

        # Stage 2
        results = self.stage2_correlation(candidates, results, library_signals)

        # ── P-20260828-004: IC bootstrap 置信区间 (影子模式: 只输出不改变判定) ──
        n_boot_computed = 0
        n_ci_below_zero = 0
        for i, cand in enumerate(candidates):
            series = cand.get("_ic_series")
            if not series:
                continue
            ci = block_bootstrap_ic_ci(series)
            if ci["n_obs"] == 0:
                continue
            results[i].ic_boot_mean = ci["mean"]
            results[i].ic_ci95_low = ci["ci95_low"]
            results[i].ic_ci95_high = ci["ci95_high"]
            results[i].ic_boot_n = ci["n_obs"]
            n_boot_computed += 1
            if ci["ci95_low"] < 0:
                # CI 下限 < 0 → IC 符号统计上不显著 (影子: 仅计数)
                n_ci_below_zero += 1
        if n_boot_computed > 0:
            print(f"  [P-004影子] IC bootstrap CI: {n_boot_computed} 候选 | "
                  f"CI下限<0: {n_ci_below_zero} (仅统计, 不改判定)")

        # ── P-20260828-001: Δlag 滞后衰减 (影子模式: 只输出不改变判定) ──
        n_delta_computed = sum(1 for i, cand in enumerate(candidates)
                               if cand.get("ic_lag1") is not None or cand.get("delta_lag") is not None)
        n_decay_risk = sum(1 for r in results if r.high_decay_risk)
        if n_delta_computed > 0:
            print(f"  [P-001影子] Δlag 滞后衰减: {n_delta_computed} 候选有 lag1 IC | "
                  f"high_decay_risk(Δlag>0.015 或 Δlag/IC>50%): {n_decay_risk} "
                  f"(仅标记, 不改判定)")

        # Stage 3
        results = self.stage3_batch_dedup(candidates, results)

        # Stage 4
        results = self.stage4_oos_replacement(candidates, results, library_factors)

        # Stage 5 (v0.5.2): 联合正向过滤 + Calmar
        # 轻量模式: 降低阈值适配 IC-scale 代理数据
        if s5_lightweight:
            orig_calmar = self.calmar_threshold
            self.calmar_threshold = 0.3  # ICIR 0.3 适配 IC-scale
            results = self.stage5_joint_filter(candidates, results, s5_filter)
            self.calmar_threshold = orig_calmar
        else:
            results = self.stage5_joint_filter(candidates, results, s5_filter)

        # Stage 6 (v0.6 实验): 多重检验门禁
        # 影子模式(默认): s6_passed 恒 True → 下游判定与 v0.5.2 完全一致 (rollback 安全)
        results = self.stage6_multiple_testing(candidates, results)

        # 综合评分
        # v0.5.2: 轻量 S5 时禁止绕过 — 必须 S5 通过才 eligible
        # v0.6: S6 生效模式时 eligible 还需 s6_passed (影子模式恒真, 行为不变)
        for i, r in enumerate(results):
            stages_passed = sum([r.s1_passed, r.s2_passed, r.s3_passed, r.s4_passed, r.s5_passed])
            r.total_stages_passed = stages_passed
            # gate_level: 序列门语义 (S1→S2→...→S5 连续通过的最深阶段)
            seq = [r.s1_passed, r.s2_passed, r.s3_passed, r.s4_passed, r.s5_passed]
            r.gate_level = 0
            for j, ok in enumerate(seq):
                if not ok:
                    break
                r.gate_level = j + 1
            if s5_lightweight:
                # 轻量 S5 成本低 → 严格要求 5/5
                r.eligible_for_jq = (stages_passed >= 5) and r.s6_policy_allows
            else:
                # 真实 S5 路径: 原 `>= 4` 的历史理由是"真实 S5 成本高 → 4/5 即可"。
                # 🔴 W6 审计 (2026-09-12): 因 S2/S3/S4 几乎不拒 (S3 失效、S4 空转),
                #   `>=4` 等价于"只要 S1 过就进 JQ", S5 失败的 B 级因子照样入队 —
                #   而 S5 是**唯一**真正收口的关口 (实测 S4→S5 只剩 12.8%)。
                # 开关: env FUXI_ELIGIBLE_REQUIRE_S5 = on(默认) | off。
                # 🔴 v0.7.27 翻默认 off→on (用户 2026-09-16 决策)。
                #   用户口径"S5 没通过即无实盘价值" ⇒ 必须 5/5。
                #   ⚠️ 副作用 (已充分披露): JQ 队列 121→7 (实测), 近期归零 ⇒
                #   D+ 闭环**断供**。这是**有意**的收紧: 宁可无产出, 不让
                #   S5 失败候选流入 JQ 冒充"通过"。回退: 显式设 env=off。
                _req_s5 = os.environ.get("FUXI_ELIGIBLE_REQUIRE_S5", "on").strip().lower()
                _need = 5 if _req_s5 in ("on", "1", "true", "yes", "enforce") else 4
                r.eligible_for_jq = (stages_passed >= _need) and r.s6_policy_allows

            # A semantic plan may only narrow eligibility. A research feature
            # has a different objective (fixed-book increment), regardless of
            # its standalone S1-S5 readings. Unknown tagged uses fail closed.
            tagged_use = candidates[i].get("semantic_target_use")
            if ((tagged_use and tagged_use != "promotion_candidate")
                    or candidates[i].get("semantic_plan_violations")):
                r.eligible_for_jq = False
            # Static quality observations never grant eligibility. Shadowed
            # hard defects and unresolved mechanism reviews both downlock JQ.
            quality = candidates[i].get("formula_quality") or {}
            if (quality.get("hard_codes") or quality.get("review_codes")
                    or candidates[i].get("formula_quality_review_required")):
                r.eligible_for_jq = False

            if stages_passed == 5:
                r.final_grade = "A"
            elif stages_passed == 4:
                r.final_grade = "B"
            elif stages_passed == 3:
                r.final_grade = "C"
            else:
                r.final_grade = "D"

        # P-20260915-009: S5 口径模式 (供 s5_caliber.alert 判定告警口径)
        _cm_mode = next((r.s5_calmar_mode for r in results if r.s5_calmar_mode), "")

        # 汇总
        summary = {
            "total_candidates": len(candidates),
            "stage1_passed": sum(1 for r in results if r.s1_passed),
            "stage2_passed": sum(1 for r in results if r.s2_passed),
            "stage3_passed": sum(1 for r in results if r.s3_passed),
            "stage4_passed": sum(1 for r in results if r.s4_passed),
            "stage2_valid_passed": sum(1 for r in results if r.s2_passed and r.s2_eval_status == "valid"),
            "stage3_valid_passed": sum(1 for r in results if r.s3_passed and r.s3_eval_status == "valid"),
            "stage4_valid_passed": sum(1 for r in results if r.s4_passed and r.s4_eval_status == "valid"),
            "stage5_passed": sum(1 for r in results if r.s5_passed),
            "stage5_valid_passed": sum(1 for r in results if r.s5_passed and r.s5_eval_status == "valid"),
            "stage6_passed": sum(1 for r in results if r.s6_passed),      # v0.6
            "stage6_valid_passed": sum(1 for r in results if r.s6_passed and r.s6_eval_status == "valid"),
            "stage6_policy_allows": sum(1 for r in results if r.s6_policy_allows),
            "eligible_for_jq": sum(1 for r in results if r.eligible_for_jq),
            "s5_mode": "lightweight_ic" if s5_lightweight else ("real_s5" if s5_filter else "precomputed"),
            "ic_boot_computed": n_boot_computed,      # P-20260828-004 影子统计
            "delta_lag_computed": n_delta_computed,   # P-20260828-001 影子统计
            "delta_lag_high_decay_risk": n_decay_risk,
            "ic_ci_low_below_zero": n_ci_below_zero,  # P-20260828-004 影子统计
            "s4_fallback_to_s1ic": sum(1 for r in results if r.s4_fallback_to_s1ic),  # v0.6 伪OOS统计
            "evaluation_status": {
                stage: {
                    status: sum(
                        1 for r in results
                        if getattr(r, f"{stage}_eval_status") == status
                    )
                    for status in sorted({
                        getattr(r, f"{stage}_eval_status") for r in results
                    })
                }
                for stage in ("s1", "s2", "s3", "s4", "s5", "s6")
            },
            "s6_computed": sum(1 for r in results if r.s6_computed),      # v0.6
            "s6_mode": "gate" if self._s6_gate_enabled else "shadow",     # v0.6
            "grade_distribution": {
                "A": sum(1 for r in results if r.final_grade == "A"),
                "B": sum(1 for r in results if r.final_grade == "B"),
                "C": sum(1 for r in results if r.final_grade == "C"),
                "D": sum(1 for r in results if r.final_grade == "D"),
            },
            # ── W2 (2026-09-12): S1 样本量守卫读数 (归因 + 报表用) ──
            "s1_min_ic_days": self.min_ic_days,
            "s1_low_sample_rejected": int(getattr(self, "_s1_low_sample", 0)),
            "s1_ic_eval_failed": int(getattr(self, "_s1_eval_failed", 0)),
            "s1_n_days_median": int(getattr(self, "_s1_w2_stats", {}).get("n_days_median", -1)),
            # ── W3 (2026-09-12): 已证伪方向屏蔽读数 ──
            "s1_forbidden_hits": int(getattr(self, "_s1_forbidden_hits", 0)),
            "s1_forbidden_rejected": int(getattr(self, "_s1_forbidden_rejected", 0)),
            # ── M10 (2026-09-27): AST 结构重复读数 ──
            "s1_struct_dup_hits": int(getattr(self, "_s1_struct_dup_hits", 0)),
            "s1_struct_dup_rejected": int(getattr(self, "_s1_struct_dup_rejected", 0)),
            # ── W6 (2026-09-12): S2–S4 空转归因 (诚实口径) ──
            "s2_default_zero_passed": int(getattr(self, "_s2_default_zero", 0)),
            "s3_signal_missing_passed": int(getattr(self, "_s3_signal_missing", 0)),
            "s3_skeleton_dup": int(getattr(self, "_s3_skel_dup", 0)),
            "s3_skeleton_rejected": int(getattr(self, "_s3_skel_rejected", 0)),
            "s4_lib_meta_missing_passed": int(getattr(self, "_s4_lib_meta_missing", 0)),
            # ── M8 (2026-09-26): S2/S3/S4 空通过影子分母 (五态) ──
            # 口径: "若未来把缺数一律改判 reject 将拦下的条数" — 影子拦截数,
            #   不参与裁决, 仅供误杀率 A/B 与分母审计。
            "m8_shadow_reject_s2": sum(
                1 for r in results
                if r.s2_passed and r.s2_eval_status == "data_insufficient"),
            "m8_shadow_reject_s3": sum(
                1 for r in results
                if r.s3_passed and r.s3_pairs_missing > 0),
            "m8_shadow_reject_s4_replacement": sum(
                1 for r in results
                if r.s4_passed and (r.s4_lib_meta_missing
                                    or r.s4_replacement_eval_status == "data_insufficient")),
            "eligible_require_s5": str(os.environ.get("FUXI_ELIGIBLE_REQUIRE_S5", "on")).lower(),
            # ── W5 (2026-09-12): S5 口径留痕 (诊断, 不影响裁决) ──
            # P-20260915-009: 告警口径修正。vol_scaled (默认) 的分母已由 |maxDD|
            #   换成**年化波动率** (IR 型) ⇒ 设计上与窗口长度无关 (蒙特卡洛实测:
            #   legacy 随窗口漂移 2.89x, IR span 仅 0.033)。故 2025(243日) 与
            #   2026(159日) 必然 unmatched, 旧实现天天打「不可比」⇒ 告警疲劳,
            #   且会误导 S5 干旱归因 (误以为存在口径成因)。现仅在窗口敏感口径
            #   (legacy / common_window) 下告警; vol_scaled 下 window_matched
            #   字段保留供诊断, alert 置空。
            "s5_caliber": {
                "note": "口径字段不影响裁决, 仅用于诊断/复现",
                "calmar_mode": _cm_mode,
                "calmar_threshold_effective": next(
                    (r.s5_calmar_threshold_effective for r in results
                     if r.s5_calmar_mode), 1.0),
                "window_days_2025": next((r.s5_window_days_2025 for r in results
                                          if r.s5_window_days_2025), 0),
                "window_days_2026": next((r.s5_window_days_2026 for r in results
                                          if r.s5_window_days_2026), 0),
                "window_matched": next((r.s5_window_matched for r in results
                                        if r.s5_window_matched is not None), None),
                "n_window_unmatched": int(getattr(self, "_s5_window_unmatched", 0)),
                "alert": ("" if _cm_mode == "vol_scaled" else (
                    "⚠️ S5 两窗口交易日数不对齐 ⇒ Calmar 不可比"
                    if getattr(self, "_s5_window_unmatched", 0) > 0 else "")),
            },
            "timestamp": datetime.now().isoformat(),
        }

        return results, summary

    def print_summary(self, results: List[ValidationResult], summary: Dict):
        """打印验证摘要"""
        print("=" * 60)
        print("  Multi-Stage Validation Summary")
        print("=" * 60)
        print(f"\n  📊 总计: {summary['total_candidates']} 个候选")
        print(f"  📊 各阶段通过:")
        for stage in ["stage1", "stage2", "stage3", "stage4", "stage5", "stage6"]:
            passed = summary[f"{stage}_passed"]
            bar = "🟢" * passed + "🔴" * (summary["total_candidates"] - passed)
            stage_labels = {
                "stage1": "S1 快速IC",
                "stage2": "S2 库内去重",
                "stage3": "S3 批次去重",
                "stage4": "S4 OOS+替换",
                "stage5": "S5 联合过滤+Calmar",
                "stage6": f"S6 多重检验[{summary.get('s6_mode', 'shadow')}]",
            }
            print(f"    {stage_labels[stage]}: {bar} ({passed})")

        print(f"\n  🎯 JQ 候选: {summary['eligible_for_jq']} 个")
        print(f"  📋 评级: A={summary['grade_distribution']['A']} "
              f"B={summary['grade_distribution']['B']} "
              f"C={summary['grade_distribution']['C']} "
              f"D={summary['grade_distribution']['D']}")

        # 列出 A/B 级因子
        a_b_results = [r for r in results if r.final_grade in ("A", "B")]
        if a_b_results:
            print(f"\n  🏆 A/B 级因子:")
            for r in a_b_results:
                print(f"    [{r.final_grade}] {r.factor_name}: "
                      f"S1={r.s1_passed} S2={r.s2_passed} S3={r.s3_passed} "
                      f"S4={r.s4_passed} S5={r.s5_passed} S6={r.s6_passed}")

        # v0.6: S4 伪OOS 与 S6 影子统计提示
        n_fb = summary.get("s4_fallback_to_s1ic", 0)
        if n_fb > 0:
            print(f"\n  [v0.6影子] S4 伪OOS回退(S1 IC代OOS): {n_fb} 候选 (只统计, 不改判定)")
        if summary.get("s6_computed", 0) > 0:
            n_rej = summary["total_candidates"] - summary["stage6_passed"]
            tag = "硬拦截" if summary.get("s6_mode") == "gate" else "只输出"
            print(f"  [v0.6] S6 多重检验: {summary['s6_computed']} 可计算 | "
                  f"S6拒绝 {n_rej} ({tag})")


# ── 北向资金数据加载 (v0.5 P-002) ───────────────────────

def load_moneyflow_data(data_dir: Path = DATA_DIR) -> Optional[pd.DataFrame]:
    """加载沪深港通资金流数据，用于 S5 评估中注入北向资金因子"""
    mf_path = data_dir / "raw" / "moneyflow_hsgt.csv"
    if not mf_path.exists():
        return None
    try:
        df = pd.read_csv(mf_path)
        df['trade_date'] = pd.to_datetime(df['trade_date'], errors='coerce')
        return df
    except Exception:
        return None


# ── IC 序列缓存 (v0.5 P-004) ──────────────────────────────

def cache_ic_series(factor_name: str, ic_series, cache_path: Optional[Path] = None):
    """S5 评估后将因子 IC 序列缓存到 decay_monitor
    
    在 S5 的 evaluate() 末尾调用，每个因子的 IC 序列会自动累积。
    """
    if cache_path is None:
        cache_path = DATA_DIR / "cache" / "ic_history.json"
    try:
        from decay_monitor import get_decay_monitor
        dm = get_decay_monitor()
        dm.update_ic_cache(factor_name, ic_series)
    except ImportError:
        pass  # decay_monitor 未导入则不缓存


# ── 便捷函数 ──────────────────────────────────────────────

def compute_sector_ic(
    factor_signals: np.ndarray,
    returns: np.ndarray,
    sector_labels: List[str],
    min_stocks_per_sector: int = 5,
) -> Dict:
    """
    P-016 行业轮动验证模式: 计算因子在行业维度的 Rank IC。

    对行业轮动类因子 (paradigm=行业轮动)，应按行业维度评估因子效果。
    每个行业内计算 rank IC，然后取均值作为行业维度 IC。

    Parameters
    ----------
    factor_signals: (n_stocks,) 因子信号值
    returns: (n_stocks,) 对应未来收益率
    sector_labels: (n_stocks,) 申万行业标签
    min_stocks_per_sector: 行业内最少股票数

    Returns
    -------
    {
        "sector_ic": float,           # 行业内IC均值
        "sector_ic_std": float,       # 行业内IC标准差
        "sector_icir": float,         # 行业维度ICIR = mean/std
        "n_sectors_valid": int,       # 有效行业数
        "per_sector_ic": Dict[str, float],  # 每个行业的IC
        "top_sectors": List[str],     # IC最高的前3行业
        "bottom_sectors": List[str],  # IC最低的前3行业
    }
    """
    from scipy.stats import rankdata

    # 按行业分组
    sectors = {}
    for i, label in enumerate(sector_labels):
        if label not in sectors:
            sectors[label] = {"signals": [], "returns": []}
        sectors[label]["signals"].append(factor_signals[i])
        sectors[label]["returns"].append(returns[i])

    per_sector_ic = {}
    valid_sectors = 0

    for label, data in sectors.items():
        sig = np.array(data["signals"])
        ret = np.array(data["returns"])
        valid = ~(np.isnan(sig) | np.isnan(ret))
        if valid.sum() < min_stocks_per_sector:
            continue
        sig_valid = sig[valid]
        ret_valid = ret[valid]

        # 行业内 rank IC
        sig_rank = rankdata(sig_valid) / len(sig_valid)
        ret_rank = rankdata(ret_valid) / len(ret_valid)
        ic = np.corrcoef(sig_rank, ret_rank)[0, 1]
        if not np.isnan(ic):
            per_sector_ic[label] = ic
            valid_sectors += 1

    if valid_sectors < 2:
        return {
            "sector_ic": 0.0, "sector_ic_std": 0.0, "sector_icir": 0.0,
            "n_sectors_valid": valid_sectors,
            "per_sector_ic": per_sector_ic,
            "top_sectors": [], "bottom_sectors": [],
        }

    ic_values = list(per_sector_ic.values())
    sector_ic = np.mean(ic_values)
    sector_ic_std = np.std(ic_values)
    sector_icir = sector_ic / sector_ic_std if sector_ic_std > 0 else 0.0

    # 排序
    sorted_sectors = sorted(per_sector_ic.items(), key=lambda x: x[1], reverse=True)
    top_sectors = [s[0] for s in sorted_sectors[:3]]
    bottom_sectors = [s[0] for s in sorted_sectors[-3:]]

    return {
        "sector_ic": round(sector_ic, 4),
        "sector_ic_std": round(sector_ic_std, 4),
        "sector_icir": round(sector_icir, 4),
        "n_sectors_valid": valid_sectors,
        "per_sector_ic": {k: round(v, 4) for k, v in per_sector_ic.items()},
        "top_sectors": top_sectors,
        "bottom_sectors": bottom_sectors,
    }


# ── 便捷函数 ──────────────────────────────────────────────

def quick_validate(candidates: List[Dict]) -> Tuple[List[ValidationResult], Dict]:
    """快速验证 (无需库信号)"""
    validator = MultiStageValidator()
    return validator.validate(candidates)


# ── 测试 ──────────────────────────────────────────────────

if __name__ == "__main__":
    # 模拟测试
    mock_candidates = [
        {
            "factor_name": "test_momentum_strong",
            "formula": "ts_delta(close, 20) / ts_std(close, 20)",
            "paradigm": "动量反转",
            "hypothesis": "20日动量有预测力",
            "ic": 0.045,
            "icir": 0.52,
            "max_corr": 0.35,
            "max_corr_factor": "existing_momentum_v1",
        },
        {
            "factor_name": "test_liquidity_weak",
            "formula": "ts_mean(volume, 5)",
            "paradigm": "资金流",
            "ic": 0.008,
            "icir": 0.12,
            "max_corr": 0.55,
        },
        {
            "factor_name": "test_high_corr_reject",
            "formula": "ts_corr(close, volume, 10)",
            "paradigm": "流动性×微观结构",
            "ic": 0.035,
            "icir": 0.40,
            "max_corr": 0.72,
            "max_corr_factor": "existing_volume_corr",
        },
    ]

    results, summary = quick_validate(mock_candidates)
    MultiStageValidator().print_summary(results, summary)
