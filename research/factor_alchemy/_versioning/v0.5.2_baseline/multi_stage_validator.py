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

    # Stage 2
    s2_max_corr: float = 0.0     # 与库的最大相关系数
    s2_max_corr_factor: str = ""
    s2_passed: bool = False
    s2_reason: str = ""

    # Stage 3
    s3_max_batch_corr: float = 0.0  # 批次内最大相关系数
    s3_passed: bool = False
    s3_reason: str = ""

    # Stage 4
    s4_oos_ic: float = 0.0
    s4_oos_icir: float = 0.0
    s4_replacement: bool = False    # 是否替换了库中已有因子
    s4_replaced_factor: str = ""
    s4_passed: bool = False
    s4_reason: str = ""

    # Stage 5 (v0.3 新增): 联合正向过滤 + Calmar
    s5_year1_excess: float = 0.0    # 年度1超额收益
    s5_year2_excess: float = 0.0    # 年度2超额收益
    s5_calmar: float = 0.0          # Calmar比率
    s5_passed: bool = False
    s5_reason: str = ""

    # Overall
    final_grade: str = ""          # A/B/C/D
    total_stages_passed: int = 0
    eligible_for_jq: bool = False  # 是否提交 JQ

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
            "s1": {"ic": self.s1_ic, "icir": self.s1_icir, "passed": self.s1_passed, "reason": self.s1_reason},
            "s2": {"max_corr": self.s2_max_corr, "max_corr_factor": self.s2_max_corr_factor, "passed": self.s2_passed, "reason": self.s2_reason},
            "s3": {"max_batch_corr": self.s3_max_batch_corr, "passed": self.s3_passed, "reason": self.s3_reason},
            "s4": {"oos_ic": self.s4_oos_ic, "oos_icir": self.s4_oos_icir, "replacement": self.s4_replacement, "passed": self.s4_passed, "reason": self.s4_reason},
            "s5": {"year1_excess": self.s5_year1_excess, "year2_excess": self.s5_year2_excess, "calmar": self.s5_calmar, "passed": self.s5_passed, "reason": self.s5_reason},
            "ic_boot": {"mean": round(self.ic_boot_mean, 5), "ci95_low": round(self.ic_ci95_low, 5), "ci95_high": round(self.ic_ci95_high, 5), "n_obs": self.ic_boot_n},
            "delta_lag": {"ic_lag1": round(self.s1_ic_lag1, 5), "delta_lag": round(self.delta_lag, 5), "high_decay_risk": self.high_decay_risk},
            "final_grade": self.final_grade,
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
        icir_threshold: float = 0.3,
        corr_threshold: float = 0.5,
        batch_corr_threshold: float = 0.7,
        replacement_k: float = 1.1,   # 新因子ICIR ≥ k × 旧因子ICIR才替换
        # S5 参数 (v0.3 新增)
        calmar_threshold: float = 1.0,       # Calmar比率最低阈值
        year1_label: str = "2025",           # 年度1标签
        year2_label: str = "2026",           # 年度2标签
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
        self._library_signals: Optional[Dict[str, np.ndarray]] = None

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
        """
        for i, cand in enumerate(candidates):
            ic = abs(float(cand.get("ic", cand.get("daily_ic", 0)) or 0))
            icir = abs(float(cand.get("icir", cand.get("daily_icir", 0)) or 0))

            if ic == 0 and icir == 0:
                # 无预计算 IC — 拒绝（没有real数据不能信任）
                results[i].s1_passed = False
                results[i].s1_reason = "❌ 无IC数据（status=reserve/未评估）— Stage1拒绝"
                continue

            results[i].s1_ic = ic
            results[i].s1_icir = icir

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
                    fails.append(f"IC={ic:.4f}<{self.ic_threshold}")
                if icir < self.icir_threshold:
                    fails.append(f"ICIR={icir:.3f}<{self.icir_threshold}")
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

            # 优先使用候选自带的 max_corr
            max_corr = float(cand.get("max_corr", cand.get("max_correlation", 0)) or 0)
            max_corr_factor = cand.get("max_corr_factor", cand.get("max_correlation_factor", ""))

            # 如果库信号可用，重新计算
            if library_signals and cand.get("signal"):
                cand_signal = np.array(cand["signal"])
                for lib_name, lib_signal in library_signals.items():
                    valid = ~(np.isnan(cand_signal) | np.isnan(lib_signal))
                    if valid.sum() < 20:
                        continue
                    corr = abs(np.corrcoef(cand_signal[valid], lib_signal[valid])[0, 1])
                    if corr > max_corr:
                        max_corr = corr
                        max_corr_factor = lib_name

            results[i].s2_max_corr = max_corr
            results[i].s2_max_corr_factor = max_corr_factor

            if max_corr <= self.corr_threshold:
                results[i].s2_passed = True
                results[i].s2_reason = f"max_corr={max_corr:.3f} (≤{self.corr_threshold}) ✅"
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

        if len(passed_idx) <= 1:
            for i in passed_idx:
                results[i].s3_passed = True
                results[i].s3_reason = "批次内无需去重 (≤1个候选)"
            return results

        # 按 ICIR 排序 (高优先)
        passed_idx.sort(key=lambda i: results[i].s1_icir, reverse=True)

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
                if sig_a is not None and sig_b is not None:
                    valid = ~(np.isnan(sig_a) | np.isnan(sig_b))
                    if valid.sum() >= 20:
                        corr = abs(np.corrcoef(sig_a[valid], sig_b[valid])[0, 1])
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

        for i in passed_idx:
            if results[i].s3_reason == "":
                results[i].s3_passed = True
                results[i].s3_reason = "批次内无高相关冗余 ✅"

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
        for i, cand in enumerate(candidates):
            if not all([results[i].s1_passed, results[i].s2_passed, results[i].s3_passed]):
                continue

            # OOS 验证 (使用候选自带的 OOS 指标)
            oos_ic = abs(float(cand.get("oos_ic", cand.get("ic", 0)) or 0))
            oos_icir = abs(float(cand.get("oos_icir", cand.get("icir", 0)) or 0))
            results[i].s4_oos_ic = oos_ic
            results[i].s4_oos_icir = oos_icir

            # 替换检查
            max_corr_factor = results[i].s2_max_corr_factor
            if max_corr_factor and library_factors and max_corr_factor in library_factors:
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
                results[i].s4_reason = "无替换需求或库中无对应因子 ✅"

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
            else:
                # 使用预计算数据 (或默认值回退)
                self._apply_stage5_from_data(cand, results, i)

        # 批量 S5 实时计算
        if real_compute_candidates and s5_filter is not None:
            print(f"  [S5] Computing real joint filter for {len(real_compute_candidates)} candidates...")
            s5_results = s5_filter.validate_batch(real_compute_candidates, verbose=True)
            for j, s5r in enumerate(s5_results):
                idx = real_compute_indices[j]
                self._apply_stage5_from_s5result(s5r, results, idx)

        return results

    def _apply_stage5_from_data(
        self, cand: Dict, results: List[ValidationResult], i: int
    ):
        """从预计算数据应用 S5 过滤（兼容旧格式）"""
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
        results[i].s5_year1_excess = s5r.excess_2025
        results[i].s5_year2_excess = s5r.excess_2026
        results[i].s5_calmar = max(s5r.calmar_2025, s5r.calmar_2026)
        results[i].s5_passed = s5r.passed
        results[i].s5_reason = s5r.reason

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
                formula=c.get("formula", ""),
                paradigm=c.get("paradigm", ""),
                hypothesis=c.get("hypothesis", ""),
            )
            for i, c in enumerate(candidates)
        ]

        # Stage 1
        results = self.stage1_fast_ic(candidates, results)

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

        # 综合评分
        # v0.5.2: 轻量 S5 时禁止绕过 — 必须 S5 通过才 eligible
        for r in results:
            stages_passed = sum([r.s1_passed, r.s2_passed, r.s3_passed, r.s4_passed, r.s5_passed])
            r.total_stages_passed = stages_passed
            if s5_lightweight:
                # 轻量 S5 成本低 → 严格要求 5/5
                r.eligible_for_jq = stages_passed >= 5
            else:
                r.eligible_for_jq = stages_passed >= 4  # 真实 S5 成本高 → 4/5 即可

            if stages_passed == 5:
                r.final_grade = "A"
            elif stages_passed == 4:
                r.final_grade = "B"
            elif stages_passed == 3:
                r.final_grade = "C"
            else:
                r.final_grade = "D"

        # 汇总
        summary = {
            "total_candidates": len(candidates),
            "stage1_passed": sum(1 for r in results if r.s1_passed),
            "stage2_passed": sum(1 for r in results if r.s2_passed),
            "stage3_passed": sum(1 for r in results if r.s3_passed),
            "stage4_passed": sum(1 for r in results if r.s4_passed),
            "stage5_passed": sum(1 for r in results if r.s5_passed),
            "eligible_for_jq": sum(1 for r in results if r.eligible_for_jq),
            "s5_mode": "lightweight_ic" if s5_lightweight else ("real_s5" if s5_filter else "precomputed"),
            "ic_boot_computed": n_boot_computed,      # P-20260828-004 影子统计
            "delta_lag_computed": n_delta_computed,   # P-20260828-001 影子统计
            "delta_lag_high_decay_risk": n_decay_risk,
            "ic_ci_low_below_zero": n_ci_below_zero,  # P-20260828-004 影子统计
            "grade_distribution": {
                "A": sum(1 for r in results if r.final_grade == "A"),
                "B": sum(1 for r in results if r.final_grade == "B"),
                "C": sum(1 for r in results if r.final_grade == "C"),
                "D": sum(1 for r in results if r.final_grade == "D"),
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
        for stage in ["stage1", "stage2", "stage3", "stage4", "stage5"]:
            passed = summary[f"{stage}_passed"]
            bar = "🟢" * passed + "🔴" * (summary["total_candidates"] - passed)
            stage_labels = {
                "stage1": "S1 快速IC",
                "stage2": "S2 库内去重",
                "stage3": "S3 批次去重",
                "stage4": "S4 OOS+替换",
                "stage5": "S5 联合过滤+Calmar",
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
                      f"S1={r.s1_passed} S2={r.s2_passed} S3={r.s3_passed} S4={r.s4_passed} S5={r.s5_passed}")


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
