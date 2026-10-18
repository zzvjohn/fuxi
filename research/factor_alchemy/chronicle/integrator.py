"""
Chronicle 集成器 — 从 run_fa.py 各阶段收集数据并写入 Chronicle
==============================================================
在 run_fa.py 中:
    from chronicle.integrator import ChronicleIntegrator
    ci = ChronicleIntegrator(experiment_id="fa_v7.9_20260714", config={...})
    ci.log_phase3(ic_summary, factor_corr_matrix, ...)
    ci.log_ga(pop_size, generations, pareto_front, ...)
    ci.log_phase5(best_weights_all, portfolio_results_all)
    ci.finish(success=True, duration_s=elapsed)
"""
import time
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Any

from chronicle.db import ExperimentChronicle


class ChronicleIntegrator:
    """将 run_fa.py 的实验数据自动写入 Chronicle"""

    def __init__(self, experiment_id: str, config: dict = None,
                 db_path: str = None):
        self.experiment_id = experiment_id
        self.chronicle = ExperimentChronicle(db_path)
        self.chronicle.start_experiment(experiment_id, config or {})
        self.phase_timings = {}
        self._phase_start = time.time()
        self._started_at = datetime.now()

    # ── Phase 时间追踪 ──

    def mark_phase(self, phase_name: str):
        """标记一个 Phase 完成，记录耗时"""
        elapsed = time.time() - self._phase_start
        self.phase_timings[phase_name] = elapsed
        self.chronicle.update_phase_timing(self.experiment_id, phase_name, elapsed)
        self._phase_start = time.time()  # 重置下一个 Phase

    # ── Phase 3: 单因子评估 ──

    def log_phase3(self, ic_summary: pd.DataFrame,
                   factor_corr_matrix: pd.DataFrame = None,
                   stability_scores: dict = None,
                   screening_results: dict = None):
        """
        ic_summary: DataFrame with index=factor_id, columns包括:
            ICIR, IC_mean, IC_std, rank_autocorr, top_decile_retention,
            cv_ic, ic_half_life, estimated_turnover, decile_monotonicity,
            wfa_survival_rate
        """
        records = []
        for factor_id, row in ic_summary.iterrows():
            # 安全转换: 处理 NaN/None 值
            def _sf(val, default=0.0):
                """safe float: None/NaN → default"""
                if val is None:
                    return default
                try:
                    return float(val)
                except (ValueError, TypeError):
                    return default

            records.append({
                "factor_id": str(factor_id),
                "category": str(row.get("category", "")),
                "icir_mean": _sf(row.get("ICIR")),
                "icir_std": _sf(row.get("ICIR_std")),
                "ic_mean": _sf(row.get("IC_mean")),
                "rank_autocorr": _sf(row.get("rank_autocorr")),
                "top_decile_retention": _sf(row.get("top_decile_retention")),
                "cv_ic": _sf(row.get("cv_ic")),
                "ic_half_life": _sf(row.get("ic_half_life")),
                "estimated_turnover": _sf(row.get("estimated_turnover")),
                "decile_monotonicity": _sf(row.get("decile_monotonicity")),
                "wfa_survival_rate": _sf(row.get("wfa_survival_rate")),
                "regime_bull_icir": _sf(row.get("regime_bull_icir")) if pd.notna(row.get("regime_bull_icir", np.nan)) else None,
                "regime_bear_icir": _sf(row.get("regime_bear_icir")) if pd.notna(row.get("regime_bear_icir", np.nan)) else None,
                "regime_volatile_icir": _sf(row.get("regime_volatile_icir")) if pd.notna(row.get("regime_volatile_icir", np.nan)) else None,
                "regime_calm_icir": _sf(row.get("regime_calm_icir")) if pd.notna(row.get("regime_calm_icir", np.nan)) else None,
                "passed_screening": screening_results.get(str(factor_id), False) if screening_results else False,
                "status": "active",
            })

        count = self.chronicle.log_factor_performances(self.experiment_id, records)
        print(f"  [Chronicle] Phase 3: 写入 {count} 个因子性能")
        self.mark_phase("Phase3_单因子评估")

    # ── Phase 4: NSGA-II ──

    def log_ga(self, pop_size: int, generations: int, n_factors: int,
               pareto_front: dict, convergence_q: list = None,
               convergence_s: list = None, convergence_c: list = None,
               ga_duration_s: float = 0):
        """写入 NSGA-II 运行结果"""
        pf_objs = pareto_front.get("objectives", np.empty((0, 3)))
        if len(pf_objs) > 0:
            q_min, q_max = float(pf_objs[:, 0].min()), float(pf_objs[:, 0].max())
            s_min, s_max = float(pf_objs[:, 1].min()), float(pf_objs[:, 1].max())
            c_min, c_max = float(pf_objs[:, 2].min()), float(pf_objs[:, 2].max())
        else:
            q_min = q_max = s_min = s_max = c_min = c_max = 0

        self.chronicle.log_ga_run(self.experiment_id, {
            "pop_size": pop_size,
            "generations": generations,
            "n_factors": n_factors,
            "final_pareto_size": len(pf_objs),
            "pareto_q_range": f"[{q_min:.3f}, {q_max:.3f}]",
            "pareto_s_range": f"[{s_min:.3f}, {s_max:.3f}]",
            "pareto_c_range": f"[{c_min:.3f}, {c_max:.3f}]",
            "convergence_q": convergence_q or [],
            "convergence_s": convergence_s or [],
            "convergence_c": convergence_c or [],
            "ga_duration_s": ga_duration_s,
        })
        print(f"  [Chronicle] Phase 4: GA 完成, Pareto={len(pf_objs)} 解")
        self.mark_phase("Phase4_NSGA-II")

    # ── Phase 4.5 + 5: 策略评估 & 组合 ──

    def log_strategies(self, best_weights_all: dict,
                        portfolio_results_all: dict = None):
        """
        best_weights_all: {strat_name: {weights, composite, icir, stab, estimated_to, ...}}
        portfolio_results_all: {strat_name: {stats: {total_return, cagr, sharpe, ...}}}
        """
        strat_labels = {
            "high_quality": "高质量(Q↑)",
            "high_stable": "高稳定(S↑)",
            "high_sharpe": "高收益(C↑)",
            "balanced": "均衡(调和)",
        }

        for strat_name, info in best_weights_all.items():
            w = info.get("weights", {})
            icir_data = info.get("icir", {})
            stab_data = info.get("stab", {})
            port = (portfolio_results_all or {}).get(strat_name, {})
            port_stats = port.get("stats", {})

            self.chronicle.log_strategy_result(self.experiment_id, strat_name, {
                "factor_weights": {k: float(v) for k, v in w.items()},
                "quality_obj": float(stab_data.get("icir_quality_score", 0) or 0),
                "stability_obj": float(stab_data.get("total_stability_score", 0) or 0),
                "sharpe_obj": float(info.get("estimated_sharpe", 0) or 0),
                "icir_train": float(icir_data.get("icir", 0) or 0),
                "estimated_turnover": float(info.get("estimated_to", 0) or 0),
                "turnover_penalty": float(info.get("turnover_penalty", 0) or 0),
                "passed_dimensions": int(info.get("passed_count", 0) or 0),
                "phase5_total_return": float(port_stats.get("total_return")) if port_stats.get("total_return") is not None else None,
                "phase5_cagr": float(port_stats.get("cagr")) if port_stats.get("cagr") is not None else None,
                "phase5_sharpe": float(port_stats.get("sharpe")) if port_stats.get("sharpe") is not None else None,
                "phase5_max_dd": float(port_stats.get("max_drawdown")) if port_stats.get("max_drawdown") is not None else None,
                "phase5_avg_turnover": float(port_stats.get("avg_turnover")) if port_stats.get("avg_turnover") is not None else None,
            })
            print(f"  [Chronicle] Phase 4.5: 策略 {strat_name} 写入")

        self.mark_phase("Phase5_虚拟组合")

    # ── Phase 5.5 + 7: 全样本复核 + WF ──

    def log_wf_results(self, strategy_results: dict):
        """更新策略的 WF 验证结果"""
        for strat_name, result in strategy_results.items():
            self.chronicle.log_strategy_result(self.experiment_id, strat_name, {
                "phase55_icir": result.get("phase55_icir"),
                "phase56_sharpe_2026": result.get("phase56_sharpe_2026"),
                "oos_icir_mean": result.get("oos_icir_mean"),
                "wf_anchored_score": result.get("wf_anchored_score"),
                "wf_rolling_score": result.get("wf_rolling_score"),
                "overfitting_risk": result.get("overfitting_risk", "UNKNOWN"),
            })
        print(f"  [Chronicle] Phase 7: WF 结果写入 ({len(strategy_results)} 策略)")

    # ── 完成 ──

    def finish(self, success: bool = True, error_msg: str = "",
               extra_notes: dict = None, sync_to_vault: bool = True):
        """标记实验完成, 可选同步到 Vault"""
        total_duration = (datetime.now() - self._started_at).total_seconds()
        self.chronicle.complete_experiment(
            self.experiment_id,
            success=success,
            error_msg=error_msg,
            phase_timings=self.phase_timings,
            duration_s=total_duration,
        )
        status = "SUCCESS" if success else "FAILED"
        print(f"\n  [Chronicle] 实验 {self.experiment_id} {status} "
              f"(总耗时 {total_duration/60:.1f}min)")

        # ★ Phase 1 集成: 自动同步到 Vault
        if sync_to_vault and success:
            try:
                from chronicle.bridge import create_bridge
                bridge = create_bridge()
                result = bridge.full_sync(self.experiment_id)
                print(f"  [Bridge] → Vault 同步完成: "
                      f"{result['synced_factors']} 因子 "
                      f"(+{len(result.get('new_factors',[]))}新 "
                      f"Δ{len(result.get('updated_factors',[]))}更)")
            except Exception as _be:
                print(f"  [Bridge] → Vault 同步失败 (非致命): {_be}")
