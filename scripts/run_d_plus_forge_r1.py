# -*- coding: utf-8 -*-
"""
D+ 蒸馏手动触发 — Forge Round1 四个 S5 通过因子的 JQ 实测结果注入
================================================================
JQ 实测 (2021-01-01 ~ 2026-07-30, 用户回测, 日志在 E:\\log\\):
  1. forge_r1_vol_zscore_inv_stability        : +44.32%, Sharpe 0.128, maxDD 40.54%, JQ IC +0.0009 / ICIR +0.026
  2. forge_r1_vol_double_zscore_inv_stability : +52.00%, Sharpe 0.172, maxDD 32.58%, JQ IC +0.0019 / ICIR +0.059
  3. forge_r1_open_min_vol_squared            :  -4.10%, Sharpe -0.189, maxDD 45.73%, JQ IC -0.0061 / ICIR -0.072
  4. forge_r1_volume_dual_rank                : -20.11%, Sharpe -0.351, maxDD 50.45%, JQ IC -0.0268 / ICIR -0.303

这 4 个因子来自 forge_round1 实验 (scripts/forge_round1_experiment.py),
不在 trajectory_log 中, 直接构造 jq_backtest_result 调用 RalphLoop.jq_feedback()
(新建轨迹 + form_from_jq + distill_motif + evolve), 完成后标记 d_plus_applied=true。

用法: python scripts/run_d_plus_forge_r1.py
"""
import sys
import json
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "research" / "factor_alchemy"))

TRAJ_LOG = ROOT / "data" / "trajectory_log.json"


def build_jq_result() -> dict:
    def infer_operators(formula: str):
        ops = []
        kws = {"rolling": "rolling", "rank": "rank", "std": "rolling_std",
               "mean": "rolling_mean", "min": "rolling_min", "sum": "rolling_sum",
               "sqrt": "sqrt", "abs": "abs", "/": "division", "*": "multiplication",
               "+": "addition", "-": "subtraction", ">": "comparison", "1e-6": "epsilon_guard"}
        for kw, op in kws.items():
            if kw in formula.lower():
                ops.append(op)
        return ops

    factors = [
        {
            "factor_name": "forge_r1_vol_zscore_inv_stability",
            "formula": "(1/(zscore12(volume)+1e-6)).rolling(20).std().rolling(5).mean().rolling(5).sum()",
            "hypothesis": "成交量 12日 zscore 倒数的波动平稳度: 量能平稳(低zscore倒数波动)的股票后续更优",
            "paradigm": "筹码分布",
            "category": "chip_distribution",
            "operators_used": infer_operators("(1/((volume-mean12)/std12+1e-6)).rolling(20).std().rolling(5).mean().rolling(5).sum()"),
            "local_ic": None,
            "local_icir": None,
            "forge_icir": 0.157,
            "s5_excess_2025": 3.95, "s5_excess_2026": 4.06,
            "s5_calmar_2025": 1.29, "s5_calmar_2026": 2.17,
            "jq_return": 44.32,
            "jq_sharpe": 0.128,
            "jq_maxdd": 40.54,
            "jq_ic": 0.0009,
            "jq_icir": 0.0261,
            "jq_composite_contribution": "neutral",
            "root_cause": (
                "JQ IC +0.0009≈0, 选股 alpha 归零。收益 +44.32% 主要来自小盘/量价风格 beta 暴露。"
                "local S5 excess 3.95%/4.06% (Calmar 1.29/2.17) 未能在 JQ 上转化为截面 alpha。"
                "Sharpe 0.128 远低于阈值, 评级 MARGINAL。"
                "量能平稳性信号在 A 股小盘池的截面区分力不足 — 方向正确(收益为正)但无 alpha。"
            ),
        },
        {
            "factor_name": "forge_r1_vol_double_zscore_inv_stability",
            "formula": "(1/(zscore12(zscore40(volume))+1e-6)).rolling(20).std().rolling(5).mean().rolling(5).sum()",
            "hypothesis": "成交量 40日 zscore 的再 zscore(双重标准化) 倒数的波动平稳度",
            "paradigm": "筹码分布",
            "category": "chip_distribution",
            "operators_used": infer_operators("(1/(((volume-mean40)/std40-mean12)/std12+1e-6)).rolling(20).std().rolling(5).mean().rolling(5).sum()"),
            "local_ic": None,
            "local_icir": None,
            "forge_icir": 0.206,
            "s5_excess_2025": 3.31, "s5_excess_2026": 3.38,
            "s5_calmar_2025": 1.30, "s5_calmar_2026": 2.79,
            "jq_return": 52.00,
            "jq_sharpe": 0.172,
            "jq_maxdd": 32.58,
            "jq_ic": 0.0019,
            "jq_icir": 0.0590,
            "jq_composite_contribution": "neutral",
            "root_cause": (
                "JQ IC +0.0019≈0 — 双重 zscore 平稳性与单重版本无本质差异, 均为 beta 收益。"
                "收益 +52% 是 4 因子中最高但 Sharpe 0.172 仍极低 (未达 >0.4 阈值), 评级 MARGINAL。"
                "maxDD 32.58% 是 4 因子中最低, 说明量能平稳组合有小盘低波风格暴露。"
                "结论: Forge 进化出的'双重标准化'变体未产生增量 alpha, 复杂化无效。"
            ),
        },
        {
            "factor_name": "forge_r1_open_min_vol_squared",
            "formula": "(zscore60(sqrt(rolling_min(open,30)).rolling(20).std()))**2",
            "hypothesis": "开盘价 30日最低值的 20日波动率 zscore 平方: 开盘价下限波动加速信号",
            "paradigm": "筹码分布",
            "category": "chip_distribution",
            "operators_used": infer_operators("((sqrt(max(rolling_min(open,30),0)).rolling(20).std()-mean60)/std60)**2"),
            "local_ic": None,
            "local_icir": None,
            "forge_icir": 0.206,
            "s5_excess_2025": 4.87, "s5_excess_2026": 5.52,
            "s5_calmar_2025": 2.25, "s5_calmar_2026": 2.14,
            "jq_return": -4.10,
            "jq_sharpe": -0.189,
            "jq_maxdd": 45.73,
            "jq_ic": -0.0061,
            "jq_icir": -0.0724,
            "jq_composite_contribution": "negative",
            "root_cause": (
                "local S5 最强因子 (excess 4.87%/5.52%, Calmar 2.25/2.14, 4因子中最高) → JQ 收益 -4.1%, IC -0.006 方向反转。"
                "又一次 'local 上优化越狠 JQ 越差' 的铁律实证: S5 在周频数据上对 Calmar 的筛选等价于对噪声的过拟合。"
                "zscore 平方构造(尾部放大) 在日频执行下信号衰减归零。"
                "教训: S5 Calmar 排名靠前 ≠ JQ 生存概率高, Forge 的 S5 质量门把最'好看'的因子留下, 恰好是 JQ 上最差的。"
            ),
        },
        {
            "factor_name": "forge_r1_volume_dual_rank",
            "formula": "(3+volume).rolling(30).rank(pct=True).rolling(26).rank(pct=True)",
            "hypothesis": "成交量 30日分位的再 26日分位: 量能相对位置的二阶趋势",
            "paradigm": "尾部风险",
            "category": "tail_risk",
            "operators_used": infer_operators("(3+volume).rolling(30).rank(pct=True).rolling(26).rank(pct=True)"),
            "local_ic": None,
            "local_icir": None,
            "forge_icir": 0.0,
            "s5_excess_2025": 3.93, "s5_excess_2026": 3.20,
            "s5_calmar_2025": 1.34, "s5_calmar_2026": 2.58,
            "jq_return": -20.11,
            "jq_sharpe": -0.351,
            "jq_maxdd": 50.45,
            "jq_ic": -0.0268,
            "jq_icir": -0.3033,
            "jq_composite_contribution": "negative",
            "root_cause": (
                "JQ IC -0.0268 / ICIR -0.303, 4 因子中最强负信号 — 双重 rank 量能趋势在 JQ 上是明确的反向指标。"
                "forge_icir=0.0 已预示该因子 local 无信号, S5 excess 3.93%/3.20% 是 S5 窗口过拟合产物。"
                "收益 -20.11% 触发硬禁止阈值。"
                "教训: Forge 引擎中 ICIR=0 的候选即使 S5 通过也必须拦截 — 质量门缺少'ICIR 下限'导致漏网。"
            ),
        },
    ]

    returns = [f["jq_return"] for f in factors]
    sharpes = [f["jq_sharpe"] for f in factors]
    maxdds = [f["jq_maxdd"] for f in factors]
    return {
        "batch_id": f"forge_r1_jq_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "timestamp": datetime.now().isoformat(),
        "composite_return": round(sum(returns) / len(returns), 2),
        "composite_sharpe": round(sum(sharpes) / len(sharpes), 3),
        "composite_maxdd": round(sum(maxdds) / len(maxdds), 2),
        "factors": factors,
    }


def mark_new_traj_d_plus():
    data = json.load(open(TRAJ_LOG, encoding="utf-8"))
    now = datetime.now().isoformat()
    count = 0
    for t in data.get("trajectories", []):
        if t.get("phase") == "jq_feedback" and not t.get("d_plus_applied"):
            t["d_plus_applied"] = True
            t["d_plus_at"] = now[:19]
            count += 1
    data["updated_at"] = now
    json.dump(data, open(TRAJ_LOG, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"  标记 {count} 条新轨迹 d_plus_applied=true")
    return count


def main():
    print("=" * 60)
    print("  D+ 蒸馏: Forge Round1 S5 通过因子 JQ 实测结果注入")
    print("=" * 60)

    jq_result = build_jq_result()
    print(f"  composite: {jq_result['composite_return']}% / "
          f"Sharpe {jq_result['composite_sharpe']} / MDD {jq_result['composite_maxdd']}%")
    for f in jq_result["factors"]:
        print(f"    {f['factor_name']}: JQ {f['jq_return']:+.2f}% / {f['jq_sharpe']:+.3f} "
              f"/ MDD {f['jq_maxdd']}% | JQ IC {f['jq_ic']:+.4f} | {f['jq_composite_contribution']}")

    from ralph_loop import RalphLoop
    from experience_memory import get_memory

    memory = get_memory()
    rl = RalphLoop(memory=memory)
    result = rl.jq_feedback(jq_backtest_result=jq_result)

    marked = mark_new_traj_d_plus()

    print(f"\n{'='*60}")
    print(f"  D+ 完成")
    print(f"    轨迹更新:      {result.get('trajectories_updated', 0)}")
    print(f"    硬禁止方向:    +{result.get('hard_forbidden_added', 0)}")
    print(f"    软负警告:      +{result.get('soft_warnings_added', 0)}")
    print(f"    JQ确认成功:    +{result.get('jq_success_confirmed', 0)}")
    print(f"    d_plus 标记:   {marked}")
    print(f"    Memory 记录:   {memory.data['stats'].get('total_attempts')}")
    print(f"{'='*60}")

    summary = {
        "batch_id": jq_result["batch_id"],
        "composite": {
            "return": jq_result["composite_return"],
            "sharpe": jq_result["composite_sharpe"],
            "maxdd": jq_result["composite_maxdd"],
        },
        "result": {k: result.get(k) for k in
                   ("trajectories_updated", "hard_forbidden_added",
                    "soft_warnings_added", "jq_success_confirmed")},
        "factors": [
            {"factor_name": f["factor_name"], "jq_return": f["jq_return"],
             "jq_sharpe": f["jq_sharpe"], "jq_ic": f["jq_ic"],
             "jq_icir": f["jq_icir"],
             "contribution": f["jq_composite_contribution"]}
            for f in jq_result["factors"]
        ],
    }
    out = ROOT / "data" / "jq_feedback_forge_r1_batch.json"
    json.dump(summary, open(out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"  摘要已存: {out.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
