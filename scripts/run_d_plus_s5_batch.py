# -*- coding: utf-8 -*-
"""
D+ 蒸馏手动触发 — Stage2 三个 S5 通过因子的 JQ 实测结果注入
================================================================
JQ 实测 (2021-01-01 ~ 2026-07-30, 用户回测, 日志在 E:\\log\\):
  1. event_driven_convexity_fade : +60.26%, Sharpe 0.219, maxDD 32.47%, JQ IC +0.052 / ICIR +0.599
  2. max_drawdown_duration      : -39.93%, Sharpe -0.587, maxDD 53.37%, JQ IC -0.026 / ICIR -0.151
  3. small_cap_liquidity_quality: +40.28%, Sharpe 0.109, maxDD 36.85%, JQ IC +0.0009 / ICIR +0.031

这 3 个因子来自 Stage2 因子池 (passed_factor_pool.csv), 不在 trajectory_log 中,
故不走 trigger_d_plus.py 的轨迹扫描, 直接构造 jq_backtest_result 调用
RalphLoop.jq_feedback() (新建轨迹 + form_from_jq + distill_motif + evolve),
完成后为新轨迹标记 d_plus_applied=true。

用法: python scripts/run_d_plus_s5_batch.py
"""
import sys
import json
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "research" / "factor_alchemy"))

META = json.load(open(ROOT / "_dplus_tmp_meta.json", encoding="utf-8"))
TRAJ_LOG = ROOT / "data" / "trajectory_log.json"


def build_jq_result() -> dict:
    def infer_operators(formula: str):
        ops = []
        kws = {"pct_change": "pct_change", "rolling": "rolling", "rank": "rank",
               "std": "rolling_std", "mean": "rolling_mean", "min": "rolling_min",
               "max": "rolling_max", "clip": "clip", "fillna": "fillna",
               "abs": "abs", "/": "division", "*": "multiplication",
               "+": "addition", "-": "subtraction", ">": "comparison"}
        for kw, op in kws.items():
            if kw in formula.lower():
                ops.append(op)
        return ops

    factors = [
        {
            "factor_name": "event_driven_convexity_fade",
            "formula": META["event_driven_convexity_fade"]["formula"],
            "hypothesis": META["event_driven_convexity_fade"]["hypothesis"],
            "paradigm": "事件驱动凸性退潮",
            "category": "sentiment",
            "operators_used": infer_operators(META["event_driven_convexity_fade"]["formula"]),
            "local_ic": META["event_driven_convexity_fade"]["ic_mean"],
            "local_icir": META["event_driven_convexity_fade"]["icir"],
            "jq_return": 60.26,
            "jq_sharpe": 0.219,
            "jq_maxdd": 32.47,
            "jq_ic": 0.0521,
            "jq_icir": 0.5992,
            "jq_composite_contribution": "positive",
            "root_cause": (
                "JQ IC +0.052/ICIR +0.599 方向一致且强于 local IC +0.028 — 逆拥挤退潮在JQ真实环境有效。"
                "收益 +60.26% 为正但 Sharpe 0.219 低 (拥挤崩盘信号本身高波动), "
                "未达 JQ_PASSED 阈值 (jq_ret>50 且 sharpe>0.4), 评级 WEAK。"
                "是 3 个 Stage2 因子中唯一 local 与 JQ 方向一致的正案例。"
            ),
        },
        {
            "factor_name": "max_drawdown_duration",
            "formula": META["max_drawdown_duration"]["formula"],
            "hypothesis": META["max_drawdown_duration"]["hypothesis"],
            "paradigm": "极端事件韧性",
            "category": "extreme_events",
            "operators_used": infer_operators(META["max_drawdown_duration"]["formula"]),
            "local_ic": META["max_drawdown_duration"]["ic_mean"],
            "local_icir": META["max_drawdown_duration"]["icir"],
            "jq_return": -39.93,
            "jq_sharpe": -0.587,
            "jq_maxdd": 53.37,
            "jq_ic": -0.0248,
            "jq_icir": -0.1513,
            "jq_composite_contribution": "negative",
            "root_cause": (
                "方向完全反转: local IC +0.125/ICIR 1.67 (3因子中最强) → JQ IC -0.026/ICIR -0.151。"
                "关键根因是口径差异: local S5 验证在周频数据上 rolling(120)=120周≈2.4年高点贴近度, "
                "JQ 日频执行 rolling(120)=120天, 信号语义根本不同。"
                "高 local ICIR 因子在 JQ 上崩塌 — local→JQ gap 放大器 (FRI ρ=+0.22 规律再证)。"
                "教训: 带长窗口(>=60)的因子送 JQ 前必须确认周频→日频窗口换算, 否则验证与执行不是同一信号。"
            ),
        },
        {
            "factor_name": "small_cap_liquidity_quality",
            "formula": META["small_cap_liquidity_quality"]["formula"],
            "hypothesis": META["small_cap_liquidity_quality"]["hypothesis"],
            "paradigm": "小盘流动性共振",
            "category": "small_bull_breakthrough",
            "operators_used": infer_operators(META["small_cap_liquidity_quality"]["formula"]),
            "local_ic": META["small_cap_liquidity_quality"]["ic_mean"],
            "local_icir": META["small_cap_liquidity_quality"]["icir"],
            "jq_return": 40.28,
            "jq_sharpe": 0.109,
            "jq_maxdd": 36.85,
            "jq_ic": 0.0009,
            "jq_icir": 0.0308,
            "jq_composite_contribution": "neutral",
            "root_cause": (
                "JQ IC +0.0009≈0 — 选股 alpha 归零, 与 local IC +0.030 一致地弱。"
                "收益 +40.28% 几乎全部来自小盘 beta 暴露 (2021-2026 小盘风格), 非因子 alpha。"
                "Sharpe 0.109 极低, 评级 MARGINAL。"
                "教训: Amihud 改善×动量门在小盘域内无增量选股力, 与 '流动性×微观结构是A股alpha主源' 不矛盾 — "
                "但需要中大盘池或更强的截面条件, 小盘 beta 掩盖了 alpha 缺失。"
            ),
        },
    ]

    returns = [f["jq_return"] for f in factors]
    sharpes = [f["jq_sharpe"] for f in factors]
    maxdds = [f["jq_maxdd"] for f in factors]
    return {
        "batch_id": f"s5_stage2_jq_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "timestamp": datetime.now().isoformat(),
        "composite_return": round(sum(returns) / len(returns), 2),
        "composite_sharpe": round(sum(sharpes) / len(sharpes), 3),
        "composite_maxdd": round(sum(maxdds) / len(maxdds), 2),
        "factors": factors,
    }


def mark_new_traj_d_plus():
    """jq_feedback 新建的轨迹 (phase=jq_feedback) 标记 d_plus_applied=true"""
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
    print("  D+ 蒸馏: Stage2 S5 通过因子 JQ 实测结果注入")
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

    # 输出精简摘要供归档
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
    out = ROOT / "data" / "jq_feedback_s5_stage2_batch.json"
    json.dump(summary, open(out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"  摘要已存: {out.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
