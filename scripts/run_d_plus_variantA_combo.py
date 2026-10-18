# -*- coding: utf-8 -*-
"""
D+ 蒸馏手动触发 — 变体A 6因子复合 JQ 实测结果注入 (2026-08-21)
================================================================
JQ 实测 (2021-01-01 ~ 2026-07-30, 用户回测, 日志 E:\\log\\variantA_gpevo006_6factor.txt):
  组合: +173.32%, Sharpe 0.685, MaxDD 32.91%
  组成: gpevo006(-open) + overnight5 + tvma20 + dollar20 + tstd_cv + nearhigh
       (MMR 全池搜索冠军, rank 等权, top60 周频)

单因子 JQ IC/ICIR (日志 2021-12-27 首次报告, n=47 周, 早期快照非全周期):
  gpevo006   RankIC=+0.0468 ICIR=+0.2962 +IC%=59.6%
  overnight5 RankIC=+0.0037 ICIR=+0.0489 +IC%=53.2%
  tvma20     RankIC=+0.0376 ICIR=+0.3288 +IC%=59.6%
  dollar20   RankIC=+0.0733 ICIR=+0.4753 +IC%=66.0%
  tstd_cv    RankIC=+0.0149 ICIR=+0.1389 +IC%=53.2%
  nearhigh   RankIC=-0.0098 ICIR=-0.0660 +IC%=46.8%

注意:
  1. v1 执行层存在 order_target_value 差额<100股拒单 bug (3478 笔失败, ~20% 失败率),
     本结果是在执行层缺陷下的绩效; 修复版 v2 已生成, 待重跑裁决。
  2. 单因子 jq_return/jq_sharpe 用组合口径 (成员未单独回测), root_cause 已注明。

用法: python scripts/run_d_plus_variantA_combo.py
"""
import sys
import json
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "research" / "factor_alchemy"))

TRAJ_LOG = ROOT / "data" / "trajectory_log.json"

COMBO_RET = 173.32
COMBO_SR = 0.685
COMBO_MDD = 32.91

# 6 因子: (name, formula, hypothesis, paradigm, category, jq_ic, jq_icir, +ic_pct)
FACTORS = [
    ("gpevo006", "-open_p",
     "低开盘价: 开盘价绝对值低的股票后续收益更高 (GP 育种 gp_evo_006 单因子 JQ +134.3%/SR0.54)",
     "价格形态", "microstructure", 0.0468, 0.2962, 59.6),
    ("overnight5", "(open_p/close_p.shift(1)-1).rolling(5).sum()",
     "隔夜跳空 5 日累计: 持续小幅低开 (跳空负值累积大) 的股票有均值回复空间",
     "隔夜动量", "microstructure", 0.0037, 0.0489, 53.2),
    ("tvma20", "-(close_p*volume_p/((close_p*volume_p).rolling(20).mean())).clip(0,10)",
     "缩量: 20日成交额萎缩 (换手下降) 的股票被低估, 王者底仓成员",
     "流动性", "liquidity", 0.0376, 0.3288, 59.6),
    ("dollar20", "-log((close_p*volume_p).rolling(20).mean())",
     "低成交额: 20日平均成交额对数值取负, 低关注度股票有超额 (A股流动性溢价)",
     "流动性", "liquidity", 0.0733, 0.4753, 66.0),
    ("tstd_cv", "-(volume_p.rolling(20).std()/volume_p.rolling(20).mean())",
     "低量波动: 成交量变异系数取负, 量能稳定的股票风险调整后收益更高",
     "低波动", "low_volatility", 0.0149, 0.1389, 53.2),
    ("nearhigh", "-(close_p.rolling(10).mean()/close_p.rolling(60).max())",
     "远离60日高点: 当前价远离近期高点 (回调充分) 的股票有反转空间",
     "均值回复", "mean_reversion", -0.0098, -0.0660, 46.8),
]


def infer_operators(formula: str):
    ops = []
    kws = {"pct_change": "pct_change", "rolling": "rolling", "rank": "rank",
           "std": "rolling_std", "mean": "rolling_mean", "min": "rolling_min",
           "max": "rolling_max", "clip": "clip", "log": "log",
           "shift": "shift", "abs": "abs", "/": "division", "*": "multiplication",
           "+": "addition", "-": "subtraction"}
    for kw, op in kws.items():
        if kw in formula.lower():
            ops.append(op)
    return ops


def build_jq_result() -> dict:
    factors = []
    for name, formula, hypothesis, paradigm, category, jq_ic, jq_icir, ic_pct in FACTORS:
        factors.append({
            "factor_name": name,
            "formula": formula,
            "hypothesis": hypothesis,
            "paradigm": paradigm,
            "category": category,
            "operators_used": infer_operators(formula),
            "local_ic": None,
            "local_icir": None,
            # 组合成员未单独回测, 用组合口径并在 root_cause 注明
            "jq_return": COMBO_RET,
            "jq_sharpe": COMBO_SR,
            "jq_maxdd": COMBO_MDD,
            "jq_ic": jq_ic,
            "jq_icir": jq_icir,
            "jq_ic_note": f"n=47 周早期快照 (2021-12-27), +IC%={ic_pct}%; 全周期报告 v1 缺失(ic_reported 一次性标志 bug), v2 已修复",
            "jq_composite_contribution": "positive" if jq_icir > 0.1 else ("neutral" if jq_icir > -0.1 else "negative"),
            "root_cause": (
                f"变体A 6因子复合成员 (MMR 全池搜索冠军, rank 等权 top60 周频)。"
                f"组合 JQ 实测 {COMBO_RET}%/Sharpe {COMBO_SR}/MaxDD {COMBO_MDD}% "
                f"(2021-01~2026-07)。成员单独回测未跑, 收益口径=组合。"
                f"JQ 单因子 IC {jq_ic:+.4f}/ICIR {jq_icir:+.3f} 为 n=47 周早期快照。"
                f"注意: v1 执行层有 3478 笔 <100股 拒单 (~20% 失败率), "
                f"真实组合绩效以 v2 重跑为准。"
            ),
        })
    return {
        "batch_id": f"variantA_combo_jq_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "timestamp": datetime.now().isoformat(),
        "composite_return": COMBO_RET,
        "composite_sharpe": COMBO_SR,
        "composite_maxdd": COMBO_MDD,
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
    print("  D+ 蒸馏: 变体A 6因子复合 JQ 实测结果注入")
    print("=" * 60)

    jq_result = build_jq_result()
    print(f"  composite: {jq_result['composite_return']}% / "
          f"Sharpe {jq_result['composite_sharpe']} / MDD {jq_result['composite_maxdd']}%")
    for f in jq_result["factors"]:
        print(f"    {f['factor_name']}: JQ IC {f['jq_ic']:+.4f} / ICIR {f['jq_icir']:+.3f} "
              f"| {f['jq_composite_contribution']}")

    from ralph_loop import RalphLoop
    from experience_memory import get_memory

    memory = get_memory()
    rl = RalphLoop(memory=memory)
    result = rl.jq_feedback(jq_backtest_result=jq_result)
    print(json.dumps(result, ensure_ascii=False, indent=1, default=str))

    out = ROOT / "data" / "jq_feedback_variantA_combo.json"
    json.dump({"input": jq_result, "feedback_result": result},
              open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
    print(f"  结果已存: {out}")

    mark_new_traj_d_plus()
    print("D+ 闭环完成")


if __name__ == "__main__":
    main()
