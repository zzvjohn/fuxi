# -*- coding: utf-8 -*-
"""
D+ 蒸馏手动触发 — 情绪×日内 5因子复合 JQ 实测注入 (2026-08-23)
==================================================================
JQ 实测 (2021-01-01 ~ 2026-07-30, 用户回测, 文件 jq_s5_pass_sentiment_intraday_composite_standalone.py):
  复合组合: +131.41%, Sharpe 0.588, MaxDD 32.03%
  (5因子 = overnight_raw/overnight5/intraday_rev/lhb_freq_20d/overreaction,
   等权 rank和, top30/池800/周频周一open, lookback 30)

裁决: PASSED (Sharpe 0.588 >= 0.5 立项门槛) — 情绪×日内族首个成员成立。
  1. 复合有效: 显著强于组成单因子 (overreaction +75.47%/0.396, lhb_freq_20d +64.59%/0.093);
  2. 族间定位: 弱于价量族最优形态 (打散v1 +157.81%/0.69) 但族不同 (情绪×日内 vs 价量),
     提供低相关增量维度, 值得入库扩展 MMR 候选空间;
  3. 日频语境回响: 该复合在日频框架因换手成本证伪 (P0.5: 单边日换手76%), 周频是其可存活形态;
  4. 模板可复用: jq_s5_pass 执行层 + 短持有族公式可批量生成同类单因子。

成员口径说明:
  - overreaction: 单因子 JQ 实测 +75.47%/0.396/26% (MARGINAL), 用真实单因子口径;
  - lhb_freq_20d: 单因子 JQ 实测 +64.59%/0.093 (MARGINAL), 用真实单因子口径;
  - overnight5: King core4 成员 (组合 +182.57%/0.69), 无独立单因子验证, 组合口径注明;
  - overnight_raw / intraday_rev: 本次首次入组合, 组合口径注明。

用法: python scripts/run_d_plus_sentiment_combo.py
"""
import sys
import json
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "research" / "factor_alchemy"))

TRAJ_LOG = ROOT / "data" / "trajectory_log.json"
CHAMPIONS = ROOT / "data" / "jq_validated_champions.json"

COMBO_RET = 131.41
COMBO_SR = 0.588
COMBO_MDD = 32.03

LESSON = (
    "情绪×日内族首个成员成立: 5因子等权rank和 JQ +131.41%/0.588/32.03% (2021-2026), "
    "Sharpe 0.588 通过 0.5 立项门槛。复合显著强于组成单因子 (overreaction +75.47%/0.396, "
    "lhb +64.59%/0.093) → 低相关短持有因子复合有效。族间定位: 弱于价量族 v1 (+157.81%/0.69) "
    "但提供低相关增量维度。日频框架因换手成本证伪 (P0.5 单边日换手76%), 周频是可存活形态。"
)

# 5 因子: (name, formula(库五变量格式), hypothesis, paradigm, category,
#          jq_return, jq_sharpe, jq_maxdd, contribution, caliber)
FACTORS = [
    ("overnight_raw", "open_p/close_p.shift(1)-1.0",
     "隔夜跳空单日: 今日开盘相对昨收的隔夜收益, 捕捉隔夜动量/情绪延续。首次入组合",
     "隔夜动量", "microstructure", COMBO_RET, COMBO_SR, COMBO_MDD, "positive", "combo"),
    ("overnight5", "(open_p/close_p.shift(1)-1.0).rolling(5).sum()",
     "隔夜跳空5日累计: 持续小幅低开的股票有均值回复空间。King core4 成员 (组合 +182.57%)",
     "隔夜动量", "microstructure", COMBO_RET, COMBO_SR, COMBO_MDD, "positive", "combo"),
    ("intraday_rev", "-(close_p/open_p-1.0)",
     "日内反转: 当日日内涨幅取负, A股经典日内反转效应 (P0 实测 IC(1d)=+0.020)。首次入组合",
     "日内反转", "microstructure", COMBO_RET, COMBO_SR, COMBO_MDD, "positive", "combo"),
    ("lhb_freq_20d", "-lhb_flag.rolling(20).sum()",
     "龙虎榜上榜频次取负: 20日内上榜次数越多越回避 (游资兑现风险)。单因子 JQ +64.59%/0.093 (MARGINAL, MaxDD 未记录置 0)",
     "事件驱动", "event", 64.59, 0.093, 0.0, "positive", "real"),
    ("overreaction", "-close_p.pct_change(1).rolling(20).max()",
     "过度反应反转: 20日内最大单日涨幅取负, 行为金融。单因子 JQ +75.47%/0.396 (MARGINAL)",
     "行为金融", "behavioral", 75.47, 0.396, 26.0, "neutral", "real"),
]


def infer_operators(formula: str):
    ops = []
    kws = {"pct_change": "pct_change", "rolling": "rolling", "rank": "rank",
           "std": "rolling_std", "mean": "rolling_mean", "min": "rolling_min",
           "max": "rolling_max", "clip": "clip", "log": "log",
           "shift": "shift", "abs": "abs", "/": "division", "*": "multiplication",
           "+": "addition", "-": "subtraction", "corr": "rolling_corr"}
    for kw, op in kws.items():
        if kw in formula.lower():
            ops.append(op)
    return ops


def build_jq_result() -> dict:
    factors = []
    for name, formula, hypothesis, paradigm, category, jq_ret, jq_sr, jq_mdd, contrib, caliber in FACTORS:
        if caliber == "real":
            note = (f"真实单因子口径 (独立 JQ 回测): {jq_ret}% / Sharpe {jq_sr} / MaxDD {jq_mdd}%。")
        else:
            note = (f"组合口径 (本次复合 JQ {COMBO_RET}%/{COMBO_SR}/{COMBO_MDD}%, "
                    f"无独立单因子验证, 该数字不代表单因子真实水平, 防误升级/误入库)。")
        factors.append({
            "factor_name": name,
            "formula": formula,
            "hypothesis": hypothesis,
            "paradigm": paradigm,
            "category": category,
            "operators_used": infer_operators(formula),
            "local_ic": None,
            "local_icir": None,
            "jq_return": jq_ret,
            "jq_sharpe": jq_sr,
            "jq_maxdd": jq_mdd,
            "jq_ic": None,
            "jq_icir": None,
            "jq_ic_note": ("复合 JQ 已输出滞后法单因子 IC (回测日志), 待用户回传后补录; "
                           "本地 mock 已证: 复合 IC +0.0664/ICIR 0.665, "
                           "overreaction 子因子 +0.0894 与 JQ 实测对拍一致"),
            "jq_composite_contribution": contrib,
            "root_cause": (
                f"情绪×日内复合成员 (5因子等权 rank和, top30/池800/周频周一open, lookback 30)。"
                f"组合 JQ 实测 {COMBO_RET}%/Sharpe {COMBO_SR}/MaxDD {COMBO_MDD}% "
                f"(2021-01~2026-07)。{note} {LESSON}"
            ),
        })
    return {
        "batch_id": f"sentiment_intraday_combo_jq_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "timestamp": datetime.now().isoformat(),
        "composite_return": COMBO_RET,
        "composite_sharpe": COMBO_SR,
        "composite_maxdd": COMBO_MDD,
        "combo_note": LESSON,
        "factors": factors,
    }


def record_combo_in_champions():
    if not CHAMPIONS.exists():
        return
    data = json.load(open(CHAMPIONS, encoding="utf-8"))
    existing = {str(it.get("name", "")) for it in data if isinstance(it, dict)}
    name = "sentiment_intraday_5f_composite"
    if name in existing:
        print("  champions 注册表已存在, 跳过")
        return
    data.append({
        "name": name,
        "paradigm": "sentiment_combo",
        "jq_return": COMBO_RET,
        "jq_sharpe": COMBO_SR,
        "jq_maxdd": COMBO_MDD,
        "note": "情绪×日内 5因子等权rank和 (overnight_raw/overnight5/intraday_rev/lhb_freq_20d/overreaction), "
                "top30/池800/周频。首个情绪×日内族成员, Sharpe 0.588 过 0.5 门槛; "
                "强于组成单因子 (overreaction 75.47/0.396, lhb 64.59/0.093)",
        "recorded_at": datetime.now().isoformat()[:19],
    })
    json.dump(data, open(CHAMPIONS, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("  champions 注册表新增: sentiment_intraday_5f_composite")


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
    print("  D+ 蒸馏: 情绪×日内 5因子复合 JQ 实测注入")
    print("=" * 60)

    jq_result = build_jq_result()
    print(f"  composite: {jq_result['composite_return']}% / "
          f"Sharpe {jq_result['composite_sharpe']} / MDD {jq_result['composite_maxdd']}%")
    for f in jq_result["factors"]:
        print(f"    {f['factor_name']}: JQ {f['jq_return']}% / {f['jq_sharpe']} "
              f"| {f['jq_composite_contribution']}")

    from ralph_loop import RalphLoop
    from experience_memory import get_memory

    memory = get_memory()
    rl = RalphLoop(memory=memory)
    result = rl.jq_feedback(jq_backtest_result=jq_result)
    print(json.dumps(result, ensure_ascii=False, indent=1, default=str))

    out = ROOT / "data" / "jq_feedback_sentiment_combo.json"
    json.dump({"input": jq_result, "feedback_result": result},
              open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
    print(f"  结果已存: {out}")

    record_combo_in_champions()
    mark_new_traj_d_plus()
    print("D+ 闭环完成")


if __name__ == "__main__":
    main()
