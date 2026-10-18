# -*- coding: utf-8 -*-
"""
D+ 蒸馏手动触发 — 打散重组 v3 (10因子 rank和 + 波动率目标管理) JQ 实测注入 (2026-08-22)
========================================================================================
JQ 实测 (2021-01-01 ~ 2026-07-30, 用户回测, 文件 jq_scatter_v3_10factor_voltarget.py):
  v3 组合: +126.75%, Sharpe 0.604, MaxDD 31.2%
  vs v1 (同组合无波动率管理): +157.81% / 0.690 / 30.67%   → 三指标全面劣化
  vs King 基线:               +182.57% / 0.690 / 35.41%

裁决: 波动率目标管理 (Barroso-SantaClara 规则化, σ_t=0.040/12周窗/scale∈[0.3,1.0]) 证伪。
机制归因:
  1. scale 上限 1.0 只降不加 → 实为"降杠杆保险"而非目标波动;
  2. A 股价量 alpha 的收益集中在高波动期 (急涨急跌), 降仓恰好砍在 alpha 释放段;
  3. 12 周窗口波动率估计滞后, 追不上波动聚集, 回撤左侧降仓已来不及 → MaxDD 反升 0.53pp。
定论: "组合学天花板"成立 — 现有全价量族因子池的重组上限 = King (+182.57%/0.69)
      或 v1 形态 (+157.81%/0.69/-30.67%), 组合内再工程 (打散/分组/rank积/波动率管理)
      均无法突破; 主攻方向必须切新 alpha 源 (行业轮动/情绪/筹码分布/北向大宗)。

成员口径说明:
  - gp002_pricevol: 单因子 JQ 实测 +143.46%/0.51 (Breed BEST), 用真实单因子口径;
  - overreaction:   单因子 JQ 实测 +75.47%/0.396 (MARGINAL), 用真实单因子口径;
  - 其余 8 成员本次未单独回测, 收益口径=组合 126.75/0.604/31.2 (root_cause 注明
    各自所属最强验证口径: King core4 182.57 / FA Composite 146.13 / 历史单因子)。

用法: python scripts/run_d_plus_scatter_v3_combo.py
"""
import sys
import json
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "research" / "factor_alchemy"))

TRAJ_LOG = ROOT / "data" / "trajectory_log.json"
CHAMPIONS = ROOT / "data" / "jq_validated_champions.json"

COMBO_RET = 126.75
COMBO_SR = 0.604
COMBO_MDD = 31.2

VOLTARGET_LESSON = (
    "波动率目标管理证伪: v3(+126.75%/0.604/31.2%) vs v1(+157.81%/0.69/30.67%) 三指标全劣。"
    "scale≤1.0 只降不加 + 12周窗滞后 + A股alpha集中于高波动期 → 降仓砍收益不砍回撤。"
    "组合学天花板定论成立, 主攻方向切新 alpha 源。"
)

# 10 因子: (name, formula(库五变量格式), hypothesis, paradigm, category,
#          jq_return, jq_sharpe, jq_maxdd, contribution)
FACTORS = [
    ("dollar20", "-log((close_p*volume_p).rolling(20).mean())",
     "低成交额: 20日平均成交额对数值取负, 低关注度股票有超额 (A股流动性溢价)。King core4 成员",
     "流动性", "liquidity", COMBO_RET, COMBO_SR, COMBO_MDD, "positive"),
    ("overnight5", "(open_p/close_p.shift(1)-1.0).rolling(5).sum()",
     "隔夜跳空5日累计: 持续小幅低开的股票有均值回复空间。King core4 成员",
     "隔夜动量", "microstructure", COMBO_RET, COMBO_SR, COMBO_MDD, "positive"),
    ("tstd_cv", "-(volume_p.rolling(20).std()/volume_p.rolling(20).mean())",
     "低量波动: 成交量变异系数取负, 量能稳定的股票风险调整后收益更高。King core4 成员",
     "低波动", "low_volatility", COMBO_RET, COMBO_SR, COMBO_MDD, "positive"),
    ("tvma20", "-(close_p*volume_p/((close_p*volume_p).rolling(20).mean())).clip(0,10)",
     "缩量: 20日成交额萎缩的股票被低估。King core4 成员",
     "流动性", "liquidity", COMBO_RET, COMBO_SR, COMBO_MDD, "positive"),
    ("fa_turnover_std", "-((volume_p/volume_p.rolling(20).mean())).rolling(20).std()",
     "换手率波动: 换手比率20日标准差取负。FA v5.2 成员 (组合 +146.13%)",
     "低波动", "low_volatility", COMBO_RET, COMBO_SR, COMBO_MDD, "positive"),
    ("fa_vroc_12", "volume_p.pct_change(12)",
     "量变化率: 12日成交量变化率。FA v5.2 成员 (组合 +146.13%)",
     "量价关系", "price_volume", COMBO_RET, COMBO_SR, COMBO_MDD, "positive"),
    ("gp002_pricevol", "high_p-close_p.pct_change(20).shift(20)",
     "价量关系: GP 育种冠军, 单因子 JQ +143.46%/0.51 (Breed BEST)",
     "价量关系", "price_volume", 143.46, 0.51, 43.17, "positive"),
    ("gp_micro_7db3", "(close_p-low_p)/((close_p*volume_p-open_p)+0.001)",
     "微观结构: GP 育种产物, 历史单因子 JQ +129.4% (sharpe 未知, 本次组合口径)",
     "微观结构", "microstructure", COMBO_RET, COMBO_SR, COMBO_MDD, "positive"),
    ("overreaction", "-close_p.pct_change(1).rolling(20).max()",
     "过度反应反转: 20日内最大单日涨幅取负, 行为金融。单因子 JQ +75.47%/0.396 (MARGINAL)",
     "行为金融", "behavioral", 75.47, 0.396, 26.0, "neutral"),
    ("gp_evo_003", "-volume_p.rolling(120).corr(close_p).pct_change(10)",
     "量价相关变化率: 120日量价相关性10日变化取负。历史单因子 JQ +73.52% (本次组合口径)",
     "量价关系", "price_volume", COMBO_RET, COMBO_SR, COMBO_MDD, "positive"),
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
    for name, formula, hypothesis, paradigm, category, jq_ret, jq_sr, jq_mdd, contrib in FACTORS:
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
            "jq_ic_note": "scatter v3 组合回测未输出单因子IC (组合策略口径), 已知证据见 root_cause",
            "jq_composite_contribution": contrib,
            "root_cause": (
                f"打散重组 scatter v3 成员 (10因子贪心低相关筛选 ρ<0.5, 等权 rank和, "
                f"top30/池800/周频周一open + 波动率目标管理)。"
                f"组合 JQ 实测 {COMBO_RET}%/Sharpe {COMBO_SR}/MaxDD {COMBO_MDD}% "
                f"(2021-01~2026-07)。" + VOLTARGET_LESSON
            ),
        })
    return {
        "batch_id": f"scatter_v3_voltarget_combo_jq_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "timestamp": datetime.now().isoformat(),
        "composite_return": COMBO_RET,
        "composite_sharpe": COMBO_SR,
        "composite_maxdd": COMBO_MDD,
        "combo_note": VOLTARGET_LESSON,
        "factors": factors,
    }


def record_combo_in_champions():
    """组合级结果写入 champions 注册表 (含 v1/v2 系列完整证据链)"""
    if not CHAMPIONS.exists():
        return
    data = json.load(open(CHAMPIONS, encoding="utf-8"))
    existing = {str(it.get("name", "")) for it in data if isinstance(it, dict)}
    combos = [
        ("scatter_v1_10factor_ranksum", 157.81, 0.690, 30.67,
         "10因子 rank和 (无波动率管理), 打散系列最优形态"),
        ("scatter_v2a_10factor_rankprod", 95.22, 0.42, 32.1,
         "10因子 rank积 → 区分度坍塌 (0.5^10≈0.001)"),
        ("scatter_v2b_10factor_grouped", 117.18, None, None,
         "core4 rank积 + 外围6因子 rank和 0.5/0.5等权"),
        ("scatter_v2c_10factor_rankprod_top50", 119.64, None, None,
         "rank积 + top50 扩池"),
        ("scatter_v2d_10factor_geomean", 95.22, None, None,
         "几何平均 ≡ rank积排序等价 (与 v2a 逐位一致)"),
        ("scatter_v3_10factor_voltarget", 126.75, 0.604, 31.2,
         "rank和 + 波动率目标管理 → 三指标劣于 v1, 波动率管理证伪"),
    ]
    added = 0
    for name, ret, sr, mdd, note in combos:
        if name in existing:
            continue
        data.append({
            "name": name,
            "paradigm": "scatter_combo",
            "jq_return": ret,
            "jq_sharpe": sr,
            "jq_maxdd": mdd,
            "note": note,
            "recorded_at": datetime.now().isoformat()[:19],
        })
        added += 1
    if added:
        json.dump(data, open(CHAMPIONS, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
        print(f"  champions 注册表新增 {added} 条组合记录 (v1/v2 系列补录 + v3)")
    else:
        print("  champions 注册表无新增 (已存在)")


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
    print("  D+ 蒸馏: 打散重组 v3 (波动率目标管理) JQ 实测注入")
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

    out = ROOT / "data" / "jq_feedback_scatter_v3_combo.json"
    json.dump({"input": jq_result, "feedback_result": result},
              open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
    print(f"  结果已存: {out}")

    record_combo_in_champions()
    mark_new_traj_d_plus()
    print("D+ 闭环完成")


if __name__ == "__main__":
    main()
