# -*- coding: utf-8 -*-
"""
parse_shadow_log.py — 解析影子运行日志 → 复合级训练数据集

输入: JQ 日志文件 (含 [SHADOW_RESULT] 行)
输出:
  1. 袖子最终表现表 (total_return/sharpe/maxdd)
  2. FRI vs 影子表现的相关性 (FRI 有没有预测力的第一次直接测量)
  3. 追加复合级样本到 trial_log.jsonl (供 RidgeUCB 重训)
"""
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

ROOT = Path(__file__).resolve().parents[3]
INJECTED_JSON = ROOT / "data" / "injected_factors.json"
TRIAL_LOG = Path(__file__).parent / "trial_log.jsonl"
COMPOSITE_LOG = Path(__file__).parent / "composite_trials.jsonl"

# 袖子 → (factor_a, factor_b, 注入因子名 or None)
SLEEVE_MAP = {
    "king_comp1": ("overnight_5d", "tvma_20", None),
    "king_comp2": ("dollar_vol_20d", "turnover_std_cv", None),
    "king_comp3": ("money_flow_20", "ret_3m", None),
    "king_comp4": ("nf_ff7913", "tvma_20", None),
    "king_comp5": ("f_", "dollar_vol_20d", None),
    "king_comp6": ("nf_02a304", "money_flow_20", None),
    "inj_harvey_coskew": ("harvey_siddique_coskew", "tvma_20", "harvey_siddique_coskew"),
    "inj_idio_tail_hedge": ("idiosyncratic_tail_hedge_premium", "dollar_vol_20d", "idiosyncratic_tail_hedge_premium"),
    "inj_max_dd_duration": ("max_drawdown_duration", "overnight_5d", "max_drawdown_duration"),
    "inj_capital_efficiency": ("capital_efficiency_proxy", "dollar_vol_20d", "capital_efficiency_proxy"),
    "inj_hl_vol_spread": ("hl_volatility_spread_regime_stable", "dollar_vol_20d", "hl_volatility_spread_regime_stable"),
    "inj_lottery_suppress": ("lottery_demand_suppression_score", "tvma_20", "lottery_demand_suppression_score"),
    "inj_earnings_pre_drift": ("earnings_pre_drift_alignment", "turnover_std_cv", "earnings_pre_drift_alignment"),
    "inj_diffusion_mom": ("diffusion_index_momentum", "dollar_vol_20d", "diffusion_index_momentum"),
    "inj_vol_crowding_div": ("volume_crowding_divergence", "overnight_5d", "volume_crowding_divergence"),
    "inj_smallcap_liq_quality": ("small_cap_liquidity_quality", "ret_3m", "small_cap_liquidity_quality"),
    "inj_event_convexity": ("event_driven_convexity_fade", "overnight_5d", "event_driven_convexity_fade"),
}


def parse_log(log_path):
    """解析最后一行 SHADOW_RESULT → {sid: (total, sharpe, mdd)}"""
    last_line = None
    with open(log_path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            if "SHADOW_RESULT" in line:
                last_line = line
    if last_line is None:
        raise ValueError("日志中无 SHADOW_RESULT")
    results = {}
    for m in re.finditer(r"(\w+)=(-?[\d.]+)/(-?[\d.]+)/(-?[\d.]+)", last_line):
        sid, total, sharpe, mdd = m.groups()
        if sid.startswith(("king_", "inj_")):
            results[sid] = (float(total), float(sharpe), float(mdd))
    return results


def main(log_path):
    results = parse_log(log_path)
    with open(INJECTED_JSON, encoding="utf-8") as f:
        inj_db = {i["factor_name"]: i for i in json.load(f)["injected"]}

    # ── 1. 表现表 ──
    print(f"{'袖子':<28} {'收益':>8} {'Sharpe':>7} {'MaxDD':>8}  {'FRI':>6} {'等级':>4}")
    print("-" * 75)
    rows = []
    for sid, (total, sharpe, mdd) in sorted(results.items(), key=lambda x: -x[1][0]):
        _, _, inj_name = SLEEVE_MAP[sid]
        fri = inj_db[inj_name]["fri"] if inj_name else None
        grade = inj_db[inj_name]["fri_grade"] if inj_name else ""
        fri_str = f"{fri:.3f}" if fri is not None else "  —"
        dead = " ☠️死亡" if total == 0 and sharpe == 0 else ""
        star = " 🏆" if total > 0.6 else ""
        print(f"{sid:<28} {total:>8.3f} {sharpe:>7.2f} {mdd:>8.3f}  {fri_str:>6} {grade:>4}{dead}{star}")
        rows.append((sid, total, sharpe, mdd, fri))

    # ── 2. FRI vs 影子表现相关性 ──
    inj_rows = [(fri, total) for sid, total, _, _, fri in rows
                if fri is not None and not (total == 0)]
    if len(inj_rows) >= 5:
        fris = np.array([r[0] for r in inj_rows])
        rets = np.array([r[1] for r in inj_rows])
        from scipy import stats as sstats
        spearman = sstats.spearmanr(fris, rets)
        pearson = np.corrcoef(fris, rets)[0, 1]
        print(f"\n=== FRI 预测力直接测量 (n={len(inj_rows)} 注入袖子, 排除死亡) ===")
        print(f"Spearman ρ(FRI, 影子收益) = {spearman.correlation:+.3f} (p={spearman.pvalue:.3f})")
        print(f"Pearson  r(FRI, 影子收益) = {pearson:+.3f}")
        verdict = ("负预测力!" if spearman.correlation < -0.2 else
                   "无预测力" if abs(spearman.correlation) <= 0.2 else "正预测力")
        print(f"结论: FRI 对 JQ 影子表现 = {verdict}")

    # ── 3. H1 vs H2 裁决 ──
    print(f"\n=== H1(FRI加权选错) vs H2(占位符bug) 裁决 ===")
    king_best = max(t for sid, t, _, _, _ in rows if sid.startswith("king_"))
    inj_good = [(sid, t) for sid, t, _, _, fri in rows
                if sid.startswith("inj_") and t > king_best]
    print(f"王者最佳袖子: {king_best:+.3f}")
    print(f"超过王者最佳的注入袖子: {len(inj_good)} 个")
    for sid, t in inj_good:
        print(f"  🏆 {sid}: {t:+.3f}")
    if inj_good:
        print("→ H2 成立: 真实实现的注入因子能赢, #3轮失败主因含占位符bug")
    else:
        print("→ H1 成立: 注入因子即使真实实现也全输, FRI加权确实选错")

    # ── 4. 追加复合级样本 ──
    n_new = 0
    with open(COMPOSITE_LOG, "a", encoding="utf-8") as f:
        for sid, total, sharpe, mdd, fri in rows:
            a, b, inj_name = SLEEVE_MAP[sid]
            rec = {
                "type": "composite_sleeve",
                "sleeve_id": sid,
                "factor_a": a, "factor_b": b,
                "injected_factor": inj_name,
                "fri": fri,
                "total_return": total, "sharpe": sharpe, "maxdd": mdd,
                "source": "shadow_eval_v1",
                "date": datetime.now().strftime("%Y-%m-%d"),
                "note": "top20单复合周频, NaN bug影响(comp3/comp6死亡, 其余袖子轻度退化)",
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_new += 1
    print(f"\n→ {COMPOSITE_LOG.name}: 追加 {n_new} 条复合级样本")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/e/log.txt")
