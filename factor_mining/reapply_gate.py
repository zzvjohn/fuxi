"""
对已落盘的 JQ 候选重新应用修正后的 JQ 门（无需重跑 GA/GFlowNet）
=================================================================
修复背景：icir 是比值型指标（ic_mean/ic_std），std→0 时伪 ICIR 爆炸，
会穿透「|icir|>=0.3」验收门。修正 = 在 JQ 门加 ic_std 下限（>=0.04，
原始 97 因子 ic_std min=0.0469）+ grammar 禁 div(X,X) 自除。

本脚本：读 results/jq_eval/{nsga2,gflownet}_candidates.json，
从 ic_series 重算 ic_std，重跑 jq_gate，回写 ic_std/jq_passed/jq_checks，
并更新 jq_summary.json 的 n_jq_pass 与 jq_gate 口径。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

RESULTS = Path(__file__).parent / "results" / "jq_eval"

MIN_N_WEEKS = 15
MIN_ABS_ICIR = 0.3
MIN_PLUS_IC_PCT = 0.55
MAX_TURNOVER = 0.5
MIN_IC_STD = 0.04


def reapply(cands):
    n_pass = 0
    n_pseudo = 0
    for c in cands:
        ic = np.array(c["ic_series"], dtype=np.float64)
        ic_f = ic[np.isfinite(ic)]
        ic_std = float(np.std(ic_f, ddof=1)) if len(ic_f) > 1 else 0.0
        c["ic_std"] = ic_std
        checks = {
            "n_weeks>=15": bool(c["n_weeks"] >= MIN_N_WEEKS),
            "|icir|>=0.3": bool(abs(c["icir"]) >= MIN_ABS_ICIR),
            "+IC%>=0.55": bool(c["plus_ic_pct"] >= MIN_PLUS_IC_PCT),
            "turnover<=0.5": bool(c["turnover"] <= MAX_TURNOVER),
            "ic_std>=0.04": bool(ic_std >= MIN_IC_STD),
        }
        c["jq_passed"] = all(checks.values())
        c["jq_checks"] = checks
        if ic_std < MIN_IC_STD:
            n_pseudo += 1
        if c["jq_passed"]:
            n_pass += 1
    return n_pass, n_pseudo


def main():
    nsga = json.load(open(RESULTS / "nsga2_candidates.json", encoding="utf-8"))["candidates"]
    gf = json.load(open(RESULTS / "gflownet_candidates.json", encoding="utf-8"))["candidates"]

    ns_pass, ns_pseudo = reapply(nsga)
    gf_pass, gf_pseudo = reapply(gf)

    print(f"NSGA-II: {ns_pass}/{len(nsga)} 过门, 其中伪(std<0.04)被剔除 {ns_pseudo} 个")
    print(f"GFlowNet: {gf_pass}/{len(gf)} 过门, 其中伪(std<0.04)被剔除 {gf_pseudo} 个")

    # 回写 candidates
    json.dump({"candidates": nsga}, open(RESULTS / "nsga2_candidates.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2, default=str)
    json.dump({"candidates": gf}, open(RESULTS / "gflownet_candidates.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2, default=str)

    # 更新 summary
    s = json.load(open(RESULTS / "jq_summary.json", encoding="utf-8"))
    s["jq_gate"]["min_ic_std"] = MIN_IC_STD
    s["nsga2"]["n_jq_pass"] = ns_pass
    s["nsga2"]["pass_rate"] = ns_pass / max(1, len(nsga))
    s["gflownet"]["n_jq_pass"] = gf_pass
    s["gflownet"]["pass_rate"] = gf_pass / max(1, len(gf))
    json.dump(s, open(RESULTS / "jq_summary.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2, default=str)
    print(f"\nsummary 已更新：NSGA-II n_jq_pass={ns_pass}, GFlowNet n_jq_pass={gf_pass}")


if __name__ == "__main__":
    main()
