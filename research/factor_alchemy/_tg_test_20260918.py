# -*- coding: utf-8 -*-
"""Smoke-test trade_gate_shadow on a few formulas (single panel load)."""
import json, os, sys
sys.path.insert(0, r"e:\quant\research\factor_alchemy")

import trade_gate_shadow as tg

FORMULAS = [
    # 1) forge gen4 (高换手负面读数)
    ("forge_gen4_r1",
     "-(np.log(np.maximum(volume_p.rolling(10).sum(), 1e-8)).rolling(5).std().rolling(60).mean().rolling(10).rank(pct=True))"),
    # 2) 低换手慢因子 (60 日反转) —— 预期 drag 低
    ("slow_rev_60",
     "-(close / close.shift(60) - 1.0)"),
    # 3) 高换手短动量 —— 预期 drag 高
    ("fast_mom_2",
     "(close / close.shift(2) - 1.0)"),
    # 4) 常量因子 (无区分度, 应被 nlargest 抽查到但 alpha≈0)
    ("turnover_level",
     "-(turnover_rate.rolling(20).mean())"),
]

out = []
for nm, f in FORMULAS:
    try:
        r = tg.factor_check(f, nm)
    except Exception as e:
        r = {"name": nm, "error": f"{type(e).__name__}: {e}"}
    out.append(r)
    print(json.dumps(r, ensure_ascii=False, indent=2, default=str))
    print("-" * 70)

print("\n=== summary table ===")
print("%-22s %5s %6s %9s %8s %9s %8s" % ("name", "n", "turn", "gross_ann", "net15", "drag30", "flags"))
for r in out:
    if "error" in r:
        print("%-22s ERROR %s" % (r["name"], r["error"][:50]))
        continue
    dr = r.get("drag_ratio_30")
    fl = []
    if r["fragile_30bp"]: fl.append("脆断")
    if r["steep_sensitivity"]: fl.append("陡降")
    if r["high_drag_30bp"]: fl.append("高拖累")
    if not r["reliable"]: fl.append("样本不足")
    print("%-22s %5d %6.2f %8.1f%% %8.1f%% %9s %s" % (
        r["name"], r["n_periods"], r["turnover_mean"], r["gross_ann"] * 100,
        r["net_ann"].get("15", float("nan")) * 100,
        ("%.0f%%" % (dr * 100)) if dr is not None else "n/a",
        ",".join(fl) or "-"))
