# -*- coding: utf-8 -*-
"""Evaluate the 4 new 分歧度 factors (novel_237-240) through FactorICComputer."""
import os
import sys
import json

FA = r"e:\quant\research\factor_alchemy"
sys.path.insert(0, FA)
os.environ["FUXI_LHB_TRI_CHANNEL"] = "on"

from factor_ic_computer import FactorICComputer  # noqa: E402
import importlib.util  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "dfh", r"e:\quant\scripts\daily_factor_hypothesis.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
LIB = {e["id"]: e for e in m.NOVEL_FACTOR_LIBRARY if isinstance(e, dict)}

comp = FactorICComputer(lookback_days=500)
comp._load_data()

print()
print("--- novel_237..240 端到端 eval ---")
out = {}
for fid in ("novel_237", "novel_238", "novel_239", "novel_240"):
    e = LIB[fid]
    nm = e["name"]
    try:
        df = comp._eval_formula(e["formula"], nm[:40])
    except Exception as ex:
        print("  %-14s %-34s EXCEPTION %s: %s" % (fid, nm, type(ex).__name__, ex))
        out[fid] = {"ok": False, "err": f"{type(ex).__name__}: {ex}"}
        continue
    if df is None or len(df) == 0:
        print("  %-14s %-34s -> None/空 (静默丢弃)" % (fid, nm))
        out[fid] = {"ok": False, "rows": 0}
        continue
    fv = df["factor_value"]
    n_nn = int(fv.notna().sum())
    uniq = int(fv.nunique())
    std = float(fv.std()) if n_nn else 0.0
    # 收敛性: 常量/近常量因子是死的
    dead = uniq <= 2 or std < 1e-9
    print("  %-14s %-34s rows=%-8d nonNaN=%-8d uniq=%-5d std=%.5f %s"
          % (fid, nm, len(df), n_nn, uniq, std, "[!! 近常量=死因子]" if dead else ""))
    out[fid] = {"ok": True, "rows": len(df), "nonnan": n_nn, "uniq": uniq,
                "std": std, "dead": dead}

print()
print("JSON_RESULT " + json.dumps(out, ensure_ascii=False, default=str))
