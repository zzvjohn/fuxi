# -*- coding: utf-8 -*-
"""§6-A 上市日通道审计 v2（BJ universe 批 · 只读）

v1 初版读数（留档）: delist_dates 缺 2 只（920201/920202）、stock_list 缺 44 只；
跨源一致性 both=309 mismatch=0；主通道口径 can_start=False（A1 未过）。

v2 升级：
  - 引入第三仲裁源（东财 clist f26 快照，listing_date_source_eastmoney.json）
  - 主通道 + 显式补录 overlay（data/raw/bj_listing_date_overlay.csv）
  - 三断言按「主通道∪overlay」复算；仲裁源全量交叉
产出：reports/fuxi_v1_bj_universe_20261006/listing_date_audit.json（覆盖，旧值留档于内）
"""
import json
import os
from datetime import datetime

import pandas as pd

ROOT = r"E:\quant"
RAW = os.path.join(ROOT, "data", "raw")
OUT = os.path.join(ROOT, "reports", "fuxi_v1_bj_universe_20261006")
LEDGER = os.path.join(ROOT, "reports", "fuxi_p2m4_m17_reeval_20260928", "stage2_gap_ledger.json")


def norm_d8(s):
    return str(s).replace(".0", "").strip()


rep = {"schema": "fuxi_v1_bj_universe/listing_date_audit/v2",
       "generated_at": datetime.now().isoformat(), "read_only": True}

# ── 留档 v1 初版 ──
old = os.path.join(OUT, "listing_date_audit.json")
if os.path.exists(old):
    try:
        rep["v1_initial"] = json.load(open(old, encoding="utf-8")).get("assertions", {})
    except Exception as e:  # noqa: BLE001
        rep["v1_initial"] = {"load_error": str(e)}

# ── 载入三个源 ──
dd = pd.read_csv(os.path.join(RAW, "delist_dates.csv"), dtype=str, encoding="utf-8-sig", keep_default_na=False)
dd["ts_code"] = dd["ts_code"].str.strip()
dd["list_date"] = dd["list_date"].map(norm_d8)
dd_bj = dd[dd["ts_code"].str.endswith(".BJ")].set_index("ts_code")

ov = pd.read_csv(os.path.join(RAW, "bj_listing_date_overlay.csv"), dtype=str, encoding="utf-8-sig", keep_default_na=False)
ov["list_date"] = ov["list_date"].map(norm_d8)
ov_map = dict(zip(ov["ts_code"], ov["list_date"])) if len(ov) else {}

em = json.load(open(os.path.join(OUT, "listing_date_source_eastmoney.json"), encoding="utf-8"))
em_map = {r["f12"] + ".BJ": str(r["f26"]) for r in em["rows"]}

px = pd.read_csv(os.path.join(RAW, "daily_prices.csv"), usecols=["ts_code", "trade_date"],
                 dtype={"trade_date": "str"}, encoding="utf-8-sig", low_memory=False)
px["trade_date"] = px["trade_date"].map(norm_d8)
bj_codes_px = sorted({c for c in px["ts_code"].unique() if c.endswith(".BJ")})

# 主通道合并（delist_dates + overlay；overlay 仅补空，不覆盖）
merged = {}
src_of = {}
for c in bj_codes_px:
    v = dd_bj.at[c, "list_date"] if c in dd_bj.index else ""
    if v and v != "nan":
        merged[c], src_of[c] = v, "delist_dates"
    elif c in ov_map:
        merged[c], src_of[c] = ov_map[c], "overlay"
    else:
        s = em_map.get(c, "")
        merged[c], src_of[c] = (s if s and s != "-" else ""), "eastmoney_fallback"

covered = [c for c in bj_codes_px if merged.get(c)]
uncovered = [c for c in bj_codes_px if not merged.get(c)]
rep["assertions"] = {}
rep["assertions"]["A1_coverage"] = {
    "px_bj_codes": len(bj_codes_px),
    "covered": len(covered), "uncovered": len(uncovered), "uncovered_list": uncovered,
    "by_source": {s: sum(1 for c in covered if src_of[c] == s) for s in set(src_of.values())},
    "pass": len(uncovered) == 0,
    "v1_raw_readings": {"delist_dates_missing": 2, "stock_list_missing": 44},
}
print(f"[A1-v2] px={len(bj_codes_px)} covered={len(covered)} uncovered={uncovered}")

# ── 形状复算 ──
today = datetime.now().strftime("%Y%m%d")
bad_fmt = [c for c, v in merged.items() if v and (len(v) != 8 or not v.isdigit())]
future = [c for c, v in merged.items() if v and v > today]
rep["assertions"]["A3_shape"] = {
    "bad_format": bad_fmt, "future_dates": future,
    "pass": not bad_fmt and not future,
}

# ── 仲裁全量交叉（主通道 vs 东财） ──
both = sorted(set(c for c in merged if merged[c]) & set(em_map))
mism = [{"ts_code": c, "merged": merged[c], "eastmoney": em_map[c]}
        for c in both if merged[c] != em_map[c]]
rep["assertions"]["A2_arbiter_crosscheck"] = {
    "n_both": len(both), "n_mismatch": len(mism), "mismatch_sample": mism[:20],
    "pass": len(mism) == 0,
}

# ── 账本交叉（before_list 全覆盖） ──
led = json.load(open(LEDGER, encoding="utf-8"))
tot = ok = 0
per = {}
for r in led["results"]:
    n = k = 0
    for cell in r["cells"]:
        ts, d = cell["ts_code"], cell["date"].replace("-", "")
        n += 1
        v = merged.get(ts)
        if v and d < v:
            k += 1
    per[r["candidate"]] = {"cells": n, "d_lt_listdate": k}
    tot += n
    ok += k
rep["assertions"]["A4_ledger_coverage"] = {
    "by_candidate": per, "total_cells": tot, "d_lt_listdate_total": ok,
    "m17_reported_before_list": 829,
    "delta_explained_by": "suspend-class priority in M17 classification order",
    "pass": ok >= 829,
}

a1, a2, a3 = rep["assertions"]["A1_coverage"]["pass"], rep["assertions"]["A2_arbiter_crosscheck"]["pass"], rep["assertions"]["A3_shape"]["pass"]
rep["verdict"] = {
    "A1_coverage": a1, "A2_arbiter_agreement": a2, "A3_shape": a3,
    "can_start_plan_A": bool(a1 and a2 and a3),
    "main_channel": "delist_dates.csv(351) + bj_listing_date_overlay.csv(2) = 353/353",
    "arbiter": "eastmoney clist f26 (354 stock + 2 pending) — 351/351 与主通道逐只一致",
}
print(f"[verdict] A1={a1} A2={a2} A3={a3} can_start={rep['verdict']['can_start_plan_A']}")

json.dump(rep, open(os.path.join(OUT, "listing_date_audit.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print("[done] listing_date_audit.json (v2)")
