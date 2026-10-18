# -*- coding: utf-8 -*-
"""合并东财 4 页快照 + 校验 + 与本地 delist_dates 全量对账（§6-A 第三方仲裁）"""
import json
import os
from datetime import datetime

OUT = r"E:\quant\reports\fuxi_v1_bj_universe_20261006"
SRC = [os.path.join(OUT, f"_em_raw_pn{i}.json") for i in range(1, 5)]

rows = []
for p in SRC:
    part = json.load(open(p, encoding="utf-8"))
    print(f"[load] {os.path.basename(p)}: {len(part)} rows")
    rows.extend(part)

# ── 校验1 总数与重复 ──
codes = [r["f12"] for r in rows]
dup = {c for c in codes if codes.count(c) > 1}
print(f"[check] total={len(rows)} uniq={len(set(codes))} dup={sorted(dup)}")

# ── 校验2 分类 ──
def cls(r):
    c = r["f12"]
    if c.startswith("81"):
        return "other"
    if r["f26"] in ("-", None):
        return "not_listed"
    return "stock"

buckets = {"stock": [], "not_listed": [], "other": []}
for r in rows:
    buckets[cls(r)].append(r)
print(f"[check] stock={len(buckets['stock'])} not_listed={len(buckets['not_listed'])} other={len(buckets['other'])}")
print("  not_listed:", [(r['f12'], r['f14']) for r in buckets['not_listed']])
print("  other:", [(r['f12'], r['f14']) for r in buckets['other']])

# ── 快照落盘 ──
tf = [r for r in rows if cls(r) != "other"]
snap = {
    "schema": "fuxi_bj_listing_dates/eastmoney/v1",
    "fetched_at": datetime.now().isoformat(),
    "source": "push2.eastmoney.com clist (fs=m:0+t:81+s:2048 北交所板块)",
    "fields": "f12=code, f14=name, f26=list_date(YYYYMMDD)",
    "n_reported_total": 359,
    "n_rows_transcribed": len(rows),
    "n_transferables": len(tf),
    "bucket_counts": {k: len(v) for k, v in buckets.items()},
    "rows": sorted(tf, key=lambda x: str(x["f12"])),
}
sp = os.path.join(OUT, "listing_date_source_eastmoney.json")
json.dump(snap, open(sp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"[write] snapshot -> {sp} ({len(tf)} transferable rows)")

# ── 与本地 delist_dates 对账 ──
import pandas as pd
dd = pd.read_csv(r"E:\quant\data\raw\delist_dates.csv", dtype=str, encoding="utf-8-sig", keep_default_na=False)
dd["ts_code"] = dd["ts_code"].astype(str).str.strip()
dd["list_date"] = dd["list_date"].astype(str).str.replace(r"\.0$", "", regex=True).str.strip()
dd_bj = dd[dd["ts_code"].str.endswith(".BJ")].set_index("ts_code")

em = {}
for r in tf:
    em[r["f12"] + ".BJ"] = str(r["f26"])

# 交集对账
both = sorted(set(em) & set(dd_bj.index))
mism = []
for c in both:
    if em[c] != dd_bj.at[c, "list_date"]:
        mism.append((c, em[c], dd_bj.at[c, "list_date"]))
em_only = sorted(set(em) - set(dd_bj.index))
dd_only = sorted(set(dd_bj.index) - set(em))

print(f"[xcheck] both={len(both)} mismatch={len(mism)}")
for m in mism[:30]:
    print("   MISMATCH", m)
print(f"[xcheck] em_only({len(em_only)}):", em_only[:30])
print(f"[xcheck] dd_only({len(dd_only)}):", dd_only[:30])

# 追加到快照
snap["crosscheck_vs_delist_dates"] = {
    "both": len(both), "mismatch": len(mism),
    "mismatch_sample": [{"ts_code": c, "eastmoney": a, "delist_dates": b} for c, a, b in mism[:30]],
    "em_only": em_only, "dd_only": dd_only,
}
json.dump(snap, open(sp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("[done]")
