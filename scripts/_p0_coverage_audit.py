# -*- coding: utf-8 -*-
"""P0 前置：五族数据窗口内覆盖审计（只读）"""
import os, sys, csv
from collections import Counter

ROOT = r"E:/quant"
RAW = os.path.join(ROOT, "data", "raw")

FILES = [
    ("block_trade_full", os.path.join(RAW, "block_trade_full.csv"), "trade_date"),
    ("hk_hold", os.path.join(RAW, "hk_hold.csv"), "trade_date"),
    ("idx300", os.path.join(RAW, "index_000300_daily.csv"), "trade_date"),
    ("idx905", os.path.join(RAW, "index_000905_daily.csv"), "trade_date"),
    ("sw_daily", os.path.join(RAW, "sw_daily.csv"), "trade_date"),
    ("forecast", os.path.join(RAW, "forecast.csv"), None),
]

def norm(d):
    d = str(d).strip()
    return d.replace("-", "")

for name, path, dcol in FILES:
    if not os.path.exists(path):
        print("[%s] MISSING %s" % (name, path)); continue
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        r = csv.reader(f)
        try:
            hdr = next(r)
        except StopIteration:
            print("[%s] EMPTY" % name); continue
        if dcol is None:
            for cand in ("trade_date", "ann_date", "end_date", "date"):
                if cand in hdr:
                    dcol = cand; break
        idx = hdr.index(dcol) if dcol in hdr else None
        ds = Counter(); n = 0; bad = 0
        for row in r:
            n += 1
            if idx is None or idx >= len(row):
                bad += 1; continue
            ds[norm(row[idx])] += 1
    keys = sorted(ds)
    print("[%s] file=%s rows=%d col=%s bad=%d" % (name, os.path.basename(path), n, dcol, bad))
    if keys:
        print("   date_min=%s  date_max=%s  n_dates=%d" % (keys[0], keys[-1], len(keys)))
        print("   最近8日: %s" % ", ".join(keys[-8:]))
        print("   最早4日: %s" % ", ".join(keys[:4]))
    else:
        print("   NO DATES")
