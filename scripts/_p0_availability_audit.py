# -*- coding: utf-8 -*-
"""P0 前置二：五族在 87 个冻结截面上的逐截面可得性（只读）。

判定口径：
* 逐截面「有无该族数据」=> 决定该族能否进入 IC 上限测量；
* 行业族另需检查 sw_daily 的行业指数代码能否与 sw_classification_L1 的 index_code 对上；
* 基准族是市场级时间序列，天然无截面方差，此处只报可得性。
"""
import csv
import json
import os
import sys
from collections import Counter, defaultdict

ROOT = "E:/quant"
RAW = os.path.join(ROOT, "data", "raw")

proto = json.load(open(os.path.join(ROOT, "reports/external_info_20260924/protocol.json"),
                       encoding="utf-8"))
STAMPS = sorted(proto["signal_dates"])
print("stamps=%d  %s .. %s" % (len(STAMPS), STAMPS[0], STAMPS[-1]))

WINDOW = [s.replace("-", "") for s in STAMPS]
WSET = set(WINDOW)

# --- block_trade -----------------------------------------------------------
bt_days = Counter()
bt_codes = defaultdict(set)
with open(os.path.join(RAW, "block_trade_full.csv"), encoding="utf-8-sig", newline="") as f:
    r = csv.DictReader(f)
    for row in r:
        d = str(row["trade_date"]).strip().replace("-", "")[:8]
        bt_days[d] += 1
        if d in WSET:
            bt_codes[d].add(row["ts_code"])
covered_bt = [s for s in WINDOW if s in bt_days]
print("\n[block_trade] 窗口内截面 %d/%d；缺 %d 个: %s"
      % (len(covered_bt), len(STAMPS), len(STAMPS) - len(covered_bt),
         ",".join(s for s in WINDOW if s not in bt_days) or "-"))
if covered_bt:
    ns = [len(bt_codes[s]) for s in covered_bt]
    print("  每截面涉及股票数 min=%d 中位=%d max=%d" % (min(ns), sorted(ns)[len(ns)//2], max(ns)))

# --- hk_hold --------------------------------------------------------------
hk_days = Counter()
with open(os.path.join(RAW, "hk_hold.csv"), encoding="utf-8-sig", newline="") as f:
    r = csv.DictReader(f)
    for row in r:
        d = str(row["trade_date"]).strip().replace("-", "")[:8]
        hk_days[d] += 1
covered_hk = [s for s in WINDOW if s in hk_days]
print("\n[hk_hold] 窗口内截面 %d/%d" % (len(covered_hk), len(STAMPS)))
print("  窗口内命中的截面: %s" % (",".join(covered_hk) or "none"))
print("  全表最大日=%s" % max(hk_days))

# --- index ----------------------------------------------------------------
def idx_days(name):
    p = os.path.join(RAW, name)
    out = {}
    with open(p, encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            d = str(row.get("trade_date", "")).strip().replace("-", "")[:8]
            if d:
                out[d] = row.get("close")
    return out

for nm in ("index_000300_daily.csv", "index_000905_daily.csv"):
    d = idx_days(nm)
    hit = [s for s in WINDOW if s in d]
    print("\n[%s] 窗口内截面 %d/%d  max=%s" % (nm, len(hit), len(STAMPS), max(d)))

# --- sw_daily -------------------------------------------------------------
sw_days = Counter()
sw_codes = set()
with open(os.path.join(RAW, "sw_daily.csv"), encoding="utf-8-sig", newline="") as f:
    r = csv.DictReader(f)
    for row in r:
        d = str(row["trade_date"]).strip().replace("-", "")[:8]
        sw_days[d] += 1
        sw_codes.add(row["ts_code"])
covered_sw = [s for s in WINDOW if s in sw_days]
print("\n[sw_daily] 窗口内截面 %d/%d  max=%s" % (len(covered_sw), len(STAMPS), max(sw_days)))
print("  代码数=%d 例: %s" % (len(sw_codes), ",".join(sorted(sw_codes)[:8])))
print("  窗口内命中截面: %s ... %s" % (",".join(covered_sw[:4]), ",".join(covered_sw[-4:])))

# --- sw_classification_L1 -------------------------------------------------
cls_codes = set()
cls_rows = 0
with open(os.path.join(ROOT, "data/sw_classification_L1.csv"), encoding="utf-8-sig",
          newline="") as f:
    r = csv.DictReader(f)
    for row in r:
        cls_rows += 1
        cls_codes.add(row["index_code"])
print("\n[sw_classification_L1] rows=%d codes=%d" % (cls_rows, len(cls_codes)))
inter = cls_codes & sw_codes
print("  与 sw_daily 代码交集=%d  (classification 独有 %d / sw_daily 独有 %d)"
      % (len(inter), len(cls_codes - sw_codes), len(sw_codes - cls_codes)))
print("  classification 例: %s" % ",".join(sorted(cls_codes)[:8]))
