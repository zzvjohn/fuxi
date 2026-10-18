# -*- coding: utf-8 -*-
"""P0 前置三：大宗交易 / 行业日线 的日期稀疏性诊断（只读）。

问题：窗口内只有 38/87 命中。要区分两种可能：
  (a) 该表本身就是「按需稀疏」——只含部分交易日；
  (b) 采集缺口——某段日期整体缺失。
诊断：把两表的日期序列按月统计「有数据的日期数」，并与该月交易日数对比。
"""
import csv
import os
from collections import Counter, defaultdict

ROOT = "E:/quant"
RAW = os.path.join(ROOT, "data", "raw")

cal = []
with open(os.path.join(RAW, "trade_calendar.csv"), encoding="utf-8-sig", newline="") as f:
    for row in csv.DictReader(f):
        if row.get("exchange") == "SSE" and row.get("is_open") == "1":
            cal.append(str(row["cal_date"]).strip().replace("-", "")[:8])
cal = sorted(set(cal))
print("交易日历: %d 日 %s .. %s" % (len(cal), cal[0], cal[-1]))

LO, HI = "20241201", "20260901"
cal_w = [d for d in cal if LO <= d <= HI]
print("窗口区间 %s..%s 交易日 %d" % (LO, HI, len(cal_w)))

def table_days(path, col):
    ds = set()
    with open(path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            v = str(row.get(col, "")).strip().replace("-", "")[:8]
            if v:
                ds.add(v)
    return ds

for name, path, col in (("block_trade", os.path.join(RAW, "block_trade_full.csv"), "trade_date"),
                        ("sw_daily", os.path.join(RAW, "sw_daily.csv"), "trade_date")):
    ds = table_days(path, col)
    inw = sorted(d for d in ds if LO <= d <= HI)
    print("\n=== %s：全表日期 %d，窗口区间内 %d / 交易日 %d" % (name, len(ds), len(inw), len(cal_w)))
    allmonths = defaultdict(int)
    for d in ds:
        allmonths[d[:6]] += 1
    print("  全表按月有数据天数:")
    for m in sorted(allmonths):
        print("    %s  %d" % (m, allmonths[m]))
    hit = set(inw)
    print("  窗口区间命中日期（前后各20）: %s" % (",".join(inw[:20])))
    print("                                ... %s" % (",".join(inw[-20:])))
    miss = [d for d in cal_w if d not in hit]
    print("  缺失交易日 %d 个；连续缺失段（>=3 日）:")
    run = []
    prev = None
    segs = []
    for d in miss:
        if prev is None or cal_w.index(d) == cal_w.index(prev) + 1:
            run.append(d)
        else:
            if len(run) >= 3:
                segs.append((run[0], run[-1], len(run)))
            run = [d]
        prev = d
    if len(run) >= 3:
        segs.append((run[0], run[-1], len(run)))
    for a, b, n in segs[:25]:
        print("    %s .. %s  (%d)" % (a, b, n))
