# -*- coding: utf-8 -*-
"""2026-09-19 基线快照 + 备份 (跑主流水线之前)。

1) 备份 data/trajectory_log.json  (之前不覆盖 9-18 的备份)
2) 落基线指纹 JSON: 键集/键频/D+ 计数/富字段覆盖
3) 第 5 项核验: 因子库 axis=1 统计 + 5 条广播修复条目逐条读出
"""
import io
import json
import shutil
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

ROOT = Path(r'e:/quant')
TRAJ = ROOT / "data" / "trajectory_log.json"
LIB = ROOT / "data" / "unified_factor_pool.csv"
STAMP = "20260919_predround"
BACKUP = ROOT / "data" / f"trajectory_log.json.bak_{STAMP}"
FP = ROOT / "data" / f"_dplus_baseline_{STAMP}.json"

print("=" * 72)
print("1) 备份")
print("=" * 72)
shutil.copy2(TRAJ, BACKUP)
print(f"   {BACKUP}  ({BACKUP.stat().st_size} B)")

print()
print("=" * 72)
print("2) 基线指纹 (主流水线开跑前)")
print("=" * 72)
raw = TRAJ.read_bytes()
with open(TRAJ, 'r', encoding='utf-8') as f:
    data = json.load(f)
trajs = data["trajectories"]

keyfreq = Counter()
for t in trajs:
    keyfreq.update(t.keys())

RICH7 = ["jq_maxdd", "jq_ic", "jq_icir", "jq_rating",
         "jq_plus_ic_pct", "jq_backtest_period", "jq_notes"]
marked = [t["trajectory_id"] for t in trajs if t.get("d_plus_applied")]
src_map = {t["trajectory_id"]: t.get("d_plus_source", "") for t in trajs if t.get("d_plus_applied")}

fp = {
    "captured_at": datetime.now().isoformat(),
    "phase": "pre_round",
    "file_size": len(raw),
    "n_trajectories": len(trajs),
    "n_keys_in_union": len(keyfreq),
    "keyfreq": dict(keyfreq),
    "jq_validated": sum(1 for t in trajs if t.get("jq_validated")),
    "d_plus_applied": len(marked),
    "d_plus_debt": sum(1 for t in trajs if t.get("jq_validated") and not t.get("d_plus_applied")),
    "d_plus_source_dist": dict(Counter(t.get("d_plus_source", "") for t in trajs if t.get("d_plus_applied"))),
    "rich7_nonempty": sum(1 for t in trajs
                          if any(t.get(k) not in (None, "", [], {}) for k in RICH7)),
    "lessons_key": keyfreq.get("lessons", 0),
    "marked_ids": sorted(marked),
    "marked_source_map": src_map,
}
with open(FP, 'w', encoding='utf-8', newline='') as f:
    json.dump(fp, f, ensure_ascii=False, indent=2)
print(f"   指纹已落盘: {FP}")
print(f"   n_trajectories      = {fp['n_trajectories']}")
print(f"   键并集              = {fp['n_keys_in_union']}")
print(f"   jq_validated        = {fp['jq_validated']}")
print(f"   d_plus_applied      = {fp['d_plus_applied']}")
print(f"   欠账                = {fp['d_plus_debt']}")
print(f"   source 分布          = {fp['d_plus_source_dist']}")
print(f"   富7字段非空覆盖      = {fp['rich7_nonempty']}")
print(f"   lessons 键频         = {fp['lessons_key']}   <-- 0 = schema 尚未经 _save() 走一遍")

print()
print("=" * 72)
print("3) 第 5 项: 因子库 axis=1 统计 + 5 条广播修复条目")
print("=" * 72)
import csv
import re
AXIS1 = re.compile(r'\.\s*\w+\s*\([^()]*\baxis\s*=\s*1\b')
rows = list(csv.DictReader(open(LIB, 'r', encoding='utf-8-sig')))
print(f"   因子库条目 = {len(rows)}")
cols = list(rows[0].keys())
print(f"   列名 = {cols}")
fcol = next((c for c in cols if c.lower() in ('formula', 'expression')), None)
ncol = next((c for c in cols if c.lower() in ('name', 'factor_name')), None)
n_ax1 = sum(1 for r in rows if fcol and AXIS1.search(r.get(fcol) or ""))
print(f"   axis=1 命中 = {n_ax1}")
print(f"\n   -- 5 条广播修复条目逐条读出 --")
for fid in ["novel_040", "novel_070", "novel_073", "novel_099", "novel_203"]:
    hit = [r for r in rows if (r.get(ncol) or "") == fid]
    if not hit:
        print(f"   *** {fid} 未命中")
        continue
    r = hit[0]
    fx = r.get(fcol) or ""
    print(f"   {fid}: axis0={'axis=0' in fx or 'axis = 0' in fx} len={len(fx)}")
    print(f"      {fx[:220]}")
print("\nDONE")
