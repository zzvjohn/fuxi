# -*- coding: utf-8 -*-
"""2026-09-19 收尾核查: 行尾 / 缩进 / 全键频 / 新增轨迹 / 多次 _save 存活轨迹。只读。"""
import io
import json
import sys
from collections import Counter
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

ROOT = Path(r'e:/quant')
TRAJ = ROOT / "data" / "trajectory_log.json"
BAK = ROOT / "data" / "trajectory_log.json.bak_20260919_predround"

raw = TRAJ.read_bytes()
crlf = raw.count(b'\r\n')
bare = raw.count(b'\n') - crlf
print("=" * 70)
print("行尾 / 缩进 (项目铁律: 按文件实测再写)")
print("=" * 70)
print(f"   当前 trajectory_log.json: {len(raw)} B")
print(f"   CRLF={crlf}  裸LF={bare}  -> {'OK 无裸LF' if bare == 0 else '*** 混行尾!'}")
first = raw.decode('utf-8').splitlines()
print(f"   行数={len(first)}  第1行缩进={len(first[1]) - len(first[1].lstrip())}  "
      f"(TrajectoryLogger._save 约定 indent=2)")
print(f"   备份: CRLF={BAK.read_bytes().count(b'\r\n')}")

with open(TRAJ, 'r', encoding='utf-8') as f:
    d = json.load(f)
t = d["trajectories"]
print(f"\n   updated_at={d['updated_at']}  n={len(t)}")

kf = Counter()
for x in t:
    kf.update(x.keys())
print()
print("=" * 70)
print("全键频 (33)")
print("=" * 70)
for k in sorted(kf):
    flag = "" if kf[k] == len(t) else f"  <-- 非全量 ({kf[k]}/{len(t)})"
    print(f"   {k:26s} {kf[k]:5d}{flag}")

print()
print("=" * 70)
print("新增轨迹 (本轮 4 条)")
print("=" * 70)
with open(BAK, 'r', encoding='utf-8') as f:
    tbak = json.load(f)["trajectories"]
old_ids = {x['trajectory_id'] for x in tbak}
new = [x for x in t if x['trajectory_id'] not in old_ids]
for x in new:
    print(f"   {x['trajectory_id']}  paradigm={x.get('paradigm')!r}  "
          f"outcome={x.get('final_outcome')!r}  d_plus={x.get('d_plus_applied')}  "
          f"jq_validated={x.get('jq_validated')}")

print()
print("=" * 70)
print("schema 三处对齐复核 (dataclass / to_dict / from_dict)")
print("=" * 70)
sys.path.insert(0, str(ROOT / "research" / "factor_alchemy"))
import dataclasses
from trajectory_logger import MiningTrajectory

dc = {f.name for f in dataclasses.fields(MiningTrajectory)}
probe = MiningTrajectory(trajectory_id="probe_roundtrip")
td = set(probe.to_dict().keys())
lib = set(kf.keys())
print(f"   dataclass 字段 {len(dc)}")
print(f"   to_dict 键     {len(td)}")
print(f"   to_dict - dataclass = {sorted(td - dc)}   (应仅 _eval_log)")
print(f"   库键 - to_dict      = {sorted(lib - td)}   (应为空)")
print(f"   库键 - dataclass    = {sorted(lib - dc)}")
# 往返
import tempfile
tmp = Path(tempfile.mkdtemp()) / "t.json"
probe.jq_ic = 1.25
probe.d_plus_applied = True
probe.d_plus_source = "roundtrip"
probe.lessons = [{"lesson": "x"}]
rt = MiningTrajectory.from_dict(probe.to_dict())
ok = (rt.jq_ic == 1.25 and rt.d_plus_applied and rt.d_plus_source == "roundtrip"
      and rt.lessons == [{"lesson": "x"}])
print(f"   往返 (含 9 富字段) = {'PASS' if ok else '*** FAIL'}")
print("\nDONE")
