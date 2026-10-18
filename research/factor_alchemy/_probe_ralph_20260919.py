# -*- coding: utf-8 -*-
"""主流水线进度探针 (只读): 进程状态 + 日志 + trajectory_log.json 变化。"""
import io
import json
import os
import sys
import time
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

ROOT = Path(r'e:/quant')
LOG = ROOT / "research" / "factor_alchemy" / "_ralph_round_dplus_20260919.log"
TRAJ = ROOT / "data" / "trajectory_log.json"


def ts(p):
    return time.strftime('%H:%M:%S', time.localtime(p.stat().st_mtime))


print(f"[{time.strftime('%H:%M:%S')}] 进度探针")
print("-" * 60)

import psutil
procs = []
for p in psutil.process_iter(['pid', 'name', 'cmdline', 'create_time',
                              'memory_info', 'cpu_percent']):
    try:
        cl = ' '.join(p.info['cmdline'] or [])
    except Exception:
        continue
    if p.info['pid'] == os.getpid():
        continue
    if 'run_v4_pipeline.py' in cl or '--ralph' in cl:
        el = time.time() - p.info['create_time']
        rss = p.info['memory_info'].rss / 1e9
        try:
            cpu = p.cpu_percent(interval=0.5)
        except Exception:
            cpu = -1
        procs.append((p.info['pid'], round(el), round(rss, 2), cpu))
print(f"流水线进程: {len(procs)}")
for pid, el, rss, cpu in procs:
    print(f"   PID {pid}  已运行 {el}s  RSS {rss} GB  CPU {cpu}%")
if not procs:
    print("   *** 无进程 (可能已结束)")

print()
print(f"日志: {LOG.stat().st_size} B, mtime {ts(LOG)}")
txt = LOG.read_bytes().decode('utf-8', 'replace')
lines = txt.splitlines()
print(f"   行数 {len(lines)}")
print("   --- 尾部 30 行 ---")
for ln in lines[-30:]:
    print("   | " + ln)

print()
print(f"trajectory_log.json: {TRAJ.stat().st_size} B, mtime {ts(TRAJ)}")
try:
    with open(TRAJ, 'r', encoding='utf-8') as f:
        d = json.load(f)
    t = d["trajectories"]
    print(f"   n_trajectories = {len(t)}")
    print(f"   d_plus_applied = {sum(1 for x in t if x.get('d_plus_applied'))}")
    print(f"   lessons 键     = {sum(1 for x in t if 'lessons' in x)}")
    print(f"   键并集         = {len({k for x in t for k in x})}")
except Exception as e:
    print(f"   *** 读取失败 (可能正在写入): {e}")
