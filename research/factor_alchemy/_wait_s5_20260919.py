# -*- coding: utf-8 -*-
"""轮询等待 S5 axis=1 批次完成 (或超时), 打印进度。

用法: python _wait_s5_20260919.py [max_seconds]
退出码: 0 = 已完成 (progress == n/n); 2 = 超时 (仍未完成)
"""
import io
import json
import sys
import time
from pathlib import Path

RES = Path(r'e:/quant/data/s5_axis1_rerun_result_default.json')
DEADLINE = time.time() + (float(sys.argv[1]) if len(sys.argv) > 1 else 540.0)
last = None
while True:
    try:
        d = json.loads(RES.read_text(encoding="utf-8"))
        prog = str(d.get("progress") or "")
        cur, tot = (prog.split("/") + ["0"])[:2]
        cur, tot = int(cur), int(tot)
        n = len(d.get("results") or [])
        n_pass = sum(1 for r in d["results"] if r.get("new_passed"))
        n_skip = sum(1 for r in d["results"] if r.get("eval_status") == "not_evaluated")
        line = "t=%s progress=%s n=%d PASS=%d 未评估=%d" % (
            time.strftime("%H:%M:%S"), prog, n, n_pass, n_skip)
        if line != last:
            print(line, flush=True)
            last = line
        if cur >= tot and tot > 0:
            print("DONE: %d/%d" % (cur, tot), flush=True)
            sys.exit(0)
    except Exception as e:
        print("poll err: %s: %s" % (type(e).__name__, e), flush=True)
    if time.time() > DEADLINE:
        print("TIMEOUT: 仍未完成 (progress=%s)" % (last or "?"), flush=True)
        sys.exit(2)
    time.sleep(20)
