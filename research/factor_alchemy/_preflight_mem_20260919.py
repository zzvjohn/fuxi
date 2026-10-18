# -*- coding: utf-8 -*-
"""2026-09-19 记账前置探测: 目标文件行尾 / 是否存在 / MEMORY.md 限额。只读。"""
import io
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

ROOT = Path(r'e:/quant')
targets = [
    ROOT / "reports" / "提案落地_20260918_D+字段根治与轴语义修复.md",
    ROOT / ".workbuddy" / "memory" / "MEMORY.md",
    ROOT / ".workbuddy" / "memory" / "_fuxi_landing_details.md",
    ROOT / ".workbuddy" / "memory" / "2026-09-19.md",
    ROOT / ".workbuddy" / "memory" / "2026-09-18.md",
    ROOT / "CHANGELOG.md",
]
print(f"{'file':52s} {'存在':>4s} {'size':>9s} {'CRLF':>8s} {'裸LF':>7s}")
for p in targets:
    if not p.exists():
        print(f"{p.name:52s} {'NO':>4s}")
        continue
    raw = p.read_bytes()
    crlf = raw.count(b'\r\n')
    bare = raw.count(b'\n') - crlf
    print(f"{p.name:52s} {'YES':>4s} {len(raw):>9d} {crlf:>8d} {bare:>7d}")

print()
p = ROOT / ".workbuddy" / "memory" / "MEMORY.md"
txt = p.read_text(encoding='utf-8')
nospace = len("".join(txt.split()))
print(f"MEMORY.md raw len = {len(txt)}")
print(f"MEMORY.md 去空白  = {nospace}  (限额 3000)  余量 = {3000 - nospace}")

print()
p2 = ROOT / "reports" / "提案落地_20260918_D+字段根治与轴语义修复.md"
if p2.exists():
    t = p2.read_text(encoding='utf-8')
    print(f"9-18 报告 行数={len(t.splitlines())}")
    print("   --- 最后 20 行 ---")
    for ln in t.splitlines()[-20:]:
        print("   | " + ln)
