# -*- coding: utf-8 -*-
"""2026-09-19 第 5 项复核: NOVEL_FACTOR_LIBRARY + 正交库 的 axis=1 现状 + 5 条广播修复真身。

只读。对标 _rank_axis_layer_audit_20260918.py, 但用当前磁盘状态。
"""
import io
import json
import re
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

ROOT = Path(r'e:/quant')
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "research" / "factor_alchemy"))

print("=" * 72)
print("A. NOVEL_FACTOR_LIBRARY (scripts/daily_factor_hypothesis.py)")
print("=" * 72)
import daily_factor_hypothesis as dh

NOVEL = getattr(dh, 'NOVEL_FACTOR_LIBRARY', [])
print(f"   条目数 = {len(NOVEL)}")

AXIS1 = re.compile(r'\.\s*\w+\s*\([^()]*\baxis\s*=\s*1\b')
RANK_AX1 = re.compile(r'\.\s*rank\s*\([^()]*\baxis\s*=\s*1\b')

n_ax1 = n_rank_ax1 = 0
for e in NOVEL:
    f = e.get('formula', '') or ''
    if AXIS1.search(f):
        n_ax1 += 1
    if RANK_AX1.search(f):
        n_rank_ax1 += 1
print(f"   axis=1 命中      = {n_ax1}   (9-18 记: 31)")
print(f"   其中 .rank 形式  = {n_rank_ax1}")

print("\n   -- 5 条广播修复条目 (9-18 声称改为 .sub(..., axis=0)) --")
by_id = {}
for e in NOVEL:
    for k in ('id', 'name', 'factor_name'):
        if k in e:
            by_id[str(e[k])] = e
            break
for fid in ["novel_040", "novel_070", "novel_073", "novel_099", "novel_203"]:
    e = by_id.get(fid)
    if not e:
        # 退一步: 在列表里按 id 字段模糊找
        e = next((x for x in NOVEL if fid in json.dumps(x, ensure_ascii=False)[:200]), None)
    if not e:
        print(f"   *** {fid} 未命中 (库内 id 样例: {list(by_id)[:5]})")
        continue
    f = e.get('formula', '') or ''
    has_ax0 = 'axis=0' in f.replace(' ', '')
    print(f"\n   {fid}  axis=0? {has_ax0}   len={len(f)}")
    print(f"   {'-'*66}")
    for ln in f.splitlines():
        print(f"      {ln}")

print()
print("=" * 72)
print("B. library_orthogonality_state.json (正交库)")
print("=" * 72)
p = ROOT / "data" / "library_orthogonality_state.json"
if p.exists():
    with open(p, 'r', encoding='utf-8') as fh:
        st = json.load(fh)
    facs = st.get('factors', [])
    print(f"   条目数 = {len(facs)}   (9-18 记: 999)")
    k1 = list(facs[0].keys()) if facs else []
    print(f"   键 = {k1}")
    fc = next((c for c in ('formula', 'expression', 'expr') if k1 and c in k1), None)
    la = sum(1 for x in facs if fc and AXIS1.search(x.get(fc) or ''))
    print(f"   axis=1 命中 = {la}   (9-18 记: 421)")
else:
    print("   *** 文件不存在")

print("\nDONE")
