# -*- coding: utf-8 -*-
"""2026-09-19 差分: 主流水线首次 _save() 前后 trajectory_log.json 到底变了什么。

只读。备份 = 跑前 (5,427,358 B) / 当前 = 跑中。
"""
import io
import json
import sys
from collections import Counter
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

ROOT = Path(r'e:/quant')
CUR = ROOT / "data" / "trajectory_log.json"
BAK = ROOT / "data" / "trajectory_log.json.bak_20260919_predround"


def load(p):
    raw = p.read_bytes()
    with open(p, 'r', encoding='utf-8') as f:
        d = json.load(f)
    return raw, d


raw_b, db = load(BAK)
raw_c, dc = load(CUR)
tb, tc = db["trajectories"], dc["trajectories"]

print("=" * 74)
print("体积 / 行尾")
print("=" * 74)
print(f"   备份 {len(raw_b):>9d} B  CRLF={raw_b.count(b'\\r\\n'):>7d}  裸LF={raw_b.count(b'\\n') - raw_b.count(b'\\r\\n'):>7d}")
print(f"   当前 {len(raw_c):>9d} B  CRLF={raw_c.count(b'\\r\\n'):>7d}  裸LF={raw_c.count(b'\\n') - raw_c.count(b'\\r\\n'):>7d}")
print(f"   净减 {len(raw_b) - len(raw_c)} B")
print(f"   行首缩进 width: 备份首行 {len(raw_b.splitlines()[1]) - len(raw_b.splitlines()[1].lstrip())}, "
      f"当前 {len(raw_c.splitlines()[1]) - len(raw_c.splitlines()[1].lstrip())}")

print()
print("=" * 74)
print("顶层 & 记录数")
print("=" * 74)
print(f"   备份 keys {sorted(db.keys())}")
print(f"   当前 keys {sorted(dc.keys())}")
print(f"   备份 n={len(tb)}  版本={db.get('version')}  updated={db.get('updated_at')}")
print(f"   当前 n={len(tc)}  版本={dc.get('version')}  updated={dc.get('updated_at')}")

print()
print("=" * 74)
print("键频差分")
print("=" * 74)
kb, kc = Counter(), Counter()
for t in tb:
    kb.update(t.keys())
for t in tc:
    kc.update(t.keys())
allk = sorted(set(kb) | set(kc))
print(f"   {'key':26s} {'备份':>8s} {'当前':>8s}  变化")
for k in allk:
    a, b = kb.get(k, 0), kc.get(k, 0)
    if a != b:
        print(f"   {k:26s} {a:>8d} {b:>8d}  {'+' if b>a else ''}{b-a}")
print("   (仅列变化项)")

print()
print("=" * 74)
print("记录级: 键集合是否被裁剪")
print("=" * 74)
id2b = {t['trajectory_id']: t for t in tb}
id2c = {t['trajectory_id']: t for t in tc}
lost_ids = set(id2b) - set(id2c)
new_ids = set(id2c) - set(id2b)
print(f"   消失的 id {len(lost_ids)}  新增的 id {len(new_ids)}  {sorted(new_ids)[:6]}")

common = sorted(set(id2b) & set(id2c))
removed_keys = Counter()
added_keys = Counter()
for tid in common:
    a = set(id2b[tid]) - set(id2c[tid])
    b = set(id2c[tid]) - set(id2b[tid])
    removed_keys.update(a)
    added_keys.update(b)
print(f"   记录内被删除的键: {dict(removed_keys)}")
print(f"   记录内被新增的键: {dict(added_keys)}")

print()
print("=" * 74)
print("内容体量差分 (解释体积净减)")
print("=" * 74)
FIELDS = ['hypothesis', 'expression', 'code', 'backtest']


def blen(t):
    tot = 0
    for f in FIELDS:
        s = t.get(f)
        if isinstance(s, dict):
            tot += len(s.get('content') or '')
    return tot


sb = sum(blen(t) for t in tb)
sc = sum(blen(t) for t in tc)
print(f"   4 个子步骤 content 总字符: 备份 {sb:,}  ->  当前 {sc:,}   (差 {sc - sb:+,})")

eb = sum(len(json.dumps(t.get('_eval_log') or [], ensure_ascii=False)) for t in tb)
ec = sum(len(json.dumps(t.get('_eval_log') or [], ensure_ascii=False)) for t in tc)
print(f"   _eval_log 总字符:          备份 {eb:,}  ->  当前 {ec:,}   (差 {ec - eb:+,})")

# 逐记录 content 长度差异
trunc = 0
grown = 0
for tid in common:
    x, y = blen(id2b[tid]), blen(id2c[tid])
    if y < x:
        trunc += 1
    elif y > x:
        grown += 1
print(f"   content 变短的记录 {trunc} / 变长 {grown} / 相同 {len(common)-trunc-grown}")
print("\nDONE")
