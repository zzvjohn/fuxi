# -*- coding: utf-8 -*-
"""MEMORY.md v45 落盘: 临时文件 → 目标文件, 显式 CRLF (禁 Edit, 会整文件转 LF)。"""
import hashlib
import re
from pathlib import Path

SRC = Path(r'e:/quant/research/factor_alchemy/_mem_v45_20260919.md')
DST = Path(r'e:/quant/.workbuddy/memory/MEMORY.md')

src_bytes_before = DST.read_bytes()
print('写入前 MEMORY.md bytes =', len(src_bytes_before),
      'md5 =', hashlib.md5(src_bytes_before).hexdigest()[:12])


def nw(s):
    return len(''.join(s.split()))


new = SRC.read_text(encoding='utf-8')
old = src_bytes_before.decode('utf-8')

# ── 断言: 新内容必须覆盖全部旧锚点 (防静默丢条目) ──
MUST_KEEP = [
    'TIE_RULE=on', 'COST_BPS=15', 'MIN_COVERAGE=0.9', 'TRADABLE_GATE', 'ADJ_PRICE',
    'ELIGIBLE_REQUIRE_S5', 'SPARSE_DIAG', 'DATE_CEIL=auto', 'FIRSTDAY', 'IDX_ALIGN',
    'FORMULA_RESOLVE', 'AUX_LAG', 'vol_scaled', 'S5_IR_THRESHOLD=0.7', 'STYLE_GUARD',
    'version_bump.py', '--rebuild', 'ALL_SWITCH_ENVS', 'setdefault',
    'newline=', 'CRLF', '纯 LF', 'Edit 会整文件转 LF', 'helper 定义了必须调用',
    '反引号', 'fail-open', 'off⊆on', 'INFO', '假 FAIL', '假 PASS', '假阴性',
    '必需通道交集末端', '结构化字段', '0.6082pp', 'psutil', '进度不动',
    'd_plus_*', '三处同步', 'axis=1', '7 类', 'legacy_e2e', '同族须穷举',
    '2026 腿', 'Q9', 'MIN_COVERAGE=0.9', '池依赖', '单通道剥离消融',
    '解释器', 'Python314', 'D:/RAG/venv',
]
miss = [k for k in MUST_KEEP if k not in new]
print('锚点校验:', f'{len(MUST_KEEP)-len(miss)}/{len(MUST_KEEP)}',
      'MISS =', miss)

n_old, n_new = nw(old), nw(new)
print(f'去空白: {n_old} -> {n_new}  (delta {n_new-n_old}, 余量 {3000-n_new})')
assert n_new <= 3000, '🔴 超限, 禁止落盘'
assert not miss, '🔴 有锚点丢失, 禁止落盘'

# ── 显式 CRLF 落盘 (newline='\\r\\n' 会把每个 \\n 译成 CRLF) ──
with open(DST, 'w', encoding='utf-8', newline='\r\n') as fh:
    fh.write(new)

# ── 回读验证 (禁只信写成功) ──
b = DST.read_bytes()
t = b.decode('utf-8')
crlf = b.count(b'\r\n')
bare = b.count(b'\n') - crlf
print()
print('回读: bytes =', len(b))
print('  CRLF =', crlf, ' 裸 LF =', bare, '  (裸 LF 必须为 0)')
print('  去空白 =', nw(t), '  行数 =', len(t.splitlines()))
print('  首行 =', t.splitlines()[0])
print('  末行 =', t.splitlines()[-1][:80])
assert bare == 0, '🔴 出现裸 LF'
assert nw(t) == n_new, '🔴 往返不一致'
assert t.splitlines() == new.splitlines(), '🔴 逐行不一致'
print()
print('PASS: 落盘完成, CRLF 保持, 内容逐行一致, 锚点无丢失')
