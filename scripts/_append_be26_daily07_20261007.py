# -*- coding: utf-8 -*-
"""daily07 追加 BE.26 收口条目（二进制追加，CRLF-only）。

- 目标: .workbuddy/memory/2026-10-07.md（纯 CRLF 实测：crlf=21, bareLF=0, rrn=0）
- 结构: 尾部单 CRLF 收尾；节间单空行 ⇒ 追加块 = 空行 + 标题 + 空行 + 3 条
- 纪律: open('ab') 二进制追加预编码 bytes；写后四项对账 + 回读断言。
"""
import sys

P = '.workbuddy/memory/2026-10-07.md'

LINES = [
    '',
    '## BE.26 收口：MEMORY.md 同量压缩（2989→2971）+ RAG 收口（daily07 入库）（2026-10-07）',
    '',
    '- **MEMORY.md 同量压缩**：17 处编辑；去空白 2989→2971（余量 29）；备份 `MEMORY.md.bak_40r_20261007`；四项对账 crlf 52 / bareLF 0 / rrn 0；17 条回读断言全过。首版 dry-run 3018 超限 ⇒ 追加 5 处压缩后 2971。',
    '- **RAG 收口**：`SOURCES` 手动新增 `daily07`（硬编码日期不自动滚动）；`--only daily07` +3 chunk（锚点 725→727）；`--verify` 38/38 可达、缺失 0；`--anchor BE.24/BE.25` 直取命中本单元；selfcheck ok=726/727（唯一例外 `_intro`，既有契约）。',
    '- 归因注记：用「catalog.generated 时间戳 vs 源 mtime」判定 details(01:20)/changelog(01:19) 已被 01:22 批收录 ⇒ 只补 daily07，不重入 details（跨源同名 BE.7/BE.8 占位不扰动）。',
]

BLOCK = ''.join(ln + '\r\n' for ln in LINES).encode('utf-8')

raw = open(P, 'rb').read()
b0 = (len(raw), raw.count(b'\r\n'), raw.count(b'\n') - raw.count(b'\r\n'), raw.count(b'\r\r\n'))
assert raw.endswith(b'\r\n'), 'file must end with CRLF'
assert b0[2] == 0 and b0[3] == 0, 'pre-state not CRLF-only'
assert b'BE.26' not in raw, 'BE.26 already present'

with open(P, 'ab') as fh:
    fh.write(BLOCK)

rb = open(P, 'rb').read()
b1 = (len(rb), rb.count(b'\r\n'), rb.count(b'\n') - rb.count(b'\r\n'), rb.count(b'\r\r\n'))
print('before bytes=%d crlf=%d bareLF=%d rrn=%d' % b0)
print('after  bytes=%d crlf=%d bareLF=%d rrn=%d' % b1)
print('delta  bytes=%+d crlf=%+d' % (b1[0] - b0[0], b1[1] - b0[1]))
assert b1[2] == 0 and b1[3] == 0, 'post-state not CRLF-only'
assert b1[1] - b0[1] == len(LINES), 'crlf delta mismatch'
txt = rb.decode('utf-8')
assert txt.count('## BE.26') == 1, 'BE.26 heading count != 1'
assert '2989→2971' in txt and 'ok=726/727' in txt, 'key phrase missing'
print('OK: BE.26 appended; crlf delta=%d == %d lines; 读回断言全过' % (b1[1] - b0[1], len(LINES)))
sys.exit(0)
