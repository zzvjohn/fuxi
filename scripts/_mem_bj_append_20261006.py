# -*- coding: utf-8 -*-
"""_mem_bj_append_20261006.py — BJ 方案 A 记忆追加（daily06 BE.22 / details BE.23）

纪律（fuxi-memory-rag 铁律 #15）:
  ① 二进制追加 open('ab')，禁用全文件行尾规范化；
  ② 块尾空行显式保留（块内 LF 归一后转 CRLF，尾部补一个 CRLF ⇒ 文件末 '\\r\\n\\r\\n'）；
  ③ 幂等前置断言（daily 不含 '## BE.22 '；details 不含 '### BE.23 '）；
  ④ 写后四项对账（bytes / CRLF / 裸LF / rrn）+ 锚点唯一性断言。
用法: python _mem_bj_append_20261006.py           # dry-run
      python _mem_bj_append_20261006.py --apply
"""
import sys
from pathlib import Path

MEM = Path(r'E:\quant\.workbuddy\memory')
JOBS = [
    (MEM / '_tmp_bj_daily22_20261006.md', MEM / '2026-10-06.md', '## BE.22 '),
    (MEM / '_tmp_bj_details23_20261006.md', MEM / '_fuxi_landing_details.md', '### BE.23 '),
]


def norm_block(p):
    t = p.read_text(encoding='utf-8')
    t = t.replace('\r\n', '\n').replace('\r', '\n')
    t = t.rstrip('\n') + '\n'
    return t.replace('\n', '\r\n') + '\r\n'   # 尾部 CRLF + 空行 CRLF


def stat(b):
    return dict(bytes=len(b), crlf=b.count(b'\r\n'),
                bare=b.count(b'\n') - b.count(b'\r\n'), rrn=b.count(b'\r\r\n'))


def main():
    apply = '--apply' in sys.argv
    blocks = []
    for src, dst, anchor in JOBS:
        assert src.exists(), 'missing tmp: %s' % src
        blk = norm_block(src)
        tgt = dst.read_bytes()
        if anchor.encode('utf-8') in tgt:
            print('[skip] 目标已含锚点 %r -> %s' % (anchor.strip(), dst.name))
            continue
        assert tgt.endswith(b'\r\n\r\n'), '目标不以 CRLF 空行结尾: %s' % dst.name
        blocks.append((src, dst, anchor, blk))

    for src, dst, anchor, blk in blocks:
        b = blk.encode('utf-8')
        print('[plan] %-34s -> %-26s +%d bytes (crlf+%d)' %
              (src.name, dst.name, len(b), b.count(b'\r\n')))
        if apply:
            before = stat(dst.read_bytes())
            with open(dst, 'ab') as f:
                f.write(b)
            after = stat(dst.read_bytes())
            print('       %s: bytes %d->%d | crlf %d->%d | bare %d | rrn %d'
                  % (dst.name, before['bytes'], after['bytes'],
                     before['crlf'], after['crlf'], after['bare'], after['rrn']))
            txt = dst.read_bytes().decode('utf-8')
            assert txt.count(anchor) == 1, 'anchor 非唯一: %r x%d' % (anchor, txt.count(anchor))
            assert after['bare'] == 0 and after['rrn'] == 0, '行尾异常'
            assert after['bytes'] - before['bytes'] == len(b), '增量不符'
    if not apply:
        print('[dry-run] 未写盘。加 --apply 落地。')
    return 0


if __name__ == '__main__':
    sys.exit(main())
