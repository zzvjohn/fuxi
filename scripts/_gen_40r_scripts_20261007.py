# -*- coding: utf-8 -*-
"""生成 40 轮批三件套脚本（受控复制 + 精确替换, 保持纯 LF）。

来源: reports/fuxi_v1_p003_fullrun_20261006/{extract_campaign_candidates.py, run_s5_p003batch.py, analyze_p003batch.py}
目标: reports/fuxi_v1_40rounds_20261007/{extract_campaign_candidates_40r.py, run_s5_40r_batch.py, analyze_40r_batch.py}

纪律: 每处替换 assert count 精确; 写后 py_compile + 行尾四项对账。
"""
import io
import os
import re
import py_compile

BASE = 'e:/quant'
SRC = os.path.join(BASE, 'reports', 'fuxi_v1_p003_fullrun_20261006')
DST = os.path.join(BASE, 'reports', 'fuxi_v1_40rounds_20261007')

RUN_NEW = 'reports/fuxi_v1_40rounds_20261007'
RUN_OLD = 'reports/fuxi_v1_p003_fullrun_20261006'

JOBS = [
    # (src_name, dst_name, [(old, new, expected_count), ...])
    ('extract_campaign_candidates.py', 'extract_campaign_candidates_40r.py', [
        (RUN_OLD, RUN_NEW, 2),
        ('"""P003 落地后完整运行批：从 campaign bundles 抽取候选公式总表 -> master_formulas.json',
         '"""伏羲 40 轮迭代批：从 campaign bundles 抽取候选公式总表 -> master_formulas.json', 1),
        ("'sources': ['p003_fullrun_20261006'],", "'sources': ['b40r_20261007'],", 1),
        ("'generated_at': '2026-10-06',", "'generated_at': '2026-10-07',", 1),
        ("'note': 'P003 落地后完整运行批（adaptive, seed 17）候选公式总表; unique by formula_hash',",
         "'note': '伏羲 40 轮迭代运行批（adaptive, seed 17, rounds=40）候选公式总表; unique by formula_hash',", 1),
    ]),
    ('run_s5_p003batch.py', 'run_s5_40r_batch.py', [
        (RUN_OLD, RUN_NEW, 1),
        ('"""P003 批 S5 组合层评测运行器（复用 144 批翻译函数保证口径一致）。',
         '"""40 轮批 S5 组合层评测运行器（复用 144 批翻译函数保证口径一致）。', 1),
        ("print('mark=%s (getter实测)  | P003 翻后默认档' % mark, flush=True)",
         "print('mark=%s (getter实测)  | v0.7.130 现行默认档' % mark, flush=True)", 1),
    ]),
    ('analyze_p003batch.py', 'analyze_40r_batch.py', [
        (RUN_OLD, RUN_NEW, 1),
        ('"""P003 批（11 条）S5 读数分析：三分类 + would_pass + 与 144 批对照 + 干旱判读。',
         '"""40 轮批 S5 读数分析：三分类 + would_pass + 与 144 批对照 + 干旱判读。', 1),
        ("用法: python analyze_p003batch.py", "用法: python analyze_40r_batch.py", 1),
        ("out = {'batch': 'p003_fullrun_20261006', 'n': len(recs), 'items': []}",
         "out = {'batch': 'b40r_20261007', 'n': len(recs), 'items': []}", 1),
        ("print('P003 批（%d 条）", "print('40 轮批（%d 条）", 1),
    ]),
]


def main():
    for src_name, dst_name, edits in JOBS:
        sp = os.path.join(SRC, src_name)
        dp = os.path.join(DST, dst_name)
        raw = io.open(sp, encoding='utf-8', newline='').read()
        t = raw
        for old, new, exp in edits:
            cnt = t.count(old)
            assert cnt == exp, 'count mismatch for %r in %s: got %d expected %d' % (
                old[:60], src_name, cnt, exp)
            t = t.replace(old, new)
        data = t.encode('utf-8')
        with io.open(dp, 'wb') as fh:
            fh.write(data)
        py_compile.compile(dp, doraise=True)
        rb = io.open(dp, 'rb').read()
        crlf = rb.count(b'\r\n')
        bare = rb.count(b'\n') - crlf
        rrn = rb.count(b'\r\r\n')
        print('[ok] %-36s bytes=%6d crlf=%4d bareLF=%4d rrn=%d  py_compile=OK'
              % (dst_name, len(rb), crlf, bare, rrn))


if __name__ == '__main__':
    main()
