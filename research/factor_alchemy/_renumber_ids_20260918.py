"""提案号去冲突 (2026-09-18)

背景: 本批两项修复在代码注释里写的 P-20260918-005 / -006 已被并行会话
（公式重放截断点 / 运行时因果重放校验）占用 ⇒ 本批重新编号:
  · D+ schema 结构性断供修复   : P-20260918-005 -> P-20260918-021
  · .rank(axis=1) 跨层轴缺陷修复: P-20260918-006 -> P-20260918-022

🔴 纪律: ralph_loop.py / s5_joint_filter.py 是 CRLF, 禁用 Edit (会静默转 LF)
   ⇒ 统一走「读字节 -> 解码 -> 替换 -> 按原行尾写回」文件通道。
"""
import os
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(BASE))

# (相对路径, {旧号: 新号})
JOBS = [
    ('research/factor_alchemy/trajectory_logger.py', {'P-20260918-005': 'P-20260918-021'}),
    ('research/factor_alchemy/trigger_d_plus.py',    {'P-20260918-005': 'P-20260918-021'}),
    ('research/factor_alchemy/ralph_loop.py',        {'P-20260918-005': 'P-20260918-021'}),
    ('research/factor_alchemy/s5_joint_filter.py',   {'P-20260918-006': 'P-20260918-022'}),
]

REPORT = []
ok = True
for rel, mapping in JOBS:
    p = os.path.join(ROOT, rel)
    raw = open(p, 'rb').read()
    crlf = raw.count(b'\r\n')
    bare_lf = raw.count(b'\n') - crlf
    nl = '\r\n' if crlf and bare_lf == 0 else ('\r\n' if crlf > bare_lf else '\n')
    txt = raw.decode('utf-8')
    n = {}
    for old, new in mapping.items():
        c = txt.count(old)
        n[old] = c
        if c:
            txt = txt.replace(old, new)
    # 写回: 按原行尾
    out = txt.encode('utf-8')
    if nl == '\r\n':
        # 解码后文本仍是 CRLF 形态, 无需转换; 但为稳妥做一次规范化
        out = txt.replace('\r\n', '\n').replace('\n', '\r\n').encode('utf-8')
    with open(p, 'wb') as f:
        f.write(out)
    # 复核
    raw2 = open(p, 'rb').read()
    crlf2 = raw2.count(b'\r\n')
    bare2 = raw2.count(b'\n') - crlf2
    t2 = raw2.decode('utf-8')
    left = sum(t2.count(o) for o in mapping)
    # 🔴 断言须匹配真实形态: 本批同时改了 CRLF 文件与**原生 LF** 文件,
    #    判据只能是「行尾与输入逐类相等」, 不能要求 bareLF==0
    #    (2026-09-18 首版即因此误报 FAIL)。
    eol_same = (crlf2 == crlf and bare2 == bare_lf)
    REPORT.append((rel, n, f'CRLF {crlf}->{crlf2} bareLF {bare_lf}->{bare2}',
                   f'残留旧号={left}', f'行尾不变={eol_same}'))
    if (not eol_same) or left != 0:
        ok = False

print('=== 去冲突重编号 ===')
for rel, n, eol, left, same in REPORT:
    print(f'  {rel}')
    print(f'      替换 {n}  |  {eol}  |  {left}  |  {same}')
print('RESULT =', 'ALL GREEN' if ok else 'FAIL')
sys.exit(0 if ok else 1)
