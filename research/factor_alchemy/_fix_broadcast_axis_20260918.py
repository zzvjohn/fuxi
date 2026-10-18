# -*- coding: utf-8 -*-
"""修复 NOVEL_FACTOR_LIBRARY 中 DataFrame − Series 广播对齐写错的公式 (2026-09-18, P-20260918-006)

缺陷类: `df - df.mean(axis=1)` — 左 DataFrame(日期×股票), 右 Series(索引=日期)。
pandas 的 `df - series` **按列对齐**, Series 索引(日期) 与 df 列(股票) 不匹配
⇒ 整表 NaN ⇒ 截面求值返回空表 ⇒ 因子**恒死且完全静默**。
证据: 4 条在正交库中 icir=0.0/status=reserve; e2e 测试截面路径产出 0 行、无任何日志。
正解: `.sub(series, axis=0)` 按**行**对齐。

先全库静态扫描同类写法, 再只改 NOVEL_FACTOR_LIBRARY 的命中项。
CRLF 保真。
"""
import io
import re
import json
import shutil
import py_compile

P = r'e:\quant\scripts\daily_factor_hypothesis.py'
LIB = r'e:\quant\data\library_orthogonality_state.json'

# 形状: 减法 `-` 之后出现 axis=1 归约 (mean/std/sum/median)。
# ⚠️ 这是**粗预筛**, 会低估: 首版用 `[^()+-]*` 限制左操作数, 只抓到 5 条中的 2 条
# (漏 novel_073 的括号嵌套与 novel_099 的 `.mean(axis=1).rolling(60).sum()` 链式)。
# **权威判据是运行时检测** (s5_joint_filter 新增的"全 NaN 诊断" + e2e 双路径测试),
# 本正则仅用于人工巡检。
SUSPECT = re.compile(r'\s-\s[^=!<>]*?\b(mean|std|sum|median)\s*\(\s*axis\s*=\s*1\s*\)')

FIXES = [
    ("novel_040",
     "close_p.pct_change(10) - close_p.mean(axis=1).pct_change(10)",
     "close_p.pct_change(10).sub(close_p.mean(axis=1).pct_change(10), axis=0)"),
    ("novel_070",
     "-(1 - (close_p.pct_change() - close_p.mean(axis=1).pct_change()).abs().rolling(20).mean())",
     "-(1 - (close_p.pct_change().sub(close_p.mean(axis=1).pct_change(), axis=0)).abs().rolling(20).mean())"),
    ("novel_073",
     "(close_p / close_p.shift(20) - 1) - (close_p.mean(axis=1) / close_p.mean(axis=1).shift(20) - 1)",
     "(close_p / close_p.shift(20) - 1).sub(close_p.mean(axis=1) / close_p.mean(axis=1).shift(20) - 1, axis=0)"),
    ("novel_099",
     # 该条源码是**跨行隐式字符串拼接**, 匹配串须按源码两行写 (不能按 eval 后的合并串)
     "'(close_p.pct_change(5) - close_p.pct_change().mean(axis=1).rolling(5).sum()) / '\n"
     "                   '((close_p.pct_change(60) - close_p.pct_change().mean(axis=1).rolling(60).sum()).abs() + 0.001)',",
     "'(close_p.pct_change(5).sub(close_p.pct_change().mean(axis=1).rolling(5).sum(), axis=0)) / '\n"
     "                   '((close_p.pct_change(60).sub(close_p.pct_change().mean(axis=1).rolling(60).sum(), axis=0)).abs() + 0.001)',"),
    ("novel_203",
     "(close_p.pct_change(20) - close_p.pct_change(20).mean(axis=1))",
     "(close_p.pct_change(20).sub(close_p.pct_change(20).mean(axis=1), axis=0))"),
]


def scan():
    print('=' * 74)
    print('静态扫描: DataFrame − Series(axis=1 归约) 广播对齐隐患')
    print('=' * 74)
    import sys
    sys.path.insert(0, r'e:/quant/scripts')
    import daily_factor_hypothesis as dh
    hits = []
    for e in dh.NOVEL_FACTOR_LIBRARY:
        f = str(e.get('formula', ''))
        if SUSPECT.search(f):
            hits.append(('NOVEL', e.get('id'), e.get('name'), f))
    print('  NOVEL_FACTOR_LIBRARY 命中: %d 条' % len(hits))
    for src, i, n, f in hits:
        print('    %-10s %-32s %s' % (i, str(n)[:32], f[:110]))
    lib = json.load(io.open(LIB, encoding='utf-8'))['factors']
    lh = []
    for f_ in lib:
        e = str(f_.get('expression', ''))
        if SUSPECT.search(e):
            lh.append((f_.get('name'), f_.get('status'), f_.get('icir'), e))
    print('  正交库命中: %d 条 (仅报告, 不改历史记录)' % len(lh))
    for n, s, i, e in lh[:20]:
        print('    %-40s status=%-10s icir=%-8s %s' % (str(n)[:40], s, i, e[:70]))
    return hits, lh


def patch():
    shutil.copy2(P, P + '.bak_20260918_axis')
    print('backup:', P.split('\\')[-1] + '.bak_20260918_axis')
    raw = open(P, 'rb').read()
    s = raw.decode('utf-8')
    n0 = s.count('\r\n')
    print('before: bytes=%d CRLF=%d bareLF=%d' % (len(raw), n0, s.count('\n') - n0))
    lf = s.replace('\r\n', '\n')
    for fid, old, new in FIXES:
        c = lf.count(old)
        print('  [%s] %-10s occurrences=%d' % ('OK ' if c == 1 else 'FAIL', fid, c))
        if c != 1:
            raise SystemExit('ABORT %s: expected 1 got %d' % (fid, c))
        lf = lf.replace(old, new, 1)
    out = lf.replace('\n', '\r\n')
    open(P, 'wb').write(out.encode('utf-8'))
    raw2 = open(P, 'rb').read()
    s2 = raw2.decode('utf-8')
    n = s2.count('\r\n')
    nl = s2.count('\n') - n
    print('after : bytes=%d CRLF=%d bareLF=%d' % (len(raw2), n, nl))
    assert nl == 0
    py_compile.compile(P, doraise=True)
    print('py_compile OK')
    for fid, old, new in FIXES:
        print('  verify %-10s old_left=%d new_found=%d' % (fid, s2.count(old), s2.count(new)))
        assert s2.count(old) == 0 and s2.count(new) == 1


if __name__ == '__main__':
    import sys
    hits, lh = scan()
    print()
    if '--apply' in sys.argv:
        patch()
    else:
        print('  [dry] 加 --apply 执行修复')
