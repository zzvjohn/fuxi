# -*- coding: utf-8 -*-
"""补丁: _evaluate_cross_section 动态字段补全时排除键列 (2026-09-18)"""
import py_compile

P = r'e:\quant\research\factor_alchemy\s5_joint_filter.py'

OLD = "\n".join([
    "        for _tok in sorted(set(re.findall(r'[A-Za-z_][A-Za-z0-9_]*', formula))):",
    "            if _tok in df.columns and _tok not in fields:",
    "                fields.append(_tok)",
])
NEW = "\n".join([
    "        _KEY_COLS = ('trade_date', 'ts_code')   # 键列不参与 pivot (非数值语义)",
    "        for _tok in sorted(set(re.findall(r'[A-Za-z_][A-Za-z0-9_]*', formula))):",
    "            if (_tok in df.columns and _tok not in fields",
    "                    and _tok not in _KEY_COLS):",
    "                fields.append(_tok)",
])


def main():
    raw = open(P, 'rb').read()
    s = raw.decode('utf-8')
    lf = s.replace('\r\n', '\n')
    print('occurrences:', lf.count(OLD))
    assert lf.count(OLD) == 1, 'expected 1'
    lf = lf.replace(OLD, NEW, 1)
    out = lf.replace('\n', '\r\n')
    open(P, 'wb').write(out.encode('utf-8'))
    b = open(P, 'rb').read()
    print('CRLF=%d bareLF=%d' % (b.count(b'\r\n'), b.count(b'\n') - b.count(b'\r\n')))
    assert b.count(b'\n') - b.count(b'\r\n') == 0
    py_compile.compile(P, doraise=True)
    print('py_compile OK')
    t = b.decode('utf-8')
    print('probe _KEY_COLS:', t.count('_KEY_COLS'))
    print('probe _AXIS1_CALL_RE:', t.count('_AXIS1_CALL_RE'))


if __name__ == '__main__':
    main()
