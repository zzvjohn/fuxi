# -*- coding: utf-8 -*-
"""行尾工具: 核验并按需把目标文件转为 CRLF (只碰行尾, 不动内容)。

用法: python _lf2crlf_20261006.py <path> [<path> ...] [--apply]
不带 --apply 时只报告; 带 --apply 时仅在检出裸 LF 时重写 (先备份到 .lf_bak)。
"""
import os
import shutil
import sys


def report(p):
    with open(p, 'rb') as fh:
        data = fh.read()
    crlf = data.count(b'\r\n')
    lf = data.count(b'\n') - crlf
    cr = data.count(b'\r') - crlf
    return data, crlf, lf, cr


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    apply = '--apply' in sys.argv
    for p in args:
        if not os.path.exists(p):
            print('MISSING', p)
            continue
        data, crlf, lf, cr = report(p)
        print('%-60s crlf=%-5d bare_lf=%-4d bare_cr=%-3d bytes=%d'
              % (os.path.basename(p), crlf, lf, cr, len(data)))
        if apply and (lf or cr):
            shutil.copy2(p, p + '.lf_bak')
            fixed = data.replace(b'\r\n', b'\n').replace(b'\r', b'\n').replace(b'\n', b'\r\n')
            with open(p, 'wb') as fh:
                fh.write(fixed)
            _, c2, l2, r2 = report(p)
            print('   -> APPLIED crlf=%d bare_lf=%d bare_cr=%d (backup %s.lf_bak)'
                  % (c2, l2, r2, os.path.basename(p)))


if __name__ == '__main__':
    main()
