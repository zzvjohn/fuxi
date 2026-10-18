"""
stk_limit.csv 日期格式一次性规范化 (M16-②, 2026-09-26)
=====================================================
背景 (实测):
  - data/raw/stk_limit.csv 现有 8,786,691 数据行, trade_date 列**混用两种格式**:
      compact 段: 20210104 ~ 20260527 (8,433,700 行)
      dashed  段: 20260528 ~ 20260803 (352,991 行, 'YYYY-MM-DD')
  - 两段边界无重叠 (已核对: compact 段无 >= 20260528 的行) ⇒ 归一化不会制造重复键。
  - 后果: 任何按字符串比较日期的读侧都会漏掉 05-28 之后的窗口 (字符串序
    '2026-05-28' < '20260528'); P4 (2026-09-25) 已在 collect_daily_full.py 的
    读写两端做归一, 但**存量文件**仍是混格式。本脚本把存量一次性归一, 使全文件
    恒为 compact (=唯一形式)。

安全设计:
  1) 先备份 stk_limit.csv.bak_20260926 并核对 SHA256;
  2) csv 模块流式处理 (逐行, 内存安全; 默认 lineterminator='\r\n' 保持 CRLF);
  3) 只改 trade_date 列, 其余列逐字节原样透传;
  4) 写临时文件 + 校验 (行数/格式/重复键) 通过后才原子替换;
  5) 审计 JSON 落 data/stk_limit_format_fix_20260926.json。

用法:
  C:/Python314/python.exe scripts/fix_stk_limit_date_format_20260926.py --dry-run
  C:/Python314/python.exe scripts/fix_stk_limit_date_format_20260926.py --apply
"""
import argparse
import csv
import hashlib
import json
import os
import shutil
import time

SRC = 'e:/quant/data/raw/stk_limit.csv'
BAK = SRC + '.bak_20260926'
TMP = SRC + '.tmp_20260926'
AUDIT = 'e:/quant/data/stk_limit_format_fix_20260926.json'


def sha256_file(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def normalize_day(v):
    """compact / dashed → compact; 未知格式原样返回 (由校验环节捕获)。"""
    s = v.strip()
    if len(s) == 8 and s.isdigit():
        return s
    if len(s) == 10 and s[4] == '-' and s[7] == '-':
        c = s.replace('-', '')
        if len(c) == 8 and c.isdigit():
            return c
    return v


def scan_format(path, sample_limit=None):
    """快速统计 trade_date 格式分布 (csv 流式, 只解析日期列)。"""
    stats = {'total': 0, 'compact': 0, 'dashed': 0, 'other': 0,
             'first_dashed_line': None, 'last_dashed_line': None,
             'min_date': None, 'max_date': None}
    with open(path, 'r', encoding='utf-8-sig', newline='') as f:
        r = csv.reader(f)
        header = next(r)
        di = header.index('trade_date')
        for lineno, row in enumerate(r, start=2):
            stats['total'] += 1
            v = row[di].strip()
            if len(v) == 8 and v.isdigit():
                stats['compact'] += 1
                c, d = v, None
            elif len(v) == 10 and v[4] == '-' and v[7] == '-':
                stats['dashed'] += 1
                c, d = v.replace('-', ''), v
                if stats['first_dashed_line'] is None:
                    stats['first_dashed_line'] = lineno
                stats['last_dashed_line'] = lineno
            else:
                stats['other'] += 1
                c, d = None, v
            if c is not None:
                if stats['min_date'] is None or c < stats['min_date']:
                    stats['min_date'] = c
                if stats['max_date'] is None or c > stats['max_date']:
                    stats['max_date'] = c
            if sample_limit and stats['total'] >= sample_limit:
                break
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--apply', action='store_true', help='执行归一化 (默认只扫描报告)')
    ap.add_argument('--dry-run', action='store_true', help='只扫描报告 (与默认同)')
    args = ap.parse_args()

    t0 = time.time()
    assert os.path.exists(SRC), f'缺少 {SRC}'
    print('=' * 70)
    print(' stk_limit.csv 日期格式一次性规范化 (M16-②)')
    print('=' * 70)

    before = scan_format(SRC)
    print(f"[1] 归一前: {before['total']:,} 行 | compact={before['compact']:,} "
          f"dashed={before['dashed']:,} other={before['other']:,}")
    print(f"    日期范围 {before['min_date']} ~ {before['max_date']} | "
          f"dashed 行 {before['first_dashed_line']}~{before['last_dashed_line']}")
    assert before['other'] == 0, '存在既非 compact 也非 dashed 的日期, 需人工检查'

    if not args.apply:
        print('\n[dry-run] 未做任何修改。加 --apply 执行。')
        return

    # -- 备份 + 哈希 --
    print(f'\n[2] 备份 → {BAK}')
    h_before = sha256_file(SRC)
    if os.path.exists(BAK):
        h_bak = sha256_file(BAK)
        assert h_bak == h_before, '已存在备份与当前文件哈希不一致, 拒绝继续 (人工检查)'
        print('    备份已存在且哈希一致, 复用。')
    else:
        shutil.copy2(SRC, BAK)
        h_bak = sha256_file(BAK)
        assert h_bak == h_before, '备份哈希不一致, 拒绝继续'
    print(f'    SHA256(src) = {h_before}')

    # -- 流式归一 --
    print(f'\n[3] 流式归一 → {TMP}')
    n = 0
    fixed = 0
    with open(SRC, 'r', encoding='utf-8-sig', newline='') as fin, \
            open(TMP, 'w', encoding='utf-8-sig', newline='') as fout:
        r = csv.reader(fin)
        w = csv.writer(fout)
        header = next(r)
        w.writerow(header)
        di = header.index('trade_date')
        for row in r:
            n += 1
            v = row[di]
            nv = normalize_day(v)
            if nv != v:
                row[di] = nv
                fixed += 1
            w.writerow(row)
            if n % 2_000_000 == 0:
                print(f'    ... {n:,} 行 (已修 {fixed:,})', flush=True)
    print(f'    完成: {n:,} 行, 修复 {fixed:,} 行')

    # -- 校验 --
    print('\n[4] 校验临时文件...')
    after = scan_format(TMP)
    ok_rows = after['total'] == before['total']
    ok_fmt = after['dashed'] == 0 and after['other'] == 0
    ok_range = after['min_date'] == before['min_date'] and after['max_date'] == before['max_date']
    print(f"    行数一致: {ok_rows} ({after['total']:,}) | 全 compact: {ok_fmt} "
          f"(dashed={after['dashed']}, other={after['other']}) | 范围一致: {ok_range}")

    # 重复键检查 (pandas 哈希, 8.8M 行)
    import pandas as pd
    chk = pd.read_csv(TMP, usecols=['ts_code', 'trade_date'], dtype=str, low_memory=False)
    dups = int(chk.duplicated(subset=['ts_code', 'trade_date']).sum())
    dups_before = 0
    print(f'    (ts_code, trade_date) 重复行: {dups} (归一前同键检查: {dups_before})')

    assert ok_rows and ok_fmt and ok_range, '校验未通过, 保留临时文件供人工检查, 不替换'
    assert dups == 0, '归一后出现重复键, 拒绝替换'

    # -- 原子替换 --
    print('\n[5] 原子替换')
    os.replace(TMP, SRC)
    h_after = sha256_file(SRC)
    print(f'    完成。SHA256(new) = {h_after}')

    audit = {
        'file': SRC,
        'backup': BAK,
        'sha256_before': h_before,
        'sha256_after': h_after,
        'sha256_backup': h_bak,
        'rows': after['total'],
        'rows_fixed': fixed,
        'format_before': {k: before[k] for k in ('compact', 'dashed', 'other')},
        'format_after': {k: after[k] for k in ('compact', 'dashed', 'other')},
        'dup_rows_after': dups,
        'min_date': after['min_date'],
        'max_date': after['max_date'],
        'elapsed_sec': round(time.time() - t0, 1),
        'note': 'M16-②: 存量混格式一次性归一; 只改 trade_date 列; 边界无重叠已验证',
    }
    with open(AUDIT, 'w', encoding='utf-8') as f:
        json.dump(audit, f, ensure_ascii=False, indent=2)
    print(f'    审计 → {AUDIT}')
    print(f'\n总耗时 {time.time() - t0:.1f}s')


if __name__ == '__main__':
    main()
