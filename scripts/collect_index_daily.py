"""
指数日线增量更新 — 000985 中证全指 / 000905 中证500 / 000300 沪深300 (M16-①)
==========================================================================
背景 (2026-09-26 实测):
  · 000985/000905 此前无稳定日更入口, 冻结在 2026-09-09; 000300 冻结在 2026-09-10。
  · S5 的 FUXI_S5_DATE_CEIL=auto 取「价格 ∩ 基准指数」必需通道交集末端
    (research/factor_alchemy/s5_joint_filter.py: ~L1051) ⇒ 指数冻结会把**裁决窗**
    钉死在指数末端 (实测落后价格末端 11 个交易日)。
  · 本脚本把三指数纳入日更 (由 stage0_daily_update_v3.py Phase 1.6 调用)。

格式契约 (只追加新行, 不改写历史行; 与各文件现役格式逐列一致):
  index_000985_daily.csv / index_000905_daily.csv:
      ts_code,trade_date(YYYYMMDD),open,high,low,close,vol,amount,pct_chg
  index_000300_daily.csv:
      ts_code,trade_date(YYYY-MM-DD),close,open,high,low,pre_close,change,pct_chg,vol,amount,date(YYYYMMDD)
  说明: 000985 近年段使用 '000985.CSI' 后缀 (旧段为 .SH, 两段日期不重叠, 不做改写);
        000300 行情列为 Tushare index_daily 标准 11 字段 + 冗余 date 列。

规则:
  · 增量: 从存量最大日期次日拉取到今日; 幂等 (已存在的 (ts_code,date) 不重复追加);
  · 只追加 (mode='a', header=False, utf-8-sig, CRLF 与存量一致);
  · --dry-run 只报告缺口。

用法:
  C:/Python314/python.exe scripts/collect_index_daily.py [--dry-run] [--only 000985]
"""
from __future__ import annotations

import argparse
import os
import time
from datetime import datetime

import pandas as pd

RAW = 'e:/quant/data/raw'

F9 = 'ts_code,trade_date,open,high,low,close,vol,amount,pct_chg'
F11 = 'ts_code,trade_date,close,open,high,low,pre_close,change,pct_chg,vol,amount'

COL9 = ['ts_code', 'trade_date', 'open', 'high', 'low', 'close', 'vol', 'amount', 'pct_chg']
COL12 = ['ts_code', 'trade_date', 'close', 'open', 'high', 'low', 'pre_close',
         'change', 'pct_chg', 'vol', 'amount', 'date']

SPECS = [
    {'key': '000985', 'name': 'index_000985',
     'path': os.path.join(RAW, 'index_000985_daily.csv'),
     'codes': ['000985.CSI', '000985.SH'], 'fields': F9, 'date_fmt': 'compact', 'cols': COL9},
    {'key': '000905', 'name': 'index_000905',
     'path': os.path.join(RAW, 'index_000905_daily.csv'),
     'codes': ['000905.SH'], 'fields': F9, 'date_fmt': 'compact', 'cols': COL9},
    {'key': '000300', 'name': 'index_000300',
     'path': os.path.join(RAW, 'index_000300_daily.csv'),
     'codes': ['000300.SH'], 'fields': F11, 'date_fmt': 'dashed', 'cols': COL12},
]


def norm_date(v):
    s = str(v).strip()
    if not s or s.lower() in ('nan', 'none', 'nat'):
        return None
    if '.' in s:
        s = s.split('.')[0]
    s = s.replace('-', '').replace('/', '')
    return s if len(s) == 8 and s.isdigit() else None


def read_existing(spec):
    """返回 (keys_set((ts_code, yyyymmdd)), last_yyyymmdd)。文件缺失时 last=None。"""
    path = spec['path']
    if not os.path.exists(path):
        return set(), None
    df = pd.read_csv(path, dtype=str, encoding='utf-8-sig')
    nd = df['trade_date'].map(norm_date)
    keys = set(zip(df['ts_code'], nd))
    keys.discard((None, None))
    last = nd.dropna().max()
    return keys, last


def update_one(pro, spec, dry_run=False, log=print):
    keys, last = read_existing(spec)
    today = datetime.now().strftime('%Y%m%d')
    if last is None:
        msg = f"[{spec['name']}] ⚠️ 文件缺失或无可解析日期 → 跳过 (本脚本只做增量, 不建全量)"
        log('  ' + msg)
        return {'name': spec['name'], 'status': 'no_base', 'added': 0, 'last': None,
                'fetched': 0, 'dup_skipped': 0}
    start = (pd.Timestamp(last) + pd.Timedelta(days=1)).strftime('%Y%m%d')
    if start > today:
        log(f"[{spec['name']}] ✅ 已最新 (末端 {last})")
        return {'name': spec['name'], 'status': 'current', 'added': 0, 'last': last,
                'fetched': 0, 'dup_skipped': 0}
    if dry_run:
        log(f"[{spec['name']}] 🔍 缺口 {start} ~ {today} (存量末端 {last})")
        return {'name': spec['name'], 'status': 'dry_run', 'added': 0, 'last': last,
                'fetched': 0, 'dup_skipped': 0}

    fetched, used = None, None
    for code in spec['codes']:
        try:
            fetched = pro.index_daily(ts_code=code, start_date=start, end_date=today,
                                      fields=spec['fields'])
        except Exception as e:  # noqa: BLE001
            log(f"  [{spec['name']}] ⚠️ {code} 调用异常: {str(e)[:90]}")
            fetched = None
        time.sleep(0.4)
        if fetched is not None and len(fetched) > 0:
            used = code
            break
    if fetched is None or len(fetched) == 0:
        log(f"[{spec['name']}] ⚪ 无新数据 ({start} ~ {today}; tried {spec['codes']})")
        return {'name': spec['name'], 'status': 'no_data', 'added': 0, 'last': last,
                'fetched': 0, 'dup_skipped': 0}

    n_fetched = len(fetched)
    new = fetched.copy()
    new['_nd'] = new['trade_date'].map(norm_date)
    bad = int(new['_nd'].isna().sum())
    if bad:
        log(f"  [{spec['name']}] ⚠️ {bad} 行日期不可解析, 已剔除")
        new = new[new['_nd'].notna()]
    # 同批去重
    new = new.drop_duplicates(subset=['ts_code', '_nd'], keep='last')
    # 与存量去重 (幂等) — 注意: 列名 '_nd' 下划线开头会被 itertuples 重命名, 用 zip 迭代
    dup_mask = [(c, d) in keys for c, d in zip(new['ts_code'], new['_nd'])]
    n_dup = int(sum(dup_mask))
    new = new[[not d for d in dup_mask]]
    if len(new) == 0:
        log(f"[{spec['name']}] ✅ 无新增 (API {n_fetched} 行均为已有; dup={n_dup})")
        return {'name': spec['name'], 'status': 'no_new', 'added': 0, 'last': last,
                'fetched': n_fetched, 'dup_skipped': n_dup}

    # 目标格式
    if spec['date_fmt'] == 'dashed':
        new['trade_date'] = new['_nd'].map(lambda s: f'{s[:4]}-{s[4:6]}-{s[6:]}')
        new['date'] = new['_nd']
    else:
        new['trade_date'] = new['_nd']
    out = new[spec['cols']].sort_values('trade_date')
    out.to_csv(spec['path'], mode='a', header=False, index=False, encoding='utf-8-sig')

    new_last = new['_nd'].max()
    log(f"[{spec['name']}] ✅ +{len(out)} 行 (source={used}), 末端 {last} → {new_last}"
        + (f" | 剔除重复 {n_dup}" if n_dup else ''))
    return {'name': spec['name'], 'status': 'ok', 'added': len(out), 'last': new_last,
            'fetched': n_fetched, 'dup_skipped': n_dup, 'source': used}


def update_indices(pro, dry_run=False, only=None, verbose=True):
    """入口: 更新三指数文件; 返回逐文件摘要 list。"""
    log = print if verbose else (lambda *a, **k: None)
    res = []
    for spec in SPECS:
        if only and spec['key'] != only:
            continue
        res.append(update_one(pro, spec, dry_run=dry_run, log=log))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--only', type=str, default=None, help='只更新一个 (000985/000905/000300)')
    args = ap.parse_args()

    import sys
    sys.path.insert(0, 'e:/quant')
    from credentials import get_tushare_api

    print('=' * 70)
    print('指数日线增量更新 (000985/000905/000300) — M16-①')
    print('=' * 70)
    pro = get_tushare_api()
    res = update_indices(pro, dry_run=args.dry_run, only=args.only)
    print('-' * 70)
    for r in res:
        print(f"  {r['name']:16s} status={r['status']:8s} +{r['added']:>3} 行  末端={r['last']}")


if __name__ == '__main__':
    from pathlib import Path as _FuxiPath
    import sys as _FuxiSys
    _FuxiSys.path.insert(0, str(_FuxiPath(__file__).resolve().parents[1]))
    from fuxi_v1.legacy_writers import legacy_cli_guard as _FuxiGuard
    _fuxi_guard = _FuxiGuard(__file__)
    main()
