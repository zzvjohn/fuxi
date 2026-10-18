"""
申万行业日线 (sw_daily) 增量更新 (M16-③)
========================================
背景 (2026-09-26 实测):
  · data/raw/sw_daily.csv 冻结在 2026-04-24, 此前无稳定日更入口
    (v3 日更链不含 sw_daily; 旧补采为月度一次性脚本)。
  · 行业基准用于行业中性化/拥挤度诊断; v0.7.83 已证"申万历史行业缺口过大,
    不作行业收益定量裁决"——本脚本把**增量通道**恢复, 不重构历史缺口。

格式契约 (与存量逐列一致, 只追加):
  ts_code,trade_date(YYYYMMDD),close,vol,amount

规则:
  · 缺口 = 交易日历中 > 存量末端 的已发生交易日 (Tushare trade_cal, SSE);
  · 逐日拉取 (单日实测 ~439 行), 幂等: 已存在的 (ts_code,date) 不重复追加;
  · --max-days N 限制单次补采上限 (默认不限, 稳态下缺口 ≤ 数个交易日);
  · --dry-run 只报告缺口。

用法:
  C:/Python314/python.exe scripts/collect_sw_daily.py [--dry-run] [--max-days 400]
"""
from __future__ import annotations

import argparse
import os
import time
from datetime import datetime

import pandas as pd

RAW = 'e:/quant/data/raw'
PATH = os.path.join(RAW, 'sw_daily.csv')
FIELDS = 'ts_code,trade_date,close,vol,amount'
COLS = ['ts_code', 'trade_date', 'close', 'vol', 'amount']


def norm_date(v):
    s = str(v).strip()
    if not s or s.lower() in ('nan', 'none', 'nat'):
        return None
    if '.' in s:
        s = s.split('.')[0]
    s = s.replace('-', '').replace('/', '')
    return s if len(s) == 8 and s.isdigit() else None


def update_sw_daily(pro, dry_run=False, max_days=None, verbose=True):
    """入口: 增量更新 sw_daily.csv; 返回摘要 dict。"""
    log = print if verbose else (lambda *a, **k: None)
    today = datetime.now().strftime('%Y%m%d')

    if not os.path.exists(PATH):
        log('[sw_daily] ⚠️ 文件缺失 → 跳过 (本脚本只做增量)')
        return {'name': 'sw_daily', 'status': 'no_base', 'added': 0, 'last': None}

    df = pd.read_csv(PATH, usecols=['trade_date'], dtype=str, encoding='utf-8-sig')
    nd = df['trade_date'].map(norm_date)
    last = nd.dropna().max()
    if last is None:
        log('[sw_daily] ⚠️ 无可解析日期 → 跳过')
        return {'name': 'sw_daily', 'status': 'no_base', 'added': 0, 'last': None}

    start = (pd.Timestamp(last) + pd.Timedelta(days=1)).strftime('%Y%m%d')
    if start > today:
        log(f'[sw_daily] ✅ 已最新 (末端 {last})')
        return {'name': 'sw_daily', 'status': 'current', 'added': 0, 'last': last}

    # 交易日历 (缺口日)
    cal = None
    try:
        cal = pro.trade_cal(exchange='SSE', start_date=start, end_date=today, is_open='1')
        time.sleep(0.4)
    except Exception as e:  # noqa: BLE001
        log(f'[sw_daily] ❌ 交易日历获取失败: {str(e)[:90]}')
        return {'name': 'sw_daily', 'status': 'cal_error', 'added': 0, 'last': last}
    miss = sorted(cal['cal_date'].astype(str).unique()) if cal is not None and len(cal) else []
    if not miss:
        log(f'[sw_daily] ⚪ 无缺口交易日 ({start} ~ {today})')
        return {'name': 'sw_daily', 'status': 'no_data', 'added': 0, 'last': last}
    if max_days:
        miss = miss[:max_days]
    if dry_run:
        log(f'[sw_daily] 🔍 缺口 {len(miss)} 个交易日: {miss[0]} ~ {miss[-1]} (存量末端 {last})')
        return {'name': 'sw_daily', 'status': 'dry_run', 'added': 0, 'last': last,
                'missing_days': len(miss)}

    # 已有键集合 (幂等)
    ex = pd.read_csv(PATH, usecols=['ts_code', 'trade_date'], dtype=str, encoding='utf-8-sig')
    ex_keys = set(zip(ex['ts_code'], ex['trade_date'].map(norm_date)))

    parts = []
    fails = []
    for d in miss:
        try:
            got = pro.sw_daily(trade_date=d, fields=FIELDS)
        except Exception as e:  # noqa: BLE001
            fails.append(d)
            log(f'    {d}: 调用异常 {str(e)[:60]}')
            time.sleep(0.5)
            continue
        time.sleep(0.35)
        if got is None or len(got) == 0:
            fails.append(d)
            continue
        parts.append(got)
    if not parts:
        log(f'[sw_daily] ⚪ 无新数据 ({len(miss)} 个缺口日全部空/失败)')
        return {'name': 'sw_daily', 'status': 'no_data', 'added': 0, 'last': last,
                'missing_days': len(miss), 'fails': len(fails)}

    new = pd.concat(parts, ignore_index=True)
    new['trade_date'] = new['trade_date'].astype(str)
    new = new.drop_duplicates(subset=['ts_code', 'trade_date'], keep='last')
    dup_mask = [ (r.ts_code, r.trade_date) in ex_keys for r in new.itertuples() ]
    n_dup = int(sum(dup_mask))
    new = new[[not x for x in dup_mask]]
    if len(new) == 0:
        log(f'[sw_daily] ✅ 无新增 (dup={n_dup})')
        return {'name': 'sw_daily', 'status': 'no_new', 'added': 0, 'last': last}
    out = new[COLS].sort_values(['trade_date', 'ts_code'])
    out.to_csv(PATH, mode='a', header=False, index=False, encoding='utf-8-sig')

    new_last = out['trade_date'].max()
    per_day = out.groupby('trade_date').size()
    log(f'[sw_daily] ✅ +{len(out):,} 行 / {out["trade_date"].nunique()} 日 '
        f'(每日中位 {int(per_day.median())} 行), 末端 {last} → {new_last}'
        + (f' | 失败日 {len(fails)}' if fails else '')
        + (f' | 剔除重复 {n_dup}' if n_dup else ''))
    return {'name': 'sw_daily', 'status': 'ok', 'added': len(out), 'last': new_last,
            'days': int(out['trade_date'].nunique()), 'dup_skipped': n_dup,
            'fails': len(fails)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--max-days', type=int, default=None)
    args = ap.parse_args()

    import sys
    sys.path.insert(0, 'e:/quant')
    from credentials import get_tushare_api

    print('=' * 70)
    print('申万行业日线 (sw_daily) 增量更新 — M16-③')
    print('=' * 70)
    pro = get_tushare_api()
    r = update_sw_daily(pro, dry_run=args.dry_run, max_days=args.max_days)
    print('-' * 70)
    print(f"  {r['name']} status={r['status']} +{r['added']} 行  末端={r['last']}")


if __name__ == '__main__':
    from pathlib import Path as _FuxiPath
    import sys as _FuxiSys
    _FuxiSys.path.insert(0, str(_FuxiPath(__file__).resolve().parents[1]))
    from fuxi_v1.legacy_writers import legacy_cli_guard as _FuxiGuard
    _fuxi_guard = _FuxiGuard(__file__)
    main()
