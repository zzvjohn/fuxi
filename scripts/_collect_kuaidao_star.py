# -*- coding: utf-8 -*-
"""科创板补齐 (P-20260911-kd-fixup, 2026-09-11)

问题: data/raw/kuaidao_gaps/ 的 income / cashflow / balancesheet 三表只采了
      5150 只 (沿用 CF 策略股票池口径, 排除 688/689/8/4 开头), 而 daily_prices
      全市场面板有 5773 只 → **618 只科创板 (688) 在三表中完全缺失**,
      导致全部 kd_* 基本面因子在这 11% 的股票上是 NaN, 被迫退出横截面。
      (事件/状态表 holdernumber/holdertrade/dividend/repurchase/top10 是月区间
       全市场批量采集, 已含科创板, 无需补。)

做法: 对「daily_prices 中有、但三表里没有」的代码, 逐股补采全历史并 append,
      再按 (ts_code, end_date) 去重 (report_type=1 合并报表优先)。

断点续传: _star_state.json
用法: python scripts/_collect_kuaidao_star.py [income|cashflow|balancesheet|all]
"""
import sys, os, io, time, json, datetime
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, 'e:/quant')
from credentials import get_tushare_api

OUT = 'e:/quant/data/raw/kuaidao_gaps'
STATE = os.path.join(OUT, '_star_state.json')
os.makedirs(OUT, exist_ok=True)
pro = get_tushare_api()

START, END = '20180101', '20260930'

TABLES = [
    ('income.csv', 'income',
     'ts_code,ann_date,f_ann_date,end_date,report_type,revenue,operate_profit,'
     'total_profit,income_tax,n_income,n_income_attr_p'),
    ('cashflow.csv', 'cashflow',
     'ts_code,ann_date,f_ann_date,end_date,report_type,'
     'n_cashflow_act,c_pay_acq_const_fiolta,c_fr_sale_sg'),
    ('balancesheet.csv', 'balancesheet',
     'ts_code,ann_date,f_ann_date,end_date,report_type,total_assets,money_cap,'
     'trad_asset,notes_receiv,accounts_receiv,goodwill,st_borr,lt_borr,bond_payable,'
     'adv_receipts,contract_liab'),
]


def load_state():
    return json.load(open(STATE, encoding='utf-8')) if os.path.exists(STATE) else {}


def save_state(st):
    json.dump(st, open(STATE, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)


def fetch_retry(fn, desc, tries=4):
    for i in range(tries):
        try:
            df = fn()
            return None if df is None or len(df) == 0 else df
        except Exception as e:
            msg = str(e)[:120]
            if any(k in msg for k in ('每分钟', '抱歉', '2000次', '访问频率', '频率')):
                w = 15 + 10 * i
                print('  [限速] %s 第%d次 -> sleep %ds' % (desc, i + 1, w)); time.sleep(w)
            else:
                print('  [重试] %s 第%d次: %s' % (desc, i + 1, msg)); time.sleep(2.0)
    return None


def _append_csv(path, frames):
    if not frames:
        return
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(path, mode='a', header=not os.path.exists(path), index=False)


def _dedupe(path, keys):
    d = pd.read_csv(path, dtype=str, keep_default_na=False)
    before = len(d)
    if 'report_type' in d.columns:
        d['_r'] = np.where(d['report_type'] == '1', 0, 1)
        d = d.sort_values(keys + ['_r']).drop_duplicates(subset=keys, keep='first').drop(columns=['_r'])
    else:
        d = d.drop_duplicates(subset=keys, keep='first')
    d.to_csv(path, index=False)
    print('  去重 %s: %d -> %d 行' % (os.path.basename(path), before, len(d)))


def targets(fname):
    """daily_prices 里有、但该表里没有的代码。"""
    all_codes = set(pd.read_csv('e:/quant/data/raw/daily_prices.csv', usecols=['ts_code'],
                                dtype=str)['ts_code'].dropna().unique())
    try:
        have = set(pd.read_csv(os.path.join(OUT, fname), usecols=['ts_code'],
                               dtype=str)['ts_code'].dropna().unique())
    except Exception:
        have = set()
    miss = sorted(all_codes - have)
    # 只补真实股票前缀 (排除指数/基金类 000xxx.CSI 等)
    miss = [c for c in miss if c.split('.')[-1] in ('SH', 'SZ', 'BJ')]
    return miss


def run(st, only=None):
    for fname, api, fields in TABLES:
        if only and only != api:
            continue
        path = os.path.join(OUT, fname)
        codes = targets(fname)
        key = 'star_done_' + api
        done = set(st.get(key, []))
        todo = [c for c in codes if c not in done]
        print('=' * 62)
        print('[科创板补齐] %-16s 缺 %d 只, 待采 %d 只' % (fname, len(codes), len(todo)))
        print('=' * 62)
        if not todo:
            continue
        buf, n, t0 = [], 0, time.time()
        for i, c in enumerate(todo):
            df = fetch_retry(lambda c=c: getattr(pro, api)(ts_code=c, start_date=START,
                                                          end_date=END, fields=fields),
                             '%s %s' % (api, c))
            if df is not None:
                buf.append(df); n += len(df)
            done.add(c)
            if len(buf) >= 200:
                _append_csv(path, buf); buf = []
                st[key] = sorted(done); save_state(st)
            if (i + 1) % 100 == 0 or i == len(todo) - 1:
                el = time.time() - t0
                print('  %s %d/%d rows=%d (%.0fs)' % (api, i + 1, len(todo), n, el))
            time.sleep(0.12)
        _append_csv(path, buf)
        st[key] = sorted(done); save_state(st)
        _dedupe(path, ['ts_code', 'end_date'])
        print('  %s 完成: 新增 %d 行' % (fname, n))


if __name__ == '__main__':
    st = load_state()
    only = sys.argv[1] if len(sys.argv) > 1 else None
    if only == 'all':
        only = None
    t0 = time.time()
    run(st, only)
    print('\n全部完成, 用时 %.0fs' % (time.time() - t0))
