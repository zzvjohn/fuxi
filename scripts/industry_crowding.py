# -*- coding: utf-8 -*-
"""P-20260902-003: 行业级拥挤度三因子监控 (纯监控, 零干预)

三因子 (申万一级行业, 等权合成):
  1. 融资拥挤: 行业融资余额 20 日净流入 / 行业 20 日成交额
  2. 换手率拥挤: 行业换手率(成交额加权) 60 日历史分位
  3. 成交占比拥挤: 行业成交额占全市场比重 20 日变化率

输出: data/industry_crowding.json (Top-5/Bottom-5 + 全行业得分)
用法: python scripts/industry_crowding.py [--days 120]
"""
import sys
import json
import argparse
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
OUT_FILE = DATA_DIR / 'industry_crowding.json'

MARGIN_NET_DAYS = 20      # 融资净流入窗口
TURNOVER_Z_WINDOW = 60    # 换手率历史分位窗口
SHARE_CHG_DAYS = 20       # 成交占比变化率窗口


def load_industry_map() -> pd.DataFrame:
    """个股 → 申万一级行业映射 (当前成分: out_date 为空)"""
    m = pd.read_csv(
        DATA_DIR / 'raw' / 'sw_index_member_all.csv',
        dtype={'ts_code': str},
        usecols=['l1_name', 'ts_code', 'out_date'],
    )
    m = m[m['out_date'].isna()][['ts_code', 'l1_name']]
    return m.drop_duplicates(subset=['ts_code'])


def load_daily(days: int) -> pd.DataFrame:
    df = pd.read_csv(
        DATA_DIR / 'raw' / 'daily_prices.csv',
        dtype={'ts_code': str, 'trade_date': str},
        usecols=['ts_code', 'trade_date', 'amount'],
    )
    df['trade_date'] = pd.to_datetime(df['trade_date'], format='mixed')
    end = df['trade_date'].max()
    df = df[df['trade_date'] >= end - timedelta(days=days)]
    df = df.drop_duplicates(subset=['ts_code', 'trade_date'])
    return df


def load_margin(days: int) -> pd.DataFrame:
    df = pd.read_csv(
        DATA_DIR / 'raw' / 'margin_detail.csv',
        dtype={'ts_code': str, 'trade_date': str},
        usecols=['ts_code', 'trade_date', 'rzye'],
    )
    df['trade_date'] = pd.to_datetime(df['trade_date'], format='mixed')
    end = df['trade_date'].max()
    df = df[df['trade_date'] >= end - timedelta(days=days)]
    return df.drop_duplicates(subset=['ts_code', 'trade_date'])


def load_turnover(days: int) -> pd.DataFrame:
    df = pd.read_csv(
        DATA_DIR / 'raw' / 'daily_basic.csv',
        dtype={'ts_code': str, 'trade_date': str},
        usecols=['ts_code', 'trade_date', 'turnover_rate'],
    )
    df['trade_date'] = pd.to_datetime(df['trade_date'], format='mixed')
    end = df['trade_date'].max()
    df = df[df['trade_date'] >= end - timedelta(days=days)]
    return df.drop_duplicates(subset=['ts_code', 'trade_date'])


def compute():
    ind_map = load_industry_map()
    print(f'[行业拥挤] 映射: {len(ind_map)} 只个股 → {ind_map["l1_name"].nunique()} 个申万一级行业')

    daily = load_daily(140)
    margin = load_margin(140)
    turn = load_turnover(140)
    last_date = daily['trade_date'].max().strftime('%Y-%m-%d')
    print(f'[行业拥挤] daily {len(daily)} 行 / margin {len(margin)} 行 / turnover {len(turn)} 行, 最新 {last_date}')

    # ── 因子1: 行业融资余额 20 日净流入 / 行业成交额 ──
    mg = margin.merge(ind_map, on='ts_code', how='inner')
    m_ind = mg.groupby(['l1_name', 'trade_date'])['rzye'].sum().unstack(0).sort_index()
    m_net = (m_ind - m_ind.shift(MARGIN_NET_DAYS)).iloc[-1]  # 最新日 20 日净流入

    dg = daily.merge(ind_map, on='ts_code', how='inner')
    d_ind = dg.groupby(['l1_name', 'trade_date'])['amount'].sum().unstack(0).sort_index()
    ind_amt_20d = d_ind.iloc[-MARGIN_NET_DAYS:].sum()  # 近 20 日行业成交额合计

    f1 = (m_net / ind_amt_20d.replace(0, np.nan)).reindex(d_ind.columns)

    # ── 因子2: 行业换手率 (成交额加权) 60 日历史分位 ──
    tg = turn.merge(ind_map, on='ts_code', how='inner')
    tg = tg.merge(daily[['ts_code', 'trade_date', 'amount']], on=['ts_code', 'trade_date'], how='left')
    tg['w_turn'] = tg['turnover_rate'] * tg['amount'].fillna(0)
    wsum = tg.groupby(['l1_name', 'trade_date'])['amount'].sum()
    t_wavg = tg.groupby(['l1_name', 'trade_date'])['w_turn'].sum() / wsum.replace(0, np.nan)
    t_wide = t_wavg.unstack(0).sort_index()
    t_latest = t_wide.iloc[-1]
    f2 = pd.Series({
        ind: (t_wide[ind].dropna().tail(TURNOVER_Z_WINDOW) <= t_latest[ind]).mean()
        for ind in t_wide.columns
    })

    # ── 因子3: 行业成交额占全市场比重 20 日变化率 ──
    mkt_sum = d_ind.sum(axis=1)
    share = d_ind.div(mkt_sum, axis=0)
    share_cur = share.iloc[-MARGIN_NET_DAYS:].mean()
    share_prev = share.iloc[-2 * MARGIN_NET_DAYS:-MARGIN_NET_DAYS].mean()
    f3 = (share_cur - share_prev) / share_prev.replace(0, np.nan)

    # ── 等权合成 (z-score) ──
    def z(s: pd.Series) -> pd.Series:
        s = pd.to_numeric(s, errors='coerce')
        mu, sd = s.mean(), s.std()
        return (s - mu) / sd if sd and sd > 1e-12 else s * 0

    score = (z(f1) + z(f2) + z(f3)).dropna().sort_values(ascending=False)
    top5 = [{'industry': k, 'score': round(v, 3), 'f1_finance': _f(v), 'f2_turnover': _f(f2.get(k)),
             'f3_share_chg': _f(f3.get(k))} for k, v in score.head(5).items()]
    bot5 = [{'industry': k, 'score': round(v, 3), 'f1_finance': _f(f1.get(k)), 'f2_turnover': _f(f2.get(k)),
             'f3_share_chg': _f(f3.get(k))} for k, v in score.tail(5).items()]

    out = {
        'date': last_date,
        'generated_at': datetime.now().isoformat(),
        'proposal': 'P-20260902-003',
        'top5_crowded': top5,
        'bottom5_least_crowded': bot5,
        'n_industries': int(len(score)),
        'note': '纯监控零干预 (影子起步, 不接入选股裁决)',
    }
    OUT_FILE.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')

    print(f'[行业拥挤] Top-5 拥挤行业:')
    for t in top5:
        print(f'   {t["industry"]:<8} {t["score"]:+.2f}  (融资{t["f1_finance"]:+.2f} 换手分位{t["f2_turnover"]:.2f} 占比变化{t["f3_share_chg"]:+.2f})')
    print(f'[行业拥挤] Bottom-5 最不拥挤:')
    for t in bot5:
        print(f'   {t["industry"]:<8} {t["score"]:+.2f}  (融资{t["f1_finance"]:+.2f} 换手分位{t["f2_turnover"]:.2f} 占比变化{t["f3_share_chg"]:+.2f})')
    print(f'  → {OUT_FILE}')


def _f(v):
    return round(float(v), 4) if v is not None and not np.isnan(v) else None


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='行业级拥挤度监控 (纯监控)')
    ap.add_argument('--days', type=int, default=140, help='回溯自然日数')
    args = ap.parse_args()
    compute()
