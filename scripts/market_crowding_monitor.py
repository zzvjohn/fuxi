# -*- coding: utf-8 -*-
"""P-20260815-003: 市场级拥挤度监控 (纯监控版 — 只输出读数, 不干预管线)

两个指标 (每日运行, 读本地日线数据):
  1. 前5%成交额集中度: 每日成交额前5%个股之和 / 全市场总额, 取 20 日均值
     阈值参考: 山证策略 2026-08: 前5%占比 54.9% (拥挤高位); >50% 提醒
  2. 多空波动率比: 5日动量排名 Top-8 (多头) / Bottom-8 (空头) 等权日收益,
     20日滚动波动率之比 (华泰口径, 阈值 1.5; 2026-07 踩踏前达 2.8)

输出: data/market_crowding.json + stdout 读数 (仅提醒, 无状态写入管线)
用法: python scripts/market_crowding_monitor.py [--days 90]
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
OUT_FILE = DATA_DIR / 'market_crowding.json'
DAILY_PRICES = DATA_DIR / 'raw' / 'daily_prices.csv'

TOP_K = 8          # 多空组合个股数
MOM_DAYS = 5       # 动量窗口
VOL_WINDOW = 20    # 波动率滚动窗口
TOP_PCT = 0.05     # 成交额集中度分位
TH_CONCENTRATION = 0.50   # 集中度提醒阈值
TH_LS_VOL_RATIO = 1.5     # 多空波动率比提醒阈值


def load_daily(days: int) -> pd.DataFrame:
    df = pd.read_csv(
        DAILY_PRICES,
        dtype={'ts_code': str, 'trade_date': str},
        usecols=['ts_code', 'trade_date', 'close', 'amount'],
    )
    df['trade_date'] = pd.to_datetime(df['trade_date'], format='mixed')
    end = df['trade_date'].max()
    start = end - timedelta(days=days)
    df = df[(df['trade_date'] >= start) & (df['trade_date'] <= end)]
    df = df.drop_duplicates(subset=['ts_code', 'trade_date'])
    return df.sort_values(['ts_code', 'trade_date'])


def compute_concentration(daily: pd.DataFrame) -> float:
    """前5%成交额集中度 (20日均值)"""
    agg = daily.groupby('trade_date')['amount'].agg(['sum', 'count'])
    top_sum = daily.groupby('trade_date').apply(
        lambda g: g.nlargest(max(1, int(len(g) * TOP_PCT)), 'amount')['amount'].sum(),
        include_groups=False,
    )
    ratio = top_sum / agg['sum']
    ratio = ratio.replace([np.inf, -np.inf], np.nan).dropna()
    if len(ratio) < VOL_WINDOW:
        return float(ratio.mean()) if len(ratio) else np.nan
    return float(ratio.rolling(VOL_WINDOW).mean().iloc[-1])


def compute_ls_vol_ratio(daily: pd.DataFrame) -> float:
    """Top-8/Bottom-8 动量多空组合 20日滚动波动率比 (华泰口径代理)"""
    pivot = daily.pivot(index='trade_date', columns='ts_code', values='close')
    # 至少要有 60 个有效交易日的股票才参与
    valid = pivot.columns[pivot.count() >= 60]
    pivot = pivot[valid]
    ret = pivot.pct_change(fill_method=None)
    mom = pivot / pivot.shift(MOM_DAYS) - 1

    long_ret, short_ret = [], []
    for i in range(MOM_DAYS, len(mom)):
        m = mom.iloc[i].dropna()
        if len(m) < 40:
            continue
        top = m.nlargest(TOP_K).index
        bot = m.nsmallest(TOP_K).index
        r_next = ret.iloc[i + 1] if i + 1 < len(ret) else None
        if r_next is None:
            break
        long_ret.append(r_next[top].mean())
        short_ret.append(r_next[bot].mean())

    if len(long_ret) < VOL_WINDOW:
        return np.nan
    lr = pd.Series(long_ret).rolling(VOL_WINDOW).std().iloc[-1]
    sr = pd.Series(short_ret).rolling(VOL_WINDOW).std().iloc[-1]
    if not sr or np.isnan(sr) or sr == 0:
        return np.nan
    return float(lr / sr)


def main():
    ap = argparse.ArgumentParser(description='市场级拥挤度监控 (纯监控)')
    ap.add_argument('--days', type=int, default=90, help='回溯自然日数')
    args = ap.parse_args()

    if not DAILY_PRICES.exists():
        print('[市场拥挤] daily_prices.csv 不存在, 跳过')
        sys.exit(0)

    daily = load_daily(args.days)
    last_date = daily['trade_date'].max().strftime('%Y-%m-%d')
    print(f'[市场拥挤监控] 数据 {len(daily)} 行, 最新交易日 {last_date}')

    concentration = compute_concentration(daily)
    ls_vol_ratio = compute_ls_vol_ratio(daily)

    alerts = []
    if not np.isnan(concentration):
        print(f'  前5%成交额集中度 (20日均): {concentration:.1%}'
              + (f"  ⚠️ 超阈值 {TH_CONCENTRATION:.0%}" if concentration > TH_CONCENTRATION else ''))
        if concentration > TH_CONCENTRATION:
            alerts.append('concentration_high')
    else:
        print('  前5%成交额集中度: 数据不足')
    if not np.isnan(ls_vol_ratio):
        print(f'  多空波动率比 (Top-8/Bottom-8, 20日滚动): {ls_vol_ratio:.2f}'
              + (f"  ⚠️ 超阈值 {TH_LS_VOL_RATIO}" if ls_vol_ratio > TH_LS_VOL_RATIO else ''))
        if ls_vol_ratio > TH_LS_VOL_RATIO:
            alerts.append('ls_vol_ratio_high')
    else:
        print('  多空波动率比: 数据不足')

    out = {
        'date': last_date,
        'generated_at': datetime.now().isoformat(),
        'top5_concentration_20d': None if np.isnan(concentration) else round(float(concentration), 4),
        'ls_vol_ratio_20d': None if np.isnan(ls_vol_ratio) else round(float(ls_vol_ratio), 4),
        'thresholds': {'concentration': TH_CONCENTRATION, 'ls_vol_ratio': TH_LS_VOL_RATIO},
        'alerts': alerts,
        'note': '纯监控指标, 不干预管线 (P-20260815-003 用户指定: 只提醒不干预)',
    }
    OUT_FILE.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'  → {OUT_FILE}')
    if alerts:
        print(f'  🔥 提醒: {" / ".join(alerts)} — 仅提醒, 不触发任何管线干预')


if __name__ == '__main__':
    main()
