# -*- coding: utf-8 -*-
"""P-20260826-003: ETF 申赎分歧度数据源接入 — 日度 ETF 份额采集

Tushare 5000 积分接口 (2026-08-26 实测全通):
  - fund_basic(market='E')  : ETF 静态列表 (约 2900 只)
  - fund_share(trade_date)  : 全市场 ETF 当日份额 (单次调用全市场)
  - fund_daily(trade_date)  : 全市场 ETF 日行情

产物:
  - data/raw/fund_basic_etf.csv  静态列表 (刷新式)
  - data/raw/fund_share.csv      日度份额 (增量 append, smart_append_csv)
  - data/raw/fund_daily.csv      ETF 日行情 (增量 append)

用途 (P-20260826-003): 行业/宽基 ETF 份额 5 日净申赎变化率 = 机构-个人参与者分歧度代理。
因子公式桥接安排在 Stage 3 (阶段一先落地数据管道)。

用法: python scripts/stage0_fund_share_collect.py [--days 30]
"""
import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'scripts'))
try:
    from smart_append_csv import smart_append_csv
except Exception:
    def smart_append_csv(df, path, keys=None):  # 兜底
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        if Path(path).exists():
            old = pd.read_csv(path, low_memory=False)
            df = pd.concat([old, df]).drop_duplicates()
        df.to_csv(path, index=False, encoding='utf-8-sig')
        return len(df)

DATA_RAW = PROJECT_ROOT / 'data' / 'raw'
FUND_BASIC_PATH = DATA_RAW / 'fund_basic_etf.csv'
FUND_SHARE_PATH = DATA_RAW / 'fund_share.csv'
FUND_DAILY_PATH = DATA_RAW / 'fund_daily.csv'


def get_pro():
    import tushare as ts
    return ts.pro_api()


def collect_fund_basic(pro):
    """ETF 静态列表 (刷新式覆盖)"""
    df = pro.fund_basic(market='E')
    if df is None or df.empty:
        raise RuntimeError('fund_basic(market=E) 返回空')
    keep = ['ts_code', 'name', 'fund_type', 'benchmark', 'invest_type',
            'list_date', 'issue_date', 'm_fee', 'status']
    df = df[[c for c in keep if c in df.columns]].copy()
    df.to_csv(FUND_BASIC_PATH, index=False, encoding='utf-8-sig')
    return df


def collect_by_day(pro, api: str, end_date: str, days: int, tscode_first: str):
    """按自然日逐日拉取全市场快照 (fund_share/fund_daily 均支持 trade_date 全市场)"""
    rows = []
    d = datetime.strptime(end_date, '%Y%m%d')
    fails = 0
    for i in range(days):
        day = (d - timedelta(days=i)).strftime('%Y%m%d')
        try:
            if api == 'fund_share':
                chunk = pro.fund_share(trade_date=day)
            else:
                chunk = pro.fund_daily(trade_date=day)
            if chunk is not None and len(chunk):
                rows.append(chunk)
        except Exception as e:
            fails += 1
            if fails <= 3:
                print(f'  [warn] {api} {day} 失败: {str(e)[:80]}')
    if not rows:
        raise RuntimeError(f'{api} 无任何成功日期')
    df = pd.concat(rows, ignore_index=True)
    df = df.drop_duplicates(subset=[c for c in ['ts_code', 'trade_date'] if c in df.columns])
    return df


def main():
    ap = argparse.ArgumentParser(description='ETF 份额/行情日度采集 (P-20260826-003)')
    ap.add_argument('--days', type=int, default=30, help='回填自然日数')
    ap.add_argument('--end-date', type=str, default=None, help='截止日 YYYYMMDD, 默认今天')
    args = ap.parse_args()

    pro = get_pro()
    end_date = args.end_date or datetime.now().strftime('%Y%m%d')

    print(f'[P-003] ETF 数据采集 (回填 {args.days} 自然日, 截止 {end_date})')

    # 1. 静态列表
    try:
        basic = collect_fund_basic(pro)
        print(f'[fund_basic] ETF 列表 {len(basic)} 只 → {FUND_BASIC_PATH.name}')
    except Exception as e:
        print(f'[fund_basic] 失败 (不阻塞): {str(e)[:100]}')

    # 2. 份额
    share = collect_by_day(pro, 'fund_share', end_date, args.days, 'ts_code')
    n_share = smart_append_csv(share, FUND_SHARE_PATH, keys=['ts_code', 'trade_date'])
    latest = share['trade_date'].max()
    print(f'[fund_share] 新增 {len(share)} 行 (累计 {n_share}), 最新 {latest} → {FUND_SHARE_PATH.name}')

    # 3. 行情
    daily = collect_by_day(pro, 'fund_daily', end_date, args.days, 'ts_code')
    n_daily = smart_append_csv(daily, FUND_DAILY_PATH, keys=['ts_code', 'trade_date'])
    latest_d = daily['trade_date'].max()
    print(f'[fund_daily] 新增 {len(daily)} 行 (累计 {n_daily}), 最新 {latest_d} → {FUND_DAILY_PATH.name}')


if __name__ == '__main__':
    from pathlib import Path as _FuxiPath
    import sys as _FuxiSys
    _FuxiSys.path.insert(0, str(_FuxiPath(__file__).resolve().parents[1]))
    from fuxi_v1.legacy_writers import legacy_cli_guard as _FuxiGuard
    _fuxi_guard = _FuxiGuard(__file__)
    main()
