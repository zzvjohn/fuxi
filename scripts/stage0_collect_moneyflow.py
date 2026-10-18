"""
Stage 0: 每日数据采集 — moneyflow 订单流数据
从 Tushare 采集每日资金流向（小单/大单/超大单买卖量），
支持增量更新。
"""
import sys
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
import time
import os
from pathlib import Path
from datetime import datetime, timedelta
from credentials import get_tushare_api

DATA_DIR = Path('data')
RAWFILE = DATA_DIR / 'raw' / 'moneyflow_daily.csv'

def get_existing_stocks():
    """获取 daily_prices 中的股票列表"""
    dp = pd.read_csv(DATA_DIR / 'raw' / 'daily_prices.csv',
                     usecols=['ts_code'], dtype={'ts_code': str})
    return sorted(dp['ts_code'].unique().tolist())

def get_existing_moneyflow():
    """加载已有的 moneyflow 数据"""
    if RAWFILE.exists():
        mf = pd.read_csv(RAWFILE, dtype={'ts_code': str, 'trade_date': str})
        return mf
    return pd.DataFrame()

def collect_moneyflow(pro, stocks, start_date='20210101', end_date=None, batch_size=50):
    """
    批量采集资金流向数据。
    Tushare moneyflow 接口每次最多返回4000条/200token。
    按股票分批采集，每50只一批。
    """
    if end_date is None:
        end_date = datetime.now().strftime('%Y%m%d')
    
    all_data = []
    total = len(stocks)
    failed = []
    
    for i in range(0, total, batch_size):
        batch = stocks[i:i+batch_size]
        ts_codes = ','.join(batch)
        
        try:
            df = pro.moneyflow(
                ts_code=ts_codes,
                start_date=start_date,
                end_date=end_date,
            )
            if df is not None and len(df) > 0:
                all_data.append(df)
            print(f'  [{i+batch_size}/{total}] OK: {len(df) if df is not None else 0} rows')
        except Exception as e:
            print(f'  [{i+batch_size}/{total}] FAIL: {e}')
            failed.extend(batch)
        
        # Rate limiting
        if (i + batch_size) % 200 == 0:
            time.sleep(1)  # Throttle after every 4 batches
        else:
            time.sleep(0.1)
    
    if all_data:
        result = pd.concat(all_data, ignore_index=True)
        result.drop_duplicates(subset=['ts_code', 'trade_date'], inplace=True)
        return result, failed
    return pd.DataFrame(), failed

def merge_and_save(new_data, existing):
    """合并新数据和已有数据，去重保存"""
    if existing.empty:
        combined = new_data
    else:
        combined = pd.concat([existing, new_data], ignore_index=True)
        combined.drop_duplicates(subset=['ts_code', 'trade_date'], inplace=True)
    
    combined.sort_values(['ts_code', 'trade_date'], inplace=True)
    RAWFILE.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(RAWFILE, index=False)
    return combined

def main():
    print('=' * 60)
    print('  Stage 0: moneyflow 资金流向数据采集')
    print('=' * 60)
    
    pro = get_tushare_api()
    
    # 获取股票列表 (从 daily_prices)
    stocks = get_existing_stocks()
    print(f'\n股票池: {len(stocks)} 只')
    
    # 加载已有数据确定增量范围
    existing = get_existing_moneyflow()
    if not existing.empty:
        last_date = existing['trade_date'].max()
        print(f'已有数据: {len(existing):,} 行, 最新日期: {last_date}')
        start_date = (datetime.strptime(last_date, '%Y%m%d') - timedelta(days=3)).strftime('%Y%m%d')
        print(f'增量采集: {start_date} ~ today')
    else:
        print('首次采集 — 全量拉取')
        start_date = '20210101'
    
    # 采集
    print(f'\n开始采集 (batch_size=50)...')
    new_data, failed = collect_moneyflow(pro, stocks, start_date=start_date, batch_size=50)
    
    if new_data.empty:
        print('\n无新数据')
        return
    
    print(f'\n新数据: {len(new_data):,} 行')
    print(f'失败股票: {len(failed)} 只')
    
    # 去重合并保存
    combined = merge_and_save(new_data, existing)
    print(f'合并后总计: {len(combined):,} 行')
    print(f'日期范围: {combined["trade_date"].min()} ~ {combined["trade_date"].max()}')
    print(f'股票数: {combined["ts_code"].nunique()}')
    
    # 统计最近日期
    latest = combined['trade_date'].max()
    latest_count = len(combined[combined['trade_date'] == latest])
    print(f'最新日期 {latest}: {latest_count} 只股票')
    
    print(f'\n保存至: {RAWFILE}')

if __name__ == '__main__':
    from pathlib import Path as _FuxiPath
    import sys as _FuxiSys
    _FuxiSys.path.insert(0, str(_FuxiPath(__file__).resolve().parents[1]))
    from fuxi_v1.legacy_writers import legacy_cli_guard as _FuxiGuard
    _fuxi_guard = _FuxiGuard(__file__)
    main()
