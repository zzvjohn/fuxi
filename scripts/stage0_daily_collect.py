"""
Stage 0: 每日数据采集 — 伏羲 v0.5
每天自动采集/更新以下数据:
  1. moneyflow_daily — 逐股资金流向
  2. forecast — 业绩预告
  3. report_rc — 分析师报告 (兜底补采)

调用方式: python scripts/stage0_daily_collect.py [--full]
  --full: 全量重新采集 (默认: 增量更新)
"""
import sys
sys.path.insert(0, '.')
import pandas as pd
import time
from pathlib import Path
from datetime import datetime, timedelta
from credentials import get_tushare_api

DATA_DIR = Path('data') / 'raw'

def collect_moneyflow_incremental(pro, stocks, existing_file, lookback_days=5):
    """增量采集最近的 moneyflow 数据"""
    existing = pd.DataFrame()
    if existing_file.exists():
        existing = pd.read_csv(existing_file, dtype={'ts_code': str, 'trade_date': str})
        last_date = existing['trade_date'].max()
        start_date = (datetime.strptime(str(last_date), '%Y%m%d') - timedelta(days=lookback_days)).strftime('%Y%m%d')
        print(f'  moneyflow: 增量 {start_date} ~ today (已有 {len(existing):,} rows)')
    else:
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y%m%d')
        print(f'  moneyflow: 首次采集 {start_date} ~ today')
    
    all_new = []
    total = len(stocks)
    batch_size = 50
    
    for i in range(0, total, batch_size):
        batch = stocks[i:i+batch_size]
        try:
            df = pro.moneyflow(ts_code=','.join(batch), start_date=start_date)
            if df is not None and len(df) > 0:
                all_new.append(df)
        except Exception as e:
            pass
        time.sleep(0.06)
    
    if not all_new:
        print('  moneyflow: 无新数据')
        return
    
    new_data = pd.concat(all_new, ignore_index=True)
    new_data.drop_duplicates(subset=['ts_code', 'trade_date'], inplace=True)
    
    if not existing.empty:
        combined = pd.concat([existing, new_data], ignore_index=True)
        combined.drop_duplicates(subset=['ts_code', 'trade_date'], inplace=True)
    else:
        combined = new_data
    
    combined.sort_values(['ts_code', 'trade_date'], inplace=True)
    combined.to_csv(existing_file, index=False)
    print(f'  moneyflow: +{len(new_data):,} new, total {len(combined):,} rows')


def collect_forecast_incremental(pro, existing_file, lookback_days=90):
    """采集分析师业绩预告 (增量)"""
    existing = pd.DataFrame()
    if existing_file.exists():
        existing = pd.read_csv(existing_file, dtype={'ts_code': str, 'ann_date': str})
        last_date = existing['ann_date'].max()
        start_date = (datetime.strptime(str(last_date), '%Y%m%d') - timedelta(days=lookback_days)).strftime('%Y%m%d')
        print(f'  forecast: 增量 {start_date} ~ today (已有 {len(existing):,} rows)')
    else:
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y%m%d')
        print(f'  forecast: 首次采集 {start_date} ~ today')
    
    try:
        # forecast API supports ann_date (single date), loop through range
        all_new = []
        d = datetime.strptime(start_date, '%Y%m%d')
        end_d = datetime.now()
        while d <= end_d:
            date_str = d.strftime('%Y%m%d')
            try:
                df = pro.forecast(ann_date=date_str)
                if df is not None and len(df) > 0:
                    all_new.append(df)
            except Exception:
                pass
            d += timedelta(days=1)
            time.sleep(0.05)
        
        if not all_new:
            print('  forecast: 无新数据')
            return
        new_data = pd.concat(all_new, ignore_index=True)
        new_data.drop_duplicates(subset=['ts_code', 'ann_date'], inplace=True)
        
        if not existing.empty:
            combined = pd.concat([existing, new_data], ignore_index=True)
            combined.drop_duplicates(subset=['ts_code', 'ann_date'], inplace=True)
        else:
            combined = new_data
        
        combined.sort_values(['ts_code', 'ann_date'], inplace=True)
        combined.to_csv(existing_file, index=False)
        print(f'  forecast: +{len(new_data):,} new, total {len(combined):,} rows')
    except Exception as e:
        print(f'  forecast: FAILED ({e})')


def collect_report_rc_incremental(pro, existing_file, lookback_days=30):
    """采集分析师投研报告 (增量)"""
    existing = pd.DataFrame()
    if existing_file.exists():
        existing = pd.read_csv(existing_file, dtype={'ts_code': str, 'ann_date': str})
        last_date = existing['ann_date'].max()
        start_date = (datetime.strptime(str(last_date), '%Y%m%d') - timedelta(days=lookback_days)).strftime('%Y%m%d')
        print(f'  report_rc: 增量 {start_date} ~ today (已有 {len(existing):,} rows)')
    else:
        start_date = (datetime.now() - timedelta(days=180)).strftime('%Y%m%d')
        print(f'  report_rc: 首次采集 {start_date} ~ today')
    
    try:
        new_data = pro.report_rc(start_date=start_date)
        if new_data is None or len(new_data) == 0:
            print('  report_rc: 无新数据')
            return
        
        if not existing.empty:
            combined = pd.concat([existing, new_data], ignore_index=True)
            combined.drop_duplicates(subset=['ts_code', 'ann_date', 'report_type'], inplace=True)
        else:
            combined = new_data
        
        combined.sort_values(['ts_code', 'ann_date'], inplace=True)
        combined.to_csv(existing_file, index=False)
        print(f'  report_rc: +{len(new_data):,} new, total {len(combined):,} rows')
    except Exception as e:
        print(f'  report_rc: FAILED ({e})')


def main(full=False, moneyflow_only=False):
    mode_label = "资金流增量" if moneyflow_only else ("全量" if full else "增量")
    print('=' * 60)
    print(f'  Stage 0: 每日数据采集 [{mode_label}] — {datetime.now().strftime("%Y-%m-%d %H:%M")}')
    print('=' * 60)
    
    pro = get_tushare_api()
    
    # 获取股票池
    dp_path = DATA_DIR / 'daily_prices.csv'
    if dp_path.exists():
        dp = pd.read_csv(dp_path, usecols=['ts_code'], dtype={'ts_code': str})
        stocks = sorted(dp['ts_code'].unique().tolist())
        print(f'\n股票池: {len(stocks)} 只 (from daily_prices)')
    else:
        print('⚠️ daily_prices.csv 不存在，跳过')
        return
    
    # 1. Moneyflow (每日更新)
    print('\n--- 1. moneyflow 资金流向 ---')
    mf_path = DATA_DIR / 'moneyflow_daily.csv'
    if full:
        mf_path.unlink(missing_ok=True)
    collect_moneyflow_incremental(pro, stocks, mf_path)
    
    if moneyflow_only:
        print('\n' + '=' * 60)
        print('  Stage 0 [资金流增量] 完成')
        print('=' * 60)
        return
    
    # 2. Forecast (每周更新)
    print('\n--- 2. forecast 业绩预告 ---')
    fc_path = DATA_DIR / 'forecast.csv'
    if not fc_path.exists() or full:
        collect_forecast_incremental(pro, fc_path)
    else:
        print('  forecast: 已存在，跳过 (增量间隔90天)')
    
    # 3. Analyst reports (每周更新; 8:00 全量任务已通过 quant-data-collector 采集)
    print('\n--- 3. report_rc 分析师报告 ---')
    rc_path = DATA_DIR / 'report_rc.csv'
    if not rc_path.exists() or full:
        collect_report_rc_incremental(pro, rc_path)
    else:
        print('  report_rc: 已存在，跳过 (增量间隔30天)')
    
    print('\n' + '=' * 60)
    print('  Stage 0 完成')
    print('=' * 60)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--full', action='store_true', help='全量重新采集')
    parser.add_argument('--moneyflow-only', action='store_true', help='仅采集资金流向 (17:00 增量)')
    args = parser.parse_args()
    main(full=args.full, moneyflow_only=args.moneyflow_only)
