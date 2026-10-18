"""
数据更新脚本: 补充2026年5-6月数据
================================
从 Tushare 拉取 daily, adj_factor, daily_basic 的增量数据
并追加到本地 CSV 文件。

使用方法:
  python update_data_may_june.py

依赖: credentials.py (Tushare token), tushare
"""
import sys, os
from pathlib import Path
import pandas as pd
from datetime import datetime

# 添加工作区路径
sys.path.insert(0, str(Path(__file__).parent.parent))
from credentials import get_tushare_api

RAW_DIR = Path('e:/quant/data/raw')
START_DATE = '20260501'
END_DATE = '20260625'

def get_new_data(file_name, api_func, trade_date_col='trade_date'):
    """
    读取本地已有数据的最后日期, 拉取增量并追加
    """
    fpath = RAW_DIR / file_name
    if not fpath.exists():
        print(f'  [SKIP] {file_name} 不存在')
        return
    
    # 读取本地最后日期
    try:
        df_existing = pd.read_csv(fpath, encoding='utf-8-sig', low_memory=False)
        if trade_date_col in df_existing.columns:
            df_existing[trade_date_col] = pd.to_datetime(
                df_existing[trade_date_col], errors='coerce'
            )
            last_date = df_existing[trade_date_col].max()
            print(f'  {file_name}: 本地最后日期 = {last_date.strftime("%Y%m%d")}')
        else:
            last_date = pd.Timestamp('2021-01-01')
            print(f'  {file_name}: 无日期列, 重新全量')
    except Exception as e:
        print(f'  {file_name}: 读取失败 ({e}), 重新全量')
        last_date = pd.Timestamp('2021-01-01')
    
    if last_date >= pd.Timestamp(END_DATE):
        print(f'  {file_name}: 数据已是最新, 跳过')
        return
    
    # 调用 get_tushare_api 获取数据
    try:
        api = get_tushare_api()
        print(f'  [OK] Tushare 连接成功')
    except Exception as e:
        print(f'  [ERROR] Tushare 连接失败: {e}')
        print(f'  请先确保 credentials.py 配置正确, 且 Tushare 5000积分可用')
        return

    fetch_start = (last_date + pd.Timedelta(days=1)).strftime('%Y%m%d')
    
    if file_name == 'daily_prices.csv':
        print(f'  拉取 {fetch_start} ~ {END_DATE} 日线数据...')
        try:
            new = api.daily(
                start_date=fetch_start, end_date=END_DATE,
                fields='ts_code,trade_date,open,high,low,close,pre_close,change,pct_chg,vol,amount'
            )
            if new is not None and len(new) > 0:
                new.rename(columns={'trade_date': 'trade_date'}, inplace=True)
                combined = pd.concat([df_existing, new], ignore_index=True)
                combined.drop_duplicates(subset=['ts_code', 'trade_date'], keep='last', inplace=True)
                combined.to_csv(fpath, index=False, encoding='utf-8-sig')
                print(f'  [OK] daily_prices.csv: +{len(new)}条, 总计{len(combined)}条')
            else:
                print(f'  [WARN] 无新数据')
        except Exception as e:
            print(f'  [ERROR] daily_prices: {e}')
    
    elif file_name == 'adj_factor.csv':
        print(f'  拉取 {fetch_start} ~ {END_DATE} 复权因子...')
        try:
            new = api.adj_factor(
                start_date=fetch_start, end_date=END_DATE,
                fields='ts_code,trade_date,adj_factor'
            )
            if new is not None and len(new) > 0:
                combined = pd.concat([df_existing, new], ignore_index=True)
                combined.drop_duplicates(subset=['ts_code', 'trade_date'], keep='last', inplace=True)
                combined.to_csv(fpath, index=False, encoding='utf-8-sig')
                print(f'  [OK] adj_factor.csv: +{len(new)}条, 总计{len(combined)}条')
            else:
                print(f'  [WARN] 无新数据')
        except Exception as e:
            print(f'  [ERROR] adj_factor: {e}')
    
    elif file_name == 'daily_basic.csv':
        print(f'  拉取 {fetch_start} ~ {END_DATE} 日频估值...')
        try:
            new = api.daily_basic(
                start_date=fetch_start, end_date=END_DATE,
                fields='ts_code,trade_date,turnover_rate,turnover_rate_f,volume_ratio,pe_ttm,pb,ps_ttm,dv_ttm,total_mv,circ_mv'
            )
            if new is not None and len(new) > 0:
                combined = pd.concat([df_existing, new], ignore_index=True)
                combined.drop_duplicates(subset=['ts_code', 'trade_date'], keep='last', inplace=True)
                combined.to_csv(fpath, index=False, encoding='utf-8-sig')
                print(f'  [OK] daily_basic.csv: +{len(new)}条, 总计{len(combined)}条')
            else:
                print(f'  [WARN] 无新数据')
        except Exception as e:
            print(f'  [ERROR] daily_basic: {e}')

    elif file_name == 'index_000300_daily.csv':
        print(f'  拉取 {fetch_start} ~ {END_DATE} 沪深300日线...')
        try:
            new = api.index_daily(
                ts_code='000300.SH',
                start_date=fetch_start, end_date=END_DATE,
                fields='ts_code,trade_date,close,open,high,low,vol,amount'
            )
            if new is not None and len(new) > 0:
                combined = pd.concat([df_existing, new], ignore_index=True)
                combined.drop_duplicates(subset=['ts_code', 'trade_date'], keep='last', inplace=True)
                combined.to_csv(fpath, index=False, encoding='utf-8-sig')
                print(f'  [OK] index_000300_daily.csv: +{len(new)}条, 总计{len(combined)}条')
            else:
                print(f'  [WARN] 无新数据')
        except Exception as e:
            print(f'  [ERROR] index_000300: {e}')


if __name__ == '__main__':
    print('=' * 60)
    print(f'  数据增量更新: {START_DATE} ~ {END_DATE}')
    print(f'  时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print('=' * 60)
    
    # 按顺序更新四个文件
    for fname in ['daily_prices.csv', 'adj_factor.csv', 'daily_basic.csv', 'index_000300_daily.csv']:
        get_new_data(fname, api_func=None)
        print()
    
    print('更新完成!')
