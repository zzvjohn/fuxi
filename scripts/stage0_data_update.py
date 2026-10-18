"""
Stage 0 数据增量更新脚本
========================
按 PROJECT.md 定义的 6 类数据，从 Tushare 增量拉取最新数据并追加到本地 CSV。

特点:
- 增量模式：自动检测每张表的最后日期，只拉取新数据
- 容错：单表失败不影响其他表继续
- 去重：按 (ts_code, date) 自动去重
- report_rc 分批：10次/h 频率限制下安全拉取

用法:
    python scripts/stage0_data_update.py                # 增量更新所有表
    python scripts/stage0_data_update.py --dry-run      # 仅检查，不拉取
    python scripts/stage0_data_update.py --only report_rc  # 仅更新指定表
"""
import sys
import os
from pathlib import Path
import pandas as pd
import time
from datetime import datetime, timedelta
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent))
from credentials import get_tushare_api

RAW_DIR = Path('e:/quant/data/raw')

# ============ 配置 ============
TABLES = {
    'daily_basic': {
        'file': 'daily_basic.csv',
        'api_method': 'daily_basic',
        'api_params_template': lambda start, end: {
            'start_date': start, 'end_date': end,
            'fields': 'ts_code,trade_date,turnover_rate,turnover_rate_f,volume_ratio,pe,pe_ttm,pb,pb_mrq,ps,ps_ttm,dv_ratio,dv_ttm,total_mv,circ_mv'
        },
        'key_cols': ['ts_code', 'trade_date'],
        'date_col': 'trade_date',
    },
    'adj_factor': {
        'file': 'adj_factor.csv',
        'api_method': 'adj_factor',
        'api_params_template': lambda start, end: {
            'start_date': start, 'end_date': end,
            'fields': 'ts_code,trade_date,adj_factor'
        },
        'key_cols': ['ts_code', 'trade_date'],
        'date_col': 'trade_date',
    },
    'stk_factor': {
        'file': 'stk_factor.csv',
        'api_method': 'stk_factor',
        'api_params_template': lambda start, end: {
            'start_date': start, 'end_date': end,
        },
        'key_cols': ['ts_code', 'trade_date'],
        'date_col': 'trade_date',
    },
    'fina_indicator': {
        'file': 'fina_indicator.csv',
        'api_method': 'fina_indicator',
        'api_params_template': lambda start, end: {
            'start_date': start, 'end_date': end,
            'fields': 'ts_code,ann_date,end_date,roe,roe_dt,netprofit_yoy,or_yoy,grossprofit_margin,netprofit_margin,debt_to_assets'
        },
        'key_cols': ['ts_code', 'ann_date', 'end_date'],
        'date_col': 'ann_date',
    },
    'moneyflow_hsgt': {
        'file': 'moneyflow_hsgt.csv',
        'api_method': 'moneyflow_hsgt',
        'api_params_template': lambda start, end: {
            'start_date': start, 'end_date': end,
        },
        'key_cols': ['trade_date'],
        'date_col': 'trade_date',
    },
    'index_000300': {
        'file': 'index_000300_daily.csv',
        'api_method': 'index_daily',
        'api_params_template': lambda start, end: {
            'ts_code': '000300.SH',
            'start_date': start, 'end_date': end,
            'fields': 'ts_code,trade_date,close,open,high,low,vol,amount,pct_chg'
        },
        'key_cols': ['ts_code', 'trade_date'],
        'date_col': 'trade_date',
    },
}


def parse_safe_date(series):
    """安全解析多种日期格式: 整数20260430 / 字符串'20260430' / 日期对象"""
    if series.dtype in ['int64', 'int32', 'float64']:
        # Tushare 返回整数格式 YYYYMMDD
        return pd.to_datetime(series.astype(str), format='%Y%m%d', errors='coerce')
    else:
        return pd.to_datetime(series, errors='coerce')


def get_local_state(fpath, date_col, key_cols):
    """读取本地文件,返回 (existing_df, last_date_str)"""
    if not fpath.exists():
        print(f'    [NEW] {fpath.name} 不存在, 将全量拉取')
        return pd.DataFrame(), '20210101'
    
    try:
        df = pd.read_csv(fpath, encoding='utf-8-sig', low_memory=False)
        if date_col in df.columns:
            df[date_col] = parse_safe_date(df[date_col])
            last_date = df[date_col].max()
            if pd.isna(last_date):
                last_date = pd.Timestamp('2021-01-01')
            return df, last_date.strftime('%Y%m%d')
        return df, '20210101'
    except Exception as e:
        print(f'    [WARN] 读取失败 ({e}), 将全量重拉')
        return pd.DataFrame(), '20210101'


def incremental_update(pro, table_name, cfg, dry_run=False):
    """增量更新一张表"""
    fpath = RAW_DIR / cfg['file']
    method = cfg['api_method']
    date_col = cfg['date_col']
    key_cols = cfg['key_cols']
    
    existing_df, last_date_str = get_local_state(fpath, date_col, key_cols)
    fetch_start = (pd.Timestamp(last_date_str) + pd.Timedelta(days=1)).strftime('%Y%m%d')
    fetch_end = datetime.now().strftime('%Y%m%d')
    
    if fetch_start >= fetch_end:
        print(f'    [SKIP] {cfg["file"]} 已是最新 ({last_date_str})')
        return {'table': table_name, 'file': cfg['file'], 'status': 'SKIPPED',
                'new_rows': 0, 'total_rows': len(existing_df), 'last_date': last_date_str}
    
    if dry_run:
        print(f'    [DRY-RUN] {cfg["file"]}: 需拉取 {fetch_start} ~ {fetch_end}')
        return {'table': table_name, 'file': cfg['file'], 'status': 'DRY_RUN',
                'new_rows': 0, 'total_rows': len(existing_df), 'last_date': last_date_str}
    
    print(f'    [PULL] {cfg["file"]}: {fetch_start} ~ {fetch_end} ...')
    
    try:
        params = cfg['api_params_template'](fetch_start, fetch_end)
        api_func = getattr(pro, method)
        new = api_func(**params)
        time.sleep(0.4)
    except Exception as e:
        print(f'    [ERROR] API 调用失败: {e}')
        return {'table': table_name, 'file': cfg['file'], 'status': 'ERROR',
                'new_rows': 0, 'total_rows': len(existing_df), 'error': str(e)}
    
    if new is None or len(new) == 0:
        print(f'    [EMPTY] 无新数据')
        return {'table': table_name, 'file': cfg['file'], 'status': 'NO_DATA',
                'new_rows': 0, 'total_rows': len(existing_df), 'last_date': last_date_str}
    
    # 统一日期格式
    if date_col in new.columns:
        new[date_col] = pd.to_datetime(new[date_col], errors='coerce')
    
    # 合并 + 去重
    if len(existing_df) > 0:
        combined = pd.concat([existing_df, new], ignore_index=True)
    else:
        combined = new
    
    dup_cols = [c for c in key_cols if c in combined.columns]
    if dup_cols:
        before = len(combined)
        combined.drop_duplicates(subset=dup_cols, keep='last', inplace=True)
        removed = before - len(combined)
    else:
        removed = 0
    
    combined.to_csv(fpath, index=False, encoding='utf-8-sig')
    
    # 获取最后日期
    if date_col in combined.columns:
        final_last = combined[date_col].max().strftime('%Y%m%d') if not combined[date_col].max() is pd.NaT else last_date_str
    else:
        final_last = last_date_str
    
    print(f'    [OK] {cfg["file"]}: +{len(new)}条 (去重-{removed}), 总计{len(combined)}条, 最新{final_last}')
    
    return {'table': table_name, 'file': cfg['file'], 'status': 'OK',
            'new_rows': len(new), 'total_rows': len(combined), 'last_date': final_last,
            'dup_removed': removed}


def update_report_rc(pro, dry_run=False):
    """
    更新 report_rc.csv — 分批拉取（10次/h 限制）
    
    report_rc 需要按日期或按 ts_code 分批拉取。
    采用按日期范围分批策略，每批拉取约 2 周的数据。
    每 5 批后等待 6 分钟以遵守频率限制。
    """
    fpath = RAW_DIR / 'report_rc.csv'
    table_name = 'report_rc'
    
    if not fpath.exists():
        print(f'    [NEW] report_rc.csv 不存在')
        existing_df = pd.DataFrame()
        last_date_str = '20250101'  # 从2025年1月开始
    else:
        try:
            existing_df = pd.read_csv(fpath, encoding='utf-8-sig', low_memory=False)
            if 'report_date' in existing_df.columns or 'trade_date' in existing_df.columns:
                date_col = 'report_date' if 'report_date' in existing_df.columns else 'trade_date'
                existing_df[date_col] = pd.to_datetime(existing_df[date_col], errors='coerce')
                last_date = existing_df[date_col].max()
                last_date_str = last_date.strftime('%Y%m%d') if not pd.isna(last_date) else '20250101'
            elif 'report_date' in str(existing_df.columns.tolist()):  # 可能有编码问题
                # 找包含 report_date 的列
                for c in existing_df.columns:
                    if 'report' in c.lower() and 'date' in c.lower():
                        existing_df[c] = pd.to_datetime(existing_df[c], errors='coerce')
                        last_date = existing_df[c].max()
                        last_date_str = last_date.strftime('%Y%m%d') if not pd.isna(last_date) else '20250101'
                        break
                else:
                    last_date_str = '20250601'
            else:
                last_date_str = '20250601'
        except Exception as e:
            print(f'    [WARN] 读取 report_rc 失败 ({e})')
            existing_df = pd.DataFrame()
            last_date_str = '20250601'
    
    fetch_end = datetime.now().strftime('%Y%m%d')
    fetch_start_dt = pd.Timestamp(last_date_str) + pd.Timedelta(days=1)
    
    if fetch_start_dt > pd.Timestamp(fetch_end):
        print(f'    [SKIP] report_rc 已是最新 ({last_date_str})')
        return {'table': table_name, 'file': 'report_rc.csv', 'status': 'SKIPPED',
                'new_rows': 0, 'total_rows': len(existing_df), 'last_date': last_date_str}
    
    if dry_run:
        print(f'    [DRY-RUN] report_rc: {fetch_start_dt.strftime("%Y%m%d")} ~ {fetch_end}')
        return {'table': table_name, 'file': 'report_rc.csv', 'status': 'DRY_RUN'}
    
    # 按 14 天一批
    batch_size = 14
    current = fetch_start_dt
    all_new = []
    batch_count = 0
    
    while current <= pd.Timestamp(fetch_end):
        batch_end = min(current + pd.Timedelta(days=batch_size - 1), pd.Timestamp(fetch_end))
        s = current.strftime('%Y%m%d')
        e = batch_end.strftime('%Y%m%d')
        
        print(f'    [BATCH {batch_count+1}] report_rc {s} ~ {e} ...', end=' ')
        
        try:
            new = pro.report_rc(
                start_date=s, end_date=e,
                fields='ts_code,name,report_date,report_title,report_type,classify,'
                       'org_name,author_name,quarter,op_rt,op_pr,tp,np,eps,pe,rd,roe,'
                       'ev_ebitda,rating,max_price,min_price'
            )
            time.sleep(0.5)
            
            if new is not None and len(new) > 0:
                all_new.append(new)
                print(f'+{len(new)}条')
            else:
                print('0条')
        except Exception as e:
            print(f'[ERR] {e}')
            # 继续下一批
            time.sleep(3)
        
        batch_count += 1
        current = batch_end + pd.Timedelta(days=1)
        
        # 每5批等待6分钟（10次/h 限制）
        if batch_count % 5 == 0 and current <= pd.Timestamp(fetch_end):
            wait_sec = 360
            print(f'    [WAIT] 频率限制, 等待 {wait_sec//60} 分钟...')
            time.sleep(wait_sec)
    
    if not all_new:
        print(f'    [EMPTY] report_rc 无新数据')
        return {'table': table_name, 'file': 'report_rc.csv', 'status': 'NO_DATA',
                'total_rows': len(existing_df), 'last_date': last_date_str}
    
    new_df = pd.concat(all_new, ignore_index=True)
    
    # 合并 + 去重
    if len(existing_df) > 0:
        combined = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        combined = new_df
    
    # 去重（按 ts_code + report_date + quarter）
    dup_cols = ['ts_code', 'report_date', 'quarter']
    dup_cols = [c for c in dup_cols if c in combined.columns]
    if dup_cols:
        before = len(combined)
        combined.drop_duplicates(subset=dup_cols, keep='last', inplace=True)
        removed = before - len(combined)
    else:
        removed = 0
    
    combined.to_csv(fpath, index=False, encoding='utf-8-sig')
    
    final_last = 'N/A'
    if 'report_date' in combined.columns:
        combined['report_date'] = pd.to_datetime(combined['report_date'], errors='coerce')
        final_last = combined['report_date'].max().strftime('%Y%m%d')
    
    print(f'    [OK] report_rc.csv: +{len(new_df)}条 (去重-{removed}), 总计{len(combined)}条, 最新{final_last}')
    
    return {'table': table_name, 'file': 'report_rc.csv', 'status': 'OK',
            'new_rows': len(new_df), 'total_rows': len(combined), 'last_date': final_last,
            'dup_removed': removed, 'batches': batch_count}


def main():
    parser = argparse.ArgumentParser(description='Stage 0 数据增量更新')
    parser.add_argument('--dry-run', action='store_true', help='仅检查, 不拉取')
    parser.add_argument('--only', type=str, help='仅更新指定表 (daily_basic/adj_factor/stk_factor/fina_indicator/moneyflow_hsgt/report_rc/index_000300)')
    parser.add_argument('--skip-report-rc', action='store_true', help='跳过 report_rc (耗时最长)')
    args = parser.parse_args()
    
    print('=' * 60)
    print(f'  Stage 0 数据增量更新')
    print(f'  时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    if args.dry_run:
        print(f'  [DRY RUN 模式]')
    print('=' * 60)
    
    # Tushare 连接
    try:
        pro = get_tushare_api()
        print('[OK] Tushare 连接成功\n')
    except Exception as e:
        print(f'[FATAL] Tushare 连接失败: {e}')
        sys.exit(1)
    
    results = []
    
    # 常规表增量更新
    for table_name, cfg in TABLES.items():
        if args.only and args.only != table_name:
            continue
        print(f'--- {table_name}: {cfg["file"]} ---')
        result = incremental_update(pro, table_name, cfg, dry_run=args.dry_run)
        results.append(result)
        print()
    
    # report_rc 单独处理（分批 + 频率限制）
    if not args.skip_report_rc and (not args.only or args.only == 'report_rc'):
        print(f'--- report_rc: report_rc.csv ---')
        result = update_report_rc(pro, dry_run=args.dry_run)
        results.append(result)
        print()
    
    # 汇总
    print('=' * 60)
    print('  汇总报告')
    print('=' * 60)
    for r in results:
        if r['status'] == 'OK':
            print(f'  ✅ {r["file"]:30s} +{r["new_rows"]:>6d}条  总计{r.get("total_rows", 0):>8d}条  最新{r.get("last_date", "N/A")}')
        elif r['status'] == 'SKIPPED':
            print(f'  ⏭️  {r["file"]:30s} 已最新 ({r.get("last_date", "N/A")})')
        elif r['status'] == 'DRY_RUN':
            print(f'  🔍 {r["file"]:30s} 需更新 (最近 {r.get("last_date", "N/A")})')
        elif r['status'] == 'NO_DATA':
            print(f'  ⚪ {r["file"]:30s} 无新数据')
        else:
            print(f'  ❌ {r["file"]:30s} 错误: {r.get("error", "unknown")}')
    
    print(f'\n完成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')


if __name__ == '__main__':
    main()
