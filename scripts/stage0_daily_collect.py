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


def check_moneyflow_coverage(pro, existing_file, threshold=1000, recent_days=60,
                             auto_repair=True, max_repair=200):
    """覆盖率自检 (P-20260916-003)

    背景: 2026-09-16 实测发现 2025-08-08 ~ 2026-02-05 的 122 个交易日每日仅残留
    2~4 行(尾部残片), 根因 = 历史回补走「长区间 + ts_code 分批」查询, 单次返回量被
    Tushare 上限截断, 旧日期只剩尾部。日频增量路径本身安全(50 只 × 5 日 ≈ 250 行),
    但任何一次性长区间回补都会重蹈覆辙 ⇒ 必须用「按日行数」而非「最新日期」判断完整性。

    判据: 某交易日行数 < threshold(默认 1000, 远低于满市场 ~5150~5550) 即视为缺口。
    修复: 该日改用 trade_date= 单日口径重采(实测单日 5548 行完整返回, 不截断)。
    """
    if not existing_file.exists():
        print('  [覆盖自检] 文件不存在, 跳过')
        return {}
    cnt = pd.read_csv(existing_file, usecols=['trade_date'],
                      dtype={'trade_date': str}).groupby('trade_date').size()
    base = int(cnt.tail(recent_days).median()) if len(cnt) else 0
    bad = {str(d): int(n) for d, n in cnt.items() if int(n) < threshold}
    print(f'  [覆盖自检] 交易日 {len(cnt)} 个 | 近期中位数 {base} | 阈值 {threshold} | '
          f'缺口日 {len(bad)} 个')
    if not bad:
        print('  [覆盖自检] OK 无覆盖缺口')
        return {}
    tail = sorted(bad.items())[-20:]
    print('  [覆盖自检] ⚠️ 缺口日(最近20): ' + ', '.join(f'{d}:{n}' for d, n in tail))
    if not auto_repair:
        return bad

    targets = sorted(bad)[-max_repair:]
    print(f'  [覆盖自检] 触发按日重采 {len(targets)} 个交易日 ...')
    fixed = []
    for j, d in enumerate(targets, 1):
        try:
            nd = pro.moneyflow(trade_date=d)
            if nd is not None and len(nd) >= threshold:
                fixed.append(nd)
            else:
                print(f'    {d} 重采仅 {0 if nd is None else len(nd)} 行, 保留原状')
        except Exception as e:
            print(f'    {d} 重采失败: {e}')
        time.sleep(0.35)
        if j % 20 == 0:
            print(f'    ... {j}/{len(targets)}')
    if not fixed:
        print('  [覆盖自检] 无成功重采, 数据保持原状')
        return bad

    new = pd.concat(fixed, ignore_index=True)
    new['trade_date'] = new['trade_date'].astype(str)
    new['ts_code'] = new['ts_code'].astype(str)
    old = pd.read_csv(existing_file, dtype={'ts_code': str, 'trade_date': str})
    keep = ~old['trade_date'].isin(set(new['trade_date'].unique()))
    combined = pd.concat([old[keep], new], ignore_index=True)
    combined.drop_duplicates(subset=['ts_code', 'trade_date'], inplace=True)
    combined.sort_values(['ts_code', 'trade_date'], inplace=True)
    combined.to_csv(existing_file, index=False)
    print(f'  [覆盖自检] 已重采修复 {len(fixed)} 日 / 新增 {len(new):,} 行 / '
          f'总计 {len(combined):,} 行')
    remain = {str(d): int(n) for d, n in combined.groupby('trade_date').size().items()
              if int(n) < threshold}
    print(f'  [覆盖自检] 修复后剩余缺口日 {len(remain)} 个')
    return remain


def collect_forecast_incremental(pro, existing_file, lookback_days=90):
    """采集分析师业绩预告 (增量)

    P2-2a 修复 (2026-09-28): 原实现三处缺陷会破坏已并入的 PIT 数据
      (1) 去重键 (ts_code, ann_date) 为弱键 —— 同一公告日对不同报告期
          (end_date) 的多条预告会被压成一行 (并入后实测 5 行);
      (2) sort_values 默认 quicksort 不稳定 + 未含 end_date ⇒ 行序不确定;
      (3) to_csv 默认 LF 会把 CRLF 主表静默改写, 且默认类型推断往返
          会改写数值字符串形态。
    现改为: 三键去重 + kind='stable' 三键排序 + 行尾保持 + dtype=str 保真往返。
    """
    existing = pd.DataFrame()
    if existing_file.exists():
        # 全字符读取: 避免类型推断往返改写数值形态 (字节保真)
        existing = pd.read_csv(existing_file, dtype=str, keep_default_na=False)
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
        _K3 = ['ts_code', 'ann_date', 'end_date']
        new_data = pd.concat(all_new, ignore_index=True)
        new_data.drop_duplicates(subset=_K3, inplace=True)
        
        if not existing.empty:
            combined = pd.concat([existing, new_data], ignore_index=True)
            combined.drop_duplicates(subset=_K3, inplace=True)
        else:
            combined = new_data
        
        combined.sort_values(_K3, inplace=True, kind='stable')
        # 行尾保持 (P2-2a): 既有文件 CRLF 时不得改写为 LF
        nl = '\n'
        try:
            with open(existing_file, 'rb') as _f:
                if b'\r\n' in _f.read(65536):
                    nl = '\r\n'
        except Exception:
            pass
        with open(existing_file, 'w', encoding='utf-8', newline='') as _f:
            combined.to_csv(_f, index=False, lineterminator=nl)
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

    # 1b. 覆盖完整性自检 (P-20260916-003: 「最新日期」不能代表「覆盖完整」)
    print('\n--- 1b. moneyflow 覆盖完整性自检 ---')
    check_moneyflow_coverage(pro, mf_path)

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
    from pathlib import Path as _FuxiPath
    import sys as _FuxiSys
    _FuxiSys.path.insert(0, str(_FuxiPath(__file__).resolve().parents[1]))
    from fuxi_v1.legacy_writers import legacy_cli_guard as _FuxiGuard
    _fuxi_guard = _FuxiGuard(__file__)
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--full', action='store_true', help='全量重新采集')
    parser.add_argument('--moneyflow-only', action='store_true', help='仅采集资金流向 (17:00 增量)')
    parser.add_argument('--check-coverage', action='store_true',
                        help='只做 moneyflow 覆盖完整性自检 (P-20260916-003)')
    parser.add_argument('--no-repair', action='store_true', help='自检只告警不重采')
    args = parser.parse_args()
    if args.check_coverage:
        _pro = get_tushare_api()
        check_moneyflow_coverage(_pro, DATA_DIR / 'moneyflow_daily.csv',
                                 auto_repair=not args.no_repair)
    else:
        main(full=args.full, moneyflow_only=args.moneyflow_only)
