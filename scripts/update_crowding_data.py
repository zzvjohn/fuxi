#!/usr/bin/env python3
"""
每日数据更新 + 拥挤度监控 一体化脚本
=====================================
1. 从 Tushare 增量拉取最新日线数据
2. 增量拉取涨跌停价格数据
3. 追加到本地 CSV（去重）
4. 运行拥挤度监控
5. 对比上次结果，报告趋势变化

用法:
    python scripts/update_crowding_data.py            # 默认增量更新
    python scripts/update_crowding_data.py --full     # 全量重新拉取（慎用）
    python scripts/update_crowding_data.py --monitor-only  # 仅运行监控（不更新数据）
"""

import pandas as pd
import numpy as np
import os
import sys
import time
import json
from datetime import datetime, timedelta

# ============ 路径配置 ============
WORKSPACE = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(WORKSPACE, 'data', 'raw')
REPORT_DIR = os.path.join(WORKSPACE, 'reports')
CROWDING_SCRIPT = os.path.join(WORKSPACE, 'strategies', 'final', 'crowding_monitor.py')

PRICES_FILE = os.path.join(DATA_DIR, 'daily_prices.csv')
LIMITS_FILE = os.path.join(DATA_DIR, 'stk_limit.csv')
TREND_FILE = os.path.join(REPORT_DIR, 'crowding_trend.json')

# Tushare 速率控制（5000积分 = 300次/分钟）
REQUEST_INTERVAL = 0.35  # 秒
MAX_RETRIES = 5
RETRY_DELAY = 120  # 秒


def get_tushare_pro():
    """延迟导入以加快 --monitor-only 模式"""
    sys.path.insert(0, WORKSPACE)
    from credentials import get_tushare_api
    return get_tushare_api()


def get_latest_date(filepath, date_col='trade_date'):
    """获取CSV文件中最新的日期（只读日期列，高效，兼容多种格式）"""
    if not os.path.exists(filepath):
        return None
    try:
        dates = pd.read_csv(
            filepath, dtype={'ts_code': str, date_col: str},
            usecols=[date_col], encoding='utf-8-sig'
        )[date_col]
        return pd.to_datetime(dates, format='mixed').max()
    except Exception:
        return None


def get_existing_dates(filepath, date_col='trade_date'):
    """获取CSV中已有的所有日期集合（兼容 int/str/datetime 多种格式）"""
    if not os.path.exists(filepath):
        return set()
    # 只读日期列，节省内存；先读为str避免parse_dates对int格式失效
    dates = pd.read_csv(
        filepath, dtype={'ts_code': str, date_col: str},
        usecols=[date_col], encoding='utf-8-sig'
    )[date_col].drop_duplicates()
    # 统一转换为 date 对象
    parsed = pd.to_datetime(dates, format='mixed').dt.date
    return set(parsed)


def pull_daily_prices(pro, start_date, end_date):
    """从 Tushare 拉取日线数据"""
    print(f"\n{'='*60}")
    print(f"  拉取日线数据: {start_date} ~ {end_date}")
    print(f"{'='*60}")

    # 获取交易日历
    print("  获取交易日历...")
    trade_cal = pro.trade_cal(
        exchange='SSE',
        start_date=start_date.strftime('%Y%m%d'),
        end_date=end_date.strftime('%Y%m%d'),
        is_open='1'
    )
    trade_dates = sorted(trade_cal['cal_date'].tolist())
    print(f"  共 {len(trade_dates)} 个交易日")

    existing_dates = get_existing_dates(PRICES_FILE)
    new_dates = [d for d in trade_dates if datetime.strptime(d, '%Y%m%d').date() not in existing_dates]

    if not new_dates:
        print("  ✅ 日线数据已是最新，无需更新")
        return

    print(f"  需拉取 {len(new_dates)} 个新交易日: {new_dates[0]} ~ {new_dates[-1]}")

    all_frames = []
    for i, trade_date in enumerate(new_dates):
        for attempt in range(MAX_RETRIES):
            try:
                df = pro.daily(trade_date=trade_date)
                if df is not None and len(df) > 0:
                    all_frames.append(df)
                print(f"  [{i+1}/{len(new_dates)}] {trade_date}: {len(df) if df is not None else 0} 条")
                break
            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    print(f"  ⚠️ {trade_date} 第{attempt+1}次失败: {e}，{RETRY_DELAY}s后重试...")
                    time.sleep(RETRY_DELAY)
                else:
                    print(f"  ❌ {trade_date} 全部重试失败: {e}")

        time.sleep(REQUEST_INTERVAL)  # 速率控制

    if all_frames:
        new_data = pd.concat(all_frames, ignore_index=True)
        # 只保留与本地CSV一致的列，过滤掉Tushare可能多返回的change等列
        keep_cols = ['ts_code', 'trade_date', 'open', 'high', 'low',
                     'close', 'pre_close', 'pct_chg', 'vol', 'amount']
        new_data = new_data[[c for c in keep_cols if c in new_data.columns]]
        # Tushare 返回的 trade_date 是字符串，需要转换
        if new_data['trade_date'].dtype == object:
            new_data['trade_date'] = pd.to_datetime(new_data['trade_date'])

        # 追加到现有文件
        header = not os.path.exists(PRICES_FILE)
        new_data.to_csv(PRICES_FILE, mode='a', header=header, index=False, encoding='utf-8-sig')
        print(f"\n  ✅ 已追加 {len(new_data):,} 条日线记录")
    else:
        print(f"\n  ⚠️ 未拉取到任何新数据")


def pull_stk_limits(pro, start_date, end_date):
    """从 Tushare 拉取涨跌停价格数据"""
    print(f"\n{'='*60}")
    print(f"  拉取涨跌停数据: {start_date} ~ {end_date}")
    print(f"{'='*60}")

    existing_dates = get_existing_dates(LIMITS_FILE)

    # 获取交易日历
    trade_cal = pro.trade_cal(
        exchange='SSE',
        start_date=start_date.strftime('%Y%m%d'),
        end_date=end_date.strftime('%Y%m%d'),
        is_open='1'
    )
    trade_dates = sorted(trade_cal['cal_date'].tolist())
    new_dates = [d for d in trade_dates if datetime.strptime(d, '%Y%m%d').date() not in existing_dates]

    if not new_dates:
        print("  ✅ 涨跌停数据已是最新，无需更新")
        return

    print(f"  需拉取 {len(new_dates)} 个新交易日")

    all_frames = []
    for i, trade_date in enumerate(new_dates):
        for attempt in range(MAX_RETRIES):
            try:
                df = pro.stk_limit(trade_date=trade_date)
                if df is not None and len(df) > 0:
                    all_frames.append(df)
                print(f"  [{i+1}/{len(new_dates)}] {trade_date}: {len(df) if df is not None else 0} 条")
                break
            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    print(f"  ⚠️ {trade_date} 第{attempt+1}次失败: {e}，{RETRY_DELAY}s后重试...")
                    time.sleep(RETRY_DELAY)
                else:
                    print(f"  ❌ {trade_date} 全部重试失败: {e}")

        time.sleep(REQUEST_INTERVAL)

    if all_frames:
        new_data = pd.concat(all_frames, ignore_index=True)
        # 强制列序与本地CSV一致（Tushare返回列序可能不同）
        keep_cols = ['ts_code', 'trade_date', 'up_limit', 'down_limit']
        new_data = new_data[[c for c in keep_cols if c in new_data.columns]]
        if new_data['trade_date'].dtype == object:
            new_data['trade_date'] = pd.to_datetime(new_data['trade_date'])

        header = not os.path.exists(LIMITS_FILE)
        new_data.to_csv(LIMITS_FILE, mode='a', header=header, index=False, encoding='utf-8-sig')
        print(f"\n  ✅ 已追加 {len(new_data):,} 条涨跌停记录")
    else:
        print(f"\n  ⚠️ 未拉取到任何新涨跌停数据")


def run_crowding_monitor():
    """运行拥挤度监控"""
    print(f"\n{'='*60}")
    print(f"  运行拥挤度监控")
    print(f"{'='*60}")

    # 用 subprocess 执行，保持独立性
    import subprocess

    result = subprocess.run(
        [sys.executable, CROWDING_SCRIPT],
        cwd=WORKSPACE,
        capture_output=True, text=True, timeout=300
    )

    if result.returncode != 0:
        print(f"  ❌ 拥挤度监控运行失败:\n{result.stderr}")
        return None

    print(result.stdout)

    # 解析最新结果
    factor_file = os.path.join(REPORT_DIR, 'crowding_factors.csv')
    if os.path.exists(factor_file):
        df = pd.read_csv(factor_file, parse_dates=['trade_date'])
        # 取最后一个有mom_rolling_sharpe的有效行（最新日无次日数据导致NaN）
        valid = df.dropna(subset=['crowding_score', 'mom_rolling_sharpe'])
        if len(valid) == 0:
            valid = df.dropna(subset=['crowding_score'])
        if len(valid) == 0:
            return None
        latest = valid.iloc[-1]
        return {
            'date': str(latest['trade_date']),
            'crowding_score': float(latest['crowding_score']),
            'crowding_level': str(latest['crowding_level']),
            'mom_sharpe': float(latest['mom_rolling_sharpe']),
            'rev_sharpe': float(latest['rev_rolling_sharpe']),
            'mom_vol': float(latest['mom_vol']),
            'rev_vol': float(latest['rev_vol']),
            'corr': float(latest['mom_rev_corr']),
            'small_share': float(latest['small_turnover_share']),
            'timestamp': datetime.now().isoformat()
        }

    return None


def compare_trend(current, previous_file):
    """对比当前结果与上次结果，报告趋势变化"""
    if not current:
        return

    if not os.path.exists(previous_file):
        # 首次运行，保存基准
        with open(previous_file, 'w', encoding='utf-8') as f:
            json.dump([current], f, ensure_ascii=False, indent=2)
        print(f"\n  📌 首次记录，已保存基准到 {previous_file}")
        return

    with open(previous_file, 'r', encoding='utf-8') as f:
        history = json.load(f)

    prev = history[-1] if history else None

    print(f"\n{'='*60}")
    print(f"  趋势变化对比")
    print(f"{'='*60}")
    print(f"  上次: {prev.get('date','?') if prev else '无'}")
    print(f"  本次: {current['date']}")
    print(f"  {'─'*50}")

    changes = []
    for key, label, fmt, unit in [
        ('crowding_score', '综合拥挤度', '.2f', ''),
        ('mom_sharpe', '动量夏普', '.2f', ''),
        ('rev_sharpe', '反转夏普', '.2f', ''),
        ('mom_vol', '动量波动率', '.1%', ''),
        ('rev_vol', '反转波动率', '.1%', ''),
        ('corr', '因子相关性', '.3f', ''),
        ('small_share', '小市值成交占比', '.1%', ''),
    ]:
        curr_val = current.get(key)
        prev_val = prev.get(key) if prev else None

        if prev_val is not None and curr_val is not None:
            delta = curr_val - prev_val
            arrow = '↑' if delta > 0 else '↓' if delta < 0 else '→'
            print(f"  {label:12s}: {prev_val:{fmt}} → {curr_val:{fmt}} ({arrow}{abs(delta):{fmt}})")
            changes.append({'metric': key, 'label': label, 'prev': prev_val, 'curr': curr_val, 'delta': delta})
        else:
            print(f"  {label:12s}: {curr_val:{fmt}} (首次)")

    # 关键预警
    print(f"\n  【关键预警】")
    warnings = []
    if current.get('crowding_score', 0) > 0.5:
        warnings.append(f"⚠️ 拥挤度 {current['crowding_score']:.2f} > 0.5，中度拥挤")
    if current.get('rev_sharpe', 0) < -0.5:
        warnings.append(f"⚠️ 反转夏普 {current['rev_sharpe']:.2f} < -0.5，反转因子持续恶化")
    if current.get('mom_sharpe', 0) > 0.5:
        warnings.append(f"✅ 动量夏普 {current['mom_sharpe']:.2f} > 0.5，动量环境良好")
    if current.get('corr', 0) > 0.7:
        warnings.append(f"⚠️ 因子相关性 {current['corr']:.3f} > 0.7，同向拥挤风险")
    if not warnings:
        warnings.append("✅ 无重大预警")

    for w in warnings:
        print(f"  {w}")

    # 追加历史记录
    history.append(current)
    # 只保留最近90天
    history = history[-90:]
    with open(previous_file, 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    print(f"\n  📁 趋势记录已保存: {previous_file} (共 {len(history)} 条)")


def main():
    monitor_only = '--monitor-only' in sys.argv
    full_refresh = '--full' in sys.argv

    if not monitor_only:
        pro = get_tushare_pro()

        # 确定日期范围
        if full_refresh:
            start_date = pd.Timestamp('2021-01-01')
        else:
            # 增量：只读日期列获取最新日期（避免全量读取580万行）
            last_date = get_latest_date(PRICES_FILE)
            if last_date is not None:
                start_date = last_date + timedelta(days=1)
            else:
                start_date = pd.Timestamp('2021-01-01')

        end_date = pd.Timestamp.now()

        if start_date > end_date:
            print("✅ 数据已是最新，无需更新")
        else:
            pull_daily_prices(pro, start_date, end_date)
            pull_stk_limits(pro, start_date, end_date)

    # 运行监控
    current = run_crowding_monitor()
    if current:
        compare_trend(current, TREND_FILE)

    print(f"\n{'='*60}")
    print(f"  完成: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
