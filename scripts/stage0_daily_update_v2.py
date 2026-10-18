"""
Stage 0 每日数据更新 v2
======================
修正版 — 处理 fina_indicator 必须 ts_code 的限制
"""
import sys
sys.path.insert(0, 'e:/quant')
from credentials import get_tushare_api
import pandas as pd
import numpy as np
import time
import os
import csv
from datetime import datetime, timedelta

BASE = 'e:/quant/data/raw'
TODAY = '20260705'

pro = get_tushare_api()
time.sleep(0.3)

# ============================================================
# 工具函数
# ============================================================

def get_max_date_from_csv(path, date_col):
    """高效获取CSV中最大日期值（遍历所有行找max）"""
    if not os.path.exists(path) or os.path.getsize(path) < 100:
        return None
    max_val = None
    with open(path, 'r', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        header = next(reader)
        idx = header.index(date_col)
        for row in reader:
            try:
                val = row[idx].strip()
                if not val or val == 'nan':
                    continue
                if max_val is None or val > max_val:
                    max_val = val
            except (IndexError, ValueError):
                continue
    return max_val

def get_line_count(path):
    """获取有效行数（不含header空行）"""
    if not os.path.exists(path):
        return 0
    count = 0
    with open(path, 'r', encoding='utf-8-sig') as f:
        next(f)  # skip header
        for line in f:
            if line.strip():
                count += 1
    return count

def smart_append_csv(path, df_new, dedup_keys):
    """
    智能追加到CSV：去重后只写新行
    返回 (新增数, 总行数, 最新日期str)
    """
    if df_new is None or len(df_new) == 0:
        return 0, get_line_count(path), None
    
    old_count = get_line_count(path)
    
    # 统一类型
    for k in dedup_keys:
        if k in df_new.columns:
            df_new[k] = df_new[k].astype(str)
    
    if old_count > 0:
        # 读已有数据的去重键
        existing = pd.read_csv(path, usecols=dedup_keys, encoding='utf-8-sig', dtype=str)
        # merge去重
        merged = df_new.merge(existing, on=dedup_keys, how='left', indicator=True)
        new_only = merged[merged['_merge'] == 'left_only'].drop(columns=['_merge'])
    else:
        new_only = df_new.copy()
    
    if len(new_only) == 0:
        return 0, old_count, None

    # v0.9.1 (2026-08-19): 追加前列对齐 — 防止 API 返回字段漂移污染存量文件
    # 事故实证: margin_detail 202608 增量 Tushare 返回 10 列, 直接 append 到 7 列
    # 文件尾部 → 13290 行字段错位, 后续 read_csv 全表加载失败。
    if old_count > 0:
        try:
            with open(path, 'r', encoding='utf-8-sig') as f:
                header_line = f.readline().strip()
            existing_cols = [c.strip() for c in header_line.split(',')]
            missing = [c for c in existing_cols if c not in new_only.columns]
            if missing:
                print(f"    ⚠️ [smart_append] 新数据缺列 {missing}, 填 NaN 追加", flush=True)
                for c in missing:
                    new_only[c] = pd.NA
            extra = [c for c in new_only.columns if c not in existing_cols]
            if extra:
                print(f"    ⚠️ [smart_append] 新数据多列 {extra}, 已丢弃 (按存量列序对齐)", flush=True)
                new_only = new_only[existing_cols]
        except Exception as _e:
            print(f"    ⚠️ [smart_append] 列对齐检查失败(继续原样追加): {_e}", flush=True)

    # 追加
    write_header = (old_count == 0)
    new_only.to_csv(path, mode='a', index=False, header=write_header, encoding='utf-8-sig')
    
    new_count = len(new_only)
    total = get_line_count(path)
    
    # 找最新日期
    date_col = dedup_keys[1] if len(dedup_keys) > 1 else dedup_keys[0]
    max_date = str(new_only[date_col].max()) if date_col in new_only.columns else None
    
    return new_count, total, max_date


def safe_call(func, **kwargs):
    """API包装，3次重试"""
    for attempt in range(3):
        try:
            result = func(**kwargs)
            time.sleep(0.35)
            return result
        except Exception as e:
            err = str(e)
            if any(kw in err for kw in ['频率', '限流', 'frequency', '超限']):
                print(f"    ⏳ 频率限制，等待65秒...")
                time.sleep(65)
            elif attempt < 2:
                print(f"    ⚠️ 重试 {attempt+1}/3: {err[:80]}")
                time.sleep(3)
            else:
                print(f"    ❌ 3次失败: {err[:100]}")
                return None


# ============================================================
# Phase 1: 5张快速表
# ============================================================

print("=" * 70)
print(" Stage 0 每日数据更新")
print(f" 执行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 70)

results = {}

# 获取交易日历（只调一次）
cal = safe_call(pro.trade_cal, exchange='SSE', start_date='20260701', end_date=TODAY, is_open='1')
cal_dates = sorted(cal['cal_date'].astype(str).unique()) if cal is not None else []

# ============================================================
# 1. daily — 日线行情
# ============================================================
print("\n[1/6] daily — 日线行情（后复权）")
try:
    old_cnt = get_line_count(f'{BASE}/daily_prices.csv')
    max_d = get_max_date_from_csv(f'{BASE}/daily_prices.csv', 'trade_date') or '20200101'
    print(f"  存量: {old_cnt:,}条 | 最新: {max_d}")
    
    todo_dates = [d for d in cal_dates if d > max_d]
    total_new = 0
    
    if not todo_dates:
        print(f"  无新交易日（最新交易日={cal_dates[-1] if cal_dates else 'N/A'}）")
    else:
        for td in todo_dates:
            df = safe_call(pro.daily, trade_date=td)
            if df is not None and len(df) > 0:
                n, t, mx = smart_append_csv(f'{BASE}/daily_prices.csv', df, ['ts_code', 'trade_date'])
                total_new += n
                print(f"    {td}: +{n}条")
            else:
                print(f"    {td}: 空")
    
    cnt = get_line_count(f'{BASE}/daily_prices.csv')
    results['daily'] = f'+{total_new}条 / 总计{cnt:,}条 / 最新{get_max_date_from_csv(f"{BASE}/daily_prices.csv", "trade_date")}'
    print(f"  ✅ {results['daily']}")
except Exception as e:
    results['daily'] = f'❌ {str(e)[:80]}'
    print(f"  ❌ {e}")

# ============================================================
# 2. daily_basic — 估值指标
# ============================================================
print("\n[2/6] daily_basic — 估值指标")
try:
    old_cnt = get_line_count(f'{BASE}/daily_basic.csv')
    max_d = get_max_date_from_csv(f'{BASE}/daily_basic.csv', 'trade_date') or '20200101'
    print(f"  存量: {old_cnt:,}条 | 最新: {max_d}")
    
    todo_dates = [d for d in cal_dates if d > max_d]
    total_new = 0
    
    if not todo_dates:
        print(f"  无新交易日")
    else:
        for td in todo_dates:
            df = safe_call(pro.daily_basic, trade_date=td)
            if df is not None and len(df) > 0:
                n, t, mx = smart_append_csv(f'{BASE}/daily_basic.csv', df, ['ts_code', 'trade_date'])
                total_new += n
                print(f"    {td}: +{n}条")
            else:
                print(f"    {td}: 空")
    
    cnt = get_line_count(f'{BASE}/daily_basic.csv')
    results['daily_basic'] = f'+{total_new}条 / 总计{cnt:,}条 / 最新{get_max_date_from_csv(f"{BASE}/daily_basic.csv", "trade_date")}'
    print(f"  ✅ {results['daily_basic']}")
except Exception as e:
    results['daily_basic'] = f'❌ {str(e)[:80]}'
    print(f"  ❌ {e}")

# ============================================================
# 3. stk_factor — 技术因子
# ============================================================
print("\n[3/6] stk_factor — 技术因子")
try:
    old_cnt = get_line_count(f'{BASE}/stk_factor.csv')
    max_d = get_max_date_from_csv(f'{BASE}/stk_factor.csv', 'trade_date') or '20200101'
    print(f"  存量: {old_cnt:,}条 | 最新: {max_d}")
    
    todo_dates = [d for d in cal_dates if d > max_d]
    total_new = 0
    
    if not todo_dates:
        print(f"  无新交易日")
    else:
        for td in todo_dates:
            df = safe_call(pro.stk_factor, trade_date=td)
            if df is not None and len(df) > 0:
                n, t, mx = smart_append_csv(f'{BASE}/stk_factor.csv', df, ['ts_code', 'trade_date'])
                total_new += n
                print(f"    {td}: +{n}条")
            else:
                print(f"    {td}: 空")
    
    cnt = get_line_count(f'{BASE}/stk_factor.csv')
    results['stk_factor'] = f'+{total_new}条 / 总计{cnt:,}条 / 最新{get_max_date_from_csv(f"{BASE}/stk_factor.csv", "trade_date")}'
    print(f"  ✅ {results['stk_factor']}")
except Exception as e:
    results['stk_factor'] = f'❌ {str(e)[:80]}'
    print(f"  ❌ {e}")

# ============================================================
# 4. moneyflow_hsgt — 北向资金
# ============================================================
print("\n[4/6] moneyflow_hsgt — 北向资金")
try:
    old_cnt = get_line_count(f'{BASE}/moneyflow_hsgt.csv')
    max_d = get_max_date_from_csv(f'{BASE}/moneyflow_hsgt.csv', 'trade_date') or '20140101'
    print(f"  存量: {old_cnt}条 | 最新: {max_d}")
    
    df = safe_call(pro.moneyflow_hsgt, start_date=max_d, end_date=TODAY)
    if df is not None and len(df) > 0:
        n, t, mx = smart_append_csv(f'{BASE}/moneyflow_hsgt.csv', df, ['trade_date'])
        cnt = get_line_count(f'{BASE}/moneyflow_hsgt.csv')
        results['moneyflow_hsgt'] = f'+{n}条 / 总计{cnt}条 / 最新{get_max_date_from_csv(f"{BASE}/moneyflow_hsgt.csv", "trade_date")}'
    else:
        results['moneyflow_hsgt'] = f'+0条 / 总计{old_cnt}条 / 最新{max_d}'
    
    print(f"  ✅ {results['moneyflow_hsgt']}")
except Exception as e:
    results['moneyflow_hsgt'] = f'❌ {str(e)[:80]}'
    print(f"  ❌ {e}")

# ============================================================
# 5. fina_indicator — 财务指标（逐股API，跳过批量更新）
# ============================================================
print("\n[5/6] fina_indicator — 财务指标")
old_cnt = get_line_count(f'{BASE}/fina_indicator.csv')
max_d = get_max_date_from_csv(f'{BASE}/fina_indicator.csv', 'ann_date') or 'N/A'
print(f"  存量: {old_cnt}条 | 最新ann_date: {max_d}")
print(f"  ⚠️ fina_indicator 要求 ts_code 参数，无法批量全市场更新")
print(f"  ℹ️  如需更新特定股票，使用: pro.fina_indicator(ts_code='XXX.SH', end_date='20260331')")
results['fina_indicator'] = f'跳过（逐股API）/ 存量{old_cnt}条 / 最新{max_d}'

# ============================================================
# 6. report_rc — 券商研报盈利预测 ⚠️ 重点：13个月缺口
# ============================================================
print("\n" + "=" * 70)
print("[6/6] report_rc — 券商研报 ⚠️ 13个月缺口 | 剩余9次/天")
print("=" * 70)

try:
    old_cnt = get_line_count(f'{BASE}/report_rc.csv')
    max_d = get_max_date_from_csv(f'{BASE}/report_rc.csv', 'report_date') or '20250701'
    print(f"  存量: {old_cnt}条 | 最新: {max_d}")
    
    # 13个月分9批，每批约1.4个月
    # 计算批次日期
    start_dt = datetime.strptime(max_d, '%Y%m%d') + timedelta(days=1)
    end_dt = datetime.strptime(TODAY, '%Y%m%d')
    total_days = (end_dt - start_dt).days
    batch_days = total_days // 9 + 1
    
    batches = []
    current = start_dt
    batch_num = 0
    while current <= end_dt and batch_num < 9:
        batch_end = min(current + timedelta(days=batch_days), end_dt)
        batches.append((current.strftime('%Y%m%d'), batch_end.strftime('%Y%m%d')))
        current = batch_end + timedelta(days=1)
        batch_num += 1
    
    print(f"  分{batch_num}批拉取，每批约{batch_days}天")
    for i, (s, e) in enumerate(batches):
        print(f"    批次{i+1}: {s} ~ {e}")
    print()
    
    total_new = 0
    rc_success = 0
    rc_fail = 0
    
    for i, (start, end) in enumerate(batches):
        batch_label = f"批次{i+1}/{len(batches)}"
        print(f"  {batch_label}: {start} ~ {end} ...", end=' ')
        
        df = safe_call(pro.report_rc, start_date=start, end_date=end)
        
        if df is not None and len(df) > 0:
            n, t, mx = smart_append_csv(f'{BASE}/report_rc.csv', df, ['ts_code', 'report_date', 'report_title', 'org_name'])
            total_new += n
            rc_success += 1
            print(f"+{n}条 (总计{t}条)")
        elif df is not None and len(df) == 0:
            print(f"0条（该时段无研报）")
            rc_success += 1
        else:
            print(f"❌ 失败")
            rc_fail += 1
            # 如果是频率限制，等待
            if i < len(batches) - 1:
                time.sleep(2)
    
    cnt = get_line_count(f'{BASE}/report_rc.csv')
    results['report_rc'] = f'+{total_new}条 / 总计{cnt}条 / {rc_success}/{len(batches)}批成功 / 最新{get_max_date_from_csv(f"{BASE}/report_rc.csv", "report_date")}'
    print(f"\n  ✅ {results['report_rc']}")
    
except Exception as e:
    results['report_rc'] = f'❌ {str(e)[:100]}'
    print(f"  ❌ {e}")

# ============================================================
# 最终汇总
# ============================================================
print("\n" + "=" * 70)
print(" Stage 0 更新完成 — 汇总")
print("=" * 70)
for name, status in results.items():
    print(f"  [{name}] {status}")
print("=" * 70)
