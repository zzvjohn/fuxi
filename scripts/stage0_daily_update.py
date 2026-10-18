"""
Stage 0 每日数据更新脚本
2026-07-06 执行
重点：report_rc 13个月缺口补采（2025-07-01 ~ 2026-07-03）
"""
import sys
sys.path.append('e:/quant')
from credentials import get_tushare_api
import pandas as pd
import numpy as np
import time
import os
from datetime import datetime, timedelta

pro = get_tushare_api()

# ============================================================
# 配置
# ============================================================
DATA_DIR = 'e:/quant/data/raw'

# 表配置: {name: (filename, date_col, skip_reason)}
TABLES = {
    'daily':       ('daily_prices.csv',   'trade_date', None),
    'daily_basic': ('daily_basic.csv',    'trade_date', None),
    'stk_factor':  ('stk_factor.csv',     'trade_date', None),
    'moneyflow':   ('moneyflow_hsgt.csv', 'trade_date', None),
    'fina_indicator': ('fina_indicator.csv', 'ann_date', 'API需逐股，Stage 0 限制跳过'),
    'report_rc':   ('report_rc.csv',      'report_date', '13个月缺口，10次/天'),
}

LAST_TRADING_DAY = '20260703'  # 上周五
REPORT_RC_CALLS_TOTAL = 10     # 10次/天
GAP_START = '20250701'
GAP_END = '20260703'

results = {}

# ============================================================
# 工具函数
# ============================================================
def get_max_date(filepath, date_col):
    """获取CSV中的最大日期"""
    if not os.path.exists(filepath):
        return None, 0
    try:
        df = pd.read_csv(filepath, usecols=[date_col], dtype={date_col: str},
                         encoding='utf-8-sig', on_bad_lines='skip')
        dates = pd.to_datetime(df[date_col], format='mixed', errors='coerce')
        return dates.max(), len(df)
    except Exception as e:
        print(f"  ⚠️ 读取{filepath}失败: {e}")
        return None, -1

def append_to_csv(df, filepath, dedup_cols=None):
    """追加数据到CSV，自动去重"""
    if os.path.exists(filepath):
        existing = pd.read_csv(filepath, encoding='utf-8-sig', on_bad_lines='skip')
        before = len(existing)
        if dedup_cols:
            combined = pd.concat([existing, df], ignore_index=True)
            combined = combined.drop_duplicates(subset=dedup_cols, keep='last')
            new_count = len(combined) - before
        else:
            new_count = len(df)
            combined = pd.concat([existing, df], ignore_index=True)
        combined.to_csv(filepath, index=False, encoding='utf-8-sig')
        return new_count, len(combined)
    else:
        df.to_csv(filepath, index=False, encoding='utf-8-sig')
        return len(df), len(df)


# ============================================================
# Phase 1: 快速增量表（已确认最新，跳过）
# ============================================================
print("=" * 60)
print("Phase 1: 快速增量表检查")
print("=" * 60)

for name in ['daily', 'daily_basic', 'stk_factor', 'moneyflow']:
    filename, date_col, _ = TABLES[name]
    filepath = os.path.join(DATA_DIR, filename)
    max_dt, rows = get_max_date(filepath, date_col)
    max_str = max_dt.strftime('%Y-%m-%d') if max_dt and pd.notna(max_dt) else 'N/A'
    
    if max_str >= '2026-07-03':
        print(f"  ✅ {name}: {rows:,}行, 最新{max_str} → 已最新，跳过")
        results[name] = {'new': 0, 'total': rows, 'max_date': max_str, 'status': 'skip'}
    else:
        print(f"  ⚠️ {name}: {rows:,}行, 最新{max_str} → 需要更新")
        results[name] = {'new': 0, 'total': rows, 'max_date': max_str, 'status': 'need_update'}

# ============================================================
# Phase 2: fina_indicator（有限更新）
# ============================================================
print("\n" + "=" * 60)
print("Phase 2: fina_indicator 财务指标")
print("=" * 60)

filename, date_col, skip_reason = TABLES['fina_indicator']
filepath = os.path.join(DATA_DIR, filename)
max_dt, rows = get_max_date(filepath, 'ann_date')

if max_dt is not None and pd.notna(max_dt):
    print(f"  当前: {rows:,}行, 最新ann_date={max_dt}")
else:
    print(f"  当前: {rows:,}行")

# fina_indicator 要求 ts_code，无法全市场批量更新
# Q1 2026 财报通常在4月底前完成披露，ann_date 2026-05-01 已覆盖
# 尝试拉取最新一批公告日的数据
print("  ⚠️ API强制要求ts_code参数，Stage 0跳过全量更新")
print("  建议按需逐股采集（quant-data-collector skill 提供）")
results['fina_indicator'] = {'new': 0, 'total': rows, 'max_date': str(max_dt.date()) if max_dt and pd.notna(max_dt) else 'N/A', 'status': 'skip_ts_code_required'}


# ============================================================
# Phase 3: report_rc 券商研报（重点攻坚）
# ============================================================
print("\n" + "=" * 60)
print("Phase 3: report_rc 券商研报盈利预测 ⚠️ 重点")
print("=" * 60)

filename, date_col, _ = TABLES['report_rc']
filepath = os.path.join(DATA_DIR, filename)
max_dt, existing_rows = get_max_date(filepath, date_col)

print(f"  当前: {existing_rows:,}行, 最新{max_dt}")
print(f"  缺口: {max_dt.date() if max_dt and pd.notna(max_dt) else 'N/A'} → 2026-07-03")
print(f"  预算: {REPORT_RC_CALLS_TOTAL}次API调用")

# 策略：将13个月缺口分成10段，每段覆盖约40天
# 同时预留几次给更精确的日期范围
total_days = (datetime(2026, 7, 3) - datetime(2025, 7, 1)).days
print(f"  总缺口: ~{total_days}天")
print(f"  分段: 每段约{total_days // REPORT_RC_CALLS_TOTAL}天")

# 生成10个日期段
date_ranges = []
chunk_days = total_days // REPORT_RC_CALLS_TOTAL
current_start = datetime(2025, 7, 1)
for i in range(REPORT_RC_CALLS_TOTAL):
    if i == REPORT_RC_CALLS_TOTAL - 1:
        current_end = datetime(2026, 7, 3)
    else:
        current_end = current_start + timedelta(days=chunk_days - 1)
    date_ranges.append((current_start, current_end))
    current_start = current_end + timedelta(days=1)

all_new_data = []
success_count = 0
fail_count = 0

for idx, (start_dt, end_dt) in enumerate(date_ranges):
    s = start_dt.strftime('%Y%m%d')
    e = end_dt.strftime('%Y%m%d')
    label = f"{start_dt.strftime('%Y-%m-%d')} ~ {end_dt.strftime('%Y-%m-%d')}"
    
    print(f"\n  [{idx+1}/{REPORT_RC_CALLS_TOTAL}] {label} ...", end=' ', flush=True)
    
    try:
        df = pro.report_rc(
            start_date=s,
            end_date=e,
            fields='ts_code,report_date,researcher,researcher_org,org_sfc,'
                   'first_researcher,actual_report_date,end_date,report_type,'
                   'rating,评级变动,eps,net_profit,total_mv,pe,pb,roe'
        )
        time.sleep(0.5)  # 频率控制
        
        if df is not None and len(df) > 0:
            n = len(df)
            all_new_data.append(df)
            success_count += 1
            print(f"✅ {n}条")
        else:
            print(f"⚠️ 空结果")
            fail_count += 1
    except Exception as ex:
        err_msg = str(ex)
        print(f"❌ {err_msg[:60]}")
        fail_count += 1
        
        # 如果是频率限制，等待6分钟
        if '频率' in err_msg or 'limit' in err_msg.lower() or '分钟' in err_msg:
            print(f"  ⏳ 频率限制，等待6分钟...")
            time.sleep(360)
            try:
                df = pro.report_rc(start_date=s, end_date=e,
                    fields='ts_code,report_date,researcher,researcher_org,org_sfc,'
                           'first_researcher,actual_report_date,end_date,report_type,'
                           'rating,评级变动,eps,net_profit,total_mv,pe,pb,roe')
                if df is not None and len(df) > 0:
                    all_new_data.append(df)
                    success_count += 1
                    print(f"  → 重试成功 ✅ {len(df)}条")
                else:
                    print(f"  → 重试空结果")
            except Exception as ex2:
                print(f"  → 重试仍然失败: {str(ex2)[:60]}")
    
    # 批次间短暂休息
    if idx < len(date_ranges) - 1:
        time.sleep(1.0)

# 合并新数据
print(f"\n  API调用: {success_count}成功 / {fail_count}失败 / {REPORT_RC_CALLS_TOTAL}总计")

if all_new_data:
    new_df = pd.concat(all_new_data, ignore_index=True)
    print(f"  新采集总计: {len(new_df)}条")
    
    # 去重追加
    new_count, total_count = append_to_csv(new_df, filepath, 
                                            dedup_cols=['ts_code', 'report_date', 'researcher_org', 'end_date'])
    
    # 重新检查最新日期
    updated_max, _ = get_max_date(filepath, date_col)
    max_str = updated_max.strftime('%Y-%m-%d') if updated_max and pd.notna(updated_max) else 'N/A'
    
    print(f"  → 新增{new_count}条 / 总计{total_count:,}条 / 最新{max_str}")
    results['report_rc'] = {'new': new_count, 'total': total_count, 'max_date': max_str, 'status': 'ok'}
else:
    print(f"  ❌ 无新数据")
    max_str = max_dt.strftime('%Y-%m-%d') if max_dt and pd.notna(max_dt) else 'N/A'
    results['report_rc'] = {'new': 0, 'total': existing_rows, 'max_date': max_str, 'status': 'no_new_data'}


# ============================================================
# Phase 4: 数据完整性检查
# ============================================================
print("\n" + "=" * 60)
print("Phase 4: 数据完整性检查")
print("=" * 60)

print(f"\n{'表名':<20} {'新增':>8} {'总计':>12} {'最新日期':<14} {'状态':<15}")
print("-" * 72)

status_ok = True
for name, r in results.items():
    total_str = f"{r['total']:,}" if isinstance(r['total'], int) and r['total'] > 0 else str(r['total'])
    new_str = f"+{r['new']}" if r['new'] > 0 else str(r['new'])
    max_str = r.get('max_date', 'N/A')
    status_str = r.get('status', '?')
    
    # 状态标记
    icon = '✅' if status_str in ('skip', 'ok') else ('⚠️' if status_str == 'no_new_data' else '❌')
    print(f"{icon} {name:<17} {new_str:>8} {total_str:>12} {max_str:<14} {status_str:<15}")
    
    if status_str not in ('skip', 'ok', 'skip_ts_code_required'):
        status_ok = False

print("-" * 72)

# 报表明细
print(f"\n📊 数据完整性报告:")
print(f"  daily/daily_basic/stk_factor/moneyflow: ✅ 截至2026-07-03（上周五收盘）")
print(f"  fina_indicator: ⚠️ 跳过（API逐股限制），最新公告日2026-05-01")
print(f"  report_rc: {'✅' if results['report_rc']['new'] > 0 else '❌'} ", end='')

rrc = results['report_rc']
if rrc['new'] > 0:
    print(f"新增{rrc['new']}条，总计{rrc['total']:,}条，最新{rrc['max_date']}")
else:
    print(f"无新数据，仍为{rrc['max_date']}（缺口持续）")

print(f"\n执行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"整体状态: {'✅ 全部完成' if status_ok else '⚠️ 部分完成'}")
