"""
Stage 0 每日数据更新 v3
======================
v3 修复:
  - .meta 缓存: 避免 13M+ 行全量扫描（~20s → 0.01s）
  - 日期格式归一化: 处理 YYYY-MM-DD / YYYYMMDD / YYYYMMDD.0 混用
  - 严格列校验: pandas 只读日期列，不依赖 csv.reader 全量遍历
  - report_rc 分离: 独立运行，用 --report-rc 标志触发
  - 自动 flush: 每步输出后强制刷新

用法:
  python -u scripts/stage0_daily_update_v3.py              # 5张快速表
  python -u scripts/stage0_daily_update_v3.py --report-rc  # 包含 report_rc (请确认日额度充足!)
  python -u scripts/stage0_daily_update_v3.py --rebuild-meta  # 重建元数据缓存
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from credentials import get_tushare_api
import pandas as pd
import numpy as np
import time
import os
import json
import csv
import re
from datetime import datetime, timedelta

# ============================================================
# 配置
# ============================================================
BASE = str(Path(__file__).resolve().parent.parent / 'data' / 'raw')
META_PATH = os.path.join(BASE, '.meta.json')
# 动态取当天日期（修复原硬编码 20260705 导致无法拉取新交易日的 bug）
TODAY = datetime.now().strftime('%Y%m%d')
REPORT_RC_ENABLED = '--report-rc' in sys.argv
REBUILD_META = '--rebuild-meta' in sys.argv
FLUSH = lambda: sys.stdout.flush()

# ============================================================
# .meta 缓存系统
# ============================================================

def load_meta():
    """加载元数据缓存"""
    if os.path.exists(META_PATH):
        with open(META_PATH, 'r') as f:
            return json.load(f)
    return {}

def save_meta(meta):
    """保存元数据缓存"""
    os.makedirs(os.path.dirname(META_PATH), exist_ok=True)
    with open(META_PATH, 'w') as f:
        json.dump(meta, f, indent=2)

def normalize_date(val):
    """将各种日期格式统一为 YYYYMMDD 字符串"""
    val = str(val).strip()
    if not val or val.lower() == 'nan':
        return None
    # 20260618.0 → 20260618
    if '.' in val:
        val = val.split('.')[0]
    # 2021-01-04 → 20210104
    val = val.replace('-', '')
    # 只保留数字
    digits = ''.join(c for c in val if c.isdigit())
    if len(digits) == 8:
        return digits
    return None

def get_max_date_from_csv_fast(path, date_col):
    """
    高效获取 CSV 中 date_col 的最大值（只读日期列）
    返回: (max_date_YYYYMMDD, total_rows, is_cached)
    """
    if not os.path.exists(path) or os.path.getsize(path) < 100:
        return None, 0, False
    
    # 只读两列: ts_code + date_col，大幅减少内存
    cols_to_read = [date_col]
    # 先读 header 确认列存在
    try:
        header = pd.read_csv(path, nrows=0, encoding='utf-8-sig').columns.tolist()
        if date_col not in header:
            # 可能列名不同，检查常见替代
            for alt in ['trade_date', 'ann_date', 'report_date']:
                if alt in header:
                    date_col = alt
                    break
            else:
                return None, 0, False
        
        # 只读日期列（用 chunk 避免内存爆炸）
        total = 0
        max_val = None
        chunks = pd.read_csv(
            path, usecols=[date_col], encoding='utf-8-sig',
            chunksize=200000, dtype=str
        )
        for chunk in chunks:
            total += len(chunk)
            series = chunk[date_col].dropna()
            if len(series) == 0:
                continue
            # 归一化所有日期
            normalized = series.apply(normalize_date).dropna()
            if len(normalized) > 0:
                chunk_max = normalized.max()
                if max_val is None or chunk_max > max_val:
                    max_val = chunk_max
        
        return max_val, total, False
    except Exception as e:
        print(f"    ⚠️ 日期扫描失败: {e}", flush=True)
        return None, 0, False

def get_table_meta(table_name, csv_path, date_col):
    """
    获取表的最新日期和行数（优先从缓存读）
    返回: (max_date_str, row_count)
    """
    meta = load_meta()
    
    if not REBUILD_META and table_name in meta:
        cached = meta[table_name]
        file_mtime = os.path.getmtime(csv_path) if os.path.exists(csv_path) else 0
        if cached.get('mtime') == file_mtime and cached.get('max_date'):
            # 缓存命中 + 文件未变
            return cached['max_date'], cached.get('row_count', 0)
    
    # 缓存未命中，扫描文件
    max_date, row_count, _ = get_max_date_from_csv_fast(csv_path, date_col)
    
    # 更新缓存
    meta[table_name] = {
        'max_date': max_date,
        'row_count': row_count,
        'mtime': os.path.getmtime(csv_path) if os.path.exists(csv_path) else 0,
        'updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    save_meta(meta)
    
    return max_date, row_count

def update_table_meta(table_name, csv_path, new_max_date, new_row_count):
    """采集完成后更新缓存"""
    meta = load_meta()
    meta[table_name] = {
        'max_date': new_max_date,
        'row_count': new_row_count,
        'mtime': os.path.getmtime(csv_path) if os.path.exists(csv_path) else 0,
        'updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    save_meta(meta)

def get_line_count(path):
    """快速行数统计"""
    if not os.path.exists(path):
        return 0
    count = 0
    with open(path, 'r', encoding='utf-8-sig') as f:
        next(f)
        for _ in f:
            count += 1
    return count

# ============================================================
# 增量更新逻辑
# ============================================================

def smart_append_csv(path, df_new, dedup_keys):
    """
    智能追加到 CSV: 去重后只写新行
    返回: (新增数, 总行数, 最新日期str)
    """
    if df_new is None or len(df_new) == 0:
        return 0, get_line_count(path), None
    
    old_count = get_line_count(path)
    
    # 统一类型
    for k in dedup_keys:
        if k in df_new.columns:
            df_new[k] = df_new[k].astype(str)
    
    if old_count > 0:
        existing = pd.read_csv(path, usecols=dedup_keys, encoding='utf-8-sig', dtype=str)
        merged = df_new.merge(existing, on=dedup_keys, how='left', indicator=True)
        new_only = merged[merged['_merge'] == 'left_only'].drop(columns=['_merge'])
    else:
        new_only = df_new.copy()
    
    if len(new_only) == 0:
        return 0, old_count, None

    # v0.9.1 (2026-08-19): 追加前列对齐 — 防止 API 返回字段漂移污染存量文件
    # 事故实证: margin_detail 202608 增量 Tushare 返回 10 列 (多 rzche/rqmcl/rzrqye),
    # 直接 append 到 7 列文件尾部 → 13290 行字段错位, 后续 read_csv 全表加载失败。
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
    date_col = dedup_keys[1] if len(dedup_keys) > 1 else dedup_keys[0]
    max_date_raw = str(new_only[date_col].max()) if date_col in new_only.columns else None
    max_date = normalize_date(max_date_raw)
    
    return new_count, total, max_date

def _safe_func_name(func):
    """安全获取函数名（兼容 functools.partial / bound method）"""
    name = getattr(func, '__name__', '')
    if not name:
        name = getattr(getattr(func, 'func', None), '__name__', '')
    return name

def safe_call(func, **kwargs):
    """API 包装: 3次重试 + 频率限制检测"""
    fname = _safe_func_name(func)
    for attempt in range(3):
        try:
            result = func(**kwargs)
            time.sleep(0.35)
            return result
        except Exception as e:
            err = str(e)
            if any(kw in err for kw in ['频率', '限流', 'frequency', '超限']):
                if 'report_rc' in fname and '天' in err:
                    print(f"    🔴 日额度已耗尽 (report_rc 10次/天)，跳过后续批次", flush=True)
                    raise SystemExit(0)  # 不再浪费等待时间
                print(f"    ⏳ 频率限制，等待 360 秒(6分钟)...", flush=True)
                time.sleep(360)
            elif attempt < 2:
                print(f"    ⚠️ 重试 {attempt+1}/3: {err[:80]}", flush=True)
                time.sleep(3)
            else:
                print(f"    ❌ 3次失败: {err[:100]}", flush=True)
                return None

# ============================================================
# 主流程
# ============================================================

print("=" * 70, flush=True)
print(f" Stage 0 每日数据更新 v3", flush=True)
print(f" 执行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
print(f" report_rc: {'启用' if REPORT_RC_ENABLED else '跳过 (加 --report-rc 启用)'}", flush=True)
print(f" 重建缓存: {'是' if REBUILD_META else '否'}", flush=True)
print("=" * 70, flush=True)

pro = get_tushare_api()
time.sleep(0.3)

results = {}

# 交易日历
cal = safe_call(pro.trade_cal, exchange='SSE', start_date='20260701', end_date=TODAY, is_open='1')
cal_dates = sorted(cal['cal_date'].astype(str).unique()) if cal is not None else []
print(f"\n📅 近期交易日: {cal_dates}", flush=True)

# ============================================================
# Phase 1: 5 张快速表
# ============================================================

tables = [
    ('daily_prices', 'data/raw/daily_prices.csv', 'trade_date',
     lambda trade_date: pro.daily(trade_date=trade_date)),
    ('daily_basic', 'data/raw/daily_basic.csv', 'trade_date',
     lambda trade_date: pro.daily_basic(trade_date=trade_date)),
    ('stk_factor', 'data/raw/stk_factor.csv', 'trade_date',
     lambda trade_date: pro.stk_factor(trade_date=trade_date)),
]
labels = ['日线行情', '估值指标', '技术因子']

for (name, path, date_col, fetch_fn), label in zip(tables, labels):
    print(f"\n[{name}] {label}", flush=True)
    try:
        old_max, old_cnt = get_table_meta(name, path, date_col)
        print(f"  存量: {old_cnt:>12,} 行 | 最新: {old_max}", flush=True)
        
        todo_dates = [d for d in cal_dates if old_max is None or d > old_max]
        
        if not todo_dates:
            print(f"  ✅ 无需更新（最新交易日={cal_dates[-1] if cal_dates else 'N/A'}）", flush=True)
            results[name] = f"✅ +0 / {old_cnt:,}行 / {old_max}"
            continue
        
        total_new = 0
        for td in todo_dates:
            df = safe_call(fetch_fn, trade_date=td)
            if df is not None and len(df) > 0:
                n, t, mx = smart_append_csv(path, df, ['ts_code', date_col])
                total_new += n
                print(f"    {td}: +{n} 条", flush=True)
            else:
                print(f"    {td}: 无数据", flush=True)
        
        # 更新缓存
        final_max, final_cnt = get_table_meta(name, path, date_col)
        # force refresh
        if REBUILD_META or total_new > 0:
            final_max, final_cnt = get_max_date_from_csv_fast(path, date_col)[:2]
            update_table_meta(name, path, final_max, final_cnt)
        
        results[name] = f"✅ +{total_new} / {final_cnt:,}行 / {final_max}"
        print(f"  {results[name]}", flush=True)
        
    except Exception as e:
        results[name] = f"❌ {str(e)[:80]}"
        print(f"  ❌ {e}", flush=True)

# --- moneyflow_hsgt ---
name, path, date_col = 'moneyflow_hsgt', 'data/raw/moneyflow_hsgt.csv', 'trade_date'
print(f"\n[{name}] 北向资金", flush=True)
try:
    old_max, old_cnt = get_table_meta(name, path, date_col)
    if old_max is None:
        old_max = '20140101'
    print(f"  存量: {old_cnt} 行 | 最新: {old_max}", flush=True)
    
    df = safe_call(pro.moneyflow_hsgt, start_date=old_max, end_date=TODAY)
    if df is not None and len(df) > 0:
        n, t, mx = smart_append_csv(path, df, [date_col])
        update_table_meta(name, path, mx or old_max, t)
        results[name] = f"✅ +{n} / {t}行 / {mx or old_max}"
    else:
        results[name] = f"✅ +0 / {old_cnt}行 / {old_max}"
    print(f"  {results[name]}", flush=True)
except Exception as e:
    results[name] = f"❌ {str(e)[:80]}"
    print(f"  ❌ {e}", flush=True)

# ============================================================
# Phase 1.5: 龙虎榜 + 两融 + 北向持股 (2026-08-14 新增, P-026/P-001/P-002 落地)
# ============================================================

def update_daily_by_trade_date(name, path, fetch_fn, dedup_keys, label, max_missing_days=5):
    """通用逐日增量: 补存量最大日期后的最近 N 个交易日"""
    old_max, old_cnt = get_table_meta(name, path, 'trade_date')
    print(f"  存量: {old_cnt:>12,} 行 | 最新: {old_max}", flush=True)
    todo_dates = [d for d in cal_dates if old_max is None or d > old_max][:max_missing_days]
    if not todo_dates:
        print(f"  ✅ 无需更新", flush=True)
        results[name] = f"✅ +0 / {old_cnt:,}行 / {old_max}"
        return
    total_new = 0
    for td in todo_dates:
        df = safe_call(fetch_fn, trade_date=td)
        if df is not None and len(df) > 0:
            n, t, mx = smart_append_csv(path, df, dedup_keys)
            total_new += n
            print(f"    {td}: +{n} 条", flush=True)
        else:
            print(f"    {td}: 无数据", flush=True)
    if total_new > 0:
        final_max, final_cnt = get_max_date_from_csv_fast(path, 'trade_date')[:2]
        update_table_meta(name, path, final_max, final_cnt)
    else:
        final_max, final_cnt = old_max, old_cnt
    results[name] = f"✅ +{total_new} / {final_cnt:,}行 / {final_max}"
    print(f"  {results[name]}", flush=True)

# --- top_list 龙虎榜每日明细 ---
print(f"\n[top_list] 龙虎榜明细", flush=True)
try:
    update_daily_by_trade_date(
        'top_list', 'data/raw/top_list.csv',
        lambda trade_date: pro.top_list(trade_date=trade_date, limit=1000),
        ['ts_code', 'trade_date', 'reason'], '龙虎榜')
except Exception as e:
    results['top_list'] = f"❌ {str(e)[:80]}"
    print(f"  ❌ {e}", flush=True)

# --- top_inst 龙虎榜机构席位 ---
print(f"\n[top_inst] 龙虎榜机构席位", flush=True)
try:
    update_daily_by_trade_date(
        'top_inst', 'data/raw/top_inst.csv',
        lambda trade_date: pro.top_inst(trade_date=trade_date, limit=2000),
        ['ts_code', 'trade_date', 'exalter', 'net_buy'], '机构席位')
except Exception as e:
    results['top_inst'] = f"❌ {str(e)[:80]}"
    print(f"  ❌ {e}", flush=True)

# --- margin_detail 两融明细 ---
print(f"\n[margin_detail] 两融明细", flush=True)
try:
    update_daily_by_trade_date(
        'margin_detail', 'data/raw/margin_detail.csv',
        lambda trade_date: pro.margin_detail(trade_date=trade_date),
        ['ts_code', 'trade_date'], '两融')
except Exception as e:
    results['margin_detail'] = f"❌ {str(e)[:80]}"
    print(f"  ❌ {e}", flush=True)

# --- hk_hold 北向持股 (已停更, 不做每日增量) ---
print(f"\n[hk_hold] 北向个股持股", flush=True)
hk_old_max, hk_old_cnt = get_table_meta('hk_hold', 'data/raw/hk_hold.csv', 'trade_date')
print(f"  存量: {hk_old_cnt:>12,} 行 | 最新: {hk_old_max}", flush=True)
print(f"  ⚠️ 北向(SH/SZ)明细自 2025 年起停更 (数据源停止披露), 无每日增量可补", flush=True)
print(f"  如需补历史: python scripts/hk_hold_update.py --start 20230101 --end 20241231", flush=True)
results['hk_hold'] = f"⚠️ 北向停更 / {hk_old_cnt:,}行 / {hk_old_max}"

# --- fina_indicator ---
print(f"\n[fina_indicator] 财务指标", flush=True)
old_max, old_cnt = get_table_meta('fina_indicator', 'data/raw/fina_indicator.csv', 'ann_date')
print(f"  存量: {old_cnt} 行 | 最新 ann_date: {old_max}", flush=True)
print(f"  ⚠️ API 要求 ts_code 逐股，Stage 0 不批量采集", flush=True)
results['fina_indicator'] = f"⚠️ 跳过 / {old_cnt}行 / {old_max}"

# ============================================================
# Phase 2: report_rc (仅当 --report-rc 启用)
# ============================================================

if REPORT_RC_ENABLED:
    print("\n" + "=" * 70, flush=True)
    print(" [report_rc] 券商研报盈利预测", flush=True)
    print("=" * 70, flush=True)
    
    name, path = 'report_rc', 'data/raw/report_rc.csv'
    try:
        old_max, old_cnt = get_table_meta(name, path, 'report_date')
        if old_max is None:
            old_max = '20250701'
        print(f"  存量: {old_cnt} 行 | 最新: {old_max}", flush=True)
        
        # 计算批次（每批 ~1.5 月，最多 9 批）
        start_dt = datetime.strptime(old_max, '%Y%m%d') + timedelta(days=1)
        end_dt = datetime.strptime(TODAY, '%Y%m%d')
        total_days = (end_dt - start_dt).days
        batch_count = min(9, max(1, total_days // 40 + 1))
        batch_days = total_days // batch_count + 1
        
        batches = []
        current = start_dt
        for _ in range(batch_count):
            batch_end = min(current + timedelta(days=batch_days), end_dt)
            if current <= end_dt:
                batches.append((current.strftime('%Y%m%d'), batch_end.strftime('%Y%m%d')))
            current = batch_end + timedelta(days=1)
        
        print(f"  分 {len(batches)} 批拉取（report_rc 限制 10次/小时，每批间隔6分钟）", flush=True)
        for i, (s, e) in enumerate(batches):
            print(f"    批次 {i+1}/{len(batches)}: {s} ~ {e}", flush=True)
        print(flush=True)
        
        total_new = 0
        success = 0
        
        for i, (start, end) in enumerate(batches):
            label = f"批次 {i+1}/{len(batches)}"
            print(f"  {label}: {start} ~ {end} ...", end=' ', flush=True)

            # 批次间隔 6 分钟，遵守 report_rc 10次/小时 限制（首个批次前也等，避免紧接上次额度）
            if i > 0:
                print(f"    ⏳ 批次间隔等待 360 秒(6分钟)...", flush=True)
                time.sleep(360)

            df = safe_call(pro.report_rc, start_date=start, end_date=end)

            if df is not None and len(df) > 0:
                n, t, mx = smart_append_csv(path, df, ['ts_code', 'report_date', 'report_title', 'org_name'])
                total_new += n
                success += 1
                print(f"+{n} 条 (总计 {t} 条)", flush=True)
            elif df is not None:
                print(f"0 条", flush=True)
                success += 1
            else:
                # 调用失败（通常额度耗尽）：本小时已无额度，保存进度并结束，留给下一小时/次日
                print(f"❌ 调用失败，疑似本小时额度耗尽，保存进度后退出", flush=True)
                break
        
        final_max, final_cnt = get_table_meta(name, path, 'report_date')
        if total_new > 0:
            final_max_scan, final_cnt_scan = get_max_date_from_csv_fast(path, 'report_date')[:2]
            update_table_meta(name, path, final_max_scan, final_cnt_scan)
        results[name] = f"✅ +{total_new} / {final_cnt:,}行 / {success}/{len(batches)}批成功 / {final_max}"
        print(f"\n  {results[name]}", flush=True)
        
    except SystemExit:
        results[name] = f"🔴 日额度耗尽 / {old_cnt}行 / {old_max}"
        print(f"  {results[name]}", flush=True)
    except Exception as e:
        results[name] = f"❌ {str(e)[:100]}"
        print(f"  ❌ {e}", flush=True)
else:
    old_max, old_cnt = get_table_meta('report_rc', 'data/raw/report_rc.csv', 'report_date')
    results['report_rc'] = f"⏭️ 跳过 / {old_cnt}行 / {old_max} (用 --report-rc 启用)"

# ============================================================
# 最终汇总
# ============================================================
print("\n" + "=" * 70, flush=True)
print(" Stage 0 更新完成 — 汇总", flush=True)
print("=" * 70, flush=True)
for name, status in results.items():
    print(f"  [{name}] {status}", flush=True)

# 保存最终元数据
meta = load_meta()
meta['last_run'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
meta['last_success'] = all('❌' not in v and '🔴' not in v for v in results.values())
save_meta(meta)
print(f"\n📁 元数据已保存: {META_PATH}", flush=True)
print("=" * 70, flush=True)
