"""
龙虎榜数据补采 + 每日增量 (top_list + top_inst)
================================================
P-026 落地依赖: 龙虎榜 top_list (每日明细) + top_inst (机构席位)

用法:
  python scripts/top_list_update.py --start 20240801    # 历史补采 (默认近2年)
  python scripts/top_list_update.py --days 5            # 每日增量: 补最近N个交易日
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from credentials import get_tushare_api
import pandas as pd
import time
import os
from datetime import datetime, timedelta

BASE = str(Path(__file__).resolve().parent.parent / 'data' / 'raw')
TOP_LIST_PATH = os.path.join(BASE, 'top_list.csv')
TOP_INST_PATH = os.path.join(BASE, 'top_inst.csv')
FLUSH = lambda: sys.stdout.flush()

DAYS = None
START = None
for i, a in enumerate(sys.argv):
    if a == '--days' and i + 1 < len(sys.argv):
        DAYS = int(sys.argv[i + 1])
    if a == '--start' and i + 1 < len(sys.argv):
        START = sys.argv[i + 1]


def normalize_date(val):
    val = str(val).strip()
    if not val or val.lower() == 'nan':
        return None
    if '.' in val:
        val = val.split('.')[0]
    val = val.replace('-', '')
    digits = ''.join(c for c in val if c.isdigit())
    return digits if len(digits) == 8 else None


def get_max_date(path):
    if not os.path.exists(path) or os.path.getsize(path) < 100:
        return None
    try:
        df = pd.read_csv(path, usecols=['trade_date'], dtype=str, encoding='utf-8-sig')
        return df['trade_date'].apply(normalize_date).max()
    except Exception as e:
        print(f"  ⚠️ 读取 {path} 失败: {e}", flush=True)
        return None


def smart_append(path, df_new, dedup_keys):
    """去重追加, 返回 (新增数, 总行数)"""
    if df_new is None or len(df_new) == 0:
        return 0, 0
    for k in dedup_keys:
        df_new[k] = df_new[k].astype(str)
    old_exists = os.path.exists(path) and os.path.getsize(path) > 100
    if old_exists:
        existing = pd.read_csv(path, usecols=dedup_keys, dtype=str, encoding='utf-8-sig')
        merged = df_new.merge(existing, on=dedup_keys, how='left', indicator=True)
        new_only = merged[merged['_merge'] == 'left_only'].drop(columns=['_merge'])
    else:
        new_only = df_new.copy()
    if len(new_only) == 0:
        return 0, sum(1 for _ in open(path, encoding='utf-8-sig')) - 1 if old_exists else 0
    write_header = not old_exists
    new_only.to_csv(path, mode='a', index=False, header=write_header, encoding='utf-8-sig')
    total = sum(1 for _ in open(path, encoding='utf-8-sig')) - 1
    return len(new_only), total


def safe_call(func, **kwargs):
    for attempt in range(3):
        try:
            r = func(**kwargs)
            time.sleep(0.35)
            return r
        except Exception as e:
            err = str(e)
            if any(k in err for k in ['频率', '限流', 'frequency', '超限']):
                print(f"    ⏳ 频率限制，等待 360 秒...", flush=True)
                time.sleep(360)
            elif attempt < 2:
                print(f"    ⚠️ 重试 {attempt+1}/3: {err[:80]}", flush=True)
                time.sleep(3)
            else:
                print(f"    ❌ 3次失败: {err[:100]}", flush=True)
                return None


def get_trade_days(pro, start, end):
    cal = safe_call(pro.trade_cal, exchange='SSE', start_date=start, end_date=end, is_open='1')
    if cal is None or len(cal) == 0:
        return []
    return sorted(cal['cal_date'].astype(str).unique())


def main():
    pro = get_tushare_api()
    today = datetime.now().strftime('%Y%m%d')

    for name, path in [('top_list', TOP_LIST_PATH), ('top_inst', TOP_INST_PATH)]:
        old_max = get_max_date(path)
        print(f"\n[{name}] 存量最新: {old_max}", flush=True)

        if DAYS:
            # 每日增量模式: 从存量最大日期后补最近 N 个交易日
            start = old_max or (datetime.now() - timedelta(days=60)).strftime('%Y%m%d')
            todo = get_trade_days(pro, start, today)
            todo = [d for d in todo if old_max is None or d > old_max][:DAYS]
        else:
            # 历史补采模式
            start = START or (datetime.now() - timedelta(days=730)).strftime('%Y%m%d')
            todo = get_trade_days(pro, start, today)
            todo = [d for d in todo if old_max is None or d > old_max]

        print(f"  待补交易日: {len(todo)} 天 ({todo[0] if todo else '-'} → {todo[-1] if todo else '-'})", flush=True)

        api = pro.top_list if name == 'top_list' else pro.top_inst
        total_new = 0
        fails = 0
        for i, td in enumerate(todo):
            df = safe_call(api, trade_date=td, limit=1000)
            if df is not None and len(df) > 0:
                n, t = smart_append(path, df, ['ts_code', 'trade_date', 'exalter'] if name == 'top_inst' else ['ts_code', 'trade_date', 'reason'])
                total_new += n
            else:
                fails += 1
            if (i + 1) % 50 == 0:
                print(f"    {td}: 进度 {i+1}/{len(todo)} (+{total_new})", flush=True)

        new_max = get_max_date(path)
        print(f"  ✅ [{name}] +{total_new} 条 / 失败 {fails} 天 / 最新 {new_max}", flush=True)


if __name__ == '__main__':
    main()
