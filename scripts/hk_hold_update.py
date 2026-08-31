"""
北向个股持股 hk_hold 历史补采
================================
P-002 三路资金分歧升级依赖: 北向个股持股明细 (沪股通SH + 深股通SZ)

重要约束 (2026-08-14 实测):
  - Tushare hk_hold 北向(SH/SZ)明细自 2025 年起停更 (数据源停止披露),
    2025+ 仅剩 HK 港股通(南向)数据。
  - 可用窗口: 2017-03 ~ 2024-12-31 (2024-08-19 当日无北向数据)。
  - 因此本脚本默认补采 2023-01-01 ~ 2024-12-31 (北向完整2年窗口),
    2025+ 无法补采, P-002 三路分歧只能用 2024 年样本做历史验证。

用法:
  python scripts/hk_hold_update.py --start 20230101 --end 20241231   # 历史补采
  python scripts/hk_hold_update.py --start 20170301 --end 20241231   # 全量历史
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from credentials import get_tushare_api
import pandas as pd
import time
import os
from datetime import datetime

BASE = str(Path(__file__).resolve().parent.parent / 'data' / 'raw')
HK_HOLD_PATH = os.path.join(BASE, 'hk_hold.csv')
FLUSH = lambda: sys.stdout.flush()

START = '20230101'
END = '20241231'
for i, a in enumerate(sys.argv):
    if a == '--start' and i + 1 < len(sys.argv):
        START = sys.argv[i + 1]
    if a == '--end' and i + 1 < len(sys.argv):
        END = sys.argv[i + 1]


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
    except Exception:
        return None


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


def main():
    pro = get_tushare_api()
    old_max = get_max_date(HK_HOLD_PATH)
    print(f"[hk_hold] 存量最新: {old_max}", flush=True)

    # 交易日历
    cal = safe_call(pro.trade_cal, exchange='SSE', start_date=START, end_date=END, is_open='1')
    if cal is None or len(cal) == 0:
        print("❌ 交易日历获取失败", flush=True)
        return
    todo = sorted(cal['cal_date'].astype(str).unique())
    todo = [d for d in todo if old_max is None or d > old_max]
    print(f"  待补交易日: {len(todo)} 天 ({todo[0] if todo else '-'} → {todo[-1] if todo else '-'})", flush=True)

    total_new = 0
    fails = 0
    north_zero = 0
    for i, td in enumerate(todo):
        df = safe_call(pro.hk_hold, trade_date=td, limit=6000)
        if df is not None and len(df) > 0:
            # 只保留北向 (沪股通SH + 深股通SZ), 过滤港股通HK
            north = df[df['exchange'].isin(['SH', 'SZ'])]
            if len(north) > 0:
                north = north.copy()
                for c in ['trade_date', 'ts_code', 'code']:
                    if c in north.columns:
                        north[c] = north[c].astype(str)
                old_exists = os.path.exists(HK_HOLD_PATH) and os.path.getsize(HK_HOLD_PATH) > 100
                if old_exists:
                    existing = pd.read_csv(HK_HOLD_PATH, usecols=['ts_code', 'trade_date'], dtype=str, encoding='utf-8-sig')
                    merged = north.merge(existing, on=['ts_code', 'trade_date'], how='left', indicator=True)
                    new_only = merged[merged['_merge'] == 'left_only'].drop(columns=['_merge'])
                else:
                    new_only = north
                if len(new_only) > 0:
                    new_only.to_csv(HK_HOLD_PATH, mode='a', index=False, header=not old_exists, encoding='utf-8-sig')
                    total_new += len(new_only)
            else:
                north_zero += 1
        else:
            fails += 1
        if (i + 1) % 40 == 0:
            print(f"    {td}: 进度 {i+1}/{len(todo)} (+{total_new})", flush=True)

    new_max = get_max_date(HK_HOLD_PATH)
    print(f"  ✅ [hk_hold] +{total_new} 条 / 失败 {fails} 天 / 北向停更日 {north_zero} 天 / 最新 {new_max}", flush=True)


if __name__ == '__main__':
    main()
