"""
Stage 0 补采 — 剩余3表 (fina_indicator, report_rc, moneyflow_hsgt)
- stk_factor: 已最新跳过
- daily_basic: 已完成(需后续dedup)
"""
import sys
sys.path.append('e:/quant')
from credentials import get_tushare_api
import pandas as pd
import time
import os
from datetime import datetime, timedelta

pro = get_tushare_api()
DATA_DIR = 'e:/quant/data/raw'
TODAY = datetime.now().strftime('%Y%m%d')

results = []

def get_last_date(filepath, date_col):
    if not os.path.exists(filepath) or os.path.getsize(filepath) < 100:
        return None
    try:
        df = pd.read_csv(filepath, usecols=[date_col], low_memory=False)
        vals = pd.to_numeric(df[date_col], errors='coerce').dropna()
        if len(vals) == 0:
            return None
        return int(vals.max())
    except:
        return None

def stream_append(filepath, df_new):
    if len(df_new) == 0:
        return 0
    if os.path.exists(filepath) and os.path.getsize(filepath) > 100:
        header = pd.read_csv(filepath, nrows=0).columns.tolist()
        new_cols = df_new.columns.tolist()
        if header == new_cols:
            # Get "before" count from just reading first column
            before = len(pd.read_csv(filepath, usecols=[header[0]], low_memory=False))
            df_new.to_csv(filepath, mode='a', index=False, header=False, encoding='utf-8-sig')
            return len(df_new)
        else:
            print(f"  ⚠️ 列不匹配，填充NaN后追加", flush=True)
            for c in header:
                if c not in new_cols:
                    df_new[c] = None
            df_new = df_new[header]
            before = len(pd.read_csv(filepath, usecols=[header[0]], low_memory=False))
            df_new.to_csv(filepath, mode='a', index=False, header=False, encoding='utf-8-sig')
            return len(df_new)
    else:
        df_new.to_csv(filepath, index=False, encoding='utf-8-sig')
        return len(df_new)

def fetch_retry(func, **kwargs):
    for attempt in range(3):
        try:
            return func(**kwargs)
        except Exception as e:
            msg = str(e)
            if any(k in msg for k in ['频率','QPS','429','limit']):
                print(f"  等6分钟...", flush=True)
                time.sleep(360)
                continue
            if attempt < 2:
                time.sleep((attempt+1)*3)
    return None

# ============================================================
# 4. fina_indicator
# ============================================================
print("="*60)
print("[4] fina_indicator - 财务指标")
fi_path = f'{DATA_DIR}/fina_indicator.csv'
fi_last = get_last_date(fi_path, 'ann_date')
fi_start = '20260501'  # 补最后一个季度
if fi_last and fi_last > 20260430:
    fi_start = str(fi_last + 1)
    if len(fi_start) == 7: fi_start = '0' + fi_start

print(f"  本地: {fi_last}, 拉取: {fi_start}~{TODAY}")

all_new = []
current = datetime.strptime(fi_start, '%Y%m%d')
end = datetime.strptime(TODAY, '%Y%m%d')
while current <= end:
    batch_end = current + timedelta(days=30)
    if batch_end > end: batch_end = end
    cs, ce = current.strftime('%Y%m%d'), batch_end.strftime('%Y%m%d')
    print(f"  {cs}~{ce} ...", end=' ', flush=True)
    df = fetch_retry(pro.fina_indicator,
                     start_date=cs, end_date=ce,
                     fields='ts_code,ann_date,end_date,roe,roe_dt,roa,roa_dt,'
                            'netprofit_yoy,or_yoy,grossprofit_margin,netprofit_margin,'
                            'debt_to_assets,eps')
    if df is not None and len(df) > 0:
        all_new.append(df)
        print(f"{len(df)}条", flush=True)
    else:
        print("0条", flush=True)
    time.sleep(0.5)
    current = batch_end + timedelta(days=1)

if all_new:
    df_all = pd.concat(all_new, ignore_index=True)
    added = stream_append(fi_path, df_all)
    total = len(pd.read_csv(fi_path, usecols=[0], low_memory=False))
    latest = df_all['ann_date'].max()
    print(f"  ✅ fina_indicator: +{added}条 -> {total:,}条 / ann_date={latest}")
    results.append(('fina_indicator', added, total, latest))
else:
    total = len(pd.read_csv(fi_path, usecols=[0], low_memory=False))
    print(f"  ✅ fina_indicator: 无新数据, {total:,}条")
    results.append(('fina_indicator', 0, total, fi_last))

time.sleep(1)

# ============================================================
# 5. report_rc (10次/h限制)
# ============================================================
print("\n"+"="*60)
print("[5] report_rc - 券商研报 ⚠️ 重点")
rc_path = f'{DATA_DIR}/report_rc.csv'
rc_last = get_last_date(rc_path, 'report_date')
rc_start = '20250701'  # 本地最新20250630
if rc_last and rc_last > 20250630:
    rc_start = str(rc_last + 1)
    if len(rc_start) == 7: rc_start = '0' + rc_start

print(f"  本地最新: {rc_last}, 拉取: {rc_start}~{TODAY} (最多10批)")

rc_new = []
current = datetime.strptime(rc_start, '%Y%m%d')
end = datetime.strptime(TODAY, '%Y%m%d')
batch = 0
while current <= end and batch < 10:
    batch += 1
    batch_end = current + timedelta(days=6)
    if batch_end > end: batch_end = end
    cs, ce = current.strftime('%Y%m%d'), batch_end.strftime('%Y%m%d')
    print(f"  [{batch}/10] {cs}~{ce} ...", end=' ', flush=True)
    df = fetch_retry(pro.report_rc,
                     start_date=cs, end_date=ce,
                     fields='ts_code,name,report_date,report_title,report_type,classify,'
                            'org_name,author_name,rating,rating_change,eps,net_profit,'
                            'total_mv,pe,pb,roe,report_year,end_date')
    if df is not None and len(df) > 0:
        rc_new.append(df)
        print(f"{len(df)}条", flush=True)
        time.sleep(0.5)
    else:
        print("0条", flush=True)
        time.sleep(1)
    current = batch_end + timedelta(days=1)

if batch == 10 and current <= end:
    print(f"  ⚠️ 达10批上限, 剩余{current.strftime('%Y%m%d')}~{TODAY} 下次继续")

if rc_new:
    df_all = pd.concat(rc_new, ignore_index=True)
    # report_rc可能有不同列的版本, 直接用pandas自动合并
    if os.path.exists(rc_path) and os.path.getsize(rc_path) > 100:
        existing = pd.read_csv(rc_path, low_memory=False)
        all_cols = list(dict.fromkeys(list(existing.columns) + list(df_all.columns)))
        # 规范化列
        for c in all_cols:
            if c not in existing.columns: existing[c] = None
            if c not in df_all.columns: df_all[c] = None
        existing = existing[all_cols]
        df_all = df_all[all_cols]
        combined = pd.concat([existing, df_all], ignore_index=True)
        combined.to_csv(rc_path, index=False, encoding='utf-8-sig')
        total = len(combined)
        added = total - len(existing)
    else:
        df_all.to_csv(rc_path, index=False, encoding='utf-8-sig')
        total = len(df_all)
        added = total
    latest = df_all['report_date'].max() if 'report_date' in df_all.columns else rc_last
    print(f"  ✅ report_rc: +{added}条 -> {total:,}条 / 最新{latest}")
    results.append(('report_rc', added, total, latest))
else:
    total = len(pd.read_csv(rc_path, usecols=[0], low_memory=False))
    print(f"  ✅ report_rc: 无新数据, {total:,}条")
    results.append(('report_rc', 0, total, rc_last))

time.sleep(1)

# ============================================================
# 6. moneyflow_hsgt
# ============================================================
print("\n"+"="*60)
print("[6] moneyflow_hsgt - 北向资金")
mf_path = f'{DATA_DIR}/moneyflow_hsgt.csv'
mf_last = get_last_date(mf_path, 'trade_date')
mf_start = str(mf_last + 1) if mf_last else '20240101'
if len(mf_start) == 7: mf_start = '0' + mf_start

print(f"  本地: {mf_last}, 拉取: {mf_start}~{TODAY}")

all_new = []
current = datetime.strptime(mf_start, '%Y%m%d')
end = datetime.strptime(TODAY, '%Y%m%d')
while current <= end:
    batch_end = current + timedelta(days=29)
    if batch_end > end: batch_end = end
    cs, ce = current.strftime('%Y%m%d'), batch_end.strftime('%Y%m%d')
    print(f"  {cs}~{ce} ...", end=' ', flush=True)
    df = fetch_retry(pro.moneyflow_hsgt, start_date=cs, end_date=ce)
    if df is not None and len(df) > 0:
        all_new.append(df)
        print(f"{len(df)}条", flush=True)
    else:
        print("0条", flush=True)
    time.sleep(0.3)
    current = batch_end + timedelta(days=30)

if all_new:
    df_all = pd.concat(all_new, ignore_index=True)
    added = stream_append(mf_path, df_all)
    total = len(pd.read_csv(mf_path, usecols=[0], low_memory=False))
    latest = df_all['trade_date'].max() if 'trade_date' in df_all.columns else mf_last
    print(f"  ✅ moneyflow_hsgt: +{added}条 -> {total:,}条 / 最新{latest}")
    results.append(('moneyflow_hsgt', added, total, latest))
else:
    total = len(pd.read_csv(mf_path, usecols=[0], low_memory=False))
    print(f"  ✅ moneyflow_hsgt: 无新数据, {total:,}条")
    results.append(('moneyflow_hsgt', 0, total, mf_last))

# ============================================================
# 汇总
# ============================================================
print("\n" + "="*60)
print("📋 最终汇总")
print("="*60)
summary = []
for name, added, total, latest in results:
    print(f"  {name}: +{added}条 -> {total:,}条 / 最新{latest}")
    summary.append(f"{name}+{added}")
print(f"\n📊 补采完成: {' '.join(summary)}")
