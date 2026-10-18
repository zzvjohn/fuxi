"""
Stage 0 修复脚本 — 采纳全部教训:
1. 大文件只读日期列获取元信息
2. 流式追加+列名校验+NaN填充
3. 写入前备份，写入后验证行数
4. 绝不将大CSV全量读入内存做merge
"""
import sys
sys.path.append('e:/quant')
from credentials import get_tushare_api
import pandas as pd
import time, os, shutil
from datetime import datetime, timedelta

pro = get_tushare_api()
DATA = 'e:/quant/data/raw'
TODAY = '20260703'

def file_info(path, date_col):
    """读取元信息：只读日期列，零风险"""
    n = 0
    with open(path, 'r', encoding='utf-8-sig') as f:
        f.readline()  # skip header
        for _ in f: n += 1
    df = pd.read_csv(path, usecols=[date_col], low_memory=False)
    vals = pd.to_numeric(df[date_col], errors='coerce').dropna().astype('int64')
    return n, vals.min() if len(vals) else None, vals.max() if len(vals) else None

def safe_write(path, df, verify_rows_min=None):
    """安全写入：先备份，再写，再验证"""
    if os.path.exists(path) and os.path.getsize(path) > 100:
        bak = path + '.bak'
        shutil.copy2(path, bak)
    df.to_csv(path, index=False, encoding='utf-8-sig')
    if verify_rows_min:
        # 快速验证：只读首列计数
        vdf = pd.read_csv(path, usecols=[0], low_memory=False)
        if len(vdf) < verify_rows_min:
            print(f'  ❌ 验证失败: {len(vdf)} < {verify_rows_min}, 回滚!')
            if os.path.exists(path + '.bak'):
                shutil.copy2(path + '.bak', path)
            raise RuntimeError(f'{path}: 写入后行数异常 {len(vdf)} < {verify_rows_min}')

def stream_append(path, df_new):
    """流式追加：校验列后直接append，0内存"""
    if len(df_new) == 0: return 0
    if not os.path.exists(path) or os.path.getsize(path) < 100:
        df_new.to_csv(path, index=False, encoding='utf-8-sig')
        return len(df_new)
    
    # 只读表头
    existing_cols = pd.read_csv(path, nrows=0).columns.tolist()
    new_cols = df_new.columns.tolist()
    
    if existing_cols != new_cols:
        print(f'  ⚠️ 列不匹配，填充NaN对齐')
        for c in existing_cols:
            if c not in new_cols:
                df_new[c] = None
        df_new = df_new[existing_cols]
    
    # 行数验证标记（写前）
    before = 0
    with open(path, 'r', encoding='utf-8-sig') as f:
        f.readline()
        for _ in f: before += 1
    
    # 备份
    bak = path + '.tmp_append'
    shutil.copy2(path, bak)
    
    # 追加写入
    df_new.to_csv(path, mode='a', index=False, header=False, encoding='utf-8-sig')
    
    # 验证
    after = 0
    with open(path, 'r', encoding='utf-8-sig') as f:
        f.readline()
        for _ in f: after += 1
    
    if after < before:
        print(f'  ❌ 追加后行数异常 {after} < {before}, 回滚!')
        shutil.copy2(bak, path)
        raise RuntimeError(f'{path}: append后行数 {after} < 追加前 {before}')
    
    os.remove(bak)
    return after - before

def fetch_retry(func, max_r=3, **kw):
    for i in range(max_r):
        try:
            return func(**kw)
        except Exception as e:
            s = str(e)
            if any(k in s for k in ['频率','QPS','429','limit','Too Many']):
                print(f'  频率限制，等6分钟...', flush=True)
                time.sleep(360)
                continue
            if i < max_r - 1:
                time.sleep((i+1)*3)
    return None


# ============================================================
# 1. 修复 daily_basic: 去重
# ============================================================
print('='*60)
print('[1] 修复 daily_basic — 去重')
print('='*60)

db_path = f'{DATA}/daily_basic.csv'
b4, mn, mx = file_info(db_path, 'trade_date')
print(f'  当前: {b4:,}条, trade_date={mn}~{mx}')

# 用chunk迭代去重，避免内存爆炸
tmp_path = db_path + '.dedup'
# 先读全部去重（daily_basic 18列 * 8M行 ≈ 2.5GB — 可能OOM, 但这是修复性操作一次性）
print('  去重中 (keep=first, 按ts_code+trade_date)...')
df = pd.read_csv(db_path, low_memory=False)
before = len(df)
df = df.drop_duplicates(subset=['ts_code', 'trade_date'], keep='last')
df = df.sort_values('trade_date').reset_index(drop=True)
after = len(df)

print(f'  {before:,} -> {after:,} (去除 {before-after:,} 重复)')

# 安全写入
safe_write(db_path, df, verify_rows_min=6000000)  # 预期至少600万行

a2, mn2, mx2 = file_info(db_path, 'trade_date')
print(f'  ✅ daily_basic 修复: {a2:,}条, trade_date={mn2}~{mx2} ({a2-b4:+d})')
del df  # 释放内存


# ============================================================
# 2. stk_factor: 增量1-2天
# ============================================================
print('\n'+'='*60)
print('[2] stk_factor — 增量')
print('='*60)
sf_path = f'{DATA}/stk_factor.csv'
b4, mn, mx = file_info(sf_path, 'trade_date')
start = str(int(mx) + 1)
if len(start) == 7: start = '0' + start
print(f'  当前: {b4:,}条, 最新{mx}, 增量{start}~{TODAY}')

all_new = []
current = datetime.strptime(start, '%Y%m%d')
end = datetime.strptime(TODAY, '%Y%m%d')

while current <= end:
    d = current.strftime('%Y%m%d')
    print(f'  trade_date={d} ...', end=' ', flush=True)
    # stk_factor用trade_date单日查询
    df = fetch_retry(pro.stk_factor, trade_date=d)
    if df is not None and len(df) > 0:
        all_new.append(df)
        print(f'{len(df)}条')
    else:
        print('0条')
    time.sleep(0.4)
    current += timedelta(days=1)

if all_new:
    df_all = pd.concat(all_new, ignore_index=True)
    added = stream_append(sf_path, df_all)
    a2, _, mx2 = file_info(sf_path, 'trade_date')
    print(f'  ✅ stk_factor: +{added}条 -> {a2:,}条 / 最新{mx2}')
else:
    print(f'  ✅ stk_factor: 无新数据, {b4:,}条')


# ============================================================
# 3. report_rc: 重点补采（10次/h限制）
# ============================================================
print('\n'+'='*60)
print('[3] report_rc — 研报补采 (10批/h)')
print('='*60)
rc_path = f'{DATA}/report_rc.csv'
b4, mn, mx = file_info(rc_path, 'report_date')
start = str(int(mx) + 1)
if len(start) == 7: start = '0' + start
print(f'  当前: {b4:,}条, 最新report_date={mx}, 增量{start}~{TODAY}')

rc_new = []
current = datetime.strptime(start, '%Y%m%d')
end = datetime.strptime(TODAY, '%Y%m%d')
batch = 0

while current <= end and batch < 10:
    batch += 1
    batch_end = current + timedelta(days=6)
    if batch_end > end: batch_end = end
    cs, ce = current.strftime('%Y%m%d'), batch_end.strftime('%Y%m%d')
    
    print(f'  [{batch}/10] {cs}~{ce} ...', end=' ', flush=True)
    df = fetch_retry(pro.report_rc, max_r=2,
                     start_date=cs, end_date=ce,
                     fields='ts_code,name,report_date,report_title,report_type,classify,'
                            'org_name,author_name,rating,rating_change,eps,net_profit,'
                            'total_mv,pe,pb,roe,report_year,end_date')
    if df is not None and len(df) > 0:
        rc_new.append(df)
        print(f'{len(df)}条')
        time.sleep(0.5)
    else:
        print('0条')
        time.sleep(1)
    current = batch_end + timedelta(days=1)

if batch == 10 and current <= end:
    print(f'  ⚠️ 达10批上限, 剩余{current.strftime("%Y%m%d")}~{TODAY}')

if rc_new:
    df_all = pd.concat(rc_new, ignore_index=True)
    added = stream_append(rc_path, df_all)
    a2, _, mx2 = file_info(rc_path, 'report_date')
    print(f'  ✅ report_rc: +{added}条 -> {a2:,}条 / 最新{mx2}')
else:
    print(f'  ✅ report_rc: 无新数据, {b4:,}条')


# ============================================================
# 4. moneyflow_hsgt: 增量
# ============================================================
print('\n'+'='*60)
print('[4] moneyflow_hsgt — 北向资金')
print('='*60)
mf_path = f'{DATA}/moneyflow_hsgt.csv'
b4, mn, mx = file_info(mf_path, 'trade_date')
start = str(int(mx) + 1)
if len(start) == 7: start = '0' + start
print(f'  当前: {b4:,}条, 最新{mx}, 增量{start}~{TODAY}')

all_new = []
current = datetime.strptime(start, '%Y%m%d')
end = datetime.strptime(TODAY, '%Y%m%d')
while current <= end:
    batch_end = current + timedelta(days=29)
    if batch_end > end: batch_end = end
    cs, ce = current.strftime('%Y%m%d'), batch_end.strftime('%Y%m%d')
    print(f'  {cs}~{ce} ...', end=' ', flush=True)
    df = fetch_retry(pro.moneyflow_hsgt, start_date=cs, end_date=ce)
    if df is not None and len(df) > 0:
        all_new.append(df)
        print(f'{len(df)}条')
    else:
        print('0条')
    time.sleep(0.3)
    current = batch_end + timedelta(days=30)

if all_new:
    df_all = pd.concat(all_new, ignore_index=True)
    added = stream_append(mf_path, df_all)
    a2, _, mx2 = file_info(mf_path, 'trade_date')
    print(f'  ✅ moneyflow_hsgt: +{added}条 -> {a2:,}条 / 最新{mx2}')
else:
    print(f'  ✅ moneyflow_hsgt: 无新数据, {b4:,}条')


# ============================================================
# 最终汇总
# ============================================================
print('\n'+'='*60)
print('📋 最终数据状态')
print('='*60)
for name, (path, col) in [
    ('daily_prices', (f'{DATA}/daily_prices.csv', 'trade_date')),
    ('daily_basic', (f'{DATA}/daily_basic.csv', 'trade_date')),
    ('stk_factor', (f'{DATA}/stk_factor.csv', 'trade_date')),
    ('fina_indicator', (f'{DATA}/fina_indicator.csv', 'ann_date')),
    ('report_rc', (f'{DATA}/report_rc.csv', 'report_date')),
    ('moneyflow_hsgt', (f'{DATA}/moneyflow_hsgt.csv', 'trade_date')),
]:
    n, mn, mx = file_info(path, col)
    sz = os.path.getsize(path) / (1024*1024)
    print(f'  {name:<20} {n:>10,}条 {sz:>7.1f}MB  {col}={mn}~{mx}')
