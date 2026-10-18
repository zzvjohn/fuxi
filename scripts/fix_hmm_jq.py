#!/usr/bin/env python
"""Fix placeholder factor implementations in HMM ElasticNet JQ strategy files."""
from pathlib import Path

JQ_DIR = Path('e:/quant/strategies/drafts')

# ── Real JQ implementations ──

COMPUTE_ASSET_GROWTH = '''
def compute_asset_growth(stocks, context):
    """YoY asset growth from balance sheet. Higher growth = positive signal."""
    n = len(stocks)
    arr = np.full(n, np.nan)
    try:
        q = query(balance.code, balance.total_assets).filter(
            balance.code.in_(stocks))
        df_cur = get_fundamentals(q, date=context.previous_date)
        import datetime as _dt
        prev_date = context.previous_date - _dt.timedelta(days=365)
        df_prev = get_fundamentals(q, date=prev_date)
    except Exception:
        df_cur = df_prev = None
    for i, s in enumerate(stocks):
        if df_cur is None or df_prev is None:
            continue
        r_cur = df_cur[df_cur['code'] == s]
        r_prev = df_prev[df_prev['code'] == s]
        if r_cur.empty or r_prev.empty:
            continue
        ta_cur = r_cur['total_assets'].iloc[0]
        ta_prev = r_prev['total_assets'].iloc[0]
        if pd.notna(ta_cur) and pd.notna(ta_prev) and ta_prev > 0:
            arr[i] = (ta_cur - ta_prev) / ta_prev
    valid = np.isfinite(arr)
    return arr, valid
'''

COMPUTE_MAX_RET_1M = '''
def compute_max_ret_1m(stocks, price_data):
    """-max(20d daily return). Reversal: high max ret = overbought = negative."""
    n = len(stocks)
    arr = np.full(n, np.nan)
    cd = price_data.get('close', {})
    for i, s in enumerate(stocks):
        if s not in cd:
            continue
        c = np.array(cd[s], dtype=float)
        c = c[np.isfinite(c) & (c > 0)]
        if len(c) < 11:
            continue
        daily_ret = np.diff(c[-21:]) / c[-21:-1]
        arr[i] = -np.nanmax(daily_ret[-20:]) if len(daily_ret) >= 10 else np.nan
    valid = np.isfinite(arr)
    return arr, valid
'''

COMPUTE_DOLLAR_VOL_STABILITY = '''
def compute_dollar_vol_stability(stocks, price_data):
    """CV(close*volume, 20d), negated. Stable dollar vol = positive signal."""
    n = len(stocks)
    arr = np.full(n, np.nan)
    cd = price_data.get('close', {})
    vd = price_data.get('volume', {})
    for i, s in enumerate(stocks):
        if s not in cd or s not in vd:
            continue
        c = np.array(cd[s], dtype=float)
        v = np.array(vd[s], dtype=float)
        mask = np.isfinite(c) & np.isfinite(v) & (c > 0) & (v > 0)
        c_m, v_m = c[mask], v[mask]
        if len(c_m) < 11:
            continue
        dv = (c_m * v_m)[-20:]
        mean_dv = np.nanmean(dv)
        std_dv = np.nanstd(dv)
        if mean_dv == 0:
            continue
        arr[i] = -std_dv / mean_dv
    valid = np.isfinite(arr)
    return arr, valid
'''

COMPUTE_EARNINGS_VOLUME_DRIFT = '''
def compute_earnings_volume_drift(stocks, price_data):
    """Shrink volume * pos return = smart money buying (缩量上涨)."""
    n = len(stocks)
    arr = np.full(n, np.nan)
    cd = price_data.get('close', {})
    vd = price_data.get('volume', {})
    for i, s in enumerate(stocks):
        if s not in cd or s not in vd:
            continue
        c = np.array(cd[s], dtype=float)
        v = np.array(vd[s], dtype=float)
        mask = np.isfinite(c) & np.isfinite(v) & (c > 0) & (v > 0)
        c_m, v_m = c[mask], v[mask]
        if len(c_m) < 11:
            continue
        ret_5d = c_m[-1] / c_m[-6] - 1
        if ret_5d <= 0:
            arr[i] = 0.0
            continue
        ma5_now = np.nanmean(v_m[-5:])
        ma5_prev = np.nanmean(v_m[-10:-5]) if len(v_m) >= 10 else ma5_now
        if ma5_prev == 0:
            arr[i] = 0.0
            continue
        vol_chg = ma5_now / ma5_prev - 1
        ret_clipped = np.clip(ret_5d, -0.3, 0.3)
        vol_clipped = np.clip(-vol_chg, -1.0, 2.0)
        arr[i] = ret_clipped * vol_clipped
    valid = np.isfinite(arr)
    return arr, valid
'''

# ── Fix File function ──

def fix_file(fpath):
    src = fpath.read_text(encoding='utf-8')
    modified = False
    
    # Find placeholder compute functions and replace
    placeholder_patterns = {
        'asset_growth': (COMPUTE_ASSET_GROWTH, 'asset_growth'),
        'max_ret_1m': (COMPUTE_MAX_RET_1M, 'max_ret_1m'),
        'dollar_vol_stability': (COMPUTE_DOLLAR_VOL_STABILITY, 'dollar_vol_stability'),
        'earnings_volume_drift': (COMPUTE_EARNINGS_VOLUME_DRIFT, 'earnings_volume_drift'),
    }
    
    for fname, (new_impl, check_name) in placeholder_patterns.items():
        # Find function start
        fn_start = f'def compute_{fname}('
        if fn_start not in src:
            continue
        
        # Check if it's a placeholder (returns np.full(n, np.nan))
        pos = src.find(fn_start)
        # Find next def or end of file
        next_pos = src.find('\ndef ', pos + len(fn_start))
        if next_pos == -1:
            next_pos = len(src)
        
        func_src = src[pos:next_pos]
        if 'np.full(n, np.nan)' in func_src:
            # It's a placeholder, replace
            src = src[:pos] + new_impl.strip() + '\n' + src[next_pos:]
            modified = True
            print(f'  ✅ {fname}: placeholder → real implementation')
        else:
            print(f'  ✓ {fname}: already has real implementation')
    
    if modified:
        fpath.write_text(src, encoding='utf-8')
        # Validate
        try:
            compile(src, str(fpath), 'exec')
            print(f'  ✅ compile OK after fixes')
        except SyntaxError as e:
            print(f'  ❌ compile FAILED: {e}')
            return False
    return True

# ── Main ──

files = list(JQ_DIR.glob('fa_hmm_elasticnet_state_*.py'))
print(f'Fixing {len(files)} HMM ElasticNet JQ files...\n')

for fpath in sorted(files):
    print(f'--- {fpath.name} ---')
    fix_file(fpath)
    print()

print('Done!')
