# -*- coding: utf-8 -*-
"""Update registry_generated.py with real JQ implementations for 7 factors."""
import re
import textwrap
import sys
import os

# --- Define proper JQ implementations for each factor ---
IMPLEMENTATIONS = {}

IMPLEMENTATIONS['atr_14'] = '''\
def compute_atr_14(stocks, price_data):
    """ATR(14) normalized by close, negated. Low ATR = positive signal."""
    n = len(stocks)
    arr = np.full(n, np.nan)
    hd = price_data.get('high', {})
    ld = price_data.get('low', {})
    cd = price_data.get('close', {})
    for i, s in enumerate(stocks):
        if s not in hd or s not in ld or s not in cd:
            continue
        h = np.array(hd[s], dtype=float)
        l = np.array(ld[s], dtype=float)
        c = np.array(cd[s], dtype=float)
        if len(h) < 16 or len(l) < 16 or len(c) < 16:
            continue
        tr = np.zeros(len(c) - 1)
        for t in range(1, len(c)):
            tr[t - 1] = max(
                h[t] - l[t],
                abs(h[t] - c[t - 1]),
                abs(l[t] - c[t - 1]))
        if len(tr) < 14:
            continue
        alpha = 2.0 / 15.0
        ema = tr[0]
        for t in range(1, len(tr)):
            ema = alpha * tr[t] + (1 - alpha) * ema
        c_now = c[-1]
        if c_now == 0 or np.isnan(c_now) or np.isnan(ema):
            continue
        arr[i] = -ema / c_now
    valid = np.isfinite(arr)
    return arr, valid'''

IMPLEMENTATIONS['bp'] = '''\
def compute_bp(stocks, context):
    """Book-to-Price = 1/pb_ratio. Higher BP = value = positive signal."""
    n = len(stocks)
    arr = np.full(n, np.nan)
    try:
        q = query(valuation.code, valuation.pb_ratio).filter(
            valuation.code.in_(stocks))
        df = get_fundamentals(q, date=context.previous_date)
    except:
        return arr, np.zeros(n, dtype=bool)
    if df is None or len(df) == 0:
        return arr, np.zeros(n, dtype=bool)
    stock_idx = {s: i for i, s in enumerate(stocks)}
    for _, row in df.iterrows():
        code = row['code']
        pb = row.get('pb_ratio')
        if code in stock_idx and pb is not None and not np.isnan(pb) and pb > 0:
            arr[stock_idx[code]] = 1.0 / pb
    valid = np.isfinite(arr)
    return arr, valid'''

IMPLEMENTATIONS['ep'] = '''\
def compute_ep(stocks, context):
    """Earnings-to-Price = 1/pe_ratio. Higher EP = value = positive signal."""
    n = len(stocks)
    arr = np.full(n, np.nan)
    try:
        q = query(valuation.code, valuation.pe_ratio).filter(
            valuation.code.in_(stocks))
        df = get_fundamentals(q, date=context.previous_date)
    except:
        return arr, np.zeros(n, dtype=bool)
    if df is None or len(df) == 0:
        return arr, np.zeros(n, dtype=bool)
    stock_idx = {s: i for i, s in enumerate(stocks)}
    for _, row in df.iterrows():
        code = row['code']
        pe = row.get('pe_ratio')
        if code in stock_idx and pe is not None and not np.isnan(pe) and pe > 0:
            arr[stock_idx[code]] = 1.0 / pe
    valid = np.isfinite(arr)
    return arr, valid'''

IMPLEMENTATIONS['volume_stability'] = '''\
def compute_volume_stability(stocks, price_data):
    """Volume CV over 20d, negated. Stable volume = positive signal."""
    n = len(stocks)
    arr = np.full(n, np.nan)
    vd = price_data.get('volume', {})
    for i, s in enumerate(stocks):
        if s not in vd:
            continue
        v = np.array(vd[s], dtype=float)
        v_clean = v[np.isfinite(v) & (v > 0)]
        if len(v_clean) < 10:
            continue
        roll = v_clean[-20:] if len(v_clean) >= 20 else v_clean
        std_v = np.nanstd(roll)
        mean_v = np.nanmean(roll)
        if mean_v == 0:
            continue
        arr[i] = -std_v / mean_v
    valid = np.isfinite(arr)
    return arr, valid'''

IMPLEMENTATIONS['range_consistency'] = '''\
def compute_range_consistency(stocks, price_data):
    """Range CV over 20d, negated. Consistent range = positive signal."""
    n = len(stocks)
    arr = np.full(n, np.nan)
    hd = price_data.get('high', {})
    ld = price_data.get('low', {})
    for i, s in enumerate(stocks):
        if s not in hd or s not in ld:
            continue
        h = np.array(hd[s], dtype=float)
        l = np.array(ld[s], dtype=float)
        min_len = min(len(h), len(l))
        if min_len < 10:
            continue
        rng = np.abs(h[-min_len:] - l[-min_len:])
        roll = rng[-20:] if len(rng) >= 20 else rng
        std_r = np.nanstd(roll)
        mean_r = np.nanmean(roll)
        if mean_r == 0:
            continue
        arr[i] = -std_r / mean_r
    valid = np.isfinite(arr)
    return arr, valid'''

IMPLEMENTATIONS['high_low_range'] = '''\
def compute_high_low_range(stocks, price_data):
    """20d avg (H-L)/close, negated. Narrow range = positive signal."""
    n = len(stocks)
    arr = np.full(n, np.nan)
    hd = price_data.get('high', {})
    ld = price_data.get('low', {})
    cd = price_data.get('close', {})
    for i, s in enumerate(stocks):
        if s not in hd or s not in ld or s not in cd:
            continue
        h = np.array(hd[s], dtype=float)
        l = np.array(ld[s], dtype=float)
        c = np.array(cd[s], dtype=float)
        min_len = min(len(h), len(l), len(c))
        if min_len < 10:
            continue
        h_arr = h[-min_len:]
        l_arr = l[-min_len:]
        c_arr = c[-min_len:]
        daily_range = (h_arr - l_arr) / c_arr
        valid_range = daily_range[np.isfinite(daily_range) & (c_arr > 0)]
        if len(valid_range) < 10:
            continue
        roll = valid_range[-20:] if len(valid_range) >= 20 else valid_range
        avg_range = np.nanmean(roll)
        if np.isnan(avg_range):
            continue
        arr[i] = -avg_range
    valid = np.isfinite(arr)
    return arr, valid'''

IMPLEMENTATIONS['ret_6m'] = '''\
def compute_ret_6m(stocks, price_data):
    """6-month return reversal. Negative past return = positive signal."""
    n = len(stocks)
    arr = np.full(n, np.nan)
    cd = price_data.get('close', {})
    for i, s in enumerate(stocks):
        if s not in cd:
            continue
        c = np.array(cd[s], dtype=float)
        c_clean = c[np.isfinite(c) & (c > 0)]
        if len(c_clean) < 131:
            continue
        ret = c_clean[-1] / c_clean[-126] - 1.0
        arr[i] = -ret
    valid = np.isfinite(arr)
    return arr, valid'''


def main():
    filepath = os.path.join(os.path.dirname(__file__), 'registry_generated.py')
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    updated_count = 0
    for fname, impl in IMPLEMENTATIONS.items():
        # Indent the impl to match the ''' wrapper (12 spaces inside dict)
        impl_indented = textwrap.indent(impl, '            ')
        new_jq_block = "        'jq_compute_function': '''\n" + impl_indented + "\n        ''',"

        # Replace the jq_compute_function block within the factor's dict entry
        pattern = (
            r"('" + re.escape(fname) + r"':\s*\{.*?)"
            r"'jq_compute_function':\s*'''.*?''',"
        )
        replacement = r"\g<1>" + new_jq_block
        new_content, n = re.subn(pattern, replacement, content, count=1, flags=re.DOTALL)
        if n > 0:
            content = new_content
            updated_count += 1
            print(f'  [OK] Updated jq_compute_function for {fname}')
        else:
            print(f'  [FAIL] Could not find {fname} in registry_generated.py')

    # Fix bp/ep jq_data from 'price' to 'fundamental'
    for fname in ['bp', 'ep']:
        pattern = r"('" + re.escape(fname) + r"':\s*\{.*?)'jq_data':\s*'price'"
        replacement = r"\g<1>'jq_data': 'fundamental'"
        new_content, n = re.subn(pattern, replacement, content, count=1, flags=re.DOTALL)
        if n > 0:
            content = new_content
            print(f'  [OK] Changed {fname} jq_data: price -> fundamental')

    # Write back
    with open(filepath, 'w', encoding='utf-8', newline='\n') as f:
        f.write(content)

    print(f'\nUpdated {updated_count}/7 factor implementations')

    # Verify by importing
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from jq_generator.registry import JQ_FACTOR_REGISTRY
    for fname in IMPLEMENTATIONS:
        meta = JQ_FACTOR_REGISTRY[fname]
        code = meta['jq_compute_function']
        has_placeholder = 'placeholder' in code.lower() or 'Auto-generated placeholder' in code
        length = len(code)
        status = 'PLACEHOLDER' if has_placeholder else 'REAL'
        print(f'  {fname}: len={length}, {status}, data={meta["jq_data"]}')


if __name__ == '__main__':
    main()
