"""
kuaidao_full_market_v1.py
快刀斩股选股策略 Skill —— 全量 A 股回归测试（约 5550 只）

Tushare 接口能力实测 (2026-09-08):
- 可批量: stock_basic / daily_basic(按日) / top10_holders(按period) /
          stk_holdernumber / stk_holdertrade / trade_cal
- 必 ts_code 逐股: income / balancesheet / cashflow / fina_indicator / dividend
  -> 5 接口 × 5500 只 并发 5 线程 + 断点续拉, 约 45-60 分钟

行业口径: 门控1/s6 用申万一级 (sw_industry_map.csv, 31 行业全覆盖);
peer 排名用申万二级 (stock_industry_map.csv, 110 行业)
[与 25 只样本版差异: 样本版细分行业键(白酒/光伏等)在全量版统一为申万一级]

用法:
  python kuaidao_full_market_v1.py --phase a    # 只拉数据 (bulk + per_stock)
  python kuaidao_full_market_v1.py --phase b    # 组装 + 打分 (依赖 Phase A 缓存)
"""
import sys, os, json, time, traceback, threading
from pathlib import Path
sys.path.insert(0, 'e:/quant/scripts')
sys.path.insert(0, 'e:/quant')
os.chdir('e:/quant')

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from credentials import get_tushare_api

import kuaidao_test_v1 as K   # 复用门控/打分函数

FULL_DIR = Path('e:/quant/data/kuaidao_test/full_market')
FULL_DIR.mkdir(parents=True, exist_ok=True)
PER_STOCK_DIR = FULL_DIR / 'per_stock'
PER_STOCK_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR = Path('e:/quant/data/kuaidao_test')
RUN_TS = datetime.now().strftime('%Y%m%d_%H%M%S')
TODAY = '20260908'          # 与 market_snapshot 缓存日期一致

# ------------------------------------------------------------
# Phase A1: 批量拉取 (可批量接口)
# ------------------------------------------------------------
def safe_call(fn, *args, retries=2, **kwargs):
    for i in range(retries + 1):
        try:
            df = fn(*args, **kwargs)
            return df if df is not None else pd.DataFrame()
        except Exception as e:
            if i == retries:
                raise
            time.sleep(0.6)

def fetch_paged(pro, fn, params, fields, page_size=5000):
    frames = []
    off = 0
    while True:
        df = safe_call(fn, **params, fields=fields, limit=page_size, offset=off)
        if df is None or len(df) == 0:
            break
        frames.append(df)
        if len(df) < page_size:
            break
        off += page_size
        time.sleep(0.12)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).drop_duplicates()

def bulk_load(pro, cache_name, fetch_fn):
    cp = FULL_DIR / cache_name
    if cp.exists():
        df = pd.read_csv(cp, dtype=str)
        print(f'  [cache] {cache_name}: {len(df)} rows', flush=True)
        return df
    df = fetch_fn(pro)
    if len(df) > 0:
        df.to_csv(cp, index=False)
    print(f'  [fetch] {cache_name}: {len(df)} rows', flush=True)
    return df

def fetch_bulk_part(pro):
    out = {}
    t0 = time.time()

    def f_basic(p):
        frames = []
        for off in (0, 5000, 10000):
            df = safe_call(p.stock_basic, list_status='L', fields='ts_code,name,industry,list_date', limit=5000, offset=off)
            if len(df) == 0: break
            frames.append(df)
            if len(df) < 5000: break
            time.sleep(0.12)
        return pd.concat(frames, ignore_index=True).drop_duplicates() if frames else pd.DataFrame()
    out['stock_basic'] = bulk_load(pro, 'full_stock_basic.csv', f_basic)

    def f_top10(p):
        frames = []
        for period in ['20251231', '20260630']:
            off = 0
            while True:
                df = safe_call(p.top10_holders, period=period, limit=2000, offset=off,
                               fields='ts_code,ann_date,end_date,holder_name,hold_ratio,hold_amount')
                if df is None or len(df) == 0: break
                frames.append(df)
                if len(df) < 2000: break
                off += 2000
                time.sleep(0.25)
            time.sleep(0.12)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    out['top10'] = bulk_load(pro, 'full_top10.csv', f_top10)

    def f_holder(p):
        return fetch_paged(p, p.stk_holdernumber,
                           {'start_date': '20250901', 'end_date': '20260908'},
                           'ts_code,ann_date,end_date,holder_num')
    out['holder'] = bulk_load(pro, 'full_holder.csv', f_holder)

    def f_trade(p):
        return fetch_paged(p, p.stk_holdertrade,
                           {'start_date': '20250901', 'end_date': '20260908'},
                           'ts_code,ann_date,holder_name,holder_type,in_de,change_vol,change_ratio')
    out['trade'] = bulk_load(pro, 'full_trade.csv', f_trade)

    def f_cal(p):
        return safe_call(p.trade_cal, exchange='SSE', start_date='20260101', end_date='20260908',
                         fields='exchange,cal_date,is_open')
    out['trade_cal'] = bulk_load(pro, 'full_trade_cal.csv', f_cal)

    def f_pshist(p):
        cal = out['trade_cal']
        open_days = sorted(cal[cal['is_open'].astype(str) == '1']['cal_date'].tolist(), reverse=True)
        open_days = [d for d in open_days if d <= TODAY][:130]
        print(f'    [pshist] 需拉 {len(open_days)} 个交易日 daily_basic...', flush=True)
        frames = []
        for i, td in enumerate(open_days, 1):
            df = safe_call(p.daily_basic, trade_date=td, limit=8000, offset=0,
                           fields='ts_code,trade_date,close,ps_ttm')
            if len(df) > 0:
                frames.append(df)
            if i % 20 == 0 or i == len(open_days):
                print(f'      {i}/{len(open_days)} 日 ({time.time()-t0:.0f}s)', flush=True)
            time.sleep(0.10)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    out['pshist'] = bulk_load(pro, 'full_ps_hist.csv', f_pshist)

    print(f'[Phase A1 bulk done] {time.time()-t0:.0f}s', flush=True)
    return out

# ------------------------------------------------------------
# Phase A2: 逐股拉取 (ts_code 必填的 5 接口, 并发 + 断点续拉)
# ------------------------------------------------------------
PER_STOCK_SPECS = {
    'fina':   dict(api='fina_indicator', limit=40, extra={},
                   fields='ts_code,end_date,ann_date,roe,grossprofit_margin,netprofit_margin,tr_yoy,netprofit_yoy,bps'),
    'income': dict(api='income', limit=24, extra={'report_type': '1'},
                   fields='ts_code,end_date,total_revenue,n_income,n_income_attr_p'),
    'bal':    dict(api='balancesheet', limit=12, extra={'report_type': '1'},
                   fields='ts_code,end_date,money_cap,trad_asset,st_borr,lt_borr,bond_payable,non_cur_liab_due_1y,'
                          'accounts_receiv,contract_liab,adv_receipts,goodwill,total_assets,total_liab,total_share'),
    'cf':     dict(api='cashflow', limit=16, extra={'report_type': '1'},
                   fields='ts_code,end_date,c_fr_sale_sg,n_cashflow_act,c_pay_acq_const_fiolta,free_cashflow,net_profit'),
    'div':    dict(api='dividend', limit=50, extra={},
                   fields='ts_code,end_date,ann_date,cash_div,cash_div_tax,div_proc'),
}

def per_stock_cache(kind, code):
    """返回已有缓存路径; 25 只样本缓存在 OUT_DIR 根, 全量缓存在 PER_STOCK_DIR"""
    for p in (PER_STOCK_DIR / f'{kind}_{code}.csv', OUT_DIR / f'{kind}_{code}.csv'):
        if p.exists():
            return p
    return None

def _worker(kind, codes, lock, counter, t0):
    spec = PER_STOCK_SPECS[kind]
    pro = get_tushare_api()   # 每线程独立实例
    api_fn = getattr(pro, spec['api'])
    done, empty, fail = 0, 0, 0
    for code in codes:
        if per_stock_cache(kind, code) is not None:
            continue
        try:
            df = api_fn(ts_code=code, limit=spec['limit'], fields=spec['fields'], **spec['extra'])
            if df is None or len(df) == 0:
                (PER_STOCK_DIR / f'{kind}_{code}.csv').touch()   # 空标记, 避免重拉
                empty += 1
            else:
                df.to_csv(PER_STOCK_DIR / f'{kind}_{code}.csv', index=False)
                done += 1
        except Exception as e:
            fail += 1   # 不写文件, 下一轮重试
            if fail <= 5:
                print(f'  [warn] {kind} {code}: {str(e)[:60]}', flush=True)
            time.sleep(1.0)
        with lock:
            counter[0] += 1
            if counter[0] % 500 == 0:
                print(f'  [A2] {counter[0]}/{counter[1]} 只已处理 ({time.time()-t0:.0f}s) '
                      f'| {kind} 有效{done} 空{empty} 失败{fail}', flush=True)
        time.sleep(0.45)
    return kind, done, empty, fail

def fetch_per_stock_part(codes):
    """并发逐股拉取 5 接口。codes: 全部股票代码列表"""
    t0 = time.time()
    total = 0
    for kind in PER_STOCK_SPECS:
        miss = sum(1 for c in codes if per_stock_cache(kind, c) is None)
        total += miss
        print(f'  [A2] {kind}: 待拉 {miss}/{len(codes)} 只', flush=True)
    if total == 0:
        print('  [A2] 全部已缓存, 跳过', flush=True)
        return
    lock = threading.Lock()
    counter = [0, total]
    with ThreadPoolExecutor(max_workers=5) as ex:
        futures = [ex.submit(_worker, kind, codes, lock, counter, t0) for kind in PER_STOCK_SPECS]
        for fut in as_completed(futures):
            kind, done, empty, fail = fut.result()
            print(f'  [A2 done] {kind}: 有效{done} 空{empty} 失败{fail}', flush=True)
    print(f'[Phase A2 per_stock done] {time.time()-t0:.0f}s', flush=True)

# ------------------------------------------------------------
# Phase B: 组装每只股票的 data dict (与 collect_one 同构)
# ------------------------------------------------------------
def _f(v):
    try:
        fv = float(v)
        return fv if pd.notna(fv) else None
    except Exception:
        return None

def _read_per_stock(code, kind):
    p = per_stock_cache(kind, code)
    if p is None or p.stat().st_size < 20:
        return None
    try:
        return pd.read_csv(p, dtype=str)
    except Exception:
        return None

def build_universe(bulk, market_snap_rows, stock_ind_map, sw_l1_map, market_fina):
    t0 = time.time()
    basic = bulk['stock_basic']
    basic_idx = {str(r['ts_code']): r for _, r in basic.iterrows()}

    top10_by_code = {k: g for k, g in bulk['top10'].groupby('ts_code')} if len(bulk['top10']) else {}
    holder_by_code = {k: g for k, g in bulk['holder'].groupby('ts_code')} if len(bulk['holder']) else {}
    trade_by_code = {k: g for k, g in bulk['trade'].groupby('ts_code')} if len(bulk['trade']) else {}
    ps_by_code = {k: g for k, g in bulk['pshist'].groupby('ts_code')} if len(bulk['pshist']) else {}
    print(f'  bulk 索引构建完成 {time.time()-t0:.0f}s, 开始组装...', flush=True)

    all_data = {}
    codes = sorted(set(market_snap_rows.keys()) | set(stock_ind_map.keys()))
    for code in codes:
        br = basic_idx.get(code)
        if br is None:
            continue
        d = {'ts_code': code, 'name': str(br['name']), 'expected': ''}
        d['industry_l1'] = str(stock_ind_map.get(code, br.get('industry', '')))
        d['sw_l1'] = str(sw_l1_map.get(code, ''))
        d['list_date'] = str(br.get('list_date', ''))
        snap = market_snap_rows.get(code)
        if snap:
            d['price_date'] = TODAY
            d['close'] = snap.get('close')
            d['total_mv_yi'] = (snap.get('total_mv') or 0) / 1e4 if snap.get('total_mv') else None
            d['pe_ttm'] = snap.get('pe_ttm')
            d['pb'] = snap.get('pb')
            d['ps_ttm'] = snap.get('ps_ttm')
        # ---- 财务指标近3年年报 (fina_indicator 逐股缓存) ----
        fa = []
        g = _read_per_stock(code, 'fina')
        if g is not None:
            g = g[g['end_date'].astype(str).str.endswith('1231')].drop_duplicates('end_date') \
                .sort_values('end_date', ascending=False).head(3)
            for _, r in g.iterrows():
                fa.append({'end_date': r['end_date'], 'roe': _f(r.get('roe')),
                           'gross_margin': _f(r.get('grossprofit_margin')),
                           'net_margin': _f(r.get('netprofit_margin')),
                           'rev_yoy': _f(r.get('tr_yoy')), 'profit_yoy': _f(r.get('netprofit_yoy'))})
        d['fina_annual'] = fa
        # ---- 利润表 ----
        ia, quarters = [], []
        g = _read_per_stock(code, 'income')
        if g is not None:
            g = g.drop_duplicates('end_date').sort_values('end_date', ascending=False)
            annual = g[g['end_date'].astype(str).str.endswith('1231')].head(3)
            for _, r in annual.iterrows():
                ia.append({'end_date': r['end_date'], 'revenue': _f(r.get('total_revenue')),
                           'net_profit': _f(r.get('n_income'))})
            for _, r in g.iterrows():
                if _f(r.get('total_revenue')):
                    quarters.append({'end_date': r['end_date'], 'revenue': _f(r.get('total_revenue'))})
        d['income_annual'] = ia
        d['income_quarters'] = quarters
        # ---- 资产负债表 ----
        bal = None
        g = _read_per_stock(code, 'bal')
        if g is not None:
            g = g.drop_duplicates('end_date').sort_values('end_date', ascending=False)
            r = g.iloc[0]
            ts_raw = _f(r.get('total_share'))
            bal = {'end_date': r['end_date'], 'monetary': _f(r.get('money_cap')), 'trad_asset': _f(r.get('trad_asset')),
                   'st_loan': _f(r.get('st_borr')), 'lt_loan': _f(r.get('lt_borr')), 'bond_pay': _f(r.get('bond_payable')),
                   'ltb_1y': _f(r.get('non_cur_liab_due_1y')), 'ar': _f(r.get('accounts_receiv')),
                   'contract_liab': _f(r.get('contract_liab')), 'advance_receipts': _f(r.get('adv_receipts')),
                   'goodwill': _f(r.get('goodwill')), 'total_assets': _f(r.get('total_assets')),
                   'total_liab': _f(r.get('total_liab')),
                   # Tushare balancesheet.total_share 单位=股 (实测行动教育 1.19亿股 与 top10 持股量级一致)
                   'total_share': ts_raw}
        d['balance_latest'] = bal
        if bal and bal.get('total_share'):
            d['total_share'] = bal['total_share']
        # ---- 现金流 ----
        cf = []
        g = _read_per_stock(code, 'cf')
        if g is not None:
            g = g.drop_duplicates('end_date').sort_values('end_date', ascending=False).head(4)
            for _, r in g.iterrows():
                cf.append({'end_date': r['end_date'], 'cf_from_sales': _f(r.get('c_fr_sale_sg')),
                           'ocf': _f(r.get('n_cashflow_act')), 'capex': _f(r.get('c_pay_acq_const_fiolta')),
                           'fcf': _f(r.get('free_cashflow'))})
        d['cf_annual'] = cf
        # ---- 分红 ----
        divs = []
        g = _read_per_stock(code, 'div')
        if g is not None:
            for _, r in g.iterrows():
                divs.append({'end_date': str(r.get('end_date', '')), 'cash_div': _f(r.get('cash_div'))})
        d['dividends'] = divs
        # ---- 股东人数 ----
        d['holder_latest'] = None
        g = holder_by_code.get(code)
        if g is not None:
            g = g.sort_values('end_date', ascending=False)
            r = g.iloc[0]
            hn = _f(r.get('holder_num'))
            d['holder_latest'] = {'end_date': str(r['end_date']), 'holder_num': int(hn) if hn else None}
        # ---- 前10大股东 ----
        d['top10_holders'] = None
        g = top10_by_code.get(code)
        if g is not None:
            latest = g['end_date'].astype(str).max()
            g = g[g['end_date'].astype(str) == latest].sort_values('hold_ratio', ascending=False).head(10)
            d['top10_holders'] = [{'name': r['holder_name'], 'hold_ratio': _f(r.get('hold_ratio')) or 0,
                                   'hold_amount': _f(r.get('hold_amount')) or 0, 'end_date': latest}
                                  for _, r in g.iterrows()]
        # ---- 增减持 ----
        g = trade_by_code.get(code)
        d['trades_3m'] = len(g) if g is not None else 0
        d['trades_in_n'] = int((g['in_de'].astype(str).str.upper() == 'IN').sum()) if g is not None else 0
        d['trades_de_n'] = int((g['in_de'].astype(str).str.upper() == 'DE').sum()) if g is not None else 0
        # ---- PS 历史序列 ----
        d['_ps_hist'] = None
        g = ps_by_code.get(code)
        if g is not None:
            vals = [_f(v) for v in g['ps_ttm'].tolist()]
            vals = [v for v in vals if v is not None and 0 < v < 500]
            d['_ps_hist'] = vals if len(vals) >= 30 else None
        all_data[code] = d
    print(f'  组装完成 {len(all_data)} 只 ({time.time()-t0:.0f}s)', flush=True)
    return all_data

# ------------------------------------------------------------
# Phase C: 打分 (monkeypatch 三处后复用 K 的函数)
# ------------------------------------------------------------
PS_HIST_MEM = {}
ORIG_INDUSTRY_KEY = K.industry_key   # 保存原函数, 防 patch 后递归

def industry_key_full(data):
    sw = data.get('sw_l1')
    if sw:
        return sw
    return ORIG_INDUSTRY_KEY(data)

def ps_hist_from_mem(pro, ts_code, data):
    vals = PS_HIST_MEM.get(ts_code)
    if vals and len(vals) >= 30:
        return vals
    return None

PEER_RANKS = {}

def prebuild_peer_ranks(stock_ind_map, market_fina):
    from collections import defaultdict
    by_ind = defaultdict(list)
    for c, ind in stock_ind_map.items():
        if c in market_fina:
            by_ind[ind].append(c)
    ranks = {}
    for ind, codes in by_ind.items():
        for m in ['roe', 'gross_margin', 'net_margin']:
            valid = [(c, market_fina[c].get(m)) for c in codes if market_fina[c].get(m) is not None]
            valid.sort(key=lambda x: x[1], reverse=True)
            ranks[(ind, m)] = {c: r for r, (c, v) in enumerate(valid, 1)}
    return ranks

def peer_rank_fast(ts_code, ind_l1, stock_ind_map, market_fina, metric):
    idx = PEER_RANKS.get((ind_l1, metric))
    if idx is None:
        return None, 0
    return idx.get(ts_code), len(idx)

def patch_k():
    K.industry_key = industry_key_full
    K.fetch_ps_history = ps_hist_from_mem
    K._peer_rank_metric = peer_rank_fast

def score_all(all_data, market_snap, stock_ind_map, market_fina):
    t0 = time.time()
    gate_results, scorecards = {}, []
    n = len(all_data)
    for i, (code, d) in enumerate(all_data.items(), 1):
        PS_HIST_MEM[code] = d.get('_ps_hist')
        g1 = K.gate_industry(d)
        g2 = K.gate_company(d, all_data, market_snap, stock_ind_map, market_fina)
        # 传 dummy pro: PS 路径会走 monkeypatch 后的 ps_hist_from_mem (忽略 pro)
        g3 = K.gate_price(d, pro='DUMMY')
        gate_results[code] = {'industry': g1, 'company': g2, 'price': g3}
        if any(g.get('pass') is False for g in (g1, g2, g3)):
            scorecards.append({'code': code, 'name': d.get('name', ''), 'gating': gate_results[code],
                               'scores': {}, 'summary': {'decision': '门控淘汰'}})
            continue
        sc = {
            's1_revenue_profit': K.s1_revenue_profit(d),
            's2_roe': K.s2_roe(d),
            's3_balance': K.s3_balance(d),
            's4_cashflow': K.s4_cashflow(d),
            's5_dividend': K.s5_dividend(d),
            's6_industry': K.s6_industry(d),
            's7_leader': K.s7_leader(d, all_data, market_snap, stock_ind_map, market_fina),
            's8_equity_change': K.s8_equity_change(d),
            's9_per_capita': K.s9_per_capita(d),
            's10_valuation': K.s10_valuation(d, market_snap),
        }
        pos = sum(1 for v in sc.values() if v and v[0] == 1)
        neg = sum(1 for v in sc.values() if v and v[0] == -1)
        na = [k for k, v in sc.items() if v is None]
        total = sum(v[0] for v in sc.values() if v and v[0] is not None)
        decision = K.decide(pos, neg, total)
        scorecards.append({
            'code': code, 'name': d.get('name', ''),
            'industry_l1': d.get('industry_l1', ''), 'sw_l1': d.get('sw_l1', ''),
            'is_cyclical': K.is_cyclical(d), 'is_stable_profit': K.is_stable_profit(d),
            'gating': gate_results[code],
            'scores': {k: {'score': v[0] if v else None, 'evidence': v[1] if v else None} for k, v in sc.items()},
            'summary': {'positive': pos, 'negative': neg, 'na_items': na, 'total': total, 'decision': decision},
        })
        if i % 1000 == 0 or i == n:
            print(f'  打分进度 {i}/{n} ({time.time()-t0:.0f}s)', flush=True)
    return gate_results, scorecards

def run_report(all_data, gate_results, scorecards, market_snap):
    from collections import Counter
    rep = {'run_ts': RUN_TS, 'n_stocks': len(scorecards)}
    g1v = Counter(str(gate_results[c]['industry'].get('verdict', 'NA')) for c in gate_results)
    g2pass = Counter(str(gate_results[c]['company'].get('pass')) for c in gate_results)
    g2top3 = Counter(str(gate_results[c]['company'].get('top3_count', 'NA')) for c in gate_results)
    g3m = Counter(str(gate_results[c]['price'].get('method', 'NA')) for c in gate_results)
    g3v = Counter(str(gate_results[c]['price'].get('verdict', f"NA:{gate_results[c]['price'].get('detail','')[:30]}")) for c in gate_results)
    rep['gate1_verdict'] = dict(g1v)
    rep['gate2_pass'] = dict(g2pass)
    rep['gate2_top3'] = dict(g2top3)
    rep['gate3_method'] = dict(g3m)
    rep['gate3_verdict'] = dict(g3v)
    sdist = {}
    for s in scorecards:
        for k, v in s['scores'].items():
            sdist.setdefault(k, Counter())
            sdist[k][v['score']] += 1
    rep['score_dist'] = {k: dict(v) for k, v in sdist.items()}
    rep['decision'] = dict(Counter(s['summary'].get('decision') for s in scorecards))
    rep['stable_profit'] = dict(Counter(s.get('is_stable_profit') for s in scorecards))
    rep['cyclical'] = dict(Counter(s.get('is_cyclical') for s in scorecards))
    codes = list(all_data.keys())
    rep['n_bse_or_b'] = sum(1 for c in codes if c.startswith(('8', '4', '92')) or c[:3] in ('900', '200'))
    rep['n_star'] = sum(1 for c in codes if c.startswith('688'))
    rep['n_gem'] = sum(1 for c in codes if c.startswith('300'))
    g2na = Counter(gate_results[c]['company'].get('detail', '') for c in gate_results
                   if gate_results[c]['company'].get('pass') is None)
    g3na = Counter(gate_results[c]['price'].get('detail', '') for c in gate_results
                   if gate_results[c]['price'].get('pass') is None)
    rep['gate2_na_reasons'] = dict(g2na.most_common(15))
    rep['gate3_na_reasons'] = dict(g3na.most_common(15))
    # s9 异常 corner case: 流通股本<=0
    rep['s9_abnormal'] = [s['code'] for s in scorecards
                          if s['scores'].get('s9_per_capita', {}).get('score') is None
                          and '流通股本' in str(s['scores'].get('s9_per_capita', {}).get('evidence', ''))]
    passed = [s for s in scorecards if s['summary'].get('decision') in ('TOP15候选', '直通候选池')]
    passed.sort(key=lambda x: x['summary']['total'], reverse=True)
    rep['direct_n'] = sum(1 for s in passed if s['summary']['decision'] == '直通候选池')
    rep['top30'] = [{'code': s['code'], 'name': s['name'], 'sw_l1': s.get('sw_l1', ''),
                     'total': s['summary']['total'], 'pos': s['summary']['positive'],
                     'neg': s['summary']['negative'], 'decision': s['summary']['decision']}
                    for s in passed[:30]]
    return rep

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase', default='a', choices=['a', 'b', 'all'])
    ap.add_argument('--limit', type=int, default=0, help='只测试前 N 只 (调试)')
    args = ap.parse_args()

    if args.phase in ('a', 'all'):
        print('=== Phase A1: 批量拉取 (可批量接口) ===', flush=True)
        pro = get_tushare_api()
        bulk = fetch_bulk_part(pro)
        codes = sorted(set(bulk['stock_basic']['ts_code'].tolist()))
        print(f'=== Phase A2: 逐股拉取 ({len(codes)} 只 × 5 接口, 并发+断点续拉) ===', flush=True)
        fetch_per_stock_part(codes)
        print('Phase A 完成', flush=True)
        if args.phase == 'a':
            return

    print('=== Phase B: 组装 + Phase C: 打分 ===', flush=True)
    bulk = {name: (pd.read_csv(FULL_DIR / fname, dtype=str) if (FULL_DIR / fname).exists() else pd.DataFrame())
            for name, fname in [('stock_basic', 'full_stock_basic.csv'), ('top10', 'full_top10.csv'),
                                ('holder', 'full_holder.csv'), ('trade', 'full_trade.csv'),
                                ('trade_cal', 'full_trade_cal.csv'), ('pshist', 'full_ps_hist.csv')]}
    snap_df = pd.read_csv(OUT_DIR / 'market_snapshot.csv', dtype=str)
    msnap = K._snapshot_from_df(snap_df)
    # 补 close/total_mv (K._snapshot_from_df 只存估值字段, 丢弃了价格与市值)
    for _, r in snap_df.iterrows():
        code = str(r['ts_code'])
        if code in msnap['rows']:
            try:
                c = float(r['close'])
                mv = float(r['total_mv'])
                msnap['rows'][code]['close'] = c if pd.notna(c) else None
                msnap['rows'][code]['total_mv'] = mv if pd.notna(mv) else None
            except Exception:
                pass
    # 行业映射 (三层: 申万 L1 is_new=Y -> Tushare 行业名映射 -> 空)
    basic = bulk['stock_basic']
    ind_map = dict(zip(basic['ts_code'].astype(str), basic['industry'].astype(str)))   # 二级(peer 分组用)
    swl1_df = pd.read_csv('e:/quant/data/sw_classification_L1.csv', dtype=str)
    l1_cur = swl1_df[swl1_df['is_new'] == 'Y']
    sw_l1 = dict(zip(l1_cur['con_code'], l1_cur['industry_name']))
    with open('e:/quant/data/tushare_ind_to_sw_l1.json', 'r', encoding='utf-8') as f:
        ts2sw = json.load(f)
    ts2sw.pop('_note', None)
    n_mapped = 0
    for code, ind2 in ind_map.items():
        if code not in sw_l1 and ind2 and ind2 != 'nan':
            sw_l1[code] = ts2sw.get(ind2, '')
            if sw_l1[code]:
                n_mapped += 1
    print(f'  sw_l1: 申万 {len(l1_cur)} + Tushare行业名兜底 {n_mapped} = {len(sw_l1)} 只', flush=True)
    mf = K.fetch_full_market_fina(None, stock_codes=None)   # 从缓存读
    all_data = build_universe(bulk, msnap['rows'], ind_map, sw_l1, mf)
    if args.limit > 0:
        all_data = {k: all_data[k] for k in list(all_data.keys())[:args.limit]}
    patch_k()
    global PEER_RANKS
    PEER_RANKS = prebuild_peer_ranks(ind_map, mf)
    gate_results, scorecards = score_all(all_data, msnap, ind_map, mf)
    rep = run_report(all_data, gate_results, scorecards, msnap)
    with open(OUT_DIR / f'scorecard_full_{RUN_TS}.json', 'w', encoding='utf-8') as f:
        json.dump({'report': rep, 'scorecards': scorecards}, f, ensure_ascii=False, indent=1, default=str)
    rows = [{'code': s['code'], 'name': s['name'], 'sw_l1': s.get('sw_l1', ''), 'ind_l2': s.get('industry_l1', ''),
             'stable': s.get('is_stable_profit'), 'cyclical': s.get('is_cyclical'),
             'total': s['summary'].get('total'), 'pos': s['summary'].get('positive'),
             'neg': s['summary'].get('negative'), 'na': len(s['summary'].get('na_items', [])),
             'decision': s['summary'].get('decision')} for s in scorecards]
    pd.DataFrame(rows).to_csv(OUT_DIR / f'full_ranking_{RUN_TS}.csv', index=False, encoding='utf-8-sig')
    print('\n===== 全量统计 =====', flush=True)
    for k in ('n_stocks', 'decision', 'stable_profit', 'gate1_verdict', 'gate2_pass', 'gate3_method', 'gate3_verdict'):
        print(f'{k}: {rep.get(k)}', flush=True)
    print(f"\n✅ scorecard_full_{RUN_TS}.json / full_ranking_{RUN_TS}.csv", flush=True)

if __name__ == '__main__':
    main()
