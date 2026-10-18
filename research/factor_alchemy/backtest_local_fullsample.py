# -*- coding: utf-8 -*-
"""
本地全样本回测: no_regime_allf vs stage_2 (同方法对照)
- 复刻 JQ 文件合成逻辑: 截面 z-score 加权 -> Top-30 周频再平衡
- no_regime_allf: 因子 ICIR-signed(高=好) + 正权重(abs)
- stage_2:       原始因子 + 签名权重(如 bargaining_power_proxy=-0.2452)
- 成本: JQ 对齐 万1双边佣金 + 千1印花(卖) + 0.3%滑点 => 满换手往返 0.72%
- 市值过滤: 流通市值 >= 5亿 (亿元)
前向收益 = close_weekly.shift(-1)/close_weekly - 1
"""
import sys, time, pickle, json, warnings
import numpy as np, pandas as pd
from datetime import datetime
sys.path.insert(0, 'research/factor_alchemy')
from data.loader import load_all_data
from evaluation.ic_analysis import compute_ic_icir
warnings.filterwarnings('ignore')

t0 = time.time()
def log(m): print(m, flush=True)

log('[1] loading data (slow, ~10min)...')
data = load_all_data('2021-01-01', '2026-06-25')
price_data = data['price_data']; valuation_data = data['valuation_data']
close = price_data.get('close')
close_weekly = close.resample('W').last(); close_weekly.index = pd.to_datetime(close_weekly.index)
forward_returns = close_weekly.shift(-1) / close_weekly - 1
log('   close_weekly %s  t=%.0fs' % (close_weekly.shape, time.time()-t0))

mcap_weekly = None
if valuation_data is not None:
    col = 'circ_mv' if 'circ_mv' in valuation_data.columns else ('market_cap' if 'market_cap' in valuation_data.columns else None)
    if col:
        m = valuation_data[['code', 'trade_date', col]].drop_duplicates()
        m['trade_date'] = pd.to_datetime(m['trade_date'])
        mp = m.pivot(index='trade_date', columns='code', values=col).resample('W').last()
        mp.index = pd.to_datetime(mp.index)
        mcap_weekly = mp
        log('   mcap_weekly %s' % (mcap_weekly.shape,))

log('[2] loading factor cache...')
with open('research/factor_alchemy/output/factor_cache.pkl', 'rb') as f:
    cache = pickle.load(f)

def zscore_cross(series):
    mu = series.mean(); sd = series.std()
    if sd == 0 or np.isnan(sd): return series * 0.0
    return (series - mu) / sd

def icir_sign(fname):
    f = cache[fname]
    cols = [c for c in f.columns if c in forward_returns.columns]
    ff = f.loc[forward_returns.index, cols]
    fr = forward_returns.loc[forward_returns.index, cols]
    try:
        r = compute_ic_icir(ff, fr, min_samples=10)
        icir = r.get('icir', 0) if isinstance(r, dict) else 0
    except Exception:
        icir = 0
    if icir is None or np.isnan(icir): icir = 0
    return 1.0 if icir >= 0 else -1.0

def run_backtest(factors, weights, signed_weights=False, topn=30, mcap_min=5.0):
    weeks = forward_returns.index
    signs = {f: (icir_sign(f) if not signed_weights else 1.0) for f in factors}
    fmats = {}
    for f in factors:
        ff = cache[f]
        cols = [c for c in ff.columns if c in forward_returns.columns]
        fmats[f] = ff.loc[weeks, cols]
    mw = mcap_weekly.reindex(index=weeks, columns=forward_returns.columns) if mcap_weekly is not None else None
    port_ret = []; holdings = set()
    for wk in weeks:
        comp = pd.Series(0.0, index=forward_returns.columns)
        for f in factors:
            s = fmats[f].loc[wk].reindex(comp.index)
            if not signed_weights:
                s = signs[f] * s
            comp += weights[f] * zscore_cross(s)
        if mw is not None:
            mc = mw.loc[wk].reindex(comp.index)
            comp = comp.where(mc >= mcap_min, other=np.nan)
        comp = comp.dropna()
        if len(comp) < topn:
            holdings = set(); continue
        target = set(comp.nlargest(topn).index)
        cost = 0.0
        if len(holdings) > 0:
            sell = holdings - target; buy = target - holdings
            turnover = (len(sell) + len(buy)) / (2.0 * topn)
            cost = turnover * 0.0072
        fr = forward_returns.loc[wk].reindex(comp.index)
        valid = [s for s in target if s in fr.index and pd.notna(fr[s])]
        if len(valid) < topn // 2:
            holdings = set(); continue
        ret = fr[valid].mean()
        port_ret.append(ret - cost)
        holdings = target
    pr = pd.Series(port_ret)
    if len(pr) == 0:
        return dict(annual=0.0, sharpe=0.0, maxdd=0.0, n=0, cum_end=0.0)
    cum = (1 + pr).cumprod() - 1
    ann = pr.mean() * 52
    sd = pr.std() * np.sqrt(52)
    sh = ann / sd if sd > 0 else 0
    mdd = (cum - cum.cummax()).min()
    return dict(annual=float(ann), sharpe=float(sh), maxdd=float(mdd), n=int(len(pr)), cum_end=float(cum.iloc[-1]))

nr_weights = {'relative_spread_proxy':0.3062,'bargaining_power_proxy':0.2457,'min_ret_1m':0.1786,'avg_turnover_3m':0.1650,'atr_14':0.1044}
log('[3] backtest no_regime_allf (ICIR-signed + 正权重)...')
nr_res = run_backtest(list(nr_weights.keys()), nr_weights, signed_weights=False)
log('   no_regime_allf: ' + json.dumps({k: round(v,4) if isinstance(v,float) else v for k,v in nr_res.items()}))

s2_weights = {'amihud_illiq':0.0813,'bargaining_power_proxy':-0.2452,'vol_3m':0.3135,'capital_efficiency_proxy':0.1014,'dollar_vol_20d':0.2586}
log('[4] backtest stage_2 (原始因子 + 签名权重)...')
s2_res = run_backtest(list(s2_weights.keys()), s2_weights, signed_weights=True)
log('   stage_2: ' + json.dumps({k: round(v,4) if isinstance(v,float) else v for k,v in s2_res.items()}))

out = {'no_regime_allf': nr_res, 'stage_2': s2_res,
       'note': '本地全样本回测(净成本0.72%满换手往返, 5亿流通市值过滤). 绝对数字受本地模拟器偏差影响, 仅组合间比值可参考. stage_2 JQ真回测=+141%.'}
with open('research/factor_alchemy/output/local_backtest_no_regime_vs_stage2.json', 'w') as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
log('DONE t=%.0fs' % (time.time()-t0))
