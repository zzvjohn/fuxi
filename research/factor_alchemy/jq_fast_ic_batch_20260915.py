# -*- coding: utf-8 -*-
"""
JQ 单因子 IC 快验 (批量, 无组合模拟) — 自动生成 2026-09-15 20:31
=========================================================
用途: 周频 rank IC/ICIR 第二道证据门 (分离力分析: 仅 JQ 口径指标有裁决力)
口径: 中证800 (000300+000905 并集), 1300 日线, 每 5 日网格周频 IC,
      因子用日线口径计算 (与完整回测 standalone 同一公式口径), fwd 5 日收益为标签
输出: 日志 [FAST_IC] <name> ic=.. icir=.. n=.. hint=..  ← 本地 --ingest 解析回灌
成本: 无订单/无费用/无持仓 → 秒级~分钟级, 远快于组合回测
"""
from jqdata import *
import numpy as np
import pandas as pd
from scipy.stats import rankdata

def _rolling_rank_pct_vec(a, window, min_periods=None):
    """pandas>=1.4 Rolling.rank(pct=True) 的 numpy 等价实现 (JQ 老 pandas 兼容)
    - min_periods=None → 默认 = window (窗口含 NaN 则该位输出 NaN)
    - min_periods=N   → 窗口内非 NaN 计数 >= N 才计算, pct 分母 = 非 NaN 计数
    """
    from numpy.lib.stride_tricks import as_strided
    a = np.asarray(a, dtype=float)
    n = len(a)
    out = np.full(n, np.nan)
    if n < window:
        return out
    if min_periods is None:
        min_periods = window
    m = n - window + 1
    x = as_strided(a, shape=(m, window), strides=(a.strides[0], a.strides[0]))
    nonnan = (~np.isnan(x)).sum(axis=1)
    ok = nonnan >= min_periods
    last = x[:, -1]
    less = (x < last[:, None]).sum(axis=1)
    eq = (x == last[:, None]).sum(axis=1)
    rank = less + (eq + 1.0) / 2.0
    pct = rank / np.maximum(nonnan, 1)
    pct[~ok] = np.nan
    out[window - 1:] = pct
    return out


def _roll_rank_apply(x, window, min_periods=None):
    """v0.6.2: rolling(W).rank(pct=True) 的 Series/DataFrame 自适应等价实现。
    - DataFrame: 逐列沿时间轴滚动排名 (与 pandas df.rolling(W).rank(pct=True) 语义一致)
    - Series:   直接委托 _rolling_rank_pct_vec
    修复: 宽表 DataFrame 公式 (.rolling(30).rank(pct=True)) 曾被按一维翻译
    → 形状错位 → valid=0 事故 (2026-08-29 forge_gen1_square_square_tsmin)
    """
    if isinstance(x, pd.DataFrame):
        out = pd.DataFrame(np.nan, index=x.index, columns=x.columns)
        for c in x.columns:
            out[c] = _rolling_rank_pct_vec(
                x[c].values.astype(float), window, min_periods)
        return out
    return pd.Series(_rolling_rank_pct_vec(
        np.asarray(x, dtype=float), window, min_periods), index=x.index)




def _rank_pct(obj):
    """pandas .rank(pct=True, na_option='keep') 的 JQ 兼容实现 (axis=0 逐列时序排名)。

    语义与本地 S5 引擎 (pandas 3.x): pct = rank / n_non_na, 单值列 -> 1.0, NaN 保持 NaN。
    """
    if isinstance(obj, pd.DataFrame):
        out = pd.DataFrame(np.nan, index=obj.index, columns=obj.columns)
        for c in obj.columns:
            out[c] = _rank_pct(obj[c])
        return out
    v = obj.values.astype(float)
    mask = np.isfinite(v)
    out = np.full(len(v), np.nan)
    k = int(mask.sum())
    if k >= 1:
        from scipy.stats import rankdata
        r = rankdata(v[mask])
        out[mask] = r / k
    return pd.Series(out, index=obj.index)



_EXPR_factor_forge_gen4_gp_gen4_r1_ts_rank_ts_mean_ts_std_log_ts = "-(_roll_rank_apply((np.log(np.maximum(volume_p.rolling(10).sum(), 1e-8)).rolling(5).std().rolling(60).mean()), 10))"
_GLOBALS_factor_forge_gen4_gp_gen4_r1_ts_rank_ts_mean_ts_std_log_ts = {
    "np": np, "pd": pd,
    "abs": np.abs, "sqrt": np.sqrt, "log": np.log, "log1p": np.log1p,
    "exp": np.exp, "sign": np.sign,
    "maximum": np.maximum, "minimum": np.minimum,
    "where": np.where, "clip": np.clip,
    "range": range, "len": len, "int": int, "float": float,
    "list": list, "dict": dict, "tuple": tuple,
    "_rolling_rank_pct_vec": _rolling_rank_pct_vec,
    "_roll_rank_apply": _roll_rank_apply,
    "_rank_pct": _rank_pct,
}


def factor_forge_gen4_gp_gen4_r1_ts_rank_ts_mean_ts_std_log_ts(close_df, extra_dfs=None):
    """(fast-IC 批量快验)
    公式: -(_roll_rank_apply((np.log(np.maximum(volume_p.rolling(10).sum(), 1e-8)).rolling(5).std().rolling(60).mean()), 10))
    全序列版本 (fast-IC): 返回完整 dates x stocks DataFrame"""
    nan_df = pd.DataFrame(np.nan, index=close_df.index, columns=close_df.columns)
    ex = extra_dfs or {}
    def _pick(key):
        df = ex.get(key)
        if df is None:
            return nan_df
        return df.reindex(index=close_df.index, columns=close_df.columns)
    _LOCALS = {
        "close": close_df, "open": _pick("open"), "high": _pick("high"),
        "low": _pick("low"), "volume": _pick("volume"), "amount": _pick("amount"),
        "close_p": close_df, "open_p": _pick("open"), "high_p": _pick("high"),
        "low_p": _pick("low"), "volume_p": _pick("volume"), "amount_p": _pick("amount"),
        "returns": close_df.pct_change(),
    }
    val = eval(_EXPR_factor_forge_gen4_gp_gen4_r1_ts_rank_ts_mean_ts_std_log_ts, _GLOBALS_factor_forge_gen4_gp_gen4_r1_ts_rank_ts_mean_ts_std_log_ts, _LOCALS)
    if isinstance(val, pd.Series):
        val = pd.DataFrame({'factor_forge_gen4_gp_gen4_r1_ts_rank_ts_mean_ts_std_log_ts': val}, index=close_df.index)
    if not isinstance(val, pd.DataFrame):
        val = pd.DataFrame(np.full(close_df.shape, float(val)),
                            index=close_df.index, columns=close_df.columns)
    return val.reindex(index=close_df.index, columns=close_df.columns)


FACTOR_LIST = [["forge_gen4_gp_gen4_r1_ts_rank(ts_mean(ts_std(log(ts", "factor_forge_gen4_gp_gen4_r1_ts_rank_ts_mean_ts_std_log_ts"]]
FACTOR_FIELDS = {"forge_gen4_gp_gen4_r1_ts_rank(ts_mean(ts_std(log(ts": ["volume"]}

def _spearman(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 20:
        return np.nan, int(m.sum())
    ra = rankdata(a[m])
    rb = rankdata(b[m])
    n = len(ra)
    ca = ra - ra.mean()
    cb = rb - rb.mean()
    va = float((ca * ca).sum())
    vb = float((cb * cb).sum())
    if va <= 0 or vb <= 0:
        return np.nan, n
    return float((ca * cb).sum()) / np.sqrt(va * vb), n

def initialize(context):
    log.info('[FAST-IC] start, building universe...')
    univ = []
    for idx in ["000300.XSHG", "000905.XSHG"]:
        try:
            univ += list(get_index_stocks(idx))
        except Exception as e:
            log.warn('[FAST-IC] index fail %s: %s' % (idx, e))
    univ = sorted(set(univ))
    if len(univ) < 300:
        log.error('[FAST-IC] universe too small (%d), abort' % len(univ))
        return
    log.info('[FAST-IC] universe=%d, end=%s, count=%d' % (len(univ), '2026-08-28', 1300))
    for name, fname in FACTOR_LIST:
        try:
            close = get_price(univ, end_date='2026-08-28', count=1300,
                              fields=['close'], fq='pre', skip_paused=False, panel=False)
            close = close.reindex(columns=univ)
            extras = {}
            for fld in FACTOR_FIELDS.get(name, []):
                if fld == 'close':
                    continue
                df = get_price(univ, end_date='2026-08-28', count=1300,
                               fields=[fld], fq='pre', skip_paused=False, panel=False)
                extras[fld] = df.reindex(index=close.index, columns=univ)
            val = globals()[fname](close, extras)
            fwd = close.shift(-5) / close - 1.0
            grid = np.arange(0, len(val), 5)
            ics = []
            for i in grid:
                ic, _n = _spearman(val.iloc[i].values, fwd.iloc[i].values)
                if np.isfinite(ic):
                    ics.append(ic)
            ics = np.asarray(ics, dtype=float)
            n_weeks = int(np.isfinite(ics).sum())
            if n_weeks >= 20:
                ic = float(np.nanmean(ics))
                sd = float(np.nanstd(ics, ddof=1))
                icir = ic / sd if sd > 0 else np.nan
            else:
                ic, icir = np.nan, np.nan
            hint = 'LONG' if ic > 0 else ('SHORT' if ic < 0 else 'FLAT')
            log.info('[FAST_IC] %s ic=%.4f icir=%.4f n=%d hint=%s'
                     % (name, ic, icir, n_weeks, hint))
        except Exception as e:
            log.info('[FAST_IC] %s ERROR %s' % (name, str(e)[:160]))

def handle_data(context, data):
    pass
