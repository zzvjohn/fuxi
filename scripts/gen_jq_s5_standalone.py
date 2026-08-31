# -*- coding: utf-8 -*-
"""
S5 通过因子 → 单因子 JQ 回测代码批量生成器
==========================================
仿照 jq_gp_breed_*_standalone.py 模板 (P-001 合规: 输出 per-factor Rank IC/ICIR)。
- 周频调仓, top-80 等权做多 (方向与 S5JointFilter.backtest_factor 一致: ascending=False 选最高因子值)
- 滞后 IC 法: 上周因子值 → 本周收益 spearman
- 防未来: get_price(end_date=context.previous_date)
- 停牌/涨停/ST 过滤; Panel 兼容; OrderCost+FixedSlippage

用法:
  python scripts/gen_jq_s5_standalone.py            # 生成所有注册因子
  python scripts/gen_jq_s5_standalone.py --name xxx  # 只生成指定因子
输出: research/factor_alchemy/jq_s5_pass_<name>_standalone.py
"""

import os
import re
import sys

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "research", "factor_alchemy")

# ════════════════════════════════════════════════════════════
# 因子注册表: 每项含 (JQ 函数体, 因子中文含义, lookback, local 指标注释)
# ════════════════════════════════════════════════════════════
FACTORS = {
    "max_drawdown_duration": {
        "meaning": "高点贴近度均值: (close/120日高点).20日均值 — 长期横盘后接近前高的股票, 突破势能强",
        "lookback": 160,
        "local_note": "IC=0.1246 ICIR=1.665 +IC%=93.3% | S5: excess_25=7.70% excess_26=4.37% Calmar_25=2.59 Calmar_26=1.69",
        "func": '''def factor_max_drawdown_duration(stocks, price_data):
    """(close / rolling_max(120)).rolling(20).mean() -- 高点贴近度; HIGH = 做多"""
        # 2026-08-17: 向量化契约兼容 shim — 从宽表 DataFrame 重建 price_data 字典
    price_data = {"close": {s: close_df[s].values for s in close_df.columns}}
    if extra_dfs:
        for _k, _df in extra_dfs.items():
            price_data[_k] = {s: _df[s].values for s in _df.columns}
    n = len(stocks)
    arr = np.full(n, np.nan); valid = np.zeros(n, dtype=bool)
    closed = price_data.get('close', {})
    for i, s in enumerate(stocks):
        close_arr = closed.get(s)
        if close_arr is None or len(close_arr) < 130: continue
        c = pd.Series(np.array(close_arr, dtype=float))
        try:
            roll_max = c.rolling(120).max()
            res = (c / roll_max).rolling(20).mean()
            v = res.dropna()
            arr[i] = v.iloc[-1] if len(v) > 0 else np.nan
            valid[i] = not pd.isna(arr[i])
        except Exception: continue
    return arr, valid''',
    },
    "small_cap_liquidity_quality": {
        "meaning": "小市值流动性改善: Amihud非流动性改善 × 小市值权重 × 20日动量门 — 优质小盘突破",
        "lookback": 110,
        "local_note": "IC=0.0299 ICIR=0.301 +IC%=60.5% | S5: excess_25=9.72% excess_26=7.49% Calmar_25=1.26 Calmar_26=2.46",
        "func": '''def factor_small_cap_liquidity_quality(stocks, price_data):
    """Amihud流动性改善 x 小市值权重 x 动量门 -- HIGH = 做多"""
        # 2026-08-17: 向量化契约兼容 shim — 从宽表 DataFrame 重建 price_data 字典
    price_data = {"close": {s: close_df[s].values for s in close_df.columns}}
    if extra_dfs:
        for _k, _df in extra_dfs.items():
            price_data[_k] = {s: _df[s].values for s in _df.columns}
    n = len(stocks)
    arr = np.full(n, np.nan); valid = np.zeros(n, dtype=bool)
    closed = price_data.get('close', {}); volumed = price_data.get('volume', {})
    # Step 1: 逐股票时间序列 (amihud 改善 x 动量门)
    ts_vals = {}
    for s in stocks:
        close_arr = closed.get(s); vol_arr = volumed.get(s)
        if close_arr is None or vol_arr is None or len(close_arr) < 70: continue
        c = pd.Series(np.array(close_arr, dtype=float))
        v = pd.Series(np.array(vol_arr, dtype=float))
        try:
            ret = c.pct_change().fillna(0)
            dv = (v * c).replace(0, np.nan)
            amihud = ret.abs() / dv.clip(lower=1e6)
            amihud_20 = amihud.rolling(20).mean()
            amihud_60 = amihud.rolling(60).mean()
            amihud_imp = -(amihud_20 / amihud_60.replace(0, np.nan) - 1).clip(-0.8, 0.5)
            mom_20 = c.pct_change(20).clip(-0.3, 0.3)
            mom_signal = (mom_20 > 0).astype(float)
            composite = (amihud_imp * mom_signal).rolling(5, min_periods=2).mean()
            mcap_proxy = (c.rolling(20).mean() * v.rolling(60).mean()).iloc[-1]
            fv = composite.iloc[-1]
            if pd.isna(fv): continue
            ts_vals[s] = (fv, mcap_proxy)
        except Exception: continue
    if len(ts_vals) < 10: return arr, valid
    # Step 2: 横截面市值排名 (代理市值越小, small_weight 越大)
    from scipy.stats import rankdata
    keys = list(ts_vals.keys())
    mcap_arr = np.array([ts_vals[s][1] for s in keys])
    mcap_ok = np.isfinite(mcap_arr)
    if int(mcap_ok.sum()) >= 2:
        ranks = rankdata(mcap_arr[mcap_ok])
        pct = (ranks - 1) / (len(ranks) - 1)
        small_w_all = np.full(len(keys), 1.0)
        small_w_all[mcap_ok] = np.clip(1.0 - pct, 0.2, 1.0)
    else:
        small_w_all = np.full(len(keys), 1.0)
    for j, s in enumerate(keys):
        arr[j] = ts_vals[s][0] * small_w_all[j]
        valid[j] = not pd.isna(arr[j])
    return arr, valid''',
    },
    "event_driven_convexity_fade": {
        "meaning": "事件拥挤退潮: 量能极端(>均量+2σ)近期爆发 × 5日动量追涨(>3%) → 做空拥挤泡沫, HIGH=无拥挤",
        "lookback": 90,
        "local_note": "IC=0.0279 ICIR=0.376 +IC%=63.2% | S5: excess_25=6.41% excess_26=3.87% Calmar_25=1.19 Calmar_26=2.00",
        "func": '''def factor_event_driven_convexity_fade(stocks, price_data):
    """-mean5(量能极端 x 动量追涨) -- 拥挤泡沫退潮; HIGH = 做多(无拥挤)"""
        # 2026-08-17: 向量化契约兼容 shim — 从宽表 DataFrame 重建 price_data 字典
    price_data = {"close": {s: close_df[s].values for s in close_df.columns}}
    if extra_dfs:
        for _k, _df in extra_dfs.items():
            price_data[_k] = {s: _df[s].values for s in _df.columns}
    n = len(stocks)
    arr = np.full(n, np.nan); valid = np.zeros(n, dtype=bool)
    closed = price_data.get('close', {}); volumed = price_data.get('volume', {})
    for i, s in enumerate(stocks):
        close_arr = closed.get(s); vol_arr = volumed.get(s)
        if close_arr is None or vol_arr is None or len(close_arr) < 50: continue
        c = pd.Series(np.array(close_arr, dtype=float))
        v = pd.Series(np.array(vol_arr, dtype=float))
        try:
            vol_ma20 = v.rolling(20).mean()
            vol_std20 = v.rolling(20).std()
            extreme = (v > vol_ma20 + 2.0 * vol_std20).astype(float)
            extreme_recent = extreme.rolling(3).max()
            mom_5 = c.pct_change(5).clip(-0.2, 0.2)
            crowd_chase = (mom_5 > 0.03).astype(float)
            bubble = extreme_recent * crowd_chase
            final = -(bubble.rolling(5, min_periods=2).mean())
            fv = final.iloc[-1]
            arr[i] = fv if not pd.isna(fv) else np.nan
            valid[i] = not pd.isna(arr[i])
        except Exception: continue
    return arr, valid''',
    },
    # ── Forge Round1 试验 S5 通过因子 (2026-08-13) ──
    # S5 回测口径: 见 data/forge_round1_s5_results.json 中 pandas_expression
    "forge_r1_vol_zscore_inv_stability": {
        "meaning": "量能zscore倒数波动稳定性: 1/(成交量12日zscore) 的20日波动率 → 5日均值 → 5日求和。量能持续平稳(不极端放量/缩量)的股票得分高。HIGH=做多",
        "lookback": 80,
        "local_note": "Forge ICIR=0.878 | S5: excess_25=3.95% excess_26=4.06% Calmar_25=1.29 Calmar_26=2.17",
        "extra_fields": ["volume"],
        "func": '''def factor_forge_r1_vol_zscore_inv_stability(stocks, price_data):
    """1/(volume 12日zscore) 的20日std -> 5日均值 -> 5日求和 -- 量能平稳性; HIGH = 做多"""
        # 2026-08-17: 向量化契约兼容 shim — 从宽表 DataFrame 重建 price_data 字典
    price_data = {"close": {s: close_df[s].values for s in close_df.columns}}
    if extra_dfs:
        for _k, _df in extra_dfs.items():
            price_data[_k] = {s: _df[s].values for s in _df.columns}
    n = len(stocks)
    arr = np.full(n, np.nan); valid = np.zeros(n, dtype=bool)
    volumed = price_data.get('volume', {})
    for i, s in enumerate(stocks):
        vol_arr = volumed.get(s)
        if vol_arr is None or len(vol_arr) < 45: continue
        v = pd.Series(np.array(vol_arr, dtype=float))
        try:
            z12 = (v - v.rolling(12).mean()) / (v.rolling(12).std() + 1e-6)
            invz = 1.0 / (z12 + 1e-6)
            res = invz.rolling(20).std().rolling(5).mean().rolling(5).sum()
            fv = res.iloc[-1]
            arr[i] = fv if not pd.isna(fv) else np.nan
            valid[i] = not pd.isna(arr[i])
        except Exception: continue
    return arr, valid''',
    },
    "forge_r1_vol_double_zscore_inv_stability": {
        "meaning": "成交量双层zscore倒数波动稳定性: 量能40日zscore再12日zscore → 取倒数 → 20日波动率 → 5日均值 → 5日求和。双重标准化后的量能平稳性。HIGH=做多",
        "lookback": 120,
        "local_note": "Forge ICIR=0.812 | S5: excess_25=3.31% excess_26=3.38% Calmar_25=1.30 Calmar_26=2.79",
        "extra_fields": ["volume"],
        "func": '''def factor_forge_r1_vol_double_zscore_inv_stability(stocks, price_data):
    """1/(zscore12(zscore40(volume))) 的20日std -> 5日均值 -> 5日求和; HIGH = 做多"""
        # 2026-08-17: 向量化契约兼容 shim — 从宽表 DataFrame 重建 price_data 字典
    price_data = {"close": {s: close_df[s].values for s in close_df.columns}}
    if extra_dfs:
        for _k, _df in extra_dfs.items():
            price_data[_k] = {s: _df[s].values for s in _df.columns}
    n = len(stocks)
    arr = np.full(n, np.nan); valid = np.zeros(n, dtype=bool)
    volumed = price_data.get('volume', {})
    for i, s in enumerate(stocks):
        vol_arr = volumed.get(s)
        if vol_arr is None or len(vol_arr) < 85: continue
        v = pd.Series(np.array(vol_arr, dtype=float))
        try:
            z40 = (v - v.rolling(40).mean()) / (v.rolling(40).std() + 1e-6)
            z12 = (z40 - z40.rolling(12).mean()) / (z40.rolling(12).std() + 1e-6)
            invz = 1.0 / (z12 + 1e-6)
            res = invz.rolling(20).std().rolling(5).mean().rolling(5).sum()
            fv = res.iloc[-1]
            arr[i] = fv if not pd.isna(fv) else np.nan
            valid[i] = not pd.isna(arr[i])
        except Exception: continue
    return arr, valid''',
    },
    "forge_r1_open_min_vol_squared": {
        "meaning": "价格下限波动平方: sqrt(30日最低价) 的20日波动率 → 60日zscore → 平方。S5验证时open为close代理, 故此处用close复刻同一口径。HIGH=做多",
        "lookback": 150,
        "local_note": "Forge ICIR=0.206 | S5: excess_25=4.87% excess_26=5.52% Calmar_25=2.25 Calmar_26=2.14 (open:=close代理口径)",
        "extra_fields": [],
        "func": '''def factor_forge_r1_open_min_vol_squared(stocks, close_df, extra_dfs=None):
    """square(zscore60(sqrt(rolling_min(close,30)).rolling(20).std()))
    S5验证时 open/high/low 均为 close 代理 -> 此处用 close 忠实复刻口径; HIGH = 做多"""
    # 2026-08-17: 向量化契约兼容 shim — 从宽表 DataFrame 重建 price_data 字典
    price_data = {"close": {s: close_df[s].values for s in close_df.columns}}
    if extra_dfs:
        for _k, _df in extra_dfs.items():
            price_data[_k] = {s: _df[s].values for s in _df.columns}
    n = len(stocks)
    arr = np.full(n, np.nan); valid = np.zeros(n, dtype=bool)
    closed = price_data.get('close', {})
    for i, s in enumerate(stocks):
        close_arr = closed.get(s)
        if close_arr is None or len(close_arr) < 115: continue
        c = pd.Series(np.array(close_arr, dtype=float))
        try:
            base = np.sqrt(np.maximum(c.rolling(30).min(), 0.0))
            sd = base.rolling(20).std()
            mu60 = sd.rolling(60, min_periods=20).mean()
            sg60 = sd.rolling(60, min_periods=20).std()
            z60 = (sd - mu60) / (sg60 + 1e-6)
            res = z60 * z60
            fv = res.iloc[-1]
            arr[i] = fv if not pd.isna(fv) else np.nan
            valid[i] = not pd.isna(arr[i])
        except Exception: continue
    return arr, valid''',
    },
    "forge_r1_volume_dual_rank": {
        "meaning": "成交量双重rank趋势: (3+成交量) 的30日分位 → 再26日分位。量能相对位置的二阶趋势。HIGH=做多",
        "lookback": 90,
        "local_note": "Forge ICIR=0.0 | S5: excess_25=3.93% excess_26=3.20% Calmar_25=1.34 Calmar_26=2.58",
        "extra_fields": ["volume"],
        "func": '''def _rolling_rank_pct_vec(a, window):
    """pandas 1.4+ Rolling.rank(pct=True, method='average') 的向量化等价实现
    JQ 平台老 pandas 没有 Rolling.rank, 用 stride 滑窗 + 秩统计替代; 窗口含 NaN -> 该位输出 NaN"""
    from numpy.lib.stride_tricks import as_strided
    n = len(a)
    out = np.full(n, np.nan)
    if n < window:
        return out
    m = n - window + 1
    x = as_strided(a, shape=(m, window), strides=(a.strides[0], a.strides[0]))
    last = x[:, -1][:, None]
    nan_win = np.isnan(x).any(axis=1)
    less = (x < last).sum(axis=1)
    eq = (x == last).sum(axis=1)          # 含自身, >= 1
    rank = less + (eq + 1.0) / 2.0        # method='average'
    pct = rank / float(window)
    pct[nan_win] = np.nan
    out[window - 1:] = pct
    return out


def factor_forge_r1_volume_dual_rank(stocks, close_df, extra_dfs=None):
    """(3+volume).rolling(30).rank(pct) 再 rolling(26).rank(pct) -- 成交量双重rank; HIGH = 做多
    JQ 兼容版: Rolling.rank 为 pandas>=1.4 API, 用 _rolling_rank_pct_vec 等价替代"""
    # 2026-08-17: 向量化契约兼容 shim — 从宽表 DataFrame 重建 price_data 字典
    price_data = {"close": {s: close_df[s].values for s in close_df.columns}}
    if extra_dfs:
        for _k, _df in extra_dfs.items():
            price_data[_k] = {s: _df[s].values for s in _df.columns}
    n = len(stocks)
    arr = np.full(n, np.nan); valid = np.zeros(n, dtype=bool)
    volumed = price_data.get('volume', {})
    for i, s in enumerate(stocks):
        vol_arr = volumed.get(s)
        if vol_arr is None or len(vol_arr) < 60: continue
        try:
            v = pd.Series(np.array(vol_arr, dtype=float))
            a = (3.0 + v).values.astype(float)  # 保留 NaN: 窗口含 NaN -> 该位 NaN (与 S5 本地验证口径一致)
            r30 = _rolling_rank_pct_vec(a, 30)
            res = _rolling_rank_pct_vec(r30, 26)
            fv = res[-1]
            arr[i] = fv if not np.isnan(fv) else np.nan
            valid[i] = not np.isnan(arr[i])
        except Exception: continue
    return arr, valid''',
    },
}


# ════════════════════════════════════════════════════════════
# 公共模板 (基于 jq_gp_breed_000_standalone.py, 已验证可过 JQ)
# ════════════════════════════════════════════════════════════
TEMPLATE = '''# -*- coding: utf-8 -*-
"""
@FNAME@ 单因子独立回测 (S5 通过因子)
因子含义: @MEANING@
Local: @LOCAL_NOTE@
生成时间: @GEN_TIME@ | 方向: HIGH=做多 (与 S5JointFilter.backtest_factor ascending=False 一致)
P-001 合规: 输出 per-factor Rank IC/ICIR
"""

import numpy as np
import pandas as pd

def initialize(context):
    g.stock_num = 80
    g.lookback = @LOOKBACK@
    g.trade_days = 0

    # --- IC 收集 (滞后法: 本周算上周因子→本周收益的秩相关) ---
    g.ic_records = []
    g.last_factor_stocks = []
    g.last_factor_values = None
    g.last_closes = {}
    g.ic_reported = False
    # --- 暴露监控 (P-20260814-004): 每期因子值横截面 std 时序 ---
    g.exposure_records = []   # [(week, held_std, universe_std), ...]

    set_benchmark("000905.XSHG")
    set_option("use_real_price", True)
    set_option("avoid_future_data", True)
    log.set_level("order", "error")
    set_order_cost(OrderCost(open_tax=0, close_tax=0.001,
                              open_commission=0.0001, close_commission=0.0001),
                   type="stock")
    set_slippage(FixedSlippage(0.003))
    run_weekly(rebalance, 1, time="open", reference_security="000300.XSHG")
    log.info("@FNAME@ 单因子 (S5通过, 滞后IC) 启动")


# ═══ 辅助 ═══
def rank_pct(arr, valid):
    n = len(arr)
    valid_arr = arr[valid]
    if len(valid_arr) < 3:
        result = np.full(n, 0.5); result[~valid] = 0.5; return result
    from scipy.stats import rankdata
    ranks = rankdata(valid_arr)
    pct = (ranks - 1) / (len(valid_arr) - 1)
    result = np.full(n, 0.5); result[valid] = pct; result[~valid] = 0.5
    return result


# ═══ 因子: @FNAME@ ═══
@FUNC@
# ═══ 因子映射 ═══
FACTOR_FUNC_MAP = {"@FNAME@": @FNAME_SAFE@}


# ═══ IC/ICIR 报告 ═══
def _print_ic_report():
    """P-001 合规: 输出单因子 Rank IC 和 ICIR"""
    if len(g.ic_records) == 0:
        log.info("[IC] @FNAME@: NO IC RECORDS")
        return

    ic_arr = np.array(g.ic_records)
    ic_mean = float(np.mean(ic_arr))
    ic_std  = float(np.std(ic_arr))
    icir    = ic_mean / ic_std if ic_std > 0 else 0.0
    ic_pos  = float(np.mean(ic_arr > 0))

    log.info("=" * 55)
    log.info("  @FNAME@  Per-Factor IC/ICIR Report (P-001)")
    log.info("=" * 55)
    log.info("  Rank IC mean:       %+.4f" % ic_mean)
    log.info("  Rank IC std:         %.4f" % ic_std)
    log.info("  ICIR (mean/std):    %+.4f" % icir)
    log.info("  IC > 0 ratio:       %.1f%%" % (ic_pos * 100))
    log.info("  Total IC samples:   %d" % len(ic_arr))
    log.info("=" * 55)


# ═══ 暴露监控报告 (P-20260814-004) ═══
def _print_exposure_report():
    """因子暴露波动率监控: 持仓股因子值横截面 std 的时序统计。

    背景: 2026-07 量化集体回撤 (幻方单月-22.15%), 先行指标 = 动量因子波动率翻数倍。
    用途: (1) 提前预警因子波动放大; (2) 供 D+ 蒸馏拥挤维度的归因输入。
    """
    if len(g.exposure_records) < 10:
        log.info("[EXPOSURE] @FNAME@: 样本不足 (%d < 10), 跳过暴露报告" % len(g.exposure_records))
        return
    held = np.array([e[1] for e in g.exposure_records], dtype=float)
    univ = np.array([e[2] for e in g.exposure_records], dtype=float)
    held = held[~np.isnan(held)]; univ = univ[~np.isnan(univ)]
    if len(held) < 10:
        return
    # 基线 = 前 20 期 (或前一半); 最新 = 最近 4 期均值
    base_n = min(20, max(4, len(held) // 2))
    base = held[:base_n]; recent = held[-4:]
    base_mean, recent_mean = float(np.mean(base)), float(np.mean(recent))
    ratio = recent_mean / base_mean if base_mean > 0 else float("nan")
    log.info("=" * 55)
    log.info("  @FNAME@  Factor Exposure Report (P-004)")
    log.info("=" * 55)
    log.info("  held std baseline(前%d期): %.4f" % (base_n, base_mean))
    log.info("  held std recent (近4期):   %.4f" % recent_mean)
    log.info("  universe std mean:         %.4f" % float(np.mean(univ)))
    log.info("  exposure vol ratio:        %.2fx" % ratio)
    if ratio >= 2.0:
        log.info("  [WARN] 因子暴露波动率放大 %.1fx! 拥挤/踩踏风险, 建议降权或暂停" % ratio)
    elif ratio >= 1.5:
        log.info("  [WATCH] 因子暴露波动率上升 %.1fx, 进入观察名单" % ratio)
    else:
        log.info("  [OK] 暴露波动率稳定")
    log.info("=" * 55)


# ═══ 周度调仓 ═══
def rebalance(context):
    import gc
    g.trade_days += 1
    if g.trade_days < 5:
        return

    prev_date = context.previous_date

    # ═══════════════════════════════════════════
    # Step 0: 股票池过滤 (lhb v2 修复版: 逐只 try, 单只异常剔除不污染池子)
    # ═══════════════════════════════════════════
    universe_all = get_all_securities(["stock"], prev_date).index.tolist()
    universe_all = [s for s in universe_all if s.startswith(("0","3","6"))][:2000]

    filtered = []
    try:
        cd = get_current_data()
        for s in universe_all:
            try:
                if cd[s].paused or cd[s].is_st:
                    continue
                if cd[s].high_limit <= cd[s].last_price:
                    continue
                filtered.append(s)
            except Exception:
                continue          # 单只数据异常 → 剔除, 不污染池子
    except Exception:
        filtered = list(universe_all)   # 整体异常才回退 (保持可用性)
    universe = [s for s in filtered if s.startswith(("0","3","6"))][:@UNIV_CAP@]
    if len(universe) < g.stock_num:
        return

    # ═══════════════════════════════════════════
    # Step 1: 单次 get_price(close) + 条件字段加载 (v3 性能骨架)
    #   原版 3 次全市场查询 (snap count=2 / close count=lb / volume count=lb)
    #   → 1 次 close + 按需 extra_fields; IC 快照复用本数据最后一行
    # ═══════════════════════════════════════════
    lb = g.lookback
    px_c = get_price(universe, count=lb, end_date=prev_date, frequency="daily",
                     fields="close", skip_paused=False, fq="pre")
    _P = getattr(pd, "Panel", None)
    if _P is not None and isinstance(px_c, _P):
        px_c = px_c["close"] if "close" in getattr(px_c, "items", []) else px_c.minor_xs("close")
    if px_c is None or px_c.shape[0] < 22 or px_c.shape[1] == 0:
        if not getattr(g, "diag_no_data", False):
            log.info("[DIAG] get_price 空/不足 at %s shape=%s — 数据加载问题?" % (
                prev_date, getattr(px_c, "shape", None)))
            g.diag_no_data = True
        return

    close_df = px_c.reindex(columns=universe)   # 缺失列 → NaN 列 → valid=False 自动排除
    valid_u = [s for s in universe
               if s in close_df.columns and int((~close_df[s].isna()).sum()) >= @MINLEN@]
    if len(valid_u) < g.stock_num:
        return

    extra_dfs = {}
@EXTRA_LOAD@
    # 2026-08-17: 因子计算异常必须打日志 (旧版 silent valid=0 掩盖了 per-stock eval 事故)
    try:
        arr, v = FACTOR_FUNC_MAP["@FNAME@"](valid_u, close_df, extra_dfs)
    except Exception as _fe:
        if not getattr(g, "diag_factor_err", False):
            log.info("[DIAG] @FNAME@ 因子计算异常: %r at %s (data shape=%s, univ=%d) — 公式与 JQ 环境不兼容?" % (
                _fe, prev_date, getattr(px_c, "shape", None), len(valid_u)))
            g.diag_factor_err = True
        return
    nv = int(np.sum(v))
    if nv < 10:
        if not getattr(g, "diag_low_valid", False):
            log.info("[DIAG] @FNAME@ valid=%d (<10) at %s — 不交易 (因子计算失败?)" % (nv, prev_date))
            g.diag_low_valid = True
        return

    rp = rank_pct(arr, v)

    # ═══════════════════════════════════════════
    # Step 1.5: 快照 + 滞后 IC 计算 (上周因子 → 本周收益)
    #   v3: 快照复用 px_c 最后一行 (prev_date 收盘), 不再单独全市场查询
    # ═══════════════════════════════════════════
    snap_closes = {}
    for s in px_c.columns:
        last_v = px_c[s].iloc[-1]
        if np.isfinite(last_v):
            snap_closes[s] = float(last_v)

    if len(g.last_factor_stocks) > 0 and g.last_factor_values is not None:
        fwd_rets = []
        factor_vals = []
        for i, s in enumerate(g.last_factor_stocks):
            prev_close = g.last_closes.get(s)
            curr_close = snap_closes.get(s)
            fv = g.last_factor_values[i]
            if (prev_close and prev_close > 0
                and curr_close and curr_close > 0
                and not np.isnan(fv)):
                ret = (curr_close / prev_close) - 1.0
                fwd_rets.append(ret)
                factor_vals.append(fv)

        if len(fwd_rets) >= 30:
            try:
                from scipy.stats import spearmanr
                ic, _ = spearmanr(factor_vals, fwd_rets)
                if ic == ic:               # 非 NaN
                    g.ic_records.append(float(ic))
            except Exception:
                pass

    # ═══════════════════════════════════════════
    # Step 2: 保存快照供下周 IC 计算
    # ═══════════════════════════════════════════
    g.last_factor_stocks = valid_u
    g.last_factor_values = rp.copy()
    g.last_closes = {}
    for s in valid_u:
        if s in close_df.columns and s in snap_closes:
            g.last_closes[s] = snap_closes[s]

    # ═══════════════════════════════════════════
    # Step 3: 选股 & 调仓 (HIGH = 做多: 取最高 rp)
    #   lhb v2 修复版: ①停牌股下单前再拦一道 ②平仓 order_target(s,0) 清仓语义
    #   ③开仓开盘价口径 ④现金不足提前 break
    # ═══════════════════════════════════════════
    comp = np.where(v, -rp, np.inf)
    topn = min(g.stock_num, int(np.sum(v)))
    top_idx = np.argsort(comp)[:topn]
    top_stocks = [valid_u[i] for i in top_idx if comp[i] < np.inf]

    # P-20260814-004: 记录本期暴露 (持仓股因子值 std + 全池因子值 std)
    try:
        held_vals = rp[top_idx][:len(top_stocks)]
        g.exposure_records.append((g.trade_days,
                                   float(np.std(held_vals)) if len(held_vals) > 1 else 0.0,
                                   float(np.std(rp[v])) if int(np.sum(v)) > 1 else 0.0))
        if len(g.exposure_records) > 500:  # 内存防御 (5.5年周频 ~290 期)
            g.exposure_records = g.exposure_records[-400:]
    except Exception:
        pass

    cd_order = get_current_data()

    # ── 平仓: 不在新持仓名单里的股票清仓 ──
    for s in list(context.portfolio.positions.keys()):
        if s not in top_stocks:
            try:
                if cd_order[s].paused:          # 停牌卖不出, 下周再试
                    continue
                pos = context.portfolio.positions[s]
                if pos.total_amount < 100:
                    continue
                order_target(s, 0)              # 清仓语义: 整单卖出含零股
            except Exception:
                pass

    if len(top_stocks) == 0:
        return

    # ── 开仓: 等权买入 ──
    w = context.portfolio.total_value / len(top_stocks)
    ordered = 0
    for s in top_stocks:
        try:
            if context.portfolio.available_cash < w * 0.5:
                break                           # 现金不足(前面平仓失败连锁), 停止开仓
            d = cd_order[s]
            if d.paused or d.is_st:             # 停牌/ST 不买
                continue
            if d.high_limit <= d.last_price:    # 已涨停买不进
                continue
            ref_px = d.day_open if (d.day_open and d.day_open > 0) else d.last_price
            if ref_px <= 0 or w / ref_px < 100: # 开盘价口径判断能否买100股
                continue
            order_target_value(s, w)
            ordered += 1
        except Exception:
            continue

    # ═══════════════════════════════════════════
    # Step 4: 日志 & 定期 IC 报告
    # ═══════════════════════════════════════════
    latest_ic = g.ic_records[-1] if g.ic_records else 0.0
    log.info("@FNAME@ | week=%d | ordered=%d/%d | stocks=%d | IC=%.4f | total_ic=%d" % (
        g.trade_days, ordered, len(top_stocks), int(np.sum(v)),
        latest_ic, len(g.ic_records)))

    if len(g.ic_records) >= 20 and g.trade_days % 26 == 0:
        _print_ic_report()
        _print_exposure_report()
        g.ic_reported = True

    gc.collect()
'''


def _fname_safe(name):
    """因子名 → JQ 因子函数名 (factor_<name>)"""
    return "factor_" + name


def generate(name):
    meta = FACTORS[name]
    from datetime import datetime
    code = TEMPLATE
    code = code.replace("@FNAME@", name)
    code = code.replace("@FNAME_SAFE@", _fname_safe(name))
    code = code.replace("@MEANING@", meta["meaning"])
    code = code.replace("@LOOKBACK@", str(meta["lookback"]))
    # 5.5万行/周 铁律反推池子规模 (注册因子 lookback 90-160, 池子相应收紧)
    nf = 1 + len(meta.get("extra_fields", []) or [])
    univ_cap = max(400, min(1500, int(55000 / (meta["lookback"] * nf) // 50 * 50)))
    code = code.replace("@UNIV_CAP@", str(univ_cap))
    code = code.replace("@LOCAL_NOTE@", meta["local_note"])
    code = code.replace("@GEN_TIME@", datetime.now().strftime("%Y-%m-%d %H:%M"))
    code = code.replace("@FUNC@", meta["func"].strip("\n"))
    code = code.replace("@MINLEN@", "50")

    # 按需加载额外字段 (如 volume) → extra_dfs (向量化契约)
    extra = ""
    for fld in meta.get("extra_fields", []) or []:
        extra += '''
    px_x = get_price(universe, count=lb, end_date=prev_date, frequency="daily",
                     fields="%s", skip_paused=False, fq="pre")
    if _P is not None and isinstance(px_x, _P):
        px_x = px_x["%s"] if "%s" in getattr(px_x, "items", []) else px_x.minor_xs("%s")
    if px_x is not None and px_x.shape[0] > 0 and px_x.shape[1] > 0:
        extra_dfs["%s"] = px_x.reindex(columns=universe)''' % (fld, fld, fld, fld, fld)
    code = code.replace("@EXTRA_LOAD@", extra)

    out_path = os.path.normpath(os.path.join(OUT_DIR, f"jq_s5_pass_{name}_standalone.py"))
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(code)
    return out_path


if __name__ == "__main__":
    only = None
    if "--name" in sys.argv:
        only = sys.argv[sys.argv.index("--name") + 1]

    for nm in FACTORS:
        if only and nm != only:
            continue
        p = generate(nm)
        # 本地语法校验
        with open(p, encoding="utf-8") as f:
            compile(f.read(), p, "exec")
        print(f"[OK] {os.path.basename(p)}")
    print("全部生成完毕")
