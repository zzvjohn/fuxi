# -*- coding: utf-8 -*-
"""
shadow_jq.py — JQ 影子并行评估文件生成器

一次 JQ 回测 = K 个候选复合袖子 + 6 个王者对照袖子并行模拟。
不开真实订单: 每周计算信号 → top20 等权虚拟组合 → 下周用真实价格实现收益。
输出: 每周 [SHADOW_RESULT] 全量表 (total_return/sharpe/maxdd per sleeve)。

关键设计:
  - 注入因子袖子使用本地 passed_factor_pool.csv 的真实公式翻译,
    不是 pipeline JQ 生成器的占位符 fallback (20260804_001621 的 bug:
    三个注入因子全是同一个"价格/MA20偏离"占位符 → -41pp 归因存疑)
  - 所有袖子共享同一 universe/同一价格数据/同一周 → 天然配对,
    袖子间差异剔除 market regime 噪声
  - 需要市场截面量的因子 (harvey/idio/diffusion/crowding/earnings)
    通过 price_data['_mkt_ret'] 缓存预计算一次
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

ROOT = Path(__file__).resolve().parents[3]
KING_FILE = ROOT / "research" / "factor_alchemy" / "output" / "fa_alpha_agent_v3_jq_20260802_130337.py"

# ── 从王者文件提取的因子 (真实实现, JQ已验证) ──────────────────
KING_FACTORS = [
    "compute_dollar_vol_20d", "compute_money_flow_20", "compute_overnight_5d",
    "compute_ret_3m", "compute_turnover_std_cv", "compute_tvma_20",
    "compute_nf_ff7913", "compute_f_", "compute_nf_02a304",
]
HELPERS = ["rank_pct", "zscore"]

# ── 袖子定义: (sleeve_id, factor_a, factor_b, group) ──────────
# group: king=王者对照, inj=注入因子(真实公式)
SLEEVES = [
    # ── 王者 6 复合 (对照组) ──
    ("king_comp1", "overnight_5d", "tvma_20", "king"),
    ("king_comp2", "dollar_vol_20d", "turnover_std_cv", "king"),
    ("king_comp3", "money_flow_20", "ret_3m", "king"),
    ("king_comp4", "nf_ff7913", "tvma_20", "king"),
    ("king_comp5", "f_", "dollar_vol_20d", "king"),
    ("king_comp6", "nf_02a304", "money_flow_20", "king"),
    # ── 注入因子 × 最佳互补 V3 伙伴 (真实公式翻译) ──
    ("inj_harvey_coskew", "harvey_siddique_coskew", "tvma_20", "inj"),
    ("inj_idio_tail_hedge", "idiosyncratic_tail_hedge_premium", "dollar_vol_20d", "inj"),
    ("inj_max_dd_duration", "max_drawdown_duration", "overnight_5d", "inj"),
    ("inj_capital_efficiency", "capital_efficiency_proxy", "dollar_vol_20d", "inj"),
    ("inj_hl_vol_spread", "hl_volatility_spread_regime_stable", "dollar_vol_20d", "inj"),
    ("inj_lottery_suppress", "lottery_demand_suppression_score", "tvma_20", "inj"),
    ("inj_earnings_pre_drift", "earnings_pre_drift_alignment", "turnover_std_cv", "inj"),
    ("inj_diffusion_mom", "diffusion_index_momentum", "dollar_vol_20d", "inj"),
    ("inj_vol_crowding_div", "volume_crowding_divergence", "overnight_5d", "inj"),
    ("inj_smallcap_liq_quality", "small_cap_liquidity_quality", "ret_3m", "inj"),
    ("inj_event_convexity", "event_driven_convexity_fade", "overnight_5d", "inj"),
]

# ── 注入因子真实实现 (从本地 passed_factor_pool.csv formula 翻译为 per-stock JQ) ──
# 统一约定: 返回值大 → 做多 (方向已在内部处理)
INJECTED_IMPLS = {
"harvey_siddique_coskew": '''
def compute_harvey_siddique_coskew(stocks, price_data):
    """协偏度: E[(ri-ui)(rm-um)^2]/(std_i*mean((rm-um)^2)), 60日.
    负协偏度=崩盘敏感=风险补偿 → 信号=-coskew (Harvey-Siddique 2000)"""
    import numpy as np
    n = len(stocks)
    close = price_data.get('close', {})
    mkt = _ensure_mkt_ret(stocks, price_data, 61)
    arr = np.full(n, np.nan)
    valid = np.zeros(n, dtype=bool)
    for i, s in enumerate(stocks):
        c = close.get(s)
        if c is None or len(c) < 61 or mkt is None:
            continue
        c = np.array(c[-61:], dtype=float)
        ri = np.diff(c) / np.where(c[:-1] == 0, np.nan, c[:-1])
        rm = mkt[-60:]
        k = min(len(ri), len(rm))
        if k < 40:
            continue
        ri, rm = ri[-k:], rm[-k:]
        ok = ~(np.isnan(ri) | np.isnan(rm))
        if int(np.sum(ok)) < 40:
            continue
        ri, rm = ri[ok], rm[ok]
        ri_c = ri - np.mean(ri)
        rm_c = rm - np.mean(rm)
        denom = np.std(ri) * np.mean(rm_c ** 2)
        if denom < 1e-12:
            continue
        coskew = np.mean(ri_c * rm_c ** 2) / denom
        arr[i] = -coskew
        valid[i] = True
    return arr, valid
''',
"idiosyncratic_tail_hedge_premium": '''
def compute_idiosyncratic_tail_hedge_premium(stocks, price_data):
    """CAPM残差尾部: 60日beta回归残差的最差5日均值.
    特质左尾越深→补偿越高 → 信号=-worst5_mean"""
    import numpy as np
    n = len(stocks)
    close = price_data.get('close', {})
    mkt = _ensure_mkt_ret(stocks, price_data, 61)
    arr = np.full(n, np.nan)
    valid = np.zeros(n, dtype=bool)
    for i, s in enumerate(stocks):
        c = close.get(s)
        if c is None or len(c) < 61 or mkt is None:
            continue
        c = np.array(c[-61:], dtype=float)
        ri = np.diff(c) / np.where(c[:-1] == 0, np.nan, c[:-1])
        rm = mkt[-60:]
        k = min(len(ri), len(rm))
        if k < 40:
            continue
        ri, rm = ri[-k:], rm[-k:]
        ok = ~(np.isnan(ri) | np.isnan(rm))
        if int(np.sum(ok)) < 40:
            continue
        ri, rm = ri[ok], rm[ok]
        mvar = np.var(rm)
        if mvar < 1e-12:
            continue
        beta = np.cov(ri, rm)[0, 1] / mvar
        resid = ri - beta * rm
        worst5 = np.mean(np.sort(resid)[:5])
        arr[i] = -worst5
        valid[i] = True
    return arr, valid
''',
"max_drawdown_duration": '''
def compute_max_drawdown_duration(stocks, price_data):
    """回撤恢复度: mean_20(close / max_120(close)).
    已从长期回撤中修复的股票(韌性信號)有alpha → 值大=修复好=做多"""
    import numpy as np
    n = len(stocks)
    close = price_data.get('close', {})
    arr = np.full(n, np.nan)
    valid = np.zeros(n, dtype=bool)
    for i, s in enumerate(stocks):
        c = close.get(s)
        if c is None or len(c) < 120:
            continue
        c = np.array(c[-120:], dtype=float)
        cmax = np.nanmax(c)
        if not np.isfinite(cmax) or cmax <= 0:
            continue
        ratio = c / cmax
        arr[i] = float(np.nanmean(ratio[-20:]))
        valid[i] = True
    return arr, valid
''',
"capital_efficiency_proxy": '''
def compute_capital_efficiency_proxy(stocks, price_data):
    """资本效率: |ret_20| / mean(dollar_vol_20) * 1e8.
    单位成交额产出的价格变动效率 (Amihud的反面)"""
    import numpy as np
    n = len(stocks)
    close = price_data.get('close', {})
    volume = price_data.get('volume', {})
    arr = np.full(n, np.nan)
    valid = np.zeros(n, dtype=bool)
    for i, s in enumerate(stocks):
        c, v = close.get(s), volume.get(s)
        if c is None or v is None or len(c) < 41 or len(v) < 41:
            continue
        c = np.array(c[-41:], dtype=float)
        v = np.array(v[-41:], dtype=float)
        ret20 = c[-1] / c[-21] - 1.0 if c[-21] > 0 else np.nan
        dv = np.nanmean(v[-20:] * c[-20:])
        if not np.isfinite(ret20) or not np.isfinite(dv) or dv <= 0:
            continue
        arr[i] = abs(ret20) / dv * 1e8
        valid[i] = True
    return arr, valid
''',
"hl_volatility_spread_regime_stable": '''
def compute_hl_volatility_spread_regime_stable(stocks, price_data):
    """Parkinson高低价波动率 vs 收盘波动率的价差比率 (20日).
    vm_diff毒因子替代: (parkinson-close_vol)/(parkinson+close_vol)"""
    import numpy as np
    n = len(stocks)
    close = price_data.get('close', {})
    high = price_data.get('high', {})
    low = price_data.get('low', {})
    arr = np.full(n, np.nan)
    valid = np.zeros(n, dtype=bool)
    for i, s in enumerate(stocks):
        c, h, l = close.get(s), high.get(s), low.get(s)
        if c is None or h is None or l is None or len(c) < 25:
            continue
        c = np.array(c[-25:], dtype=float)
        h = np.array(h[-25:], dtype=float)
        l = np.array(l[-25:], dtype=float)
        l_safe = np.where(l <= 0, np.nan, l)
        hl_ratio = np.log(h / l_safe)
        park = np.nanmean((hl_ratio ** 2) / (4 * np.log(2)))
        park = np.sqrt(max(park, 0.0))
        ri = np.diff(c[-21:]) / np.where(c[-21:-1] == 0, np.nan, c[-21:-1])
        cv = np.nanstd(ri)
        if not np.isfinite(park) or not np.isfinite(cv) or (park + cv) < 1e-12:
            continue
        arr[i] = (park - cv) / (park + cv)
        valid[i] = True
    return arr, valid
''',
"lottery_demand_suppression_score": '''
def compute_lottery_demand_suppression_score(stocks, price_data):
    """MAX效应: 过去20日最大单日收益取负.
    高MAX=彩票需求→高估 → 做多低MAX (信号=-max_ret_20d)"""
    import numpy as np
    n = len(stocks)
    close = price_data.get('close', {})
    arr = np.full(n, np.nan)
    valid = np.zeros(n, dtype=bool)
    for i, s in enumerate(stocks):
        c = close.get(s)
        if c is None or len(c) < 21:
            continue
        c = np.array(c[-21:], dtype=float)
        ri = np.diff(c) / np.where(c[:-1] == 0, np.nan, c[:-1])
        ri = ri[~np.isnan(ri)]
        if len(ri) < 15:
            continue
        arr[i] = -float(np.max(ri))
        valid[i] = True
    return arr, valid
''',
"earnings_pre_drift_alignment": '''
def compute_earnings_pre_drift_alignment(stocks, price_data):
    """盈余前动量对齐: 5日超额动量(vs截面) x 量能趋势(vol_ma5/vol_ma20).
    强者恒强在量能确认下加速"""
    import numpy as np
    n = len(stocks)
    close = price_data.get('close', {})
    volume = price_data.get('volume', {})
    # 截面5日动量均值 (预计算)
    mom5_all = []
    for s in stocks[:800]:
        c = close.get(s)
        if c is not None and len(c) >= 6 and c[-6] > 0:
            mom5_all.append(c[-1] / c[-6] - 1.0)
    xsec_mom = float(np.nanmean(mom5_all)) if mom5_all else 0.0
    arr = np.full(n, np.nan)
    valid = np.zeros(n, dtype=bool)
    for i, s in enumerate(stocks):
        c, v = close.get(s), volume.get(s)
        if c is None or v is None or len(c) < 21 or len(v) < 21:
            continue
        c = np.array(c[-21:], dtype=float)
        v = np.array(v[-21:], dtype=float)
        if c[-6] <= 0:
            continue
        mom5 = c[-1] / c[-6] - 1.0
        vma5, vma20 = np.nanmean(v[-5:]), np.nanmean(v[-20:])
        if vma20 <= 0:
            continue
        vol_trend = vma5 / vma20
        arr[i] = (mom5 - xsec_mom) * vol_trend
        valid[i] = True
    return arr, valid
''',
"diffusion_index_momentum": '''
def compute_diffusion_index_momentum(stocks, price_data):
    """扩散指数动量: 个股20日动量 x 市场扩散度z-score.
    扩散度=市场above_ma20比例的60日z (截面预计算缓存)"""
    import numpy as np
    n = len(stocks)
    close = price_data.get('close', {})
    dz = _ensure_diffusion_z(stocks, price_data)
    arr = np.full(n, np.nan)
    valid = np.zeros(n, dtype=bool)
    for i, s in enumerate(stocks):
        c = close.get(s)
        if c is None or len(c) < 21 or dz is None:
            continue
        c = np.array(c[-21:], dtype=float)
        if c[-21] <= 0:
            continue
        mom20 = c[-1] / c[-21] - 1.0
        arr[i] = mom20 * dz
        valid[i] = True
    return arr, valid
''',
"volume_crowding_divergence": '''
def compute_volume_crowding_divergence(stocks, price_data):
    """成交量拥挤度背离: 个股vol_z(ma20 vs ma60) - 截面均值, clip±2"""
    import numpy as np
    n = len(stocks)
    volume = price_data.get('volume', {})
    # 第一遍: 算所有股票 vol_z 求截面均值
    zs = {}
    zvals = []
    for s in stocks:
        v = volume.get(s)
        if v is None or len(v) < 60:
            continue
        v = np.array(v[-60:], dtype=float)
        ma20, ma60 = np.nanmean(v[-20:]), np.nanmean(v)
        if ma60 <= 0:
            continue
        z = (ma20 - ma60) / ma60
        zs[s] = z
        zvals.append(z)
    xsec = float(np.nanmean(zvals)) if zvals else 0.0
    arr = np.full(n, np.nan)
    valid = np.zeros(n, dtype=bool)
    for i, s in enumerate(stocks):
        if s not in zs:
            continue
        arr[i] = float(np.clip(zs[s] - xsec, -2, 2))
        valid[i] = True
    return arr, valid
''',
"small_cap_liquidity_quality": '''
def compute_small_cap_liquidity_quality(stocks, price_data):
    """小盘流动性质量: -(amihud_20/amihud_60 - 1).
    amihud改善(流动性变好)的小盘股在突破时弹性更强"""
    import numpy as np
    n = len(stocks)
    close = price_data.get('close', {})
    volume = price_data.get('volume', {})
    arr = np.full(n, np.nan)
    valid = np.zeros(n, dtype=bool)
    for i, s in enumerate(stocks):
        c, v = close.get(s), volume.get(s)
        if c is None or v is None or len(c) < 61 or len(v) < 61:
            continue
        c = np.array(c[-61:], dtype=float)
        v = np.array(v[-61:], dtype=float)
        ri = np.abs(np.diff(c) / np.where(c[:-1] == 0, np.nan, c[:-1]))
        dv = v[1:] * c[1:]
        dv = np.where(dv < 1e6, np.nan, dv)
        amihud = ri / dv
        a20, a60 = np.nanmean(amihud[-20:]), np.nanmean(amihud)
        if not np.isfinite(a20) or not np.isfinite(a60) or a60 <= 0:
            continue
        arr[i] = -(a20 / a60 - 1.0)
        valid[i] = True
    return arr, valid
''',
"event_driven_convexity_fade": '''
def compute_event_driven_convexity_fade(stocks, price_data):
    """事件凸性退潮: 极端放量(>ma20+2std)后3日内5日动量取负.
    事件冲击后追涨人群被反向 → 信号=-(mom5 x extreme_recent)"""
    import numpy as np
    n = len(stocks)
    close = price_data.get('close', {})
    volume = price_data.get('volume', {})
    arr = np.full(n, np.nan)
    valid = np.zeros(n, dtype=bool)
    for i, s in enumerate(stocks):
        c, v = close.get(s), volume.get(s)
        if c is None or v is None or len(c) < 26 or len(v) < 26:
            continue
        c = np.array(c[-26:], dtype=float)
        v = np.array(v[-26:], dtype=float)
        ma20, sd20 = np.nanmean(v[-25:-5]), np.nanstd(v[-25:-5])
        extreme_recent = 1.0 if np.nanmax(v[-3:]) > ma20 + 2.0 * sd20 else 0.0
        if c[-6] <= 0:
            continue
        mom5 = float(np.clip(c[-1] / c[-6] - 1.0, -0.2, 0.2))
        arr[i] = -mom5 * extreme_recent
        valid[i] = True
    return arr, valid
''',
}

# ── 引擎模板片段 ─────────────────────────────────────────────
MKT_HELPERS = '''
def _ensure_mkt_ret(stocks, price_data, need):
    """等权市场日收益, 缓存到 price_data['_mkt_ret'] (只算一次)"""
    import numpy as np
    cached = price_data.get('_mkt_ret')
    if cached is not None and len(cached) >= need - 1:
        return cached
    close = price_data.get('close', {})
    rets = []
    for s in stocks[:800]:
        c = close.get(s)
        if c is None or len(c) < need:
            continue
        c = np.array(c[-need:], dtype=float)
        r = np.diff(c) / np.where(c[:-1] == 0, np.nan, c[:-1])
        rets.append(r)
    if len(rets) < 20:
        price_data['_mkt_ret'] = None
        return None
    mkt = np.nanmean(np.array(rets), axis=0)
    price_data['_mkt_ret'] = mkt
    return mkt

def _ensure_diffusion_z(stocks, price_data):
    """市场扩散度(above_ma20比例)的60日z-score, 缓存.
    注: 单时点横截面无60日历史, 用截面扩散度相对0.5的偏离近似"""
    import numpy as np
    cached = price_data.get('_diff_z')
    if cached is not None:
        return cached
    close = price_data.get('close', {})
    above = 0
    total = 0
    for s in stocks[:800]:
        c = close.get(s)
        if c is None or len(c) < 20:
            continue
        ma20 = np.nanmean(np.array(c[-20:], dtype=float))
        if ma20 > 0:
            total += 1
            if c[-1] > ma20:
                above += 1
    if total < 50:
        price_data['_diff_z'] = None
        return None
    diff = above / float(total)
    z = float(np.clip((diff - 0.5) / 0.15, -2, 2))
    price_data['_diff_z'] = z
    return z
'''

ENGINE = '''
def initialize(context):
    import numpy as np
    g.stock_num = 20          # 每袖子 top20 (小组合, 信号区分度高)
    g.warmup_weeks = 5
    g.trade_days = 0
    g.sleeve_nav = {__NAV_INIT__}
    g.sleeve_prev = {__PREV_INIT__}   # 上周持仓
    g.sleeve_weekly_rets = {__RETS_INIT__}
    g.sleeve_cost = {__COST_INIT__}   # 本周换手成本率 (下周实现时扣除)
    g.lookback = 130          # max_drawdown_duration 需要120日窗口
    set_benchmark('000300.XSHG')
    set_option('use_real_price', True)
    set_option('avoid_future_data', True)
    log.set_level('order', 'error')
    run_weekly(shadow_run, 1, time='open', reference_security='000300.XSHG')

SLEEVE_DEFS = __SLEEVE_DEFS__

def _sleeve_stats(nav_series):
    import numpy as np
    if len(nav_series) < 3:
        return 0.0, 0.0, 0.0
    nav = np.array(nav_series, dtype=float)
    rets = np.diff(nav) / nav[:-1]
    total = nav[-1] / nav[0] - 1.0
    sd = np.std(rets)
    sharpe = float(np.mean(rets) / sd * np.sqrt(52)) if sd > 1e-12 else 0.0
    peak = np.maximum.accumulate(nav)
    mdd = float(np.min(nav / peak - 1.0))
    return float(total), sharpe, mdd

def shadow_run(context):
    import numpy as np
    import gc
    g.trade_days += 1
    prev_date = context.previous_date

    # ── Universe (与王者同一过滤) ──
    universe = get_all_securities(['stock'], prev_date).index.tolist()
    cd = get_current_data()
    CHUNK = 500
    filtered = []
    for k in range(0, len(universe), CHUNK):
        chunk = universe[k:k+CHUNK]
        try:
            for s in chunk:
                if cd[s].paused:
                    continue
                if cd[s].is_st:
                    continue
                if cd[s].high_limit <= cd[s].last_price:
                    continue
                filtered.append(s)
        except Exception:
            filtered.extend(chunk)
    universe = [s for s in filtered if s.startswith(('0', '3', '6'))][:2000]
    if len(universe) < g.stock_num * 3:
        return

    # ── 价格数据 (与王者同一取法 + Panel兼容) ──
    lb = g.lookback
    px = {}
    for field in ['close', 'open', 'high', 'low', 'volume']:
        d = get_price(universe, count=lb, end_date=prev_date,
                      frequency='daily', fields=field,
                      skip_paused=False, fq='pre')
        _Panel = getattr(pd, 'Panel', None)
        if _Panel is not None and isinstance(d, _Panel):
            d = d[field] if field in getattr(d, 'items', []) else d.minor_xs(field)
        px[field] = d
    price_data = {
        f: {s: px[f][s].values for s in universe if s in px[f].columns}
        for f in ['close', 'open', 'high', 'low', 'volume']
    }
    del px
    gc.collect()

    valid_universe = [s for s in universe
                      if len(price_data['close'].get(s, [])) >= 125]
    if len(valid_universe) < g.stock_num * 2:
        return

    # ── 1. 实现上周持仓收益 (周频 close-to-close) ──
    if g.trade_days > 1:
        for sid, a_name, b_name in SLEEVE_DEFS:
            prev_hold = g.sleeve_prev.get(sid, [])
            if not prev_hold:
                continue
            rets = []
            for s in prev_hold:
                c = price_data['close'].get(s)
                if c is None or len(c) < 6:
                    continue
                c6 = np.array(c[-6:], dtype=float)
                if c6[0] > 0:
                    rets.append(c6[-1] / c6[0] - 1.0)
            if not rets:
                continue
            gross = float(np.mean(rets))
            # 换手成本: 与上周持仓不重叠部分 x 0.0072 (买卖双边)
            gross_ret = gross
            nav = g.sleeve_nav[sid]
            new_nav = nav[-1] * (1.0 + gross_ret - g.sleeve_cost.get(sid, 0.0))
            nav.append(new_nav)
            g.sleeve_weekly_rets[sid].append(gross_ret - g.sleeve_cost.get(sid, 0.0))

    if g.trade_days < g.warmup_weeks:
        # warmup 期: 只建仓不统计 (信号窗口不足)
        for sid, a_name, b_name in SLEEVE_DEFS:
            g.sleeve_prev[sid] = []
        # 仍需要跑一遍因子计算让 mkt 缓存就绪, 跳过
        return

    # ── 2. 计算各袖子信号 → 新持仓 ──
    factor_cache = {}
    for sid, a_name, b_name in SLEEVE_DEFS:
        if a_name not in factor_cache:
            factor_cache[a_name] = FACTOR_FUNCS[a_name](valid_universe, price_data)
        if b_name not in factor_cache:
            factor_cache[b_name] = FACTOR_FUNCS[b_name](valid_universe, price_data)
        a, va = factor_cache[a_name]
        b, vb = factor_cache[b_name]
        valid = va & vb
        ra = rank_pct(a, valid)
        rb = rank_pct(b, valid)
        # NaN 防护: 停牌日数据缺失导致因子值 NaN, 若进入 score,
        # argsort 会把 NaN 排在最前占掉 top-N 槽位 → 袖子死亡 (v1 comp3/comp6 教训)
        raw = ra * rb
        raw = np.where(np.isnan(raw), -1.0, raw)
        score = np.where(valid, raw, -1.0)
        top_idx = np.argsort(score)[-g.stock_num:][::-1]
        new_hold = [valid_universe[i] for i in top_idx if score[i] > -1]
        # 换手率 → 下周成本
        prev_hold = set(g.sleeve_prev.get(sid, []))
        if prev_hold and new_hold:
            overlap = len(prev_hold & set(new_hold))
            turnover = 1.0 - overlap / float(g.stock_num)
        else:
            turnover = 1.0 if new_hold else 0.0
        g.sleeve_cost[sid] = turnover * 0.0072
        g.sleeve_prev[sid] = new_hold

    # ── 3. 每周输出全量统计 (看最后一行即可) ──
    parts = []
    for sid, a_name, b_name in SLEEVE_DEFS:
        total, sharpe, mdd = _sleeve_stats(g.sleeve_nav[sid])
        parts.append('%s=%.3f/%.2f/%.3f' % (sid, total, sharpe, mdd))
    log.info('[SHADOW_RESULT] week=%d %s' % (g.trade_days, ' '.join(parts)))
'''


def _extract_functions(source_text, func_names):
    """从 JQ 文件源码中提取指定函数定义文本."""
    out = {}
    for name in func_names:
        m = re.search(
            rf"(def {re.escape(name)}\(.*?\n(?:    .*\n|\n)*?)(?=\ndef |\Z)",
            source_text,
        )
        if not m:
            raise ValueError(f"函数提取失败: {name}")
        out[name] = m.group(1).rstrip() + "\n"
    return out


def generate_shadow_jq(out_path=None):
    king_src = KING_FILE.read_text(encoding="utf-8")

    # 提取 helper + 王者因子实现
    helpers = _extract_functions(king_src, HELPERS)
    king_factors = _extract_functions(king_src, KING_FACTORS)

    # money_flow_20: 完全手写, 绕过提取 (v1/v2/v3 均死于提取版, comp3/comp6 恒为 0)
    # 逻辑: 典型价格×成交额 的 20 日均值 vs 当日值的变化率取负
    king_factors["compute_money_flow_20"] = '''
def compute_money_flow_20(stocks, price_data):
    """资金流偏移(取负) — 手写版, 绕过源文件提取"""
    import numpy as np
    high = price_data.get('high', {})
    low = price_data.get('low', {})
    close = price_data.get('close', {})
    volume = price_data.get('volume', {})
    n = len(stocks)
    arr = np.full(n, np.nan)
    valid = np.zeros(n, dtype=bool)
    for i, s in enumerate(stocks):
        h_arr = high.get(s)
        l_arr = low.get(s)
        c_arr = close.get(s)
        v_arr = volume.get(s)
        if h_arr is None or l_arr is None or c_arr is None or v_arr is None:
            continue
        if len(h_arr) < 25 or len(l_arr) < 25 or len(c_arr) < 25 or len(v_arr) < 25:
            continue
        h = np.array(h_arr[-25:], dtype=float)
        l_ = np.array(l_arr[-25:], dtype=float)
        c = np.array(c_arr[-25:], dtype=float)
        v = np.array(v_arr[-25:], dtype=float)
        # typical price weighted by volume: (H+L+C)/3 * V
        tp_hist = (h[-20:] + l_[-20:] + c[-20:]) * v[-20:] / 3.0
        tp_curr = (h[-1] + l_[-1] + c[-1]) * v[-1] / 3.0
        tp_mean = np.nanmean(tp_hist)  # nanmean 跳过停牌 NaN
        if not np.isfinite(tp_mean) or tp_mean == 0:
            continue
        arr[i] = -(tp_curr - tp_mean)  # 绝对偏移取负 (与原始王者公式一致)
        valid[i] = True
    return arr, valid
'''

    # 收集袖子需要的全部因子
    needed = set()
    for _, a, b, _grp in SLEEVES:
        needed.add(a)
        needed.add(b)

    funcs = []
    funcs.extend(helpers.values())
    factor_names = []
    for name in sorted(needed):
        cname = f"compute_{name}"
        if cname in king_factors:
            funcs.append(king_factors[cname])
        elif name in INJECTED_IMPLS:
            funcs.append(INJECTED_IMPLS[name].strip() + "\n")
        else:
            raise ValueError(f"无实现: {name}")
        factor_names.append(name)

    # 引擎占位符替换
    nav_init = ", ".join(f"'{sid}': [1.0]" for sid, _, _, _ in SLEEVES)
    prev_init = ", ".join(f"'{sid}': []" for sid, _, _, _ in SLEEVES)
    rets_init = ", ".join(f"'{sid}': []" for sid, _, _, _ in SLEEVES)
    cost_init = ", ".join(f"'{sid}': 0.0" for sid, _, _, _ in SLEEVES)
    sleeve_defs_repr = "(" + ", ".join(
        f"('{sid}', '{a}', '{b}')" for sid, a, b, _ in SLEEVES
    ) + ")"

    engine = (ENGINE
              .replace("__NAV_INIT__", nav_init)
              .replace("__PREV_INIT__", prev_init)
              .replace("__RETS_INIT__", rets_init)
              .replace("__COST_INIT__", cost_init)
              .replace("__SLEEVE_DEFS__", sleeve_defs_repr))

    # FACTOR_FUNCS 映射
    func_map = "FACTOR_FUNCS = {\n" + "\n".join(
        f"    '{name}': compute_{name}," for name in factor_names
    ) + "\n}\n"

    header = (
        "# -*- coding: utf-8 -*-\n"
        "# SHADOW 并行评估: 17袖子 (6王者对照 + 11注入真实公式) 虚拟组合模拟\n"
        "# 由 evaluator/shadow_jq.py 生成; 每周输出 [SHADOW_RESULT] 全量表\n"
        "# 注意: 本文件不开真实订单, 仅用于候选复合的配对评估\n"
        "import datetime\n"
        "import pandas as pd\n"
        "import numpy as np\n"
    )

    full = (header + "\n"
            + "\n".join(funcs) + "\n"
            + MKT_HELPERS + "\n"
            + func_map + "\n"
            + engine)

    # 语法验证 (JQ 铁律)
    compile(full, str(out_path or "shadow_jq.py"), "exec")

    if out_path:
        Path(out_path).write_text(full, encoding="utf-8")
        print(f"✅ 影子评估文件已生成: {out_path}")
        print(f"   袖子数: {len(SLEEVES)} (king=6, inj=11)")
        print(f"   因子数: {len(factor_names)} | 文件大小: {len(full)/1024:.1f}KB")
    return full


if __name__ == "__main__":
    out = ROOT / "research" / "factor_alchemy" / "output" / "shadow_eval_v2.py"
    generate_shadow_jq(out)
