# -*- coding: utf-8 -*-
"""
V4 Paradigm JQ Strategy v2: 王者底仓 + V4范式扩展
==================================================
生成于: 2026-08-05 (v2 fix: 2026-08-05 01:30)
来源: V4 Pipeline (paradigm_v4 → library_orthogonality → factor_model_cooptim)

策略逻辑:
  - 底仓: 保留王者4个最强复合 (overnight×tvma / dollar_vol×turnover_std / nf_ff7913×tvma / f_×dollar_vol)
  - 扩展: 新增3个V4范式复合覆盖未覆盖领域:
      V4-A: 情绪×日内   — intraday_amplitude × overnight_reversal
      V4-B: 截面交互     — volume_concentration × money_flow
      V4-C: 行业轮动     — industry_relative_strength × ret_3m
  - Ensemble: 等权 rank-product (7个复合乘积)
  - 频率: 周频, warmup=5周

v2 修复 (2026-08-05):
  - FIX#1: g.initial_capital → context.portfolio.total_value (组合增长后资金不闲置)
  - FIX#2: 增加 100股最小检查 + skip逻辑 (消除"开仓数量不能小于 100"错误)

v3 修复 (2026-08-05 02:43):
  - FIX#3: 买侧改用 order_target (份额制) 替代 order_target_value (价值制) — 消除 value→shares→delta 二次转换中的 <100 漏洞
  - FIX#4: 增加 delta-aware 跳过: 目标与现有持仓差异 < 100股时维持现状
  - FIX#5: 卖侧改用 order_target(s, 0) 份额制清仓
  - FIX#3: 增加 ordered/skipped 日志 (可监控下单成功率)

V4 诊断依据:
  - 8/21范式未覆盖 → 优先选择高优先级(情绪×日内, 截面交互, 行业轮动)
  - Red Sea: 🟢GREEN (中位corr 0.038) → 库远未饱和, 新范式无撞墙风险
  - 禁止区域: 0 → 无已知失败区域

王者基准: +182.57% / Sharpe 0.69 / MaxDD -35.41%
v1 结果: +116.32% / Sharpe 0.73 / MaxDD -22.89% (资金闲置+委托失败 → Sharpe虚高)
"""

import datetime
import pandas as pd
from jqlib.technical_analysis import *


def initialize(context):
    g.stock_num = 80
    g.limit_pct = 0.10
    g.trade_days = 0
    set_benchmark('000300.XSHG')
    set_option('use_real_price', True)
    log.set_level('order', 'error')
    log.set_level("system", "error")  # P-20260902-016: 抑制 order 函数内部仓位查询的 WARNING ("在 positions 中不存在" 刷屏)
    log.set_level('strategy', 'info')
    set_order_cost(OrderCost(open_tax=0, close_tax=0.001,
                              open_commission=0.0001, close_commission=0.0001),
                   type='stock')
    set_slippage(FixedSlippage(0.003))
    run_weekly(rebalance, 1, time='open', reference_security='000300.XSHG')


# ═══════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════

def rank_pct(arr, valid):
    '''截面百分位排名, NaN被忽略'''
    import numpy as np
    n = len(arr)
    valid_arr = arr[valid]
    if len(valid_arr) < 3:
        result = np.full(n, 0.5)
        result[~valid] = 0.5
        return result
    from scipy.stats import rankdata
    ranks = rankdata(valid_arr)
    pct = (ranks - 1) / (len(valid_arr) - 1)
    result = np.full(n, 0.5)
    result[valid] = pct
    result[~valid] = 0.5
    return result


# ═══════════════════════════════════════════════════════════
# 王者底仓因子 (4个, 来自king baseline)
# ═══════════════════════════════════════════════════════════

def compute_dollar_vol_20d(stocks, price_data):
    """成交额对数(取负) — 流动性维度"""
    import numpy as np
    close = price_data.get('close', {})
    volume = price_data.get('volume', {})
    n = len(stocks)
    arr = np.full(n, np.nan)
    valid = np.zeros(n, dtype=bool)
    for i, s in enumerate(stocks):
        if close.get(s) is None or volume.get(s) is None:
            continue
        c = np.array(close.get(s, []), dtype=float)
        v = np.array(volume.get(s, []), dtype=float)
        if len(c) < 25 or len(v) < 25:
            continue
        dv = np.mean(c[-20:] * v[-20:])
        arr[i] = -np.log(dv + 1) if dv > 0 else 0
        valid[i] = True
    return arr, valid


def compute_overnight_5d(stocks, price_data):
    """隔夜累计5日 — 情绪×日内关联"""
    import numpy as np
    close = price_data.get('close', {})
    open_ = price_data.get('open', {})
    n = len(stocks)
    arr = np.full(n, np.nan)
    valid = np.zeros(n, dtype=bool)
    for i, s in enumerate(stocks):
        if close.get(s) is None or open_.get(s) is None:
            continue
        c = np.array(close.get(s, []), dtype=float)
        o = np.array(open_.get(s, []), dtype=float)
        if len(c) < 7 or len(o) < 7:
            continue
        gap = o[-5:] / np.roll(c, 1)[-5:] - 1
        arr[i] = np.sum(gap)
        valid[i] = True
    return arr, valid


def compute_tvma_20(stocks, price_data):
    """量价趋势偏移(取负) — 趋势+流动性"""
    import numpy as np
    close = price_data.get('close', {})
    volume = price_data.get('volume', {})
    n = len(stocks)
    arr = np.full(n, np.nan)
    valid = np.zeros(n, dtype=bool)
    for i, s in enumerate(stocks):
        if close.get(s) is None or volume.get(s) is None:
            continue
        c = np.array(close.get(s, []), dtype=float)
        v = np.array(volume.get(s, []), dtype=float)
        if len(c) < 25 or len(v) < 25:
            continue
        cv = c[-20:] * v[-20:]
        ma_cv = np.mean(cv)
        curr_cv = c[-1] * v[-1]
        arr[i] = -(curr_cv / ma_cv) if ma_cv > 0 else 0
        valid[i] = True
    return arr, valid


def compute_turnover_std_cv(stocks, price_data):
    """换手变异系数(取负) — 流动性稳定性"""
    import numpy as np
    volume = price_data.get('volume', {})
    n = len(stocks)
    arr = np.full(n, np.nan)
    valid = np.zeros(n, dtype=bool)
    for i, s in enumerate(stocks):
        if volume.get(s) is None:
            continue
        v = np.array(volume.get(s, []), dtype=float)
        if len(v) < 25:
            continue
        std_v = np.nanstd(v[-20:])
        mean_v = np.nanmean(v[-20:])
        arr[i] = -(std_v / mean_v) if mean_v > 0 else 0
        valid[i] = True
    return arr, valid


def compute_nf_ff7913(stocks, price_data):
    """5日/60日均量截面取负 — 筹码分布proxy"""
    import numpy as np
    n = len(stocks)
    volume = price_data.get('volume', {})
    arr = np.full(n, np.nan)
    valid = np.zeros(n, dtype=bool)
    W = 65
    for i, s in enumerate(stocks):
        if volume.get(s) is None or len(volume.get(s, [])) < W:
            continue
        v = np.array(volume.get(s, []), dtype=float)
        ma5 = np.nanmean(v[-5:])
        ma60 = np.nanmean(v[-60:])
        if ma60 > 0:
            arr[i] = -(ma5 / ma60)
        else:
            arr[i] = -1.0
        valid[i] = True
    return arr, valid


def compute_f_(stocks, price_data):
    """最小日收益(取负)/波动率 — 尾部风险溢价"""
    import numpy as np
    n = len(stocks)
    close = price_data.get('close', {})
    arr = np.full(n, np.nan)
    valid = np.zeros(n, dtype=bool)
    for i, s in enumerate(stocks):
        if close.get(s) is None:
            continue
        c = np.array(close.get(s, []), dtype=float)
        if len(c) < 67:
            continue
        rets = c[1:] / c[:-1] - 1
        recent = rets[-20:]
        min_ret = np.nanmin(recent)
        std_ret = np.nanstd(recent)
        if std_ret > 0:
            arr[i] = -min_ret / std_ret
        else:
            arr[i] = 0.0
        valid[i] = True
    return arr, valid


def compute_money_flow_20(stocks, price_data):
    """资金流偏移(取负) — 资金流"""
    import numpy as np
    high = price_data.get('high', {})
    low = price_data.get('low', {})
    close = price_data.get('close', {})
    volume = price_data.get('volume', {})
    n = len(stocks)
    arr = np.full(n, np.nan)
    valid = np.zeros(n, dtype=bool)
    for i, s in enumerate(stocks):
        if any(high.get(s) is None or len(high.get(s, [])) < 25 for _ in [1]):
            continue
        h = np.array(high.get(s, []), dtype=float)
        l_ = np.array(low.get(s, []), dtype=float)
        c = np.array(close.get(s, []), dtype=float)
        v = np.array(volume.get(s, []), dtype=float)
        if len(h) < 25 or len(l_) < 25:
            continue
        typical = (h[-20:] + l_[-20:] + c[-20:]) * v[-20:] / 3
        curr_tp = (h[-1] + l_[-1] + c[-1]) * v[-1] / 3
        ma_tp = np.mean(typical)
        arr[i] = -(curr_tp - ma_tp)
        valid[i] = True
    return arr, valid


# ═══════════════════════════════════════════════════════════
# V4 范式因子 (3个新增, 覆盖未覆盖领域)
# ═══════════════════════════════════════════════════════════

def compute_intraday_amplitude(stocks, price_data):
    """
    V4-A: 情绪×日内范式
    日内振幅 = 周均(high-low)/open, 取负(低振幅=有主力控盘=alpha)
    A股经济学: T+1制度下, 高振幅=散户博弈/无主力, 低振幅=筹码锁定/机构控盘
    """
    import numpy as np
    high = price_data.get('high', {})
    low = price_data.get('low', {})
    open_ = price_data.get('open', {})
    n = len(stocks)
    arr = np.full(n, np.nan)
    valid = np.zeros(n, dtype=bool)
    for i, s in enumerate(stocks):
        if any(x.get(s) is None or len(x.get(s, [])) < 7 for x in [high, low, open_]):
            continue
        h = np.array(high.get(s, []), dtype=float)
        l_ = np.array(low.get(s, []), dtype=float)
        o = np.array(open_.get(s, []), dtype=float)
        # 日振幅
        daily_range = (h[-5:] - l_[-5:]) / np.where(o[-5:] > 0, o[-5:], 1)
        avg_range = np.nanmean(daily_range)
        arr[i] = -avg_range  # 低振幅=正面信号
        valid[i] = True
    return arr, valid


def compute_overnight_reversal(stocks, price_data):
    """
    V4-A companion: 隔夜反转
    隔夜跳空的反转倾向。隔夜高开→日内回落=负面, 隔夜低开→日内回升=正面。
    捕捉A股集合竞价机制产生的隔夜-日内分离定价。
    """
    import numpy as np
    close = price_data.get('close', {})
    open_ = price_data.get('open', {})
    n = len(stocks)
    arr = np.full(n, np.nan)
    valid = np.zeros(n, dtype=bool)
    for i, s in enumerate(stocks):
        if close.get(s) is None or open_.get(s) is None:
            continue
        c = np.array(close.get(s, []), dtype=float)
        o = np.array(open_.get(s, []), dtype=float)
        if len(c) < 7 or len(o) < 7:
            continue
        # 隔夜跳空: (open_t / close_{t-1} - 1)
        prev_c = np.roll(c, 1)[-5:]
        ov_gap = o[-5:] / np.where(prev_c > 0, prev_c, 1) - 1
        # 日内走势: (close_t / open_t - 1)
        intra = c[-5:] / np.where(o[-5:] > 0, o[-5:], 1) - 1
        # 反转信号: 隔夜跳空与日内走势反向=正面(定价分离修复)
        # reversal = -(ov_gap * intra)  # 负=反向, 取负后正=有修复
        reversal_signal = []
        for g, it in zip(ov_gap, intra):
            if g > 0 and it < 0:  # 高开低走 = 需求不足
                reversal_signal.append(-abs(it))
            elif g < 0 and it > 0:  # 低开高走 = 错杀修复
                reversal_signal.append(abs(it))
            else:
                reversal_signal.append(0)
        arr[i] = np.mean(reversal_signal) if reversal_signal else 0
        valid[i] = True
    return arr, valid


def compute_volume_concentration(stocks, price_data):
    """
    V4-B: 截面交互范式
    成交量集中度 = 近期5日量 / 20日均量 — 放量集中=资金关注
    与money_flow配对形成量×价的截面交互。
    经济学: 放量+资金流入=机构建仓(正面), 放量+资金流出=出货(负面)
    本因子单独: 放量集中度取负(偏好缩量稳定期, 等待放量), 由money_flow提供方向
    """
    import numpy as np
    volume = price_data.get('volume', {})
    n = len(stocks)
    arr = np.full(n, np.nan)
    valid = np.zeros(n, dtype=bool)
    for i, s in enumerate(stocks):
        if volume.get(s) is None:
            continue
        v = np.array(volume.get(s, []), dtype=float)
        if len(v) < 25:
            continue
        short_ma = np.nanmean(v[-5:])
        long_ma = np.nanmean(v[-20:])
        if long_ma > 0:
            concentration = short_ma / long_ma
            # 取负: 偏好正常/缩量的股票(不被炒作), 由money_flow筛选方向
            arr[i] = -concentration
        else:
            arr[i] = -1.0
        valid[i] = True
    return arr, valid


def compute_ret_3m(stocks, price_data):
    """3月收益(取负=反转) — 与行业轮动配对"""
    import numpy as np
    close = price_data.get('close', {})
    n = len(stocks)
    arr = np.full(n, np.nan)
    valid = np.zeros(n, dtype=bool)
    for i, s in enumerate(stocks):
        if close.get(s) is None:
            continue
        c = np.array(close.get(s, []), dtype=float)
        if len(c) < 65:
            continue
        arr[i] = -(c[-1] / c[-61] - 1)
        valid[i] = True
    return arr, valid


def compute_industry_relative_strength(stocks, price_data):
    """
    V4-C: 行业轮动范式
    行业内的相对强度 = 个股动量 / 行业平均动量
    A股经济学: 行业轮动明显, 同一行业内选相对强的股票。
    JQ实现: 用get_industry获取行业 → 计算行业内排名。
    注意: get_industry在周频调度中获取的是最新分类, 可能变化。
    """
    import numpy as np
    close = price_data.get('close', {})
    n = len(stocks)
    arr = np.full(n, np.nan)
    valid = np.zeros(n, dtype=bool)

    # 计算每只股票的20日动量
    stock_mom = {}
    for i, s in enumerate(stocks):
        if close.get(s) is None:
            continue
        c = np.array(close.get(s, []), dtype=float)
        if len(c) < 25:
            continue
        stock_mom[s] = c[-1] / c[-21] - 1

    if len(stock_mom) < 30:
        return arr, valid

    # 获取行业分类
    try:
        industry_map = get_industry([s for s in stocks if s in stock_mom])
    except Exception:
        industry_map = {}

    # 计算行业平均动量
    ind_mom = {}
    for s, mom in stock_mom.items():
        ind = industry_map.get(s, {}).get('sw_l1', {}).get('industry_name', '未知')
        if ind not in ind_mom:
            ind_mom[ind] = []
        ind_mom[ind].append(mom)

    ind_avg = {}
    for ind, moms in ind_mom.items():
        if len(moms) >= 3:
            ind_avg[ind] = np.nanmean(moms)

    # 计算相对强度 = 个股动量 - 行业平均
    for i, s in enumerate(stocks):
        if s not in stock_mom:
            continue
        ind = industry_map.get(s, {}).get('sw_l1', {}).get('industry_name', '未知')
        avg_mom = ind_avg.get(ind)
        if avg_mom is not None:
            arr[i] = stock_mom[s] - avg_mom  # 超额动量=相对强度, 取正
            valid[i] = True

    return arr, valid


# ═══════════════════════════════════════════════════════════
# 复合得分 (7个: 4底仓 + 3 V4)
# ═══════════════════════════════════════════════════════════

def compute_score_K1_overnight_tvma(stocks, price_data):
    """K1: 隔夜累计 × 量价趋势 (王者最强复合)"""
    a, va = compute_overnight_5d(stocks, price_data)
    b, vb = compute_tvma_20(stocks, price_data)
    valid = va & vb
    ra = rank_pct(a, valid)
    rb = rank_pct(b, valid)
    return np.where(valid, ra * rb, 0.5), valid


def compute_score_K2_dollar_turnover(stocks, price_data):
    """K2: 成交额 × 换手稳定性 (王者次强)"""
    a, va = compute_dollar_vol_20d(stocks, price_data)
    b, vb = compute_turnover_std_cv(stocks, price_data)
    valid = va & vb
    ra = rank_pct(a, valid)
    rb = rank_pct(b, valid)
    return np.where(valid, ra * rb, 0.5), valid


def compute_score_K3_nf_tvma(stocks, price_data):
    """K3: 缩量筹码 × 量价趋势 (筹码分布维度)"""
    a, va = compute_nf_ff7913(stocks, price_data)
    b, vb = compute_tvma_20(stocks, price_data)
    valid = va & vb
    ra = rank_pct(a, valid)
    rb = rank_pct(b, valid)
    return np.where(valid, ra * rb, 0.5), valid


def compute_score_K4_f_dollar(stocks, price_data):
    """K4: 尾部风险 × 流动性 (下行保护+流动性)"""
    a, va = compute_f_(stocks, price_data)
    b, vb = compute_dollar_vol_20d(stocks, price_data)
    valid = va & vb
    ra = rank_pct(a, valid)
    rb = rank_pct(b, valid)
    return np.where(valid, ra * rb, 0.5), valid


def compute_score_V4A_intraday_overnight(stocks, price_data):
    """
    V4-A: 情绪×日内范式
    intraday_amplitude × overnight_reversal
    经济学: 低日内振幅+隔夜跳空反转=信息在集合竞价中被充分消化+日内无噪音交易
    """
    a, va = compute_intraday_amplitude(stocks, price_data)
    b, vb = compute_overnight_reversal(stocks, price_data)
    valid = va & vb
    ra = rank_pct(a, valid)
    rb = rank_pct(b, valid)
    return np.where(valid, ra * rb, 0.5), valid


def compute_score_V4B_concentration_moneyflow(stocks, price_data):
    """
    V4-B: 截面交互范式
    volume_concentration × money_flow
    经济学: 量价截面交互 — 缩量+资金流入=机构悄然建仓, 放量+资金流出=主力出货
    一个放量维度×一个资金方向维度, 交叉定价。
    """
    a, va = compute_volume_concentration(stocks, price_data)
    b, vb = compute_money_flow_20(stocks, price_data)
    valid = va & vb
    ra = rank_pct(a, valid)
    rb = rank_pct(b, valid)
    return np.where(valid, ra * rb, 0.5), valid


def compute_score_V4C_industry_momentum(stocks, price_data):
    """
    V4-C: 行业轮动范式
    industry_relative_strength × ret_3m (反转)
    经济学: 行业内的相对强势股+中期反转=行业轮动中的轮涨机会
    选在行业内跑赢但整体回调的股票(下一轮行业龙头)
    """
    a, va = compute_industry_relative_strength(stocks, price_data)
    b, vb = compute_ret_3m(stocks, price_data)
    valid = va & vb
    ra = rank_pct(a, valid)
    rb = rank_pct(b, valid)
    return np.where(valid, ra * rb, 0.5), valid


def compute_ensemble(stocks, price_data):
    """等权rank-product ensemble (7个复合)"""
    import numpy as np
    n = len(stocks)
    composite = np.ones(n) * 0.5
    valid = np.ones(n, dtype=bool)

    score_funcs = [
        (compute_score_K1_overnight_tvma, "K1: 隔夜×量价趋势"),
        (compute_score_K2_dollar_turnover, "K2: 成交额×换手"),
        (compute_score_K3_nf_tvma, "K3: 缩量筹码×量价"),
        (compute_score_K4_f_dollar, "K4: 尾部风险×流动性"),
        (compute_score_V4A_intraday_overnight, "V4-A: 情绪×日内"),
        (compute_score_V4B_concentration_moneyflow, "V4-B: 截面交互"),
        (compute_score_V4C_industry_momentum, "V4-C: 行业轮动"),
    ]

    count = 0
    for func, label in score_funcs:
        score, v = func(stocks, price_data)
        n_v = int(np.sum(v))
        if n_v > 0:
            composite[v] *= score[v]
            count += 1
            valid = valid & v

    return composite, valid


# ═══════════════════════════════════════════════════════════
# 主调仓函数
# ═══════════════════════════════════════════════════════════

def rebalance(context):
    import numpy as np
    import gc

    g.trade_days += 1
    # warmup=5周 (周频scheduler下 trade_days=调度次数)
    if g.trade_days < 5:
        return

    current_dt = context.current_dt
    prev_date = context.previous_date

    # Universe: 全A股过滤ST/停牌/涨停
    universe = get_all_securities(['stock'], prev_date).index.tolist()
    CHUNK = 500
    filtered = []
    for k in range(0, len(universe), CHUNK):
        chunk = universe[k:k+CHUNK]
        try:
            cd = get_current_data()
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
    if len(universe) < g.stock_num:
        return

    # 数据加载
    lookback = 70
    field = 'close'
    px_close = get_price(universe, count=lookback, end_date=prev_date,
                         frequency='daily', fields=field,
                         skip_paused=False, fq='pre')
    field = 'open'
    px_open = get_price(universe, count=lookback, end_date=prev_date,
                        frequency='daily', fields=field,
                        skip_paused=False, fq='pre')
    field = 'high'
    px_high = get_price(universe, count=lookback, end_date=prev_date,
                        frequency='daily', fields=field,
                        skip_paused=False, fq='pre')
    field = 'low'
    px_low = get_price(universe, count=lookback, end_date=prev_date,
                       frequency='daily', fields=field,
                       skip_paused=False, fq='pre')
    field = 'volume'
    px_volume = get_price(universe, count=lookback, end_date=prev_date,
                          frequency='daily', fields=field,
                          skip_paused=False, fq='pre')

    # Panel兼容处理
    _Panel = getattr(pd, 'Panel', None)
    if _Panel is not None:
        for px, fname in [(px_close, 'close'), (px_open, 'open'),
                           (px_high, 'high'), (px_low, 'low'),
                           (px_volume, 'volume')]:
            if isinstance(px, _Panel):
                items = getattr(px, 'items', [])
                if fname in items:
                    px = px[fname]
                else:
                    px = px.minor_xs(fname)
            if fname == 'close':
                px_close = px
            elif fname == 'open':
                px_open = px
            elif fname == 'high':
                px_high = px
            elif fname == 'low':
                px_low = px
            elif fname == 'volume':
                px_volume = px

    price_data = {
        'close': {s: px_close[s].values for s in universe if s in px_close.columns},
        'open': {s: px_open[s].values for s in universe if s in px_open.columns},
        'high': {s: px_high[s].values for s in universe if s in px_high.columns},
        'low': {s: px_low[s].values for s in universe if s in px_low.columns},
        'volume': {s: px_volume[s].values for s in universe if s in px_volume.columns},
    }

    valid_universe = [s for s in universe
                      if len(price_data['close'].get(s, [])) >= 65
                      and len(price_data['volume'].get(s, [])) >= 65]
    if len(valid_universe) < g.stock_num:
        return

    # V4 Ensemble
    composite, valid = compute_ensemble(valid_universe, price_data)
    composite_arr = np.where(valid, composite, -1)

    topn = g.stock_num
    top_idx = np.argsort(composite_arr)[-topn:][::-1]
    top_stocks = [valid_universe[i] for i in top_idx if composite_arr[i] > -1]

    # ── 卖出不在持仓列表的股票 ──
    # 用 order_target (share-based) 而非 order_target_value (value-based)
    # 后者 value→shares→delta 两步转换中 delta 可能 <100 导致失败
    sold = 0
    for s in list(context.portfolio.positions.keys()):
        if s not in top_stocks:
            try:
                pos = context.portfolio.positions[s]
                amt = int(pos.total_amount)
                if amt < 100:
                    continue
                order_target(s, 0)  # share-based, 回避 value→shares 转换
                sold += 1
            except Exception:
                pass

    if len(top_stocks) == 0:
        return

    # ── 等权配置 (delta-aware, order_target 份额制, 杜绝 value→shares 转换) ──
    weight_per_stock = context.portfolio.total_value / len(top_stocks)
    ordered = 0
    skipped_price = 0
    skipped_delta = 0
    for s in top_stocks:
        try:
            last_px = get_current_data()[s].last_price
            if last_px <= 0:
                continue

            # 目标份额(整手向下取整)
            target_shares = max(0, int(weight_per_stock / last_px / 100) * 100)
            current_shares = 0
            if s in context.portfolio.positions:
                current_shares = int(context.portfolio.positions[s].total_amount)

            # 跳过条件:
            # 1. 目标不足1手且无持仓 → 无法开仓
            if target_shares == 0 and current_shares == 0:
                skipped_price += 1
                continue
            # 2. 目标与持仓差异不足1手 → 维持现状
            delta = abs(target_shares - current_shares)
            if 0 < delta < 100:
                skipped_delta += 1
                continue
            # 3. 无变化
            if delta == 0:
                continue
        except Exception:
            continue

        # 份额制: order_target 直接设置目标份额, 无 value→shares 转换
        order_target(s, target_shares)
        ordered += 1

    if skipped_price > 0 or skipped_delta > 0:
        log.info('ordered %d/%d stocks (skipped: %d overpriced, %d delta<1lot)'
                 % (ordered, len(top_stocks), skipped_price, skipped_delta))
