# -*- coding: utf-8 -*-
"""
王者 (King) v3 修正版 —— 宇宙截断 + 滑点成本修复
================================================================================
基线: research/factor_alchemy/output/fa_alpha_agent_v3_jq_20260802_130337.py
      自报读数 +182.57% / Sharpe 0.69 / MDD -35.41%（2021-01 ~ 2026-07 周频 top-80 等权）

本文件只改两个轴。因子公式、6 复合结构、rank_product、top-80 等权、先卖后买、
一手预算自检 —— 全部逐行保持原样。

────────────────────────────────────────────────────────────────────────────
【修复 A · 宇宙截断】（原版第 355 行）
  原版: universe = [s for s in filtered if s.startswith(('0','3','6'))][:2000]
  get_all_securities 返回按代码升序（0xxxxx < 3xxxxx < 6xxxxx），截断点只到 300428.SZ
  ⇒ 只剩 0 开头 1581 只 + 3 开头 419 只；6 开头沪市（600/601/603/605/688）
    共 2406 只全体被丢弃。（按同构代码表实测，本地 5783 只口径）
  真实 JQ 列表里 0 开头单类就超过 2000 只，6 开头必然全军覆没 —— 与本地结论同向。
  修复: UNIV_CAP = 0（不截断）。设 2000 可原样复现旧读数做 A/B。

【修复 B · 滑点成本】（原版第 18 行）
  原版: set_slippage(FixedSlippage(0.003))
  聚宽语义: set_slippage 的参数是「双边」价差，撮合按 X/2 每边扣
  ⇒ 0.003 双边 = 0.0015 元/股/边，被 0.01 元最小价格变动吞掉，实际滑点 ≈ 0。
  参照: PriceRelatedSlippage(0.002) = 每边 0.1%（双边 0.2%）。
  修复: COST_PROFILE 显式选档，默认 'real2'。

  往返成本 = 2×每边滑点 + 2×每边佣金 + 卖出印花税
  ┌──────────┬────────────────────────────────┬──────────┬────────────┬──────────────────┐
  │ 档位      │ set_slippage                   │ 名义往返  │ 150万实际   │ 说明              │
  ├──────────┼────────────────────────────────┼──────────┼────────────┼──────────────────┤
  │ 'off'    │ 不设，佣金/税清零               │ 0        │ 0          │ 纯毛收益（诊断）   │
  │ 'orig'   │ FixedSlippage(0.003)           │ ~12 bp   │ ~15.3 bp   │ 复现原版（滑点≈0） │
  │ 'std15'  │ PriceRelatedSlippage(0.0003)   │ 15 bp    │ ~18.3 bp   │ 项目标准 COST_BPS=15│
  │ 'real2'  │ PriceRelatedSlippage(0.002)    │ 32 bp    │ ~35.3 bp   │ 默认：0.1%/边      │
  │ 'cons67' │ PriceRelatedSlippage(0.0055)   │ 67 bp    │ ~70.3 bp   │ 小盘保守口径       │
  └──────────┴────────────────────────────────┴──────────┴────────────┴──────────────────┘
  ※ "150万实际"把 OrderCost 默认 min_commission=5 元/笔算进去：
    150万 / 80 只 ≈ 1.875 万元/笔，万1 佣金 = 1.875 元 < 5 元 ⇒ 按 5 元收
    ⇒ 实际佣金 2.67bp/边（不是 1bp）。资金 ≥ 400 万时每笔 ≥ 5 万，最低佣金失效。

────────────────────────────────────────────────────────────────────────────
【预注册判读规则】先定规则再看结果，避免事后解释

  1) 首行日志 [KINGFIX] 必须显示 COST_PROFILE=real2、往返成本 ≈ 35bp。
     若显示 orig 或 ~15bp ⇒ 成本档没生效，本次读数作废。
  2) 首期日志 univ 列应显著 > 2000（预期 3500~4800），6xx 列应 > 2000。
     若 univ=2000 且 6xx=0 ⇒ UNIV_CAP 没生效。
  3) 生死线：真版结构 T 的盈亏平衡往返成本 ≈ 34.7 bp（周毛收益 / 周单边换手）。
     real2 档在 150 万资金下 ≈ 35.3 bp —— 正好压在线上。
     ⇒ 预注册结论：修复后大概率落在 -15% ~ +20%，而不是 +182%。
     ⇒ 若跑出 +150% 以上：先查 (1)(2)；都正常则是 JQ 撮合/复权与本地代理的系统性差异
       （本地代理用未复权价、周相位取周五，JQ 用前复权、周相位取周一），
       需另做归因，不能当"成本不重要"的证据。
  4) 资金放大到 ≥400 万可让 5 元最低佣金失效，往返成本回落到 ≈32bp。
     这不是改策略，是改执行规模 —— 值得单独跑一次对照。

  本地代理引擎（272-273 周）同结构跨成本档读数，仅作方向参照：
    无成本 +124.2% | orig +57% | std15 +46% | real2 -1.6% | cons67 -56.5%

【跑法】
  网页回测设置：频率=天 | 资金=1500000 | 基准=000300.XSHG
               区间=2021-01-01 ~ 2026-07-30（与原版同窗）
  建议先跑 2021-01-01 ~ 2021-04-01 冒烟，核对首行与首期日志后再跑全程。
  日志字段：reb=期数 univ=过滤后宇宙数(6xx=其中的沪市) top=入选数 sel=实际下单数
            turn=本期单边换手 cum_turn=累计 cost_bp=本期成本 cum_cost_bp=累计成本 cum=净值累计

【与原版的全部差异清单】（除以下 4 条外逐行相同）
  1. UNIV_CAP 开关（默认 0 = 修复）
  2. COST_PROFILE 开关（默认 real2 = 修复）
  3. get_price 改为分块加载（等价结果，降低单次查询规模）—— 非行为变更
  4. get_current_data() 在买入循环外取一次（等价数据）—— 非行为变更
  另注：原版把 500 只一批的 try/except 兜底会整批保留未过滤标的，本文件保持原样未改。
  另注：close_tax 固定单值，无法按 2023-08-28 印花税减半做时点切换；默认 0.001（保守）。
        想按新税率跑，把 COST_PROFILES 各档的 tax 改成 0.0005（往返成本各降 5bp）。
  另注：未设 set_option('avoid_future_data', True)（原版也没有）。信号用 previous_date、
        执行用当日开盘，时点本身安全。
"""

import datetime
import numpy as np
import pandas as pd

from jqlib.technical_analysis import *

# ══════════════════════ 修复开关（只动这两处即完成修复） ══════════════════════
UNIV_CAP     = 0          # 0 = 全宇宙（修复后） | 2000 = 复现原版 [:2000] 截断
COST_PROFILE = 'real2'    # off | orig | std15 | real2 | cons67
CHUNK        = 1000       # get_price 分块（仅性能，不改结果）

COST_PROFILES = {
    #                 set_slippage            每边滑点  每边佣金   印花税
    'off':    dict(slip=None,              side=0.0,      comm=0.0,    tax=0.0),
    'orig':   dict(slip=('fixed', 0.003),  side=0.0,      comm=0.0001, tax=0.001),
    'std15':  dict(slip=('ratio', 0.0003), side=0.00015,  comm=0.0001, tax=0.001),
    'real2':  dict(slip=('ratio', 0.002),  side=0.001,    comm=0.0001, tax=0.001),
    'cons67': dict(slip=('ratio', 0.0055), side=0.00275,  comm=0.0001, tax=0.001),
}
N_TOP = 80                # 原版 g.stock_num = 80
MIN_COMMISSION = 5.0      # 原版未写，用聚宽 OrderCost 默认值 5 元/笔（显式化，行为不变）


# ══════════════════════════════ 以下为原版逐行副本 ══════════════════════════════
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

def zscore(arr, valid):
    import numpy as np
    v = arr[valid]
    if len(v) < 3:
        return np.zeros(len(arr))
    m, s = np.mean(v), np.std(v)
    if s < 1e-10:
        return np.zeros(len(arr))
    result = np.zeros(len(arr))
    result[valid] = (v - m) / s
    return result

def compute_dollar_vol_20d(stocks, price_data):
    """成交额对数(取负)"""
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

def compute_money_flow_20(stocks, price_data):
    """资金流偏移(取负)"""
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

def compute_overnight_5d(stocks, price_data):
    """隔夜累计5日"""
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

def compute_ret_3m(stocks, price_data):
    """3月收益(取负=反转)"""
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

def compute_turnover_std_cv(stocks, price_data):
    """换手变异系数(取负)"""
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

def compute_tvma_20(stocks, price_data):
    """量价趋势偏移(取负)"""
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

def compute_nf_ff7913(stocks, price_data):
    """5日/60日均量截面取负。长期缩量=筹码锁定/惜售, A股底部特征。"""
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
    """最小日收益(取负)/波动率。极端下跌相对波动水平, 捕捉尾部风险溢价。"""
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
        # 下行风险指标
        min_ret = np.nanmin(recent)
        std_ret = np.nanstd(recent)
        if std_ret > 0:
            arr[i] = -min_ret / std_ret
        else:
            arr[i] = 0.0
        valid[i] = True
    return arr, valid

def compute_nf_02a304(stocks, price_data):
    """截面取负((取负(20日最小收益))/波动率)。极端下行风险相对总风险的定价。高比率=被过度惩罚。"""
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
        # 下行风险指标
        min_ret = np.nanmin(recent)
        std_ret = np.nanstd(recent)
        if std_ret > 0:
            arr[i] = -min_ret / std_ret
        else:
            arr[i] = 0.0
        valid[i] = True
    return arr, valid

def compute_score_comp1_overnight_tvma(stocks, price_data):
    """隔夜累计x量价趋势: V3最强复合"""
    a, va = compute_overnight_5d(stocks, price_data)
    b, vb = compute_tvma_20(stocks, price_data)
    valid = va & vb
    ra = rank_pct(a, valid)
    rb = rank_pct(b, valid)
    score = np.where(valid, ra * rb, 0.5)
    return score, valid

def compute_score_comp2_dollar_turnover(stocks, price_data):
    """成交额x换手稳定性: V3次强"""
    a, va = compute_dollar_vol_20d(stocks, price_data)
    b, vb = compute_turnover_std_cv(stocks, price_data)
    valid = va & vb
    ra = rank_pct(a, valid)
    rb = rank_pct(b, valid)
    score = np.where(valid, ra * rb, 0.5)
    return score, valid

def compute_score_comp3_moneyflow_ret3m(stocks, price_data):
    """资金流x动量反转: V3原value_momentum简化版"""
    a, va = compute_money_flow_20(stocks, price_data)
    b, vb = compute_ret_3m(stocks, price_data)
    valid = va & vb
    ra = rank_pct(a, valid)
    rb = rank_pct(b, valid)
    score = np.where(valid, ra * rb, 0.5)
    return score, valid

def compute_score_comp4_tvma_20(stocks, price_data):
    """5日/60日均量截面取负。长期缩量=筹码锁定/惜售, A股底部特征。v1/v2缺 x tvma_20: 互补评分=10."""
    a, va = compute_nf_ff7913(stocks, price_data)
    b, vb = compute_tvma_20(stocks, price_data)
    valid = va & vb
    ra = rank_pct(a, valid)
    rb = rank_pct(b, valid)
    score = np.where(valid, ra * rb, 0.5)
    return score, valid

def compute_score_comp5_dollar_vol_20d(stocks, price_data):
    """最小日收益(取负)/波动率。极端下跌相对波动水平, 捕捉尾部风险溢价。 x dollar_vol_20d: 互补评分=1"""
    a, va = compute_f_(stocks, price_data)
    b, vb = compute_dollar_vol_20d(stocks, price_data)
    valid = va & vb
    ra = rank_pct(a, valid)
    rb = rank_pct(b, valid)
    score = np.where(valid, ra * rb, 0.5)
    return score, valid

def compute_score_comp6_money_flow_20(stocks, price_data):
    """截面取负((取负(20日最小收益))/波动率)。极端下行风险相对总风险的定价。高 x money_flow_20: 互补"""
    a, va = compute_nf_02a304(stocks, price_data)
    b, vb = compute_money_flow_20(stocks, price_data)
    valid = va & vb
    ra = rank_pct(a, valid)
    rb = rank_pct(b, valid)
    score = np.where(valid, ra * rb, 0.5)
    return score, valid

def compute_ensemble(stocks, price_data):
    import numpy as np
    n = len(stocks)
    composite = np.ones(n) * 0.5
    valid = np.ones(n, dtype=bool)
    score_funcs = [
        (compute_score_comp1_overnight_tvma,),
        (compute_score_comp2_dollar_turnover,),
        (compute_score_comp3_moneyflow_ret3m,),
        (compute_score_comp4_tvma_20,),
        (compute_score_comp5_dollar_vol_20d,),
        (compute_score_comp6_money_flow_20,),
    ]
    count = 0
    for (func,) in score_funcs:
        score, v = func(stocks, price_data)
        n_v = int(np.sum(v))
        if n_v > 0:
            composite[v] *= score[v]
            count += 1
            valid = valid & v
    return composite, valid


# ══════════════════════════════ 修复部分 ══════════════════════════════
def _load_field(universe, field, count, end_date, chunk=None):
    """分块取价再横向拼接。与原版单次 get_price 结果等价，仅降低单次查询规模。"""
    chunk = chunk or CHUNK
    parts = []
    for k in range(0, len(universe), chunk):
        sub = universe[k:k + chunk]
        d = get_price(sub, count=count, end_date=end_date, frequency='daily',
                      fields=field, skip_paused=False, fq='pre')
        if not isinstance(d, pd.DataFrame):
            try:
                d = d[field]
            except Exception:
                continue
        if d is not None and len(getattr(d, 'columns', [])) > 0:
            parts.append(d)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, axis=1)


def initialize(context):
    g.stock_num = 80
    g.limit_pct = 0.10                 # 原版遗留，未使用（保持原样）
    g.initial_capital = context.portfolio.total_value
    g.trade_days = 0

    set_benchmark('000300.XSHG')
    set_option('use_real_price', True)
    log.set_level('order', 'error')
    log.set_level("system", "error")  # P-20260902-016: 抑制 order 函数内部仓位查询的 WARNING ("在 positions 中不存在" 刷屏)
    log.set_level('strategy', 'info')

    # ---- 修复 B：显式成本档 ----
    prof = COST_PROFILES[COST_PROFILE]
    set_order_cost(OrderCost(open_tax=0, close_tax=prof['tax'],
                             open_commission=prof['comm'],
                             close_commission=prof['comm'],
                             min_commission=MIN_COMMISSION),
                   type='stock')
    if prof['slip'] is not None:
        kind, val = prof['slip']
        if kind == 'fixed':
            set_slippage(FixedSlippage(val))
        else:
            set_slippage(PriceRelatedSlippage(val))

    g.rt_cost = 2.0 * (prof['side'] + prof['comm']) + prof['tax']   # 名义往返成本(率)
    g.cum_turn = 0.0
    g.cum_cost = 0.0
    g.prev_top = set()
    g.reb = 0

    log.info('[KINGFIX] 启动 | UNIV_CAP=%s | COST_PROFILE=%s | 滑点=%s | 佣金=%.4f/边 '
             '| 印花税=%.4f | 名义往返成本=%.1fbp'
             % (UNIV_CAP if UNIV_CAP else '全宇宙', COST_PROFILE, prof['slip'],
                prof['comm'], prof['tax'], g.rt_cost * 1e4))

    run_weekly(rebalance, 1, time='open', reference_security='000300.XSHG')


def rebalance(context):
    import numpy as np
    import gc
    g.trade_days += 1
    if g.trade_days < 5:
        return
    g.reb += 1
    current_dt = context.current_dt
    prev_date = context.previous_date

    # ---------- Universe ----------（修复 A：默认不截断）
    universe = get_all_securities(['stock'], prev_date).index.tolist()
    CHUNK_U = 500
    filtered = []
    for k in range(0, len(universe), CHUNK_U):
        chunk = universe[k:k + CHUNK_U]
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
    universe = [s for s in filtered if s.startswith(('0', '3', '6'))]
    if UNIV_CAP:
        universe = universe[:UNIV_CAP]          # 复现原版：截点 ≈ 300428.SZ，砍掉全部 6xx
    n_univ = len(universe)
    n_sh = sum(1 for s in universe if s[0] == '6')
    if n_univ < g.stock_num:
        return

    # ---------- Load price data（分块，等价结果） ----------
    lookback = 70
    px_close  = _load_field(universe, 'close',  lookback, prev_date)
    px_open   = _load_field(universe, 'open',   lookback, prev_date)
    px_high   = _load_field(universe, 'high',   lookback, prev_date)
    px_low    = _load_field(universe, 'low',    lookback, prev_date)
    px_volume = _load_field(universe, 'volume', lookback, prev_date)

    price_data = {
        'close': {s: px_close[s].values for s in universe if s in px_close.columns},
        'open': {s: px_open[s].values for s in universe if s in px_open.columns},
        'high': {s: px_high[s].values for s in universe if s in px_high.columns},
        'low': {s: px_low[s].values for s in universe if s in px_low.columns},
        'volume': {s: px_volume[s].values for s in universe if s in px_volume.columns},
    }
    del px_close, px_open, px_high, px_low, px_volume

    valid_universe = [s for s in universe if len(price_data['close'].get(s, [])) >= 65]
    if len(valid_universe) < g.stock_num:
        return

    # ---------- 合成与选股（原版逐行） ----------
    composite, valid = compute_ensemble(valid_universe, price_data)
    composite_arr = np.where(valid, composite, -1)

    topn = g.stock_num
    top_idx = np.argsort(composite_arr)[-topn:][::-1]
    top_stocks = [valid_universe[i] for i in top_idx if composite_arr[i] > -1]

    # ---------- 诊断：单边换手 / 成本（不改行为，只记数） ----------
    cur = set(top_stocks)
    if g.prev_top:
        turn = len(g.prev_top - cur) / float(len(g.prev_top))
    else:
        turn = 1.0
    g.cum_turn += turn
    cost_pct = turn * g.rt_cost
    g.cum_cost += cost_pct
    g.prev_top = cur

    # ---------- Sell ----------（原版逐行）
    for s in list(context.portfolio.positions.keys()):
        if s not in top_stocks:
            try:
                pos = context.portfolio.positions[s]
                if pos.total_amount < 100:
                    continue
            except Exception:
                pass
            order_target_value(s, 0)

    if len(top_stocks) == 0:
        return

    # ---------- Buy ----------（原版逐行；get_current_data 提到循环外，等价）
    weight = context.portfolio.total_value / len(top_stocks)
    cd_now = get_current_data()
    ordered = 0
    for s in top_stocks:
        try:
            last_px = cd_now[s].last_price
            if last_px <= 0 or weight / last_px < 100:
                continue
        except Exception:
            continue
        order_target_value(s, weight)
        ordered += 1

    nav = context.portfolio.total_value
    log.info('[KINGFIX] reb=%d univ=%d(6xx=%d) top=%d sel=%d turn=%.3f cum_turn=%.1f '
             'cost_bp=%.1f cum_cost_bp=%.0f nav=%.0f cum=%+.2f%%'
             % (g.reb, n_univ, n_sh, len(top_stocks), ordered, turn, g.cum_turn,
                cost_pct * 1e4, g.cum_cost * 1e4, nav,
                (nav / g.initial_capital - 1.0) * 100))
    gc.collect()
