# -*- coding: utf-8 -*-
"""
v3_overreaction_reversal 单因子独立回测 (S5 通过因子, v3 高性能修复版)
因子含义: 过度反应反转: 取负(20日最大单日涨幅)。极端涨幅=过度反应, 后续反转(Bondt-Thaler)
Local: S5: grade=B calmar=1.86 | 来源=forge (种子重检)
方向: HIGH=做多 (与 S5JointFilter.backtest_factor ascending=False 一致)
P-001 合规: 输出 per-factor Rank IC/ICIR

═══ v3 根因排查与修复 (2026-08-17) ═══
【根因】JQ 回测"无组合收益" = 回测撞 JQ 平台 1800 秒超时限制被强制终止
  (引擎规则: "function runs more than 1800 seconds 即关闭", 进程内存上限 3G)
  v2 每周期数据量: get_price(2000只 x 80天) x 2字段 ≈ 32 万行 + 2000 次逐只 eval,
  2018-2026 全窗 ~440 周 → 远超平台时限 → 回测被 kill, 收益显示 0。
  本地全市场模拟实证 (2021-2026, 271周): 因子与执行层逻辑本身正常,
  IC mean=+0.089 ICIR=+0.68 IC>0=77%, 模拟组合 +29.5% 各年均为正。
【修复】数据量降至 lhb_freq_20d 同量级 (~4.5万行/周期, lhb 实测可跑完全窗):
  ① 向量化因子: 一次 DataFrame.pct_change().rolling(20).max() 替代 2000 次逐只 eval
     (与逐只版数学完全等价: rolling(20).max() 只依赖最近 21 个交易日)
  ② 单次 get_price: 只拉 close (公式不用 volume), IC 快照复用同一份数据
     (删掉原版 3 次全市场查询中的 2 次)
  ③ 池子 1500 只 (与 lhb 同口径) + lookback 80→30 (因子值不变, 见上)
  ④ 执行层 = lhb v2 修复版 (逐只try过滤 / order_target清仓 / day_open口径 / 现金break)
  ⑤ DIAG 日志: 前几周详细状态 + 每周期一行 + 每半年 IC/ICIR 报告
     (若仍无收益, 请把 JQ 日志中 [DIAG] 行与周度日志贴回, 便于定位)

回测区间建议: 2018-01-01 ~ 2026-08-13 (与 lhb 全窗对比对齐, 含 2018 熊市)
"""

import numpy as np
import pandas as pd


def initialize(context):
    g.stock_num = 80
    g.lookback = 30          # v3: 30 天足够 (rolling20 只需最近 21 个交易日)
    g.trade_days = 0

    # --- IC 收集 (滞后法: 上周因子 -> 本周收益的秩相关) ---
    g.ic_records = []
    g.last_factor_stocks = []
    g.last_factor_values = None
    g.last_closes = {}
    g.ic_reported = False

    set_benchmark("000905.XSHG")
    set_option("use_real_price", True)
    set_option("avoid_future_data", True)
    log.set_level("order", "error")
    log.set_level("system", "error")  # P-20260902-016: 抑制 order 函数内部仓位查询的 WARNING ("在 positions 中不存在" 刷屏)
    set_order_cost(OrderCost(open_tax=0, close_tax=0.001,
                              open_commission=0.0001, close_commission=0.0001),
                   type="stock")
    set_slippage(FixedSlippage(0.003))
    run_weekly(rebalance, 1, time="open", reference_security="000300.XSHG")
    log.info("v3_overreaction_reversal 单因子 (S5通过, v3向量化轻量) 启动")


# ═══ 辅助 ═══
def _rankdata_impl(a):
    """rankdata 降级实现: scipy 不可用时自实现 (等价 rankdata(valid_arr))"""
    try:
        from scipy.stats import rankdata
        return rankdata(a)
    except Exception:
        order = np.argsort(np.argsort(a))
        return order.astype(float) + 1.0


def rank_pct(arr, valid):
    n = len(arr)
    valid_arr = arr[valid]
    if len(valid_arr) < 3:
        return np.full(n, 0.5)
    ranks = _rankdata_impl(valid_arr)
    pct = (ranks - 1) / (len(valid_arr) - 1)
    result = np.full(n, 0.5)
    result[valid] = pct
    return result


# ═══ 因子: v3_overreaction_reversal (v3: 向量化) ═══
_EXPR = "-close_p.pct_change(1).rolling(20).max()"

def factor_v3_overreaction_reversal(stocks, close_df):
    """过度反应反转: 取负(20日最大单日涨幅)。极端涨幅=过度反应, 后续反转(Bondt-Thaler)
    公式 (S5 通过口径): -close_p.pct_change(1).rolling(20).max()
    向量化: 全池一次 DataFrame 计算取最后一行, 与逐只 eval 数学完全等价
    HIGH = 做多 (与 S5JointFilter.backtest_factor ascending=False 一致)"""
    rets = close_df.pct_change()          # 逐列 pct_change (fill 行为与 Series 一致)
    factor = -rets.rolling(20).max()      # 逐列 rolling max, 只依赖最近 21 个交易日
    last = factor.iloc[-1]                # prev_date 当期的因子值
    arr = last.reindex(stocks).values.astype(float)
    valid = np.isfinite(arr)
    return arr, valid


# ═══ 因子映射 ═══
FACTOR_FUNC_MAP = {"v3_overreaction_reversal": factor_v3_overreaction_reversal}


# ═══ IC/ICIR 报告 ═══
def _print_ic_report():
    """P-001 合规: 输出单因子 Rank IC 和 ICIR"""
    if len(g.ic_records) == 0:
        log.info("[IC] v3_overreaction_reversal: NO IC RECORDS")
        return

    ic_arr = np.array(g.ic_records)
    ic_mean = float(np.mean(ic_arr))
    ic_std  = float(np.std(ic_arr))
    icir    = ic_mean / ic_std if ic_std > 0 else 0.0
    ic_pos  = float(np.mean(ic_arr > 0))

    log.info("=" * 55)
    log.info("  v3_overreaction_reversal  Per-Factor IC/ICIR Report (P-001)")
    log.info("=" * 55)
    log.info("  Rank IC mean:       %+.4f" % ic_mean)
    log.info("  Rank IC std:         %.4f" % ic_std)
    log.info("  ICIR (mean/std):    %+.4f" % icir)
    log.info("  IC > 0 ratio:       %.1f%%" % (ic_pos * 100))
    log.info("  Total IC samples:   %d" % len(ic_arr))
    log.info("=" * 55)


# ═══ 周度调仓 ═══
def rebalance(context):
    import gc
    g.trade_days += 1
    if g.trade_days < 5:
        return

    prev_date = context.previous_date

    # ═══════════════════════════════════════════
    # Step 1: 股票池过滤 (lhb v2 修复版: 逐只 try, 单只异常剔除不污染池子)
    # ═══════════════════════════════════════════
    universe_all = get_all_securities(["stock"], prev_date).index.tolist()
    universe_all = [s for s in universe_all if s.startswith(("0", "3", "6"))][:2000]

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
    universe = [s for s in filtered if s.startswith(("0", "3", "6"))][:1500]
    if len(universe) < g.stock_num:
        return

    # ═══════════════════════════════════════════
    # Step 1.5: 单次 get_price(close) + 向量化因子 (v3 核心优化)
    #   原版 3 次全市场查询 (snap count=2 / close count=80 / volume count=80)
    #   → 1 次 (close count=30); volume 公式不用, IC 快照复用本数据最后一行
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

    arr, v = FACTOR_FUNC_MAP["v3_overreaction_reversal"](universe, close_df)
    nv = int(np.sum(v))
    if nv < 10:
        if not getattr(g, "diag_low_valid", False):
            log.info("[DIAG] v3_overreaction_reversal valid=%d (<10) at %s — 不交易 (因子计算失败?)" % (nv, prev_date))
            g.diag_low_valid = True
        return
    rp = rank_pct(arr, v)

    # ═══════════════════════════════════════════
    # Step 0': 滞后 IC 计算 (上周因子 → 本周收益)
    #   v3: 快照复用 close_df 最后一行 (prev_date 收盘), 不再单独全市场查询
    # ═══════════════════════════════════════════
    snap_closes = {}
    for s in close_df.columns:
        last_v = close_df[s].iloc[-1]
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
    g.last_factor_stocks = universe
    g.last_factor_values = rp.copy()
    g.last_closes = {}
    for s in universe:
        if s in snap_closes:
            g.last_closes[s] = snap_closes[s]

    # ═══════════════════════════════════════════
    # Step 3: 选股 & 调仓 (HIGH = 做多: 取最高 rp)
    #   lhb v2 修复版: ①停牌股下单前再拦一道 ②平仓 order_target(s,0) 清仓语义
    #   ③开仓开盘价口径 ④现金不足提前 break
    # ═══════════════════════════════════════════
    comp = np.where(v, -rp, np.inf)
    topn = min(g.stock_num, nv)
    top_idx = np.argsort(comp)[:topn]
    top_stocks = [universe[i] for i in top_idx if comp[i] < np.inf]

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
            # P-20260901-008: 差额微调保护 — order_target_value 对已持仓股票按
            # |target_value - current_value| 差额下单, 差额<100股时 JQ 报
            # "开仓/平仓数量不能小于 100" (2026-09-01 9因子回测累计3.5万次失败)。
            # 微调差额不足一手时保持现状, 消除刷屏级委托失败。
            _pos = context.portfolio.positions.get(s)
            if _pos is not None and getattr(_pos, "total_amount", 0) > 0:
                _delta = int(w / ref_px) - int(_pos.total_amount)
                if abs(_delta) < 100:
                    continue
            order_target_value(s, w)
            ordered += 1
        except Exception:
            continue

    # ═══════════════════════════════════════════
    # Step 4: 日志 & 定期 IC 报告
    # ═══════════════════════════════════════════
    latest_ic = g.ic_records[-1] if g.ic_records else 0.0
    if g.trade_days <= 8:
        # 前几周详细日志: 便于诊断"无交易/无收益"
        log.info("v3_overreaction_reversal | week=%d | valid=%d | top=%d | ordered=%d/%d | 首只=%s(rp=%.3f) | IC=%.4f" % (
            g.trade_days, nv, len(top_stocks), ordered, len(top_stocks),
            top_stocks[0] if top_stocks else "-",
            float(rp[top_idx[0]]) if len(top_idx) > 0 else 0.0,
            latest_ic))
    else:
        log.info("v3_overreaction_reversal | week=%d | ordered=%d/%d | stocks=%d | IC=%.4f | total_ic=%d" % (
            g.trade_days, ordered, len(top_stocks), nv,
            latest_ic, len(g.ic_records)))

    if len(g.ic_records) >= 20 and g.trade_days % 26 == 0:
        _print_ic_report()
        g.ic_reported = True

    gc.collect()
