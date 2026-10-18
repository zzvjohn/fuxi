# -*- coding: utf-8 -*-
"""
lhb_freq_20d 单因子独立回测 (本地 IC 验证因子, JQ 已定级: JQ_MARGINAL)
因子含义: 近20交易日龙虎榜上榜次数取负 — 游资炒作聚集的股票未来收益更差 (A股炒作后反转)
Local: ICIR=-0.495 (2023-2026, 856天, 400股; 10/20/60日窗口交叉验证 -0.42~-0.50, 全部显著)
数据源: get_billboard_list (2010至今, 盘后20:00/22:00更新; 防未来: end_date=context.previous_date)
方向: HIGH=做多 (与 S5JointFilter.backtest_factor ascending=False 一致; 公式取负后高值=上榜次数少)
P-001 合规: 输出 per-factor Rank IC/ICIR

JQ 实测 (2021-01-01~2026-07-30, 首轮): 累计+67% / Sharpe 0.2662 / MaxDD -35.25%
  周频滞后 Rank IC: mean=+0.0654 std=0.1028 ICIR=+0.6365 IC>0=76.1% (N=280, 半年度快照无衰减)
执行修复 (2026-08-15 v2):
  ① Step1 池过滤逐只 try — 原 except 整 chunk 绕过过滤, 停牌股混入 (首轮 12164 次订单失败根因之一)
  ② 平仓改 order_target(s,0) 清仓语义 — 分红送股零股一次性卖出 (首轮"平仓数量<100"失败根因)
  ③ 开仓用开盘价判断 w/px<100 + 停牌/涨停拦截 (原 last_price 口径, 高开卡线失败)
  ④ 现金不足提前 break (防"下单数量为0"错误)
本轮请在 JQ 平台设置回测区间: 2018-01-01 ~ 2026-08-13 (与 breed 对比基准对齐, 含 2018 熊市)
"""

import numpy as np
import pandas as pd

def initialize(context):
    g.stock_num = 80
    g.trade_days = 0

    # --- IC 收集 (滞后法: 上周因子 -> 本周收益的秩相关) ---
    g.ic_records = []
    g.last_factor_stocks = []
    g.last_factor_values = None
    g.last_closes = {}

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
    log.info("lhb_freq_20d 单因子 (龙虎榜游资活跃度反转, 滞后IC) 启动")


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


# ═══ 因子: lhb_freq_20d ═══
def _norm_code(c):
    """代码格式归一化: 无论返回纯数字还是带后缀, 统一成 000001.XSHE 格式"""
    c = str(c)
    if "." in c:
        return c
    if c.startswith(("6", "9", "5")):
        return c + ".XSHG"
    return c + ".XSHE"


def factor_lhb_freq_20d(stocks, prev_date):
    """-lhb_flag.rolling(20).sum() 的 JQ 等价: 近20交易日龙虎榜上榜次数取负; HIGH = 做多"""
    n = len(stocks)
    arr = np.full(n, np.nan); valid = np.zeros(n, dtype=bool)
    if len(stocks) == 0:
        return arr, valid

    # 20个交易日 ≈ 40个自然日缓冲
    start = pd.Timestamp(prev_date) - pd.Timedelta(days=40)
    try:
        df = get_billboard_list(stock_list=None, start_date=start,
                                end_date=prev_date, count=None)
    except Exception as e:
        if not getattr(g, "diag_err", False):
            log.info("[DIAG] get_billboard_list FAIL: %r" % (e,))
            g.diag_err = True
        return arr, valid
    if df is None or len(df) == 0:
        if not getattr(g, "diag_empty", False):
            log.info("[DIAG] get_billboard_list EMPTY for %s ~ %s" % (start, prev_date))
            g.diag_empty = True
        return arr, valid
    if not getattr(g, "diag_ok", False):
        log.info("[DIAG] get_billboard_list OK rows=%d cols=%s code_sample=%s direction=%s" % (
            len(df), list(df.columns),
            df["code"].head(3).tolist() if "code" in df.columns else "NO_code_col",
            df["direction"].unique().tolist() if "direction" in df.columns else "NO_dir_col"))
        g.diag_ok = True

    # direction 官方取值为大写 ALL/BUY/SELL (兼容其他大小写)
    try:
        d = df[df["direction"].astype(str).str.upper() == "ALL"]
        if len(d) == 0:
            d = df
    except Exception:
        d = df
    try:
        d = d.copy()
        d["_c"] = d["code"].map(_norm_code)
        c = d.drop_duplicates(["_c", "day"]).groupby("_c").size()
    except Exception:
        try:
            c = d.groupby("code").size()
        except Exception:
            return arr, valid

    for i, s in enumerate(stocks):
        arr[i] = -float(c.get(s, 0))
        valid[i] = True
    return arr, valid


# ═══ 因子映射 ═══
FACTOR_FUNC_MAP = {"lhb_freq_20d": factor_lhb_freq_20d}


# ═══ IC/ICIR 报告 ═══
def _print_ic_report():
    """P-001 合规: 输出单因子 Rank IC 和 ICIR"""
    if len(g.ic_records) == 0:
        log.info("[IC] lhb_freq_20d: NO IC RECORDS")
        return

    ic_arr = np.array(g.ic_records)
    ic_mean = float(np.mean(ic_arr))
    ic_std  = float(np.std(ic_arr))
    icir    = ic_mean / ic_std if ic_std > 0 else 0.0
    ic_pos  = float(np.mean(ic_arr > 0))

    log.info("=" * 55)
    log.info("  lhb_freq_20d  Per-Factor IC/ICIR Report (P-001)")
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
    # Step 0: 滞后 IC 计算 (上周因子 -> 本周收益)
    # ═══════════════════════════════════════════
    universe_all = get_all_securities(["stock"], prev_date).index.tolist()
    universe_all = [s for s in universe_all if s.startswith(("0","3","6"))][:2000]

    px_c_snap = get_price(universe_all, count=2, end_date=prev_date,
                          frequency="daily", fields="close",
                          skip_paused=False, fq="pre")
    _P = getattr(pd, "Panel", None)
    if _P is not None and isinstance(px_c_snap, _P):
        px_c_snap = px_c_snap["close"] if "close" in getattr(px_c_snap, "items", []) else px_c_snap.minor_xs("close")

    snap_closes = {}
    for s in universe_all:
        if s in px_c_snap.columns and len(px_c_snap[s].values) > 0:
            snap_closes[s] = float(px_c_snap[s].values[-1])

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
                g.ic_records.append(float(ic))
            except:
                pass

    # ═══════════════════════════════════════════
    # Step 1: 股票池过滤 + 因子计算
    # 修复(2026-08-15): 原 except 会把整个 chunk 绕过过滤 → 停牌股混入池子
    # 现在逐只 try: 数据异常的单只股票直接剔除; 整体过滤失败才回退全量
    # ═══════════════════════════════════════════
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
    universe = [s for s in filtered if s.startswith(("0","3","6"))][:1500]
    if len(universe) < g.stock_num:
        return

    arr, v = FACTOR_FUNC_MAP["lhb_freq_20d"](universe, prev_date)
    if int(np.sum(v)) < 10:
        if not getattr(g, "diag_low_valid", False):
            log.info("[DIAG] lhb_freq_20d valid=%d (<10) at %s — 不交易 (数据源异常?)" % (int(np.sum(v)), prev_date))
            g.diag_low_valid = True
        return

    rp = rank_pct(arr, v)

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
    # 修复(2026-08-15): ①停牌股下单前再拦一道(双保险) ②平仓改 order_target(s,0)
    #   清仓语义(分红送股零股一次性卖出, 不再被"数量<100"卡死)
    #   ③开仓用开盘价判断能否买入100股(原用 last_price, 高开卡线失败)
    # ═══════════════════════════════════════════
    comp = np.where(v, -rp, np.inf)
    topn = min(g.stock_num, int(np.sum(v)))
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
            except:
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
        except:
            continue

    # ═══════════════════════════════════════════
    # Step 4: 日志 & 定期 IC 报告
    # ═══════════════════════════════════════════
    latest_ic = g.ic_records[-1] if g.ic_records else 0.0
    log.info("lhb_freq_20d | week=%d | ordered=%d/%d | stocks=%d | IC=%.4f | total_ic=%d" % (
        g.trade_days, ordered, len(top_stocks), int(np.sum(v)),
        latest_ic, len(g.ic_records)))

    if len(g.ic_records) >= 20 and g.trade_days % 26 == 0:
        _print_ic_report()

    gc.collect()
