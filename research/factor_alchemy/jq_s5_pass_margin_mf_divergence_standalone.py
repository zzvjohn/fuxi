# -*- coding: utf-8 -*-
"""
⚠️⚠️ 2026-08-14 JQ 探针确认: 本文件无法在回测引擎运行 ⚠️⚠️
  get_money_flow / get_money_flow_pro / get_mtss 均仅研究环境可用
  回测环境 NameError: name 'get_money_flow' is not defined (资金流两个函数名都试过)
  两融/资金流数据回测引擎无任何替代 API (get_price/attribute_history 均不含)
  → 该因子只能走「JQ 研究环境手动纸质回测」(代码: jq_research_margin_paper_backtest.py) 或降级为仅本地 IC 证据
margin_mf_divergence 单因子独立回测 (本地 IC 验证因子, 待 JQ 定级)
因子含义: 主力净流入与融资余额的60交易日相关性取负 — 融资盘与主力资金背离=分歧预警 (P-002 降级两路版)
背景: P-002 三路资金分歧的北向个股数据 2024-08-19 后停更 → 近端主力方案=两路 (主力资金 vs 两融)
数据源: get_money_flow (net_amount_main=主力净流入额) + get_mtss (fin_value=融资余额)
        均为盘后披露, 防未来: end_date=context.previous_date
方向: HIGH=做多 (公式取负后高值=背离程度大)
P-001 合规: 输出 per-factor Rank IC/ICIR
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
    log.info("margin_mf_divergence 单因子 (主力vs两融背离, 滞后IC) 启动")


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


# ═══ 因子: margin_mf_divergence ═══
def _norm_code(c):
    """代码格式归一化: 无论返回纯数字还是带后缀, 统一成 000001.XSHE 格式"""
    c = str(c)
    if "." in c:
        return c
    if c.startswith(("6", "9", "5")):
        return c + ".XSHG"
    return c + ".XSHE"


def factor_margin_mf_divergence(stocks, prev_date):
    """-net_mf_amount.rolling(60).corr(rzye) 的 JQ 等价:
       主力净流入(net_amount_main) 与 融资余额(fin_value) 最近60个共同交易日的相关性取负"""
    n = len(stocks)
    arr = np.full(n, np.nan); valid = np.zeros(n, dtype=bool)
    if len(stocks) == 0:
        return arr, valid

    # 90个交易日 ≈ 130个自然日缓冲 (rolling(60).corr 需要 60 个共同观测)
    start = pd.Timestamp(prev_date) - pd.Timedelta(days=130)

    # 资金流: 先试回测内置 get_money_flow, 失败再试 get_money_flow_pro
    mf = None
    try:
        mf = get_money_flow(stocks, start_date=start, end_date=prev_date,
                            fields=["date", "sec_code", "net_amount_main"])
    except Exception as e:
        if not getattr(g, "diag_mf_err", False):
            log.info("[DIAG] get_money_flow FAIL: %r" % (e,))
            g.diag_mf_err = True
        try:
            mf = get_money_flow_pro(stocks, start_date=start, end_date=prev_date,
                                    fields=["date", "sec_code", "net_amount_main"])
        except Exception as e2:
            if not getattr(g, "diag_mfpro_err", False):
                log.info("[DIAG] get_money_flow_pro FAIL: %r" % (e2,))
                g.diag_mfpro_err = True
            mf = None
    if mf is not None and len(mf) > 0 and not getattr(g, "diag_mf_ok", False):
        log.info("[DIAG] money_flow OK rows=%d cols=%s" % (len(mf), list(mf.columns)))
        g.diag_mf_ok = True

    try:
        mt = get_mtss(stocks, start_date=start, end_date=prev_date,
                      fields=["date", "sec_code", "fin_value"])
    except Exception as e:
        if not getattr(g, "diag_mt_err", False):
            log.info("[DIAG] get_mtss FAIL: %r" % (e,))
            g.diag_mt_err = True
        mt = None
    if mt is not None and len(mt) > 0 and not getattr(g, "diag_mt_ok", False):
        log.info("[DIAG] get_mtss OK rows=%d cols=%s" % (len(mt), list(mt.columns)))
        g.diag_mt_ok = True

    if mf is None or mt is None or len(mf) == 0 or len(mt) == 0:
        return arr, valid
    try:
        mf = mf.copy(); mt = mt.copy()
        mf["_c"] = mf["sec_code"].map(_norm_code)
        mt["_c"] = mt["sec_code"].map(_norm_code)
        mf["_d"] = pd.to_datetime(mf["date"])
        mt["_d"] = pd.to_datetime(mt["date"])
        # 显式循环构造 {code: (values, dates)} (groupby.agg 不能返回数组)
        mf_map = {}
        for c, grp in mf.groupby("_c"):
            mf_map[c] = (grp["net_amount_main"].astype(float).values, grp["_d"].values)
        mt_map = {}
        for c, grp in mt.groupby("_c"):
            mt_map[c] = (grp["fin_value"].astype(float).values, grp["_d"].values)
    except Exception:
        return arr, valid

    for j, s in enumerate(stocks):
        try:
            if s not in mf_map or s not in mt_map:
                continue
            a, da = mf_map[s]
            b, db = mt_map[s]
            if len(a) < 60 or len(b) < 60:
                continue
            sa = pd.Series(a, index=da)
            sb = pd.Series(b, index=db)
            sa = sa[~sa.index.duplicated()]
            sb = sb[~sb.index.duplicated()]
            merged = pd.concat([sa, sb], axis=1, join="inner").dropna()
            if len(merged) < 60:
                continue
            x = merged.iloc[-60:, 0].values.astype(float)
            y = merged.iloc[-60:, 1].values.astype(float)
            if np.std(x) < 1e-12 or np.std(y) < 1e-12:
                continue
            c = float(np.corrcoef(x, y)[0, 1])
            if np.isnan(c):
                continue
            arr[j] = -c
            valid[j] = True
        except Exception:
            continue
    return arr, valid


# ═══ 因子映射 ═══
FACTOR_FUNC_MAP = {"margin_mf_divergence": factor_margin_mf_divergence}


# ═══ IC/ICIR 报告 ═══
def _print_ic_report():
    """P-001 合规: 输出单因子 Rank IC 和 ICIR"""
    if len(g.ic_records) == 0:
        log.info("[IC] margin_mf_divergence: NO IC RECORDS")
        return

    ic_arr = np.array(g.ic_records)
    ic_mean = float(np.mean(ic_arr))
    ic_std  = float(np.std(ic_arr))
    icir    = ic_mean / ic_std if ic_std > 0 else 0.0
    ic_pos  = float(np.mean(ic_arr > 0))

    log.info("=" * 55)
    log.info("  margin_mf_divergence  Per-Factor IC/ICIR Report (P-001)")
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
    # ═══════════════════════════════════════════
    filtered = []
    CHUNK = 500
    for k in range(0, len(universe_all), CHUNK):
        chunk = universe_all[k:k+CHUNK]
        try:
            cd = get_current_data()
            for s in chunk:
                if cd[s].paused or cd[s].is_st: continue
                if cd[s].high_limit <= cd[s].last_price: continue
                filtered.append(s)
        except Exception:
            filtered.extend(chunk)
    universe = [s for s in filtered if s.startswith(("0","3","6"))][:800]
    if len(universe) < g.stock_num:
        return

    arr, v = FACTOR_FUNC_MAP["margin_mf_divergence"](universe, prev_date)
    if int(np.sum(v)) < 10:
        if not getattr(g, "diag_low_valid", False):
            log.info("[DIAG] margin_mf_divergence valid=%d (<10) at %s — 不交易 (数据源异常?)" % (int(np.sum(v)), prev_date))
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
    # ═══════════════════════════════════════════
    comp = np.where(v, -rp, np.inf)
    topn = min(g.stock_num, int(np.sum(v)))
    top_idx = np.argsort(comp)[:topn]
    top_stocks = [universe[i] for i in top_idx if comp[i] < np.inf]

    for s in list(context.portfolio.positions.keys()):
        if s not in top_stocks:
            try:
                pos = context.portfolio.positions[s]
                if pos.total_amount < 100: continue
                order_target_value(s, 0)
            except:
                pass

    if len(top_stocks) == 0:
        return

    w = context.portfolio.total_value / len(top_stocks)
    ordered = 0
    for s in top_stocks:
        try:
            lp = get_current_data()[s].last_price
            if lp <= 0 or w / lp < 100: continue
            # P-20260901-008: 差额微调保护 (差额<100股跳过, 消除"数量不能小于100"刷屏)
            _pos = context.portfolio.positions.get(s)
            if _pos is not None and getattr(_pos, "total_amount", 0) > 0:
                _delta = int(w / lp) - int(_pos.total_amount)
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
    log.info("margin_mf_divergence | week=%d | ordered=%d/%d | stocks=%d | IC=%.4f | total_ic=%d" % (
        g.trade_days, ordered, len(top_stocks), int(np.sum(v)),
        latest_ic, len(g.ic_records)))

    if len(g.ic_records) >= 20 and g.trade_days % 26 == 0:
        _print_ic_report()

    gc.collect()
