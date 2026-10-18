# -*- coding: utf-8 -*-
"""
vwap_dev_ratio_5_60 反转版 单因子独立回测 (P-20260902-001 跟进立项)
因子含义: VWAP短长比偏离 = (5日VWAP / 60日VWAP) - 1
方向: LOW = 做多 (选因子值最低 = 5日VWAP 相对 60日VWAP 最低的股票)
风险过滤: 60日回撤 DD 过滤 (dd60 > -0.45 才入池, 剔除深跌崩溃股;
          掩码外置 +inf 而非 0 → 无 0 平局优先选股陷阱)

立项依据 (P-001 干净源复评结论, 2026-09-02):
- 做多版 JQ -46.91% 证伪 (jq_failed_direction), 做多硬禁维持
- 干净源: 日频 IC -0.0563 / ICIR 0.356 (n=1309); 周频 IC -0.0379 / ICIR 0.272 (n=220)
- 反转方向周频 ICIR 0.272 >= tau_w=0.15 → 反转版候选
- 9-1 复评记录: 反转版组合 MDD 61.7% 超红线 → 立项附带 60日回撤 DD 过滤
- 本文件双 IC 口径 (全池信号 IC vs 过滤池信号 IC) 用于验证过滤不破坏 alpha

# ═══ 版本记录 ═══
# v1 2026-09-02: 反转版立项。LOW=做多 + 60日回撤过滤 + 双IC口径 + VERSION标识 + sum遮蔽自检。
# v1.1 2026-09-02: P-20260902-016 加 log.set_level("system","error") 抑制
#   order_target_value 对未持仓标的内部仓位查询导致的 WARNING 刷屏
#   (JQ 官方帖 joinquant.com/post/13861 方法#1)。
生成时间: 2026-09-02 | 执行层: lhb_freq_20d v2 修复版骨架
"""

import numpy as np
import pandas as pd

VERSION = "v1.1-20260902-vwap_reverse_ddfilter"

def initialize(context):
    # 首行版本对账 (JQ 侧文件 ≠ 本地文件 事故教训: 贴日志第一行可验证版本)
    log.info("[VERSION] %s — 若此行为空或不是 v1.1-20260902, 说明 JQ 上文件未更新!" % VERSION)
    # sum 遮蔽自检: JQ 沙箱把 numpy 注入 globals, 裸 sum = np.sum
    _st = type(sum({1: 1.0}.values())).__name__
    log.info("[SELFTEST] sum(dict_values) 返回类型=%s (float=正常, dict_values=被numpy遮蔽)" % _st)

    g.stock_num = 80
    g.lookback = 145
    g.trade_days = 0
    g.dd_threshold = -0.45   # 60日回撤过滤阈值: dd60 > -0.45 才入池 (剔除深跌崩溃股)

    # --- IC 收集 (滞后法: 本周算上周因子→本周收益的秩相关), 双口径 ---
    g.ic_records_full = []      # 全池信号 IC (无过滤)
    g.ic_records_sel = []       # 过滤池信号 IC (DD 过滤后)
    g.last_factor_stocks = []
    g.last_factor_values = None   # 信号值 = -rank_pct(因子) (信号高 = 做多, 与选股方向一致)
    g.last_filter_mask = None     # 上期 DD 过滤掩码 (bool array, 对齐 last_factor_stocks)
    g.last_closes = {}
    g.ic_reported = False
    # --- 暴露监控 (P-20260814-004): 每期信号值横截面 std 时序 ---
    g.exposure_records = []   # [(week, held_std, universe_std), ...]

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
    log.info("vwap_dev_ratio_5_60 反转版 (LOW=做多, DD过滤, 双IC口径) 启动")


# ═══ 辅助 ═══
def _sum_vals(seq):
    _t = 0.0
    for _v in seq:
        _t = _t + _v
    return _t


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


# ═══ 因子: vwap_dev_ratio_5_60 (原式, 继承做多版验证过的 eval 向量化实现) ═══
_EXPR = "(amount.rolling(5).sum() / volume.rolling(5).sum()) / (amount.rolling(60).sum() / volume.rolling(60).sum()) - 1"
_GLOBALS = {
    "np": np, "pd": pd,
    "abs": np.abs, "sqrt": np.sqrt, "log": np.log, "log1p": np.log1p,
    "exp": np.exp, "sign": np.sign,
    "maximum": np.maximum, "minimum": np.minimum,
    "where": np.where, "clip": np.clip,
    "range": range, "len": len, "int": int, "float": float,
    "list": list, "dict": dict, "tuple": tuple,
}
def factor_vwap_dev_ratio_5_60(stocks, close_df, extra_dfs=None):
    """VWAP短长比偏离 (原式, 与做多版一致)
    向量化实现: 全池一次 DataFrame 运算, 取最后一行横截面。
    本文件在 rebalance 层做方向反转 (LOW=做多), 因子函数本身不改。"""
    nan_df = pd.DataFrame(np.nan, index=close_df.index, columns=close_df.columns)
    ex = extra_dfs or {}
    def _pick(key):
        df = ex.get(key)
        if df is None:
            return nan_df
        return df.reindex(index=close_df.index, columns=close_df.columns)
    _LOCALS = {
        "close": close_df, "open": _pick("open"), "high": _pick("high"),
        "low": _pick("low"), "volume": _pick("volume"), "amount": _pick("money"),
        "close_p": close_df, "open_p": _pick("open"), "high_p": _pick("high"),
        "low_p": _pick("low"), "volume_p": _pick("volume"), "amount_p": _pick("money"),
        "returns": close_df.pct_change(),
    }
    val = eval(_EXPR, _GLOBALS, _LOCALS)
    if isinstance(val, pd.DataFrame):
        last = val.iloc[-1]
    elif isinstance(val, pd.Series):
        last = val
    else:
        last = pd.Series(np.full(len(stocks), float(val)), index=stocks)
    arr = last.reindex(stocks).values.astype(float)
    valid = np.isfinite(arr)
    return arr, valid
# ═══ 因子映射 ═══
FACTOR_FUNC_MAP = {"vwap_dev_ratio_5_60": factor_vwap_dev_ratio_5_60}


# ═══ IC/ICIR 报告 (双口径) ═══
def _report_ic_block(label, recs):
    if len(recs) == 0:
        log.info("  [%s] NO IC RECORDS" % label)
        return
    ic_arr = np.array(recs, dtype=float)
    ic_mean = float(np.mean(ic_arr))
    ic_std = float(np.std(ic_arr))
    icir = ic_mean / ic_std if ic_std > 0 else 0.0
    ic_pos = float(np.mean(ic_arr > 0))
    log.info("  [%s] IC mean=%+.4f  std=%.4f  ICIR=%+.4f  IC>0=%.1f%%  n=%d" % (
        label, ic_mean, ic_std, icir, ic_pos * 100, len(ic_arr)))


def _print_ic_report():
    """P-001 合规: 输出单因子信号口径 Rank IC/ICIR (双口径)"""
    if len(g.ic_records_full) == 0:
        log.info("[IC] vwap_dev_ratio_5_60_reverse: NO IC RECORDS")
        return
    log.info("=" * 55)
    log.info("  vwap_dev_ratio_5_60_reverse  Per-Factor IC/ICIR (信号口径, LOW=做多)")
    log.info("=" * 55)
    _report_ic_block("FULL   ", g.ic_records_full)
    _report_ic_block("SEL_DD ", g.ic_records_sel)
    if len(g.ic_records_sel) >= 10 and len(g.ic_records_full) >= 10:
        f = np.mean(np.array(g.ic_records_sel, dtype=float))
        s = np.mean(np.array(g.ic_records_full, dtype=float))
        log.info("  DD过滤 alpha 增益: %.4f -> %.4f (%+.4f)" % (s, f, f - s))
    log.info("=" * 55)


# ═══ 暴露监控报告 (P-20260814-004) ═══
def _print_exposure_report():
    if len(g.exposure_records) < 10:
        log.info("[EXPOSURE] 样本不足 (%d < 10), 跳过暴露报告" % len(g.exposure_records))
        return
    held = np.array([e[1] for e in g.exposure_records], dtype=float)
    univ = np.array([e[2] for e in g.exposure_records], dtype=float)
    held = held[~np.isnan(held)]; univ = univ[~np.isnan(univ)]
    if len(held) < 10:
        return
    base_n = min(20, max(4, len(held) // 2))
    base = held[:base_n]; recent = held[-4:]
    base_mean, recent_mean = float(np.mean(base)), float(np.mean(recent))
    ratio = recent_mean / base_mean if base_mean > 0 else float("nan")
    log.info("=" * 55)
    log.info("  vwap_dev_ratio_5_60_reverse  Factor Exposure Report")
    log.info("=" * 55)
    log.info("  held std baseline(前%d期): %.4f" % (base_n, base_mean))
    log.info("  held std recent (近4期):   %.4f" % recent_mean)
    log.info("  universe std mean:         %.4f" % float(np.mean(univ)))
    log.info("  exposure vol ratio:        %.2fx" % ratio)
    if ratio >= 2.0:
        log.info("  [WARN] 因子暴露波动率放大 %.1fx! 拥挤/踩踏风险" % ratio)
    elif ratio >= 1.5:
        log.info("  [WATCH] 因子暴露波动率上升 %.1fx" % ratio)
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
    universe = [s for s in filtered if s.startswith(("0", "3", "6"))][:400]
    if len(universe) < g.stock_num:
        return

    # ═══════════════════════════════════════════
    # Step 1: get_price (close/money/volume, 与做多版一致, 已验证可在 JQ 跑通)
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
               if s in close_df.columns and int((~close_df[s].isna()).sum()) >= 70]
    if len(valid_u) < g.stock_num:
        return

    extra_dfs = {}

    px_x = get_price(universe, count=lb, end_date=prev_date, frequency="daily",
                     fields="money", skip_paused=False, fq="pre")
    if _P is not None and isinstance(px_x, _P):
        px_x = px_x["money"] if "money" in getattr(px_x, "items", []) else px_x.minor_xs("money")
    if px_x is not None and px_x.shape[0] > 0 and px_x.shape[1] > 0:
        extra_dfs["money"] = px_x.reindex(columns=universe)
    px_x = get_price(universe, count=lb, end_date=prev_date, frequency="daily",
                     fields="volume", skip_paused=False, fq="pre")
    if _P is not None and isinstance(px_x, _P):
        px_x = px_x["volume"] if "volume" in getattr(px_x, "items", []) else px_x.minor_xs("volume")
    if px_x is not None and px_x.shape[0] > 0 and px_x.shape[1] > 0:
        extra_dfs["volume"] = px_x.reindex(columns=universe)

    # 因子计算 (原式, 向量化)
    try:
        arr, v = FACTOR_FUNC_MAP["vwap_dev_ratio_5_60"](valid_u, close_df, extra_dfs)
    except Exception as _fe:
        if not getattr(g, "diag_factor_err", False):
            log.info("[DIAG] vwap_dev_ratio_5_60 因子计算异常: %r at %s (data shape=%s, univ=%d)" % (
                _fe, prev_date, getattr(px_c, "shape", None), len(valid_u)))
            g.diag_factor_err = True
        return
    nv = int(np.sum(v))
    if nv < 10:
        if not getattr(g, "diag_low_valid", False):
            log.info("[DIAG] vwap_dev_ratio_5_60 valid=%d (<10) at %s — 不交易 (因子计算失败?)" % (nv, prev_date))
            g.diag_low_valid = True
        return

    # ═══════════════════════════════════════════
    # Step 1.3: 60日回撤 DD 过滤 (反转版新增)
    #   dd60 = close / close.rolling(60).max() - 1 (负值, 深跌更负)
    #   掩码: dd60 > g.dd_threshold(-0.45) → 剔除 60 日回撤超 45% 的崩溃股
    #   🔴 掩码外置 +inf 而非 0 (smallcap 双雄 0 平局选股事故教训)
    # ═══════════════════════════════════════════
    dd60 = close_df / close_df.rolling(60).max() - 1.0
    dd_last = dd60.iloc[-1].reindex(valid_u)
    dd_mask = np.asarray(dd_last > g.dd_threshold, dtype=bool)
    # 无 dd 数据的股票 (NaN) → 不满足 > -0.45 → 自动出池, 安全
    n_in_mask = int(np.sum(dd_mask & v))
    if n_in_mask < g.stock_num:
        if not getattr(g, "diag_dd_mask", False):
            log.info("[DIAG] DD过滤后池内 %d < %d at %s — 过滤过严? (阈值 %.2f)" % (
                n_in_mask, g.stock_num, prev_date, g.dd_threshold))
            g.diag_dd_mask = True
        return

    rp = rank_pct(arr, v)
    sig = -rp   # 信号值: 反转方向 (信号高 = 做多, 与选股方向一致)

    # ═══════════════════════════════════════════
    # Step 1.5: 快照 + 滞后 IC 计算 (上周信号 → 本周收益), 双口径
    # ═══════════════════════════════════════════
    snap_closes = {}
    for s in px_c.columns:
        last_v = px_c[s].iloc[-1]
        if np.isfinite(last_v):
            snap_closes[s] = float(last_v)

    if len(g.last_factor_stocks) > 0 and g.last_factor_values is not None:
        fwd_rets = []
        factor_vals = []
        sel_vals = []
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
                if g.last_filter_mask is not None and bool(g.last_filter_mask[i]):
                    sel_vals.append(fv)
        if len(fwd_rets) >= 30:
            try:
                from scipy.stats import spearmanr
                ic, _ = spearmanr(factor_vals, fwd_rets)
                if ic == ic:               # 非 NaN (常数输入 → nan 丢弃)
                    g.ic_records_full.append(float(ic))
            except Exception:
                pass
        # 过滤池口径: 只用上期 DD 掩码内的股票算 IC
        if len(sel_vals) >= 30:
            try:
                from scipy.stats import spearmanr
                sel_rets = []
                for i, s in enumerate(g.last_factor_stocks):
                    prev_close = g.last_closes.get(s)
                    curr_close = snap_closes.get(s)
                    if (prev_close and prev_close > 0
                        and curr_close and curr_close > 0
                        and g.last_filter_mask is not None
                        and bool(g.last_filter_mask[i])):
                        sel_rets.append((curr_close / prev_close) - 1.0)
                ic2, _ = spearmanr(sel_vals, sel_rets)
                if ic2 == ic2:
                    g.ic_records_sel.append(float(ic2))
            except Exception:
                pass

    # ═══════════════════════════════════════════
    # Step 2: 保存快照供下周 IC 计算
    # ═══════════════════════════════════════════
    g.last_factor_stocks = valid_u
    g.last_factor_values = sig.copy()
    g.last_filter_mask = dd_mask.copy()
    g.last_closes = {}
    for s in valid_u:
        if s in close_df.columns and s in snap_closes:
            g.last_closes[s] = snap_closes[s]

    # ═══════════════════════════════════════════
    # Step 3: 选股 & 调仓 (LOW = 做多: 取最低因子值 = 最高信号)
    #   comp = np.where(v & dd_mask, rp, +inf) 升序 → 掩码内最低因子值优先
    #   🔴 掩码外 +inf 排最后, 无 0 平局陷阱
    # ═══════════════════════════════════════════
    comp = np.where(v & dd_mask, rp, np.inf)
    topn = min(g.stock_num, n_in_mask)
    top_idx = np.argsort(comp, kind="mergesort")[:topn]
    top_stocks = [valid_u[i] for i in top_idx if comp[i] < np.inf]

    # P-20260814-004: 记录本期暴露
    try:
        held_vals = sig[top_idx][:len(top_stocks)]
        g.exposure_records.append((g.trade_days,
                                   float(np.std(held_vals)) if len(held_vals) > 1 else 0.0,
                                   float(np.std(sig[v])) if int(np.sum(v)) > 1 else 0.0))
        if len(g.exposure_records) > 500:
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

    # ── 开仓: 等权买入 (lhb v2 修复版: 开盘价口径/停牌涨停再拦/现金break/差额微调保护) ──
    w = context.portfolio.total_value / len(top_stocks)
    ordered = 0
    for s in top_stocks:
        try:
            if context.portfolio.available_cash < w * 0.5:
                break                           # 现金不足, 停止开仓
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
            # "开仓/平仓数量不能小于 100"。微调差额不足一手时保持现状, 消除刷屏级委托失败。
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
    latest_ic_f = g.ic_records_full[-1] if g.ic_records_full else 0.0
    latest_ic_s = g.ic_records_sel[-1] if g.ic_records_sel else 0.0
    log.info("vwap_reverse | week=%d | ordered=%d/%d | valid=%d mask_in=%d | ICfull=%.4f ICsel=%.4f" % (
        g.trade_days, ordered, len(top_stocks), nv, n_in_mask,
        latest_ic_f, latest_ic_s))

    if len(g.ic_records_full) >= 20 and g.trade_days % 26 == 0:
        _print_ic_report()
        _print_exposure_report()
        g.ic_reported = True

    gc.collect()
