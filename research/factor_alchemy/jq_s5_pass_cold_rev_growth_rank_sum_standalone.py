# -*- coding: utf-8 -*-
"""
cold_rev_growth_rank_sum 单因子独立回测 (Stage2 S1 PASS → S5 未通过, 用户指定观测性 JQ 实测)
================================================================================
因子含义: 近20交易日龙虎榜上榜次数(取负)的截面排名 + 营业总收入同比增速的截面排名
  = "避开游资炒作热度" + "高营收增速" 的等权合成。HIGH = 做多 (与 S5 一致)。
S5 公式 (原文, 逐字): (-lhb_flag.rolling(20).sum()).rank(pct=True, axis=1)
                        + tr_yoy.rank(pct=True, axis=1)

⚠️ 裁决状态 (如实披露, 勿误读):
  - S1 (Stage2 单因子): PASS — ic_mean=+0.0304, ICIR=+0.391, +IC%=60.3%
  - S5 (全量池组合层): **未通过** — 口径 default (DATE_CEIL=auto→2026-09-09, top_n=80 周频等权)
      excess_2025 = +17.55%  excess_2026 = +8.28%   (两年超额均为正 ✅)
      calmar_2025 = 1.77     calmar_2026 = 0.42 ❌  (阈值 0.70, 差 0.28)
      argmax_decile = Q10 (单调性 8/9)
    ⇒ 卡点唯一: 2026 腿风险调整后收益。本文件用于**真实回测独立验证**, 
      属绕过 ELIGIBLE_REQUIRE_S5=on 的观测性回测, 结果不改变因子库准入状态。
  - 详细归因: reports/S5四因子送检归因_20260920.md

数据源 (两个, 均为盘后/PIT, 无前视):
  ① get_billboard_list — 龙虎榜 (盘后 20:00/22:00 更新) → end_date=context.previous_date
  ② get_fundamentals(indicator) — 财报同比增速, date=previous_date 取该日已公告最新一期 (PIT)
     ⚠️ 口径近似说明: S5 侧 tr_yoy 取自 Tushare fina_indicator
        (营业总收入同比增长率), JQ 侧 inc_revenue_year_on_year 为营业收入同比
        — 两者统计口径接近但样本源不同, 属**近似复刻**, 见文件尾"口径差异"节。

方向: HIGH = 做多 (与 S5JointFilter.backtest_factor ascending=False 一致)
P-001 合规: 输出 per-factor Rank IC/ICIR (周频滞后法)

建议回测区间:
  A) 2025-01-01 ~ 2026-09-09 — 与 S5 两个年腿 (2025/2026) 严格对齐, 便于对照 excess/calmar
  B) 2020-01-01 ~ 2026-09-09 — 长样本看跨周期稳健性 (2018 起为更长基准)
"""

import numpy as np
import pandas as pd

VERSION = "v1-20260920-首版(cold_rev_growth_rank_sum, 龙虎榜避热度+营收增速)"


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
    log.set_level("system", "error")  # P-20260902-016: 抑制 order 内部仓位查询 WARNING 刷屏
    set_order_cost(OrderCost(open_tax=0, close_tax=0.001,
                              open_commission=0.0001, close_commission=0.0001),
                   type="stock")
    set_slippage(FixedSlippage(0.003))
    run_weekly(rebalance, 1, time="open", reference_security="000300.XSHG")
    log.info("[VERSION] %s — 若此行为空或版本号不符, 说明 JQ 上文件未更新!" % VERSION)
    log.info("cold_rev_growth_rank_sum 单因子 (龙虎榜避热度 + 营收增速, 截面等权合成) 启动")


# ═══ 辅助 ═══
def _sum_vals(seq):
    """JQ 沙箱裸 sum() 被 numpy 遮蔽 (P-20260901-012), 统一用手动累加"""
    _t = 0.0
    for _v in seq:
        _t = _t + _v
    return _t


def rank_pct(arr, valid):
    """截面百分比排名: 仅在 valid 内排名, 无效位置给 0.5 (与 S5 .rank(pct=True) 语义一致)"""
    n = len(arr)
    valid_arr = arr[valid]
    if len(valid_arr) < 3:
        result = np.full(n, 0.5)
        return result
    from scipy.stats import rankdata
    ranks = rankdata(valid_arr)
    pct = (ranks - 1) / (len(valid_arr) - 1)
    result = np.full(n, 0.5)
    result[valid] = pct
    return result


def _norm_code(c):
    """代码格式归一化: 无论返回纯数字还是带后缀, 统一成 000001.XSHE 格式"""
    c = str(c)
    if "." in c:
        return c
    if c.startswith(("6", "9", "5")):
        return c + ".XSHG"
    return c + ".XSHE"


# ═══ 数据源 ①: 龙虎榜近20交易日上榜次数 ═══
def _fetch_lhb_count(prev_date):
    """近20交易日 (自然日缓冲 40 天, 与 jq_s5_pass_lhb_freq_20d 既有已验证口径一致)
    上榜次数 = 同日同股去重后计数 (对齐 S5: lhb_flag=('net_amount','size'))
    返回 {code: count}"""
    out = {}
    start = pd.Timestamp(prev_date) - pd.Timedelta(days=40)
    try:
        df = get_billboard_list(stock_list=None, start_date=start,
                                end_date=prev_date, count=None)
    except Exception as e:
        if not getattr(g, "diag_lhb_err", False):
            log.info("[DIAG] get_billboard_list FAIL: %r" % (e,))
            g.diag_lhb_err = True
        return out
    if df is None or len(df) == 0:
        if not getattr(g, "diag_lhb_empty", False):
            log.info("[DIAG] get_billboard_list EMPTY for %s ~ %s" % (start, prev_date))
            g.diag_lhb_empty = True
        return out
    if not getattr(g, "diag_lhb_ok", False):
        log.info("[DIAG] get_billboard_list OK rows=%d cols=%s code_sample=%s direction=%s" % (
            len(df), list(df.columns),
            df["code"].head(3).tolist() if "code" in df.columns else "NO_code_col",
            df["direction"].unique().tolist() if "direction" in df.columns else "NO_dir_col"))
        g.diag_lhb_ok = True

    # direction 官方取值大写 ALL/BUY/SELL (兼容其他大小写)
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
        for k, v in c.items():
            out[_norm_code(k)] = float(v)
    except Exception:
        try:
            for k, v in d.groupby("code").size().items():
                out[_norm_code(k)] = float(v)
        except Exception:
            return {}
    return out


# ═══ 数据源 ②: 财报同比增速 (PIT, 一次查全市场再本地过滤) ═══
def _fetch_yoy(prev_date):
    """get_fundamentals(date=prev_date) → prev_date 已公告的最新一期财报 (无前视)
    一次查询全市场 (不按 code.in_() 分块: 分块 300/次 × 5 块/周 × 286 周
    易触发 JQ 引擎缓慢/超时; 全市场单次查询仅 2 列, 开销可忽略)
    返回 {code: 营业总收入同比增长率(%)}"""
    out = {}
    try:
        q = query(indicator.code, indicator.inc_revenue_year_on_year)
        df = get_fundamentals(q, date=prev_date)
    except Exception as e:
        if not getattr(g, "diag_fund_err", False):
            log.info("[DIAG] get_fundamentals FAIL: %r" % (e,))
            g.diag_fund_err = True
        return out
    if df is None or len(df) == 0:
        if not getattr(g, "diag_fund_empty", False):
            log.info("[DIAG] get_fundamentals EMPTY at %s" % (prev_date,))
            g.diag_fund_empty = True
        return out
    if not getattr(g, "diag_fund_ok", False):
        log.info("[DIAG] get_fundamentals OK rows=%d cols=%s" % (len(df), list(df.columns)))
        g.diag_fund_ok = True
    try:
        col = "inc_revenue_year_on_year"
        for _i, row in df.iterrows():
            _c = _norm_code(row["code"])
            _v = row[col]
            try:
                _f = float(_v)
            except Exception:
                continue
            if np.isfinite(_f):
                out[_c] = _f
    except Exception as e:
        if not getattr(g, "diag_fund_iter", False):
            log.info("[DIAG] get_fundamentals iter FAIL: %r" % (e,))
            g.diag_fund_iter = True
        return {}
    return out


# ═══ 因子: cold_rev_growth_rank_sum ═══
def factor_cold_rev_growth_rank_sum(stocks, prev_date):
    """(-lhb_flag.rolling(20).sum()).rank(pct,axis=1) + tr_yoy.rank(pct,axis=1)
    两个分项各自截面排名 (pct) 后相加; HIGH = 做多"""
    n = len(stocks)
    arr = np.full(n, np.nan)
    valid = np.zeros(n, dtype=bool)
    if len(stocks) == 0:
        return arr, valid

    # 分项 ①: 近20日龙虎榜上榜次数取负 (未上榜=0, 全池有效)
    lhb = _fetch_lhb_count(prev_date)
    a_arr = np.full(n, np.nan)
    a_valid = np.zeros(n, dtype=bool)
    for i, s in enumerate(stocks):
        a_arr[i] = -float(lhb.get(s, 0.0))
        a_valid[i] = True

    # 分项 ②: 营业总收入同比增速 (仅已披露的股票有效)
    yoy = _fetch_yoy(prev_date)
    b_arr = np.full(n, np.nan)
    b_valid = np.zeros(n, dtype=bool)
    for i, s in enumerate(stocks):
        v = yoy.get(s)
        if v is not None and np.isfinite(v):
            b_arr[i] = float(v)
            b_valid[i] = True

    a_rp = rank_pct(a_arr, a_valid)
    b_rp = rank_pct(b_arr, b_valid)

    # 合成: 两分项均有效才计入 (对齐 S5 两列均需非 NaN 才相加)
    for i in range(n):
        if a_valid[i] and b_valid[i]:
            arr[i] = a_rp[i] + b_rp[i]
            valid[i] = True
    return arr, valid


# ═══ 因子映射 ═══
FACTOR_FUNC_MAP = {"cold_rev_growth_rank_sum": factor_cold_rev_growth_rank_sum}


# ═══ IC/ICIR 报告 ═══
def _print_ic_report():
    """P-001 合规: 输出单因子 Rank IC 和 ICIR"""
    if len(g.ic_records) == 0:
        log.info("[IC] cold_rev_growth_rank_sum: NO IC RECORDS")
        return
    ic_arr = np.array(g.ic_records)
    ic_mean = float(np.mean(ic_arr))
    ic_std = float(np.std(ic_arr))
    icir = ic_mean / ic_std if ic_std > 0 else 0.0
    ic_pos = float(np.mean(ic_arr > 0))
    log.info("=" * 55)
    log.info("  cold_rev_growth_rank_sum  Per-Factor IC/ICIR Report (P-001)")
    log.info("=" * 55)
    log.info("  Rank IC mean:       %+.4f" % ic_mean)
    log.info("  Rank IC std:         %.4f" % ic_std)
    log.info("  ICIR (mean/std):    %+.4f" % icir)
    log.info("  IC > 0 ratio:       %.1f%%" % (ic_pos * 100))
    log.info("  Total IC samples:   %d" % len(ic_arr))
    log.info("=" * 55)


# ═══ 收盘价快照 (兼容 长表/Panel/MultiIndex 三形态, 见 debt 模板 2026-09-03 修复) ═══
def _snapshot_closes(universe_all, prev_date):
    out = {}
    try:
        px = get_price(universe_all, count=2, end_date=prev_date,
                       frequency="daily", fields="close",
                       skip_paused=False, fq="pre")
    except Exception as e:
        if not getattr(g, "diag_px_err", False):
            log.info("[DIAG] get_price snapshot FAIL: %r" % (e,))
            g.diag_px_err = True
        return out
    if px is None or len(px) == 0:
        return out
    try:
        # 形态 ①: Panel
        _P = getattr(pd, "Panel", None)
        if _P is not None and isinstance(px, _P):
            px = px["close"]
        # 形态 ②: 长表 (time/code/value) — panel=False 时的返回, 必须 pivot
        elif "time" in getattr(px, "columns", []) and "code" in list(px.columns):
            _vcol = "close" if "close" in px.columns else px.columns[-1]
            px = px.pivot_table(index="time", columns="code", values=_vcol, aggfunc="first")
        # 形态 ③: MultiIndex(field, code) 宽表
        elif hasattr(px, "columns") and getattr(px.columns, "nlevels", 1) > 1:
            _lvl0 = list(px.columns.get_level_values(0).unique())
            if "close" in _lvl0:
                px = px["close"]
    except Exception as e:
        if not getattr(g, "diag_px_shape", False):
            log.info("[DIAG] get_price shape normalize FAIL: %r cols=%s" % (e, list(getattr(px, "columns", []))[:6]))
            g.diag_px_shape = True
        return out
    try:
        for s in universe_all:
            if s in px.columns and len(px[s].values) > 0:
                out[s] = float(px[s].values[-1])
    except Exception:
        return out
    return out


# ═══ 周度调仓 ═══
def rebalance(context):
    import gc
    g.trade_days += 1
    if g.trade_days < 5:
        return

    prev_date = context.previous_date

    # ═══════ Step 0: 滞后 IC (上周因子 → 本周收益) ═══════
    universe_all = get_all_securities(["stock"], prev_date).index.tolist()
    universe_all = [s for s in universe_all if s.startswith(("0", "3", "6"))][:2000]
    snap_closes = _snapshot_closes(universe_all, prev_date)

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
                fwd_rets.append((curr_close / prev_close) - 1.0)
                factor_vals.append(fv)
        if len(fwd_rets) >= 30:
            try:
                from scipy.stats import spearmanr
                ic, _ = spearmanr(factor_vals, fwd_rets)
                if not np.isnan(ic):          # 常数输入 → nan, 记为无效不污染 ICIR
                    g.ic_records.append(float(ic))
            except Exception:
                pass

    # ═══════ Step 1: 池过滤 (逐只 try, 单只异常剔除不污染池子) ═══════
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
                continue
    except Exception:
        filtered = list(universe_all)
    universe = [s for s in filtered if s.startswith(("0", "3", "6"))][:1500]
    if len(universe) < g.stock_num:
        return

    arr, v = FACTOR_FUNC_MAP["cold_rev_growth_rank_sum"](universe, prev_date)
    if int(np.sum(v)) < 10:
        if not getattr(g, "diag_low_valid", False):
            log.info("[DIAG] cold_rev_growth_rank_sum valid=%d (<10) at %s — 不交易 (数据源异常?)"
                     % (int(np.sum(v)), prev_date))
            g.diag_low_valid = True
        return

    rp = rank_pct(arr, v)

    # ═══════ Step 2: 保存快照供下周 IC ═══════
    g.last_factor_stocks = universe
    g.last_factor_values = rp.copy()
    g.last_closes = {}
    for s in universe:
        if s in snap_closes:
            g.last_closes[s] = snap_closes[s]

    # ═══════ Step 3: 选股 & 调仓 (HIGH = 做多: 取最高 rp) ═══════
    comp = np.where(v, -rp, np.inf)
    topn = min(g.stock_num, int(np.sum(v)))
    top_idx = np.argsort(comp)[:topn]
    top_stocks = [universe[i] for i in top_idx if comp[i] < np.inf]

    cd_order = get_current_data()

    # ── 平仓: 不在新名单里 → 清仓 (order_target(s,0) 语义含零股) ──
    for s in list(context.portfolio.positions.keys()):
        if s not in top_stocks:
            try:
                if cd_order[s].paused:
                    continue
                pos = context.portfolio.positions[s]
                if pos.total_amount < 100:
                    continue
                order_target(s, 0)
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
                break
            d = cd_order[s]
            if d.paused or d.is_st:
                continue
            if d.high_limit <= d.last_price:
                continue
            ref_px = d.day_open if (d.day_open and d.day_open > 0) else d.last_price
            if ref_px <= 0 or w / ref_px < 100:
                continue
            # P-20260901-008: 差额微调保护 (差额<100股 JQ 拒单)
            _pos = context.portfolio.positions.get(s)
            if _pos is not None and getattr(_pos, "total_amount", 0) > 0:
                _delta = int(w / ref_px) - int(_pos.total_amount)
                if abs(_delta) < 100:
                    continue
            order_target_value(s, w)
            ordered += 1
        except Exception:
            continue

    # ═══════ Step 4: 日志 & 定期 IC 报告 ═══════
    latest_ic = g.ic_records[-1] if g.ic_records else 0.0
    log.info("cold_rev_growth_rank_sum | week=%d | ordered=%d/%d | stocks=%d | IC=%.4f | total_ic=%d" % (
        g.trade_days, ordered, len(top_stocks), int(np.sum(v)),
        latest_ic, len(g.ic_records)))

    if len(g.ic_records) >= 20 and g.trade_days % 26 == 0:
        _print_ic_report()

    gc.collect()


# ═══════════════════════════════════════════════════════════════════════
# 口径差异备忘 (S5 本地引擎 vs 本 JQ 文件)
# ─────────────────────────────────────────────────────────────────────
# 1. 基本面字段: S5 用 Tushare fina_indicator.tr_yoy (营业总收入同比, ann_date asof +1日),
#    本文件用 JQ indicator.inc_revenue_year_on_year (营业收入同比). 统计口径接近但样本源不同。
# 2. 排名池: S5 在**全池 5000+** 上 rank(pct, axis=1); 本文件在过滤后池 (~1500) 上排名。
#    排名池更小 → pct 分布更粗, 但相对排序不变; 对 top-80 选股影响有限。
# 3. 频率: S5 为日频调仓 (fwd5); 本文件为周频调仓 (run_weekly)。local→JQ 存在 1.4~4.6x 系统性折扣。
# 4. 龙虎榜窗口: 本文件用 40 自然日缓冲 (~28 交易日, 略宽于 20 交易日) — 与既有已验证
#    jq_s5_pass_lhb_freq_20d 口径一致 (宁可略宽不漏信号)。
# ⇒ 本文件结论仅用于判断"该因子在真实交易成本/成交约束下是否仍有超额",
#    是 JQ 唯一真相源口径; 与 S5 读数不可直接数值比对。
# ═══════════════════════════════════════════════════════════════════════
