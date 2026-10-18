# -*- coding: utf-8 -*-
"""
打散重组 v1b 全天候版: 10因子低相关 rank 积复合 + 双门控
════════════════════════════════════════════════════════════════
【v1b 改动】在现有复合 (rank 积 "与"门) 基础上加指数刹车风控, 复合逻辑零改动:
  指数刹车 (GATE_INDEX, 默认开): 中证1000 (000852.XSHG) 15交易日跌幅 <= -12%
     → 本次调仓清仓+空仓一周 (来源: 外部参考策略审计通过的指数风控项,
     原为"清仓1天", 周频尺度映射为"空仓1周")
     本地预验证: 2021-2026 触发 33 天 (2022:8/2024:7/2025:4/2026:14),
     全部落在真实崩盘窗口 (2022-04/2024-01微盘/2025-04关税/2026-06) — 频率合理
  拥挤度门控 (GATE_CROWDING, 默认关): 机制保留但日期表置空 —
     本地实测现定义拥挤度指标(集中度>50%全历史0天; 多空波动率比>1.5占53%交易日
     且集中在2023-2025盈利年, 与刹车零重叠)不适合作门控, 详见 CROWDING_DATES 注释
【开关对比回测法】同一文件跑两次:
   GATE_INDEX=False → 复现无门控基线
   GATE_INDEX=True  → 全天候版
   目标: maxDD 下降且收益不回吐太多; 日志末行输出门控统计供直接对照
【注意】门控触发时跳过选股与 IC 快照, 单因子 IC 样本数会略少 (属预期)
【参数】池 800 / top 30 / lookback 130 / 周频周一open / 等权rank复合
【回测建议区间】2021-01-01 ~ 2026-07-30
【超时自救】若撞 1800s: g.universe_size 降至 500
【结果反馈】回测完成后按 jq_return 百分比格式 (如 124.78) 触发 trigger_d_plus
"""

import gc

import numpy as np
import pandas as pd

# ══════════════ 参数 ══════════════
TOP_N = 30
IC_FACTORS = [
    "dollar20", "overnight5", "tstd_cv", "tvma20",
    "fa_turnover_std", "fa_vroc_12",
    "gp002_pricevol", "gp_micro_7db3",
    "overreaction", "gp_evo_003",
]

# ══════════════ 门控参数 (v1b 新增) ══════════════
GATE_INDEX = True       # 指数刹车: 中证1000 15日跌幅<=-12% → 清仓+空仓一周
GATE_CROWDING = False   # 拥挤度门控: 机制保留, 日期表置空(实测不适合作门控, 见下)
IDX_BRAKE_CODE = "000852.XSHG"   # 中证1000
IDX_BRAKE_WIN = 15
IDX_BRAKE_TH = -0.12

# 拥挤度旗标日集合 (本地预计算 2021-2026, JQ 只查表不重算 — 铁律21 信号表直通哲学)
# ⚠️ 本地实测结论 (2026-08-24, _a4_gate_dates.py): 现定义的两个拥挤度指标均不适合作门控:
#   - 前5%成交额集中度20日均>50%: 全历史 0 天触发 (山证54.9%口径不可复现/口径不同) → 死指标
#   - Top-8/Bottom-8 多空波动率比>1.5: 53%交易日触发, 且集中 2023-2025 (策略盈利年),
#     与指数刹车触发日零重叠 → 是"动量活跃期"标记而非崩盘刹车, 作门控会砍掉盈利年
# 因此默认置空 (纯监控保留, 见 scripts/market_crowding_monitor.py); 机制保留供未来更好代理
CROWDING_DATES = set()


def initialize(context):
    g.stock_num = TOP_N
    g.universe_size = 800
    g.lookback = 130                     # max窗口60×2+10
    g.trade_days = 0

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

    # IC 快照 (滞后法: 上周因子rp → 本周收益 spearmanr)
    g.ic_records = {f: [] for f in IC_FACTORS}
    g.last_stocks = None
    g.last_values = {f: None for f in IC_FACTORS}
    g.last_closes = {}
    g.last_report_week = 0

    # v1b: 门控统计
    g.gate_stats = {"index": 0, "crowding": 0, "weeks_gated": 0, "weeks_total": 0}

    log.info("打散重组v1b: 10因子rank积复合 + 双门控(idx=%s crowding=%s) | top%d 启动" % (
        "ON" if GATE_INDEX else "OFF", "ON" if GATE_CROWDING else "OFF", TOP_N))


# ══════════════ 辅助 ══════════════
def _rankdata_impl(a):
    try:
        from scipy.stats import rankdata
        return rankdata(a)
    except Exception:
        return np.argsort(np.argsort(a)).astype(float) + 1.0


def rank_pct(arr, valid):
    """铁律15: pct = rank/n_valid, 无效值填 0.5"""
    n = len(arr)
    valid_arr = arr[valid]
    if len(valid_arr) < 3:
        return np.full(n, 0.5)
    ranks = _rankdata_impl(valid_arr)
    pct = (ranks - 1) / (len(valid_arr) - 1)
    result = np.full(n, 0.5)
    result[valid] = pct
    return result


def _unwrap_panel(px, field):
    """铁律3: Panel 兼容; 新版 pandas 直接返回 DataFrame"""
    _P = getattr(pd, "Panel", None)
    if _P is not None and isinstance(px, _P):
        if field in getattr(px, "items", []):
            return px[field]
        return px.minor_xs(field)
    if isinstance(px, pd.DataFrame):
        return px
    return None


# ══════════════ 10 因子定义 (全部向量化宽表) ══════════════
def factor_dollar20(d):
    dv = (d["close"] * d["volume"]).rolling(20, min_periods=10).mean()
    dv = dv.where(dv > 0)          # 停牌日 volume=0 → NaN → invalid (防 log(0) 警告)
    return -np.log(dv)


def factor_overnight5(d):
    return (d["open"] / d["close"].shift(1) - 1.0).rolling(5, min_periods=3).sum()


def factor_tstd_cv(d):
    return -(d["volume"].rolling(20, min_periods=10).std()
             / d["volume"].rolling(20, min_periods=10).mean())


def factor_tvma20(d):
    num = d["close"] * d["volume"]
    den = num.rolling(20, min_periods=10).mean()
    den = den.where(den > 0)       # 防全停牌分母=0 的 divide 警告
    return -num / den


def factor_fa_turnover_std(d):
    den = d["volume"].rolling(20, min_periods=10).mean()
    den = den.where(den > 0)       # 防分母=0 的 divide 警告
    ratio = d["volume"] / den
    return -ratio.rolling(20, min_periods=10).std()


def factor_fa_vroc_12(d):
    return d["volume"].pct_change(12)


def factor_gp002_pricevol(d):
    return d["high"] - d["close"].pct_change(20).shift(20)


def factor_gp_micro_7db3(d):
    return (d["close"] - d["low"]) / ((d["close"] * d["volume"] - d["open"]) + 0.001)


def factor_overreaction(d):
    return -d["close"].pct_change(1).rolling(20).max()


def factor_gp_evo_003(d):
    return -d["volume"].rolling(120).corr(d["close"]).pct_change(10)


FACTOR_FUNCS = {
    "dollar20": factor_dollar20,
    "overnight5": factor_overnight5,
    "tstd_cv": factor_tstd_cv,
    "tvma20": factor_tvma20,
    "fa_turnover_std": factor_fa_turnover_std,
    "fa_vroc_12": factor_fa_vroc_12,
    "gp002_pricevol": factor_gp002_pricevol,
    "gp_micro_7db3": factor_gp_micro_7db3,
    "overreaction": factor_overreaction,
    "gp_evo_003": factor_gp_evo_003,
}


# ══════════════ 调仓 ══════════════
def _check_gates(context):
    """v1b 双门控: 触发则清仓+空仓一周。返回 True = 本次调仓被跳过"""
    g.gate_stats["weeks_total"] += 1
    prev_date = context.previous_date
    reason = None
    if GATE_INDEX:
        try:
            idx_px = get_price(IDX_BRAKE_CODE, end_date=prev_date,
                               count=IDX_BRAKE_WIN + 1, fields=["close"], fq="pre")
            if len(idx_px) >= IDX_BRAKE_WIN + 1:
                ret15 = float(idx_px["close"].iloc[-1] / idx_px["close"].iloc[0] - 1.0)
                if ret15 <= IDX_BRAKE_TH:
                    reason = "index_brake_%.1f%%" % (ret15 * 100)
        except Exception as e:
            log.warn("gate index fetch fail: %s" % str(e)[:60])
    if reason is None and GATE_CROWDING:
        if str(prev_date) in CROWDING_DATES:
            reason = "crowding"
    if reason is None:
        return False
    # 清仓 + 空仓一周
    g.gate_stats["index" if reason.startswith("index") else "crowding"] += 1
    g.gate_stats["weeks_gated"] += 1
    cd = get_current_data()
    sold = 0
    for x in list(context.portfolio.positions.keys()):
        try:
            if cd[x].paused:
                continue
            pos = context.portfolio.positions[x]
            if pos.total_amount < 100:
                continue
            order_target(x, 0)
            sold += 1
        except Exception:
            continue
    log.info("GATE %s | %s | clear=%d | gated_weeks=%d" % (
        reason, prev_date, sold, g.gate_stats["weeks_gated"]))
    return True


def rebalance(context):
    g.trade_days += 1
    prev_date = context.previous_date

    # ── Step 0 (v1b): 门控检查 ──
    if _check_gates(context):
        return

    # ── Step 1: 池过滤 (逐只 try, 单只异常剔除不污染池) ──
    universe_all = get_all_securities(["stock"], prev_date).index.tolist()
    universe_all = [x for x in universe_all if x.startswith(("0", "3", "6"))][:g.universe_size]
    filtered = []
    try:
        cd = get_current_data()
        for x in universe_all:
            try:
                if cd[x].paused or cd[x].is_st:
                    continue
                if "ST" in cd[x].name or "*" in cd[x].name:
                    continue
                filtered.append(x)
            except Exception:
                continue
    except Exception:
        filtered = universe_all
    if len(filtered) < 100:
        log.info("pool too small: %d" % len(filtered))
        return
    universe = filtered[:g.universe_size]

    # ── Step 2: 数据加载 (单次多字段 get_price) ──
    fields = ["open", "high", "low", "close", "volume"]
    try:
        px = get_price(universe, end_date=prev_date, count=g.lookback,
                       frequency="daily", fields=fields,
                       skip_paused=False, fq="pre")
    except Exception as e:
        log.info("get_price fail: %s" % str(e)[:80])
        return

    data = {}
    for f in fields:
        d = _unwrap_panel(px, f)
        if d is None:
            log.info("field %s missing" % f)
            return
        data[f] = d

    snap_closes = data["close"].iloc[-1].to_dict()

    # ── Step 3: 计算 10 因子 (全部宽表向量化) ──
    n = len(universe)
    rp = {}
    valid_map = {}
    for f in IC_FACTORS:
        wide = FACTOR_FUNCS[f](data)
        s = wide.iloc[-1].reindex(universe)
        vals = s.values.astype(float)
        valid = np.isfinite(vals)
        rp[f] = rank_pct(vals, valid)
        valid_map[f] = valid

    # 至少一半因子有效才继续
    min_valid = min(int(np.sum(v)) for v in valid_map.values())
    if min_valid < 10:
        log.info("valid factor too few: %d" % min_valid)
        return

    # ── Step 4: rank 积复合 (King 方法论, v2a: "与"门 — 所有维度都靠前才高分) ──
    # rank 积 = Π rp[f], 任一因子 rp≈0 则复合分清零, 选出"全方位优秀"的股票
    # 与 v1 (rank 和 = "或"门) 的区别: 偏科股票 (2-3因子极高+其余平庸) 被排除
    comp = np.ones(n)
    for f in IC_FACTORS:
        comp = comp * rp[f]

    scores = pd.Series(comp, index=universe)

    # ── Step 5: IC 快照 (滞后法) ──
    if g.last_stocks is not None:
        from scipy.stats import spearmanr
        for f in IC_FACTORS:
            lv_arr = g.last_values.get(f)
            if lv_arr is None:
                continue
            fvals, rets = [], []
            for i, x in enumerate(g.last_stocks):
                pc = g.last_closes.get(x)
                cc = snap_closes.get(x)
                fv = lv_arr[i] if i < len(lv_arr) else np.nan
                if pc and cc and pc > 0 and cc > 0 and not np.isnan(fv):
                    fvals.append(fv)
                    rets.append(cc / pc - 1.0)
            if len(fvals) >= 30:
                try:
                    icv, _ = spearmanr(fvals, rets)
                    if icv == icv:
                        g.ic_records[f].append(float(icv))
                except Exception:
                    pass

    g.last_stocks = list(universe)
    g.last_values = {f: rp[f].copy() for f in IC_FACTORS}
    g.last_closes = dict(snap_closes)

    # ── Step 6: Top-N 选股 ──
    scores_sorted = scores.sort_values(ascending=False)
    targets = scores_sorted.index[:TOP_N].tolist()
    target_set = set(targets)

    # ── Step 7: 平仓 (v2: order_target 清仓语义) ──
    port = context.portfolio.positions
    cd_order = get_current_data()
    for x in list(port.keys()):
        if x not in target_set:
            try:
                if cd_order[x].paused:
                    continue
                pos = port[x]
                if pos.total_amount < 100:
                    continue
                order_target(x, 0)
            except Exception:
                pass

    # ── Step 8: 开仓 (day_open 口径 + 手数取整 + 现金 break) ──
    total_value = context.portfolio.total_value
    unit = total_value / TOP_N
    ordered = 0
    for x in targets:
        try:
            if context.portfolio.available_cash < unit * 0.5:
                break
            d = cd_order[x]
            if d.paused or d.is_st:
                continue
            if d.high_limit <= d.last_price:
                continue
            ref_px = d.day_open if (d.day_open and d.day_open > 0) else d.last_price
            if ref_px <= 0 or unit / ref_px < 100:
                continue
            lots = int(unit / ref_px / 100) * 100
            if lots < 100:
                continue
            order_target(x, lots)
            ordered += 1
        except Exception:
            continue

    # ── Step 9: 日志 & 半年报 ──
    if g.trade_days <= 8 or g.trade_days % 13 == 0:
        log.info("打散v1 | week=%d | targets=%d ordered=%d | cash=%.0f" % (
            g.trade_days, len(targets), ordered,
            context.portfolio.available_cash))
    if g.trade_days - g.last_report_week >= 26:
        g.last_report_week = g.trade_days
        log.info("=" * 60)
        log.info("  打散重组v1 半年报 (P-001)  week=%d" % g.trade_days)
        log.info("=" * 60)
        _print_ic_report()
        log.info("=" * 60)

    gc.collect()


# ══════════════ IC 报告 ══════════════
def _print_ic_report():
    log.info("  ── 单因子 IC/ICIR (P-001) ──")
    for f in IC_FACTORS:
        ics = g.ic_records[f]
        if len(ics) < 4:
            log.info("    %-20s IC 样本不足 (%d期)" % (f, len(ics)))
            continue
        arr = np.array(ics)
        ic_mean = float(np.mean(arr))
        ic_std = float(np.std(arr))
        icir = ic_mean / ic_std if ic_std > 1e-8 else 0.0
        pct_pos = float(np.mean(arr > 0))
        log.info("    %-20s IC=%+.4f ICIR=%+.3f +IC%%=%.0f%% (%d期)" % (
            f, ic_mean, icir, pct_pos * 100, len(arr)))


# ══════════════ 收尾报告 ══════════════
def on_strategy_end(context):
    log.info("=" * 60)
    log.info("  打散重组v1b 最终报告 — 结果请按 jq_return 百分比格式反馈 D+")
    log.info("=" * 60)
    _print_ic_report()
    gs = g.gate_stats
    log.info("  [门控] weeks=%d gated=%d (index=%d crowding=%d) | idx:%s crowd:%s" % (
        gs["weeks_total"], gs["weeks_gated"], gs["index"], gs["crowding"],
        "ON" if GATE_INDEX else "OFF", "ON" if GATE_CROWDING else "OFF"))
    log.info("  [提示] 对照: v1 +157.81%/SR0.69/MaxDD-30.67%; 目标=maxDD下降且收益不回吐")
    log.info("=" * 60)
