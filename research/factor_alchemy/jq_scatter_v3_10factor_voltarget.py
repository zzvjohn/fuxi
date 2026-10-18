# -*- coding: utf-8 -*-
"""
打散重组 v3: 10因子低相关 rank 和复合 + 波动率目标管理
════════════════════════════════════════════════════════════════
【方法论】贪心低相关筛选 (FactorMiner Ψ正交化思想) + rank 和复合 (v1 最优形态) + 波动率目标管理
  ① 从 14 个 JQ 验证 OHLCV 因子出发
  ② 本地计算两两 Spearman 相关矩阵 (2024-07-01~2026-08-20, 521天×5276只)
  ③ 按 JQ 收益降序贪心筛选: 与已选因子 max_ρ < 0.5 才选入
  ④ 最终 10 个低相关因子, 等权 rank 和复合 → 单一打分 → top 30
  ⑤ 波动率目标管理 (Barroso & Santa-Clara 2015 / Moreira & Muir 2017 规则化版)

【v3 改动】在 v1 (JQ 实测 +157.81%/SR0.69/MaxDD30.67%) 上加波动率目标管理
  背景: 9 次 JQ 实验证据链收敛 → v1 rank和 = 打散最优形态 (Sharpe 持平 King,
        MaxDD 更优)。King 的收益优势 (24.76pp) 来自承担更多波动 (DD 35.41% vs 30.67%)。
  假设: v1 的仓位按规则化波动率目标缩放后, 高波动期降仓 → MaxDD 进一步压缩,
        Sharpe 持平或略升 (避开的恐慌跌幅 > 牺牲的低波动涨幅)。
  数学: scale_t = clip(σ_target / σ_realized, 0.30, 1.00)
        σ_realized = 组合近 12 周周收益 std (周频框架, 无前视: 只用已完成周)
        σ_target = 0.040 (年化 ≈28.8%, 锚定 v1 自身历史波动 ~27%)
        A股无杠杆 → 封顶 1.0 (低波动期满仓, 高波动期降仓, 下限 0.30 防清仓)
  纯规则化零拟合: σ_target 锚定 v1 已实现波动, 非参数搜索产物。

【v2 系列结论 (供对照)】
  v2a rank积 +95.22% | v2b 分组 +117.18% | v2c rank积top50 +119.64% | v2d 几何平均 ≡v2a
  → 全部弱于 v1 rank和 +157.81%: rank 积 10 因子连乘区分度坍塌 (0.5^10≈0.001);
    几何平均与 rank 积是严格单调变换 → 选股排序完全相同 (排序等价性铁律)。

【筛选结果】 (相关系数来自本地 2026-08-20 截面)
  1. dollar20          (King, ρ<0.5 vs 所有已选)
  2. overnight5        (King, ρ<0.5)
  3. tstd_cv           (King, ρ<0.5)
  4. tvma20            (King, ρ<0.5)
  5. fa_turnover_std   (FA, ρ<0.5)
  6. fa_vroc_12        (FA, ρ<0.5)
  7. gp002_pricevol    (GP, ρ<0.5)
  8. gp_micro_7db3     (GP, ρ<0.5)
  9. overreaction      (行为金融, ρ<0.5)
  10. gp_evo_003       (GP, ρ<0.5)

【被淘汰的高相关因子】
  gp000_meanrev    — 与 gp002 ρ=1.000 (完全同源)
  gp_evo006_negopen — 与 gp002 ρ=-1.000 (完全反向同源)
  gp_evo004_opencov — 与 gp002 ρ=-0.907
  gp_breed_053     — 与 fa_turnover_std ρ=0.993

【关键发现】GP Breed 冠军因子间高度同源 (gp000/gp002/evo006 相关性 ±1.0),
  打散重组的核心价值 = 识别并合并这些冗余, 让每类 alpha 只计一次权重。

【参数】池 800 / top 30 / lookback 130 / 周频周一open / rank和复合 + 波动率目标管理
【回测建议区间】2021-01-01 ~ 2026-07-30 (与 King 基准对齐)
【对照基准】King 6复合 +182.57%/0.69/-35.41%; v1 (无波动率管理) +157.81%/0.69/-30.67%
【裁决标准】若 v3 Sharpe ≥ 0.69 且 MaxDD 明显 < 30.67% → 波动率管理有效;
  若 v3 收益/Sharpe 双降 → "组合学天花板"定论成立, 主攻方向切新 alpha 源
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

    # 波动率目标管理状态 (周频, 无前视: 只用已完成周的 total_value 变化)
    g.weekly_returns = []            # 组合周收益序列 (最近 12 个, 保序)
    g.last_value = None              # 上次调仓时 total_value
    g.vol_scale = 1.0                # 当前仓位缩放 (仅本次调仓生效)

    log.info("打散重组v3: rank和 + 波动率目标管理 | top%d 启动" % TOP_N)


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
def rebalance(context):
    g.trade_days += 1
    prev_date = context.previous_date

    # ── Step 0: 波动率目标管理 (规则化, 无前视) ──
    # ① 先记账: 用 total_value 变化记录"已完成周"的组合收益 (含交易摩擦, 量级可忽略)
    tv = context.portfolio.total_value
    if g.last_value is not None and g.last_value > 0:
        wr = tv / g.last_value - 1.0
        g.weekly_returns.append(float(wr))
        if len(g.weekly_returns) > 12:
            g.weekly_returns = g.weekly_returns[-12:]
    g.last_value = tv
    # ② 再算 scale: σ_realized = 近12周周收益样本std; 周数不足或波动极低 → 满仓
    g.vol_scale = 1.0
    if len(g.weekly_returns) >= 4:
        sigma = float(np.std(np.array(g.weekly_returns), ddof=1))
        if sigma > 1e-6:
            raw = 0.040 / sigma          # 目标周波动 4% ≈ 年化 28.8%
            g.vol_scale = min(max(raw, 0.30), 1.00)

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

    # ── Step 4: 等权 rank 和复合 (v1 最优形态, JQ +157.81%/SR0.69 实证) ──
    # v2a/v2d 教训: 10因子 rank积区分度坍塌 (0.5^10≈0.001), 几何平均与rank积排序等价
    # rank 和 = "或"门, 分散化最宽的复合形态, 已证是 10 因子规模下的最优
    comp = np.zeros(n)
    for f in IC_FACTORS:
        comp = comp + rp[f]
    comp = comp / len(IC_FACTORS)

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

    # ── Step 8: 开仓 (day_open 口径 + 手数取整 + 现金 break + 波动率缩放) ──
    total_value = context.portfolio.total_value
    unit = total_value * g.vol_scale / TOP_N     # 高波动期降仓, 余现金留存
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
        sigma = float(np.std(np.array(g.weekly_returns), ddof=1)) if len(g.weekly_returns) >= 4 else 0.0
        log.info("打散v3 | week=%d | targets=%d ordered=%d | vol_scale=%.2f (σ_week=%.4f) | cash=%.0f" % (
            g.trade_days, len(targets), ordered, g.vol_scale, sigma,
            context.portfolio.available_cash))
    if g.trade_days - g.last_report_week >= 26:
        g.last_report_week = g.trade_days
        log.info("=" * 60)
        log.info("  打散重组v3 半年报 (P-001)  week=%d" % g.trade_days)
        log.info("=" * 60)
        _print_ic_report()
        log.info("  [波动率管理] 近12周σ=%.4f 当前scale=%.2f" % (
            float(np.std(np.array(g.weekly_returns), ddof=1)) if len(g.weekly_returns) >= 4 else 0.0,
            g.vol_scale))
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
    log.info("  打散重组v3 (rank和+波动率目标管理) 最终报告 — 结果按 jq_return 百分比格式反馈 D+")
    log.info("=" * 60)
    _print_ic_report()
    log.info("  [波动率管理统计] 周收益样本=%d 最终scale=%.2f" % (
        len(g.weekly_returns), g.vol_scale))
    log.info("  [对照基准] King +182.57%/SR0.69/MaxDD-35.41% | v1(无管理) +157.81%/0.69/-30.67%")
    log.info("=" * 60)
