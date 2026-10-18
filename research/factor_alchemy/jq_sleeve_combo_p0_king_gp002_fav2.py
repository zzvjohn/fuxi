# -*- coding: utf-8 -*-
"""
══════════════════════════════════════════════════════════════════
三袖组合 P0 — King core4 × GP002 × FA Composite v2  (逆波动率分配)
══════════════════════════════════════════════════════════════════
生成: 2026-08-21 | 架构: 四层组合架构 P0 (策略袖子层)

三袖 (同一账户按袖记账, 跨袖互斥选股防重复持仓):
  袖A 价量周频袖   King core4: overnight5 × tvma20 × dollar20 × tstd_cv
                  等权 rank-product (王者方法论, +182.57%/0.69 血统)
  袖B GP价量袖    gp_breed_002: high - close.pct_change(20).shift(20)
                  向量化宽表 (JQ 实测 +143.46%/0.51 BEST)
  袖C FA复合袖    FA v2 原样移植: 4复合+earnings_volume_drift, ICIR 加权
                  (JQ 实测 +146.13%/0.68)

资金分配 (每4周重算): 逆波动率 w_i = (1/σ_i)/Σ(1/σ_j), σ_i=袖近26周周收益std
  数据<12周时等权 1/3; 单袖权重裁剪 [15%, 50%] (防独大, 规则化非拟合)

跨袖互斥: A→B→C 顺序选股, 后袖跳过前袖已选 — 重叠信号去冗余 (MMR 精神)

工程铁律合规:
  骨架 = jq_s5_pass_lhb_freq_20d_standalone.py v2 修复版 (逐只try过滤/
    order_target清仓/day_open口径/现金break/手数取整100股)
  向量化宽表 (铁律14) / rank_pct 铁律15 / get_price 铁律2 / get_fundamentals
  date=previous_date 铁律7 / 禁换手率API(volume代理, 铁律1) / lookback=130
  (=max窗口60×2+10, 铁律) / 单次多字段 get_price
P-001 合规: 三袖全部 10 个因子独立滞后 Rank IC/ICIR (半年报)
P-20260814-004 轻量版: 每周记录 袖收益/袖暴露(组合加权合成分)/全市场
  top5%成交额集中度, 报告期输出波动率与拥挤度摘要 (供 D+ 蒸馏)

【参数】全池2000 / A: 子池500·top30 (King验证形态) / B: 全池2000·top80 (GP002验证形态)
        / C: 000985成分+mcap≥5亿·top30 (FA v2验证形态) / lookback 130 / 周频周一open
【v2 修复 (2026-08-21, JQ v1 首测 +77.49%/SR0.35 归因)】
  v1 败因 = 参数漂移: topN 12/12/18 (验证形态 30/80/30) → 广度坍缩 (Grinold IR=IC×√BR,
  GP002 广度 80→12 理论 IR 损失 2.6×) + 互斥选股稀释 FA 候选池。
  v2: ①三袖恢复各自 JQ 验证的池/topN/rank 参照系 ②取消互斥 → 三袖独立选股,
  重叠股合并股数统一下单 (保留各袖独立性) ③逆波动率分配保留 (该层假设独立于袖内实现)
【回测建议区间】2021-01-01 ~ 2026-07-30 (与单因子 JQ 基准对齐)
【对照基准】King 6复合 +182.57%/0.69/-35.41% (三袖须≥此才升级); v1 实测 +77.49%/0.35/-35.13%
【超时自救】若撞 1800s: g.universe_size 降至 1500, 或区间减半
【结果反馈】回测完成后按 jq_return 百分比格式 (如 124.78) 触发 trigger_d_plus
"""

import numpy as np
import pandas as pd

# ══════════════ 参数 ══════════════
SLEEVES = ["A", "B", "C"]
SLEEVE_LABEL = {"A": "King_core4", "B": "GP002", "C": "FA_v2"}
SLEEVE_TOPN = {"A": 30, "B": 80, "C": 30}      # 各袖 JQ 验证形态 (v2 恢复)
FACTOR_SLEEVE = {"overnight5": "A", "tvma20": "A", "dollar20": "A",
                 "tstd_cv": "A", "gp002": "B"}  # FA 5 信号 → C (下方补充)
# FA v2 ICIR 训练权重 (保持已验证设计原样)
FA_ICIR = {
    "comp1_overnight_tvma":  0.7719,
    "comp2_dollar_turnover": 0.7472,
    "comp3_moneyflow_value": 0.6946,
    "comp4_retopen_skew":    0.6843,
    "earnings_volume_drift": 1.0679,
}
# JQ 沙箱坑: sum(dict.values()) 返回 dict_values 本身 (TypeError), 必须显式累加
_FA_WSUM = 0.0
for _k in FA_ICIR:
    _FA_WSUM += float(FA_ICIR[_k])
FA_W = {}
for _k in FA_ICIR:
    FA_W[_k] = float(FA_ICIR[_k]) / _FA_WSUM

IC_FACTORS = (["overnight5", "tvma20", "dollar20", "tstd_cv", "gp002"]
              + list(FA_ICIR.keys()))
for _k in FA_ICIR:
    FACTOR_SLEEVE[_k] = "C"


def initialize(context):
    g.stock_num = 140                    # 三袖合计 max (30+80+30)
    g.universe_size = 2000               # GP002 验证池口径
    g.pool_a_size = 500                  # King 验证池口径
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

    # IC 快照 (滞后法: 上周因子rp → 本周收益 spearmanr; 因子值与其袖池对齐)
    g.ic_records = {f: [] for f in IC_FACTORS}
    g.last_stocks = None
    g.last_pools = {}                     # {袖: 池}, 因子快照的参照系
    g.last_values = {f: None for f in IC_FACTORS}
    g.last_closes = {}
    g.last_report_week = 0

    # 袖记账
    g.sleeve_w = {s: 1.0 / 3.0 for s in SLEEVES}   # 逆波动率前等权
    g.sleeve_rets = {s: [] for s in SLEEVES}
    g.sleeve_values = {s: None for s in SLEEVES}   # 上周袖市值
    g.sleeve_holdings = {s: [] for s in SLEEVES}   # 当前袖持仓
    g.sleeve_expo = {s: [] for s in SLEEVES}       # 袖暴露 (组合加权合成分)
    g.crowding_hist = []                           # top5%成交额集中度

    log.info("三袖组合P0: King_core4(12) × GP002(12) × FA_v2(18) | 逆波动率 | 互斥选股 启动")


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


# ══════════════ 袖A 因子: King core4 (向量化宽表) ══════════════
def factor_overnight5(d):
    return (d["open"] / d["close"].shift(1) - 1.0).rolling(5, min_periods=3).sum()


def factor_tvma20(d):
    m = d["close"] * d["volume"]
    return -((m / m.rolling(20, min_periods=10).mean()).clip(0, 10))


def factor_dollar20(d):
    m = d["close"] * d["volume"]
    return -np.log(m.rolling(20, min_periods=10).mean().replace(0, np.nan))


def factor_tstd_cv(d):
    v = d["volume"]
    return -(v.rolling(20, min_periods=10).std()
             / v.rolling(20, min_periods=10).mean().replace(0, np.nan))


# ══════════════ 袖B 因子: gp_breed_002 (向量化宽表化, 铁律14) ══════════════
def factor_gp002(d):
    """high - close.pct_change(20).shift(20)  动量回调幅度"""
    return d["high"] - d["close"].pct_change(20).shift(20)


# ══════════════ 袖C 因子: FA v2 全套 (宽表向量化移植) ══════════════
def _fa_base(d, pb_map, stocks):
    """返回 dict: 因子名 → 最后行 Series (index=股票)"""
    o, h = d["open"], d["high"]
    l, c, v = d["low"], d["close"], d["volume"]
    money = c * v
    tp = (h + l + c) / 3.0
    mf = tp * v
    daily_ret = c.pct_change()

    def last(df):
        return df.iloc[-1].reindex(stocks)

    base = {}
    base["dollar_vol_20d"] = last(-np.log(
        money.rolling(20, min_periods=10).mean().replace(0, np.nan)))
    base["turnover_std_cv"] = last(-(v.rolling(20, min_periods=10).std()
                                     / v.rolling(20, min_periods=10).mean().replace(0, np.nan)))
    base["overnight_5d"] = last((o / c.shift(1) - 1.0).rolling(5, min_periods=3).sum())
    base["tvma_20"] = last(-((money / money.rolling(20, min_periods=20).mean()
                              .replace(0, np.nan))).clip(0, 10))
    base["money_flow_20"] = last(-(mf / mf.rolling(20, min_periods=20).mean()
                                   .replace(0, np.nan) - 1.0))
    base["ret_3m"] = last(-(c / c.shift(60) - 1.0))
    base["ret_open_2d"] = last((o / c.shift(1) - 1.0).rolling(2, min_periods=2).mean())
    base["skewness_20"] = last(-daily_ret.rolling(20, min_periods=10).skew())

    bp = pd.Series({s: (1.0 / pb_map[s] if (s in pb_map and pb_map.get(s)
                                            and pb_map[s] > 0) else np.nan)
                    for s in stocks})
    base["bp"] = bp

    # earnings_volume_drift: clip(ret5,±0.3)×clip(-vol_chg,-1,2), ret5>0 only
    ret5 = c / c.shift(5) - 1.0
    m5 = v.rolling(5, min_periods=3).mean()
    m5p = m5.shift(5)
    vol_chg = (m5 / m5p.replace(0, np.nan) - 1.0).fillna(0)
    sig = ret5.clip(-0.3, 0.3) * (-vol_chg).clip(-1, 2)
    base["earnings_volume_drift"] = last(sig.where(ret5 > 0, 0))
    return base


def _rp_series(s):
    """Series 版 rank_pct (铁律15 语义: pct=rank/n_valid)"""
    s2 = s.dropna()
    if len(s2) < 3:
        return s
    ranks = s2.rank(method="average")
    pct = (ranks - 1.0) / (len(s2) - 1.0)
    return pct.reindex(s.index)


def _fa_composites(base):
    """FA v2: 4 rank积复合 + evd → {信号名: rank值 Series}"""
    r = {k: _rp_series(base[k]) for k in base}
    r_vm = _rp_series(r["bp"] * r["ret_3m"])            # value_momentum 嵌套复合
    return {
        "comp1_overnight_tvma":  r["overnight_5d"] * r["tvma_20"],
        "comp2_dollar_turnover": r["dollar_vol_20d"] * r["turnover_std_cv"],
        "comp3_moneyflow_value": r["money_flow_20"] * r_vm,
        "comp4_retopen_skew":    r["ret_open_2d"] * r["skewness_20"],
        "earnings_volume_drift": r["earnings_volume_drift"],
    }


# ══════════════ IC/ICIR 报告 (P-001) ══════════════
def _print_ic_report():
    log.info("── 袖级收益/波动 (逆波动率分配依据) ──")
    for s in SLEEVES:
        r = np.array(g.sleeve_rets[s])
        if len(r) >= 4:
            sd = float(np.std(r))
            ann = float(np.mean(r)) * 52.0
            log.info("  袖%s %-12s n=%3d  周收益均值=%+.4f  σ_w=%.4f  年化≈%+.1f%%  w=%.2f" % (
                s, SLEEVE_LABEL[s], len(r), float(np.mean(r)), sd,
                ann * 100.0, g.sleeve_w[s]))
        else:
            log.info("  袖%s %-12s 样本不足 (n=%d)" % (s, SLEEVE_LABEL[s], len(r)))
    if len(g.crowding_hist) >= 4:
        ch = np.array(g.crowding_hist)
        log.info("── 拥挤度代理 (top5%%成交额集中度, 阈值50%%) ──")
        log.info("  last=%.3f  mean=%.3f  max=%.3f  触及50%%次数=%d/%d" % (
            float(ch[-1]), float(np.mean(ch)), float(np.max(ch)),
            int(np.sum(ch > 0.50)), len(ch)))
    log.info("── 每因子滞后 Rank IC/ICIR (P-001) ──")
    for f in IC_FACTORS:
        recs = g.ic_records.get(f, [])
        if len(recs) == 0:
            log.info("  [IC] %-22s NO IC RECORDS" % f)
            continue
        arr = np.array(recs)
        mean, std = float(np.mean(arr)), float(np.std(arr))
        icir = mean / std if std > 0 else 0.0
        log.info("  [IC] %-22s n=%4d  ICmean=%+.4f  ICIR=%+.4f  IC>0=%.1f%%" % (
            f, len(arr), mean, icir, float(np.mean(arr > 0)) * 100.0))


# ══════════════ 袖权重: 逆波动率 (规则化, 每4周重算) ══════════════
def _update_sleeve_weights():
    if min(len(g.sleeve_rets[s]) for s in SLEEVES) < 12:
        g.sleeve_w = {s: 1.0 / 3.0 for s in SLEEVES}
        return
    inv = {}
    for s in SLEEVES:
        sd = float(np.std(np.array(g.sleeve_rets[s][-26:])))
        inv[s] = 1.0 / max(sd, 1e-4)
    tot = 0.0
    for s in SLEEVES:
        tot += inv[s]
    w = {s: inv[s] / tot for s in SLEEVES}
    # 裁剪 [0.15, 0.50] 后归一 — 防单袖独大/清零
    # 注意: 单次 clip→normalize 会再次突破上限 (实测 0.54), 故迭代至收敛
    for _ in range(10):
        w = {s: min(max(w[s], 0.15), 0.50) for s in SLEEVES}
        t = 0.0
        for s in SLEEVES:
            t += w[s]
        if abs(t - 1.0) < 1e-6:
            break
        w = {s: w[s] / t for s in SLEEVES}
    g.sleeve_w = w


# ══════════════ 选股 helper ══════════════
def _top_from_scores(scores, valid, n, used):
    """scores: Series(index=股票); 返回 top-n 股票列表, 跳过 used 与无效值"""
    s = scores.dropna()
    s = s[[x for x in s.index if (valid.get(x, False) and x not in used)]]
    s = s.sort_values(ascending=False)
    return list(s.index[:n])


# ══════════════ 周度调仓 (骨架 v2 修复版) ══════════════
def rebalance(context):
    import gc
    g.trade_days += 1
    if g.trade_days < 3:
        return
    prev_date = context.previous_date

    # ── Step 0: 袖收益记账 (上周袖市值 → 本周) + 拥挤度留待数据后算 ──
    port = context.portfolio.positions
    cur_values = {}
    for s in SLEEVES:
        v_sum = 0.0
        for st in g.sleeve_holdings[s]:
            if st in port:
                v_sum += port[st].value
        cur_values[s] = v_sum
        if (g.sleeve_values[s] is not None and g.sleeve_values[s] > 1.0):
            g.sleeve_rets[s].append(cur_values[s] / g.sleeve_values[s] - 1.0)

    # ── Step 1: 池过滤 (v2: 逐只 try, 单只异常剔除不污染池) ──
    universe_all = get_all_securities(["stock"], prev_date).index.tolist()
    universe_all = [x for x in universe_all if x.startswith(("0", "3", "6"))][:g.universe_size]
    filtered = []
    try:
        cd = get_current_data()
        for x in universe_all:
            try:
                if cd[x].paused or cd[x].is_st:
                    continue
                if cd[x].high_limit <= cd[x].last_price:
                    continue
                filtered.append(x)
            except Exception:
                continue
    except Exception:
        filtered = list(universe_all)
    universe = filtered[:g.universe_size]
    if len(universe) < g.stock_num:
        return

    # ── Step 2: 单次多字段 get_price (铁律2/14: 宽表向量化) ──
    lb = g.lookback
    px = get_price(universe, count=lb, end_date=prev_date, frequency="daily",
                   fields=["open", "high", "low", "close", "volume"],
                   skip_paused=False, fq="pre")
    data = {}
    for f in ["open", "high", "low", "close", "volume"]:
        df = _unwrap_panel(px, f)
        if df is None:
            if not getattr(g, "diag_field_%s" % f, False):
                log.info("[DIAG] 字段 %s 缺失 at %s" % (f, prev_date))
                setattr(g, "diag_field_%s" % f, True)
            return
        data[f] = df
    close_df = data["close"]
    if close_df.shape[0] < 65 or close_df.shape[1] == 0:
        if not getattr(g, "diag_no_data", False):
            log.info("[DIAG] get_price 不足 at %s shape=%s" % (prev_date, close_df.shape))
            g.diag_no_data = True
        return

    # 拥挤度代理: top5% 20日均成交额集中度 (P-20260814-004 轻量版)
    try:
        amt = (close_df * data["volume"]).rolling(20, min_periods=10).mean().iloc[-1]
        amt = amt.dropna()
        if len(amt) > 50:
            thr = amt.quantile(0.95)
            g.crowding_hist.append(float(amt[amt >= thr].sum() / amt.sum()))
    except Exception:
        pass

    # ── Step 3: 三袖因子计算 (全部宽表向量化; 各袖用自己验证形态的 rank 参照系) ──
    # 袖A: King 验证形态 = 过滤池前 500 截断
    pool_a = universe[:g.pool_a_size]
    # 袖B: GP002 验证形态 = 全池 2000 (= universe)
    # 袖C: FA v2 验证形态 = 000985 成分 + mcap≥5亿
    fa_series = {
        "overnight5": factor_overnight5(data).iloc[-1],
        "tvma20": factor_tvma20(data).iloc[-1],
        "dollar20": factor_dollar20(data).iloc[-1],
        "tstd_cv": factor_tstd_cv(data).iloc[-1],
    }
    # 袖B: gp002
    gp_s = factor_gp002(data).iloc[-1]

    # fundamentals: PB (FA bp 因子) + 市值 (袖C 池过滤), 铁律7: date=previous_date
    pb_map = {}
    mc_map = {}
    try:
        q = query(valuation.code, valuation.pb_ratio, valuation.market_cap).filter(
            valuation.code.in_(universe))
        val = get_fundamentals(q, date=prev_date)
        if val is not None and len(val) > 0:
            pb_map = dict(zip(val["code"], val["pb_ratio"]))
            mc_map = dict(zip(val["code"], val["market_cap"]))
    except Exception:
        pb_map = {}
        mc_map = {}

    idx_stocks = None
    try:
        idx_stocks = set(get_index_stocks("000985.XSHG", date=prev_date))
    except Exception:
        idx_stocks = None
    pool_c = [x for x in universe if mc_map.get(x, 0.0) >= 5.0]
    if idx_stocks is not None and len(pool_c) > 0:
        in_idx = [x for x in pool_c if x in idx_stocks]
        if len(in_idx) >= 100:           # 成分过滤有效才用, 否则退回纯市值过滤
            pool_c = in_idx
    if len(pool_c) < 100:
        pool_c = [x for x in universe if mc_map.get(x, 0.0) >= 5.0]

    fa_base = _fa_base(data, pb_map, universe)
    fa_comp = _fa_composites(fa_base)

    # rank 化 — 各袖在自己验证池上 rank (参照系保真, v2 关键修复)
    def _to_rp(series, pool):
        vals = series.reindex(pool).values.astype(float)
        valid = np.isfinite(vals)
        return rank_pct(vals, valid), valid

    rp_a = {}
    for k in ["overnight5", "tvma20", "dollar20", "tstd_cv"]:
        rp_a[k], _ = _to_rp(fa_series[k], pool_a)
    rp_gp, v_gp = _to_rp(gp_s, universe)
    rp_c = {}
    v_c = {}
    for k in FA_ICIR:
        rp_c[k], v_c[k] = _to_rp(fa_comp[k], pool_c)

    if int(np.sum(v_gp)) < 10 and all(int(np.sum(v)) < 10 for v in rp_a.values()):
        return

    # 袖A 合成: 等权 rank 积 (王者方法论, 在 pool_a 参照系上)
    na = len(pool_a)
    comp_a = np.ones(na)
    for k in rp_a:
        comp_a = comp_a * rp_a[k]
    # 袖C 合成: ICIR 加权 (FA v2 已验证设计, 在 pool_c 参照系上)
    nc = len(pool_c)
    comp_c = np.zeros(nc)
    for k in FA_ICIR:
        comp_c = comp_c + FA_W[k] * rp_c[k]

    s_a = pd.Series(comp_a, index=pool_a)
    s_b = pd.Series(rp_gp, index=universe)
    s_c = pd.Series(comp_c, index=pool_c)
    pools = {"A": pool_a, "B": universe, "C": pool_c}

    # ── Step 4: 滞后 IC 更新 (上周 rp → 本周收益, P-001) ──
    snap_closes = {}
    for x in close_df.columns:
        lv = close_df[x].iloc[-1]
        if np.isfinite(lv):
            snap_closes[x] = float(lv)
    if g.last_stocks is not None:
        from scipy.stats import spearmanr
        for f in IC_FACTORS:
            lv_arr = g.last_values.get(f)
            pool = g.last_pools.get(FACTOR_SLEEVE[f])
            if lv_arr is None or pool is None:
                continue
            fvals, rets = [], []
            for i, x in enumerate(pool):
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

    # ── Step 5: 保存本周快照 (因子值与其袖池对齐) ──
    g.last_stocks = list(universe)
    g.last_pools = pools
    g.last_values = {
        "overnight5": rp_a["overnight5"].copy(),
        "tvma20": rp_a["tvma20"].copy(),
        "dollar20": rp_a["dollar20"].copy(),
        "tstd_cv": rp_a["tstd_cv"].copy(),
        "gp002": rp_gp.copy(),
    }
    for k in FA_ICIR:
        g.last_values[k] = rp_c[k].copy()
    g.last_closes = {}
    for x in universe:
        if x in snap_closes:
            g.last_closes[x] = snap_closes[x]

    # ── Step 6: 逆波动率权重 (每4周) ──
    if g.trade_days % 4 == 0:
        _update_sleeve_weights()

    # ── Step 7: 三袖独立选股 (v2: 取消互斥 — 各袖在验证形态下独立选股,
    #    重叠股在 Step 9 合并股数下单, 保留各袖独立性) ──
    valid_map_b = {x: bool(v_gp[i]) for i, x in enumerate(universe)}
    v_c_first = v_c["comp1_overnight_tvma"]
    valid_map_c = {x: bool(v_c_first[i]) for i, x in enumerate(pool_c)}
    targets = {}
    targets["A"] = _top_from_scores(s_a, {x: True for x in pool_a},
                                    SLEEVE_TOPN["A"], set())
    targets["B"] = _top_from_scores(s_b, valid_map_b, SLEEVE_TOPN["B"], set())
    targets["C"] = _top_from_scores(s_c, valid_map_c, SLEEVE_TOPN["C"], set())

    all_targets = set()
    for s in SLEEVES:
        all_targets |= set(targets[s])
    if len(all_targets) == 0:
        return

    # ── Step 8: 平仓 (v2: order_target 清仓语义, 不在合计持仓即卖) ──
    cd_order = get_current_data()
    for x in list(port.keys()):
        if x not in all_targets:
            try:
                if cd_order[x].paused:
                    continue
                pos = port[x]
                if pos.total_amount < 100:
                    continue
                order_target(x, 0)
            except Exception:
                pass

    # ── Step 9: 合并股数下单 (v2: 重叠股 lots 相加 → 统一 order_target;
    #    day_open 口径 + 手数取整 + 现金 break) ──
    total_value = context.portfolio.total_value
    all_lots = {}                          # x → [lots, ref_px]
    for s in SLEEVES:
        if len(targets[s]) == 0:
            continue
        budget = g.sleeve_w[s] * total_value
        unit = budget / len(targets[s])
        for x in targets[s]:
            try:
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
                if x in all_lots:
                    all_lots[x][0] += lots
                else:
                    all_lots[x] = [lots, ref_px]
            except Exception:
                continue

    ordered = 0
    for x in sorted(all_lots.keys()):
        try:
            lots, ref_px = all_lots[x]
            if context.portfolio.available_cash < lots * ref_px * 0.5:
                break                     # 现金不足, 停止开仓
            order_target(x, lots)
            ordered += 1
        except Exception:
            continue

    # ── Step 10: 袖暴露记录 (组合加权合成分, P-20260814-004) + 更新记账 ──
    # 重叠股记账归首个持有袖 (A>B>C), 防止市值双计污染逆波动率权重
    owner = {}
    for s in SLEEVES:
        for x in targets[s]:
            if x not in owner:
                owner[x] = s
    s_a_dict = s_a.to_dict()
    s_b_dict = s_b.to_dict()
    s_c_dict = s_c.to_dict()
    score_map = {"A": s_a_dict, "B": s_b_dict, "C": s_c_dict}
    for s in SLEEVES:
        vals = [score_map[s].get(x, np.nan) for x in targets[s]]
        vals = [x for x in vals if x == x]
        g.sleeve_expo[s].append(float(np.mean(vals)) if vals else np.nan)
        g.sleeve_holdings[s] = [x for x in targets[s] if owner[x] == s]
        g.sleeve_values[s] = cur_values[s]

    # ── Step 11: 日志 & 半年报 ──
    if g.trade_days <= 8 or g.trade_days % 13 == 0:
        log.info("三袖P0 | week=%d | A=%d B=%d C=%d | ordered=%d | w(A,B,C)=(%.2f,%.2f,%.2f) | cash=%.0f" % (
            g.trade_days, len(targets["A"]), len(targets["B"]), len(targets["C"]),
            ordered, g.sleeve_w["A"], g.sleeve_w["B"], g.sleeve_w["C"],
            context.portfolio.available_cash))
    if g.trade_days - g.last_report_week >= 26:
        g.last_report_week = g.trade_days
        log.info("=" * 60)
        log.info("  三袖组合P0 半年报 (P-001 + P-20260814-004)  week=%d" % g.trade_days)
        log.info("=" * 60)
        for s in SLEEVES:
            expo = np.array([x for x in g.sleeve_expo[s] if x == x])
            if len(expo) >= 8:
                log.info("  袖%s %-12s 暴露均值=%.4f  暴露σ(20w滚动末段)=%.4f" % (
                    s, SLEEVE_LABEL[s], float(np.mean(expo)), float(np.std(expo[-20:]))))
        _print_ic_report()
        log.info("=" * 60)

    gc.collect()


# ══════════════ 收尾报告 (回测结束自动调用) ══════════════
def on_strategy_end(context):
    log.info("=" * 60)
    log.info("  三袖组合P0 最终报告 — 结果请按 jq_return 百分比格式反馈 D+")
    log.info("=" * 60)
    log.info("  最终权重 w(A,B,C)=(%.2f, %.2f, %.2f)" % (
        g.sleeve_w["A"], g.sleeve_w["B"], g.sleeve_w["C"]))
    _print_ic_report()
    log.info("  [提示] 对照基准: King 6复合 +182.57%/SR0.69/MaxDD-35.41%")
    log.info("=" * 60)
