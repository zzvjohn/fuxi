# -*- coding: utf-8 -*-
r"""
【修正版 v6】smallcap 低放量池内 20 日反转 — 单因子口径 A/B 实验
================================================================
立项依据 (2026-09-01 组合回测诊断):
  袖A ratio_extreme   JQ +88.85%/SR0.387/ICIR 0.593
  袖B concentration   JQ +85.51%/SR0.370/ICIR 0.602
  组合 50/50          JQ +43.79%/SR0.134/DD27.43%  ← ELIMINATED

🔴 诊断中发现的因子语义真相 (比组合失败更严重):
  两因子公式:
    A: signal = -ret20_rank * (vr_rank  > 0.8)
    B: signal = -ret20_rank * (amt_rank > 0.8) * close_rank
  (1) 两因子的"反转部分"完全相同: 都是 -ret20_rank, 只依赖 close。
      差异仅在掩码(量比 vs 额集中度), 故周IC相关 0.691、持仓重叠 85% 属必然。
  (2) 掩码外 signal 恒 = 0, 掩码内 signal ∈ [-1,0)。
      comp = -signal 升序取 top30 → 0 永远排在负值前 → **top30 100% 落在掩码外**,
      而掩码外 signal 全为 0、彼此无法区分 → 实际靠 universe 代码序取前 30。
  (3) 因此: 两单因子 JQ 收益的本质 = "低放量股票池的池子收益"(低量比/低额集中度溢价),
      **反转信号从未参与选股**; 正 IC(0.059/0.060) 衡量的是掩码效应, 非反转效应。

【v6 要回答的核心问题】在低放量池内, 20 日反转到底有没有 alpha?
  → A/B 单变量实验 (同一池、同一周、同一执行层):
     组A (实际交易): 池内按 -ret20_rank 排序取 top30 (反转真正参与选股)
     组B (仅统计)  : 池内按代码序取前 30 (原版口径的等价物, 无反转信息)
     两组等权 1/30, 半年报输出累计收益对比 → 一次回测分离"池效应"与"反转效应"
  → 判据: A 显著 > B 则反转有效(因子名副其实); A ≈ B 则反转无效(因子=低放量溢价);
          A < B 则反转有害(应取反转反向或改用动量)

【口径】池 500(0/3/6开头) / 选股池=双低放量共识(~maskA & ~maskB) / top30 等权 3.33%
        / lookback 65 / 周频 / 基准 000905
【🔴 执行层铁律】持仓回到 30 只、单只权重 1/30=3.33% (与单因子文件同口径):
   P-20260901-008 差额保护(|delta|<100股拒单) 的杀伤力与单只权重成反比 —
   组合 v1 把非重叠股权重压到 1/60=1.67% → 成交率崩到 50.1%(单因子 98.7%),
   一半调仓指令从未执行, 是组合收益腰斩的主因。30只/3.33% 口径下该保护几乎无害。
【回测区间】2021-01-01 ~ 2026-07-30 (与单因子/组合严格对齐, 281 周)
【P-001 合规】输出 IC_reversal(池内反转) 与 IC_mask(低放量溢价对照) 双轨 IC/ICIR
【版本】v6 2026-09-01  立项 (诊断驱动, 非组合 — 单因子口径修正)
      v6.1 2026-09-01  P-20260901-013: 删除 get_price 的 panel=False 参数。
        🔴 JQ get_price 无 panel 参数 — v6 首跑真实 JQ "无收益"根因:
        第一周调仓即 TypeError(unexpected keyword argument 'panel') → 从未建仓 → 面板 0 收益。
        组合 v5 / 单因子文件均无此参数且真实跑通, v6 误加属引入性 bug (stub 的 get_price
        有 panel 兜底参数故本地全通, 掩盖了真实 JQ 签名不兼容)。
"""
import numpy as np
import pandas as pd

VERSION = "v6.1-20260901 (低放量池内反转 A/B · 30只/3.33% · P-012禁裸sum · P-013删panel参数)"


def initialize(context):
    g.stock_num = 30          # 持仓数 (与单因子同口径, 保证执行层成交率)
    g.lookback = 65
    g.universe_size = 500
    g.trade_days = 0
    g.ic_records = {"reversal": [], "mask_lowvol": []}
    g.last_values = {"reversal": None, "mask_lowvol": None}
    g.last_stocks = None
    g.last_closes = {}
    g.last_groups = None      # {"A": {s: w}, "B": {s: w}} 供下周 A/B 收益对账
    g.last_report_week = 0
    g.weekly_rets = {"A": [], "B": []}
    g.skip_stat = {}          # 执行层跳过原因计数器 (诊断)

    set_benchmark("000905.XSHG")
    set_option("use_real_price", True)
    set_option("avoid_future_data", True)
    set_order_cost(OrderCost(open_tax=0, close_tax=0.001,
                             open_commission=0.0001, close_commission=0.0001,
                             min_commission=5), type="stock")
    set_slippage(FixedSlippage(0.003))

    _st = type(np.sum({1: 1.0}.values())).__name__
    log.info("[SELFTEST] np.sum(dict_values) 返回类型=%s (JQ 沙箱 sum 被 numpy 遮蔽的旁证)" % _st)
    log.info("[VERSION] %s — 若此行不是 v6, 说明 JQ 上文件未更新!" % VERSION)
    log.info("[AB-TEST] 组A=池内反转排序top30(交易) | 组B=池内代码序top30(仅统计) | 池500 启动")
    run_weekly(rebalance, 1, time="open", reference_security="000300.XSHG")


# ═══ 辅助 ═══
def _sum_vals(seq):
    """P-20260901-012: 手动累加, 禁止裸 sum() (JQ 沙箱 sum=numpy.sum 会静默返回 dict_values)"""
    _t = 0.0
    for _v in seq:
        _t = _t + _v
    return _t


def _unwrap_panel(px, field):
    """get_price 返回 Panel/DataFrame(MultiIndex)/dict 三层兼容拆字段"""
    _P = getattr(pd, "Panel", None)
    if _P is not None and isinstance(px, _P):
        items = getattr(px, "items", [])
        if field in items:
            return px[field]
        try:
            return px.minor_xs(field)
        except Exception:
            return None
    if isinstance(px, dict):
        return px.get(field)
    if hasattr(px, "columns"):
        cols = px.columns
        if isinstance(cols, pd.MultiIndex):
            if field in cols.get_level_values(0):
                return px[field]
            if len(cols.levels) > 1 and field in cols.get_level_values(1):
                return px.xs(field, axis=1, level=1)
        return px
    return px


def _bump(key, n=1):
    g.skip_stat[key] = g.skip_stat.get(key, 0) + n


# ═══ IC/ICIR 报告 (P-001: 双轨) ═══
def _print_ic_report():
    label = {"reversal": "池内反转(核心)", "mask_lowvol": "低放量溢价(对照)"}
    for name in ["reversal", "mask_lowvol"]:
        recs = g.ic_records.get(name, [])
        if len(recs) == 0:
            log.info("[IC] %-16s NO IC RECORDS" % name)
            continue
        ic_arr = np.array(recs)
        ic_mean = float(np.mean(ic_arr))
        ic_std = float(np.std(ic_arr))
        icir = ic_mean / ic_std if ic_std > 0 else 0.0
        ic_pos = float(np.mean(ic_arr > 0))
        log.info("[IC] %-16s n=%4d  ICmean=%+.4f  ICIR=%+.4f  IC>0=%.1f%%" % (
            "%s(%s)" % (name, label.get(name, "")), len(ic_arr), ic_mean, icir, ic_pos * 100))


def _print_ab_report():
    """A/B 组累计收益对比 — 分离池效应与反转效应"""
    out = []
    for k in ("A", "B"):
        rets = np.array(g.weekly_rets.get(k, []))
        if len(rets) < 10:
            out.append("%s=数据不足" % k)
            continue
        mean_r = float(np.mean(rets))
        std_r = float(np.std(rets))
        sharpe = mean_r / std_r * np.sqrt(52) if std_r > 0 else 0.0
        cum = float(np.prod(1 + rets) - 1)
        nav = np.cumprod(1 + rets)
        mdd = float((nav / np.maximum.accumulate(nav) - 1).min())
        out.append("%s 累计=%+.1f%% SR=%+.3f MDD=%+.1f%%" % (k, cum * 100, sharpe, mdd * 100))
        g.__setattr__("ab_%s_cum" % k, cum)
    log.info("[AB] n=%d | %s" % (len(g.weekly_rets.get("A", [])), " | ".join(out)))
    ca = getattr(g, "ab_A_cum", None)
    cb = getattr(g, "ab_B_cum", None)
    if ca is not None and cb is not None:
        diff = ca - cb
        verdict = ("反转有效(A>B)" if diff > 0.15 else
                   "反转有害(A<B)" if diff < -0.15 else
                   "反转无效(A≈B, 因子=低放量溢价)")
        log.info("[AB-VERDICT] A-B=%+.1fpp → %s" % (diff * 100, verdict))


# ═══ 周度调仓 ═══
def rebalance(context):
    g.trade_days += 1
    if g.trade_days < 5:
        return
    prev_date = context.previous_date

    # ── Step 1: 池过滤 (0/3/6 开头, 停牌/ST/涨停剔除) ──
    universe_all = get_all_securities(["stock"], prev_date).index.tolist()
    universe_all = [s for s in universe_all if s.startswith(("0", "3", "6"))][:2000]
    filtered = []
    try:
        cd0 = get_current_data()
        for s in universe_all:
            try:
                if cd0[s].paused or cd0[s].is_st:
                    continue
                if cd0[s].high_limit <= cd0[s].last_price:
                    continue
                filtered.append(s)
            except Exception:
                continue
    except Exception:
        filtered = list(universe_all)
    universe = [s for s in filtered if s.startswith(("0", "3", "6"))][:g.universe_size]
    if len(universe) < g.stock_num:
        return

    # ── Step 2: get_price (单次多字段: close + volume + money) ──
    # P-20260901-013: 🔴 JQ get_price 无 panel 参数! v6 曾误加 panel=False →
    # 真实 JQ 第一周调仓即 TypeError(got an unexpected keyword argument 'panel')
    # → 从未建仓 → 面板 0 收益。组合 v5 / 单因子文件均无此参数且真实跑通。
    # JQ 返回 DataFrame(columns=MultiIndex(field, security)), _unwrap_panel 处理。
    lb = g.lookback
    px = get_price(universe, count=lb, end_date=prev_date, frequency="daily",
                   fields=["close", "volume", "money"], skip_paused=False, fq="pre")
    close_df = _unwrap_panel(px, "close")
    if close_df is None or close_df.shape[0] < 22 or close_df.shape[1] == 0:
        if not getattr(g, "diag_no_data", False):
            log.info("[DIAG] get_price 空/不足 at %s shape=%s" % (prev_date, getattr(close_df, "shape", None)))
            g.diag_no_data = True
        return
    data = {"close": close_df}
    for f in ["volume", "money"]:
        df = _unwrap_panel(px, f)
        if df is None:
            if not getattr(g, "diag_missing_field", False):
                log.info("[DIAG] 字段 %s 缺失 at %s" % (f, prev_date))
                g.diag_missing_field = True
            return
        data[f] = df

    valid_u = [s for s in universe
               if s in close_df.columns and int((~close_df[s].isna()).sum()) >= 30]
    if len(valid_u) < g.stock_num:
        return

    # ── Step 3: 信号计算 (低放量共识池 + 池内反转) ──
    try:
        close_df = data["close"]
        vol = data["volume"].reindex(index=close_df.index, columns=close_df.columns)
        amt = data["money"].reindex(index=close_df.index, columns=close_df.columns)

        ret20 = close_df.pct_change(20)
        vr = vol / (vol.rolling(20).mean() + 1e-8)
        amt_conc = amt.rolling(5).mean() / (amt.rolling(20).mean() + 1e-8)

        ret20_rank = ret20.rank(pct=True, axis=1)
        vr_rank = vr.rank(pct=True, axis=1)
        amt_rank = amt_conc.rank(pct=True, axis=1)

        last_ret20_rank = ret20_rank.iloc[-1].reindex(valid_u).values.astype(float)
        last_vr_rank = vr_rank.iloc[-1].reindex(valid_u).values.astype(float)
        last_amt_rank = amt_rank.iloc[-1].reindex(valid_u).values.astype(float)
    except Exception as _e:
        if not getattr(g, "diag_calc", False):
            log.info("[DIAG] 信号计算异常: %r at %s" % (_e, prev_date))
            g.diag_calc = True
        return

    fin = np.isfinite(last_ret20_rank) & np.isfinite(last_vr_rank) & np.isfinite(last_amt_rank)

    # 选股池 = 双低放量共识 (两掩码之外的交集: 非放量 且 非高集中度)
    pool_mask = fin & (last_vr_rank <= 0.8) & (last_amt_rank <= 0.8)
    n_pool = int(np.sum(pool_mask))
    if n_pool < g.stock_num:
        # 共识池不足时降级为单一低放量条件, 保证可比性与连续性
        pool_mask = fin & (last_vr_rank <= 0.8)
        n_pool = int(np.sum(pool_mask))
        if n_pool < g.stock_num:
            return

    # 组A 信号: 池内 -ret20_rank (越超跌分越高); 池外 -inf 不参与
    sig_rev = np.where(pool_mask, -last_ret20_rank, np.nan)
    # 组B 对照信号: 池外 NaN, 池内记 0 (常值) → argsort 稳定排序落到代码序
    sig_code = np.where(pool_mask, 0.0, np.nan)

    # ── Step 4: 滞后 IC (上周信号 → 本周收益 spearman) ──
    snap_closes = {}
    for s in close_df.columns:
        lv = close_df[s].iloc[-1]
        if np.isfinite(lv):
            snap_closes[s] = float(lv)

    if g.last_stocks is not None:
        _ic_pair("reversal", g.last_stocks, g.last_values.get("reversal"), snap_closes)
        _ic_pair("mask_lowvol", g.last_stocks, g.last_values.get("mask_lowvol"), snap_closes)

    # ── Step 4.5: A/B 组周收益对账 (上轮权重 × 上周收盘 → 本周收盘) ──
    if g.last_groups is not None and len(g.last_closes) > 0:
        for k in ("A", "B"):
            wmap = g.last_groups.get(k) or {}
            prev_parts = []
            cur_parts = []
            for s in wmap:
                pc = g.last_closes.get(s)
                cc = snap_closes.get(s)
                if pc and cc and pc > 0 and cc > 0:
                    prev_parts.append(wmap[s])
                    cur_parts.append(wmap[s] * (cc / pc))
            if prev_parts and _sum_vals(prev_parts) > 0:
                nav_prev = _sum_vals(prev_parts)
                nav_cur = _sum_vals(cur_parts) + (1.0 - nav_prev)
                g.weekly_rets[k].append(float(nav_cur / nav_prev - 1.0))

    g.last_stocks = list(valid_u)
    g.last_values = {
        "reversal": np.where(pool_mask, -last_ret20_rank, np.nan),
        # 低放量溢价对照: 池内=1, 池外=0 (原版掩码效应的等价度量)
        "mask_lowvol": np.where(fin, pool_mask.astype(float), np.nan),
    }
    g.last_closes = {s: snap_closes[s] for s in valid_u if s in snap_closes}

    # ── Step 5: 选股 (组A 交易 / 组B 仅统计) ──
    comp_a = np.where(np.isfinite(sig_rev), -sig_rev, np.inf)
    comp_b = np.where(np.isfinite(sig_code), -sig_code, np.inf)
    # P-20260901-009: JQ 旧 numpy 不支持 kind="stable", 用 mergesort (稳定排序语义等价)
    idx_a = np.argsort(comp_a, kind="mergesort")[:g.stock_num]
    idx_b = np.argsort(comp_b, kind="mergesort")[:g.stock_num]
    group_a = [valid_u[i] for i in idx_a if comp_a[i] < np.inf]
    group_b = [valid_u[i] for i in idx_b if comp_b[i] < np.inf]
    if len(group_a) < 10:
        return

    target = {}
    w_each = 1.0 / len(group_a)          # 3.33% (与单因子同口径 → 执行层成交率有保障)
    for s in group_a:
        target[s] = w_each
    g.last_groups = {
        "A": dict(target),
        "B": {s: (1.0 / len(group_b)) for s in group_b} if group_b else {},
    }

    # ── Step 6: 执行 (lhb v2 修复版骨架 + P-20260901-008 差额保护 + 分类诊断) ──
    cd_order = get_current_data()

    for s in list(context.portfolio.positions.keys()):
        if s in target:
            continue
        try:
            if cd_order[s].paused:
                _bump("sell_paused")
                continue
            pos = context.portfolio.positions[s]
            if pos.total_amount < 100:
                _bump("sell_lt100")
                continue
            order_target(s, 0)
        except Exception:
            _bump("sell_exc")
            continue

    total_value = context.portfolio.total_value
    ordered = 0
    for s in target:
        try:
            if context.portfolio.available_cash < total_value * w_each * 0.5:
                _bump("buy_cash_break")
                break
            d = cd_order[s]
            if d.paused or d.is_st:
                _bump("buy_paused_st")
                continue
            if d.high_limit <= d.last_price:
                _bump("buy_limit_up")
                continue
            ref_px = d.day_open if (d.day_open and d.day_open > 0) else d.last_price
            if ref_px <= 0 or total_value * w_each / ref_px < 100:
                _bump("buy_lt100shares")
                continue
            # P-20260901-008: 差额微调保护 (30只/3.33% 口径下几乎无害)
            _pos = context.portfolio.positions.get(s)
            if _pos is not None and getattr(_pos, "total_amount", 0) > 0:
                _delta = int(total_value * w_each / ref_px) - int(_pos.total_amount)
                if abs(_delta) < 100:
                    _bump("buy_delta_lt100")
                    continue
            order_target_value(s, total_value * w_each)
            ordered += 1
        except Exception:
            _bump("buy_exc")
            continue

    # ── Step 7: 日志 & 定期报告 ──
    log.info("reversal_inpool | week=%d | 池内=%d | 目标=%d | 成交=%d (%.0f%%)" % (
        g.trade_days, n_pool, len(target), ordered, 100.0 * ordered / max(len(target), 1)))

    if g.trade_days - g.last_report_week >= 26:
        g.last_report_week = g.trade_days
        log.info("=" * 55)
        log.info("  v6 低放量池内反转 A/B 半年报 (P-001 双轨 IC)")
        log.info("=" * 55)
        _print_ic_report()
        _print_ab_report()
        _sk = sorted(g.skip_stat.items(), key=lambda kv: -kv[1])
        if _sk:
            log.info("[SKIP] " + "  ".join("%s=%d" % (k, v) for k, v in _sk[:6]))
        log.info("=" * 55)


def _ic_pair(name, last_stocks, lv_arr, snap_closes):
    if lv_arr is None:
        return
    factor_vals = []
    rets = []
    for i, s in enumerate(last_stocks):
        pc = g.last_closes.get(s)
        cc = snap_closes.get(s)
        fv = lv_arr[i] if i < len(lv_arr) else np.nan
        if pc and cc and pc > 0 and cc > 0 and not np.isnan(fv):
            factor_vals.append(fv)
            rets.append((cc / pc) - 1.0)
    if len(factor_vals) >= 30:
        try:
            from scipy.stats import spearmanr
            ic, _ = spearmanr(factor_vals, rets)
            if ic == ic:
                g.ic_records.setdefault(name, []).append(float(ic))
        except Exception:
            pass
