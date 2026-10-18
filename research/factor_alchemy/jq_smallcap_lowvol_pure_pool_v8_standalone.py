# -*- coding: utf-8 -*-
r"""
【v8】smallcap 低放量溢价·干净复刻版 — 显式掩码内稳定代码序 top30
================================================================
立项依据 (2026-09-01 v7 面板终局 ELIMINATED 后):
  三变体证据链闭环 (同区间 2021-01~2026-07-30, 全部真实 JQ 面板):
    代码序 (单因子文件, vr<=0.8 补集, 0平局bug副产品)  +88.85% / SR0.387 / DD34.48%  ← 唯一真实alpha
    量比升序 (v7, 双低放量池内连续排序)               +40.00% / SR0.100 / DD45.63%  ← ELIMINATED
    反转排名 (v6A, 双低放量池内反转排序)              +12.98% / SR-0.066 / DD46.97%  ← ELIMINATED
  IC 终局: lowvol_rank(池内连续排序) ICIR 0.018=纯噪声 vs mask_lowvol(二值池) ICIR 0.617 强
  → 核心结论: 低放量溢价 = 纯池选择效应(二值剔除极端放量), 池内任何信号排序 ICIR≈0 且摧毁收益;
    单因子文件的"代码序bug"实际无意实现了"池内零信息选择"=最优形态

🔴 v8 目标: 把唯一赢家从 bug 副产品转正为显式干净实现, 一箭双雕:
  ① 验证 +88.85% 可复现 (若 v8 显著低于 88.85% → 单因子收益含隐藏因素, 深挖)
  ② 若可复现 → v8 即低放量溢价的正式实盘形态 (无 0 平局陷阱/无内部nav/IC 合规)

【v8 口径】池 500(0/3/6开头) / 选股池 = vr_rank<=0.8 单条件补集 (对齐唯一赢家 ratio_extreme)
          / 池内 comp=0 常数 + mergesort 稳定排序 → 代码序 top30 (显式掩码内: 池外=np.inf)
          / 等权 1/30 / lookback 65 / 周频 / 基准 000905 / 区间 2021-01-01 ~ 2026-07-30
【P-001 合规】输出 IC_mask_lowvol(交易信号=二值池本身) 单轨 IC/ICIR
【🔴 P-20260901-014 铁律】无内部自算净值 — 面板为唯一裁决口径, 半年报只报 [IC]+[SKIP]
【版本】v8 2026-09-01  立项 (v7 ELIMINATED 后的干净复刻/转正版)
      v7 前史: 池内量比升序 +40% 证伪, 见原文件头; v6 前史: 池内反转 +12.98% 证伪
"""
import numpy as np
import pandas as pd

VERSION = "v8-20260901 (低放量溢价干净复刻版 · 掩码内代码序top30 · 无内部nav · P-014)"


def initialize(context):
    g.stock_num = 30          # 持仓数 (与单因子同口径, 保证执行层成交率)
    g.lookback = 65
    g.universe_size = 500
    g.trade_days = 0
    g.ic_records = {"mask_lowvol": []}
    g.last_values = {"mask_lowvol": None}
    g.last_stocks = None
    g.last_closes = {}
    g.last_report_week = 0
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
    log.info("[VERSION] %s — 若此行不是 v8, 说明 JQ 上文件未更新!" % VERSION)
    log.info("[MODE] 低放量溢价干净复刻版: 掩码内代码序top30 | 池500 | 无内部nav(P-014) 启动")
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
    label = {"mask_lowvol": "低放量二值(交易=池选择本身)"}
    for name in ["mask_lowvol"]:
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

        vr = vol / (vol.rolling(20).mean() + 1e-8)

        vr_rank = vr.rank(pct=True, axis=1)

        last_vr_rank = vr_rank.iloc[-1].reindex(valid_u).values.astype(float)
    except Exception as _e:
        if not getattr(g, "diag_calc", False):
            log.info("[DIAG] 信号计算异常: %r at %s" % (_e, prev_date))
            g.diag_calc = True
        return

    fin = np.isfinite(last_vr_rank)

    # 选股池 = vr_rank<=0.8 单条件补集 (对齐唯一赢家 ratio_extreme 掩码口径)
    pool_mask = fin & (last_vr_rank <= 0.8)
    n_pool = int(np.sum(pool_mask))
    if n_pool < g.stock_num:
        return

    # v8 选股信号: 池内常数 0 (零信息) + 池外 NaN → mergesort 稳定排序保序 = 代码序
    # 显式掩码内选股 (P-20260901-006 教训: 掩码外必须 NaN/inf, 严禁 0 平局优先)
    sig_pool = np.where(pool_mask, 0.0, np.nan)

    # ── Step 4: 滞后 IC (上周信号 → 本周收益 spearman) ──
    snap_closes = {}
    for s in close_df.columns:
        lv = close_df[s].iloc[-1]
        if np.isfinite(lv):
            snap_closes[s] = float(lv)

    if g.last_stocks is not None:
        _ic_pair("mask_lowvol", g.last_stocks, g.last_values.get("mask_lowvol"), snap_closes)


    g.last_stocks = list(valid_u)
    g.last_values = {
        # v8 交易信号 = 低放量二值池本身: 池内=1, 池外=0
        "mask_lowvol": np.where(fin, pool_mask.astype(float), np.nan),
    }
    g.last_closes = {s: snap_closes[s] for s in valid_u if s in snap_closes}

    # ── Step 5: 选股 (池内代码序 top30, 显式掩码内) ──
    comp = np.where(np.isfinite(sig_pool), -sig_pool, np.inf)
    # P-20260901-009: JQ 旧 numpy 不支持 kind="stable", 用 mergesort (稳定排序语义等价)
    idx = np.argsort(comp, kind="mergesort")[:g.stock_num]
    group = [valid_u[i] for i in idx if comp[i] < np.inf]
    if len(group) < 10:
        return

    target = {}
    w_each = 1.0 / len(group)           # 3.33% (与单因子同口径 → 执行层成交率有保障)
    for s in group:
        target[s] = w_each

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
    log.info("lowvol_pure_pool | week=%d | 池内=%d | 目标=%d | 成交=%d (%.0f%%)" % (
        g.trade_days, n_pool, len(target), ordered, 100.0 * ordered / max(len(target), 1)))

    if g.trade_days - g.last_report_week >= 26:
        g.last_report_week = g.trade_days
        log.info("=" * 55)
        log.info("  v8 低放量溢价干净复刻版 半年报 (P-001 单轨 IC, 无内部nav)")
        log.info("=" * 55)
        _print_ic_report()
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
