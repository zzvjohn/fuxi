# -*- coding: utf-8 -*-
"""
【组合】smallcap 双雄 50/50 双袖组合: 20日反转×量比极值 + 20日反转×额集中度×市值
MMR 组合验证 (2026-09-01): 两因子均为 JQ 单因子近PASS (MARGINAL),
  袖A ratio_extreme  +88.85%/SR0.387/ICIR 0.593
  袖B concentration  +85.51%/SR0.370/ICIR 0.602
  本地预检: 周IC相关0.691(低于共线硬阈值0.8); 50/50合成ICIR 0.650;
           本地top30持仓重叠85%, 组合收益端≈单因子 (本地折扣口径, 仅方向参考)
【组合方式】双袖 50/50: 每袖独立选 top30 各持一半资金, 重叠股权重叠加(=1/30)
【🔴 口径真相 (2026-09-01 本地审计实锤)】两因子均为乘法掩码结构, 掩码外 signal=0 恒大于
  掩码内负值 → comp=-signal 升序取 top30 时 100% 选中掩码外股票 (本地验证 0 值占比 100%、
  掩码内占比 0%)。即 JQ 单因子收益的本质 = "剔除量比/额集中度前20%股票的池子收益"
  (低放量溢价), 而非"小盘20日反转激活"。IC 正号方向 = 掩码外跑赢掩码内, 与选股一致。
  本组合文件严格保持与已回测单因子相同的选股口径 (0 平局优先 + kind=stable 代码序),
  使组合结果与两单因子 JQ 结果可比。修正版 (直白低放量溢价) 另行立项。
【参数】池 500 (0/3/6 开头) / 每袖 top30 / lookback 65 / 周频 / 基准 000905
【回测建议区间】2021-01-01 ~ 2026-07-30 (与单因子 JQ 对齐)
【P-001 合规】每袖因子滞后法 Rank IC/ICIR 半年报 + 组合周收益统计
【性能】单次 get_price 3 字段 65×500×3=9.75万单元格 (mmr6 实测 14 万安全)
【版本】
  v1 2026-09-01  初版 (双袖50/50)
  v2 2026-09-01  P-20260901-009: argsort kind="stable"→"mergesort" (JQ 旧 numpy 不支持)
  v3 2026-09-01  P-20260901-010: w_each 移出循环 (50/50 权重修复) + 每周期 DIAG 日志 + 零持仓告警
  v4 2026-09-01  P-20260901-011: w_sum 加 float() 防呆 + 启动日志打印版本号
                  (诊断已证伪, 见 v5)
  v5 2026-09-01  P-20260901-012 (真根因): 🔴 JQ 沙箱把 numpy 注入用户命名空间,
                  裸 sum() 实为 np.sum → np.sum(dict.values()) 原样返回 dict_values
                  (本机 numpy 2.4.6 实测复现), np.sum(generator) 直接 TypeError。
                  全文件禁用裸 sum(), 改用 _sum_vals() 手动累加; 启动加 [SELFTEST]
                  日志打印 sum(dict_values) 的实际返回类型, 供 JQ 侧实锤。
"""
import numpy as np
import pandas as pd

VERSION = "v5-20260901c (P-009 mergesort + P-010 50/50 + P-012 禁裸sum)"


def initialize(context):
    g.stock_num = 30          # 每袖持仓数
    g.lookback = 65
    g.universe_size = 500
    g.trade_days = 0
    g.ic_records = {"ratio_extreme": [], "concentration": []}
    g.last_values = {"ratio_extreme": None, "concentration": None}
    g.last_stocks = None
    g.last_closes = {}
    g.last_report_week = 0
    g.weekly_rets = []        # 组合周收益 (对账用)
    g.zero_pos_weeks = 0      # 连续零持仓周数 (P-20260901-010 告警)

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
    # P-20260901-012 SELFTEST: 故意用裸 sum 探测 JQ 环境 (返回 float=正常, dict_values=被 numpy 遮蔽)
    _st = type(sum({1: 1.0}.values())).__name__
    log.info("[SELFTEST] sum(dict_values) 返回类型=%s %s" % (
        _st, "(sum 正常)" if _st in ("float", "int") else
             "<-- sum 被 numpy 遮蔽! 代码已用 _sum_vals() 免疫, 无需处理"))
    log.info("[VERSION] %s — 若此行不是 v5, 说明 JQ 上文件未更新!" % VERSION)
    log.info("smallcap双雄50/50: ratio_extreme + concentration | 双袖各top30 | 池500 启动")


# ═══ 辅助 ═══
def _sum_vals(seq):
    """P-20260901-012 🔴 JQ 沙箱铁律: 禁止使用内建 sum()。
    JQ 把 numpy 注入用户命名空间, 裸 sum() 实际是 np.sum:
      · np.sum(dict.values()) → 原样返回 dict_values 对象 (静默错误, 最危险)
      · np.sum(generator)     → numpy>=1.25 直接抛 TypeError
      · np.sum(list)          → np.float64 (侥幸无害)
    一律手动累加, 语义等价于内建 sum 且完全免疫遮蔽。
    本机 numpy 2.4.6 已实测复现上述两类行为。"""
    _t = 0.0
    for _v in seq:
        _t = _t + _v
    return _t


def _rankdata_impl(a):
    """rankdata 降级实现: scipy 不可用时自实现"""
    try:
        from scipy.stats import rankdata
        return rankdata(a)
    except Exception:
        return np.argsort(np.argsort(a)).astype(float) + 1.0


def rank_pct(arr, valid):
    """rank(pct=True) 兼容实现: pct = rank/n_valid (与 pandas 截面 rank 语义一致)"""
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
            names = [c for c in cols.names if c]
            if field in cols.get_level_values(0):
                return px[field]
            if len(cols.levels) > 1 and field in cols.get_level_values(1):
                return px.xs(field, axis=1, level=1)
        return px
    return px


# ═══ 因子 (宽表向量化, 与单因子 JQ 文件口径严格一致) ═══

def _factor_ratio_extreme(stocks, data):
    """袖A: 20日反转×量比极值  signal = -ret20_rank * (vr_rank > 0.8)
    JQ单因子: +88.85%/SR0.387/ICIR 0.593  HIGH=做多"""
    close_df = data["close"]
    vol = data["volume"]
    vol = vol.reindex(index=close_df.index, columns=close_df.columns)
    ret20 = close_df.pct_change(20)
    vol_ratio = vol / (vol.rolling(20).mean() + 1e-8)
    ret20_rank = ret20.rank(pct=True, axis=1)
    vr_rank = vol_ratio.rank(pct=True, axis=1)
    signal = -ret20_rank * (vr_rank > 0.8).astype(float)
    last = signal.iloc[-1]
    arr = last.reindex(stocks).values.astype(float)
    valid = np.isfinite(arr)
    return arr, valid


def _factor_concentration(stocks, data):
    """袖B: 20日反转×额集中度极值×市值  signal = -ret20_rank * (amt_rank>0.8) * close_rank
    JQ单因子: +85.51%/SR0.370/ICIR 0.602  HIGH=做多"""
    close_df = data["close"]
    amount = data["money"]
    amount = amount.reindex(index=close_df.index, columns=close_df.columns)
    ret20 = close_df.pct_change(20)
    amt_conc = amount.rolling(5).mean() / (amount.rolling(20).mean() + 1e-8)
    ret20_rank = ret20.rank(pct=True, axis=1)
    amt_rank = amt_conc.rank(pct=True, axis=1)
    close_rank = close_df.rank(pct=True, axis=1)
    signal = (-ret20_rank) * (amt_rank > 0.8).astype(float) * close_rank
    last = signal.iloc[-1]
    arr = last.reindex(stocks).values.astype(float)
    valid = np.isfinite(arr)
    return arr, valid


FACTOR_FUNC_MAP = {
    "ratio_extreme": _factor_ratio_extreme,
    "concentration": _factor_concentration,
}
FACTOR_NAMES = ["ratio_extreme", "concentration"]


# ═══ IC/ICIR 报告 (P-001) ═══
def _print_ic_report():
    for name in FACTOR_NAMES:
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
            name, len(ic_arr), ic_mean, icir, ic_pos * 100))


def _print_combo_report():
    """组合周收益统计 (对账: 与 JQ 最终指标段互验)"""
    rets = np.array(g.weekly_rets)
    if len(rets) < 10:
        return
    mean_r = float(np.mean(rets))
    std_r = float(np.std(rets))
    sharpe = mean_r / std_r * np.sqrt(52) if std_r > 0 else 0.0
    cum = np.prod(1 + rets) - 1
    nav = np.cumprod(1 + rets)
    mdd = float((nav / np.maximum.accumulate(nav) - 1).min())
    log.info("[COMBO] n=%d 周均收益=%+.4f%%  年化Sharpe=%+.3f  累计=%+.1f%%  MaxDD=%+.1f%%" % (
        len(rets), mean_r * 100, sharpe, cum * 100, mdd * 100))


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
    lb = g.lookback
    px = get_price(universe, count=lb, end_date=prev_date, frequency="daily",
                   fields=["close", "volume", "money"], skip_paused=False, fq="pre")
    close_df = _unwrap_panel(px, "close")
    if close_df is None or close_df.shape[0] < 22 or close_df.shape[1] == 0:
        if not getattr(g, "diag_no_data", False):
            log.info("[DIAG] get_price 空/不足 at %s shape=%s — 数据加载问题?" % (
                prev_date, getattr(close_df, "shape", None)))
            g.diag_no_data = True
        return
    data = {"close": close_df}
    for f in ["volume", "money"]:
        df = _unwrap_panel(px, f)
        if df is None:
            if not getattr(g, "diag_missing_field", False):
                log.info("[DIAG] 字段 %s 缺失 at %s — 检查 fields 返回结构" % (f, prev_date))
                g.diag_missing_field = True
            return
        data[f] = df

    valid_u = [s for s in universe
               if s in close_df.columns and int((~close_df[s].isna()).sum()) >= 30]
    if len(valid_u) < g.stock_num:
        return

    # ── Step 3: 逐袖因子计算 + 选股 (每袖 top30, HIGH=做多 → 取 signal 最大) ──
    sigs = {}
    for name in FACTOR_NAMES:
        try:
            arr, v = FACTOR_FUNC_MAP[name](valid_u, data)
        except Exception as _fe:
            if not getattr(g, "diag_factor_err_%s" % name, False):
                log.info("[DIAG] %s 因子计算异常: %r at %s" % (name, _fe, prev_date))
                setattr(g, "diag_factor_err_%s" % name, True)
            continue
        nv = int(np.sum(v))
        if nv < 10:
            if not getattr(g, "diag_low_valid_%s" % name, False):
                log.info("[DIAG] %s valid=%d (<10) at %s — 该袖本周不参与" % (name, nv, prev_date))
                setattr(g, "diag_low_valid_%s" % name, True)
            continue
        sigs[name] = (arr, v)

    if len(sigs) == 0:
        return

    # ── Step 4: 滞后 IC (上周各袖 signal → 本周收益 spearman) ──
    snap_closes = {}
    for s in close_df.columns:
        lv = close_df[s].iloc[-1]
        if np.isfinite(lv):
            snap_closes[s] = float(lv)

    if g.last_stocks is not None:
        for name in FACTOR_NAMES:
            lv_arr = g.last_values.get(name)
            if lv_arr is None:
                continue
            factor_vals = []
            rets = []
            for i, s in enumerate(g.last_stocks):
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

    # ── Step 4.5: 组合周收益记录 (对账, 用上一轮快照: 上轮权重×上周收盘 → 本周收盘) ──
    if g.last_stocks is not None and len(g.last_closes) > 0:
        _prev_weights = getattr(g, "last_weights", {})
        _prev_nav_parts = []
        _cur_nav_parts = []
        for s in _prev_weights:
            pc = g.last_closes.get(s)
            cc = snap_closes.get(s)
            if pc and cc and pc > 0 and cc > 0:
                _prev_nav_parts.append(_prev_weights[s])
                _cur_nav_parts.append(_prev_weights[s] * (cc / pc))
        if _prev_nav_parts and _sum_vals(_prev_nav_parts) > 0:
            _nav_prev = _sum_vals(_prev_nav_parts)
            _nav_cur = _sum_vals(_cur_nav_parts) + (1.0 - _nav_prev)  # 现金部分不涨
            g.weekly_rets.append(float(_nav_cur / _nav_prev - 1.0))

    # ── Step 5: 保存快照 (覆盖为本周, 供下周 IC/收益对账) ──
    g.last_stocks = list(valid_u)
    g.last_values = {name: sigs[name][0].copy() for name in sigs}
    g.last_closes = {s: snap_closes[s] for s in valid_u if s in snap_closes}

    # ── Step 6: 双袖 50/50 权重图 (重叠股权重叠加) ──
    sleeve_tops = {}
    for name in sigs:
        arr, v = sigs[name]
        # comp 升序取前 N = signal 最大; 掩码因子 0 平局 → kind=stable 按代码序, 可复现
        comp = np.where(np.isfinite(arr), -arr, np.inf)
        topn = min(g.stock_num, int(np.sum(v)))
        # P-20260901-009: JQ 环境 numpy 旧版不支持 kind="stable" (需 numpy>=1.15),
        # 稳定排序改用 mergesort (语义等价: 0 平局按原始顺序, 选股可复现)。
        top_idx = np.argsort(comp, kind="mergesort")[:topn]
        sleeve_tops[name] = [valid_u[i] for i in top_idx if comp[i] < np.inf]

    if len(sleeve_tops) < 1:
        return

    weight_map = {}
    n_active = len([n for n in sleeve_tops if len(sleeve_tops[n]) > 0])
    if n_active == 0:
        return
    # P-20260901-010: w_each 必须在统计完 n_active 之后统一计算。
    # 旧版在循环内递增 n_active 再算 w_each → 第一袖 1/30、第二袖 1/60,
    # 归一后 67/33 而非 50/50。正确: 每袖各 1/n_active 资金, 袖内等权。
    for name in sleeve_tops:
        top_list = sleeve_tops[name]
        if len(top_list) == 0:
            continue
        w_each = 1.0 / n_active / len(top_list)  # 每袖 50% 资金等权
        for s in top_list:
            weight_map[s] = weight_map.get(s, 0.0) + w_each
    # 归一 (若仅单袖有效则自动升为 100% 资金)
    # P-20260901-012: 用 _sum_vals() 手动累加, 禁止裸 sum()。
    # v4 曾误诊为"JQ 侧文件少了 sum() 五字符" — 已证伪: JQ 报错行号与本地逐行吻合,
    # 真因是 JQ 把 numpy 注入用户命名空间, 裸 sum() = np.sum, 而
    # np.sum(dict_values) 原样返回 dict_values (本机 numpy 2.4.6 实测复现)。
    # v3 的 'dict_values <= int' 崩溃与 v4 的 float() 报错为同一根因。
    w_sum = _sum_vals(weight_map.values())
    if w_sum <= 0:
        return
    target = {s: w / w_sum for s, w in weight_map.items()}
    g.last_weights = dict(target)   # 供下周组合周收益对账

    # ── Step 8: 执行 (lhb v2 修复版骨架 + P-20260901-008 差额保护) ──
    cd_order = get_current_data()

    # 平仓: 不在目标组合的
    for s in list(context.portfolio.positions.keys()):
        if s in target:
            continue
        try:
            if cd_order[s].paused:
                continue
            pos = context.portfolio.positions[s]
            if pos.total_amount < 100:
                continue
            order_target(s, 0)
        except Exception:
            continue

    total_value = context.portfolio.total_value
    ordered = 0
    for s in target:
        try:
            if context.portfolio.available_cash < total_value * target[s] * 0.5:
                break
            d = cd_order[s]
            if d.paused or d.is_st:
                continue
            if d.high_limit <= d.last_price:
                continue
            ref_px = d.day_open if (d.day_open and d.day_open > 0) else d.last_price
            if ref_px <= 0 or total_value * target[s] / ref_px < 100:
                continue
            # P-20260901-008: 差额微调保护 — 差额<100股跳过
            _pos = context.portfolio.positions.get(s)
            if _pos is not None and getattr(_pos, "total_amount", 0) > 0:
                _delta = int(total_value * target[s] / ref_px) - int(_pos.total_amount)
                if abs(_delta) < 100:
                    continue
            order_target_value(s, total_value * target[s])
            ordered += 1
        except Exception:
            continue

    # ── Step 9: 日志 & 定期报告 ──
    # P-20260901-010: 诊断日志改为每周期打印 (旧版仅前8周, 出了前8周就无法定位"无收益").
    # 与单因子文件每周 "ordered=…/stocks=…" 日志对齐, JQ 日志可全程追踪下单数。
    log.info("smallcap_duo | week=%d | 有效袖=%d/%d | 持仓数=%d | ordered=%d | 重叠=%d" % (
        g.trade_days, len(sleeve_tops), len(FACTOR_NAMES), len(target), ordered,
        len(set(sleeve_tops.get("ratio_extreme", [])) & set(sleeve_tops.get("concentration", [])))))
    # P-20260901-010: 连续零持仓告警 — 建仓期后仍空仓 = 执行层断链, 醒目暴露
    if len(context.portfolio.positions) == 0:
        g.zero_pos_weeks = getattr(g, "zero_pos_weeks", 0) + 1
        if g.zero_pos_weeks >= 3 and g.trade_days >= 8:
            log.info("[ALERT] 连续 %d 周零持仓 at week=%d (target=%d只) — 执行层断链? 检查上方 ordered/现金/涨停过滤" % (
                g.zero_pos_weeks, g.trade_days, len(target)))
    else:
        g.zero_pos_weeks = 0
    if g.trade_days - g.last_report_week >= 26:
        g.last_report_week = g.trade_days
        log.info("=" * 55)
        log.info("  smallcap双雄50/50 组合  每袖因子 IC/ICIR 半年报 (P-001)")
        log.info("=" * 55)
        _print_ic_report()
        log.info("-" * 55)
        _print_combo_report()
        log.info("=" * 55)
