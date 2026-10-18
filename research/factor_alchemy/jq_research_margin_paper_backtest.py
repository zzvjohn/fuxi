# -*- coding: utf-8 -*-
"""
JQ 研究环境手动纸质回测 — 两融/资金流三因子（回测引擎无此数据, 用研究环境真实 JQ 数据模拟周频调仓）
================================================================================
背景: 2026-08-14 jq_api_probe.py 实测: get_mtss / get_money_flow / get_money_flow_pro
      在【回测引擎】NameError（仅研究环境提供, 见 JQData 文档 id=9852）;
      回测引擎的 get_price/attribute_history 不含两融/资金流 → 无替代 API。
      本文件在【研究环境 notebook】运行, 用真实 JQ 数据手动复刻回测引擎的周频调仓:

  · 数据源   get_mtss(个股两融日频, fin_value≈rzye, fin_sec_value-fin_value≈rqye)
            get_money_flow(主力资金流, net_amount_main≈net_mf_amount)
            get_price(行情 close/open) / get_trade_days / get_index_stocks
  · 股票池   回测起点中证500成分 (get_index_stocks(date=START), 降低生存者偏差;
            注: 中途退市股价格缺失→持仓自然剔除, 属乐观偏差, 已知)
  · 调仓     每 5 个交易日; 决策只用 t-1 及以前数据 (shift(1) 防未来);
            开盘价成交 (镜像回测引擎 run_weekly(time="open"))
  · 成本     双边滑点 0.3%×2 + 佣金 0.01%×2 + 印花税 0.1% = 0.72% × 换手率
            (与回测引擎 OrderCost(0, 0.001, 0.0001, 0.0001)+FixedSlippage(0.003) 对齐)
  · 因子     margin_short_ratio / margin_balance_chg_21d / margin_mf_divergence
            HIGH=做多: 取因子值最高 TOPN 只 (与回测引擎 rank_pct 选股方向一致)
  · P-001    每因子输出周频滞后 Rank IC / ICIR / IC>0 占比

用法: 整体粘贴到 JQ 研究环境 notebook 的一个 cell 运行; 预计 2~6 分钟。
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# ═══ 环境兼容: 数据 API 导入 (2026-08-14 修复 NameError) ═══
# 新版网页研究环境: API 在 jqdata 模块, 必须显式导入 (老版自动注入);
# 本地 Jupyter + JQData SDK: 用 jqdatasdk 模块。
_loaded_api = False
try:
    from jqdata import *              # 聚宽研究环境 (get_trade_days/get_mtss/get_money_flow...)
    _loaded_api = True
except Exception:
    pass
if not _loaded_api:
    try:
        from jqdatasdk import *       # 本地 JQData SDK
        _loaded_api = True
        import jqdatasdk as _sdk
        try:
            if not _sdk.is_auth():
                print("⚠️ 本地 JQData 未登录: 请先执行 jqdatasdk.auth('手机号','密码') 再重跑")
        except Exception:
            pass
    except Exception:
        pass                          # 老版研究环境: 函数已自动注入

# ══════════════════ 参数 ══════════════════
START = "2018-01-01"          # 回测开始
END   = "2026-08-13"          # 回测结束（数据最后日期）
BENCH = "000905.XSHG"         # 基准: 中证500（与回测引擎一致）
TOPN  = 50                    # 持仓数 ≈ 池的10%
COST  = 0.0072                # 单次换手双边成本合计（滑点0.3%×2+佣金0.02%+印花税0.1%）
RUN   = ["margin_short_ratio", "margin_balance_chg_21d", "margin_mf_divergence"]
CHUNK = 50                    # 数据 API 分块大小（股/次）

# ══════════════════ 1. 交易日历 & 股票池 ══════════════════
try:
    trade_days = pd.to_datetime(get_trade_days(start_date=START, end_date=END))
except NameError:
    _all = pd.to_datetime(list(get_all_trade_days()))     # 兜底: 全量交易日截取区间
    trade_days = _all[(_all >= pd.Timestamp(START)) & (_all <= pd.Timestamp(END))]
pool = [s for s in get_index_stocks(BENCH, date=START) if s.startswith(("0", "3", "6"))]
print("[1/5] trade_days=%d pool=%d" % (len(trade_days), len(pool)))

# ══════════════════ 2. 行情（全池 close/open） ══════════════════
def _load_px_panel(fields):
    """加载全池行情并转成宽面板 (index=交易日, columns=股票代码)。

    ⚠️ 2026-08-15 踩坑: JQ 研究环境 get_price(多股票 list, panel=False)
    返回的是【长表】 (columns = time/code/field, 行数 = 交易日×股票数),
    不是面板! 长表直接 .reindex(trade_days) 会得到全 NaN 的 (2090, 3) 空骨架
    (3 = time/code/field 三列), 导致 0 有效调仓。必须先 pivot 成面板。
    """
    df = get_price(pool, start_date=START, end_date=END, frequency="daily",
                   fields=list(fields), skip_paused=False, fq="pre", panel=False)
    tcol = "time" if "time" in df.columns else "date"
    ccol = next((c for c in ("code", "sec_code") if c in df.columns), None)
    out = {}
    if ccol is not None:
        for f in fields:
            try:
                px = df.pivot(index=tcol, columns=ccol, values=f)
            except ValueError:                       # 重复 (time, code) 兜底
                px = df.pivot_table(index=tcol, columns=ccol, values=f, aggfunc="first")
            px.index = pd.to_datetime(px.index)
            out[f] = px.reindex(trade_days).ffill()
    else:                                            # 兼容: 已是宽表的老版本
        df = df.copy()
        df.index = pd.to_datetime(df.index)
        df = df.reindex(trade_days).ffill()
        for f in fields:
            out[f] = df[f]
    return out

PX = _load_px_panel(["open", "close"])
O = PX["open"]              # 开盘价面板 (成交价)
C = PX["close"]             # 收盘价面板 (IC 计算用)

B = get_price(BENCH, start_date=START, end_date=END, frequency="daily",
              fields=["close", "open"], skip_paused=False, fq="pre", panel=False)
B = B.reindex(trade_days).ffill()
print("[2/5] 行情加载完成 C=%s O=%s" % (C.shape, O.shape))

# ST 过滤（get_extras 可用则启用）
st_shift = None
try:
    st = get_extras("is_st", security_list=pool, start_date=START, end_date=END)
    if st is not None and len(st):
        st_shift = st.reindex(trade_days).ffill().fillna(False).shift(1)
        print("    is_st 过滤启用 (%d 列)" % st.shape[1])
    else:
        print("    get_extras 返回空, 跳过 ST 过滤")
except Exception as e:
    print("    get_extras 不可用 (%r), 跳过 ST 过滤" % (e,))

# ══════════════════ 3. 两融数据（get_mtss, 日频） ══════════════════
def load_chunked(loader, stocks, fields):
    frames = []
    for k in range(0, len(stocks), CHUNK):
        try:
            df = loader(stocks[k:k+CHUNK], start_date=START, end_date=END, fields=fields)
        except Exception as e:
            print("    [DATA] chunk %d FAIL: %r" % (k, e))
            continue
        if df is not None and len(df):
            frames.append(df)
    if not frames:
        raise RuntimeError("数据加载全部失败, 检查研究环境账号权限")
    d = pd.concat(frames, ignore_index=True)
    d["date"] = pd.to_datetime(d["date"])
    return d

print("[3/5] 加载 get_mtss (个股两融, 日频) ...")
mtss = load_chunked(get_mtss, pool, ["date", "sec_code", "fin_value", "fin_sec_value"])
fin_panel = mtss.pivot_table(index="date", columns="sec_code",
                             values="fin_value").reindex(trade_days).ffill()
sec_bal = (mtss.pivot_table(index="date", columns="sec_code",
                            values="fin_sec_value").reindex(trade_days).ffill()
           - fin_panel).clip(lower=0.0)          # 融券余额(元)代理 ≈ rqye
print("[3/5] mtss rows=%d 覆盖=%d | fin_panel=%s" % (len(mtss), mtss["sec_code"].nunique(), fin_panel.shape))

mf_panel = None
if "margin_mf_divergence" in RUN:
    print("[3b/5] 加载 get_money_flow (主力资金流) ...")
    mf = load_chunked(get_money_flow, pool, ["date", "sec_code", "net_amount_main"])
    mf_panel = mf.pivot_table(index="date", columns="sec_code",
                              values="net_amount_main").reindex(trade_days).ffill()
    print("[3b/5] money_flow rows=%d 覆盖=%d" % (len(mf), mf["sec_code"].nunique()))

# ══════════════════ 4. 因子日面板 ══════════════════
factor_panels = {}
if "margin_short_ratio" in RUN:
    # rzye/(rqye+1e-6) 的 JQ 版: fin_value / max(fin_sec_value-fin_value, 0)
    factor_panels["margin_short_ratio"] = fin_panel / (sec_bal + 1e-6)
if "margin_balance_chg_21d" in RUN:
    # -rzye.diff(21): JQ 日频直接 diff(21) ≈ 一个月
    chg = fin_panel - fin_panel.shift(21)
    factor_panels["margin_balance_chg_21d"] = (-chg).where(fin_panel.rolling(22).count() >= 22)
if "margin_mf_divergence" in RUN:
    # -rolling(60).corr(net_mf_amount, rzye): 向量化滚动相关
    x = mf_panel; y = fin_panel
    mx = x.rolling(60).mean(); my = y.rolling(60).mean()
    cov = ((x - mx) * (y - my)).rolling(60).mean()
    corr = cov / (x.rolling(60).std() * y.rolling(60).std())
    factor_panels["margin_mf_divergence"] = -corr
print("[4/5] 因子日面板完成:", {k: v.shape for k, v in factor_panels.items()})

# ══════════════════ 5. 周频纸质回测 ══════════════════
reb_days = trade_days[5::5]          # 镜像引擎 g.trade_days<5 warmup
print("[5/5] 周频回测: %d 次调仓, 持仓 TOP%d" % (len(reb_days) - 1, TOPN))

bench_rets = []
for i in range(len(reb_days) - 1):
    bench_rets.append(B["open"].loc[reb_days[i+1]] / B["open"].loc[reb_days[i]] - 1.0)
bench_eq = np.cumprod(1.0 + np.array(bench_rets))


def report(name, rets, ic_list, tos, n_active, n_ic_nan=0):
    rets = np.array(rets)
    eq = np.cumprod(1.0 + rets)
    total = eq[-1] - 1.0
    n = len(rets)
    ann = (1.0 + total) ** (52.14 / n) - 1.0
    sd = rets.std()
    sharpe = float(rets.mean() / sd * np.sqrt(52.14)) if sd > 0 else 0.0
    mdd = float((eq / np.maximum.accumulate(eq) - 1.0).min())
    ics = np.array([x for x in ic_list if not (isinstance(x, float) and np.isnan(x))])
    if len(ics):
        ic_mean = ics.mean(); ic_std = ics.std()
        icir = ic_mean / ic_std if ic_std > 0 else 0.0
    else:
        ic_mean = ic_std = icir = float("nan")
    active_tos = np.array(tos) if len(tos) else np.array([1.0])
    print("=" * 60)
    print("  %s  纸质回测报告" % name)
    print("=" * 60)
    print("  有效调仓: %d/%d 周 (%.0f%%)  ← 覆盖率低说明数据稀疏, 结果不可信"
          % (n_active, n, n_active / n * 100.0))
    print("  累计收益: %+.2f%%   (基准中证500: %+.2f%%)" % (total * 100, (bench_eq[-1] - 1) * 100))
    print("  年化: %+.2f%%   周胜率: %.1f%%" % (ann * 100, (rets > 0).mean() * 100))
    print("  Sharpe: %.2f    MaxDD: %.2f%%    平均换手: %.0f%%" % (sharpe, mdd * 100, active_tos.mean() * 100))
    print("  --- P-001 单因子 IC ---")
    print("  Rank IC mean: %+.4f   std: %.4f   ICIR: %+.4f   IC>0占比: %.1f%%   有效样本: %d"
          % (ic_mean, ic_std, icir, (ics > 0).mean() * 100, len(ics)))
    if n_ic_nan:
        print("  ⚠ IC nan 样本: %d 个 (spearmanr 常数输入/唯一值过少导致, 已从统计中剔除)"
              % n_ic_nan)
    print()
    return dict(name=name, total=total, ann=ann, sharpe=sharpe, mdd=mdd, icir=icir,
                n=n, n_active=n_active, n_ic_nan=n_ic_nan)


summary = []
for fname, fpanel in factor_panels.items():
    f_used = fpanel.shift(1)          # 防未来: 决策日 t 只用 t-1 及以前数据
    held, port_rets, ic_list, tos, n_active, n_ic_nan = [], [], [], [], 0, 0
    for i in range(len(reb_days) - 1):
        t = reb_days[i]; t_next = reb_days[i+1]
        fvals = f_used.loc[t].dropna()
        if st_shift is not None:
            try:
                fvals = fvals[~st_shift.loc[t].reindex(fvals.index).fillna(False).astype(bool)]
            except Exception:
                pass
        if len(fvals) == 0:
            port_rets.append(0.0); continue
        sel = fvals.sort_values(ascending=False).index[:TOPN]
        p0 = O.loc[t].reindex(sel).dropna()
        p1 = O.loc[t_next].reindex(sel).dropna()
        common = p0.index.intersection(p1.index)
        if len(common) == 0:
            port_rets.append(0.0); continue
        wk = p1[common] / p0[common] - 1.0
        turnover = 1.0 - len(set(held) & set(common)) / float(len(common))
        port_rets.append(float(wk.mean()) - COST * turnover)
        tos.append(turnover)
        n_active += 1
        held = list(common)
        # 滞后 IC: t 时因子 vs t→t_next 全池 close-to-close 收益
        fr = C.loc[t_next] / C.loc[t] - 1.0
        both = fvals.index.intersection(fr.dropna().index)
        if len(both) >= 30:
            ic, _ = spearmanr(fvals[both].values, fr[both].values)
            if isinstance(ic, float) and np.isnan(ic):
                n_ic_nan += 1            # 常数/趋同因子输入 → nan, 不纳入统计
            else:
                ic_list.append(float(ic))
        if i % 100 == 0:
            print("    ... %s 调仓 %d/%d" % (fname, i, len(reb_days) - 1))
    summary.append(report(fname, port_rets, ic_list, tos, n_active, n_ic_nan))

print("=" * 60)
print("  三因子横向对比")
print("=" * 60)
for r in sorted(summary, key=lambda x: -x["sharpe"]):
    print("  %-26s 有效%3d周  累计%+8.2f%%  年化%+7.2f%%  Sharpe %.2f  MaxDD %.2f%%  ICIR %+.3f"
          % (r["name"], r["n_active"], r["total"] * 100, r["ann"] * 100,
             r["sharpe"], r["mdd"] * 100, r["icir"]))
print("  基准中证500: 累计 %+.2f%%" % ((bench_eq[-1] - 1) * 100))
