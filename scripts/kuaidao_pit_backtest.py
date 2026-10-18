# -*- coding: utf-8 -*-
"""快刀选股 可执行策略 —— 独立 PIT 回测 + 前视审计电池 (2026-09-11)

目标
----
不依赖 `kuaidao_timing_backtest_v4` 引擎, 从 `pool_ohlc` 原始**后复权** OHLC 独立重实现
"季频池刷新 + 15 只等权买入持有 + 离池卖出 / 暴露 k" 的完整回测,
并**证明无未来数据** (前视审计电池 LA-1 ~ LA-6)。

设计口径 (来自 output/kuaidao_strategy_design_20260911.md)
--------------------------------------------------------
① 选池   数据截止 = run_date (4-30 / 8-31 / 10-31 披露截止日) 收盘
② 建仓   eff = 严格晚于 run_date 的第一个交易日, MOC 收盘成交
③ 持有   期内不动作 (无止损/止盈/择时)
④ 换出   生效日前一交易日收盘卖出 (等价: 上一期净值走到 eff-1)
⑤ 仓位   15 只等权; 唯一风控旋钮 = 暴露 k (k=1 全暴露; 0<k<1 余下持现金零息)
⑥ 收益   建仓日**仅当日新进者**不计收益(跨期保留者按其原始建仓日); 停牌计入分母(稀释)

前视审计电池
------------
LA-1 时间序审计      : run_date < eff, 给出每期决策->执行的时间账本
LA-2 信息集边界审计  : 扫描选池源码, 所有数据 join 是否都被 run_date 截断
LA-3 泄漏价值量化    : (a) 正向对照——注入前视后 NAV 是否"看得见"地爆掉(探测器灵敏度)
                       (b) 同收盘前视 (决策日 close 成交, 经典同 bar 泄漏)
                       (c) 不可交易缺口 (决策收盘->执行价, 我们买不到)
LA-4 执行滞后敏感性  : 建仓整体推迟 1/2/3/5/10 日
LA-5 宇宙审计        : list_date<=run_date / 退市股缺失(幸存者) / 停牌稀释
LA-6 收益口径自检    : 日期范围严格 ∈[eff_k, eff_{k+1}) / 首期首日=0 / 无越界价格

输出
----
data/kuaidao_test/full_backtest/pit/{nav_pit.csv, stats_pit.json, audit_pit.json}
"""
import sys, os, json, re
from collections import OrderedDict
import numpy as np
import pandas as pd

ROOT = 'e:/quant'
os.chdir(ROOT)

FM = 'data/kuaidao_test/full_market'
OUT = 'data/kuaidao_test/full_backtest'
OUTP = f'{OUT}/pit'
os.makedirs(OUTP, exist_ok=True)

BT_START_CUT = '20240430'     # 与 v4 一致: 日历取 (20240430, 20260910]
END_D = '20260910'
RF = 0.02
COST_BP = 30.0               # 往返成本 (bp) —— 设计文档 T2 口径

PERIOD_META = [('20240430', '20231231', 'P2023'), ('20240831', '20240630', 'P2024H1'),
               ('20241031', '20240930', 'P2024Q3'), ('20250430', '20241231', 'P2024'),
               ('20250831', '20250630', 'P2025H1'), ('20251031', '20250930', 'P2025Q3'),
               ('20260430', '20251231', 'P2025'), ('20260831', '20260630', 'P2026H1')]

QUARTERLY = [('20240430', 'P2023'), ('20240831', 'P2024H1'), ('20241031', 'P2024Q3'),
             ('20250430', 'P2024'), ('20250831', 'P2025H1'), ('20251031', 'P2025Q3'),
             ('20260430', 'P2025'), ('20260831', 'P2026H1')]

REF = dict(total=0.3258022699389107, sharpe=0.6011053442098477, mdd=-0.25984994227161007)


# =====================================================================================
# 0. 数据
# =====================================================================================
def load_calendar():
    b = pd.read_csv('data/raw/index_000300_daily.csv', dtype=str)
    dates = sorted(b['trade_date'].str.replace('-', '').tolist())
    return [d for d in dates if BT_START_CUT < d <= END_D]


def load_pool(tag):
    df = pd.read_csv(f'{OUT}/pool_{tag}.csv', dtype={'code': str})
    return [c for c in df['code'].dropna().tolist() if c]


CAL = load_calendar()
CIDX = {d: i for i, d in enumerate(CAL)}
NCAL = len(CAL)


def build_matrices():
    """逐只读取**该股自己的**序列算收益 (与 v4 `prev_close=c.shift(1)` 同义), 再对齐到 CAL.

    R[i,j] = close[t]/close[t-1] - 1  (t = cal[i]);  NaN = 不可得
    A[i,j] = 1 若 close(t) 存在
    """
    codes = sorted({c for _, t in QUARTERLY for c in load_pool(t)})
    R = pd.DataFrame(np.nan, index=CAL, columns=codes, dtype=float)
    A = pd.DataFrame(0.0, index=CAL, columns=codes, dtype=float)
    O = pd.DataFrame(np.nan, index=CAL, columns=codes, dtype=float)
    for c in codes:
        f = f'{FM}/pool_ohlc/{c}.csv'
        if not os.path.exists(f):
            continue
        d = pd.read_csv(f, dtype={'trade_date': str}).sort_values('trade_date').drop_duplicates('trade_date')
        d = d.set_index('trade_date')
        idx = [x for x in d.index if x in CIDX]
        cl = d.loc[idx, 'close'].astype(float)
        r = cl / cl.shift(1) - 1.0
        A.loc[idx, c] = 1.0
        R.loc[idx, c] = r.to_numpy(float)
        O.loc[idx, c] = d.loc[idx, 'open'].astype(float).to_numpy(float)
    return R, A, O, codes


# =====================================================================================
# 1. 独立模拟器
# =====================================================================================
def eff_of(run_date):
    later = [d for d in CAL if d > run_date]
    return min(later) if later else None


def _period_slice(pi):
    eff0 = eff_of(QUARTERLY[pi][0])
    i0 = CIDX[eff0]
    i1 = CIDX[eff_of(QUARTERLY[pi + 1][0])] if pi + 1 < len(QUARTERLY) else NCAL
    return i0, i1


def run_basket(Rm, Am, i0, i1, cols, w, zero_first=None, k=1.0):
    """期内等权 (或给定权重) 逐日累积; A 掩码决定分母. 返回 rets 数组与日净值."""
    subR = Rm[i0:i1, cols]
    subA = Am[i0:i1, cols]
    n = len(cols)
    prev_avail = np.zeros(n)
    rets = np.empty(i1 - i0)
    for t in range(i1 - i0):
        prev_avail = np.maximum(prev_avail, subA[t])
        den = float(w @ prev_avail)
        rj = np.nan_to_num(subR[t], nan=0.0)
        if t == 0 and zero_first is not None:
            rj = rj * (1.0 - zero_first)
        r = float((rj * subA[t] * w).sum()) / den if den > 1e-12 else 0.0
        rets[t] = r
    return rets


def simulate(R, A, k=1.0, mode='moc_newcomer', cost_bp=0.0, lag=0):
    """季频等权 BH.
    mode='moc_newcomer' : 建仓日只对**当日新进者**归零 (与 v4 口径一致 => 应复现 +32.58%)
    mode='moc_all'      : 建仓日对全部持仓归零 (全清仓再买回的严格 MOC)
    """
    Rm, Am = R.to_numpy(float), A.to_numpy(float)
    cidx = {c: i for i, c in enumerate(R.columns)}
    nav = 1.0
    rows, periods, ledger = [], [], []
    prev_members, prev_entry = [], {}
    for pi, (rd, tag) in enumerate(QUARTERLY):
        i0, i1 = _period_slice(pi)
        i0 = min(i0 + lag, NCAL - 1)
        i1 = min(i1 + lag, NCAL)
        if i1 <= i0:
            continue
        eff = CAL[i0]
        sel = [c for c in load_pool(tag) if c in cidx]
        cols = [cidx[c] for c in sel]
        n = len(cols)
        w = np.full(n, 1.0 / n)
        prev_set = set(prev_members)
        entry = {c: (prev_entry[c] if c in prev_entry else eff) for c in sel}
        zero = np.array([1.0 if (mode == 'moc_all' or entry[c] == eff) else 0.0 for c in sel])
        if cost_bp > 0 and pi > 0:
            n_new = sum(1 for c in sel if c not in prev_set)
            nav *= (1.0 - (n_new / n) * (cost_bp / 1e4))
        rets = run_basket(Rm, Am, i0, i1, cols, w, zero_first=zero)
        navs = nav * np.cumprod(1.0 + k * rets)
        nav = float(navs[-1])
        for t, v in zip(CAL[i0:i1], navs):
            rows.append(dict(date=t, nav=float(v), period=tag))
        periods.append(dict(eff=eff, tag=tag, i0=i0, i1=i1, n=n, sel=sel, nav_end=float(nav),
                            n_new=(0 if pi == 0 else sum(1 for c in sel if c not in prev_set))))
        ledger.append(dict(tag=tag, run_date=rd, eff=eff, n=n, first_ret=float(k * rets[0])))
        prev_members, prev_entry = list(sel), dict(entry)
    return pd.Series({r['date']: r['nav'] for r in rows}).sort_index(), periods, ledger


def perf(nav_series, rf=RF):
    r = nav_series.pct_change().dropna()
    total = float(nav_series.iloc[-1] / nav_series.iloc[0] - 1)
    cagr = float((nav_series.iloc[-1] / nav_series.iloc[0]) ** (250 / max(len(nav_series), 1)) - 1)
    sharpe = float((r.mean() - rf / 250) / r.std() * np.sqrt(250)) if r.std() > 0 else float('nan')
    dd = nav_series / nav_series.cummax() - 1
    return dict(total=total, cagr=cagr, sharpe=sharpe, mdd=float(dd.min()),
                vol=float(r.std() * np.sqrt(250)))


def period_rets(periods):
    out, prev = [], 1.0
    for p in periods:
        out.append(dict(tag=p['tag'], eff=p['eff'], ret=p['nav_end'] / prev - 1.0))
        prev = p['nav_end']
    return out


# =====================================================================================
# 2. 前视审计电池
# =====================================================================================
def la1_time_order(ledger):
    bad, rows = [], []
    for lg in ledger:
        rd, eff = lg['run_date'], lg['eff']
        gap_cal = (pd.Timestamp(eff) - pd.Timestamp(rd)).days
        prior_idx = [i for i, d in enumerate(CAL) if d <= rd]
        n_td = CIDX[eff] - (prior_idx[-1] if prior_idx else -1)
        rows.append(dict(tag=lg['tag'], run_date=rd, eff=eff, gap_cal_days=gap_cal,
                         trading_days_between=n_td - 1))
        if not (rd < eff):
            bad.append(lg['tag'])
    return dict(pass_=(len(bad) == 0), violations=bad, ledger=rows)


def la2_information_set():
    """扫描选池源码, 所有数据 join 是否都被 run_date 截断."""
    files = ['scripts/kuaidao_full_backtest_v1.py', 'scripts/kuaidao_quarterly_scores.py']
    src = {f: open(f, encoding='utf-8').read() for f in files}
    both = '\n'.join(src.values())
    # 正常守卫 (<= run_date / <= report_end)
    guards = {
        'ann_date<=cutoff': re.findall(r"ann_date'\]\.astype\(str\)\s*<=\s*(?:rd|P\['run_date'\]|re_)", both),
        'trade_date<=cutoff': re.findall(r"trade_date'\]\.astype\(str\)\s*<=\s*(?:rd|P\['run_date'\]|re_)", both),
        'start_date<=run_date': re.findall(r"start_date'\]\.astype\(str\)\s*<=\s*P\['run_date'\]", both),
        'end_date<=report_end': re.findall(r"end_date'\]\.astype\(str\)\s*<=\s*(?:re_|P\['report_end'\])", both),
        'dedup ts_code+trade_date': re.findall(r"drop_duplicates\(subset=\['ts_code',\s*'trade_date'\]\)", both),
    }
    # 危险谓词 (> cutoff = 未来)
    danger = {
        'ann_date>cutoff': re.findall(r"ann_date'\]\.astype\(str\)\s*>\s*(?:rd|P\['run_date'\])", both),
        'trade_date>cutoff': re.findall(r"trade_date'\]\.astype\(str\)\s*>\s*(?:rd|P\['run_date'\])", both),
    }
    # 未被任何 cutoff 约束的裸取数 (人工确认项)
    naked = re.findall(r"mt\['(db|nm|div|trades|rep|top10|holder)'\]\s*$", both, re.M)
    cnt = {k: len(v) for k, v in guards.items()}
    return dict(files=files, guard_counts=cnt, danger_predicates={k: len(v) for k, v in danger.items()},
                naked_tables=['mt[' + x + ']' for x in set(naked)],
                pass_=(all(c > 0 for c in cnt.values()) and not any(len(v) for v in danger.values())),
                note='全部 ann_date/trade_date/start_date/end_date join 均以披露截止日为上界; 无 > cutoff 谓词; db 已按 ts_code+trade_date 去重')


def la3_leak_value(R, A, base_perf):
    """(a) 正向对照 (b) 同 bar 前视能白捡多少."""
    Rm, Am = R.to_numpy(float), A.to_numpy(float)
    cidx = {c: i for i, c in enumerate(R.columns)}
    res = {}

    # ---- (b) 同 bar 前视 = 决策日收盘成交: 白捡的就是"执行日当天"的收益 ----
    #      合法口径必须等到 eff 当日 MOC 成交, 因此 eff 日 (从决策收盘到执行价) 这段收益拿不到
    gaps = []
    for rd, tag in QUARTERLY:
        eff = eff_of(rd)
        if eff is None:
            continue
        prior = [d for d in CAL if d <= rd]
        if not prior:
            continue                     # 首期决策日早于回测起点 (20240430 < CAL[0])
        d0, i1 = prior[-1], CIDX[eff]
        sel = [c for c in load_pool(tag) if c in cidx]
        cols = [cidx[c] for c in sel]
        seg = Rm[CIDX[d0] + 1:i1 + 1, cols]      # 严格 (d0 收盘 -> eff 收盘]
        ok = np.isfinite(seg).all(axis=0)
        g = float(np.prod(1.0 + seg[:, ok], axis=0).mean() - 1.0) if ok.any() else np.nan
        gaps.append(dict(tag=tag, from_=d0, to=eff, n_days=int(seg.shape[0]),
                         forfeited_ret=g, n=int(ok.sum())))
    res['same_bar_forfeit'] = dict(
        note='合法必须 eff 当日 MOC 成交 => "决策收盘->执行价"这段收益**拿不到**; 这里量化的就是它',
        per_period=gaps,
        cum=float(np.prod([1.0 + g['forfeited_ret'] for g in gaps if np.isfinite(g['forfeited_ret'])]) - 1.0),
        mean=float(np.nanmean([g['forfeited_ret'] for g in gaps])))

    # ---- (a) 正向对照: 每期换成"期内实现收益最好"的 15 只 (完全前视) ----
    nav_or = 1.0
    picks = []
    for pi, (rd, tag) in enumerate(QUARTERLY):
        i0, i1 = _period_slice(pi)
        seg = Rm[i0:i1]
        n_ok = np.isfinite(seg).sum(axis=0)
        cum = np.prod(1.0 + np.nan_to_num(seg), axis=0)
        cum = np.where(n_ok > max(5, (i1 - i0) * 0.5), cum, -np.inf)
        top = np.argsort(-cum)[:15]
        keep = [int(k) for k in top if np.isfinite(cum[k])] or [int(k) for k in top]
        rets = run_basket(Rm, Am, i0, i1, keep, np.full(len(keep), 1.0 / len(keep)))
        nav_or *= float(np.prod(1.0 + rets))
        picks.append(dict(tag=tag, codes=[R.columns[k] for k in keep],
                          seg_ret=float(np.mean([cum[k] - 1 for k in keep]))))
    res['positive_control_oracle'] = dict(
        note='每期选"期内实现收益最高"的15只 (完全前视) — 探测器灵敏度对照',
        total=float(nav_or - 1.0), vs_baseline_pp=float((nav_or - 1.0 - base_perf['total']) * 100),
        picks=picks)
    return res


def la7_evidence_years(periods_meta):
    """LA-7: **数据级**前视审计 —— 扫描打分明细的证据链 (scores_*.csv 的 ev/scores 文本),
    提取所有财年, 断言没有任何财年晚于该期 report_end (披露截止).

    这是不依赖选池代码、直接审"喂进模型的数据"的独立检查。
    """
    YRE = re.compile(r'(?<![0-9.])(20(?:1[0-9]|2[0-9]))(?![0-9.])')
    rows, viol = [], []
    for rd, re_, tag in periods_meta:
        f = f'{OUT}/scores_{tag}.csv'
        if not os.path.exists(f):
            continue
        sc = pd.read_csv(f, dtype={'code': str})
        reY = int(str(re_)[:4])
        n_bad, mx = 0, None
        for _, r in sc.iterrows():
            s = str(r.get('ev', ''))
            ys = {int(y) for y in YRE.findall(s)}
            if not ys:
                continue
            mx = max(mx or 0, max(ys))
            bad = [y for y in ys if y > reY]
            if bad:
                n_bad += 1
                viol.append(dict(tag=tag, code=r['code'], years=sorted(bad), ev=s[:160]))
        rows.append(dict(tag=tag, run_date=rd, report_end=str(re_), n_candidates=int(len(sc)),
                         max_year_seen=mx, n_rows_with_future_year=n_bad))
    return dict(pass_=(len(viol) == 0), violations=viol, by_period=rows,
                note='证据链中出现的财年一律 <= 该期 report_end; 越界即判前视')


def la5_universe(R, A, codes, periods):
    sb = pd.read_csv(f'{FM}/full_stock_basic.csv', dtype={'ts_code': str, 'list_date': str})
    ld = sb.drop_duplicates('ts_code').set_index('ts_code')['list_date'].to_dict()
    viol = []
    for rd, tag in QUARTERLY:
        for c in load_pool(tag):
            l = str(ld.get(c, ''))
            if l and l > rd:
                viol.append(dict(tag=tag, code=c, list_date=l, run_date=rd))
    cols = list(pd.read_csv(f'{FM}/full_stock_basic.csv', nrows=1).columns)
    susp = []
    for p in periods:
        sel = [c for c in p['sel'] if c in R.columns]
        seg = A.loc[CAL[p['i0']:p['i1']], sel].to_numpy(float)
        miss = (seg < 0.5).sum(axis=0)
        for c, m in zip(sel, miss):
            if m > 0:
                susp.append(dict(tag=p['tag'], code=c, missing_days=int(m), n_days=int(len(seg))))
    maxrun = 0
    for c in codes:
        v = A[c].to_numpy(float) > 0.5
        run = mx = 0
        for x in v:
            run = 0 if x else run + 1
            mx = max(mx, run)
        maxrun = max(maxrun, mx)
    return dict(list_date_violations=viol, pass_list_date=(len(viol) == 0),
                stock_basic_cols=cols,
                delisted_coverage=dict(
                    note='full_stock_basic.csv 为**当前**快照, 无 list_status 字段 => 退市股不可见, 存在幸存者偏差(方向: 高估)',
                    has_list_status=('list_status' in cols)),
                suspended_in_holding=susp, n_suspended_cases=len(susp),
                max_consecutive_no_price_days=int(maxrun))


def la6_returns_check(periods, ledger, R):
    covered = sorted(set(d for p in periods for d in CAL[p['i0']:p['i1']]))
    ok_range = (min(covered) == CAL[0]) and (max(covered) == CAL[-1])
    no_overlap = all(periods[i]['i1'] == periods[i + 1]['i0'] for i in range(len(periods) - 1))
    first_zero = abs(ledger[0]['first_ret']) < 1e-12
    last_px = max(R.index)
    return dict(coverage_start=min(covered), coverage_end=max(covered),
                cal_start=CAL[0], cal_end=CAL[-1], pass_coverage=bool(ok_range),
                pass_no_gap_no_overlap=bool(no_overlap), pass_first_period_day0_zero=bool(first_zero),
                pass_no_future_price=bool(last_px <= CAL[-1]),
                periods_covered=len(periods), n_cal_days=NCAL)


# =====================================================================================
# 3. 主流程
# =====================================================================================
def main():
    print('=' * 78)
    print('[load] calendar + pool_ohlc (后复权, 逐只自算收益) ...', flush=True)
    R, A, O, codes = build_matrices()
    print(f'  cal={NCAL} ({CAL[0]}..{CAL[-1]})  pool-union codes={len(codes)}')

    base, periods, ledger = simulate(R, A, k=1.0, mode='moc_newcomer')
    P = perf(base)
    d_tot = (P['total'] - REF['total']) * 100
    print(f"\n[独立复现] B_quarter_bh  total={P['total']:+.2%}  sharpe={P['sharpe']:.4f}  mdd={P['mdd']:.2%}")
    print(f"[v4 参考 ]                total={REF['total']:+.2%}  sharpe={REF['sharpe']:.4f}  mdd={REF['mdd']:.2%}")
    print(f"[差异    ]                Δtotal={d_tot:+.4f}pp  Δsharpe={P['sharpe']-REF['sharpe']:+.6f}  "
          f"Δmdd={(P['mdd']-REF['mdd'])*100:+.4f}pp   {'✅ 精确复现' if abs(d_tot) < 0.05 else '⚠ 偏差待查'}")

    print('\n[变体] 建仓日归零口径 / 暴露 k / 成本')
    var = {}
    for nm, kw in [('B_moc_newcomer(采纳)', dict(mode='moc_newcomer')),
                   ('B_moc_all(全清仓)', dict(mode='moc_all'))]:
        s, _, _ = simulate(R, A, **kw)
        var[nm] = perf(s)
        print(f'  {nm:22s} total={var[nm]["total"]:+8.2%}  sharpe={var[nm]["sharpe"]:.3f}  mdd={var[nm]["mdd"]:7.2%}')
    ktab = {}
    for k in [0.4, 0.5, 0.6, 0.61, 0.65, 0.7, 0.8, 0.9, 1.0]:
        s, _, _ = simulate(R, A, k=k, mode='moc_newcomer')
        ktab[str(k)] = perf(s)
        print(f'  k={k:<5} total={ktab[str(k)]["total"]:+8.2%}  sharpe={ktab[str(k)]["sharpe"]:.3f}  '
              f'mdd={ktab[str(k)]["mdd"]:7.2%}')
    sc, _, _ = simulate(R, A, k=1.0, mode='moc_newcomer', cost_bp=COST_BP)
    cost_perf = perf(sc)
    print(f' 成本 {COST_BP:.0f}bp            total={cost_perf["total"]:+8.2%}  Δ={(cost_perf["total"]-P["total"])*100:+.2f}pp')

    b = pd.read_csv('data/raw/index_000300_daily.csv', dtype=str)
    b['d'] = b['trade_date'].str.replace('-', '')
    b = b[(b['d'] > BT_START_CUT) & (b['d'] <= END_D)].set_index('d')
    bnav = b['close'].astype(float)
    bnav = bnav / bnav.iloc[0]
    bench = perf(bnav)
    print(f'\n[基准] 沪深300  total={bench["total"]:+.2%}  sharpe={bench["sharpe"]:.3f}  mdd={bench["mdd"]:7.2%}')
    print(f'[超额] total {100*(P["total"]-bench["total"]):+.2f}pp   sharpe {P["sharpe"]-bench["sharpe"]:+.3f}')

    pr = period_rets(periods)
    print('\n[分期] 策略 vs 沪深300')
    print(f'{"period":9s} {"eff":9s} {"n":>3s} {"ret":>9s} {"bench":>9s} {"excess":>9s}')
    bser = bnav.reindex(CAL).ffill()
    for p, rr in zip(periods, pr):
        rb = float(bser.iloc[p['i1'] - 1] / bser.iloc[p['i0']] - 1)
        print(f'{rr["tag"]:9s} {rr["eff"]:9s} {p["n"]:3d} {rr["ret"]:>+8.2%} {rb:>+8.2%} {rr["ret"]-rb:>+8.2%}')

    print('\n' + '=' * 78 + '\n[前视审计电池]\n' + '=' * 78)
    Ad = {}
    Ad['LA1'] = la1_time_order(ledger)
    print(f"LA-1 时间序 (run_date < eff): {'PASS' if Ad['LA1']['pass_'] else 'FAIL'}  {len(Ad['LA1']['ledger'])} 期")
    for r in Ad['LA1']['ledger']:
        print(f'   {r["tag"]:9s} run={r["run_date"]} -> eff={r["eff"]}  间隔 {r["gap_cal_days"]:2d} 天 / '
              f'{r["trading_days_between"]} 个交易日')

    Ad['LA2'] = la2_information_set()
    print(f"\nLA-2 信息集边界: {'PASS' if Ad['LA2']['pass_'] else 'FAIL'}")
    print(f'   守卫计数 {Ad["LA2"]["guard_counts"]}')
    print(f'   危险谓词(> cutoff) {Ad["LA2"]["danger_predicates"]}')

    Ad['LA3'] = la3_leak_value(R, A, P)
    o = Ad['LA3']['positive_control_oracle']
    g = Ad['LA3']['same_bar_forfeit']
    print(f"\nLA-3a 正向对照(完全前视选股): total={o['total']:+.1%} vs 基线 {P['total']:+.1%} ⇒ "
          f"Δ{o['vs_baseline_pp']:+.0f}pp  — 探测器{'灵敏 ✅ (可见泄漏)' if o['vs_baseline_pp'] > 100 else '⚠不灵敏'}")
    print(f"LA-3b 同 bar 前视上限(决策收盘->执行价, 我们拿不到): 累计 {g['cum']:+.2%}  "
          f"均值 {g['mean']:+.3%}/期  ⇒ {'可忽略 ✅' if abs(g['cum']) < 0.05 else '⚠ 需要复核'}")
    for x in g['per_period']:
        print(f'   {x["tag"]:9s} {x["from_"]}->{x["to"]}  放弃收益={x["forfeited_ret"]:+.2%}  n={x["n"]}')

    Ad['LA4'] = {}
    print('\nLA-4 执行滞后 (同口径基准 = lag0):')
    b0 = None
    for L in [0, 1, 2, 3, 5, 10]:
        s, _, _ = simulate(R, A, k=1.0, mode='moc_newcomer', lag=L)
        pf = perf(s)
        if b0 is None:
            b0 = pf['total']
        Ad['LA4'][f'lag{L}'] = dict(total=pf['total'], delta_pp=(pf['total'] - b0) * 100)
        print(f'   lag{L:<3d} total={pf["total"]:+8.2%}  Δ={Ad["LA4"][f"lag{L}"]["delta_pp"]:+7.2f}pp')

    Ad['LA5'] = la5_universe(R, A, codes, periods)
    print(f"\nLA-5 宇宙审计: list_date 违规={len(Ad['LA5']['list_date_violations'])} "
          f"{'PASS' if Ad['LA5']['pass_list_date'] else 'FAIL'} | "
          f"退市股可见={Ad['LA5']['delisted_coverage']['has_list_status']} | "
          f"持仓期停牌案例={Ad['LA5']['n_suspended_cases']} 最长连续无价={Ad['LA5']['max_consecutive_no_price_days']}日")

    Ad['LA6'] = la6_returns_check(periods, ledger, R)
    print(f"LA-6 收益口径: 覆盖 {Ad['LA6']['coverage_start']}..{Ad['LA6']['coverage_end']} "
          f"({'PASS' if Ad['LA6']['pass_coverage'] else 'FAIL'}) | "
          f"无缺口无重叠={'PASS' if Ad['LA6']['pass_no_gap_no_overlap'] else 'FAIL'} | "
          f"首期首日=0 {'PASS' if Ad['LA6']['pass_first_period_day0_zero'] else 'FAIL'} | "
          f"无越界价={'PASS' if Ad['LA6']['pass_no_future_price'] else 'FAIL'}")

    Ad['LA7'] = la7_evidence_years(PERIOD_META)
    print(f"\nLA-7 证据链财年审计(数据级): {'PASS' if Ad['LA7']['pass_'] else 'FAIL'}  "
          f"越界 {len(Ad['LA7']['violations'])} 行")
    for x in Ad['LA7']['by_period']:
        print(f'   {x["tag"]:9s} run={x["run_date"]} report_end={x["report_end"]} '
              f'候选={x["n_candidates"]:4d}  证据链最大财年={x["max_year_seen"]}  '
              f'越界行={x["n_rows_with_future_year"]}')

    tot_new = sum(p['n_new'] for p in periods)
    years = NCAL / 250
    turns = (tot_new / 15) / years
    print(f'\n[换手] 新进合计={tot_new} 只 ({tot_new/15:.2f} 组合当量)  年化 {turns:.2f} 次  '
          f'30bp 年化成本 {turns*COST_BP/1e4:.2%}')

    ann = OrderedDict()
    for y in ['2024', '2025', '2026']:
        idx = [d for d in base.index if d.startswith(y)]
        if len(idx) < 5:
            continue
        s = base[idx] / base[idx].iloc[0]
        bs = bser[idx] / bser[idx].iloc[0]
        ann[y] = dict(strategy=float(s.iloc[-1] - 1), bench=float(bs.iloc[-1] - 1),
                      excess=float(s.iloc[-1] - bs.iloc[-1]))
        print(f'[分年] {y}: 策略 {ann[y]["strategy"]:+.2%}  基准 {ann[y]["bench"]:+.2%}  超额 {ann[y]["excess"]:+.2%}')

    out = pd.DataFrame(dict(nav=base, bench=bser.reindex(base.index).ffill()))
    out.index.name = 'date'
    out.to_csv(f'{OUTP}/nav_pit.csv')
    json.dump(dict(
        contract='选池数据<=run_date; 建仓=eff(首个>run_date交易日) MOC; 建仓日仅新进者收益归零; 停牌计入分母',
        engine_check=dict(independent=P, v4_reference=REF, d_total_pp=d_tot,
                          d_sharpe=P['sharpe'] - REF['sharpe'], d_mdd_pp=(P['mdd'] - REF['mdd']) * 100),
        perf=P, variants=var, k_frontier=ktab,
        cost=dict(bp=COST_BP, perf=cost_perf, delta_pp=(cost_perf['total'] - P['total']) * 100),
        bench=bench, periods=pr, annual=ann,
        turnover=dict(total_new=tot_new, portfolio_turns=tot_new / 15, per_year=turns,
                      ann_cost_pct=turns * COST_BP / 1e4),
    ), open(f'{OUTP}/stats_pit.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=str)
    json.dump(Ad, open(f'{OUTP}/audit_pit.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=str)
    print(f'\n[saved] {OUTP}/nav_pit.csv  stats_pit.json  audit_pit.json')


if __name__ == '__main__':
    main()
