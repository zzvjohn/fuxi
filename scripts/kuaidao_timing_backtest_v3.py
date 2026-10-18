# -*- coding: utf-8 -*-
"""快刀斩股 日频择时回测引擎 v3 (2026-09-11)
策略结构: 选股池按财报披露季度刷新(与 v2 相同 8 期池), 池内个股用日频技术信号决定买卖。
用户钦定执行标准 v3 (2026-09-11 修改):
  买入: ①下跌到近一年最低价附近(±3%)买 ②持续下跌并跌破支撑位(120日线)买
        ③一阳穿4线(收盘价一天突破5&10&20&30日均线)买
  卖出: ①上涨到前高(近一年最高价附近±3%)卖 ②压力位(上涨趋势且接近120日线)卖 ③50%收益止盈
无前视清单:
  - L250/R250 = 过去250交易日(不含当日)最低/最高 -> shift(1); 触发用 t 日 close (收盘已知, MOC 合法)
  - MA5/10/20/30/120 含当日 (买入/卖出均为收盘市价单, 收盘时均线已知)
  - 跌破 MA120 = 昨收>=MA120[t-1] 且今收<MA120[t] (穿越事件) 且 5 日下跌 (持续下跌)
  - 压力位 = 收盘从下方接近 MA120 (close<MA120 且距 3% 以内) 且 5 日上涨 (上涨趋势)
  - 50% 止盈: close >= 买入成交价 x 1.50
  - 买单/卖单全部 MOC 收盘成交 (v2 的预挂限价单口径弃用, 因新卖条件含 t 日 close)
  - 建仓日贡献0; 当日卖出不回购; 换仓间隙与 v1/v2 一致
价格: 后复权 OHLC (pool_ohlc/, 已补历史至 2022-12-01 供 250 日窗口暖机)
变体: A=年频池+BH(校验) / B=季频池+BH / E2=年频池+择时v3 / C2=季频池+择时v3(主角) / C2net=含成本
旧规格 C 结果引用 v2 的 stats.json 对照。
"""
import sys, os, json
import numpy as np
import pandas as pd

sys.path.insert(0, 'e:/quant/scripts')
os.chdir('e:/quant')

FM = 'data/kuaidao_test/full_market'
OUT = 'data/kuaidao_test/full_backtest'
OUT3 = f'{OUT}/timing3'
END_D = '20260910'
YEAR_N = 250              # 近一年 = 250 交易日
NEAR_TOL = 0.03           # 高低点"附近"±3%
MA_NEAR = 0.03            # 距 120 日线 3% 以内视为"接近"
TAKE_PROFIT = 0.50        # 50% 收益止盈
BUY_COST, SELL_COST = 0.0003, 0.0008

ANNUAL = [('20240430', 'P2023'), ('20250430', 'P2024'), ('20260430', 'P2025')]
QUARTERLY = sorted(ANNUAL + [('20240831', 'P2024H1'), ('20241031', 'P2024Q3'),
                             ('20250831', 'P2025H1'), ('20251031', 'P2025Q3'), ('20260831', 'P2026H1')])

_PXC, _IND = {}, {}


# ---------------- 数据加载 ----------------
def load_calendar():
    b = pd.read_csv('data/raw/index_000300_daily.csv', dtype=str)
    dates = sorted(b['trade_date'].str.replace('-', '').tolist())
    return [d for d in dates if '20240430' < d <= END_D]


def load_pool(tag):
    f = f'{OUT}/pool_{tag}.csv'
    if not os.path.exists(f):
        return []
    df = pd.read_csv(f, dtype={'code': str})
    return [c for c in df['code'].dropna().tolist() if c]


def get_px(c):
    if c not in _PXC:
        f = f'{FM}/pool_ohlc/{c}.csv'
        _PXC[c] = None
        if os.path.exists(f):
            d = pd.read_csv(f, dtype={'trade_date': str}).sort_values('trade_date').drop_duplicates('trade_date')
            _PXC[c] = d.set_index('trade_date')
            _IND[c] = add_indicators(_PXC[c])
    return _PXC[c]


def get_ind(c):
    get_px(c)
    return _IND.get(c)


def add_indicators(d):
    """预计算 v3 信号量. 买入/卖出均为 MOC, 均线含当日; L250/R250 为过去一年(不含当日)."""
    c, o = d['close'], d['open']
    ind = pd.DataFrame(index=d.index)
    for n in (5, 10, 20, 30, 120):
        ind[f'ma{n}'] = c.rolling(n).mean()
    ind['ma_min'] = ind[['ma5', 'ma10', 'ma20', 'ma30']].min(axis=1)
    ind['ma_max'] = ind[['ma5', 'ma10', 'ma20', 'ma30']].max(axis=1)
    ind['L250'] = d['low'].rolling(YEAR_N).min().shift(1)    # 近一年最低(不含当日)
    ind['R250'] = d['high'].rolling(YEAR_N).max().shift(1)   # 近一年最高(不含当日)
    ind['close5'] = c.shift(5)                                # 5 日趋势基准
    ind['ma120_prev'] = ind['ma120'].shift(1)
    ind['prev_close'] = c.shift(1)
    return ind


# ---------------- 单变体回测 ----------------
def run_variant(name, schedule, mode, net=False, verbose=True):
    """schedule: [(run_date, pool_tag)]; mode: 'bh'|'timing2'"""
    cal = load_calendar()
    pools = {}
    for rd, tag in schedule:
        codes = load_pool(tag)
        pools[rd] = set(codes) if codes else None
    eff_map = {}
    for rd, _ in schedule:
        later = [d for d in cal if d > rd]
        eff_map[rd] = min(later) if later else None

    state = {}          # code -> 'FLAT'|'LONG'
    entered_at = {}     # code -> 建仓日 (建仓日贡献0)
    entry_px = {}       # code -> 买入成交价 (50%止盈基准)
    pending_bh = set()
    members = set()
    nav = 1.0
    last_px = {}
    nav_rows, trades, period_rows = [], [], []
    nb1 = nb2 = nb3 = ns1 = ns2 = ns3 = 0
    cur_period = None

    for t in cal:
        # ---- 1. 换仓日 ----
        due = [(rd, tag) for rd, tag in schedule if eff_map[rd] == t]
        for rd, tag in due:
            new_pool = pools.get(rd) or members
            new_pool = set(new_pool)
            leavers = members - new_pool
            newcomers = new_pool - members
            for c in leavers:
                state.pop(c, None)
                entered_at.pop(c, None)
                entry_px.pop(c, None)
                pending_bh.discard(c)
            for c in newcomers:
                state[c] = 'FLAT'
                entered_at.pop(c, None)
                entry_px.pop(c, None)
                if mode == 'bh':
                    pending_bh.add(c)
            members = new_pool
            if cur_period is not None:
                period_rows[-1]['nav_end'] = nav
            period_rows.append(dict(eff=t, tag=tag, n=len(members),
                                    n_in=len(newcomers), n_out=len(leavers), nav_end=None))
            cur_period = True
            if verbose:
                print(f'[{t}] refresh {tag}: pool={len(members)} in={len(newcomers)} out={len(leavers)}', flush=True)
        # ---- 2. bh 待建仓 ----
        sold_today = set()
        for c in list(pending_bh):
            d = get_px(c)
            if d is not None and t in d.index:
                fill = d.loc[t, 'close'] * (1 + BUY_COST) if net else d.loc[t, 'close']
                trades.append(dict(date=t, code=c, side='BUY', px=round(float(fill), 4), reason='建仓'))
                state[c] = 'LONG'
                entered_at[c] = t
                entry_px[c] = fill
                pending_bh.discard(c)
        # ---- 3. 持仓收益 + 卖出信号 + 买入信号 ----
        contribs = {}
        n_t = 0
        for c in members:
            d = get_px(c)
            if d is None:
                continue
            ind = get_ind(c)
            in_t = t in d.index
            if in_t:
                last_px[c] = float(d.loc[t, 'close'])
            if c not in last_px:
                continue
            n_t += 1
            if state.get(c) == 'LONG':
                if in_t:
                    r, i_ = d.loc[t], ind.loc[t]
                    prev_close = i_['prev_close']
                    if np.isfinite(prev_close) and prev_close > 0:
                        sold = False
                        if mode == 'timing2' and entered_at.get(c) != t:
                            close = r['close']
                            r250, ma120, close5 = i_['R250'], i_['ma120'], i_['close5']
                            # S1 前高(近一年最高±3%): close 进入 R250 下方3%区间或以上
                            if np.isfinite(r250) and close >= r250 * (1 - NEAR_TOL):
                                fill = close * (1 - SELL_COST) if net else close
                                contribs[c] = fill / prev_close - 1
                                state[c] = 'FLAT'
                                sold_today.add(c)
                                trades.append(dict(date=t, code=c, side='SELL', px=round(float(fill), 4), reason='前高R250'))
                                ns1 += 1
                                sold = True
                            # S2 压力位: 上涨趋势(5日上涨)且收盘自下方接近120日线(3%以内)
                            elif (np.isfinite(ma120) and np.isfinite(close5) and close < ma120
                                  and close >= ma120 * (1 - MA_NEAR) and close > close5):
                                fill = close * (1 - SELL_COST) if net else close
                                contribs[c] = fill / prev_close - 1
                                state[c] = 'FLAT'
                                sold_today.add(c)
                                trades.append(dict(date=t, code=c, side='SELL', px=round(float(fill), 4), reason='压力位MA120'))
                                ns2 += 1
                                sold = True
                            # S3 50% 止盈
                            elif np.isfinite(entry_px.get(c, np.nan)) and close >= entry_px[c] * (1 + TAKE_PROFIT):
                                fill = close * (1 - SELL_COST) if net else close
                                contribs[c] = fill / prev_close - 1
                                state[c] = 'FLAT'
                                sold_today.add(c)
                                trades.append(dict(date=t, code=c, side='SELL', px=round(float(fill), 4), reason='50%止盈'))
                                ns3 += 1
                                sold = True
                        if not sold:
                            contribs[c] = 0.0 if entered_at.get(c) == t else d.loc[t, 'close'] / prev_close - 1
            # ---- 买入信号 (日初空仓者, MOC 收盘成交; 建仓日不计收益; 当日已卖出不回购) ----
            if mode == 'timing2' and state.get(c) == 'FLAT' and in_t and c not in sold_today:
                r, i_ = d.loc[t], ind.loc[t]
                close = r['close']
                l250, ma120, ma120p, close5 = i_['L250'], i_['ma120'], i_['ma120_prev'], i_['close5']
                reason = None
                # B1 下跌到近一年最低价附近(±3%): 收盘落入 L250 上方3%内(含创新低) 且当日下跌
                if (np.isfinite(l250) and close <= l250 * (1 + NEAR_TOL) and close < i_['prev_close']):
                    reason = '一年低点附近'
                # B2 持续下跌并跌破120日线: 穿越事件(昨收在线上、今收在线下)且5日下跌
                elif (np.isfinite(ma120) and np.isfinite(ma120p) and np.isfinite(close5)
                      and close < ma120 and i_['prev_close'] >= ma120p and close < close5):
                    reason = '跌破MA120'
                # B3 一阳穿4线: 收盘价一天突破 5&10&20&30 日均线 (open<min 蕴含阳线)
                elif (np.isfinite(i_['ma30']) and r['open'] < i_['ma_min'] and close > i_['ma_max']):
                    reason = '一阳穿4线'
                if reason:
                    fill = close * (1 + BUY_COST) if net else close
                    state[c] = 'LONG'
                    entered_at[c] = t
                    entry_px[c] = fill
                    trades.append(dict(date=t, code=c, side='BUY', px=round(float(fill), 4), reason=reason))
                    if reason == '一年低点附近':
                        nb1 += 1
                    elif reason == '跌破MA120':
                        nb2 += 1
                    else:
                        nb3 += 1
        # ---- 4. 组合日收益 ----
        r_t = sum(contribs.values()) / n_t if n_t else 0.0
        nav *= (1 + r_t)
        nav_rows.append(dict(date=t, nav=nav,
                             n_pos=sum(1 for c in members if state.get(c) == 'LONG'), n_mem=n_t))
    if period_rows:
        period_rows[-1]['nav_end'] = nav
    open_pos = [c for c in members if state.get(c) == 'LONG']
    return dict(name=name, nav_rows=nav_rows, trades=trades, period_rows=period_rows,
                sig=dict(b1=nb1, b2=nb2, b3=nb3, s1=ns1, s2=ns2, s3=ns3), open_pos=open_pos)


# ---------------- 绩效 ----------------
def perf(nav_series, rf=0.02):
    r = nav_series.pct_change().dropna()
    total = float(nav_series.iloc[-1] / nav_series.iloc[0] - 1)
    cagr = float((nav_series.iloc[-1] / nav_series.iloc[0]) ** (250 / max(len(nav_series), 1)) - 1)
    rf_d = rf / 250
    sharpe = float((r.mean() - rf_d) / r.std() * np.sqrt(250)) if r.std() > 0 else float('nan')
    dd = nav_series / nav_series.cummax() - 1
    return dict(total=total, cagr=cagr, sharpe=sharpe, mdd=float(dd.min()))


def roundtrip_stats(trades):
    tr = pd.DataFrame(trades)
    if len(tr) == 0:
        return dict(n_rt=0, win=None, avg_hold=None, n_buy=0, n_sell=0)
    tr = tr.sort_values('date').reset_index(drop=True)
    open_buys, rts = {}, []
    for _, r in tr.iterrows():
        c = r['code']
        if r['side'] == 'BUY':
            open_buys.setdefault(c, []).append((r['date'], r['px'], r['reason']))
        else:
            if open_buys.get(c):
                bd, bpx, _ = open_buys[c].pop(0)
                rts.append(dict(code=c, buy_date=bd, sell_date=r['date'], buy_px=bpx, sell_px=r['px'],
                                ret=r['px'] / bpx - 1, buy_reason=_, sell_reason=r['reason']))
    n_done = len(rts)
    win = float(np.mean([x['ret'] > 0 for x in rts])) if n_done else None
    cal = load_calendar()
    cidx = {d: i for i, d in enumerate(cal)}
    holds = [cidx[x['sell_date']] - cidx[x['buy_date']] for x in rts if x['buy_date'] in cidx and x['sell_date'] in cidx]
    avg_hold = float(np.mean(holds)) if holds else None
    return dict(n_rt=n_done, win=win, avg_hold=avg_hold, n_buy=int((tr['side'] == 'BUY').sum()),
                n_sell=int((tr['side'] == 'SELL').sum()), rts=rts)


def main():
    variants = [
        ('A_annual_bh', ANNUAL, 'bh', False),
        ('B_quarter_bh', QUARTERLY, 'bh', False),
        ('E2_annual_timing2', ANNUAL, 'timing2', False),
        ('C2_quarter_timing', QUARTERLY, 'timing2', False),
        ('C2net_quarter_timing', QUARTERLY, 'timing2', True),
    ]
    results = {}
    for name, sched, mode, net in variants:
        print(f'\n===== {name} =====', flush=True)
        res = run_variant(name, sched, mode, net)
        s = pd.DataFrame(res['nav_rows']).set_index('date')['nav']
        res['nav_series'] = s
        res['perf'] = perf(s)
        df = pd.DataFrame(res['nav_rows'])
        res['exposure'] = float((df['n_pos'] / df['n_mem'].clip(lower=1)).mean())
        res['rt'] = roundtrip_stats(res['trades']) if res['trades'] else dict(n_rt=0, win=None, avg_hold=None, n_buy=0, n_sell=0, rts=[])
        results[name] = res
        print(f"[{name}] total={res['perf']['total']:+.2%} sharpe={res['perf']['sharpe']:.2f} "
              f"mdd={res['perf']['mdd']:.1%} expo={res['exposure']:.0%} trades={len(res['trades'])} "
              f"sig={res['sig']} rt={res['rt']['n_rt']}", flush=True)
    # 基准
    cal = load_calendar()
    b = pd.read_csv('data/raw/index_000300_daily.csv', dtype=str)
    b['d'] = b['trade_date'].str.replace('-', '')
    b = b[b['d'].isin(cal)].sort_values('d')
    bs = pd.Series(b['close'].astype(float).values, index=b['d'].values)
    bp = perf(bs)
    # 输出
    os.makedirs(OUT3, exist_ok=True)
    navs = pd.DataFrame({name: res['nav_series'] for name, res in results.items()})
    navs['bench'] = bs / bs.iloc[0]
    navs.index.name = 'date'
    navs.to_csv(f'{OUT3}/nav_variants3.csv')
    stats = {name: dict(perf=res['perf'], exposure=res['exposure'], sig=res['sig'],
                        rt={k: v for k, v in res['rt'].items() if k != 'rts'},
                        n_open=len(res['open_pos']), open_pos=res['open_pos'],
                        periods=res['period_rows']) for name, res in results.items()}
    stats['bench'] = bp
    # 旧规格 v2 结果引用对照
    try:
        old = json.load(open(f'{OUT}/timing/stats.json', encoding='utf-8'))
        stats['_v2_ref'] = {k: dict(perf=old[k]['perf'], exposure=old[k]['exposure'], rt=old[k]['rt'])
                            for k in ('C_quarter_timing', 'E_annual_timing', 'A_annual_bh', 'B_quarter_bh')}
    except Exception as e:
        print('v2 ref load fail:', e)
    with open(f'{OUT3}/stats3.json', 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=1, default=str)
    for name, res in results.items():
        if res['trades']:
            pd.DataFrame(res['trades']).to_csv(f'{OUT3}/trades3_{name}.csv', index=False)
        if res['rt'].get('rts'):
            pd.DataFrame(res['rt']['rts']).to_csv(f'{OUT3}/roundtrips3_{name}.csv', index=False)
    print('\n[bench]', json.dumps(bp, ensure_ascii=False), flush=True)
    print('[saved] timing3/nav_variants3.csv / stats3.json / trades3_*.csv / roundtrips3_*.csv', flush=True)


if __name__ == '__main__':
    main()
