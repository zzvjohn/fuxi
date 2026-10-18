# -*- coding: utf-8 -*-
"""快刀斩股 日频择时回测引擎 v2 (2026-09-10)
策略结构: 选股池按财报披露季度刷新(4-30年报 / 8-31中报 / 10-31三季报), 池内个股用日频技术信号决定买卖。
用户钦定执行标准:
  买入: 前低买/支撑位买(60日最低价或3日确认摆动低点回踩企稳) 或 一阳穿4线(5&10&20&30日均线)
  卖出: 前高卖/压力位卖(60日最高价阻力, 限价单预挂, high>=R60 成交于 min(R60, open))
无前视清单:
  - MA/S60/R60/摆动低点 全部仅用 t-1 及以前数据 (rolling 后 shift(1))
  - 卖单价格(R60)在 t-1 收盘已知 -> t 日开盘前挂限价单, 成交价=min(R60, 当日open) 标准撮合; 仅阻力在昨收上方时有效
  - 买单信号含 t 日收盘 -> 收盘市价单(MOC)以 close[t] 成交, 建仓日不计收益
  - 选股池数据 ann_date<=run_date (季度打分引擎继承 v1)
  - 换仓: 离池股最后收益日=生效日前一交易日, 生效日收盘等权买入新成员, 生效日当日行情双方不计 (与 v1 间隙口径一致)
价格: 后复权 OHLC (pool_ohlc/, close x adj_factor)
组合日收益 = 池内成员收益等权均值(空仓/停牌成员计0, 分母=有价成员数, 与 v1 口径一致)
变体: A=年频池+买入持有(复算v1校验) / B=季频池+买入持有 / E=年频池+择时 / C=季频池+择时(用户规格,主角)
      D=C+破位止损(敏感性) / Cnet=C+交易成本(佣金0.03%双边+印花0.05%卖出)
"""
import sys, os, json
import numpy as np
import pandas as pd

sys.path.insert(0, 'e:/quant/scripts')
os.chdir('e:/quant')

FM = 'data/kuaidao_test/full_market'
OUT = 'data/kuaidao_test/full_backtest'
END_D = '20260910'
WINDOW_DAYS = 60          # 前低/前高回看窗口(交易日)
PIV_CONFIRM = 3           # 摆动低点确认天数
PIV_LOOKBACK = 120        # 摆动低点有效回看
TOUCH_TOL = 0.015         # 支撑触及容差 1.5%
STOP_FACT = 0.97          # 破位止损: 收盘 < 60日最低x0.97
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
    """在真实行序列上预计算信号量 (全部不含当日)"""
    c, o, h, l = d['close'], d['open'], d['high'], d['low']
    ind = pd.DataFrame(index=d.index)
    for n in (5, 10, 20, 30):
        ind[f'ma{n}'] = c.rolling(n).mean()
    ind['ma_min'] = ind[['ma5', 'ma10', 'ma20', 'ma30']].min(axis=1)
    ind['ma_max'] = ind[['ma5', 'ma10', 'ma20', 'ma30']].max(axis=1)
    ind['S60'] = l.rolling(WINDOW_DAYS).min().shift(1)        # 前低: 过去60交易日(不含当日)最低
    ind['R60'] = h.rolling(WINDOW_DAYS).max().shift(1)        # 前高: 过去60交易日(不含当日)最高
    roll_min = l.rolling(2 * PIV_CONFIRM + 1, center=True).min()
    is_piv = (l == roll_min) & l.notna()
    piv = l.shift(PIV_CONFIRM).where(is_piv.shift(PIV_CONFIRM))  # t-3 为摆动低点 -> t 日确认可用
    ind['piv_low'] = piv.ffill(limit=PIV_LOOKBACK)
    ind['prev_close'] = c.shift(1)
    return ind


# ---------------- 单变体回测 ----------------
def run_variant(name, schedule, mode, stop=False, net=False, verbose=True):
    """schedule: [(run_date, pool_tag)]; mode: 'bh'|'timing'"""
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
    entered_at = {}     # code -> 建仓日 (建仓日贡献0, 次日起算收益)
    pending_bh = set()
    members = set()
    nav = 1.0
    last_px = {}
    nav_rows, trades, period_rows = [], [], []
    sig_a = sig_b = 0
    cur_period = None   # (eff_day, tag, start_nav)

    for t in cal:
        # ---- 1. 换仓日: 离池股在生效日前一交易日收盘已卖出(当日贡献0), 新成员生效日收盘建仓(贡献0) ----
        due = [(rd, tag) for rd, tag in schedule if eff_map[rd] == t]
        for rd, tag in due:
            new_pool = pools.get(rd) or members      # 池子缺失/为空 -> 沿用旧池
            new_pool = set(new_pool)
            leavers = members - new_pool
            newcomers = new_pool - members
            for c in leavers:
                state.pop(c, None)
                entered_at.pop(c, None)
                pending_bh.discard(c)
            for c in newcomers:
                state[c] = 'FLAT'
                entered_at.pop(c, None)
                if mode == 'bh':
                    pending_bh.add(c)
            members = new_pool
            if cur_period is not None:
                period_rows[-1]['nav_end'] = nav      # 上一段终点
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
                pending_bh.discard(c)
        # ---- 3. 持仓收益 + 买入信号 ----
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
                        r60, s60 = i_['R60'], i_['S60']
                        sold = False
                        if mode == 'timing' and entered_at.get(c) != t:
                            # 前高/压力位卖: 限价单(min(R60,open)), 仅当阻力在昨收上方(真压力)
                            if np.isfinite(r60) and r60 > prev_close and r['high'] >= r60:
                                fill = min(r60, r['open'])
                                fill = fill * (1 - SELL_COST) if net else fill
                                contribs[c] = fill / prev_close - 1
                                state[c] = 'FLAT'
                                sold_today.add(c)
                                trades.append(dict(date=t, code=c, side='SELL', px=round(float(fill), 4), reason='前高R60'))
                                sold = True
                            # 破位止损 (敏感性变体D)
                            elif stop and np.isfinite(s60) and r['close'] < s60 * STOP_FACT:
                                fill = r['close'] * (1 - SELL_COST) if net else r['close']
                                contribs[c] = fill / prev_close - 1
                                state[c] = 'FLAT'
                                sold_today.add(c)
                                trades.append(dict(date=t, code=c, side='SELL', px=round(float(fill), 4), reason='破位止损'))
                                sold = True
                        if not sold:
                            contribs[c] = 0.0 if entered_at.get(c) == t else r['close'] / prev_close - 1
                # 停牌日: 贡献0 (分母含)
            # ---- 买入信号 (日初空仓者, 收盘市价单; 建仓日不计收益; 当日已卖出不回购) ----
            if mode == 'timing' and state.get(c) == 'FLAT' and in_t and c not in sold_today:
                r, i_ = d.loc[t], ind.loc[t]
                if np.isfinite(i_['ma30']) and np.isfinite(i_['S60']) and np.isfinite(i_['prev_close']):
                    # A 一阳穿4线 (严格): 阳线实体从四线之下穿到四线之上
                    buy_a = (r['close'] > r['open'] and r['open'] < i_['ma_min'] and r['close'] > i_['ma_max'])
                    # B 前低/支撑位: 回踩60日最低或摆动低点, 收盘收回且较昨收上涨
                    up = r['close'] > i_['prev_close']
                    b1 = r['low'] <= i_['S60'] * (1 + TOUCH_TOL) and r['close'] > i_['S60'] and up
                    b2 = np.isfinite(i_['piv_low']) and i_['piv_low'] > 0 and r['low'] <= i_['piv_low'] * (1 + TOUCH_TOL) and r['close'] > i_['piv_low'] and up
                    if buy_a or b1 or b2:
                        fill = r['close'] * (1 + BUY_COST) if net else r['close']
                        state[c] = 'LONG'
                        reason = '一阳穿4线' if buy_a else ('前低支撑' if b1 else '摆动低点支撑')
                        trades.append(dict(date=t, code=c, side='BUY', px=round(float(fill), 4), reason=reason))
                        if buy_a:
                            sig_a += 1
                        else:
                            sig_b += 1
        # ---- 4. 组合日收益 ----
        r_t = sum(contribs.values()) / n_t if n_t else 0.0
        nav *= (1 + r_t)
        nav_rows.append(dict(date=t, nav=nav,
                             n_pos=sum(1 for c in members if state.get(c) == 'LONG'), n_mem=n_t))
    if period_rows:
        period_rows[-1]['nav_end'] = nav       # 最后一段终点 = 期末
    open_pos = [c for c in members if state.get(c) == 'LONG']
    return dict(name=name, nav_rows=nav_rows, trades=trades, period_rows=period_rows,
                sig_a=sig_a, sig_b=sig_b, open_pos=open_pos)


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
                                ret=r['px'] / bpx - 1, sell_reason=r['reason']))
    n_done = len(rts)
    win = float(np.mean([x['ret'] > 0 for x in rts])) if n_done else None
    # 平均持有交易日
    cal = load_calendar()
    cidx = {d: i for i, d in enumerate(cal)}
    holds = [cidx[x['sell_date']] - cidx[x['buy_date']] for x in rts if x['buy_date'] in cidx and x['sell_date'] in cidx]
    avg_hold = float(np.mean(holds)) if holds else None
    return dict(n_rt=n_done, win=win, avg_hold=avg_hold, n_buy=int((tr['side'] == 'BUY').sum()),
                n_sell=int((tr['side'] == 'SELL').sum()))


def main():
    variants = [
        ('A_annual_bh', ANNUAL, 'bh', False, False),
        ('B_quarter_bh', QUARTERLY, 'bh', False, False),
        ('E_annual_timing', ANNUAL, 'timing', False, False),
        ('C_quarter_timing', QUARTERLY, 'timing', False, False),
        ('D_quarter_timing_stop', QUARTERLY, 'timing', True, False),
        ('Cnet_quarter_timing', QUARTERLY, 'timing', False, True),
    ]
    results = {}
    for name, sched, mode, stop, net in variants:
        print(f'\n===== {name} =====', flush=True)
        res = run_variant(name, sched, mode, stop, net)
        s = pd.DataFrame(res['nav_rows']).set_index('date')['nav']
        res['nav_series'] = s
        res['perf'] = perf(s)
        df = pd.DataFrame(res['nav_rows'])
        res['exposure'] = float((df['n_pos'] / df['n_mem'].clip(lower=1)).mean())
        res['rt'] = roundtrip_stats(res['trades']) if res['trades'] else dict(n_rt=0, win=None, avg_hold=None, n_buy=0, n_sell=0)
        results[name] = res
        print(f"[{name}] total={res['perf']['total']:+.2%} sharpe={res['perf']['sharpe']:.2f} "
              f"mdd={res['perf']['mdd']:.1%} expo={res['exposure']:.0%} trades={len(res['trades'])} "
              f"sigA={res['sig_a']} sigB={res['sig_b']} rt={res['rt']['n_rt']}", flush=True)
    # 基准 (同一交易日序列)
    cal = load_calendar()
    b = pd.read_csv('data/raw/index_000300_daily.csv', dtype=str)
    b['d'] = b['trade_date'].str.replace('-', '')
    b = b[b['d'].isin(cal)].sort_values('d')
    bs = pd.Series(b['close'].astype(float).values, index=b['d'].values)
    bp = perf(bs)
    # 输出
    os.makedirs(f'{OUT}/timing', exist_ok=True)
    navs = pd.DataFrame({name: res['nav_series'] for name, res in results.items()})
    navs['bench'] = bs / bs.iloc[0]
    navs.index.name = 'date'
    navs.to_csv(f'{OUT}/timing/nav_variants.csv')
    stats = {name: dict(perf=res['perf'], exposure=res['exposure'], sig_a=res['sig_a'], sig_b=res['sig_b'],
                        rt=res['rt'], n_open=len(res['open_pos']), open_pos=res['open_pos'],
                        periods=res['period_rows']) for name, res in results.items()}
    stats['bench'] = bp
    with open(f'{OUT}/timing/stats.json', 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=1, default=str)
    for name, res in results.items():
        if res['trades']:
            pd.DataFrame(res['trades']).to_csv(f'{OUT}/timing/trades_{name}.csv', index=False)
    print('\n[bench]', json.dumps(bp, ensure_ascii=False), flush=True)
    print('[saved] timing/nav_variants.csv / stats.json / trades_*.csv', flush=True)


if __name__ == '__main__':
    main()
