# -*- coding: utf-8 -*-
"""快刀斩股 日频择时回测引擎 v4 (2026-09-11)

Step1 改造实装: A-4 对照框架 + C-1 出场改造 + N-1 离池条件化。
选股/建仓逻辑 **完全不动** (与 v3 一致), 只动出场分支与离池处理。

入场 (与 v3 相同, 两态)
  'bh'  : 换池生效日收盘全员等权买入, 持有
  'sig' : 日频空仓者按 B1/B2/B3 信号 MOC 买入
     B1 下跌到近一年最低价附近(±3%) 且当日下跌
     B2 持续下跌并跌破 MA120 (穿越事件 + 5日下跌)
     B3 一阳穿4线 (open < min(ma5,ma10,ma20,ma30) 且 close > max(...))
  reentry=True: 'bh' 模式下场内空仓者也可用 B1/B2/B3 回补

出场
  'none': 永不卖出 (纯 BH 基线)
  'old' : v3 原规格 —— S1 前高R250(±3%) / S2 压力位MA120 / S3 50%止盈
  'c1'  : C-1 改造规格 ——
         ① ATR(14) 2× chandelier 移动止损, **只升不降** (ratchet)
         ② 三级成本保护 (取与①的较高者作为止损线):
              maxgain >= +30% -> 锁定 +10%
              maxgain >= +15% -> 保本 (0%)
              其余           -> 硬底线 −9%
         ③ T+3 水下 3% 砍 (建仓后第3个交易日收盘 < entry×0.97)
         ④ 60 交易日时间止损
         全部为 MOC 收盘市价单, 卖出价 = close×(1−sell_cost)

N-1 离池条件化 (修 L2 "日历无条件卖出")
  换池日离池: 若 still LONG 且 close > MA60 则**保留仓位**, 移至 legacy 池继续
  按自身出场规则退出; 否则按原逻辑直接平掉。上限 MAX_LEGACY 只 (按 close/MA60−1 降序优先)。
  legacy 标的重新入池时无缝并回 (保仓位), 不作人为换手。
  分母: 跟踪只数 n_t = |members ∪ legacy|; 权重 1/n_t (等权, gross ≤ 1)。

前视清单: L250/R250 = 过去250日(不含当日) shift(1); 均线/ATR 含当日;
  hh(建仓后最高价) = 含当日 high; 所有信号用 t 日与更早信息, MOC 成交。
  ⚠ 禁 lag0 —— 实测 1 天前视可凭空造出 +249pp。

A-4 输出: 每个变体输出 member_rets / n_pos / n_track, 由 main 统一生成
  随机暴露带 (2000 路径) 与 暴露等价解析基准 scaled_bh, 并给出 verdict。

输出: data/kuaidao_test/full_backtest/timing4/{nav_variants4.csv,stats4.json,
      trades4_*.csv,roundtrips4_*.csv,bench4.json}
"""
import sys, os, json
import numpy as np
import pandas as pd

sys.path.insert(0, 'e:/quant/scripts')
os.chdir('e:/quant')
from kuaidao_bench_framework import random_exposure_band, scaled_bh_nav, verdict  # noqa: E402

FM = 'data/kuaidao_test/full_market'
OUT = 'data/kuaidao_test/full_backtest'
OUT4 = f'{OUT}/timing4'
END_D = '20260910'
YEAR_N = 250              # 近一年 = 250 交易日
NEAR_TOL = 0.03           # 高低点"附近"±3%
MA_NEAR = 0.03            # 距 120 日线 3% 以内视为"接近"
TAKE_PROFIT = 0.50        # (仅 old 规格) 50% 收益止盈
# --- C-1 参数 ---
ATR_N, ATR_MULT = 14, 2.0
HARD_STOP = 0.09          # −9% 硬底线
LOCK1_GAIN, LOCK1_FLOOR = 0.15, 0.00   # +15% -> 保本
LOCK2_GAIN, LOCK2_FLOOR = 0.30, 0.10   # +30% -> 锁 +10%
T3_DAYS, T3_DD = 3, 0.03  # T+3 水下 3%
TIME_STOP_DAYS = 60       # 60 交易日时间止损
MAX_LEGACY = 5            # N-1 离池保留上限
COST11 = (0.0003, 0.0008)  # v3 原口径, 合计 11bp/回合
COST15 = (0.0005, 0.0010)  # 报告硬性前置, 合计 15bp/回合

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
    """预计算 v3/v4 信号量. 买卖均 MOC, 均线/ATR 含当日; L250/R250 为过去一年(不含当日)."""
    c, o = d['close'], d['open']
    ind = pd.DataFrame(index=d.index)
    for n in (5, 10, 20, 30, 60, 120):
        ind[f'ma{n}'] = c.rolling(n).mean()
    ind['ma_min'] = ind[['ma5', 'ma10', 'ma20', 'ma30']].min(axis=1)
    ind['ma_max'] = ind[['ma5', 'ma10', 'ma20', 'ma30']].max(axis=1)
    ind['L250'] = d['low'].rolling(YEAR_N).min().shift(1)    # 近一年最低(不含当日)
    ind['R250'] = d['high'].rolling(YEAR_N).max().shift(1)   # 近一年最高(不含当日)
    ind['close5'] = c.shift(5)
    ind['ma120_prev'] = ind['ma120'].shift(1)
    ind['prev_close'] = c.shift(1)
    # ATR(14) — 简单均值口径 (SMA), 明确可复现
    pc = c.shift(1)
    tr = pd.concat([(d['high'] - d['low']), (d['high'] - pc).abs(), (d['low'] - pc).abs()], axis=1).max(axis=1)
    ind['atr'] = tr.rolling(ATR_N).mean()
    return ind


# ---------------- C-1 出场状态机 (可组合消融) ----------------
# spec: atr_mult(ATR倍数|None) / hard(硬止损比例|None) / t3(bool) / time_stop(天数|None) / cost_protect(bool)
DEFAULT_C1 = dict(atr_mult=ATR_MULT, hard=HARD_STOP, t3=True,
                  time_stop=TIME_STOP_DAYS, cost_protect=True)


def pos_init(t, px, hi, atr, spec):
    stop = -np.inf
    if spec.get('hard'):
        stop = px * (1 - spec['hard'])
    if spec.get('atr_mult') and np.isfinite(atr):
        stop = max(stop, hi - spec['atr_mult'] * atr)
    return dict(entry_t=t, entry_px=px, hh=hi, stop=stop)


def pos_update(p, hi, atr, spec):
    """只升不降: stop_t = max(stop_{t-1}, hh_t - m*ATR_t, 成本保护底线)."""
    p['hh'] = max(p['hh'], hi)
    floors = [p['stop']]
    if spec.get('cost_protect'):
        mg = p['hh'] / p['entry_px'] - 1
        if mg >= LOCK2_GAIN:
            floors.append(p['entry_px'] * (1 + LOCK2_FLOOR))
        elif mg >= LOCK1_GAIN:
            floors.append(p['entry_px'] * (1 + LOCK1_FLOOR))
        elif spec.get('hard'):
            floors.append(p['entry_px'] * (1 - spec['hard']))
    if spec.get('atr_mult') and np.isfinite(atr):
        floors.append(p['hh'] - spec['atr_mult'] * atr)
    p['stop'] = max(floors)
    return p['stop']


def c1_sell(close, p, i, days_held, spec):
    """返回出场原因 (None = 不出场)."""
    if np.isfinite(p['stop']) and close < p['stop']:
        if p['stop'] >= p['entry_px']:
            return '锁利/保本止损'
        if spec.get('hard') and abs(p['stop'] - p['entry_px'] * (1 - spec['hard'])) < 1e-9:
            return '硬止损'
        return 'ATR移动止损'
    if spec.get('t3') and days_held == T3_DAYS and close < p['entry_px'] * (1 - T3_DD):
        return 'T+3水下3%'
    if spec.get('time_stop') and days_held >= spec['time_stop']:
        return '时间止损'
    return None


# ---------------- 单变体回测 ----------------
def run_variant(name, schedule, entry='bh', exit_rule='none', cond_exit=False,
                reentry=False, cost=(0.0, 0.0), exit_spec=None, placebo_holds=None, verbose=True):
    """entry: 'bh'|'sig'; exit_rule: 'none'|'old'|'c1'|'placebo'; cond_exit: N-1; reentry: 空仓回补; cost: (buy, sell)
    exit_spec: 仅 exit_rule='c1' 时生效, 可组合规则 (见 DEFAULT_C1)
    placebo_holds: 仅 exit_rule='placebo' 时生效, {(code, entry_date): 目标持有交易日数}
                   —— 用于 A-4 安慰剂: 保持入场日与持有期分布不变, 只随机重排"谁持有几天"。"""
    buy_cost, sell_cost = cost
    spec = dict(exit_spec) if exit_spec else dict(DEFAULT_C1)
    ph = placebo_holds or {}
    cal = load_calendar()
    cidx = {d: i for i, d in enumerate(cal)}
    pools = {}
    for rd, tag in schedule:
        codes = load_pool(tag)
        pools[rd] = set(codes) if codes else None
    eff_map = {}
    for rd, _ in schedule:
        later = [d for d in cal if d > rd]
        eff_map[rd] = min(later) if later else None

    state = {}          # code -> 'FLAT'|'LONG'
    pos = {}            # code -> C-1 持仓状态
    entered_at = {}     # code -> 建仓日 (建仓日贡献0)
    entry_px = {}       # code -> 买入成交价 (old: 50%止盈基准)
    pending_bh = set()
    members = set()
    legacy = set()      # N-1 离池保留
    nav = 1.0
    last_px = {}
    nav_rows, trades, period_rows = [], [], []
    member_rets, n_pos_arr, n_track_arr, r_bh_arr = [], [], [], []
    nb1 = nb2 = nb3 = ns = 0
    sell_reasons = {}
    legacy_kept = legacy_dropped = 0
    cur_period = None

    for t in cal:
        ti = cidx[t]
        # ---- 1. 换池日 ----
        due = [(rd, tag) for rd, tag in schedule if eff_map[rd] == t]
        for rd, tag in due:
            new_pool = pools.get(rd) or members
            new_pool = set(new_pool)
            leavers = members - new_pool
            newcomers = new_pool - members
            for c in leavers:
                d, ind = get_px(c), get_ind(c)
                was_long = state.get(c) == 'LONG'
                keep = False
                if cond_exit and was_long and d is not None and t in d.index:
                    ma60 = ind.loc[t, 'ma60']
                    if np.isfinite(ma60) and float(d.loc[t, 'close']) > float(ma60):
                        keep = True
                if keep:
                    legacy.add(c)
                    legacy_kept += 1
                else:
                    if was_long:
                        legacy_dropped += 1
                    state.pop(c, None); pos.pop(c, None)
                    entered_at.pop(c, None); entry_px.pop(c, None)
                    pending_bh.discard(c)
                    legacy.discard(c)
            # legacy 上限 (按 close/MA60−1 降序, 强者优先保留)
            if len(legacy) > MAX_LEGACY:
                def _stg(cc):
                    dd, ii = get_px(cc), get_ind(cc)
                    if dd is None or t not in dd.index or ii is None:
                        return -9e9
                    m = ii.loc[t, 'ma60']
                    return (float(dd.loc[t, 'close']) / float(m) - 1) if (np.isfinite(m) and m) else -9e9
                for c in sorted(list(legacy), key=_stg)[:len(legacy) - MAX_LEGACY]:
                    legacy.discard(c)
                    state.pop(c, None); pos.pop(c, None)
                    entered_at.pop(c, None); entry_px.pop(c, None)
                    legacy_dropped += 1
            for c in newcomers:
                if c in legacy and state.get(c) == 'LONG':
                    legacy.discard(c)          # 无缝并回, 保留仓位
                    continue
                state[c] = 'FLAT'
                pos.pop(c, None); entered_at.pop(c, None); entry_px.pop(c, None)
                if entry == 'bh':
                    pending_bh.add(c)
            members = new_pool
            if cur_period is not None:
                period_rows[-1]['nav_end'] = nav
            period_rows.append(dict(eff=t, tag=tag, n=len(members), n_in=len(newcomers),
                                    n_out=len(leavers), n_legacy=len(legacy), nav_end=None))
            cur_period = True
            if verbose:
                print(f'[{t}] refresh {tag}: pool={len(members)} in={len(newcomers)} '
                      f'out={len(leavers)} legacy={len(legacy)}', flush=True)
        # ---- 2. bh 待建仓 ----
        sold_today = set()
        for c in list(pending_bh):
            d, ind = get_px(c), get_ind(c)
            if d is not None and t in d.index:
                fill = float(d.loc[t, 'close']) * (1 + buy_cost)
                trades.append(dict(date=t, code=c, side='BUY', px=round(fill, 4), reason='建仓'))
                state[c] = 'LONG'
                entered_at[c] = t
                entry_px[c] = fill
                pos[c] = pos_init(t, fill, float(d.loc[t, 'high']), float(ind.loc[t, 'atr']), spec)
                pending_bh.discard(c)
        # ---- 3. 持仓收益 + 卖出信号 + 买入信号 ----
        contribs, bar_r = {}, {}
        for c in sorted(members | legacy):
            d, ind = get_px(c), get_ind(c)
            if d is None:
                continue
            in_t = t in d.index
            if in_t:
                last_px[c] = float(d.loc[t, 'close'])
            if c not in last_px:
                continue
            bar_r[c] = 0.0
            if state.get(c) == 'LONG':
                if in_t:
                    r, i_ = d.loc[t], ind.loc[t]
                    prev_close = i_['prev_close']
                    if np.isfinite(prev_close) and prev_close > 0:
                        close = float(r['close'])
                        sold, reason = False, None
                        if entered_at.get(c) != t:
                            if exit_rule == 'c1':
                                p = pos.get(c)
                                if p is None:
                                    p = pos[c] = pos_init(entered_at.get(c) or t, entry_px[c],
                                                          float(r['high']), float(i_['atr']), spec)
                                pos_update(p, float(r['high']), float(i_['atr']), spec)
                                reason = c1_sell(close, p, i_, ti - cidx[entered_at.get(c) or t], spec)
                            elif exit_rule == 'placebo':
                                pdays = ph.get((c, entered_at.get(c)))
                                if pdays is not None and (ti - cidx[entered_at.get(c)]) >= pdays:
                                    reason = 'placebo'
                            elif exit_rule == 'old':
                                r250, ma120, close5 = i_['R250'], i_['ma120'], i_['close5']
                                if np.isfinite(r250) and close >= r250 * (1 - NEAR_TOL):
                                    reason = '前高R250'
                                elif (np.isfinite(ma120) and np.isfinite(close5) and close < ma120
                                      and close >= ma120 * (1 - MA_NEAR) and close > close5):
                                    reason = '压力位MA120'
                                elif np.isfinite(entry_px.get(c, np.nan)) and close >= entry_px[c] * (1 + TAKE_PROFIT):
                                    reason = '50%止盈'
                            if reason:
                                fill = close * (1 - sell_cost)
                                contribs[c] = fill / prev_close - 1
                                bar_r[c] = close / prev_close - 1
                                state[c] = 'FLAT'
                                sold_today.add(c)
                                trades.append(dict(date=t, code=c, side='SELL', px=round(fill, 4), reason=reason))
                                sell_reasons[reason] = sell_reasons.get(reason, 0) + 1
                                ns += 1
                                sold = True
                                if c in legacy:      # legacy 出场即出表
                                    legacy.discard(c)
                        if not sold:
                            rr = 0.0 if entered_at.get(c) == t else close / prev_close - 1
                            contribs[c] = rr
                            bar_r[c] = rr
            elif in_t and np.isfinite(ind.loc[t, 'prev_close']) and ind.loc[t, 'prev_close'] > 0:
                bar_r[c] = float(d.loc[t, 'close']) / float(ind.loc[t, 'prev_close']) - 1
            # ---- 买入信号 (空仓者 MOC 收盘成交; 建仓日不计收益; 当日已卖出不回购) ----
            can_buy = (entry == 'sig' or (reentry and c in members))
            if (can_buy and state.get(c) == 'FLAT' and in_t and c not in sold_today):
                r, i_ = d.loc[t], ind.loc[t]
                close = float(r['close'])
                l250, ma120, ma120p, close5 = i_['L250'], i_['ma120'], i_['ma120_prev'], i_['close5']
                reason = None
                if (np.isfinite(l250) and close <= l250 * (1 + NEAR_TOL) and close < i_['prev_close']):
                    reason = '一年低点附近'
                elif (np.isfinite(ma120) and np.isfinite(ma120p) and np.isfinite(close5)
                      and close < ma120 and i_['prev_close'] >= ma120p and close < close5):
                    reason = '跌破MA120'
                elif (np.isfinite(i_['ma30']) and float(r['open']) < i_['ma_min'] and close > i_['ma_max']):
                    reason = '一阳穿4线'
                if reason:
                    fill = close * (1 + buy_cost)
                    state[c] = 'LONG'
                    entered_at[c] = t
                    entry_px[c] = fill
                    if exit_rule == 'c1':
                        pos[c] = pos_init(t, fill, float(r['high']), float(i_['atr']), spec)
                    trades.append(dict(date=t, code=c, side='BUY', px=round(fill, 4), reason=reason))
                    if reason == '一年低点附近':
                        nb1 += 1
                    elif reason == '跌破MA120':
                        nb2 += 1
                    else:
                        nb3 += 1
        # ---- 4. 组合日收益 ----
        n_t = len(bar_r)
        nup = sum(1 for c in (members | legacy) if state.get(c) == 'LONG')
        r_t = sum(contribs.values()) / n_t if n_t else 0.0
        rb = (sum(bar_r.values()) / n_t) if n_t else 0.0     # 全跟踪标的等权 (BH 日收益)
        nav *= (1 + r_t)
        nav_rows.append(dict(date=t, nav=nav, n_pos=nup, n_mem=n_t))
        member_rets.append([bar_r[c] for c in sorted(bar_r)])
        n_pos_arr.append(nup)
        n_track_arr.append(n_t)
        r_bh_arr.append(rb)
    if period_rows:
        period_rows[-1]['nav_end'] = nav
    open_pos = [c for c in (members | legacy) if state.get(c) == 'LONG']
    return dict(name=name, nav_rows=nav_rows, trades=trades, period_rows=period_rows,
                member_rets=member_rets, n_pos=n_pos_arr, n_track=n_track_arr, r_bh=r_bh_arr,
                sig=dict(b1=nb1, b2=nb2, b3=nb3, sells=ns, sell_reasons=sell_reasons,
                         legacy_kept=legacy_kept, legacy_dropped=legacy_dropped),
                open_pos=open_pos)


# ---------------- 绩效 ----------------
def perf(nav_series, rf=0.02):
    r = nav_series.pct_change().dropna()
    total = float(nav_series.iloc[-1] / nav_series.iloc[0] - 1)
    cagr = float((nav_series.iloc[-1] / nav_series.iloc[0]) ** (250 / max(len(nav_series), 1)) - 1)
    rf_d = rf / 250
    sharpe = float((r.mean() - rf_d) / r.std() * np.sqrt(250)) if r.std() > 0 else float('nan')
    dd = nav_series / nav_series.cummax() - 1
    return dict(total=total, cagr=cagr, sharpe=sharpe, mdd=float(dd.min()))


def roundtrip_stats(trades, cal):
    tr = pd.DataFrame(trades)
    if len(tr) == 0:
        return dict(n_rt=0, win=None, avg_hold=None, med_hold=None, n_buy=0, n_sell=0, rts=[])
    tr = tr.sort_values('date').reset_index(drop=True)
    open_buys, rts = {}, []
    for _, r in tr.iterrows():
        c = r['code']
        if r['side'] == 'BUY':
            open_buys.setdefault(c, []).append((r['date'], r['px'], r['reason']))
        else:
            if open_buys.get(c):
                bd, bpx, br = open_buys[c].pop(0)
                rts.append(dict(code=c, buy_date=bd, sell_date=r['date'], buy_px=bpx, sell_px=r['px'],
                                ret=r['px'] / bpx - 1, buy_reason=br, sell_reason=r['reason']))
    n_done = len(rts)
    win = float(np.mean([x['ret'] > 0 for x in rts])) if n_done else None
    cidx = {d: i for i, d in enumerate(cal)}
    holds = [cidx[x['sell_date']] - cidx[x['buy_date']] for x in rts
             if x['buy_date'] in cidx and x['sell_date'] in cidx]
    avg_hold = float(np.mean(holds)) if holds else None
    med_hold = float(np.median(holds)) if holds else None
    # 右尾: 单笔 >+50% 的占比与平均收益
    tails = [x['ret'] for x in rts if x['ret'] > 0.5]
    return dict(n_rt=n_done, win=win, avg_hold=avg_hold, med_hold=med_hold,
                n_buy=int((tr['side'] == 'BUY').sum()), n_sell=int((tr['side'] == 'SELL').sum()),
                n_tail50=len(tails), avg_ret=float(np.mean([x['ret'] for x in rts])) if n_done else None,
                rts=rts)


# ---------------- 主流程 ----------------
def main():
    variants = [
        # name,               schedule,   entry,  exit,  cond,  re,   cost
        ('A_a_bh',           ANNUAL,    'bh',  'none', False, False, (0.0, 0.0)),   # 校验 ~+36.99%
        ('B_q_bh',           QUARTERLY, 'bh',  'none', False, False, (0.0, 0.0)),   # 基线   ~+39.14%
        ('C_q_sig_old',      QUARTERLY, 'sig', 'old',  False, False, (0.0, 0.0)),   # 旧择时 ~+4.73%
        ('N1_q_sig_old',     QUARTERLY, 'sig', 'old',  True,  False, (0.0, 0.0)),   # N-1 @旧出场
        ('X1_q_bh_c1',       QUARTERLY, 'bh',  'c1',   False, False, (0.0, 0.0)),   # C-1 单向(无回补)
        ('X2_q_bh_c1_re',    QUARTERLY, 'bh',  'c1',   False, True,  (0.0, 0.0)),   # C-1 + 信号回补
        ('X3_q_sig_c1',      QUARTERLY, 'sig', 'c1',   False, False, (0.0, 0.0)),   # C-1 @信号入场
        ('N2_q_sig_c1',      QUARTERLY, 'sig', 'c1',   True,  False, (0.0, 0.0)),   # +N-1
        ('N3_q_bh_c1_re',    QUARTERLY, 'bh',  'c1',   True,  True,  (0.0, 0.0)),   # +N-1
        ('N4_q_bh_none',     QUARTERLY, 'bh',  'none', True,  False, (0.0, 0.0)),   # N-1 单因子(纯L2)
        ('X2net11_q_bh_c1_re', QUARTERLY, 'bh', 'c1',  False, True,  COST11),
        ('X3net15_q_sig_c1', QUARTERLY, 'sig', 'c1',   False, False, COST15),
        ('N2net15_q_sig_c1', QUARTERLY, 'sig', 'c1',   True,  False, COST15),
        ('N3net15_q_bh_c1',  QUARTERLY, 'bh',  'c1',   True,  True,  COST15),
    ]
    cal = load_calendar()
    results = {}
    for row in variants:
        name, sched, entry, ex, cond, re, cost = row[:7]
        espec = row[7] if len(row) > 7 else None
        print(f'\n===== {name} (entry={entry} exit={ex} N1={cond} re={re} cost={cost} spec={espec}) =====', flush=True)
        res = run_variant(name, sched, entry, ex, cond, re, cost, exit_spec=espec)
        s = pd.DataFrame(res['nav_rows']).set_index('date')['nav']
        res['nav_series'] = s
        res['perf'] = perf(s)
        df = pd.DataFrame(res['nav_rows'])
        res['exposure'] = float((df['n_pos'] / df['n_mem'].clip(lower=1)).mean())
        res['rt'] = roundtrip_stats(res['trades'], cal)
        # A-4 基准
        band = random_exposure_band(res['member_rets'], res['n_pos'], n_paths=2000)
        res['band'] = band
        sb = scaled_bh_nav(res['r_bh'], res['n_pos'], res['n_track'])
        res['scaled_bh_total'] = float(sb[-1] - 1)
        res['scaled_bh_series'] = pd.Series(sb, index=s.index)
        res['verdict'] = verdict(res['perf']['total'], band)
        results[name] = res
        v = res['verdict']
        print(f"[{name}] total={res['perf']['total']:+.2%} sharpe={res['perf']['sharpe']:.2f} "
              f"mdd={res['perf']['mdd']:.1%} expo={res['exposure']:.0%} trades={len(res['trades'])} "
              f"rt={res['rt']['n_rt']} win={res['rt']['win']} "
              f"| scaledBH={res['scaled_bh_total']:+.2%} | randband p5={band['pct']['p5']:+.2%} "
              f"p95={band['pct']['p95']:+.2%} z={v['z']:+.2f} beat95={v['beat_hi']}", flush=True)
    # 基准指数
    b = pd.read_csv('data/raw/index_000300_daily.csv', dtype=str)
    b['d'] = b['trade_date'].str.replace('-', '')
    b = b[b['d'].isin(cal)].sort_values('d')
    bs = pd.Series(b['close'].astype(float).values, index=b['d'].values)
    bp = perf(bs)
    # 输出
    os.makedirs(OUT4, exist_ok=True)
    navs = pd.DataFrame({name: res['nav_series'] for name, res in results.items()})
    for name in ('X1_q_bh_c1', 'X2_q_bh_c1_re', 'X3_q_sig_c1', 'N3_q_bh_c1_re'):
        navs[f'{name}__scaledBH'] = results[name]['scaled_bh_series']
    navs['bench_hs300'] = bs / bs.iloc[0]
    navs.index.name = 'date'
    navs.to_csv(f'{OUT4}/nav_variants4.csv')
    # 分期限度超额
    base = results['B_q_bh']['period_rows']
    per_out = {}
    for name, res in results.items():
        pr = res['period_rows']
        rows, prev = [], 1.0
        for i, p in enumerate(pr):
            nv = float(p['nav_end']) if p['nav_end'] is not None else np.nan
            rows.append(dict(eff=p['eff'], tag=p['tag'], ret=nv / prev - 1))
            prev = nv
        per_out[name] = rows
    stats = {name: dict(perf=res['perf'], exposure=res['exposure'], sig=res['sig'],
                        rt={k: v for k, v in res['rt'].items() if k != 'rts'},
                        band=dict(pct=res['band']['pct'], mean=res['band']['mean'], std=res['band']['std']),
                        scaled_bh_total=res['scaled_bh_total'], verdict=res['verdict'],
                        n_open=len(res['open_pos']), open_pos=res['open_pos'],
                        periods=res['period_rows'], period_returns=per_out[name]) for name, res in results.items()}
    # 分期超额 (vs B_q_bh)
    bmap = {r['eff']: r['ret'] for r in per_out['B_q_bh']}
    for name in results:
        for r in per_out[name]:
            r['excess_vs_B'] = r['ret'] - bmap.get(r['eff'], np.nan)
    stats['bench_hs300'] = bp
    stats['_period_excess'] = per_out
    with open(f'{OUT4}/stats4.json', 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=1, default=str)
    with open(f'{OUT4}/bench4.json', 'w', encoding='utf-8') as f:
        json.dump({name: dict(band=dict(pct=r['band']['pct'], mean=r['band']['mean'], std=r['band']['std']),
                              scaled_bh_total=r['scaled_bh_total'], verdict=r['verdict'],
                              exposure=r['exposure'], perf=r['perf'])
                   for name, r in results.items()}, f, ensure_ascii=False, indent=1, default=str)
    for name, res in results.items():
        if res['trades']:
            pd.DataFrame(res['trades']).to_csv(f'{OUT4}/trades4_{name}.csv', index=False)
        if res['rt'].get('rts'):
            pd.DataFrame(res['rt']['rts']).to_csv(f'{OUT4}/roundtrips4_{name}.csv', index=False)
    print('\n[bench HS300]', json.dumps(bp, ensure_ascii=False), flush=True)
    print('[saved] timing4/nav_variants4.csv / stats4.json / bench4.json / trades4_*.csv', flush=True)


if __name__ == '__main__':
    main()
