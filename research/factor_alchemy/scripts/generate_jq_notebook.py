"""
JQ 研究笔记本生成器
=====================
扫描 JQ_FACTOR_REGISTRY, 为所有占位符因子生成 JQ 实现,
输出完整的 .py 笔记本 (Cell 格式, 可直接在 JQ 研究平台运行)

输出文件: research/factor_alchemy/output/jq_research_notebook.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── 因子实现模板 ──

PRICE_FACTOR_IMPLS = {
    # ═══ 收益类 ═══
    'ret_1m': '''
    close = price_data['close']
    n = len(stocks)
    arr = np.full(n, np.nan)
    for i, s in enumerate(stocks):
        c = close[s]
        if len(c) >= 21:
            arr[i] = c[-1] / c[-21] - 1
    valid = np.isfinite(arr)''',
    'ret_3m': '''
    close = price_data['close']
    n = len(stocks)
    arr = np.full(n, np.nan)
    for i, s in enumerate(stocks):
        c = close[s]
        if len(c) >= 63:
            arr[i] = c[-1] / c[-63] - 1
    valid = np.isfinite(arr)''',
    'ret_12m': '''
    close = price_data['close']
    n = len(stocks)
    arr = np.full(n, np.nan)
    for i, s in enumerate(stocks):
        c = close[s]
        if len(c) >= 252:
            arr[i] = c[-1] / c[-252] - 1
    valid = np.isfinite(arr)''',
    'ret_1m_skip1m': '''
    close = price_data['close']
    n = len(stocks)
    arr = np.full(n, np.nan)
    for i, s in enumerate(stocks):
        c = close[s]
        if len(c) >= 42:
            arr[i] = c[-22] / c[-42] - 1
    valid = np.isfinite(arr)''',
    'short_rev_5d': '''
    close = price_data['close']
    n = len(stocks)
    arr = np.full(n, np.nan)
    for i, s in enumerate(stocks):
        c = close[s]
        if len(c) >= 5:
            arr[i] = -(c[-1] / c[-5] - 1)  # 负收益=正因子 (反转)
    valid = np.isfinite(arr)''',
    'max_ret_1m': '''
    close = price_data['close']
    n = len(stocks)
    arr = np.full(n, np.nan)
    for i, s in enumerate(stocks):
        c = close[s]
        if len(c) >= 21:
            daily_ret = c[-21:] / np.roll(c[-21:], 1) - 1
            daily_ret[0] = 0
            arr[i] = np.nanmax(daily_ret)
    valid = np.isfinite(arr)''',
    'price_1m': '''
    close = price_data['close']
    n = len(stocks)
    arr = np.full(n, np.nan)
    for i, s in enumerate(stocks):
        c = close[s]
        if len(c) >= 21:
            arr[i] = c[-1] / c[-21] - 1
    valid = np.isfinite(arr)''',
    'rank_1m': '''
    close = price_data['close']
    n = len(stocks)
    arr = np.full(n, np.nan)
    rets = []
    idxs = []
    for i, s in enumerate(stocks):
        c = close[s]
        if len(c) >= 21:
            rets.append(c[-1] / c[-21] - 1)
            idxs.append(i)
    if rets:
        from scipy.stats import rankdata
        ranks = rankdata(rets)
        for j, idx in enumerate(idxs):
            arr[idx] = ranks[j] / len(rets)
    valid = np.isfinite(arr)''',

    # ═══ 波动率类 ═══
    'vol_1m': '''
    close = price_data['close']
    n = len(stocks)
    arr = np.full(n, np.nan)
    for i, s in enumerate(stocks):
        c = close[s]
        if len(c) >= 21:
            rets = c[-21:] / np.roll(c[-21:], 1) - 1
            rets[0] = 0
            arr[i] = np.nanstd(rets) * np.sqrt(252)
    valid = np.isfinite(arr)''',
    'vol_3m': '''
    close = price_data['close']
    n = len(stocks)
    arr = np.full(n, np.nan)
    for i, s in enumerate(stocks):
        c = close[s]
        if len(c) >= 63:
            rets = c[-63:] / np.roll(c[-63:], 1) - 1
            rets[0] = 0
            arr[i] = np.nanstd(rets) * np.sqrt(252)
    valid = np.isfinite(arr)''',
    'downside_vol': '''
    close = price_data['close']
    n = len(stocks)
    arr = np.full(n, np.nan)
    for i, s in enumerate(stocks):
        c = close[s]
        if len(c) >= 63:
            rets = c[-63:] / np.roll(c[-63:], 1) - 1
            rets[0] = 0
            neg_rets = rets[rets < 0]
            if len(neg_rets) > 0:
                arr[i] = np.nanstd(neg_rets) * np.sqrt(252)
    valid = np.isfinite(arr)''',
    'idio_vol': '''
    close = price_data['close']
    n = len(stocks)
    arr = np.full(n, np.nan)
    for i, s in enumerate(stocks):
        c = close[s]
        if len(c) >= 63:
            rets = c[-63:] / np.roll(c[-63:], 1) - 1
            rets[0] = 0
            if hasattr(price_data, 'market_ret') and price_data.market_ret is not None:
                mkt = price_data.market_ret[-63:]
                resid = rets - mkt
            else:
                resid = rets - np.nanmean(rets)
            arr[i] = np.nanstd(resid) * np.sqrt(252)
    valid = np.isfinite(arr)''',
    'volatility_of_volatility': '''
    close = price_data['close']
    n = len(stocks)
    arr = np.full(n, np.nan)
    for i, s in enumerate(stocks):
        c = close[s]
        if len(c) >= 63:
            rets = c[-63:] / np.roll(c[-63:], 1) - 1
            rets[0] = 0
            rolling_vol = np.array([np.nanstd(rets[max(0,j-20):j+1]) for j in range(20, len(rets))])
            if len(rolling_vol) > 0:
                arr[i] = np.nanstd(rolling_vol)
    valid = np.isfinite(arr)''',

    # ═══ 高阶矩 ═══
    'skewness_20': '''
    close = price_data['close']
    n = len(stocks)
    arr = np.full(n, np.nan)
    for i, s in enumerate(stocks):
        c = close[s]
        if len(c) >= 20:
            rets = c[-20:] / np.roll(c[-20:], 1) - 1
            rets[0] = 0
            std = np.nanstd(rets)
            if std > 0:
                arr[i] = np.nanmean((rets - np.nanmean(rets))**3) / (std**3)
    valid = np.isfinite(arr)''',
    'kurtosis_20': '''
    close = price_data['close']
    n = len(stocks)
    arr = np.full(n, np.nan)
    for i, s in enumerate(stocks):
        c = close[s]
        if len(c) >= 20:
            rets = c[-20:] / np.roll(c[-20:], 1) - 1
            rets[0] = 0
            std = np.nanstd(rets)
            if std > 0:
                arr[i] = np.nanmean((rets - np.nanmean(rets))**4) / (std**4) - 3
    valid = np.isfinite(arr)''',
    'sharpe_20': '''
    close = price_data['close']
    n = len(stocks)
    arr = np.full(n, np.nan)
    for i, s in enumerate(stocks):
        c = close[s]
        if len(c) >= 20:
            rets = c[-20:] / np.roll(c[-20:], 1) - 1
            rets[0] = 0
            std = np.nanstd(rets)
            if std > 0:
                arr[i] = np.nanmean(rets) / std
    valid = np.isfinite(arr)''',

    # ═══ 技术指标 ═══
    'rsi_14': '''
    close = price_data['close']
    n = len(stocks)
    arr = np.full(n, np.nan)
    for i, s in enumerate(stocks):
        c = close[s]
        if len(c) >= 15:
            delta = np.diff(c[-15:])
            gain = np.nanmean(delta[delta > 0]) if np.any(delta > 0) else 0
            loss = -np.nanmean(delta[delta < 0]) if np.any(delta < 0) else 1e-10
            rs = gain / loss if loss > 0 else 100
            arr[i] = 100 - 100 / (1 + rs)
    valid = np.isfinite(arr)''',
    'boll_pct_b': '''
    close = price_data['close']
    n = len(stocks)
    arr = np.full(n, np.nan)
    for i, s in enumerate(stocks):
        c = close[s]
        if len(c) >= 20:
            ma = np.nanmean(c[-20:])
            std = np.nanstd(c[-20:])
            if std > 0:
                arr[i] = (c[-1] - (ma - 2*std)) / (4*std)
    valid = np.isfinite(arr)''',
    'bias_20': '''
    close = price_data['close']
    n = len(stocks)
    arr = np.full(n, np.nan)
    for i, s in enumerate(stocks):
        c = close[s]
        if len(c) >= 20:
            ma = np.nanmean(c[-20:])
            if ma > 0:
                arr[i] = (c[-1] - ma) / ma
    valid = np.isfinite(arr)''',
    'psy_12': '''
    close = price_data['close']
    n = len(stocks)
    arr = np.full(n, np.nan)
    for i, s in enumerate(stocks):
        c = close[s]
        if len(c) >= 12:
            up_days = np.sum(np.diff(c[-13:]) > 0)
            arr[i] = up_days / 12 * 100
    valid = np.isfinite(arr)''',
    'roc_20': '''
    close = price_data['close']
    n = len(stocks)
    arr = np.full(n, np.nan)
    for i, s in enumerate(stocks):
        c = close[s]
        if len(c) >= 20:
            arr[i] = (c[-1] - c[-20]) / c[-20] * 100
    valid = np.isfinite(arr)''',
    'cci_14': '''
    high, low, close = price_data.get('high',{}), price_data.get('low',{}), price_data['close']
    n = len(stocks)
    arr = np.full(n, np.nan)
    for i, s in enumerate(stocks):
        c = close[s]; h = high.get(s, c); l_ = low.get(s, c)
        if len(c) >= 14 and len(h) >= 14 and len(l_) >= 14:
            tp = (h[-14:] + l_[-14:] + c[-14:]) / 3
            ma = np.nanmean(tp)
            md = np.nanmean(np.abs(tp - ma))
            if md > 0:
                arr[i] = (tp[-1] - ma) / (0.015 * md)
    valid = np.isfinite(arr)''',
    'plrc_12': '''
    close = price_data['close']
    n = len(stocks)
    arr = np.full(n, np.nan)
    for i, s in enumerate(stocks):
        c = close[s]
        if len(c) >= 12:
            up = np.sum(np.maximum(np.diff(c[-13:]), 0))
            down = np.sum(np.abs(np.minimum(np.diff(c[-13:]), 0)))
            total = up + down
            if total > 0:
                arr[i] = up / total
    valid = np.isfinite(arr)''',
    'beta': '''
    close = price_data['close']
    n = len(stocks)
    arr = np.full(n, np.nan)
    for i, s in enumerate(stocks):
        c = close[s]
        if len(c) >= 252 and hasattr(price_data, 'market_ret') and price_data.market_ret is not None and len(price_data.market_ret) >= 252:
            stock_rets = c[-252:] / np.roll(c[-252:], 1) - 1
            stock_rets[0] = 0
            mkt = price_data.market_ret[-252:]
            valid_mask = np.isfinite(stock_rets) & np.isfinite(mkt)
            if np.sum(valid_mask) >= 60:
                cov = np.cov(stock_rets[valid_mask], mkt[valid_mask])[0,1]
                var = np.var(mkt[valid_mask])
                if var > 0:
                    arr[i] = cov / var
    valid = np.isfinite(arr)''',

    # ═══ 量能类 ═══
    'avg_turnover_1m': '''
    volume = price_data.get('volume', {})
    n = len(stocks)
    arr = np.full(n, np.nan)
    for i, s in enumerate(stocks):
        v = volume.get(s, np.array([]))
        if len(v) >= 21:
            arr[i] = np.nanmean(v[-21:])
    valid = np.isfinite(arr)''',
    'avg_turnover_3m': '''
    volume = price_data.get('volume', {})
    n = len(stocks)
    arr = np.full(n, np.nan)
    for i, s in enumerate(stocks):
        v = volume.get(s, np.array([]))
        if len(v) >= 63:
            arr[i] = np.nanmean(v[-63:])
    valid = np.isfinite(arr)''',
    'turnover_to_vol': '''
    volume = price_data.get('volume', {})
    close = price_data['close']
    n = len(stocks)
    arr = np.full(n, np.nan)
    for i, s in enumerate(stocks):
        v = volume.get(s, np.array([]))
        c = close[s]
        if len(v) >= 21 and len(c) >= 21:
            tv = np.nanmean(v[-21:])
            rets = c[-21:] / np.roll(c[-21:], 1) - 1; rets[0] = 0
            vol = np.nanstd(rets)
            if vol > 0:
                arr[i] = tv / vol
    valid = np.isfinite(arr)''',
    'volume_ratio': '''
    volume = price_data.get('volume', {})
    n = len(stocks)
    arr = np.full(n, np.nan)
    for i, s in enumerate(stocks):
        v = volume.get(s, np.array([]))
        if len(v) >= 6:
            ma5 = np.nanmean(v[-6:-1])
            if ma5 > 0:
                arr[i] = v[-1] / ma5
    valid = np.isfinite(arr)''',
    'vroc_12': '''
    volume = price_data.get('volume', {})
    n = len(stocks)
    arr = np.full(n, np.nan)
    for i, s in enumerate(stocks):
        v = volume.get(s, np.array([]))
        if len(v) >= 12 and v[-12] > 0:
            arr[i] = (v[-1] - v[-12]) / v[-12]
    valid = np.isfinite(arr)''',
    'vr_26': '''
    volume = price_data.get('volume', {})
    close = price_data['close']
    n = len(stocks)
    arr = np.full(n, np.nan)
    for i, s in enumerate(stocks):
        v = volume.get(s, np.array([]))
        c = close[s]
        if len(v) >= 26 and len(c) >= 26:
            up_vol = np.sum(v[-26:][np.diff(c[-27:-1]) > 0]) if len(c) >= 27 else 0
            down_vol = np.sum(v[-26:][np.diff(c[-27:-1]) < 0]) if len(c) >= 27 else 1e-10
            if down_vol > 0:
                arr[i] = up_vol / down_vol
    valid = np.isfinite(arr)''',
    'vpt': '''
    volume = price_data.get('volume', {})
    close = price_data['close']
    n = len(stocks)
    arr = np.full(n, np.nan)
    for i, s in enumerate(stocks):
        v = volume.get(s, np.array([]))
        c = close[s]
        if len(v) >= 26 and len(c) >= 27:
            chg = c[-26:] / np.roll(c[-26:], 1) - 1; chg[0] = 0
            arr[i] = np.nansum(v[-26:] * chg)
    valid = np.isfinite(arr)''',
    'davol_20': '''
    volume = price_data.get('volume', {})
    n = len(stocks)
    arr = np.full(n, np.nan)
    for i, s in enumerate(stocks):
        v = volume.get(s, np.array([]))
        if len(v) >= 20:
            arr[i] = v[-1] / np.nanmean(v[-20:]) - 1
    valid = np.isfinite(arr)''',
    'dollar_vol_stability': '''
    volume = price_data.get('volume', {})
    close = price_data['close']
    n = len(stocks)
    arr = np.full(n, np.nan)
    for i, s in enumerate(stocks):
        v = volume.get(s, np.array([]))
        c = close[s]
        if len(v) >= 20 and len(c) >= 20:
            dv = np.array([v[j] * c[j] for j in range(-20, 0)])
            arr[i] = np.nanstd(dv) / (np.nanmean(dv) + 1e-10)
    valid = np.isfinite(arr)''',
    'amihud_illiq': '''
    amount = price_data.get('amount', {})
    close = price_data['close']
    n = len(stocks)
    arr = np.full(n, np.nan)
    for i, s in enumerate(stocks):
        amt = amount.get(s, np.array([]))
        c = close[s]
        if len(amt) >= 20 and len(c) >= 20:
            rets = np.abs(c[-20:] / np.roll(c[-20:], 1) - 1); rets[0] = 0
            ratio = rets / (amt[-20:] + 1)
            arr[i] = np.nanmean(ratio) * 1e8
    valid = np.isfinite(arr)''',
    'abnormal_turnover': '''
    volume = price_data.get('volume', {})
    n = len(stocks)
    arr = np.full(n, np.nan)
    for i, s in enumerate(stocks):
        v = volume.get(s, np.array([]))
        if len(v) >= 20:
            arr[i] = (v[-1] - np.nanmean(v[-20:])) / (np.nanstd(v[-20:]) + 1e-10)
    valid = np.isfinite(arr)''',

    # ═══ 开盘/日内 ═══
    'intraday_reversal': '''
    open_p = price_data.get('open', {})
    close = price_data['close']
    n = len(stocks)
    arr = np.full(n, np.nan)
    for i, s in enumerate(stocks):
        o = open_p.get(s, np.array([]))
        c = close[s]
        if len(o) >= 1 and len(c) >= 1:
            arr[i] = -(c[-1] - o[-1]) / (o[-1] + 0.001)  # 负值=日内跌→明天反转涨
    valid = np.isfinite(arr)''',
    'opening_gap_momentum': '''
    open_p = price_data.get('open', {})
    close = price_data['close']
    n = len(stocks)
    arr = np.full(n, np.nan)
    for i, s in enumerate(stocks):
        o = open_p.get(s, np.array([]))
        c = close[s]
        if len(o) >= 2 and len(c) >= 2:
            arr[i] = o[-1] / c[-2] - 1  # 今日开盘相对昨收的缺口
    valid = np.isfinite(arr)''',
    'high_52w_dist': '''
    high = price_data.get('high', {})
    close = price_data['close']
    n = len(stocks)
    arr = np.full(n, np.nan)
    for i, s in enumerate(stocks):
        h = high.get(s, np.array([]))
        c = close[s]
        if len(h) >= 252 and len(c) >= 1:
            h52 = np.nanmax(h[-252:])
            if h52 > 0:
                arr[i] = (c[-1] - h52) / h52
    valid = np.isfinite(arr)''',
    'high_52w_rank': '''
    close = price_data['close']
    n = len(stocks)
    arr = np.full(n, np.nan)
    for i, s in enumerate(stocks):
        c = close[s]
        if len(c) >= 252:
            h52 = np.nanmax(c[-252:])
            l52 = np.nanmin(c[-252:])
            if h52 > l52:
                arr[i] = (c[-1] - l52) / (h52 - l52)
    valid = np.isfinite(arr)''',

    # ═══ 趋势 ═══
    'trend_persistence_score': '''
    close = price_data['close']
    n = len(stocks)
    arr = np.full(n, np.nan)
    for i, s in enumerate(stocks):
        c = close[s]
        if len(c) >= 60:
            up = np.sum(np.diff(c[-60:]) > 0)
            down = np.sum(np.diff(c[-60:]) < 0)
            total = up + down
            if total > 0:
                consec = 0; max_consec = 0
                prev_up = None
                for j in range(-60+1, 0):
                    curr_up = c[j] > c[j-1]
                    if prev_up is not None and curr_up == prev_up:
                        consec += 1
                    else:
                        consec = 1
                    max_consec = max(max_consec, consec)
                    prev_up = curr_up
                arr[i] = (up - down) / total * max_consec / 10
    valid = np.isfinite(arr)''',
    'tvma_20': '''
    volume = price_data.get('volume', {})
    close = price_data['close']
    n = len(stocks)
    arr = np.full(n, np.nan)
    for i, s in enumerate(stocks):
        v = volume.get(s, np.array([]))
        c = close[s]
        if len(v) >= 20 and len(c) >= 20:
            total_v = np.nansum(v[-20:])
            if total_v > 0:
                w_price = np.nansum(v[-20:] * c[-20:]) / total_v
                arr[i] = c[-1] / w_price - 1
    valid = np.isfinite(arr)''',
    'trend_smoothness': '''
    close = price_data['close']
    n = len(stocks)
    arr = np.full(n, np.nan)
    for i, s in enumerate(stocks):
        c = close[s]
        if len(c) >= 60:
            rets = c[-60:] / np.roll(c[-60:], 1) - 1; rets[0] = 0
            sign_changes = np.sum(np.diff(np.sign(rets[10:])) != 0)
            arr[i] = -sign_changes / 50  # 负值=更平滑=更高分
    valid = np.isfinite(arr)''',
    'max_drawdown_duration': '''
    close = price_data['close']
    n = len(stocks)
    arr = np.full(n, np.nan)
    for i, s in enumerate(stocks):
        c = close[s]
        if len(c) >= 60:
            peak = c[-60]; trough = c[-60]; dd_days = 0; max_dd_days = 0
            for p in c[-59:]:
                if p >= peak:
                    peak = p; trough = p; dd_days = 0
                else:
                    if p < trough:
                        trough = p
                    dd_days += 1
                    max_dd_days = max(max_dd_days, dd_days)
            arr[i] = -max_dd_days / 60  # 负值=回撤短=更高分
    valid = np.isfinite(arr)''',

    # ═══ 规模 ═══
    'ln_mcap': '''
    # 需要市值数据, 在 notebook 中从 get_fundamentals 获取
    n = len(stocks)
    arr = np.full(n, np.nan)
    if hasattr(price_data, 'mcap') and price_data.mcap is not None:
        for i, s in enumerate(stocks):
            if s in price_data.mcap:
                arr[i] = np.log(price_data.mcap[s] + 1)
    valid = np.isfinite(arr)''',
    'ln_circulating_mcap': '''
    n = len(stocks)
    arr = np.full(n, np.nan)
    if hasattr(price_data, 'circ_mcap') and price_data.circ_mcap is not None:
        for i, s in enumerate(stocks):
            if s in price_data.circ_mcap:
                arr[i] = np.log(price_data.circ_mcap[s] + 1)
    valid = np.isfinite(arr)''',

    # ═══ Power类 ═══
    'bull_power': '''
    high = price_data.get('high', {})
    close = price_data['close']
    n = len(stocks)
    arr = np.full(n, np.nan)
    for i, s in enumerate(stocks):
        h = high.get(s, np.array([]))
        c = close[s]
        if len(h) >= 13 and len(c) >= 13:
            ema = c[-1]
            alpha = 2/14
            for p in c[-14:-1][::-1]:
                ema = alpha * p + (1-alpha) * ema
            arr[i] = h[-1] - ema
    valid = np.isfinite(arr)''',
    'bear_power': '''
    low = price_data.get('low', {})
    close = price_data['close']
    n = len(stocks)
    arr = np.full(n, np.nan)
    for i, s in enumerate(stocks):
        l_ = low.get(s, np.array([]))
        c = close[s]
        if len(l_) >= 13 and len(c) >= 13:
            ema = c[-1]
            alpha = 2/14
            for p in c[-14:-1][::-1]:
                ema = alpha * p + (1-alpha) * ema
            arr[i] = ema - l_[-1]  # ema高于最低价=多头
    valid = np.isfinite(arr)''',
    'streak': '''
    close = price_data['close']
    n = len(stocks)
    arr = np.full(n, np.nan)
    for i, s in enumerate(stocks):
        c = close[s]
        if len(c) >= 2:
            if c[-1] > c[-2]:
                streak = 1
                for j in range(-3, -1):
                    if j >= -len(c) and c[j] > c[j-1]:
                        streak += 1
                    else:
                        break
                arr[i] = streak
            elif c[-1] < c[-2]:
                streak = -1
                for j in range(-3, -1):
                    if j >= -len(c) and c[j] < c[j-1]:
                        streak -= 1
                    else:
                        break
                arr[i] = streak
            else:
                arr[i] = 0
    valid = np.isfinite(arr)''',
    'panic_selling': '''
    volume = price_data.get('volume', {})
    close = price_data['close']
    n = len(stocks)
    arr = np.full(n, np.nan)
    for i, s in enumerate(stocks):
        v = volume.get(s, np.array([]))
        c = close[s]
        if len(v) >= 20 and len(c) >= 20:
            ret = c[-1] / c[-2] - 1
            avg_v = np.nanmean(v[-20:])
            if avg_v > 0 and ret < -0.02:
                arr[i] = -(v[-1] / avg_v)  # 放量下跌=负因子(恐慌), 取负后高分=非恐慌
    valid = np.isfinite(arr)''',
}

FUNDAMENTAL_FACTOR_IMPLS = {
    'cfp': '''
    # 经营性现金流/市值
    n = len(stocks)
    arr = np.full(n, np.nan)
    if hasattr(price_data, 'fundamentals') and price_data.fundamentals is not None:
        for i, s in enumerate(stocks):
            f = price_data.fundamentals.get(s, {})
            ocf = f.get('ocf', np.nan)
            mcap = f.get('market_cap', np.nan)
            if np.isfinite(ocf) and np.isfinite(mcap) and mcap > 0:
                arr[i] = ocf / mcap
    valid = np.isfinite(arr)''',
    'dp': '''
    n = len(stocks)
    arr = np.full(n, np.nan)
    if hasattr(price_data, 'fundamentals') and price_data.fundamentals is not None:
        for i, s in enumerate(stocks):
            f = price_data.fundamentals.get(s, {})
            dps = f.get('dividend_per_share', np.nan)
            price = f.get('price', np.nan)
            if np.isfinite(dps) and np.isfinite(price) and price > 0:
                arr[i] = dps / price
    valid = np.isfinite(arr)''',
    'roic': '''
    n = len(stocks)
    arr = np.full(n, np.nan)
    if hasattr(price_data, 'fundamentals') and price_data.fundamentals is not None:
        for i, s in enumerate(stocks):
            f = price_data.fundamentals.get(s, {})
            nopat = f.get('nopat', np.nan)
            ic = f.get('invested_capital', np.nan)
            if np.isfinite(nopat) and np.isfinite(ic) and ic > 0:
                arr[i] = nopat / ic
    valid = np.isfinite(arr)''',
    'asset_growth': '''
    n = len(stocks)
    arr = np.full(n, np.nan)
    if hasattr(price_data, 'fundamentals') and price_data.fundamentals is not None:
        for i, s in enumerate(stocks):
            f = price_data.fundamentals.get(s, {})
            ta = f.get('total_assets', np.nan)
            ta_q = f.get('total_assets_q4', np.nan)
            if np.isfinite(ta) and np.isfinite(ta_q) and ta_q > 0:
                arr[i] = ta / ta_q - 1
    valid = np.isfinite(arr)''',
    'asset_turnover': '''
    n = len(stocks)
    arr = np.full(n, np.nan)
    if hasattr(price_data, 'fundamentals') and price_data.fundamentals is not None:
        for i, s in enumerate(stocks):
            f = price_data.fundamentals.get(s, {})
            rev = f.get('revenue', np.nan)
            ta = f.get('total_assets', np.nan)
            if np.isfinite(rev) and np.isfinite(ta) and ta > 0:
                arr[i] = rev / ta
    valid = np.isfinite(arr)''',
    'ocf_quality': '''
    n = len(stocks)
    arr = np.full(n, np.nan)
    if hasattr(price_data, 'fundamentals') and price_data.fundamentals is not None:
        for i, s in enumerate(stocks):
            f = price_data.fundamentals.get(s, {})
            ocf = f.get('ocf', np.nan)
            ni = f.get('net_income', np.nan)
            if np.isfinite(ocf) and np.isfinite(ni) and abs(ni) > 1e6:
                arr[i] = ocf / ni
    valid = np.isfinite(arr)''',
    'debt_coverage': '''
    n = len(stocks)
    arr = np.full(n, np.nan)
    if hasattr(price_data, 'fundamentals') and price_data.fundamentals is not None:
        for i, s in enumerate(stocks):
            f = price_data.fundamentals.get(s, {})
            ebitda = f.get('ebitda', np.nan)
            interest = f.get('interest_expense', np.nan)
            if np.isfinite(ebitda) and np.isfinite(interest) and interest > 0:
                arr[i] = ebitda / interest
    valid = np.isfinite(arr)''',
    'accruals': '''
    n = len(stocks)
    arr = np.full(n, np.nan)
    if hasattr(price_data, 'fundamentals') and price_data.fundamentals is not None:
        for i, s in enumerate(stocks):
            f = price_data.fundamentals.get(s, {})
            ni = f.get('net_income', np.nan)
            ocf = f.get('ocf', np.nan)
            ta = f.get('total_assets', np.nan)
            if np.isfinite(ni) and np.isfinite(ocf) and np.isfinite(ta) and ta > 0:
                arr[i] = (ni - ocf) / ta
    valid = np.isfinite(arr)''',
}

NEUT_FACTORS = [
    'abnormal_turnover_neut', 'atr_14_neut', 'avg_turnover_1m_neut',
    'bp_neut', 'davol_20_neut', 'dollar_vol_20d_neut',
    'downside_vol_neut', 'max_ret_1m_neut', 'ret_3m_neut',
    'vol_1m_neut', 'vol_3m_neut',
]

PRICE_FACTOR_IMPLS['skew_1m'] = '''
    close = price_data['close']
    n = len(stocks)
    arr = np.full(n, np.nan)
    for i, s in enumerate(stocks):
        c = close[s]
        if len(c) >= 21:
            rets = c[-21:] / np.roll(c[-21:], 1) - 1
            rets[0] = 0
            std = np.nanstd(rets)
            if std > 0:
                arr[i] = np.nanmean((rets - np.nanmean(rets))**3) / (std**3)
    valid = np.isfinite(arr)'''

COMPLEX_SKIP = [
    'scc_network_centrality',  # 需要图计算
    'attention_decay',          # 需要复杂衰减模型
    'bargaining_power_proxy',   # 需要上下游数据
    'capital_efficiency_proxy', # 需要多表联查
    'cashflow_matching_proxy',  # 需要现金流表
    'earnings_consistency_proxy', # 需要多年数据
    'earnings_quality_proxy',   # 需要多指标合成
    'operational_efficiency_proxy', # 需要多表
    'post_earnings_stability',  # 需要事件日期
]
