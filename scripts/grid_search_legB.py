# -*- coding: utf-8 -*-
"""
腿B纯反转策略 — 本地网格搜索
=============================================================
目标：在本地数据上搜索最佳参数组合，目标函数 = Sharpe
严格无未来数据：所有因子计算只用 rebalance_date 之前的数据

优化参数：
  WEIGHT_MODE:    A(IC加权) / B(等权) / C(ICIR^2加权)
  RESID_MOM_WIN:  [40, 60, 80]
  ILLIQ_WIN:      [15, 20, 30]
  MOM_1M_WIN:     [15, 20, 30]
  LOWVOL_FILTER:  [True, False]
  STOCK_NUM:      [15, 20, 25]
  TRAILING_STOP:  [-0.15, -0.18, -0.20]

合计: 3*3*3*3*2*3*3 = 1458 组合

优化策略：预计算所有窗口因子，每个参数组合只跑轻量级组合模拟
=============================================================
"""
import sys, os, time, itertools, warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ==================== 配置 ====================
RAW_DIR  = 'e:/quant/data/raw'
OUT_DIR  = 'e:/quant/reports'
MCAP_MIN = 5  # 市值下限(亿) — 固定

WEIGHTS = {
    'A': (0.40, 0.30, 0.30),
    'B': (0.333, 0.333, 0.334),
    'C': (0.47, 0.27, 0.26),
}

PARAM_SPACE = {
    'WEIGHT_MODE':    ['A', 'B', 'C'],
    'RESID_MOM_WIN':  [40, 60, 80],
    'ILLIQ_WIN':      [15, 20, 30],
    'MOM_1M_WIN':     [15, 20, 30],
    'LOWVOL_FILTER':  [True, False],
    'STOCK_NUM':      [15, 20, 25],
    'TRAILING_STOP':  [-0.15, -0.18, -0.20],
}

# ==================== 数据加载 ====================

def load_data():
    """加载并预计算所有数据"""
    print("=" * 60)
    print("加载数据...")

    # 日线数据
    raw = pd.read_csv(os.path.join(RAW_DIR, 'daily_prices.csv'),
                      dtype={'trade_date': str})
    raw['trade_date'] = pd.to_datetime(raw['trade_date'], format='mixed')

    # 基本面
    basic = pd.read_csv(os.path.join(RAW_DIR, 'daily_basic.csv'),
                        dtype={'trade_date': str})
    basic['trade_date'] = pd.to_datetime(basic['trade_date'], format='mixed')

    # 指数
    idx = pd.read_csv(os.path.join(RAW_DIR, 'index_000300_daily.csv'),
                      dtype={'trade_date': str})
    idx['trade_date'] = pd.to_datetime(idx['trade_date'], format='mixed')
    idx = idx.sort_values('trade_date')
    idx_close = idx['close'].values.astype(np.float64)
    idx_dates = idx['trade_date'].values

    print(f"  日线: {len(raw):,} 行, {raw['ts_code'].nunique()} 只股票")
    print(f"  基本面: {len(basic):,} 行")
    print(f"  沪深300: {len(idx)} 天, {idx['trade_date'].min().date()} ~ {idx['trade_date'].max().date()}")

    # 转换为 wide 格式（仅价格数据）
    print("\n转换 wide 格式...")
    close_wide = raw.pivot_table('close', 'trade_date', 'ts_code', aggfunc='last')
    vol_wide   = raw.pivot_table('vol', 'trade_date', 'ts_code', aggfunc='last')
    price_codes = close_wide.columns.tolist()

    # 基本面数据 pivot
    mcap_wide = basic.pivot_table('total_mv', 'trade_date', 'ts_code', aggfunc='last')
    pb_wide   = basic.pivot_table('pb', 'trade_date', 'ts_code', aggfunc='last')
    pe_wide   = basic.pivot_table('pe', 'trade_date', 'ts_code', aggfunc='last')
    ps_wide   = basic.pivot_table('ps', 'trade_date', 'ts_code', aggfunc='last')

    # 对齐日期：基本面可能和价格日期不完全重合
    price_dates = close_wide.index
    # 对基本面做 reindex 到价格日期 + ffill
    mcap_wide = mcap_wide.reindex(price_dates, method=None)
    pb_wide   = pb_wide.reindex(price_dates, method=None)
    pe_wide   = pe_wide.reindex(price_dates, method=None)
    ps_wide   = ps_wide.reindex(price_dates, method=None)

    # 找出价格和基本面都有数据的共同股票
    fund_codes = set(mcap_wide.columns) & set(pb_wide.columns) & set(pe_wide.columns) & set(ps_wide.columns)
    common_codes = [c for c in price_codes if c in fund_codes]

    # 排除 688/689 科创板（与策略一致）
    common_codes = [c for c in common_codes
                    if not c.startswith('688') and not c.startswith('689')]

    print(f"  价格股票: {len(price_codes)}, 基本面: {len(fund_codes)}, 共同: {len(common_codes)}")

    # 只保留共同代码
    close_v = close_wide[common_codes].values.astype(np.float64)
    vol_v   = vol_wide[common_codes].values.astype(np.float64)
    mcap_v  = mcap_wide[common_codes].values.astype(np.float64)
    pb_v    = pb_wide[common_codes].values.astype(np.float64)
    pe_v    = pe_wide[common_codes].values.astype(np.float64)
    ps_v    = ps_wide[common_codes].values.astype(np.float64)

    # ffill 基本面（发布后持续有效）
    for arr in [mcap_v, pb_v, pe_v, ps_v]:
        mask = ~np.isnan(arr)
        idx_valid = np.where(mask.any(axis=0), mask.argmax(axis=0), 0)
        for j in range(arr.shape[1]):
            col = arr[:, j]
            good = ~np.isnan(col)
            if np.any(good):
                indices = np.where(good)[0]
                values  = col[indices]
                for k in range(len(arr)):
                    pos = np.searchsorted(indices, k, side='right') - 1
                    if pos >= 0:
                        arr[k, j] = values[pos]

    print(f"  对齐后矩阵: {close_v.shape} (days x stocks)")

    return {
        'close_v': close_v, 'vol_v': vol_v,
        'mcap_v': mcap_v, 'pb_v': pb_v, 'pe_v': pe_v, 'ps_v': ps_v,
        'idx_close': idx_close, 'idx_dates': idx_dates,
        'dates': price_dates.values,
        'codes': common_codes,
    }


# ==================== 因子预计算 ====================

def zscore(x):
    m = np.nanmedian(x)
    s = np.nanstd(x)
    return np.where(s > 0, (x - m) / s, 0)


def precompute_factors(data, resid_mom_wins, illiq_wins, mom_wins):
    """预计算所有窗口的因子值，返回 {window_key: (factor_values_per_day, day_indices)}"""

    close_v  = data['close_v']
    vol_v    = data['vol_v']
    idx_close = data['idx_close']
    dates     = data['dates']
    idx_dates = data['idx_dates']
    codes     = data['codes']

    # 只取价格和指数都有数据的日期
    common_end = min(dates[-1], idx_dates[-1])
    n_days = np.searchsorted(dates, common_end, side='right')
    if n_days == 0:
        n_days = len(dates)

    n_stocks = close_v.shape[1]
    print(f"\n预计算因子 (共 {n_days} 天, {n_stocks} 只股票)...")

    # 预计算日收益率矩阵
    daily_ret = close_v[1:n_days] / close_v[0:n_days-1] - 1  # (n_days-1, stocks)

    # 预计算每日 abs_ret/vol (用于 illiq)
    vol_daily = vol_v[1:n_days]  # (n_days-1, stocks)
    abs_ret_div_vol = np.where(vol_daily > 0, np.abs(daily_ret) / vol_daily, 0)
    logret = np.log(np.abs(daily_ret) + 1)
    logret = np.where(np.isfinite(logret), logret, 0)

    all_factors = {}
    max_win = max(max(resid_mom_wins), max(illiq_wins), max(mom_wins))

    print(f"  因子窗口: resid_mom={resid_mom_wins}, illiq={illiq_wins}, mom_1m={mom_wins}")

    # 对每个窗口组合预计算
    total_combos = len(resid_mom_wins) * len(illiq_wins) * len(mom_wins)
    done = 0
    t0 = time.time()

    for rm_win in resid_mom_wins:
        for il_win in illiq_wins:
            for mm_win in mom_wins:
                done += 1
                key = (rm_win, il_win, mm_win)

                # 需要的最小回看天数
                need_bars = max(rm_win, il_win, mm_win) + 2

                # 预计算残差动量 (需要指数日对应)
                # 对每个可计算日，回看 rm_win 天
                n_valid = n_days - need_bars
                if n_valid <= 0:
                    all_factors[key] = {'day_idx': np.array([], dtype=int)}
                    continue

                f_resid_mom = np.full((n_valid, n_stocks), np.nan)
                f_illiq     = np.full((n_valid, n_stocks), np.nan)
                f_mom       = np.full((n_valid, n_stocks), np.nan)

                for d in range(need_bars, n_days):
                    out_idx = d - need_bars

                    # resid_mom: 回看 rm_win 天
                    ret_window = daily_ret[d-rm_win:d]
                    # 跳过全部为 NaN 的股票
                    valid_stocks = np.sum(~np.isnan(ret_window), axis=0) >= max(10, rm_win//2)
                    if not np.any(valid_stocks):
                        continue

                    stock_mean = np.nanmean(ret_window, axis=0)

                    # 找对应指数日
                    this_date = dates[d]
                    idx_pos = np.searchsorted(idx_dates, this_date)
                    if idx_pos >= len(idx_dates) or idx_dates[idx_pos] > this_date:
                        idx_pos -= 1
                    if idx_pos < rm_win + 1:
                        continue

                    idx_ret = idx_close[idx_pos-rm_win+1:idx_pos+1] / idx_close[idx_pos-rm_win:idx_pos] - 1
                    idx_var = np.var(idx_ret)
                    idx_mean = np.mean(idx_ret)

                    if idx_var > 1e-10:
                        idx_ctr = idx_ret - idx_mean
                        stock_ctr = ret_window - stock_mean[np.newaxis, :]
                        cov_sm = np.nansum(idx_ctr[:, np.newaxis] * stock_ctr, axis=0) / rm_win
                        beta = cov_sm / idx_var
                    else:
                        beta = np.ones(n_stocks)

                    alpha = stock_mean - beta * idx_mean
                    f_resid_mom[out_idx] = alpha

                    # illiq: 回看 il_win 天
                    illiq_mean = np.nanmean(abs_ret_div_vol[d-il_win:d], axis=0)
                    f_illiq[out_idx] = illiq_mean

                    # mom_1m: 跳过最近1天，回看 mm_win 天
                    if d >= mm_win + 2:
                        mom_val = close_v[d-1] / close_v[d-mm_win-2] - 1
                        f_mom[out_idx] = mom_val

                all_factors[key] = {
                    'day_idx': np.arange(need_bars, n_days),
                    'alpha': f_resid_mom,
                    'illiq': f_illiq,
                    'mom': f_mom,
                }

                if done % 3 == 0:
                    elapsed = time.time() - t0
                    eta = elapsed / done * (total_combos - done)
                    print(f"\r  [{done}/{total_combos}] {elapsed:.0f}s elapsed, ETA {eta:.0f}s", end='')

    print(f"\n  预计算完成，耗时 {time.time()-t0:.0f}s")
    return all_factors


# ==================== 组合模拟 ====================

def simulate_portfolio(data, all_factors, params):
    """对一组参数模拟组合表现"""

    close_v   = data['close_v']
    vol_v     = data['vol_v']
    mcap_v    = data['mcap_v']
    pb_v      = data['pb_v']
    pe_v      = data['pe_v']
    ps_v      = data['ps_v']
    dates     = data['dates']
    codes     = data['codes']

    rm_win    = params['RESID_MOM_WIN']
    il_win    = params['ILLIQ_WIN']
    mm_win    = params['MOM_1M_WIN']
    weight_mode = params['WEIGHT_MODE']
    lowvol     = params['LOWVOL_FILTER']
    stock_num  = params['STOCK_NUM']
    trail_stop = params['TRAILING_STOP']

    w1, w2, w3 = WEIGHTS[weight_mode]
    need_bars = max(rm_win, il_win, mm_win) + 2
    factor_key = (rm_win, il_win, mm_win)

    if factor_key not in all_factors:
        return None

    fac = all_factors[factor_key]
    day_indices = fac['day_idx']
    n_periods = len(day_indices)

    if n_periods < 10:
        return None

    # 找周一的索引
    monday_mask = np.array([
        pd.Timestamp(dates[i]).weekday() == 0
        for i in day_indices
    ])

    rebalance_days = day_indices[monday_mask]
    n_rebalance = len(rebalance_days)

    if n_rebalance < 5:
        return None

    # 获取每个rebalance日的因子值
    n_stocks = len(codes)
    portfolio_returns = np.zeros(n_rebalance)
    daily_values = [1.0]       # 记录每日组合价值（近似）
    max_price_tracker = {}     # 移动止损跟踪
    held_since = {}

    current_holdings = set()

    for ri, day_idx in enumerate(rebalance_days):
        row = np.searchsorted(day_indices, day_idx)
        if row >= len(fac['alpha']):
            continue

        # 获取该日的因子值
        alpha_now  = fac['alpha'][row]
        illiq_now  = fac['illiq'][row]
        mom_now    = fac['mom'][row]

        # 基本面过滤
        mcap_now   = mcap_v[day_idx]
        pb_now     = pb_v[day_idx]
        pe_now     = pe_v[day_idx]
        ps_now     = ps_v[day_idx]

        fund_ok = (
            (mcap_now >= MCAP_MIN) &
            (pb_now > 0) & (pe_now > 0) & (ps_now > 0) &
            ~np.isnan(mcap_now) & ~np.isnan(pb_now) & ~np.isnan(pe_now) & ~np.isnan(ps_now)
        )

        # 因子有效
        factor_ok = (
            ~np.isnan(alpha_now) & ~np.isnan(illiq_now) & ~np.isnan(mom_now)
        )

        # 低波过滤
        if lowvol:
            logret_win = np.log(np.abs(close_v[day_idx-20:day_idx] / close_v[day_idx-21:day_idx-1] - 1) + 1)
            logret_win = np.where(np.isfinite(logret_win), logret_win, 0)
            stock_vol = np.nanstd(logret_win, axis=0)
            vol_cutoff = np.percentile(stock_vol[~np.isnan(stock_vol) & fund_ok], 75)
            lowvol_ok = (stock_vol <= vol_cutoff) & ~np.isnan(stock_vol)
        else:
            lowvol_ok = np.ones(n_stocks, dtype=bool)

        # 综合过滤
        all_ok = fund_ok & factor_ok & lowvol_ok
        n_ok = np.sum(all_ok)

        if n_ok < stock_num * 2:
            continue

        # Z-score 标准化（在有效股票内）
        def zs_local(x):
            m = np.nanmedian(x)
            s = np.nanstd(x)
            return np.where(s > 0, (x - m) / s, 0)

        score = (w1 * zs_local(-alpha_now[all_ok]) +
                 w2 * zs_local(-illiq_now[all_ok]) +
                 w3 * zs_local(-mom_now[all_ok]))

        # 选 top N
        rank_order = np.argsort(-score)
        selected_idx = np.where(all_ok)[0][rank_order[:stock_num]]
        new_holdings = set(selected_idx)

        # 计算本周收益（持有 new_holdings 到下个周一）
        if ri < n_rebalance - 1:
            next_day = rebalance_days[ri + 1]
            holding_ret = np.zeros(len(selected_idx))
            for j, si in enumerate(selected_idx):
                p_start = close_v[day_idx, si]
                p_end   = close_v[next_day, si]
                if p_start > 0 and not np.isnan(p_start) and not np.isnan(p_end):
                    # 移动止损
                    if si in max_price_tracker:
                        mx = max_price_tracker[si]
                        if mx > 0 and p_end / mx - 1 < trail_stop:
                            holding_ret[j] = trail_stop
                            max_price_tracker.pop(si, None)
                            continue
                    holding_ret[j] = p_end / p_start - 1
                    # 更新最高价
                    if si in max_price_tracker:
                        max_price_tracker[si] = max(max_price_tracker[si], p_end)
                    else:
                        max_price_tracker[si] = max(p_start, p_end)
                else:
                    holding_ret[j] = 0

            weekly_ret = np.nanmean(holding_ret)
            portfolio_returns[ri] = weekly_ret
        else:
            portfolio_returns[ri] = 0

    # 计算指标
    valid_rets = portfolio_returns[portfolio_returns != 0]
    if len(valid_rets) < 10:
        return None

    total_ret = np.prod(1 + valid_rets) - 1
    n_weeks   = len(valid_rets)
    ann_factor = 52
    ann_ret   = (1 + total_ret) ** (ann_factor / n_weeks) - 1
    ann_vol   = np.std(valid_rets) * np.sqrt(ann_factor)
    sharpe    = (ann_ret - 0.02) / ann_vol if ann_vol > 0 else 0

    # MaxDD
    cum = np.cumprod(1 + valid_rets)
    peak = np.maximum.accumulate(cum)
    maxdd = np.min(cum / peak - 1)

    return {
        'sharpe': round(sharpe, 4),
        'total_ret': round(total_ret * 100, 1),
        'ann_ret': round(ann_ret * 100, 1),
        'ann_vol': round(ann_vol * 100, 1),
        'maxdd': round(maxdd * 100, 1),
        'n_weeks': n_weeks,
        'win_rate': round(np.mean(valid_rets > 0) * 100, 1),
    }


# ==================== 主流程 ====================

def main():
    t_total = time.time()

    # 1. 加载数据
    data = load_data()

    # 2. 预计算因子
    all_factors = precompute_factors(
        data,
        resid_mom_wins=PARAM_SPACE['RESID_MOM_WIN'],
        illiq_wins=PARAM_SPACE['ILLIQ_WIN'],
        mom_wins=PARAM_SPACE['MOM_1M_WIN'],
    )

    # 3. 生成参数组合
    param_keys = list(PARAM_SPACE.keys())
    param_values = [PARAM_SPACE[k] for k in param_keys]
    param_combos = list(itertools.product(*param_values))
    total = len(param_combos)
    print(f"\n参数组合总数: {total}")
    print(f"开始评估...")

    # 4. 评估每个组合
    results = []
    t_eval = time.time()
    for i, combo in enumerate(param_combos):
        params = dict(zip(param_keys, combo))

        # 跳过低效组合：高换手窗口+低波过滤
        result = simulate_portfolio(data, all_factors, params)
        if result is not None:
            result['params'] = params
            results.append(result)

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t_eval
            speed = elapsed / (i + 1)
            eta = speed * (total - i - 1)
            print(f"\r  [{i+1}/{total}] {elapsed:.0f}s, {speed:.2f}s/combo, ETA {eta:.0f}s", end='')

    print(f"\n  评估完成，耗时 {time.time()-t_eval:.0f}s")

    # 5. 排序输出
    if not results:
        print("ERROR: 无有效结果！")
        return

    results.sort(key=lambda x: x['sharpe'], reverse=True)

    print("\n" + "=" * 60)
    print("TOP 20 参数组合 (按 Sharpe 排序)")
    print("=" * 60)

    header = f"{'Rank':>4} {'Sharpe':>7} {'收益%':>8} {'MaxDD%':>7} {'年化%':>8} {'胜率%':>6} {'周数':>5}  WEIGHT RM_WIN IL_WIN MM_WIN LOWVOL STK_N STOP"
    print(header)
    print("-" * len(header))

    top20 = results[:20]
    for rank, r in enumerate(top20, 1):
        p = r['params']
        print("%4d %7.3f %8.1f %7.1f %8.1f %6.1f %5d  %-6s %6d %6d %6d %-6s %5d %5.2f" % (
            rank, r['sharpe'], r['total_ret'], r['maxdd'], r['ann_ret'],
            r['win_rate'], r['n_weeks'],
            p['WEIGHT_MODE'], p['RESID_MOM_WIN'], p['ILLIQ_WIN'], p['MOM_1M_WIN'],
            str(p['LOWVOL_FILTER']), p['STOCK_NUM'], p['TRAILING_STOP']
        ))

    # 6. 保存完整结果
    output_file = os.path.join(OUT_DIR, 'grid_search_legB_results.csv')
    df_out = pd.DataFrame([{
        'sharpe': r['sharpe'],
        'total_ret': r['total_ret'],
        'ann_ret': r['ann_ret'],
        'ann_vol': r['ann_vol'],
        'maxdd': r['maxdd'],
        'win_rate': r['win_rate'],
        'n_weeks': r['n_weeks'],
        **{k: r['params'][k] for k in param_keys}
    } for r in results])
    df_out.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n完整结果已保存: {output_file}")

    # 7. 参数重要性分析
    print("\n" + "=" * 60)
    print("参数重要性 (各组最优 Sharpe 均值)")
    print("=" * 60)
    for pk in param_keys:
        group_means = df_out.groupby(pk)['sharpe'].agg(['mean', 'max', 'count'])
        group_means = group_means.sort_values('max', ascending=False)
        print(f"\n{pk}:")
        for val, row in group_means.iterrows():
            print(f"  {str(val):10s}  avg={row['mean']:.3f}  best={row['max']:.3f}  n={int(row['count'])}")

    t_elapsed = time.time() - t_total
    print(f"\n总耗时: {t_elapsed:.0f}s ({t_elapsed/60:.1f}min)")
    print("=" * 60)


if __name__ == '__main__':
    main()
