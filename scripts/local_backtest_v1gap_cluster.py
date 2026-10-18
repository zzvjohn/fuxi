# -*- coding: utf-8 -*-
"""
本地回测：v1_gap (7因子月频) vs 聚类增强反转 (周频)
使用 backtest_engine.py 的 DataLoader + BacktestEngine
"""
import sys, os
sys.path.insert(0, r'e:\quant\strategies\final')
from backtest_engine import DataLoader, BacktestEngine
import pandas as pd
import numpy as np

RAW_DIR = r'e:\quant\data\raw'

# ============================================================
# 策略一：v1_gap — 7因子月频复合 (Top30等权)
# ============================================================
def select_v1_gap(date, loader):
    """
    7因子月频选股：
    - 反转2: mom_1m (21日), mom_3m (63日) — 取负（反转）
    - 低波: vol_1m (21日波动率倒数)
    - 估值2: pb, ps — 取负（低估值）
    - 换手: turnover (20日均换手)
    - 隔夜反转: gap_rev (今日开盘/昨日收盘 - 1)
    等权合成 → Top30
    """
    daily = loader.load_daily()
    basic = loader.load_daily_basic()
    
    if len(daily) == 0 or len(basic) == 0:
        return []
    
    daily['trade_date'] = pd.to_datetime(daily['trade_date'], format='mixed')
    basic['trade_date'] = pd.to_datetime(basic['trade_date'], format='mixed')
    
    # 取当前日期数据
    day_p = daily[daily['trade_date'] == date].copy()
    
    if len(day_p) < 100:
        return []
    
    # 过滤：有效价格、非ST(简单用成交额)
    day_p = day_p[(day_p['close'] > 0) & (day_p['amount'] > 0)]
    
    # 合并估值数据
    day_b = basic[basic['trade_date'] == date][['ts_code', 'pb', 'ps', 'total_mv', 'turnover_rate']].copy()
    df = day_p.merge(day_b, on='ts_code', how='inner')
    
    if len(df) < 100:
        return []
    
    # 计算前几个月动量（需要历史数据）
    # 回看3个月的数据
    lookback_dates = sorted(daily['trade_date'].unique())
    idx_3m = None
    for i in range(len(lookback_dates) - 1, -1, -1):
        if lookback_dates[i] <= date:
            idx_now = i
            break
    
    # 1月前 (约21交易日)
    idx_1m = max(0, idx_now - 21)
    date_1m = lookback_dates[idx_1m]
    p_1m = daily[daily['trade_date'] == date_1m][['ts_code', 'close']].rename(columns={'close': 'close_1m'})
    
    # 3月前 (约63交易日)
    idx_3m = max(0, idx_now - 63)
    date_3m = lookback_dates[idx_3m]
    p_3m = daily[daily['trade_date'] == date_3m][['ts_code', 'close']].rename(columns={'close': 'close_3m'})
    
    df = df.merge(p_1m, on='ts_code', how='left')
    df = df.merge(p_3m, on='ts_code', how='left')
    
    # --- 因子1: 反转 mom_1m (取负) ---
    df['mom_1m'] = df['close'] / df['close_1m'] - 1
    df['f_rev_1m'] = -df['mom_1m']
    
    # --- 因子2: 反转 mom_3m (取负) ---
    df['mom_3m'] = df['close'] / df['close_3m'] - 1
    df['f_rev_3m'] = -df['mom_3m']
    
    # --- 因子3: 低波 vol_1m ---
    # 计算21日波动率
    vol_data = daily[(daily['trade_date'] >= date_1m) & (daily['trade_date'] <= date)]
    vol_std = vol_data.groupby('ts_code')['pct_chg'].std().reset_index()
    vol_std.columns = ['ts_code', 'vol_1m']
    df = df.merge(vol_std, on='ts_code', how='left')
    df['f_lowvol'] = -df['vol_1m']  # 低波=负波动率
    
    # --- 因子4-5: 估值 pb, ps (取负) ---
    df['f_pb'] = -df['pb']
    df['f_ps'] = -df['ps']
    
    # --- 因子6: 换手率 ---
    df['f_turnover'] = -df['turnover_rate']  # 反转：低换手更优
    
    # --- 因子7: 隔夜反转 gap_rev ---
    df['gap_rev'] = df['open'] / df['pre_close'] - 1
    df['f_gap'] = -df['gap_rev']  # 低开→反弹
    
    # 标准化各因子后等权合成
    factor_cols = ['f_rev_1m', 'f_rev_3m', 'f_lowvol', 'f_pb', 'f_ps', 'f_turnover', 'f_gap']
    for c in factor_cols:
        df[c] = df[c].fillna(df[c].median())
        # 去极端值
        q1, q99 = df[c].quantile(0.01), df[c].quantile(0.99)
        df[c] = df[c].clip(q1, q99)
        # 标准化
        std = df[c].std()
        if std > 0:
            df[c] = (df[c] - df[c].mean()) / std
    
    # 等权合成
    df['score'] = df[factor_cols].mean(axis=1)
    df = df.dropna(subset=['score'])
    
    top30 = df.nlargest(30, 'score')
    return top30['ts_code'].tolist()


# ============================================================
# 策略二：聚类增强反转 — 行业动量 + 行业内分桶反转 (周频Top15)
# ============================================================
def select_cluster_reversal(date, loader):
    """
    聚类增强反转：
    1. 行业动量(75日)选Top20%行业
    2. 强势行业内计算10日反转信号
    3. 行业内按技术特征分6个bucket（模拟K-means聚类）
    4. 每bucket独立反转排名
    5. 汇总选Top15
    """
    daily = loader.load_daily()
    stocks_df = loader.load_stock_list()
    
    if len(daily) == 0:
        return []
    
    daily['trade_date'] = pd.to_datetime(daily['trade_date'], format='mixed')
    
    # 股票池：去科创板
    pool = stocks_df[~stocks_df['ts_code'].str.startswith(('688', '689'))]
    pool_codes = pool['ts_code'].tolist()
    
    # 当前日期数据
    all_dates = sorted(daily['trade_date'].unique())
    
    # 找当前日期索引
    idx_now = None
    for i, d in enumerate(all_dates):
        if d == date:
            idx_now = i
            break
    if idx_now is None:
        return []
    
    # --- Step 1: 行业动量 (75日) ---
    idx_75d = max(0, idx_now - 75)
    date_75d = all_dates[idx_75d]
    
    p_now = daily[daily['trade_date'] == date][['ts_code', 'close']].rename(columns={'close': 'close_now'})
    p_75d = daily[daily['trade_date'] == date_75d][['ts_code', 'close']].rename(columns={'close': 'close_75d'})
    
    mom_df = p_now.merge(p_75d, on='ts_code', how='inner')
    mom_df = mom_df[mom_df['close_75d'] > 0]
    mom_df['stock_return'] = mom_df['close_now'] / mom_df['close_75d'] - 1
    
    # 合并行业
    mom_df = mom_df.merge(stocks_df[['ts_code', 'sw_ind_name']], on='ts_code', how='left')
    mom_df = mom_df.dropna(subset=['sw_ind_name'])
    
    # 行业平均动量
    ind_mom = mom_df.groupby('sw_ind_name')['stock_return'].median()
    top_n = max(1, int(len(ind_mom) * 0.20))
    top_industries = ind_mom.nlargest(top_n).index.tolist()
    
    # --- Step 2: 强势行业内反转信号 ---
    idx_10d = max(0, idx_now - 10)
    date_10d = all_dates[idx_10d]
    p_10d = daily[daily['trade_date'] == date_10d][['ts_code', 'close']].rename(columns={'close': 'close_10d'})
    
    rev_df = p_now.merge(p_10d, on='ts_code', how='inner')
    rev_df = rev_df[rev_df['close_10d'] > 0]
    rev_df['reversal'] = -(rev_df['close_now'] / rev_df['close_10d'] - 1)  # 反转
    
    # 只保留强势行业中的股票
    rev_df = rev_df.merge(stocks_df[['ts_code', 'sw_ind_name']], on='ts_code', how='left')
    rev_df = rev_df[rev_df['sw_ind_name'].isin(top_industries)]
    
    if len(rev_df) < 15:
        # 行业不够，直接取全市场反转Top15
        rev_df = rev_df.dropna(subset=['reversal'])
        return rev_df.nlargest(15, 'reversal')['ts_code'].tolist()
    
    # --- Step 3: 行业内分桶（模拟聚类） ---
    # 计算技术特征
    idx_21d = max(0, idx_now - 21)
    date_21d = all_dates[idx_21d]
    idx_5d = max(0, idx_now - 5)
    date_5d = all_dates[idx_5d]
    
    p_21d = daily[daily['trade_date'] == date_21d][['ts_code', 'close']].rename(columns={'close': 'close_21d'})
    p_5d = daily[daily['trade_date'] == date_5d][['ts_code', 'close']].rename(columns={'close': 'close_5d'})
    
    feat = rev_df.merge(p_21d, on='ts_code', how='left')
    feat = feat.merge(p_5d, on='ts_code', how='left')
    feat['mom_5d'] = feat['close_now'] / feat['close_5d'] - 1
    feat['mom_21d'] = feat['close_now'] / feat['close_21d'] - 1
    
    # 波动率 (简化)
    vol_data = daily[(daily['trade_date'] >= date_21d) & (daily['trade_date'] <= date)]
    vol_std = vol_data.groupby('ts_code')['pct_chg'].std().reset_index()
    vol_std.columns = ['ts_code', 'vol_21d']
    feat = feat.merge(vol_std, on='ts_code', how='left')
    feat['vol_21d'] = feat['vol_21d'].fillna(0)
    
    # 综合特征评分（等权）
    feat_cols = ['mom_5d', 'mom_21d', 'vol_21d', 'reversal']
    for c in feat_cols:
        feat[c] = feat[c].fillna(0).clip(feat[c].quantile(0.01), feat[c].quantile(0.99))
    
    # 分桶（不依赖sklearn，用分位数）
    feat['cluster_score'] = feat[feat_cols].mean(axis=1)
    
    candidates = []
    for ind in top_industries:
        ind_feat = feat[feat['sw_ind_name'] == ind].copy()
        if len(ind_feat) < 3:
            continue
        
        # 行业内按cluster_score分4~6个桶
        n_buckets = min(6, len(ind_feat) // 3)
        if n_buckets < 2:
            n_buckets = 2
        
        try:
            ind_feat['bucket'] = pd.qcut(ind_feat['cluster_score'], n_buckets, labels=False, duplicates='drop')
        except Exception:
            ind_feat['bucket'] = 0
        
        # 每桶取反转最强的
        for b in ind_feat['bucket'].unique():
            bucket_stocks = ind_feat[ind_feat['bucket'] == b]
            best = bucket_stocks.nlargest(2, 'reversal')
            candidates.extend(best['ts_code'].tolist())
    
    if not candidates:
        return rev_df.nlargest(15, 'reversal')['ts_code'].tolist()
    
    # 去重后取Top15
    candidates = list(dict.fromkeys(candidates))  # 保持顺序去重
    return candidates[:15]


# ============================================================
# 运行回测
# ============================================================
if __name__ == '__main__':
    import time
    t0 = time.time()
    
    loader = DataLoader(clean_price_limits=True)
    
    start = '2021-06-01'
    end = '2026-04-20'
    
    strategies = [
        ('v1_gap 7因子月频', select_v1_gap, 'M', 30),
        ('聚类增强反转 周频', select_cluster_reversal, 'W', 15),
    ]
    
    all_results = {}
    
    for name, func, freq, num in strategies:
        print(f"\n{'#'*60}")
        print(f"# 策略: {name}")
        print(f"{'#'*60}")
        
        engine = BacktestEngine(loader, initial_capital=1000000)
        try:
            results = engine.run(
                select_func=func,
                start_date=start,
                end_date=end,
                rebalance_freq=freq,
                stock_num=num,
                trailing_stop=-0.15,
            )
            engine.print_report()
            all_results[name] = results
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
    
    elapsed = time.time() - t0
    print(f"\n总耗时: {elapsed:.1f}s")
