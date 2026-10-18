# -*- coding: utf-8 -*-
"""Regime gate 一期：概率软门 + 确认缓冲 + 分位数档位敏感性。

基于基准指数 (000985) 的日频数据计算「失效 regime 概率」，输出周频仓位权重。
严格无前视：每个调仓日只用该日及之前的指数数据。

设计 (见 PRE_REGISTRATION.md)：
  - 趋势分量: trend = close / MA(120) - 1
  - 波动分量: vol_ann = idx_return.rolling(20).std() * sqrt(252)
  - 失效分数: f_trend = sigmoid(-trend/s_trend), f_vol = sigmoid((vol_ann-vol_q)/s_vol)
  - f = f_trend * f_vol        (AND 语义: 趋势向下 且 高波动 才失效)
  - 软门权重: w_raw = 1 - f
  - 确认缓冲: f 连续 n_confirm 周 > 0.5 才生效降仓 (confirm), w = 1 - f * confirm
  - 阈值敏感性: vol_q 分位数三档 (0.70/0.75/0.80) 检验 cal26 稳定性
"""
import numpy as np
import pandas as pd


def _sigmoid(x):
    # clip 防 exp 溢出 (x 典型 ±20 内安全, 仍加护栏)
    x = np.clip(x, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-x))


def compute_regime_weights(index_df, rebalance_dates,
                           n_trend=120, n_vol=20, n_confirm=2,
                           s_trend=0.03, s_vol=0.05, vol_q=0.75):
    """返回 w_reb: Series(index=rebalance_dates[:-1])，每个持有期的仓位权重 ∈ [0,1]。

    index_df: 含 trade_date / close / idx_return 列 (S5 的 self._index_df)。
    rebalance_dates: 周频调仓日 (与 backtest_factor 完全一致的序列)。
    """
    idx = index_df[['trade_date', 'close', 'idx_return']].drop_duplicates('trade_date')
    idx = idx.sort_values('trade_date').set_index('trade_date')
    close = idx['close'].astype(float)
    ret = idx['idx_return'].astype(float)

    trend = close / close.rolling(n_trend).mean() - 1.0
    vol_ann = ret.rolling(n_vol).std() * np.sqrt(252.0)
    # 扩展窗口分位数 (无前视): 只用截至当日的历史
    vol_q_series = vol_ann.expanding(min_periods=max(n_vol, 40)).quantile(vol_q)

    f_trend = _sigmoid(-trend / s_trend)
    f_vol = _sigmoid((vol_ann - vol_q_series) / s_vol)
    # NaN (早期数据不足) → 视为不失效 (w=1 满仓), 保守不误伤
    f = (f_trend * f_vol).clip(0.0, 1.0).fillna(0.0)

    # 周频对齐 (rebalance 日取当日 f, 缺失 ffill)
    f_reb = f.reindex(rebalance_dates).ffill()
    # 确认缓冲: 连续 n_confirm 周 f > 0.5 才「生效」降仓 (滞后换准确)
    confirm = f_reb.rolling(n_confirm, min_periods=n_confirm).apply(
        lambda x: bool((x > 0.5).all()), raw=True).fillna(0.0)
    w_reb = (1.0 - f_reb * confirm).clip(0.0, 1.0)

    # 只保留持有期起始 (最后一个 rebalance 是 exit, 无对应持有期)
    return w_reb.iloc[:-1]


def apply_gate(port_ret, w_reb):
    """gated_ret = port_ret * 周频仓位权重 (ffill 到日频)。"""
    w_daily = w_reb.reindex(port_ret.index, method='ffill').fillna(1.0)
    return port_ret * w_daily


def summarize_gate(w_reb, rebalance_dates):
    """gate 行为摘要: 降仓周数 / 平均权重 / 各年度权重均值。"""
    if w_reb is None or len(w_reb) == 0:
        return {}
    idx = pd.DatetimeIndex(rebalance_dates[:-1])
    w = pd.Series(w_reb.values, index=idx)
    by_year = w.groupby(w.index.year).agg(['mean', 'min'])
    return {
        'n_periods': int(len(w)),
        'n_reduced': int((w < 0.5).sum()),
        'pct_reduced': float((w < 0.5).mean()),
        'mean_w': float(w.mean()),
        'min_w': float(w.min()),
        'by_year': {str(y): {'mean': round(float(r['mean']), 4),
                             'min': round(float(r['min']), 4)}
                    for y, r in by_year.iterrows()},
    }
