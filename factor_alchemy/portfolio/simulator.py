"""
虚拟投资组合模拟器 (v3 — JQ对齐版)
===================================
周频调仓, 等权Top30, QMT标准成本

v3 修复 (2026-07-20):
  - BugFix: turnover过滤后重算added/removed → 成本对齐实际持仓 (v2在换手超限时用旧的added/removed)
  - Slippage模型: 改为per-share绝对金额 (JQ FixedSlippage(0.003元/股)对齐)
  - 平均股价估算: 用于per-share → 百分比成本换算
  - 保留旧版slippage_pct参数兼容

v2 修复 (2026-07-16):
  - 存活偏差修复: factor_df 对齐 price_df 有效股票
  - 成本模型: 双边QMT对齐
  - 市值过滤: mcap_df + mcap_min
"""
import numpy as np
import pandas as pd


class PortfolioSimulator:
    """周频调仓虚拟投资组合 (v3 对齐JQ)"""

    def __init__(self, factor_df, price_df, top_n=30, weighting='equal',
                 max_turnover=0.50, commission=0.0001, stamp_tax=0.001,
                 slippage=0.003, min_commission=5, debug=False,
                 mcap_df=None, mcap_min=None,
                 slippage_mode='per_share'):
        """
        Parameters
        ----------
        factor_df : pd.DataFrame
            综合因子得分, index=date, columns=stocks
        price_df : pd.DataFrame
            价格数据 (hfq), index=date, columns=stocks
        top_n : int
            持仓数
        weighting : str
            'equal' 或 'factor' (因子分加权)
        max_turnover : float
            最大单边换手
        commission : float
            佣金率 (默认万1)
        stamp_tax : float
            印花税率 (仅卖出, 默认千1)
        slippage : float
            slippage_mode='per_share'时: 元/股 (JQ FixedSlippage默认0.003)
            slippage_mode='percent'时: 百分比 (旧版兼容, 默认0.3%)
        min_commission : float
            最低佣金 (默认5元)
        debug : bool
            是否打印每期诊断
        mcap_df : pd.DataFrame, optional
            市值数据, index=date, columns=stocks (单位须与mcap_min一致)
        mcap_min : float, optional
            最低市值阈值 (单位须与mcap_df一致)
        slippage_mode : str
            'per_share' (默认, JQ对齐) 或 'percent' (旧版兼容)
        """
        self.factor_df = factor_df
        self.price_df = price_df
        self.top_n = top_n
        self.weighting = weighting
        self.max_turnover = max_turnover
        self.commission = commission
        self.stamp_tax = stamp_tax
        self.slippage = slippage
        self.min_commission = min_commission
        self.debug = debug
        self.mcap_df = mcap_df
        self.mcap_min = mcap_min
        self.slippage_mode = slippage_mode

    def _slippage_pct(self, price):
        """per-share slippage → 百分比"""
        if self.slippage_mode == 'per_share':
            return self.slippage / max(price, 0.01)
        else:
            return self.slippage  # 旧版百分比模式

    def run(self):
        """
        运行模拟

        Returns
        -------
        dict with keys: nav, returns, turnover, positions, stats
        """
        dates = sorted(self.factor_df.index)

        if len(dates) < 2:
            return {'stats': {'error': '日期不足'}}

        # ★ v2: 预过滤 — 仅保留 price_df 中有有效价格的股票 (消除ffill存活偏差)
        valid_mask = self.price_df.notna() & (self.price_df > 0)
        # 对每个调仓日, 只用当天有有效价格的股票
        aligned_factor = self.factor_df.where(self.price_df.notna() & (self.price_df > 0))

        # ★ v3 (2026-07-20): 市值过滤 — 对齐JQ mcap_min限制
        if self.mcap_df is not None and self.mcap_min is not None:
            mcap_aligned = self.mcap_df.reindex(index=self.factor_df.index,
                                                columns=self.factor_df.columns)
            aligned_factor = aligned_factor.where(mcap_aligned >= self.mcap_min)

        nav = pd.Series(1.0, index=[dates[0]])
        weekly_returns = pd.Series(dtype=float)
        turnover_series = pd.Series(dtype=float)
        positions_history = {}

        current_holdings = set()

        for i in range(len(dates) - 1):
            rebalance_date = dates[i]
            forward_date = dates[i + 1]

            # 选股 — 用对齐后的因子值
            signal = aligned_factor.loc[rebalance_date].dropna()
            if len(signal) < self.top_n:
                if self.debug and len(signal) > 0:
                    print(f"[{rebalance_date}] 有效股票={len(signal)} < top_n={self.top_n}, 跳过")
                continue

            top_stocks = signal.nlargest(self.top_n).index.tolist()

            # ★ v2: 二次过滤 — 调仓日和持有期价格都必须有效
            if rebalance_date in self.price_df.index and forward_date in self.price_df.index:
                prices_reb = self.price_df.loc[rebalance_date]
                prices_fwd = self.price_df.loc[forward_date]
                valid_top = [s for s in top_stocks
                           if s in prices_reb.index and s in prices_fwd.index
                           and not np.isnan(prices_reb[s]) and not np.isnan(prices_fwd[s])
                           and prices_reb[s] > 0 and prices_fwd[s] > 0]
                if len(valid_top) < max(3, self.top_n // 3):
                    if self.debug:
                        print(f"[{rebalance_date}] valid_top={len(valid_top)} < {max(3, self.top_n//3)}, 跳过")
                    continue
                top_stocks = valid_top[:self.top_n]

            # ---- v3: 换手计算 + 过滤 + 重算 added/removed ----
            new_holdings_raw = set(top_stocks[:self.top_n])

            if len(current_holdings) > 0:
                added_raw = new_holdings_raw - current_holdings
                removed_raw = current_holdings - new_holdings_raw
                turnover_raw = (len(added_raw) + len(removed_raw)) / len(current_holdings)

                if turnover_raw > self.max_turnover:
                    # 换手限制: 保留信号最强的旧持仓
                    keep_from_current = sorted(
                        current_holdings & set(signal.index),
                        key=lambda s: signal.get(s, -np.inf), reverse=True)
                    max_new = int(self.top_n * self.max_turnover)
                    keep = set(keep_from_current[:self.top_n - max_new])
                    new_from_signal = [
                        s for s in top_stocks if s not in keep][:max_new]
                    new_holdings = keep | set(new_from_signal)
                else:
                    new_holdings = new_holdings_raw
            else:
                new_holdings = new_holdings_raw
                turnover_raw = 1.0

            # ★★★ v3 BugFix: 基于 FILTERED new_holdings 重算 added/removed ★★★
            added = new_holdings - current_holdings
            removed = current_holdings - new_holdings
            turnover = (len(added) + len(removed)) / max(1, len(current_holdings)) \
                       if len(current_holdings) > 0 else 1.0

            # ---- 收益计算 ----
            if rebalance_date in self.price_df.index and forward_date in self.price_df.index:
                prices_reb = self.price_df.loc[rebalance_date]
                prices_fwd = self.price_df.loc[forward_date]

                valid_stocks = [s for s in new_holdings
                               if s in prices_reb.index and s in prices_fwd.index]

                if len(valid_stocks) == 0:
                    if self.debug:
                        print(f"[{rebalance_date}] 无可交易持仓, 跳过")
                    continue

                weight = 1.0 / self.top_n
                port_ret = 0.0

                # ★ v3: 记录平均股价用于 per-share slippage 换算
                avg_price = 0.0
                n_priced = 0
                for s in valid_stocks:
                    stock_ret = prices_fwd[s] / prices_reb[s] - 1
                    port_ret += weight * stock_ret
                    avg_price += prices_reb[s]
                    n_priced += 1
                avg_price = avg_price / n_priced if n_priced > 0 else 1.0

                # ★ v3: per-share slippage → 百分比 (或旧版百分比模式)
                slip_pct = self._slippage_pct(avg_price)

                n_added = len(added & set(valid_stocks))
                n_removed = len(removed & set(valid_stocks))

                buy_cost = n_added * weight * (self.commission + slip_pct)
                sell_cost = n_removed * weight * (self.commission + self.stamp_tax + slip_pct)
                total_cost = buy_cost + sell_cost

                if self.debug:
                    print(f"[{rebalance_date.date()}] hold={len(current_holdings)}→{len(new_holdings)} "
                          f"turn={turnover:.0%} n_add={n_added} n_rem={n_removed} "
                          f"avg_px={avg_price:.1f} slip={slip_pct:.4%} "
                          f"cost={total_cost:.4%} gross_ret={port_ret:.4%}")

                port_ret -= total_cost

                # 更新净值
                if not np.isnan(port_ret):
                    new_nav = nav.iloc[-1] * (1 + port_ret)
                    nav.loc[forward_date] = new_nav
                    weekly_returns.loc[forward_date] = port_ret
                    turnover_series.loc[forward_date] = turnover

                current_holdings = set(valid_stocks)
                positions_history[forward_date] = list(valid_stocks)

        # 统计指标
        stats = self._compute_stats(nav, weekly_returns, turnover_series)

        return {
            'nav': nav,
            'returns': weekly_returns,
            'turnover': turnover_series,
            'positions': positions_history,
            'stats': stats,
        }

    def _compute_stats(self, nav, returns, turnover):
        """计算统计指标"""
        if len(returns) < 10:
            return {}

        valid_nav = nav.dropna()
        if len(valid_nav) < 2:
            return {'total_return': np.nan, 'cagr': np.nan, 'sharpe': np.nan,
                    'max_drawdown': np.nan, 'win_rate': np.nan, 'avg_turnover': np.nan}

        total_return = valid_nav.iloc[-1] / valid_nav.iloc[0] - 1

        n_weeks = len(returns)
        years = n_weeks / 52

        cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

        if returns.std() > 0:
            weekly_sharpe = returns.mean() / returns.std()
            sharpe = weekly_sharpe * np.sqrt(52)
        else:
            sharpe = 0

        cum = nav / nav.iloc[0]
        running_max = cum.cummax()
        drawdown = cum / running_max - 1
        max_dd = drawdown.min()

        calmar = cagr / abs(max_dd) if max_dd < 0 else 0

        win_rate = (returns > 0).mean()
        avg_turnover = turnover.mean() if len(turnover) > 0 else 0

        return {
            'total_return': total_return,
            'cagr': cagr,
            'sharpe': sharpe,
            'max_drawdown': max_dd,
            'calmar': calmar,
            'win_rate': win_rate,
            'avg_turnover': avg_turnover,
            'n_weeks': n_weeks,
            'years': years,
        }


# ============================================================
# 兼容性包装: 旧调用方式依然可用
# ============================================================

def run_portfolio_simulation(factor_df, price_df, top_n=30, weighting='equal',
                              max_turnover=0.50, commission=0.0001, stamp_tax=0.001,
                              slippage=0.003, debug=False,
                              mcap_df=None, mcap_min=None,
                              slippage_mode='per_share'):
    """便捷函数: 运行一次组合模拟并返回统计指标 (兼容旧API)

    注意: slippage_mode='per_share' 时 slippage 为元/股 (JQ FixedSlippage 默认0.003);
           slippage_mode='percent' 时 slippage 为百分比 (旧版默认0.3%).
    """
    sim = PortfolioSimulator(
        factor_df=factor_df, price_df=price_df,
        top_n=top_n, weighting=weighting,
        max_turnover=max_turnover, commission=commission,
        stamp_tax=stamp_tax, slippage=slippage,
        min_commission=5, debug=debug,
        mcap_df=mcap_df, mcap_min=mcap_min,
        slippage_mode=slippage_mode,
    )
    result = sim.run()
    return result['stats']
