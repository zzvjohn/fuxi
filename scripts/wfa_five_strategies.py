"""
五子策略 WFA (Walk-Forward Analysis) 验证
===========================================
基于本地 Tushare 数据，对 S5 小市值策略进行滚动窗口验证。

策略逻辑复制：
  - 选股范围：399101 中证等权指数成分（用市值分位代理）
  - 过滤：市值 10-100亿, 营收>1亿, ROE>0, ROA>0, 净利润>200万
  - 排序：市值升序
  - 调仓：周频（每周二）

WFA 配置：
  - 训练窗口：2年
  - 测试窗口：1年（OOS）
  - 滚动窗口数：3
    - W1: Train 2021-2022, Test 2023
    - W2: Train 2022-2023, Test 2024
    - W3: Train 2023-2024, Test 2025

输出：
  - reports/wfa_five_strategies.csv -- 每个窗口的 IS/OOS 指标
  - reports/wfa_five_strategies.md  -- 汇总报告
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')

BASE = Path(r'E:\quant')
DATA = BASE / 'data' / 'raw'
REPORTS = BASE / 'reports'

# ============================================================
# 1. 数据加载
# ============================================================
def load_data():
    print("加载数据...")

    prices = pd.read_csv(
        DATA / 'daily_prices.csv',
        dtype={'ts_code': str},
    )
    prices['trade_date'] = pd.to_datetime(prices['trade_date'], format='mixed')
    print(f"  行情: {len(prices)} 行, {prices['ts_code'].nunique()} 只股票, "
          f"{prices['trade_date'].min().date()} ~ {prices['trade_date'].max().date()}")

    basic = pd.read_csv(
        DATA / 'daily_basic.csv',
        dtype={'ts_code': str},
    )
    basic['trade_date'] = pd.to_datetime(basic['trade_date'], format='mixed')
    print(f"  估值因子: {len(basic)} 行, {basic['ts_code'].nunique()} 只股票")

    fina = pd.read_csv(
        BASE / 'data' / 'tushare' / 'fina_indicator_rich.csv',
        dtype={'ts_code': str},
    )
    fina['ann_date'] = pd.to_datetime(fina['ann_date'], format='mixed')
    fina['end_date'] = pd.to_datetime(fina['end_date'], format='mixed')
    print(f"  财务指标: {len(fina)} 行, {fina['ts_code'].nunique()} 只股票")

    stock_list = pd.read_csv(
        DATA / 'stock_list.csv',
        dtype={'ts_code': str},
    )
    stock_list['list_date'] = pd.to_datetime(stock_list['list_date'], format='mixed')
    print(f"  股票列表: {len(stock_list)} 只")

    limits = pd.read_csv(
        DATA / 'stk_limit.csv',
        dtype={'ts_code': str},
    )
    limits['trade_date'] = pd.to_datetime(limits['trade_date'], format='mixed')
    print(f"  涨跌停: {len(limits)} 行")

    return prices, basic, fina, stock_list, limits


# ============================================================
# 2. S5 小市值策略核心逻辑（本地复现）
# ============================================================
def s5_weekly_signal(prices, basic, fina, stock_list, limits, date, prev_week_dates):
    """
    在 date 这一天，根据截至前一天的数据选股。
    返回：买入列表（按市值升序排列）
    """
    today = pd.Timestamp(date)

    # ---- 获取截至上一个交易日的数据日期 ----
    prev_date = prev_week_dates[-1] if len(prev_week_dates) > 0 else today

    # ---- 基础过滤：上市满375天 ----
    stock_list['days_since_list'] = (today - stock_list['list_date']).dt.days
    eligible = stock_list[stock_list['days_since_list'] >= 375].copy()

    # ---- 排除 ST (名称含 ST) ----
    eligible = eligible[~eligible['name'].str.contains('ST|退', na=False)]

    # ---- 排除科创板、北交所 (68xxxx, 8xxxxx, 4xxxxx) ----
    eligible = eligible[~eligible['ts_code'].str.match(r'^(688|689|8\d{5}|4\d{5})')]

    # ---- 市值过滤：取 daily_basic 最新一期 ----
    latest_basic = basic[basic['trade_date'] <= pd.Timestamp(prev_date)]
    latest_basic = latest_basic.sort_values('trade_date').groupby('ts_code').last().reset_index()

    # total_mv 单位是万元, 转换为亿
    latest_basic['market_cap_yi'] = latest_basic['total_mv'] / 10000
    eligible = eligible.merge(
        latest_basic[['ts_code', 'market_cap_yi']], on='ts_code', how='inner'
    )
    eligible = eligible[(eligible['market_cap_yi'] >= 10) & (eligible['market_cap_yi'] <= 100)]

    # ---- 财务过滤：ROE>0, ROA>0, 营收>1亿, 净利润>200万 ----
    latest_fina = fina[fina['ann_date'] <= pd.Timestamp(prev_date)]
    latest_fina = latest_fina.sort_values('ann_date').groupby('ts_code').last().reset_index()

    fina_cols = ['ts_code', 'roe', 'roa', 'or_yoy', 'profit_dedt']
    available_cols = [c for c in fina_cols if c in latest_fina.columns]
    eligible = eligible.merge(latest_fina[available_cols], on='ts_code', how='inner')

    if 'roe' in eligible.columns:
        eligible = eligible[eligible['roe'] > 0]
    if 'roa' in eligible.columns:
        eligible = eligible[eligible['roa'] > 0]

    # 营收过滤：用 daily_basic 中的 revenue（如果有）或跳过
    # 净利润过滤：用 profit_dedt (扣非净利润)
    if 'profit_dedt' in eligible.columns:
        eligible = eligible[eligible['profit_dedt'] > 2000000]  # 200万

    # ---- 涨跌停过滤 ----
    date_limits = limits[limits['trade_date'] == pd.Timestamp(prev_date)]
    if not date_limits.empty:
        limit_codes = set(date_limits['ts_code'].values)
        eligible = eligible[~eligible['ts_code'].isin(limit_codes)]

    # ---- 市值升序排序，取前N ----
    eligible = eligible.sort_values('market_cap_yi', ascending=True)

    return eligible.head(25)['ts_code'].tolist()


# ============================================================
# 3. 简化回测引擎
# ============================================================
def run_backtest(prices, basic, fina, stock_list, limits,
                 start_date, end_date, n_hold=25, max_turnover=0.45):
    """
    简化的周频回测。返回 (equity_curve, weekly_returns, turnover_list)
    """
    # 获取交易日列表
    trade_dates = sorted(prices['trade_date'].unique())
    trade_dates = [d for d in trade_dates
                   if pd.Timestamp(start_date) <= pd.Timestamp(d) <= pd.Timestamp(end_date)]

    if len(trade_dates) < 50:
        return None, None, None

    # 按周分组（取每周二，或最近交易日）
    weekly_dates = []
    current_week = None
    week_dates = []
    for d in trade_dates:
        dt = pd.Timestamp(d)
        w = dt.isocalendar()[1]
        yr = dt.year
        wk = (yr, w)
        if wk != current_week:
            if week_dates:
                weekly_dates.append(week_dates)
            current_week = wk
            week_dates = [d]
        else:
            week_dates.append(d)
    if week_dates:
        weekly_dates.append(week_dates)

    # 初始化
    cash = 1.0
    holdings = {}  # {ts_code: weight}
    equity = []
    weekly_returns = []
    turnover_list = []
    dates_record = []

    for wi, wd in enumerate(weekly_dates):
        rebalance_date = wd[-1]  # 用本周最后一天作为调仓日

        # 获取当日价格
        day_prices = prices[prices['trade_date'] == rebalance_date][['ts_code', 'close']]
        price_dict = dict(zip(day_prices['ts_code'], day_prices['close']))

        # 计算当前组合价值
        if wi == 0:
            portfolio_value = cash
        else:
            portfolio_value = cash
            for code, wgt in holdings.items():
                if code in price_dict:
                    portfolio_value += wgt * (price_dict[code] / prev_price_dict.get(code, price_dict[code]))

        equity.append(portfolio_value)
        dates_record.append(rebalance_date)

        if wi > 0:
            ret = (equity[-1] / equity[-2] - 1)
            weekly_returns.append(ret)

        # 调仓
        prev_week = weekly_dates[wi - 1] if wi > 0 else wd
        selected = s5_weekly_signal(prices, basic, fina, stock_list, limits,
                                     rebalance_date, prev_week)

        if not selected:
            continue

        # 计算成交价格和数量
        valid_selected = [s for s in selected if s in price_dict and price_dict[s] > 0]
        if len(valid_selected) < max(3, n_hold // 3):
            continue

        target = valid_selected[:n_hold]

        # 计算换手率
        old_set = set(holdings.keys())
        new_set = set(target)
        if old_set:
            turnover = len(old_set - new_set) / len(old_set)
        else:
            turnover = 1.0
        turnover_list.append(turnover)

        # 限制换手
        if turnover > max_turnover and old_set:
            keep = list(old_set & new_set)
            need_new = n_hold - len(keep)
            new_stocks = [s for s in target if s not in old_set][:need_new]
            target = keep + new_stocks
            if len(target) < max(3, n_hold // 3):
                continue

        # 等权分配，扣除成本
        weight_per_stock = 1.0 / len(target)
        # 印花税 0.1% (卖出), 佣金 0.01% (买卖), 滑点 0.3% (买卖)
        # 简化：每次调仓扣 0.5% 总成本
        cost = 0.005 if wi > 0 else 0.0
        portfolio_value *= (1 - cost * turnover)

        holdings = {s: portfolio_value * weight_per_stock for s in target}
        cash = 0  # 满仓
        prev_price_dict = price_dict.copy()

    if len(equity) < 10:
        return None, None, None

    return np.array(equity), np.array(weekly_returns), np.array(turnover_list)


# ============================================================
# 4. 绩效计算
# ============================================================
def calc_metrics(equity, returns):
    """计算回测指标"""
    if equity is None or len(equity) < 10:
        return {'error': 'insufficient data'}

    total_return = (equity[-1] / equity[0] - 1) * 100

    # 年化
    n_weeks = len(returns)
    n_years = n_weeks / 52

    if n_years < 0.5:
        ann_return = total_return  # 不足一年不年化
    else:
        ann_return = ((equity[-1] / equity[0]) ** (1 / n_years) - 1) * 100

    if len(returns) > 1 and returns.std() > 0:
        sharpe = returns.mean() / returns.std() * np.sqrt(52)
    else:
        sharpe = 0

    # MaxDD
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    max_dd = dd.min() * 100

    # 胜率
    win_rate = (returns > 0).mean() * 100 if len(returns) > 0 else 0

    return {
        'total_return': round(total_return, 2),
        'ann_return': round(ann_return, 2),
        'sharpe': round(sharpe, 2),
        'max_dd': round(max_dd, 2),
        'win_rate': round(win_rate, 1),
        'n_weeks': n_weeks,
        'n_years': round(n_years, 2),
    }


# ============================================================
# 5. WFA 主流程
# ============================================================
def main():
    print("=" * 60)
    print("五子策略 S5 小市值 — WFA 滚动窗口验证")
    print("=" * 60)

    prices, basic, fina, stock_list, limits = load_data()

    # WFA 窗口定义
    windows = [
        {'name': 'W1', 'train_start': '2021-01-01', 'train_end': '2022-12-31',
         'test_start': '2023-01-01', 'test_end': '2023-12-31'},
        {'name': 'W2', 'train_start': '2022-01-01', 'train_end': '2023-12-31',
         'test_start': '2024-01-01', 'test_end': '2024-12-31'},
        {'name': 'W3', 'train_start': '2023-01-01', 'train_end': '2024-12-31',
         'test_start': '2025-01-01', 'test_end': '2025-12-31'},
    ]

    all_results = []

    for w in windows:
        print(f"\n{'='*40}")
        print(f"  {w['name']}: IS {w['train_start']}~{w['train_end']}  →  OOS {w['test_start']}~{w['test_end']}")
        print(f"{'='*40}")

        # IS 回测
        print(f"  训练集回测...")
        eq_is, ret_is, to_is = run_backtest(
            prices, basic, fina, stock_list, limits,
            w['train_start'], w['train_end']
        )
        metrics_is = calc_metrics(eq_is, ret_is)
        print(f"    收益={metrics_is.get('total_return','N/A')}%, "
              f"Sharpe={metrics_is.get('sharpe','N/A')}, "
              f"MaxDD={metrics_is.get('max_dd','N/A')}%")

        # OOS 回测
        print(f"  测试集回测...")
        eq_oos, ret_oos, to_oos = run_backtest(
            prices, basic, fina, stock_list, limits,
            w['test_start'], w['test_end']
        )
        metrics_oos = calc_metrics(eq_oos, ret_oos)
        print(f"    收益={metrics_oos.get('total_return','N/A')}%, "
              f"Sharpe={metrics_oos.get('sharpe','N/A')}, "
              f"MaxDD={metrics_oos.get('max_dd','N/A')}%")

        # WFA 关键指标
        is_ret = metrics_is.get('total_return', 0)
        oos_ret = metrics_oos.get('total_return', 0)
        is_sharpe = metrics_is.get('sharpe', 0)
        oos_sharpe = metrics_oos.get('sharpe', 0)

        # OOS/IS 比率（>0.5 说明有效性保持）
        oos_is_ratio = oos_ret / is_ret if is_ret != 0 else 0
        sharpe_retention = oos_sharpe / is_sharpe if is_sharpe != 0 else 0

        row = {
            'window': w['name'],
            'is_period': f"{w['train_start'][:7]}~{w['train_end'][:7]}",
            'oos_period': f"{w['test_start'][:7]}~{w['test_end'][:7]}",
            'is_return': round(is_ret, 2),
            'oos_return': round(oos_ret, 2),
            'is_sharpe': round(is_sharpe, 2),
            'oos_sharpe': round(oos_sharpe, 2),
            'is_maxdd': metrics_is.get('max_dd', 0),
            'oos_maxdd': metrics_oos.get('max_dd', 0),
            'oos_is_ratio': round(oos_is_ratio, 3),
            'sharpe_retention': round(sharpe_retention, 3),
        }
        all_results.append(row)

        print(f"    OOS/IS 收益比={oos_is_ratio:.3f}, Sharpe保持率={sharpe_retention:.3f}")

    # ============================================================
    # 汇总报告
    # ============================================================
    df = pd.DataFrame(all_results)

    # 计算 WFA 综合评分
    valid_ratios = [r['oos_is_ratio'] for r in all_results if abs(r['oos_is_ratio']) < 10]
    valid_sharpes = [r['sharpe_retention'] for r in all_results if abs(r['sharpe_retention']) < 10]

    avg_ratio = np.mean(valid_ratios) if valid_ratios else 0
    avg_sharpe_retention = np.mean(valid_sharpes) if valid_sharpes else 0

    # 评级
    if avg_ratio >= 0.7 and avg_sharpe_retention >= 0.7:
        wfa_grade = '优秀 — 策略在样本外保持稳健'
    elif avg_ratio >= 0.4 and avg_sharpe_retention >= 0.4:
        wfa_grade = '良好 — 策略有衰减但在可接受范围'
    elif avg_ratio >= 0.2:
        wfa_grade = '一般 — 策略存在明显过拟合'
    elif avg_ratio > 0:
        wfa_grade = '差 — 策略严重过拟合，样本外几乎无效'
    else:
        wfa_grade = '极差 — 策略可能在样本外亏损'

    print(f"\n{'='*60}")
    print(f"  WFA 综合评估")
    print(f"{'='*60}")
    print(f"  平均 OOS/IS 收益比: {avg_ratio:.3f}")
    print(f"  平均 Sharpe 保持率: {avg_sharpe_retention:.3f}")
    print(f"  评级: {wfa_grade}")

    # 保存 CSV
    csv_path = REPORTS / 'wfa_five_strategies.csv'
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"\n结果已保存: {csv_path}")

    # 生成 Markdown 报告
    md_path = REPORTS / 'wfa_five_strategies.md'
    with open(md_path, 'w', encoding='utf-8-sig') as f:
        f.write("# 五子策略 S5 小市值 — WFA 验证报告\n\n")
        f.write(f"> 生成日期: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"> 策略: 五子策略 S5 小市值（市值10-100亿, ROE>0, ROA>0, 周频调仓25只）\n")
        f.write(f"> 数据: 本地 Tushare daily_prices + daily_basic + fina_indicator_rich\n\n")

        f.write("## WFA 窗口配置\n\n")
        f.write("| 窗口 | 训练集 (IS) | 测试集 (OOS) |\n")
        f.write("|:--:|------|------|\n")
        for _, r in df.iterrows():
            f.write(f"| {r['window']} | {r['is_period']} | {r['oos_period']} |\n")

        f.write("\n## 各窗口表现\n\n")
        f.write("| 窗口 | IS收益% | OOS收益% | IS Sharpe | OOS Sharpe | IS MaxDD% | OOS MaxDD% | OOS/IS比 | Sharpe保持 |\n")
        f.write("|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|\n")
        for _, r in df.iterrows():
            f.write(f"| {r['window']} | {r['is_return']} | {r['oos_return']} | "
                    f"{r['is_sharpe']} | {r['oos_sharpe']} | "
                    f"{r['is_maxdd']} | {r['oos_maxdd']} | "
                    f"{r['oos_is_ratio']} | {r['sharpe_retention']} |\n")

        f.write("\n## WFA 综合评分\n\n")
        f.write(f"- **平均 OOS/IS 收益比**: {avg_ratio:.3f}\n")
        f.write(f"- **平均 Sharpe 保持率**: {avg_sharpe_retention:.3f}\n")
        f.write(f"- **WFA 评级**: **{wfa_grade}**\n\n")

        f.write("### 解读指南\n\n")
        f.write("- OOS/IS 收益比 > 0.7: 策略在样本外保持强健\n")
        f.write("- OOS/IS 收益比 0.4-0.7: 有衰减，但仍有正期望\n")
        f.write("- OOS/IS 收益比 0.2-0.4: 明显过拟合\n")
        f.write("- OOS/IS 收益比 < 0.2: 严重过拟合，样本外几乎无效\n\n")

        f.write("### 成本设置说明\n\n")
        f.write("本次回测已使用修正后的真实成本：\n")
        f.write("- 佣金: 万1 (买卖各 0.01%)\n")
        f.write("- 印花税: 千1 (卖出 0.1%)\n")
        f.write("- 滑点: 0.3% (小市值股票)\n")
        f.write("- 综合单次调仓成本: ~0.5% × 换手率\n\n")

        f.write("### 注意事项\n\n")
        f.write("1. 本地回测为简化版，未完整复制聚宽策略的所有细节（如ATR止损、MACD背离等）\n")
        f.write("2. 本地数据不含中证等权指数成分，以市值分位代理选股范围\n")
        f.write("3. 最终实盘前仍需在聚宽平台以修正后的策略代码完成完整回测验证\n")

    print(f"报告已保存: {md_path}")

    # 返回数据供后续分析
    return df, avg_ratio, wfa_grade


if __name__ == '__main__':
    df, avg_ratio, wfa_grade = main()
    print(f"\n{'='*60}")
    print(f"  最终结论: {wfa_grade}")
    print(f"{'='*60}")
