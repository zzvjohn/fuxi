"""
每周因子假设实验 — 基于 XQuant Ch9 方法论
=========================================
核心流程: 观察市场 → 提出因子假设 → IC验证 → 纳入/推翻
设计理念: "节奏比"努力"重要 — 一周一次因子实验, 坏假设尽早推翻, 好假设尽快上线"

来源: 《XQuant: 人人都是量化交易员》第9章 "持续寻找盈利机会"
创建: 2026-06-14
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 配置
# ============================================================
REPORT_DIR = Path(__file__).parent.parent / 'reports'
REPORT_DIR.mkdir(exist_ok=True)

# 预设因子假设模板 (每周选一个方向, 也可自定义)
PRESET_HYPOTHESES = [
    {
        'name': 'reversal_5d',
        'label': '5日反转',
        'hypothesis': '过去一周跌最多的股票下周会反弹',
        'logic': '行为金融: 过度反应后的均值回归',
        'formula': '-close_p.pct_change(5)',
        'direction': 'long',  # 做多 = 因子值大 → 高收益
        'icir_threshold': 0.3,
        'ic_pct_threshold': 0.55,
    },
    {
        'name': 'volume_price_divergence',
        'label': '量价背离',
        'hypothesis': '价格上涨但成交量萎缩的股票后续走弱',
        'logic': '技术分析: 量价背离预示趋势衰竭',
        'formula': '-(close_p.pct_change(10) / (volume_p.rolling(10).mean() / volume_p.rolling(20).mean()))',
        'direction': 'long',
        'icir_threshold': 0.3,
        'ic_pct_threshold': 0.55,
    },
    {
        'name': 'gap_up_persistence',
        'label': '跳空缺口持续性',
        'hypothesis': '近期出现向上跳空缺口的股票有持续动量',
        'logic': '市场微观结构: 缺口反映信息冲击, 市场需要时间消化',
        'formula': '(open_p - close_p.shift(1)) / close_p.shift(1)',
        'direction': 'long',
        'icir_threshold': 0.3,
        'ic_pct_threshold': 0.55,
    },
    {
        'name': 'low_vol_anomaly',
        'label': '低波动异常',
        'hypothesis': '过去一月波动率最低的股票未来收益更高',
        'logic': '低波动异象: A股散户偏好高波动, 导致低波被低估',
        'formula': '-close_p.pct_change().rolling(20).std()',
        'direction': 'long',
        'icir_threshold': 0.3,
        'ic_pct_threshold': 0.55,
    },
    {
        'name': 'turnover_shock',
        'label': '换手率突变',
        'hypothesis': '换手率突然放大(相比过去一月均值)的股票短期有超额',
        'logic': '流动性冲击: 异常放量反映信息进入市场',
        'formula': 'volume_p.rolling(5).mean() / volume_p.rolling(20).mean()',
        'direction': 'long',
        'icir_threshold': 0.3,
        'ic_pct_threshold': 0.55,
    },
    {
        'name': 'amihud_illiq',
        'label': 'Amihud非流动性',
        'hypothesis': '非流动性高的股票未来收益更高(流动性溢价)',
        'logic': '流动性溢价理论: Amihud(2002)',
        'formula': 'abs(close_p.pct_change()) / (volume_p * close_p) * 1e8',
        'direction': 'long',
        'icir_threshold': 0.3,
        'ic_pct_threshold': 0.55,
    },
    {
        'name': 'short_term_reversal_intraday',
        'label': '日内振幅反转',
        'hypothesis': '日内振幅(high/low-1)大的股票次日倾向于下跌',
        'logic': '过度投机: 日内大幅波动反映多空激烈博弈, 次日均值回归',
        'formula': '-(high_p / low_p - 1)',
        'direction': 'long',
        'icir_threshold': 0.3,
        'ic_pct_threshold': 0.55,
    },
    {
        'name': '52w_high_proximity',
        'label': '52周高点距离',
        'hypothesis': '接近52周高点的股票有持续动量(锚定效应)',
        'logic': '行为金融: 投资者对52周高点有锚定, 突破后继续上行',
        'formula': 'close_p / close_p.rolling(252).max()',
        'direction': 'long',
        'icir_threshold': 0.3,
        'ic_pct_threshold': 0.55,
    },
    # ---- 第2轮新增 (2026-06-14, 基于近期市场和Ch9方法论) ----
    {
        'name': 'overnight_gap',
        'label': '隔夜收益率',
        'hypothesis': '隔夜跳空高开(收盘→次日开盘)的股票短期有动量(信息在非交易时段进入)',
        'logic': '市场微观结构: 隔夜收益反映信息不对称, 高开有持续动量',
        'formula': '(open_p - close_p.shift(1)) / close_p.shift(1)',
        'direction': 'long',
        'icir_threshold': 0.3,
        'ic_pct_threshold': 0.55,
    },
    {
        'name': 'idiosyncratic_vol',
        'label': '特质波动率(IVOL)',
        'hypothesis': '特质波动率(剔除市场和行业后)低的股票未来收益更高',
        'logic': 'Ang et al.(2006)低IVOL异象: 个股特有风险未被定价反而被高估',
        'formula': '-close_p.pct_change().rolling(20).std() / close_p.pct_change().rolling(20).mean()',  # 简化版: CV
        'direction': 'long',
        'icir_threshold': 0.3,
        'ic_pct_threshold': 0.55,
    },
    {
        'name': 'tail_risk_5d',
        'label': '尾部风险(极端损失)',
        'hypothesis': '过去一月出现过极端单日跌幅(<-5%)的股票短期反弹',
        'logic': '行为金融: 极端损失后的均值回归, 恐慌过后的修复',
        'formula': '-close_p.pct_change().rolling(20).min()',  # 简化: 最差单日收益
        'direction': 'long',
        'icir_threshold': 0.3,
        'ic_pct_threshold': 0.55,
    },
    {
        'name': 'volume_concentration',
        'label': '放量集中度反转',
        'hypothesis': '过去5日成交量集中放大的股票(相比20日均量)短期走弱(过度关注后的均值回归)',
        'logic': '行为金融: Barinov(2014)注意力驱动交易导致过度反应后反转',
        'formula': '-(volume_p.rolling(5).max() / volume_p.rolling(20).mean())',
        'direction': 'long',
        'icir_threshold': 0.3,
        'ic_pct_threshold': 0.55,
    },
    {
        'name': 'max_ret_1m',
        'label': '上月最大单日涨幅',
        'hypothesis': '上月出现极端单日涨幅的股票后续走弱(彩票型偏好反转)',
        'logic': 'Bali et al.(2011) MAX效应: 散户追逐极端正收益导致高估后回落',
        'formula': '-close_p.pct_change().rolling(20).max()',
        'direction': 'long',
        'icir_threshold': 0.3,
        'ic_pct_threshold': 0.55,
    },
    {
        'name': 'price_delay',
        'label': '价格延迟反应',
        'hypothesis': '对市场信息反应慢的股票未来收益更低',
        'logic': 'Hou & Moskowitz(2005): 信息传递效率低的股票有折价',
        'formula': '-(close_p.pct_change(1) / close_p.pct_change(5).shift(-5)).abs()',  # 简化: 1日动量vs5日后动量比值
        'direction': 'long',
        'icir_threshold': 0.3,
        'ic_pct_threshold': 0.55,
    },
    {
        'name': 'short_interest_proxy',
        'label': '恐慌抛售代理',
        'hypothesis': '高成交+价格下跌的股票短期可能形成底部(恐慌盘出清)',
        'logic': '行为金融: 恐慌抛售+高换手=筹码换手充分, 短期修复概率高',
        'formula': '-(volume_p.rolling(3).mean() * (close_p.pct_change(3) < 0).astype(float).replace(0, np.nan))',
        'direction': 'long',
        'icir_threshold': 0.3,
        'ic_pct_threshold': 0.55,
    },
    {
        'name': 'vol_mean_reversion',
        'label': '波动率均值回归',
        'hypothesis': '过去5日波动率相比20日波动率突然放大的股票短期有反弹(恐慌后修复)',
        'logic': '波动率聚类+均值回归: 短波>>长波时往往是恐慌顶点',
        'formula': '-(close_p.pct_change().rolling(5).std() / close_p.pct_change().rolling(20).std())',
        'direction': 'long',
        'icir_threshold': 0.3,
        'ic_pct_threshold': 0.55,
    },
]


def compute_ic_icir(factor_values, forward_returns):
    """计算截面IC和ICIR (参考FA evaluation.ic_analysis)"""
    # rank IC: Spearman rank correlation per cross-section
    ic_series = []
    common_dates = factor_values.index.intersection(forward_returns.index)
    
    for dt in sorted(common_dates):
        fv = factor_values.loc[dt].dropna()
        fr = forward_returns.loc[dt].dropna()
        common = fv.index.intersection(fr.index)
        if len(common) < 30:
            continue
        ic = pd.Series(fv[common]).rank().corr(pd.Series(fr[common]).rank())
        if not np.isnan(ic):
            ic_series.append((dt, ic))
    
    if not ic_series:
        return {'ic_mean': np.nan, 'ic_std': np.nan, 'icir': np.nan,
                '+ic%': np.nan, 'ic_series': pd.Series(dtype=float)}
    
    ic_vals = pd.Series([x[1] for x in ic_series],
                        index=[x[0] for x in ic_series])
    ic_mean = ic_vals.mean()
    ic_std = ic_vals.std()
    icir = ic_mean / ic_std if ic_std > 0 else np.nan
    ic_pct_pos = (ic_vals > 0).mean()
    
    return {
        'ic_mean': ic_mean, 'ic_std': ic_std, 'icir': abs(icir),
        '+ic%': ic_pct_pos, 'ic_series': ic_vals,
    }


def run_weekly_hypothesis(hypothesis_idx=None, custom_hypothesis=None):
    """
    执行周频因子假设实验
    
    参数:
        hypothesis_idx: 使用预设假设的索引 (0-7), None则运行全部
        custom_hypothesis: 自定义假设 dict, 格式同 PRESET_HYPOTHESES
    """
    from credentials import get_tushare_api
    
    today = datetime.now()
    week_start = today - timedelta(days=today.weekday())
    date_str = today.strftime('%Y%m%d')
    
    print("=" * 70)
    print(f"  每周因子假设实验 — {today.strftime('%Y-%m-%d')} (第{today.isocalendar()[1]}周)")
    print("  方法论: XQuant Ch9 — 因子假设 → IC验证 → 纳入/推翻")
    print("=" * 70)
    
    # ---- 步骤1: 获取最近数据 ----
    print("\n[Step 1] 获取数据...")
    
    # 使用本地大盘数据 (daily_prices.csv, ~5.9M行, 5727只)
    local_path = Path(__file__).parent.parent / 'data' / 'raw' / 'daily_prices.csv'
    end_dt = today.strftime('%Y-%m-%d')
    start_dt = (today - timedelta(days=90)).strftime('%Y-%m-%d')
    
    try:
        # 只加载最近90天数据
        daily = pd.read_csv(
            local_path,
            dtype={'ts_code': str, 'trade_date': str},
            usecols=['ts_code', 'trade_date', 'open', 'high', 'low', 'close', 'vol']
        )
        daily = daily[(daily['trade_date'] >= start_dt) & (daily['trade_date'] <= end_dt)]
        print(f"  获取日线: {len(daily)} 条 (本地 daily_prices.csv)")
    except Exception as e:
        print(f"  [WARN] 本地数据加载失败: {e}, 尝试Tushare...")
        pro = get_tushare_api()
        daily = pro.daily(start_date=start_dt, end_date=end_dt,
                         fields='ts_code,trade_date,open,high,low,close,vol,amount')
        if daily is None or len(daily) == 0:
            print(f"  [ERROR] 无法获取数据")
            return None
        print(f"  获取日线: {len(daily)} 条 (Tushare)")
    
    daily['trade_date'] = pd.to_datetime(daily['trade_date'])
    daily = daily.sort_values(['ts_code', 'trade_date'])
    
    # 构建面板
    close_df = daily.pivot(index='trade_date', columns='ts_code', values='close')
    open_df = daily.pivot(index='trade_date', columns='ts_code', values='open')
    high_df = daily.pivot(index='trade_date', columns='ts_code', values='high')
    low_df = daily.pivot(index='trade_date', columns='ts_code', values='low')
    vol_df = daily.pivot(index='trade_date', columns='ts_code', values='vol')
    
    # 过滤: 至少需要40个交易日
    min_days = 40
    valid_codes = close_df.columns[close_df.count() >= min_days]
    close_df = close_df[valid_codes]
    open_df = open_df[valid_codes]
    high_df = high_df[valid_codes]
    low_df = low_df[valid_codes]
    vol_df = vol_df[valid_codes]
    
    print(f"  有效股票数: {len(valid_codes)} (要求≥{min_days}个交易日)")
    
    # 前向收益 (5日 / 1周)
    forward_5d = close_df.shift(-5) / close_df - 1
    
    # ---- 步骤2: 计算因子并评估IC ----
    print("\n[Step 2] 因子假设IC评估...")
    
    if custom_hypothesis:
        hypotheses = [custom_hypothesis]
    elif hypothesis_idx is not None:
        hypotheses = [PRESET_HYPOTHESES[hypothesis_idx]]
    else:
        hypotheses = PRESET_HYPOTHESES
    
    results = []
    
    for hyp in hypotheses:
        print(f"\n  假设: {hyp['label']} — {hyp['hypothesis']}")
        print(f"  逻辑: {hyp['logic']}")
        
        # 计算因子值 (简化的 eval 方式, 实际应用中应使用安全的因子计算)
        try:
        # 构建计算上下文 (避免与内置函数名冲突)
            close_p = close_df
            open_p = open_df
            high_p = high_df
            low_p = low_df
            volume_p = vol_df
            
            # 安全 eval (使用 _p 后缀避免与 Python 内置 open 冲突)
            factor_raw = eval(hyp['formula'])
            
            # 缩尾 (1%/99%)
            for col in factor_raw.columns:
                col_data = factor_raw[col].dropna()
                if len(col_data) > 50:
                    lo, hi = col_data.quantile(0.01), col_data.quantile(0.99)
                    factor_raw[col] = factor_raw[col].clip(lo, hi)
            
            # 截面标准化 (z-score)
            factor_z = factor_raw.subtract(factor_raw.mean(axis=1), axis=0) \
                               .divide(factor_raw.std(axis=1), axis=0)
            
            # IC评估
            result = compute_ic_icir(factor_z, forward_5d)
            
            results.append({
                **hyp,
                **result,
                'n_dates': len(result.get('ic_series', [])),
                'n_stocks_avg': int(factor_z.notna().sum(axis=1).mean()) if hasattr(factor_z, 'notna') else 0,
            })
            
            # 判断是否通过
            passed = (not np.isnan(result['icir']) and 
                     result['icir'] >= hyp['icir_threshold'] and
                     result['+ic%'] >= hyp['ic_pct_threshold'])
            
            status = "[PASS] 通过 (纳入候选池)" if passed else "[FAIL] 未通过 (推翻或需更多数据)"
            
            print(f"    IC_mean: {result['ic_mean']:+.4f}  "
                  f"ICIR: {result['icir']:.3f}  "
                  f"+IC%: {result['+ic%']:.1%}  "
                  f"n={result['n_dates']}截面")
            print(f"    结论: {status}")
            
        except Exception as e:
            print(f"    [ERROR] {e}")
            results.append({**hyp, 'error': str(e)})
    
    # ---- 步骤3: 汇总报告 ----
    print("\n" + "=" * 70)
    print("[Step 3] 汇总报告")
    print("=" * 70)
    
    valid_results = [r for r in results if 'icir' in r and not np.isnan(r['icir'])]
    
    if valid_results:
        print(f"\n{'假设':20s} {'ICIR':>8s} {'IC_mean':>8s} {'+IC%':>7s} {'状态':>6s}")
        print("-" * 60)
        for r in sorted(valid_results, key=lambda x: -x['icir']):
            passed = r['icir'] >= r['icir_threshold'] and r['+ic%'] >= r['ic_pct_threshold']
            status = 'PASS' if passed else 'FAIL'
            print(f"{r['label']:20s} {r['icir']:+7.3f} {r['ic_mean']:+7.4f} "
                  f"{r['+ic%']:6.1%} {status:>6s}")
        
        n_passed = sum(1 for r in valid_results 
                       if r['icir'] >= r['icir_threshold'] and r['+ic%'] >= r['ic_pct_threshold'])
        print(f"\n  通过率: {n_passed}/{len(valid_results)}")
    
    # ---- 步骤4: 保存报告 ----
    report_file = REPORT_DIR / f'weekly_factor_hypothesis_{date_str}.csv'
    report_df = pd.DataFrame(results)
    if 'ic_series' in report_df.columns:
        report_df = report_df.drop(columns=['ic_series'])
    report_df.to_csv(report_file, index=False, encoding='utf-8-sig')
    print(f"\n  报告已保存: {report_file}")
    
    return results


def suggest_next_hypothesis():
    """
    分析最近市场特征, 建议下周的因子方向
    
    参考 Ch9: "观察市场最近的特征 → 提一个新因子假设"
    """
    from credentials import get_tushare_api
    
    today = datetime.now()
    pro = get_tushare_api()
    
    # 获取指数数据判断市场状态
    print("\n[市场特征快照]")
    try:
        idx = pro.index_daily(ts_code='000300.SH',
                              start_date=(today - timedelta(days=60)).strftime('%Y%m%d'),
                              end_date=today.strftime('%Y%m%d'))
        if len(idx) > 0:
            idx = idx.sort_values('trade_date')
            ret_5d = idx['close'].pct_change(5).iloc[-1]
            ret_20d = idx['close'].pct_change(20).iloc[-1]
            vol_20d = idx['close'].pct_change().rolling(20).std().iloc[-1]
            
            print(f"  沪深300: 5日收益={ret_5d:+.2%}  20日收益={ret_20d:+.2%}  "
                  f"20日波动率={vol_20d:.2%}")
            
            suggestions = []
            if abs(ret_5d) > 0.05:
                suggestions.append("市场剧烈波动 -> 建议测试反转因子 (reversal_5d)")
            if vol_20d > 0.02:
                suggestions.append("高波动环境 -> 建议测试低波因子 (low_vol_anomaly)")
            if ret_20d > 0.05:
                suggestions.append("上升趋势 -> 建议测试动量延续因子 (52w_high)")
            if ret_20d < -0.03:
                suggestions.append("下跌趋势 -> 建议测试超跌反弹因子 (short_term_reversal)")
            
            if suggestions:
                print("\n  [>>] 下周因子建议 (基于当前市场特征):")
                for s in suggestions:
                    print(f"    * {s}")
            else:
                print("\n  [>>] 市场处于中性状态, 建议轮换测试预设因子池")
                
    except Exception as e:
        print(f"  获取指数数据失败: {e}")
    
    print("\n  [预设因子池] (8个方向):")
    for i, hyp in enumerate(PRESET_HYPOTHESES):
        print(f"    [{i}] {hyp['label']:25s} -- {hyp['hypothesis'][:40]}...")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='每周因子假设实验 (XQuant Ch9方法)')
    parser.add_argument('--idx', type=int, default=None, 
                       help='运行预设假设索引 (0-7), 默认运行全部')
    parser.add_argument('--suggest', action='store_true',
                       help='分析市场特征并建议下周因子方向')
    parser.add_argument('--all', action='store_true',
                       help='运行全部预设假设 (默认)')
    
    args = parser.parse_args()
    
    if args.suggest:
        suggest_next_hypothesis()
    elif args.idx is not None:
        run_weekly_hypothesis(hypothesis_idx=args.idx)
    else:
        # default: run all
        run_weekly_hypothesis(hypothesis_idx=None)
