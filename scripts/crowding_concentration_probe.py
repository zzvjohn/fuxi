# -*- coding: utf-8 -*-
"""P-20260906-002: 拥挤集中度分层验证 (Chincarini et al 2026 JBF 移植, 纯影子)。

问题: 库内因子/策略的异常收益是否集中在最拥挤的股票中? 若集中, 则收益 = 拥挤
风险补偿而非 alpha (Chincarini, Lazo-Paz & Moneta (2026): 剔除拥挤股后 11 个经典
anomaly 全部失效)。

方法 (A 股 Days-ADV 代理):
  1. 拥挤度代理 x3: 20日均换手率(daily_basic turnover_rate) / 20日均成交额 /
     20日均成交量
  2. 每日横截面将股票按拥挤度分 4 层 (Q4=最拥挤)
  3. 对每个主力因子: 多头 = 当日因子值 top 20%, 按拥挤层分解多头次日收益
  4. 集中度 = 最高拥挤层收益贡献 / 多头总收益 (时间平均); >0.7 判定疑似风险补偿

零干预: 只输出 data/crowding_concentration_probe.json, 不修改任何裁决状态。
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'research' / 'factor_alchemy'))

DATA_RAW = PROJECT_ROOT / 'data' / 'raw'
OUT_PATH = PROJECT_ROOT / 'data' / 'crowding_concentration_probe.json'

N_FACTORS = 8          # 主力因子数 (JQ 已验证 + daily_icir Top)
LOOKBACK_DAYS = 270    # 因子计算窗口
LONG_PCT = 0.20        # 多头 = 因子 top 20%
CONCENTRATION_TH = 0.70  # 最高拥挤层贡献占比阈值


def load_universe():
    """加载价格面板 + 换手率数据, 返回 (daily, basic)。"""
    daily = pd.read_csv(
        DATA_RAW / 'daily_prices.csv',
        dtype={'ts_code': str, 'trade_date': str},
        usecols=['ts_code', 'trade_date', 'close', 'vol', 'amount'],
    )
    daily['trade_date'] = pd.to_datetime(daily['trade_date'], format='mixed')
    daily = daily.drop_duplicates(subset=['ts_code', 'trade_date'], keep='last')

    basic = pd.read_csv(
        DATA_RAW / 'daily_basic.csv',
        dtype={'ts_code': str, 'trade_date': str},
        usecols=['ts_code', 'trade_date', 'turnover_rate'],
    )
    basic['trade_date'] = pd.to_datetime(basic['trade_date'], format='mixed')
    basic = basic.drop_duplicates(subset=['ts_code', 'trade_date'], keep='last')
    return daily, basic


def pick_main_factors():
    """从 unified_factor_pool 选主力因子: JQ 已验证且 icir 有效优先, 再按 daily_icir 补足。"""
    pool = pd.read_csv(PROJECT_ROOT / 'data' / 'unified_factor_pool.csv')
    pool = pool[pool['formula'].notna() & (pool['formula'].astype(str).str.len() > 5)]
    pool = pool[pool['daily_icir'].notna()]  # 剔除 icir 缺失的 v3_xxx 占位行
    jq = pool[pool['jq_verified'].notna()].copy()
    rest = pool[~pool.index.isin(jq.index)].copy()
    jq = jq.sort_values('daily_icir', ascending=False)
    rest = rest.sort_values('daily_icir', ascending=False)
    sel = pd.concat([jq, rest]).head(N_FACTORS)
    return sel[['name', 'formula', 'daily_icir', 'jq_verified']].to_dict('records')


def compute_factor_values(icc, formula):
    """用复用实例求因子值 (长表), 失败返回 None。"""
    try:
        fv = icc._eval_formula(formula, 'probe')
    except Exception as e:
        print(f'    ⚠️ 公式执行失败: {type(e).__name__}: {str(e)[:100]}')
        return None
    if fv is None or len(fv) == 0:
        return None
    fv = fv.rename(columns={'factor_value': 'fv'})
    return fv[['ts_code', 'trade_date', 'fv']]


def concentration_by_crowding(factor_df, daily, basic, crowd_col, crowd_df):
    """单因子 × 单拥挤代理: 剔除式集中度 (Chincarini 口径)。

    R_all  = 多头组合平均日收益 (全部层)
    R_excl = 剔除最高拥挤层后的多头平均日收益
    loss   = (R_all - R_excl) / |R_all|  → 剔除拥挤股后的收益损失比例
    loss 中位数 > 0.7 且 R_all>0 ⇒ 收益几乎全来自拥挤股 = 疑似风险补偿
    """
    # 拥挤度原始值 + 20 日均值
    if crowd_col == 'turnover':
        c = crowd_df[['ts_code', 'trade_date', 'turnover_rate']].rename(
            columns={'turnover_rate': 'crowd'})
        c['crowd20'] = c.groupby('ts_code')['crowd'].transform(
            lambda s: s.rolling(20, min_periods=5).mean())
    else:
        src_col = 'vol' if crowd_col == 'volume' else crowd_col
        c = daily[['ts_code', 'trade_date', src_col]].rename(
            columns={src_col: 'crowd'})
        c['crowd20'] = c.groupby('ts_code')['crowd'].transform(
            lambda s: s.rolling(20, min_periods=5).mean())

    m = factor_df.merge(c, on=['ts_code', 'trade_date'], how='inner')
    m = m.merge(daily[['ts_code', 'trade_date', 'close']],
                on=['ts_code', 'trade_date'], how='inner')
    m = m.sort_values(['ts_code', 'trade_date'])
    m['fwd_ret'] = m.groupby('ts_code')['close'].shift(-1) / m['close'] - 1
    m = m.dropna(subset=['fv', 'crowd20', 'fwd_ret'])

    daily_loss = []
    quartile_ret = {f'Q{i}': [] for i in range(1, 5)}
    n_days = 0
    for date, g in m.groupby('trade_date'):
        if len(g) < 100:
            continue
        n_long = max(1, int(len(g) * LONG_PCT))
        long_ = g.nlargest(n_long, 'fv')
        try:
            long_['q'] = pd.qcut(long_['crowd20'], 4, labels=False,
                                 duplicates='drop') + 1
        except ValueError:
            continue
        q_ret = long_.groupby('q')['fwd_ret'].mean()
        r_all = float(long_['fwd_ret'].mean())
        top_q = int(q_ret.index.max())
        r_excl = float(long_[long_['q'] < top_q]['fwd_ret'].mean()) \
            if (long_['q'] < top_q).any() else np.nan
        if np.isnan(r_excl) or abs(r_all) < 1e-6:
            continue
        daily_loss.append((r_all - r_excl) / abs(r_all))
        for qi in range(1, 5):
            quartile_ret[f'Q{qi}'].append(q_ret.get(qi, np.nan))
        n_days += 1

    if not daily_loss:
        return None
    med_loss = float(np.nanmedian(daily_loss))
    r_all_mean = float(np.nanmean(
        [np.nanmean(v) for v in quartile_ret.values() if v])) if n_days else 0.0
    return {
        'top_quartile_contribution': round(med_loss, 4),
        'metric': '剔除最高拥挤层后的多头收益损失比例 (中位数)',
        'n_days': n_days,
        'mean_ret_by_quartile': {k: round(float(np.nanmean(v)), 5)
                                 if v and not all(np.isnan(x) for x in v) else None
                                 for k, v in quartile_ret.items()},
        'suspected_risk_premium': bool(med_loss > CONCENTRATION_TH),
    }


def main():
    print('=' * 70)
    print('  拥挤集中度分层验证 (P-20260906-002, Chincarini 2026 移植, 影子)')
    print('=' * 70)
    factors = pick_main_factors()
    print(f'  主力因子 {len(factors)} 个:')
    for f in factors:
        print(f"    - {f['name']} (ICIR={f['daily_icir']}, jq={f['jq_verified']})")

    daily, basic = load_universe()
    print(f'  数据: 价格 {len(daily):,} 行 / 换手率 {len(basic):,} 行')

    # 单实例复用 (避免每因子全量重载 ~7M 行)
    from factor_ic_computer import FactorICComputer
    icc = FactorICComputer(lookback_days=LOOKBACK_DAYS)

    results = []
    for f in factors:
        fv = compute_factor_values(icc, f['formula'])
        if fv is None:
            results.append({'factor': f['name'], 'status': 'eval_failed'})
            continue
        print(f"  [{f['name']}] 因子值 {len(fv):,} 行, 分层计算中...")
        entry = {'factor': f['name'], 'daily_icir': f['daily_icir'],
                 'jq_verified': f['jq_verified'], 'crowding': {}}
        for crowd_col, crowd_df in [('turnover', basic), ('amount', daily),
                                    ('volume', daily)]:
            r = concentration_by_crowding(fv, daily, basic, crowd_col, crowd_df)
            entry['crowding'][crowd_col] = r
            if r:
                print(f"    {crowd_col:9s}: 剔除顶层损失 {r['top_quartile_contribution']:.1%}"
                      f" {'🔴疑似风险补偿' if r['suspected_risk_premium'] else ''}")
        results.append(entry)

    out = {
        'generated_at': pd.Timestamp.now().isoformat(),
        'method': 'Days-ADV A股代理: 20日均换手率/成交额/成交量 横截面4分层, '
                  '多头=factor top20%, 指标=剔除最高拥挤层后的多头收益损失比例(中位数)',
        'threshold': CONCENTRATION_TH,
        'factors': results,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT_PATH, 'w', encoding='utf-8'), ensure_ascii=False,
              indent=2, default=str)
    n_susp = sum(1 for r in results
                 for c in r.get('crowding', {}).values()
                 if c and c.get('suspected_risk_premium'))
    print(f'\n  完成: {OUT_PATH.name} ({len(results)} 因子, {n_susp} 例疑似风险补偿)')


if __name__ == '__main__':
    main()
