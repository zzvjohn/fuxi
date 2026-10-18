# -*- coding: utf-8 -*-
"""P-20260904-003 + P-20260908-005: 难做指数状态影子 (纯监控, 零干预)

研报《基于市场状态划分的机器学习ALPHA因子构建》(2026-09 前沿扫描 A 维度):
  难做指数与 AGRU 基准因子未来 5 日 IC 秩相关 -24.66%, 十分组单调。

三因子 (申万一级行业 + 个股截面, 全部只用当日及历史数据):
  f1 行业成交额集中度 HHI:  当日申万一级行业成交额占比的赫芬达尔指数 Σ(share²)
                            (P-005 升级: 原 Top-3 近似 → 全分布 HHI, 对齐研报口径)
  f2 个股正收益贡献头部集中度: 当日正收益前 10% 个股的贡献 (pct_chg × amount)
                            占全部正贡献总和比重
                            (P-005 升级: 原行业级 Top-3 → 个股级前 10%, 对齐研报口径)
  f3 个股收益截面离散度:   全市场当日 pct_chg 标准差
难做指数 = 三因子各自 60 日 z-score 等权合成 (越高越难做)

验证 (影子): 难做指数(t) 与 20 日动量因子当日截面 fwd5 rank IC(t) 的 Spearman 秩相关
  研报锚点 -24.66%; 本影子判定线 |ρ| >= 0.10 即有区分度, ρ < -0.15 强区分。
  注意: 研报 IC 用 AGRU(机器学习因子), 本影子用线性 20 日动量近似, 系数不可直接比。

输出: data/market_doability_index.json (日序列 + 验证统计 + 最新读数)
用法: python scripts/market_doability_index.py [--days 760]
"""
import sys
import json
import argparse
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
OUT_FILE = DATA_DIR / 'market_doability_index.json'

Z_WINDOW = 60          # z-score 窗口
FWD_DAYS = 5           # fwd5 收益
MOM_DAYS = 20          # 基准动量因子窗口
IC_RHO_WARN = 0.10     # 秩相关区分度判定线 (|ρ|)
Z_PCTL_ALERT = 0.8     # 难做指数 60 日分位提醒线


def load_data(days: int):
    """加载 daily_prices + 行业映射, 返回 (daily_df, ind_map)"""
    df = pd.read_csv(
        DATA_DIR / 'raw' / 'daily_prices.csv',
        dtype={'ts_code': str, 'trade_date': str},
        usecols=['ts_code', 'trade_date', 'close', 'amount', 'pct_chg'],
    )
    df['trade_date'] = pd.to_datetime(df['trade_date'], format='mixed')
    end = df['trade_date'].max()
    df = df[df['trade_date'] >= end - timedelta(days=days)]
    df = df.drop_duplicates(subset=['ts_code', 'trade_date'])
    for c in ['close', 'amount', 'pct_chg']:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    m = pd.read_csv(
        DATA_DIR / 'raw' / 'sw_index_member_all.csv',
        dtype={'ts_code': str},
        usecols=['l1_name', 'ts_code', 'out_date'],
    )
    m = m[m['out_date'].isna()][['ts_code', 'l1_name']]
    m = m.drop_duplicates(subset=['ts_code'])
    return df, m


def compute_doability(df: pd.DataFrame, ind_map: pd.DataFrame):
    """三因子合成难做指数日序列。"""
    d = df.merge(ind_map, on='ts_code', how='left')

    # ── f1: 行业成交额集中度 HHI = Σ(share²) ──
    ind_amt = d.groupby(['trade_date', 'l1_name'])['amount'].sum().reset_index()
    tot_amt = ind_amt.groupby('trade_date')['amount'].sum().rename('tot_amt')
    ind_amt = ind_amt.merge(tot_amt, on='trade_date', how='left')
    ind_amt['share'] = ind_amt['amount'] / ind_amt['tot_amt']
    f1 = (ind_amt.groupby('trade_date')['share']
          .apply(lambda s: float((s ** 2).sum())).rename('f1_hhi'))

    # ── f2: 个股正收益贡献头部集中度 (前 10% 个股贡献占比) ──
    d2 = d[['trade_date', 'ts_code', 'pct_chg', 'amount']].copy()
    d2['pos_contrib'] = d2['pct_chg'].clip(lower=0) * d2['amount'] / 100.0
    pos_all = (d2[d2['pos_contrib'] > 0]
               .groupby('trade_date')['pos_contrib'].sum().rename('sum_pos'))
    d2 = d2[d2['pos_contrib'] > 0].copy()
    d2['rk'] = d2.groupby('trade_date')['pos_contrib'].rank(
        ascending=False, pct=True)
    head = d2[d2['rk'] <= 0.10].groupby('trade_date')['pos_contrib'].sum()
    f2 = (head / pos_all).rename('f2_head10_conc').replace([np.inf, -np.inf], np.nan)

    # ── f3: 个股收益截面离散度 ──
    f3 = d.groupby('trade_date')['pct_chg'].std().rename('f3_disp')

    comp = pd.concat([f1, f2, f3], axis=1)
    # 三因子各自 60 日 z-score 等权
    z = ((comp - comp.rolling(Z_WINDOW).mean())
         / comp.rolling(Z_WINDOW).std().replace(0, np.nan))
    doability = z.mean(axis=1, skipna=True).rename('doability')
    return comp, z, doability


def compute_fwd5_ic(df: pd.DataFrame, days_back: int):
    """20 日动量因子每日截面 fwd5 rank IC 序列 (矩阵化)。

    返回 Series(index=trade_date, name='ic')。仅在验证端使用未来数据,
    不进入难做指数计算 (影子脚本, 无前视污染)。
    """
    codes = df['ts_code'].unique()
    piv = df.pivot_table(index='trade_date', columns='ts_code',
                         values='close', aggfunc='last')
    piv = piv.reindex(columns=codes).astype(np.float32)
    closes = piv.values
    # 20 日动量信号 (T×N)
    mom = np.full_like(closes, np.nan, dtype=np.float32)
    mom[MOM_DAYS:] = closes[MOM_DAYS:] / closes[:-MOM_DAYS] - 1.0
    # fwd5 收益
    fwd = np.full_like(closes, np.nan, dtype=np.float32)
    fwd[:-FWD_DAYS] = closes[FWD_DAYS:] / closes[:-FWD_DAYS] - 1.0

    def _rank_ic(a, b):
        mask = np.isfinite(a) & np.isfinite(b)
        if mask.sum() < 100:
            return np.nan
        ra = pd.Series(a[mask]).rank().values
        rb = pd.Series(b[mask]).rank().values
        if np.std(ra) == 0 or np.std(rb) == 0:
            return np.nan
        return float(np.corrcoef(ra, rb)[0, 1])

    ic_vals = [_rank_ic(mom[i], fwd[i]) for i in range(len(closes))]
    return pd.Series(ic_vals, index=piv.index, name='ic')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--days', type=int, default=760)
    args = ap.parse_args()

    print('[难做指数] 加载数据...')
    df, ind_map = load_data(args.days)
    print(f'[难做指数] daily {len(df)} 行, 行业映射 {ind_map["l1_name"].nunique()} 个一级行业, '
          f'最新 {df["trade_date"].max().date()}')

    comp, z, doability = compute_doability(df, ind_map)
    print(f'[难做指数] 序列 {len(doability)} 日, 最新 {doability.index.max().date()}')

    print('[难做指数] 计算 fwd5 IC 验证序列 (矩阵化)...')
    ic = compute_fwd5_ic(df, args.days)

    # 对齐: 难做指数(t) vs 基准因子 fwd5 IC(t)
    aligned = pd.concat([doability, ic], axis=1).dropna()
    rho = aligned['doability'].corr(aligned['ic'], method='spearman') if len(aligned) > 30 else np.nan
    pear = aligned['doability'].corr(aligned['ic']) if len(aligned) > 30 else np.nan

    latest = doability.dropna().iloc[-1] if doability.notna().any() else np.nan
    pctl60 = None
    if doability.notna().any():
        tail = doability.dropna().tail(Z_WINDOW)
        pctl60 = float((tail <= latest).mean())

    alerts = []
    if pctl60 is not None and pctl60 > Z_PCTL_ALERT:
        alerts.append(f'doability_pctl60_{pctl60:.0%}_high')
    if rho is not None and not np.isnan(rho) and abs(rho) >= IC_RHO_WARN:
        alerts.append(f'ic_rho_{rho:+.3f}_distinct')

    out = {
        "date": str(doability.index.max().date()),
        "generated_at": pd.Timestamp.now().isoformat(),
        "proposal": "P-20260904-003 + P-20260908-005",
        "n_days": int(len(doability)),
        "doability_latest": None if pd.isna(latest) else round(float(latest), 4),
        "doability_pctl_60d": None if pctl60 is None else round(pctl60, 4),
        "components_latest": {
            k: (None if pd.isna(v) else round(float(v), 4))
            for k, v in comp.iloc[-1].items()
        },
        "validation": {
            "benchmark": f"{MOM_DAYS}d momentum fwd{FWD_DAYS} rank IC",
            "n_aligned": int(len(aligned)),
            "spearman_rho": None if rho is None or pd.isna(rho) else round(float(rho), 4),
            "pearson_r": None if pear is None or pd.isna(pear) else round(float(pear), 4),
            "threshold_abs": IC_RHO_WARN,
            "research_anchor": -0.2466,  # 研报: AGRU 因子 IC 与难做指数秩相关
        },
        "series_30d": {
            str(k.date()): (None if pd.isna(v) else round(float(v), 3))
            for k, v in doability.dropna().tail(30).items()
        },
        "alerts": alerts,
    }
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT_FILE, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print(f'[难做指数] 落盘 {OUT_FILE}')
    print(f'  最新难做指数: {out["doability_latest"]} (60日分位 {out["doability_pctl_60d"]})')
    print(f'  Spearman(难做指数, 动量fwd5 IC) = {out["validation"]["spearman_rho"]} '
          f'(n={out["validation"]["n_aligned"]}, 研报锚点 -0.2466)')
    for a in alerts:
        print(f'  ⚠️ {a}')


if __name__ == '__main__':
    main()
