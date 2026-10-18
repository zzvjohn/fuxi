# -*- coding: utf-8 -*-
"""P-20260908-001: 分域建模评估通道 (市值×行业) + BH-FDR 多重检验校正

对单因子公式做 市值(4层) × 申万一级行业(~31域) 分域 fwd5 IC 评估:
  1. 全市场 IC/ICIR/+IC% (与 FactorICComputer 同口径)
  2. 每域 IC 序列 → t 统计量 → 近似正态 p 值 → BH-FDR 校正 (q≤0.10 显著)
  3. 置换噪声基线: 因子值横截面打乱 1 次, 重算 null 域 ICIR 95% 分位
  4. domain_specific 判定: 显著域 ≥1 ∧ 全市场 ICIR<0.1 ∧ 显著域 ICIR>null 上限
     (仅打标签, 不裁决 PASS — 铁则①JQ 唯一真相源, 铁则④ domain配对>统计筛选)

用法: python research/factor_alchemy/domain_split_evaluator.py --name X --formula "..." 
输出: data/domain_split_eval_<name>.json
"""
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
PRICE_PATH = DATA_DIR / 'raw' / 'daily_prices.csv'
BASIC_PATH = DATA_DIR / 'raw' / 'daily_basic.csv'
SW_PATH = DATA_DIR / 'raw' / 'sw_index_member_all.csv'
OUT_DIR = DATA_DIR / 'domain_split'

LOOKBACK_DAYS = 270
FWD = 5
MIN_DOMAIN_STOCKS = 30      # 单日域内最少样本
MIN_DOMAIN_DAYS = 30        # 域内最少有效日
FDR_Q = 0.10                # BH-FDR 阈值
FULL_ICIR_WEAK = 0.10       # 全市场"弱"判定线
N_PERM = 1                  # 置换次数 (成本考量; 可扩展)


def _load_price():
    df = pd.read_csv(PRICE_PATH, low_memory=False)
    if pd.api.types.is_integer_dtype(df["trade_date"]):
        df["trade_date"] = pd.to_datetime(df["trade_date"].astype(str), format='%Y%m%d')
    else:
        df["trade_date"] = pd.to_datetime(df["trade_date"], format='mixed')
    df = df.dropna(subset=["trade_date"])
    df = df.drop_duplicates(subset=["ts_code", "trade_date"], keep="last")
    start = df["trade_date"].max() - pd.Timedelta(days=LOOKBACK_DAYS)
    df = df[df["trade_date"] >= start]
    for c in ["open", "high", "low", "close", "vol", "amount"]:
        if c in df.columns and f"{c}_p" not in df.columns:
            df[f"{c}_p"] = df[c]
    df["volume_p"] = df["vol"]
    return df


def _load_mv():
    db = pd.read_csv(BASIC_PATH, low_memory=False, usecols=["ts_code", "trade_date", "total_mv"])
    if pd.api.types.is_integer_dtype(db["trade_date"]):
        db["trade_date"] = pd.to_datetime(db["trade_date"].astype(str), format='%Y%m%d')
    else:
        db["trade_date"] = pd.to_datetime(db["trade_date"], format='mixed')
    db = db.dropna(subset=["trade_date"])
    return db


def _load_sw():
    m = pd.read_csv(SW_PATH, dtype={'ts_code': str}, usecols=['l1_name', 'ts_code', 'out_date'])
    m = m[m['out_date'].isna()][['ts_code', 'l1_name']]
    return m.drop_duplicates(subset=['ts_code'])


def _rank_ic_vec(a: np.ndarray, b: np.ndarray) -> float:
    """两向量 rank IC (spearman), 输入已对齐且含 NaN"""
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < MIN_DOMAIN_STOCKS:
        return np.nan
    ra = pd.Series(a[mask]).rank().values.astype(np.float64)
    rb = pd.Series(b[mask]).rank().values.astype(np.float64)
    if np.std(ra) == 0 or np.std(rb) == 0:
        return np.nan
    return float(np.corrcoef(ra, rb)[0, 1])


def _bh_fdr(pvals: np.ndarray, q: float = FDR_Q) -> np.ndarray:
    """BH-FDR 校正, 返回与输入同长的显著布尔数组 (NaN p 视为不显著)"""
    n = len(pvals)
    valid = ~np.isnan(pvals)
    order = np.argsort(pvals)
    adj = np.full(n, np.nan)
    m = valid.sum()
    if m == 0:
        return np.zeros(n, dtype=bool)
    # BH: p_(i) <= i/m * q
    prev = 1.0
    for rank_i in range(m - 1, -1, -1):
        idx = order[rank_i]
        cur = pvals[idx] * m / (rank_i + 1)
        prev = min(prev, cur)
        adj[idx] = prev
    return adj <= q


def eval_formula_in_env(formula: str) -> pd.DataFrame:
    """宽表评估公式 (FactorICComputer 自加载数据, 长表返回转宽表)"""
    from factor_ic_computer import FactorICComputer
    comp = FactorICComputer(price_path=PRICE_PATH, lookback_days=LOOKBACK_DAYS,
                            forward_period=FWD)
    long_df = comp._eval_formula(formula, "domain_probe")
    if long_df is None or len(long_df) == 0:
        return None
    # _eval_formula 返回长表 [ts_code, trade_date, factor_value] → 宽表
    wide = long_df.pivot_table(index='trade_date', columns='ts_code',
                               values='factor_value', aggfunc='last')
    return wide.astype(np.float32)


def build_wide(df: pd.DataFrame, col: str) -> pd.DataFrame:
    piv = df.pivot_table(index='trade_date', columns='ts_code', values=col, aggfunc='last')
    return piv.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', required=True)
    ap.add_argument('--formula', required=True)
    ap.add_argument('--fdr-q', type=float, default=FDR_Q)
    ap.add_argument('--n-perm', type=int, default=N_PERM)
    args = ap.parse_args()

    print(f'[分域评估] {args.name}: 加载数据...')
    df = _load_price()
    mv = _load_mv()
    sw = _load_sw()
    df = df.merge(sw, on='ts_code', how='left')
    df = df.merge(mv, on=['ts_code', 'trade_date'], how='left')
    print(f'  prices {len(df):,} 行, 行业覆盖 {df["l1_name"].notna().mean():.1%}, '
          f'市值覆盖 {df["total_mv"].notna().mean():.1%}')

    # 全市场因子宽表
    print('[分域评估] 评估因子公式...')
    fw = eval_formula_in_env(args.formula)  # T×N
    if fw is None:
        print('[分域评估] ❌ 公式评估失败 (eval 返回空)')
        sys.exit(1)
    fw = fw.astype(np.float32)
    closes = build_wide(df, 'close')
    fwd = closes.shift(-FWD) / closes - 1.0
    fwd = fwd.astype(np.float32)
    common_dates = fw.index.intersection(fwd.index)
    fw, fwd = fw.loc[common_dates], fwd.loc[common_dates]
    # 列对齐 (eval 宽表股票覆盖可能与 close 宽表不一致)
    fw, fwd = fw.align(fwd, join='inner', axis=1)
    codes = fw.columns
    F = fw.values.astype(np.float64)
    R = fwd.values.astype(np.float64)

    # ── 全市场 IC ──
    full_ics = [_rank_ic_vec(F[t], R[t]) for t in range(F.shape[0])]
    full_ics = np.array(full_ics)
    full_valid = full_ics[~np.isnan(full_ics)]
    full_ic_mean = float(np.mean(full_valid)) if len(full_valid) else np.nan
    full_icir = (full_ic_mean / float(np.std(full_valid))) if len(full_valid) > 1 else np.nan
    full_plus = float((full_valid > 0).mean()) if len(full_valid) else np.nan

    # ── 域构造: 市值 4 分位 × 行业 ──
    mv_wide = build_wide(df, 'total_mv').reindex(index=common_dates, columns=codes)
    sw_map = (df.drop_duplicates('ts_code').set_index('ts_code')['l1_name']
              .reindex(codes))
    industries = sw_map.dropna().unique().tolist()
    mv_q = np.full_like(F, np.nan)
    M = mv_wide.values.astype(np.float64)
    for t in range(M.shape[0]):
        row = M[t]
        valid = np.isfinite(row)
        if valid.sum() < 100:
            continue
        qs = pd.Series(row).rank(pct=True).values
        mv_q[t] = np.floor(qs * 4).clip(0, 3)

    # 域列表: (cap_layer, industry)
    domains = [(c, i) for c in range(4) for i in industries]
    ind_idx = np.array([list(industries).index(v) if v in industries else -1
                        for v in sw_map.reindex(codes).values])
    domain_ic = {}   # (c, i) -> [ic...]
    domain_n = {}
    for t in range(F.shape[0]):
        ft, rt = F[t], R[t]
        cap = mv_q[t]
        for c in range(4):
            for i_idx, i in enumerate(industries):
                mask = (cap == c) & (ind_idx == i_idx)
                if mask.sum() < MIN_DOMAIN_STOCKS:
                    continue
                ic = _rank_ic_vec(ft[mask], rt[mask])
                if np.isnan(ic):
                    continue
                domain_ic.setdefault((c, i), []).append(ic)
                domain_n[(c, i)] = domain_n.get((c, i), 0) + 1

    # 域统计 → p 值
    domain_stats = []
    for (c, i), ics in domain_ic.items():
        if len(ics) < MIN_DOMAIN_DAYS:
            continue
        arr = np.array(ics)
        mean = float(np.mean(arr))
        std = float(np.std(arr))
        n = len(arr)
        icir = mean / std if std > 0 else 0.0
        tstat = mean / (std / np.sqrt(n - 1)) if (std > 0 and n > 1) else 0.0
        # 正态近似双尾 p
        from math import erfc, sqrt
        pval = float(erfc(abs(tstat) / sqrt(2.0)))
        domain_stats.append({
            'cap_layer': int(c), 'industry': i, 'n_days': n,
            'ic_mean': round(mean, 5), 'icir': round(icir, 5),
            'tstat': round(tstat, 3), 'p': round(pval, 6),
        })
    domain_stats.sort(key=lambda x: x['p'])

    # BH-FDR 校正
    pvals = np.array([x['p'] for x in domain_stats], dtype=float)
    sig = _bh_fdr(pvals, args.fdr_q)
    for x, s in zip(domain_stats, sig):
        x['bh_significant'] = bool(s)

    # ── 置换噪声基线 ──
    null_icirs = []
    rng = np.random.default_rng(42)
    for _ in range(args.n_perm):
        Fp = F.copy()
        for t in range(Fp.shape[0]):
            row = Fp[t]
            valid = np.isfinite(row)
            shuffled = rng.permutation(row[valid])
            Fp[t, valid] = shuffled
        perm_domain_icir = []
        for (c, i), _ in domain_ic.items():
            if len(domain_ic[(c, i)]) < MIN_DOMAIN_DAYS:
                continue
            ics = []
            for t in range(Fp.shape[0]):
                cap = mv_q[t]
                mask = (cap == c) & (ind_idx == i)
                if mask.sum() < MIN_DOMAIN_STOCKS:
                    continue
                ic = _rank_ic_vec(Fp[t][mask], R[t][mask])
                if not np.isnan(ic):
                    ics.append(ic)
            if len(ics) >= MIN_DOMAIN_DAYS:
                arr = np.array(ics)
                if np.std(arr) > 0:
                    perm_domain_icir.append(float(np.mean(arr) / np.std(arr)))
        if perm_domain_icir:
            null_icirs.extend(perm_domain_icir)
    null_icir_95 = (float(np.percentile(null_icirs, 95))
                    if null_icirs else np.nan)

    # ── domain_specific 判定 ──
    sig_domains = [x for x in domain_stats if x['bh_significant']]
    sig_above_null = [x for x in sig_domains
                      if abs(x['icir']) > null_icir_95] if not np.isnan(null_icir_95) else sig_domains
    full_weak = (not np.isnan(full_icir)) and abs(full_icir) < FULL_ICIR_WEAK
    is_domain_specific = (len(sig_above_null) >= 1) and full_weak

    out = {
        'factor_name': args.name,
        'formula': args.formula,
        'generated_at': pd.Timestamp.now().isoformat(),
        'full_market': {
            'ic_mean': round(full_ic_mean, 5), 'icir': round(full_icir, 5),
            'plus_ic_pct': round(full_plus, 4), 'n_days': int(len(full_valid)),
            'weak_threshold': FULL_ICIR_WEAK, 'is_weak': bool(full_weak),
        },
        'fdr_q': args.fdr_q,
        'n_domains_tested': len(domain_stats),
        'n_domains_significant': len(sig_domains),
        'significant_domains': sig_domains[:10],
        'permutation': {
            'n_perm': args.n_perm, 'null_icir_p95': round(null_icir_95, 5)
            if not np.isnan(null_icir_95) else None,
            'n_null_domains': int(len(null_icirs)),
        },
        'domain_specific': bool(is_domain_specific),
        'judge_rule': ('显著域(BH-FDR q≤0.10 ∧ ICIR>null p95) ≥1 ∧ 全市场|ICIR|<0.10 '
                       '→ domain_specific 标签 (仅标签, 不裁决 PASS)'),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f'domain_split_eval_{args.name}.json'
    json.dump(out, open(out_path, 'w', encoding='utf-8'), ensure_ascii=False,
              indent=1, default=str)
    print(f'[分域评估] 全市场 IC={full_ic_mean:.4f}/ICIR={full_icir:.3f}/+IC%={full_plus:.1%}')
    print(f'[分域评估] 域测试 {len(domain_stats)} 个, BH-FDR 显著 {len(sig_domains)} 个, '
          f'null ICIR p95={null_icir_95:.3f}')
    print(f'[分域评估] domain_specific 判定: {is_domain_specific}')
    if sig_domains:
        for x in sig_domains[:5]:
            print(f'   显著域: cap{x["cap_layer"]}×{x["industry"]} '
                  f'ICIR={x["icir"]:+.3f} (p={x["p"]:.2e}, n={x["n_days"]})')
    print(f'[分域评估] 落盘 {out_path}')


if __name__ == '__main__':
    main()
