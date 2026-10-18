"""
因子分域 — 前提证伪测试 (Domain IC Heterogeneity Test)
========================================================

目标: 在**我们的数据**(2021-2026)上验证"因子分域"框架的核心前提 ——
**同一因子在不同 DS 域中的 RankIC 是否结构性分化**。

若成立 (如报告所述 BP 高DS域 IC 7% vs 低DS域 2.7%; RET20 单调递增;
RET240 方向翻转), 则分域选股框架值得投入改造 FA 流水线。
若不成立, 则该效应可能是特定数据集产物, 需重新评估。

测试因子 (直接对应报告案例):
    - bp        : 账面市值比 = 1/pb   (价值因子, 报告称高DS域IC=7.0%)
    - ret_20    : 20日收益 (短期反转, 报告称 3.5%→9.3% 单调递增)
    - ret_240   : 240日收益 (长期动量, 报告称 1.0%→-3.4% 方向翻转)
    - ln_mcap   : 市值本身 (检验规模效应是否随DS变化)

用法:
    cd e:/quant/research/factor_alchemy
    python scripts/domain_ic_test.py
"""

import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data.loader import load_all_data
from ga.domain_split import compute_ds_panel, rankic_per_domain

OOS_SPLIT = pd.Timestamp('2025-01-01')
N_DOMAINS = 3


def main():
    t0 = time.time()
    print("=" * 70)
    print("  因子分域 — 前提证伪测试 (Domain IC Heterogeneity)")
    print("=" * 70)

    # ---------- 1. 加载数据 ----------
    print("\n[1] 加载数据 (2021-01-01 ~ 2026-06-25) ...")
    data = load_all_data('2021-01-01', '2026-06-25')
    close = data['price_data']['close']
    val = data['valuation_data']  # 长表: code, trade_date, pb, market_cap, ...
    print(f"    价格面板: {close.shape} | 估值长表: {val.shape}")

    # ---------- 2. 周频收益 & 市值 ----------
    print("\n[2] 构建周频面板 ...")
    close_w = close.resample('W').last()
    close_w.index = pd.to_datetime(close_w.index)
    fwd_ret = close_w.shift(-1) / close_w - 1

    # 估值 pivot → 周频
    val2 = val.copy()
    val2['trade_date'] = pd.to_datetime(val2['trade_date'])
    pb_w = val2.pivot(index='trade_date', columns='code', values='pb').resample('W').last()
    mv_w = val2.pivot(index='trade_date', columns='code', values='market_cap').resample('W').last()
    mv_w = np.log(mv_w.clip(lower=1))

    # 对齐列
    common = close_w.columns.intersection(pb_w.columns).intersection(mv_w.columns)
    close_w = close_w[common]
    fwd_ret = fwd_ret[common]
    pb_w = pb_w[common]
    mv_w = mv_w[common]

    # ---------- 3. 计算测试因子 ----------
    print("[3] 计算测试因子 ...")
    bp = (1.0 / pb_w).where(pb_w > 0)
    ret_20 = close_w.pct_change(20)
    ret_240 = close_w.pct_change(240)
    ln_mcap = mv_w.copy()
    factors = {
        'bp(价值)': bp,
        'ret_20(短期反转)': ret_20,
        'ret_240(长期动量)': ret_240,
        'ln_mcap(市值)': ln_mcap,
    }
    print(f"    因子面板: {bp.shape}")

    # ---------- 4. 计算 DS 分域 ----------
    print(f"[4] 计算 DS 分域 (n_domains={N_DOMAINS}, 控制因子=市值) ...")
    ds_panel, domain_panel = compute_ds_panel(
        fwd_ret, {'size': mv_w}, n_domains=N_DOMAINS
    )
    # 分域覆盖度 (有效样本占比)
    coverage = domain_panel.notna().mean().mean()
    print(f"    分域有效覆盖度: {coverage:.1%}")
    # 各域股票数占比 (全样本均值)
    dom_share = domain_panel.apply(lambda r: r.value_counts(normalize=True).to_dict()
                                    if r.notna().any() else {}, axis=1)
    print(f"    各域占比(末周): {domain_panel.iloc[-1].value_counts(normalize=True).round(3).to_dict()}")

    # ---------- 5. 各域 RankIC (训练期) ----------
    print(f"\n[5] 各因子 × 各域 RankIC (训练期 < {OOS_SPLIT.strftime('%Y-%m-%d')})")
    print("-" * 70)
    train_mask = domain_panel.index < OOS_SPLIT
    dom_train = domain_panel.loc[train_mask]
    fwd_train = fwd_ret.loc[train_mask]

    results = {}
    for fname, fdf in factors.items():
        fdf_t = fdf.loc[train_mask]
        res = rankic_per_domain(fdf_t, fwd_train, dom_train, n_domains=N_DOMAINS)
        results[fname] = res
        print(f"\n  ▶ {fname}")
        print(f"    {'域':>4s} {'IC_mean':>10s} {'ICIR':>8s} {'n_weeks':>8s}")
        for d in range(N_DOMAINS):
            r = res.loc[d]
            ic = r['IC_mean']
            icir = r['ICIR']
            ic_s = f"{ic:+.4f}" if pd.notna(ic) else "  n/a"
            ir_s = f"{icir:+.2f}" if pd.notna(icir) else " n/a"
            print(f"    {d:>4d} {ic_s:>10s} {ir_s:>8s} {int(r['n_weeks']):>8d}")
        # 单调性检查
        ics = [res.loc[d]['IC_mean'] for d in range(N_DOMAINS)]
        if all(pd.notna(x) for x in ics):
            mono = "单调递增 ✅" if ics[2] > ics[1] > ics[0] else (
                   "单调递减 ✅" if ics[0] > ics[1] > ics[2] else "非单调")
            spread = ics[2] - ics[0]
            print(f"    → 域间IC跨度: {spread:+.4f} | {mono}")

    # ---------- 6. 结论 ----------
    print("\n" + "=" * 70)
    print("[6] 结论")
    print("=" * 70)
    bp_res = results.get('bp(价值)')
    if bp_res is not None:
        ics = [bp_res.loc[d]['IC_mean'] for d in range(N_DOMAINS)]
        if all(pd.notna(x) for x in ics):
            print(f"  BP: 低DS域IC={ics[0]:+.4f} → 高DS域IC={ics[2]:+.4f} "
                  f"(报告claim: 2.7%→7.0%)")
            print(f"  {'✅ 前提成立' if ics[2] > ics[0] else '⚠ 前提不成立'}: "
                  f"BP的IC随DS递增")
    print(f"\n  总耗时: {time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()
