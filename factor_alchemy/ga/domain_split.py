"""
因子分域模块 (Domain Splitting) — Factor Alchemy v7.5 新模块
============================================================

用途: 实现"因子分域选股框架"的核心第一步 —— 横截面快分域 (DS / Departure from
Significance, 偏离显著性)。

理论依据 (三个皮匠《2026量化因子分域报告》):
    同一因子在不同"域"中的预测能力截然不同。BP在高DS域 RankIC=7.0%(ICIR 2.24),
    低DS域仅2.7%; RET20(短期反转)单调递增 3.5%→9.3%; RET240(长期动量)单调递减
    1.0%→-3.4%(方向翻转)。中证500分域增强年化超额+8.8% vs 全域5.2%。

DS 定义 (本模块 v1, 可扩展):
    在每个再平衡日 t, 对横截面做 OLS:
        r_i = beta0 + Σ_k beta_k * ctrl_{k,i} + e_i
    残差 e_i = 个股收益中无法被"共同因子"(市值/估值等)解释的特发性部分。
    DS_i = |e_i| / std(e across stocks at week t)
    DS 越高 → 收益越"说不清" → 个股特异性因子效应(价值/反转)越强。

分域: 每个再平衡日对 DS 做分位切分 (qcut) → 0..n_domains-1 (0=低DS域)。

数据流约定 (与 run_fa.py 一致):
    所有面板均为 DataFrame, index=datetime(周), columns=ts_code(股票)。
    缺失值用 NaN 表示。

作者: Quant (2026-07-10)
"""

from __future__ import annotations
import numpy as np
import pandas as pd


def compute_ds_panel(
    returns: pd.DataFrame,
    controls: dict[str, pd.DataFrame],
    n_domains: int = 3,
    clip_ds: float = 5.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    计算每个再平衡日的 DS (Departure from Significance) 面板与分域标签面板。

    参数
    ----
    returns : DataFrame (date×stock)
        用作被解释变量的收益面板 (通常用周频前向收益 forward_returns)。
    controls : dict
        控制因子面板, 键为因子名, 值为 date×stock 面板。
        例: {'size': mcap_log_df}。至少需1个控制因子。
    n_domains : int
        分域数 (默认3: 低/中/高 DS)。
    clip_ds : float
        DS 的截尾倍数 (防止极端残差主导)。

    返回
    ----
    ds_panel : DataFrame (date×stock) 标准化绝对残差 DS
    domain_panel : DataFrame (date×stock) 整数域标签 0..n_domains-1 (0=低DS)
    """
    if not controls:
        raise ValueError("controls 不能为空, 至少需要1个控制因子(如市值)")

    dates = returns.index
    # 对齐所有面板到同一股票集 (取 returns 的列)
    stocks = returns.columns
    ctrl_aligned = {}
    for name, df in controls.items():
        ctrl_aligned[name] = df.reindex(index=dates, columns=stocks)

    r = returns.reindex(columns=stocks)
    ds_rows = []
    dom_rows = []

    for t in dates:
        y = r.loc[t].values.astype(float)  # 被解释变量
        valid = ~np.isnan(y)
        if valid.sum() < max(30, n_domains * 20):
            # 样本不足, 该周全 NaN
            ds_rows.append(pd.Series(np.nan, index=stocks))
            dom_rows.append(pd.Series(np.nan, index=stocks))
            continue

        # 构造设计矩阵 [1, ctrl1, ctrl2, ...]
        X_parts = [np.ones(valid.sum())]
        ok = valid.copy()
        for name, df in ctrl_aligned.items():
            xc = df.loc[t].values.astype(float)
            xc_v = xc[valid]
            # 该控制因子在该周自身缺失的, 整行剔除
            ok[valid] = ok[valid] & ~np.isnan(xc_v)
        if ok.sum() < max(30, n_domains * 20):
            ds_rows.append(pd.Series(np.nan, index=stocks))
            dom_rows.append(pd.Series(np.nan, index=stocks))
            continue

        yv = y[ok]
        X = np.column_stack([np.ones(ok.sum())] +
                            [ctrl_aligned[n].loc[t].values.astype(float)[ok]
                             for n in ctrl_aligned])
        try:
            beta, *_ = np.linalg.lstsq(X, yv, rcond=None)
            pred = X @ beta
            e = yv - pred
            sd = np.nanstd(e)
            if sd < 1e-12:
                sd = 1.0
            ds_vec = np.full(ok.shape, np.nan)
            ds_vec[ok] = np.clip(np.abs(e) / sd, 0, clip_ds)
            # 分域: 对有效 DS 做分位切分
            dom_vec = np.full(ok.shape, np.nan)
            ds_valid = ds_vec[ok]
            try:
                # qcut 返回 0..n_domains-1
                dom_valid = pd.qcut(ds_valid, n_domains, labels=False, duplicates='drop')
                # 若 duplicates='drop' 导致域数< n_domains, 做安全映射
                dom_vec[ok] = dom_valid
            except Exception:
                dom_vec[ok] = 0
            # 写回完整 stocks 维度
            ds_full = pd.Series(np.nan, index=stocks)
            dom_full = pd.Series(np.nan, index=stocks)
            ds_full[ok] = ds_vec[ok]
            dom_full[ok] = dom_vec[ok]
            ds_rows.append(ds_full)
            dom_rows.append(dom_full)
        except Exception:
            ds_rows.append(pd.Series(np.nan, index=stocks))
            dom_rows.append(pd.Series(np.nan, index=stocks))
            continue

    ds_panel = pd.DataFrame(ds_rows, index=dates, columns=stocks)
    domain_panel = pd.DataFrame(dom_rows, index=dates, columns=stocks)
    return ds_panel, domain_panel


def split_universe_by_domain(
    panel: pd.DataFrame,
    domain_panel: pd.DataFrame,
    domain_id: int,
) -> pd.DataFrame:
    """
    按域标签提取子集: 域外股票置 NaN (即该域不参与)。

    返回
    ----
    masked : DataFrame 仅保留属于 domain_id 的截面点, 其余 NaN
    """
    mask = (domain_panel == domain_id)
    return panel.where(mask)


def rankic_per_domain(
    factor: pd.DataFrame,
    fwd_ret: pd.DataFrame,
    domain_panel: pd.DataFrame,
    n_domains: int = 3,
) -> pd.DataFrame:
    """
    计算每个因子在各域的横截面 RankIC 序列的均值与 ICIR。

    返回
    ----
    res : DataFrame, index=域标签(0..n_domains-1), columns=['IC_mean','ICIR','n']
        同时附 'IC_series' 列 (dict of domain->Series) 供调用方进一步分析。
    """
    ic_series = {}
    for d in range(n_domains):
        ics = []
        idxs = []
        for t in factor.index:
            dom_t = domain_panel.loc[t]
            m = (dom_t == d).values
            fv = factor.loc[t].values.astype(float)[m]
            rv = fwd_ret.loc[t].values.astype(float)[m]
            good = ~(np.isnan(fv) | np.isnan(rv))
            if good.sum() >= 30:
                # scipy-free Spearman: pearson on ranks
                fr = pd.Series(fv[good]).rank()
                rr = pd.Series(rv[good]).rank()
                if fr.std() > 0 and rr.std() > 0:
                    rho = fr.corr(rr)
                    if not np.isnan(rho):
                        ics.append(rho)
                        idxs.append(t)
        ic_series[d] = pd.Series(ics, index=idxs) if idxs else pd.Series(dtype=float)

    rows = []
    for d in range(n_domains):
        s = ic_series[d].dropna()
        if len(s) > 5:
            ic_mean = s.mean()
            icir = ic_mean / (s.std() / np.sqrt(len(s))) if s.std() > 0 else np.nan
        else:
            ic_mean = np.nan
            icir = np.nan
        rows.append({'domain': d, 'IC_mean': ic_mean, 'ICIR': icir, 'n_weeks': len(s)})
    res = pd.DataFrame(rows).set_index('domain')
    res['IC_series'] = [ic_series[d] for d in range(n_domains)]
    return res


def combine_domain_scores(
    domain_scores: list[pd.DataFrame],
    domain_icir: list[float] | None = None,
    method: str = 'icir_weight',
) -> pd.DataFrame:
    """
    域间组合: 融合各域独立优化的 composite 分数。

    参数
    ----
    domain_scores : list[DataFrame]
        各域产出的 composite 分数面板 (date×stock, 域外为 NaN)。
    domain_icir : list[float] | None
        各域 OOS ICIR (用于 meta 权重)。None 则等权。
    method : str
        'icir_weight' : 按域 OOS |ICIR| 加权 (ICIR高→权重高)
        'equal'       : 等权
        'union_top'   : 各域取自身 z-score 后取并集 (不加权, 适合持仓拼接)

    返回
    ----
    combined : DataFrame (date×stock) 融合后的统一 composite
    """
    if method == 'equal':
        w = np.ones(len(domain_scores)) / len(domain_scores)
    elif method == 'icir_weight':
        if domain_icir is None:
            w = np.ones(len(domain_scores)) / len(domain_scores)
        else:
            a = np.array([max(0.0, abs(x)) for x in domain_icir])
            s = a.sum()
            w = a / s if s > 0 else np.ones(len(domain_scores)) / len(domain_scores)
    else:  # union_top: 各域独立 z-score 后合并 (求和)
        out = None
        for sc in domain_scores:
            z = (sc - sc.mean(axis=1, skipna=True)) / sc.std(axis=1, skipna=True)
            out = z if out is None else out.add(z, fill_value=0.0)
        return out

    combined = None
    for sc, wi in zip(domain_scores, w):
        sc_w = sc * wi
        combined = sc_w if combined is None else combined.add(sc_w, fill_value=0.0)
    return combined


if __name__ == '__main__':
    # --- 合成数据冒烟测试 ---
    np.random.seed(0)
    dates = pd.date_range('2024-01-01', periods=20, freq='W')
    stocks = [f'S{i:03d}' for i in range(200)]
    idx = pd.Index(dates)
    cols = stocks
    # 收益: 含市值暴露 + 噪声
    size = pd.DataFrame(np.random.randn(20, 200), index=idx, columns=cols)
    true_ret = 0.3 * size + np.random.randn(20, 200) * 0.5
    # DS 高的股票: 给一个额外因子效应
    ds_panel, dom = compute_ds_panel(true_ret, {'size': size}, n_domains=3)
    print('DS panel shape:', ds_panel.shape)
    print('Domain counts per week (last):')
    print(dom.iloc[-1].value_counts().sort_index())
    assert dom.notna().sum().sum() > 0
    print('SMOKE TEST PASSED')
