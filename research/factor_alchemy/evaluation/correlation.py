"""
因子相关性分析 (向量化加速版)
"""
import numpy as np
import pandas as pd


def factor_correlation_matrix(factor_dict, method='spearman'):
    """
    计算因子间相关性矩阵（向量化，极快）
    
    方法: 逐截面做截面秩，然后用 np.corrcoef 批量算相关，跨时间平均。
    避免 scipy.spearmanr 的大矩阵开销。
    """
    factor_names = list(factor_dict.keys())
    n_factors = len(factor_names)
    if n_factors < 2:
        return pd.DataFrame()

    # 1. 找公共日期
    all_dates = sorted(set.intersection(*[set(df.index) for df in factor_dict.values()])
                       if factor_dict else set())
    if not all_dates:
        return pd.DataFrame()

    corr_accum = np.zeros((n_factors, n_factors))
    n_periods = 0

    for date in all_dates:
        # 2. 收集该截面所有因子 → 矩阵 (stocks × factors)
        rows = []
        valid_names = []
        for name in factor_names:
            df = factor_dict[name]
            if date not in df.index:
                continue
            row = df.loc[date]
            if row.notna().sum() < 30:
                continue
            rows.append(row)
            valid_names.append(name)

        if len(rows) < 2:
            continue

        # 对齐所有行的索引 (不同因子可能有不同的有效股票集)
        common_idx = rows[0].index
        for r in rows[1:]:
            common_idx = common_idx.intersection(r.index)
        if len(common_idx) < 30:
            continue
        aligned_rows = [r.reindex(common_idx).values for r in rows]
        
        # 检查所有行长度是否一致
        lengths = [len(arr) for arr in aligned_rows]
        if len(set(lengths)) > 1:
            # 截断到最短长度
            min_len = min(lengths)
            aligned_rows = [arr[:min_len] for arr in aligned_rows]
        
        try:
            data = np.array(aligned_rows).T  # (stocks, k)
        except (ValueError, Exception):
            continue
        if data.shape[0] < 30:
            continue

        # 3. 筛掉全 NaN 的行 → 填充为 0 (排名时 NaN 自动处理)
        valid_rows = ~np.all(np.isnan(data), axis=1)
        data = data[valid_rows]
        if len(data) < 30:
            continue

        # 4. 截面排名 (Spearman = rank 后的 Pearson)
        if method == 'spearman':
            # nanrank: rank ignoring NaN, NaN stays
            ranked = np.full_like(data, np.nan)
            for j in range(data.shape[1]):
                col = data[:, j]
                mask = ~np.isnan(col)
                ranked[mask, j] = (pd.Series(col[mask]).rank() - 1).values
            data = ranked
            del ranked

        # 5. 删除仍有 NaN 的行 (股票在某些因子上无值)
        complete = ~np.any(np.isnan(data), axis=1)
        data_complete = data[complete]
        del data
        if len(data_complete) < 30:
            continue

        # 6. 批量算相关 (np.corrcoef, C-加速)
        corr = np.corrcoef(data_complete.T)

        # 7. 映射回全因子矩阵
        if len(valid_names) == n_factors:
            corr_accum += corr
            n_periods += 1
        else:
            idx_map = [factor_names.index(n) for n in valid_names]
            for i, ni in enumerate(idx_map):
                for j, nj in enumerate(idx_map):
                    corr_accum[ni, nj] += corr[i, j]
            n_periods += 1

    if n_periods == 0:
        return pd.DataFrame()

    avg_corr = corr_accum / n_periods
    result = pd.DataFrame(avg_corr, index=factor_names, columns=factor_names)
    # values 在 numpy 2.x 可能只读, 用逐元素赋值
    for i in range(len(result)):
        result.iloc[i, i] = 1.0
    return result


def check_multicollinearity(corr_matrix, threshold=0.7):
    """检查多重共线性"""
    if corr_matrix.empty:
        return []
    high_corr_pairs = []
    n = len(corr_matrix)
    names = corr_matrix.index.tolist()
    for i in range(n):
        for j in range(i + 1, n):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > threshold:
                high_corr_pairs.append((names[i], names[j], corr_val))
    return sorted(high_corr_pairs, key=lambda x: abs(x[2]), reverse=True)


def compute_vif(corr_matrix):
    """计算方差膨胀因子"""
    if corr_matrix.empty:
        return {}
    try:
        inv_corr = np.linalg.inv(corr_matrix.values)
        vif = np.diag(inv_corr)
        return {name: vif[i] for i, name in enumerate(corr_matrix.index)}
    except Exception:
        return {}
