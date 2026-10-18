# -*- coding: utf-8 -*-
"""
DL Factor Composer v1 — 神经网络因子组合发现
=============================================
用浅层 MLP 自动学习 85+ 因子的最优非线性组合，替代人工穷举 pairwise 乘积。

设计:
  输入: N 个截面标准化因子 (z-score per week)
  模型: 3 层 MLP (N→64→32→16→1) + ReLU + Dropout
  输出: 股票得分 (标量)
  损失: MSE on rank-transformed forward 1w return
  训练: 时间序列 CV (2021-2023 train, 2024 val, 2025-2026 test)
  样本: ~780K (train), 每(周, 股票)为独立样本

与 V3 对比:
  V3 = 穷举 3570 对 rank(A)×rank(B) → ICIR 打分 → 选 5 个 → 加权求和
  DL  = 自动学 N 个因子的高阶交互 + 非线性变换 → 1 个得分

运行时: Python 3.13 + PyTorch (CPU, 低内存)
"""

from __future__ import annotations

import gc
import math
import os
import pickle
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings('ignore')

# ── 路径 ──
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / 'output'
OUTPUT_DIR.mkdir(exist_ok=True)
sys.path.insert(0, str(SCRIPT_DIR))

# ── 配置 ──
SEED = 42
BATCH_SIZE = 8192
LR = 3e-4
EPOCHS = 50
PATIENCE = 10
DROP_OUT = 0.2
HIDDEN = [64, 32, 16]

# 时间划分
TRAIN_END = '2023-12-31'
VAL_END   = '2024-12-31'
# TEST_END implicit: 2026-06

# ── 禁投令因子 (JQ 实盘已证伪)
EXCLUDE_FACTORS = {
    'vm_diff',               # 毒因子
    'bargaining_power_proxy', # JQ 死重
    'relative_spread_proxy',  # JQ 死重
    'avg_turnover_3m',        # JQ 死重
    'min_ret_1m',             # JQ 死重
    'atr_14',                 # JQ 死重
}

# 额外需排除的因子 (形状不兼容, 基本面频次不一致)
EXCLUDE_FACTORS |= {
    # 274 周 (财报频率), 与 286 周量价因子不对齐
    'accruals', 'asset_growth', 'asset_turnover', 'debt_coverage',
    'earnings_growth_yoy', 'earnings_stability', 'gross_margin',
    'net_margin', 'ocf_quality', 'rev_growth_yoy', 'roa', 'roe', 'roic',
    # 股票数偏少
    'dp',
}

torch.manual_seed(SEED)
np.random.seed(SEED)


# ═══════════════════════════════════════════════════════════════════
# 1. 数据加载
# ═══════════════════════════════════════════════════════════════════
def load_data():
    t0 = time.time()
    cache_path = OUTPUT_DIR / 'factor_cache.pkl'
    print(f'[1/5] 加载因子缓存: {cache_path}')
    factor_dfs = pickle.load(open(cache_path, 'rb'))
    print(f'  原始: {len(factor_dfs)} 个因子')

    # 排除禁投令因子
    for ex in EXCLUDE_FACTORS:
        factor_dfs.pop(ex, None)

    # 只保留 (286, ~5700+) 的量价因子 (排除小样本/低频)
    good = {}
    for name, df in factor_dfs.items():
        if df.shape[0] < 280:  # 至少 280 周
            continue
        if df.shape[1] < 4000:  # 至少 4000 股票
            continue
        good[name] = df

    factor_names = sorted(good.keys())
    print(f'  保留: {len(factor_names)} 个因子 (排除 {len(EXCLUDE_FACTORS)} 个)')
    print(f'  因子列表: {factor_names}')

    # 找公共索引 (周) 和公共列 (股票)
    common_index = None
    common_cols = None
    for name in factor_names:
        df = good[name]
        if common_index is None:
            common_index = df.index
            common_cols = set(df.columns)
        else:
            common_index = common_index.intersection(df.index)
            common_cols = common_cols.intersection(set(df.columns))

    common_cols = sorted(common_cols)
    print(f'  公共周数: {len(common_index)} | 公共股票: {len(common_cols)}')
    print(f'  周范围: {common_index[0]} ~ {common_index[-1]}')

    # 构建 3D 数组: (n_weeks, n_stocks, n_factors)
    n_weeks = len(common_index)
    n_stocks = len(common_cols)
    n_factors = len(factor_names)

    X = np.full((n_weeks, n_stocks, n_factors), np.nan, dtype=np.float32)
    print(f'  分配内存: {X.nbytes / 1024 / 1024:.0f} MB')

    col_to_idx = {c: i for i, c in enumerate(common_cols)}
    for fi, name in enumerate(factor_names):
        df = good[name].loc[common_index, common_cols]
        X[:, :, fi] = df.values.astype(np.float32)

    del good, factor_dfs, col_to_idx
    gc.collect()

    print(f'  数据加载完成 | 耗时: {time.time()-t0:.1f}s')
    return X, factor_names, common_index, common_cols


def compute_weekly_returns():
    """从 daily_prices + adj_factor 计算周频收益 (直接读CSV, 不用 load_all_data)."""
    t0 = time.time()
    print('  [returns] 计算周频收益...')
    raw_dir = SCRIPT_DIR.parent.parent / 'data' / 'raw'

    # 只读 close 列 + adj_factor, 跳过 daily_basic/fina
    print('    加载 daily_prices (close only)...')
    price_raw = pd.read_csv(raw_dir / 'daily_prices.csv',
                            usecols=['ts_code', 'trade_date', 'close'],
                            dtype={'ts_code': str, 'trade_date': str})
    price_raw['trade_date'] = pd.to_datetime(price_raw['trade_date'], format='mixed')
    print(f'      {len(price_raw):,} 行')

    print('    加载 adj_factor...')
    adj_raw = pd.read_csv(raw_dir / 'adj_factor.csv',
                          usecols=['ts_code', 'trade_date', 'adj_factor'],
                          dtype={'ts_code': str})
    # 确保日期类型一致
    adj_raw['trade_date'] = pd.to_datetime(adj_raw['trade_date'], format='mixed')
    # 去重
    adj_raw = adj_raw.drop_duplicates(subset=['ts_code', 'trade_date'])

    # merge + 后复权
    merged = price_raw.merge(adj_raw, on=['ts_code', 'trade_date'], how='left')
    merged['adj_factor'] = merged.groupby('ts_code')['adj_factor'].transform(
        lambda x: x.bfill().ffill().fillna(1.0))
    merged['close_hfq'] = merged['close'] * merged['adj_factor']
    # 去重 (数据可能有重复的 ts_code×trade_date)
    before = len(merged)
    merged = merged.drop_duplicates(subset=['ts_code', 'trade_date'])
    if before != len(merged):
        print(f'      去重: {before:,} → {len(merged):,} 行')
    del price_raw, adj_raw
    gc.collect()

    # pivot → weekly close → returns
    close_df = merged.pivot(index='trade_date', columns='ts_code', values='close_hfq')
    del merged
    gc.collect()

    weekly_close = close_df.resample('W').last()
    weekly_ret = weekly_close.pct_change().dropna(how='all')
    forward_ret = weekly_ret.shift(-1)

    print(f'    周频收益: {weekly_ret.shape} | 前向收益: {forward_ret.shape}')
    print(f'    耗时: {time.time()-t0:.1f}s')
    return weekly_ret, forward_ret


# ═══════════════════════════════════════════════════════════════════
# 2. 数据处理
# ═══════════════════════════════════════════════════════════════════
def prepare_dataset(X, factor_names, common_index, common_cols,
                    forward_ret):
    """构建训练/验证/测试数据集."""
    t0 = time.time()
    print(f'\n[2/5] 数据预处理...')

    n_weeks, n_stocks, n_factors = X.shape

    # 对齐周
    fwd_weeks = forward_ret.index
    common_weeks = common_index.intersection(fwd_weeks)
    week_mask = np.array([w in common_weeks for w in common_index])
    X = X[week_mask]
    common_weeks_sorted = sorted(common_weeks)
    print(f'  对齐后周数: {len(common_weeks_sorted)}')

    # 对齐股票
    fwd_cols = set(forward_ret.columns)
    stock_mask = np.array([c in fwd_cols for c in common_cols])
    X = X[:, stock_mask, :]
    aligned_cols = [c for i, c in enumerate(common_cols) if stock_mask[i]]
    n_stocks = len(aligned_cols)
    print(f'  对齐后股票: {n_stocks}')

    # 前向收益矩阵
    Y_raw = forward_ret.loc[common_weeks_sorted, aligned_cols].values.astype(np.float32)

    # 截面标准化: per-week, per-factor z-score
    for w in range(len(common_weeks_sorted)):
        for f in range(n_factors):
            vals = X[w, :, f]
            mask = np.isfinite(vals)
            if mask.sum() < 50:
                X[w, :, f] = 0.0
                continue
            mu = np.nanmean(vals)
            sigma = np.nanstd(vals)
            if sigma < 1e-8:
                X[w, :, f] = 0.0
            else:
                X[w, mask, f] = (vals[mask] - mu) / sigma
                X[w, ~mask, f] = 0.0

    # 收益处理: rank transform within week
    Y_rank = np.zeros_like(Y_raw)
    for w in range(len(common_weeks_sorted)):
        vals = Y_raw[w]
        mask = np.isfinite(vals)
        if mask.sum() < 20:
            continue
        # rank within week [0, 1]
        valid_vals = vals[mask]
        ranks = pd.Series(valid_vals).rank(pct=True).values
        Y_rank[w, mask] = ranks.astype(np.float32)

    # 划分数据
    week_dates = pd.to_datetime(common_weeks_sorted)
    train_idx = np.where(week_dates <= TRAIN_END)[0]
    val_idx   = np.where((week_dates > TRAIN_END) & (week_dates <= VAL_END))[0]
    test_idx  = np.where(week_dates > VAL_END)[0]

    print(f'  训练: {len(train_idx)} 周 | 验证: {len(val_idx)} 周 | 测试: {len(test_idx)} 周')
    print(f'  训练样本: {len(train_idx) * n_stocks / 1e6:.1f}M')

    # 构建 numpy arrays
    X_train = X[train_idx].reshape(-1, n_factors)
    Y_train = Y_rank[train_idx].reshape(-1)
    X_val   = X[val_idx].reshape(-1, n_factors)
    Y_val   = Y_rank[val_idx].reshape(-1)
    X_test  = X[test_idx].reshape(-1, n_factors)
    Y_test  = Y_rank[test_idx].reshape(-1)

    # 过滤 NaN
    def filter_nan(x, y):
        mask = np.isfinite(x).all(axis=1) & np.isfinite(y)
        return x[mask], y[mask]

    X_train, Y_train = filter_nan(X_train, Y_train)
    X_val, Y_val     = filter_nan(X_val, Y_val)
    X_test, Y_test   = filter_nan(X_test, Y_test)

    print(f'  训练: {X_train.shape[0]/1e3:.0f}K 样本 | '
          f'验证: {X_val.shape[0]/1e3:.0f}K | '
          f'测试: {X_test.shape[0]/1e3:.0f}K')
    print(f'  耗时: {time.time()-t0:.1f}s')

    return (X_train, Y_train, X_val, Y_val, X_test, Y_test,
            n_factors, week_dates, train_idx, val_idx, test_idx,
            aligned_cols, X, Y_raw)


# ═══════════════════════════════════════════════════════════════════
# 3. 模型定义
# ═══════════════════════════════════════════════════════════════════
class FactorMLP(nn.Module):
    """浅层 MLP: n_factors → H[0] → H[1] → H[2] → 1"""

    def __init__(self, n_in: int, hidden: list[int], dropout: float = 0.2):
        super().__init__()
        layers = []
        prev = n_in
        for h in hidden:
            layers.extend([
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def spearman_loss(y_pred, y_true):
    """可微 Spearman 近 (Pearson on rank)."""
    # Pearson = cov / (std_x * std_y)
    x = y_pred - y_pred.mean()
    y = y_true - y_true.mean()
    cov = (x * y).mean()
    sx = x.std() + 1e-8
    sy = y.std() + 1e-8
    return -cov / (sx * sy)


# ═══════════════════════════════════════════════════════════════════
# 4. 训练
# ═══════════════════════════════════════════════════════════════════
def train_model(X_train, Y_train, X_val, Y_val, n_factors, device):
    """训练 MLP."""
    t0 = time.time()
    print(f'\n[3/5] 训练 DL 因子组合模型...')
    print(f'  架构: {n_factors}→{HIDDEN}→1 | Dropout={DROP_OUT} | LR={LR:.0e} | BS={BATCH_SIZE}')

    model = FactorMLP(n_factors, HIDDEN, DROP_OUT).to(device)
    print(f'  参数量: {sum(p.numel() for p in model.parameters()):,}')

    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(Y_train, dtype=torch.float32))
    val_ds = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(Y_val, dtype=torch.float32))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE * 2, shuffle=False,
                              num_workers=0, pin_memory=False)

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        n_batches = 0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            pred = model(bx)
            loss = criterion(pred, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_loss += loss.item()
            n_batches += 1

        scheduler.step()
        train_loss /= max(n_batches, 1)

        # 验证
        model.eval()
        val_loss = 0.0
        val_spearman = 0.0
        n_val = 0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                pred = model(bx)
                val_loss += criterion(pred, by).item()
                val_spearman += -spearman_loss(pred, by).item()
                n_val += 1

        val_loss /= max(n_val, 1)
        val_spearman /= max(n_val, 1)

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            # 保存最佳模型
            torch.save(model.state_dict(),
                       str(OUTPUT_DIR / 'dl_composer_v1.pth'))
        else:
            patience_counter += 1

        if epoch == 1 or epoch % 5 == 0 or epoch == EPOCHS:
            print(f'  Epoch {epoch:3d}/{EPOCHS} | '
                  f'Train Loss: {train_loss:.6f} | '
                  f'Val Loss: {val_loss:.6f} | '
                  f'Val Spearman: {val_spearman:.4f} | '
                  f'LR: {scheduler.get_last_lr()[0]:.2e}')

        if patience_counter >= PATIENCE:
            print(f'  早停 @ Epoch {epoch} (best={best_epoch})')
            break

    # 恢复最佳模型
    model.load_state_dict(torch.load(str(OUTPUT_DIR / 'dl_composer_v1.pth'),
                                     weights_only=True))
    print(f'  训练完成 | 最佳 Epoch: {best_epoch} | 耗时: {time.time()-t0:.1f}s')
    return model


# ═══════════════════════════════════════════════════════════════════
# 5. 评估
# ═══════════════════════════════════════════════════════════════════
def evaluate(model, X_test, Y_test, device):
    """测试集评估."""
    print(f'\n[4/5] 测试集评估...')
    model.eval()
    with torch.no_grad():
        x_t = torch.tensor(X_test, dtype=torch.float32).to(device)
        y_t = torch.tensor(Y_test, dtype=torch.float32).to(device)
        pred = model(x_t)
        mse = nn.MSELoss()(pred, y_t).item()
        sp = -spearman_loss(pred, y_t).item()
    print(f'  Test MSE: {mse:.6f}')
    print(f'  Test Spearman ρ: {sp:.4f}')
    return sp


# ═══════════════════════════════════════════════════════════════════
# 6. 组合回测
# ═══════════════════════════════════════════════════════════════════
def portfolio_backtest(model, X_data, Y_raw, week_dates,
                       test_idx, aligned_cols, device, n_factors):
    """在测试集上做 Top-30 周频组合回测."""
    print(f'\n[5/5] 组合回测...')

    TOP_N = 30
    COST = 0.00144  # 周频双边成本 ~14.4bp

    model.eval()
    portfolio_returns = []
    dates = []

    test_weeks = week_dates[test_idx]
    print(f'  测试区间: {test_weeks[0].date()} ~ {test_weeks[-1].date()}')

    for i, w in enumerate(test_idx):
        if w >= len(X_data) - 1:
            break
        week_x = X_data[w]  # (n_stocks, n_factors)
        week_y = Y_raw[w + 1] if w + 1 < len(Y_raw) else Y_raw[w]

        # 筛选有效样本
        valid = np.isfinite(week_x).all(axis=1) & np.isfinite(week_y)
        if valid.sum() < TOP_N:
            continue

        # 预测得分
        x_t = torch.tensor(week_x[valid], dtype=torch.float32).to(device)
        with torch.no_grad():
            scores = model(x_t).cpu().numpy()

        # 选 Top N
        idx = np.argsort(scores)[-TOP_N:]
        selected_ret = week_y[valid][idx]

        # 等权组合收益 (扣除成本)
        port_ret = np.nanmean(selected_ret) - COST
        portfolio_returns.append(port_ret)
        dates.append(week_dates[w])

    if not portfolio_returns:
        print('  ⚠️ 无有效回测数据')
        return

    rets = pd.Series(portfolio_returns, index=dates, name='DL_Composite')

    # 统计
    annual_ret = (1 + rets.mean()) ** 52 - 1
    annual_vol = rets.std() * math.sqrt(52)
    sharpe = annual_ret / annual_vol if annual_vol > 0 else 0
    cumulative = (1 + rets).cumprod()
    max_dd = (cumulative / cumulative.cummax() - 1).min()

    print(f'  周频回测 (Top {TOP_N}, 等权, 扣成本 {COST*10000:.0f}bp)')
    print(f'  年化收益: {annual_ret*100:+.1f}%')
    print(f'  年化波动: {annual_vol*100:.1f}%')
    print(f'  Sharpe:   {sharpe:.2f}')
    print(f'  MaxDD:    {max_dd*100:.1f}%')
    print(f'  累计收益: {(cumulative.iloc[-1]-1)*100:+.1f}%')

    # 保存回测序列
    rets.to_csv(OUTPUT_DIR / 'dl_composer_v1_returns.csv', header=True)
    print(f'  收益序列已保存 → output/dl_composer_v1_returns.csv')

    return {
        'annual_ret': float(annual_ret),
        'sharpe': float(sharpe),
        'max_dd': float(max_dd),
        'n_weeks': len(rets),
    }


# ═══════════════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════════════
def main():
    print('=' * 60)
    print('  DL Factor Composer v1')
    print('  神经网络因子组合发现')
    print(f'  时间: {pd.Timestamp.now()}')
    print('=' * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'  设备: {device}')

    # 1. 加载数据
    X, factor_names, common_index, common_cols = load_data()
    weekly_ret, forward_ret = compute_weekly_returns()
    print(f'  周频收益: {weekly_ret.shape} | 前向收益: {forward_ret.shape}')

    # 2. 预处理
    (X_train, Y_train, X_val, Y_val, X_test, Y_test,
     n_factors, week_dates, train_idx, val_idx, test_idx,
     aligned_cols, X_data, Y_raw) = prepare_dataset(
         X, factor_names, common_index, common_cols, forward_ret)

    # 3. 训练
    model = train_model(X_train, Y_train, X_val, Y_val, n_factors, device)

    # 4. 评估
    test_sp = evaluate(model, X_test, Y_test, device)

    # 5. 回测
    bt_stats = portfolio_backtest(
        model, X_data, Y_raw, week_dates,
        test_idx, aligned_cols, device, n_factors)

    # ── 因子重要性 (简单线性近似) ──
    print(f'\n[因子重要性] 输入权重 (第一层线性系数 L1 norm):')
    with torch.no_grad():
        w1 = model.net[0].weight.abs().sum(dim=0).cpu().numpy()
        top_idx = np.argsort(w1)[-20:][::-1]
        for rank, idx in enumerate(top_idx, 1):
            print(f'  {rank:2d}. {factor_names[idx]:35s} | score={w1[idx]:.3f}')

    # ── 保存结果 ──
    result = {
        'model': 'DL_Factor_Composer_v1',
        'n_factors': n_factors,
        'factor_names': factor_names,
        'hidden': HIDDEN,
        'dropout': DROP_OUT,
        'test_spearman': float(test_sp),
    }
    if bt_stats:
        result.update(bt_stats)

    import json
    with open(OUTPUT_DIR / 'dl_composer_v1_result.json', 'w') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f'\n✓ 完成 | 模型: output/dl_composer_v1.pth | 结果: output/dl_composer_v1_result.json')


if __name__ == '__main__':
    main()
