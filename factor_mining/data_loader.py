"""
共享数据加载 + 统一评估口径
============================
两目标 NSGA-II 与 GFlowNet 两个方法共用同一份数据与同一套评估口径，
确保对比结果只反映「优化器」差异，而非「数据/口径」差异。

数据源（均为现成缓存，秒级加载，不重算因子）:
  - e:/quant/research/factor_alchemy/output/factor_cache.pkl   (dict: 97因子, 周×股)
  - e:/quant/research/factor_alchemy/output/forward_returns.pkl (下周收益, 周×股)

统一评估口径 evaluate_composite:
  - OOS 70:30 时间切分
  - 方向自动检测（IC 均值符号）
  - Top30 等权，算三个指标: ICIR / 毛Sharpe / 周换手率
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

import os

# 数据路径可配置：优先读环境变量（供每日自动化/不同数据口径覆盖），
# 默认指向 factor_alchemy 的现成缓存（秒级加载，不重算因子）。
FACTOR_ALCHEMY = Path(os.environ.get("FUXI_FACTOR_ALCHEMY", "e:/quant/research/factor_alchemy"))
FACTOR_CACHE = Path(os.environ.get(
    "FUXI_FACTOR_CACHE", str(FACTOR_ALCHEMY / "output" / "factor_cache.pkl")))
FORWARD_RETURNS = Path(os.environ.get(
    "FUXI_FORWARD_RETURNS", str(FACTOR_ALCHEMY / "output" / "forward_returns.pkl")))

TOP_N = 30
OOS_SPLIT = 0.70          # 前 70% 训练 / 后 30% 外样本
ANNUALIZE = 52            # 周频年化


# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------
def load_factor_dict() -> dict[str, pd.DataFrame]:
    with open(FACTOR_CACHE, "rb") as f:
        return pickle.load(f)


def load_forward_returns() -> pd.DataFrame:
    return pd.read_pickle(FORWARD_RETURNS)


def load_aligned() -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """加载并做完整行列对齐（不同因子日期/股票覆盖可能不一致，财务因子滞后）。"""
    factor_dict = load_factor_dict()
    fr = load_forward_returns()
    # 行（日期）对齐：所有因子 + 前向收益的 index 交集
    common_idx = fr.index
    for df in factor_dict.values():
        common_idx = common_idx.intersection(df.index)
    # 列（股票）对齐
    common_cols = set(fr.columns)
    for df in factor_dict.values():
        common_cols &= set(df.columns)
    common_cols = sorted(common_cols)
    fr = fr.loc[common_idx, common_cols]
    aligned = {k: v.loc[common_idx, common_cols] for k, v in factor_dict.items() if not v.empty}
    return aligned, fr


# ---------------------------------------------------------------------------
# 统一评估口径
# ---------------------------------------------------------------------------
def _cross_sectional_ic(comp: np.ndarray, fr: np.ndarray, min_stocks: int = 50) -> np.ndarray:
    """逐周截面 Spearman rank IC（float64 精算）。comp/fr 形状 (n_weeks, n_stocks)。"""
    n_weeks, n_stocks = comp.shape
    ic_vals = np.full(n_weeks, np.nan)
    for w in range(n_weeks):
        c = comp[w]
        f = fr[w]
        valid = np.isfinite(c) & np.isfinite(f)
        if valid.sum() < min_stocks:
            continue
        rk_c = rankdata(c[valid])
        rk_f = rankdata(f[valid])
        n = len(rk_c)
        d2 = np.sum((rk_c - rk_f) ** 2)
        ic_vals[w] = 1.0 - 6.0 * d2 / (n * (n * n - 1))
    return ic_vals


def evaluate_composite(
    comp: pd.DataFrame,
    forward_returns: pd.DataFrame,
    top_n: int = TOP_N,
    oos_split: float = OOS_SPLIT,
) -> dict | None:
    """
    统一评估：对截面因子值 comp 做方向感知的 Top-N 等权回测口径。

    返回 dict（均基于 OOS 段）:
      icir            : 截面 Rank IC 的均值/标准差
      ic_mean         : IC 均值
      gross_sharpe    : 毛年化 Sharpe（未扣成本）
      turnover        : 周换手率（Top-N 持仓 overlap 口径）
      direction       : +1 做多 / -1 做空（由 IC 符号决定）
      n_weeks         : 有效周数
      score_ret       : 收益目标（毛Sharpe clip 到 [0,1]）
      score_turnover  : 换手目标（1 - turnover）
    """
    if comp is None or comp.empty:
        return None
    common_idx = comp.index.intersection(forward_returns.index)
    common_cols = comp.columns.intersection(forward_returns.columns)
    if len(common_idx) < 40 or len(common_cols) < 50:
        return None

    C = comp.loc[common_idx, common_cols].astype(np.float64).values
    F = forward_returns.loc[common_idx, common_cols].astype(np.float64).values
    n_weeks, n_stocks = C.shape

    # 70:30 OOS 时间切分
    split = int(n_weeks * oos_split)
    if n_weeks - split < 20:
        split = n_weeks - 20
    C_oos = C[split:]
    F_oos = F[split:]

    # 方向检测（OOS 段 IC 均值符号）
    ic_all = _cross_sectional_ic(C_oos, F_oos)
    avg_ic = np.nanmean(ic_all) if np.isfinite(ic_all).any() else 0.0
    direction = 1 if avg_ic >= 0 else -1

    # 截面 rank 后选 Top-N（方向感知）
    n_oos = C_oos.shape[0]
    top_idx = np.zeros((n_oos, top_n), dtype=int)
    for w in range(n_oos):
        row = C_oos[w].copy()
        row[~np.isfinite(row)] = -np.inf
        order = np.argsort(-row) if direction >= 0 else np.argsort(row)
        top_idx[w] = order[:top_n]

    # 毛收益（下周收益等权；停牌导致全 NaN 的周跳过）
    gross_rets = np.full(n_oos - 1, np.nan)
    for w in range(n_oos - 1):
        row = F_oos[w, top_idx[w]]
        if np.isfinite(row).sum() < 5:
            continue
        gross_rets[w] = np.nanmean(row)

    # 换手（Top-N 持仓 overlap 口径）
    mask = np.zeros((n_oos, n_stocks), dtype=bool)
    for w in range(n_oos):
        mask[w, top_idx[w]] = True
    overlap = (mask[:-1] & mask[1:]).sum(axis=1)
    turnovers = 1.0 - overlap / top_n

    gross = gross_rets[np.isfinite(gross_rets)]
    if len(gross) < 15:
        return None
    mean_g = np.mean(gross)
    std_g = np.std(gross, ddof=1)
    gross_sharpe = (mean_g / std_g * np.sqrt(ANNUALIZE)) if std_g > 1e-9 else 0.0

    ic_f = ic_all[np.isfinite(ic_all)]
    ic_mean = np.mean(ic_f) if len(ic_f) else 0.0
    ic_std = np.std(ic_f, ddof=1) if len(ic_f) > 1 else 1.0
    icir = ic_mean / ic_std if ic_std > 1e-9 else 0.0

    turnover = float(np.nanmean(turnovers))

    score_ret = float(np.clip(gross_sharpe / 2.0, 0.0, 1.0))
    score_turnover = float(np.clip(1.0 - turnover, 0.0, 1.0))

    return {
        "icir": float(icir),
        "ic_mean": float(ic_mean),
        "ic_std": float(ic_std),
        "gross_sharpe": float(gross_sharpe),
        "turnover": turnover,
        "direction": direction,
        "n_weeks": len(gross),
        "score_ret": score_ret,
        "score_turnover": score_turnover,
    }


# ---------------------------------------------------------------------------
# IC 均值主导的复合收益分（两方法共用，保证收益口径一致）
# ---------------------------------------------------------------------------
IC_MEAN_SCALE = 0.05    # |ic_mean| = 5% 记满分（原始 97 因子 p75≈5%，有区分度）
SHARPE_SCALE = 2.0      # 毛Sharpe = 2.0 记满分
W_IC = 0.7              # IC 均值权重（主导）；毛Sharpe 辅助保留 Top30 尾部能力


def ic_strength_return_score(r: dict, w_ic: float = W_IC) -> float:
    """IC 均值主导的复合收益分。

    - score_ic = clip(|ic_mean|/IC_MEAN_SCALE, 0, 1)：截面 IC 均值强度，
      直接度量「全截面单调排序能力」。
      🔴 用 ic_mean 而非 icir=ic_mean/ic_std —— 线性加权可构造 ic_std≈0 的组合
      把 icir 除到无穷大（伪 ICIR），GA 追 |icir| 会全收敛到这些稳定反向的伪信号
      （实测：改 |ICIR| 主导后 NSGA-II median ICIR 反恶化到 -2.05、正占比 0%）。
      方向由 direction 检测在 evaluate_composite 内自动翻转，故取绝对值。
    - score_ret = clip(gross_sharpe/SHARPE_SCALE, 0, 1)：毛Sharpe 辅助，
      保留 Top30 尾部选股能力（实盘 Top-K 选股场景需要）。
    - 返回 w_ic*score_ic + (1-w_ic)*score_ret ∈ [0,1]
    """
    score_ic = float(np.clip(abs(r["ic_mean"]) / IC_MEAN_SCALE, 0.0, 1.0))
    score_ret = float(np.clip(r["gross_sharpe"] / SHARPE_SCALE, 0.0, 1.0))
    return w_ic * score_ic + (1.0 - w_ic) * score_ret


# ---------------------------------------------------------------------------
# 贝叶斯收缩 ICIR + softplus 软门（替代硬 ic_std 阈值的一刀切）
# ---------------------------------------------------------------------------
# ICIR=ic_mean/ic_std 的爆炸来自分母 std→0。硬阈值 ic_std>=0.04 是「外科手术式一刀切」，
# 这里用两个算法层等价物替代：
#   (1) 贝叶斯收缩分母：ic_mean/(ic_std+λ_prior)，λ_prior 数据驱动（原始因子池 ic_std
#       最小值），天然软下界——ic_std→0 时 icir→ic_mean/λ，不再爆炸，且连续可微；
#   (2) softplus 软门：reward 里的乘性软惩罚，低 ic_std 被平滑压低，无 if-else 硬跳变。
LAMBDA_PRIOR = 0.047   # 原始 97 因子 ic_std 最小值 0.0469（经验贝叶斯先验）


def shrinkage_icir(ic_mean: float, ic_std: float, lambda_prior: float = LAMBDA_PRIOR) -> float:
    """贝叶斯收缩 ICIR = ic_mean/(ic_std+λ)。分母加数据驱动先验，天然软下界，替代硬阈值。"""
    return ic_mean / (ic_std + lambda_prior)


def softplus_gate(ic_std: float, k: float = 10.0, c: float = 0.03) -> float:
    """softplus 软门 ∈ (0,1]：ic_std 远高于 c 时≈1，接近/低于 c 时平滑下降。

    用于 reward 的乘性软惩罚（低 ic_std 伪信号被平滑压低），替代 if-else 硬跳变。
    归一化到 ic_std=0.12（原始因子中位数）处 gate=1，并 clip 到 ≤1（不奖励高波动）。
    """
    ref = np.log1p(np.exp(k * (0.12 - c)))
    return float(min(1.0, np.log1p(np.exp(k * (ic_std - c))) / ref))


# ---------------------------------------------------------------------------
# 因子组合辅助（NSGA-II 用线性 rank 加权；GFlowNet 用表达式树求值）
# ---------------------------------------------------------------------------
def cross_sectional_rank(df: pd.DataFrame) -> pd.DataFrame:
    """截面百分位 rank → [0,1]（float64）。"""
    return df.rank(axis=1, pct=True).astype(np.float64)


def linear_combine(factor_dict: dict[str, pd.DataFrame], weights: dict[str, float]) -> pd.DataFrame:
    """截面 rank 到 [0,1] 后的加权平均组合。weights 已归一化。"""
    parts = []
    for name, w in weights.items():
        if name not in factor_dict or abs(w) < 1e-9:
            continue
        parts.append(cross_sectional_rank(factor_dict[name]) * w)
    if not parts:
        return pd.DataFrame()
    return sum(parts)
