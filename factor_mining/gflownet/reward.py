"""
GFlowNet reward 函数
====================
reward 与两目标 NSGA-II 的收益口径一致（IC 均值主导复合收益），
并额外加「softplus 软门 × 强换手惩罚 × 表达式复杂度惩罚」:
  score_ret       = ic_strength_return_score × softplus_gate(ic_std)
                    (IC 均值主导 + softplus 软门平滑惩罚低 ic_std 伪信号, 替代硬阈值)
  score_turnover  = clip(1 - TURNOVER_PENALTY × turnover, 0, 1)   (强换手惩罚)
  score_complexity= complexity_score(n_tokens)   (分段: 罚 n=1 原始单因子与 >7 深树, 奖励 2~5)
  score           = score_ret × score_turnover × score_complexity
  R(x)            = exp(beta × score)            (beta 放大区分度, 缓解 reward 稀疏)

表达式求值 → evaluate_composite（同一评估函数）→ 映射为正 reward。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from ..data_loader import evaluate_composite, ic_strength_return_score, softplus_gate
except ImportError:  # 兼容平铺脚本运行
    from data_loader import evaluate_composite, ic_strength_return_score, softplus_gate

DEFAULT_BETA = 8.0
TURNOVER_PENALTY = 2.0     # turnover=0.5 时换手分归 0；turnover=0.16 → 0.68（原口径 0.84）
TYPE_PENALTY = 0.1         # 每个 soft 类型违规乘 exp(-0.1)：1 个扣 ~10%、2 个 ~18%（温和，不误杀合法交互）


def complexity_score(n_tokens: int) -> float:
    """分段复杂度分（2026-10-21 修订，替代旧的 exp(-0.3×(n-1)) 单调"越简单越高"）。

    旧式单调惩罚把策略推向 n_tokens=1（直接吐回最强原始单因子），62% 样本退化成
    单叶子 → E2 独立性判冗余，GFlowNet 产出近乎全灭。修订为**只罚退化与过深**：
      - n <= 1        : 0.30  原始单因子（吐回池因子，E2 判冗余）→ 重罚
      - 2 <= n <= 5   : 1.00  奖励区间（真复合式，E2 独立性高）
      - 6 <= n <= 7   : 0.70  温和（接近上限，不鼓励但允许）
      - n >= 8        : 0.30  过深（不可解释/过拟合）→ 重罚
    刻意**不整体反向**（避免单调奖励复杂度 → 推向 max_len 深树 / 灌水凑 token）。
    """
    if n_tokens <= 1:
        return 0.30
    if n_tokens <= 5:
        return 1.00
    if n_tokens <= 7:
        return 0.70
    return 0.30


def _type_violations(grammar, token_ids):
    """返回 (hard, soft) 类型违规列表。

    - FUXI_TYPE_GATE=on：hard 违规 → 编译拒绝（R=0），off 时不查 hard；
    - FUXI_TYPE_SOFT_PENALTY=on：soft 违规 → 乘性软惩罚，off 时不查 soft；
    - 两者都 off 时直接返回 ([], [])，训练行为逐位不变（A/A 纪律）。

    复用 fuxi_v1.types_core 的 preorder 前端推断（同一规则表），保证 factor_mining
    与 fuxi_v1 两侧类型纪律一致。
    """
    gate = os.environ.get("FUXI_TYPE_GATE", "off") == "on"
    soft_penalty = os.environ.get("FUXI_TYPE_SOFT_PENALTY", "off") == "on"
    if not gate and not soft_penalty:
        return [], []
    try:
        from fuxi_v1.types_core import infer_prefix, hard_violations, soft_violations, resolve
    except ImportError:
        # factor_mining 独立运行时，把 types_core 的父目录（e:/quant）加入 path
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from fuxi_v1.types_core import infer_prefix, hard_violations, soft_violations, resolve
    # token_ids 是 int id 序列，先经 id2token 转回字符串（叶子=因子名，算子=算子名）
    tokens = [grammar.id2token[t] for t in token_ids]
    ops = set(grammar.binary + grammar.unary)
    _, vs = infer_prefix(tokens, lambda n: resolve(n, kind="factor"), lambda t: t in ops)
    hard = hard_violations(vs) if gate else []
    soft = soft_violations(vs) if soft_penalty else []
    return hard, soft


def expression_to_dataframe(comp: np.ndarray, forward_returns: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(comp, index=forward_returns.index, columns=forward_returns.columns)


def compute_reward(grammar, token_ids, forward_returns, beta=DEFAULT_BETA):
    """给定一个完整表达式（token_ids），返回标量 reward ≥ 0。"""
    if not token_ids:
        return 0.0  # 空序列 = 采样失败（不完整表达式），R=0
    # 类型前置：gate=on 时语义非法表达式直接 R=0（与求值失败同路径，训练循环零改动）。
    hard, soft = _type_violations(grammar, token_ids)
    if hard:
        return 0.0
    comp = grammar.evaluate(token_ids)
    if comp is None or not np.isfinite(comp).any():
        return 0.0
    comp_df = expression_to_dataframe(comp, forward_returns)
    r = evaluate_composite(comp_df, forward_returns)
    if r is None:
        return 0.0
    score_ret = ic_strength_return_score(r) * softplus_gate(r["ic_std"])
    score_turnover = float(np.clip(1.0 - TURNOVER_PENALTY * r["turnover"], 0.0, 1.0))
    score_complexity = complexity_score(len(token_ids))
    # P2 软惩罚：soft 违规（跨 family 相加、horizon 跨档、序数参与算术等）温和乘性压低，
    # 只影响 reward 面，不硬拒 —— 可疑组合被平滑降权，留给 E 层与训练信号裁决。
    score_type = float(np.exp(-TYPE_PENALTY * len(soft)))
    score = score_ret * score_turnover * score_complexity * score_type
    if score <= 0:
        return 0.0
    return float(np.exp(beta * score))


def compute_full_metrics(grammar, token_ids, forward_returns):
    """完整指标（后处理用）：返回 evaluate_composite 的完整 dict + 表达式字符串。"""
    comp = grammar.evaluate(token_ids)
    comp_df = expression_to_dataframe(comp, forward_returns)
    r = evaluate_composite(comp_df, forward_returns)
    return r
