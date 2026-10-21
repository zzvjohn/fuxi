"""
GFlowNet 训练 + 采样
====================
轨迹平衡(TB)训练：采样轨迹 → 算 reward → TB 损失 → 更新前向策略与 logZ。
训练后采样一批表达式（去重），输出多样化候选因子。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

from .grammar import Grammar, sequence_to_string
from .model import GFlowNetPolicy
from .reward import compute_reward, compute_full_metrics, TURNOVER_PENALTY, complexity_score

try:
    from ..data_loader import ic_strength_return_score
except ImportError:  # 兼容平铺脚本运行
    from data_loader import ic_strength_return_score


def _type_mask_enabled():
    return os.environ.get("FUXI_TYPE_MASK", "off") == "on"


def _types_core():
    """懒加载 types_core 模块（factor_mining 独立运行时加 sys.path 兜底）。"""
    try:
        from fuxi_v1 import types_core
    except ImportError:
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from fuxi_v1 import types_core
    return types_core


def _make_type_state(grammar):
    """构造 GFlowNet preorder 采样的增量类型状态（FUXI_TYPE_MASK=on 时用）。"""
    tc = _types_core()
    ops = set(grammar.binary + grammar.unary)
    return tc.TypeState(lambda n: tc.resolve(n, kind="factor"), lambda t: t in ops)


def _apply_type_mask(type_state, grammar, acts):
    """用类型掩码收窄合法动作集：acts（token id）→ 类型合法子集（token id）。

    只收窄、不改变已完成轨迹的 on-policy 语义（采样/训练共用同一掩码，见 R-A）。
    """
    tc = _types_core()
    tokens = [grammar.id2token[i] for i in acts]
    legal_tokens = tc.mask_actions(type_state, tokens)
    return [grammar.token2id[t] for t in legal_tokens]


def sample_trajectory(policy, grammar, greedy=False):
    """采样一个完整表达式轨迹，返回 (token_ids, log_prob_sum)。"""
    seq = []
    log_p_sum = 0.0
    slots = 1
    length = 0
    max_len = grammar.max_len
    # P3 类型动作掩码（FUXI_TYPE_MASK=on 时用类型栈收窄动作集）
    type_state = _make_type_state(grammar) if _type_mask_enabled() else None
    while slots > 0 and length < max_len:
        acts = grammar.valid_actions(slots, length)
        if not acts:
            break
        if type_state is not None:
            masked = _apply_type_mask(type_state, grammar, acts)
            # 死锁保护：掩码收窄后若为空（如 add(rank(X),?) 第二槽期望 ranked 但叶子无
            # ranked form、且 length 接近 max_len 已无算子可选），回退到未掩码集，
            # 避免采样提前终止产生不完整序列。违规由 compute_reward 的 type_check R=0 兜底。
            if masked:
                acts = masked
        logits = policy.next_logits(torch.tensor(seq, dtype=torch.long))
        mask = torch.full((policy.n_tokens,), float("-inf"))
        mask[acts] = 0.0
        logits = logits + mask
        probs = F.softmax(logits, dim=0)
        if greedy:
            a = int(torch.argmax(probs))
        else:
            a = int(torch.multinomial(probs, 1))
        log_p_sum = log_p_sum + torch.log(probs[a] + 1e-12)  # 保持 tensor 以保留梯度
        seq.append(a)
        length += 1
        if type_state is not None:
            type_state.step(grammar.id2token[a])
        if a >= grammar.leaf_start:
            slots -= 1
        elif a < grammar.n_binary:
            slots += 1
        # 一元算子 slots 不变
    if slots > 0:
        # 不完整序列（length 到上限但槽未填满）：返回空，视为采样失败（R=0）。
        return [], log_p_sum
    return seq, log_p_sum


def train(policy, grammar, forward_returns, n_episodes=300, batch_size=16,
          lr=1e-3, beta=8.0, seed=0, verbose=True):
    torch.manual_seed(seed)
    np.random.seed(seed)
    opt = Adam(policy.parameters(), lr=lr)

    for ep in range(n_episodes):
        batch = [sample_trajectory(policy, grammar) for _ in range(batch_size)]
        rewards = []
        for seq, _ in batch:
            rewards.append(compute_reward(grammar, seq, forward_returns, beta))
        rewards = torch.tensor(rewards, dtype=torch.float32)

        logR = torch.log(rewards.clamp(min=1e-8))
        logp = torch.stack([lp for _, lp in batch])  # 保留梯度
        loss = ((policy.logZ + logp - logR) ** 2).mean()

        opt.zero_grad()
        loss.backward()
        opt.step()

        if verbose and (ep % 20 == 0 or ep == n_episodes - 1):
            print(f"  ep {ep+1}/{n_episodes} | loss={loss.item():.3f} "
                  f"logZ={policy.logZ.item():.3f} maxR={rewards.max().item():.2f} "
                  f"meanR={rewards.mean().item():.2f}")

    return policy


def sample_diverse(policy, grammar, forward_returns, n_samples=300):
    """训练后采样去重，返回 [(token_ids, metrics, expr_str)]，按 reward 降序。"""
    seen = {}
    attempts = 0
    while len(seen) < n_samples and attempts < n_samples * 8:
        attempts += 1
        seq, _ = sample_trajectory(policy, grammar)
        if len(seq) == 0:
            continue
        key = tuple(seq)
        if key in seen:
            continue
        seen[key] = seq

    out = []
    for key, seq in seen.items():
        m = compute_full_metrics(grammar, seq, forward_returns)
        if m is None:
            continue
        score_ret = ic_strength_return_score(m)
        score_turnover = float(np.clip(1.0 - TURNOVER_PENALTY * m["turnover"], 0.0, 1.0))
        score_complexity = complexity_score(len(seq))
        out.append({
            "token_ids": seq,
            "expr": sequence_to_string(seq, grammar),
            "icir": m["icir"],
            "gross_sharpe": m["gross_sharpe"],
            "turnover": m["turnover"],
            "score_ret": m["score_ret"],
            "score_turnover": m["score_turnover"],
            "n_tokens": len(seq),
            "reward": score_ret * score_turnover * score_complexity,
        })
    out.sort(key=lambda x: -x["reward"])
    return out
