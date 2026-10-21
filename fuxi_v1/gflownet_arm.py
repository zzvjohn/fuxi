# -*- coding: utf-8 -*-
"""GFlowNet 生产臂（第 6 臂）：生产 DSL 空间（10 字段）的离散因子生成器。

定位
====
把 factor_mining 沙箱里的 GFlowNet 正式提升为生产 ``fuxi_v1`` 的一臂，输出**生产
DSL 空间的 AST**（叶子 = ``{"field": name}``、算子 = 生产 CATALOG 算子），从而走与
``fixed/refine/combine`` 完全相同的 ``search.factor_spec → compile_ast → E0-E5`` 链路。

与量纲语法系统（types_core）的契合
==================================
- 叶子类型 = ``FIELD_REGISTRY``（10 字段，``resolve(name, kind="field")``）；
- 算子规则 = 同一张 ``types_core.rules`` 规则表（add/sub 跨 form/dim 硬拒、div 分母
  signed 硬拒、log 非无量纲/非正硬拒……），采样侧用 ``TypeState + mask_actions`` 做
  增量类型掩码，产出侧由 ``compile_ast`` 的 UNITS 量纲 + ``FUXI_TYPE_GATE`` 类型闸门
  兜底 —— 两侧共用同一规则表，不漂移（R5 治理的核心）。

算子范围（第一版）
==================
- 二元：``add / sub / mul / div``
- 一元：``neg / rank / zscore / abs / square / log``
- 共 10 个，全部是生产 CATALOG 的**非时序算子**（无 window），且都在
  ``types_core.rules._RULES`` 中，``TypeState`` 增量掩码完整支持。

时序算子（``lag/return/mean/std/min/max/delta``）+ window 是**明确的后续增强**：需要
扩展 ``TypeState`` 支持 window 感知 + grammar 增加 window token，属独立子任务（见
模块 docstring 末尾 TODO）。第一版刻意不含，保证「架构闭环 + 类型契合」可验证落地。

采样 / 训练
===========
- ``sample_asts``：无 policy 时用「随机 + 类型掩码」采样（等价 typed random，但用
  GFlowNet 的槽位驱动语法），有 policy 时按前向策略采样；产出唯一 AST。
- ``train_prod_gflownet``：离线训练入口（lazy import torch），reward 用生产 snapshot
  的截面 Rank IC（复用 ``evaluation.factor_statistics`` 口径），训练 ``GFlowNetPolicy``。
  torch 仅此函数需要；generate 主路径不 import torch，保证 fuxi_v1 其余模块零 torch 依赖。

Local: 手写 (2026-10-21, 用户指令「真正把 GFlowNet 注册成生产第 6 臂、进 UCB」)
"""

from __future__ import annotations

import os
import random

from .contracts import digest, require

# 生产 DSL 非时序算子（与 types_core.rules 的 BINARY_ARITH / 一元算子对齐）。
PROD_BINARY = ("add", "sub", "mul", "div")
PROD_UNARY = ("neg", "rank", "zscore", "abs", "square", "log")

DEFAULT_MAX_LEN = 12          # token 数上限（< CATALOG max_nodes=64，留足编译预算）
DEFAULT_BETA = 8.0            # reward 放大系数（与 factor_mining 训练一致）
DEFAULT_MECHANISM = "gflownet_discrete_structure"


def _type_mask_default():
    """生产臂默认启用类型掩码（契合量纲语法系统），可用 FUXI_TYPE_MASK=off 关闭做 A/A。"""
    return os.environ.get("FUXI_TYPE_MASK", "on") == "on"


class ProdGrammar:
    """生产 DSL 空间（字段叶子 + 非时序算子）的 GFlowNet 语法。

    preorder 前缀 token 序列表示表达式树；``to_ast`` 把 token 序列还原成生产 AST
    （dict 嵌套），可直接交给 ``search.factor_spec`` / ``dsl.compile_ast``。
    """

    def __init__(self, field_names, max_len=DEFAULT_MAX_LEN):
        self.field_names = list(field_names)
        self.binary = list(PROD_BINARY)
        self.unary = list(PROD_UNARY)
        self.max_len = max_len

        self.tokens = self.binary + self.unary + self.field_names
        self.token2id = {t: i for i, t in enumerate(self.tokens)}
        self.id2token = {i: t for i, t in enumerate(self.tokens)}

        self.n_binary = len(self.binary)
        self.n_unary = len(self.unary)
        self.n_leaf = len(self.field_names)
        self.n_tokens = len(self.tokens)
        self.leaf_start = self.n_binary + self.n_unary   # 叶子 id 起始

    # ------------------------------------------------------------------
    # 动作空间（槽位驱动，与 factor_mining/gflownet/grammar.py 同思路）
    # ------------------------------------------------------------------
    def valid_actions(self, slots, length):
        """剩余槽位 + 已用长度 → 合法 token id 列表（空 = terminal）。"""
        if slots <= 0:
            return []
        budget = self.max_len - length
        acts = list(range(self.leaf_start, self.n_tokens))          # 叶子恒合法
        if budget >= slots + 1:
            acts += list(range(self.n_binary, self.leaf_start))     # 一元算子
        if budget >= slots + 2:
            acts += list(range(0, self.n_binary))                   # 二元算子
        return acts

    def is_terminal(self, slots):
        return slots <= 0

    # ------------------------------------------------------------------
    # token 序列 → 生产 AST
    # ------------------------------------------------------------------
    def to_ast(self, token_ids):
        """preorder token 序列 → 生产 AST（dict）。返回 None 若序列不完整/非法。"""
        idx = [0]
        try:
            ast = self._parse(token_ids, idx)
        except (IndexError, KeyError):
            return None
        if idx[0] != len(token_ids):
            return None
        return ast

    def _parse(self, token_ids, idx):
        tid = token_ids[idx[0]]
        idx[0] += 1
        if tid >= self.leaf_start:
            return {"field": self.id2token[tid]}
        tok = self.id2token[tid]
        if tid < self.n_binary:
            left = self._parse(token_ids, idx)
            right = self._parse(token_ids, idx)
            return {"op": tok, "args": [left, right]}
        return {"op": tok, "args": [self._parse(token_ids, idx)]}

    def sequence_to_string(self, token_ids):
        """token 序列 → 可读表达式字符串（可解释性输出）。"""
        idx = [0]

        def parse():
            tid = token_ids[idx[0]]
            idx[0] += 1
            if tid >= self.leaf_start:
                return self.id2token[tid]
            tok = self.id2token[tid]
            if tid < self.n_binary:
                return "%s(%s, %s)" % (tok, parse(), parse())
            return "%s(%s)" % (tok, parse())

        return parse()


# ---------------------------------------------------------------------------
# 类型掩码（types_core TypeState + mask_actions，resolver = FIELD_REGISTRY）
# ---------------------------------------------------------------------------
def _make_type_state(grammar):
    from .types_core import TypeState, resolve
    ops = set(grammar.binary + grammar.unary)
    return TypeState(lambda n: resolve(n, kind="field"), lambda t: t in ops)


def _apply_type_mask(type_state, grammar, acts):
    from .types_core import mask_actions
    tokens = [grammar.id2token[i] for i in acts]
    legal = mask_actions(type_state, tokens)
    return [grammar.token2id[t] for t in legal]


# ---------------------------------------------------------------------------
# 采样
# ---------------------------------------------------------------------------
def _sample_sequence(grammar, rng, *, policy=None, type_state=None, use_type_mask=True):
    """采样一个完整表达式 token 序列；返回 (token_ids, logp_sum)。

    - token_ids 空 = 失败/不完整表达式（slots 未归零）。
    - ``logp_sum`` 仅在 ``policy`` 非 None 时为 torch 标量（保留梯度，供 TB 训练）；
      随机采样时为 None（无梯度语义）。

    ``policy`` 为 ``GFlowNetPolicy``（torch，可选）；None 时等概率随机 + 类型掩码。
    ``type_state`` 为 types_core.TypeState（None 时惰性构造）。
    """
    seq = []
    logp_sum = None
    slots = 1
    length = 0
    if use_type_mask and type_state is None:
        type_state = _make_type_state(grammar)
    # TypeState 是「单条轨迹」的状态机：每次采样必须从干净状态出发，否则上一条
    # 轨迹遗留的 values/frames 会污染掩码（让 add/sub 跨量纲漏过）。见 test_sample_asts。
    if type_state is not None:
        type_state.reset()
    if policy is None:
        # 随机采样：每一步从类型合法动作集中均匀抽一个。
        while slots > 0 and length < grammar.max_len:
            acts = grammar.valid_actions(slots, length)
            if not acts:
                break
            if type_state is not None:
                masked = _apply_type_mask(type_state, grammar, acts)
                if masked:
                    acts = masked
                else:
                    # 类型掩码下无合法动作 = 轨迹死胡同：放弃本轨迹（rejection sampling），
                    # 不得回退到未掩码动作集（否则 add/sub 跨量纲会漏过）。
                    break
            a = rng.choice(acts)
            seq.append(a)
            length += 1
            if type_state is not None:
                type_state.step(grammar.id2token[a])
            if a >= grammar.leaf_start:
                slots -= 1
            elif a < grammar.n_binary:
                slots += 1
    else:
        import torch
        import torch.nn.functional as F
        logp_sum = torch.tensor(0.0)
        while slots > 0 and length < grammar.max_len:
            acts = grammar.valid_actions(slots, length)
            if not acts:
                break
            if type_state is not None:
                masked = _apply_type_mask(type_state, grammar, acts)
                if masked:
                    acts = masked
                else:
                    # 同上：类型掩码死胡同 → 拒绝本轨迹。
                    break
            logits = policy.next_logits(torch.tensor(seq, dtype=torch.long))
            mask = torch.full((policy.n_tokens,), float("-inf"))
            mask[acts] = 0.0
            probs = F.softmax(logits + mask, dim=0)
            a = int(torch.multinomial(probs, 1))
            logp_sum = logp_sum + torch.log(probs[a] + 1e-12)   # 保留梯度
            seq.append(a)
            length += 1
            if type_state is not None:
                type_state.step(grammar.id2token[a])
            if a >= grammar.leaf_start:
                slots -= 1
            elif a < grammar.n_binary:
                slots += 1
    if slots > 0:
        return [], logp_sum
    return seq, logp_sum


def sample_asts(fields, *, count=8, seed=0, policy=None, max_len=DEFAULT_MAX_LEN,
                use_type_mask=None):
    """采样 ``count`` 个**唯一**的生产 AST（去重按 digest(ast)）。

    ``fields``：生产 snapshot 的字段 catalog dict（如 ``payload["fields"]``），叶子 =
    ``sorted(fields.keys())``。返回 AST 列表（已去重，可能 < count）。
    """
    field_names = sorted(fields.keys())
    require(len(field_names) >= 2, "gflownet arm requires at least two fields")
    grammar = ProdGrammar(field_names, max_len=max_len)
    if use_type_mask is None:
        use_type_mask = _type_mask_default()
    rng = random.Random(seed)
    type_state = _make_type_state(grammar) if use_type_mask else None

    asts = []
    seen = set()
    attempts = 0
    while len(asts) < count and attempts < count * 20:
        attempts += 1
        seq, _ = _sample_sequence(grammar, rng, policy=policy,
                                   type_state=type_state, use_type_mask=use_type_mask)
        if not seq:
            continue
        ast = grammar.to_ast(seq)
        if ast is None:
            continue
        key = digest(ast)
        if key in seen:
            continue
        seen.add(key)
        asts.append(ast)
    return asts


# ---------------------------------------------------------------------------
# 训练入口（离线增强，lazy torch；reward = 生产 snapshot 截面 Rank IC）
# ---------------------------------------------------------------------------
def _ic_reward_from_statistics(factor_stats, n_tokens):
    """把 E1 factor_statistics 的 spearman mean/std 映射为 [0,1] 收益分。

    复用生产 E1 口径：ic_mean 取 |spearman.mean|（方向无关强度），ic_std 走贝叶斯收缩
    （分母加先验防伪 ICIR 爆炸），复杂度分段罚退化/过深。返回 float ∈ [0,1]。
    """
    if factor_stats is None or factor_stats.get("eval_status") != "valid":
        return 0.0
    sp = factor_stats.get("metrics", {}).get("spearman", {}) or {}
    ic_mean = sp.get("mean")
    ic_std = sp.get("sample_sd")
    if ic_mean is None or ic_std is None:
        return 0.0
    # 贝叶斯收缩 ICIR（分母加先验，替代硬阈值；先验取 0.047，与 factor_mining 口径一致）
    LAMBDA_PRIOR = 0.047
    strength = abs(ic_mean) / (ic_std + LAMBDA_PRIOR)
    score_ic = min(1.0, strength / 0.2)   # shrink_icir=0.2 记满分（对齐 JQ 门阈值）
    score_complexity = _complexity_score(n_tokens)
    return float(score_ic * score_complexity)


def _complexity_score(n_tokens):
    """分段复杂度分（与 factor_mining/gflownet/reward.py 同口径）。"""
    if n_tokens <= 1:
        return 0.30
    if n_tokens <= 5:
        return 1.00
    if n_tokens <= 7:
        return 0.70
    return 0.30


def train_prod_gflownet(snapshot, labels, fields, *, n_episodes=300, batch_size=16,
                        lr=1e-3, beta=DEFAULT_BETA, seed=0, max_len=DEFAULT_MAX_LEN,
                        research_clock=None, policy=None):
    """离线训练生产 DSL 空间的 GFlowNet 前向策略（TB 损失）。

    参数：
      snapshot / labels：生产 snapshot + 标签（``campaign`` 已构造）；用于算截面 Rank IC。
      fields：snapshot 字段 catalog。
    返回：训练好的 (policy, grammar)。

    注意：需 torch（qlib312 有 2.13.0+cpu）；generate 主路径不依赖此函数。
    """
    import numpy as np
    import torch
    import torch.nn.functional as F
    from torch.optim import Adam

    from .dsl import evaluate_ast, compile_ast
    from .evaluation import DEFAULT_POLICY, factor_statistics

    field_names = sorted(fields.keys())
    grammar = ProdGrammar(field_names, max_len=max_len)
    if policy is None:
        from .gflownet_policy import GFlowNetPolicy
        policy = GFlowNetPolicy(grammar.n_tokens)

    torch.manual_seed(seed)
    np.random.seed(seed)
    opt = Adam(policy.parameters(), lr=lr)

    def reward_of(seq):
        ast = grammar.to_ast(seq)
        if ast is None:
            return 0.0
        try:
            compile_ast(ast, fields)          # 量纲 + 类型闸门兜底
            scores = evaluate_ast(ast, snapshot)
        except Exception:
            return 0.0
        try:
            clock = research_clock or (snapshot.sessions[-1] + "T20:00:00+08:00")
            stats = factor_statistics(scores, snapshot, labels, DEFAULT_POLICY, clock)
        except Exception:
            return 0.0
        return _ic_reward_from_statistics(stats, len(seq))

    for ep in range(n_episodes):
        batch = []
        for i in range(batch_size):
            seq, logp = _sample_sequence(grammar, random.Random(seed * 1000 + ep * batch_size + i),
                                         policy=policy, use_type_mask=True)
            batch.append((seq, logp))
        rewards = torch.tensor([reward_of(s) for s, _ in batch], dtype=torch.float32)
        logR = torch.log(rewards.clamp(min=1e-8))
        logp = torch.stack([lp for _, lp in batch])   # policy 分支恒为 tensor，保留梯度
        loss = ((policy.logZ + logp - logR) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if ep % 20 == 0 or ep == n_episodes - 1:
            print("  ep %d/%d | loss=%.3f logZ=%.3f maxR=%.3f meanR=%.3f" % (
                ep + 1, n_episodes, loss.item(), policy.logZ.item(),
                rewards.max().item(), rewards.mean().item()))
    return policy, grammar


# 惰性导出：避免 fuxi_v1 其余模块在 import 本文件时被迫依赖 torch。
def make_policy(n_tokens, *, embed_dim=64, hidden_dim=128):
    """构造 GFlowNetPolicy（lazy import torch）。"""
    from .gflownet_policy import GFlowNetPolicy
    return GFlowNetPolicy(n_tokens, embed_dim=embed_dim, hidden_dim=hidden_dim)
