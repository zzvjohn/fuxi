# -*- coding: utf-8 -*-
"""GFlowNet 前向策略模型 + 轨迹平衡(TB)损失（生产臂专用，自包含）。

与 factor_mining/gflownet/model.py 同结构，但独立放在 fuxi_v1 下，避免生产侧依赖
factor_mining 沙箱。torch 仅在本模块被 import 时才加载（gflownet_arm.py 惰性调用）。

状态 = 当前前缀 token 序列（preorder）。
前向策略 P_F(a|s) = softmax(GRU(embed(seq)) + 动作掩码)。
后向策略：序列生成中「去掉末 token 回到父状态」是确定性的 ⇒ P_B=1 ⇒ log P_B=0。
TB 损失:  L(τ) = ( logZ + Σ_t log P_F(a_t|s_t) − log R(x) )²
"""

from __future__ import annotations

import torch
import torch.nn as nn


class GFlowNetPolicy(nn.Module):
    def __init__(self, n_tokens, embed_dim=64, hidden_dim=128):
        super().__init__()
        self.n_tokens = n_tokens
        self.embed = nn.Embedding(n_tokens, embed_dim)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.h0 = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.head = nn.Linear(hidden_dim, n_tokens)
        self.logZ = nn.Parameter(torch.zeros(1))

    def next_logits(self, seq):
        """seq: (L,) 长整型前缀 token 序列（可空）。返回 (V,) logits。"""
        if len(seq) == 0:
            hidden = self.h0  # (1,1,H)
            return self.head(hidden[0, 0])
        emb = self.embed(seq.unsqueeze(0))          # (1,L,E)
        out, _ = self.gru(emb, self.h0)             # out (1,L,H)
        return self.head(out[0, -1])                # (V,)
