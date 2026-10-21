"""
因子表达式语法空间 + 求值
==========================
GFlowNet 生成的对象 = 因子表达式树（preorder 前缀序列表示）。

词汇表:
  - 二元算子: add / sub / mul / div
  - 一元算子: rank(截面百分位) / neg(取反) / ts_zscore(时序标准化)
  - 叶子: 97 个预计算因子

生成规则（槽位驱动，与 AlphaGen 的语法合法约束同思路）:
  - 初始剩余槽位 slots = 1
  - 二元算子: 消耗 1 槽 + 产生 2 槽 (净 +1)
  - 一元算子: 消耗 1 槽 + 产生 1 槽 (净 0)
  - 叶子:     消耗 1 槽 (净 -1)
  - slots == 0 时表达式完成；max_len 限制总 token 数

金融约束:
  - div 做安全除法（分母近 0 → NaN，评估时由截面 rank 的 NaN 处理兜底）
  - max_len 限制表达式复杂度（防过拟合/不可解释）
"""
from __future__ import annotations

import numpy as np

BINARY_OPS = ["add", "sub", "mul", "div"]
UNARY_OPS = ["rank", "neg", "ts_zscore"]


class Grammar:
    def __init__(self, factor_names, max_len=9):
        self.factor_names = list(factor_names)
        self.binary = list(BINARY_OPS)
        self.unary = list(UNARY_OPS)
        self.max_len = max_len

        self.tokens = self.binary + self.unary + self.factor_names
        self.token2id = {t: i for i, t in enumerate(self.tokens)}
        self.id2token = {i: t for i, t in enumerate(self.tokens)}

        self.n_binary = len(self.binary)
        self.n_unary = len(self.unary)
        self.n_leaf = len(self.factor_names)
        self.n_tokens = len(self.tokens)
        self.leaf_start = self.n_binary + self.n_unary  # 叶子 id 起始

        # 预加载叶子因子值 (float64, n_weeks × n_stocks)
        self._factor_arrays = None

    # ------------------------------------------------------------------
    # 动作空间
    # ------------------------------------------------------------------
    def valid_actions(self, slots, length):
        """给定剩余槽位与已用长度，返回合法 token id 列表（空 = terminal）。"""
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
    # 表达式求值
    # ------------------------------------------------------------------
    def set_factor_arrays(self, factor_dict):
        """把 DataFrame 字典转成 {name: float64 ndarray} 供快速求值。"""
        self._factor_arrays = {
            name: factor_dict[name].astype(np.float64).values
            for name in self.factor_names
        }

    def evaluate(self, token_ids):
        """token_ids: preorder 前缀序列（完整表达式）。返回 (n_weeks, n_stocks) ndarray。"""
        self._idx = 0
        arr = self._parse(token_ids)
        return arr

    def _parse(self, token_ids):
        tid = token_ids[self._idx]
        self._idx += 1
        if tid >= self.leaf_start:
            return self._factor_arrays[self.id2token[tid]]
        elif tid < self.n_binary:
            left = self._parse(token_ids)
            right = self._parse(token_ids)
            return self._apply_binary(tid, left, right)
        else:
            child = self._parse(token_ids)
            return self._apply_unary(tid, child)

    def _apply_binary(self, tid, x, y):
        op = self.id2token[tid]
        if op == "add":
            return x + y
        if op == "sub":
            return x - y
        if op == "mul":
            return x * y
        if op == "div":
            # 🔴 禁自除：div(X, X) 会产生恒等/常量因子 → ic_std≈0 → 伪 ICIR 爆炸穿透 JQ 门。
            # 同一 ndarray 对象（同一叶子）直接判 NaN；值相同但对象不同（如 div(rank(X), rank(X))）
            # 由评估层 ic_std 下限兜底。
            if x is y:
                return np.full(x.shape, np.nan, dtype=np.float64)
            denom = np.where(np.abs(y) < 1e-12, np.nan, y)
            with np.errstate(invalid="ignore", divide="ignore"):
                return x / denom
        raise ValueError(f"unknown binary op {op}")

    def _apply_unary(self, tid, x):
        op = self.id2token[tid]
        if op == "neg":
            return -x
        if op == "rank":
            return self._cross_sectional_rank(x)
        if op == "ts_zscore":
            return self._time_series_zscore(x)
        raise ValueError(f"unknown unary op {op}")

    @staticmethod
    def _cross_sectional_rank(x):
        """逐行截面百分位 rank → [0,1]，NaN 保留。"""
        out = np.full(x.shape, np.nan, dtype=np.float64)
        for w in range(x.shape[0]):
            row = x[w]
            valid = np.isfinite(row)
            if valid.sum() < 2:
                continue
            rk = np.argsort(np.argsort(row[valid]))
            out[w, valid] = rk.astype(np.float64) / (valid.sum() - 1)
        return out

    @staticmethod
    def _time_series_zscore(x):
        """逐列时序 z-score，NaN 保留。"""
        out = np.full(x.shape, np.nan, dtype=np.float64)
        for c in range(x.shape[1]):
            col = x[:, c]
            valid = np.isfinite(col)
            if valid.sum() < 5:
                continue
            mu = np.mean(col[valid])
            sd = np.std(col[valid], ddof=0)
            if sd < 1e-12:
                continue
            out[valid, c] = (col[valid] - mu) / sd
        return out


def sequence_to_string(token_ids, grammar):
    """把 preorder 序列还原成可读字符串（用于可解释性输出）。"""
    idx = [0]

    def parse():
        tid = token_ids[idx[0]]
        idx[0] += 1
        if tid >= grammar.leaf_start:
            return grammar.id2token[tid]
        if tid < grammar.n_binary:
            l = parse()
            r = parse()
            return f"{grammar.id2token[tid]}({l}, {r})"
        return f"{grammar.id2token[tid]}({parse()})"

    return parse()
