# -*- coding: utf-8 -*-
"""
scorer.py — RidgeUCB: 带收缩先验的岭回归 + LinUCB 探索奖励

score(x) = ŷ(x) + β · σ_e · σ(x)
  ŷ(x)  : 预测的 JQ 累计收益 (从试验历史学习)
  σ(x)  : 预测不确定性 (特征空间覆盖不足 → 大)
  β     : 探索系数, 决定"试未知"的积极性

核心性质:
  - 所有权重从 0 出发 (收缩先验 λ 大), 证据推动权重移动
  - FRI 等 local 特征的权重是**学出来的**, 不是手拍的 — 若 FRI 无 JQ
    预测力, 权重自动收敛到 ~0; 这正是 "评分函数可以犯错但可继续优化"
  - 滚动半衰期加权: 近期试验权重更高 (非平稳性适应)
"""
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np


class RidgeUCB:
    def __init__(self, n_features, lam=10.0, beta=1.0, half_life_days=30.0):
        """
        lam: 收缩强度 (先验精度). 大 → 权重需要更多证据才能离开 0
        beta: 探索系数. 0=纯利用, 1=平衡, 2=激进探索
        half_life_days: 试验样本的时间半衰期
        """
        self.n_features = n_features
        self.lam = float(lam)
        self.beta = float(beta)
        self.half_life_days = float(half_life_days)
        # 在线累积量
        self.A = np.eye(n_features) * self.lam   # 精度矩阵
        self.b = np.zeros(n_features)            # 加权 XY
        self.x_mean = np.zeros(n_features)
        self.x_std = np.ones(n_features)
        self.y_mean = 0.0
        self.sigma_e = 1.0                       # 残差标准差
        self.n_samples = 0
        self.feature_names = None

    # ── 标准化 ──────────────────────────────────────────────
    def fit_scaler(self, X):
        self.x_mean = X.mean(axis=0)
        std = X.std(axis=0)
        std[std < 1e-8] = 1.0
        self.x_std = std

    def _norm(self, x):
        return (np.asarray(x, dtype=float) - self.x_mean) / self.x_std

    # ── 训练 ────────────────────────────────────────────────
    def fit(self, X, y, dates=None):
        """
        X: (n, p) 特征矩阵, y: (n,) JQ 累计收益 %
        dates: ISO 日期字符串列表, 用于时间半衰期加权 (None=等权)
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n = len(y)
        self.fit_scaler(X)
        Xn = (X - self.x_mean) / self.x_std
        self.y_mean = float(y.mean())
        yc = y - self.y_mean

        # 时间权重
        if dates is not None:
            now = datetime.now()
            weights = []
            for d in dates:
                try:
                    age = (now - datetime.fromisoformat(d[:10])).days
                except Exception:
                    age = 365
                weights.append(0.5 ** (max(age, 0) / self.half_life_days))
            w = np.asarray(weights)
        else:
            w = np.ones(n)

        # 加权岭回归: A = λI + Σ w x x', b = Σ w x y
        self.A = np.eye(self.n_features) * self.lam
        self.b = np.zeros(self.n_features)
        for i in range(n):
            xi = Xn[i]
            self.A += w[i] * np.outer(xi, xi)
            self.b += w[i] * xi * yc[i]
        self.n_samples = n

        # 残差 std (用于把 σ(x) 换算到收益单位)
        w_coef = self.coef()
        resid = yc - Xn @ w_coef
        dof = max(n - 2, 1)
        self.sigma_e = float(np.sqrt(np.sum(w * resid ** 2) / dof))
        return self

    def coef(self):
        """标准化空间中的权重 (对应标准化特征)"""
        return np.linalg.solve(self.A, self.b)

    # ── 预测 ────────────────────────────────────────────────
    def predict(self, x):
        """返回 (预测收益%, 不确定性σ(收益单位))"""
        xn = self._norm(x)
        w = self.coef()
        mean = self.y_mean + float(xn @ w)
        A_inv = np.linalg.inv(self.A)
        sigma = self.sigma_e * math.sqrt(max(float(xn @ A_inv @ xn), 0.0))
        return mean, sigma

    def score(self, x):
        """LinUCB 选择分 = 预测 + β·σ (用于决定测什么, 越大越优先)"""
        mean, sigma = self.predict(x)
        return mean + self.beta * sigma

    # ── 报告 ────────────────────────────────────────────────
    def weight_table(self):
        """按 |w| 排序的权重表 (标准化空间)"""
        w = self.coef()
        names = self.feature_names or [f"f{i}" for i in range(self.n_features)]
        rows = sorted(zip(names, w), key=lambda r: -abs(r[1]))
        return rows

    # ── 持久化 ──────────────────────────────────────────────
    def save(self, path):
        state = {
            "n_features": self.n_features,
            "lam": self.lam,
            "beta": self.beta,
            "half_life_days": self.half_life_days,
            "A": self.A.tolist(),
            "b": self.b.tolist(),
            "x_mean": self.x_mean.tolist(),
            "x_std": self.x_std.tolist(),
            "y_mean": self.y_mean,
            "sigma_e": self.sigma_e,
            "n_samples": self.n_samples,
            "feature_names": self.feature_names,
            "saved_at": datetime.now().isoformat(),
        }
        Path(path).write_text(json.dumps(state, ensure_ascii=False, indent=2),
                              encoding="utf-8")

    @classmethod
    def load(cls, path):
        state = json.loads(Path(path).read_text(encoding="utf-8"))
        obj = cls(state["n_features"], state["lam"], state["beta"],
                  state["half_life_days"])
        obj.A = np.array(state["A"])
        obj.b = np.array(state["b"])
        obj.x_mean = np.array(state["x_mean"])
        obj.x_std = np.array(state["x_std"])
        obj.y_mean = state["y_mean"]
        obj.sigma_e = state["sigma_e"]
        obj.n_samples = state["n_samples"]
        obj.feature_names = state.get("feature_names")
        return obj
