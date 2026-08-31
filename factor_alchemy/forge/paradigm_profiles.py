# -*- coding: utf-8 -*-
"""
Forge 范式定向 profiles (v0.7 P-025)
=====================================
成熟范式清单与终端/算子加权配置。

成熟范式定义依据 (2026-08-13):
  - MAB expected_reward > 0: 筹码分布(0.55) / 动量反转(0.35) / 下行保护(0.15) /
    流动性×微观结构(0.08 + d_liq 0.20) / 波动率适应(d_vol 0.15)
  - JQ 已验证 breed: 价量关系(breed002 BEST +143%) / 均值回复(breed000 PASS +125%)
  - 王者 AlphaAgent v3 复合构成: 尾部风险 (筹码+尾部)

使用方式 (FactorForge):
    forge = FactorForge(data, paradigm_profiles=MATURE_PARADIGM_PROFILES, ...)
每个个体随机分配到一个范式, 初始化/交叉/变异继承范式标签,
树生成时按 profile 加权采样终端与算子。适应度不做任何范式扭曲。

Round 1c 实测 (pop=200, gens=5, depth=6, FSA+质量门全开):
  S5 通过 4/18 = 22% (gp_breed 基线 ~10.4%), 全部为流动性/微观结构方向。
"""

# {范式名: {terminal_weights: {终端: 权重}, op_weights: {算子: 权重}}}
# 终端: open/high/low/close/volume/vwap/returns
# 算子: forge.primitives 中算术 + ts_* + rank/zscore/scale
MATURE_PARADIGM_PROFILES = {
    "筹码分布": {
        "terminal_weights": {"volume": 2.5, "close": 2.0, "vwap": 1.5},
        "op_weights": {"ts_rank": 2.5, "ts_zscore": 2.0, "ts_mean": 1.5,
                       "ts_max": 1.5, "sub": 1.5, "ts_std": 1.0},
    },
    "动量反转": {
        "terminal_weights": {"returns": 3.0, "close": 2.0, "volume": 1.0},
        "op_weights": {"ts_delta": 2.0, "ts_pct": 2.0, "ts_rank": 2.0,
                       "ts_mean": 1.5, "ts_std": 1.0, "ts_zscore": 1.0},
    },
    "流动性×微观结构": {
        "terminal_weights": {"volume": 3.0, "vwap": 2.0, "close": 1.5},
        "op_weights": {"ts_std": 2.0, "div": 2.0, "ts_sum": 2.0,
                       "ts_mean": 1.5, "ts_pct": 1.5, "log": 1.0},
    },
    "下行保护": {
        "terminal_weights": {"low": 2.5, "close": 2.0, "high": 1.5},
        "op_weights": {"ts_min": 2.5, "ts_std": 2.0, "ts_max": 1.5,
                       "sub": 1.5, "div": 1.5, "ts_zscore": 1.0},
    },
    "波动率适应": {
        "terminal_weights": {"close": 2.0, "returns": 2.0, "high": 1.5, "low": 1.5},
        "op_weights": {"ts_std": 3.0, "ts_zscore": 2.0, "ts_min": 1.5,
                       "ts_max": 1.5, "div": 1.5, "ts_mean": 1.0},
    },
    "价量关系": {
        "terminal_weights": {"volume": 3.0, "close": 2.0, "vwap": 2.0},
        "op_weights": {"div": 2.0, "sub": 1.5, "ts_mean": 1.5, "ts_std": 1.5,
                       "ts_sum": 1.5, "mul": 1.0},
    },
    "均值回复": {
        "terminal_weights": {"close": 3.0, "returns": 2.0},
        "op_weights": {"ts_zscore": 2.5, "ts_mean": 2.0, "ts_std": 1.5,
                       "sub": 1.5, "rank": 1.5, "div": 1.0},
    },
    "尾部风险": {
        "terminal_weights": {"returns": 2.5, "close": 2.0, "volume": 1.5},
        "op_weights": {"ts_std": 2.5, "ts_min": 2.0, "ts_max": 2.0,
                       "ts_zscore": 1.5, "abs": 1.0, "div": 1.0},
    },
}
