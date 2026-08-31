# -*- coding: utf-8 -*-
"""
features.py — 候选策略 → 特征向量 φ

设计原则:
  - 特征刻意保持少 (~10 维), 防止小样本下过拟合
  - 只允许 domain/构造/local筛选 三层特征, 禁止喂 raw 净值曲线形状
  - local 特征 (FRI/Ψ/ICIR) 保留但权重由 scorer 从 JQ 结果学习, 不手拍

特征分两层粒度:
  - strategy level: 整个 6 复合组合 → 预测 JQ 收益 (用于回填历史 & 最终组合评分)
  - composite level: 单个复合 → 用于影子并行评估后的训练 (v2, 待影子数据)
"""
import json
import re
from pathlib import Path

# V3 基准原始因子 (所有策略共享的 comp1-3 原料)
V3_BASE_RAW = {
    "overnight_5d", "tvma_20", "dollar_vol_20d",
    "turnover_std_cv", "money_flow_20", "ret_3m",
}

FEATURE_NAMES = [
    "freq_weekly",          # 周频=1 (月频为基准)
    "freq_daily",           # 日频=1
    "n_composites_norm",    # 复合数 / 6
    "n_injected",           # 注入因子复合数 (0-3)
    "mean_fri_injected",    # 注入复合的平均 FRI (无注入=0)
    "max_fri_injected",     # 注入复合的最大 FRI
    "mean_novelty_injected",# 注入复合的平均 novelty
    "n_new_llm",            # 新 LLM 生成复合数 (非V3基准, 非注入)
    "tail_dim_count",       # 含尾部风险维度的新复合数 (从docstring关键词)
    "chip_dim_count",       # 含筹码分布维度的新复合数
]

_TAIL_KEYWORDS = ["尾部", "极端", "回撤", "下行", "崩盘", "协偏"]
_CHIP_KEYWORDS = ["筹码", "缩量", "锁定", "惜售", "持仓成本"]


def parse_jq_strategy(path):
    """解析 JQ 策略文件, 提取频率/复合结构/原始因子名."""
    text = Path(path).read_text(encoding="utf-8")

    freq_m = re.search(r"run_(weekly|monthly|daily)", text)
    frequency = freq_m.group(1) if freq_m else "unknown"

    # 匹配每个 compute_score_compN 的 docstring + 两个原始因子
    comp_pattern = re.findall(
        r'def (compute_score_comp\w+)\(stocks, price_data\):\s*"""([^"]*)"""'
        r"\s*a, va = compute_(\w+)\(stocks, price_data\)"
        r"\s*b, vb = compute_(\w+)\(stocks, price_data\)",
        text,
    )
    composites = [
        {"func": func, "doc": doc, "raw_a": raw_a, "raw_b": raw_b}
        for func, doc, raw_a, raw_b in comp_pattern
    ]
    return {"frequency": frequency, "composites": composites, "path": str(path)}


def _load_injected_db(injected_json_path):
    p = Path(injected_json_path)
    if not p.exists():
        return {}
    with open(p, encoding="utf-8") as f:
        data = json.load(f)
    return {item["factor_name"]: item for item in data.get("injected", [])}


def build_strategy_features(parsed, injected_json_path):
    """
    从解析结果构建策略级特征向量。
    返回 (feature_vector: list[float], meta: dict)
    """
    injected_db = _load_injected_db(injected_json_path)
    comps = parsed["composites"]
    n_comps = len(comps)

    n_injected = 0
    fris, novelties = [], []
    n_new_llm = 0
    tail_count, chip_count = 0, 0

    for c in comps:
        raws = {c["raw_a"], c["raw_b"]}
        new_raws = raws - V3_BASE_RAW
        if not new_raws:
            continue  # V3 保留复合 (comp1-3), 不提供区分信息

        # 新复合: 注入 or LLM
        inj_hits = [r for r in new_raws if r in injected_db]
        if inj_hits:
            n_injected += 1
            for r in inj_hits:
                fris.append(float(injected_db[r].get("fri", 0)))
                novelties.append(float(injected_db[r].get("fri_novelty", 0)))
        else:
            n_new_llm += 1

        doc = c["doc"]
        if any(k in doc for k in _TAIL_KEYWORDS):
            tail_count += 1
        if any(k in doc for k in _CHIP_KEYWORDS):
            chip_count += 1

    vec = [
        1.0 if parsed["frequency"] == "weekly" else 0.0,
        1.0 if parsed["frequency"] == "daily" else 0.0,
        n_comps / 6.0,
        float(n_injected),
        float(sum(fris) / len(fris)) if fris else 0.0,
        float(max(fris)) if fris else 0.0,
        float(sum(novelties) / len(novelties)) if novelties else 0.0,
        float(n_new_llm),
        float(tail_count),
        float(chip_count),
    ]
    meta = {
        "frequency": parsed["frequency"],
        "n_composites": n_comps,
        "n_injected": n_injected,
        "n_new_llm": n_new_llm,
    }
    assert len(vec) == len(FEATURE_NAMES), f"特征维度不匹配: {len(vec)} vs {len(FEATURE_NAMES)}"
    return vec, meta
