# -*- coding: utf-8 -*-
"""
AlphaAgent: LLM驱动因子挖掘 + 三重约束 + EFS进化循环
=====================================================
基于论文:
- AlphaAgent (KDD 2025): LLM-driven alpha mining + 三重正则化
- EFS (arXiv:2507.17211): Evolutionary Factor Search with LLMs
- Adaptive Alpha Weighting with PPO (arXiv:2509.01393)

核心创新: LLM 生成因子公式（非配对现有因子）, 三重约束过滤:
  1. Originality: AST 相似度 vs 现有因子池
  2. Hypothesis-Factor Alignment: LLM 自我评估经济逻辑
  3. Complexity Control: AST 节点数硬上限

EFS进化循环: Gen0(V3种子) → LLM生成 → 三重过滤 → ICIR评估 →
               Top-K幸存者 → 知识蒸馏 → 下一轮提示 → 重复

集成到 Forge 基础设施: 复用 ExprNode / parse_expression / ExpressionEvaluator / Primitives
"""

import json
import re
import time
import os
import sys
import gc
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict

# ─── 项目路径 ───
PROJ_DIR = Path(__file__).parent
sys.path.insert(0, str(PROJ_DIR))

from forge.primitives import (
    Primitive, PRIMITIVE_BY_NAME, INPUT_PRIMITIVES,
    ARITHMETIC_PRIMITIVES, TIMESERIES_PRIMITIVES, CROSS_SECTION_PRIMITIVES,
    WINDOW_SIZES,
)
from forge.expression import (
    ExprNode, ExpressionEvaluator, parse_expression, simplify_tree,
)


# ═══════════════════════════════════════════════════════════
# 配置
# ═══════════════════════════════════════════════════════════

@dataclass
class AlphaAgentConfig:
    """AlphaAgent 配置"""
    # 三重约束阈值
    originality_threshold: float = 0.70      # AST相似度 > 此值 → 拒绝
    hypothesis_min_score: int = 3            # LLM自我评分 < 此值 → 拒绝 (1-5)
    max_complexity: int = 28                 # AST节点数硬上限

    # EFS进化
    n_generations: int = 4                   # 进化代数
    n_candidates_per_gen: int = 25           # 每代LLM生成候选数
    n_survivors: int = 6                     # 每代幸存因子数（进入知识蒸馏）
    icir_threshold: float = 0.25             # 幸存者最低|ICIR|

    # LLM
    llm_model: str = "deepseek-v4-pro"       # 使用的LLM模型
    llm_temperature_gen: float = 0.8         # 生成阶段温度（高多样性）
    llm_temperature_eval: float = 0.3        # 评估阶段温度（低随机性）

    # 因子池
    seed_factor_exprs: List[str] = field(default_factory=lambda: [
        # V3 11个底因子的 Forge 表达式
        "ts_sum(sub(div(open, ts_delta(close, 1)), 1), 5)",             # overnight_5d
        "neg(div(mul(close, volume), ts_mean(mul(close, volume), 20)))", # tvma_20
        "neg(log(ts_mean(mul(close, volume), 20)))",                     # dollar_vol_20d
        "neg(div(ts_std(volume, 20), ts_mean(volume, 20)))",             # turnover_std_cv
        "neg(sub(div(mul(add(add(high,low),close), volume), 3), ts_mean(div(mul(add(add(high,low),close), volume), 3), 20)))", # money_flow_20
        "neg(sub(div(close, ts_delta(close, 60)), 1))",                  # ret_3m
        "ts_mean(sub(div(open, ts_delta(close, 1)), 1), 2)",             # ret_open_2d
        "neg(ts_std(returns, 20))",                                       # skewness_20代理
        "sub(div(open, ts_delta(close, 1)), 1)",                         # gap_up
        "neg(div(sub(high, low), add(close, 0.001)))",                   # relative_spread
        "inv(div(close, 1))",                                             # bp (1/PB) — 需要PB数据, 此处占位
    ])

    # 输出目录
    output_dir: str = ""


# ═══════════════════════════════════════════════════════════
# V3 因子池 — 用于Originality检查的参照集
# ═══════════════════════════════════════════════════════════

V3_FACTOR_POOL: List[ExprNode] = []


def init_v3_pool(existing_exprs: List[str] = None):
    """初始化V3因子池（用于Originality检查）"""
    global V3_FACTOR_POOL
    V3_FACTOR_POOL = []
    exprs = existing_exprs or AlphaAgentConfig().seed_factor_exprs
    for e in exprs:
        try:
            node = parse_expression(e)
            if node:
                V3_FACTOR_POOL.append(node)
        except Exception:
            pass
    return len(V3_FACTOR_POOL)


# ═══════════════════════════════════════════════════════════
# 约束1: Originality Enforcement — AST相似度
# ═══════════════════════════════════════════════════════════

def _collect_subtree_hashes(node: ExprNode) -> Set[str]:
    """收集树的所有子树hash（用于Jaccard相似度）"""
    hashes = {node.to_string()}
    for c in node.children:
        hashes.update(_collect_subtree_hashes(c))
    return hashes


def ast_similarity(node_a: ExprNode, node_b: ExprNode) -> float:
    """
    计算两个因子AST的Jaccard子树相似度。
    取值范围 [0, 1], 0=完全不同, 1=完全相同。
    """
    hashes_a = _collect_subtree_hashes(node_a)
    hashes_b = _collect_subtree_hashes(node_b)
    if not hashes_a or not hashes_b:
        return 0.0
    intersection = len(hashes_a & hashes_b)
    union = len(hashes_a | hashes_b)
    return intersection / union if union > 0 else 0.0


def check_originality(candidate: ExprNode,
                      pool: List[ExprNode] = None,
                      threshold: float = 0.70) -> Tuple[bool, float, str]:
    """
    检查候选因子的原创性 vs 现有因子池。

    Returns:
        (is_original, max_similarity, closest_match_name)
    """
    pool = pool or V3_FACTOR_POOL
    if not pool:
        return True, 0.0, ""

    max_sim = 0.0
    closest = ""
    for ref in pool:
        sim = ast_similarity(candidate, ref)
        if sim > max_sim:
            max_sim = sim
            closest = ref.to_string()[:60]

    is_original = max_sim < threshold
    return is_original, max_sim, closest


# ═══════════════════════════════════════════════════════════
# 约束3: Complexity Control
# ═══════════════════════════════════════════════════════════

def check_complexity(node: ExprNode, max_nodes: int = 28) -> Tuple[bool, int]:
    """
    AST节点数硬上限检查。
    """
    n = node.size()
    return n <= max_nodes, n


# ═══════════════════════════════════════════════════════════
# LLM 因子生成 Prompt
# ═══════════════════════════════════════════════════════════

AVAILABLE_PRIMITIVES_DESC = """
### 输入变量 (叶子节点)
- open, high, low, close, volume, vwap, returns

### 算术算子
- add(a,b), sub(a,b), mul(a,b), div(a,b)
- sqrt(x), abs(x), log(x), neg(x), inv(x), square(x), sign(x)

### 时间序列算子 (rolling, 第二个参数为窗口大小: 3,5,10,12,20,26,30,40,60)
- ts_sum(x, w)     : 滚动求和
- ts_mean(x, w)    : 滚动均值
- ts_std(x, w)     : 滚动标准差
- ts_min(x, w)     : 滚动最小值
- ts_max(x, w)     : 滚动最大值
- ts_delta(x, w)   : 滚动差值 (x[t] - x[t-w])
- ts_pct(x, w)     : 滚动变化率
- ts_rank(x, w)    : 滚动排名百分位 (0~1)
- ts_zscore(x, w)  : 滚动z-score
- ts_ema_decay(x, w): EMA偏离 (类似MACD)

### 截面算子 (跨股票排名)
- rank(x)   : 截面排名百分位 (0~1)
- zscore(x) : 截面z-score
- scale(x)  : 截面min-max归一化
"""


def build_generation_prompt(v3_factors_summary: str,
                            survivors_summary: str = "",
                            generation: int = 1) -> str:
    """
    构建LLM因子生成Prompt。

    Args:
        v3_factors_summary: V3现有因子池摘要
        survivors_summary: 上代幸存因子摘要（Gen2+使用）
        generation: 当前代数
    """

    v3_section = f"""
## V3 已验证因子池 (作为多样性参照, 请生成不同的因子)

{v3_factors_summary}
"""

    survivors_section = ""
    if survivors_summary and generation >= 2:
        survivors_section = f"""
## 上一代幸存因子 (知识蒸馏 — 这些因子在本地数据上表现优秀, 启发你的设计)

{survivors_summary}

**重要**: 不要复制上述因子。从它们的设计模式中汲取灵感:
- 哪些算子组合有效？
- 哪些市场逻辑被验证？
- 哪些窗口大小偏好出现？
"""

    prompt = f"""# AlphaAgent: 生成A股量化因子公式 (第{generation}代)

你是一位A股量化因子研究员。请基于以下算子和数据源，生成10个新的因子公式。

## 可用算子与数据
{AVAILABLE_PRIMITIVES_DESC}
{v3_section}
{survivors_section}
## 输出格式

每个因子一行，格式：
```
<表达式> | <因子名> | <经济逻辑> | <预期方向>
```

其中:
- 表达式: 使用上述算子的数学公式，如 `ts_zscore(div(close, ts_mean(close, 20)), 60)`
- 因子名: 简洁描述
- 经济逻辑: 1-2句解释为什么这个因子应该预测收益
- 预期方向: + 或 -

## 设计要求

1. **多样性**: 覆盖以下维度:
   - 流动性/微观结构 (价量关系、冲击成本代理)
   - 动量/反转 (趋势强度、极值回归)
   - 波动率/尾部风险 (波动不对称、尾部特征)
   - 资金流 (量价背离、资金流入流出)
   - 情绪/行为 (开盘跳空、隔夜效应、涨停跌停代理)

2. **复杂度控制**: 表达式节点数控制在8-20之间, 避免过于简单(没有区分度)或过于复杂(过拟合)

3. **截面信息**: 至少一半的因子应使用 rank() 或 zscore() 做截面标准化

4. **原创性**: 避免与V3因子高度重复, 寻找新的市场逻辑维度

请直接输出10个因子, 每行一个, 严格按照上述格式。"""

    return prompt


def build_alignment_prompt(factor_expr: str, factor_rationale: str) -> str:
    """
    构建Hypothesis-Factor Alignment评估Prompt。
    LLM自我评估因子的经济逻辑一致性。
    """
    return f"""# 因子经济逻辑一致性评估

请评估以下A股量化因子的经济逻辑是否合理。

## 因子
- 公式: `{factor_expr}`
- 声称逻辑: {factor_rationale}

## 评估维度 (1-5分)
1. 因果关系: 因子变化是否直接影响预期收益? (1=纯统计相关, 5=明确因果)
2. 经济学直觉: 是否能用市场微观结构/行为金融/风险定价解释? (1=无法解释, 5=教科书级逻辑)
3. 单调性: 因子值单调映射到预期收益是否合理? (1=无单调关系, 5=明确单调)
4. 稳健性: 逻辑是否依赖特定市场条件? (1=高度条件依赖, 5=跨条件稳健)
5. 拥挤度风险: 该逻辑是否容易被套利/拥挤? (1=极易拥挤, 5=难以套利)

## 输出格式
```
评分: X/5
[[详细分析: 每个维度1-2句]]
通过: YES 或 NO
```

评分>=3且通过=YES的因子才会被保留。"""


# ═══════════════════════════════════════════════════════════
# 候选因子数据结构
# ═══════════════════════════════════════════════════════════

@dataclass
class AlphaCandidate:
    """一个LLM生成的候选因子"""
    id: str
    expression: str
    name: str
    rationale: str
    direction: str = "+"       # + 或 -

    # AST
    ast_node: ExprNode = None

    # 三重约束结果
    originality_ok: bool = False
    originality_max_sim: float = 1.0
    originality_closest: str = ""
    alignment_ok: bool = False
    alignment_score: int = 0
    complexity_ok: bool = False
    complexity_nodes: int = 0

    # 评估
    icir: float = 0.0
    ic_mean: float = 0.0
    ic_std: float = 0.0
    hit_rate: float = 0.0
    nan_ratio: float = 0.0

    # 元数据
    generation: int = 0
    passed_triple: bool = False


# ═══════════════════════════════════════════════════════════
# AlphaAgent 主类
# ═══════════════════════════════════════════════════════════

class AlphaAgent:
    """
    LLM驱动因子挖掘引擎。

    工作流:
    1. 初始化: 加载V3因子池, 准备数据
    2. 生成: LLM生成候选因子表达式
    3. 三重过滤: Originality + Alignment + Complexity
    4. 评估: 计算ICIR
    5. 进化: 选择幸存者 → 知识蒸馏 → 下一轮
    """

    def __init__(self, config: AlphaAgentConfig = None):
        self.config = config or AlphaAgentConfig()

        # 因子池
        init_v3_pool(self.config.seed_factor_exprs)
        self.pool: List[ExprNode] = list(V3_FACTOR_POOL)
        self.all_candidates: List[AlphaCandidate] = []
        self.survivors_by_gen: Dict[int, List[AlphaCandidate]] = {}

        # 数据 (延迟加载)
        self.data: Dict[str, np.ndarray] = {}
        self.forward_returns: np.ndarray = None
        self.evaluator: ExpressionEvaluator = None

    # ── 数据加载 ──

    def load_data(self, data_dir: str = None, n_stocks: int = 500,
                   n_periods: int = 260) -> bool:
        """
        加载缓存因子数据。

        优先从 factor_alchemy/data/cache/ 读取已有因子数据。
        如果不可用, 尝试从 Tushare raw data 构建。
        """
        # 尝试从缓存加载
        cache_dir = PROJ_DIR / "data" / "cache"
        factor_cache = cache_dir / "factors_weekly_forward.parquet"

        if factor_cache.exists():
            try:
                import pandas as pd
                df = pd.read_parquet(factor_cache)
                print(f"  [AlphaAgent] 从缓存加载: {len(df)} 行")

                # 提取价格数据和收益率
                available_cols = set(df.columns)
                field_map = {
                    'open': 'open', 'high': 'high', 'low': 'low',
                    'close': 'close', 'volume': 'volume',
                }

                n = len(df['stock'].unique()) if 'stock' in df.columns else n_stocks
                t = len(df['date'].unique()) if 'date' in df.columns else n_periods

                stocks = df['stock'].unique()[:n] if 'stock' in df.columns else None

                for fname, col in field_map.items():
                    if col in available_cols:
                        pivoted = df.pivot(index='date', columns='stock', values=col)
                        self.data[fname] = pivoted.values.astype(float)
                        print(f"    {fname}: {self.data[fname].shape}")

                # 收益率
                if 'forward_return_1w' in available_cols:
                    pivoted = df.pivot(index='date', columns='stock',
                                       values='forward_return_1w')
                    self.forward_returns = pivoted.values.astype(float)
                    print(f"    forward_return: {self.forward_returns.shape}")

                if self.data:
                    self.evaluator = ExpressionEvaluator(self.data)
                    return True

            except Exception as e:
                print(f"  [AlphaAgent] 缓存加载失败: {e}")

        # Fallback: 尝试 data/loader
        try:
            from data.loader import load_factor_cache
            cache = load_factor_cache()
            if cache:
                # 提取 returns
                if 'returns' in cache:
                    self.forward_returns = cache['returns'].values
                self.data = cache
                self.evaluator = ExpressionEvaluator(self.data)
                return True
        except Exception:
            pass

        print("  [AlphaAgent] 无法加载数据, 将跳过ICIR评估")
        return False

    # ── 表达式解析 ──

    def parse_factor(self, expr_str: str) -> Optional[ExprNode]:
        """安全解析因子表达式字符串 → ExprNode"""
        expr_str = expr_str.strip()
        try:
            node = parse_expression(expr_str)
            if node and node.size() > 1:  # 至少不是纯终端
                return node
        except Exception:
            pass
        return None

    # ── LLM调用接口 ──

    def _call_llm(self, prompt: str, temperature: float = 0.8,
                  max_tokens: int = 3000) -> str:
        """
        调用 DeepSeek LLM API（伏羲统一客户端）。

        原实现: 将 prompt 写入文件供用户手动喂给 LLM。
        现在: 通过 llm_client.py 直接调用 DeepSeek v4-pro API。
        """
        try:
            from llm_client import get_llm_client
            client = get_llm_client()
            response = client.chat_with_system(
                system_prompt="你是 A 股量化因子研究员，专精于设计具有经济逻辑支撑的因子表达式。",
                user_prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response
        except Exception as e:
            # Fallback: 写入文件
            print(f"  [AlphaAgent] LLM API 调用失败: {e}，降级为文件模式")
            out_dir = Path(self.config.output_dir) if self.config.output_dir else PROJ_DIR / "output"
            out_dir.mkdir(parents=True, exist_ok=True)
            prompt_file = out_dir / f"alpha_agent_prompt_gen{getattr(self, '_gen_counter', 1)}.txt"
            with open(prompt_file, "w", encoding="utf-8") as f:
                f.write(prompt)
            print(f"  [AlphaAgent] Prompt已写入: {prompt_file}")
            return f"PROMPT_FILE:{prompt_file}"

    def generate_factors(self, v3_summary: str,
                         survivors_summary: str = "",
                         generation: int = 1) -> List[AlphaCandidate]:
        """调用LLM生成候选因子"""
        prompt = build_generation_prompt(v3_summary, survivors_summary, generation)
        response = self._call_llm(prompt, self.config.llm_temperature_gen)

        # 解析LLM响应
        candidates = self._parse_factor_response(response, generation)
        return candidates

    def _parse_factor_response(self, response: str, generation: int) -> List[AlphaCandidate]:
        """
        解析LLM响应中的因子。
        支持两种格式:
        1. 直接文本: expression | name | rationale | direction
        2. 文件路径: PROMPT_FILE:xxx → 需要用户手动提供响应
        """
        candidates = []

        # 如果是prompt文件标记, 跳过解析
        if response.startswith("PROMPT_FILE:"):
            return candidates

        # 逐行解析
        for line in response.strip().split("\n"):
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("```"):
                continue

            # 跳过标题行
            if line.startswith("表达式") or line.startswith("因子"):
                continue

            # 按 | 分割
            parts = [p.strip() for p in line.split("|")]
            if len(parts) < 3:
                # 尝试其他分隔符
                parts = [p.strip() for p in line.split("\t")]

            if len(parts) >= 4:
                expr, name, rationale, direction = parts[0], parts[1], parts[2], parts[3]
            elif len(parts) >= 3:
                expr, name, rationale = parts[0], parts[1], parts[2]
                direction = "+"
            else:
                # 可能是纯表达式
                expr = parts[0].strip()
                if not expr or '(' not in expr:
                    continue
                name = f"alpha_gen{generation}_{len(candidates):03d}"
                rationale = "LLM生成"
                direction = "+"

            # 清理表达式中的反引号
            expr = expr.replace("`", "").strip()

            cid = f"alpha_gen{generation}_{len(candidates):03d}"
            candidates.append(AlphaCandidate(
                id=cid, expression=expr, name=name,
                rationale=rationale, direction=direction,
                generation=generation,
            ))

        return candidates

    # ── 手动添加候选 (用于用户提供因子的场景) ──

    def add_manual_factors(self, factor_specs: List[dict],
                           generation: int = 1) -> List[AlphaCandidate]:
        """
        手动添加因子候选（绕过LLM调用）。
        factor_specs: [{"expression": "rank(div(close, ts_mean(close,20)))",
                        "name": "价格偏离度",
                        "rationale": "价格相对20日均线偏离, 截面排名",
                        "direction": "+"}, ...]
        """
        candidates = []
        for i, spec in enumerate(factor_specs):
            cid = f"alpha_gen{generation}_{i:03d}"
            candidates.append(AlphaCandidate(
                id=cid,
                expression=spec["expression"],
                name=spec.get("name", f"factor_{i}"),
                rationale=spec.get("rationale", ""),
                direction=spec.get("direction", "+"),
                generation=generation,
            ))
        return candidates

    # ── 三重约束过滤 ──

    def apply_triple_constraints(self, candidates: List[AlphaCandidate],
                                 verbose: bool = True) -> List[AlphaCandidate]:
        """
        对候选因子列表应用三重约束过滤。

        约束1: Originality — AST相似度 vs 因子池
        约束2: Alignment — LLM自我评估经济逻辑
        约束3: Complexity — AST节点数硬上限
        """
        passed = []

        for c in candidates:
            # 先解析AST
            node = self.parse_factor(c.expression)
            if node is None:
                if verbose:
                    print(f"  [SKIP] {c.id}: 无法解析表达式 '{c.expression[:50]}'")
                continue
            c.ast_node = node

            # 约束1: Originality
            is_orig, max_sim, closest = check_originality(
                node, self.pool, self.config.originality_threshold)
            c.originality_ok = is_orig
            c.originality_max_sim = max_sim
            c.originality_closest = closest
            if not is_orig and verbose:
                print(f"  [ORIG] {c.id}: sim={max_sim:.2f} vs {closest[:40]}... — 拒绝")

            # 约束3: Complexity
            is_simple, n_nodes = check_complexity(node, self.config.max_complexity)
            c.complexity_ok = is_simple
            c.complexity_nodes = n_nodes
            if not is_simple and verbose:
                print(f"  [COMPLEX] {c.id}: {n_nodes} nodes > {self.config.max_complexity} — 拒绝")

            # 约束2: Alignment (简化版 — 基于rationale长度和关键词做粗筛)
            # 完整版需要LLM调用, 这里先用启发式
            c.alignment_ok, c.alignment_score = self._heuristic_alignment(c)
            if not c.alignment_ok and verbose:
                print(f"  [ALIGN] {c.id}: score={c.alignment_score} < {self.config.hypothesis_min_score} — 拒绝")

            # 三重通过
            c.passed_triple = c.originality_ok and c.alignment_ok and c.complexity_ok
            if c.passed_triple:
                passed.append(c)

        if verbose:
            print(f"  [Triple] {len(passed)}/{len(candidates)} 通过三重约束")
        return passed

    def _heuristic_alignment(self, candidate: AlphaCandidate) -> Tuple[bool, int]:
        """
        启发式经济逻辑一致性评分 (替代LLM评估)。
        基于rationale文本质量和因子结构评分。
        """
        rationale = candidate.rationale
        expr = candidate.expression
        score = 1

        # 长度不够 → 降分
        if len(rationale) < 15:
            return False, 1

        # 关键词检测 — 经济逻辑相关
        econ_keywords = [
            "流动性", "动量", "反转", "波动", "风险", "溢价", "拥挤",
            "资金流", "情绪", "行为", "错杀", "套利", "博弈", "机构",
            "散户", "筹码", "冲击", "成本", "趋势", "背离", "加速",
            "liquidity", "momentum", "reversal", "volatility", "risk",
            "premium", "crowding", "flow", "sentiment", "behavioral",
            "arbitrage", "institutional", "retail", "cost", "trend",
            "divergence", "acceleration",
        ]
        keyword_hits = sum(1 for kw in econ_keywords if kw.lower() in rationale.lower())
        score += min(keyword_hits, 3)

        # 因子结构质量 — 检查是否使用了截面算子
        if "rank(" in expr:
            score += 1
        if "zscore(" in expr:
            score += 1
        if "ts_" in expr:
            score += 1

        # 表达式太简单 → 降分
        if candidate.ast_node and candidate.ast_node.size() < 4:
            score -= 2

        capped = max(1, min(5, score))
        return capped >= self.config.hypothesis_min_score, capped

    # ── ICIR评估 ──

    def evaluate_candidates(self, candidates: List[AlphaCandidate],
                            verbose: bool = True) -> List[AlphaCandidate]:
        """计算候选因子的ICIR"""
        if self.evaluator is None or self.forward_returns is None:
            print("  [Eval] 无数据, 跳过ICIR计算")
            return candidates

        for c in candidates:
            if c.ast_node is None:
                c.ast_node = self.parse_factor(c.expression)
            if c.ast_node is None:
                continue

            try:
                factor_values = self.evaluator.evaluate(c.ast_node)
            except Exception:
                continue

            if factor_values is None:
                continue

            c.nan_ratio = np.mean(np.isnan(factor_values))

            # 计算ICIR
            icir, ic_mean, ic_std, hit_rate = self._compute_stats(factor_values)
            c.icir = icir
            c.ic_mean = ic_mean
            c.ic_std = ic_std
            c.hit_rate = hit_rate

        if verbose:
            best = max(candidates, key=lambda x: abs(x.icir)) if candidates else None
            if best:
                print(f"  [Eval] {len(candidates)} 候选, 最佳 ICIR={best.icir:+.3f} ({best.name})")

        return candidates

    def _compute_stats(self, factor_values: np.ndarray) -> Tuple[float, float, float, float]:
        """计算ICIR, IC_mean, IC_std, hit_rate"""
        fwd = self.forward_returns
        if factor_values.shape != fwd.shape:
            return 0.0, 0.0, 0.0, 0.0

        T, N = factor_values.shape
        if T < 10 or N < 30:
            return 0.0, 0.0, 0.0, 0.0

        fvals = np.asarray(factor_values, dtype=float)
        rvals = np.asarray(fwd, dtype=float)

        valid = np.isfinite(fvals) & np.isfinite(rvals)
        valid_count = valid.sum(axis=1)

        # 行均值填充NaN
        fmean = np.nanmean(np.where(valid, fvals, np.nan), axis=1, keepdims=True)
        rmean = np.nanmean(np.where(valid, rvals, np.nan), axis=1, keepdims=True)
        fvals = np.where(valid, fvals, np.broadcast_to(fmean, fvals.shape))
        rvals = np.where(valid, rvals, np.broadcast_to(rmean, rvals.shape))

        from scipy.stats import rankdata
        fr = rankdata(fvals, axis=1)
        rr = rankdata(rvals, axis=1)

        fc = fr - fr.mean(axis=1, keepdims=True)
        rc = rr - rr.mean(axis=1, keepdims=True)
        num = (fc * rc).sum(axis=1)
        den = np.sqrt((fc ** 2).sum(axis=1) * (rc ** 2).sum(axis=1))
        ics = np.divide(num, den, out=np.full(T, np.nan), where=den > 0)
        ics = np.where(valid_count >= 30, ics, np.nan)
        ics_valid = ics[np.isfinite(ics)]

        if len(ics_valid) < 10:
            return 0.0, 0.0, 0.0, 0.0

        ic_mean = np.mean(ics_valid)
        ic_std = np.std(ics_valid, ddof=1)
        icir = ic_mean / max(ic_std, 0.001)
        hit_rate = np.mean(ics_valid > 0)

        return icir, ic_mean, ic_std, hit_rate

    # ── EFS 进化循环 ──

    def evolution_loop(self, verbose: bool = True) -> Dict[int, List[AlphaCandidate]]:
        """
        EFS进化循环主循环。

        每一代:
        1. LLM生成候选因子
        2. 三重约束过滤
        3. ICIR评估
        4. 选择幸存者 (|ICIR| >= threshold, 选Top-K)
        5. 知识蒸馏 → 下一轮提示
        """
        # V3摘要
        v3_summary = self._build_v3_summary()
        survivors_summary = ""

        for gen in range(1, self.config.n_generations + 1):
            if verbose:
                print(f"\n{'='*60}")
                print(f"  EFS 第{gen}代 / {self.config.n_generations}")
                print(f"{'='*60}")

            # 1. LLM生成
            candidates = self.generate_factors(v3_summary, survivors_summary, gen)

            if not candidates:
                print(f"  [Gen{gen}] LLM未返回候选, 跳过")
                continue

            print(f"  [Gen{gen}] LLM生成 {len(candidates)} 个候选")

            # 2. 三重约束
            passed = self.apply_triple_constraints(candidates, verbose=verbose)
            self.all_candidates.extend(candidates)

            if not passed:
                print(f"  [Gen{gen}] 无候选通过三重约束, 跳过")
                continue

            # 3. ICIR评估
            passed = self.evaluate_candidates(passed, verbose=verbose)

            # 4. 选择幸存者
            survivors = [
                c for c in passed
                if abs(c.icir) >= self.config.icir_threshold
            ]
            survivors.sort(key=lambda x: -abs(x.icir))
            survivors = survivors[:self.config.n_survivors]

            self.survivors_by_gen[gen] = survivors
            if verbose and survivors:
                print(f"  [Gen{gen}] {len(survivors)} 幸存者:")
                for s in survivors:
                    print(f"    {s.id}: ICIR={s.icir:+.3f} | {s.name} | {s.expression[:50]}")

            # 5. 加入因子池 (用于原始性检查)
            for s in survivors:
                if s.ast_node:
                    self.pool.append(s.ast_node)

            # 6. 知识蒸馏摘要
            survivors_summary = self._build_survivors_summary(gen)

        return self.survivors_by_gen

    def _build_v3_summary(self) -> str:
        """构建V3因子池摘要"""
        lines = []
        for i, node in enumerate(V3_FACTOR_POOL[:len(V3_FACTOR_POOL)]):
            expr = node.to_string()
            lines.append(f"{i+1}. `{expr[:80]}`")
        return "\n".join(lines)

    def _build_survivors_summary(self, gen: int) -> str:
        """构建幸存因子知识蒸馏摘要"""
        survivors = self.survivors_by_gen.get(gen, [])
        if not survivors:
            return ""

        lines = [f"## 第{gen}代幸存因子 (ICIR >= {self.config.icir_threshold})\n"]
        for s in survivors:
            lines.append(
                f"- **{s.name}**: `{s.expression[:60]}`\n"
                f"  ICIR={s.icir:+.3f}, 逻辑: {s.rationale[:80]}"
            )

        # 提取模式总结
        if len(survivors) >= 2:
            patterns = self._extract_patterns(survivors)
            lines.append(f"\n### 设计模式总结\n{patterns}")

        return "\n".join(lines)

    def _extract_patterns(self, survivors: List[AlphaCandidate]) -> str:
        """从幸存因子中提取设计模式"""
        op_counts = defaultdict(int)
        input_counts = defaultdict(int)
        window_sizes = []

        for s in survivors:
            if s.ast_node is None:
                continue
            expr = s.expression
            for op_name in PRIMITIVE_BY_NAME:
                if op_name in expr:
                    op_counts[op_name] += 1
            for inp in INPUT_PRIMITIVES:
                if inp.name in expr:
                    input_counts[inp.name] += 1
            # 提取窗口大小
            w_matches = re.findall(r'ts_\w+\(.*?,\s*(\d+)\)', expr)
            for w in w_matches:
                window_sizes.append(int(w))

        top_ops = sorted(op_counts.items(), key=lambda x: -x[1])[:3]
        top_inputs = sorted(input_counts.items(), key=lambda x: -x[1])[:3]

        patterns = []
        if top_ops:
            patterns.append(f"频繁算子: {', '.join(f'{op}({n}次)' for op, n in top_ops)}")
        if top_inputs:
            patterns.append(f"频繁输入: {', '.join(f'{inp}({n}次)' for inp, n in top_inputs)}")
        if window_sizes:
            from statistics import median
            patterns.append(f"中位窗口: {int(median(window_sizes))}")

        return "\n".join(f"- {p}" for p in patterns)

    # ── 导出 ──

    def export_results(self):
        """导出所有候选和幸存因子到JSON"""
        out_dir = Path(self.config.output_dir) if self.config.output_dir else PROJ_DIR / "output"
        out_dir.mkdir(parents=True, exist_ok=True)

        # 导出所有候选
        all_data = []
        for c in self.all_candidates:
            all_data.append({
                "id": c.id,
                "expression": c.expression,
                "name": c.name,
                "rationale": c.rationale,
                "direction": c.direction,
                "generation": c.generation,
                "passed_triple": c.passed_triple,
                "originality_ok": c.originality_ok,
                "originality_max_sim": c.originality_max_sim,
                "alignment_ok": c.alignment_ok,
                "alignment_score": c.alignment_score,
                "complexity_ok": c.complexity_ok,
                "complexity_nodes": c.complexity_nodes,
                "icir": c.icir,
                "ic_mean": c.ic_mean,
                "ic_std": c.ic_std,
                "hit_rate": c.hit_rate,
                "nan_ratio": c.nan_ratio,
            })

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        all_file = out_dir / f"alpha_agent_all_{timestamp}.json"
        with open(all_file, "w", encoding="utf-8") as f:
            json.dump(all_data, f, ensure_ascii=False, indent=2)

        # 导出幸存者
        survivors_data = {}
        for gen, survs in self.survivors_by_gen.items():
            survivors_data[f"gen_{gen}"] = [
                {
                    "id": s.id,
                    "expression": s.expression,
                    "name": s.name,
                    "rationale": s.rationale,
                    "icir": s.icir,
                    "ic_mean": s.ic_mean,
                    "ic_std": s.ic_std,
                    "hit_rate": s.hit_rate,
                    "complexity_nodes": s.complexity_nodes,
                }
                for s in survs
            ]

        surv_file = out_dir / f"alpha_agent_survivors_{timestamp}.json"
        with open(surv_file, "w", encoding="utf-8") as f:
            json.dump(survivors_data, f, ensure_ascii=False, indent=2)

        print(f"\n  [Export] 全部候选: {all_file}")
        print(f"  [Export] 幸存因子: {surv_file}")
        return all_file, surv_file


# ═══════════════════════════════════════════════════════════
# CLI入口
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    config = AlphaAgentConfig(
        n_generations=1,
        n_candidates_per_gen=1,
        output_dir=str(PROJ_DIR / "output"),
    )

    agent = AlphaAgent(config)

    # 加载数据
    data_loaded = agent.load_data()
    if not data_loaded:
        print("警告: 无法加载数据, 将仅运行结构验证")

    # 测试: 手动添加一些候选因子
    test_factors = [
        {
            "expression": "neg(ts_zscore(div(close, ts_mean(close, 20)), 60))",
            "name": "价格偏离度反转",
            "rationale": "价格偏离20日均线的长期z-score, 极端偏离后反转。流动性/微观结构驱动。",
            "direction": "+",
        },
        {
            "expression": "rank(div(sub(ts_max(high, 20), close), add(ts_std(close, 20), 1e-6)))",
            "name": "回落幅度",
            "rationale": "当前价格距离20日最高价的回落幅度(标准化), 捕捉短期超卖反弹。均值回归逻辑。",
            "direction": "+",
        },
        {
            "expression": "sub(ts_pct(volume, 5), ts_pct(close, 5))",
            "name": "量价背离",
            "rationale": "成交量增速vs价格增速的差异, 正的量价背离通常预示趋势加速或反转。资金流信号。",
            "direction": "-",
        },
        {
            "expression": "neg(div(ts_mean(volume, 5), ts_mean(volume, 20)))",
            "name": "缩量",
            "rationale": "近期成交量相对20日均量萎缩, 缩量下跌后可能反转。筹码锁定信号。",
            "direction": "+",
        },
        {
            "expression": "rank(mul(ts_zscore(close, 20), neg(ts_std(returns, 20))))",
            "name": "动量×低波",
            "rationale": "价格动量强度×负波动率, 捕捉稳健上涨趋势。低波动异象+动量增强。",
            "direction": "+",
        },
    ]

    candidates = agent.add_manual_factors(test_factors)
    print(f"\n添加 {len(candidates)} 个测试因子")

    # 三重约束
    passed = agent.apply_triple_constraints(candidates)
    print(f"\n三重约束: {len(passed)}/{len(candidates)} 通过")

    # ICIR评估
    if data_loaded:
        passed = agent.evaluate_candidates(passed)
        for c in passed:
            print(f"  {c.id}: ICIR={c.icir:+.3f} IC_mean={c.ic_mean:+.3f} "
                  f"hit={c.hit_rate:.1%} {c.name}")

    # 导出
    agent.export_results()
