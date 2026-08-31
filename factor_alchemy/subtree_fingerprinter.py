# -*- coding: utf-8 -*-
"""
Subtree Fingerprinter — FSA (频繁子树规避) 机制
==================================================

对标中金 CICC Loop Engineering 报告中的 FSA (Frequent Subtree Avoidance) 机制。

核心功能:
  1. 从因子表达式（Python 代码 / DSL 字符串）提取结构骨架指纹
  2. 扫描因子库，统计每个结构骨架的出现频率
  3. 超过阈值（默认 15%）的骨架列入禁止列表
  4. 在生成阶段自动排除被禁骨架
  5. 支持动态阈值（根据库规模调整）

设计原理:
  - 传统的相关性过滤只在因子值层面去重，无法发现 sub(ma(overnight,60), ma(close,20))
    和 sub(ma(overnight,80), ma(close,30)) 本质上是同一结构骨架的不同参数变体
  - FSA 在结构层面去重：将具体字段名/窗口参数抽象为占位符，只保留算子结构
  - 如 "sub(ma(FIELD, N1), ma(FIELD, N2))" 是一个骨架，
    不管 FIELD 是 overnight 还是 close，N1 是 20 还是 60

集成路径:
  - Ralph Loop G 阶段: 生成前检查 FSA 禁止列表，排除被禁骨架
  - MetaController auto_cycle: 每次循环后扫描库，更新 FSA 状态
  - Experience Memory: 将 FSA 状态作为 P_fail 的一部分持久化

用法:
    from subtree_fingerprinter import SubtreeFingerprinter

    fsa = SubtreeFingerprinter()
    
    # 扫描因子库
    report = fsa.scan_library(factors, threshold=0.15)
    
    # 检查单个表达式是否被禁
    is_forbidden, reason = fsa.check_forbidden("sub(ma(overnight,60), ma(close,20))")
    
    # 获取禁止骨架列表（注入 LLM 生成约束）
    forbidden_list = fsa.get_forbidden_for_generation()
"""

import re
import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import Counter

# ── 默认路径 ──────────────────────────────────────────────
DATA_DIR = Path(__file__).parent.parent.parent / "data"
FSA_STATE_PATH = DATA_DIR / "fsa_state.json"


# ── 算子/字段识别正则 ────────────────────────────────────

# 时间序列算子（会引入窗口参数）
TS_OPERATORS = [
    "ma", "ts_mean", "ts_std", "ts_min", "ts_max", "ts_sum",
    "ts_delta", "ts_rank", "ts_skewness", "ts_kurtosis", "ts_corr",
    "ts_cov", "ts_regression", "ts_decay_linear", "ts_delay",
    "ema", "wma", "delta", "roc",
]

# 截面算子
CS_OPERATORS = [
    "rank", "rank_cs", "scale", "zscore", "cs_rank", "cs_zscore",
    "demean", "normalize",
]

# 算术/逻辑算子
MATH_OPERATORS = [
    "add", "sub", "mul", "div", "pow", "sqrt", "abs", "log", "sign",
    "min", "max", "if_else", "clip",
]

# 所有算子
ALL_OPERATORS = TS_OPERATORS + CS_OPERATORS + MATH_OPERATORS

# 字段名（在表达式中出现的量价字段）
FIELD_NAMES = [
    "open_p", "high_p", "low_p", "close_p", "volume_p", "amount_p",
    "open", "high", "low", "close", "volume", "amount", "vwap",
    "overnight", "intraday", "returns", "amplitude", "turnover",
    "market_cap", "mv", "float_mv", "up_shadow", "down_shadow",
    "hl_ratio", "atr", "swing", "gap", "limit", "bid_ask",
    "buy_vol", "sell_vol", "buy_amount", "sell_amount",
    "roe", "eps", "pe", "pb", "roa", "gross_margin", "net_margin",
    "moneyflow_lg", "moneyflow_sm", "moneyflow_elg",
]

# 窗口参数模式（数字）
WINDOW_PATTERN = re.compile(r'\b(\d+)\b')


@dataclass
class SubtreeSkeleton:
    """表达式树的结构骨架"""
    skeleton: str           # 抽象后的骨架，如 "sub(ma(FIELD,N1), ma(FIELD,N2))"
    original_expressions: List[str] = field(default_factory=list)  # 原始表达式列表
    count: int = 0
    forbidden: bool = False
    forbidden_at: Optional[str] = None


@dataclass
class FSAReport:
    """FSA 扫描报告"""
    timestamp: str
    total_factors: int
    total_skeletons: int
    forbidden_count: int
    forbidden_skeletons: List[str]
    max_concentration: float       # 最高集中度
    max_concentration_skeleton: str
    warnings: List[str] = field(default_factory=list)


class SubtreeFingerprinter:
    """
    FSA 频繁子树规避器。

    从因子的 Python 代码表达式 / DSL 字符串中提取结构骨架，
    维护禁止列表，防止因子库同质化。
    """

    def __init__(
        self,
        state_path: Path = FSA_STATE_PATH,
        default_threshold: float = 0.15,   # 默认 15% 阈值
        min_factors_to_trigger: int = 10,   # 最少因子数才触发
    ):
        self.state_path = state_path
        self.threshold = default_threshold
        self.min_factors_to_trigger = min_factors_to_trigger
        self.skeletons: Dict[str, SubtreeSkeleton] = {}
        self._load_state()

    # ═══════════════════════════════════════════════════════════
    # 表达式解析与指纹提取
    # ═══════════════════════════════════════════════════════════

    def parse_expression(self, expression: str) -> str:
        """
        将原始因子表达式解析为结构骨架。

        转换规则:
          - 所有字段名 → "FIELD"
          - 所有窗口参数 → "N{i}"（按出现顺序编号）
          - 保留算子结构

        示例:
          "sub(ma(overnight, 60), ma(close, 20))"
            → "sub(ma(FIELD,N1), ma(FIELD,N2))"

          "ts_delta(ts_rank(volume, 120), 20)"
            → "ts_delta(ts_rank(FIELD,N1), N2)"

          "rank(ts_mean(close, 60) - ts_mean(close, 20))"
            → "rank(sub(ma(FIELD,N1), ma(FIELD,N2)))"

        Parameters
        ----------
        expression: 因子表达式（Python 代码字符串或 DSL）

        Returns
        -------
        skeleton: 抽象后的结构骨架
        """
        expr = expression.strip()

        # Step 1: 提取所有数字参数（窗口大小、阈值等）
        numbers = WINDOW_PATTERN.findall(expr)
        # 去重但保持顺序
        seen_nums = {}
        unique_nums = []
        for n in numbers:
            if n not in seen_nums:
                seen_nums[n] = f"N{len(seen_nums) + 1}"
                unique_nums.append(n)

        # 替换数字 → 编号占位符（从大到小替换，避免 120 被替换为 N12 的问题）
        nums_sorted = sorted(unique_nums, key=lambda x: int(x), reverse=True)
        for n in nums_sorted:
            placeholder = seen_nums[n]
            # 只替换独立数字（被非字母非数字包围的数字）
            expr = re.sub(rf'\b{n}\b', placeholder, expr)

        # Step 2: 提取所有字段名，替换为 "FIELD"
        # 按长度降序排序，避免 "close" 在 "float_close" 中被部分替换
        fields_sorted = sorted(FIELD_NAMES, key=len, reverse=True)
        for field in fields_sorted:
            pattern = rf'\b{re.escape(field)}\b'
            expr = re.sub(pattern, 'FIELD', expr, flags=re.IGNORECASE)

        # Step 3: 规范化空格
        expr = re.sub(r'\s+', '', expr)

        # Step 4: 标准化算子名称（aliases → canonical）
        ALIAS_MAP = {
            "ts_mean": "ma", "ts_delta": "delta",
            "ts_std": "std", "ts_rank": "rank_ts",
            "ts_corr": "corr", "ts_min": "min_ts",
            "ts_max": "max_ts", "ts_sum": "sum_ts",
            "rank_cs": "rank", "cs_rank": "rank",
            "cs_zscore": "zscore",
        }
        for alias, canonical in ALIAS_MAP.items():
            expr = re.sub(rf'\b{re.escape(alias)}\b', canonical, expr)

        return expr

    def fingerprint(self, expression: str) -> str:
        """快捷方法：expression → skeleton"""
        return self.parse_expression(expression)

    # ═══════════════════════════════════════════════════════════
    # 因子库扫描
    # ═══════════════════════════════════════════════════════════

    def scan_library(
        self,
        factors: List[Dict],
        threshold: Optional[float] = None,
        persist: bool = True,
    ) -> FSAReport:
        """
        扫描因子库，统计每个结构骨架的出现频率，标记超过阈值的骨架。

        Parameters
        ----------
        factors: 因子列表，每个包含 'factor_name', 'formula' (或 'expression')
        threshold: 禁止阈值（默认 self.threshold = 15%）
        persist: 是否持久化到文件

        Returns
        -------
        FSAReport: 扫描报告
        """
        if threshold is None:
            threshold = self.threshold

        n_total = len(factors)
        if n_total < self.min_factors_to_trigger:
            return FSAReport(
                timestamp=datetime.now().isoformat(),
                total_factors=n_total,
                total_skeletons=0,
                forbidden_count=0,
                forbidden_skeletons=[],
                max_concentration=0.0,
                max_concentration_skeleton="",
                warnings=[f"因子数 {n_total} < 触发阈值 {self.min_factors_to_trigger}，跳過FSA扫描"]
            )

        # 提取所有骨架
        skeleton_counter = Counter()
        skeleton_expressions: Dict[str, List[str]] = {}

        for f in factors:
            expr = f.get("formula", f.get("expression", ""))
            name = f.get("factor_name", f.get("name", "unknown"))
            if not expr:
                continue

            try:
                skeleton = self.parse_expression(expr)
            except Exception:
                continue

            skeleton_counter[skeleton] += 1
            skeleton_expressions.setdefault(skeleton, []).append(
                f"{name}: {expr[:80]}"
            )

        # 计算阈值对应的绝对数量
        threshold_count = max(int(n_total * threshold), 2)

        # 更新骨架状态
        self.skeletons = {}
        warnings = []
        forbidden = []
        max_conc = 0.0
        max_conc_skel = ""

        for skeleton, count in skeleton_counter.most_common():
            conc = count / n_total
            is_forbidden = count >= threshold_count

            skel = SubtreeSkeleton(
                skeleton=skeleton,
                original_expressions=skeleton_expressions.get(skeleton, [])[:5],
                count=count,
                forbidden=is_forbidden,
                forbidden_at=datetime.now().isoformat() if is_forbidden else None,
            )
            self.skeletons[skeleton] = skel

            if is_forbidden:
                forbidden.append(skeleton)
                warnings.append(
                    f"⚠️ FSA 触发: '{skeleton}' 出现 {count}/{n_total} "
                    f"({conc:.1%}) ≥ 阈值 {threshold:.0%}"
                )

            if conc > max_conc:
                max_conc = conc
                max_conc_skel = skeleton

        # 持久化
        if persist:
            self._save_state()

        return FSAReport(
            timestamp=datetime.now().isoformat(),
            total_factors=n_total,
            total_skeletons=len(skeleton_counter),
            forbidden_count=len(forbidden),
            forbidden_skeletons=forbidden,
            max_concentration=max_conc,
            max_concentration_skeleton=max_conc_skel,
            warnings=warnings,
        )

    # ═══════════════════════════════════════════════════════════
    # 禁止检查
    # ═══════════════════════════════════════════════════════════

    def check_forbidden(self, expression: str) -> Tuple[bool, str]:
        """
        检查表达式是否使用了被禁止的结构骨架。

        Returns
        -------
        (is_forbidden, reason)
        """
        if not expression or not self.skeletons:
            return False, ""

        try:
            skeleton = self.parse_expression(expression)
        except Exception:
            return False, ""

        if skeleton in self.skeletons and self.skeletons[skeleton].forbidden:
            skel = self.skeletons[skeleton]
            return True, (
                f"FSA 禁止: 骨架 '{skeleton}' 出现 {skel.count} 次，"
                f"超过阈值，已被冻结"
            )

        return False, ""

    def check_expression(self, expression: str) -> bool:
        """快捷检查：是否被禁止（返回 True=被禁止）"""
        is_forbidden, _ = self.check_forbidden(expression)
        return is_forbidden

    def get_forbidden_skeletons(self) -> List[str]:
        """获取所有被禁止的骨架"""
        return [s for s, skel in self.skeletons.items() if skel.forbidden]

    def get_forbidden_for_generation(self) -> str:
        """
        获取禁止骨架的文本描述，用于注入 LLM 生成约束。

        Returns
        -------
        str: 格式化的禁止骨架列表
        """
        forbidden = self.get_forbidden_skeletons()
        if not forbidden:
            return ""

        # 显示前5个
        lines = ["## ⚠️ 以下因子结构骨架已被 FSA 冻结（请勿生成）："]
        for i, skel in enumerate(forbidden[:5], 1):
            skel_info = self.skeletons[skel]
            lines.append(f"{i}. `{skel}` ({skel_info.count} 个变体)")
            if skel_info.original_expressions:
                lines.append(f"   示例: `{skel_info.original_expressions[0][:80]}`")

        if len(forbidden) > 5:
            lines.append(f"... 另有 {len(forbidden) - 5} 个被禁骨架")

        return "\n".join(lines)

    # ═══════════════════════════════════════════════════════════
    # 动态阈值调整
    # ═══════════════════════════════════════════════════════════

    def adjust_threshold(
        self, n_factors: int, search_stage: str = "exploration"
    ) -> float:
        """
        根据因子库规模动态调整阈值。

        - 探索期（库小）: 阈值宽松，允许更多骨架共存
        - 饱和期（库大）: 阈值收紧，更积极地触发 FSA

        对标中金报告：15% 阈值在 69 因子库中 ≈10个同骨架，在更大库中需降低。
        """
        if search_stage == "cold_start":
            return 0.30    # 冷启动：30%，几乎不触发
        elif n_factors < 30:
            return 0.20    # 探索期：20%
        elif n_factors < 70:
            return 0.15    # 积累期：15%（中金默认值）
        elif n_factors < 150:
            return 0.12    # 成熟期：12%
        else:
            return 0.08    # 饱和期：8%

    # ═══════════════════════════════════════════════════════════
    # 因子哈希（用于检查点系统）
    # ═══════════════════════════════════════════════════════════

    @staticmethod
    def expression_hash(expression: str) -> str:
        """计算因子表达式的哈希指纹（用于去重/检查点）"""
        skeleton = SubtreeFingerprinter()
        try:
            normalized = skeleton.parse_expression(expression)
        except Exception:
            normalized = expression.strip().replace(" ", "")
        return hashlib.sha256(normalized.encode()).hexdigest()[:16]

    @staticmethod
    def factor_hash(factor: Dict) -> str:
        """计算因子完整哈希（表达式 + 参数）"""
        expr = factor.get("formula", factor.get("expression", ""))
        return SubtreeFingerprinter.expression_hash(expr)

    # ═══════════════════════════════════════════════════════════
    # 状态持久化
    # ═══════════════════════════════════════════════════════════

    def _save_state(self):
        """持久化 FSA 状态到 JSON"""
        state = {
            "_updated": datetime.now().isoformat(),
            "threshold": self.threshold,
            "min_factors_to_trigger": self.min_factors_to_trigger,
            "skeletons": {},
        }
        for skel_str, skel_obj in self.skeletons.items():
            state["skeletons"][skel_str] = {
                "skeleton": skel_obj.skeleton,
                "count": skel_obj.count,
                "forbidden": skel_obj.forbidden,
                "forbidden_at": skel_obj.forbidden_at,
                "sample_expressions": skel_obj.original_expressions[:3],
            }

        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, ensure_ascii=False)

    def _load_state(self):
        """从 JSON 恢复 FSA 状态"""
        if not self.state_path.exists():
            return

        try:
            with open(self.state_path, "r", encoding="utf-8") as f:
                state = json.load(f)

            self.threshold = state.get("threshold", self.threshold)
            self.min_factors_to_trigger = state.get(
                "min_factors_to_trigger", self.min_factors_to_trigger
            )

            for skel_str, skel_data in state.get("skeletons", {}).items():
                self.skeletons[skel_str] = SubtreeSkeleton(
                    skeleton=skel_data.get("skeleton", skel_str),
                    original_expressions=skel_data.get("sample_expressions", []),
                    count=skel_data.get("count", 0),
                    forbidden=skel_data.get("forbidden", False),
                    forbidden_at=skel_data.get("forbidden_at"),
                )
        except Exception:
            pass  # 状态文件损坏则从零开始

    # ═══════════════════════════════════════════════════════════
    # 统计与查询
    # ═══════════════════════════════════════════════════════════

    def get_status(self) -> Dict:
        """获取 FSA 当前状态摘要"""
        total = len(self.skeletons)
        forbidden = sum(1 for s in self.skeletons.values() if s.forbidden)
        return {
            "total_skeletons": total,
            "forbidden_skeletons": forbidden,
            "threshold": self.threshold,
            "top_skeletons": sorted(
                [
                    {"skeleton": s, "count": sk.count, "forbidden": sk.forbidden}
                    for s, sk in self.skeletons.items()
                ],
                key=lambda x: x["count"],
                reverse=True,
            )[:10],
        }

    def get_diversity_score(self) -> float:
        """计算因子库的骨架多样性分数（0-1，越高越多样）"""
        if not self.skeletons:
            return 1.0

        counts = [sk.count for sk in self.skeletons.values()]
        total = sum(counts)
        if total == 0:
            return 1.0

        # 使用 normalized entropy 作为多样性指标
        import math
        entropy = -sum(
            (c / total) * math.log(c / total, 2)
            for c in counts if c > 0
        )
        max_entropy = math.log(len(counts), 2) if len(counts) > 1 else 1.0
        return entropy / max_entropy if max_entropy > 0 else 1.0


# ── 便捷函数 ──────────────────────────────────────────────

_default_fsa: Optional[SubtreeFingerprinter] = None


def get_fsa() -> SubtreeFingerprinter:
    global _default_fsa
    if _default_fsa is None:
        _default_fsa = SubtreeFingerprinter()
    return _default_fsa


# ── 测试 ──────────────────────────────────────────────────

if __name__ == "__main__":
    fsa = SubtreeFingerprinter()

    # 测试表达式解析
    test_expressions = [
        "sub(ma(overnight, 60), ma(close, 20))",
        "sub(ma(overnight, 80), ma(close, 30))",
        "ts_delta(ts_rank(volume, 120), 20)",
        "rank(ts_mean(close, 60) - ts_mean(close, 20))",
        "div(sub(ma(overnight, 60), ma(close, 20)), ts_std(amplitude, 30))",
        "ma(delta(overnight, 5), 40)",
        "rank(ma(down_shadow, 60) - ts_mean(hl_ratio, 20))",
    ]

    print("=" * 60)
    print("  FSA Subtree Fingerprinter 测试")
    print("=" * 60)

    for expr in test_expressions:
        skeleton = fsa.parse_expression(expr)
        print(f"\n  原始: {expr}")
        print(f"  骨架: {skeleton}")
        print(f"  哈希: {fsa.expression_hash(expr)}")

    # 测试库扫描
    print("\n" + "=" * 60)
    print("  因子库扫描测试")
    print("=" * 60)

    mock_factors = [
        {"factor_name": f"f{i}", "formula": expr}
        for i, expr in enumerate(test_expressions * 5)  # 模拟 35 个因子
    ]

    report = fsa.scan_library(mock_factors, threshold=0.10, persist=False)
    print(f"\n  总因子: {report.total_factors}")
    print(f"  骨架数: {report.total_skeletons}")
    print(f"  被禁数: {report.forbidden_count}")
    print(f"  最大集中: {report.max_concentration:.1%} ({report.max_concentration_skeleton[:60]})")
    for w in report.warnings:
        print(f"  {w}")

    # 测试禁止检查
    print("\n" + "=" * 60)
    print("  禁止检查测试")
    print("=" * 60)

    forbidden_context = fsa.get_forbidden_for_generation()
    print(forbidden_context if forbidden_context else "  无禁止骨架")

    print(f"\n  多样性分数: {fsa.get_diversity_score():.3f}")
