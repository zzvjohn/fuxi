# -*- coding: utf-8 -*-
"""
Factor Review Agent — Sub-agent 生成与审查分离
====================================================

对标中金 CICC Loop Engineering 报告的 Sub-agent 分离设计:
  - 生成和审查使用不同的 LLM 调用（独立 Skill、独立上下文）
  - 防止单一模型「自我说服」
  - 对标 FactorMiner 的 Multi-Agent Debate（提案者 vs 批判者）

核心功能:
  1. FactorReviewAgent: 独立审查者 — 只看因子表达式 + 经济逻辑 + 回测结果
  2. 生成 Agent 不参与审查，审查 Agent 不接触生成过程
  3. 可配置的审查标准（边界条件、经济合理性、过拟合风险）
  4. 产出结构化的审查报告 + 修改建议

设计原则:
  - 审查者与生成者完全隔离（不同 Skill context）
  - 审查者可独立运行（不依赖外部 LLM 时用规则审查）
  - 规则模式（无 LLM）: 检查表达式合法性、复杂度、越界操作
  - LLM 模式（有 LLM）: 评估经济逻辑合理性、机制新颖性

集成路径:
  - Ralph Loop G 阶段: 生成后调用 review_batch 过滤低质量候选
  - MetaController: 在提案阶段用审查结果调整建议
  - 独立使用: 对任意因子池做质量审计

用法:
    from factor_review_agent import FactorReviewAgent
    
    reviewer = FactorReviewAgent(mode="rule")  # 或 mode="llm"
    
    # 审查单个因子
    report = reviewer.review_single({
        "factor_name": "test",
        "formula": "sub(ma(overnight, 60), ma(close, 20))",
        "hypothesis": "跳空溢价与日内趋势背离",
    })
    
    # 批量审查
    results = reviewer.review_batch(candidates)
"""

import re
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field


# ── 审查规则 ──────────────────────────────────────────

@dataclass
class ReviewRule:
    """单条审查规则"""
    name: str
    category: str          # "syntax" | "semantic" | "economic" | "risk"
    check_fn: Callable
    severity: str = "ERROR"  # ERROR / WARNING / INFO
    description: str = ""


@dataclass
class ReviewReport:
    """审查报告"""
    factor_name: str
    passed: bool
    score: float            # 0.0 - 1.0
    issues: List[Dict] = field(default_factory=list)
    warnings: List[Dict] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    dimensions: Dict = field(default_factory=dict)
    reviewed_at: str = field(default_factory=lambda: datetime.now().isoformat())


class FactorReviewAgent:
    """
    因子审查 Agent — 独立于生成过程的审查者。

    两种模式:
      - "rule": 纯规则审查（快速、确定性、无 LLM 成本）
      - "llm": 规则 + LLM 精判（慢、需要 LLM 调用、但有经济逻辑评估）

    对标中金: 审查阶段对候选表达式做规则过滤 —
      外层截面算子自动退化、同质算子简化、跨量纲运算拒绝、最小复杂度门槛。
      随机抽样 5 个候选交由 LLM 精判。
    """

    # 禁止的算子组合（无实际意义的组合）
    FORBIDDEN_COMBOS = [
        ("rank", "rank"),      # 嵌套 rank 无意义
        ("zscore", "zscore"),  # 嵌套 zscore 无意义
        ("abs", "abs"),        # 嵌套 abs 冗余
        ("sign", "sign"),      # 嵌套 sign 无意义
    ]

    # 跨量纲禁止组合（如 sub(volume, close) 无意义）
    CROSS_UNIT_PAIRS = {
        ("volume", "close"),
        ("volume", "overnight"),
        ("amount", "close"),
        ("turnover", "close"),
        ("market_cap", "volume"),
    }

    # 最小复杂度阈值
    MIN_NODES = 2
    MAX_NODES = 15
    MIN_DEPTH = 1
    MAX_DEPTH = 6

    def __init__(
        self,
        mode: str = "rule",
        llm_fn: Optional[Callable] = None,
        strict: bool = True,
    ):
        """
        Parameters
        ----------
        mode: "rule" | "llm"
        llm_fn: LLM 调用函数 (mode="llm" 时需要)
        strict: 严格模式 — ERROR 级别问题直接拒绝
        """
        self.mode = mode
        self.llm_fn = llm_fn
        self.strict = strict
        self.rules = self._build_rules()
        self.stats = {"total_reviewed": 0, "total_passed": 0, "total_rejected": 0}

    def _build_rules(self) -> List[ReviewRule]:
        """构建审查规则集"""
        return [
            # ── 语法规则 ──
            ReviewRule(
                name="non_empty_expression",
                category="syntax",
                check_fn=self._check_non_empty,
                severity="ERROR",
                description="表达式不能为空",
            ),
            ReviewRule(
                name="balanced_parentheses",
                category="syntax",
                check_fn=self._check_parentheses,
                severity="ERROR",
                description="括号必须平衡",
            ),
            ReviewRule(
                name="valid_operators",
                category="syntax",
                check_fn=self._check_valid_operators,
                severity="ERROR",
                description="算子必须在允许列表中",
            ),

            # ── 语义规则 ──
            ReviewRule(
                name="no_nested_redundant",
                category="semantic",
                check_fn=self._check_no_nested_redundant,
                severity="WARNING",
                description="禁止无意义的嵌套 rank/abs/sign",
            ),
            ReviewRule(
                name="no_cross_unit_operations",
                category="semantic",
                check_fn=self._check_cross_unit,
                severity="ERROR",
                description="禁止跨量纲运算",
            ),
            ReviewRule(
                name="complexity_bounds",
                category="semantic",
                check_fn=self._check_complexity,
                severity="WARNING",
                description=f"节点数应在 [{self.MIN_NODES}, {self.MAX_NODES}] 范围内",
            ),

            # ── 经济规则 ──
            ReviewRule(
                name="has_field_reference",
                category="economic",
                check_fn=self._check_has_field,
                severity="ERROR",
                description="表达式必须引用至少一个价量字段",
            ),
            ReviewRule(
                name="has_temporal_structure",
                category="economic",
                check_fn=self._check_temporal_structure,
                severity="WARNING",
                description="表达式应有时间序列结构（ma/delta/std 等）",
            ),
            ReviewRule(
                name="not_pure_constant",
                category="economic",
                check_fn=self._check_not_pure_constant,
                severity="ERROR",
                description="表达式不能是纯常数",
            ),

            # ── 风险规则 ──
            ReviewRule(
                name="no_self_reference",
                category="risk",
                check_fn=self._check_no_self_ref,
                severity="ERROR",
                description="不能引用自身（未来函数）",
            ),
        ]

    # ═══════════════════════════════════════════════════════════
    # 审查入口
    # ═══════════════════════════════════════════════════════════

    def review_single(self, factor: Dict) -> ReviewReport:
        """
        审查单个因子。

        Parameters
        ----------
        factor: {
            "factor_name": str,
            "formula": str (或 "expression"),
            "hypothesis": str (可选),
            "paradigm": str (可选),
        }

        Returns
        -------
        ReviewReport
        """
        name = factor.get("factor_name", factor.get("name", "unknown"))
        formula = factor.get("formula", factor.get("expression", ""))
        hypothesis = factor.get("hypothesis", "")

        report = ReviewReport(factor_name=name, passed=True, score=1.0)

        # 应用规则
        total_weight = 0.0
        weighted_score = 0.0

        for rule in self.rules:
            try:
                passed, detail = rule.check_fn(factor)
            except Exception as e:
                passed = False
                detail = f"规则执行异常: {e}"

            issue = {
                "rule": rule.name,
                "category": rule.category,
                "severity": rule.severity,
                "passed": passed,
                "detail": detail,
            }

            if not passed:
                if rule.severity == "ERROR":
                    report.issues.append(issue)
                    # 严格模式下 ERROR 直接失败
                    if self.strict:
                        report.passed = False
                elif rule.severity == "WARNING":
                    report.warnings.append(issue)
                else:
                    report.warnings.append(issue)

            # 评分
            weight = {"ERROR": 3, "WARNING": 1, "INFO": 0}.get(rule.severity, 1)
            total_weight += weight
            if passed:
                weighted_score += weight

        # 计算分数
        if total_weight > 0:
            report.score = weighted_score / total_weight

        # 规则未完全通过
        if report.issues:
            report.passed = False

        # 规则全部通过但可能有 warnings
        # 如果通过了所有 ERROR 规则，可以视为通过
        if report.issues and not any(
            i["severity"] == "ERROR" for i in report.issues
        ):
            if not self.strict:
                report.passed = True  # 宽容模式

        # LLM 精判 (mode="llm" 时)
        if self.mode == "llm" and self.llm_fn and report.passed:
            try:
                llm_review = self._llm_review(factor, report)
                report.dimensions["llm_review"] = llm_review
                # LLM 的评估作为参考，不影响规则审查的结论
                if llm_review.get("reject", False):
                    report.warnings.append({
                        "rule": "LLM_RISK_FLAG",
                        "category": "economic",
                        "severity": "WARNING",
                        "passed": False,
                        "detail": llm_review.get("reason", "LLM 标记为高风险"),
                    })
            except Exception as e:
                report.warnings.append({
                    "rule": "LLM_REVIEW_FAILED",
                    "category": "syntax",
                    "severity": "INFO",
                    "passed": True,
                    "detail": f"LLM 审查调用失败: {e}",
                })

        # 生成建议
        report.suggestions = self._generate_suggestions(factor, report)

        # 更新统计
        self.stats["total_reviewed"] += 1
        if report.passed:
            self.stats["total_passed"] += 1
        else:
            self.stats["total_rejected"] += 1

        return report

    def review_batch(
        self, candidates: List[Dict], sample_llm_n: int = 5
    ) -> List[ReviewReport]:
        """
        批量审查。

        Parameters
        ----------
        candidates: 候选因子列表
        sample_llm_n: LLM 精判的抽样数量（默认 5，对标中金）

        Returns
        -------
        审查报告列表
        """
        reports = []

        # 规则审查（全部）
        for cand in candidates:
            report = self.review_single(cand)
            reports.append(report)

        # LLM 精判（抽样）
        if self.mode == "llm" and self.llm_fn and sample_llm_n > 0:
            # 选审查通过但分数较低的候选做 LLM 精判
            passed_reports = [
                (i, r) for i, r in enumerate(reports) if r.passed
            ]
            passed_reports.sort(key=lambda x: x[1].score)

            for i, r in passed_reports[:sample_llm_n]:
                try:
                    llm_review = self._llm_review(candidates[i], r)
                    r.dimensions["llm_review"] = llm_review
                except Exception:
                    pass

        return reports

    def get_review_context_for_generation(self) -> str:
        """
        获取审查规则摘要，注入生成 Agent 作为约束。
        
        对标中金: 审查阶段拦截无意义或越界表达式。
        """
        lines = ["## 因子审查约束（生成时必须遵守）："]
        for rule in self.rules:
            if rule.severity == "ERROR":
                lines.append(f"- [{rule.category}] {rule.description}")
        return "\n".join(lines)

    # ═══════════════════════════════════════════════════════════
    # 审查规则实现
    # ═══════════════════════════════════════════════════════════

    @staticmethod
    def _check_non_empty(factor: Dict) -> Tuple[bool, str]:
        expr = factor.get("formula", factor.get("expression", ""))
        if not expr or not expr.strip():
            return False, "表达式为空"
        return True, "OK"

    @staticmethod
    def _check_parentheses(factor: Dict) -> Tuple[bool, str]:
        expr = factor.get("formula", factor.get("expression", ""))
        depth = 0
        for ch in expr:
            if ch == '(':
                depth += 1
            elif ch == ')':
                depth -= 1
            if depth < 0:
                return False, f"括号不匹配: 在位置遇到意外的 ')'"
        if depth != 0:
            return False, f"括号不匹配: 未闭合的 '(' (深度={depth})"
        return True, "OK"

    @staticmethod
    def _check_valid_operators(factor: Dict) -> Tuple[bool, str]:
        expr = factor.get("formula", factor.get("expression", ""))
        # 提取所有函数调用名（包括 pandas 方法链: .rolling(...), .shift(...)）
        funcs = re.findall(r'(?:\.)?([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', expr)
        KNOWN_OPS = {
            # DSL operators
            "ma", "ts_mean", "ts_std", "ts_min", "ts_max", "ts_sum",
            "ts_delta", "ts_rank", "ts_skewness", "ts_kurtosis",
            "ts_corr", "ts_cov", "ts_decay_linear", "ts_delay",
            "ema", "wma", "delta", "roc",
            "rank", "rank_cs", "scale", "zscore", "cs_zscore",
            "add", "sub", "mul", "div", "pow", "sqrt", "abs", "log", "sign",
            "min", "max", "if_else", "clip",
            # Pandas method calls (real factors use these)
            "rolling", "shift", "mean", "std", "pct_change", "astype",
            "sum", "max", "min", "round", "abs", "rank", "fillna",
            "replace", "clip", "where", "apply", "transform",
        }
        ALL_KNOWN_FIELDS = {
            "open", "open_p", "high", "high_p", "low", "low_p",
            "close", "close_p", "volume", "volume_p", "amount", "amount_p",
            "vwap", "overnight", "intraday", "returns", "amplitude", "turnover",
            "market_cap", "mv", "float_mv", "up_shadow", "down_shadow",
            "hl_ratio", "atr", "swing", "gap",
        }
        unknown = [f for f in funcs if f not in KNOWN_OPS and f not in ALL_KNOWN_FIELDS]
        if unknown:
            return False, f"未知算子/字段: {unknown}"
        return True, "OK"

    def _check_no_nested_redundant(self, factor: Dict) -> Tuple[bool, str]:
        expr = factor.get("formula", factor.get("expression", ""))
        for op1, op2 in self.FORBIDDEN_COMBOS:
            if f"{op1}({op2}(" in expr.replace(" ", ""):
                return False, f"无意义的嵌套: {op1}({op2}(...))"
        return True, "OK"

    def _check_cross_unit(self, factor: Dict) -> Tuple[bool, str]:
        expr = factor.get("formula", factor.get("expression", ""))
        expr_clean = expr.replace(" ", "").lower()

        # 简化的跨量纲检查：如果 sub/mul/div 的参数包含不同量纲的字段
        for op in ["sub(", "mul(", "div(", "add("]:
            idx = expr_clean.find(op)
            if idx >= 0:
                # 检查该操作的参数中是否包含跨量纲对
                inner = expr_clean[idx + len(op):]
                depth = 0
                args = ""
                for ch in inner:
                    if ch == '(':
                        depth += 1
                    elif ch == ')':
                        depth -= 1
                        if depth < 0:
                            break
                    elif ch == ',' and depth == 0:
                        args += "|"
                        continue
                    if depth >= 0:
                        args += ch

                parts = args.split("|")
                for (f1, f2) in self.CROSS_UNIT_PAIRS:
                    found_f1 = any(f1 in p for p in parts)
                    found_f2 = any(f2 in p for p in parts)
                    if found_f1 and found_f2:
                        return False, f"跨量纲运算: {f1} 与 {f2} 不可直接 {op.strip('(')}"
        return True, "OK"

    def _check_complexity(self, factor: Dict) -> Tuple[bool, str]:
        expr = factor.get("formula", factor.get("expression", ""))

        # 数节点（算子调用数 + 字段引用数）
        func_calls = len(re.findall(r'(?:\.)?[a-zA-Z_][a-zA-Z0-9_]*\s*\(', expr))
        field_refs = len(re.findall(
            r'\b(open_p|high_p|low_p|close_p|volume_p|amount_p|'
            r'open|high|low|close|volume|amount|overnight|intraday|'
            r'returns|amplitude|turnover|market_cap|mv|up_shadow|down_shadow|'
            r'hl_ratio|atr|swing|gap|buy_vol|sell_vol)\b',
            expr, re.IGNORECASE
        ))

        nodes = func_calls + field_refs

        if nodes < self.MIN_NODES:
            return False, f"过于简单: 节点数={nodes} < {self.MIN_NODES}（可能是裸字段引用）"
        if nodes > self.MAX_NODES:
            return False, f"过于复杂: 节点数={nodes} > {self.MAX_NODES}（过拟合风险高）"
        return True, f"节点数={nodes} OK"

    @staticmethod
    def _check_has_field(factor: Dict) -> Tuple[bool, str]:
        expr = factor.get("formula", factor.get("expression", ""))
        fields = re.findall(
            r'\b(open_p|high_p|low_p|close_p|volume_p|amount_p|'
            r'open|high|low|close|volume|amount|vwap|overnight|intraday|'
            r'returns|amplitude|turnover|market_cap|mv|up_shadow|down_shadow|'
            r'hl_ratio|atr|swing|gap|buy_vol|sell_vol)\b',
            expr, re.IGNORECASE
        )
        if not fields:
            return False, "表达式未引用任何价量字段"
        return True, f"引用字段: {list(set(fields))}"

    @staticmethod
    def _check_temporal_structure(factor: Dict) -> Tuple[bool, str]:
        expr = factor.get("formula", factor.get("expression", ""))
        temporal_ops = [
            "ma(", "ts_mean(", "ts_std(", "ts_delta(", "ts_rank(",
            "delta(", "ema(", "wma(", "roc(", "ts_delay(",
            # Pandas equivalents
            ".rolling(", ".shift(", ".pct_change(",
        ]
        has_temporal = any(op in expr.lower().replace(" ", "") for op in temporal_ops)
        if not has_temporal:
            return False, "无时间序列结构（建议至少使用 rolling/shift/pct_change 之一）"
        return True, "OK"

    @staticmethod
    def _check_not_pure_constant(factor: Dict) -> Tuple[bool, str]:
        expr = factor.get("formula", factor.get("expression", ""))
        # 去掉空白后检查是否所有内容都是数字和运算符
        cleaned = re.sub(r'[0-9+\-*/().,\s]', '', expr)
        if not cleaned:  # 只剩数字和运算符 = 纯常数
            return False, "表达式是纯常数（无实际预测力）"
        return True, "OK"

    @staticmethod
    def _check_no_self_ref(factor: Dict) -> Tuple[bool, str]:
        expr = factor.get("formula", factor.get("expression", ""))
        # 检查是否有类似 "factor_value" 或 "returns" 这类未来数据的引用
        dangerous = ["future", "next", "forward", "shift(-", "lead"]
        for d in dangerous:
            if d in expr.lower():
                return False, f"可能包含未来信息: '{d}'"
        return True, "OK"

    # ═══════════════════════════════════════════════════════════
    # LLM 精判
    # ═══════════════════════════════════════════════════════════

    def _llm_review(
        self, factor: Dict, rule_report: ReviewReport
    ) -> Dict:
        """
        LLM 精判 — 评估经济逻辑和机制新颖性。

        对标中金: 随机抽样 5 个候选交由 LLM 精判，
        检查表达式边界条件是否合理。
        """
        if not self.llm_fn:
            return {"reject": False, "reason": "LLM 未配置"}

        prompt = self._build_llm_review_prompt(factor, rule_report)
        try:
            response = self.llm_fn(prompt)
            return response
        except Exception as e:
            return {"reject": False, "reason": f"LLM 调用异常: {e}"}

    def _build_llm_review_prompt(
        self, factor: Dict, rule_report: ReviewReport
    ) -> str:
        """构建 LLM 审查 prompt"""
        name = factor.get("factor_name", "unknown")
        formula = factor.get("formula", factor.get("expression", ""))
        hypothesis = factor.get("hypothesis", "")

        return f"""作为独立的因子审查专家，请评估以下候选因子：

**因子名**: {name}
**表达式**: `{formula}`
**经济假设**: {hypothesis}

**规则审查结果**:
- 通过: {rule_report.passed}
- 分数: {rule_report.score:.2f}
- 警告: {[w['detail'] for w in rule_report.warnings[:3]]}

请从以下维度评估（返回 JSON）：
1. 经济逻辑合理性: 该因子是否有清晰的经济学解释？
2. 机制新颖性: 该因子与常见因子（如动量、反转、波动率）相比是否有新意？
3. 过拟合风险: 表达式复杂度是否过高？（深度>4 为高风险）
4. 边界条件: 在极端市场情况下，因子值是否会爆炸？

返回格式:
{{
  "economic_sound": true/false,
  "novelty": "high/medium/low",
  "overfit_risk": "high/medium/low",
  "boundary_issues": ["... issues ..."],
  "reject": true/false,
  "reason": "如果拒绝，简要说明原因"
}}"""

    # ═══════════════════════════════════════════════════════════
    # 建议生成
    # ═══════════════════════════════════════════════════════════

    def _generate_suggestions(
        self, factor: Dict, report: ReviewReport
    ) -> List[str]:
        """根据审查结果生成改进建议"""
        suggestions = []

        for issue in report.issues:
            if "跨量纲" in issue.get("detail", ""):
                suggestions.append("建议: 使用同量纲字段 (如 volume 与 amount)")
            elif "嵌套" in issue.get("detail", ""):
                suggestions.append("建议: 移除无意义的嵌套 rank/abs/sign")
            elif "简单" in issue.get("detail", ""):
                suggestions.append("建议: 增加时间序列操作 (ma/delta/std)")
            elif "复杂" in issue.get("detail", ""):
                suggestions.append("建议: 简化表达式，减少嵌套层数")
            elif "未知算子" in issue.get("detail", ""):
                suggestions.append("建议: 使用标准算子库中的算子")
            elif "纯常数" in issue.get("detail", ""):
                suggestions.append("建议: 表达式必须引用至少一个价量字段")

        for warn in report.warnings:
            if "无时间序列" in warn.get("detail", ""):
                if "建议" not in str(suggestions):
                    suggestions.append("建议: 添加 ma/delta/std 等时间序列操作以便捕捉时序模式")

        return suggestions

    # ═══════════════════════════════════════════════════════════
    # 统计
    # ═══════════════════════════════════════════════════════════

    def get_stats(self) -> Dict:
        """获取审查统计"""
        return {
            **self.stats,
            "pass_rate": (
                self.stats["total_passed"] / max(self.stats["total_reviewed"], 1)
            ),
        }


# ── 便捷函数 ──────────────────────────────────────────────

_default_reviewer: Optional[FactorReviewAgent] = None


def get_reviewer(mode: str = "rule") -> FactorReviewAgent:
    global _default_reviewer
    if _default_reviewer is None:
        _default_reviewer = FactorReviewAgent(mode=mode)
    return _default_reviewer


def quick_review(factor: Dict) -> ReviewReport:
    """快速审查单个因子"""
    return get_reviewer().review_single(factor)


# ── 测试 ──────────────────────────────────────────────────

if __name__ == "__main__":
    reviewer = FactorReviewAgent(mode="rule")

    print("=" * 60)
    print("  Factor Review Agent 测试")
    print("=" * 60)

    test_factors = [
        {
            "factor_name": "good_overnight_divergence",
            "formula": "sub(ma(overnight, 60), ma(close, 20))",
            "hypothesis": "跳空溢价与日内收盘价的趋势背离",
        },
        {
            "factor_name": "bad_nested_rank",
            "formula": "rank(rank(close))",
            "hypothesis": "双重排名测试",
        },
        {
            "factor_name": "bad_cross_unit",
            "formula": "sub(volume, close)",
            "hypothesis": "量价相减",
        },
        {
            "factor_name": "bad_too_simple",
            "formula": "close",
            "hypothesis": "裸收盘价",
        },
        {
            "factor_name": "good_complex",
            "formula": "div(sub(ma(overnight, 60), ma(close, 20)), ts_std(amplitude, 30))",
            "hypothesis": "跳空溢价差异除以振幅波动，衡量信号信噪比",
        },
    ]

    for f in test_factors:
        report = reviewer.review_single(f)
        status = "✅" if report.passed else "❌"
        print(f"\n  {status} {f['factor_name']}")
        print(f"    分数: {report.score:.2f}")
        if report.issues:
            for i in report.issues:
                print(f"    ❌ [{i['severity']}] {i['detail']}")
        if report.warnings:
            for w in report.warnings:
                print(f"    ⚠️ [{w['severity']}] {w['detail']}")
        if report.suggestions:
            for s in report.suggestions:
                print(f"    💡 {s}")

    print(f"\n  统计: {reviewer.get_stats()}")
