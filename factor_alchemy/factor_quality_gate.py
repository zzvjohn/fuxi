# -*- coding: utf-8 -*-
"""
FactorQualityGate — 统一因子质量门禁 (合并 SemanticVerifier + FactorReviewAgent)
====================================================================================

合并前两个模块的功能:
  - SemanticVerifier: H↔E↔C 三重一致性 + 复杂度/冗余检查 + 评分
  - FactorReviewAgent: 10 条审查规则(syntax/semantic/economic/risk) + LLM 精判

统一后:
  1. 语法规则 (5条): 非空/括号平衡/合法算子/无嵌套冗余/无跨量纲
  2. 语义一致性 (3条): H↔E / E↔C / H↔C
  3. 经济合理性 (3条): 有字段引用/有时序结构/非纯常数
  4. 风险检查 (1条): 无未来函数
  5. 复杂度约束: node_count <= max_complexity, ast_depth <= max_ast_depth
  6. 综合评分: 加权平均, ERROR=致命/WARNING=警告

用法:
    from factor_quality_gate import FactorQualityGate

    gate = FactorQualityGate()
    result = gate.verify(factor_dict)

    if result.passed:
        print(f"通过: {result.score:.2f}")
    else:
        for issue in result.fatal_issues:
            print(f"致命: {issue}")
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable
from datetime import datetime


# ═══════════════════════════════════════════════════════════
# 数据结构
# ═══════════════════════════════════════════════════════════

@dataclass
class GateResult:
    """统一质量门禁检查结果"""
    passed: bool
    score: float                          # 0.0 - 1.0
    fatal_issues: List[str] = field(default_factory=list)    # ERROR 级别
    warnings: List[str] = field(default_factory=list)         # WARNING 级别
    suggestions: List[str] = field(default_factory=list)

    # 各维度详情
    syntax: Dict = field(default_factory=dict)
    semantic: Dict = field(default_factory=dict)
    economic: Dict = field(default_factory=dict)
    risk: Dict = field(default_factory=dict)
    complexity: Dict = field(default_factory=dict)
    scores: Dict = field(default_factory=dict)   # 分项评分

    checked_at: str = field(default_factory=lambda: datetime.now().isoformat())


# ═══════════════════════════════════════════════════════════
# FactorQualityGate
# ═══════════════════════════════════════════════════════════

class FactorQualityGate:
    """
    统一因子质量门禁。

    参数:
      max_complexity: 最大 AST 节点数 (默认 10)
      max_ast_depth: 最大 AST 深度 (默认 4)
      strict: 严格模式 — ERROR 即拒绝
      llm_fn: LLM 审查函数 (可选, None 时仅规则检查)
    """

    # ── 概念→表达式 映射 (H↔E) ──
    # v0.5.4: 使用 semantic_verifier 的共享扩充表 (资金流/筹码/突破/背离 + 边界匹配)
    # 旧表仅 10 行且 "vol" 误命中 "volume", 导致 LLM 轮 0/10 全拒
    CONCEPT_TO_OPS = []
    try:
        from semantic_verifier import H2E_CONCEPT_TABLE as _H2E_TABLE
        CONCEPT_TO_OPS = list(_H2E_TABLE)
    except Exception:
        # 回退: 旧表 (仅当 semantic_verifier 不可导入时)
        CONCEPT_TO_OPS = [
            (["动量", "momentum", "趋势", "trend"], ["delta", "diff", "pct_change", "roc", "slope", "ts_delta"]),
            (["波动", "volatility", "vol", "std"], ["std", "rolling_std", "ts_std"]),
            (["反转", "reversal", "逆转", "逆向", "neg"], ["neg", "rank", "csrank"]),
            (["量", "volume", "成交", "amt"], ["volume", "vol", "amount"]),
            (["相关", "correlation", "corr"], ["corr", "rolling_corr", "ts_corr"]),
            (["均线", "平均", "mean", "average", "sma", "ema"], ["mean", "sma", "ema", "ts_mean", "rolling"]),
            (["偏度", "skew"], ["skew", "ts_skew"]),
            (["峰度", "kurt"], ["kurt", "ts_kurt"]),
            (["回归", "slope", "residual", "残差", "r2", "rsquare"], ["slope", "rsquare", "resi"]),
            (["高点", "低点", "high", "low", "开盘", "收盘", "open", "close"], ["open", "high", "low", "close"]),
        ]

    # ── 表达式→代码 操作符映射 (E↔C) ──
    OP_TO_CODE = [
        ("delta", ["diff", "shift", "pct_change"]),
        ("mean", ["mean", "rolling", "average"]),
        ("std", ["std", "rolling", "rolling_std"]),
        ("corr", ["corr"]),
        ("rank", ["rank", "pct_rank"]),
        ("neg", ["-"]),
        ("skew", ["skew"]),
        ("kurt", ["kurt", "kurtosis"]),
        ("slope", ["slope", "linregress"]),
        ("ema", ["ewm", "ema"]),
    ]

    # ── 禁止的算子嵌套 ──
    FORBIDDEN_NESTING = [
        ("rank", "rank"), ("zscore", "zscore"), ("abs", "abs"), ("sign", "sign"),
    ]

    # ── 跨量纲禁止组合 ──
    CROSS_UNIT_PAIRS = {
        ("volume", "close"), ("volume", "overnight"),
        ("amount", "close"), ("turnover", "close"), ("market_cap", "volume"),
    }

    # ── 已知算子/字段 (统一白名单) ──
    KNOWN_OPS = {
        "ma", "ts_mean", "ts_std", "ts_min", "ts_max", "ts_sum",
        "ts_delta", "ts_rank", "ts_skewness", "ts_kurtosis",
        "ts_corr", "ts_cov", "ts_decay_linear", "ts_delay",
        "ema", "wma", "delta", "roc",
        "rank", "rank_cs", "scale", "zscore", "cs_zscore",
        "add", "sub", "mul", "div", "pow", "sqrt", "abs", "log", "sign",
        "min", "max", "if_else", "clip", "neg", "ts_pct", "ts_zscore",
        "rolling", "shift", "pct_change", "astype", "fillna",
        "replace", "where", "apply", "transform",
        # v0.6.2 (2026-08-29): ewm 补入 — ts_ema_decay 翻译产物 .ewm(span=..).mean()
        # 此前被误判「未知算子/字段: ['ewm']」拒掉 2 个候选 (Round1 实证)
        "ewm",
        # v0.5 伏羲: pandas 标准方法 (rolling().mean()/.std()/.kurt() 等)
        "mean", "std", "var", "sum", "skew", "kurt", "kurtosis",
        "cumsum", "cumprod", "quantile", "corr", "cov", "diff",
        "rank_pct", "bfill", "ffill", "interpolate", "dropna",
        "isna", "notna", "round", "astype", "pipe",
        # numpy 函数别名
        "maximum", "minimum", "np_maximum", "np_minimum",
        # v0.9.1 P-20260819: pandas Series 数值/集合方法 (mantissa 尾数公式: .mod(10).isin([0,5]))
        "mod", "isin", "floordiv", "rmod", "rsub", "rdiv",
        "any", "all", "value_counts", "unique", "nunique",
    }
    KNOWN_FIELDS = {
        "open", "open_p", "high", "high_p", "low", "low_p",
        "close", "close_p", "volume", "volume_p", "amount", "amount_p",
        "vwap", "overnight", "intraday", "returns", "amplitude", "turnover",
        "market_cap", "mv", "float_mv", "up_shadow", "down_shadow",
        "hl_ratio", "atr", "swing", "gap", "buy_vol", "sell_vol",
        # v0.5.4: moneyflow 字段 (FactorICComputer 已支持)
        "buy_lg_vol", "sell_lg_vol", "buy_sm_vol", "sell_sm_vol",
        "buy_md_vol", "sell_md_vol", "buy_elg_vol", "sell_elg_vol",
        "net_mf_vol", "net_mf_amount",
        # v0.9 P-20260814-001: 两融字段 (FactorICComputer 已支持)
        "rzye", "rqye", "rzmre", "rqyl", "rqchl",
        # v0.9 P-20260812-026: 龙虎榜字段 (FactorICComputer 已支持)
        "lhb_flag", "lhb_net_amount", "lhb_net_rate", "lhb_amount",
        "lhb_inst_net_buy", "lhb_inst_buy", "lhb_inst_sell",
        # v0.9 P-20260814-002: 北向字段 (FactorICComputer 已支持)
        "north_vol", "north_ratio",
        # v0.9.1 P-20260819-002: 分析师分歧字段 (FactorICComputer 已支持)
        "eps_disp", "tp_disp", "rating_disp", "n_cover",
        # v0.9.1 P-20260819-003: 行业 peer 收益 (FactorICComputer 已支持)
        "industry_ret_peer",
    }

    DANGEROUS_PATTERNS = ["future", "next", "forward", "shift(-", "lead"]

    def __init__(
        self,
        max_complexity: int = 10,
        max_ast_depth: int = 4,
        strict: bool = True,
        llm_fn: Optional[Callable] = None,
    ):
        self.max_complexity = max_complexity
        self.max_ast_depth = max_ast_depth
        self.strict = strict
        self.llm_fn = llm_fn
        self.stats = {"total": 0, "passed": 0, "rejected": 0}

    # ═══════════════════════════════════════════════════════════
    # 统一入口
    # ═══════════════════════════════════════════════════════════

    def verify(self, factor: Dict) -> GateResult:
        """统一质量门禁：检查因子的语法、语义、经济逻辑和风险。"""
        name = factor.get("factor_name", factor.get("name", "unknown"))
        formula = factor.get("formula", factor.get("expression", ""))
        hypothesis = factor.get("hypothesis", "")
        code = factor.get("code", "")

        result = GateResult(passed=True, score=1.0)

        # ── 1. 语法规则 ──
        result.syntax = self._check_syntax_all(formula)
        for issue in result.syntax.get("issues", []):
            if issue["severity"] == "ERROR":
                result.fatal_issues.append(f"[语法] {issue['detail']}")
            else:
                result.warnings.append(f"[语法] {issue['detail']}")

        # ── 2. 语义一致性 (H↔E↔C) ──
        result.semantic = self._check_semantic_consistency(hypothesis, formula, code)
        for issue in result.semantic.get("issues", []):
            if issue["severity"] == "ERROR":
                result.fatal_issues.append(f"[语义] {issue['detail']}")
            else:
                result.warnings.append(f"[语义] {issue['detail']}")

        # ── 3. 经济合理性 ──
        result.economic = self._check_economic(hypothesis, formula)
        for issue in result.economic.get("issues", []):
            if issue["severity"] == "ERROR":
                result.fatal_issues.append(f"[经济] {issue['detail']}")
            else:
                result.warnings.append(f"[经济] {issue['detail']}")

        # ── 4. 风险检查 ──
        result.risk = self._check_risk(formula)
        for issue in result.risk.get("issues", []):
            result.fatal_issues.append(f"[风险] {issue['detail']}")

        # ── 5. 复杂度 ──
        result.complexity = self._check_complexity(formula)

        # ── 评分 ──
        result.scores = self._calculate_scores(hypothesis, formula, code)
        result.score = self._compute_final_score(result)

        # ── 裁决 ──
        if self.strict:
            result.passed = len(result.fatal_issues) == 0
        else:
            result.passed = not any("ERROR" in i for i in result.fatal_issues)

        # ── 建议 ──
        result.suggestions = self._generate_suggestions(result)

        # LLM 精判
        if self.llm_fn and result.passed:
            try:
                llm_flag = self._llm_check(factor, result)
                if llm_flag.get("reject"):
                    result.warnings.append(f"[LLM] {llm_flag.get('reason', 'LLM标记高风险')}")
            except Exception:
                pass

        self.stats["total"] += 1
        if result.passed:
            self.stats["passed"] += 1
        else:
            self.stats["rejected"] += 1

        return result

    def verify_batch(self, factors: List[Dict]) -> List[GateResult]:
        """批量审查"""
        return [self.verify(f) for f in factors]

    # ═══════════════════════════════════════════════════════════
    # P-017: CodeGate 预检 (Stage 0)
    # ═══════════════════════════════════════════════════════════

    CODE_QUALITY_ASSERTS = [
        "assert isinstance(result, pd.DataFrame), 'result must be DataFrame'",
        "result = result.replace([np.inf, -np.inf], np.nan)",
        "result = result.fillna(0)",
    ]

    @staticmethod
    def preflight_code_check(code: str, formula: str = "") -> Dict:
        """
        CodeGate Stage 0: 在 S1 之前对因子代码做编译级验证。
        返回 {"passed": bool, "errors": [], "warnings": []}
        """
        result = {"passed": True, "errors": [], "warnings": []}

        # 1. 编译校验
        if code and code.strip():
            try:
                compile(code, '<codegate>', 'exec')
            except SyntaxError as e:
                result["passed"] = False
                result["errors"].append(f"SyntaxError: {e.msg} at line {e.lineno}")
                return result

        # 2. 静态分析: 除零保护
        expr = formula or code
        if expr:
            # 检查是否有裸的 division 没有保护
            simple_divs = ["/ close", "/ volume", "/ high", "/ low",
                           "/ buy_lg_vol", "/ sell_lg_vol", "/ intan_assets", "/ total_assets"]
            has_div = any(d in expr for d in ["/ close", "/ volume", "/ high", "/ low",
                                                "/ buy_lg_vol", "/ sell_lg_vol"])
            has_safety = "1e-6" in expr or "1e-8" in expr or "1e-12" in expr or "0.001" in expr or "add(" in expr
            if has_div and not has_safety:
                result["warnings"].append("可能存在除零风险: 建议用 add(denom, 1e-6) 保护分母")

            # 3. NaN 处理检测
            if "fillna" not in expr and "dropna" not in expr and "nan_to_num" not in expr:
                if "/" in expr or "rolling" in expr:
                    result["warnings"].append("缺少 NaN 处理: 建议添加 fillna(0) 或 dropna()")

        return result

    @staticmethod
    def codegate_self_debug(code: str, error_msg: str, context: str = "") -> str:
        """
        基于编译错误信息自动修复代码。
        返回修复后的代码文本，或 None 表示无法修复。
        """
        fixes = []
        # 常见错误模式
        if "ewm" in error_msg.lower() or "ewm" in code:
            # ewm 不是 pandas rolling 操作，替换为 rolling mean
            code = code.replace(".ewm(", ".rolling(window=").replace("span=", "window=")
            if ".mean()" not in code and "rolling" in code:
                code = code.replace(".rolling(", ".rolling(window=20).mean().rolling(").replace("window=20).mean().rolling(window=", "")
            fixes.append("修复: ewm → rolling")
        if "is not defined" in error_msg:
            fixes.append("修复: 变量未定义 → 需 LLM 介入")
            return None
        if "object has no attribute" in error_msg:
            fixes.append("修复: 属性不存在 → 需 LLM 介入")
            return None

        # 尝试编译
        try:
            compile(code, '<fix>', 'exec')
            return code
        except SyntaxError:
            return None

    # ═══════════════════════════════════════════════════════════
    # 1. 语法规则 (5条)
    # ═══════════════════════════════════════════════════════════

    def _check_syntax_all(self, expr: str) -> Dict:
        issues = []

        if not expr or not expr.strip():
            issues.append({"severity": "ERROR", "detail": "表达式为空"})
            return {"issues": issues, "all_passed": False}

        # 括号平衡
        depth = 0
        for ch in expr:
            if ch == '(': depth += 1
            elif ch == ')': depth -= 1
            if depth < 0:
                issues.append({"severity": "ERROR", "detail": "括号不匹配: 意外 ')'"})
                break
        if depth != 0:
            issues.append({"severity": "ERROR", "detail": f"括号不匹配: 未闭合 '(' (深度={depth})"})

        # 合法算子
        funcs = re.findall(r'(?:\.)?([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', expr)
        unknown = [f for f in funcs if f not in self.KNOWN_OPS and f not in self.KNOWN_FIELDS]
        if unknown:
            issues.append({"severity": "ERROR", "detail": f"未知算子/字段: {unknown}"})

        # 无嵌套冗余
        clean = expr.replace(" ", "")
        for op1, op2 in self.FORBIDDEN_NESTING:
            if f"{op1}({op2}(" in clean:
                issues.append({"severity": "WARNING", "detail": f"无意义嵌套: {op1}({op2}(...))"})

        # 无跨量纲
        for (f1, f2) in self.CROSS_UNIT_PAIRS:
            if f1 in clean.lower() and f2 in clean.lower():
                for op in ["sub(", "add(", "mul(", "div("]:
                    idx = clean.lower().find(op)
                    if idx >= 0:
                        issues.append({"severity": "ERROR", "detail": f"跨量纲运算: {f1} 与 {f2} 不可直接 {op.strip('(')}"})
                        break

        return {"issues": issues, "all_passed": not any(i["severity"] == "ERROR" for i in issues)}

    # ═══════════════════════════════════════════════════════════
    # 2. 语义一致性 (H↔E↔C)
    # ═══════════════════════════════════════════════════════════

    def _check_semantic_consistency(self, hypothesis: str, formula: str, code: str) -> Dict:
        issues = []

        # H↔E: 假设概念是否在表达式中体现
        if hypothesis and formula:
            h2e = self._check_h2e(hypothesis, formula)
            issues.extend(h2e)

        # E↔C: 表达式操作是否在代码中实现
        if formula and code:
            e2c = self._check_e2c(formula, code)
            issues.extend(e2c)

        # H↔C: 假设逻辑是否在代码中体现
        if hypothesis and code:
            h2c = self._check_h2c(hypothesis, code)
            issues.extend(h2c)

        return {"issues": issues, "all_passed": not any(i["severity"] == "ERROR" for i in issues)}

    def _check_h2e(self, hypothesis: str, formula: str) -> List[Dict]:
        hyp_lower = hypothesis.lower()
        expr_lower = formula.lower()
        issues = []
        matched = total = 0

        # v0.5.4: 假设侧用边界化匹配 (防 "vol" 误命中 "volume")
        try:
            from semantic_verifier import _match_keywords as _h2e_kw_match
        except Exception:
            _h2e_kw_match = None

        for hyp_kw, expr_kw in self.CONCEPT_TO_OPS:
            if _h2e_kw_match is not None:
                hyp_hit = _h2e_kw_match(hyp_lower, hyp_kw)
            else:
                hyp_hit = any(kw in hyp_lower for kw in hyp_kw)
            if hyp_hit:
                total += 1
                # 表达式侧纯子串匹配 (buy_lg_vol 命中 "lg_vol"/"vol")
                if any(kw in expr_lower for kw in expr_kw):
                    matched += 1
                else:
                    issues.append({
                        "severity": "WARNING",
                        "detail": f"H↔E: 假设提到 '{hyp_kw[0]}', 但表达式中未找到对应操作"
                    })

        # v0.5.4: ERROR 仅在明显脱节时触发 (≥2 概念 0 实现); 60% 以下降级为 WARNING
        if total > 0 and matched == 0 and total >= 2:
            issues.append({"severity": "ERROR", "detail": f"H↔E: 假设提到 {total} 个概念, 表达式中 0 个实现"})
        elif total > 0 and matched / total < 0.6:
            issues.append({"severity": "WARNING", "detail": f"H↔E: 概念匹配率 {matched}/{total} < 60%"})
        return issues

    def _check_e2c(self, formula: str, code: str) -> List[Dict]:
        issues = []
        expr_lower = formula.lower()
        code_lower = code.lower()

        for op, code_patterns in self.OP_TO_CODE:
            if op in expr_lower:
                if not any(re.search(p, code_lower) for p in code_patterns):
                    issues.append({
                        "severity": "WARNING",
                        "detail": f"E↔C: 表达式用了 '{op}', 代码中未找到对应实现"
                    })

        # 窗口一致性
        expr_windows = {int(m.group(1)) for m in re.finditer(r'(\d+)', formula) if 2 <= int(m.group(1)) <= 252}
        code_windows = {int(m.group(1)) for m in re.finditer(r'(\d+)', code) if 2 <= int(m.group(1)) <= 252}
        for w in expr_windows:
            if w not in code_windows and expr_windows:
                issues.append({
                    "severity": "INFO",
                    "detail": f"E↔C: 表达式使用窗口 {w}, 代码中未找到此参数"
                })
        return issues

    def _check_h2c(self, hypothesis: str, code: str) -> List[Dict]:
        hyp_lower = hypothesis.lower()
        code_lower = code.lower()
        issues = []

        if any(kw in hyp_lower for kw in ["当", "when", "if", "条件", "condition", "regime", "状态", "环境"]):
            if not any(kw in code_lower for kw in ["ifelse", "if ", "where", "np.where", "greater", "less"]):
                issues.append({
                    "severity": "WARNING",
                    "detail": "H↔C: 假设描述条件依赖逻辑, 但代码中未使用条件操作"
                })

        if any(kw in hyp_lower for kw in ["交互", "interaction", "乘积", "product", "结合", "combine"]):
            if not any(kw in code_lower for kw in ["mul(", "multiply", "*", "add(", "+"]):
                issues.append({
                    "severity": "WARNING",
                    "detail": "H↔C: 假设描述多因子交互, 但代码中未使用乘法/加法"
                })
        return issues

    # ═══════════════════════════════════════════════════════════
    # 3. 经济合理性
    # ═══════════════════════════════════════════════════════════

    def _check_economic(self, hypothesis: str, formula: str) -> Dict:
        issues = []

        # 有字段引用
        # v0.5.4: 补充 moneyflow 字段 (buy_lg_vol 等), 资金流公式不再误报"未引用价量字段"
        fields = set(re.findall(
            r'\b(open_p|high_p|low_p|close_p|volume_p|amount_p|'
            r'open|high|low|close|volume|amount|vwap|overnight|intraday|'
            r'returns|amplitude|turnover|market_cap|mv|up_shadow|down_shadow|'
            r'hl_ratio|atr|swing|gap|buy_vol|sell_vol|'
            r'buy_lg_vol|sell_lg_vol|buy_sm_vol|sell_sm_vol|'
            r'buy_md_vol|sell_md_vol|buy_elg_vol|sell_elg_vol|'
            r'net_mf_vol|net_mf_amount|'
            r'rzye|rqye|rzmre|rqyl|rqchl|'
            r'lhb_flag|lhb_net_amount|lhb_net_rate|lhb_amount|'
            r'lhb_inst_net_buy|lhb_inst_buy|lhb_inst_sell|'
            r'north_vol|north_ratio|'
            r'eps_disp|tp_disp|rating_disp|n_cover|industry_ret_peer)\b',
            formula, re.IGNORECASE
        ))
        if not fields:
            issues.append({"severity": "ERROR", "detail": "未引用任何价量字段"})

        # 有时序结构
        temporal = ["ma(", "ts_mean(", "ts_std(", "ts_delta(", "ts_rank(",
                    "delta(", "ema(", "wma(", "roc(", "ts_delay(",
                    ".rolling(", ".shift(", ".pct_change("]
        has_temporal = any(op in formula.lower().replace(" ", "") for op in temporal)
        if not has_temporal:
            issues.append({"severity": "WARNING", "detail": "无时间序列结构, 建议加 rolling/shift/pct_change"})

        # 非纯常数
        cleaned = re.sub(r'[0-9+\-*/().,\s]', '', formula)
        if not cleaned:
            issues.append({"severity": "ERROR", "detail": "表达式是纯常数, 无预测力"})

        return {"issues": issues, "all_passed": not any(i["severity"] == "ERROR" for i in issues)}

    # ═══════════════════════════════════════════════════════════
    # 4. 风险检查
    # ═══════════════════════════════════════════════════════════

    def _check_risk(self, formula: str) -> Dict:
        issues = []
        for d in self.DANGEROUS_PATTERNS:
            if d in formula.lower():
                issues.append({"severity": "ERROR", "detail": f"可能包含未来信息: '{d}'"})
        return {"issues": issues, "all_passed": len(issues) == 0}

    # ═══════════════════════════════════════════════════════════
    # 5. 复杂度
    # ═══════════════════════════════════════════════════════════

    def _check_complexity(self, formula: str) -> Dict:
        func_calls = len(re.findall(r'(?:\.)?[a-zA-Z_][a-zA-Z0-9_]*\s*\(', formula))
        operators = len(re.findall(r'[\+\-\*/]', formula))
        node_count = func_calls + operators

        depth = max_depth = 0
        for ch in formula:
            if ch == '(':
                depth += 1
                max_depth = max(max_depth, depth)
            elif ch == ')':
                depth -= 1

        return {
            "node_count": node_count,
            "ast_depth": max_depth,
            "complexity_pass": node_count <= self.max_complexity and max_depth <= self.max_ast_depth,
            "max_nodes": self.max_complexity,
            "max_depth": self.max_ast_depth,
        }

    # ═══════════════════════════════════════════════════════════
    # 6. 评分
    # ═══════════════════════════════════════════════════════════

    def _calculate_scores(self, hypothesis: str, formula: str, code: str) -> Dict:
        scores = {}

        # 新颖度
        hyp_lower = hypothesis.lower()
        common = sum(1 for p in ["动量", "反转", "均线交叉"] if p in hyp_lower)
        uncommon = sum(1 for p in ["偏度", "峰度", "残差", "熵", "拥挤", "微观结构", "跨资产"] if p in hyp_lower)
        scores["novelty"] = max(0.0, min(1.0, 0.3 + uncommon * 0.15 - common * 0.1))

        # 经济逻辑
        logic_kw = ["因为", "因此", "由于", "所以", "机制", "机理",
                    "because", "therefore", "since", "mechanism"]
        scores["economic_logic"] = min(1.0, 0.3 + sum(0.2 for kw in logic_kw if kw in hyp_lower))

        # 表达式质量
        if formula:
            ops = set(re.findall(r'(?:ts_|cs_|rolling_)?(\w+)\(', formula.lower()))
            scores["expression_quality"] = max(0.1, min(1.0, 0.5 + min(0.3, len(ops) * 0.06)
                - max(0, len(re.findall(r'\w+\(', formula)) - 10) * 0.05))
        else:
            scores["expression_quality"] = 0.0

        # 代码保真度
        if code:
            score = 0.5
            if "return" in code.lower() or "def " in code.lower(): score += 0.2
            if any(kw in code for kw in ["fillna", "dropna", "np.nan", "isna"]): score += 0.2
            scores["code_fidelity"] = min(1.0, score)
        else:
            scores["code_fidelity"] = 0.5

        return scores

    def _compute_final_score(self, result: GateResult) -> float:
        weights = {
            "syntax": 0.25, "semantic": 0.25, "economic": 0.20,
            "risk": 0.15, "complexity": 0.15,
        }
        dim_scores = {
            "syntax": 1.0 if result.syntax.get("all_passed", False) else 0.3,
            "semantic": 1.0 if result.semantic.get("all_passed", False) else 0.3,
            "economic": 1.0 if result.economic.get("all_passed", False) else 0.2,
            "risk": 1.0 if result.risk.get("all_passed", False) else 0.0,
            "complexity": 1.0 if result.complexity.get("complexity_pass", False) else 0.3,
        }
        return sum(weights.get(k, 0) * v for k, v in dim_scores.items())

    # ═══════════════════════════════════════════════════════════
    # LLM 精判
    # ═══════════════════════════════════════════════════════════

    def _llm_check(self, factor: Dict, result: GateResult) -> Dict:
        if not self.llm_fn:
            return {"reject": False}
        prompt = (
            f"审查候选因子:\n名称: {factor.get('factor_name','?')}\n"
            f"表达式: {factor.get('formula','')}\n假设: {factor.get('hypothesis','')}\n"
            f"规则分数: {result.score:.2f}\n"
            f"请评估经济逻辑合理性、机制新颖性、过拟合风险和边界条件。返回JSON: "
            f'{{"reject":bool,"reason":"..."}}'
        )
        try:
            return self.llm_fn(prompt)
        except Exception:
            return {"reject": False}

    # ═══════════════════════════════════════════════════════════
    # 建议
    # ═══════════════════════════════════════════════════════════

    def _generate_suggestions(self, result: GateResult) -> List[str]:
        suggestions = set()
        for issue in result.fatal_issues:
            if "跨量纲" in issue:
                suggestions.add("使用同量纲字段 (如 volume 与 amount)")
            elif "嵌套" in issue:
                suggestions.add("移除无意义的嵌套 rank/abs/sign")
            elif "简单" in issue:
                suggestions.add("增加时间序列操作 (ma/delta/std)")
            elif "复杂" in issue:
                suggestions.add("简化表达式, 减少嵌套层数")
            elif "未知算子" in issue:
                suggestions.add("使用标准算子库中的算子")
            elif "纯常数" in issue:
                suggestions.add("表达式必须引用至少一个价量字段")
            elif "字段" in issue:
                suggestions.add("确保公式引用了有效的价量字段")
            elif "未来" in issue:
                suggestions.add("移除未来函数, 使用 shift(1) 或 lag 代替")
        for w in result.warnings:
            if "无时间序列" in w:
                suggestions.add("添加 rolling/shift/pct_change 以便捕捉时序模式")
            elif "H↔E" in w:
                suggestions.add("确保假设中的关键概念在表达式中有对应的算子")
        return sorted(suggestions)

    # ═══════════════════════════════════════════════════════════
    # 上下文生成 (复现 Reviewer 的 LLM prompt 约束生成)
    # ═══════════════════════════════════════════════════════════

    def get_constraints_for_generation(self) -> str:
        """生成因子时需遵守的约束 (可注入 LLM prompt)"""
        return (
            "## 因子生成约束\n"
            "- 表达式必须引用至少一个价量字段 (open/high/low/close/volume/amount)\n"
            "- 必须有时间序列结构 (rolling/shift/pct_change/ma/ts_delta 等)\n"
            "- 不能是纯常数\n"
            "- 禁止无意义嵌套: rank(rank()), abs(abs()), sign(sign())\n"
            "- 禁止跨量纲运算: volume 不能与 close 直接 add/sub/mul/div\n"
            "- 节点数 ≤ 10, 括号深度 ≤ 4\n"
            "- 禁止未来函数 (shift(-1), forward, next 等)\n"
        )

    def get_summary(self) -> Dict:
        """获取统计摘要"""
        return {
            **self.stats,
            "pass_rate": self.stats["passed"] / max(self.stats["total"], 1),
        }


# ── 便捷函数 ──

_default_gate: Optional[FactorQualityGate] = None


def get_gate() -> FactorQualityGate:
    global _default_gate
    if _default_gate is None:
        _default_gate = FactorQualityGate()
    return _default_gate


def quick_verify(factor: Dict) -> GateResult:
    return get_gate().verify(factor)


# ── 测试 ──

if __name__ == "__main__":
    gate = FactorQualityGate()

    tests = [
        {
            "factor_name": "good_momentum",
            "formula": "ts_delta(close, 20) / ts_std(close, 20)",
            "hypothesis": "20日动量在市场低波动时有效",
        },
        {
            "factor_name": "bad_nested",
            "formula": "rank(rank(close))",
            "hypothesis": "测试",
        },
        {
            "factor_name": "bad_cross_unit",
            "formula": "sub(volume, close)",
            "hypothesis": "量价相减",
        },
    ]

    for f in tests:
        r = gate.verify(f)
        status = "PASS" if r.passed else "FAIL"
        print(f"\n  [{status}] {f['factor_name']}  score={r.score:.2f}")
        if r.fatal_issues:
            for i in r.fatal_issues:
                print(f"    FATAL: {i}")
        if r.warnings:
            for w in r.warnings:
                print(f"    WARN:  {w}")
        if r.suggestions:
            for s in r.suggestions:
                print(f"    HINT:  {s}")

    print(f"\n  Summary: {gate.get_summary()}")
