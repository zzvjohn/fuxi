# -*- coding: utf-8 -*-
"""
Factor Expression Tree — 因子表达式树表示 + GP 交叉育种
===========================================================

对标中金 CICC Loop Engineering 报告中的五维演化策略，实现:
  1. 因子 DSL (Domain Specific Language) 解析 → 表达式树 (AST)
  2. 五种演化操作: mutate(变异) / crossover(交叉) / perturb(扰动) / 
     random(随机) / llm_guide(LLM引导)
  3. 从成功模板的交叉育种
  4. 与 FSA (频繁子树规避) 集成

DSL 语法（类 Lisp 风格，兼容 Python 字符串）:
  Factor  = Field | Operator(params, Factor...)
  Field   = open | high | low | close | volume | overnight | ...
  Operator = ma | delta | sub | mul | div | rank | ts_std | ...

表达式树节点:
  ExpressionNode
    ├── OperatorNode(operator, children, params)
    ├── FieldNode(field_name)
    └── ConstantNode(value)

设计原则:
  - 表达式树化后可以遍历、修改子树、交换子树、计算深度
  - 与 FSA 无缝集成：树 → 骨架指纹
  - 支持从 LLM 生成的代码字符串解析为树

集成路径:
  - Ralph Loop G 阶段: 替代纯 LLM 生成，引入 GP 操作
  - 预算分配: mutate 25% / crossover 25% / perturb 15% / random 15% / llm 20%

用法:
    from factor_expression_tree import FactorExpressionTree, GPBreeder
    
    tree = FactorExpressionTree().parse("sub(ma(overnight, 60), ma(close, 20))")
    
    breeder = GPBreeder()
    child = breeder.crossover(tree1, tree2)
    variant = breeder.mutate(tree, fsa=fsa)
"""

import re
import random
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from collections import deque


# ── 算子/字段定义 ────────────────────────────────────

# 参数化算子（带窗口/参数）— 转为有序列表确保确定性
PARAMETRIC_OPERATORS = {
    "ma", "ts_mean", "ts_std", "ts_min", "ts_max", "ts_sum",
    "ts_delta", "ts_rank", "ts_skewness", "ts_kurtosis",
    "ts_regression", "ts_decay_linear", "ts_delay",
    "ema", "wma", "delta", "roc", "rolling",
    "ts_corr", "ts_cov",
    # v0.6.1 (2026-08-29): ts_pct/ts_zscore 补入 — 原集合遗漏导致
    # ts_pct(close, 5) 的窗口 5 不被提取为 param, to_pandas_infix 里
    # n=params[0] 回退 20 → 嵌套在 ts_corr 内时窗口错误传播
    # (ts_corr(x, ts_pct(close,5), 20) → close.pct_change(20) 而非 5)
    "ts_pct", "ts_zscore",
}
PARAMETRIC_OPERATORS_LIST = sorted(PARAMETRIC_OPERATORS)  # 确定性采样

# 二元算子
BINARY_OPERATORS = {
    "add", "sub", "mul", "div", "pow", "min", "max",
    "if_else", "clip", "corr",
}
BINARY_OPERATORS_LIST = sorted(BINARY_OPERATORS)

# 一元算子
UNARY_OPERATORS = {
    "rank", "rank_cs", "scale", "zscore", "cs_zscore",
    "sqrt", "abs", "log", "sign", "neg", "demean",
    "normalize",
}
UNARY_OPERATORS_LIST = sorted(UNARY_OPERATORS)

# 所有算子
ALL_OPERATORS = PARAMETRIC_OPERATORS | BINARY_OPERATORS | UNARY_OPERATORS | {"ts_pct", "ts_zscore"}

# 字段名
FIELD_NAMES = {
    "open", "high", "low", "close", "volume", "amount", "vwap",
    "overnight", "intraday", "returns", "amplitude", "turnover",
    "market_cap", "mv", "float_mv", "up_shadow", "down_shadow",
    "hl_ratio", "atr", "swing", "gap",
    "buy_vol", "sell_vol", "buy_lg_vol", "sell_lg_vol",
    "buy_elg_vol", "sell_elg_vol", "buy_sm_vol", "sell_sm_vol",
    # v0.10 (2026-08-25 红利审计): 估值 + 财务质量字段
    "pe_ttm", "pb", "dv_ratio",
    "roe_dt", "tr_yoy", "netprofit_yoy",
}

# v0.3.1: S5 评估器能提供的字段（含 _p 后缀和衍生字段）
# 这些是 daily_prices.csv 可推导出来的真实数据列
S5_AVAILABLE_FIELDS = {
    "close", "close_p", "open", "open_p", "high", "high_p",
    "low", "low_p", "volume", "volume_p", "amount", "amount_p",
    "overnight", "overnight_p", "amplitude", "amplitude_p",
    "returns", "returns_p", "turnover", "turnover_p",
    "hl_ratio", "hl_ratio_p",
    # v0.10 (2026-08-25 红利审计): 估值字段 (daily_basic 慢变量, 按股 ffill)
    "pe_ttm", "pe_ttm_p", "pb", "pb_p", "dv_ratio", "dv_ratio_p",
    # v0.10: 财务质量字段 (fina_indicator, ann_date 公告日 asof 无前视)
    "roe_dt", "roe_dt_p", "tr_yoy", "tr_yoy_p", "netprofit_yoy", "netprofit_yoy_p",
}

# v0.3.1: S5 eval 中使用的 Python/pandas 内置词（不是字段引用）
S5_EVAL_KEYWORDS = {
    "rolling", "shift", "pct_change", "mean", "std", "skew", "kurt",
    "corr", "cov", "rank", "apply", "lambda", "abs", "sqrt", "log",
    "np", "pd", "range", "len", "sum", "min", "max", "int", "float",
    "True", "False", "None", "not", "and", "or", "if", "else",
    "astype", "nan", "inf", "idx", "values", "index", "array",
    "iloc", "loc", "head", "tail", "dropna", "fillna", "isna",
    "exp", "sign", "clip", "where", "log1p", "round",
    "sub", "div", "mul", "add", "neg",
    "cumsum", "cumprod", "argsort", "sort_values", "rank_pct",
    "pipe", "fillna", "bfill", "ffill", "interpolate",
    "diff", "corrcoef", "x", "raw", "ewm", "span", "polyfit",
    "arange", "argsort", "quantile", "cumsum", "cumprod",
    "replace",
    # v0.5 伏羲: 添加所有 DSL 操作符, 避免 _validate_field_refs 将 ForgeDSL 算子误判为未知字段
    "ma", "ts_mean", "ts_std", "ts_min", "ts_max", "ts_sum",
    "ts_delta", "ts_rank", "ts_skewness", "ts_kurtosis",
    "ts_regression", "ts_decay_linear", "ts_delay",
    "ts_corr", "ts_cov", "ts_zscore",
    "ema", "wma", "delta", "roc",
    "pow", "gt", "lt", "gte", "lte", "eq", "neq", "if_else",
    "rank_cs", "zscore", "cs_zscore", "scale", "demean",
    "normalize", "sqrt", "log1p", "sign", "abs",
    # 常见字段简写（避免误判为未知）
    "pct", "pct_chg", "ret", "vwap", "mv",
    # numpy 函数别名
    "maximum", "minimum",
}


# ── 表达式树节点 ──────────────────────────────────────

@dataclass
class ExprNode:
    """表达式树节点基类"""
    def to_expression(self) -> str:
        raise NotImplementedError

    def to_skeleton(self) -> str:
        """生成结构骨架（字段名→FIELD，窗口→N{i}）"""
        raise NotImplementedError

    def node_count(self) -> int:
        raise NotImplementedError

    def depth(self) -> int:
        raise NotImplementedError

    def children(self) -> List["ExprNode"]:
        raise NotImplementedError

    def clone(self) -> "ExprNode":
        raise NotImplementedError

    def to_pandas_infix(self) -> str:
        """将表达式树转为 pandas infix 格式（用于 S5 评估等场景）"""
        raise NotImplementedError

    def get_fields(self) -> Set[str]:
        """收集子树中所有字段引用（用于兼容性检查）"""
        raise NotImplementedError

    def get_all_subtrees(self) -> List["ExprNode"]:
        """收集所有子树（用于交叉操作）"""
        result = [self]
        for child in self.children():
            result.extend(child.get_all_subtrees())
        return result

    def replace_child(self, old: "ExprNode", new: "ExprNode") -> bool:
        """替换子节点，返回是否成功"""
        return False


@dataclass
class FieldNode(ExprNode):
    """字段节点: close, volume, overnight, ..."""
    name: str

    def to_expression(self) -> str:
        return self.name

    def to_skeleton(self) -> str:
        return "FIELD"

    def to_pandas_infix(self) -> str:
        return self.name

    def node_count(self) -> int:
        return 1

    def depth(self) -> int:
        return 1

    def children(self) -> List[ExprNode]:
        return []

    def clone(self) -> "FieldNode":
        return FieldNode(name=self.name)

    def get_fields(self) -> Set[str]:
        return {self.name}


@dataclass
class ConstantNode(ExprNode):
    """常数节点: 1.0, 0.5, ..."""
    value: float

    def to_expression(self) -> str:
        if self.value == int(self.value):
            return str(int(self.value))
        return f"{self.value:.4f}"

    def to_skeleton(self) -> str:
        return "N"

    def to_pandas_infix(self) -> str:
        if self.value == int(self.value):
            return str(int(self.value))
        return f"{self.value:.4f}"

    def node_count(self) -> int:
        return 1

    def depth(self) -> int:
        return 1

    def children(self) -> List[ExprNode]:
        return []

    def clone(self) -> "ConstantNode":
        return ConstantNode(value=self.value)

    def get_fields(self) -> Set[str]:
        return set()


@dataclass
class OperatorNode(ExprNode):
    """算子节点: ma(...), sub(...), rank(...), ..."""
    operator: str
    operands: List[ExprNode] = field(default_factory=list)
    params: List[float] = field(default_factory=list)  # 窗口参数等

    def to_expression(self) -> str:
        args = self.operands[:]
        # 如果有参数（窗口等），附加到表达式
        if self.params:
            args_str = ", ".join(
                [a.to_expression() for a in args] +
                [str(int(p)) if p == int(p) else f"{p:.1f}" for p in self.params]
            )
        else:
            args_str = ", ".join(a.to_expression() for a in args)

        return f"{self.operator}({args_str})"

    def to_pandas_infix(self) -> str:
        """将 DSL 算子节点转为 pandas infix 表达式（用于 S5 eval 等场景）
        
        v0.3.1: 增加异常保护 — 任何转换异常都 fallback 到 DSL 格式
        """
        op = self.operator
        try:
            args = [a.to_pandas_infix() for a in self.operands]
        except Exception:
            # 子节点转换失败 → 退回 DSL
            return self.to_expression()
        
        params = self.params
        n = int(params[0]) if params else 20

        # ── 二元算术 ──
        if op == "add": return f"({args[0]} + {args[1]})"
        if op == "sub": return f"({args[0]} - {args[1]})"
        if op == "mul": return f"({args[0]} * {args[1]})"
        if op == "div": return f"({args[0]} / {args[1]})"
        if op == "pow": return f"({args[0]} ** {args[1]})"
        if op == "gt": return f"({args[0]} > {args[1]})"
        if op == "lt": return f"({args[0]} < {args[1]})"
        if op == "gte": return f"({args[0]} >= {args[1]})"
        if op == "lte": return f"({args[0]} <= {args[1]})"
        if op == "eq": return f"({args[0]} == {args[1]})"
        if op == "neq": return f"({args[0]} != {args[1]})"
        if op == "neg": return f"(-{args[0]})"

        # ── 一元算子 ──
        if op == "abs": return f"np.abs({args[0]})"
        if op == "sqrt": return f"np.sqrt({args[0]})"
        if op == "log": return f"np.log({args[0]})"
        if op == "log1p": return f"np.log1p({args[0]})"
        if op == "exp": return f"np.exp({args[0]})"
        if op == "sign": return f"np.sign({args[0]})"
        if op == "demean": return f"({args[0]} - {args[0]}.rolling({n}).mean())"
        if op == "normalize":
            return f"(({args[0]} - {args[0]}.rolling({n}).min()) / ({args[0]}.rolling({n}).max() - {args[0]}.rolling({n}).min()).clip(1e-10))"
        if op == "scale": return f"(({args[0]} - {args[0]}.mean()) / {args[0]}.std().clip(1e-10))"
        if op == "zscore": return f"(({args[0]} - {args[0]}.rolling({n}).mean()) / {args[0]}.rolling({n}).std().clip(1e-10))"
        if op == "ts_zscore": return f"(({args[0]} - {args[0]}.rolling({n}).mean()) / {args[0]}.rolling({n}).std().clip(1e-10))"
        if op == "cs_zscore": return f"(({args[0]} - {args[0]}.rolling({n}).mean()) / {args[0]}.rolling({n}).std().clip(1e-10))"
        if op == "rank": return f"{args[0]}.rank(pct=True)"
        if op == "rank_cs": return f"{args[0]}.rank(pct=True)"
        if op == "rolling":
            return f"{args[0]}.rolling({int(n)})"
        if op == "shift": return f"{args[0]}.shift({int(n)})"
        if op == "pct_change": return f"{args[0]}.pct_change({int(n)})"
        if op == "diff": return f"{args[0]}.diff({int(n)})"
        if op == "astype": return f"{args[0]}.astype(float)"

        # ── 二元 numpy ──
        if op == "max" and len(args) >= 2: return f"np.maximum({args[0]}, {args[1]})"
        if op == "max" and len(args) == 1: return f"{args[0]}.max()"  # v0.3.2: 不重复加 rolling
        if op == "min" and len(args) >= 2: return f"np.minimum({args[0]}, {args[1]})"
        if op == "min" and len(args) == 1: return f"{args[0]}.min()"  # v0.3.2: 不重复加 rolling
        if op == "where" and len(args) >= 3: return f"np.where({args[0]}, {args[1]}, {args[2]})"
        if op == "clip" and len(args) >= 3: return f"({args[0]}).clip({args[1]}, {args[2]})"
        if op == "clip" and len(args) >= 2: return f"({args[0]}).clip({args[1]})"
        if op == "if_else" and len(args) >= 3: return f"np.where({args[0]}, {args[1]}, {args[2]})"
        if op == "corr" and len(args) >= 2: return f"{args[0]}.rolling({int(n)}).corr({args[1]})"

        # ── 参数化算子（含窗口） ──
        # P-20260831-002: 常数参数进 ts_* 窗口 → "0.30.rolling(5)" SyntaxError
        # (invalid decimal literal), 退化为常数本身 (rank IC 对常数缩放平移不变)
        _num_literal = re.match(r'^-?\d+(\.\d+)?([eE][+-]?\d+)?$', str(args[0]).strip())
        if _num_literal and op in ("ma", "ts_mean", "ts_std", "ts_min", "ts_max",
                                   "ts_sum", "ts_skewness", "ts_kurtosis",
                                   "ts_delta", "delta", "ts_delay", "ema", "wma",
                                   "roc", "ts_pct", "ts_rank", "ts_regression",
                                   "rolling", "shift", "pct_change", "diff",
                                   "corr", "rank", "rank_cs"):
            return str(args[0]).strip()
        if op in ("ma", "ts_mean"): return f"{args[0]}.rolling({n}).mean()"
        if op == "ts_std": return f"{args[0]}.rolling({n}).std()"
        if op == "ts_min": return f"{args[0]}.rolling({n}).min()"
        if op == "ts_max": return f"{args[0]}.rolling({n}).max()"
        if op == "ts_sum": return f"{args[0]}.rolling({n}).sum()"
        if op == "ts_skewness": return f"{args[0]}.rolling({n}).skew()"
        if op == "ts_kurtosis": return f"{args[0]}.rolling({n}).kurt()"
        if op in ("ts_delta", "delta"): return f"({args[0]} - {args[0]}.shift({n}))"
        if op == "ts_delay": return f"{args[0]}.shift({n})"
        if op == "ema": return f"{args[0]}.ewm(span={n}).mean()"
        if op == "wma": return f"{args[0]}.ewm(span={n}).mean()"
        if op == "roc": return f"{args[0]}.pct_change({n})"
        if op == "ts_pct": return f"{args[0]}.pct_change({n})"
        if op == "ts_corr" and len(args) >= 2: return f"{args[0]}.rolling({n}).corr({args[1]})"
        if op == "ts_cov" and len(args) >= 2: return f"{args[0]}.rolling({n}).cov({args[1]})"
        # Single-arg fallback for binary operators: auto-complete with close_p as second arg
        if op == "ts_corr": return f"{args[0]}.rolling({n}).corr(close_p)" if args else self.to_expression()
        if op == "ts_cov": return f"{args[0]}.rolling({n}).cov(close_p)" if args else self.to_expression()
        if op == "ts_rank":
            return f"{args[0]}.rolling({n}).apply(lambda x: (pd.Series(x).rank().iloc[-1]-1)/(len(x)-1) if len(x)>1 else 0.5, raw=False)"
        if op == "ts_regression":
            return f"{args[0]}.rolling({n}).apply(lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] if len(x)>5 else 0.0, raw=True)"

        # ── method names (PANDAS_METHODS used as operators) ──
        if op == "mean" and args:
            return f"{args[0]}.mean()"
        if op == "std" and args:
            return f"{args[0]}.std()"
        if op == "sum" and args:
            return f"{args[0]}.sum()"
        if op == "skew" and args:
            return f"{args[0]}.skew()"
        if op == "kurt" and args:
            return f"{args[0]}.kurt()"
        if op == "quantile" and args:
            q = float(params[0]) if params else 0.5
            return f"{args[0]}.quantile({q})"
        if op == "cumsum" and args:
            return f"{args[0]}.cumsum()"
        if op == "cumprod" and args:
            return f"{args[0]}.cumprod()"
        if op == "fillna" and args:
            val = params[0] if params else 0
            return f"{args[0]}.fillna({val})"

        # ── fallback: return DSL (may not eval in pandas context) ──
        return self.to_expression()

    def to_skeleton(self) -> str:
        args_str = ", ".join(a.to_skeleton() for a in self.operands)
        # 参数也抽象化
        for i in range(len(self.params)):
            args_str += f", N{i+1}"
        return f"{self.operator}({args_str})"

    def node_count(self) -> int:
        return 1 + sum(o.node_count() for o in self.operands)

    def depth(self) -> int:
        if not self.operands:
            return 1
        return 1 + max(o.depth() for o in self.operands)

    def children(self) -> List[ExprNode]:
        return self.operands[:]

    def clone(self) -> "OperatorNode":
        return OperatorNode(
            operator=self.operator,
            operands=[o.clone() for o in self.operands],
            params=self.params[:],
        )

    def replace_child(self, old: ExprNode, new: ExprNode) -> bool:
        for i, child in enumerate(self.operands):
            if child is old:
                self.operands[i] = new
                return True
            if isinstance(child, OperatorNode) and child.replace_child(old, new):
                return True
        return False

    def get_fields(self) -> Set[str]:
        result = set()
        for child in self.operands:
            result.update(child.get_fields())
        return result

    def structural_fingerprint(self) -> str:
        """子树的结构指纹（用于 saliency 匹配）— operator+fields"""
        fields = sorted(self.get_fields())
        return f"{self.operator}({','.join(fields)})"


# ═══════════════════════════════════════════════════════════
# 统一解析器 (Pratt Parser) — 支持 DSL 和 Pandas infix
# ═══════════════════════════════════════════════════════════

# 价格字段名（含 _p 后缀和裸字段名）
PRICE_FIELDS = {
    "open", "open_p", "high", "high_p", "low", "low_p",
    "close", "close_p", "volume", "volume_p", "amount", "amount_p",
    "vwap", "vwap_p", "overnight", "overnight_p",
    "returns", "returns_p", "amplitude", "amplitude_p",
    "turnover", "turnover_p", "market_cap", "mv",
    "float_mv", "up_shadow", "down_shadow",
    "hl_ratio", "atr", "swing", "gap",
    "buy_vol", "sell_vol", "buy_lg_vol", "sell_lg_vol",
    "buy_elg_vol", "sell_elg_vol", "buy_sm_vol", "sell_sm_vol",
    "pre_close", "change", "pct_chg", "ret",
}

# Python builtins that appear in factor expressions
PY_BUILTINS = {"float", "int", "True", "False", "None", "lower", "upper", "other"}

# Pandas method names (all converted to OperatorNode with method name)
PANDAS_METHODS = {
    "rolling", "shift", "mean", "std", "sum", "max", "min",
    "pct_change", "astype", "abs", "where", "clip", "fillna",
    "rank", "skew", "kurt", "quantile", "cumsum", "cumprod",
    "diff", "corr", "cov",
}

# Infix operator → DSL function name mapping
INFIX_TO_DSL = {
    "+": "add", "-": "sub", "*": "mul", "/": "div",
    ">": "gt", "<": "lt", ">=": "gte", "<=": "lte",
    "==": "eq", "!=": "neq",
}

# Precedence (higher = binds tighter). 0 = lowest.
PREC = {
    "COMPARE": 1,   # > < >= <= == !=
    "ADDSUB": 2,    # + -
    "MULDIV": 3,    # * /
    "UNARY": 4,     # unary -
    "POSTFIX": 5,   # .method()
}


class FactorExpressionParser:
    """
    统一因子表达式解析器 — 支持 DSL 函数式 & Pandas infix 格式。

    使用 Pratt (top-down operator precedence) 解析器:
      - 原子: FIELD, NUMBER, 分组括号 (expr)
      - 后缀: .method(args...) 方法链
      - 一元: -expr
      - 二元: + - * / > < >= <= == != (按优先级)

    所有格式统一转换为 ExprNode AST，保持骨架一致性。
    """

    @classmethod
    def parse(cls, expression: str) -> Optional[ExprNode]:
        """解析因子表达式为 AST。支持 DSL 和 Pandas infix 两种格式。"""
        if not expression or not expression.strip():
            return None

        expr = expression.strip()

        # 多行公式预处理: 去掉注释，提取最后一个有效表达式
        expr = cls._preprocess_multiline(expr)

        # 去掉赋值前缀: "ret_1d = close.pct_change()" → "close.pct_change()"
        if "=" in expr and not any(op in expr for op in ["<=", ">=", "==", "!="]):
            parts = expr.split("=", 1)
            possible_assign = parts[0].strip()
            if re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', possible_assign):
                expr = parts[1].strip()

        # 标准化算子别名
        expr = cls._normalize_operators(expr)

        # 科学计数法规范化: 1e8 → 100000000.0, 1e-6 → 0.000001
        expr = cls._normalize_scientific(expr)

        try:
            tokens = cls._tokenize(expr)
            if not tokens:
                return None
            node, pos = cls._parse_pratt(tokens, 0, 0)
            return node
        except Exception:
            return None

    @classmethod
    def _preprocess_multiline(cls, expr: str) -> str:
        """多行公式预处理: 去掉注释行/行尾注释，去掉 import 语句，提取最终表达式。"""
        lines = expr.split('\n')
        clean_lines = []
        for line in lines:
            # 跳过 import 语句
            if re.match(r'^\s*(import|from)\s+', line):
                continue
            # 去掉行尾注释 (# ...)
            comment_pos = line.find('#')
            if comment_pos >= 0:
                line = line[:comment_pos]
            line = line.strip()
            if line:
                clean_lines.append(line)

        if not clean_lines:
            return expr

        # 如果最后一行是赋值 (var = ...) 或表达式，取它
        # 否则取最长的非赋值行（最可能是最终表达式）
        last = clean_lines[-1]
        if '=' in last and not any(op in last for op in ['<=', '>=', '==', '!=']):
            parts = last.split('=', 1)
            if re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', parts[0].strip()):
                last = parts[1].strip()

        # 如果最后一行太短 (< 10 chars)，找最长行
        if len(last) < 10 and len(clean_lines) > 1:
            longest = max(clean_lines, key=len)
            if '=' in longest and not any(op in longest for op in ['<=', '>=', '==', '!=']):
                parts = longest.split('=', 1)
                if re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', parts[0].strip()):
                    longest = parts[1].strip()
            return longest

        return last

    @classmethod
    def _normalize_scientific(cls, expr: str) -> str:
        """将科学计数法转换为普通数字: 1e8 → 100000000.0, 1e-6 → 0.000001"""
        def _replace(m):
            try:
                return str(float(m.group(0)))
            except ValueError:
                return m.group(0)
        return re.sub(r'\b\d+\.?\d*[eE][+-]?\d+\b', _replace, expr)

    # ── Tokenizer ───────────────────────────────────────

    _TOKEN_RE = re.compile(r"""
        (?:\.([a-zA-Z_][a-zA-Z0-9_]*))   # 1: DOT method name (captured with leading dot context)
        |([a-zA-Z_][a-zA-Z0-9_]*)          # 2: IDENT (field, function, builtin)
        |(\d+\.?\d*)                       # 3: NUMBER
        |(==|!=|>=|<=|[+\-*/><(),])       # 4: OPERATOR / paren / comma
        |(\S)                              # 5: UNKNOWN (catch-all)
    """, re.VERBOSE)

    @classmethod
    def _tokenize(cls, expr: str) -> List[Tuple[str, str]]:
        """分词: 完整的 token 流，保留操作符和分组信息。"""
        tokens = []
        i = 0
        while i < len(expr):
            ch = expr[i]

            # 跳过空白
            if ch in ' \t\r\n':
                i += 1
                continue

            # DOT + method name: .rolling, .shift, .mean, etc.
            if ch == '.':
                m = re.match(r'\.([a-zA-Z_][a-zA-Z0-9_]*)', expr[i:])
                if m:
                    tokens.append(("DOT", m.group(1)))
                    i += len(m.group(0))
                    continue
                else:
                    i += 1  # stray dot, skip
                    continue

            # 多字符操作符: == != >= <=
            if i + 1 < len(expr) and expr[i:i+2] in ("==", "!=", ">=", "<="):
                tokens.append(("OP", expr[i:i+2]))
                i += 2
                continue

            # 单字符操作符 + 括号
            if ch in "+-*/><":
                tokens.append(("OP", ch))
                i += 1
                continue
            if ch == '(':
                tokens.append(("LPAREN", "("))
                i += 1
                continue
            if ch == ')':
                tokens.append(("RPAREN", ")"))
                i += 1
                continue
            if ch == ',':
                tokens.append(("COMMA", ","))
                i += 1
                continue

            # 标识符
            m_ident = re.match(r'[a-zA-Z_][a-zA-Z0-9_]*', expr[i:])
            if m_ident:
                ident = m_ident.group(0)
                # 判断类型
                if ident.lower() in PRICE_FIELDS or ident in PRICE_FIELDS:
                    tokens.append(("FIELD", ident))
                elif ident in PY_BUILTINS:
                    tokens.append(("BUILTIN", ident))
                elif ident in PANDAS_METHODS:
                    tokens.append(("METHOD", ident))
                elif ident in ALL_OPERATORS:
                    tokens.append(("FUNC", ident))
                else:
                    tokens.append(("IDENT", ident))
                i += len(ident)
                continue

            # 数字
            m_num = re.match(r'\d+\.?\d*', expr[i:])
            if m_num:
                tokens.append(("NUMBER", m_num.group(0)))
                i += len(m_num.group(0))
                continue

            # 跳过无法识别的字符
            i += 1

        return tokens

    # ── Pratt Parser ─────────────────────────────────────

    @classmethod
    def _parse_pratt(
        cls, tokens: List[Tuple[str, str]], pos: int, min_prec: int
    ) -> Tuple[Optional[ExprNode], int]:
        """Pratt 解析器: 处理前缀 + 后缀/中缀循环。"""
        if pos >= len(tokens):
            return None, pos

        # ── NUD (Null Denotation): 前缀解析 ──
        node, pos = cls._parse_prefix(tokens, pos)

        # ── LED (Left Denotation): 后缀/中缀循环 ──
        while node is not None and pos < len(tokens):
            tt, tv = tokens[pos]

            # --- 中缀操作符 ---
            if tt == "OP":
                op_prec = cls._get_precedence(tv)
                if op_prec >= min_prec:
                    pos += 1
                    right, pos = cls._parse_pratt(tokens, pos, op_prec + 1)
                    if right is not None:
                        dsl_op = INFIX_TO_DSL.get(tv, tv)
                        node = OperatorNode(operator=dsl_op, operands=[node, right])
                    continue
                else:
                    break

            # --- DOT 方法链 ---
            if tt == "DOT":
                method_name = tv
                pos += 1

                # 解析方法参数
                method_args = []
                is_call = False
                if pos < len(tokens) and tokens[pos][0] == "LPAREN":
                    is_call = True
                    pos += 1  # skip (
                    while pos < len(tokens):
                        tt2, tv2 = tokens[pos]
                        if tt2 == "RPAREN":
                            pos += 1
                            break
                        if tt2 == "COMMA":
                            pos += 1
                            continue
                        # 跳过关键字参数前缀 (lower=, upper=)
                        if tt2 == "IDENT" and pos + 1 < len(tokens) and tokens[pos + 1][0] == "OP" and tokens[pos + 1][1] == "=":
                            pos += 2  # skip ident and =
                            continue
                        arg, pos = cls._parse_pratt(tokens, pos, 0)
                        if arg is not None:
                            method_args.append(arg)

                # Build method node: method(prev_node, arg1, arg2, ...)
                # Separate params (numbers) from operands (expressions)
                operands = [node]
                params = []
                for a in method_args:
                    if isinstance(a, ConstantNode):
                        params.append(a.value)
                    else:
                        operands.append(a)

                node = OperatorNode(
                    operator=method_name,
                    operands=operands,
                    params=params,
                )
                continue

            # --- 直接函数调用: func_name( → DSL 模式或 bare 方法调用 ---
            if tt in ("FUNC", "METHOD"):
                func_name = tv
                pos += 1
                if pos < len(tokens) and tokens[pos][0] == "LPAREN":
                    pos += 1  # skip (
                call_args = []
                while pos < len(tokens):
                    tt2, tv2 = tokens[pos]
                    if tt2 == "RPAREN":
                        pos += 1
                        break
                    if tt2 == "COMMA":
                        pos += 1
                        continue
                    arg, pos = cls._parse_pratt(tokens, pos, 0)
                    if arg is not None:
                        call_args.append(arg)
                operands = []
                params = []
                for a in call_args:
                    if isinstance(a, ConstantNode) and func_name in PARAMETRIC_OPERATORS:
                        params.append(a.value)
                    elif isinstance(a, ConstantNode):
                        operands.append(a)
                    else:
                        operands.append(a)
                node = OperatorNode(operator=func_name, operands=operands, params=params)
                continue

            break  # no more LED matches

        return node, pos

    @classmethod
    def _parse_prefix(
        cls, tokens: List[Tuple[str, str]], pos: int
    ) -> Tuple[Optional[ExprNode], int]:
        """前缀解析: FIELD | NUMBER | ( expr ) | - expr | FUNC (call)"""
        if pos >= len(tokens):
            return None, pos

        tt, tv = tokens[pos]

        # 字段
        if tt == "FIELD":
            return FieldNode(name=tv), pos + 1

        # 标识符/Built-in（可能是不带 _p 的字段名或 Python 内建）
        if tt in ("IDENT", "BUILTIN"):
            return FieldNode(name=tv), pos + 1

        # 方法名作为函数调用: abs(...), pct_change(...)
        if tt == "METHOD":
            func_name = tv
            pos += 1
            if pos < len(tokens) and tokens[pos][0] == "LPAREN":
                pos += 1  # 跳过 (
            call_args = []
            while pos < len(tokens):
                tt2, tv2 = tokens[pos]
                if tt2 == "RPAREN":
                    pos += 1
                    break
                if tt2 == "COMMA":
                    pos += 1
                    continue
                arg, pos = cls._parse_pratt(tokens, pos, 0)
                if arg is not None:
                    call_args.append(arg)
            operands = []
            params = []
            for a in call_args:
                if isinstance(a, ConstantNode) and func_name in PARAMETRIC_OPERATORS:
                    params.append(a.value)
                elif isinstance(a, ConstantNode):
                    operands.append(a)
                else:
                    operands.append(a)
            return OperatorNode(operator=func_name, operands=operands, params=params), pos

        # DSL 函数调用: sub(..., ...), ma(field, N), rank(...), etc.
        if tt == "FUNC":
            func_name = tv
            pos += 1
            if pos < len(tokens) and tokens[pos][0] == "LPAREN":
                pos += 1  # 跳过 (
            call_args = []
            while pos < len(tokens):
                tt2, tv2 = tokens[pos]
                if tt2 == "RPAREN":
                    pos += 1
                    break
                if tt2 == "COMMA":
                    pos += 1
                    continue
                arg, pos = cls._parse_pratt(tokens, pos, 0)
                if arg is not None:
                    call_args.append(arg)
            operands = []
            params = []
            for a in call_args:
                if isinstance(a, ConstantNode) and func_name in PARAMETRIC_OPERATORS:
                    params.append(a.value)
                elif isinstance(a, ConstantNode):
                    operands.append(a)
                else:
                    operands.append(a)
            return OperatorNode(operator=func_name, operands=operands, params=params), pos

        # 数字
        if tt == "NUMBER":
            return ConstantNode(value=float(tv)), pos + 1

        # 分组括号: ( expr )
        if tt == "LPAREN":
            pos += 1
            node, pos = cls._parse_pratt(tokens, pos, 0)
            if pos < len(tokens) and tokens[pos][0] == "RPAREN":
                pos += 1
            return node, pos

        # 一元负号: - expr
        if tt == "OP" and tv == "-":
            pos += 1
            child, pos = cls._parse_pratt(tokens, pos, PREC["UNARY"])
            if child is not None:
                return OperatorNode(operator="neg", operands=[child]), pos
            return None, pos

        # 一元正号: + expr (rare, treated as identity)
        if tt == "OP" and tv == "+":
            pos += 1
            child, pos = cls._parse_pratt(tokens, pos, PREC["UNARY"])
            return child, pos

        # 无法解析，跳过
        return None, pos + 1

    @classmethod
    def _get_precedence(cls, op: str) -> int:
        """获取中缀操作符的优先级。"""
        if op in (">", "<", ">=", "<=", "==", "!="):
            return PREC["COMPARE"]
        if op in ("+", "-"):
            return PREC["ADDSUB"]
        if op in ("*", "/"):
            return PREC["MULDIV"]
        return 0

    @classmethod
    def _normalize_operators(cls, expr: str) -> str:
        """标准化算子名称（仅用于 DSL 模式）"""
        aliases = {
            "ts_mean": "ma", "ts_delta": "delta",
            "ts_rank": "ts_rank", "cs_zscore": "zscore",
        }
        for alias, canonical in aliases.items():
            if alias != canonical:
                expr = re.sub(rf'\b{re.escape(alias)}\b', canonical, expr)
        return expr


class ChildNodeWrapper:
    """临时包装器（保留向后兼容）"""
    def __init__(self, node):
        self._node = node
    def to_expression(self):
        return self._node.to_expression() if hasattr(self._node, 'to_expression') else str(self._node)


# ═══════════════════════════════════════════════════════════
# v0.4: Directed GP — 域分类 + 先验结构
# ═══════════════════════════════════════════════════════════

# 因子域分类 (heuristic, 基于公式关键词)
DOMAIN_PATTERNS = {
    "trend_momentum": [
        r'\bpct_change\b', r'\.diff\(', r'ts_delta\b', r'\bovernight\b',
        r'returns_p?\b', r'\bdelta\b', r'corrcoef\b',
    ],
    "mean_reversion": [
        r'zscore\b', r'demean\b', r'\.rank\(', r'normalize\b',
        r'/.*\.rolling\(.*\.mean\(\)',  # 除以均线
    ],
    "volatility_size": [
        r'\.std\(\)', r'\.skew\(\)', r'\bamplitude\b', r'kurtosis\b',
    ],
    "volume_flow": [
        r'\bvolume_p?\b', r'\bamount_p?\b', r'turnover_p?\b',
        r'buy_.*vol\b', r'sell_.*vol\b', r'money_flow\b',
    ],
    "price_pattern": [
        r'hl_ratio\b', r'\bhigh_p?\b.*\blow_p?\b',
        r'\bopen_p?\b.*\bclose_p?\b', r'\bclose_p?\b.*\bopen_p?\b',
        r'up_shadow\b', r'down_shadow\b',
    ],
    "cross_sectional": [
        r'rank_cs\b', r'cs_zscore\b', r'\.rank\(pct=True\)',
        r'cross.section', r'\brank_pct\b',
    ],
}

# 应用了哪些算子（细粒度，用于转移矩阵）
OPERATOR_FAMILIES = {
    "trend": ["pct_change", "diff", "ts_delta", "returns"],
    "smoothing": ["rolling", "ewm", "ema", "ma", "ts_mean"],
    "dispersion": ["std", "skew", "kurtosis", "ts_std", "ts_skewness"],
    "rank_norm": ["rank", "zscore", "demean", "normalize", "scale"],
    "binary_arith": ["sub", "add", "mul", "div"],
    "comparison": ["gt", "lt", "gte", "lte", "clip"],
}


@dataclass
class GPPriors:
    """v0.4: 半监督引导 GP 的先验知识库
    
    由 MetaController.distill_rules() 在每轮 JQ 反馈后更新，
    注入 GPBreeder 的交叉/变异决策中。
    """
    # E: 字段加权采样 (Dirichlet-Categorical)
    field_weights: Dict[str, float] = field(default_factory=dict)
    
    # E: 算子加权采样
    operator_weights: Dict[str, float] = field(default_factory=dict)
    
    # A: 域互补对 (domain1, domain2, complementarity_score)
    domain_complementarity: List[Tuple[str, str, float]] = field(default_factory=list)
    
    # E: 窗口分布 (log-normal 拟合的参数)
    window_weights: Dict[int, float] = field(default_factory=dict)
    
    # D: Thompson Sampling 算子选择 (op_name → (successes, failures))
    operator_success: Dict[str, Tuple[int, int]] = field(default_factory=dict)
    
    # A: 模板→域标签映射
    template_domains: Dict[str, str] = field(default_factory=dict)
    
    # B: 子树显著度 (structural_fingerprint → saliency_score)
    subtree_saliency: Dict[str, float] = field(default_factory=dict)
    
    # 积木库: (pandas_expression, score)
    building_blocks: List[Tuple[str, float]] = field(default_factory=list)
    
    # 默认域（无信息先验）
    DEFAULT_DOMAIN = "unknown"

    @classmethod
    def empty(cls) -> "GPPriors":
        """创建空先验（纯随机，向后兼容）"""
        return cls()

    def has_signal(self) -> bool:
        """是否有足够的信号做引导"""
        return len(self.operator_success) >= 2 or len(self.field_weights) >= 3
    
    @classmethod
    def from_meta_rules(cls, rules: List[Any], genes: List[Any]) -> "GPPriors":
        """v0.4: 从 MetaController 的蒸馏规则 + 策略基因中提取 GP 先验
        
        Parameters
        ----------
        rules: DistilledRule 列表
        genes: StrategyGene 列表
        """
        priors = cls()
        
        # E: 从成功基因中提取字段频率
        field_counts: Dict[str, float] = {}
        operator_fam_counts: Dict[str, float] = {}
        domain_success: Dict[str, int] = {}
        window_counts: Dict[int, int] = {}
        op_success_counts: Dict[str, Tuple[int, int]] = {}  # (success, failure)
        
        jq_validated = [g for g in genes if getattr(g.fitness, 'jq_sharpe', 0) or 
                        getattr(g.fitness, 'is_jq_validated', lambda: False)()]
        
        for gene in genes:
            fitness = getattr(gene, 'fitness', None)
            if fitness is None:
                continue
            
            # 判断成功/失败
            is_success = getattr(fitness, 'is_profitable', lambda: False)()
            is_jq = getattr(fitness, 'is_jq_validated', lambda: False)()
            
            # 从 factors 中提取字段
            factors = getattr(gene.dna, 'factors', [])
            for factor_name in factors:
                formula = cls._resolve_formula(factor_name, gene)
                if formula:
                    domain = classify_factor_domain(formula)
                    domain_success[domain] = domain_success.get(domain, 0) + (1 if is_success else 0)
                    
                    # 提取字段引用
                    fields = cls._extract_fields(formula)
                    for f in fields:
                        field_counts[f] = field_counts.get(f, 0.0) + (2.0 if is_success and is_jq else 0.5)
                    
                    # 提取窗口
                    windows = cls._extract_windows(formula)
                    for w in windows:
                        window_counts[w] = window_counts.get(w, 0) + (2 if is_success else 1)
        
        # 归一化为权重
        total_f = sum(field_counts.values()) if field_counts else 1.0
        priors.field_weights = {k: v/total_f * 10 for k, v in field_counts.items()} if field_counts else {}
        
        total_w = sum(window_counts.values()) if window_counts else 1.0
        priors.window_weights = {k: max(v/total_w, 0.02) for k, v in window_counts.items()} if window_counts else {}
        
        # 域互补：successful domains 作为高互补分
        for d1 in domain_success:
            for d2 in domain_success:
                if d1 < d2:
                    score = (domain_success[d1] + domain_success[d2]) / max(domain_success.values(), default=1) * 0.5
                    priors.domain_complementarity.append((d1, d2, min(score, 1.0)))
        
        # D: Thompson Sampling 初始化
        default_beta = (1, 1)  # Beta(1,1) = Uniform
        for op_name in ["crossover", "mutate", "perturb"]:
            if op_name in op_success_counts:
                priors.operator_success[op_name] = op_success_counts[op_name]
            else:
                priors.operator_success[op_name] = default_beta
        
        return priors
    
    @staticmethod
    def _resolve_formula(factor_name: str, gene: Any) -> Optional[str]:
        """从基因中解析因子公式"""
        # 尝试多种可能的路径
        for attr in ['formula', 'expression', 'factor_formula']:
            val = getattr(gene.dna, attr, None) if hasattr(gene, 'dna') else getattr(gene, attr, None)
            if val and isinstance(val, str):
                return val
        return None
    
    @staticmethod
    def _extract_fields(formula: str) -> Set[str]:
        """从公式中提取字段引用"""
        import re
        all_words = set(re.findall(r'[a-zA-Z_]\w*', formula))
        return all_words & S5_AVAILABLE_FIELDS
    
    @staticmethod
    def _extract_windows(formula: str) -> List[int]:
        """从公式中提取窗口值"""
        import re
        windows = re.findall(r'\.rolling\((\d+)\)', formula)
        return [int(w) for w in windows]


def classify_factor_domain(formula: str) -> str:
    """v0.4: 基于公式关键词的因子域分类
    
    Returns:
        "trend_momentum" | "mean_reversion" | "volatility_size" | 
        "volume_flow" | "price_pattern" | "cross_sectional" | "mixed"
    """
    if not formula:
        return "unknown"
    
    scores = {}
    for domain, patterns in DOMAIN_PATTERNS.items():
        score = sum(1 for p in patterns if re.search(p, formula))
        scores[domain] = score
    
    # 取最高分域
    if not scores or max(scores.values()) == 0:
        return "unknown"
    
    top = sorted(scores.items(), key=lambda x: -x[1])
    
    # 如果有接近的并列第一 → mixed
    if len(top) >= 2 and top[0][1] == top[1][1] and top[0][1] > 0:
        return "mixed"
    
    return top[0][0]


# ═══════════════════════════════════════════════════════════
# GP 育种器
# ═══════════════════════════════════════════════════════════

class GPBreeder:
    """
    基于表达式树的 GP 育种器。

    对标中金五维演化策略:
      - mutate (25%): 替换子树
      - crossover (25%): 交换子树
      - perturb (15%): 微调窗口参数
      - random (15%): 数据驱动特征分布引导的随机生成
      - llm_guide (20%): LLM 机制引导生成
    """

    # 窗口参数范围
    WINDOW_RANGES = [5, 10, 15, 20, 30, 40, 50, 60, 80, 100, 120, 150, 200, 250]

    # 默认字段
    # v0.3.1: 仅使用 S5 评估器已知的字段（避免交叉育种引入模板局部变量）
    DEFAULT_FIELDS = ["close_p", "volume_p", "overnight", "amplitude", "hl_ratio",
                      "open_p", "high_p", "low_p", "amount_p", "returns", "turnover"]
    
    # v0.3.1: S5 可用字段标识符（用于验证）
    S5_FIELDS = S5_AVAILABLE_FIELDS

    def __init__(
        self,
        parser: Optional[FactorExpressionParser] = None,
        max_depth: int = 5,
        max_nodes: int = 20,
        seed: Optional[int] = None,
        priors: Optional[GPPriors] = None,  # v0.4: 半监督先验
        penalizer: Optional[Any] = None,  # P-007: 子结构频率惩罚器 (None=关闭, 行为与改造前一致)
        penalty_retry: int = 6,  # P-007: 软拒绝后重试上限 (耗尽放行防死循环)
        layered_selection: bool = True,  # P-20260826-005: 分层奖励父本采样 (False=均匀采样回退)
        edit_memory: Optional[Any] = None,  # P-20260827-001: SSPM 编辑记忆 (None=关闭, 行为与改造前一致)
        dim_prune: Optional[str] = None,  # P-20260831: off/shadow/enforce 维度预剪枝 (None=读 env)
        diversity_w: float = 0.0,  # P-20260831 P1: Alpha2 多样性折扣权重 (0=关闭; 0.10 建议)
    ):
        self.parser = parser or FactorExpressionParser()
        self.max_depth = max_depth
        self.max_nodes = max_nodes
        self.priors = priors  # v0.4: 可选的 GP 先验知识库
        self.penalizer = penalizer  # P-007: SubstructurePenalizer 实例或 None
        self.penalty_retry = penalty_retry  # P-007: 软拒绝重试次数
        self.layered_selection = layered_selection  # P-20260826-005: 分层奖励父本采样
        self.edit_memory = edit_memory  # P-20260827-001: SSPMEditMemory 或 None
        # P-20260831: 维度预剪枝状态 (shadow 只计数, enforce 拒绝硬违规子代)
        try:
            from forge import dimension_rules as _dr
            self.dim_prune = dim_prune if dim_prune is not None else _dr.get_mode()
        except ImportError:
            self.dim_prune = "off"
        self.dim_hard: int = 0
        self.dim_soft: int = 0
        # P-20260831 P1: 多样性折扣 (Alpha2 MaxCorr 移植, w>0 才启用查询)
        self.diversity_w = diversity_w
        self._div_cache = None  # JQDiversityCache 惰性创建
        if seed is not None:
            random.seed(seed)

    def update_priors(self, priors: "GPPriors"):
        """v0.4: 更新 GP 先验（通常在 JQ 反馈后调用）"""
        self.priors = priors

    # ── P-20260826-005: 分层奖励父本采样 ─────────────────

    def _layered_template_score(self, meta, precomputed_corr=None) -> float:
        """模板分层奖励得分 (L2 稳健 + L3 稀缺), clamp [0.2, 1.0]

        L2 稳健: verification_level (jq_single > jq_composite > s5_passed > stage2_only)
                 + success_rate (模板历史通过率)
        L3 稀缺: 1/(1 + 0.3 × occurrence_count) — 库内出现越多越拥挤, 越该让位
        元数据完全缺失时返回 0.5 (中性, 均匀退化)。

        precomputed_corr (2026-09-01 转正配套): _layered_template_weights 批量
        预查询的 jq_max_corr (避免逐模板单查 → 每查重 eval 整个参照库)。
        """
        lv, sr, occ = 'stage2_only', 0.0, 0
        if isinstance(meta, dict):
            lv = str(meta.get('verification_level', 'stage2_only') or 'stage2_only')
            try:
                sr = float(meta.get('success_rate', 0) or 0)
            except (TypeError, ValueError):
                sr = 0.0
            try:
                occ = int(meta.get('occurrence_count', 0) or 0)
            except (TypeError, ValueError):
                occ = 0
        elif meta is not None:
            lv = str(getattr(meta, 'verification_level', 'stage2_only') or 'stage2_only')
            try:
                sr = float(getattr(meta, 'success_rate', 0) or 0)
            except (TypeError, ValueError):
                sr = 0.0
            try:
                occ = int(getattr(meta, 'occurrence_count', 0) or 0)
            except (TypeError, ValueError):
                occ = 0
        l2_map = {'jq_single': 1.0, 'jq_composite': 0.85, 's5_passed': 0.7,
                  'stage2_only': 0.5, '': 0.4}
        l2 = l2_map.get(lv, 0.4) * 0.7 + min(max(sr, 0.0), 1.0) * 0.3
        l3 = 1.0 / (1.0 + 0.3 * occ)
        score = max(0.2, min(1.0, 0.6 * l2 + 0.4 * l3))

        # P-20260831 P1: 多样性折扣 (Alpha2 MaxCorr 移植) —
        # 与已 JQ 正面验证因子高相关的模板降采样, 引导变异远离已占用方向。
        # shadow: 只进统计不改 score; enforce: score *= max(0.5, 1 - w·corr)。
        if self.diversity_w > 0:
            try:
                from forge import diversity_discount as _dd
                if precomputed_corr is None:
                    fml = ""
                    mname = ""
                    if isinstance(meta, dict):
                        fml = str(meta.get("formula", meta.get("expression", "")) or "")
                        mname = str(meta.get("factor_name", meta.get("pattern_id", "")) or "")
                    elif meta is not None:
                        fml = str(getattr(meta, "formula",
                                          getattr(meta, "expression", "")) or "")
                        mname = str(getattr(meta, "factor_name",
                                            getattr(meta, "pattern_id", "")) or "")
                    if fml:
                        if self._div_cache is None:
                            self._div_cache = _dd.get_shared_cache()
                        precomputed_corr = self._div_cache.get_corr(fml, mname)
                if _dd.is_enforce():
                    score = _dd.apply_discount(score, precomputed_corr or 0.0,
                                               w=self.diversity_w)
            except Exception:
                pass  # 折扣任何异常都不影响主流程 (软引导)
        return score

    def _layered_template_weights(self, metas) -> List[float]:
        if not self.layered_selection:
            return [1.0] * len(metas)
        # 2026-09-01 P1 转正配套: 批量预查询 jq_max_corr —
        # 逐模板单查每查重 eval 整个 JQ 参照库 (~20 公式全市场 eval),
        # 模板池 100+ 时冷启动需数十分钟; 批量一次只 eval 参照库一遍。
        _fmls, _names = [], []
        for m in metas:
            if isinstance(m, dict):
                fml = str(m.get("formula", m.get("expression", "")) or "")
                nm = str(m.get("factor_name", m.get("pattern_id", "")) or "")
            elif m is not None:
                fml = str(getattr(m, "formula", getattr(m, "expression", "")) or "")
                nm = str(getattr(m, "factor_name", getattr(m, "pattern_id", "")) or "")
            else:
                fml, nm = "", ""
            _fmls.append(fml)
            _names.append(nm or f"tpl_{len(_names)}")
        corr_map = {}
        if self.diversity_w > 0:
            try:
                from forge import diversity_discount as _dd
                if self._div_cache is None:
                    self._div_cache = _dd.get_shared_cache()
                corr_map = self._div_cache.get_corrs_batch(_fmls, _names)
            except Exception:
                corr_map = {}
        return [self._layered_template_score(m, corr_map.get(n))
                for m, n in zip(metas, _names)]

    def _layered_parent_choice(self, trees, weights):
        """分层加权父本选择 (权重退化均匀时行为等价 random.choice)"""
        if not trees:
            return None
        if not self.layered_selection or len(trees) < 3 or len(set(round(w, 3) for w in weights)) <= 1:
            return random.choice(trees)
        return random.choices(trees, weights=weights, k=1)[0]

    # ── 变异 (Mutate) ───────────────────────────────────

    def mutate(
        self,
        tree: ExprNode,
        fsa: Any = None,
    ) -> Optional[ExprNode]:
        """
        变异: 随机选择一个子树节点，替换为随机生成的新子树。

        对标中金: 以 50% 概率从入库因子（已通过 11 项筛选）中选取父本，
        在其附近搜索成功率显著优于宽泛采样。

        Parameters
        ----------
        tree: 父本表达式树
        fsa: SubtreeFingerprinter 实例（可选，用于避免生成被禁骨架）

        Returns
        -------
        变异后的新树 (clone + mutate)
        """
        mutant = tree.clone()
        subtrees = mutant.get_all_subtrees()

        if len(subtrees) < 2:
            return None

        # 随机选一个非根子树替换
        target = random.choice(subtrees[1:])  # 排除根节点
        depth_limit = self.max_depth - target.depth()

        # 随机生成替换子树
        replacement = self._random_subtree(depth_limit, self.max_nodes)

        # 找到 target 在 mutant 中的位置并替换
        self._replace_in_tree(mutant, target, replacement)

        # FSA 检查
        if fsa and fsa.check_expression(mutant.to_expression()):
            return None  # 被禁止，拒绝变异

        return mutant

    # ── 交叉 (Crossover) ────────────────────────────────

    def crossover(
        self,
        parent1: ExprNode,
        parent2: ExprNode,
        fsa: Any = None,
    ) -> Optional[ExprNode]:
        """
        交叉: 从 parent1 中选一个子树，替换 parent2 中选的一个子树。
        
        对标中金: 交叉的意义在于融合不同来源的结构特征。

        v0.4: 三处指向性升级
          - 从 parent1 选 subtree: saliency 加权 (B)
          - 从 parent2 选 target: saliency 加权 (B)  
          - 嫁接前兼容性检查 (C)
        """
        subtrees1 = parent1.get_all_subtrees()
        subtrees2 = parent2.get_all_subtrees()

        if len(subtrees1) < 2 or len(subtrees2) < 2:
            return None

        # v0.4B: 显著度加权的 graft 子树选择
        candidates1 = [s for s in subtrees1 if s.depth() >= 1]
        if not candidates1:
            return None
        
        if self.priors and self.priors.has_signal():
            # 显著度加权采样
            saliencies = [self._subtree_saliency(s) for s in candidates1]
            total = sum(saliencies)
            if total > 0:
                probs = [s / total for s in saliencies]
                graft = candidates1[random.choices(range(len(candidates1)), weights=probs, k=1)[0]]
            else:
                graft = random.choice(candidates1)
        else:
            graft = random.choice(candidates1)
        
        child = parent2.clone()

        # v0.4C: 嫁接兼容性预检
        subtrees_child = child.get_all_subtrees()
        candidates_child = [s for s in subtrees_child if s.depth() >= 1]
        if not candidates_child:
            return None
        
        if self.priors and self.priors.has_signal():
            # 兼容性过滤 + 显著度加权
            compatible = []
            saliencies_c = []
            for s in candidates_child:
                if self._graft_compatible(graft, s, child):
                    compatible.append(s)
                    saliencies_c.append(self._subtree_saliency(s))
            
            if not compatible:
                # 全部不兼容 → 退回随机
                compatible = candidates_child
                saliencies_c = [1.0] * len(compatible)
            
            total = sum(saliencies_c)
            if total > 0:
                probs = [s / total for s in saliencies_c]
                target = compatible[random.choices(range(len(compatible)), weights=probs, k=1)[0]]
            else:
                target = random.choice(compatible)
        else:
            target = random.choice(candidates_child)

        # 深度约束
        if target.depth() + graft.depth() > self.max_depth:
            return None

        self._replace_in_tree(child, target, graft.clone())

        # FSA 检查
        if fsa and fsa.check_expression(child.to_expression()):
            return None

        return child

    # ── 参数扰动 (Perturb) ──────────────────────────────

    def perturb(
        self,
        tree: ExprNode,
        momentum_direction: int = 0,  # -1/0/+1, 来自动量追踪器
    ) -> Optional[ExprNode]:
        """
        参数扰动: 表达式结构不变，微调时间序列算子的窗口参数。

        对标中金: 调整方向由动量追踪器给出，步长随梯度强度自适应调整。

        Parameters
        ----------
        tree: 表达式树
        momentum_direction: 窗口调整方向 (-1=缩短, 0=随机, +1=延长)

        Returns
        -------
        扰动后的新树
        """
        variant = tree.clone()

        # 收集所有 parametric 算子节点
        param_nodes = []
        self._collect_parametric_nodes(variant, param_nodes)

        if not param_nodes:
            return None

        # 随机选一个 parametric 节点
        target = random.choice(param_nodes)

        # 调整窗口参数
        for i in range(len(target.params)):
            old_val = int(target.params[i])
            # 在 WINDOW_RANGES 中找相邻值
            try:
                idx = self.WINDOW_RANGES.index(old_val)
            except ValueError:
                # 不是标准窗口，就近替换
                idx = min(range(len(self.WINDOW_RANGES)),
                         key=lambda j: abs(self.WINDOW_RANGES[j] - old_val))

            if momentum_direction > 0:
                new_idx = min(idx + 1, len(self.WINDOW_RANGES) - 1)
            elif momentum_direction < 0:
                new_idx = max(idx - 1, 0)
            else:
                # 随机: ±1 或保持
                delta = random.choice([-1, 0, 1])
                new_idx = max(0, min(idx + delta, len(self.WINDOW_RANGES) - 1))

            target.params[i] = self.WINDOW_RANGES[new_idx]

        return variant

    # ── 随机生成 (Random) ───────────────────────────────

    def random_generate(
        self,
        field_weights: Optional[Dict[str, float]] = None,
        fsa: Any = None,
    ) -> Optional[ExprNode]:
        """
        随机生成：在字段/算子空间随机组合产生新因子。

        对标中金: 数据驱动特征分布引导（overnight 权重×4 等）。

        Parameters
        ----------
        field_weights: 字段权重分布（如 {"overnight": 4.0, "amplitude": 2.0}）
        fsa: 用于避免生成被禁骨架
        """
        fields = field_weights or {f: 1.0 for f in self.DEFAULT_FIELDS}
        depth = random.randint(2, self.max_depth)

        for _ in range(5):  # 最多重试5次
            tree = self._random_tree(depth, fields)
            expr = tree.to_expression()
            if not fsa or not fsa.check_expression(expr):
                return tree
        return None

    # ── 交叉育种（从模板批量生成） ─────────────────────

    def breed_from_templates(
        self,
        templates: List[Dict],  # 每个包含 'formula' 或 'expression'
        n_children: int = 10,
        fsa: Any = None,
        output_format: str = "pandas",  # "dsl" | "pandas"
        paradigm: Optional[str] = None,  # v0.9.1: MAB 方向标签 (缺失时保持 auto_breed)
    ) -> List[Dict]:
        """
        从成功模板库中交叉育种，批量生成新候选因子。

        Parameters
        ----------
        templates: 成功因子模板列表 (从 Experience Memory 获取)
        n_children: 生成数量
        fsa: FSA 实例
        output_format: "dsl" 输出 DSL 格式; "pandas" 输出 pandas infix 格式

        Returns
        -------
        新候选因子列表
        """
        # 解析所有模板为树 (P-20260826-005: 同步保留元数据供分层父本采样)
        trees = []
        tree_metas = []
        for t in templates:
            # Handle both dict and object (SuccessPattern, etc.)
            if isinstance(t, dict):
                expr = t.get("formula", t.get("expression", ""))
            elif hasattr(t, 'formula'):
                expr = getattr(t, 'formula', '')
            elif hasattr(t, 'expression'):
                expr = getattr(t, 'expression', '')
            else:
                expr = ""
            if not expr:
                continue
            tree = self.parser.parse(expr)
            if tree:
                trees.append(tree)
                tree_metas.append(t)

        if len(trees) < 2:
            return []

        # P-20260826-005: 分层父本权重 (L2 稳健 + L3 稀缺; 元数据缺失时退化为均匀)
        parent_weights = self._layered_template_weights(tree_metas)
        _w_uniq = len(set(round(w, 3) for w in parent_weights))
        if self.layered_selection and _w_uniq > 1:
            print(f"  [Layered] 分层父本采样启用: {len(trees)} 模板, "
                  f"权重范围 {min(parent_weights):.2f}~{max(parent_weights):.2f}")

        children = []
        attempts = 0
        max_attempts = n_children * 12  # v0.4: 更多重试以补偿 directed 过滤

        # v0.6: 持久化育种编号 — 跨运行不重置
        _breed_counter = self._get_next_breed_id()

        # v0.4A: 预计算模板域标签（用于语义配对）
        template_domains = {}
        if self.priors:
            for t in templates:
                if isinstance(t, dict):
                    expr = t.get("formula", t.get("expression", ""))
                elif hasattr(t, 'formula'):
                    expr = getattr(t, 'formula', '')
                elif hasattr(t, 'expression'):
                    expr = getattr(t, 'expression', '')
                else:
                    expr = ""
                if expr:
                    template_domains[expr] = classify_factor_domain(expr)

        while len(children) < n_children and attempts < max_attempts:
            attempts += 1
            # P-20260901-005: 父因子上下文 (编辑模式记忆影子, 随候选落盘)
            _parents = []
            # v0.4D: Thompson Sampling 算子选择
            # P-20260827-001: 透传 MAB 方向标签供 SSPM 条件化否决
            op = self._choose_operator_thompson(paradigm)

            if op == "crossover":
                # v0.4A: 语义父本配对 — 优先选不同域的两个模板
                if self.priors and self.priors.has_signal() and len(trees) >= 2:
                    # 加权配对：先用域互补分选第一个，再选最互补的第二个
                    candidates = random.sample(trees, min(10, len(trees)))
                    
                    # 先随机选 p1，然后按语义互补分选 p2
                    p1 = random.choice(candidates)
                    remaining = [t for t in candidates if t is not p1]
                    
                    if remaining:
                        pair_weights = [self._semantic_pair_weight(p1, t) for t in remaining]
                        total = sum(pair_weights)
                        if total > 0:
                            probs = [w / total for w in pair_weights]
                            p2 = remaining[random.choices(range(len(remaining)), weights=probs, k=1)[0]]
                        else:
                            p2 = random.choice(remaining)
                    else:
                        p1, p2 = random.choice(trees), random.choice(trees)
                else:
                    p1 = random.choice(trees)
                    p2 = random.choice(trees)
                child = self.crossover(p1, p2, fsa)
                _parents = [p1.to_expression(), p2.to_expression()]
            elif op == "mutate":
                parent = self._layered_parent_choice(trees, parent_weights)
                child = self.mutate(parent, fsa)
                _parents = [parent.to_expression()]
            else:
                parent = self._layered_parent_choice(trees, parent_weights)
                child = self.perturb(parent)
                _parents = [parent.to_expression()]

            if child and child.node_count() <= self.max_nodes:
                # v0.3.1: 输出 pandas infix 格式（兼容 S5 eval）
                if output_format == "pandas":
                    try:
                        expr = child.to_pandas_infix()
                    except Exception:
                        expr = child.to_expression()
                else:
                    expr = child.to_expression()
                
                # v0.3.2: 验证公式有效性
                if not expr or not isinstance(expr, str):
                    continue
                
                # v0.3.2: 验证 pandas 格式（不是原始 DSL）
                if output_format == "pandas":
                    if not self._is_valid_pandas_formula(expr):
                        # DSL fallback — 尝试再转一次
                        dsl_expr = child.to_expression()
                        pandas_expr = dsl_to_pandas_infix(dsl_expr)
                        if pandas_expr and self._is_valid_pandas_formula(pandas_expr):
                            expr = pandas_expr
                        else:
                            # 用 DSL 原样输出（后续 S5 的 _normalize_to_pandas 会再处理）
                            expr = dsl_expr
                
                # v0.3.2: 公式清理（修正常见的 crossover 产物）
                expr = self._cleanup_formula(expr)

                # P-20260831: 维度一致性审计 (shadow 计数 / enforce 拒绝硬违规)
                if self.dim_prune != "off":
                    try:
                        from forge import dimension_rules as _dr
                        _aud = _dr.audit_fet_node(child, record=False)
                        if _aud is not None:
                            if _aud.n_hard:
                                self.dim_hard += 1
                                if self.dim_prune == "enforce":
                                    continue
                            if _aud.n_soft:
                                self.dim_soft += 1
                    except ImportError:
                        pass
                
                # v0.3.2: 验证字段引用
                if output_format == "pandas":
                    is_valid, unknown = self._validate_field_refs(expr)
                    if not is_valid:
                        if len(unknown) <= 2 and all(u in S5_AVAILABLE_FIELDS for u in unknown):
                            pass  # 少量已知字段（可能含 _p 变体），允许通过
                        else:
                            if attempts < 5:
                                print(f"    [GP] Rejecting {child.node_count()} nodes: unknown fields {unknown}")
                            continue
                
                # v0.5 P-001: motif 级验证 — 用 Memory 已知模式过滤子代
                if fsa and getattr(fsa, '_motif_validator', None):
                    motifs = fsa.extract_fingerprints(child.to_expression())
                    try:
                        from experience_memory import get_memory
                        mem = get_memory()
                        motif_forbidden = mem.get_motif_forbidden(min_samples=3, max_rate=0.1)
                        if any(str(m) in motif_forbidden for m in motifs):
                            continue  # 子代包含已知失败 motif, 丢弃
                    except Exception:
                        pass
                
                # v0.5 P-001: GP 子代输出时带 motif 信息 (供 trajectory_logger 使用)
                child_motifs = []
                if fsa:
                    try:
                        fingerprints = fsa.extract_fingerprints(child.to_expression())
                        child_motifs = [f.fingerprint_key if hasattr(f, 'fingerprint_key') else str(f) 
                                       for f in fingerprints]
                    except Exception:
                        pass
                
                children.append({
                    "factor_name": f"gp_breed_{_breed_counter:03d}",
                    "formula": expr,
                    "expression": expr,
                    "depth": child.depth(),
                    "nodes": child.node_count(),
                    "source": f"gp_{op}",
                    "hypothesis": f"GP {op} from template pool",
                    "paradigm": paradigm or "auto_breed",  # v0.9.1: MAB 方向透传 (修复 auto_breed 断链)
                    "motifs": child_motifs,  # v0.5 P-001: 子代 motif 标注
                    "edit_meta": {"parents": _parents, "op": op},  # P-20260901-005 影子
                })
                _breed_counter += 1  # v0.6: 持久化递增

        # v0.6: 保存最新编号
        self._save_breed_counter(_breed_counter)
        # P-20260831: 维度审计摘要 (shadow 计数 / enforce 拦截)
        if self.dim_prune != "off" and (self.dim_hard or self.dim_soft):
            print(f"  [DimPrune:{self.dim_prune}] hard={self.dim_hard} "
                  f"soft={self.dim_soft} (维度一致性审计)")
        return children

    @staticmethod
    def _get_next_breed_id() -> int:
        """v0.6: 读取持久化育种编号，跨运行不重置"""
        import json, os
        # 从当前文件向上3层到 quant 项目根
        workspace = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        counter_path = os.path.join(workspace, 'data', 'breed_counter.json')
        try:
            with open(counter_path, 'r') as f:
                data = json.load(f)
            next_id = data.get('next_breed_id', 0)
        except (FileNotFoundError, json.JSONDecodeError):
            next_id = 0
        return next_id

    @staticmethod
    def _save_breed_counter(next_id: int):
        """v0.6: 保存最新育种编号"""
        import json, os
        workspace = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        counter_path = os.path.join(workspace, 'data', 'breed_counter.json')
        os.makedirs(os.path.dirname(counter_path), exist_ok=True)
        from datetime import datetime
        with open(counter_path, 'w') as f:
            json.dump({'next_breed_id': next_id, 'last_updated': datetime.now().isoformat()}, f,
                     indent=2, default=str)

    def breed_from_single_seed(
        self,
        formula: str,
        n_children: int = 10,
        fsa: Any = None,
        output_format: str = "pandas",
        motif_avoid: Optional[set] = None,
        motif_prefer: Optional[set] = None,
        paradigm: Optional[str] = None,  # v0.9.1: MAB 方向标签透传
    ) -> List[Dict]:
        """
        v0.6 EvoTraj: 从单个种子因子通过连续变异生成子代。

        用于多轮进化轨迹 (gp_evolve) — 每轮保留 best 1，
        下一轮以其为种子继续变异精炼。

        Parameters
        ----------
        formula: 种子 DSL 公式
        n_children: 生成数量
        fsa: FSA 实例
        output_format: "dsl" | "pandas"
        motif_avoid: 需要避开的 motif 集合 (P-001)
        motif_prefer: 优先的 motif 集合 (P-001)

        Returns
        -------
        子代因子列表
        """
        seed_tree = self.parser.parse(formula)
        if seed_tree is None:
            return []

        children = []
        attempts = 0
        max_attempts = n_children * 16  # v0.6: 单种子变异需要更多重试

        while len(children) < n_children and attempts < max_attempts:
            attempts += 1
            child = self.mutate(seed_tree, fsa)
            if child is None:
                continue

            expr = child.to_expression() if output_format == "dsl" else child.to_pandas_infix()
            if not expr or len(expr) < 5:
                continue

            # FSA 检查
            if fsa:
                try:
                    if fsa.check_expression(expr):
                        continue
                except Exception:
                    pass

            # P-001 motif 约束
            child_motifs = []
            if motif_avoid or motif_prefer:
                try:
                    fprints = fsa.extract_fingerprints(child.to_expression()) if fsa else []
                    child_motifs = [
                        getattr(f, 'fingerprint_key', str(f))
                        for f in fprints
                    ]
                except Exception:
                    pass
                if motif_avoid and any(m in motif_avoid for m in child_motifs):
                    continue

            # 字段引用验证
            is_valid, unknown = self._validate_field_refs(expr)
            if not is_valid:
                if attempts < 5:
                    pass
                continue

            children.append({
                "factor_name": f"gp_evo_{len(children):03d}",
                "formula": expr,
                "expression": expr,
                "depth": child.depth(),
                "nodes": child.node_count(),
                "source": "gp_evolve",
                "hypothesis": f"EvoTraj 第 N 轮变异自 {formula[:30]}...",
                "paradigm": paradigm or "auto_breed",  # v0.9.1: MAB 方向透传
                "motifs": child_motifs,
            })

        return children

    # ── v0.3.1: 字段验证 ──────────────────────────────
    
    @staticmethod
    def _validate_field_refs(formula: str) -> Tuple[bool, set]:
        """验证公式中的所有字段引用都在 S5 已知字段集中。
        
        Returns:
            (is_valid, unknown_fields_set)
        """
        import re
        
        # 移除字符串字面量
        cleaned = re.sub(r"'[^']*'", '', formula)
        cleaned = re.sub(r'"[^"]*"', '', cleaned)
        
        # 提取所有标识符
        all_words = set(re.findall(r'[a-zA-Z_]\w*', cleaned))
        
        # 过滤：数字、关键词、内置函数
        refs = all_words - S5_EVAL_KEYWORDS
        refs = {w for w in refs if not w.replace('_', '').isdigit()}
        refs = {w for w in refs if len(w) > 1}
        
        unknown = refs - S5_AVAILABLE_FIELDS
        
        return len(unknown) == 0, unknown
    
    @staticmethod
    def _cleanup_formula(formula: str) -> str:
        """v0.3.2: 清理 GP 交叉育种产生的格式问题"""
        import re
        
        # 1. 移除 'np' 作为孤立值（如 np.sign(np), np.abs(np)）
        #    匹配模式: func(np) 中的 np，但保留 np.func(...) 中的 np
        formula = re.sub(r'(?<![.\w])np(?=\s*[,\\)])', '1.0', formula)
        
        # 2. 修正过度精确的浮点数 (保留3位小数)
        def _round_float(m):
            val = float(m.group(0))
            return f'{val:.3f}'
        formula = re.sub(r'(?<!\w)(\d+\.\d{4,})(?!\w)', _round_float, formula)
        
        # 3. 移除多余的括号配对（如 )( 之间缺少运算符）
        #    这种通常是 crossover 产生的语法错误，跳过
        
        # 4. 修正 .rolling(.mean()... 这类畸形的方法链
        formula = re.sub(r'\.rolling\(\.(\w+)\(\)\)', r'.\1()', formula)
        
        return formula
    
    @staticmethod
    def _is_valid_pandas_formula(formula: str) -> bool:
        """快速检查公式是否为有效 pandas 表达式（而非原始 DSL）"""
        # DSL 特征：以算子名开头，如 sub(..., ma(..., div(...
        dsl_starters = ('sub(', 'add(', 'mul(', 'div(', 'neg(', 'abs(',
                        'ma(', 'rank(', 'ts_', 'rolling(', 'shift(', 'ema(')
        
        # 如果公式以 DSL 算子开头且不含 '.' 方法链 → 很可能是未转换的 DSL
        has_dsl_start = any(formula.strip().startswith(s) for s in dsl_starters)
        has_method_chain = '.' in formula
        
        if has_dsl_start and not has_method_chain:
            return False
        
        # 如果包含明显 DSL 关键字
        dsl_keywords = ['sub(', 'add(', 'mul(', 'div(', 'ts_mean(', 'ts_std(',
                        'ts_delta(', 'ma(', 'neg(']
        dsl_count = sum(1 for kw in dsl_keywords if kw in formula)
        if dsl_count >= 2 and '.' not in formula:
            return False
        
        return True

    # ── v0.4: Directed GP 辅助方法 ───────────────────────

    def _domain_tag(self, tree: ExprNode) -> str:
        """v0.4A: 对表达式树打域标签"""
        expr = tree.to_expression()
        return classify_factor_domain(expr)

    def _semantic_pair_weight(
        self, parent1: ExprNode, parent2: ExprNode
    ) -> float:
        """v0.4A: 计算语义配对权重（域互补分）
        
        两个父本来自不同域 → 高权重（促进多样性交叉）
        两个父本同域 → 低权重（避免同质交叉产生冗余）
        
        Returns:
            0.0~2.0 的权重值，1.0 为中性
        """
        d1 = self._domain_tag(parent1)
        d2 = self._domain_tag(parent2)
        
        # 不同域 → 高分（最大多样性）
        if d1 != d2 and d1 != "unknown" and d2 != "unknown":
            # 检查 priors 中是否有该域对的互补记录
            if self.priors and self.priors.domain_complementarity:
                for (da, db, score) in self.priors.domain_complementarity:
                    if {d1, d2} == {da, db}:
                        return 1.0 + score  # 1.0~2.0
            return 1.5  # 默认跨域加分
        
        # 同域且都有信号 → 惩罚（避免冗余）
        if d1 == d2 and d1 != "unknown":
            return 0.3  # 不禁止但降权
        
        return 1.0  # 中性

    def _subtree_saliency(self, subtree: ExprNode) -> float:
        """v0.4B: 计算子树的显著度（在历史成功中的贡献度）
        
        匹配子树的结构指纹 (operator+fields) 与 priors 中的 saliency map。
        
        Returns:
            0.0~2.0 的显著度分数，1.0 为中性
        """
        if not self.priors or not self.priors.subtree_saliency:
            return 1.0
        
        if not isinstance(subtree, OperatorNode):
            return 1.0  # Field/Constant 节点不参与 saliency
        
        fp = subtree.structural_fingerprint()
        score = self.priors.subtree_saliency.get(fp, 1.0)
        return max(score, 0.1)  # 不下探到 0

    def _graft_compatible(
        self, graft: ExprNode, target: ExprNode, parent_tree: ExprNode
    ) -> bool:
        """v0.4C: 嫁接兼容性检查
        
        规则:
        1. graft 的输入字段必须在 target 位置可获取（即存在于 parent_tree 的其他分支或全局字段中）
        2. graft 和 target 的维度不能冲突（都是 scalar 或都是 series）
        3. 禁止将不同频率的算子混入（如 intraday op 插入日频上下文）
        
        Returns:
            True if compatible, False if likely invalid
        """
        # 规则1: graft 需要的字段
        graft_fields = graft.get_fields()
        if not graft_fields:
            return True  # 没有字段引用（纯常数运算）
        
        # 检查字段是否在 S5 已知字段中（最小检查）
        if graft_fields - S5_AVAILABLE_FIELDS:
            return False  # 含有 S5 不认识的字段
        
        # 收集 parent_tree 中 target 位置以外可用的字段
        all_fields = parent_tree.get_fields()
        target_fields = target.get_fields()
        sibling_fields = all_fields - target_fields
        
        # 如果 graft 需要的字段不在 parent 的字段集中 → 嫁接可能产生残缺因子
        # 但宽松处理：如果至少有一个字段能用就放行
        usable = graft_fields & (sibling_fields | {"close_p", "open_p", "high_p", "low_p", "volume_p", "amount_p"})
        if not usable:
            return False
        
        # 规则2: 类型兼容 — 如果 graft 返回标量但 target 位置是向量上下文
        # 简化处理：如果 graft 是 rank/zscore 且 target 位置在 rolling 链中 → ok
        # (完整类型推断需要更复杂的分析，这里只做基本检查)
        
        return True

    def _choose_operator_thompson(self, paradigm: Optional[str] = None) -> str:
        """v0.4D: Thompson Sampling 选择交叉/变异/扰动操作

        每个算子维护 Beta(alpha, beta) 后验分布。
        每次选择时从后验采样，选采样值最高的。

        P-20260827-001 (SSPM): 若 (paradigm, op) 被编辑记忆非对称否决
        (hard_veto_enabled 时), 从候选集中剔除该算子; 全部被否时回退均匀采样
        (置信不足/软否决关闭时行为与改造前一致)。

        Returns:
            "crossover" | "mutate" | "perturb"
        """
        ops = ["crossover", "mutate", "perturb"]
        # P-20260827-001: SSPM 非对称否决过滤 (默认关闭=仅记录)
        if self.edit_memory is not None:
            allowed = [op for op in ops
                       if not self.edit_memory.should_veto(paradigm, op)]
            if allowed:
                ops = allowed
        if not self.priors or not self.priors.operator_success:
            return random.choice(ops)

        samples = {}
        for op in ops:
            a, b = self.priors.operator_success.get(op, (1, 1))
            # Beta 采样: Gamma(alpha,1) / (Gamma(alpha,1) + Gamma(beta,1))
            samples[op] = random.betavariate(max(a, 0.1), max(b, 0.1))

        return max(samples, key=samples.get)

    def update_operator_feedback(self, op: str, success: bool):
        """v0.4D: 更新算子选择的 Beta 后验
        
        每轮 JQ 验证后调用，成功 → alpha+1, 失败 → beta+1
        """
        if self.priors is None:
            self.priors = GPPriors.empty()
        
        a, b = self.priors.operator_success.get(op, (1, 1))
        if success:
            a += 1
        else:
            b += 1
        self.priors.operator_success[op] = (a, b)

    def _weighted_field_choice(self) -> str:
        """v0.4E: 加权字段选择（基于 priors.field_weights）"""
        if not self.priors or not self.priors.field_weights:
            return random.choice(self.DEFAULT_FIELDS)
        
        # Dirichlet-Categorical: normalize weights + pseudo-count
        fields = []
        weights = []
        for f in self.DEFAULT_FIELDS:
            w = self.priors.field_weights.get(f, 0.5)  # 默认小权重
            fields.append(f)
            weights.append(max(w, 0.01))  # 最低 0.01 避免零概率
        
        total = sum(weights)
        probs = [w / total for w in weights]
        return random.choices(fields, weights=probs, k=1)[0]

    def _weighted_operator_choice(
        self, op_type: str, depth: int, field_weights: Dict[str, float]
    ) -> Tuple[str, List[ExprNode], List[float]]:
        """v0.4E: 加权算子选择 + 生成子节点
        
        Returns:
            (operator_name, operands, params)
        """
        if op_type == "parametric":
            safe_parametric = [op for op in PARAMETRIC_OPERATORS_LIST 
                             if op not in ("rolling", "ts_regression", "ts_decay_linear")]
            
            if self.priors and self.priors.operator_weights:
                # 加权选择
                ops = []
                weights = []
                for op_name in safe_parametric[:8]:
                    w = self.priors.operator_weights.get(op_name, 0.5)
                    ops.append(op_name)
                    weights.append(max(w, 0.01))
                total = sum(weights)
                probs = [w / total for w in weights]
                op = random.choices(ops, weights=probs, k=1)[0]
            else:
                op = random.choice(safe_parametric[:8])
            
            child = self._random_tree(depth - 1, field_weights)
            
            # E: 加权窗口选择
            if self.priors and self.priors.window_weights:
                windows = list(self.priors.window_weights.keys())
                w_weights = list(self.priors.window_weights.values())
                window = random.choices(windows, weights=w_weights, k=1)[0]
            else:
                window = random.choice(self.WINDOW_RANGES)
            
            return op, [child], [float(window)]
        
        elif op_type == "binary":
            ops = ["sub", "mul", "div", "add", "max", "min"]
            if self.priors and self.priors.operator_weights:
                b_ops = [o for o in ops if o in self.priors.operator_weights]
                if b_ops:
                    weights = [max(self.priors.operator_weights[o], 0.01) for o in b_ops]
                    total = sum(weights)
                    probs = [w / total for w in weights]
                    op = random.choices(b_ops, weights=probs, k=1)[0]
                else:
                    op = random.choice(ops)
            else:
                op = random.choice(ops)
            left = self._random_tree(depth - 1, field_weights)
            right = self._random_tree(depth - 1, field_weights)
            return op, [left, right], []
        
        else:  # unary
            ops = ["rank", "zscore", "abs", "log", "neg"]
            if self.priors and self.priors.operator_weights:
                u_ops = [o for o in ops if o in self.priors.operator_weights]
                if u_ops:
                    weights = [max(self.priors.operator_weights[o], 0.01) for o in u_ops]
                    total = sum(weights)
                    probs = [w / total for w in weights]
                    op = random.choices(u_ops, weights=probs, k=1)[0]
                else:
                    op = random.choice(ops)
            else:
                op = random.choice(ops)
            child = self._random_tree(depth - 1, field_weights)
            return op, [child], []

    # ── 内部辅助 ─────────────────────────────────────────

    def _random_subtree(
        self, max_depth: int, max_nodes: int
    ) -> ExprNode:
        """生成随机子树
        
        v0.4E: 加权原语采样
        P-007: penalizer 启用时对高频子结构软拒绝并重试 (耗尽放行)
        """
        if max_depth <= 1 or max_nodes <= 1:
            if self.priors and self.priors.has_signal():
                return FieldNode(name=self._weighted_field_choice())
            return FieldNode(name=random.choice(self.DEFAULT_FIELDS))

        depth = random.randint(1, max_depth)
        for _ in range(self.penalty_retry + 1):  # P-007: +1 = 至少生成一次
            node = self._random_tree(depth, {f: 1.0 for f in self.DEFAULT_FIELDS})
            if not (self.penalizer and self.penalizer.should_reject(
                    node.to_expression(), random)):
                return node
        return node  # P-007: 重试耗尽放行, 防死循环

    def _random_tree(
        self, depth: int, field_weights: Dict[str, float]
    ) -> ExprNode:
        """递归生成随机表达式树
        
        v0.4E: 加权原语采样 — 字段/算子/窗口优先使用 priors
        """
        if depth <= 1:
            # v0.4E: 加权字段选择
            if self.priors and self.priors.has_signal():
                field = self._weighted_field_choice()
            else:
                fields = list(field_weights.keys())
                weights = list(field_weights.values())
                field = random.choices(fields, weights=weights, k=1)[0]
            return FieldNode(name=field)

        # 随机选操作符类型
        op_type = random.random()
        if op_type < 0.35:  # 35% parametric
            op, operands, params = self._weighted_operator_choice(
                "parametric", depth, field_weights
            )
            return OperatorNode(operator=op, operands=operands, params=params)
            
        elif op_type < 0.55:  # 20% binary
            op, operands, params = self._weighted_operator_choice(
                "binary", depth, field_weights
            )
            return OperatorNode(operator=op, operands=operands, params=params)
            
        elif op_type < 0.70:  # 15% unary
            op, operands, params = self._weighted_operator_choice(
                "unary", depth, field_weights
            )
            return OperatorNode(operator=op, operands=operands, params=params)
            
        else:  # 30% field
            # v0.4E: 加权字段选择
            if self.priors and self.priors.has_signal():
                field = self._weighted_field_choice()
            else:
                fields = list(field_weights.keys())
                weights = list(field_weights.values())
                field = random.choices(fields, weights=weights, k=1)[0]
            return FieldNode(name=field)

    def _replace_in_tree(
        self, root: ExprNode, target: ExprNode, replacement: ExprNode
    ) -> bool:
        """在树中替换节点"""
        if not isinstance(root, OperatorNode):
            return False

        # 检查直接子节点
        for i, child in enumerate(root.operands):
            if child is target or self._nodes_equal(child, target):
                root.operands[i] = replacement
                return True

        # 递归检查
        for child in root.operands:
            if isinstance(child, OperatorNode):
                if self._replace_in_tree(child, target, replacement):
                    return True
        return False

    def _nodes_equal(self, a: ExprNode, b: ExprNode) -> bool:
        """比较两个节点是否结构等价"""
        return a.to_expression() == b.to_expression()

    def _collect_parametric_nodes(
        self, node: ExprNode, result: List[OperatorNode]
    ):
        """收集所有 parameterized 算子节点"""
        if isinstance(node, OperatorNode) and node.operator in PARAMETRIC_OPERATORS:
            if node.params:
                result.append(node)
        for child in node.children():
            self._collect_parametric_nodes(child, result)


# ── 便捷函数 ──────────────────────────────────────────────

def parse_factor_expression(expression: str) -> Optional[ExprNode]:
    """快速解析因子表达式为树"""
    return FactorExpressionParser.parse(expression)


def dsl_to_pandas_infix(expression: str) -> Optional[str]:
    """
    将 DSL 格式表达式转为 pandas infix 格式。
    如果已经是 pandas infix 格式，返回原表达式。
    
    Examples:
        dsl_to_pandas_infix("sub(ma(overnight, 60), ma(close, 20))")
        → "(overnight.rolling(60).mean() - close.rolling(20).mean())"
        
        dsl_to_pandas_infix("close_p.rolling(20).mean()")
        → "close_p.rolling(20).mean()"  # already pandas infix
    """
    if not expression or not expression.strip():
        return None

    expr = expression.strip()

    # 快速启发式: 如果包含 .rolling( / .shift( / .pct_change( / .mean() 等，可能已是 pandas infix
    pd_markers = ['.rolling(', '.shift(', '.pct_change(', '.diff(', '.mean()', '.std()', '.abs()', '.rank(', '.ewm(']
    if any(m in expr for m in pd_markers):
        # 检查是否混有 DSL 函数调用（如 sub(ma(X,N), ...) 在 pandas 表达式内）
        dsl_markers = ['sub(', 'add(', 'mul(', 'div(', 'ma(', 'ts_mean(', 'ts_delta(', 'rank(', 'zscore(']
        has_dsl = any(expr.startswith(m) or m in expr[:20] for m in dsl_markers)
        if not has_dsl:
            return expr  # already pandas infix

    # 尝试解析并转换
    tree = FactorExpressionParser.parse(expr)
    if tree is None:
        return None

    try:
        pandas_expr = tree.to_pandas_infix()
        return pandas_expr
    except Exception:
        return None


def breed_factors_from_templates(
    templates: List[Dict],
    n_children: int = 10,
    output_format: str = "pandas",  # "dsl" or "pandas"
) -> List[Dict]:
    """快速从模板库育种"""
    breeder = GPBreeder()
    return breeder.breed_from_templates(templates, n_children, output_format=output_format)


# ── 测试 ──────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  Factor Expression Tree + GP Breeder 测试")
    print("=" * 60)

    # 测试解析
    test_exprs = [
        "sub(ma(overnight, 60), ma(close, 20))",
        "rank(ts_delta(volume, 5))",
        "div(sub(ma(overnight, 60), ma(close, 20)), ts_std(amplitude, 30))",
    ]

    for expr in test_exprs:
        tree = FactorExpressionParser.parse(expr)
        if tree:
            print(f"\n  解析: {expr}")
            print(f"  重建: {tree.to_expression()}")
            print(f"  骨架: {tree.to_skeleton()}")
            print(f"  深度: {tree.depth()}, 节点: {tree.node_count()}")
            subtrees = tree.get_all_subtrees()
            print(f"  子树: {len(subtrees)} 个")

    # 测试 GP 育种
    print("\n" + "=" * 60)
    print("  GP 育种测试")
    print("=" * 60)

    templates = [
        {"formula": "sub(ma(overnight, 60), ma(close, 20))"},
        {"formula": "rank(ts_delta(volume, 5))"},
        {"formula": "div(sub(ma(overnight, 60), ma(close, 20)), ts_std(amplitude, 30))"},
    ]

    children = breed_factors_from_templates(templates, n_children=5)
    for c in children:
        print(f"  [{c['source']}] {c['factor_name']}: {c['formula'][:60]}")

    # 测试变异
    print("\n" + "=" * 60)
    print("  变异测试")
    print("=" * 60)
    tree = FactorExpressionParser.parse("sub(ma(overnight, 60), ma(close, 20))")
    breeder = GPBreeder()
    for i in range(3):
        mutated = breeder.mutate(tree)
        if mutated:
            print(f"  变异{i+1}: {tree.to_expression()} → {mutated.to_expression()}")

    # 测试交叉
    print("\n" + "=" * 60)
    print("  交叉测试")
    print("=" * 60)
    t1 = FactorExpressionParser.parse("sub(ma(overnight, 60), ma(close, 20))")
    t2 = FactorExpressionParser.parse("rank(ts_delta(volume, 5))")
    if t1 and t2:
        crossed = breeder.crossover(t1, t2)
        if crossed:
            print(f"  交叉: ({t1.to_expression()}) × ({t2.to_expression()})")
            print(f"  结果: {crossed.to_expression()}")

    # 测试扰动
    print("\n" + "=" * 60)
    print("  参数扰动测试")
    print("=" * 60)
    for i in range(3):
        perturbed = breeder.perturb(tree, momentum_direction=0)
        if perturbed:
            print(f"  扰动{i+1}: {tree.to_expression()} → {perturbed.to_expression()}")
