"""
forge/dimension_rules.py
=========================
P-20260831: Alpha2 论文移植 — 维度一致性预剪枝 (影子 → 对照 → 转正 SOP)

核心思想 (摘自 Alpha2 §4.3.3):
  表达式树的每个子树/中间结果都携带一个"量纲"标签, 算子在其操作数
  量纲不合法时拒绝构造 — 在计算 IC 之前就把无意义的表达式整棵剪掉。

量纲体系 (6 类, 覆盖 forge 原生终端与 S5 全字段):
  PRICE   — 价格类 (元/股):   open/high/low/close/vwap
  VOLUME  — 成交量类 (股):    volume
  AMOUNT  — 金额类 (元):      amount / north_money / hgt / sgt / mv
  DIMLESS — 无量纲量:         returns/overnight/turnover/振幅/估值/财务/rank输出
  SCALAR  — 纯数值常数:       0.5 / 1e-6 / 窗口参数
  UNKNOWN — 未注册字段:       不剪枝 (宁可放过, 不误杀)

规则分级 (与对比分析报告一致):
  HARD (硬剪): add/sub 两侧量纲类别不一致 — 如 close + volume
               (currency + unit 无经济含义, Alpha2 原文示例)
  SOFT (软标记, 不剪): ① 常数平滑/平移 (x + 1e-6 数值稳定惯用法, JQ 冠军
               因子普遍使用, 硬剪会误伤) ② mul/div 产生"可计算但语义
               可疑"的量纲 (price*price, close/volume)

模式 (模块级, 环境变量 FUXI_DIM_PRUNE 覆盖):
  off     — 完全不检查 (v0.7 原行为)
  shadow  — 检查+计数, 不改变任何生成行为 (默认, 影子期)
  enforce — 硬剪生效, 生成器配合重采样 (转正后启用)

用法:
  from forge import dimension_rules as dr
  res = dr.apply_op("add", [dr.PRICE, dr.VOLUME])   # -> DimResult(hard=True)
  aud = dr.audit_forge_node(tree, record=True)      # 审计 forge ExprNode
  aud = dr.audit_fet_node(tree, record=False)       # 审计 gp_breed ExprNode
"""

import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ═══════════════════════════════════════════════════════
# 量纲常量
# ═══════════════════════════════════════════════════════

PRICE = "PRICE"
VOLUME = "VOLUME"
AMOUNT = "AMOUNT"
DIMLESS = "DIMLESS"
SCALAR = "SCALAR"
UNKNOWN = "UNKNOWN"

# ═══════════════════════════════════════════════════════
# 终端量纲注册表
# ═══════════════════════════════════════════════════════

_TERMINAL_DIMS: Dict[str, str] = {}


def register_terminal(name: str, dim: str):
    """注册终端字段的量纲 (供外部字段扩展, 如 S5 新增字段)"""
    _TERMINAL_DIMS[name] = dim
    _TERMINAL_DIMS[name.rstrip("_p")] = dim
    _TERMINAL_DIMS[name + "_p"] = dim


def _register_bulk(dim: str, names: List[str]):
    for n in names:
        register_terminal(n, dim)


# 价格类 (元/股)
_register_bulk(PRICE, ["open", "high", "low", "close", "vwap", "wap",
                       "prev_close", "adj_close"])
# 成交量类 (股/手)
_register_bulk(VOLUME, ["volume", "vol"])
# 金额类 (元)
_register_bulk(AMOUNT, ["amount", "north_money", "south_money", "hgt", "sgt",
                        "mv", "float_mv", "free_mv", "circ_mv", "total_mv"])
# 无量纲: 收益/振幅/换手/估值/财务 (均为比例或倍数)
_register_bulk(DIMLESS, [
    "returns", "ret", "pct", "pct_chg", "overnight", "amplitude",
    "hl_ratio", "turnover", "pe_ttm", "pb", "dv_ratio",
    "roe", "roe_dt", "tr_yoy", "netprofit_yoy", "eps_yoy", "bps",
    "roe_ttm", "gross_margin", "net_margin", "debt_ratio",
])


def dim_of_terminal(name: str) -> str:
    """终端字段 → 量纲; 未注册返回 UNKNOWN (不剪枝, 宁可放过)"""
    return _TERMINAL_DIMS.get(name, _TERMINAL_DIMS.get(name.rstrip("_p"), UNKNOWN))


# ═══════════════════════════════════════════════════════
# 算子规则
# ═══════════════════════════════════════════════════════

@dataclass
class DimResult:
    out: str
    hard: bool = False
    soft: bool = False
    reason: str = ""

    def __repr__(self):
        flag = "HARD" if self.hard else ("soft" if self.soft else "ok")
        return f"DimResult({self.out}, {flag}, {self.reason})"


# 时间序列/窗口类算子: 保持输入量纲
_TS_PRESERVE = {
    "ts_mean", "ts_sum", "ts_min", "ts_max", "ts_std", "ts_delta",
    "ts_delay", "ts_ema_decay", "ma", "ema", "wma", "delta",
    "shift", "rolling", "diff", "demean", "cumsum", "cumprod",
}
# 时间序列类算子: 输出无量纲
_TS_TO_DIMLESS = {
    "ts_pct", "ts_rank", "ts_zscore", "ts_skewness", "ts_kurtosis",
    "ts_corr", "ts_cov", "ts_regression", "ts_decay_linear",
    "pct_change", "roc", "corr", "cov",
}
# 截面标准化算子: 输出无量纲
_CS_TO_DIMLESS = {"rank", "rank_cs", "zscore", "cs_zscore", "scale",
                  "normalize", "ts_rank"}
# 布尔比较算子: 输出无量纲 (0/1)
_BOOL_OPS = {"gt", "lt", "gte", "lte", "eq", "neq"}
# 分支算子: 输出取分支量纲
_BRANCH_OPS = {"where", "if_else"}

_SOFT_SUSPECT = {PRICE, VOLUME, AMOUNT}  # 非无量纲类参与幂/开方/倒数 → 软标记


def apply_op(op: str, child_dims: List[str]) -> DimResult:
    """给一个算子及其子节点量纲, 返回 (输出量纲, 硬违规, 软标记, 原因)"""
    op = (op or "").lower()
    n = len(child_dims)
    d0 = child_dims[0] if n >= 1 else UNKNOWN
    d1 = child_dims[1] if n >= 2 else UNKNOWN

    # ── 时间序列/截面/布尔/分支算子 (按名称优先) ──
    if op in _TS_PRESERVE:
        return DimResult(d0, reason=f"{op} 保持量纲")
    if op in _TS_TO_DIMLESS:
        return DimResult(DIMLESS, reason=f"{op} → 无量纲")
    if op in _CS_TO_DIMLESS:
        return DimResult(DIMLESS, reason=f"{op} → 无量纲(截面标准化)")
    if op in _BOOL_OPS:
        soft = (d0 != d1) and (UNKNOWN not in (d0, d1))
        return DimResult(DIMLESS, soft=soft,
                         reason=("比较两侧量纲不一致" if soft else "布尔→无量纲"))
    if op in _BRANCH_OPS:
        db = child_dims[1] if n >= 2 else UNKNOWN
        dc = child_dims[2] if n >= 3 else UNKNOWN
        soft = (db != dc) and (UNKNOWN not in (db, dc))
        return DimResult(db, soft=soft,
                         reason=("分支量纲不一致" if soft else "分支→取分支量纲"))

    # ── 一元 ──
    if n == 1 or (n >= 1 and op in ("neg", "abs", "sign", "sqrt", "log",
                                    "log1p", "exp", "inv", "square", "cube",
                                    "demean", "max", "min")):
        if op in ("neg", "abs"):
            return DimResult(d0)
        if op == "sign":
            return DimResult(DIMLESS)
        if op in ("log", "log1p", "exp"):
            return DimResult(DIMLESS, reason=f"{op} → 无量纲(log-价格惯例)")
        if op == "sqrt":
            soft = d0 in _SOFT_SUSPECT
            return DimResult(d0, soft=soft,
                             reason=("开方非无量纲量" if soft else "开方保持量纲"))
        if op == "inv":
            if d0 in (DIMLESS, SCALAR):
                return DimResult(d0)
            if d0 in _SOFT_SUSPECT:
                return DimResult(UNKNOWN, soft=True, reason="倒数非无量纲量(1/元)")
            return DimResult(UNKNOWN)
        if op in ("square", "cube"):
            soft = d0 in _SOFT_SUSPECT
            return DimResult(d0, soft=soft,
                             reason=("幂次放大非无量纲量" if soft else "幂次保持量纲"))
        if op == "demean":
            return DimResult(d0)
        if op in ("max", "min") and n == 1:
            return DimResult(d0)

    # ── 二元 ──
    if n >= 2:
        if op in ("add", "sub"):
            if d0 == d1:
                return DimResult(d0)
            if UNKNOWN in (d0, d1):
                return DimResult(UNKNOWN, reason="含未注册量纲, 不剪枝")
            if SCALAR in (d0, d1):
                other = d1 if d0 == SCALAR else d0
                return DimResult(other, soft=True,
                                 reason="常数平滑/平移惯用法 (x±ε)")
            # P-20260831-B 校准: PRICE±DIMLESS 是 JQ 冠军实证的位置类因子
            # 惯用法 (gp_breed_002 动量回调幅度:
            #   high_p - close_p.pct_change(20).shift(20), JQ +143.46%)
            # → 降级为软标记, 禁止硬剪
            if {d0, d1} == {PRICE, DIMLESS}:
                return DimResult(PRICE, soft=True,
                                 reason="价格±收益(位置类因子, JQ冠军实证)")
            # 价格/量/金额三类之间的量纲混配保持硬剪 (Alpha2 原文场景)
            return DimResult(d0, hard=True,
                             reason=f"{op}: {d0} vs {d1} 量纲不一致")
        if op == "mul":
            if UNKNOWN in (d0, d1):
                return DimResult(UNKNOWN)
            if {d0, d1} == {PRICE, VOLUME}:
                return DimResult(AMOUNT, reason="价格×成交量 → 成交额")
            if d0 == SCALAR:
                return DimResult(d1)
            if d1 == SCALAR:
                return DimResult(d0)
            if d0 == DIMLESS:
                return DimResult(d1)
            if d1 == DIMLESS:
                return DimResult(d0)
            if d0 == d1:
                return DimResult(d0, soft=True,
                                 reason=f"{d0}×{d0} 语义可疑(平方项)")
            return DimResult(UNKNOWN, soft=True, reason=f"{d0}×{d1} 语义可疑")
        if op == "div":
            if UNKNOWN in (d0, d1):
                return DimResult(UNKNOWN)
            if d1 == SCALAR:
                return DimResult(d0)
            if d0 == d1:
                return DimResult(DIMLESS, reason="同量纲相除 → 无量纲")
            if d0 == SCALAR:
                return DimResult(UNKNOWN, soft=True, reason=f"常数/{d1} 语义可疑")
            if d1 == DIMLESS:
                return DimResult(d0)
            if (d0, d1) == (AMOUNT, VOLUME):
                return DimResult(PRICE, reason="金额/成交量 → 价格")
            if (d0, d1) == (AMOUNT, PRICE):
                return DimResult(VOLUME, reason="金额/价格 → 成交量")
            return DimResult(UNKNOWN, soft=True, reason=f"{d0}/{d1} 语义可疑")
        if op == "pow":
            if d1 in (SCALAR, DIMLESS):  # 指数是常数/无量纲
                soft = d0 in _SOFT_SUSPECT
                return DimResult(d0, soft=soft,
                                 reason=("非无量纲幂次" if soft else "幂次保持量纲"))
            return DimResult(UNKNOWN)
        if op in ("max", "min"):
            soft = (d0 != d1) and (UNKNOWN not in (d0, d1))
            return DimResult(d0, soft=soft,
                             reason=("max/min 两侧量纲不一致" if soft else ""))
        if op == "clip":
            return DimResult(d0)

    return DimResult(UNKNOWN)


# ═══════════════════════════════════════════════════════
# 模式控制与计数器
# ═══════════════════════════════════════════════════════

_MODE = None  # lazy: 首次访问读环境变量


def get_mode() -> str:
    """off | shadow | enforce; 环境变量 FUXI_DIM_PRUNE 覆盖"""
    global _MODE
    if _MODE is None:
        m = os.environ.get("FUXI_DIM_PRUNE", "shadow").strip().lower()
        _MODE = m if m in ("off", "shadow", "enforce") else "shadow"
    return _MODE


def set_mode(mode: str):
    """显式设置模式 (测试/脚本用)"""
    global _MODE
    mode = (mode or "").strip().lower()
    if mode not in ("off", "shadow", "enforce"):
        raise ValueError(f"mode must be off/shadow/enforce, got {mode}")
    _MODE = mode


class _Counters:
    """影子期统计: 硬违规/软标记按 (算子, 量纲对) 聚合"""

    def __init__(self):
        self.hard: Dict[Tuple, int] = {}
        self.soft: Dict[Tuple, int] = {}
        self.total_ops = 0

    def record(self, op: str, dims: Tuple[str, ...], hard: bool, soft: bool):
        self.total_ops += 1
        key = (op, dims)
        if hard:
            self.hard[key] = self.hard.get(key, 0) + 1
        elif soft:
            self.soft[key] = self.soft.get(key, 0) + 1

    def snapshot(self) -> Dict:
        return {
            "n_flag_events": self.total_ops,
            "n_hard_events": sum(self.hard.values()),
            "n_soft_events": sum(self.soft.values()),
            "hard_by_op": {f"{k[0]}{list(k[1])}": v
                           for k, v in
                           sorted(self.hard.items(), key=lambda kv: -kv[1])},
            "soft_by_op": {f"{k[0]}{list(k[1])}": v
                           for k, v in
                           sorted(self.soft.items(), key=lambda kv: -kv[1])},
        }

    def reset(self):
        self.hard.clear()
        self.soft.clear()
        self.total_ops = 0


COUNTERS = _Counters()


def reset_counters():
    COUNTERS.reset()


# ═══════════════════════════════════════════════════════
# 审计结果
# ═══════════════════════════════════════════════════════

@dataclass
class AuditResult:
    root_dim: str = UNKNOWN
    hard_violations: List[Tuple[str, str]] = field(default_factory=list)
    soft_marks: List[Tuple[str, str]] = field(default_factory=list)

    @property
    def n_hard(self) -> int:
        return len(self.hard_violations)

    @property
    def n_soft(self) -> int:
        return len(self.soft_marks)

    @property
    def clean(self) -> bool:
        return self.n_hard == 0


# ═══════════════════════════════════════════════════════
# 审计器 (双树适配: forge ExprNode / gp_breed ExprNode)
# ═══════════════════════════════════════════════════════

def audit_forge_node(node, record: bool = False) -> Optional[AuditResult]:
    """审计 forge ExprNode (duck-typed: .is_leaf / .primitive / .children)"""
    try:
        res = AuditResult()
        res.root_dim = _walk_forge(node, res, record, node.to_string())
        return res
    except Exception:
        return None


def _walk_forge(node, res: AuditResult, record: bool, path: str) -> str:
    if node.is_leaf:
        name = node.primitive.name
        if node.primitive.is_input:
            return dim_of_terminal(name)
        try:
            float(name)
            return SCALAR  # 数字常量
        except ValueError:
            return dim_of_terminal(name)

    op = node.primitive.name
    children = node.children

    if op.startswith("ts_"):
        d0 = _walk_forge(children[0], res, record, f"{path}/{op}")
        r = apply_op(op, [d0, SCALAR])  # 窗口参数=常数
        _note(res, record, op, (d0, SCALAR), r, path)
        return r.out

    dims = [_walk_forge(c, res, record, f"{path}/{op}") for c in children]
    r = apply_op(op, dims)
    _note(res, record, op, tuple(dims), r, path)
    return r.out


def audit_fet_node(node, record: bool = False) -> Optional[AuditResult]:
    """审计 gp_breed (factor_expression_tree) ExprNode — 延迟导入避免循环依赖"""
    try:
        from factor_expression_tree import FieldNode, ConstantNode, OperatorNode
    except ImportError:
        return None

    res = AuditResult()

    def walk(n, path: str) -> str:
        if isinstance(n, FieldNode):
            return dim_of_terminal(n.name)
        if isinstance(n, ConstantNode):
            return SCALAR
        if isinstance(n, OperatorNode):
            op = n.operator
            if op.startswith("ts_") or op in _TS_PRESERVE | _TS_TO_DIMLESS:
                dims = [walk(o, f"{path}/{op}") for o in n.operands]
                d0 = dims[0] if dims else UNKNOWN
                r = apply_op(op, [d0, SCALAR])
                _note(res, record, op, (d0, SCALAR), r, path)
                return r.out
            dims = [walk(o, f"{path}/{op}") for o in n.operands]
            r = apply_op(op, dims)
            _note(res, record, op, tuple(dims), r, path)
            return r.out
        return UNKNOWN

    try:
        res.root_dim = walk(node, "root")
        return res
    except Exception:
        return None


def _note(res: AuditResult, record: bool, op: str, dims: Tuple[str, ...],
          r: DimResult, path: str):
    if r.hard:
        res.hard_violations.append((path, r.reason))
    elif r.soft:
        res.soft_marks.append((path, r.reason))
    if record:
        COUNTERS.record(op, dims, r.hard, r.soft)


def audit_expression_string(expr: str, record: bool = False) -> Optional[AuditResult]:
    """审计任意公式字符串: 优先 gp_breed 解析器, 失败则 forge 解析器"""
    if not expr or not expr.strip():
        return None
    expr = expr.strip()
    # 优先 fet 解析器 (字段宇宙更大)
    try:
        from factor_expression_tree import parse_factor_expression
        tree = parse_factor_expression(expr)
        if tree is not None:
            return audit_fet_node(tree, record=record)
    except ImportError:
        pass
    # fallback: forge 解析器
    try:
        from forge.expression import parse_expression
        tree = parse_expression(expr)
        if tree is not None:
            return audit_forge_node(tree, record=record)
    except ImportError:
        pass
    return None


def dim_of_node(node) -> str:
    """快速取子树根量纲 (无审计记录, 生成期用; 不识别返回 UNKNOWN)"""
    res = audit_forge_node(node, record=False)
    return res.root_dim if res else UNKNOWN
