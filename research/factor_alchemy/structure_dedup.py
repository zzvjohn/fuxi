# -*- coding: utf-8 -*-
"""
AST 结构去重 (Structure-level Deduplication) — P-20260915-005
=============================================================

对标 QuantMind (qusong0627/QuantMind) / RD-Agent CoSTEER 的
`alphaagent/scenarios/qlib/regulator/factor_regulator.py` 口径。
2026-09-15 已完成**源码级复刻** (含 `factor_ast.py` 的节点计数约定)。

────────────────────────────────────────────────────────────
一、与现有去重层的分工 (互补, 不替代)
────────────────────────────────────────────────────────────
| 层            | 模块                        | 判据                    | 覆盖面                     |
|---------------|-----------------------------|-------------------------|----------------------------|
| 数值级        | library_orthogonality.py    | 因子值相关性 / 正交     | 值域相似的因子             |
| 结构级(全骨架)| subtree_fingerprinter.py    | 正则抽象骨架 **全等**   | 骨架完全相同的因子         |
| 结构级(部分)  | **本模块**                  | AST **子树包含+节点计数**| 骨架不同但共享一大块子树   |

FSA 的盲区 (2026-09-12 周对比已定位): `share` 模式阈值不可达
(max count=3 vs 需 ≥40 ⇒ 200/200 轮 `forbidden_skeletons=0`)，且只做
**全骨架精确等值** ⇒ 漏掉「整体骨架不同、但共享一大块子树」的因子。
本模块补这一层，直接呼应 S5 干旱归因里的「去重覆盖面」结论。

────────────────────────────────────────────────────────────
二、QuantMind 原口径 (逐字复刻)
────────────────────────────────────────────────────────────
`FactorRegulator.evaluate()` 的 `duplicated_subtree_size` = 候选与库内因子
**最大公共子树 (Largest Common Subtree) 的节点总数**
(实现要求两子树**同尺寸且结构相等**; 可交换算子允许左右交换)。

准入三条件 (`is_expression_acceptable`):
    cond1 = duplicated_subtree_size <= duplication_threshold   # 默认 8
    cond2 = -log(1 - free_args_ratio)   < 0.693   ⇔  free_args_ratio   < 0.5
    cond3 = -log(1 - unique_vars_ratio) < 0.693   ⇔  unique_vars_ratio < 0.5
    (另有 free_args_ratio >= 1 或 unique_vars_ratio >= 1 直接判 False)

⚠️ 计数语义**与函数名直觉相反，以源码为准** (`factor_ast.py`):
    count_free_args   = **数值常量 (NumberNode) 的个数** —— 不是"自由变量"
    count_unique_vars = **不同的数据变量名个数** (原实现仅统计 `$` 前缀者)
    count_all_nodes   = 全节点数

节点计数约定 (本模块完全复刻; 每节点计 1):
    var / num 叶子   = 1
    fn (函数调用)    = 1 + Σ 子节点
    op (二元运算)    = 1 + 左 + 右
    neg (一元负号)   = 1 + 操作数
    ite (条件表达式) = 1 + cond + true + false      (对应 ConditionalNode)
    attr / sub       = 1 + Σ 子节点

🔴 **阈值口径差异 (必须知情, 已写入报告 `parity_note`)**:
    QuantMind 函数式 `TS_MEAN($close, 20)` = 3 节点。
    伏羲库多为 pandas 方法链 `close.rolling(20).mean()` = 4 节点
    (方法接收者多一层) ⇒ 同结构 pandas 形式天然多 ~1 节点/rolling 层。
    ⇒ 本模块**不盲抄 8**: 先给库内实测分布 (`max_common_size_distribution`)，
      `DEFAULT_THRESHOLD` 按实测标定, 可用 env 覆盖。
      另**不做** rolling 链扁平化 —— 扁平化会让
      `close.rolling(20).mean()` 与 `close.rolling(20).std()` 的公共子树
      从 3 掉到 1, 丢掉的正是本模块要抓的信号。

────────────────────────────────────────────────────────────
三、🔴 符号规范化 (2026-09-12 硬屏蔽审计血泪, 必读)
────────────────────────────────────────────────────────────
伏羲历史上 51 条硬禁条目中 **13/26 组 (50%) 是方向特定的**。铁证:
`lhb_flag.rolling(20).sum()` 被禁 (**正向**)，但同条注记与 MEMORY 都写
「**取负版 JQ +64.59%**」。⇒ 结构去重若不做符号处理，会**封杀已验证赚钱的反向因子**。

本模块规则 (四条):
    1. 递归化简 `-(-x)` → `x` (奇偶消去)
    2. 只剥**最外层**符号: `sign ∈ {+1,-1}` 单独记录, **不入结构签名**
       (内部 `neg` 保留 —— `a - b` 与 `b - a` 仍是不同结构)
    3. 判重 = **结构等价 AND 符号相同** (`sign_same=True`) 才算数
    4. `sign_same=False` 的「结构镜像对」照常输出，但默认**不判重**，供人工复核
另: 变量别名统一 (`close_p`→`close`, `volume_p`→`volume`, 大小写归一等)。

────────────────────────────────────────────────────────────
四、抽象层级 (三级, 全跑, 分别读数)
────────────────────────────────────────────────────────────
    literal : 不抽象 (`close` ≠ `volume`) —— QuantMind 原口径
    field   : 变量 → `$F` (同算子结构、不同字段) —— 补 FSA 的部分骨架层
    full    : 变量 → `$F`, 数值 → `$N` (同结构、不同字段与窗口) —— 最激进

────────────────────────────────────────────────────────────
五、开关与产物
────────────────────────────────────────────────────────────
    FUXI_STRUCT_DEDUP             = off | shadow (默认) | enforce
    FUXI_STRUCT_DEDUP_THRESHOLD   = int, 最大公共子树节点数阈值
    FUXI_STRUCT_DEDUP_ABSTRACTION = literal (默认) | field | full  (仅 enforce 生效)
    FUXI_STRUCT_DEDUP_MIN_SIZE    = 入索引的最小子树节点数 (默认 3, 挡平凡签名)
    FUXI_STRUCT_DEDUP_MAX_FANOUT  = 单签名扇出保护 (默认 400, 防 O(n^2) 爆炸)

产物: data/structure_dedup_report.json
读数行: [StructDedup:shadow] 库 N 因子 | 阈值 T: 字面 X / 字段 Y / 全抽象 Z 对 | ...

回滚: 删除本文件 + 撤销 ralph_loop 钩子即完全移除; shadow 模式零裁决。
"""

from __future__ import annotations

import ast as _ast
import copy
import json
import math
import os
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

# ── 默认路径 ──────────────────────────────────────────────────
DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
LIB_STATE_PATH = DATA_DIR / "library_orthogonality_state.json"
CORR_MATRIX_PATH = DATA_DIR / "library_corr_matrix.npz"
REPORT_PATH = DATA_DIR / "structure_dedup_report.json"

# ── 阈值默认值 ────────────────────────────────────────────────
# QuantMind 原值 = 8 (函数式语法口径)。见 QUANTMIND_PARITY_THRESHOLD。
QUANTMIND_PARITY_THRESHOLD = 8

# 🔴 实测标定 (2026-09-15, 923 因子全库, 见 data/structure_dedup_report.json):
#   两个增量: ①pandas 方法链比函数式多 ~1 节点/rolling 层
#             ②阈值 8 实际命中的是「共用样板」，不是「共用判别性核心」
#   size 8 典型命中 (低精度/应放行):
#     op:add(fn:std(fn:rolling(op:sub(buy_lg_vol,sell_lg_vol),n:60)),n:1e-08)
#     —— 只是共用分母归一化样板; 两条因子的判别性分子 (rolling(10) vs rolling(20)) 不同
#   size 10 典型命中 (高精度/应拦):
#     op:lt(fn:abs(op:sub(fn:mod(close,1),fn:round(fn:mod(close,1)))),n:0.05)
#     —— 共用判别性核心谓词, 仅窗口不同 (rolling(20) vs rolling(5)) = 真近重复
#   ⇒ 标定为 10 (literal 档命中 1366 对; 8 档 2147 对含大量样板噪声)
DEFAULT_THRESHOLD = 10
DEFAULT_MIN_SIZE = 3
DEFAULT_MAX_FANOUT = 400

# ── 可交换算子 (子树等价判定允许左右交换) ─────────────────────
COMMUTATIVE_OPS: Set[str] = {
    "add", "mul", "eq", "ne", "and", "or",
    "maximum", "minimum", "min", "max",
}

# ── 变量别名统一 (保守: 只统一同义写法, 不做语义猜测) ─────────
VAR_ALIAS: Dict[str, str] = {
    "open_p": "open", "close_p": "close", "high_p": "high",
    "low_p": "low", "volume_p": "volume", "amount_p": "amount",
    "open_": "open", "close_": "close", "high_": "high",
    "low_": "low", "volume_": "volume",
    # forge DSL 旧写法
    "asset": "close",
}

# ── 模块名 (不计入数据变量, 且作为函数前缀被剥离) ─────────────
MODULE_NAMES: Set[str] = {"np", "numpy", "pd", "pandas", "math", "scipy"}

# ── 模块常量 (np.nan / np.inf / ... → 常量叶子, 不是数据字段) ──
MODULE_CONSTANTS: Set[str] = {"nan", "inf", "pi", "e", "infinity", "newaxis"}

# ── 内建类型名 (astype(float) / float(x) → 类型令牌, 不是数据字段) ──
BUILTIN_TYPE_NAMES: Set[str] = {"float", "int", "str", "bool", "object", "list", "dict"}

# ── 解析失败时的容错: 单目/双目函数名 → 算子符号 (仅用于把中缀变体归一) ──
_ABST_KEYS = ("literal", "field", "full")


# ═══════════════════════════════════════════════════════════════
# 一、规范化 CNode 树 (自 Python AST 构建)
# ═══════════════════════════════════════════════════════════════

class CNode:
    """规范化结构树节点。kind ∈ {var, num, fn, op, neg, ite, attr, sub}。"""
    __slots__ = ("kind", "label", "children")

    def __init__(self, kind: str, label: str = "", children: Optional[List["CNode"]] = None):
        self.kind = kind
        self.label = label
        self.children: List[CNode] = children or []

    def __repr__(self) -> str:  # pragma: no cover - debug only
        return f"CNode({self.kind}:{self.label}, n={len(self.children)})"


def _fmt_num(v) -> str:
    """数值常量归一: 20 / 20.0 / 2e1 → 同一字符串。"""
    try:
        f = float(v)
    except Exception:
        return str(v)
    if f == int(f) and abs(f) < 1e15:
        return str(int(f))
    return repr(f)


def _collect_local_names(node) -> Set[str]:
    """收集**非数据**的局部名: lambda 参数 + 推导式目标 + 赋值目标。

    这些名字是作用域局部绑定, 不是数据字段 ⇒ 不计入 unique_vars
    (对标 QuantMind 只统计 `$` 前缀变量的过滤)。
    """
    out: Set[str] = set()
    for n in _ast.walk(node):
        if isinstance(n, _ast.Lambda):
            a = n.args
            for arg in list(a.args) + list(a.posonlyargs) + list(a.kwonlyargs):
                out.add(arg.arg.lower())
            if a.vararg:
                out.add(a.vararg.arg.lower())
            if a.kwarg:
                out.add(a.kwarg.arg.lower())
        elif isinstance(n, (_ast.ListComp, _ast.SetComp, _ast.DictComp, _ast.GeneratorExp)):
            for gen in n.generators:
                for t in _ast.walk(gen.target):
                    if isinstance(t, _ast.Name):
                        out.add(t.id.lower())
    return out


def _canon(node, locals_: frozenset = frozenset()) -> CNode:
    """Python ast 节点 → CNode (递归)。无法归类的语法降级为 'sub' 而非抛错。

    `locals_` = 非数据局部名集合 (lambda 参数 / 推导式目标), 归为常量叶子。
    """
    if isinstance(node, _ast.Expression):
        return _canon(node.body, locals_)

    if isinstance(node, _ast.Constant):
        v = node.value
        if isinstance(v, bool):
            return CNode("num", "bool")
        if isinstance(v, (int, float)):
            return CNode("num", _fmt_num(v))
        return CNode("num", "str")          # 字符串字面量 → 常量叶子

    if isinstance(node, _ast.Name):
        name = node.id.lower()
        if name in locals_:
            return CNode("num", "local")     # lambda 参数 / 推导式变量
        if name in BUILTIN_TYPE_NAMES:
            return CNode("num", "type")      # astype(float/int/...) 的类型名
        if name in MODULE_NAMES:
            return CNode("num", "module")    # 裸模块引用
        return CNode("var", VAR_ALIAS.get(name, name))

    if isinstance(node, _ast.Call):
        args = [_canon(a, locals_) for a in node.args]
        kws = [_canon(k.value, locals_) for k in node.keywords]
        f = node.func
        if isinstance(f, _ast.Attribute):
            base = f.value
            # 模块前缀剥离: np.log(x) → fn(np.log, [x]), 不让 'np' 变成数据变量
            if isinstance(base, _ast.Name) and base.id.lower() in MODULE_NAMES:
                return CNode("fn", f"{base.id.lower()}.{f.attr.lower()}", args + kws)
            return CNode("fn", f.attr.lower(), [_canon(base, locals_)] + args + kws)
        if isinstance(f, _ast.Name):
            nm = f.id.lower()
            if nm in BUILTIN_TYPE_NAMES:               # float(x) / int(x)
                return CNode("num", "type")
            if nm in MODULE_NAMES:                     # 罕见: 直接调用模块
                return CNode("fn", nm, args + kws)
            return CNode("fn", nm, args + kws)
        return CNode("fn", "?call", args + kws)

    if isinstance(node, _ast.BinOp):
        op_name = type(node.op).__name__.lower()
        op_map = {
            "add": "add", "sub": "sub", "mult": "mul", "div": "div",
            "mod": "mod", "pow": "pow", "floordiv": "floordiv",
            "matmult": "matmul", "bitand": "bitand", "bitor": "bitor",
            "bitxor": "bitxor", "lshift": "lshift", "rshift": "rshift",
        }
        return CNode("op", op_map.get(op_name, op_name),
                     [_canon(node.left, locals_), _canon(node.right, locals_)])

    if isinstance(node, _ast.UnaryOp):
        op_name = type(node.op).__name__.lower()
        op_map = {"usub": "neg", "uadd": "uadd", "not": "not", "invert": "invert"}
        return CNode(op_map.get(op_name, op_name), "", [_canon(node.operand, locals_)])

    if isinstance(node, _ast.BoolOp):
        return CNode("op", type(node.op).__name__.lower(),
                     [_canon(v, locals_) for v in node.values])

    if isinstance(node, _ast.Compare):
        # 链式比较降级为左结合二元链, 保证节点数可加
        left = _canon(node.left, locals_)
        out = None
        for op_node, comp in zip(node.ops, node.comparators):
            nm = type(op_node).__name__.lower()
            cur = CNode("op", nm, [left, _canon(comp, locals_)])
            out = cur if out is None else CNode("op", "and", [out, cur])
            left = _canon(comp, locals_)
        return out or CNode("num", "bool")

    if isinstance(node, _ast.IfExp):
        return CNode("ite", "", [_canon(node.test, locals_), _canon(node.body, locals_),
                                 _canon(node.orelse, locals_)])

    if isinstance(node, _ast.Attribute):
        # 模块常量: np.nan / np.inf / np.pi / np.e → 常量叶子 (不是数据字段)
        if isinstance(node.value, _ast.Name) and node.value.id.lower() in MODULE_NAMES \
                and node.attr.lower() in MODULE_CONSTANTS:
            return CNode("num", "const")
        return CNode("attr", node.attr.lower(), [_canon(node.value, locals_)])

    if isinstance(node, _ast.Subscript):
        return CNode("sub", "", [_canon(node.value, locals_), _canon(node.slice, locals_)])

    if isinstance(node, (_ast.List, _ast.Tuple)):
        return CNode("op", "seq", [_canon(e, locals_) for e in node.elts])

    if isinstance(node, _ast.Lambda):
        # 参数名已入 locals_; 只保留函数体结构 + 参数个数
        kids = [_canon(node.body, locals_), CNode("num", str(len(node.args.args)))]
        return CNode("op", "lambda", kids)

    if isinstance(node, (_ast.ListComp, _ast.SetComp, _ast.GeneratorExp)):
        kids = [_canon(node.elt, locals_),
                _canon(node.generators[0].iter, locals_) if node.generators else CNode("num", "none")]
        return CNode("op", "comprehension", kids)

    if isinstance(node, _ast.DictComp):
        return CNode("op", "dictcomp",
                     [_canon(node.key, locals_), _canon(node.value, locals_)])

    if isinstance(node, _ast.Slice):
        kids = []
        for part in (node.lower, node.upper, node.step):
            kids.append(_canon(part, locals_) if part is not None else CNode("num", "none"))
        return CNode("sub", "slice", kids)

    return CNode("op", f"?{type(node).__name__.lower()}", [])


def _simplify_neg(n: CNode) -> CNode:
    """递归化简: `-(-x)` → `x` (奇偶消去)。其余内部 neg 保留。"""
    kids = [_simplify_neg(c) for c in n.children]
    cur = CNode(n.kind, n.label, kids)
    if cur.kind == "neg" and len(cur.children) == 1:
        inner = cur.children[0]
        if inner.kind == "neg" and len(inner.children) == 1:
            return inner.children[0]
    # uadd 无结构意义 → 剥掉
    if cur.kind == "uadd" and len(cur.children) == 1:
        return cur.children[0]
    return cur


def _split_sign(n: CNode) -> Tuple[int, CNode]:
    """剥离最外层符号链 → (sign, body)。内部 neg 一律保留。"""
    sign = 1
    while n.kind == "neg" and len(n.children) == 1:
        sign = -sign
        n = n.children[0]
    return sign, n


def node_count(n: CNode) -> int:
    """复刻 QuantMind `count_nodes`: 每节点计 1, 内部节点 = 1 + Σ 子节点。"""
    return 1 + sum(node_count(c) for c in n.children)


# ═══════════════════════════════════════════════════════════════
# 一.5、多行代码块 → 单表达式 (库内 ~15% 因子是多行块, 必须支持)
# ═══════════════════════════════════════════════════════════════

class _EnvInliner(_ast.NodeTransformer):
    """把局部变量名替换为其定义式 (带环检测 + 深度上限)。

    在 **Python AST 层**做内联 (而非 CNode 往返), 以保住 lambda / 属性链 /
    下标等 CNode 无法无损还原的语法。
    """

    def __init__(self, env: Dict[str, _ast.AST], seen: frozenset = frozenset(), depth: int = 0):
        self.env = env
        self.seen = seen
        self.depth = depth

    def visit_Name(self, node: _ast.Name):
        key = node.id.lower()
        if key in self.env and key not in self.seen and self.depth < 12:
            sub = copy.deepcopy(self.env[key])
            return _EnvInliner(self.env, self.seen | {key}, self.depth + 1).visit(sub)
        return node


def _collect_module_env(mod: _ast.Module) -> Tuple[Optional[_ast.AST], Dict[str, _ast.AST]]:
    """**顺序**收集顶层简单赋值 → env; 返回 (结果表达式 AST, env)。

    🔴 顺序语义: 每条赋值的 RHS 用**赋值前**的 env 解析。
    否则 `result = ret.std()` 后 `result = pd.DataFrame({c: result.values ...})`
    会把第二条自己的值内联进自身 (自我覆盖 → 结构爆炸)。
    返回的 `result` 已完全解析, 调用方无需再内联。
    """
    env: Dict[str, _ast.AST] = {}
    result = None
    last_value = None

    def _resolve(value: _ast.AST) -> _ast.AST:
        """用当前 env 解析 (env 内条目均已完全解析, 故一趟即收敛)。"""
        try:
            out = _EnvInliner(env).visit(copy.deepcopy(value))
            _ast.fix_missing_locations(out)
            return out
        except Exception:
            return value

    for st in mod.body:
        if isinstance(st, _ast.Assign):
            # 元组解包: w, mp = 60, 30
            if len(st.targets) == 1 and isinstance(st.targets[0], (_ast.Tuple, _ast.List)):
                tgts = st.targets[0].elts
                if isinstance(st.value, (_ast.Tuple, _ast.List)) and len(st.value.elts) == len(tgts):
                    for t, v in zip(tgts, st.value.elts):
                        if isinstance(t, _ast.Name):
                            env[t.id.lower()] = _resolve(v)
                    continue
            resolved = _resolve(st.value)
            for t in st.targets:
                if isinstance(t, _ast.Name):
                    env[t.id.lower()] = resolved
                    if t.id.lower() == "result":
                        result = resolved
            last_value = resolved
        elif isinstance(st, _ast.AnnAssign) and isinstance(st.target, _ast.Name):
            resolved = _resolve(st.value)
            env[st.target.id.lower()] = resolved
            if st.target.id.lower() == "result":
                result = resolved
            last_value = resolved

    if result is None:
        result = last_value
    if result is None:
        for st in reversed(mod.body):
            if isinstance(st, _ast.Expr):
                result = _resolve(st.value)
                break
    return result, env


def extract_expression(expr: str) -> str:
    """单行表达式原样返回; 多行代码块 → 内联中间变量后的单表达式文本。

    例: 'ret = close.pct_change().fillna(0)\\nresult = ret.rolling(20).std()'
        → 'close.pct_change().fillna(0).rolling(20).std()'
    """
    s = (expr or "").strip()
    if not s or "\n" not in s:
        return s
    try:
        _ast.parse(s, mode="eval")
        return s                       # 单行但语法上是表达式
    except Exception:
        pass
    try:
        mod = _ast.parse(s)
    except Exception:
        return s
    result, _env = _collect_module_env(mod)
    if result is None:
        return s
    try:
        return _ast.unparse(result)
    except Exception:
        return s


# ═══════════════════════════════════════════════════════════════
# 二、签名与子树抽取
# ═══════════════════════════════════════════════════════════════

def _sig(n: CNode, abstraction: str) -> str:
    """结构签名。abstraction ∈ {literal, field, full}。"""
    if n.kind == "var":
        return "$F" if abstraction in ("field", "full") else f"v:{n.label}"
    if n.kind == "num":
        if abstraction == "full":
            return "$N"
        return f"n:{n.label}" if n.label not in ("bool", "str", "none") else f"n:{n.label}"

    kids = [_sig(c, abstraction) for c in n.children]
    if n.kind == "op" and n.label in COMMUTATIVE_OPS:
        kids = sorted(kids)          # 可交换算子: 左右交换视为等价
    return f"{n.kind}:{n.label}({','.join(kids)})"


def _collect_subtrees(n: CNode, abstraction: str, out: Dict[str, int]) -> None:
    """收集全部子树: {签名 → 节点数}。size 由结构唯一确定。"""
    out[_sig(n, abstraction)] = node_count(n)
    for c in n.children:
        _collect_subtrees(c, abstraction, out)


def _count_number_nodes(n: CNode) -> int:
    """复刻 `count_number_nodes` = 数值常量个数 (= QuantMind 的 free_args)。"""
    if n.kind == "num":
        return 1
    return sum(_count_number_nodes(c) for c in n.children)


def _collect_var_names(n: CNode, acc: Set[str]) -> None:
    """复刻 `collect_unique_vars` = 不同数据变量名 (已剥离模块名/函数名)。"""
    if n.kind == "var":
        acc.add(n.label)
        return
    if n.kind == "num":
        return
    for c in n.children:
        _collect_var_names(c, acc)


# ═══════════════════════════════════════════════════════════════
# 三、主类
# ═══════════════════════════════════════════════════════════════

@dataclass
class ParsedFactor:
    idx: int
    name: str
    expression: str
    body: Optional[CNode]
    sign: int
    n_nodes: int = 0
    n_body_nodes: int = 0          # 剥离外层符号后的 body 节点数 (比值分母口径)
    n_free_args: int = 0
    n_unique_vars: int = 0
    is_block: bool = False
    src: str = ""
    subtrees: Dict[str, Dict[str, int]] = field(default_factory=dict)

    @property
    def _ratio_denom(self) -> int:
        """比值分母 = body 节点数 (符号归一化后), 保证 X 与 -X 比值一致。"""
        return self.n_body_nodes or self.n_nodes

    @property
    def free_args_ratio(self) -> float:
        d = self._ratio_denom
        return self.n_free_args / d if d else 0.0

    @property
    def unique_vars_ratio(self) -> float:
        d = self._ratio_denom
        return self.n_unique_vars / d if d else 0.0


class ASTStructureDeductor:
    """AST 级结构去重器 (shadow 优先, 零裁决)。"""

    def __init__(
        self,
        threshold: Optional[int] = None,
        abstraction: Optional[str] = None,
        min_size: Optional[int] = None,
        max_fanout: Optional[int] = None,
    ):
        self.mode = (os.environ.get("FUXI_STRUCT_DEDUP", "shadow") or "shadow").strip().lower()
        if self.mode not in ("off", "shadow", "enforce"):
            self.mode = "shadow"

        self.threshold = int(
            threshold if threshold is not None
            else os.environ.get("FUXI_STRUCT_DEDUP_THRESHOLD", DEFAULT_THRESHOLD)
        )
        self.abstraction = (
            abstraction or os.environ.get("FUXI_STRUCT_DEDUP_ABSTRACTION", "literal")
        ).strip().lower()
        if self.abstraction not in _ABST_KEYS:
            self.abstraction = "literal"

        self.min_size = int(
            min_size if min_size is not None
            else os.environ.get("FUXI_STRUCT_DEDUP_MIN_SIZE", DEFAULT_MIN_SIZE)
        )
        self.max_fanout = int(
            max_fanout if max_fanout is not None
            else os.environ.get("FUXI_STRUCT_DEDUP_MAX_FANOUT", DEFAULT_MAX_FANOUT)
        )

        self.factors: List[ParsedFactor] = []
        self.notes: List[str] = []
        # 索引: abstraction → {签名 → [(factor_idx, size)]}
        self._index: Dict[str, Dict[str, List[Tuple[int, int]]]] = {k: defaultdict(list) for k in _ABST_KEYS}

    # ── 解析 ──────────────────────────────────────────────

    @staticmethod
    def normalize_expression(expr: str) -> str:
        """变量别名统一 + 去首尾空白。不改变结构。"""
        s = (expr or "").strip()
        if not s:
            return s
        for alias, canonical in VAR_ALIAS.items():
            s = re.sub(rf"\b{re.escape(alias)}\b", canonical, s)
        return s

    def parse_one(self, expr: str, idx: int = 0, name: str = "") -> Optional[ParsedFactor]:
        """解析单条表达式 → ParsedFactor (失败返回 None)。

        支持两种输入: ①单行表达式 ②多行代码块 (内联中间变量后抽取 result)。
        """
        if not expr or not expr.strip():
            return None
        src = extract_expression(expr)
        if not src:
            return None
        try:
            tree = _ast.parse(self.normalize_expression(src), mode="eval")
        except Exception:
            return None
        try:
            _locals = frozenset(_collect_local_names(tree))
            body = _simplify_neg(_canon(tree, _locals))
            sign, body = _split_sign(body)
        except Exception:
            return None

        pf = ParsedFactor(idx=idx, name=name, expression=expr, body=body, sign=sign)
        pf.is_block = ("\n" in expr.strip())
        pf.src = src
        pf.n_nodes = node_count(body) + (1 if sign == -1 else 0)
        pf.n_body_nodes = node_count(body)
        pf.n_free_args = _count_number_nodes(body)
        _vars: Set[str] = set()
        _collect_var_names(body, _vars)
        pf.n_unique_vars = len(_vars)
        for ab in _ABST_KEYS:
            d: Dict[str, int] = {}
            _collect_subtrees(body, ab, d)
            pf.subtrees[ab] = d
        return pf

    # ── 建索引 ────────────────────────────────────────────

    def index_library(self, factors: Sequence[Dict]) -> Dict[str, int]:
        """把因子库灌入索引。factors 需含 expression/formula + name。"""
        self.factors = []
        n_fail = 0
        for i, f in enumerate(factors):
            expr = f.get("expression") or f.get("formula") or ""
            name = f.get("name") or f.get("factor_name") or f"f{i}"
            pf = self.parse_one(expr, idx=i, name=name)
            if pf is None:
                n_fail += 1
                continue
            pf.idx = len(self.factors)
            self.factors.append(pf)

        self._index = {k: defaultdict(list) for k in _ABST_KEYS}
        for pf in self.factors:
            for ab in _ABST_KEYS:
                for sig, size in pf.subtrees[ab].items():
                    if size >= self.min_size:
                        self._index[ab][sig].append((pf.idx, size))
        return {
            "n_input": len(factors),
            "n_parsed": len(self.factors),
            "n_parse_fail": n_fail,
            "n_sig": {ab: len(self._index[ab]) for ab in _ABST_KEYS},
        }

    # ── 候选检查 (生成端/入库端用) ────────────────────────

    def check_candidate(self, expression: str) -> Dict:
        """候选 vs 库: 返回最大公共子树读数 (shadow) 或裁决 (enforce)。"""
        pf = self.parse_one(expression, idx=-1, name="<candidate>")
        if pf is None:
            return {"ok": False, "reason": "parse_failed", "expression": expression}

        per_ab: Dict[str, Dict] = {}
        for ab in _ABST_KEYS:
            best_size, best_factor, best_sig, sign_same = 0, None, None, True
            for sig, size in pf.subtrees[ab].items():
                if size < self.min_size or size <= best_size:
                    continue
                hits = self._index[ab].get(sig) or []
                if not hits:
                    continue
                bi = max(hits, key=lambda t: t[1])
                cand = self.factors[bi[0]]
                best_size, best_factor, best_sig = size, cand.name, sig
                sign_same = (cand.sign == pf.sign)
            per_ab[ab] = {
                "max_common_size": best_size,
                "matched_factor": best_factor,
                "matched_sig": (best_sig or "")[:160],
                "sign_same": sign_same,
            }

        # QuantMind 三条件 (按 enforce 抽象层)
        chosen = per_ab[self.abstraction]
        free_ratio = pf.free_args_ratio
        uniq_ratio = pf.unique_vars_ratio
        cond1 = chosen["max_common_size"] <= self.threshold
        cond2 = (-math.log(1 - free_ratio) < 0.693) if free_ratio < 1 else False
        cond3 = (-math.log(1 - uniq_ratio) < 0.693) if uniq_ratio < 1 else False
        signs_ok = chosen["sign_same"]

        duplicated = (not cond1) and cond2 and cond3 and signs_ok
        return {
            "ok": True,
            "expression": expression,
            "sign": pf.sign,
            "n_nodes": pf.n_nodes,
            "n_free_args": pf.n_free_args,
            "n_unique_vars": pf.n_unique_vars,
            "free_args_ratio": round(free_ratio, 4),
            "unique_vars_ratio": round(uniq_ratio, 4),
            "per_abstraction": per_ab,
            "threshold": self.threshold,
            "abstraction": self.abstraction,
            "cond1_size_le_thresh": cond1,
            "cond2_free_args": cond2,
            "cond3_unique_vars": cond3,
            "sign_same": signs_ok,
            # 🔴 shadow 语义: 即使 duplicated=True 也只记录, 不阻断
            "is_structural_dup": bool(duplicated),
            "would_reject": bool(duplicated) if self.mode == "enforce" else False,
            "mode": self.mode,
        }

    # ── 库内两两自扫 (诊断主入口) ─────────────────────────

    def scan_library_pairs(self, modes: Sequence[str] = _ABST_KEYS) -> Dict:
        """库内两两「最大公共子树」扫描。返回 flagged 对 + 分位分布。"""
        sizes: Dict[str, List[int]] = {ab: [] for ab in modes}
        pair_best: Dict[str, Dict[Tuple[int, int], Tuple[int, str]]] = {ab: {} for ab in modes}

        for ab in modes:
            idx = self._index[ab]
            for sig, hits in idx.items():
                if len(hits) < 2:
                    continue
                if len(hits) > self.max_fanout:
                    self.notes.append(
                        f"{ab}: 签名扇出 {len(hits)} > {self.max_fanout}, 跳过 (sig={sig[:70]})"
                    )
                    continue
                srt = sorted(hits)
                for a in range(len(srt)):
                    i, si = srt[a]
                    for b in range(a + 1, len(srt)):
                        j, sj = srt[b]
                        if i == j:
                            continue
                        key = (i, j) if i < j else (j, i)
                        cur = pair_best[ab].get(key)
                        if cur is None or si > cur[0]:
                            pair_best[ab][key] = (si, sig)

            sizes[ab] = [v[0] for v in pair_best[ab].values()]

        def _pct(lst: List[int], q: float) -> int:
            if not lst:
                return 0
            s = sorted(lst)
            k = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
            return s[k]

        dist = {
            ab: {
                "n_pairs_with_any_shared": len(sizes[ab]),
                "p50": _pct(sizes[ab], 0.50),
                "p90": _pct(sizes[ab], 0.90),
                "p99": _pct(sizes[ab], 0.99),
                "p999": _pct(sizes[ab], 0.999),
                "max": max(sizes[ab]) if sizes[ab] else 0,
            }
            for ab in modes
        }

        # ── 组装 flagged 对 (任一抽象层超阈值) ──
        all_keys: Set[Tuple[int, int]] = set()
        for ab in modes:
            all_keys |= set(pair_best[ab].keys())

        flagged = []
        n_ge = {ab: 0 for ab in modes}
        for key in all_keys:
            i, j = key
            rows = {}
            for ab in modes:
                v = pair_best[ab].get(key)
                rows[ab] = {"size": v[0], "sig": v[1][:140]} if v else {"size": 0, "sig": ""}
                if v and v[0] >= self.threshold:
                    n_ge[ab] += 1
            hit = any(rows[ab]["size"] >= self.threshold for ab in modes)
            if not hit:
                continue
            fa, fb = self.factors[i], self.factors[j]
            flagged.append({
                "factor_a": fa.name, "factor_b": fb.name,
                "sign_a": fa.sign, "sign_b": fb.sign,
                "sign_same": fa.sign == fb.sign,
                "max_common_size": rows,
                "expr_a": fa.expression[:200], "expr_b": fb.expression[:200],
            })

        flagged.sort(
            key=lambda r: max(r["max_common_size"][ab]["size"] for ab in modes),
            reverse=True,
        )
        return {"distribution": dist, "n_pairs_ge_threshold": n_ge, "flagged_pairs": flagged}

    # ── 与数值口径交叉读数 ───────────────────────────────

    def cross_read_numeric(self, flagged_pairs: List[Dict],
                           corr_path: Path = CORR_MATRIX_PATH,
                           lib_state_path: Path = LIB_STATE_PATH) -> Dict:
        """交叉读数: 「结构重复但数值不相关」vs「数值高相关但无结构重叠」。

        这是本模块的核心价值验证 —— 直接量化两个去重层的**互补覆盖面**。
        """
        out: Dict = {
            "corr_available": False,
            "corr_universe": 0,
            "structural_dup_numeric_uncorrelated": [],
            "numeric_dup_structurally_clean": [],
        }
        try:
            import numpy as np
        except Exception as e:                       # pragma: no cover
            out["error"] = f"numpy unavailable: {e}"
            return out
        if not corr_path.exists() or not lib_state_path.exists():
            out["error"] = "corr matrix or library state missing"
            return out

        try:
            st = json.loads(lib_state_path.read_text(encoding="utf-8"))
            raw_names: List[str] = st.get("_factor_names", []) or []
            corr = np.load(corr_path)["corr_matrix"]
        except Exception as e:
            out["error"] = f"load failed: {e}"
            return out

        n_corr = int(corr.shape[0])
        # 🔴 P-018 口径: `recluster(corr, factor_names)` 才重设 `_factor_names` (L345),
        # 而 `register_factor` 是 append (L256) ⇒ 矩阵建立后的名单是**头部切片**,
        # 尾部是后续新增但未重建矩阵的因子 (实测 1173 = 250 矩阵名 + 923 追加)。
        if len(raw_names) < n_corr:
            out["error"] = (f"corr matrix {n_corr} > _factor_names {len(raw_names)}; "
                            f"无法对齐")
            return out
        names: List[str] = raw_names[:n_corr]
        out["corr_name_slice_note"] = (
            f"_factor_names {len(raw_names)} 条 > corr {n_corr} 阶 ⇒ 按 recluster() 语义取头部 "
            f"{n_corr} 条为矩阵名单 (尾部 {len(raw_names) - n_corr} 条为矩阵建立后新注册、未重建矩阵)"
        )
        if corr.shape[1] != n_corr:
            out["error"] = f"corr not square: {corr.shape}"
            return out

        out["corr_available"] = True
        out["corr_universe"] = len(names)
        pos = {n: k for k, n in enumerate(names)}

        def _corr(a: str, b: str):
            ia, ib = pos.get(a), pos.get(b)
            if ia is None or ib is None:
                return None
            return float(corr[ia, ib])

        # ① 结构重复 且 |corr| < 0.30  ⇒ 数值去重完全看不到的盲区
        for r in flagged_pairs:
            c = _corr(r["factor_a"], r["factor_b"])
            if c is None:
                continue
            if abs(c) < 0.30:
                out["structural_dup_numeric_uncorrelated"].append({
                    "factor_a": r["factor_a"], "factor_b": r["factor_b"],
                    "abs_corr": round(abs(c), 4), "sign_same": r["sign_same"],
                    "max_common_size": max(v["size"] for v in r["max_common_size"].values()),
                })
        out["structural_dup_numeric_uncorrelated"].sort(key=lambda x: x["abs_corr"])

        # ② |corr| > 0.85 但结构零重叠 ⇒ 数值去重抓到了、结构层无信号的
        flagged_set = {(r["factor_a"], r["factor_b"]) for r in flagged_pairs}
        flagged_set |= {(b, a) for a, b in flagged_set}
        n = len(names)
        for a in range(n):
            for b in range(a + 1, n):
                if abs(corr[a, b]) <= 0.85:
                    continue
                if (names[a], names[b]) in flagged_set:
                    continue
                out["numeric_dup_structurally_clean"].append({
                    "factor_a": names[a], "factor_b": names[b],
                    "abs_corr": round(abs(float(corr[a, b])), 4),
                })
        out["numeric_dup_structurally_clean"].sort(key=lambda x: -x["abs_corr"])
        out["n_structural_dup_numeric_uncorr"] = len(out["structural_dup_numeric_uncorrelated"])
        out["n_numeric_dup_structurally_clean"] = len(out["numeric_dup_structurally_clean"])
        return out

    # ── 读数行 ────────────────────────────────────────────

    @staticmethod
    def format_readout(report: Dict) -> str:
        lib = report.get("library", {})
        summ = report.get("summary", {})
        ge = summ.get("n_pairs_ge_threshold", {})
        cross = report.get("cross_read", {}) or {}
        return (
            f"[StructDedup:{report.get('mode', 'shadow')}] "
            f"库 {lib.get('n_parsed', 0)} 因子 (解析失败 {lib.get('n_parse_fail', 0)}) | "
            f"阈值 {report.get('threshold')}: "
            f"字面 {ge.get('literal', 0)} / 字段 {ge.get('field', 0)} / 全抽象 {ge.get('full', 0)} 对 | "
            f"结构重复且|corr|<0.30: {cross.get('n_structural_dup_numeric_uncorr', 'n/a')} | "
            f"数值|corr|>0.85但结构零重叠: {cross.get('n_numeric_dup_structurally_clean', 'n/a')}"
        )

    # ── 全量扫描 ──────────────────────────────────────────

    def run_scan(self, lib_state_path: Path = LIB_STATE_PATH,
                 persist: bool = True,
                 top_n: int = 30) -> Dict:
        st = json.loads(Path(lib_state_path).read_text(encoding="utf-8"))
        factors = st.get("factors", []) or []
        idxstats = self.index_library(factors)
        scan = self.scan_library_pairs()
        cross = self.cross_read_numeric(scan["flagged_pairs"], lib_state_path=lib_state_path)

        report = {
            "generated_at": datetime.now().isoformat(),
            "proposal_id": "P-20260915-005",
            "mode": self.mode,
            "abstraction_for_enforce": self.abstraction,
            "threshold": self.threshold,
            "min_size": self.min_size,
            "max_fanout": self.max_fanout,
            "parity_note": (
                "阈值口径差异: QuantMind 函数式 TS_MEAN($close,20)=3 节点; "
                "伏羲 pandas 方法链 close.rolling(20).mean()=4 节点 (方法接收者多一层)。"
                "本模块保留嵌套形式以保住 rolling 链的公共子树信号 ⇒ 阈值按库内实测标定, "
                f"未盲抄 QuantMind 的 {QUANTMIND_PARITY_THRESHOLD}。"
                f"实测: 阈值 {QUANTMIND_PARITY_THRESHOLD} 命中的多为共用归一化样板 "
                "(如 std(rolling(sub(buy_lg_vol,sell_lg_vol),60))+1e-8), 判别性分子不同 ⇒ 低精度; "
                f"阈值 {DEFAULT_THRESHOLD} 命中共用判别性核心谓词 ⇒ 高精度。"
                f"故 DEFAULT_THRESHOLD={DEFAULT_THRESHOLD}。见 max_common_size_distribution。"
            ),
            "library": {
                "state_path": str(lib_state_path),
                **idxstats,
            },
            "max_common_size_distribution": scan["distribution"],
            "summary": {
                "n_pairs_ge_threshold": scan["n_pairs_ge_threshold"],
                "n_flagged_pairs": len(scan["flagged_pairs"]),
            },
            "cross_read": cross,
            "flagged_pairs_top": scan["flagged_pairs"][:top_n],
            "notes": self.notes[:50],
            "zero_adjudication": self.mode != "enforce",
            "readout": "",
        }
        report["readout"] = self.format_readout(report)

        if persist:
            REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
            REPORT_PATH.write_text(
                json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        return report


# ═══════════════════════════════════════════════════════════════
# 便捷函数
# ═══════════════════════════════════════════════════════════════

_default: Optional[ASTStructureDeductor] = None


def get_deductor() -> ASTStructureDeductor:
    """获取默认去重器 (惰性单例)。"""
    global _default
    if _default is None:
        _default = ASTStructureDeductor()
    return _default


def check_structure(expression: str) -> Dict:
    """快捷检查单条表达式 (需先 index_library)。"""
    return get_deductor().check_candidate(expression)


# ── 单轮候选集影子读数 (供 ralph_loop 挂钩, 零裁决) ───────────
_lib_cache: Dict = {"key": None, "deductor": None}


def _get_indexed_deductor(lib_state_path: Path) -> Optional[ASTStructureDeductor]:
    """按库文件 mtime 缓存的已建索引去重器 (每轮复用, 免重复解析 900+ 因子)。"""
    try:
        key = (str(lib_state_path), Path(lib_state_path).stat().st_mtime_ns)
    except Exception:
        return None
    if _lib_cache["key"] == key and _lib_cache["deductor"] is not None:
        return _lib_cache["deductor"]
    try:
        st = json.loads(Path(lib_state_path).read_text(encoding="utf-8"))
        factors = st.get("factors", []) or []
        d = ASTStructureDeductor()
        d.index_library(factors)
    except Exception:
        return None
    _lib_cache["key"] = key
    _lib_cache["deductor"] = d
    return d


def get_indexed_deductor(lib_state_path: Optional[Path] = None) -> Optional[ASTStructureDeductor]:
    """公开包装 (M10, 2026-09-27): 带 mtime 缓存的已建索引去重器。

    供 ralph_loop 的 S1 前置打标复用 (与 shadow_readout 共享同一模块级缓存,
    同一轮内不会重复建索引)。库文件不可读时返回 None (调用方须容错)。
    """
    return _get_indexed_deductor(
        Path(lib_state_path) if lib_state_path else LIB_STATE_PATH)


def shadow_readout(candidates: Sequence, lib_state_path: Optional[Path] = None) -> Dict:
    """P-20260915-005: 本轮候选 vs 因子库的结构重复影子读数 (**零裁决**)。

    - off    : 直接返回空读数, 不做任何工作
    - shadow : 计算读数 + 打印, 但**不影响任何候选的通过/拒绝**
    - enforce: 裁决消费点在 ralph_loop S1 前置打标 (`_struct_dup_hit`,
      M10 2026-09-27 接线) + multi_stage_validator.stage1_fast_ic; 本函数
      只报读数 (与打标共享同一 deductor 缓存, 不重复建索引)。

    返回: {mode, n_candidates, n_parse_fail, n_flagged, flagged[], readout}
    """
    lp = Path(lib_state_path) if lib_state_path else LIB_STATE_PATH
    mode = (os.environ.get("FUXI_STRUCT_DEDUP", "shadow") or "shadow").strip().lower()
    if mode not in ("off", "shadow", "enforce"):
        mode = "shadow"
    out: Dict = {"mode": mode, "n_candidates": 0, "n_parse_fail": 0,
                 "n_flagged": 0, "flagged": [], "readout": ""}
    if mode == "off":
        return out

    d = _get_indexed_deductor(lp)
    if d is None:
        out["error"] = "index_unavailable"
        return out

    n_fail = 0
    flagged: List[Dict] = []
    for i, c in enumerate(candidates or []):
        expr = c if isinstance(c, str) else (
            c.get("expression") or c.get("formula") or "")
        nm = "" if isinstance(c, str) else (c.get("name") or f"cand{i}")
        r = d.check_candidate(expr)
        out["n_candidates"] += 1
        if not r.get("ok"):
            n_fail += 1
            continue
        if r.get("is_structural_dup"):
            _ch = (r.get("per_abstraction") or {}).get(r.get("abstraction"), {}) or {}
            flagged.append({
                "name": nm, "sign": r.get("sign"),
                "n_nodes": r.get("n_nodes"),
                "max_common_size": _ch.get("max_common_size"),
                "matched": _ch.get("matched_factor"),
                "matched_sig": _ch.get("matched_sig"),
            })
    out["n_parse_fail"] = n_fail
    out["n_flagged"] = len(flagged)
    out["flagged"] = flagged[:10]
    out["readout"] = (
        f"[StructDedup:{mode}] 候选 {out['n_candidates']} 条 (解析失败 {n_fail}) | "
        f"结构重复 {len(flagged)} 条"
        + (f" | {flagged[0]['name']} vs {flagged[0]['matched']} "
           f"(公共子树 {flagged[0]['max_common_size']} 节点)"
           if flagged else "")
    )
    return out


# ── 标定记录 (DEFAULT_THRESHOLD 的实证依据, 首跑后回报) ──────
# _CALIBRATION: 见 data/structure_dedup_report.json 的
#   max_common_size_distribution 与 n_pairs_ge_threshold。
#   首跑实测 (923 因子) 后确定默认阈值, 记录于 v0.7.15 CHANGELOG。


if __name__ == "__main__":  # pragma: no cover
    d = ASTStructureDeductor()
    rep = d.run_scan()
    print(rep["readout"])
    print(json.dumps(rep["max_common_size_distribution"], ensure_ascii=False, indent=2))
