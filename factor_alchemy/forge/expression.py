"""
Factor Forge 表达式树
======================
GP 因子表达式的树形表示与评估。

树结构:
  每个节点存储一个 Primitive 和子节点列表。
  叶节点: Primitive(name='close') — 输入变量
  内部节点: Primitive(name='ts_mean') + [close, 20] — 滚动均值
  root: Primitive(name='div') + [close, ts_mean(close,20)] — close/MA20

评估:
  给定数据字典 {open, high, low, close, volume, vwap, returns},
  递归评估树 → 产生 (时间×股票) 矩阵 → 截面 ICIR
"""

import re
import numpy as np
from typing import Dict, Optional, List, Tuple, Set
from forge.primitives import (
    Primitive, PRIMITIVE_BY_NAME, INPUT_PRIMITIVES,
    WINDOW_SIZES,
)

# v0.6.1 (2026-08-29): 外部注册输入终端 (Ralph Loop 种子翻译注入 S5 字段,
# 如 lhb_flag/pe_ttm 等 — Forge 原生解析器不认识的字段此前 parse_expression
# 返回 None → seed 重检翻译失败跳过)
EXTRA_TERMINALS: Set[str] = set()


class ExprNode:
    """表达式树节点"""
    __slots__ = ['primitive', 'children', '_hash']

    def __init__(self, primitive: Primitive, children: List["ExprNode"] = None):
        self.primitive = primitive
        self.children = children or []
        self._hash = None

    @property
    def is_leaf(self):
        return len(self.children) == 0

    @property
    def complexity(self):
        """递归计算表达式复杂度"""
        c = self.primitive.complexity
        for child in self.children:
            c += child.complexity
        return c

    def to_string(self) -> str:
        """转为可读表达式字符串"""
        if self.is_leaf:
            return self.primitive.name

        name = self.primitive.name
        if len(self.children) == 1:
            return f"{name}({self.children[0].to_string()})"
        elif len(self.children) == 2:
            # 检查第二个子节点是否是窗口大小
            c0 = self.children[0].to_string()
            c1 = self.children[1].to_string()
            # 时间序列: ts_mean(close, 20)
            if name.startswith("ts_"):
                return f"{name}({c0}, {c1})"
            # 算术: add(close, volume)
            return f"{name}({c0}, {c1})"
        else:
            args = ", ".join(c.to_string() for c in self.children)
            return f"{name}({args})"

    def __repr__(self):
        return f"ExprNode({self.to_string()})"

    def __hash__(self):
        if self._hash is None:
            self._hash = hash(self.to_string())
        return self._hash

    def __eq__(self, other):
        if not isinstance(other, ExprNode):
            return False
        return self.to_string() == other.to_string()

    def clone(self) -> "ExprNode":
        """深拷贝"""
        return ExprNode(
            self.primitive,
            [c.clone() for c in self.children]
        )

    def to_pandas_string(self) -> str:
        """转为 fri.py 可执行的 pandas 滚动表达式 (替代 Forge 原语表示)
        
        映射规则:
          - 算术原语 → Python 中缀表达式
          - 时间序列 → .rolling(w).mean()/.std()/.shift()/.pct_change() 等
          - 截面原语 → .rank(axis=1)/.mean(axis=1) 等
          - 终端 → close_p/high_p/low_p/volume_p 等 fri.py 变量名
        """
        if self.is_leaf:
            return self._terminal_to_pandas(self.primitive.name)

        name = self.primitive.name
        children_pd = [c.to_pandas_string() for c in self.children]

        # ── 时间序列 ──
        if name.startswith("ts_"):
            return self._ts_to_pandas(name, children_pd)

        # ── 截面 ──
        if name in ("rank", "zscore", "scale"):
            return self._cs_to_pandas(name, children_pd)

        # ── 二元算术 ──
        if len(children_pd) == 2:
            return self._binary_to_pandas(name, children_pd)

        # ── 一元算术 ──
        if len(children_pd) == 1:
            return self._unary_to_pandas(name, children_pd)

        # fallback
        return f"{name}({', '.join(children_pd)})"

    @staticmethod
    def _terminal_to_pandas(name: str) -> str:
        """终端变量名映射"""
        mapping = {
            "open": "open_p",
            "high": "high_p",
            "low": "low_p",
            "close": "close_p",
            "volume": "volume_p",
            "vwap": "volume_p",        # 无 vwap, 用 volume 代理
            "returns": "((close_p - close_p.shift(1)) / close_p.shift(1))",
        }
        return mapping.get(name, name)

    @staticmethod
    def _ts_to_pandas(name: str, args: list) -> str:
        """时间序列原语 → pandas rolling

        P-20260831-002 (2026-08-31): GP 常数叶子进 ts_* 窗口的病态个体
        (如 ts_min(0.30, 12)) 转译成 "0.30.rolling(12).min()" → SyntaxError:
        invalid decimal literal (Forge numpy fitness 路径合法, 只有 pandas
        转译炸) → S1 向量化 eval 挂起。修复: 数字字面量参数直接退化为常数
        (常数窗口聚合=常数本身; rank IC 对常数缩放/平移不变, 语义无损)。
        """
        x, w = args[0], args[1]
        x_s = str(x).strip()
        # 裸数字或带括号数字: "0.30" / "(0.30)" / "-0.5" / "(-0.5)"
        if re.match(r'^\(?-?\d+(\.\d+)?([eE][+-]?\d+)?\)?$', x_s):
            return x_s
        ts_map = {
            "ts_sum":   lambda x, w: f"{x}.rolling({w}).sum()",
            "ts_mean":  lambda x, w: f"{x}.rolling({w}).mean()",
            "ts_std":   lambda x, w: f"{x}.rolling({w}).std()",
            "ts_min":   lambda x, w: f"{x}.rolling({w}).min()",
            "ts_max":   lambda x, w: f"{x}.rolling({w}).max()",
            "ts_delta": lambda x, w: f"({x} - {x}.shift({w}))",
            "ts_pct":   lambda x, w: f"{x}.pct_change({w})",
            "ts_rank":  lambda x, w: f"{x}.rolling({w}).rank(pct=True)",
            "ts_zscore":lambda x, w: f"(({x} - {x}.rolling({w}).mean()) / ({x}.rolling({w}).std() + 1e-6))",
            "ts_ema_decay": lambda x, w: f"({x} - {x}.ewm(span={w}, adjust=False).mean())",
        }
        if name in ts_map:
            return ts_map[name](x, w)
        return f"{name}({x}, {w})"

    @staticmethod
    def _cs_to_pandas(name: str, args: list) -> str:
        """截面原语 → 真截面 pandas 表达式 (axis=1, 与 Forge fitness 语义一致)

        v0.6.2 (2026-08-29) 修复: 旧实现把 rank/zscore/scale 降级为长窗口
        时序近似 (rank → rolling(252).rank, 为旧 fri.py 逐股路径设计)。
        但 Forge 自身 fitness 用真截面 _cross_rank/_cross_zscore/_cross_scale
        (primitives.py, 沿股票轴 axis=1) → 表型≠基因型, Forge 精选的因子
        在流水线 S1 日频 IC 全灭 (max|IC|≈0.02) + rolling 近似与 IC Computer
        v0.9.4 rank 语义修正正则冲突 (TypeError: Rolling.rank axis)。

        下游两个消费者均已支持 axis=1 宽表:
          - FactorICComputer._eval_formula: 宽表直接 eval, axis=1 天然正确
          - S5JointFilter._needs_cross_section: 检测 rank(pct=True, axis=1 /
            mean(axis=1) 标记 → 走 _evaluate_cross_section 宽表路径
        """
        x = args[0]
        # P-20260831-002: 常数进截面原语同样退化 (rank(0.30) → 0.30.rank(axis=1) SyntaxError)
        x_s = str(x).strip()
        if re.match(r'^\(?-?\d+(\.\d+)?([eE][+-]?\d+)?\)?$', x_s):
            return x_s
        cs_map = {
            "rank":   lambda x: f"{x}.rank(pct=True, axis=1)",
            "zscore": lambda x: f"(({x} - {x}.mean(axis=1)) / ({x}.std(axis=1) + 1e-6))",
            "scale":  lambda x: f"(({x} - {x}.min(axis=1)) / (({x}.max(axis=1) - {x}.min(axis=1)) + 1e-6))",
        }
        if name in cs_map:
            return cs_map[name](x)
        return f"{name}({x})"

    @staticmethod
    def _binary_to_pandas(name: str, args: list) -> str:
        """二元算术 → 中缀"""
        a, b = args[0], args[1]
        bin_map = {
            "add": lambda a, b: f"({a} + {b})",
            "sub": lambda a, b: f"({a} - {b})",
            "mul": lambda a, b: f"({a} * {b})",
            "div": lambda a, b: f"({a} / ({b} + 1e-6))",
        }
        if name in bin_map:
            return bin_map[name](a, b)
        return f"{name}({a}, {b})"

    @staticmethod
    def _unary_to_pandas(name: str, args: list) -> str:
        """一元算术 → 中缀"""
        x = args[0]
        # 无需括号的情况: 终端名、.rolling() 链
        needs_paren = any(op in x for op in "+-*/") and not x.startswith("(")
        safe_x = f"({x})" if needs_paren else x

        unary_map = {
            "neg":    lambda x: f"-{x}",
            "inv":    lambda x: f"(1.0 / ({x} + 1e-6))",
            "sqrt":   lambda x: f"np.sqrt(np.maximum({x}, 0))",
            "abs":    lambda x: f"np.abs({x})",
            "log":    lambda x: f"np.log(np.maximum({x}, 1e-8))",
            "square": lambda x: f"({x} * {x})",
            "sign":   lambda x: f"np.sign({x})",
        }
        if name in unary_map:
            result = unary_map[name](safe_x)
            return result
        return f"{name}({x})"

    def size(self) -> int:
        """树中节点总数"""
        return 1 + sum(c.size() for c in self.children)

    def depth(self) -> int:
        """树的最大深度"""
        if self.is_leaf:
            return 0
        return 1 + max(c.depth() for c in self.children)

    def terminals(self) -> List[str]:
        """收集所有输入终端名称"""
        if self.is_leaf:
            return [self.primitive.name] if self.primitive.is_input else []
        result = []
        for c in self.children:
            result.extend(c.terminals())
        return list(set(result))


# ═══════════════════════════════════════════════════════
# 评估器
# ═══════════════════════════════════════════════════════

class ExpressionEvaluator:
    """在数据上评估表达式树"""

    def __init__(self, data: Dict[str, np.ndarray]):
        """
        Args:
            data: {'open': (T,N), 'high': (T,N), 'low': (T,N),
                   'close': (T,N), 'volume': (T,N),
                   'vwap': (T,N), 'returns': (T,N)}
        """
        self.data = data

    def evaluate(self, node: ExprNode) -> np.ndarray:
        """递归评估表达式树, 返回 (T,N) 数组"""
        if node.is_leaf:
            name = node.primitive.name
            # 检查是否是数字常量
            try:
                val = float(name)
                # 返回与数据同shape的常数数组
                ref = self.data.get("close")
                if ref is not None:
                    return np.full(ref.shape, val)
                return np.array([[val]])
            except ValueError:
                pass

            # 映射输入名称
            name_map = {
                "returns": "returns",
                "ret": "returns",
            }
            key = name_map.get(name, name)
            if key in self.data:
                return self.data[key].copy()
            return np.full_like(self.data.get("close", np.zeros((1,1))), np.nan)

        # 递归评估子节点
        child_vals = [self.evaluate(c) for c in node.children]

        # 处理时间序列原语的第二个参数(窗口大小)
        func = node.primitive.func
        if node.primitive.arity == 2 and node.primitive.name.startswith("ts_"):
            # 第二个子节点是窗口参数的数字或表达式
            if node.children[1].is_leaf:
                try:
                    window = float(node.children[1].primitive.name)
                except ValueError:
                    window = 20.0
            else:
                # 如果第二个子节点也是表达式, 用它的均值作为窗口
                window = max(2, min(60, np.nanmean(np.abs(child_vals[1]))))

            # 调用 func(x, window)
            return func(child_vals[0], window)

        # 普通调用
        if len(child_vals) == 1:
            return func(child_vals[0])
        elif len(child_vals) == 2:
            return func(child_vals[0], child_vals[1])
        else:
            # 不支持多于2个参数
            return np.full_like(child_vals[0], np.nan)


# ═══════════════════════════════════════════════════════
# 表达式生成器
# ═══════════════════════════════════════════════════════

def _weighted_pick(rng: np.random.RandomState, candidates: list,
                   weights: Dict = None):
    """按权重采样。weights 为 {name: weight} 或列表; 缺失项权重=1.0。

    v0.7 P-025: 范式定向初始化 — 不同范式偏好不同终端/算子。
    """
    if weights is None:
        return candidates[rng.randint(0, len(candidates))]
    if isinstance(weights, dict):
        w = np.array([float(weights.get(getattr(c, 'name', str(c)), 1.0))
                      for c in candidates])
    else:
        w = np.asarray(weights, dtype=float)
    w = np.where(w > 0, w, 0.0)
    if w.sum() <= 0:
        return candidates[rng.randint(0, len(candidates))]
    return candidates[rng.choice(len(candidates), p=w / w.sum())]


def _pick_primitive(rng: np.random.RandomState, include_ts: bool,
                    include_cs: bool, op_weights: Dict = None):
    """选取内部节点原语 (v0.7: 支持算子级范式偏置)"""
    from forge.primitives import (
        ARITHMETIC_PRIMITIVES, TIMESERIES_PRIMITIVES, CROSS_SECTION_PRIMITIVES,
    )
    candidates = list(ARITHMETIC_PRIMITIVES)
    if include_ts:
        candidates += TIMESERIES_PRIMITIVES
    if include_cs:
        candidates += CROSS_SECTION_PRIMITIVES
    return _weighted_pick(rng, candidates, op_weights)


def _pick_terminal(rng: np.random.RandomState, terminal_weights: Dict = None):
    """选取终端 (v0.7: 支持终端级范式偏置; 无偏置时沿用 15% 常量逻辑)"""
    from forge.primitives import get_random_terminal, INPUT_PRIMITIVES
    if terminal_weights is None:
        return ExprNode(get_random_terminal(rng, include_constants=True))
    # 带偏置: 10% 常量 + 90% 加权输入变量
    if rng.random() < 0.10:
        return ExprNode(get_random_terminal(rng, include_constants=True))
    return ExprNode(_weighted_pick(rng, INPUT_PRIMITIVES, terminal_weights))


# ═══════════════════════════════════════════════════════
# P-20260831: 维度一致性预剪枝门 (Alpha2 移植)
# ═══════════════════════════════════════════════════════

_dim_cache: dict = {}  # 子树字符串 → 根量纲 (生成期缓存, 避免重复遍历)


def _resolve_dim_mode(dim_prune):
    """dim_prune=None 时读模块默认模式 (env FUXI_DIM_PRUNE, 默认 shadow)"""
    if dim_prune is None:
        from forge import dimension_rules as dr
        return dr.get_mode()
    return dim_prune


def _child_dim(node: ExprNode) -> str:
    """子树根量纲 (带缓存; 超限清空防长跑内存膨胀)"""
    global _dim_cache
    if len(_dim_cache) > 200_000:
        _dim_cache.clear()
    key = node.to_string()
    if key not in _dim_cache:
        from forge import dimension_rules as dr
        _dim_cache[key] = dr.dim_of_node(node)
    return _dim_cache[key]


def _dim_gate(p_name: str, children: List[ExprNode], dim_prune: str):
    """剪枝门: 返回 (allowed, hard)。
    off: 直接放行; shadow: 只计数不拦截; enforce: hard 违规返回 allowed=False。
    """
    if dim_prune == "off":
        return True, False
    from forge import dimension_rules as dr
    dims = tuple(_child_dim(c) for c in children)
    r = dr.apply_op(p_name, list(dims))
    if r.hard:
        dr.COUNTERS.record(p_name, dims, True, False)
        return (dim_prune != "enforce"), True
    if r.soft:
        dr.COUNTERS.record(p_name, dims, False, True)
    return True, False


def _dim_retry_ok(p_name: str, children: List[ExprNode]) -> bool:
    """enforce 重采样路径: 只判断不计数 (首次违规已由 _dim_gate 计数)"""
    from forge import dimension_rules as dr
    dims = tuple(_child_dim(c) for c in children)
    return not dr.apply_op(p_name, list(dims)).hard


def grow_random_tree(rng: np.random.RandomState,
                     max_depth: int = 4,
                     include_ts: bool = True,
                     include_cs: bool = True,
                     terminal_weights: Dict = None,
                     op_weights: Dict = None,
                     dim_prune: str = None) -> ExprNode:
    """Grow 方法: 随机生成完整树 (所有分支填满到max_depth或随机停止)

    v0.7: terminal_weights/op_weights 支持范式定向生成 (P-025)。
    P-20260831: dim_prune ∈ {off, shadow, enforce} — 维度一致性预剪枝。
      shadow(默认): 只计数不拦截; enforce: hard 违规重采样, 兜底退化为终端。
    """
    dim_prune = _resolve_dim_mode(dim_prune)

    def _grow(depth: int, min_depth: int) -> ExprNode:
        if depth >= max_depth or (depth >= min_depth and rng.random() < 0.4):
            # 终端
            return _pick_terminal(rng, terminal_weights)

        # 内部节点
        p = _pick_primitive(rng, include_ts, include_cs, op_weights)
        if p.arity == 1:
            child = _grow(depth + 1, min_depth)
            return ExprNode(p, [child])
        elif p.arity == 2:
            c0 = _grow(depth + 1, min_depth)
            # 时间序列原语的第二个参数是窗口 → 固定为数字终端
            if p.name.startswith("ts_"):
                w = rng.choice(WINDOW_SIZES)
                c1 = ExprNode(
                    Primitive(str(w), None, 0, 0, is_input=False)
                )
                return ExprNode(p, [c0, c1])
            c1 = _grow(depth + 1, min_depth)
            allowed, hard = _dim_gate(p.name, [c0, c1], dim_prune)
            if hard and not allowed:
                # enforce: 重采样第二子节点, 失败则整体退化为终端
                ok = False
                for _ in range(4):
                    c1 = _grow(depth + 1, min_depth)
                    if _dim_retry_ok(p.name, [c0, c1]):
                        ok = True
                        break
                if not ok:
                    return _pick_terminal(rng, terminal_weights)
            return ExprNode(p, [c0, c1])
        return _pick_terminal(rng, terminal_weights)

    return _grow(0, 2)


def full_random_tree(rng: np.random.RandomState,
                     max_depth: int = 4,
                     terminal_weights: Dict = None,
                     op_weights: Dict = None,
                     dim_prune: str = None) -> ExprNode:
    """Full 方法: 所有叶节点在同一深度
    (v0.7: 支持范式偏置; P-20260831: dim_prune 维度预剪枝)
    """
    dim_prune = _resolve_dim_mode(dim_prune)

    def _full(depth: int) -> ExprNode:
        if depth >= max_depth:
            return _pick_terminal(rng, terminal_weights)

        p = _pick_primitive(rng, True, True, op_weights)
        if p.arity == 1:
            return ExprNode(p, [_full(depth + 1)])
        elif p.arity == 2:
            c0 = _full(depth + 1)
            if p.name.startswith("ts_"):
                w = rng.choice(WINDOW_SIZES)
                c1 = ExprNode(Primitive(str(w), None, 0, 0, is_input=False))
                return ExprNode(p, [c0, c1])
            c1 = _full(depth + 1)
            allowed, hard = _dim_gate(p.name, [c0, c1], dim_prune)
            if hard and not allowed:
                ok = False
                for _ in range(4):
                    c1 = _full(depth + 1)
                    if _dim_retry_ok(p.name, [c0, c1]):
                        ok = True
                        break
                if not ok:
                    return _pick_terminal(rng, terminal_weights)
            return ExprNode(p, [c0, c1])
        return _pick_terminal(rng, terminal_weights)

    return _full(0)


# ═══════════════════════════════════════════════════════
# 字符串 <-> 树 转换 (用于技能库持久化)
# ═══════════════════════════════════════════════════════

def parse_expression(expr_str: str) -> Optional[ExprNode]:
    """从字符串解析表达式 (简化版)

    v0.6.1 (2026-08-29): 支持外部注册输入终端 (EXTRA_TERMINALS),
    供 Ralph Loop 种子翻译路径注入 S5 已知字段 (lhb_flag 等)。
    """
    expr_str = expr_str.strip()

    # 匹配原语调用: name(args)
    m = re.match(r'^(\w+)\((.*)\)$', expr_str)
    if not m:
        # 终端
        name = expr_str.strip()
        if name in PRIMITIVE_BY_NAME:
            return ExprNode(PRIMITIVE_BY_NAME[name])
        # v0.6.1: 外部注册终端 (S5 字段等)
        if name in EXTRA_TERMINALS:
            return ExprNode(Primitive(name, None, 0, 0, is_input=True))
        # 数字常量
        try:
            float(name)
            return ExprNode(Primitive(name, None, 0, 0, is_input=False))
        except ValueError:
            return None

    name = m.group(1)
    args_str = m.group(2)

    if name not in PRIMITIVE_BY_NAME:
        return None

    p = PRIMITIVE_BY_NAME[name]
    children = _split_args(args_str, p.arity)
    child_nodes = []
    for c in children:
        node = parse_expression(c)
        if node is None:
            return None
        child_nodes.append(node)

    return ExprNode(p, child_nodes)


def _split_args(args_str: str, expected: int) -> List[str]:
    """安全分割参数 (处理嵌套括号)"""
    args = []
    depth = 0
    current = []
    for ch in args_str:
        if ch == ',' and depth == 0:
            args.append(''.join(current).strip())
            current = []
        else:
            if ch == '(':
                depth += 1
            elif ch == ')':
                depth -= 1
            current.append(ch)
    if current:
        args.append(''.join(current).strip())
    return args


# ═══════════════════════════════════════════════════════
# 修剪 / 简化
# ═══════════════════════════════════════════════════════

def simplify_tree(node: ExprNode, max_size: int = 30) -> ExprNode:
    """修剪过大的树 (截断最深分支)"""
    if node.size() <= max_size:
        return node.clone()

    # 递归修剪
    result = ExprNode(node.primitive)
    for child in node.children:
        if child.size() > max_size // 3:
            # 截断: 替换为终端
            result.children.append(
                ExprNode(
                    Primitive("close", None, 0, 0, is_input=True)
                )
            )
        else:
            result.children.append(simplify_tree(child, max_size // 3))
    return result
