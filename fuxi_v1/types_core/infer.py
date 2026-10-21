"""types_core.infer —— 类型推断双前端（AST 递归 + preorder 前缀栈式）。

- ``infer_ast``：fuxi_v1 侧，dict AST 递归（后序），叶子 = {"field": name}。
- ``infer_prefix``：factor_mining 侧，preorder 前缀 token 序列（栈式/递归下降）。
- 两者共享同一 rules.py 规则表，保证两套语法纪律不漂移（R5 治理的核心）。

自除判断（V_SELF_DIV）在此层做**结构等价**判断：div 两侧子表达式结构相同即判自除，
强于 grammar.py 现有的「同一 ndarray 对象」判断（能拦 div(rank(X), rank(X))）。
"""

from __future__ import annotations

import json

from .rules import (
    BINARY_ARITH, TEMPORAL, Violation, rule_for, _check_temporal,
)
from .spec import TypeSpec

# window -> horizon 标签（时序算子的调仓频率语义）
_WINDOW_HORIZON = (
    (5, "fast"),
    (20, "medium"),
)


def window_horizon(w):
    """窗口期数 -> horizon 标签。<=5 fast；<=20 medium；其余 slow。"""
    if w is None:
        return "none"
    for bound, h in _WINDOW_HORIZON:
        if w <= bound:
            return h
    return "slow"


def _ast_canonical(node) -> str:
    """AST 结构规范形式（自除判断用）。"""
    return json.dumps(node, sort_keys=True, default=str)


def infer_ast(node, resolver):
    """dict AST 递归推断。返回 (TypeSpec, [Violation])。

    resolver(name) -> TypeSpec：解析叶子（fuxi_v1 侧为字段名）。
    """
    violations: list[Violation] = []

    # 叶子
    if isinstance(node, dict) and "field" in node:
        return resolver(node["field"]), violations

    op = node["op"]
    args = node.get("args", [])
    check, output = rule_for(op)

    # 时序算子：输入必须是 field 叶子，且只接受单叶子
    if op in TEMPORAL:
        leaf = args[0]
        leaf_spec, vs = infer_ast(leaf, resolver)
        violations.extend(vs)
        is_leaf = isinstance(leaf, dict) and "field" in leaf
        violations.extend(_check_temporal(op, leaf_spec.form, is_leaf))
        out = output(op, leaf_spec)
        w = node.get("window")
        if w is not None:
            out = out.with_(horizon=window_horizon(w))
        return out, violations

    # 递归子节点
    child_specs = []
    for arg in args:
        spec, vs = infer_ast(arg, resolver)
        child_specs.append(spec)
        violations.extend(vs)

    # 结构自除（div）
    if op == "div" and len(args) == 2 and _ast_canonical(args[0]) == _ast_canonical(args[1]):
        violations.append(Violation("V_SELF_DIV", op, "self division (structurally identical)", "hard"))

    if check is not None:
        violations.extend(check(op, *child_specs))
    return output(op, *child_specs), violations


def infer_prefix(tokens, resolver, is_op):
    """preorder 前缀 token 序列推断。返回 (TypeSpec, [Violation])。

    - tokens：list，叶子是因子名（resolver 解析），算子在 is_op 判定内。
    - resolver(name) -> TypeSpec：解析叶子（factor_mining 侧为因子名）。
    - is_op(token) -> bool：是否算子 token。
    """
    violations: list[Violation] = []
    idx = 0

    def parse():
        nonlocal idx
        tok = tokens[idx]
        idx += 1
        if not is_op(tok):
            return resolver(tok), [], (tok,)
        op = tok
        check, output = rule_for(op)
        if op in BINARY_ARITH:
            left, vl, sl = parse()
            right, vr, sr = parse()
            vs = vl + vr
            if op == "div" and sl == sr:
                vs.append(Violation("V_SELF_DIV", op, "self division (structurally identical)", "hard"))
            if check is not None:
                vs.extend(check(op, left, right))
            return output(op, left, right), vs, (op,) + sl + sr
        # 一元算子（含时序算子统一按单参数处理；factor_mining 侧无时序算子）
        child, vc, sc = parse()
        if check is not None:
            vc.extend(check(op, child))
        return output(op, child), vc, (op,) + sc

    spec, vs, _ = parse()
    return spec, vs


def check_tree(node, resolver):
    """编译期树检查（W1 用）。返回全部 Violation（含 hard + soft）。

    调用方按 severity 过滤：hard = 拒绝，soft = 诊断。
    """
    _, violations = infer_ast(node, resolver)
    return violations


def hard_violations(violations):
    """过滤出 hard（编译拒绝）违规。"""
    return [v for v in violations if v.severity == "hard"]


def soft_violations(violations):
    """过滤出 soft（诊断/惩罚）违规。"""
    return [v for v in violations if v.severity == "soft"]


# ---------------------------------------------------------------------------
# 增量类型状态 + 动作掩码（P3，供 GFlowNet preorder 生成逐 token 约束）
# ---------------------------------------------------------------------------

class TypeState:
    """preorder 增量类型状态：显式栈模拟递归归约，逐 token 推进并检测 hard 违规。

    用途：GFlowNet 采样时，每一步用 ``mask_actions`` 试探候选 token，过滤会立即
    触发 hard 违规（add/sub 跨 form、div 分母 signed、log 非无量纲/非正等）的 token，
    让训练预算集中在类型合法空间——只收窄动作集、不改变已完成轨迹的 on-policy 语义。

    与 infer_prefix 同源（同一 rules.py 规则表），保证单次推断与增量推断结果一致。
    """

    def __init__(self, resolver, is_op):
        self.resolver = resolver
        self.is_op = is_op
        self.reset()

    def reset(self):
        self.values = []       # 已归约子表达式 TypeSpec（后进先出）
        self.frames = []       # 待完成算子帧 [op, remaining_args]
        self.hard_codes = []   # 已触发的 hard 违规码
        self.soft_codes = []   # 已触发的 soft 违规码

    def clone(self):
        c = TypeState(self.resolver, self.is_op)
        c.values = list(self.values)
        c.frames = [list(f) for f in self.frames]
        c.hard_codes = list(self.hard_codes)
        c.soft_codes = list(self.soft_codes)
        return c

    def step(self, token):
        """推进一个 token（叶子或算子），归约并累积违规码。"""
        if not self.is_op(token):
            self._emit(self.resolver(token))
        else:
            arity = 2 if token in BINARY_ARITH else 1
            self.frames.append([token, arity])

    def _emit(self, spec):
        self.values.append(spec)
        while self.frames:
            op, remaining = self.frames[-1]
            remaining -= 1
            self.frames[-1][1] = remaining
            if remaining > 0:
                break
            self.frames.pop()
            arity = 2 if op in BINARY_ARITH else 1
            args = [self.values.pop() for _ in range(arity)]
            args.reverse()
            check, output = rule_for(op)
            # 时序算子的 check 签名是 (op, form, is_leaf)，与 (op, *specs) 不同，
            # 且 factor_mining 侧无时序算子，此处跳过（时序算子不走增量掩码）。
            if check is not None and op not in TEMPORAL:
                for v in check(op, *args):
                    (self.hard_codes if v.severity == "hard" else self.soft_codes).append(v.code)
            self.values.append(output(op, *args))

    def has_hard(self):
        return bool(self.hard_codes)

    def result(self):
        """表达式完成（无剩余 frame、恰好一个归约结果）时返回最终 TypeSpec，否则 None。"""
        if not self.frames and len(self.values) == 1:
            return self.values[0]
        return None


def mask_actions(state, candidates):
    """试探式动作掩码：过滤会立即触发 hard 违规的候选 token。

    state: 当前 TypeState（已采样前缀的状态）
    candidates: 候选 token 列表（valid_actions 的槽位/长度合法集）
    返回：类型也合法的候选子集（与 valid_actions 取交集后的最终合法动作集）。
    """
    return [tok for tok in candidates if not _trial_hard(state, tok)]


def _trial_hard(state, token):
    trial = state.clone()
    trial.step(token)
    return trial.has_hard()
