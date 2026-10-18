# -*- coding: utf-8 -*-
"""Forge DSL -> pandas 统一翻译层 (P-20260918-004).

单一真源 (single source of truth)
====================================================================
本模块是**唯一**的 Forge 前缀 DSL 翻译实现 + S5 终端白名单定义处, 供两条通道共用:

  1. ``ralph_loop._translate_seed_formula``  -- 种子重检 / 门禁回填 (原实现内联在
     ralph_loop, 白名单亦定义在彼);
  2. ``factor_ic_computer._normalize_to_pandas`` -- **S1 IC / S2 库相关 / S2 候选
     相关 / 行为同质性门禁** 四条 eval 路径唯一的翻译入口。

🔴 为什么必须合并 (本轮病灶)
--------------------------------------------------------------------
两条通道此前各有一套翻译器, **覆盖范围不一致**:

  · ``dsl_to_pandas_infix`` (factor_expression_tree) 只认正则白名单里的原语
    (ts_mean|ts_std|ts_rank|ts_sum|neg|add|sub|mul|div|...), **不含** ``inv`` /
    ``square`` / ``ts_ema_decay`` / ``ts_pct`` 等;
  · Forge 表达式树 (``forge.expression.parse_expression`` + ``to_pandas_string``)
    覆盖**全部 31 个 Forge 原语**, 但此前只有 ralph 的种子重检在用。

后果是**静默零覆盖** (不报错, 只是没有面板):

    inv(ts_rank(volume, 10))
      -> _normalize_to_pandas 正则不匹配 -> 原样返回
      -> dsl_to_pandas_infix 只吐出裸 ``inv`` (不可求值)
      -> _eval_formula: NameError: name 'inv' is not defined -> None
      -> 该公式在 S1/S2/门禁 三处**都没有面板**

实测 (``_diag_forge_translate_20260918.py``, 全市场 5587 只):

    inv(ts_rank(volume, 10))   原样=None   翻译后=OK len=941400
    inv(ts_rank(volume, 20))   原样=None   翻译后=OK len=882051
    neg(ts_rank(volume, 20))   原样=OK     翻译后=OK len=882051   (本就可翻译)
    close_p / close_p.rolling(20).mean()   原样=OK  翻译后=(无需翻译)

⇒ 修法 = 把 Forge 表达式树翻译器**下沉为 eval 层兜底**, 而不是在门禁侧做事后
回填 (P-20260917-025 的回填是安全网, 受 ``FUXI_BH_LIB_BACKFILL_MAX`` 上限约束,
且对 ``_seed_recheck`` 候选豁免 ⇒ 覆盖永远是部分的)。

为什么用「AST 标识符校验」而不是「Forge 关键字黑名单」
--------------------------------------------------------------------
黑名单是**锚在命名**上的判据 (``inv(`` / ``ts_mean(`` 之类的正则), 必然漏。
本项目铁律: 判据须锚在**实现**。故本模块统一用 ``find_unknown_identifiers``
(ast 遍历, 对照 S5 eval 上下文真正提供的名字集合) 判定「翻译结果可不可求值」:
它就是 eval 环境自身的可见性检查, 不会漏。

零回归保证
--------------------------------------------------------------------
``translate()`` 只被**兜底调用** (见 ``factor_ic_computer._normalize_to_pandas``):
现有通道能产出「可求值结果」时一律原样返回, 故既有读数逐位不变。仅当现有通道
产出「空 / 不可求值」时才启用本层 ⇒ 只增覆盖, 不改已有语义。
"""
import re
from typing import List, Optional, Tuple

# ═══════════════════════════════════════════════════════════
# S5 eval 上下文的可见名字集合 (单一真源)
# ═══════════════════════════════════════════════════════════

# 与 s5_joint_filter._build_series_context / factor_ic_computer 的宽表字段对齐
S5_KNOWN_TERMINALS = {
    # 价格/量 (含无 _p 别名)
    "close", "open", "high", "low", "volume", "amount",
    "close_p", "open_p", "high_p", "low_p", "volume_p", "amount_p",
    # 衍生
    "returns", "returns_p", "overnight", "overnight_p",
    "amplitude", "amplitude_p", "turnover", "turnover_p",
    "hl_ratio", "hl_ratio_p",
    # v0.6 moneyflow
    "buy_lg_vol", "buy_lg_vol_p", "sell_lg_vol", "sell_lg_vol_p",
    "buy_sm_vol", "buy_sm_vol_p", "sell_sm_vol", "sell_sm_vol_p",
    "buy_md_vol", "buy_md_vol_p", "sell_md_vol", "sell_md_vol_p",
    "buy_elg_vol", "buy_elg_vol_p", "sell_elg_vol", "sell_elg_vol_p",
    "net_mf_vol", "net_mf_vol_p", "net_mf_amount", "net_mf_amount_p",
    # v0.9 P-20260814-001: 两融字段 (margin_detail.csv 接入)
    "rzye", "rzye_p", "rqye", "rqye_p", "rzmre", "rzmre_p",
    "rqyl", "rqyl_p", "rqchl", "rqchl_p",
    # v0.9 P-20260812-026: 龙虎榜字段 (top_list/top_inst 接入)
    "lhb_flag", "lhb_flag_p", "lhb_net_amount", "lhb_net_amount_p",
    "lhb_net_rate", "lhb_net_rate_p", "lhb_amount", "lhb_amount_p",
    "lhb_inst_net_buy", "lhb_inst_net_buy_p",
    "lhb_inst_buy", "lhb_inst_buy_p", "lhb_inst_sell", "lhb_inst_sell_p",
    # 🔴 P-20260918-001 (2026-09-18): 席位三路通道向 factor_ic_computer/S5 对齐。
    # 本集合 docstring 自称「与 s5_joint_filter._build_series_context /
    # factor_ic_computer 的宽表字段对齐」, 但实测**三方已漂移**:
    #   - 原集合缺 lhb_north_* (factor_ic_computer 自 v0.9.2 就有) 与 lhb_yz_*;
    #   - s5_joint_filter 此前连 inst 都没有 (P-20260918-001 已补)。
    # 缺项的后果 = find_unknown_identifiers 把**实际可求值**的公式判为不可求值,
    # 属 fail-closed 假阴性 (挡住合法因子被 forge 翻译), 与「判据锚在实现」铁律相悖。
    "lhb_north_net_buy", "lhb_north_net_buy_p",
    "lhb_north_buy", "lhb_north_buy_p", "lhb_north_sell", "lhb_north_sell_p",
    "lhb_yz_net_buy", "lhb_yz_net_buy_p",
    "lhb_yz_buy", "lhb_yz_buy_p", "lhb_yz_sell", "lhb_yz_sell_p",
    # v0.9 P-20260814-002: 北向持股字段 (hk_hold SH+SZ 接入, 2018-2024 历史窗口)
    "north_vol", "north_vol_p", "north_ratio", "north_ratio_p",
    # v0.6 balancesheet
    "intan_assets", "intan_assets_p", "goodwill", "goodwill_p",
    "total_assets", "total_assets_p",
}

# S5 eval 上下文中可用的函数名 (base_context + safe_builtins 并集)
S5_KNOWN_FUNCS = {
    "np", "pd",
    "abs", "sqrt", "log", "log1p", "exp", "sign",
    "maximum", "minimum", "where", "clip",
    "range", "len", "int", "float", "list", "dict", "tuple", "str", "bool",
    "min", "max", "round", "sum", "zip", "sorted", "reversed",
    "enumerate", "map", "filter", "print", "isinstance",
    "Exception", "ValueError", "TypeError",
}

# ── 遗留 pandas 函数式调用 → 方法链 (rolling_min(x, w) → (x).rolling(w).min()) ──
_LEGACY_ROLLING_METHODS = {
    "rolling_min": "min",
    "rolling_max": "max",
    "rolling_mean": "mean",
    "rolling_std": "std",
    "rolling_sum": "sum",
}

_LEGACY_ROLLING_RE = re.compile(
    r'\b(rolling_min|rolling_max|rolling_mean|rolling_std|rolling_sum)'
    r'\s*\(\s*([^,()]+)\s*,\s*(\d+)\s*\)'
)

# v0.6.2 (2026-08-29): 单参数省略字段形式 rolling_min(60) → (close).rolling(60).min()
_LEGACY_ROLLING_IMPLICIT_RE = re.compile(
    r'\b(rolling_min|rolling_max|rolling_mean|rolling_std|rolling_sum)'
    r'\s*\(\s*(\d+)\s*\)'
)


# ═══════════════════════════════════════════════════════════
# 基础工具
# ═══════════════════════════════════════════════════════════

def is_numeric_const(name: str) -> bool:
    """Forge 常量终端判断 (数字/科学计数/负数)。"""
    try:
        float(name)
        return True
    except (ValueError, TypeError):
        return False


def normalize_legacy_rolling(formula: str) -> str:
    """v0.9: 遗留函数式调用 → 方法链。

    rolling_min(close, 60)  -> (close).rolling(60).min()
    rolling_min(60)         -> (close).rolling(60).min()   # 省略字段=close
    """
    def _rep(m):
        fn, x, w = m.group(1), m.group(2).strip(), m.group(3)
        return "(%s).rolling(%s).%s()" % (x, w, _LEGACY_ROLLING_METHODS[fn])
    formula = _LEGACY_ROLLING_RE.sub(_rep, formula)

    def _rep_imp(m):
        fn, w = m.group(1), m.group(2)
        return "(close).rolling(%s).%s()" % (w, _LEGACY_ROLLING_METHODS[fn])
    return _LEGACY_ROLLING_IMPLICIT_RE.sub(_rep_imp, formula)


def canonicalize_neg_inline(formula: str) -> str:
    """v0.9: 内联负号规范化 — Forge 解析器不识别 ``-ts_min(x, 5)`` 参数形式,
    统一改写为 ``neg(ts_min(x, 5))`` (仅在 Forge 解析路径使用, 不影响原式)。

    实现: 扫描式匹配 ``-`` 紧邻 标识符``(`` 的模式, 用括号配对找到参数闭合位置,
    整体替换为 neg(<fn>(<args>)) 保证括号平衡。
    中缀减号 (close - x) 与负常数 (-0.5) 不受影响。
    """
    out = []
    i, n = 0, len(formula)
    while i < n:
        ch = formula[i]
        if ch == '-' and i + 1 < n and not (i > 0 and formula[i - 1] in "0123456789."):
            m = re.match(r'\s*([A-Za-z_]\w*)\s*\(', formula[i + 1:])
            if m:
                fn = m.group(1)
                open_pos = i + 1 + m.end() - 1  # '(' 下标
                depth, j = 1, open_pos + 1
                while j < n and depth > 0:
                    if formula[j] == '(':
                        depth += 1
                    elif formula[j] == ')':
                        depth -= 1
                    j += 1
                if depth == 0:
                    inner = formula[open_pos + 1:j - 1]
                    out.append("neg(%s(%s))" % (fn, inner))
                    i = j
                    continue
        out.append(ch)
        i += 1
    return "".join(out)


def find_unknown_identifiers(
    formula: str,
    known_terminals: Optional[set] = None,
    known_funcs: Optional[set] = None,
) -> List[str]:
    """AST 遍历找出 S5 eval 上下文中不存在的标识符 (**判据锚在实现**)。

    这是本模块判定「翻译结果可否求值」的统一标准 —— 它直接对照 eval 环境真正
    提供的名字集合, 而不是猜 Forge 关键字长什么样。

    Returns:
        ["<语法错误>"]  — 表达式本身非法
        [...]           — 未知标识符列表 (已排除 np/pd 属性链与 builtins)
        []              — 全部可解析 (可求值)
    """
    import ast
    import builtins
    try:
        tree = ast.parse(formula, mode="eval")
    except SyntaxError:
        return ["<语法错误>"]
    allowed = (known_terminals if known_terminals is not None
               else S5_KNOWN_TERMINALS) | (known_funcs if known_funcs is not None
                                           else S5_KNOWN_FUNCS)
    allowed |= {"factor", "result"}
    allowed |= set(dir(builtins))
    unknown = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            if node.id not in allowed:
                unknown.add(node.id)
    return sorted(unknown)


def is_evaluable(formula: str,
                 known_terminals: Optional[set] = None,
                 known_funcs: Optional[set] = None) -> bool:
    """翻译结果是否一定不会因「未定义名字」而 NameError。"""
    if not formula:
        return False
    return not find_unknown_identifiers(formula, known_terminals, known_funcs)


# ═══════════════════════════════════════════════════════════
# Forge 表达式树翻译 (覆盖全部 31 个 Forge 原语)
# ═══════════════════════════════════════════════════════════

def translate_forge(
    formula: str,
    known_terminals: Optional[set] = None,
) -> Tuple[str, str]:
    """Forge 前缀 DSL -> pandas 字符串 (表达式树路径)。

    Returns:
        (pandas_str, reason)
          - pandas_str 非空 -> 成功, reason=""
          - pandas_str==""  -> 失败, reason 说明原因 (不静默)
    """
    f = (formula or "").strip()
    if not f:
        return "", "空公式"
    kt = known_terminals if known_terminals is not None else S5_KNOWN_TERMINALS
    try:
        from forge.expression import parse_expression, EXTRA_TERMINALS
        # 注册 S5 已知终端 (修复 lhb_flag/pe_ttm 等 parse_expression 返回 None)
        EXTRA_TERMINALS |= kt
        tree = parse_expression(canonicalize_neg_inline(f))
        if tree is None:
            return "", "Forge 解析返回 None"
        terms = [t for t in tree.terminals() if not is_numeric_const(t)]
        unknown = [t for t in terms if t not in kt]
        if unknown:
            return "", "Forge 未知终端: " + ", ".join(sorted(set(unknown)))
        px = tree.to_pandas_string()
        if not px:
            return "", "Forge 表达式树翻译为空"
        return px, ""
    except Exception as e:  # noqa: BLE001 — 翻译失败必须回传原因, 不吞
        return "", "%s: %s" % (type(e).__name__, e)


# ═══════════════════════════════════════════════════════════
# 统一入口
# ═══════════════════════════════════════════════════════════

def translate(
    formula: str,
    known_terminals: Optional[set] = None,
    known_funcs: Optional[set] = None,
) -> Tuple[str, str]:
    """**统一翻译入口**: 任意公式 -> 可求值的 pandas 字符串。

    与 ``ralph_loop._translate_seed_formula`` 同构 (后者是它的历史副本), 按序尝试:

      路径 1  Forge 表达式树 (``parse_expression`` + ``to_pandas_string()``)
      路径 2  ``dsl_to_pandas_infix`` 正则转换 (覆盖 Forge 没有的 ts_corr 等)
      路径 3  遗留 rolling 规范化 + AST 标识符校验

    每条路径的产物都必须过 ``find_unknown_identifiers`` (即真的可求值) 才被接受,
    ⇒ 不会出现「翻译成功但 eval 仍 NameError」的假成功。

    Returns:
        (pandas_str, reason) — reason 非空表示失败原因。
    """
    f = (formula or "").strip()
    if not f:
        return "", "空公式"
    kt = known_terminals if known_terminals is not None else S5_KNOWN_TERMINALS
    kf = known_funcs if known_funcs is not None else S5_KNOWN_FUNCS

    # ── 路径 1: Forge 表达式树 (覆盖全部 31 原语, 含 inv/square) ──
    px, _why = translate_forge(f, kt)
    if px and is_evaluable(px, kt, kf):
        return px, ""

    # ── 路径 2: dsl_to_pandas_infix (覆盖 ts_corr 等 Forge 无的原语) ──
    try:
        from factor_expression_tree import dsl_to_pandas_infix
        conv = dsl_to_pandas_infix(normalize_legacy_rolling(f))
        if conv and conv != f and is_evaluable(conv, kt, kf):
            return conv, ""
    except Exception:
        pass

    # ── 结构性失败优先报「数据源停更」(非语法问题, 上游据此跳过重试) ──
    # 顺序与 ralph_loop._translate_seed_formula 一致 (北向先于通用未知标识符)。
    _north = set(re.findall(r'(?<![0-9.])[A-Za-z_]\w*', f)) & {
        "north_money", "south_money", "hgt", "sgt"}
    if _north:
        return "", ("北向资金数据已停更(hk_hold), S5 无该终端: "
                    + ", ".join(sorted(_north)))

    # ── 路径 3: 遗留 rolling + AST 标识符校验 ──
    norm = normalize_legacy_rolling(f)
    unk = find_unknown_identifiers(norm, kt, kf)
    if not unk:
        return norm, ""
    if unk == ["<语法错误>"]:
        return "", "无法解析 (非 Forge 前缀亦非合法 Python 表达式)"
    return "", "未知标识符: " + ", ".join(unk)
