"""
LLM 因子生成器: Experience Memory 反馈驱动的 LLM 因子生成。
在统一闭环中作为 GP 育种的补充生成器。
"""
import json
import hashlib
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple


_UNSET = object()
from datetime import datetime

DATA_DIR = Path("e:/quant/data")
OUTPUT_DIR = Path("e:/quant/output")

# P-20260821-001: AQuA 密封研究契约开关 (rollback: 置 False 即回退裸 prompt)
_RESEARCH_CONTRACT_ENABLED = True


# ═══════════════════════════════════════════════════════════════
# P-20260915-006: LLM 生成端强约束 schema (函数白名单 + 变量白名单)
# ═══════════════════════════════════════════════════════════════
# 来源: QuantMind / RD-Agent CoSTEER `prompts_alphaagent.yaml` 的硬编码算符表
#       (允许算子 ~60 个 + 变量仅 6 个)。
#
# 🔴 白名单构造原则 (不拍脑袋): **契约允许的变量 ∩ 库内已验证可运行的算子**。
#    - 变量: 与研究契约 (求值上下文仅五变量 + 可选资金流) 完全一致 —— 不改求值语义;
#    - 算子: 全部取自库内 923 因子实测使用过的算子 (即已被 evaluator 证明可运行),
#      **不引入任何未经库内验证的新算子** ⇒ 只会减少无效输出, 不引入新失败模式。
#    实测词表 (2026-09-15, 923 因子): rolling 1950 / mean 1175 / pct_change 937 /
#      rank 637 / std 376 / clip 325 / shift 302 / sum 276 / abs 220 / replace 209 /
#      fillna 177 / astype 159 / mul 120 / min 108 / max 100 / diff 69 / np.log 62 /
#      sub 56 / where 45 / np.sqrt 38 / div 37 / mod 35 / ewm 31 / round 30 / corr 29 /
#      add 21 / apply 20 / pow 19 / np.sign 18 / np.exp 14 / cov 9 / reshape 7 /
#      var 6 / median 6 / ffill 5 / skew 5 / isin 5 / cumsum 5
#
# 开关: FUXI_LLM_SCHEMA_CONSTRAINT = on (默认) | off (回退到 v0.7 的弱语法说明块)

LLM_SCHEMA_CORE_VARS = ["close", "open", "high", "low", "volume"]
LLM_SCHEMA_MONEYFLOW_VARS = [
    "buy_lg_vol", "sell_lg_vol", "buy_md_vol", "sell_md_vol",
    "buy_sm_vol", "sell_sm_vol", "buy_elg_vol", "sell_elg_vol", "net_mf_vol",
]
LLM_SCHEMA_WEEKLY_EXTRA_VARS = ["amount"]      # 仅周频宽表提供

LLM_SCHEMA_SERIES_METHODS = [
    "rolling(N).mean()", "rolling(N).std()", "rolling(N).sum()",
    "rolling(N).min()", "rolling(N).max()", "rolling(N).median()",
    "rolling(N).var()", "rolling(N).skew()", "rolling(N).corr(other)",
    "rolling(N).cov(other)", "rolling(N).rank()",
    "ewm(span=N).mean()", "expanding().mean()",
    "shift(N)", "diff(N)", "pct_change(N)",
    "rank(pct=True, axis=1)", "clip(lower=,upper=)", "replace(0, x)",
    "fillna(x)", "ffill()", "astype(float)", "round()", "abs()", "pow(N)",
    "cumsum()", "where(cond, x)", "isin([...])",
    "add(x)", "sub(x)", "mul(x)", "div(x)",
]

LLM_SCHEMA_NP_FUNCS = [
    "np.log", "np.log1p", "np.abs", "np.sqrt", "np.exp", "np.sign",
    "np.where", "np.maximum", "np.minimum", "np.nan_to_num", "np.isnan",
]

LLM_SCHEMA_OPERATORS = ["+", "-", "*", "/", "**", "%", ">", "<", ">=", "<=", "==", "!=", "&", "|", "~"]

MODULE_NAMES_LOWER = {"np", "numpy", "pd", "pandas", "math"}

# ── 变量别名 (🔴 已核对求值环境, 非猜测) ──────────────────────
# `factor_ic_computer.py:848-867` 的 `field_cols` 绑定的是 **_p 后缀名**
# (close_p/open_p/...), 且 `suffix_map` 把裸名 close→close_p 打通
# ⇒ `close` 与 `close_p` 在求值环境中**等价**, 两者都必须被白名单接受,
#   否则校验器会把库内主流写法误报为违规。
LLM_SCHEMA_DERIVED_VARS = ["overnight", "amplitude", "returns", "turnover", "hl_ratio"]

# 契约明确排除、但库内实际已在使用的字段族 (校验器单独归类, 便于量化"契约口径 vs 库内词汇"的差集)
LLM_SCHEMA_CONTRACT_EXCLUDED_PREFIXES = (
    "amount", "lhb_", "north_", "rz", "rq", "kd_", "ind_", "industry_",
    "eps", "roe", "pe", "pb", "circ_mv", "netprofit", "limit_up",
)


def _schema_accept_set(weekly_mode: bool) -> set:
    """白名单接受集: 契约变量 + 量价衍生 + 资金流, 并展开 `_p` 别名。"""
    base = set(_schema_vars(weekly_mode)) | set(LLM_SCHEMA_DERIVED_VARS)
    if not weekly_mode:
        base |= set(LLM_SCHEMA_MONEYFLOW_VARS)
    base |= {f + "_p" for f in list(base)}
    base |= MODULE_NAMES_LOWER | {"axis", "pct", "lower", "upper", "min_periods",
                                   "span", "other", "index", "columns", "values",
                                   "float", "int", "bool", "str", "true", "false",
                                   "none", "nan", "inf", "x"}
    return base

# 🔴 前视/语义陷阱 —— 生成端必须显式禁用
LLM_SCHEMA_FORBIDDEN = [
    ("原始价格/成交量量级的绝对值参与比较或作被除数",
     "先转相对变化或标准化: (close - close.rolling(20).mean()) / (close.rolling(20).std() + 1e-8)"),
    ("分母不做保护",
     "分母一律加 1e-8, 或用 .replace(0, np.nan) / .clip(lower=1e-8)"),
    ("未来函数: shift(-N) / diff(-N) / pct_change(-N) / rolling(center=True) / bfill()",
     "只用 shift(N) (N>0) 等历史方向算子; 禁 center=True"),
    ("if/else 与 np.where 的状态切换 (regime 切换族 JQ 多次失败 -79.5%)",
     "改成连续函数 (rank/zscore/连续权重), 不做硬开关"),
    ("forge DSL 语法 (ts_mean / ts_std / div / sub / neg / rank(x))",
     "一律用 pandas Series 方法链"),
    ("字段未在下面白名单中 (含基本面/自定义字段)", "只用白名单变量 + 量价衍生"),
    ("一个因子引用同批另一个因子的名称", "每条因子必须自包含, 不得互相引用"),
]


def _schema_constraint_enabled() -> bool:
    """P-20260915-006 开关: 默认 on; 置 off 回退到 v0.7 的弱语法说明块。"""
    v = (os.environ.get("FUXI_LLM_SCHEMA_CONSTRAINT", "on") or "on").strip().lower()
    return v not in ("off", "0", "false", "no", "disable", "disabled")


def _schema_vars(weekly_mode: bool):
    vs = list(LLM_SCHEMA_CORE_VARS)
    if weekly_mode:
        vs += LLM_SCHEMA_WEEKLY_EXTRA_VARS
    return vs


def build_schema_constraint_block(weekly_mode: bool = False, n_factors: int = 5) -> str:
    """P-20260915-006: 生成端强约束块 (函数白名单 + 变量白名单 + 前视陷阱)。"""
    lines = []
    lines.append("## 🔒 生成端强约束 schema (白名单制 — 违反者会被预检直接拦截)")
    lines.append("")
    lines.append("**① 变量白名单 (只能用这些, 其余一律不合格)**:")
    lines.append("   - 核心: " + " ".join(LLM_SCHEMA_CORE_VARS))
    if weekly_mode:
        lines.append("   - 周频额外可用: " + " ".join(LLM_SCHEMA_WEEKLY_EXTRA_VARS))
        lines.append("   - 资金流字段**周频不可用**")
    else:
        lines.append("   - 可选资金流: " + " ".join(LLM_SCHEMA_MONEYFLOW_VARS))
        lines.append("   - ⚠️ 日频**无 amount、无基本面字段** (roe/eps/pb 等一律不可用)")
    lines.append("")
    lines.append("**② 算子/方法白名单 (只允许下列写法)**:")
    for i in range(0, len(LLM_SCHEMA_SERIES_METHODS), 6):
        lines.append("   - " + "  ".join(LLM_SCHEMA_SERIES_METHODS[i:i + 6]))
    lines.append("   - 数学函数: " + "  ".join(LLM_SCHEMA_NP_FUNCS))
    lines.append("   - 运算符: " + " ".join(LLM_SCHEMA_OPERATORS))
    lines.append("")
    lines.append("**③ 量纲与数值稳定 (硬要求)**:")
    lines.append("   - 禁止把原始价格/成交量参与比较或作被除数 —— 必须先转相对变化或标准化")
    lines.append("     正确: `(close - close.rolling(20).mean()) / (close.rolling(20).std() + 1e-8)`")
    lines.append("     错误: `close / volume`、`close > 10`")
    lines.append("   - 每个分母都要加 `1e-8`, 或 `.replace(0, np.nan)` / `.clip(lower=1e-8)`")
    lines.append("")
    lines.append("**④ 前视陷阱 (踩中即废, 系统会扫描)**")
    for bad, fix in LLM_SCHEMA_FORBIDDEN:
        lines.append(f"   - ❌ {bad} ⇒ ✅ {fix}")
    lines.append("")
    lines.append(f"**⑤ 批量要求**: 本批 {n_factors} 条因子之间**不得互相引用**, 每条自包含。")
    lines.append("")
    lines.append("⚠️ 参考 (合法写法):")
    lines.append("   - `(close - close.rolling(60).min()) / (close.rolling(60).min() + 1e-8)`")
    lines.append("   - `-((high - low) / (volume.rolling(20).mean() + 1e-8)).rank(pct=True)`")
    lines.append("   - `(close.pct_change(20).rank(pct=True, axis=1) * (1 - ((close.pct_change(20) - close.pct_change(20).rolling(20).mean()) / (close.pct_change(20).rolling(20).std() + 1e-8)).abs().clip(0, 1)))`")
    return "\n".join(lines)


def validate_llm_schema(expression: str, weekly_mode: bool = False) -> list:
    """P-20260915-006: 对生成结果做白名单/前视校验, 返回违规项列表 (空 = 通过)。

    纯观测用 (生成端自检 + 违规率统计), **不参与任何裁决**。
    """
    viol = []
    e = (expression or "").strip()
    if not e:
        return ["空表达式"]
    low = e.lower()

    # 1) 前视模式
    if re.search(r"shift\s*\(\s*-", low):
        viol.append("前视: shift 负偏移")
    if re.search(r"(diff|pct_change)\s*\(\s*-", low):
        viol.append("前视: diff/pct_change 负偏移")
    if "center=true" in low.replace(" ", ""):
        viol.append("前视: rolling(center=True)")
    if re.search(r"\.bfill\s*\(", low):
        viol.append("前视: bfill()")

    # 2) forge DSL 残留
    if re.search(r"(?<![.\w])(ts_mean|ts_std|ts_min|ts_max|ts_sum|ts_rank|ts_corr|ts_delta|ts_delay|"
                 r"ts_decay_linear|div|mul|sub|add|neg)\s*\(", low):
        viol.append("语法: forge DSL 残留")
    if re.search(r"(^|[^.\w])rank\s*\(", low) and ".rank(" not in low:
        viol.append("语法: 裸 rank(x) (须 .rank(pct=True))")

    # 3) regime 硬切换
    if re.search(r"\bif\b.*\belse\b", low) and "if_else" not in low:
        viol.append("语义: if/else 状态切换")
    if "np.where" in low:
        viol.append("语义: np.where 状态切换 (仅允许用于数值保护)")

    # 4) 分母保护 (粗判: 含除法但全文无 1e-8/clip/replace/nan)
    if "/" in e and not re.search(r"1e-8|1e-08|clip|replace|nan_to_num|np\.nan", low):
        viol.append("数值: 含除法但未见分母保护 (1e-8/clip/replace)")

    # 5) 变量白名单 (只取"值位置"标识符: 排除方法名/关键字参数/模块成员)
    _METHODS = {m.split("(")[0].split(".")[-1].lower().replace("(n)", "")
                for m in LLM_SCHEMA_SERIES_METHODS}
    _METHODS |= {"rolling", "ewm", "expanding", "mean", "std", "sum", "min", "max",
                 "median", "var", "skew", "corr", "cov", "rank", "shift", "diff",
                 "pct_change", "clip", "replace", "fillna", "ffill", "astype", "round",
                 "abs", "pow", "cumsum", "where", "isin", "add", "sub", "mul", "div"}
    _METHODS |= {f.split(".")[-1] for f in LLM_SCHEMA_NP_FUNCS}
    _KW = {"true", "false", "none", "nan", "inf", "and", "or", "not", "in", "is"}
    allowed = _schema_accept_set(weekly_mode) | _METHODS | _KW
    seen_bad = []
    seen_excluded = []
    for m in re.finditer(r"(?<![\w.])([a-zA-Z_][a-zA-Z_0-9]*)", e):
        nm = m.group(1)
        n = nm.lower()
        rest = e[m.end():].lstrip()
        if rest.startswith("="):          # 关键字参数名
            continue
        if rest.startswith("("):          # 裸函数调用名 (forge DSL 已单独判)
            continue
        if n in allowed:
            continue
        if len(n) <= 1:
            continue
        if any(n.startswith(p) for p in LLM_SCHEMA_CONTRACT_EXCLUDED_PREFIXES):
            if nm not in seen_excluded:
                seen_excluded.append(nm)
        elif nm not in seen_bad:
            seen_bad.append(nm)
    for nm in seen_bad[:4]:
        viol.append(f"变量: 非白名单字段 `{nm}`")
    for nm in seen_excluded[:4]:
        viol.append(f"变量: 契约外字段 `{nm}` (库内已在用, 属口径差集非错误)")
    return viol


def _schema_system_prompt(weekly_mode: bool = False) -> str:
    """P-20260915-006: API 模式的 system prompt (强约束摘要)。"""
    base = ("你是 A 股量化因子研究员，专精于设计具有经济逻辑支撑的因子表达式。"
            "使用 pandas 风格表达式（如 close.rolling(20).mean()）。")
    if not _schema_constraint_enabled():
        return base
    vs = " ".join(_schema_vars(weekly_mode))
    return (base
            + f" 硬约束: 变量仅限 [{vs}]; "
            + "只用 pandas Series 方法链 (rolling/ewm/shift/diff/pct_change/rank/clip/"
              "replace/fillna/astype) 与 np.log/np.abs/np.sign/np.sqrt/np.exp/np.maximum/np.minimum; "
            + "禁把原始价格量级直接比较或作被除数 (先转相对变化或标准化); "
            + "每个分母加 1e-8; 禁 shift(-N)/diff(-N)/rolling(center=True)/bfill(); "
            + "禁 if-else 与 np.where 状态切换; 禁 forge ts_* 语法。")


def schema_violation_summary(candidates: list, weekly_mode: bool = False) -> str:
    """P-20260915-006: 生成结果的 schema 违规率读数 (观测用)。"""
    if not candidates:
        return ""
    n_bad = 0
    top = []
    for c in candidates:
        expr = c.get("formula") or c.get("expression") or ""
        v = validate_llm_schema(expr, weekly_mode=weekly_mode)
        if v:
            n_bad += 1
            if len(top) < 3:
                top.append(f"{c.get('factor_name') or c.get('name') or '?'}: {v[0]}")
    pct = n_bad / len(candidates) * 100
    msg = f"[LLMGen] schema 违规 {n_bad}/{len(candidates)} ({pct:.0f}%)"
    if top:
        msg += " | 例: " + "; ".join(top)
    return msg


def _load_v06_flag(key: str, default=False) -> bool:
    """读取 V06_EXPERIMENTAL 开关 (config 导入失败时静默回退默认值)"""
    try:
        from config import V06_EXPERIMENTAL
        return bool(V06_EXPERIMENTAL.get(key, default))
    except Exception:
        return default


def _resolve_failure_feedback(weekly_mode: bool = False) -> str:
    """P-20260915-007: 按 FUXI_STRUCT_FEEDBACK 开关解析失败反馈文本。

    off / shadow → 旧扁平文本 (与 v0.7 行为一致; shadow 额外落盘+打印读数)
    enforce      → 结构化 5 字段块 (Observations/Evaluation/New Hypothesis/Reasoning/Decision)
    模块缺失/异常 → 静默回退旧扁平文本 (不阻塞生成)。
    """
    try:
        import sys as _sys
        from pathlib import Path as _P
        _here = str(_P(__file__).resolve().parent)
        if _here not in _sys.path:
            _sys.path.insert(0, _here)
        from structured_feedback import build_feedback
        return build_feedback()
    except Exception:
        return _extract_recent_failures()


def _extract_recent_failures(limit: int = 3, days: int = 7) -> str:
    """从 experience_memory.attempts 提取近期失败原因 (lazy 回灌, 失败时静默)"""
    try:
        from datetime import timedelta
        em_path = DATA_DIR / "experience_memory.json"
        if not em_path.exists():
            return ""
        with open(em_path, "r", encoding="utf-8") as f:
            em = json.load(f)
        attempts = em.get("attempts", [])
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        fails = []
        for a in reversed(attempts):
            if len(fails) >= limit:
                break
            ts = a.get("timestamp", "")
            if ts < cutoff:
                continue
            outcome = a.get("outcome", "")
            if outcome in ("PASS", "PASSED", "MARGINAL"):
                continue
            name = a.get("factor_name", "?")[:50]
            lessons = a.get("lessons", [])
            lesson_txt = "; ".join(str(l)[:80] for l in lessons[:2]) if lessons else "无归因记录"
            fails.append(f"- {name} ({outcome}): {lesson_txt}")
        return "\n".join(fails)
    except Exception:
        return ""


# ═══════════════════════════════════════════════════════════════
# M7 (2026-09-26): 批内 AST 骨架去重 — 提升为公共阶段
# ═══════════════════════════════════════════════════════════════
# 病灶 (E1): `_batch_dedup_ast` 自 P-20260910-005 起只出现在 generate() 的
#   两条内路径 (L1086/L1115)。Ralph 生产主线逐轮直调 parse_response
#   (ralph_loop.py 首轮 + reviewer 二轮), 从未经过批内去重 ⇒ 同批骨架
#   完全相同的候选照常进入 S1-S5 ⇒ 评估预算重复消耗。
# 语义: 骨架指纹 = subtree_fingerprinter.parse_expression (字段→FIELD, 窗口→
#   N1/N2 按首现顺序编号, 算子别名归一); 结构重复保留先出现者 (保多样性)。
# 开关: FUXI_BATCH_AST_DEDUP = off | shadow | enforce (默认 enforce, 与
#   generate() 内历史无条件去重行为一致)。shadow 只留档不移除, 供误杀率 A/B。
# 留档: 命中的候选全部写入 evidence_out (含指纹/首现者/阶段), 不静默丢弃。
# 范围: 仅批内。reviewer 二轮本质是"同骨架定向改进", 故意不做跨轮去重。

def batch_dedup_mode() -> str:
    """解析 FUXI_BATCH_AST_DEDUP (off|shadow|enforce, 默认 enforce)。

    非法值直接抛 ValueError (fail-fast; 调用方须放在 API 降级 try 之外,
    否则配置错误会被 except 吞掉误报为 api_error)。
    """
    mode = str(os.environ.get("FUXI_BATCH_AST_DEDUP", "enforce") or "enforce").strip().lower()
    if mode not in ("off", "shadow", "enforce"):
        raise ValueError("FUXI_BATCH_AST_DEDUP must be off, shadow or enforce")
    return mode


def dedup_ast_candidates(
    candidates: List[Dict],
    evidence_out: Optional[List[Dict]] = None,
    mode: Optional[str] = None,
    stage: str = "",
) -> List[Dict]:
    """批内 AST 骨架指纹去重 (结构重复保留先出现者, 保多样性)。

    mode=None ⇒ 读取 FUXI_BATCH_AST_DEDUP。
    enforce: 移除重复项; shadow: 只记录不移除; off: 原样返回。
    evidence_out: 每条命中候选追加留档记录 (去重但不删除证据)。
    """
    if mode is None:
        mode = batch_dedup_mode()
    if mode == "off" or len(candidates) < 2:
        return candidates
    try:
        from subtree_fingerprinter import get_fsa
        fsa = get_fsa()
        seen: Dict[str, str] = {}
        out, n_dup = [], 0
        for idx, c in enumerate(candidates):
            expr = c.get("formula", c.get("expression", ""))
            if not expr:
                out.append(c)
                continue
            try:
                fp = fsa.fingerprint(expr)
            except Exception:
                fp = None
            if fp and fp in seen:
                n_dup += 1
                if evidence_out is not None:
                    evidence_out.append({
                        "stage": str(stage or ""),
                        "factor_name": str(c.get("factor_name", c.get("name", "?")))[:80],
                        "formula": str(expr)[:200],
                        "fingerprint": str(fp)[:200],
                        "dup_of": str(seen[fp])[:80],
                        "index": idx,
                        "round2": bool(c.get("_llm_reviewer_round2", False)),
                        "removed": bool(mode == "enforce"),
                    })
                if mode == "enforce":
                    continue
            if fp and fp not in seen:
                seen[fp] = str(c.get("factor_name", c.get("name", "?")))[:80]
            out.append(c)
        if n_dup:
            _suffix = f" (stage={stage})" if stage else ""
            if mode == "enforce":
                print(f"  [LLMGen] 批内 AST 去重: 移除 {n_dup} 条结构重复候选{_suffix}")
            else:
                print(f"  [LLMGen] 批内 AST 去重 (shadow): 命中 {n_dup} 条结构重复候选,"
                      f" 未移除{_suffix}")
        return out
    except Exception as e:
        print(f"  [LLMGen] 批内去重失败 (不阻塞): {e}")
        return candidates


class LLMGenerator:
    """
    基于 Experience Memory 的 LLM 因子生成器。
    
    工作流:
    1. receive_context() 接收 Memory priors (成功模板/禁止方向/FSA/库状态)
    2. build_prompt() 构建结构化 LLM prompt
    3. [手动] 用户将 prompt 提交给 LLM
    4. parse_response() 解析 LLM 响应为候选因子
    5. get_candidates() 返回标准化候选列表
    
    也支持通过 MCP/API 直接调用 (如未来接入 LLM API)。
    """
    
    def __init__(self, output_dir: Path = None):
        self.output_dir = output_dir or OUTPUT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._context = self._empty_context()
        self._semantic_plan = None
        self._last_prompt_file = None
        self._gen_counter = 0
        # P-20260830-001: 周频语境开关 (默认 False = 日频语境, 回退即关闭)
        self._weekly_mode = False
        # P-20260910-007: 范式提案模式开关 (默认关 = 21 范式冻结语义不变)
        self._proposal_mode = False
        self._proposal_hint = ""
        # P-20260912-001: CogAlpha 认知进化提示块 (思维进化 + 自适应生成), 默认不注入
        self._cog_enabled = False
        self._cog_blocks: Dict[str, str] = {}

    # ── P-20260912-001: CogAlpha 认知进化提示块 ───────────

    def set_cog_evolve(self, enabled: bool = False,
                       blocks: Optional[Dict[str, str]] = None) -> None:
        """注入思维进化 / 自适应生成提示块。

        enabled=False 时零注入 (回退裸 prompt)。blocks 形如
        {"thought_evolution": str, "adaptive": str}, 空块自动忽略。
        """
        self._cog_enabled = bool(enabled)
        self._cog_blocks = {k: v for k, v in (blocks or {}).items() if v}

    # ── P-20260910-007: 范式提案模式 ──────────────────────

    def set_proposal_mode(self, enabled: bool = True, hint: str = "") -> None:
        """开启/关闭范式提案模式。开启后 build_prompt 追加一段:
        在常规候选之外 **额外** 生成 1 条偏离既有 21 范式的候选, 第 7 字段 = 新范式名。
        偏离既有范式, 但必须可用本地数据求值 (禁捏造字段)。"""
        self._proposal_mode = bool(enabled)
        self._proposal_hint = hint or ""

    
    # ── P-20260830-001: 周频语境开关 ──────────────────────
    
    def set_weekly_mode(self, weekly: bool = True) -> None:
        """开启/关闭周频生成语境。开启后 build_prompt 注入周频语义约束,
        且 generate() 解析出的候选自动打 natural_freq=weekly 标签 (走 weekly lane)。"""
        self._weekly_mode = bool(weekly)
    
    def set_semantic_plan(self, plan=None):
        """Explicit experimental plan; None restores the historical prompt."""
        if plan is None:
            self._semantic_plan = None
        else:
            try:
                from semantic_plan import validate_plan
            except ImportError:
                from .semantic_plan import validate_plan
            self._semantic_plan = validate_plan(plan)

    def tag_semantic_candidates(self, candidates):
        if self._semantic_plan is None:
            return candidates
        try:
            from semantic_plan import tag_candidates
        except ImportError:
            from .semantic_plan import tag_candidates
        return tag_candidates(candidates, self._semantic_plan)

    # ── 上下文注入 ───────────────────────────────────────

    @staticmethod
    def _empty_context() -> Dict:
        return {
            "success_templates": [],
            "forbidden_directions": [],
            "fsa_forbidden": "",
            "library_stats": {},
            "paradigm_priorities": {},
            "mab_direction": "",
            "recent_performance": {},
            "warning_directions": [],
            "motif_rules": {},
            "rag_knowledge": {},
            "frontier_cards": "",
            "last_round_failures": "",
        }
    
    def receive_context(
        self,
        success_templates: List = _UNSET,
        forbidden_directions: List = _UNSET,
        fsa_forbidden: str = _UNSET,
        library_stats: Dict = _UNSET,
        paradigm_priorities: Dict = _UNSET,
        mab_direction: str = _UNSET,
        recent_performance: Dict = _UNSET,
        # v3.1: 软负收益警告 + Motif 规则
        warning_directions: List = _UNSET,
        motif_rules: Dict = _UNSET,
        # v0.5: RAG 知识注入
        rag_knowledge: Dict = _UNSET,
        # P-20260910-006: 前沿路径卡片 (方向层上下文, 仅灵感)
        frontier_cards: str = _UNSET,
        # P-20260821-001: AQuA 研究契约 — 上轮失败原因回灌
        last_round_failures: str = _UNSET,
        replace: bool = False,
    ):
        """Merge Experience Memory/RAG context; ``replace=True`` clears first.

        R1 (2026-09-22): omitted arguments no longer erase context received by
        earlier calls. Passing an explicit empty list/dict/string still clears
        that one field, which keeps rollback and tests deterministic.
        """
        mode = str(os.environ.get("FUXI_LLM_CONTEXT_MODE", "merge") or "merge").strip().lower()
        if mode not in ("legacy", "merge"):
            raise ValueError("FUXI_LLM_CONTEXT_MODE must be legacy or merge")
        if mode == "legacy":
            replace = True
        if replace:
            self._context = self._empty_context()
        updates = {
            "success_templates": success_templates,
            "forbidden_directions": forbidden_directions,
            "fsa_forbidden": fsa_forbidden,
            "library_stats": library_stats,
            "paradigm_priorities": paradigm_priorities,
            "mab_direction": mab_direction,
            "recent_performance": recent_performance,
            "warning_directions": warning_directions,
            "motif_rules": motif_rules,
            "rag_knowledge": rag_knowledge,
            "frontier_cards": frontier_cards,
            "last_round_failures": last_round_failures,
        }
        for key, value in updates.items():
            if value is not _UNSET:
                self._context[key] = value
    
    # ── Prompt 构建 ──────────────────────────────────────
    
    def build_prompt(self, target_paradigm: str = "", n_factors: int = 5,
                     weekly_mode: Optional[bool] = None) -> str:
        """
        构建 LLM prompt: 包含成功模板、禁止方向、FSA约束、库状态。
        返回可直接发给 LLM 的文本。

        P-20260830-001: weekly_mode=True 时注入周频语义约束 —
        表达式将在周频宽表 (W-FRI 采样) 上 eval, 窗口口径与日频完全不同,
        fwd_ret = 1 周持有; 生成公式走 weekly lane 裁决 (τ_w)。
        """
        if weekly_mode is None:
            weekly_mode = self._weekly_mode
        ctx = self._context
        parts = []
        
        if weekly_mode:
            parts.append("# A股量化因子生成任务 (周频语境 weekly lane)")
            parts.append("")
            parts.append("你是一位专业量化研究员，请基于以下经验记忆生成新的**周频因子**表达式。")
            parts.append("这些因子将在周频数据 (每周最后一个交易日采样, W-FRI) 上求值, ")
            parts.append("持有周期 = 1 周 (本周信号 → 下周收益)。")
            parts.append("")
            parts.append("## 📐 周频语境硬约束 (与日频语义完全不同, 必须遵守)")
            parts.append("1. 窗口含义: rolling(1)=1周, rolling(4)=约1月, rolling(12)=约1季度, rolling(20)=约5个月")
            parts.append("2. 周频求值环境字段: close open high low volume amount (均为周频宽表 Series)")
            parts.append("   - 周频 OHLC: open=周首日开盘, high=周内最高, low=周内最低 (真实周内路径)")
            parts.append("3. 可设计的周频特有结构: 周内路径形态 (如 (close-open)/(high-low) 光头周线), ")
            parts.append("   周度资金累积 (volume/amount 的周频滚动), 周频均值回复 (周收益的 zscore 反转)")
            parts.append("4. 禁止用日频直觉设计窗口: 周频 rolling(20) 已是 5 个月, 勿套用日频 20 日均线语义")
            parts.append("5. 表达式语法仍为 pandas Rolling 风格 (见下方语法要求)")
            parts.append("")
            # P-20260901-004: 周频原生种子 few-shot 参考 (生成侧与裁决侧频率对齐)
            try:
                from weekly_seed_pool import format_prompt_reference
                parts.append("## 🌱 周频原生种子参考 (结构风格示例, 请设计新的而非复制)")
                parts.append(format_prompt_reference(max_n=6))
                parts.append("")
            except Exception:
                pass
        else:
            parts.append("# A股量化因子生成任务")
            parts.append("")
            parts.append("你是一位专业量化研究员，请基于以下经验记忆生成新的因子表达式。")
            parts.append("表达式必须使用 pandas Series 方法链语法，如 close.rolling(20).mean()。")
            parts.append("")
        
        # Plan injection is opt-in; it cannot grant JQ eligibility.
        if self._semantic_plan is not None:
            try:
                from semantic_plan import prompt_block
            except ImportError:
                from .semantic_plan import prompt_block
            if self._semantic_plan["frequency"] != ("weekly" if weekly_mode else "daily"):
                raise ValueError("semantic plan frequency disagrees with prompt lane")
            parts.extend((prompt_block(self._semantic_plan), ""))

        # 目标范式
        if target_paradigm:
            parts.append(f"## 目标探索范式: {target_paradigm}")
            parts.append(f"请优先围绕'{target_paradigm}'范式设计因子。")
        elif ctx.get("mab_direction"):
            parts.append(f"## MAB推荐方向: {ctx['mab_direction']}")
        
        parts.append("")
        
        # 成功模板 (来自 Experience Memory — 曾经有效的模式)
        templates = ctx.get("success_templates", [])
        if templates:
            parts.append("## ✅ 历史成功模板 (建议参考这些模式)")
            parts.append("以下是过去验证有效的因子模式，可以作为变体设计参考:")
            for i, t in enumerate(templates[:10]):
                if isinstance(t, dict):
                    formula = t.get("formula", t.get("expression", ""))
                    desc = t.get("description", t.get("pattern_id", f"template_{i}"))
                    paradigm = t.get("paradigm", "")
                    occ = t.get("occurrence_count", 1)
                    jq_tag = ""
                    if t.get("jq_return"):
                        jq_tag = f" [JQ:{t['jq_return']:+.1f}%]"
                    parts.append(f"- {desc}{jq_tag}: `{formula[:120]}` (范式={paradigm}, occ={occ})")
                elif hasattr(t, 'description'):
                    formula = getattr(t, 'formula', getattr(t, 'expression', ''))
                    desc = getattr(t, 'description', str(t))
                    paradigm = getattr(t, 'pattern_id', '').split('::')[0]
                    occ = getattr(t, 'occurrence_count', 1)
                    jq_tag = ""
                    jq_ret = getattr(t, 'jq_return', None)
                    if jq_ret:
                        jq_tag = f" [JQ:{jq_ret:+.1f}%]"
                    parts.append(f"- {desc}{jq_tag}: `{str(formula)[:120]}` (范式={paradigm}, occ={occ})")
            parts.append("")
        
        # 禁止方向
        forbidden = ctx.get("forbidden_directions", [])
        if forbidden:
            parts.append("## 🚫 硬禁止方向 (绝对不能生成)")
            for d in forbidden[:8]:
                if isinstance(d, dict):
                    parts.append(f"- **{d.get('direction', d.get('name', '?'))}**: {d.get('reason', '')[:150]}")
                elif hasattr(d, 'direction'):
                    parts.append(f"- **{d.direction}**: {getattr(d, 'reason', '')[:150]}")
            parts.append("")
        
        # v3.1: 软负收益警告方向
        warnings = ctx.get("warning_directions", [])
        if warnings:
            parts.append("## ⚠️ 软负收益警告 (倾向避开这些范式)")
            for w in warnings[:5]:
                if isinstance(w, dict):
                    parts.append(f"- {w.get('description', w.get('pattern_id', '?'))[:200]}")
                elif hasattr(w, 'description'):
                    parts.append(f"- {getattr(w, 'description', str(w))[:200]}")
            parts.append("")

        # v3.1: Motif 级别规则 (P2 2026-09-05: 跨引擎降权渲染)
        motif_rules = ctx.get("motif_rules", {})
        if motif_rules:
            forbid_motifs = motif_rules.get("forbid", [])
            prefer_motifs = motif_rules.get("prefer", [])
            forbid_cross = motif_rules.get("forbid_cross", [])
            prefer_cross = motif_rules.get("prefer_cross", [])
            if forbid_motifs:
                parts.append("## 🚫 Motif 禁止规则 (样本量已达标, 硬约束)")
                for r in forbid_motifs[:5]:
                    parts.append(f"- 结构 '{r.get('motif_key', '?')[:60]}': {r.get('reason', '')}")
                parts.append("")
            # 证据门槛: 样本量不足以支撑断言的规则不再以硬约束形式注入, 只作为低置信度
            # 参考登记。0/3 与真实成功率 63% 相容, 把它写成"禁止"会系统性误杀常见结构。
            soft_forbid = motif_rules.get("soft_forbid", [])
            if soft_forbid:
                parts.append("## ⚠️ Motif 低置信度观察 (样本量不足, 仅供参考, 不构成约束)")
                for r in soft_forbid[:5]:
                    parts.append(f"- 结构 '{r.get('motif_key', '?')[:60]}': {r.get('reason', '')}"
                                 " [证据不足, 可自由使用]")
                parts.append("")
            if forbid_cross:
                parts.append("## ⚠️ Motif 软警告 (跨引擎失败模式, 降权参考, 机制不同可忽略)")
                for r in forbid_cross[:5]:
                    parts.append(f"- 结构 '{r.get('motif_key', '?')[:60]}' (来源引擎={r.get('engine', '?')}): {r.get('reason', '')}。该模式在其它生成引擎的搜索空间中失败, 若你的生成机制不同可不受此约束")
                parts.append("")
            if prefer_motifs:
                parts.append("## ✅ Motif 推荐规则 (样本量已达标)")
                for r in prefer_motifs[:5]:
                    parts.append(f"- 结构 '{r.get('motif_key', '?')[:60]}': {r.get('reason', '')}")
                parts.append("")
            soft_prefer = motif_rules.get("soft_prefer", [])
            if soft_prefer:
                parts.append("## ℹ️ Motif 低置信度偏好 (样本量不足, 仅作思路参考)")
                for r in soft_prefer[:5]:
                    parts.append(f"- 结构 '{r.get('motif_key', '?')[:60]}': {r.get('reason', '')}"
                                 " [证据不足]")
                parts.append("")
            if prefer_cross:
                parts.append("## ℹ️ Motif 跨引擎参考偏好 (低权重, 仅作思路参考)")
                for r in prefer_cross[:5]:
                    parts.append(f"- 结构 '{r.get('motif_key', '?')[:60]}' (来源引擎={r.get('engine', '?')}): {r.get('reason', '')}")
                parts.append("")

        # FSA 禁止骨架
        fsa = ctx.get("fsa_forbidden", "")
        if fsa and fsa.strip():
            parts.append("## 🚫 FSA 禁止子树骨架 (避免使用这些结构)")
            parts.append(fsa[:1000])
            parts.append("")
        
        # 库状态
        lib = ctx.get("library_stats", {})
        if lib:
            parts.append(f"## 因子库现状")
            parts.append(f"- 因子总数: {lib.get('total_factors', '?')}")
            parts.append(f"- 因子族数: {lib.get('n_clusters', '?')}")
            red_sea = lib.get("red_sea", {})
            parts.append(f"- Red Sea等级: {red_sea.get('level', '?')} (相关性中位数={red_sea.get('median_correlation', '?')})")
            parts.append("")
        
        # 范式优先级
        priorities = ctx.get("paradigm_priorities", {})
        if priorities:
            parts.append("## 探索优先级")
            if isinstance(priorities, dict):
                for p, pri in priorities.items():
                    parts.append(f"- {p}: {pri}")
            elif isinstance(priorities, list):
                for item in priorities:
                    if not isinstance(item, dict):
                        continue
                    paradigm = item.get("paradigm", "?")
                    priority = item.get("priority", "?")
                    reason = item.get("reason", "")
                    parts.append(f"- {paradigm}: {priority}" + (f" ({reason})" if reason else ""))
            parts.append("")
        
        # 近期表现
        recent = ctx.get("recent_performance", {})
        if recent:
            parts.append("## 近期表现反馈")
            for k, v in recent.items():
                parts.append(f"- {k}: {v}")
            parts.append("")
        
        # v0.5: RAG 知识注入 — 从本地知识库检索相关前沿研究
        rag = ctx.get("rag_knowledge", {})
        if rag and (rag.get("frontier_research") or rag.get("failure_patterns")):
            from rag_context import format_for_prompt
            rag_text = format_for_prompt(rag)
            if rag_text.strip():
                parts.append(rag_text)

        # P-20260910-006: 前沿→路径卡片注入 (三问标注的方向层上下文)
        _fcards = ctx.get("frontier_cards", "")
        if _fcards:
            parts.append("")
            parts.append(_fcards)
            parts.append("(以上卡片仅作方向灵感: 若某方向需要本地没有的数据, 请勿硬造字段; "
                         "优先选择'可数据化 + 可因子化'的卡片。)")

        # L1 解读层: 上轮 JQ 归因避坑提示 (2026-08-14, 软消费)
        # 只影响"怎么生成", 不影响"选谁" — 不碰筛选毒药区
        try:
            from experience_memory import get_memory as _get_mem_l1
            _frag = _get_mem_l1().build_interpretation_prompt_fragment(
                paradigm=target_paradigm,
                limit=5,
            )
            if _frag.strip():
                parts.append(_frag)
        except Exception:
            pass  # 归因不可用时不影响生成

        # ── P-20260821-001: AQuA 密封研究契约 (sealed contract) ──
        # AQuA (arXiv 2608.12841) 实证: 递归自改进核心 = 固定约束声明 +
        # 持久化验证证据回灌。此处把求值环境硬约束 + 上轮失败原因显式注入 prompt。
        # 开关 _RESEARCH_CONTRACT_ENABLED 关闭即回退裸 prompt (rollback)。
        if _RESEARCH_CONTRACT_ENABLED:
            parts.append("## 📜 研究契约 (sealed contract — 固定约束, 违反必被拦截)")
            parts.append("1. 求值上下文字段: " + ("周频 close open high low volume amount；资金流不可用，无基本面" if weekly_mode else "日频 close open high low volume (+ 可选资金流字段)；无 amount/无基本面"))
            parts.append("2. 方向一律 long; 空头信号用表达式整体负号实现 (系统按负号取反)")
            parts.append("3. .rank(pct=True) 语义 = 横截面排名 (系统强制 axis=1), 勿依赖时间轴排名")
            parts.append("4. 窗口口径: " + (
                "周频数据对齐 — rolling(1)=1周, rolling(4)=1月, rolling(12)=1季度, rolling(20)=5个月; 回测频率周频(1周持有)"
                if weekly_mode else
                "日频对齐: rolling(5)=1周, rolling(20)=1月; 回测频率日频(fwd5 持有)"))
            parts.append("5. 禁止 if-else/where 状态切换表达式 (regime 切换族多次 JQ 失败 -79.5%)")
            parts.append("")
            _fail_note = ctx.get("last_round_failures", "")
            if not _fail_note:
                # P-20260915-007: 结构化失败反馈 (FUXI_STRUCT_FEEDBACK=off|shadow|enforce)
                _fail_note = _resolve_failure_feedback(weekly_mode=weekly_mode)
            if _fail_note.strip():
                if not _fail_note.lstrip().startswith("#"):
                    parts.append("## 🔁 上轮失败原因回灌 (避免重复犯错)")
                parts.append(_fail_note)
                parts.append("")

        # ── P1-3 (v0.6 实验): 评价宪法约束注入 ──
        # 借鉴 LLM_QUANT_FACTORY 评价宪法: 让生成端提前知晓最终评价纪律
        # (增量边际 > 绝对IC、行为冗余降级、方向纪律、holdout 不透明);
        # D5(2026-09-26): 只写系统真正执行的承诺(移除"冻结冷却/尝试预算")。
        # 开关 V06_EXPERIMENTAL["g_prompt_constitution"] 关闭即回退 v0.5.2 prompt (rollback)。
        if _load_v06_flag("g_prompt_constitution"):
            parts.append("## 📜 评价宪法约束 (决定因子生死 — 请按此纪律设计)")
            parts.append("1. 【增量边际】新因子以'加入组合后的净提升'为准: 高 IC 但组合增量≈0 的因子会被拒绝; 优先设计与现有因子机制正交的信号")
            parts.append("2. 【行为冗余】与库内因子行为高度相似的变体会被标记为替代品/近似重复并降级; 同一经济机制只保留一个, 变体须改变机制而非微调参数")
            parts.append("3. 【方向纪律】每个探索方向有尝试记录(供评审追溯); 已被证伪/明确放弃的方向不要变相重试")
            parts.append("4. 【密封盲评】存在最终盲评验证区 (其划分对生成端不可见), 只有类别结论回传; 任何针对验证集调参的尝试都会被污染登记拦截")
            parts.append("5. 【可解释优先】假设必须说清经济机制, 纯统计拟合优先被淘汰")
            parts.append("")

        # ── P-20260910-005: 五种转述模式多样性引导 (CogAlpha 消融: 多样性引导单调提升 RankICIR) ──
        _modes = ["Light", "Moderate", "Creative", "Divergent", "Concrete"]
        _mode_desc = {
            "Light": "最小改动: 直接基于现有成功模板做参数/窗口微调",
            "Moderate": "自然措辞: 换一种机制表述, 骨架基本不变",
            "Creative": "研究导向: 为机制补充更深的经济解释后重新设计",
            "Divergent": "互补视角: 从相反/正交的分析视角设计同方向因子",
            "Concrete": "抽象落地: 把抽象概念转化为可直接求值的 pandas 公式",
        }
        _n_modes = min(n_factors, len(_modes))
        parts.append("## 🎭 五种转述模式 (同批多样性引导 — 防思维定式)")
        parts.append(f"本批 {n_factors} 条因子中, 前 {_n_modes} 条必须分别使用下列模式 (每条一个, 不得重复):")
        for i in range(_n_modes):
            parts.append(f"- {_modes[i]}: {_mode_desc[_modes[i]]}")
        parts.append("")

        # ── P-20260912-001: CogAlpha 认知进化 (思维进化 + 自适应生成) ──
        # 开关 FUXI_LLM_COG_EVOLVE=enforce 才注入; shadow(默认)/off 时 _cog_enabled=False
        # → 零注入, prompt 与 v0.7 完全一致 (rollback 一行)。
        _cog_parts = []
        if getattr(self, "_cog_enabled", False):
            for _k in ("adaptive", "thought_evolution"):
                _blk = self._cog_blocks.get(_k, "")
                if _blk:
                    _cog_parts.append(_blk)
        if _cog_parts:
            parts.extend(_cog_parts)

        # 输出格式要求
        parts.append(f"## 输出格式")
        parts.append(f"请生成 {n_factors} 个新因子，每行一个，格式: `表达式 | 名称 | 假设 | 方向 | 证伪条件 | 转述模式`")
        parts.append(f"其中方向为 + (做多) 或 - (做空); 转述模式取值为上述五种之一 ({'/'.join(_modes)}), 前 {_n_modes} 条须各不相同")
        parts.append("")
        parts.append("⚠️ 假设 (hypothesis) 必须与表达式严格对齐:")
        parts.append("- 假设必须具体描述表达式中实际使用的操作和字段")
        parts.append("  (如用到 rolling(5).mean 就写均线, 用到 diff/pct_change/shift 才写动量)")
        parts.append("- 禁止写表达式里没有的概念 (表达式不含 volume 时, 假设不得提放量/缩量)")
        parts.append("- 若表达式中使用了 buy_lg_vol/sell_lg_vol 等字段, 假设应提及大单/主力资金流")

        # ── P-20260910-007: 范式提案模式 — 额外 1 条偏离既有 21 范式的候选 ──
        if getattr(self, "_proposal_mode", False):
            parts.append("")
            parts.append("## 🧭 范式提案 (本轮额外 1 条 — 偏离既有范式, 独立计通道)")
            parts.append(
                "除上述常规候选外, 请**额外**给出 1 条尝试开辟**新范式**的候选 "
                "(不是既有范式内的改进, 而是此前未系统探索的结构)。"
            )
            if self._proposal_hint:
                parts.append(f"方向提示: {self._proposal_hint}")
            parts.append(
                "硬约束: ①只能用本地可求值字段 (close/open/high/low/volume 及量价衍生), "
                "禁捏造不存在字段; ②必须 pandas .rolling()/.shift()/.rank(pct=True) 风格, 禁前视; "
                "③必须给出可证伪条件。"
            )
            parts.append(
                "格式 (在第 6 字段后追加第 7 字段): "
                "`表达式 | 名称 | 假设 | 方向 | 证伪条件 | 转述模式 | 新范式名`"
            )
            parts.append("新范式名请用简短中文短语 (如 '多尺度复杂性' / '路径几何' / '跨期结构断裂')。")
        parts.append("")
        parts.append("⚠️ 证伪条件 (refutation_conditions, AQuA 式可证伪设计 — 必填, 至少 1 条):")
        parts.append("- 每个因子必须声明: 什么可观测结果会推翻这个假设")
        parts.append("- 示例: `若该因子 2023 子样本 ICIR<0 或 +IC%<50% 则假设不成立`")
        parts.append("- 示例: `若剔除量价背离日后信号消失则纯噪声`")
        parts.append("- 多个证伪条件用分号分隔; 无法证伪的因子会被降级为 low_confidence")
        parts.append("")
        parts.append("")
        # ── P-20260915-006: 生成端强约束 schema (白名单制) ──
        # 开关 FUXI_LLM_SCHEMA_CONSTRAINT=on(默认)/off; off 回退到 v0.7 弱语法说明块。
        if _schema_constraint_enabled():
            parts.append(build_schema_constraint_block(weekly_mode=weekly_mode,
                                                       n_factors=n_factors))
            parts.append("")
        else:
            parts.append("⚠️ 表达式语法要求 (pandas Rolling 风格):")
            parts.append("- 必须使用 pandas Series.rolling(N).method() 语法, 不允许 forge ts_* 风格")
            parts.append("- 可用算子: .rolling(N).mean() .rolling(N).std() .rolling(N).max() .rolling(N).min() .shift(N) .diff(N) .pct_change(N) .rank(pct=True)")
            parts.append("- 可用函数: np.log np.abs np.sign np.where np.maximum np.minimum")
            parts.append("- 变量名: close open high low volume (均为 pandas Series)")
            parts.append("- 可选资金流字段: buy_lg_vol sell_lg_vol buy_sm_vol sell_sm_vol buy_md_vol sell_md_vol buy_elg_vol sell_elg_vol net_mf_vol")
            parts.append("- 示例正确: `(close - close.rolling(60).min()) / close.rolling(60).min()`")
            parts.append("- 示例正确: `-((high - low) / (volume.rolling(20).mean() + 1)).rank(pct=True)`")
            parts.append("- 示例错误: `ts_mean(close, 20)` `div(a,b)` `neg(x)` `rank(x)` (不允许!)")
            parts.append("")
        parts.append("因子设计原则:")
        parts.append("1. 避免使用已被FSA禁止的骨架结构")
        parts.append("2. 优先在目标范式内设计，也可以跨界创新")
        parts.append("3. 表达式复杂度适中(node_count ≤ 10, depth ≤ 4)")
        parts.append("4. 有清晰的经济逻辑支撑")
        parts.append("5. 考虑A股市场特征(散户为主、高换手、政策敏感)")
        parts.append("6. 如果能基于成功模板做改进/变体，优先该路径")
        parts.append("")
        parts.append("直接输出因子列表(不要额外解释):")
        
        return "\n".join(parts)
    
    # ── 写入 prompt 文件 ─────────────────────────────────
    
    def write_prompt_file(self, prompt: str = "", target_paradigm: str = "") -> Path:
        """将 prompt 写入文件供手动提交LLM"""
        if not prompt:
            prompt = self.build_prompt(target_paradigm)
        
        self._gen_counter += 1
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"llm_gen_prompt_{ts}_r{self._gen_counter:03d}.txt"
        filepath = self.output_dir / filename
        
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(prompt)
        
        self._last_prompt_file = filepath
        print(f"  [LLMGen] Prompt → {filepath}")
        return filepath
    
    # ── 响应解析 ──────────────────────────────────────────
    
    def parse_response(self, response_text: str) -> List[Dict]:
        """
        解析 LLM 响应为候选因子列表。
        
        支持两种格式:
        1. 管道分隔: expression | name | hypothesis | direction
        2. JSON: [{"expression": ..., "name": ..., "hypothesis": ..., "direction": ...}]
        """
        candidates = []
        
        # 尝试 JSON 格式
        if response_text.strip().startswith("["):
            try:
                items = json.loads(response_text)
                for item in items:
                    cand = self._normalize_candidate(item)
                    if cand:
                        candidates.append(cand)
                if candidates:
                    return candidates
            except json.JSONDecodeError:
                pass
        
        # 管道分隔格式
        for line in response_text.strip().split("\n"):
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("//"):
                continue
            
            # 跳过 LLM 的"解释"行 (通常不是因子格式)
            if "|" not in line:
                continue
            
            parts = [p.strip() for p in line.split("|")]
            if len(parts) < 3:
                continue
            
            expression = parts[0]
            name = parts[1] if len(parts) > 1 else ""
            hypothesis = parts[2] if len(parts) > 2 else ""
            direction = parts[3].strip() if len(parts) > 3 else "+"
            # P-20260902-005: 第 5 字段 = 证伪条件 (可选, 缺失由 SemanticVerifier 降级)
            refutation_conditions = parts[4].strip() if len(parts) > 4 else ""
            # P-20260910-005: 第 6 字段 = 转述模式 (Light/Moderate/Creative/Divergent/Concrete)
            paraphrase_mode = parts[5].strip() if len(parts) > 5 else ""
            # P-20260910-007: 第 7 字段 = 新范式名 (仅范式提案模式下出现)
            proposed_paradigm = parts[6].strip() if len(parts) > 6 else ""
            
            # 2026-09-01 修复: 剥掉 markdown 列表前导符 (LLM 常以 "- expr | name | ..."
            # 或 "1. expr | ..." 输出, 前导 "- " 会被误当成公式取负 → 方向语义翻转)
            _stripped = re.sub(r'^\s*[-*•]\s+', '', expression)
            _stripped = re.sub(r'^\s*\d+[.)、]\s*', '', _stripped)
            # 剥除后非空且仍像表达式 (不以裸运算符结尾/开头异常) 才采用
            if _stripped and len(_stripped) > 2 and _stripped != expression:
                expression = _stripped
            
            # 标准化方向
            if direction in ("-", "short", "做空"):
                direction = "short"
            else:
                direction = "long"
            
            # 2026-09-01 修复: 管道分支与 JSON 分支统一走 _normalize_candidate,
            # 保证候选 dict 含 factor_name/formula 键。
            # (此前管道产物只有 expression/name, 下游 jq_candidate_details 的
            #  cand_map 按 factor_name 匹配失败 → formula 空 → JQ codegen 全 SKIP)
            cand = self._normalize_candidate({
                "expression": expression,
                "name": name,
                "hypothesis": hypothesis,
                "direction": direction,
                "refutation_conditions": refutation_conditions,
                "paraphrase_mode": paraphrase_mode,
                "proposed_paradigm": proposed_paradigm,  # P-20260910-007
            })
            if cand:
                candidates.append(cand)
        
        return candidates
    
    def parse_response_from_file(self, filepath: str) -> List[Dict]:
        """从文件读取 LLM 响应并解析"""
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
        return self.parse_response(text)
    
    # ── 归一化 ───────────────────────────────────────────
    
    def _normalize_candidate(self, item: Dict) -> Optional[Dict]:
        """将各种格式的候选字典归一化为标准格式"""
        formula = item.get("expression") or item.get("formula") or item.get("expr", "")
        if not formula:
            return None
        
        name = item.get("name") or item.get("factor_name") or f"llm_gen_{hash(formula) % 10000}"
        hypothesis = item.get("hypothesis") or item.get("rationale") or item.get("logic") or ""
        direction = item.get("direction", "+")
        
        if direction in ("-", "short", "做空"):
            direction = "short"
        else:
            direction = "long"
        
        return {
            "factor_name": name,
            "formula": formula,
            "expression": formula,
            "hypothesis": hypothesis,
            "logic": hypothesis,  # LLM 生成的可能在同一个字段里
            "direction": direction,
            "source": "llm_generated",
            "paradigm": item.get("paradigm", ""),
            # P-20260830-001: 周频语境标签透传 (weekly → weekly lane 裁决)
            "natural_freq": item.get("natural_freq") or item.get("freq") or "",
            # P-20260902-005: 证伪条件透传 (AQuA 式可证伪设计)
            "refutation_conditions": item.get("refutation_conditions", "") or "",
            # P-20260910-005: 转述模式透传 (五种转述模式多样性引导)
            "paraphrase_mode": item.get("paraphrase_mode", "") or "",
            # P-20260910-007: 范式提案透传 (非空即该候选为偏离既有范式的范式提案)
            "proposed_paradigm": item.get("proposed_paradigm", "") or "",
            "paradigm_proposal": bool(item.get("proposed_paradigm", "") or ""),
        }
    
    # ── 统一入口 ─────────────────────────────────────────
    
    def batch_dedup_ast(
        self,
        candidates: List[Dict],
        evidence: Optional[List[Dict]] = None,
        mode: Optional[str] = None,
        stage: str = "",
    ) -> List[Dict]:
        """M7: 公共批内 AST 去重入口 (Ralph 主线 / reviewer 二轮 / generate())。

        受 FUXI_BATCH_AST_DEDUP 管辖; evidence 非 None 时命中项追加留档。
        """
        return dedup_ast_candidates(candidates, evidence_out=evidence,
                                    mode=mode, stage=stage)

    def _batch_dedup_ast(self, candidates: List[Dict]) -> List[Dict]:
        """P-20260910-005 历史入口: 委托公共阶段 (M7 起统一受开关管辖)。"""
        return dedup_ast_candidates(candidates)

    def filter_quality_candidates(self, candidates: List[Dict]) -> List[Dict]:
        """Keep raw evidence; block only proven defects in enforce mode.

        FUXI_LLM_QUALITY_GATE=off|shadow|enforce (default enforce).  A review
        flag is a downlock for later promotion, not a fabricated rejection.
        """
        from formula_quality_precheck import assess_candidate

        mode = os.environ.get("FUXI_LLM_QUALITY_GATE", "enforce").strip().lower()
        if mode not in ("off", "shadow", "enforce"):
            raise ValueError("FUXI_LLM_QUALITY_GATE must be off, shadow or enforce")
        if mode == "off":
            return candidates
        kept = []
        for candidate in candidates:
            prior = candidate.get("formula_quality") or {}
            raw = candidate.get("raw_formula")
            current = candidate.get("formula") or candidate.get("expression")
            reused = (bool(raw) and prior.get("canonical_formula") == current
                      and prior.get("raw_formula_sha256")
                      == hashlib.sha256(str(raw).encode("utf-8")).hexdigest())
            if reused:
                # Ralph can call this gate after generate() has already repaired
                # rank axes. Keep the first assessment tied to the raw response.
                verdict = prior
            else:
                verdict = assess_candidate(candidate)
            candidate["formula_quality"] = verdict
            candidate["formula_quality_review_required"] = bool(verdict["review_codes"])
            if mode == "enforce" and verdict["hard_codes"]:
                print(f"  [LLMQuality] reject {candidate.get('factor_name', '?')}: "
                      + ",".join(verdict["hard_codes"]))
                continue
            if mode == "enforce" and verdict["repairs"] and not reused:
                candidate["raw_formula"] = candidate.get("formula") or candidate.get("expression")
                candidate["formula"] = verdict["canonical_formula"]
                candidate["expression"] = verdict["canonical_formula"]
            kept.append(candidate)
        return kept

    def generate(
        self,
        target_paradigm: str = "",
        n_factors: int = 5,
        mode: str = "file",  # "file" | "api" | None (仅解析已有response)
        response_text: str = "",
        weekly_mode: Optional[bool] = None,  # P-20260830-001
    ) -> Tuple[Path, List[Dict]]:
        """
        生成因子: 构建prompt → API调用或写入文件 → 解析响应。
        
        Args:
            target_paradigm: 目标探索范式
            n_factors: 期望生成的因子数
            mode: "file" 写入文件等待手动提交 | "api" 调用 DeepSeek API
            response_text: 如果已有LLM响应文本，直接解析
            weekly_mode: True 时周频语境 prompt + 候选打 natural_freq=weekly
        """
        if weekly_mode is None:
            weekly_mode = self._weekly_mode
        if response_text:
            candidates = self.parse_response(response_text)
            candidates = self._batch_dedup_ast(candidates)  # P-20260910-005
            candidates = self.tag_semantic_candidates(candidates)
            candidates = self.filter_quality_candidates(candidates)
            _v = schema_violation_summary(candidates, weekly_mode=weekly_mode)  # P-20260915-006
            if _v:
                print("  " + _v)
            if weekly_mode:
                for c in candidates:
                    c["natural_freq"] = "weekly"
            return None, candidates
        
        prompt = self.build_prompt(target_paradigm, n_factors, weekly_mode=weekly_mode)
        filepath = self.write_prompt_file(prompt, target_paradigm)
        
        if mode == "file":
            return filepath, []
        
        if mode == "api":
            # v3.1: 直接调用 DeepSeek API
            try:
                from llm_client import get_llm_client
                client = get_llm_client()
                response_text = client.chat_with_system(
                    system_prompt=_schema_system_prompt(weekly_mode=weekly_mode),
                    user_prompt=prompt,
                    temperature=0.8,
                    max_tokens=4096,
                )
                candidates = self.parse_response(response_text)
                candidates = self._batch_dedup_ast(candidates)  # P-20260910-005
                candidates = self.tag_semantic_candidates(candidates)
                candidates = self.filter_quality_candidates(candidates)
                if weekly_mode:
                    for c in candidates:
                        c["natural_freq"] = "weekly"
                print(f"  [LLMGen] API 返回 {len(response_text)} chars → 解析出 {len(candidates)} 个候选"
                      + (" (周频语境)" if weekly_mode else ""))
                _v = schema_violation_summary(candidates, weekly_mode=weekly_mode)  # P-20260915-006
                if _v:
                    print("  " + _v)
                return filepath, candidates
            except Exception as e:
                print(f"  [LLMGen] API 调用失败: {e}，降级为文件模式")
                return filepath, []
        
        return filepath, []  # 默认: 生成文件, 手动解析

    # ── v0.6 P-021: 多范式批量生成 ────────────────────────

    def generate_paradigm_batch(
        self,
        paradigms: List[str],
        n_per_paradigm: int = 3,
        mode: str = "file",  # "file" | "api"
    ) -> Dict[str, Tuple[Path, List[Dict]]]:
        """
        P-021: 对多个未覆盖范式并行生成因子公式。

        为每个目标范式构建领域专用 prompt (包含经济逻辑+典型表达式)，
        调用 LLM 生成因子公式，返回按范式分组的候选列表。

        Parameters
        ----------
        paradigms: 目标范式名列表, 如 ["行业轮动", "情绪×日内"]
        n_per_paradigm: 每个范式生成因子数
        mode: "file" 写入文件 | "api" 直接调用 DeepSeek API

        Returns
        -------
        {paradigm_name: (prompt_file_path, [candidate_dict, ...]), ...}
        """
        results = {}
        for paradigm in paradigms:
            # 构建领域专用 prompt
            prompt = self._build_domain_prompt(paradigm, n_per_paradigm)
            filepath = self.write_prompt_file(prompt, paradigm)

            candidates = []
            if mode == "api":
                try:
                    from llm_client import get_llm_client
                    client = get_llm_client()
                    response_text = client.chat_with_system(
                        system_prompt=_schema_system_prompt(),
                        user_prompt=prompt,
                        temperature=0.8,
                        max_tokens=4096,
                    )
                    candidates = self.parse_response(response_text)
                    candidates = self.filter_quality_candidates(candidates)
                    # 标注 paradigm
                    for c in candidates:
                        c["paradigm"] = paradigm
                    print(f"  [LLMGen] {paradigm}: {len(candidates)} 个候选")
                except Exception as e:
                    print(f"  [LLMGen] {paradigm} API 调用失败: {e}")

            results[paradigm] = (filepath, candidates)

        return results

    def _build_domain_prompt(self, paradigm: str, n_factors: int = 3) -> str:
        """
        构建领域专用 prompt，为每个未覆盖范式注入经济逻辑和参考表达式风格。

        基于 paradigm_v4.py 的 PARADIGMS_V4 定义，自动填充:
        - 范式经济逻辑
        - A股相关性说明
        - 典型表达式风格 (来自 CHAMPION_FACTORS 同范式模板)
        """
        from paradigm_v4 import PARADIGMS_V4

        paradigm_info = PARADIGMS_V4.get(paradigm, {})
        economic_logic = paradigm_info.get("economic_logic", "")
        a_share_relevance = paradigm_info.get("a_share_relevance", "")
        description = paradigm_info.get("description", "")

        # 从 seed_injector CHAMPION_FACTORS 中找同范式示例
        example_formulas = self._get_example_formulas(paradigm)

        ctx = self._context
        parts = []

        parts.append(f"# A股量化因子生成: {paradigm} 范式探索")
        parts.append("")
        parts.append(f"## 范式背景")
        parts.append(f"- **经济逻辑**: {economic_logic}")
        parts.append(f"- **A股相关性**: {a_share_relevance}")
        parts.append(f"- **描述**: {description}")
        parts.append("")

        if example_formulas:
            parts.append(f"## 参考表达式风格 (来自已验证的同范式因子)")
            for ex in example_formulas[:3]:
                parts.append(f"- `{ex['expression'][:120]}` — {ex['rationale'][:80]}")
            parts.append("")

        # 成功模板
        templates = ctx.get("success_templates", [])
        paradigm_templates = [t for t in templates
                            if (isinstance(t, dict) and paradigm in t.get("paradigm", "")) or
                               (hasattr(t, 'pattern_id') and paradigm in str(getattr(t, 'pattern_id', '')))]
        if paradigm_templates:
            parts.append(f"## ✅ 历史成功模板 ({paradigm} 领域)")
            for t in paradigm_templates[:3]:
                if isinstance(t, dict):
                    parts.append(f"- {t.get('description', t.get('pattern_id', ''))[:100]}: `{t.get('formula', '')[:120]}`")
            parts.append("")

        # 禁止方向
        forbidden = ctx.get("forbidden_directions", [])
        if forbidden:
            parts.append("## 🚫 禁止方向")
            for d in forbidden[:3]:
                parts.append(f"- {d.get('description', d.get('direction', ''))[:120]}")
            parts.append("")

        # 输出格式
        parts.append(f"## 输出格式")
        parts.append(f"请为 `{paradigm}` 范式生成 {n_factors} 个新因子，每行: `表达式 | 名称 | 假设 | 方向`")
        parts.append("方向为 + (做多) 或 - (做空)")
        parts.append("")
        parts.append("⚠️ 表达式要求: pandas Rolling 风格 (close.rolling(20).mean()), 不允许 forge ts_* 风格")
        parts.append("表达式复杂度适中 (node_count ≤ 10, depth ≤ 4)")
        parts.append("")
        parts.append("直接输出因子列表:")

        return "\n".join(parts)

    @staticmethod
    def _get_example_formulas(paradigm: str) -> List[Dict]:
        """从 CHAMPION_FACTORS 提取同范式示例公式"""
        try:
            from seed_injector import CHAMPION_FACTORS
        except ImportError:
            return []
        examples = []
        for f in CHAMPION_FACTORS:
            if f.get("paradigm", "") == paradigm:
                examples.append({
                    "expression": f.get("expression", f.get("formula", "")),
                    "rationale": f.get("rationale", ""),
                })
            if len(examples) >= 3:
                break
        return examples


def generate_llm_prompt_file(
    memory_priors: Dict,
    target_paradigm: str = "",
    n_factors: int = 5,
) -> Path:
    """
    快捷函数: 从 Memory priors 生成 LLM prompt 文件。
    
    Args:
        memory_priors: memory.retrieve() 的返回值
        target_paradigm: 目标范式
        n_factors: 期望生��因子数
        
    Returns:
        写入的 prompt 文件路径
    """
    gen = LLMGenerator()
    gen.receive_context(
        success_templates=memory_priors.get("success_templates", []),
        forbidden_directions=memory_priors.get("forbidden_directions", []),
        fsa_forbidden=memory_priors.get("fsa_forbidden_context", ""),
        library_stats=memory_priors.get("library_stats", {}),
    )
    return gen.write_prompt_file(target_paradigm=target_paradigm)


if __name__ == "__main__":
    # 自测: 构建一个示例 prompt
    gen = LLMGenerator()
    gen.receive_context(
        success_templates=[
            {"pattern_id": "筹码分布::[缩量,均值]", "description": "筹码锁定度因子",
             "formula": "neg(rank(div(ts_mean(volume, 10), ts_mean(volume, 60))))",
             "paradigm": "筹码分布", "occurrence_count": 6, "jq_return": 182.57},
        ],
        forbidden_directions=[
            {"direction": "ralph_no_ic_candidates", "reason": "无IC数据的因子不能直接上JQ"},
        ],
        library_stats={"total_factors": 243, "n_clusters": 156,
                       "red_sea": {"level": "green", "median_correlation": 0.038}},
    )
    
    filepath = gen.write_prompt_file(target_paradigm="筹码分布")
    print(f"Prompt written to: {filepath}")
    
    # 测试解析
    test_response = """
rank(div(sub(high, low), ts_mean(volume, 20))) | 振幅量比 | 高振幅低量=流动性溢价 | +
neg(ts_zscore(close, 60)) | 深度超卖 | 长期超卖=均值回归 | +
"""
    candidates = gen.parse_response(test_response)
    print(f"Parsed {len(candidates)} candidates:")
    for c in candidates:
        print(f"  - {c['expression'][:50]} | {c['name']} | {c['direction']}")
