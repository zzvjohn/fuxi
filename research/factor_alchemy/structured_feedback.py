# -*- coding: utf-8 -*-
"""
结构化失败反馈 (Structured Failure Feedback) — P-20260915-007
=============================================================

对标 QuantMind / RD-Agent CoSTEER `alphaagent_loop.py` 的 feedback 阶段。
其 `feedback` 步骤把每一轮的结论写成 **5 个规范字段**回灌 `trace.hist`,
下一轮 `hypothesis_gen` 读 `trace.hist[-10:]` ⇒ **失败不是垃圾, 是下一轮的输入**。

────────────────────────────────────────────────────────────
一、伏羲现状 vs 目标
────────────────────────────────────────────────────────────
现状: `llm_generator._extract_recent_failures()` 只产出**扁平文本**
      `- <因子名> (<outcome>): <lesson 前 80 字>`
      ⇒ 结构信息 (卡在哪一阶段/差多少/为什么) 在传给 LLM 时丢失。
目标: 拆成 5 个规范字段 (与 QuantMind 字段一一对应):
      Observations   ← 客观读数 (原始观测, 不含推断)
      Evaluation     ← 判据对比与结论 (哪个闸门, 差多少)
      New Hypothesis ← **可迁移的条件式知识** ("If…, then…", 对应 concise_knowledge)
      Reasoning      ← 归因层次 (表达式层/机制层/数据层/判据层)
      Decision       ← 下一轮动作 (rewrite / retry_paradigm / abandon / replace_best)

────────────────────────────────────────────────────────────
二、数据来源 (全部来自既有产物, 不新增采集)
────────────────────────────────────────────────────────────
主源: `data/trajectory_log.json` → `trajectories[]._eval_log[]`
      字段: factor_name / formula / paradigm / stage / passed / ic / icir /
            excess_25 / excess_26 / calmar_25 / calmar_26 /
            rejection_reason / formula_complexity / refutation_conditions
备源: `data/experience_memory.json` → `attempts[]`
      字段: factor_name / formula / paradigm / outcome / icir / fri / lessons / engine

🔴 口径知情 (来自 2026-09-12 版本纪律审计):
   `_eval_log.calmar_25` 字段存的**实际是 calmar_26 的值**, `calmar_26` 恒为 null
   ⇒ 本模块**只用 `rejection_reason` 文本**做判据归因, 不消费 calmar_* 数值字段,
     并在 Observations 中显式标注"字段错位待修"。

────────────────────────────────────────────────────────────
三、开关与产物
────────────────────────────────────────────────────────────
    FUXI_STRUCT_FEEDBACK = off | shadow (默认) | enforce
      off     : 完全不介入 (prompt 用旧扁平文本)
      shadow  : 计算并落盘 + 打印读数, **prompt 仍用旧扁平文本** (零生成影响)
      enforce : prompt 用结构化 5 字段块
    FUXI_STRUCT_FEEDBACK_LIMIT = 每轮回灌条数 (默认 3)
    FUXI_STRUCT_FEEDBACK_DAYS  = 回看天数 (默认 7)

产物: `data/structured_feedback_state.json` (滚动窗口, 保留最近 200 批)

回滚: 删除本文件 + 撤销 llm_generator 的 `_resolve_failure_feedback` 调用。
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
TRAJ_PATH = DATA_DIR / "trajectory_log.json"
EM_PATH = DATA_DIR / "experience_memory.json"
STATE_PATH = DATA_DIR / "structured_feedback_state.json"

# ── 5 字段名 (与 QuantMind feedback 阶段对齐) ────────────────
FIELDS = ("Observations", "Evaluation", "New Hypothesis", "Reasoning", "Decision")

DECISION_ENUM = (
    "rewrite_expression",   # 表达式层可修 → 重写
    "retry_with_paradigm",  # 机制层重复/无增量 → 换范式重试
    "abandon_direction",    # 已被实证证伪 → 放弃该方向
    "replace_best",         # 优于当前最优 → 替换
    "promote",              # 通过闸门 → 晋级
)


# ═══════════════════════════════════════════════════════════════
# 归因规则引擎 (确定性, 零 LLM 配额)
# ═══════════════════════════════════════════════════════════════

@dataclass
class AttributionRule:
    """rejection_reason 正则 → 5 字段归因。"""
    pattern: str
    layer: str                      # 表达式层 | 机制层 | 数据层 | 判据层
    decision: str
    new_hypothesis: str
    evaluation: str


# 顺序敏感: 先匹配到的规则生效 (越具体越靠前)
RULES: List[AttributionRule] = [
    AttributionRule(
        pattern=r"无周频\s*IC\s*数据",
        layer="数据层",
        decision="rewrite_expression",
        evaluation="卡在**数据可得性**而非信号强度: 该表达式在周频宽表上无法求值",
        new_hypothesis=("If 要生成周频 (weekly lane) 因子, then 必须只用周频宽表提供的字段 "
                        "(close/open/high/low/volume/amount) 且窗口按周频语义 "
                        "(rolling(4)≈1月) —— 禁引用日频专有字段与资金流字段, 否则恒被判据拒绝"),
    ),
    AttributionRule(
        pattern=r"(SyntaxError|语法|语法错误|IndentationError)",
        layer="表达式层",
        decision="rewrite_expression",
        evaluation="表达式**语法非法**, 未进入统计评估 —— 属机械错误而非 alpha 问题",
        new_hypothesis=("If 使用 pandas 方法链, then 必须保证链尾是返回 Series 的方法 "
                        "(如 .rolling().mean() 合法, .rolling().pct_change() 非法 —— "
                        "Rolling 对象没有 pct_change), 且不写多行代码块"),
    ),
    AttributionRule(
        pattern=r"(NameError|字段|未定义|不存在)",
        layer="数据层",
        decision="rewrite_expression",
        evaluation="引用了求值环境中**不存在的字段**, 表达式未进入统计评估",
        new_hypothesis=("If 表达式引用了白名单外字段, then 恒被判据拒绝 —— "
                        "变量只能用 close/open/high/low/volume (+日频资金流字段); "
                        "日频无 amount、无基本面"),
    ),
    AttributionRule(
        pattern=r"前视|lookahead|未来",
        layer="表达式层",
        decision="rewrite_expression",
        evaluation="触发**前视审计闸门**, 信号不可实现",
        new_hypothesis=("If 信号收益在 T 日盘后即可获得, then 必须按 open(T+1) 口径重建 —— "
                        "盘后披露类信号用当日 close 建仓 = 1 日前视, 表观 ICIR 会虚高数倍"),
    ),
    AttributionRule(
        pattern=r"(相关性|max_corr|冗余|近似重复|同质)",
        layer="机制层",
        decision="retry_with_paradigm",
        evaluation="被**相关性/冗余闸门**拒绝: 与库内既有因子行为高度相似, 无增量",
        new_hypothesis=("If 新因子与库内因子相关性 > 阈值, then 即使单因子 IC 优秀也无组合增量 ⇒ "
                        "必须改变**经济机制**本身 (而非微调窗口/参数), 并优先选 "
                        "机制正交的范式"),
    ),
    AttributionRule(
        pattern=r"(excess|超额).*(负|<\s*0|不足)|excess_25|excess_26",
        layer="判据层",
        decision="retry_with_paradigm",
        evaluation="通过单因子闸门但**组合超额为负** —— 即 S5 终裁 (组合层) 不通过",
        new_hypothesis=("If 因子在组合中不带来正的超额收益, then 单因子 IC/ICIR 再高也无意义 ⇒ "
                        "设计时应先问「它与现有因子组合后边际增量从哪来」, "
                        "而不是先追求单因子 IC"),
    ),
    AttributionRule(
        pattern=r"(calmar|回撤)",
        layer="判据层",
        decision="retry_with_paradigm",
        evaluation="组合**风险调整后收益 (Calmar) 不达标**",
        new_hypothesis=("If 因子只在低波动阶段有效, then 其年度 Calmar 会显著低于 1 ⇒ "
                        "优先设计与高波动阶段也成立 (或在波动率上升时不被淹没) 的信号"),
    ),
    AttributionRule(
        pattern=r"(过度简化|复杂度|过简|过度复杂)",
        layer="表达式层",
        decision="rewrite_expression",
        evaluation="**表达式复杂度闸门**拒绝 (过简或过繁)",
        new_hypothesis=("If 表达式节点数过少, then 通常是已有因子的退化形式; "
                        "If 过多, then 过拟合风险高 ⇒ 把复杂度控制在 node_count 5~15"),
    ),
    AttributionRule(
        pattern=r"(符号守卫|符号翻转|负向)",
        layer="表达式层",
        decision="rewrite_expression",
        evaluation="触发**符号守卫**: 因子实证 IC 为负, 与声明的假设方向相反",
        new_hypothesis=("If 声明方向为 long 但实证 IC 为负, then 两种可能必须区分: "
                        "①经济机制的方向假设写反了 ②该信号在 A 股确为反向 ⇒ "
                        "先按实证符号重构方向声明再重评, "
                        "**禁止用 |IC| 取绝对值掩盖方向问题** (S5 方向盲区历史坑)"),
    ),
    AttributionRule(
        pattern=r"(无\s*IC|IC\s*不足|IC\s*[=<>]|ICIR|icir|信号强度)",
        layer="判据层",
        decision="retry_with_paradigm",
        evaluation="**单因子信号强度不足** (IC/ICIR 未过闸门)",
        new_hypothesis=("If 因子的 IC 在时间上不稳定, then ICIR 会远低于 IC 均值 ⇒ "
                        "优先生成**经济机制明确、方向可先验推导**的因子, "
                        "而不是靠参数搜索凑 IC"),
    ),
]

# 已实证的 JQ 终态 → 强制决策 (优先级最高)
JQ_DECISION = {
    "JQ_FAILED": ("JQ 实测**证伪** (负收益/负 Sharpe)", "abandon_direction"),
    "JQ_FAIL": ("JQ 实测证伪", "abandon_direction"),
    "JQ_WEAK_NEGATIVE": ("JQ 实测显著负向", "abandon_direction"),
    "JQ_BROKEN": ("JQ 执行层故障 (非 alpha 问题)", "rewrite_expression"),
    "JQ_MARGINAL": ("JQ 实测边际 (不可独立交易, 可作组合候选)", "replace_best"),
    "JQ_PASSED": ("JQ 实测通过", "promote"),
}


def _classify(rejection_reason: str, stage: str = "", outcome: str = "") -> AttributionRule:
    r = (rejection_reason or "") + " " + (stage or "")
    for rule in RULES:
        if re.search(rule.pattern, r, re.IGNORECASE):
            return rule
    return AttributionRule(
        pattern="*",
        layer="判据层",
        decision="retry_with_paradigm",
        evaluation="未能在 rejection_reason 中定位到已知失败模式 (归因不足, 保守降级)",
        new_hypothesis=("If 失败原因无法归因, then 该方向不应重复投入, "
                        "应换一个经济机制明确的新范式"),
    )


# ═══════════════════════════════════════════════════════════════
# 记录构建
# ═══════════════════════════════════════════════════════════════

@dataclass
class FeedbackRecord:
    source: str
    factor_name: str
    paradigm: str
    stage: str
    formula: str
    ts: str
    fields: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return asdict(self)


def _truncate(s: str, n: int) -> str:
    s = (s or "").replace("\n", " ").strip()
    return s if len(s) <= n else s[:n] + "…"


def build_record_from_eval(e: Dict, paradigm: str = "", ts: str = "") -> FeedbackRecord:
    """从一条 `_eval_log` 条目构建 5 字段记录。"""
    stage = e.get("stage", "") or ""
    passed = bool(e.get("passed"))
    reason = e.get("rejection_reason", "") or ""
    ic = e.get("ic")
    icir = e.get("icir")
    name = e.get("factor_name", "") or "?"
    formula = e.get("formula", "") or ""

    if passed:
        rule = AttributionRule("*", "判据层", "promote", "", "该阶段通过")
        decision_why = "该阶段通过"
    else:
        rule = _classify(reason, stage)
        decision_why = reason or "(无 reason 文本)"

    obs = (f"因子 `{_truncate(name, 60)}` 在 {stage} 阶段未通过 | "
           f"formula={_truncate(formula, 90)} | ic={ic} icir={icir} | "
           f"复杂度={e.get('formula_complexity')} | paradigm={paradigm or '-'}")
    if not passed:
        obs += " | ⚠️ `_eval_log.calmar_*` 字段错位 (calmar_25 实存 26 值) 故不消费"

    ev = f"{rule.evaluation}｜原文: {_truncate(decision_why, 120)}"
    return FeedbackRecord(
        source="trajectory._eval_log",
        factor_name=name, paradigm=paradigm, stage=stage,
        formula=_truncate(formula, 140), ts=ts,
        fields={
            "Observations": obs,
            "Evaluation": ev,
            "New Hypothesis": rule.new_hypothesis or "(该失败模式暂无可迁移结论)",
            "Reasoning": f"归因层次 = {rule.layer}",
            "Decision": rule.decision,
        },
    )


def build_record_from_attempt(a: Dict) -> FeedbackRecord:
    """从 experience_memory.attempts 构建 (备源; 无 stage 信息)。"""
    outcome = (a.get("outcome") or "").upper()
    name = a.get("factor_name", "") or "?"
    lessons = a.get("lessons") or []
    fri = a.get("fri") or {}
    psi = a.get("psi") or {}

    if outcome in JQ_DECISION:
        why, decision = JQ_DECISION[outcome]
        layer = "实证层"
        nh = ("If 该方向已被 JQ 实盘证伪, then 禁止原方向重试 ⇒ "
              "只允许按实证 IC 符号翻转后重新评估 (翻转后仍不达标则彻底放弃)")
    elif outcome in ("PASS", "PASSED"):
        why, decision, layer = "通过闸门", "promote", "判据层"
        nh = "If 因子通过全部闸门, then 应作为成功模板参考其结构, 但避免做参数微调型变体"
    else:
        got = _classify(" ".join(str(l) for l in lessons), "", outcome)
        why, decision, layer = "未过闸门 (WEAK/RESERVE)", got.decision, got.layer
        nh = got.new_hypothesis

    obs = (f"因子 `{_truncate(name, 60)}` outcome={outcome} | "
           f"icir={a.get('icir')} | fri_score={fri.get('score')} grade={fri.get('grade') or '-'} | "
           f"psi_r2={psi.get('r_squared')} independence={psi.get('independence')} | "
           f"engine={a.get('engine', '-')}")
    ev = f"{why}｜lessons: {'; '.join(str(l)[:80] for l in lessons[:2]) or '(无)'}"
    return FeedbackRecord(
        source="experience_memory.attempt",
        factor_name=name, paradigm=a.get("paradigm", "") or "", stage="",
        formula=_truncate(a.get("formula", ""), 140), ts=a.get("timestamp", "") or "",
        fields={
            "Observations": obs,
            "Evaluation": ev,
            "New Hypothesis": nh or "(暂无可迁移结论)",
            "Reasoning": f"归因层次 = {layer}",
            "Decision": decision,
        },
    )


# ═══════════════════════════════════════════════════════════════
# 汇总与渲染
# ═══════════════════════════════════════════════════════════════

def collect(limit: int = 3, days: int = 7) -> List[FeedbackRecord]:
    """收集近期失败反馈 (主源 trajectory._eval_log, 备源 experience_memory)。"""
    cutoff = (datetime.now() - timedelta(days=days)).isoformat()
    recs: List[FeedbackRecord] = []

    # 主源: trajectory_log (倒序扫, 取最近失败的 eval 条目)
    try:
        d = json.loads(TRAJ_PATH.read_text(encoding="utf-8"))
        trajs = d.get("trajectories", []) if isinstance(d, dict) else d
        for tr in reversed(trajs):
            if len(recs) >= limit:
                break
            evals = tr.get("_eval_log") or []
            if not isinstance(evals, list):
                continue
            ts = tr.get("completed_at") or tr.get("created_at") or ""
            if ts and ts < cutoff:
                break
            paradigm = tr.get("paradigm", "") or ""
            # 取该轨迹第一个失败阶段 (最具信息量)
            for e in evals:
                if isinstance(e, dict) and not e.get("passed"):
                    recs.append(build_record_from_eval(e, paradigm=paradigm, ts=ts))
                    break
    except Exception as ex:                       # noqa: BLE001
        recs.append(FeedbackRecord("error", f"trajectory load failed: {ex}", "", "", "", "",
                                   {"Observations": str(ex), "Evaluation": "", "New Hypothesis": "",
                                    "Reasoning": "", "Decision": "retry_with_paradigm"}))

    # 备源: experience_memory (不足时补)
    if len(recs) < limit:
        try:
            em = json.loads(EM_PATH.read_text(encoding="utf-8"))
            for a in reversed(em.get("attempts", [])):
                if len(recs) >= limit:
                    break
                ts = a.get("timestamp", "") or ""
                if ts and ts < cutoff:
                    continue
                if (a.get("outcome") or "").upper() in ("PASS", "PASSED"):
                    continue
                recs.append(build_record_from_attempt(a))
        except Exception:
            pass

    return recs[:limit]


def render_prompt_block(recs: List[FeedbackRecord]) -> str:
    """渲染为 LLM prompt 用的结构化 5 字段块。"""
    if not recs:
        return ""
    lines = ["## 🔁 上轮失败反馈 (结构化 5 字段 — 每条都是下一轮的输入)"]
    lines.append("_字段含义: Observations=客观读数 / Evaluation=判据对比 / "
                 "New Hypothesis=可迁移条件式知识 / Reasoning=归因层次 / Decision=下一轮动作_")
    for i, r in enumerate(recs, 1):
        lines.append("")
        lines.append(f"### 反馈 {i} (来源 {r.source}{', 阶段 ' + r.stage if r.stage else ''})")
        for f in FIELDS:
            lines.append(f"- **{f}**: {r.fields.get(f, '')}")
    # 交叉汇总: 条件式知识清单 (对应 QuantMind concise_knowledge)
    nhs = [r.fields.get("New Hypothesis", "") for r in recs
           if r.fields.get("New Hypothesis", "").startswith("If")]
    if nhs:
        lines.append("")
        lines.append("### ⚠️ 本轮必须遵守的可迁移规则 (从上列失败中提炼)")
        for k, nh in enumerate(dict.fromkeys(nhs), 1):
            lines.append(f"{k}. {nh}")
    decs = [r.fields.get("Decision", "") for r in recs]
    if decs:
        lines.append("")
        lines.append("### 下一轮动作分布: " + ", ".join(f"{d}×{decs.count(d)}"
                                                        for d in dict.fromkeys(decs)))
    return "\n".join(lines)


def render_flat_fallback(limit: int = 3, days: int = 7) -> str:
    """旧扁平文本 (rollback / shadow 用, 与 v0.7 完全一致的行为)。"""
    try:
        from llm_generator import _extract_recent_failures
        return _extract_recent_failures(limit=limit, days=days)
    except Exception:
        return ""


def persist(recs: List[FeedbackRecord], mode: str, keep: int = 200) -> None:
    """落盘滚动窗口 (最多 keep 批)。"""
    try:
        state = {"_updated": datetime.now().isoformat(), "batches": []}
        if STATE_PATH.exists():
            try:
                state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
            except Exception:
                pass
        batches = state.get("batches", [])
        batches.append({
            "ts": datetime.now().isoformat(),
            "mode": mode,
            "n_records": len(recs),
            "records": [r.to_dict() for r in recs],
            "decision_counts": _decision_counts(recs),
            "layer_counts": _layer_counts(recs),
        })
        state["batches"] = batches[-keep:]
        state["_updated"] = datetime.now().isoformat()
        STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


def _decision_counts(recs: List[FeedbackRecord]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for r in recs:
        d = r.fields.get("Decision", "?")
        out[d] = out.get(d, 0) + 1
    return out


def _layer_counts(recs: List[FeedbackRecord]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for r in recs:
        lay = (r.fields.get("Reasoning", "") or "").replace("归因层次 = ", "")
        out[lay] = out.get(lay, 0) + 1
    return out


def format_readout(recs: List[FeedbackRecord], mode: str) -> str:
    dc = _decision_counts(recs)
    lc = _layer_counts(recs)
    d_txt = "/".join(f"{k}:{v}" for k, v in dc.items()) or "-"
    l_txt = "/".join(f"{k}:{v}" for k, v in lc.items()) or "-"
    return (f"[StructFeedback:{mode}] 回灌 {len(recs)} 条 | "
            f"决策 {d_txt} | 归因层 {l_txt}")


def get_mode() -> str:
    v = (os.environ.get("FUXI_STRUCT_FEEDBACK", "shadow") or "shadow").strip().lower()
    return v if v in ("off", "shadow", "enforce") else "shadow"


def build_feedback(limit: Optional[int] = None, days: Optional[int] = None,
                   mode: Optional[str] = None) -> str:
    """主入口: 按模式返回要注入 prompt 的失败反馈文本。

    - enforce: 结构化 5 字段块
    - shadow : 落盘 + 打印读数, 但返回**旧扁平文本** (零生成影响)
    - off    : 返回旧扁平文本, 不做任何额外工作
    """
    m = mode or get_mode()
    lim = int(limit if limit is not None
              else os.environ.get("FUXI_STRUCT_FEEDBACK_LIMIT", 3))
    dys = int(days if days is not None
              else os.environ.get("FUXI_STRUCT_FEEDBACK_DAYS", 7))

    if m == "off":
        return render_flat_fallback(limit=lim, days=dys)

    recs = collect(limit=lim, days=dys)
    if m == "shadow":
        persist(recs, mode=m)
        print("  " + format_readout(recs, m))
        return render_flat_fallback(limit=lim, days=dys)

    # enforce
    persist(recs, mode=m)
    print("  " + format_readout(recs, m))
    block = render_prompt_block(recs)
    return block or render_flat_fallback(limit=lim, days=dys)


if __name__ == "__main__":  # pragma: no cover
    _lim = int(os.environ.get("FUXI_STRUCT_FEEDBACK_LIMIT", 3))
    _recs = collect(limit=_lim, days=int(os.environ.get("FUXI_STRUCT_FEEDBACK_DAYS", 7)))
    print(format_readout(_recs, get_mode()))
    print()
    print(render_prompt_block(_recs))
