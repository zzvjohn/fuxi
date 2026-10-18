"""Typed, point-in-time checked hypothesis cards for factor generation.

Cards guide expression generation, but never grant S5/JQ eligibility.  An
unverified field or timing contract is an error rather than a prompt hint.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any, Dict, List


SCHEMA = "fuxi_semantic_plan/v1"
AVAILABLE = {
    "daily": {"open", "high", "low", "close", "volume"},
    "weekly": {"open", "high", "low", "close", "volume", "amount"},
}
CUTOFF = {"daily": ("post_close", "next_session"),
          "weekly": ("week_end_post_close", "next_week")}
USES = {"research_feature", "promotion_candidate"}
OBSERVABLE_NAMES = set().union(*AVAILABLE.values()) | {
    "buy_lg_vol", "sell_lg_vol", "buy_md_vol", "sell_md_vol",
    "buy_sm_vol", "sell_sm_vol", "buy_elg_vol", "sell_elg_vol",
    "net_mf_vol", "north_money", "overnight", "amplitude", "returns", "turnover", "hl_ratio",
}


def validate_plan(plan: Dict[str, Any]) -> Dict[str, Any]:
    required = ("plan_id", "frequency", "mechanism", "prediction", "falsifier",
                "baseline", "target_use", "fields", "signal_cutoff", "earliest_trade")
    missing = [key for key in required if not plan.get(key)]
    if missing:
        raise ValueError("semantic plan missing: " + ", ".join(missing))
    frequency = plan["frequency"]
    if frequency not in AVAILABLE:
        raise ValueError("unsupported semantic frequency")
    if plan["target_use"] not in USES:
        raise ValueError("target_use must be a research feature or promotion candidate")
    if (plan["signal_cutoff"], plan["earliest_trade"]) != CUTOFF[frequency]:
        raise ValueError("signal and execution timing must match the PIT contract")
    fields = plan["fields"]
    if not isinstance(fields, list) or not fields or len(fields) != len(set(fields)):
        raise ValueError("fields must be a nonempty unique list")
    unknown = set(fields) - AVAILABLE[frequency]
    if unknown:
        raise ValueError("unverified fields: " + ", ".join(sorted(unknown)))
    return dict(plan)


def load_plan(path: str | Path, plan_id: str, frequency: str) -> Dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("schema") != SCHEMA:
        raise ValueError("unknown semantic plan schema")
    matches = [p for p in data.get("plans", []) if p.get("plan_id") == plan_id]
    if len(matches) != 1:
        raise ValueError("plan_id absent or duplicated")
    plan = validate_plan(matches[0])
    if plan["frequency"] != frequency:
        raise ValueError("plan frequency disagrees with generation lane")
    return plan


def prompt_block(plan: Dict[str, Any]) -> str:
    p = validate_plan(plan)
    return ("## 已冻结的语义计划（先机制后表达式）\n"
            f"计划ID: {p['plan_id']}；用途: {p['target_use']}\n"
            f"机制: {p['mechanism']}\n"
            f"可检验预测: {p['prediction']}\n"
            f"证伪条件: {p['falsifier']}\n"
            f"对照基线: {p['baseline']}\n"
            f"仅允许字段: {', '.join(p['fields'])}；信号截止: {p['signal_cutoff']}；"
            f"最早交易: {p['earliest_trade']}。\n"
            "每条表达式必须与本计划机制逐项对应；不能虚构字段、提前使用盘后信息或把历史验证区称为未知样本。")


def tag_candidates(candidates: List[Dict[str, Any]], plan: Dict[str, Any]) -> List[Dict[str, Any]]:
    p = validate_plan(plan)
    tagged = []
    for candidate in candidates:
        formula = str(candidate.get("formula") or candidate.get("expression") or "")
        try:
            names = {n.id.removesuffix("_p") for n in ast.walk(ast.parse(formula, mode="eval"))
                     if isinstance(n, ast.Name)}
            violations = sorted((names & OBSERVABLE_NAMES) - set(p["fields"]))
        except SyntaxError:
            violations = ["invalid_expression_syntax"]
        tagged.append({**candidate, "semantic_plan_id": p["plan_id"],
                       "semantic_target_use": p["target_use"],
                       "semantic_signal_cutoff": p["signal_cutoff"],
                       "semantic_earliest_trade": p["earliest_trade"],
                       "semantic_plan_violations": violations})
    return tagged
