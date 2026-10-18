# -*- coding: utf-8 -*-
"""M1: 奖励守卫 — 「已验证经济事件 → 奖励」的唯一通道（影子半程）。

存在理由（2026-09-26 M1；reports/fuxi_alpha_generation_overhaul_20260926.md §M1）
--------------------------------------------------------------------
改造前的事实链：
  - `ralph_loop._mab_record_result` 是全库零调用的**死函数**，且 JQ 奖励入口
    没有任何五态守卫：一旦恢复接线，data_insufficient / error / not_evaluated
    的经济事件也会被换算成奖励（缺数被奖惩）；
  - 其他奖励写入点（EvoTraj streak / L1 / 生成器代理）各自内联记账，
    量纲守卫 `ensure_reward_source` 依赖每个调用点自觉；
  - 09-26 A0 已把全部 legacy 奖励写入点用 FUXI_LEGACY_* 隔离（默认 off）。

本模块把「经济事件 → 奖励写入」收敛成单一函数，并先以 **shadow** 消费：
  - 五态守卫：仅 eval_status="valid" 且数值有限的事件可产生奖励；
    data_insufficient / error / not_evaluated / precheck_rejected 永不奖惩；
    缺失或 NaN 的 jq_return 视为「无读数」（not_evaluated）——不得当 0 或负值；
  - 写前强制 `ensure_reward_source`（换源重置，禁止跨量纲混算）；
  - 默认 shadow：只落 `data/reward_guard_ledger.jsonl`（只读数不生效），
    MAB 臂零改动；写生效需显式 `FUXI_REWARD_GUARD=enforce`。

与 A0 隔离语义的关系：shadow 不写任何 MAB 字段、不复活 FUXI_LEGACY_* 门控的
旧通道；enforce 是**新增显式开关**，翻默认前须按项目铁律做 A/B 与端到端对账。

开关
--------------------------------------------------------------------
  FUXI_REWARD_GUARD = off | shadow(默认) | enforce
    off     : 完全不动作（不落台账）
    shadow  : 五态守卫 + 量纲守卫预检 + 台账（**只读数不生效**）
    enforce : 在 shadow 基础上真实写入 MAB（ensure_reward_source → 记账 → save）

行尾: 本文件为 LF（新建）。
"""
from __future__ import annotations

import json
import math
import os
from datetime import datetime
from pathlib import Path

MODES = ("off", "shadow", "enforce")
DEFAULT_MODE = "shadow"

# JQ 量纲（与 ralph_loop._mab_record_result 历史实现逐字一致）:
#   reward = clip((jq_return - 182.57) / 100 * 5 * (1 + 0.2 * streak), -1, 1)
JQ_REWARD_REF_RETURN = 182.57
JQ_REWARD_GAIN = 5.0
STREAK_BONUS = 0.2
FAILURE_RETURN_THRESHOLD = -20.0
FAILURE_SHARPE_THRESHOLD = -0.5
COOLING_AT_FAILURES = 3

_REWARDABLE_STATUS = "valid"
_LEDGER_NAME = "reward_guard_ledger.jsonl"
# research/factor_alchemy/reward_guard.py -> parents[2] == e:/quant
_DEFAULT_LEDGER = Path(__file__).resolve().parents[2] / "data" / _LEDGER_NAME


def guard_mode() -> str:
    v = str(os.environ.get("FUXI_REWARD_GUARD", "") or "").strip().lower()
    return v if v in MODES else DEFAULT_MODE


def _finite(x) -> bool:
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def _as_float_or_none(x):
    return float(x) if _finite(x) else None


def jq_reward_from_return(jq_return, streak: int = 0) -> float:
    """JQ 收益 → 奖励（唯一量纲实现；1.0 ≈ 超出王者基准 20pp）。"""
    relative = (float(jq_return) - JQ_REWARD_REF_RETURN) / 100.0
    mult = 1.0 + STREAK_BONUS * int(streak or 0)
    return max(-1.0, min(1.0, relative * JQ_REWARD_GAIN * mult))


def classify_economic_event(evidence: dict) -> dict:
    """五态守卫：返回 {rewardable, eval_status, status_source, reason}。

    evidence 契约: {"eval_status": str, "value": float|None, "source": str, ...}
      - 显式 eval_status 优先；
      - 缺失状态时仅在 value 有限时按 valid 推断，并留痕
        status_source="inferred_from_value"（绝不静默把缺状态当 valid 上报）；
      - valid 但 value 缺失/NaN → 视为 not_evaluated（无读数 ≠ 坏读数）。
    """
    ev = dict(evidence or {})
    value = ev.get("value")
    status = str(ev.get("eval_status", "") or "").strip()
    if not status:
        status = "valid" if _finite(value) else "not_evaluated"
        status_source = "inferred_from_value"
    else:
        status_source = "explicit"
    if status != _REWARDABLE_STATUS:
        return {"rewardable": False, "eval_status": status,
                "status_source": status_source,
                "reason": f"eval_status={status} 永不奖惩 (五态守卫)"}
    if not _finite(value):
        return {"rewardable": False, "eval_status": "not_evaluated",
                "status_source": status_source,
                "reason": "valid 状态但数值缺失/NaN → 视为无读数 (不得当 0)"}
    return {"rewardable": True, "eval_status": status,
            "status_source": status_source, "reason": f"rewardable({status_source})"}


def _append_ledger(ledger_path, rec: dict) -> None:
    p = Path(ledger_path) if ledger_path else _DEFAULT_LEDGER
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a", encoding="utf-8", newline="") as f:
            f.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")
    except Exception:
        pass  # 台账写失败不阻塞（守卫不得成为新故障点）


def _bookkeep_outcome(d, jq_return, jq_sharpe, streak: int) -> None:
    """legacy 记账（仅 enforce）：失败阈值 < -20 / Sharpe < -0.5；缺失值不计成败。"""
    if not (_finite(jq_return) and _finite(jq_sharpe)):
        return  # 缺失不得当 0；不许猜成败
    if float(jq_return) < FAILURE_RETURN_THRESHOLD or float(jq_sharpe) < FAILURE_SHARPE_THRESHOLD:
        d.failures = int(getattr(d, "failures", 0) or 0) + 1
        if d.failures >= COOLING_AT_FAILURES:
            d.status = "cooling"
    else:
        d.successes = int(getattr(d, "successes", 0) or 0) + 1
        _jr = list(getattr(d, "jq_results", []) or [])
        _jr.append({"return": float(jq_return), "sharpe": float(jq_sharpe),
                    "streak": int(streak or 0)})
        d.jq_results = _jr


def consume_verified_reward(
    mab,
    direction_name: str,
    *,
    evidence: dict,
    reward: float = None,
    jq_return: float = None,
    jq_sharpe: float = None,
    streak: int = 0,
    source: str = "jq",
    ledger_path=None,
) -> dict:
    """唯一奖励通道：五态守卫 → 影子/生效消费 → 追加台账。

    shadow（默认）: 只读数不生效 —— 解析臂、检查是否需换源、计算 would_reward，
                    **不修改任何 MAB 字段**，只追加台账记录。
    enforce      : 写前逐臂 `ensure_reward_source`，再执行 legacy 记账
                   （n_rewards / expected_reward / 成败 / 冷却）并 save_state。

    Returns: 台账记录 dict（status ∈ off/no_reward/no_arms/shadow_only/applied）。
    """
    mode = guard_mode()
    ev = dict(evidence or {})
    ev.setdefault("source", source)
    if "value" not in ev:
        ev["value"] = jq_return
    if reward is None and jq_return is not None and _finite(jq_return):
        reward = jq_reward_from_return(jq_return, streak)
    verdict = classify_economic_event(ev)

    rec = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "mode": mode,
        "direction": str(direction_name or ""),
        "source": source,
        "eval_status": verdict["eval_status"],
        "status_source": verdict["status_source"],
        "value": _as_float_or_none(ev.get("value")),
        "reward": _as_float_or_none(reward) if verdict["rewardable"] else None,
        "streak": int(streak or 0),
        "ref": str(ev.get("ref", "") or ""),
        "applied": False,
        "n_arms": 0,
        "arms": [],
        "source_switch_needed": [],
        "status": "",
        "reason": verdict["reason"],
    }

    if mode == "off":
        rec["status"] = "off"
        return rec  # off = 完全不动作（不落台账）

    if not verdict["rewardable"]:
        rec["status"] = "no_reward"
        _append_ledger(ledger_path, rec)
        return rec

    arms = []
    try:
        arms = list(mab.resolve_arms(direction_name) or [])
    except Exception as e:
        rec["reason"] = f"resolve_arms 失败: {type(e).__name__}: {e}"
    _arm_ids = [str(getattr(d, "direction_id", "") or getattr(d, "name", "")) for d in arms]
    if not arms:
        rec["status"] = "no_arms"
        rec["reason"] = rec["reason"] if "resolve_arms" in rec["reason"] else "无对应臂（方向未注册）"
        _append_ledger(ledger_path, rec)
        return rec

    rec["n_arms"] = len(arms)
    rec["arms"] = _arm_ids
    rec["source_switch_needed"] = [
        _id for d, _id in zip(arms, _arm_ids)
        if str(getattr(d, "reward_source", "") or "") not in ("", source)]

    if mode != "enforce":
        rec["status"] = "shadow_only"
        _append_ledger(ledger_path, rec)
        return rec

    for d in arms:
        try:
            mab.ensure_reward_source(d, source)
        except Exception:
            pass
        n = int(getattr(d, "n_rewards", 0) or 0) + 1
        d.n_rewards = n
        d.expected_reward = (float(getattr(d, "expected_reward", 0.0) or 0.0) * (n - 1)
                             + float(rec["reward"])) / n
        _bookkeep_outcome(d, jq_return, jq_sharpe, streak)
    try:
        mab.save_state()
    except Exception:
        pass
    rec["applied"] = True
    rec["status"] = "applied"
    _append_ledger(ledger_path, rec)
    return rec
