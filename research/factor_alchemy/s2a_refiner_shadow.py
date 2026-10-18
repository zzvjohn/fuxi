"""M6-A (2026-09-28): S2-a 精炼器补路影子遥测 + A/B 对照底座（只读数, 不裁决）。

背景（方案 §M6「精炼器/机制计划转正」，P2 长期）:
  · S2-a 精炼器补路 (开关 `FUXI_EVO_REFINER = off|shadow(默认)|on`) 自
    2026-09-12 起默认 shadow —— 只 `print` 一行「本应触发」，**读数不落盘**。
    后果三条:
      (a) 触发率无分母 ⇒ 跨轮无法汇总「到底多常命中周期门」;
      (b) `on` 档若启用，子代产出 / S1 通过率无归因账本 ⇒ 转正无工程证据;
      (c) A/B 对照两侧无**同一口径**的读数端点。
  · 本模块只补「留痕 + 汇总」：shadow 档零行为回归（不碰 MAB、不碰任何裁决字段），
    任何异常显式返回 status=error 并入 lines（**不抛出、不静默** —— 与 M1/M2 的
    "修静默吞异常" 同一条纪律）。

三态 / 两开关（默认即契约）:
  FUXI_EVO_REFINER        : off | shadow(默认) | on   生产门（在 ralph_loop 消费）
  FUXI_REFINER_SHADOW_LOG : on(默认) | off            本模块账本总开关
                            off ⇒ 一行不写；**不影响** FUXI_EVO_REFINER 的任何行为。

账本 `data/s2a_refiner_shadow.jsonl`（append-only；每轮 run() 至多 2 行）:
  行 1 stage="refine" : 门控读数。event ∈
        off_skipped / on_gate_miss / shadow_gate_miss / shadow_would_fire /
        on_no_seed / on_no_kids / on_refined / error
        ★ 每轮都落一行 ⇒ 「触发率」的分母**恒存在**（含未命中与未成熟）。
  行 2 stage="eval"   : 子代归因读数。**仅当**本轮批次内存在 `_source="gp_evolve"`
        候选时落盘 —— 主路 / 子代分开计数，S1 分母带 read / refused 两栏。

🔴 读数边界（M6 原文「影子读数可能被误读为经济证据 → 只作为工程对照」）:
  · 本账本全部字段是**工程产出读数**（产出数 / 通过数 / 结构新颖数），
    **不是**经济证据，不得据以调整任何阈值、注册任何候选、宣称任何 alpha;
  · `s1_passed` 只是 S1 单阶段布尔；`s1_eval_status` 五态独立记录，
    0/N **不得**读作「干净驳回」（须查 eval_status）。

签名口径（R4 结构新颖数）:
  复用 `structure_dedup._sig`（abstraction ∈ {literal, field}），
  literal = 结构 + 具体字段/常量; field = 结构骨架（字段抽象为 $F）。
  不可用时显式记 sig_mode="unavailable" + sig_error（**不降级替代**）。
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

SCHEMA = "fuxi_s2a_refiner_shadow/v1"

MODE_ENV = "FUXI_EVO_REFINER"
LOG_ENV = "FUXI_REFINER_SHADOW_LOG"
EVERY_ENV = "FUXI_EVO_REFINER_EVERY"
MIN_PULLS_ENV = "FUXI_EVO_REFINER_MIN_PULLS"

MODE_VALUES = ("off", "shadow", "on")
LOG_VALUES = ("on", "off")
DEFAULT_MODE = "shadow"
DEFAULT_LOG_MODE = "on"

ARM_SHADOW = "shadow"
ARM_ON = "on"

CHILD_SOURCE = "gp_evolve"          # _phase_generate_evo 打的 _source 标记

REFINE_EVENTS = (
    "off_skipped",          # 开关 off（显式可见, 不是"没发生"）
    "on_gate_miss",         # on 但方向未成熟 / 周期未命中
    "shadow_gate_miss",     # shadow 且本应也不触发
    "shadow_would_fire",    # shadow 且「若无开关本应触发」★ 触发率的分子
    "on_no_seed",           # on + 门命中, 但种子池为空
    "on_no_kids",           # on + 门命中 + 有种子, 但 evo 未产子代
    "on_refined",           # on + 门命中 + 产子代
    "error",                # 遥测自身异常（显式, 不静默）
)

_LEDGER_NAME = "s2a_refiner_shadow.jsonl"
DEFAULT_LEDGER = Path(__file__).resolve().parents[2] / "data" / _LEDGER_NAME

__all__ = [
    "SCHEMA", "MODE_ENV", "LOG_ENV", "EVERY_ENV", "MIN_PULLS_ENV",
    "MODE_VALUES", "LOG_VALUES", "DEFAULT_MODE", "DEFAULT_LOG_MODE",
    "ARM_SHADOW", "ARM_ON", "CHILD_SOURCE", "REFINE_EVENTS", "DEFAULT_LEDGER",
    "resolve_log_mode", "classify_refine_event", "fire_readiness", "structure_key",
    "build_refine_record", "build_eval_record", "append_shadow_entry",
    "s2a_refiner_refine_telemetry", "s2a_refiner_eval_telemetry",
    "summarize_ledger",
]


# ══════════════════════════════════════════════════════════════════
# 开关解析
# ══════════════════════════════════════════════════════════════════
def resolve_log_mode(env: Optional[Dict[str, str]] = None) -> str:
    """账本总开关。非法值回落默认 on（调用方负责可见性提示）。"""
    e = os.environ if env is None else env
    raw = str(e.get(LOG_ENV, "") or "").strip().lower()
    return raw if raw in LOG_VALUES else DEFAULT_LOG_MODE


def _gate_env_numbers(env: Dict[str, str]) -> Tuple[int, int]:
    """门参数（与 ralph_loop._s2a_refiner_gate 同默认: 3 / 5）。"""
    try:
        min_pulls = int(env.get(MIN_PULLS_ENV, "3") or 3)
    except (TypeError, ValueError):
        min_pulls = 3
    try:
        every = max(1, int(env.get(EVERY_ENV, "5") or 5))
    except (TypeError, ValueError):
        every = 5
    return min_pulls, every


# ══════════════════════════════════════════════════════════════════
# 纯函数：事件分类
# ══════════════════════════════════════════════════════════════════
def fire_readiness(gate: Optional[Dict], pulls: Optional[int]) -> Tuple[bool, Optional[bool]]:
    """(mature, period_ok) —— 与 ralph_loop `_s2a_refiner_gate` 同口径。

    ★ 生产事实: gate["mature"] 只表示「方向成熟」，**不含**周期门；周期门
    (`pulls % every == 0`) 由调用点单独判。故此处必须同时拿 pulls 才能判定
    「本应触发」—— 漏掉周期门会把非周期轮误报成 would_fire（**假读数**）。
    pulls 未知 ⇒ period_ok=None（保守: 不得据此声称会触发）。
    """
    g = gate if isinstance(gate, dict) else {}
    mature = bool(g.get("mature", False))
    raw_every = g.get("every", None)
    if raw_every is None:
        every = 5                       # 与生产默认 (FUXI_EVO_REFINER_EVERY=5) 同
    else:
        try:
            every = max(1, int(raw_every))   # 与生产 max(1, ...) 同口径 (0 ⇒ 1)
        except (TypeError, ValueError):
            every = 5
    if pulls is None:
        return mature, None
    return mature, (int(pulls) % every == 0)


def classify_refine_event(
    gate: Optional[Dict],
    *,
    pulls: Optional[int] = None,
    n_seeds: Optional[int] = None,
    n_kids: Optional[int] = None,
    error: Optional[str] = None,
) -> str:
    """把门控读数 + 本轮产出分类成 REFINE_EVENTS 之一（纯函数, 可测）。

    gate 为 `_s2a_refiner_gate` 的返回（mode/min_pulls/every/mature/probe/reason）。
    无 gate（None）视作门未求值 ⇒ on_gate_miss / shadow_gate_miss 按 mode 分派。
    """
    if error:
        return "error"
    g = gate if isinstance(gate, dict) else {}
    mode = str(g.get("mode", DEFAULT_MODE) or DEFAULT_MODE).strip().lower()
    if mode not in MODE_VALUES:
        mode = DEFAULT_MODE

    if mode == "off":
        return "off_skipped"

    mature, period_ok = fire_readiness(g, pulls)
    probe = bool(g.get("probe", False))
    if mode == "shadow":
        return ("shadow_would_fire" if (mature and period_ok is True)
                else "shadow_gate_miss")

    # mode == "on"（probe 已内含成熟 + 周期门）
    if not probe:
        return "on_gate_miss"
    if n_seeds is not None and int(n_seeds) <= 0:
        return "on_no_seed"
    if n_kids is not None and int(n_kids) <= 0:
        return "on_no_kids"
    return "on_refined"


# ══════════════════════════════════════════════════════════════════
# 纯函数：结构签名（R4）
# ══════════════════════════════════════════════════════════════════
def structure_key(expression: str, abstraction: str = "literal") -> Optional[str]:
    """单条表达式的结构签名（复用 structure_dedup._sig）。

    返回 None ⇒ 未求得（调用方须记 sig 缺口, **不得**用字符串归一化顶替）。
    """
    expr = str(expression or "").strip()
    if not expr:
        return None
    try:
        import structure_dedup as _sd
        ded = _sd.ASTStructureDeductor()
        pf = ded.parse_one(expr, 0, "")
        if pf is None or pf.body is None:
            return None
        return _sd._sig(pf.body, abstraction if abstraction in ("literal", "field", "full") else "literal")
    except Exception:
        return None


def _cand_expr(c: Any) -> str:
    if isinstance(c, dict):
        return str(c.get("formula") or c.get("expression") or "")
    return str(getattr(c, "formula", "") or getattr(c, "expression", ""))


def _cand_flag(c: Any, key: str, default: Any = None) -> Any:
    if isinstance(c, dict):
        return c.get(key, default)
    return getattr(c, key, default)


def _res_field(r: Any, key: str, default: Any = None) -> Any:
    if isinstance(r, dict):
        return r.get(key, default)
    return getattr(r, key, default)


# ══════════════════════════════════════════════════════════════════
# 纯函数：记录构造
# ══════════════════════════════════════════════════════════════════
def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def build_refine_record(
    *,
    gate: Optional[Dict],
    run_id: str = "",
    direction: str = "",
    paradigm: str = "",
    generator: str = "",
    pulls: int = 0,
    n_seeds: Optional[int] = None,
    n_kids: Optional[int] = None,
    n_main_candidates: Optional[int] = None,
    seed_sources: str = "",
    downgrade: str = "",
    error: str = "",
    ts: Optional[str] = None,
    env: Optional[Dict[str, str]] = None,
) -> Dict:
    """行 1「门控读数」记录（纯函数, 不落盘）。"""
    e = os.environ if env is None else env
    g = gate if isinstance(gate, dict) else {}
    min_pulls, every = _gate_env_numbers(e)
    mode = str(g.get("mode", "") or resolve_mode_lenient(e)).strip().lower()
    if mode not in MODE_VALUES:
        mode = DEFAULT_MODE
    event = classify_refine_event(g, pulls=pulls, n_seeds=n_seeds,
                                  n_kids=n_kids, error=error or None)
    pulls_i = int(pulls or 0)
    _mature, _period_ok = fire_readiness(g, pulls_i)
    would_fire = bool(mode == "shadow" and _mature and _period_ok is True)
    return {
        "schema": SCHEMA,
        "ts": ts or _now_iso(),
        "stage": "refine",
        "event": event,
        "arm": ARM_ON if mode == "on" else (ARM_SHADOW if mode == "shadow" else "off"),
        "mode": mode,
        "min_pulls": int(g.get("min_pulls", min_pulls) or min_pulls),
        "every": int(g.get("every", every) or every),
        "mature": bool(g.get("mature", False)),
        "probe": bool(g.get("probe", False)),
        "would_fire": would_fire,
        "pulls": pulls_i,
        "run_id": str(run_id or ""),
        "direction": str(direction or ""),
        "paradigm": str(paradigm or ""),
        "generator": str(generator or ""),
        "n_seeds": None if n_seeds is None else int(n_seeds),
        "n_kids": None if n_kids is None else int(n_kids),
        "n_main_candidates": (None if n_main_candidates is None
                              else int(n_main_candidates)),
        "seed_sources": str(seed_sources or ""),
        "gate_reason": str(g.get("reason", "") or ""),
        "downgrade": str(downgrade or ""),
        "error": str(error or ""),
    }


def resolve_mode_lenient(env: Optional[Dict[str, str]] = None) -> str:
    """生产门开关的宽松解析（非法/缺失 ⇒ 默认 shadow）。"""
    e = os.environ if env is None else env
    raw = str(e.get(MODE_ENV, "") or "").strip().lower()
    return raw if raw in MODE_VALUES else DEFAULT_MODE


def build_eval_record(
    candidates: Sequence[Any],
    results: Sequence[Any],
    *,
    run_id: str = "",
    paradigm: str = "",
    generator: str = "",
    ts: Optional[str] = None,
    env: Optional[Dict[str, str]] = None,
) -> Optional[Dict]:
    """行 2「子代归因读数」记录（纯函数）。

    仅当批次内存在 `_source == "gp_evolve"` 候选时返回记录；否则返回 None
    （避免给非精炼轮灌水；分母 = 「本应存在的子代轮」而非全部轮次）。

    归因口径:
      · 主路 / 子代按 `_source` 标记二分（`_phase_generate_evo` 在 L3511 打标）;
      · 主路候选**不带** `_source`（forge/llm 路径），故按 `!= CHILD_SOURCE` 计;
      · S1 分母带 read / refused：read = s1_eval_status == "valid"，
        refused = not_evaluated + data_insufficient + error（**并集**, 不含 valid）;
      · 结果与候选按 `zip` 索引对齐（与 learning_feedback 同口径）。
    """
    e = os.environ if env is None else env
    cands = list(candidates or [])
    res = list(results or [])
    n_main = n_child = 0
    main_sig: set = set()
    child_sig: set = set()
    sig_missing = 0
    child_stats = {"n": 0, "s1_pass": 0, "s1_read": 0, "s1_refused": 0,
                   "s1_not_evaluated": 0, "s1_data_insufficient": 0, "s1_error": 0,
                   "n_days_missing": 0, "forbidden_hit": 0, "struct_dup_hit": 0,
                   "s2_pass": 0, "s5_pass": 0, "s5_read": 0, "s5_refused": 0,
                   "s4_pass": 0}
    main_stats = {"n": 0, "s1_pass": 0, "s1_read": 0, "s1_refused": 0,
                  "s2_pass": 0, "s5_pass": 0, "s5_read": 0, "s5_refused": 0}

    for i, c in enumerate(cands):
        is_child = str(_cand_flag(c, "_source", "") or "") == CHILD_SOURCE
        expr = _cand_expr(c)
        sig = structure_key(expr, "literal")
        if sig is None:
            sig_missing += 1
        if is_child:
            n_child += 1
            child_stats["n"] += 1
            if sig is not None:
                child_sig.add(sig)
        else:
            n_main += 1
            main_stats["n"] += 1
            if sig is not None:
                main_sig.add(sig)

        r = res[i] if i < len(res) else None
        if r is None:
            if is_child:
                child_stats["s1_refused"] += 1
                child_stats["s1_error"] += 1
                child_stats["s5_refused"] += 1
            else:
                main_stats["s1_refused"] += 1
                main_stats["s5_refused"] += 1
            continue
        passed = bool(_res_field(r, "s1_passed", False))
        status = str(_res_field(r, "s1_eval_status", "not_evaluated") or
                     "not_evaluated").strip().lower()
        s2_pass = bool(_res_field(r, "s2_passed", False))
        s4_pass = bool(_res_field(r, "s4_passed", False))
        s5_pass = bool(_res_field(r, "s5_passed", False))
        s5_status = str(_res_field(r, "s5_eval_status", "not_evaluated") or
                        "not_evaluated").strip().lower()
        if is_child:
            if passed:
                child_stats["s1_pass"] += 1
            if s2_pass:
                child_stats["s2_pass"] += 1
            if s4_pass:
                child_stats["s4_pass"] += 1
            if s5_pass:
                child_stats["s5_pass"] += 1
            if s5_status == "valid":
                child_stats["s5_read"] += 1
            else:
                child_stats["s5_refused"] += 1
            if status == "valid":
                child_stats["s1_read"] += 1
            else:
                child_stats["s1_refused"] += 1
                if status == "data_insufficient":
                    child_stats["s1_data_insufficient"] += 1
                elif status == "error":
                    child_stats["s1_error"] += 1
                else:
                    child_stats["s1_not_evaluated"] += 1
            if int(_res_field(r, "s1_n_days", -1) or -1) < 0:
                child_stats["n_days_missing"] += 1
            if bool(_res_field(r, "s1_forbidden_hit", False)):
                child_stats["forbidden_hit"] += 1
            if bool(_res_field(r, "s1_struct_dup_hit", False)):
                child_stats["struct_dup_hit"] += 1
        else:
            if passed:
                main_stats["s1_pass"] += 1
            if s2_pass:
                main_stats["s2_pass"] += 1
            if s5_pass:
                main_stats["s5_pass"] += 1
            if s5_status == "valid":
                main_stats["s5_read"] += 1
            else:
                main_stats["s5_refused"] += 1
            if status == "valid":
                main_stats["s1_read"] += 1
            else:
                main_stats["s1_refused"] += 1

    if n_child == 0:
        return None

    novel_literal = len(child_sig - main_sig)
    # field 骨架新颖数（第二读数, 结构骨架层面）
    child_field: set = set()
    for c in cands:
        if str(_cand_flag(c, "_source", "") or "") == CHILD_SOURCE:
            s = structure_key(_cand_expr(c), "field")
            if s is not None:
                child_field.add(s)
    main_field: set = set()
    for c in cands:
        if str(_cand_flag(c, "_source", "") or "") != CHILD_SOURCE:
            s = structure_key(_cand_expr(c), "field")
            if s is not None:
                main_field.add(s)

    rec = {
        "schema": SCHEMA,
        "ts": ts or _now_iso(),
        "stage": "eval",
        "event": "child_attribution",
        "arm": ARM_ON if resolve_mode_lenient(e) == "on" else ARM_SHADOW,
        "mode": resolve_mode_lenient(e),
        "run_id": str(run_id or ""),
        "paradigm": str(paradigm or ""),
        "generator": str(generator or ""),
        "n_candidates": len(cands),
        "n_main": n_main,
        "n_child": n_child,
        "n_results": len(res),
        "main": main_stats,
        "child": child_stats,
        "s1_rate_main": (main_stats["s1_pass"] / main_stats["n"]) if main_stats["n"] else None,
        "s1_rate_child": (child_stats["s1_pass"] / child_stats["n"]) if child_stats["n"] else None,
        "novel_literal": novel_literal,
        "novel_field": len(child_field - main_field),
        "n_sig_main": len(main_sig),
        "n_sig_child": len(child_sig),
        "sig_missing": sig_missing,
        "sig_mode": "structure_dedup._sig/literal+field",
    }
    return rec


# ══════════════════════════════════════════════════════════════════
# 落盘（append-only, 单行 LF）
# ══════════════════════════════════════════════════════════════════
def append_shadow_entry(entry: Dict, path: Optional[Any] = None) -> str:
    """追加一行 JSON 到影子账本（append-only, 单行 LF）。返回实际路径。"""
    p = Path(path) if path else DEFAULT_LEDGER
    if p.parent and not p.parent.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(entry, ensure_ascii=False, sort_keys=True)
    with open(p, "a", encoding="utf-8", newline="\n") as f:
        f.write(line + "\n")
    return str(p)


# ══════════════════════════════════════════════════════════════════
# 接线入口 1：门控读数（每轮一行）
# ══════════════════════════════════════════════════════════════════
def s2a_refiner_refine_telemetry(
    *,
    gate: Optional[Dict],
    run_id: str = "",
    direction: str = "",
    paradigm: str = "",
    generator: str = "",
    pulls: int = 0,
    n_seeds: Optional[int] = None,
    n_kids: Optional[int] = None,
    n_main_candidates: Optional[int] = None,
    seed_sources: str = "",
    downgrade: str = "",
    error: str = "",
    ledger_path: Optional[Any] = None,
    env: Optional[Dict[str, str]] = None,
) -> Dict:
    """门控遥测写入（三态安全：off 也落一行, 保证分母存在）。

    返回 dict: schema / mode(log) / event / status / path / lines / record
    status ∈ {logged, skipped_switch_off, error}。**绝不抛出**。
    """
    e = os.environ if env is None else env
    log_mode = resolve_log_mode(e)
    rep: Dict[str, Any] = {"schema": SCHEMA, "log_mode": log_mode,
                           "status": "skipped_switch_off", "event": "",
                           "path": "", "lines": [], "record": None}
    lines: List[str] = rep["lines"]

    raw = str(e.get(LOG_ENV, "") or "").strip().lower()
    if raw and raw not in LOG_VALUES:
        lines.append(f"  [S2a] ⚠️ {LOG_ENV}={raw!r} 非法 → 按默认 "
                     f"{DEFAULT_LOG_MODE} 处理")
    if log_mode == "off":
        return rep

    try:
        rec = build_refine_record(
            gate=gate, run_id=run_id, direction=direction, paradigm=paradigm,
            generator=generator, pulls=pulls, n_seeds=n_seeds, n_kids=n_kids,
            n_main_candidates=n_main_candidates, seed_sources=seed_sources,
            downgrade=downgrade, error=error, env=e)
        rep["record"] = rec
        rep["event"] = rec["event"]
        rep["path"] = append_shadow_entry(rec, ledger_path)
        rep["status"] = "logged"
    except Exception as _te:      # 显式, 不静默
        rep["status"] = "error"
        lines.append(f"  [S2a] ⚠️ 精炼器遥测写入失败 (不阻塞): {_te}")
    return rep


# ══════════════════════════════════════════════════════════════════
# 接线入口 2：子代归因读数（仅子代轮）
# ══════════════════════════════════════════════════════════════════
def s2a_refiner_eval_telemetry(
    candidates: Sequence[Any],
    results: Sequence[Any],
    *,
    run_id: str = "",
    paradigm: str = "",
    generator: str = "",
    ledger_path: Optional[Any] = None,
    env: Optional[Dict[str, str]] = None,
) -> Dict:
    """子代归因写入。无子代 ⇒ status=no_child（**不写行**, 非错误）。绝不抛出。"""
    e = os.environ if env is None else env
    log_mode = resolve_log_mode(e)
    rep: Dict[str, Any] = {"schema": SCHEMA, "log_mode": log_mode,
                           "status": "skipped_switch_off", "event": "",
                           "path": "", "lines": [], "record": None}
    lines: List[str] = rep["lines"]
    if log_mode == "off":
        return rep
    try:
        rec = build_eval_record(candidates, results, run_id=run_id,
                                paradigm=paradigm, generator=generator, env=e)
        if rec is None:
            rep["status"] = "no_child"
            return rep
        rep["record"] = rec
        rep["event"] = rec["event"]
        rep["path"] = append_shadow_entry(rec, ledger_path)
        rep["status"] = "logged"
        _ch = rec["child"]
        lines.append(
            f"  [S2a/telemetry] 子代归因: {_ch['n']} 子代 / {rec['n_main']} 主路, "
            f"子代 S1 {_ch['s1_pass']}/{_ch['n']} (read={_ch['s1_read']}, "
            f"refused={_ch['s1_refused']}), 结构新颖 literal={rec['novel_literal']} "
            f"field={rec['novel_field']}")
    except Exception as _te:
        rep["status"] = "error"
        lines.append(f"  [S2a] ⚠️ 子代归因遥测失败 (不阻塞): {_te}")
    return rep


# ══════════════════════════════════════════════════════════════════
# 汇总（供 A/B 判定与人工查阅）
# ══════════════════════════════════════════════════════════════════
def _read_ledger(path: Optional[Any] = None) -> Tuple[List[Dict], int]:
    p = Path(path) if path else DEFAULT_LEDGER
    if not p.exists():
        return [], 0
    rows: List[Dict] = []
    n_bad = 0
    with open(p, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                rows.append(json.loads(ln))
            except Exception:
                n_bad += 1
    return rows, n_bad


def summarize_ledger(path: Optional[Any] = None,
                     arm: Optional[str] = None) -> Dict:
    """汇总账本：触发率（带分母）/ 子代产出 / S1 通过率 / 结构新颖数。

    arm ∈ {None(全部), "shadow", "on"}。
    status ∈ {no_ledger, empty, valid} —— 「没有读数」三态显式分开。
    """
    p = Path(path) if path else DEFAULT_LEDGER
    rows, n_bad = _read_ledger(p)
    if not p.exists():
        return {"schema": SCHEMA, "status": "no_ledger", "path": str(p),
                "n_lines": 0, "n_bad_lines": 0}
    if not rows:
        return {"schema": SCHEMA, "status": "empty", "path": str(p),
                "n_lines": 0, "n_bad_lines": n_bad}

    sel = [r for r in rows if (arm is None or r.get("arm") == arm)]
    refine = [r for r in sel if r.get("stage") == "refine"]
    evals = [r for r in sel if r.get("stage") == "eval"]

    # 触发率：分母 = 门被求值的轮次（不含 error）
    gated = [r for r in refine if r.get("event") != "error"]
    fired = [r for r in refine if r.get("event") == "shadow_would_fire"]
    on_refined = [r for r in refine if r.get("event") == "on_refined"]
    by_paradigm: Dict[str, Dict[str, int]] = {}
    for r in refine:
        k = str(r.get("paradigm", "") or "(none)")
        d = by_paradigm.setdefault(k, {"n": 0, "would_fire": 0, "on_refined": 0})
        d["n"] += 1
        if r.get("event") == "shadow_would_fire":
            d["would_fire"] += 1
        if r.get("event") == "on_refined":
            d["on_refined"] += 1

    n_child = sum(int((r.get("child") or {}).get("n", 0) or 0) for r in evals)
    n_child_pass = sum(int((r.get("child") or {}).get("s1_pass", 0) or 0)
                       for r in evals)
    n_child_read = sum(int((r.get("child") or {}).get("s1_read", 0) or 0)
                       for r in evals)
    n_child_ref = sum(int((r.get("child") or {}).get("s1_refused", 0) or 0)
                      for r in evals)
    n_main = sum(int(r.get("n_main", 0) or 0) for r in evals)
    n_main_pass = sum(int((r.get("main") or {}).get("s1_pass", 0) or 0)
                      for r in evals)

    def _agg(where: str, key: str) -> int:
        return sum(int((r.get(where) or {}).get(key, 0) or 0) for r in evals)

    return {
        "schema": SCHEMA,
        "status": "valid",
        "path": str(p),
        "n_lines": len(rows),
        "n_bad_lines": n_bad,
        "arm_filter": arm,
        "n_refine_rows": len(refine),
        "n_eval_rows": len(evals),
        "fire": {
            "n_gated_rounds": len(gated),
            "n_would_fire": len(fired),
            "would_fire_rate": (len(fired) / len(gated)) if gated else None,
            "n_on_refined": len(on_refined),
            "by_paradigm": by_paradigm,
            "n_error_rounds": len([r for r in refine if r.get("event") == "error"]),
        },
        "children": {
            "n_total": n_child,
            "n_s1_read": n_child_read,
            "n_s1_refused": n_child_ref,
            "n_s1_pass": n_child_pass,
            "s1_pass_rate_over_n": (n_child_pass / n_child) if n_child else None,
            "s1_pass_rate_over_read": (n_child_pass / n_child_read) if n_child_read else None,
            "novel_literal": sum(int(r.get("novel_literal", 0) or 0) for r in evals),
            "novel_field": sum(int(r.get("novel_field", 0) or 0) for r in evals),
            "n_s2_pass": _agg("child", "s2_pass"),
            "n_s4_pass": _agg("child", "s4_pass"),
            "n_s5_pass": _agg("child", "s5_pass"),
            "n_s5_read": _agg("child", "s5_read"),
            "n_s5_refused": _agg("child", "s5_refused"),
        },
        "main": {
            "n_total": n_main,
            "n_s1_pass": n_main_pass,
            "s1_rate": (n_main_pass / n_main) if n_main else None,
            "n_s2_pass": _agg("main", "s2_pass"),
            "n_s5_pass": _agg("main", "s5_pass"),
            "n_s5_read": _agg("main", "s5_read"),
        },
        "events": {k: len([r for r in refine if r.get("event") == k])
                   for k in REFINE_EVENTS},
    }
