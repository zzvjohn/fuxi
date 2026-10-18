#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stage2 PASS -> 完整 S1~S5 核验链路（执行器）
==========================================================================

存在理由（2026-09-25 立项，P-20260925-001 的落地实现）
--------------------------------------------------------------------
在本次审计前，伏羲存在一条**可跑但无执行器**的通道：
  - 通道本体: scripts/s135_candidate_check.py（自述「把 status=candidate 的因子
    重跑完整 S1-S5 管线」，内部实调 MultiStageValidator + S5JointFilter）
  - 通道调用方: **0**（全仓库 grep 无生产引用；上次运行 2026-09-08 13:32）
  - 结果: Stage2 判 PASS -> add_to_pool() -> 入池即终点，永久滞留 candidate
本脚本就是那个缺失的执行器。

与 s135_candidate_check.py 的分工
--------------------------------------------------------------------
  - `s135_candidate_check.py`  取数是 **passed_factor_pool.csv** 里 status=candidate
                              的存量（1292 行池 / 57 条 candidate），是**存量盘点**工具。
  - `stage2_s5_chain.py`（本脚本）取数是 **stage1_factor_proposals.json** 里
                              status=PASS 的**新增**条目，是**增量闭环**执行器。
    两者互补：存量用 s135 盘，新增用本脚本闭环。

四条硬纪律（对应记忆铁律）
--------------------------------------------------------------------
1. **只有真实 S5 才写裁决**。S5JointFilter 未就绪时记
   `s5_eval_status="data_insufficient"`，**绝不静默降级**为 IC 年拆分轻量代理后
   再报一个看起来像裁决的读数（「静默降级」铁律）。
2. **非破坏**。不写 `data/s135_candidate_check_result.json`（那是历史证据，
   已被覆盖过一次，见 2026-09-25 备份）。每次运行的读数写独立 run_id 目录。
3. **fail-closed 包住「读」**。因子池读取失败 -> 直接 exit 3，不产出空裁决。
4. **五态不可压布尔**。valid / data_insufficient / precheck_rejected / error /
   not_evaluated 五态原样落盘，分母带 read/refused 计数。

模式与推进路径（2026-09-26 M15 对齐后为唯一权威）
--------------------------------------------------------------------
  off      : 不动作（与接线前行为等价）
  annotate : 刷新队列 + 五态分流 + 滞留台账（零回测；**不跑 S5**），默认
  audit    : annotate + 预检 shadow 证据标注（仍不跑 S5；只标不改）
  enforce  : audit + 对队列前 N 条执行真实 S1~S5（历史别名 run 仍可传入）
  推进路径 : annotate → audit → enforce（逐级打开；annotate/audit 用于看到
             真实逐条状态与分母后再决定是否 enforce，annotate 读数不是裁决）
  🔴 语义修正（M15）：对齐之前，「annotate 不跑 S5」只是注释里的承诺，真实
     行为是只要不带 --queue-only 就会执行 S5（模式未被闸门消费）。现按模式
     闸门执行：annotate/audit 一律只标注；唯一例外是显式 --dry-run 预演。

用法
--------------------------------------------------------------------
  # 只看队列 + 五态分流 + 滞留台账（最便宜，适合每天跑）
  python scripts/stage2_s5_chain.py --queue-only

  # audit：额外附预检 shadow 证据（仍不回测）
  FUXI_STAGE2_S5_CHAIN=audit python scripts/stage2_s5_chain.py

  # enforce：真跑，对队列前 10 条做完整 S1~S5
  FUXI_STAGE2_S5_CHAIN=enforce python scripts/stage2_s5_chain.py --limit 10

  # 只跑指定因子（--only 是过滤器；执行仍需 enforce）
  FUXI_STAGE2_S5_CHAIN=enforce python scripts/stage2_s5_chain.py --only margin_buy_ratio_5d_ma

  # 预演，不跑回测（任何模式都可，显式请求）
  python scripts/stage2_s5_chain.py --limit 3 --dry-run

  # 额外把裁决回写进因子池（追加键，默认关闭）
  FUXI_STAGE2_S5_CHAIN=enforce python scripts/stage2_s5_chain.py --limit 10 --persist-to-pool

产出文件（M15）
--------------------------------------------------------------------
  data/stage2_s5_pending.json    队列快照：逐条五态 + 分母 + 滞留摘要（v2）
  data/stage2_s5_queue_age.json  滞留台账：>14 天 review_due（只报不阻塞，
                                 不自动拒绝/降级——「删除」≠「尚未发生」）

退出码
--------------------------------------------------------------------
  0 = 正常（含「队列为空」）
  2 = S5 引擎未就绪（fail-closed，未产出任何裁决）
  3 = 因子池不可读 / 结构无法识别（fail-closed）
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FA_DIR = ROOT / "research" / "factor_alchemy"
DATA_DIR = ROOT / "data"

sys.path.insert(0, str(FA_DIR))

FACTOR_POOL = DATA_DIR / "stage1_factor_proposals.json"
PENDING_QUEUE = DATA_DIR / "stage2_s5_pending.json"
VERDICT_INDEX = DATA_DIR / "stage2_s5_verdicts.json"
AGE_LEDGER = DATA_DIR / "stage2_s5_queue_age.json"
POOL_CSV = DATA_DIR / "passed_factor_pool.csv"
REPORT_DIR = ROOT / "reports" / "stage2_s5_chain"

# 开关：默认 annotate（只记队列 + 五态分流，零成本）。enforce 才真跑回测。
# off      -> 完全不动作（与接线前行为等价）
# annotate -> 刷新待办队列 + 五态分流 + 滞留台账，但不跑 S5
# audit    -> annotate + 预检 shadow 证据标注（仍不跑 S5）
# enforce  -> audit + 立即对前 N 条执行完整 S1~S5（run 是历史别名）
CHAIN_MODES = ("off", "annotate", "audit", "run", "enforce")
DEFAULT_MODE = "annotate"
DEFAULT_LIMIT = 10

# M15: 滞留评审阈值（>2 周触发评审，只报不阻塞）
RETENTION_REVIEW_DAYS = 14

# 五态（与 multi_stage_validator.ValidationResult 的 *_eval_status 语义对齐）
EVAL_VALID = "valid"
EVAL_DATA_INSUFFICIENT = "data_insufficient"
EVAL_PRECHECK_REJECTED = "precheck_rejected"
EVAL_ERROR = "error"
EVAL_NOT_EVALUATED = "not_evaluated"

_S5_HARD_ERRORS = {
    "held_price_observations_missing",
    "benchmark_dates_missing",
    "weekly_data_unavailable",
    "Price data not available",
}


def _json_safe(o):
    """float32/np 标量 → JSON 可序列化 (np.float32 JSON: default=_json_safe 铁律)。

    2026-09-28 first-enforce 暴露: write_run_dir 直写真实 readings 时遇
    np.float32 (S5 读数来自 pandas 聚合) ⇒ TypeError 整链崩、裁决不落盘。
    与 scripts/s5_fullpool_rerun.py::_json_safe 同一口径。
    """
    try:
        import numpy as _np
    except Exception:
        return str(o)
    if isinstance(o, (_np.floating, _np.integer)):
        return float(o)
    if isinstance(o, _np.bool_):
        return bool(o)
    if isinstance(o, _np.ndarray):
        return o.tolist()
    return str(o)


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def chain_mode() -> str:
    v = str(os.environ.get("FUXI_STAGE2_S5_CHAIN", "") or "").strip().lower()
    return v if v in CHAIN_MODES else DEFAULT_MODE


def canonical_mode(mode: str) -> str:
    """把历史别名归一化：run 是 enforce 的兼容别名（保留传参，不再单独存在）。"""
    return "enforce" if mode == "run" else mode


def chain_limit() -> int:
    try:
        n = int(str(os.environ.get("FUXI_STAGE2_S5_LIMIT", "") or "").strip())
        return n if n > 0 else DEFAULT_LIMIT
    except Exception:
        return DEFAULT_LIMIT


# ---------------------------------------------------------------- 读取（fail-closed）

def load_factor_pool():
    """读因子公式池。返回 (records, top_level_shape)。失败 -> raise（由 main 转 exit 3）。"""
    if not FACTOR_POOL.exists():
        raise FileNotFoundError(f"因子池不存在: {FACTOR_POOL}")
    raw = FACTOR_POOL.read_text(encoding="utf-8")
    d = json.loads(raw)
    if isinstance(d, dict):
        for k in ("proposals", "factors", "items"):
            if isinstance(d.get(k), list):
                return d[k], f"dict[{k}]"
        raise ValueError(f"dict 型但无 proposals/factors/items 列表键; 实际键={list(d.keys())}")
    if isinstance(d, list):
        return d, "list"
    raise ValueError(f"顶层类型不可识别: {type(d).__name__}")


def is_pass(p: dict) -> bool:
    """与 stage2_pass_to_stage3.py::load_pass_factors 保持同一判据，避免两处口径分裂。"""
    st = str(p.get("status", "") or "")
    if st.upper() == "PASS":
        return True
    if str(p.get("test_result", "") or "").upper() == "PASS":
        return True
    try:
        return float(p.get("icir", 0) or 0) >= 0.3
    except Exception:
        return False


def has_formula(p: dict) -> bool:
    f = p.get("formula_pandas") or p.get("formula") or p.get("expression") or ""
    return bool(str(f).strip())


def get_formula(p: dict) -> str:
    return str(p.get("formula_pandas") or p.get("formula") or p.get("expression") or "").strip()


def load_verified_names() -> set:
    """已获 S5 裁决的因子名集合：裁决索引 + 因子池 s5_checked_at 标记 + passed 池终态。"""
    done = set()
    if VERDICT_INDEX.exists():
        try:
            idx = json.loads(VERDICT_INDEX.read_text(encoding="utf-8"))
            for rec in (idx.get("verdicts", {}) or {}).values():
                if rec.get("factor_name"):
                    done.add(str(rec["factor_name"]))
        except Exception:
            pass
    # passed_factor_pool.csv 中已走到 S5 之后的终态，视为已裁
    if POOL_CSV.exists():
        try:
            with open(POOL_CSV, encoding="utf-8-sig") as f:
                for r in csv.DictReader(f):
                    if str(r.get("status", "") or "") in ("s5_passed", "s5_failed", "jq_done", "recheck_s1_passed"):
                        if r.get("name"):
                            done.add(str(r["name"]))
        except Exception:
            pass
    return done


def build_queue(records):
    """返回 (pending_list, diag)。diag 与 readings 分表（记忆铁律：运行读数与旁路诊断不同表）。"""
    verified = load_verified_names()
    pending, diag = [], {
        "pool_total": len(records),
        "status_pass": 0,
        "no_formula": [],
        "already_verified": [],
        "skipped_incomplete": [],
    }
    for p in records:
        if not isinstance(p, dict):
            continue
        if not is_pass(p):
            continue
        diag["status_pass"] += 1
        name = str(p.get("factor_name") or p.get("name") or "").strip()
        if not name:
            diag["skipped_incomplete"].append({"reason": "无 factor_name"})
            continue
        if name in verified:
            diag["already_verified"].append(name)
            continue
        if not has_formula(p):
            diag["no_formula"].append(name)
            continue
        pending.append({
            "factor_name": name,
            "formula": get_formula(p),
            "paradigm": str(p.get("paradigm", "") or ""),
            "direction": str(p.get("direction", "long") or "long"),
            "icir": float(p.get("icir", 0) or 0) if str(p.get("icir", "") or "").strip() else 0.0,
            "ic_mean": float(p.get("ic_mean", 0) or 0) if str(p.get("ic_mean", "") or "").strip() else 0.0,
            "source_proposal": str(p.get("source_proposal", "") or ""),
            "generated_at": str(p.get("generated_at", "") or ""),
        })
    return pending, diag


# ---------------------------------------------------------------- M15 五态分流 + 滞留台账

_FIVE_STATES = (EVAL_VALID, EVAL_DATA_INSUFFICIENT, EVAL_PRECHECK_REJECTED,
                EVAL_ERROR, EVAL_NOT_EVALUATED)


def _parse_iso_dt(s):
    """容错解析 ISO 时间串；不可解析 -> None（**不得当 0 岁处理**，见滞留台账）。"""
    txt = str(s or "").strip()
    if not txt:
        return None
    try:
        dt = datetime.fromisoformat(txt.replace("Z", "+00:00"))
    except Exception:
        return None
    if dt.tzinfo is not None:
        dt = dt.astimezone().replace(tzinfo=None)
    return dt


def _recency_key(rec):
    return (str(rec.get("checked_at") or ""), str(rec.get("run_id") or ""))


def load_attempt_history():
    """扫描历史 run 目录的 readings.json：因子名 -> 最近一次真实裁决。

    返回 (history, attempt_counts)。
      history[name]        = {s5_eval_status, s5_passed, run_id, checked_at, s5_reason}
      attempt_counts[name] = 出现过的 run 目录数（尝试次数，非独立实验数）

    诚实性边界：只有**真实产生过读数**（含 precheck_rejected）的条目才会出现；
    从未执行的条目保持「尚未发生」（not_evaluated），不得被伪装成拒绝，
    也不得从分母消失（记忆铁律：三种「没有读数」分开）。
    """
    hist, counts = {}, {}
    if not REPORT_DIR.exists():
        return hist, counts
    for d in sorted(REPORT_DIR.iterdir()):
        rp = d / "readings.json"
        if not d.is_dir() or not rp.exists():
            continue
        try:
            rows = json.loads(rp.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(rows, list):
            continue
        seen_here = set()
        for r in rows:
            if not isinstance(r, dict):
                continue
            name = str(r.get("factor_name") or "").strip()
            if not name or name in seen_here:
                continue
            seen_here.add(name)
            rec = {
                "s5_eval_status": str(r.get("s5_eval_status") or ""),
                "s5_passed": bool(r.get("s5_passed", False)),
                "run_id": str(r.get("run_id") or d.name),
                "checked_at": str(r.get("checked_at") or ""),
                "s5_reason": str(r.get("s5_reason", ""))[:200],
            }
            prev = hist.get(name)
            if prev is None or _recency_key(rec) >= _recency_key(prev):
                hist[name] = rec
        for name in seen_here:
            counts[name] = counts.get(name, 0) + 1
    return hist, counts


def triage_queue(pending, attempt_history, attempt_counts, preflight_shadow=False):
    """M15：对队列逐条做五态分流（annotate 出真读数与分母）。

    诚实性约束：
      - triage_state 只来自**已发生**的证据（历史裁决 / 尝试记录），绝不来自
        推测；从未执行 = not_evaluated（「尚未发生」必须留在输出里）。
      - preflight_shadow=True 只附质量证据标注，硬缺陷记为
        would_reject_under_enforce（**投影**），不改写 triage_state、不算裁决。
    返回 (annotated_pending, summary)。
    """
    annotated = []
    by_state = Counter()
    n_projected_reject = 0
    n_review_required = 0
    for item in pending:
        name = item["factor_name"]
        rec = attempt_history.get(name)
        state, source, evidence = EVAL_NOT_EVALUATED, "none", {}
        if rec:
            raw = rec.get("s5_eval_status") or ""
            state = raw if raw in _FIVE_STATES else EVAL_NOT_EVALUATED
            source = "attempt_history"
            evidence = {
                "last_run_id": rec.get("run_id"),
                "last_checked_at": rec.get("checked_at"),
                "last_passed": rec.get("s5_passed"),
                "last_reason": rec.get("s5_reason"),
                "raw_status": raw,
            }
        out = dict(item)
        out["triage_state"] = state
        out["triage_source"] = source
        out["triage_evidence"] = evidence
        out["n_attempts"] = int(attempt_counts.get(name, 0))
        if preflight_shadow:
            from candidate_preflight import precheck_candidates  # noqa: E402
            rows, _audit = precheck_candidates(
                [out], source="stage2_queue_audit", mode="shadow")
            q = ((rows[0].get("formula_quality") if rows else None) or {})
            hard = list(q.get("hard_codes") or [])
            review = list(q.get("review_codes") or [])
            out["preflight_shadow"] = {
                "hard_codes": hard,
                "review_codes": review,
                "would_reject_under_enforce": bool(hard),
                "note": "shadow 只标注证据；不是裁决，也不改变五态",
            }
            n_projected_reject += 1 if hard else 0
            n_review_required += 1 if review else 0
        annotated.append(out)
        by_state[state] += 1

    denom = {
        "tracked": len(annotated),
        "read": by_state.get(EVAL_VALID, 0),
        "read_passed": sum(1 for x in annotated if x["triage_state"] == EVAL_VALID
                           and x["triage_evidence"].get("last_passed")),
        "refused": sum(by_state.get(k, 0) for k in
                       (EVAL_DATA_INSUFFICIENT, EVAL_ERROR, EVAL_PRECHECK_REJECTED)),
        "not_yet_attempted": by_state.get(EVAL_NOT_EVALUATED, 0),
        "by_state": dict(by_state),
        "labels": {
            "read": "有可用于判定的真实读数（valid）",
            "refused": "已尝试但未产出可用读数（data_insufficient/error/precheck_rejected）",
            "not_yet_attempted": "尚未发生（not_evaluated）——必须留在分母，不得当作缺失",
        },
    }
    denom["assert_split_ok"] = (
        denom["tracked"] == denom["read"] + denom["refused"] + denom["not_yet_attempted"])
    summary = {
        "denominator": denom,
        "preflight_projection": {
            "mode": "shadow" if preflight_shadow else "not_run",
            "would_reject_under_enforce": n_projected_reject,
            "review_required": n_review_required,
            "note": "投影 ≠ 裁决；enforce 下才真正移出并记 precheck_rejected",
        },
    }
    return annotated, summary


def update_age_ledger(annotated, run_id, now=None):
    """M15：滞留时长台账（>RETENTION_REVIEW_DAYS 触发评审，只报不阻塞）。

    age 优先级：generated_at（真实进入队列时间）与上一版账本 first_seen_at
    取**较早**者；两者都不可解析 -> age_days=None、review_due=None
    （未知 ≠ 未超期），单独计入 n_age_unknown。
    """
    now = now or datetime.now()
    prev_items = {}
    if AGE_LEDGER.exists():
        try:
            prev_items = (json.loads(AGE_LEDGER.read_text(encoding="utf-8")).get("items") or {})
        except Exception:
            prev_items = {}
    items, review_due, sources = {}, [], Counter()
    n_unknown = 0
    for it in annotated:
        name = it["factor_name"]
        gen = _parse_iso_dt(it.get("generated_at"))
        prev_first = _parse_iso_dt((prev_items.get(name) or {}).get("first_seen_at"))
        cands = [(dt, tag) for dt, tag in ((gen, "generated_at"),
                                           (prev_first, "ledger_first_seen")) if dt]
        if cands:
            first_seen, age_source = min(cands, key=lambda x: x[0])
            age_days = (now - first_seen).total_seconds() / 86400.0
            due = age_days > RETENTION_REVIEW_DAYS
        else:
            first_seen, age_source, age_days, due = now, "ledger_fallback_now", None, None
            n_unknown += 1
        sources[age_source] += 1
        items[name] = {
            "first_seen_at": first_seen.isoformat(timespec="seconds"),
            "age_source": age_source,
            "generated_at": str(it.get("generated_at") or ""),
            "last_seen_at": now.isoformat(timespec="seconds"),
            "age_days": None if age_days is None else round(age_days, 2),
            "review_due": due,
            "triage_state": it["triage_state"],
            "n_attempts": int(it.get("n_attempts", 0)),
        }
        if due:
            review_due.append(name)
    known_ages = [x["age_days"] for x in items.values() if x["age_days"] is not None]
    ledger = {
        "schema": "stage2_s5_queue_age/v1",
        "updated_at": now.isoformat(timespec="seconds"),
        "run_id": run_id,
        "threshold_days": RETENTION_REVIEW_DAYS,
        "n_tracked": len(items),
        "n_review_due": len(review_due),
        "n_age_unknown": n_unknown,
        "oldest_age_days": max(known_ages) if known_ages else None,
        "age_sources": dict(sources),
        "note": "review_due=滞留超阈值，仅触发评审，不自动拒绝/降级；annotate 只报不阻塞",
        "review_due": review_due,
        "items": items,
    }
    AGE_LEDGER.write_text(json.dumps(ledger, ensure_ascii=False, indent=2, default=_json_safe),
                          encoding="utf-8", newline="")
    return ledger


# ---------------------------------------------------------------- S5 执行

def run_s5(pending, limit, run_id, dry_run=False):
    """对 pending 前 limit 条执行真实 S5。返回 (readings, engine_info)。"""
    engine = {"s5_mode": None, "is_ready": False, "note": ""}
    subset = pending[:limit]

    # A2: Stage2 and Ralph use the same opt-in formula/PIT precheck. Off is
    # exact historical candidate behavior; shadow only annotates. Enforce
    # keeps every rejected candidate visible as precheck_rejected.
    preflight_mode = os.environ.get("FUXI_STAGE2_PREFLIGHT", "off").strip().lower()
    if preflight_mode not in ("off", "shadow", "enforce"):
        raise ValueError("FUXI_STAGE2_PREFLIGHT must be off, shadow or enforce")
    from candidate_preflight import precheck_candidates
    subset, preflight_audit = precheck_candidates(
        subset, source="stage2_daily", mode=preflight_mode)
    engine["generator_preflight"] = preflight_audit
    precheck_readings = [
        {"factor_name": row["factor_name"], "formula_sha256": row["formula_sha256"],
         "s5_eval_status": EVAL_PRECHECK_REJECTED, "s5_passed": False,
         "s5_reason": ";".join(row["hard_codes"]), "s5_error": "",
         "run_id": run_id, "checked_at": datetime.now().isoformat(timespec="seconds")}
        for row in preflight_audit.get("rejected_candidates", [])
    ]

    if dry_run:
        engine.update({"s5_mode": "dry_run", "is_ready": True,
                       "note": "预演模式，未执行回测"})
        return [], engine

    if not subset and precheck_readings:
        engine.update({"s5_mode": "precheck_only", "is_ready": True,
                       "note": "all requested candidates rejected before S5"})
        return precheck_readings, engine

    try:
        from s5_joint_filter import S5JointFilter  # noqa: E402
    except Exception as e:
        engine.update({"s5_mode": "import_failed", "note": f"{type(e).__name__}: {e}"})
        return None, engine

    try:
        sf = S5JointFilter(top_n=80)
        engine["is_ready"] = bool(sf.is_ready)
    except Exception as e:
        engine.update({"s5_mode": "init_failed", "note": f"{type(e).__name__}: {e}"})
        return None, engine

    if not sf.is_ready:
        engine.update({"s5_mode": "not_ready", "note": "S5JointFilter.is_ready=False"})
        return None, engine

    engine["s5_mode"] = "real_s5"
    results = sf.validate_batch(subset, verbose=True)
    if results is None or len(results) != len(subset):
        engine.update({"s5_mode": "result_count_mismatch", "is_ready": False,
                       "note": f"requested={len(subset)} returned={len(results) if results is not None else 'None'}"})
        return None, engine

    readings = list(precheck_readings)
    for cand, r in zip(subset, results):
        err = str(getattr(r, "error", "") or "")
        reason = str(getattr(r, "reason", "") or "")
        if err in _S5_HARD_ERRORS:
            status = EVAL_DATA_INSUFFICIENT
        elif err:
            status = EVAL_ERROR
        elif getattr(r, "elapsed_sec", 0) or getattr(r, "window_days_2025", 0) or getattr(r, "window_days_2026", 0):
            status = EVAL_VALID
        elif reason.startswith("🧹 公式合理性拦截"):
            status = EVAL_PRECHECK_REJECTED
        else:
            status = EVAL_NOT_EVALUATED

        rec = {
            "factor_name": cand["factor_name"],
            "formula": cand["formula"],
            "raw_formula": cand.get("raw_formula", cand["formula"]),
            "formula_quality": cand.get("formula_quality"),
            "paradigm": cand["paradigm"],
            "s1_icir_from_pool": cand["icir"],
            "s5_eval_status": status,
            "s5_passed": bool(getattr(r, "passed", False)),
            "s5_reason": reason,
            "s5_error": err,
            "run_id": run_id,
            "checked_at": datetime.now().isoformat(timespec="seconds"),
        }
        try:
            rec.update({k: v for k, v in r.to_dict().items() if k != "formula"})
        except Exception:
            pass
        readings.append(rec)
    return readings, engine


# ---------------------------------------------------------------- 落盘

def write_run_dir(run_id, payload):
    d = REPORT_DIR / run_id
    d.mkdir(parents=True, exist_ok=True)
    (d / "readings.json").write_text(
        json.dumps(payload["readings"], ensure_ascii=False, indent=2, default=_json_safe), encoding="utf-8", newline="")
    (d / "diagnostics.json").write_text(
        json.dumps(payload["diagnostics"], ensure_ascii=False, indent=2, default=_json_safe), encoding="utf-8", newline="")
    (d / "manifest.json").write_text(
        json.dumps(payload["manifest"], ensure_ascii=False, indent=2, default=_json_safe), encoding="utf-8", newline="")
    with open(d / "verdicts.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["factor_name", "s5_eval_status", "s5_passed",
                    "excess_2025", "excess_2026", "calmar_2025", "calmar_2026",
                    "window_days_2025", "window_days_2026", "coverage_rate",
                    "s5_reason", "run_id"])
        for r in payload["readings"]:
            w.writerow([
                r.get("factor_name", ""), r.get("s5_eval_status", ""), r.get("s5_passed", ""),
                r.get("excess_2025", ""), r.get("excess_2026", ""),
                r.get("calmar_2025", ""), r.get("calmar_2026", ""),
                r.get("window_days_2025", ""), r.get("window_days_2026", ""),
                r.get("coverage_rate", ""),
                str(r.get("s5_reason", ""))[:200], r.get("run_id", ""),
            ])
    return d


def update_verdict_index(readings, run_id):
    idx = {"updated_at": "", "verdicts": {}}
    if VERDICT_INDEX.exists():
        try:
            idx = json.loads(VERDICT_INDEX.read_text(encoding="utf-8"))
            idx.setdefault("verdicts", {})
        except Exception:
            idx = {"updated_at": "", "verdicts": {}}
    for r in readings:
        name = r.get("factor_name")
        if not name:
            continue
        idx["verdicts"][name] = {
            "factor_name": name,
            "s5_eval_status": r.get("s5_eval_status"),
            "s5_passed": r.get("s5_passed"),
            "excess_2025": r.get("excess_2025"),
            "excess_2026": r.get("excess_2026"),
            "calmar_2025": r.get("calmar_2025"),
            "calmar_2026": r.get("calmar_2026"),
            "s5_reason": str(r.get("s5_reason", ""))[:300],
            "run_id": run_id,
            "checked_at": r.get("checked_at"),
        }
    idx["updated_at"] = datetime.now().isoformat(timespec="seconds")
    idx["n_verdicts"] = len(idx["verdicts"])
    VERDICT_INDEX.write_text(json.dumps(idx, ensure_ascii=False, indent=2, default=_json_safe),
                             encoding="utf-8", newline="")
    return len(idx["verdicts"])


def persist_to_pool(readings, run_id):
    """把裁决回写进因子池（**追加键，不改既有键**）。

    走项目既有的 `scripts/proposals_io.py` 通道：
      O_EXCL 跨进程文件锁 + 原子写(tmp+os.replace) + 写后读回校验 + 锁内闭环
      (load_and_modify: 读最新 → 改 → 写，杜绝 TOCTOU 丢改)。
    **刻意不自行做裸「读→改→写」** —— 那正是 2026-08-21 覆盖事故后被
    `proposals_io` 取代的旧模式（见该模块 docstring）。

    另在**锁内**先落一份备份快照，保证备份与改写同一视图。

    🔴 路径纪律（2026-09-25 修）：写入方 `proposals_io.PATH` 才是落盘位置，
    备份与回读**必须**同源指向它。历史上本函数用模块常量 FACTOR_POOL 做回读，
    而写入走 PATH ⇒ 当二者被测试替身分离时，校验的是另一个文件，
    出现 verified=True 的**假阳性**并静默污染真池。现已统一，且不一致时留痕。
    """
    if not readings:
        return {"written": 0, "note": "无裁决可回写"}

    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        import proposals_io  # noqa: E402
    except Exception as e:
        return {"written": 0, "note": f"proposals_io 不可用，放弃回写: {type(e).__name__}: {e}"}

    # 🔴 写入权威路径 = proposals_io.PATH（真正决定落盘位置的那一个）。
    #    2026-09-25 实证缺陷「写 A 读 B」：本函数经 proposals_io 写到 PATH，
    #    却用模块常量 FACTOR_POOL 备份/回读 ⇒ 校验的是**另一个文件**，
    #    verified=True 是假阳性。后果已验证：测试把 FACTOR_POOL 重定向到临时副本后，
    #    写入落到真实因子池，回读却校验临时副本并通过，真池被 TEST 数据静默污染。
    #    备份与回读一律以写入方路径为准；两者不一致时留痕而非静默。
    write_path = Path(proposals_io.PATH)
    path_mismatch = None
    if write_path != FACTOR_POOL:
        path_mismatch = (f"写入权威路径 {write_path} != 模块常量 FACTOR_POOL {FACTOR_POOL}，"
                         f"已以写入方为准（备份与回读均指向写入方路径）")

    kv = {r["factor_name"]: r for r in readings if r.get("factor_name")}
    info: dict = {"written": 0, "write_path": str(write_path)}
    if path_mismatch:
        info["NOTE_path_mismatch"] = path_mismatch

    def _mod(d):
        is_dict = isinstance(d, dict)
        if is_dict:
            key = next((k for k in ("proposals", "factors", "items")
                        if isinstance(d.get(k), list)), None)
            if key is None:
                info["note"] = "结构不可识别，中止回写"
                info["n_before"] = info["n_after"] = None
                return d
            records = d[key]
        else:
            records = d

        info["n_before"] = len(records)
        # 锁内备份（与本次改写同一视图，且**字节级**等同改写前磁盘内容）
        # 2026-09-25 修正：旧实现用 write_text(..., newline="") 会写成 LF，
        # 而因子池实际是 CRLF（Windows 文本模式产物）⇒ 备份不是字节可信备份。
        # 改为直接对锁内读取的原始字节做快照，并断言保真。
        try:
            bak = write_path.with_suffix(".json.bak_20260925_chain")
            before_bytes = write_path.read_bytes()      # 锁内 ⇒ 即本次改写前的磁盘内容
            bak.write_bytes(before_bytes)
            info["backup"] = bak.name
            info["backup_bytes"] = len(before_bytes)
            info["backup_sha256"] = hashlib.sha256(before_bytes).hexdigest()
            if bak.read_bytes() != before_bytes:        # 备份保真断言
                info["ASSERT_FAIL_BACKUP"] = "备份字节与改写前磁盘内容不一致"
        except Exception as e:
            info["backup"] = f"备份失败(不中止): {type(e).__name__}"

        for p in records:
            if not isinstance(p, dict):
                continue
            r = kv.get(str(p.get("factor_name") or ""))
            if not r:
                continue
            p["s5_eval_status"] = r.get("s5_eval_status")
            p["s5_passed"] = r.get("s5_passed")
            p["s5_reason"] = str(r.get("s5_reason", ""))[:300]
            p["s5_excess_2025"] = r.get("excess_2025")
            p["s5_excess_2026"] = r.get("excess_2026")
            p["s5_calmar_2025"] = r.get("calmar_2025")
            p["s5_calmar_2026"] = r.get("calmar_2026")
            p["s5_checked_at"] = r.get("checked_at")
            p["s5_run_id"] = run_id
            info["written"] += 1
        info["n_after"] = len(records)
        return d

    try:
        proposals_io.load_and_modify(_mod)
    except Exception as e:
        return {"written": 0, "note": f"锁内回写失败(未改动): {type(e).__name__}: {e}"}

    # ── 写后独立回读断言（不信锁内返回值） ──
    # 🔴 回读**必须以写入方路径为准**；读写不同路径时本断言退化为假阳性。
    back = json.loads(write_path.read_text(encoding="utf-8"))
    if isinstance(back, dict):
        key = next((k for k in ("proposals", "factors", "items")
                    if isinstance(back.get(k), list)), None)
        back_records = back.get(key, []) if key else []
    else:
        back_records = back
    info["n_after_readback"] = len(back_records)
    if info.get("n_before") is not None and len(back_records) != info["n_before"]:
        info["ASSERT_FAIL"] = f"回写后条数变化 {info['n_before']} -> {len(back_records)}"
    back_names = {str(p.get("factor_name") or "") for p in back_records if isinstance(p, dict)}
    lost = [n for n in kv if n not in back_names]
    if lost:
        info["ASSERT_FAIL_LOST"] = lost[:10]
    info["verified"] = not (info.get("ASSERT_FAIL") or info.get("ASSERT_FAIL_LOST")
                            or info.get("ASSERT_FAIL_BACKUP"))
    return info


# ---------------------------------------------------------------- main

def _main_under_lock():
    ap = argparse.ArgumentParser(description="Stage2 PASS -> 完整 S1~S5 核验链路")
    ap.add_argument("--limit", type=int, default=None, help=f"本次最多核验多少条（默认 env 或 {DEFAULT_LIMIT}）")
    ap.add_argument("--only", nargs="*", default=None, help="只核验指定因子名")
    ap.add_argument("--queue-only", action="store_true", help="只刷新待办队列，不跑 S5")
    ap.add_argument("--dry-run", action="store_true", help="预演，不执行回测")
    ap.add_argument("--persist-to-pool", action="store_true", help="把裁决回写进因子池（追加键）")
    ap.add_argument("--run-id", default=None, help="显式指定 run_id（默认按时间生成，不复用）")
    args = ap.parse_args()

    mode_raw = chain_mode()
    mode = canonical_mode(mode_raw)
    if mode_raw != mode:
        print(f"[chain] FUXI_STAGE2_S5_CHAIN={mode_raw} 为历史别名 -> 按 {mode} 执行")
    if mode_raw == "off" and not (args.queue_only or args.dry_run or args.only):
        print("[chain] FUXI_STAGE2_S5_CHAIN=off -> 不动作（与接线前行为等价）")
        return 0

    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    print("=" * 88)
    print(f"[Stage2 -> S5 链路] run_id={run_id}  mode={mode}")
    print("=" * 88)

    # ── 读（fail-closed） ──────────────────────────────
    try:
        records, shape = load_factor_pool()
    except Exception as e:
        print(f"[FATAL] 因子池不可读/结构不可识别 -> exit 3: {type(e).__name__}: {e}")
        return 3
    print(f"  因子池: {FACTOR_POOL.name}  shape={shape}  records={len(records)}")

    pending, diag = build_queue(records)
    if args.only:
        want = set(args.only)
        pending = [p for p in pending if p["factor_name"] in want]

    limit = args.limit or chain_limit()
    print(f"  PASS 条目        : {diag['status_pass']}")
    print(f"  已有 S5 裁决     : {len(diag['already_verified'])}")
    print(f"  无公式（跳过）   : {len(diag['no_formula'])}")
    print(f"  ==> 待核验队列   : {len(pending)}")
    if args.only:
        print(f"  --only 过滤后    : {len(pending)}")

    # ── M15: 五态分流（只消费已发生证据）+ 滞留台账 ──────────
    attempt_hist, attempt_counts = load_attempt_history()
    pending, triage = triage_queue(
        pending, attempt_hist, attempt_counts,
        preflight_shadow=(mode in ("audit", "enforce") or bool(args.dry_run)))
    retention = update_age_ledger(pending, run_id)
    denom = triage["denominator"]
    print(f"  五态分流         : {denom['by_state']}")
    print(f"  分母             : tracked={denom['tracked']} read={denom['read']} "
          f"refused={denom['refused']} not_yet_attempted={denom['not_yet_attempted']} "
          f"(split_ok={denom['assert_split_ok']})")
    pf = triage["preflight_projection"]
    if pf["mode"] != "not_run":
        print(f"  预检投影(影子)   : would_reject={pf['would_reject_under_enforce']} "
              f"review_required={pf['review_required']}")
    print(f"  滞留台账         : tracked={retention['n_tracked']} "
          f"review_due={retention['n_review_due']}(>{RETENTION_REVIEW_DAYS}天) "
          f"unknown={retention['n_age_unknown']} oldest={retention['oldest_age_days']}天 "
          f"-> {AGE_LEDGER.name}")
    if retention["n_review_due"]:
        print(f"  ⏰ 滞留评审触发: {retention['n_review_due']} 条滞留超过 {RETENTION_REVIEW_DAYS} 天"
              f"（只报不阻塞；推进路径 annotate→audit→enforce）")

    # 队列快照（便宜，每次都刷；M15 起含五态/分母/滞留字段）
    PENDING_QUEUE.write_text(json.dumps({
        "schema": "stage2_s5_pending/v2",
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "run_id": run_id,
        "mode": mode,
        "mode_requested": mode_raw,
        "n_pending": len(pending),
        "n_pass_total": diag["status_pass"],
        "n_already_verified": len(diag["already_verified"]),
        "denominator": denom,
        "preflight_projection": pf,
        "retention": {k: retention[k] for k in
                      ("threshold_days", "n_tracked", "n_review_due",
                       "n_age_unknown", "oldest_age_days")},
        "review_due": retention["review_due"],
        "pending": [{k: p.get(k) for k in (
            "factor_name", "paradigm", "icir", "generated_at",
            "triage_state", "triage_source", "triage_evidence", "n_attempts",
            "preflight_shadow")} for p in pending],
    }, ensure_ascii=False, indent=2, default=_json_safe), encoding="utf-8", newline="")
    print(f"  队列已落盘: {PENDING_QUEUE}")

    if args.queue_only:
        print("[chain] --queue-only 结束（未执行 S5）")
        return 0
    if not pending:
        print("[chain] 队列为空 -> S5 链路已闭合，无待办")
        return 0

    if mode != "enforce" and not args.dry_run:
        print(f"[chain] mode={mode} 只标注不执行（快照已含五态/分母/滞留；零回测）")
        print("[chain] 要执行真实 S5: FUXI_STAGE2_S5_CHAIN=enforce --limit N"
              "（推进路径 annotate->audit->enforce）")
        return 0

    # ── 跑（真实 S5；未就绪则 fail-closed，绝不轻量降级） ──
    readings, engine = run_s5(pending, limit, run_id, dry_run=args.dry_run)
    if readings is None:
        print(f"[FATAL] S5 引擎不可用 -> exit 2（未产出任何裁决，不做轻量降级）")
        print(f"        s5_mode={engine.get('s5_mode')} note={engine.get('note')}")
        payload = {
            "readings": [],
            "diagnostics": {**diag, "engine": engine,
                            "five_state_denominator": {
                                "attempted": 0, "read": 0, "refused": 0,
                                "reason": "engine_not_ready"}},
            "manifest": {
                "run_id": run_id, "mode": mode, "limit": limit,
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "engine": engine,
                "code_sha256": _sha256_file(Path(__file__)),
                "factor_pool_sha256": _sha256_file(FACTOR_POOL),
                "note": "fail-closed: 未产出裁决",
            },
        }
        d = write_run_dir(run_id, payload)
        print(f"  留痕已落盘: {d}")
        return 2

    # ── 五态分母（不可压布尔） ─────────────────────────
    cnt = Counter(r["s5_eval_status"] for r in readings)
    denom = {
        "attempted": len(readings),
        "read": cnt.get(EVAL_VALID, 0),
        "refused": sum(v for k, v in cnt.items() if k != EVAL_VALID),
        "by_state": dict(cnt),
    }

    print()
    print("=" * 88)
    print(f"{'因子':<52} {'状态':<20} {'S5':<5} 超额25/26")
    print("=" * 88)
    for r in readings:
        print(f"{r['factor_name']:<52} {r['s5_eval_status']:<20} "
              f"{'PASS' if r['s5_passed'] else '-':<5} "
              f"{r.get('excess_2025', 0):.2%}/{r.get('excess_2026', 0):.2%}")
    print("=" * 88)
    print(f"  分母: attempted={denom['attempted']} read={denom['read']} refused={denom['refused']}")
    print(f"  五态分布: {denom['by_state']}")

    payload = {
        "readings": readings,
        "diagnostics": {**diag, "engine": engine, "five_state_denominator": denom},
        "manifest": {
            "run_id": run_id, "mode": mode, "limit": limit,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "engine": engine,
            "code_sha256": _sha256_file(Path(__file__)),
            "factor_pool_sha256": _sha256_file(FACTOR_POOL),
            "n_readings": len(readings),
            "n_pending_after": max(0, len(pending) - len(readings)),
        },
    }
    d = write_run_dir(run_id, payload)
    print(f"  读数已落盘: {d}")

    n_idx = update_verdict_index(readings, run_id)
    print(f"  裁决索引已更新: {VERDICT_INDEX.name} (累计 {n_idx} 条)")

    if args.persist_to_pool:
        info = persist_to_pool(readings, run_id)
        print(f"  回写因子池: {info}")
    else:
        print("  回写因子池: 跳过（需 --persist-to-pool 显式开启）")

    if len(pending) > len(readings):
        print(f"  ⚠️ 队列仍有 {len(pending) - len(readings)} 条待核验，请再次运行本脚本继续")
    return 0


def main():
    """All queue/verdict writes share the Fuxi single-writer lock."""
    from run_contract import ActiveFuxiRunError, FuxiWriterLock
    try:
        with FuxiWriterLock(run_id="stage2_s5_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")):
            return _main_under_lock()
    except ActiveFuxiRunError as exc:
        print(f"[FATAL] 伏羲写入方正在运行，Stage2链停止: {exc}")
        return 3


if __name__ == "__main__":
    sys.exit(main())
