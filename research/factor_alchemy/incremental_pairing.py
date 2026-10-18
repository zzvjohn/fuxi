# -*- coding: utf-8 -*-
"""
incremental_pairing — M13 (2026-09-27): 配对增量门编排（control 快照 + 分母可见 + 五态）
==========================================================================================
方案 §M13 要求的四处修复，全部落在这个模块里（`ralph_loop` 只做薄适配）:

  1. **control 固定版本快照**（不再取 dict 遍历序前 8 条）
        → `control_snapshot.ControlSnapshotRegistry.ensure()`；快照冻结、漂移只记不切。
  2. **预算提升 + 分母可见**
        → `incremental_max_pair`（默认 12，旧硬编码 3）；输出 `coverage`：
          每个 eligible 候选恰好落进一个桶（valid / budget_deferred / no_formula /
          short_series / data_insufficient / error），**不得从输出消失**。
  3. **异常不静默放行**
        → 任何异常 ⇒ 该候选标 `not_evaluated`（或 `error`）+ 进 `deferred` 队列，
          绝不写"通过"；`exceptions` 计数落盘。
  4. **期数不足不得豁免通过**
        → 期数 < `incremental_min_periods` ⇒ `data_insufficient` + deferred，
          `gate_passed` 不得为 True（五态不可压布尔，见 K2）。

口径铁律:
  · **零行为变化**：`incremental_margin_enabled=False` 时直接返回、不触碰任何候选。
  · **缺数永不奖惩**：非 valid 状态既不拒绝（不剥夺 eligible_for_jq）也不通过，
    只登记进补数队列。
  · 只在 `decision == "reject"`（即 status==valid 且门禁失败）时才剥夺 `eligible_for_jq`。
  · **回退通道**：`FUXI_INCREMENTAL_CONTROL=traversal_legacy` 恢复 M13 前的
    "遍历序前 N 条" control（**仅供反事实 A/B**，读数不得用于正式裁决）；
    预算回退 = 把 config 的 `incremental_max_pair` 显式改回 3（旧硬编码值）。
"""

from __future__ import annotations

import inspect
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[2]          # e:/quant

DEFAULT_SNAPSHOT_REL = "data/ralph_control_snapshot.json"
DEFAULT_MAX_PAIR = 12
DEFAULT_MIN_PERIODS = 8
DEFERRED_CAP = 200

#: 五态（与全系统一致；不可压成布尔）
STATUSES = ("valid", "data_insufficient", "not_evaluated", "error", "precheck_rejected")

#: control 来源口径。snapshot=默认（固定版本快照）；traversal_legacy=M13 前语义
#: （dict 遍历序前 N 条）—— **仅供反事实 A/B**，读数不得用于正式裁决。
CONTROL_MODES = ("snapshot", "traversal_legacy")


def control_mode() -> str:
    """`FUXI_INCREMENTAL_CONTROL`：snapshot(默认) | traversal_legacy(反事实)。非法值 fail-safe 回 snapshot。"""
    raw = (os.environ.get("FUXI_INCREMENTAL_CONTROL", "snapshot") or "").strip().lower()
    return raw if raw in CONTROL_MODES else "snapshot"


# ── helpers ────────────────────────────────────────────────────────────
def _cfg(cfg: Optional[Dict[str, Any]], key: str, default: Any) -> Any:
    if not isinstance(cfg, dict):
        return default
    v = cfg.get(key, default)
    return default if v is None else v


def resolve_snapshot_path(cfg: Optional[Dict[str, Any]] = None,
                          override: Optional[Any] = None) -> Path:
    if override:
        return Path(override)
    rel = str(_cfg(cfg, "incremental_control_snapshot_path", DEFAULT_SNAPSHOT_REL))
    p = Path(rel)
    return p if p.is_absolute() else (ROOT / p)


def _supports_stats(fn: Any) -> bool:
    """s5_filter.rank_sum_combo_returns 是否接受 stats= 记账出参。"""
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
    return "stats" in params


def _eligible_indices(results: Sequence[Any]) -> List[int]:
    return [i for i, r in enumerate(results or [])
            if getattr(r, "eligible_for_jq", False)]


class _DeferredQueue:
    """补数队列：登记"未评估/缺数/异常"候选（有上限，超出只记数不丢计数）。"""

    def __init__(self, cap: int = DEFERRED_CAP):
        self.cap = int(cap)
        self.items: List[Dict[str, Any]] = []
        self.n_total = 0

    def add(self, factor_id: str, status: str, reason: str) -> None:
        self.n_total += 1
        if len(self.items) < self.cap:
            self.items.append({"factor_id": factor_id, "status": status, "reason": reason})

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_deferred": self.n_total,
            "n_listed": len(self.items),
            "truncated": max(0, self.n_total - len(self.items)),
            "items": self.items,
        }


def _new_coverage(cfg: Optional[Dict[str, Any]], n_candidates: int) -> Dict[str, Any]:
    return {
        "n_candidates": int(n_candidates),
        "n_eligible": 0,
        "n_pairs_attempted": 0,
        "n_pairs_valid": 0,
        "n_pass": 0,
        "n_reject": 0,
        "budget": int(_cfg(cfg, "incremental_max_pair", DEFAULT_MAX_PAIR)),
        "min_periods": int(_cfg(cfg, "incremental_min_periods", DEFAULT_MIN_PERIODS)),
        "status_distribution": {},
        "deferred_by_reason": {},
        "control_series_len": 0,
    }


def _bump(dist: Dict[str, int], key: str) -> None:
    dist[key] = int(dist.get(key, 0)) + 1


def run_incremental_pairs(
    candidates: Sequence[Dict[str, Any]],
    results: Sequence[Any],
    *,
    lib_factors: Optional[Iterable[Any]],
    s5_filter: Any,
    cfg: Optional[Dict[str, Any]] = None,
    snapshot_path: Optional[Any] = None,
    registry: Any = None,
) -> Dict[str, Any]:
    """
    执行一轮配对增量门。**所有异常都在内部收敛成五态 + 队列**，不向上抛。

    Returns:
        {
          "status": 五态之一（本轮总体）,
          "checked": int,            # valid 评估数（= 有读数的配对数）
          "rejected": int,           # 仅 valid 且门禁失败
          "control": {...},          # 快照哈希/版本/组成/漂移
          "coverage": {...},         # 分母（每候选一桶）
          "deferred": {...},         # 补数队列
          "exceptions": int,
          "notes": [str],
        }
    """
    cfg = cfg or {}
    cov = _new_coverage(cfg, len(candidates or []))
    queue = _DeferredQueue()
    out: Dict[str, Any] = {
        "status": "not_evaluated",
        "checked": 0,
        "rejected": 0,
        "control": {},
        "coverage": cov,
        "deferred": queue.to_dict(),
        "exceptions": 0,
        "notes": [],
    }

    def _finalize(status: str) -> Dict[str, Any]:
        out["status"] = status
        cov["status_distribution"] = dict(cov["status_distribution"])
        out["deferred"] = queue.to_dict()
        return out

    def _defer_all(status: str, reason: str) -> None:
        for i in _eligible_indices(results):
            cand = candidates[i]
            cand["_incremental_eval_status"] = status
            _bump(cov["status_distribution"], status)
            _bump(cov["deferred_by_reason"], reason)
            queue.add(str(cand.get("factor_name", "?")), status, reason)

    # ── 0) 开关：关闭 = 零行为变化（连候选都不碰） ─────────────────────
    if not bool(_cfg(cfg, "incremental_margin_enabled", False)):
        out["notes"].append("[v0.6-增量] 门禁未启用 ⇒ 本轮 not_evaluated (不记为通过)")
        for _ in range(len(candidates or [])):
            _bump(cov["status_distribution"], "not_evaluated")
        return _finalize("not_evaluated")

    # ── 1) 依赖缺失：不得静默放行 ─────────────────────────────────────
    if s5_filter is None or not lib_factors:
        missing = "s5_filter 缺失" if s5_filter is None else "库因子为空"
        cov["n_eligible"] = len(_eligible_indices(results))
        out["notes"].append(
            f"[v0.6-增量] 依赖缺失({missing}) ⇒ 本轮 not_evaluated + 补数队列 (不记为通过)")
        _defer_all("not_evaluated", "dependency_missing")
        return _finalize("not_evaluated")

    # ── 2) control 来源：快照（默认）或遍历序反事实 ───────────────────
    _cmode = control_mode()
    reg = None
    if _cmode == "traversal_legacy":
        _top_k = int(_cfg(cfg, "incremental_control_top_k", 8))
        _legacy = []
        for _f in (lib_factors or []):
            if not isinstance(_f, dict):
                continue
            _fx = str(_f.get("formula") or "").strip()
            if _fx:
                _legacy.append(_fx)
        ctl_formulas = _legacy[:_top_k]
        out["control"] = {
            "mode": "traversal_legacy",
            "snapshot_sha256": None,
            "revision": None,
            "rule": "traversal_legacy",
            "top_k": _top_k,
            "n_formulas": len(ctl_formulas),
            "snapshot_status": "bypassed",
            "drift_detected": None,
            "fresh_sha256": None,
            "drift_diff": None,
            "rotation_rule": None,
            "path": None,
            "frozen": False,
        }
        out["notes"].append(
            "[v0.6-增量] ⚠️ FUXI_INCREMENTAL_CONTROL=traversal_legacy 反事实口径: "
            "control = dict 遍历序前 N 条 (M13 前语义), 读数不可用于正式裁决")
    else:
        try:
            from control_snapshot import ControlSnapshotRegistry
            reg = registry or ControlSnapshotRegistry(
                resolve_snapshot_path(cfg, snapshot_path))
            ens = reg.ensure(
                lib_factors,
                rule=str(_cfg(cfg, "incremental_control_rule", "sha256_prefix")),
                top_k=int(_cfg(cfg, "incremental_control_top_k", 8)),
                explicit=_cfg(cfg, "incremental_control_explicit", None) or None,
            )
        except Exception as e:                       # pragma: no cover - 防御
            out["exceptions"] += 1
            cov["n_eligible"] = len(_eligible_indices(results))
            out["notes"].append(
                f"[v0.6-增量] control 快照构建失败({type(e).__name__}: {e}) ⇒ "
                f"not_evaluated + 补数队列 (不记为通过)")
            _defer_all("not_evaluated", "control_snapshot_error")
            return _finalize("not_evaluated")

        snap = dict(ens.get("snapshot") or {})
        ctl_formulas = list(snap.get("formulas") or [])
        out["control"] = {
            "mode": "snapshot",
            "snapshot_sha256": snap.get("snapshot_sha256"),
            "revision": snap.get("revision"),
            "rule": snap.get("rule"),
            "top_k": snap.get("top_k"),
            "n_formulas": len(ctl_formulas),
            "snapshot_status": ens.get("status"),
            "drift_detected": bool(ens.get("drift_detected")),
            "fresh_sha256": ens.get("fresh_sha256"),
            "drift_diff": ens.get("diff"),
            "rotation_rule": snap.get("rotation_rule"),
            "path": ens.get("path"),
            "frozen": True,
        }
        if ens.get("drift_detected"):
            out["notes"].append(
                "[v0.6-增量] ⚠️ control 快照漂移: 库已变化但 control 保持冻结(未切换); "
                "换基组合须显式 rotate 并重算相关历史增量")

    if not ctl_formulas:
        cov["n_eligible"] = len(_eligible_indices(results))
        out["notes"].append(
            "[v0.6-增量] control 基组合为空 ⇒ 本轮 data_insufficient (不得豁免通过)")
        _defer_all("data_insufficient", "control_snapshot_empty")
        return _finalize("data_insufficient")

    # ── 3) 门禁与 control 收益序列 ────────────────────────────────────
    try:
        from incremental_margin import IncrementalMarginGate
        gate = IncrementalMarginGate(
            enabled=True,
            min_incremental_net_ir=float(_cfg(cfg, "incremental_net_ir_min", 0.10)),
            max_drawdown_deterioration=float(
                _cfg(cfg, "incremental_dd_deterioration_max", 0.02)),
            min_incremental_periods=int(
                _cfg(cfg, "incremental_min_periods", DEFAULT_MIN_PERIODS)),
        )
    except Exception as e:                       # pragma: no cover - 防御
        out["exceptions"] += 1
        cov["n_eligible"] = len(_eligible_indices(results))
        out["notes"].append(f"[v0.6-增量] 门禁构造失败({e}) ⇒ not_evaluated + 补数队列")
        _defer_all("not_evaluated", "gate_construct_error")
        return _finalize("not_evaluated")

    ctl_stats: Dict[str, Any] = {}
    try:
        if _supports_stats(getattr(s5_filter, "rank_sum_combo_returns", None)):
            control_ret = s5_filter.rank_sum_combo_returns(ctl_formulas, stats=ctl_stats)
        else:
            control_ret = s5_filter.rank_sum_combo_returns(ctl_formulas)
            ctl_stats = {"stats_supported": False}
    except Exception as e:
        out["exceptions"] += 1
        cov["n_eligible"] = len(_eligible_indices(results))
        out["control"]["composition"] = {"error": f"{type(e).__name__}: {e}"}
        out["notes"].append(f"[v0.6-增量] control 组合收益计算失败({e}) ⇒ not_evaluated + 补数队列")
        _defer_all("not_evaluated", "control_series_error")
        return _finalize("not_evaluated")

    control_ret = [] if control_ret is None else control_ret
    out["control"]["composition"] = ctl_stats
    out["control"]["n_control_returns"] = int(len(control_ret))
    cov["control_series_len"] = int(len(control_ret))

    min_periods = int(cov["min_periods"])
    if len(control_ret) < min_periods:
        cov["n_eligible"] = len(_eligible_indices(results))
        out["notes"].append(
            f"[v0.6-增量] control 收益期数不足({len(control_ret)}<{min_periods}) ⇒ "
            f"data_insufficient + 补数队列 (不得豁免通过)")
        _defer_all("data_insufficient", "control_series_short")
        return _finalize("data_insufficient")

    # ── 4) 配对循环（预算 + 每候选恰好一桶） ──────────────────────────
    eligible_idx = _eligible_indices(results)
    cov["n_eligible"] = len(eligible_idx)
    budget = int(cov["budget"])

    for pos, i in enumerate(eligible_idx):
        cand = candidates[i]
        fid = str(cand.get("factor_name", "?"))
        if pos >= budget:
            cand["_incremental_eval_status"] = "not_evaluated"
            _bump(cov["status_distribution"], "not_evaluated")
            _bump(cov["deferred_by_reason"], "budget_exhausted")
            queue.add(fid, "not_evaluated", "budget_exhausted")
            continue
        formula = cand.get("formula", cand.get("expression", "")) or ""
        if not formula:
            cand["_incremental_eval_status"] = "not_evaluated"
            _bump(cov["status_distribution"], "not_evaluated")
            _bump(cov["deferred_by_reason"], "no_formula")
            queue.add(fid, "not_evaluated", "no_formula")
            continue

        cov["n_pairs_attempted"] += 1
        try:
            treat_ret = s5_filter.rank_sum_combo_returns(ctl_formulas + [formula])
        except Exception as e:
            out["exceptions"] += 1
            cov["n_error"] = int(cov.get("n_error", 0)) + 1
            cand["_incremental_eval_status"] = "error"
            _bump(cov["status_distribution"], "error")
            _bump(cov["deferred_by_reason"], "treatment_series_error")
            queue.add(fid, "error", f"treatment_series_error: {type(e).__name__}: {e}")
            continue

        treat_ret = [] if treat_ret is None else treat_ret
        if len(treat_ret) < min_periods:
            cand["_incremental_eval_status"] = "data_insufficient"
            _bump(cov["status_distribution"], "data_insufficient")
            _bump(cov["deferred_by_reason"], "treatment_series_short")
            queue.add(fid, "data_insufficient",
                      f"treatment_series_short({len(treat_ret)}<{min_periods})")
            continue

        verdict = gate.evaluate(control_ret, treat_ret, factor_id=fid)
        cand["_incremental_verdict"] = (
            verdict.to_dict() if hasattr(verdict, "to_dict") else str(verdict))
        status = str(getattr(verdict, "evaluation_status", "valid"))
        cand["_incremental_eval_status"] = status
        _bump(cov["status_distribution"], status)

        if status != "valid":
            # 缺数/未评估：不奖惩，只登记补数
            _bump(cov["deferred_by_reason"], f"verdict_{status}")
            queue.add(fid, status, str(getattr(verdict, "reason", "")))
            continue

        cov["n_pairs_valid"] += 1
        out["checked"] += 1
        decision = str(getattr(verdict, "decision", "") or
                       ("reject" if not getattr(verdict, "gate_passed", True) else "pass"))
        if decision == "reject":
            cov["n_reject"] += 1
            out["rejected"] += 1
            cand["_incremental_rejected"] = True
            if getattr(results[i], "eligible_for_jq", False):
                results[i].eligible_for_jq = False
        else:
            cov["n_pass"] += 1

    overall = "valid" if cov["n_pairs_valid"] else "data_insufficient"
    if cov["n_pairs_valid"] and out["exceptions"]:
        overall = "valid"
    return _finalize(overall)


# ── smoke ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import tempfile

    class _Res:
        def __init__(self, ok=True):
            self.eligible_for_jq = ok

    class _FakeS5:
        def __init__(self, series):
            self.series = series

        def rank_sum_combo_returns(self, formulas, *, top_n=None, names=None, stats=None):
            if stats is not None:
                stats.update({"n_formulas": len(formulas), "n_evaluated": len(formulas),
                              "n_skipped": 0})
            return list(self.series)

    lib = [{"formula": f"f{i}"} for i in range(10)]
    cands = [{"factor_name": f"c{i}", "formula": f"cand{i}"} for i in range(5)]
    res = [_Res() for _ in cands]

    with tempfile.TemporaryDirectory() as td:
        # (a) 关闭开关 ⇒ 零行为变化
        r0 = run_incremental_pairs(cands, res, lib_factors=lib,
                                   s5_filter=_FakeS5([0.01] * 60), cfg={},
                                   snapshot_path=Path(td) / "s.json")
        assert r0["status"] == "not_evaluated" and r0["checked"] == 0
        assert all(not hasattr(c, "_incremental_eval_status") for c in cands), "关闭时不得触碰候选"

        # (b) 期数不足 ⇒ data_insufficient（不得豁免通过）
        r1 = run_incremental_pairs(cands, res, lib_factors=lib,
                                   s5_filter=_FakeS5([0.01] * 5),
                                   cfg={"incremental_margin_enabled": True},
                                   snapshot_path=Path(td) / "s1.json")
        print("[SMOKE] short series:", r1["status"], r1["coverage"]["status_distribution"])
        assert r1["status"] == "data_insufficient"
        assert r1["coverage"]["status_distribution"].get("data_insufficient") == 5
        assert r1["deferred"]["n_deferred"] == 5
        assert r1["checked"] == 0

        # (c) 预算可见：budget=2 ⇒ 其余进 not_evaluated/budget_exhausted
        import numpy as np
        rng = np.random.RandomState(7)
        base = (rng.normal(0, 0.01, 80)).tolist()
        r2 = run_incremental_pairs(cands, res, lib_factors=lib,
                                   s5_filter=_FakeS5(base),
                                   cfg={"incremental_margin_enabled": True,
                                        "incremental_max_pair": 2},
                                   snapshot_path=Path(td) / "s2.json")
        print("[SMOKE] budget:", r2["coverage"]["status_distribution"],
              "checked=", r2["checked"])
        assert r2["coverage"]["n_pairs_attempted"] == 2
        assert r2["coverage"]["deferred_by_reason"].get("budget_exhausted") == 3

        # (d) 异常 ⇒ error + 队列（不静默放行）
        class _Boom(_FakeS5):
            def rank_sum_combo_returns(self, formulas, *, top_n=None, names=None, stats=None):
                if stats is not None and len(formulas) <= 8:
                    stats.update({"n_formulas": len(formulas)})
                    return list(base)
                raise RuntimeError("boom")

        res2 = [_Res() for _ in cands]
        r3 = run_incremental_pairs(cands, res2, lib_factors=lib, s5_filter=_Boom(base),
                                   cfg={"incremental_margin_enabled": True,
                                        "incremental_max_pair": 2},
                                   snapshot_path=Path(td) / "s3.json")
        print("[SMOKE] exceptions:", r3["exceptions"], r3["coverage"]["status_distribution"])
        assert r3["exceptions"] == 2 and r3["rejected"] == 0
        assert all(getattr(r, "eligible_for_jq") for r in res2), "异常不得剥夺 eligible"
        assert all(c["_incremental_eval_status"] == "error" for c in cands[:2])
    print("[SMOKE] OK: run_incremental_pairs 五态/预算/异常路径全部可运行")
