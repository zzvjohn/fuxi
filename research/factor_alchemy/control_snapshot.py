# -*- coding: utf-8 -*-
"""
control_snapshot — M13 (2026-09-27): 增量门 control 基组合的**固定版本快照**
================================================================================
背景（方案 §M13 / 依据 E1）:
    旧实现 `ralph_loop.py:757` 用 `[f.get("formula") for f in lib_factors if
    f.get("formula")][:8]` —— 取的是 **dict 遍历序前 8 条**。插入序随库增长/
    重载变化 ⇒ control 集合静默漂移，treatment 增量在不同轮次之间不可比
    （分母换了却无任何痕迹），违反"每个比较都断言实际比较数量、口径须可比"。

本模块做的事:
  1. **预注册选择规则**（写进快照文件，不在代码里偷偷改）:
       · rule="sha256_prefix"（默认）: 库内全部非空公式去重后，
         按 sha256(公式文本) 升序取前 top_k ⇒ **与遍历序无关**、可复现。
       · rule="explicit": 用显式给定的基组合列表（保序、去空、去重）。
  2. **快照冻结**：`ensure()` 默认 frozen 语义 —— 已存在快照且新算哈希不同
     ⇒ **保持旧 control 不变**（不静默切换），只记录 drift（漂移）事件。
  3. **轮换通道**：`rotate(reason=...)` 显式轮换（revision+1 + 写 update_log）。
     轮换规则预注册为："仅显式轮换；轮换后须重算全部相关历史增量读数"。
  4. **哈希入账**：`snapshot_sha256` 随 run 结果 JSON / manifest 落盘，
     供"控制组版本漂移次数（应为 0）"指标对账。

不做什么:
  · 不改任何门禁阈值；不因缺数而放行或拒绝（那是调用方的五态职责）。
  · 不写库文件以外的资产；不触碰 `.workbuddy/`。
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

SCHEMA = "fuxi_control_snapshot/v1"

DEFAULT_RULE = "sha256_prefix"
RULES = ("sha256_prefix", "explicit")
DEFAULT_TOP_K = 8

#: 预注册的更新规则（写进快照，改它本身就是一次口径变更）
ROTATION_RULE = (
    "frozen_until_explicit_rotate: 快照一经建立即冻结; 仅当调用方显式 rotate() "
    "并给出 reason 时才轮换 (revision+1); 轮换后须重算全部相关历史增量读数并标注口径版本。"
)


# ── helpers ────────────────────────────────────────────────────────────
def _norm(formula: Any) -> str:
    return str(formula or "").strip()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _canonical_payload(rule: str, top_k: int, formulas: Sequence[str]) -> Dict[str, Any]:
    return {
        "schema": SCHEMA,
        "rule": rule,
        "top_k": int(top_k),
        "formulas": list(formulas),
    }


def snapshot_sha256(rule: str, top_k: int, formulas: Sequence[str]) -> str:
    """快照内容摘要（规则 + 容量 + 公式序列 全量参与）。"""
    canon = json.dumps(_canonical_payload(rule, top_k, formulas),
                       ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return _sha256_text(canon)


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def select_formulas(
    lib_factors: Optional[Iterable[Any]],
    *,
    rule: str = DEFAULT_RULE,
    top_k: int = DEFAULT_TOP_K,
    explicit: Optional[Sequence[str]] = None,
) -> Tuple[List[str], Dict[str, Any]]:
    """
    预注册选择规则 ⇒ (formulas, meta)。

    meta 含 n_library_formulas / n_unique / n_selected / rule / top_k。
    `sha256_prefix` 与输入遍历序无关（先按文本去重，再按文本哈希排序）。
    """
    if rule not in RULES:
        raise ValueError(f"未知 control 选择规则: {rule!r} (允许: {RULES})")
    if int(top_k) <= 0:
        raise ValueError(f"top_k 必须为正整数, 得到 {top_k!r}")

    raw: List[str] = []
    for item in (lib_factors or []):
        if isinstance(item, dict):
            raw.append(_norm(item.get("formula")))
        else:
            raw.append(_norm(item))
    non_empty = [f for f in raw if f]
    unique = sorted(set(non_empty))

    if rule == "explicit":
        seq = [_norm(f) for f in (explicit or [])]
        seq = [f for f in seq if f]
        seen, picked = set(), []
        for f in seq:
            if f in seen:
                continue
            seen.add(f)
            picked.append(f)
        picked = picked[: int(top_k)]
    else:  # sha256_prefix
        picked = sorted(unique, key=lambda f: (_sha256_text(f), f))[: int(top_k)]

    meta = {
        "rule": rule,
        "top_k": int(top_k),
        "n_library_formulas": len(raw),
        "n_non_empty": len(non_empty),
        "n_unique": len(unique),
        "n_selected": len(picked),
    }
    return picked, meta


def build_snapshot(
    lib_factors: Optional[Iterable[Any]],
    *,
    rule: str = DEFAULT_RULE,
    top_k: int = DEFAULT_TOP_K,
    explicit: Optional[Sequence[str]] = None,
    revision: int = 1,
    created_at: Optional[str] = None,
    update_log: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    formulas, meta = select_formulas(lib_factors, rule=rule, top_k=top_k, explicit=explicit)
    return {
        "schema": SCHEMA,
        "rule": rule,
        "top_k": int(top_k),
        "formulas": formulas,
        "formula_sha256": [_sha256_text(f) for f in formulas],
        "snapshot_sha256": snapshot_sha256(rule, top_k, formulas),
        "revision": int(revision),
        "created_at": created_at or _now(),
        "updated_at": _now(),
        "rotation_rule": ROTATION_RULE,
        "drift_events": [],
        "update_log": list(update_log or []),
        "selection_meta": meta,
    }


def diff_formulas(old: Sequence[str], new: Sequence[str]) -> Dict[str, Any]:
    o, n = list(old or []), list(new or [])
    so, sn = set(o), set(n)
    return {
        "n_old": len(o),
        "n_new": len(n),
        "n_added": len(sn - so),
        "n_removed": len(so - sn),
        "added": sorted(sn - so)[:8],
        "removed": sorted(so - sn)[:8],
        "order_changed": [a for a, b in zip(o, n) if a != b][:1] != [] and so == sn,
    }


# ── registry ───────────────────────────────────────────────────────────
class ControlSnapshotRegistry:
    """快照读写 + 冻结语义 + 漂移计数。"""

    def __init__(self, path: Path | str):
        self.path = Path(path)

    # ---- io ----
    def load(self) -> Optional[Dict[str, Any]]:
        if not self.path.exists():
            return None
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def _write(self, snap: Dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(snap, ensure_ascii=False, indent=2, sort_keys=True)
        self.path.write_text(text + "\n", encoding="utf-8", newline="\n")

    # ---- ensure ----
    def ensure(
        self,
        lib_factors: Optional[Iterable[Any]],
        *,
        rule: str = DEFAULT_RULE,
        top_k: int = DEFAULT_TOP_K,
        explicit: Optional[Sequence[str]] = None,
        max_drift_events: int = 20,
    ) -> Dict[str, Any]:
        """
        frozen 语义: 已有快照 ⇒ 用它（新算的只用于对账，不切换 control）。

        Returns:
            {
              "status": "created" | "frozen" | "drift" | "empty_library",
              "drift_detected": bool, "created": bool,
              "snapshot": {...}, "fresh_sha256": str,
              "fresh_formulas": [...], "diff": {...} | None,
              "path": str,
            }
        """
        fresh = build_snapshot(lib_factors, rule=rule, top_k=top_k, explicit=explicit)
        stored = self.load()

        if stored is None:
            if not fresh["formulas"]:
                # 空库也要落盘（status 记空，不伪装成"已建立 control"）
                fresh["selection_meta"]["empty_library"] = True
            self._write(fresh)
            return {
                "status": "created" if fresh["formulas"] else "empty_library",
                "drift_detected": False,
                "created": True,
                "snapshot": fresh,
                "fresh_sha256": fresh["snapshot_sha256"],
                "fresh_formulas": list(fresh["formulas"]),
                "diff": None,
                "path": str(self.path),
            }

        stored_formulas = list(stored.get("formulas") or [])
        drift = stored.get("snapshot_sha256") != fresh["snapshot_sha256"]
        if drift:
            events = list(stored.get("drift_events") or [])
            events.append({
                "at": _now(),
                "stored_sha256": stored.get("snapshot_sha256"),
                "fresh_sha256": fresh["snapshot_sha256"],
                "diff": diff_formulas(stored_formulas, fresh["formulas"]),
            })
            stored["drift_events"] = events[-int(max_drift_events):]
            stored["n_drift_events"] = len(events)
            self._write(stored)
        return {
            "status": "drift" if drift else "frozen",
            "drift_detected": bool(drift),
            "created": False,
            "snapshot": stored,
            "fresh_sha256": fresh["snapshot_sha256"],
            "fresh_formulas": list(fresh["formulas"]),
            "diff": diff_formulas(stored_formulas, fresh["formulas"]) if drift else None,
            "path": str(self.path),
        }

    # ---- rotate ----
    def rotate(
        self,
        lib_factors: Optional[Iterable[Any]],
        *,
        reason: str,
        rule: str = DEFAULT_RULE,
        top_k: int = DEFAULT_TOP_K,
        explicit: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        """显式轮换 control（唯一合法的更新通道；必须先给 reason）。"""
        if not str(reason or "").strip():
            raise ValueError("rotate 必须给出 reason（预注册更新规则要求）")
        stored = self.load() or {}
        prev_rev = int(stored.get("revision") or 0)
        prev_formulas = list(stored.get("formulas") or [])
        log = list(stored.get("update_log") or [])
        log.append({
            "at": _now(),
            "from_revision": prev_rev,
            "from_sha256": stored.get("snapshot_sha256"),
            "reason": str(reason),
            "diff": diff_formulas(prev_formulas, select_formulas(
                lib_factors, rule=rule, top_k=top_k, explicit=explicit)[0]),
        })
        snap = build_snapshot(
            lib_factors, rule=rule, top_k=top_k, explicit=explicit,
            revision=prev_rev + 1, created_at=stored.get("created_at"),
            update_log=log,
        )
        snap["drift_events"] = list(stored.get("drift_events") or [])
        snap["n_drift_events"] = int(stored.get("n_drift_events") or 0)
        snap["rotated_at"] = _now()
        snap["rotation_reason"] = str(reason)
        self._write(snap)
        return {"status": "rotated", "snapshot": snap, "path": str(self.path)}


# ── smoke ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    lib = [{"formula": f"x{i}"} for i in range(12)]
    shuffled = list(reversed(lib))
    a, _ = select_formulas(lib)
    b, _ = select_formulas(shuffled)
    assert a == b, "选择规则必须与遍历序无关"
    s1 = build_snapshot(lib)["snapshot_sha256"]
    s2 = build_snapshot(shuffled)["snapshot_sha256"]
    assert s1 == s2, "快照哈希必须与遍历序无关"
    print("[SMOKE] order-independent OK:", s1[:16], "n=", len(a))

    import tempfile
    with tempfile.TemporaryDirectory() as td:
        reg = ControlSnapshotRegistry(Path(td) / "snap.json")
        r1 = reg.ensure(lib)
        r2 = reg.ensure(lib)
        r3 = reg.ensure(lib + [{"formula": "new_one"}])
        print("[SMOKE] ensure:", r1["status"], r2["status"], r3["status"],
              "drift=", r3["drift_detected"])
        assert r1["status"] == "created" and r2["status"] == "frozen"
        assert r3["status"] == "drift" and r3["drift_detected"]
        assert r3["snapshot"]["formulas"] == r1["snapshot"]["formulas"], "漂移时不得切换 control"
        r4 = reg.rotate(lib + [{"formula": "new_one"}], reason="smoke rotate")
        assert r4["snapshot"]["revision"] == 2
        print("[SMOKE] rotate -> rev", r4["snapshot"]["revision"],
              "n=", len(r4["snapshot"]["formulas"]))
    print("[SMOKE] OK: ControlSnapshotRegistry 全部路径可运行")
