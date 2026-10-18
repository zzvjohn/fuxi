# -*- coding: utf-8 -*-
"""
P-20260910-007 llm 引擎范式提案模式 (Paradigm Proposal Mode)

为已冻结的 21 范式空间提供制度化增量通道 (替代已归档的 paradigm_augmenter):
  - 每 N=3 轮 Ralph 允许 1 条偏离既有 21 范式的候选
  - 只过 S1 快速证伪 (icir≥0.1 且方向正确) 即作为 'provisional' 候选范式入空间
  - 需 S5+JQ 终裁通过才转正 (provisional 期间只占 S1 计算预算, 不影响 S5 门槛)
  - 反固着配额 (Picbreeder VLM 教训): 提案不得与近 5 轮 S1 高分候选结构同构

开关: FUXI_PARADIGM_PROPOSAL_MODE = shadow(默认) | enforce | off
  shadow  : 提案生成 + 准入判定照跑, 结果只写 shadow 日志, **不入范式空间** (零副作用)
  enforce : provisional 范式正式写入 data/paradigm_space.json
  off     : 完全关闭, 退化为 21 范式冻结 (一行回退)

状态: data/paradigm_space.json
"""
from __future__ import annotations

import ast
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ── 路径 ──────────────────────────────────────────────────────────
try:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
except Exception:
    PROJECT_ROOT = Path.cwd()
DATA_DIR = PROJECT_ROOT / "data"
SPACE_PATH = DATA_DIR / "paradigm_space.json"
SHADOW_LOG = DATA_DIR / "paradigm_proposal_shadow.json"

# ── 配置 ──────────────────────────────────────────────────────────
PROPOSAL_INTERVAL = int(os.environ.get("FUXI_PARADIGM_PROPOSAL_INTERVAL", "3"))
S1_ICIR_MIN = 0.1          # S1 快速证伪门槛
ANTI_FIXATION_MIN_DIST = float(os.environ.get("FUXI_PARADIGM_ANTI_FIX_DIST", "0.25"))
RECENT_HIGH_WINDOW = 5     # 近 N 轮 S1 高分候选
MAX_ACTIVE_PROVISIONAL = 3  # 同时存在的 provisional 范式上限 (防范式空间爆炸)


def get_mode() -> str:
    m = (os.environ.get("FUXI_PARADIGM_PROPOSAL_MODE", "shadow") or "shadow").lower()
    return m if m in ("shadow", "enforce", "off") else "shadow"


def should_propose(round_index: int, interval: int = None) -> bool:
    """每 interval 轮允许 1 条范式提案。"""
    if get_mode() == "off":
        return False
    iv = int(interval or PROPOSAL_INTERVAL)
    if iv <= 0:
        return False
    return (int(round_index) % iv) == 0


# ══════════════════════════════════════════════════════════════════
# 1. 范式空间状态
# ══════════════════════════════════════════════════════════════════

def _base_paradigms() -> List[str]:
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from paradigm_v4 import PARADIGMS_V4  # type: ignore
        return list(PARADIGMS_V4.keys())
    except Exception:
        return []


def load_space() -> Dict:
    base = _base_paradigms()
    if SPACE_PATH.exists():
        try:
            d = json.loads(SPACE_PATH.read_text(encoding="utf-8"))
            d.setdefault("base_paradigms", base)
            d.setdefault("provisional", [])
            d.setdefault("promoted", [])
            d.setdefault("retired", [])
            return d
        except Exception:
            pass
    return {
        "version": "1.0.0",
        "base_paradigms": base,
        "provisional": [],
        "promoted": [],
        "retired": [],
        "last_updated": datetime.now().isoformat(timespec="seconds"),
    }


def save_space(space: Dict):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    space["last_updated"] = datetime.now().isoformat(timespec="seconds")
    SPACE_PATH.write_text(json.dumps(space, ensure_ascii=False, indent=1), encoding="utf-8")


def active_provisional(space: Dict = None) -> List[Dict]:
    space = space or load_space()
    return [p for p in space.get("provisional", []) if p.get("status") == "provisional"]


def is_known_paradigm(name: str) -> bool:
    if not name:
        return False
    space = load_space()
    known = set(space.get("base_paradigms", [])) | set(space.get("promoted", []))
    known |= {p.get("name") for p in space.get("provisional", []) if p.get("name")}
    return name in known


# ══════════════════════════════════════════════════════════════════
# 2. 反固着: AST 结构距离
# ══════════════════════════════════════════════════════════════════

def _ast_tokens(expr: str, skeleton: bool = False) -> set:
    """把表达式转成结构 token 集合 (算子/属性/字段名/常数桶)。

    skeleton=True 时剔除数值/字符串常数 token — 用于"同骨架不同参数"的识别
    (反固着应拦截 'close.pct_change(20).rolling(5).mean()' 这类仅改窗口的同构变体)。
    """
    toks = set()
    try:
        tree = ast.parse(str(expr).strip(), mode="eval")
    except Exception:
        # 非 Python 语法 (Forge DSL) → 退化为字面 token
        for t in re.findall(r"[A-Za-z_][A-Za-z_0-9]*|\d+", str(expr)):
            if skeleton and t.isdigit():
                continue
            toks.add(t.lower())
        return toks
    for n in ast.walk(tree):
        if isinstance(n, ast.Attribute):
            toks.add("attr:" + n.attr)
        elif isinstance(n, ast.Name):
            toks.add("name:" + n.id)
        elif isinstance(n, ast.Call):
            f = n.func
            toks.add("call:" + (f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", "?")))
        elif isinstance(n, ast.Constant):
            if skeleton:
                continue
            v = n.value
            if isinstance(v, (int, float)):
                toks.add("num:" + str(int(abs(v)) if abs(v) < 1e6 else "big"))
            elif isinstance(v, str):
                toks.add("str:" + v[:12].lower())
        elif isinstance(n, ast.BinOp):
            toks.add("op:" + type(n.op).__name__)
        elif isinstance(n, ast.Compare):
            toks.add("cmp:" + "".join(type(o).__name__ for o in n.ops))
    return toks


def ast_distance(expr_a: str, expr_b: str, skeleton: bool = False) -> float:
    """1 - Jaccard(AST token 集) ∈ [0,1]; 0=完全同构, 1=完全无关。

    skeleton=True → 结构骨架距离 (忽略参数取值), 用于反固着判定。
    """
    a, b = _ast_tokens(expr_a, skeleton), _ast_tokens(expr_b, skeleton)
    if not a or not b:
        return 1.0
    inter = len(a & b)
    union = len(a | b)
    return 1.0 - (inter / union if union else 0.0)


def check_anti_fixation(formula: str, recent_high: List[Dict],
                        min_dist: float = None) -> Tuple[bool, float, str]:
    """提案不得与近 N 轮 S1 高分候选同构 (按**结构骨架**距离, 忽略参数取值)。

    返回 (passed, min_observed_distance, reason)
    """
    thr = ANTI_FIXATION_MIN_DIST if min_dist is None else float(min_dist)
    if not recent_high:
        return True, 1.0, "no recent high candidates"
    dmin = 1.0
    worst = ""
    for c in recent_high:
        f = c.get("formula") or c.get("expression") or ""
        if not f:
            continue
        d = ast_distance(formula, f, skeleton=True)
        if d < dmin:
            dmin = d
            worst = c.get("factor_name") or c.get("name") or ""
    if dmin < thr:
        return False, dmin, f"与近 {len(recent_high)} 轮高分候选 '{worst}' 结构同构 (骨架距离={dmin:.3f} < {thr})"
    return True, dmin, f"最小结构骨架距离 {dmin:.3f} ≥ {thr}"


# ══════════════════════════════════════════════════════════════════
# 3. 准入判定 (S1 快速证伪 → provisional)
# ══════════════════════════════════════════════════════════════════

def admit_provisional(
    candidate: Dict,
    icir: float,
    direction_ok: bool,
    recent_high: Optional[List[Dict]] = None,
    mode: Optional[str] = None,
) -> Dict:
    """判定一条范式提案是否准入 provisional。

    准入三条件: ①新范式名有效且不与既有范式重复 ②S1 快速证伪通过 (icir≥0.1 且方向正确)
               ③反固着配额通过。全过 → provisional (enforce) / shadow_record (shadow)。
    """
    mode = mode or get_mode()
    name = (candidate.get("proposed_paradigm") or "").strip()
    formula = candidate.get("formula") or candidate.get("expression") or ""
    verdict = {
        "name": name,
        "factor_name": candidate.get("factor_name") or candidate.get("name") or "",
        "formula": formula,
        "icir": float(icir or 0.0),
        "direction_ok": bool(direction_ok),
        "mode": mode,
        "checks": {},
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }

    if not name:
        verdict.update({"admitted": False, "reason": "缺少新范式名 (proposed_paradigm)"})
        return verdict
    if is_known_paradigm(name):
        verdict.update({"admitted": False, "reason": f"范式 '{name}' 已存在于范式空间"})
        return verdict
    verdict["checks"]["novel_name"] = True

    s1_ok = (verdict["icir"] >= S1_ICIR_MIN) and verdict["direction_ok"]
    verdict["checks"]["s1_falsification"] = s1_ok
    if not s1_ok:
        verdict.update({
            "admitted": False,
            "reason": f"S1 快速证伪未过 (icir={verdict['icir']:.3f} < {S1_ICIR_MIN} 或方向错)",
        })
        return verdict

    af_ok, dmin, af_reason = check_anti_fixation(formula, recent_high or [])
    verdict["checks"]["anti_fixation"] = af_ok
    verdict["anti_fixation_distance"] = round(dmin, 4)
    if not af_ok:
        verdict.update({"admitted": False, "reason": af_reason})
        return verdict

    space = load_space()
    if len(active_provisional(space)) >= MAX_ACTIVE_PROVISIONAL:
        verdict.update({
            "admitted": False,
            "reason": f"活跃 provisional 已达上限 {MAX_ACTIVE_PROVISIONAL}",
        })
        return verdict

    if mode == "shadow":
        verdict.update({"admitted": False, "shadow_record": True,
                        "reason": "shadow 模式: 满足全部准入条件, 仅记录不入空间"})
        return verdict

    # enforce: 正式登记 provisional
    space.setdefault("provisional", []).append({
        "name": name,
        "status": "provisional",
        "origin_factor": verdict["factor_name"],
        "formula": formula,
        "icir": verdict["icir"],
        "admitted_at": datetime.now().isoformat(timespec="seconds"),
        "requires": "S5 + JQ 终裁通过才转正",
        "anti_fixation_distance": round(dmin, 4),
    })
    save_space(space)
    verdict.update({"admitted": True, "reason": f"准入 provisional 范式 '{name}' (待 S5+JQ 转正)"})
    return verdict


def record_shadow(verdict: Dict):
    """shadow 模式落盘 (供 KPI 面板与日报统计)。"""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    store = {"records": []}
    if SHADOW_LOG.exists():
        try:
            store = json.loads(SHADOW_LOG.read_text(encoding="utf-8"))
        except Exception:
            store = {"records": []}
    store.setdefault("records", []).append(verdict)
    store["last_updated"] = datetime.now().isoformat(timespec="seconds")
    SHADOW_LOG.write_text(json.dumps(store, ensure_ascii=False, indent=1), encoding="utf-8")


def promote(name: str, evidence: str = "") -> bool:
    """provisional → promoted (S5+JQ 终裁通过后调用)。"""
    space = load_space()
    hit = None
    for p in space.get("provisional", []):
        if p.get("name") == name and p.get("status") == "provisional":
            hit = p
            break
    if not hit:
        return False
    hit["status"] = "promoted"
    hit["promoted_at"] = datetime.now().isoformat(timespec="seconds")
    hit["evidence"] = evidence
    space.setdefault("promoted", []).append(name)
    save_space(space)
    return True


def retire(name: str, reason: str = "") -> bool:
    """清退 provisional (回退计划: 一键清退, 未转正前无下游依赖)。"""
    space = load_space()
    kept, removed = [], None
    for p in space.get("provisional", []):
        if p.get("name") == name and p.get("status") == "provisional":
            removed = p
            continue
        kept.append(p)
    if not removed:
        return False
    space["provisional"] = kept
    space.setdefault("retired", []).append({
        "name": name, "reason": reason,
        "retired_at": datetime.now().isoformat(timespec="seconds"),
    })
    save_space(space)
    return True


# ══════════════════════════════════════════════════════════════════
# 4. 统计
# ══════════════════════════════════════════════════════════════════

def stats() -> Dict:
    space = load_space()
    shadow = {"records": []}
    if SHADOW_LOG.exists():
        try:
            shadow = json.loads(SHADOW_LOG.read_text(encoding="utf-8"))
        except Exception:
            pass
    recs = shadow.get("records", [])
    return {
        "mode": get_mode(),
        "base": len(space.get("base_paradigms", [])),
        "provisional_active": len(active_provisional(space)),
        "promoted": len(space.get("promoted", [])),
        "retired": len(space.get("retired", [])),
        "shadow_records": len(recs),
        "shadow_would_admit": sum(1 for r in recs if r.get("shadow_record")),
        "interval": PROPOSAL_INTERVAL,
        "s1_icir_min": S1_ICIR_MIN,
        "anti_fixation_min_dist": ANTI_FIXATION_MIN_DIST,
    }


def stats_line() -> str:
    s = stats()
    return (f"范式提案[{s['mode']}]: 空间 {s['base']}+provisional {s['provisional_active']}"
            f"+转正 {s['promoted']} | 每 {s['interval']} 轮 1 条 | "
            f"shadow 记录 {s['shadow_records']} (拟准入 {s['shadow_would_admit']})")


# ══════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════

def main():
    import argparse
    ap = argparse.ArgumentParser(description="范式提案模式 (P-20260910-007)")
    ap.add_argument("--stats", action="store_true")
    ap.add_argument("--dist", nargs=2, metavar=("EXPR_A", "EXPR_B"))
    ap.add_argument("--promote", metavar="NAME")
    ap.add_argument("--retire", metavar="NAME")
    args = ap.parse_args()

    if args.dist:
        print(f"AST 距离 = {ast_distance(args.dist[0], args.dist[1]):.4f}")
        return 0
    if args.promote:
        print("转正:", promote(args.promote, "manual CLI"))
        return 0
    if args.retire:
        print("清退:", retire(args.retire, "manual CLI"))
        return 0
    print(stats_line())
    print(json.dumps(stats(), ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
