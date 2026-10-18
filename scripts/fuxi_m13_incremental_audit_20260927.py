# -*- coding: utf-8 -*-
"""M13 (2026-09-27) 增量门修复 —— 取证 runner。

四组证据 (全部落盘 reports/fuxi_m13_incremental_audit_20260927/audit.json):

  1) control 选择规则 A/B（**真实库** data/experience_memory.json 的 success_templates）
       · 旧规则 = dict 遍历序前 8 条（M13 前实现）
       · 新规则 = sha256(formula) 前缀 8 条（预注册, 与遍历序无关）
       · 对库做 5 次固定种子打乱: 统计两规则各自产生多少个不同集合
         ⇒ 旧规则不稳定(=漂移源) / 新规则恒 1 个
  2) 快照生命周期: created → frozen → drift(**不切换**) → rotate(rev+1)
  3) 五态语义 A/B（参考实现对照, 不是改阈值）:
       旧: n<8 ⇒ gate_passed=True + "豁免"; 未启用 ⇒ gate_passed=True
       新: data_insufficient/not_evaluated ⇒ decision=deferred + gate_passed=False
  4) 分母可见性: run_incremental_pairs 在 (预算截断 / 缺公式 / 短序列 / 异常) 四种
     情形下每个 eligible 候选恰好落进一个桶, 补数队列可枚举。

注: 第 4 组用合成 S5 替身（只为验证接线与分母），**不构成任何因子有效性读数**;
    线上真读数在首个批次后从 ralph_loop 结果 JSON 的 summary.incremental_* 读取。
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FA = ROOT / "research" / "factor_alchemy"
sys.path.insert(0, str(FA))

import numpy as np  # noqa: E402

import control_snapshot as cs  # noqa: E402
import incremental_pairing as ip  # noqa: E402
from incremental_margin import IncrementalMarginGate  # noqa: E402

OUT_DIR = ROOT / "reports" / "fuxi_m13_incremental_audit_20260927"
MEMORY = ROOT / "data" / "experience_memory.json"
TOP_K = 8
N_SHUFFLE = 5


# ── 1) 真实库 → 公式池 ─────────────────────────────────────────────────
def load_library_formulas() -> dict:
    if not MEMORY.exists():
        return {"path": str(MEMORY), "exists": False, "formulas": [], "n_templates": 0}
    data = json.loads(MEMORY.read_text(encoding="utf-8"))
    raw = data.get("success_templates", [])
    if isinstance(raw, dict):
        raw = list(raw.values())
    out, seen = [], set()
    for t in raw:
        if isinstance(t, dict):
            name = t.get("pattern_id", t.get("factor_name", t.get("name", "")))
            formula = t.get("formula", t.get("expression", ""))
        else:
            name = getattr(t, "pattern_id", getattr(t, "name", ""))
            formula = getattr(t, "formula", getattr(t, "expression", ""))
        formula = str(formula or "").strip()
        if not formula or formula in seen:
            continue
        seen.add(formula)
        out.append({"name": str(name or ""), "formula": formula})
    return {"path": str(MEMORY), "exists": True,
            "n_templates": len(raw), "n_unique_formulas": len(out),
            "formulas": out}


def old_rule_select(items, top_k=TOP_K):
    """M13 前的实现: dict 遍历序前 top_k 条非空公式。"""
    return [i["formula"] for i in items if i.get("formula")][:top_k]


def stability_probe(items, *, seeds=(1, 2, 3, 4, 5)):
    old_sets, new_sets = [], []
    for s in seeds:
        shuffled = list(items)
        random.Random(s).shuffle(shuffled)
        old_sets.append(tuple(old_rule_select(shuffled)))
        new_sets.append(tuple(cs.select_formulas(shuffled)[0]))
    return {
        "n_shuffles": len(seeds),
        "old_rule": {
            "n_distinct_sets": len(set(old_sets)),
            "samples": [list(x) for x in list(dict.fromkeys(old_sets))[:3]],
        },
        "new_rule": {
            "n_distinct_sets": len(set(new_sets)),
            "sha256": sorted({cs.snapshot_sha256("sha256_prefix", TOP_K, list(x))
                              for x in new_sets}),
        },
        "verdict": ("new_rule_order_invariant" if len(set(new_sets)) == 1
                    else "new_rule_unstable"),
    }


# ── 3) 旧/新五态语义对照（参考实现） ──────────────────────────────────
def old_semantics(gate: IncrementalMarginGate, c, t, factor_id="probe"):
    """M13 前 evaluate() 的判定分支（仅用于对照; 不改阈值）。"""
    c = np.asarray(c, dtype=float)
    t = np.asarray(t, dtype=float)
    n = min(len(c), len(t))
    if n < 8:
        return {"evaluation_status": "(无此字段)", "decision": "(无此字段)",
                "gate_passed": True, "reason": f"样本不足 ({n}<8), 增量门禁豁免"}
    c, t = c[:n], t[:n]
    ir_c, ir_t = gate_ir(c, gate), gate_ir(t, gate)
    dd_c, dd_t = gate_mdd(c), gate_mdd(t)
    if not gate.enabled:
        return {"evaluation_status": "(无此字段)", "decision": "(无此字段)",
                "gate_passed": True, "reason": "增量门禁未启用 (影子观测)",
                "incremental_net_ir": ir_t - ir_c, "drawdown_deterioration": dd_t - dd_c}
    fails = []
    if ir_t - ir_c < gate.min_incremental_net_ir:
        fails.append("ir")
    if dd_t - dd_c > gate.max_drawdown_deterioration:
        fails.append("dd")
    return {"evaluation_status": "(无此字段)", "decision": "(无此字段)",
            "gate_passed": not fails,
            "reason": "旧实现: " + ("失败" if fails else "通过"),
            "incremental_net_ir": ir_t - ir_c, "drawdown_deterioration": dd_t - dd_c}


def gate_ir(r, gate):
    r = np.asarray(r, dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 4 or r.std(ddof=1) <= 0:
        return 0.0
    return float(r.mean() / r.std(ddof=1) * np.sqrt(gate.periods_per_year))


def gate_mdd(r):
    r = np.asarray(r, dtype=float)
    r = r[np.isfinite(r)]
    if r.size == 0:
        return 0.0
    cum = np.cumprod(1.0 + r)
    return float(np.max(1.0 - cum / np.maximum.accumulate(cum)))


# ── 4) 合成 S5 替身（只验证接线/分母） ────────────────────────────────
class _Res:
    def __init__(self, eligible=True):
        self.eligible_for_jq = eligible


class _SyntheticS5:
    def __init__(self, base, *, fail_on_treatment=False):
        self.base = list(base)
        self.fail_on_treatment = fail_on_treatment

    def rank_sum_combo_returns(self, formulas, *, top_n=None, names=None, stats=None):
        if stats is not None:
            stats.update({"n_formulas": len(formulas), "n_evaluated": len(formulas),
                          "n_contributed": len(formulas), "contribution_gap": 0,
                          "status": "valid", "series_len": len(self.base),
                          "probe": "synthetic_double"})
        if self.fail_on_treatment and len(formulas) > TOP_K:
            raise RuntimeError("synthetic treatment failure")
        return list(self.base)


def coverage_probe(base_series, snap_path: Path) -> dict:
    cases = {}

    # (a) 预算截断: 6 候选 / budget=2
    cands = [{"factor_name": f"c{i}", "formula": f"syn{i}"} for i in range(6)]
    res = [_Res() for _ in cands]
    r = ip.run_incremental_pairs(
        cands, res, lib_factors=[{"formula": f"lib{i}"} for i in range(10)],
        s5_filter=_SyntheticS5(base_series),
        cfg={"incremental_margin_enabled": True, "incremental_max_pair": 2},
        snapshot_path=snap_path.parent / "cov_budget.json")
    cases["budget_truncation"] = {
        "n_candidates": len(cands), "coverage": r["coverage"],
        "deferred": r["deferred"], "status": r["status"],
        "per_candidate_status": {c["factor_name"]: c.get("_incremental_eval_status")
                                 for c in cands},
    }

    # (b) 缺公式 / 短序列 / 异常 混合
    cands2 = [{"factor_name": "no_formula", "formula": ""},
              {"factor_name": "short_ok", "formula": "s1"},
              {"factor_name": "boom", "formula": "s2"}]
    res2 = [_Res() for _ in cands2]
    class _Mix(_SyntheticS5):
        def rank_sum_combo_returns(self, formulas, *, top_n=None, names=None, stats=None):
            if formulas and formulas[-1] == "s2":
                raise RuntimeError("synthetic boom")
            if formulas and formulas[-1] == "s1":
                return list(self.base[:5])          # 短序列
            return super().rank_sum_combo_returns(formulas, top_n=top_n, names=names, stats=stats)
    r2 = ip.run_incremental_pairs(
        cands2, res2, lib_factors=[{"formula": f"lib{i}"} for i in range(10)],
        s5_filter=_Mix(base_series),
        cfg={"incremental_margin_enabled": True},
        snapshot_path=snap_path.parent / "cov_mix.json")
    cases["mixed_paths"] = {
        "coverage": r2["coverage"], "deferred": r2["deferred"],
        "exceptions": r2["exceptions"], "status": r2["status"],
        "per_candidate_status": {c["factor_name"]: c.get("_incremental_eval_status")
                                 for c in cands2},
        "eligible_preserved": all(x.eligible_for_jq for x in res2),
    }
    return cases


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    audit: dict = {"schema": "fuxi_m13_incremental_audit/v1"}

    # ── 1) 真实库 control A/B ────────────────────────────────────────
    lib = load_library_formulas()
    items = lib.pop("formulas")
    audit["library"] = lib
    old_first8 = old_rule_select(items)
    new_first8, meta = cs.select_formulas(items)
    audit["control_selection"] = {
        "top_k": TOP_K,
        "n_unique_formulas": len(items),
        "old_rule_formulas": old_first8,
        "new_rule_formulas": new_first8,
        "new_rule_selection_meta": meta,
        "diff": cs.diff_formulas(old_first8, new_first8),
        "overlap_n": len(set(old_first8) & set(new_first8)),
    }
    audit["stability"] = stability_probe(items)

    # ── 2) 快照生命周期 ──────────────────────────────────────────────
    snap = OUT_DIR / "control_snapshot_audit.json"
    if snap.exists():
        snap.unlink()
    reg = cs.ControlSnapshotRegistry(snap)
    step1 = reg.ensure(items)
    step2 = reg.ensure(items)
    grown = items + [{"name": "synthetic_new", "formula": "SYNTHETIC_NEW_FORMULA_X"}]
    step3 = reg.ensure(grown)
    step4 = reg.ensure(items)
    # 确定会改变选择集的漂移源: 库大幅缩减 (sha256 前 8 名被移出池子)
    shrunk = items[:20] if len(items) > 20 else items[:max(1, len(items) // 4)]
    step5 = reg.ensure(shrunk)
    step6 = reg.ensure(items)
    audit["snapshot_lifecycle"] = {
        "created": {"status": step1["status"], "sha256": step1["snapshot"]["snapshot_sha256"],
                    "n_formulas": len(step1["snapshot"]["formulas"]),
                    "revision": step1["snapshot"]["revision"]},
        "frozen": {"status": step2["status"], "sha256": step2["snapshot"]["snapshot_sha256"],
                   "drift_detected": step2["drift_detected"]},
        "growth_no_drift": {"status": step3["status"],
                            "drift_detected": step3["drift_detected"],
                            "note": "sha256 前缀规则对新增不敏感: 新公式未挤进前 8 ⇒ 不漂移 (特性, 非缺陷)"},
        "drift_on_shrink": {"status": step5["status"],
                            "drift_detected": step5["drift_detected"],
                            "control_switched": (step5["snapshot"]["snapshot_sha256"]
                                                 != step1["snapshot"]["snapshot_sha256"]),
                            "diff": step5["diff"]},
        "frozen_again": {"status": step6["status"], "drift_detected": step6["drift_detected"],
                         "sha256": step6["snapshot"]["snapshot_sha256"]},
        "drift_events_recorded": len(cs.ControlSnapshotRegistry(snap).load().get("drift_events", [])),
    }
    rot = reg.rotate(items, reason="M13 audit: explicit rotation probe")
    audit["snapshot_lifecycle"]["rotate"] = {
        "revision": rot["snapshot"]["revision"],
        "rotation_reason": rot["snapshot"]["rotation_reason"],
        "n_update_log": len(rot["snapshot"]["update_log"]),
    }

    # ── 3) 五态语义 A/B（参考实现） ──────────────────────────────────
    rng = np.random.RandomState(11)
    noise = rng.normal(0, 0.02, 200)
    c200 = 0.003 + noise
    better = 0.006 + noise
    worse = c200 - 0.004
    probes = {
        "short_n5": (c200[:5], better[:5]),
        "empty": ([], []),
        "zero_variance": (np.zeros(60), np.zeros(60)),
        "valid_improve": (c200, better),
        "valid_degrade": (c200, worse),
    }
    gate_on = IncrementalMarginGate(enabled=True)
    gate_off = IncrementalMarginGate(enabled=False)
    sem = {}
    for name, (c, t) in probes.items():
        newv = (gate_off if name == "disabled_probe" else gate_on).evaluate(c, t, factor_id=name)
        oldv = old_semantics(gate_off if name == "disabled_probe" else gate_on, c, t, name)
        sem[name] = {"old": oldv, "new": newv.to_dict()}
    sem["disabled_probe"] = {
        "old": old_semantics(gate_off, c200, better, "disabled_probe"),
        "new": gate_off.evaluate(c200, better, factor_id="disabled_probe").to_dict(),
    }
    audit["five_state_ab"] = sem

    # ── 4) 分母可见性（合成 S5 替身） ────────────────────────────────
    base_series = (0.002 + rng.normal(0, 0.015, 120)).tolist()
    audit["coverage"] = coverage_probe(base_series, snap)
    audit["coverage_note"] = ("合成 S5 替身, 只验证接线/分母, 不构成因子有效性读数; "
                              "线上真读数见 ralph_loop 结果 JSON summary.incremental_*")

    (OUT_DIR / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
        encoding="utf-8", newline="\n")

    # ── 控制台摘要 ──────────────────────────────────────────────────
    stab = audit["stability"]
    print(f"[1] 库: templates={lib.get('n_templates')} unique={lib.get('n_unique_formulas')}")
    print(f"    旧规则 distinc sets/{stab['n_shuffles']} 打乱 = "
          f"{stab['old_rule']['n_distinct_sets']} | 新规则 = {stab['new_rule']['n_distinct_sets']}"
          f"  ⇒ {stab['verdict']}")
    print(f"    旧∩新 = {audit['control_selection']['overlap_n']}/8, "
          f"diff added={audit['control_selection']['diff']['n_added']} "
          f"removed={audit['control_selection']['diff']['n_removed']}")
    lc = audit["snapshot_lifecycle"]
    print(f"[2] 快照: {lc['created']['status']} rev{lc['created']['revision']} "
          f"→ {lc['frozen']['status']} → 增长({lc['growth_no_drift']['status']}) "
          f"→ 缩减({lc['drift_on_shrink']['status']}, 漂移={lc['drift_on_shrink']['drift_detected']}, "
          f"切换={lc['drift_on_shrink']['control_switched']}) → "
          f"{lc['frozen_again']['status']} | drift_events="
          f"{lc['drift_events_recorded']} | rotate rev{lc['rotate']['revision']}")
    for name in ("short_n5", "zero_variance", "valid_improve", "valid_degrade",
                 "disabled_probe"):
        o, n = sem[name]["old"], sem[name]["new"]
        print(f"[3] {name:<14} 旧: passed={o.get('gate_passed')} | "
              f"新: {n['evaluation_status']}/{n['decision']} passed={n['gate_passed']}")
    cb = audit["coverage"]["budget_truncation"]
    cm = audit["coverage"]["mixed_paths"]
    print(f"[4] 预算截断: {cb['coverage']['status_distribution']} "
          f"deferred={cb['deferred']['n_deferred']}")
    print(f"    混合路径: {cm['coverage']['status_distribution']} "
          f"exceptions={cm['exceptions']} eligible_preserved={cm['eligible_preserved']}")
    print(f"OUT {OUT_DIR / 'audit.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
