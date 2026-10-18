# -*- coding: utf-8 -*-
"""M10 门禁 A/B 反事实 + 顺序修复实证（2026-09-27）。

背景：M10 首批两门落地 ——
  A. forbidden_directions 公式级硬屏蔽门：默认 shadow -> enforce（配豁免白名单，
     库内 4 条冲突逐条裁决：3 保留拦截 + 1 执行层根因豁免）；
  B. AST 结构去重门：S1 前置打标接线（enforce 通道打通），默认保持 shadow
     （转正留后续批，一次只翻一门）。
  另有顺序修复：forbidden / struct 打标从「符号守卫之前」移到「守卫之后」
  （守卫 enforce 会原地替换 formula 为翻转版，守卫前打标会漏检实际提交式）。

本脚本产出四组证据（全部只读或隔离运行，不改生产状态）：
  G1 冲突裁决表     — 库内 4 条冲突逐条裁决（含黑名单证据字段 + 豁免状态）
  G2 生产坐标 A/B   — shadow vs enforce 双跑三个坐标：
      a) 全库唯一公式（unified+passed 去重）
      b) S5 在途队列（stage2_s5_pending 55 条，按名反查库内公式）
      c) 生成链提案流（stage1_factor_proposals 2530 条）
  G3 顺序修复实证   — (a) 源码锚：guard < forbidden < struct 字节序；
      (b) 黑名单原型符号敏感性：p vs -(p) 的命中（量化顺序修复的检出上界）
  G4 默认与不变量   — 默认 enforce / 回退通道 / exempt 开关 / struct_dedup
      默认 shadow + would_reject=False / S1 硬拒用 not_evaluated 字段锚

用法：python scripts/fuxi_m10_gate_ab_20260927.py
输出：reports/fuxi_m10_gate_audit_20260927/audit.json + stdout 摘要
"""
from __future__ import annotations

import contextlib
import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FA = ROOT / "research" / "factor_alchemy"
OUT = ROOT / "reports" / "fuxi_m10_gate_audit_20260927"
if str(FA) not in sys.path:
    sys.path.insert(0, str(FA))

import forbidden_gate as fg                                       # noqa: E402
import structure_dedup as sd                                       # noqa: E402

ENV_KEYS = ("FUXI_FORBIDDEN_GATE", "FUXI_FORBIDDEN_EXEMPT", "FUXI_STRUCT_DEDUP")


@contextlib.contextmanager
def env_ctx(**kv):
    """隔离环境变量（组间零污染）。"""
    saved = {k: os.environ.get(k) for k in ENV_KEYS}
    for k in ENV_KEYS:
        os.environ.pop(k, None)
    for k, v in kv.items():
        os.environ[k] = v
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def load_library_formulas():
    """全库唯一公式（unified + passed，保序去重）。"""
    seen, out = set(), []
    for path in [ROOT / "data" / "unified_factor_pool.csv",
                 ROOT / "data" / "passed_factor_pool.csv"]:
        if not path.exists():
            continue
        with open(path, encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                ex = (row.get("formula") or "").strip()
                if not ex:
                    continue
                key = ex[:200]
                if key in seen:
                    continue
                seen.add(key)
                out.append({"factor_name": (row.get("name") or "?").strip(),
                            "formula": ex, "pool": path.name,
                            "status": (row.get("status") or "").strip()})
    return out


def build_name_formula_index():
    """名字 -> 公式独立索引（先到先得，不参与公式去重，保证反向查名字完备）。"""
    idx = {}
    for path in [ROOT / "data" / "unified_factor_pool.csv",
                 ROOT / "data" / "passed_factor_pool.csv"]:
        if not path.exists():
            continue
        with open(path, encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                nm = (row.get("name") or "").strip()
                ex = (row.get("formula") or "").strip()
                if nm and ex:
                    idx.setdefault(nm, ex)
    return idx


def scan_check_all(gate, items, expr_key="formula"):
    """逐条 check，返回聚合 + 命中清单。"""
    n_blocked = n_exempted = n_enforced = 0
    hits = []
    for it in items:
        expr = (it.get(expr_key) or "").strip()
        if not expr:
            continue
        r = gate.check(expr)
        if r.get("blocked"):
            n_blocked += 1
            n_exempted += int(bool(r.get("exempted")))
            n_enforced += int(bool(r.get("enforced")))
            hits.append({"factor_name": it.get("factor_name"),
                         "similarity": r["similarity"],
                         "exact": bool(r.get("exact")),
                         "direction_id": r["direction_id"],
                         "exempted": bool(r.get("exempted")),
                         "enforced": bool(r.get("enforced"))})
    return {"n_scanned": len(items), "n_blocked": n_blocked,
            "n_exempted": n_exempted, "n_enforced": n_enforced, "hits": hits}


def main():
    started = datetime.now(timezone.utc)
    checks = {}

    # ════════════════════════════════════════════════════════════
    # G1 冲突裁决表 —— 库内 4 条冲突
    # ════════════════════════════════════════════════════════════
    lib = load_library_formulas()
    with env_ctx(FUXI_FORBIDDEN_GATE="enforce"):
        g = fg.ForbiddenGate()
        g.load()
        g1_scan = scan_check_all(g, lib)

    mem = json.loads((ROOT / "data" / "experience_memory.json").read_text(encoding="utf-8"))
    dirs = {str(d.get("direction_id")): d for d in (mem.get("forbidden_directions") or [])
            if isinstance(d, dict)}
    g1_rows = []
    for h in g1_scan["hits"]:
        d = dirs.get(h["direction_id"]) or {}
        g1_rows.append({
            "factor_name": h["factor_name"],
            "formula_head": next((f["formula"][:110] for f in lib
                                  if f["factor_name"] == h["factor_name"]), ""),
            "similarity": h["similarity"], "exact": h["exact"],
            "direction_id": h["direction_id"],
            "jq_return": d.get("jq_return"), "jq_sharpe": d.get("jq_sharpe"),
            "desc": str(d.get("description") or "")[:110],
            "verdict": ("exempted_execution_layer" if h["exempted"] else "keep_block"),
            "exempted": h["exempted"], "enforced": h["enforced"],
        })
    checks["g1_conflicts_eq4"] = len(g1_rows) == 4
    checks["g1_verdict_split_3_1"] = (
        sum(1 for r in g1_rows if r["verdict"] == "keep_block") == 3
        and sum(1 for r in g1_rows if r["verdict"] == "exempted_execution_layer") == 1)

    print("=" * 96)
    print("G1 冲突裁决表（库内 %d 唯一公式，命中 %d 条）" % (len(lib), len(g1_rows)))
    for r in g1_rows:
        print("  [%s] %s sim=%.4f dir=%s jq=%s%%/%s"
              % (r["verdict"], r["factor_name"], r["similarity"],
                 r["direction_id"], r["jq_return"], r["jq_sharpe"]))

    # ════════════════════════════════════════════════════════════
    # G2 生产坐标 A/B —— shadow vs enforce
    # ════════════════════════════════════════════════════════════
    g2 = {}

    # a) 全库
    with env_ctx(FUXI_FORBIDDEN_GATE="shadow"):
        ga = fg.ForbiddenGate()
        ga.load()
        a_lib = scan_check_all(ga, lib)
    g2["a_library"] = {"A_shadow": {k: a_lib[k] for k in
                                    ("n_scanned", "n_blocked", "n_exempted", "n_enforced")},
                       "B_enforce": {k: g1_scan[k] for k in
                                     ("n_scanned", "n_blocked", "n_exempted", "n_enforced")}}
    a_set = {h["factor_name"] for h in a_lib["hits"] if h["enforced"]}
    b_set = {h["factor_name"] for h in g1_scan["hits"] if h["enforced"]}
    g2["a_new_blocks_enforce_minus_shadow"] = sorted(b_set - a_set)
    g2["a_newly_exempted"] = sorted({h["factor_name"] for h in g1_scan["hits"] if h["exempted"]})
    checks["g2a_shadow_zero_enforced"] = a_lib["n_enforced"] == 0
    checks["g2a_enforce_new_blocks_eq3"] = len(b_set - a_set) == 3
    checks["g2a_exempt_not_forced"] = all(
        not h["enforced"] for h in g1_scan["hits"] if h["exempted"])

    # b) S5 在途队列（按名反查：库名字索引 -> 提案流 -> 真缺口）
    name_idx = build_name_formula_index()
    props_path = ROOT / "data" / "stage1_factor_proposals.json"
    props = json.loads(props_path.read_text(encoding="utf-8")).get("proposals") or []
    prop_by_name = {}
    for p in props:
        nm = p.get("factor_name")
        ex = (p.get("formula_pandas") or "").strip()
        if nm and ex:
            prop_by_name.setdefault(nm, ex)
    pend_path = ROOT / "data" / "stage2_s5_pending.json"
    pend = json.loads(pend_path.read_text(encoding="utf-8")).get("pending") or []
    pend_items, pend_missing = [], []
    n_from_lib = n_from_props = 0
    for c in pend:
        nm = c.get("factor_name")
        ex = name_idx.get(nm)
        src = "library"
        if not ex:
            ex = prop_by_name.get(nm)
            src = "proposals"
        if not ex:
            pend_missing.append(nm)
            continue
        if src == "library":
            n_from_lib += 1
        else:
            n_from_props += 1
        pend_items.append({"factor_name": nm, "formula": ex, "src": src})
    with env_ctx(FUXI_FORBIDDEN_GATE="enforce"):
        gb = fg.ForbiddenGate()
        gb.load()
        b_pend = scan_check_all(gb, pend_items)
    g2["b_s5_pending"] = {"n_pending": len(pend), "n_with_formula": len(pend_items),
                          "n_from_library": n_from_lib, "n_from_proposals": n_from_props,
                          "n_missing": len(pend_missing), "missing_names": pend_missing,
                          "n_blocked": b_pend["n_blocked"],
                          "n_enforced": b_pend["n_enforced"],
                          "hits": b_pend["hits"]}
    checks["g2b_pending_zero_hits"] = b_pend["n_blocked"] == 0
    checks["g2b_pending_formula_coverage_full"] = len(pend_items) == len(pend)

    # c) 生成链提案流（props 已在上方加载）
    prop_items = []
    for p in props:
        f = (p.get("formula_pandas") or "").strip()
        if not f:
            continue
        prop_items.append({"factor_name": p.get("factor_name"), "formula": f,
                           "status": p.get("status"), "generated_at": p.get("generated_at")})
    with env_ctx(FUXI_FORBIDDEN_GATE="enforce"):
        gc = fg.ForbiddenGate()
        gc.load()
        c_prop = scan_check_all(gc, prop_items)
    c_hits = []
    for h in c_prop["hits"]:
        src = next((p for p in prop_items if p["factor_name"] == h["factor_name"]), {})
        c_hits.append({**h, "status": src.get("status"),
                       "generated_at": str(src.get("generated_at"))[:19]})
    g2["c_proposals"] = {"n_proposals": len(props), "n_with_formula": len(prop_items),
                         "n_blocked": c_prop["n_blocked"], "n_enforced": c_prop["n_enforced"],
                         "n_exempted": c_prop["n_exempted"], "hits": c_hits}
    checks["g2c_proposals_hits_le5"] = c_prop["n_blocked"] <= 5

    print()
    print("=" * 96)
    print("G2 生产坐标 A/B（shadow vs enforce）")
    print("  a) 全库 %d 条: shadow 标记 %d / 硬拒 %d → enforce 标记 %d / 硬拒 %d"
          % (len(lib), a_lib["n_blocked"], a_lib["n_enforced"],
             g1_scan["n_blocked"], g1_scan["n_enforced"]))
    print("     新增硬拒: %s" % ", ".join(g2["a_new_blocks_enforce_minus_shadow"]))
    print("     豁免: %s" % ", ".join(g2["a_newly_exempted"]))
    print("  b) S5 在途队列 %d 条（库反查 %d / 提案流补查 %d / 缺失 %d）: 命中 %d / 硬拒 %d"
          % (len(pend), n_from_lib, n_from_props, len(pend_missing),
             b_pend["n_blocked"], b_pend["n_enforced"]))
    print("  c) 提案流 %d 条（有公式 %d）: 命中 %d / 硬拒 %d"
          % (len(props), len(prop_items), c_prop["n_blocked"], c_prop["n_enforced"]))
    for h in c_hits:
        print("     - %s sim=%.4f status=%s gen=%s"
              % (h["factor_name"], h["similarity"], h["status"], h["generated_at"]))

    # ════════════════════════════════════════════════════════════
    # G3 顺序修复实证
    # ════════════════════════════════════════════════════════════
    ralph_text = (FA / "ralph_loop.py").read_text(encoding="utf-8")
    i_guard = ralph_text.find("_apply_sign_guard(candidates")
    i_forb = ralph_text.find('cand["_forbidden_hit"] = _fr')
    i_struct = ralph_text.find('cand["_struct_dup_hit"]')
    order_ok = 0 < i_guard < i_forb < i_struct
    checks["g3a_order_guard_lt_forbidden_lt_struct"] = order_ok

    with env_ctx(FUXI_FORBIDDEN_GATE="enforce"):
        g3 = fg.ForbiddenGate()
        g3.load()
        protos = [e["expr"] for e in g3.entries.values() if e.get("expr")]
        sens = []
        for p in protos:
            rp = g3.check(p)
            rn = g3.check("-(%s)" % p)
            sens.append({"proto_head": p[:80], "sim_self": rp["similarity"],
                         "self_blocked": rp["blocked"],
                         "sim_flip": rn["similarity"], "flip_blocked": rn["blocked"]})
    n_self = sum(1 for r in sens if r["self_blocked"])
    n_flip = sum(1 for r in sens if r["flip_blocked"])
    checks["g3b_all_prototypes_self_blocked"] = n_self == len(protos)
    n_flip_escape = sum(1 for r in sens if r["self_blocked"] and not r["flip_blocked"])
    # 顺序修复场景分类（符号守卫 enforce 会原地替换 formula 为翻转版）:
    #   A 误杀修复: 候选原始式 = p(短原型) 且被翻转 -> 修复前查 p 命中(误杀反向提交式),
    #     修复后查 -(p) 不命中 -> 放行;  B 漏检修复: 原始式 = -(p) 且被翻转 ->
    #     修复前查 -(p) 不命中(漏检), 修复后查 p 命中 -> 拦截。两者均以修复后为准。
    esc_protos = [r for r in sens if r["self_blocked"] and not r["flip_blocked"]]
    scenario = {
        "n_both_blocked": sum(1 for r in sens if r["self_blocked"] and r["flip_blocked"]),
        "n_flip_escape": n_flip_escape,
        "escape_prototypes": [r["proto_head"] for r in esc_protos],
        "escape_sim_self": [r["sim_self"] for r in esc_protos],
        "escape_sim_flip": [r["sim_flip"] for r in esc_protos],
        "note": ("短骨架条目(<=4 token)加前导负号后跌破 0.95 阈值 => 翻转逃脱; "
                 "长骨架(DVS 类)符号不敏感. 修复前查守卫前形态=双向错位, "
                 "修复后查实际提交式."),
    }

    print()
    print("=" * 96)
    print("G3 顺序修复实证")
    print("  a) 源码字节序: guard@%d < forbidden@%d < struct@%d -> %s"
          % (i_guard, i_forb, i_struct, order_ok))
    print("  b) 黑名单原型 %d 条: 原式命中 %d / 翻转版命中 %d / 翻转逃脱 %d"
          % (len(protos), n_self, n_flip, n_flip_escape))
    print("     逃脱原型: %s" % ", ".join(scenario["escape_prototypes"]))

    # ════════════════════════════════════════════════════════════
    # G4 默认与不变量对账
    # ════════════════════════════════════════════════════════════
    g4 = {}
    with env_ctx():
        gd = fg.ForbiddenGate()
        gd.load()
        g4["forbidden_default_mode"] = gd.mode
        g4["forbidden_default_exempt_size"] = len(gd.exempt_entries)
        g4["exempt_path_is_default"] = str(gd.exempt_path) == str(fg.DEFAULT_EXEMPT_PATH)
    with env_ctx(FUXI_FORBIDDEN_GATE="shadow"):
        g4["rollback_shadow_mode"] = fg.ForbiddenGate().mode
    with env_ctx(FUXI_FORBIDDEN_GATE="off"):
        g4["rollback_off_mode"] = fg.ForbiddenGate().mode
    with env_ctx(FUXI_FORBIDDEN_EXEMPT="off"):
        g4["exempt_off_resolves_none"] = fg._resolve_exempt_path() is None
    g4["struct_default_is_shadow_srcanchor"] = 'os.environ.get("FUXI_STRUCT_DEDUP", "shadow")' in ralph_text

    # struct_dedup 行为：默认（无 enforce env）would_reject 必须 False
    with env_ctx():
        ded = sd.get_indexed_deductor()
        g4["struct_deductor_available"] = ded is not None
        if ded is not None:
            sample = lib[0]["formula"] if lib else "(close)"
            rc = ded.check_candidate(sample)
            g4["struct_default_mode"] = rc.get("mode")
            g4["struct_default_would_reject"] = rc.get("would_reject")

    # S1 硬拒五态字段锚
    msv_text = (FA / "multi_stage_validator.py").read_text(encoding="utf-8")
    g4["s1_forbidden_hard_reject_uses_not_evaluated"] = (
        's1_eval_status = "not_evaluated"' in msv_text)
    g4["s1_struct_dup_consumed"] = 'cand.get("_struct_dup_hit")' in msv_text

    checks["g4a_forbidden_default_enforce"] = g4["forbidden_default_mode"] == "enforce"
    checks["g4a_rollback_channels"] = (g4["rollback_shadow_mode"] == "shadow"
                                       and g4["rollback_off_mode"] == "off")
    checks["g4b_exempt_default_path_on"] = (g4["exempt_path_is_default"]
                                            and g4["forbidden_default_exempt_size"] == 1
                                            and g4["exempt_off_resolves_none"])
    checks["g4c_struct_shadow_no_reject"] = (
        g4.get("struct_default_mode") == "shadow"
        and g4.get("struct_default_would_reject") is False)
    checks["g4d_s1_not_evaluated_anchor"] = g4["s1_forbidden_hard_reject_uses_not_evaluated"]

    print()
    print("=" * 96)
    print("G4 默认与不变量")
    for k in ("forbidden_default_mode", "forbidden_default_exempt_size",
              "rollback_shadow_mode", "rollback_off_mode", "exempt_off_resolves_none",
              "struct_default_mode", "struct_default_would_reject"):
        print("  %s = %s" % (k, g4.get(k)))

    # ════════════════════════════════════════════════════════════
    # G5 struct_dedup 转正前置：误杀预评估（只读，enforce 隔离实例）
    # ════════════════════════════════════════════════════════════
    g5 = {}
    with env_ctx(FUXI_STRUCT_DEDUP="enforce"):
        ded = sd.get_indexed_deductor()
        g5["deductor_available"] = ded is not None
        if ded is not None:
            g5["mode"] = ded.mode
            g5["n_lib_factors"] = len(ded.factors)
            # 坐标: S5 在途队列（已在库再审场景 —— 自匹配高危区）
            n_dup = n_self = n_other = 0
            other_pairs = []
            for it in pend_items:
                r = ded.check_candidate(it["formula"])
                if r.get("is_structural_dup"):
                    n_dup += 1
                    mf = (r.get("per_abstraction", {}).get("literal", {}) or {}).get("matched_factor")
                    if mf == it["factor_name"]:
                        n_self += 1
                    else:
                        n_other += 1
                        other_pairs.append({"factor": it["factor_name"], "matched": mf})
            g5.update(n_pending=len(pend_items), n_dup=n_dup,
                      n_self_match=n_self, n_other_match=n_other,
                      other_pairs=other_pairs)
            # 坐标: 库内抽样基线（固定步长确定性抽样，与随机种子无关）
            lib_pairs = [(f["factor_name"], f["formula"]) for f in lib if f["formula"]]
            step = max(1, len(lib_pairs) // 40)
            sample = lib_pairs[::step][:40]
            s_dup = s_self = s_other = 0
            for nm, ex in sample:
                rr = ded.check_candidate(ex)
                if rr.get("is_structural_dup"):
                    s_dup += 1
                    mf = (rr.get("per_abstraction", {}).get("literal", {}) or {}).get("matched_factor")
                    if mf == nm:
                        s_self += 1
                    else:
                        s_other += 1
            g5.update(lib_sample_n=len(sample), lib_sample_dup=s_dup,
                      lib_sample_self=s_self, lib_sample_other=s_other)
    # 基线断言（机制性）: 确认自匹配缺口存在 = struct_dedup 转正阻断项。
    # 语义说明: 本 check 的「PASS」表示缺陷基线被复现（阻断项仍在）。
    # 若未来做 leave-self-out 修复, n_self_match/lib_sample_self 应大幅下降 —— 
    # 本 check 会转 FAIL, 即提醒复核转正前置条件已解除。
    checks["g5_struct_self_match_gap_confirmed"] = (
        g5.get("n_self_match", 0) >= 30 and g5.get("lib_sample_self", 0) >= 15)
    checks["g5_struct_other_match_small"] = (
        g5.get("n_other_match", 99) <= 5 and g5.get("lib_sample_other", 99) <= 5)

    print()
    print("=" * 96)
    print("G5 struct_dedup 转正前置（enforce 隔离跑，只读）")
    print("  库索引 %d 条 | pending 55: dup %d (self %d / other %d) | 库抽样 40: dup %d (self %d / other %d)"
          % (g5.get("n_lib_factors", 0), g5.get("n_dup", 0), g5.get("n_self_match", 0),
             g5.get("n_other_match", 0), g5.get("lib_sample_dup", 0),
             g5.get("lib_sample_self", 0), g5.get("lib_sample_other", 0)))
    print("  转正阻断项: 无 leave-self-out —— 已入库因子再审会被自身条目判重"
          "（pending 误杀率 %d/%d）" % (g5.get("n_self_match", 0), g5.get("n_pending", 0)))
    for op in g5.get("other_pairs", []):
        print("    other: %s ~ %s" % (op["factor"], op["matched"]))

    # ════════════════════════════════════════════════════════════
    # 汇总
    # ════════════════════════════════════════════════════════════
    report = {
        "schema": "fuxi_m10_gate_ab/v1",
        "generated_at": started.isoformat(timespec="seconds"),
        "batch": "M10 (P-20260927-006)",
        "checks": checks,
        "g1_conflict_adjudication": g1_rows,
        "g2_ab": g2,
        "g3_order_fix": {"byte_positions": {"guard": i_guard, "forbidden": i_forb,
                                            "struct": i_struct},
                         "order_ok": order_ok,
                         "forbidden_threshold": g3.jaccard_threshold,
                         "n_prototypes": len(protos),
                         "n_self_blocked": n_self, "n_flip_blocked": n_flip,
                         "n_flip_escape": n_flip_escape,
                         "scenario": scenario,
                         "sign_sensitivity": sens},
        "g4_defaults": g4,
        "g5_struct_dedup_preflight": g5,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1, default=str), encoding="utf-8")

    print()
    print("=" * 96)
    n_pass = sum(1 for v in checks.values() if v)
    print("checks: %d/%d 通过" % (n_pass, len(checks)))
    for k, v in checks.items():
        print("  [%s] %s" % ("PASS" if v else "FAIL", k))
    print("written: %s" % (OUT / "audit.json"))
    return 0 if n_pass == len(checks) else 1


if __name__ == "__main__":
    sys.exit(main())
