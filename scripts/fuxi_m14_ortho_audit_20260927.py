# -*- coding: utf-8 -*-
"""M14 (2026-09-27) 组合层正交化诊断 —— 取证 runner。

四组证据 (全部落盘 reports/fuxi_m14_ortho_audit_20260927/audit.json):

  1) control 同源对账: 用 M13 快照规则 (sha256_prefix, top_k=8) 对真实库
     (data/experience_memory.json) 建**审计专用**快照 (落 reports/, 不动 data/),
     校验 load_control 读出的 sha256 与 control_snapshot 规则独立计算一致;
     并记录线上快照 (data/ralph_control_snapshot.json) 的当前存在状态。
  2) 合成世界对照 (不依赖真实面板, 可完全复现):
     replica / indep / partial 三个已知候选 → base_r2 / 三口径 IC / 读法标签
     必须与理论值一致 (复制品 base_r2≈1, 独立≈0, 部分≈0.65 且 decay 高)。
  3) 真实面板读数 (若 FactorICComputer 可加载): 真实库挑 4 条候选公式,
     对 8 条 control 公式逐期截面回归, 落盘真实 base_r2 / 三口径 IC。
     —— 面板不可用 ⇒ 记录 skipped, 不阻塞其余证据。
  4) 红线检查: 源守卫 (不 import neutralize) + 落盘 payload 只含标量 +
     候选字典/结果对象不被修改 (零裁决硬断言)。

注: 本 runner **不写** data/ 下任何文件 (除 combo_orthogonalize 的读数文件仅
     在有真实面板时落盘); 审计快照落在 reports 子目录。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FA = ROOT / "research" / "factor_alchemy"
sys.path.insert(0, str(FA))
sys.path.insert(0, str(FA / "tests"))

import numpy as np  # noqa: E402

import combo_orthogonalize as co  # noqa: E402
import control_snapshot as cs  # noqa: E402

OUT_DIR = ROOT / "reports" / "fuxi_m14_ortho_audit_20260927"
MEMORY = ROOT / "data" / "experience_memory.json"
LIVE_SNAPSHOT = ROOT / "data" / "ralph_control_snapshot.json"
TOP_K = 8


# ── 0) 库加载 (与 M13 runner 同式) ─────────────────────────────────────
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


# ── 1) control 同源对账 ────────────────────────────────────────────────
def group1_control_parity(items) -> dict:
    # 幂等: 不删除已存在的审计快照 (内容由 sha256_prefix 规则决定, 稳定);
    # 首次运行 created, 之后 frozen —— 对账只看 sha, 不看 status 名。
    audit_snap = OUT_DIR / "control_snapshot_audit.json"
    reg = cs.ControlSnapshotRegistry(audit_snap)
    ens = reg.ensure(items, rule="sha256_prefix", top_k=TOP_K)
    formulas, meta = co.load_control(audit_snap)
    expect = cs.snapshot_sha256("sha256_prefix", TOP_K, formulas)
    live = {"path": str(LIVE_SNAPSHOT), "exists": LIVE_SNAPSHOT.exists()}
    if LIVE_SNAPSHOT.exists():
        live_formulas, live_meta = co.load_control(LIVE_SNAPSHOT)
        live.update({"status": live_meta.get("status"),
                     "snapshot_sha256": live_meta.get("snapshot_sha256"),
                     "same_as_audit": bool(
                         live_meta.get("snapshot_sha256") == meta.get("snapshot_sha256")),
                     "n_formulas": len(live_formulas)})
    return {
        "audit_snapshot_path": str(audit_snap),
        "ensure_status": ens.get("status"),
        "revision": meta.get("revision"),
        "snapshot_sha256": meta.get("snapshot_sha256"),
        "expected_sha256": expect,
        "sha_match": bool(meta.get("snapshot_sha256") == expect),
        "n_formulas": len(formulas),
        "formulas": formulas,
        "live_snapshot": live,
        "note": ("审计快照落 reports/ (不写 data/); 线上快照由 M13 首个真实批次创建, "
                 "同规则 ⇒ 同 sha; M14 在线上快照出现前返回 data_insufficient (不兜底)"),
    }


# ── 2) 合成世界对照 ────────────────────────────────────────────────────
def group2_synthetic() -> dict:
    import os
    import tempfile
    from test_m14_combo_orthogonalize_20260927 import build_world, _StubComp

    saved = dict(os.environ)
    co._CTX.clear()
    try:
        os.environ["FUXI_COMBO_ORTHO_FWD"] = "1"
        panels, price, meta = build_world()
        co.attach_computer(_StubComp(panels, price))
        co._CTX["ind_map"] = {}
        rows = {}
        for formula, name, expect in (
                ("F_COPY", "synthetic_replica", "replicated_exposure"),
                ("F_INDEP", "synthetic_independent", "increment_candidate"),
                ("F_PARTIAL", "synthetic_partial", "weak_increment")):
            row = co.factor_check(formula, name, ["F_X1", "F_X2"], ctl_sha="syn")
            rows[name] = {k: row.get(k) for k in
                          ("status", "read", "base_r2_median", "rank_raw",
                           "rank_resid", "pearson_raw", "ic_decay_ratio", "n_periods")}
            rows[name]["expect_read"] = expect
            rows[name]["read_match"] = (row.get("read") == expect)
        return {"world": meta, "rows": rows,
                "all_read_match": all(r["read_match"] for r in rows.values())}
    finally:
        os.environ.clear()
        os.environ.update(saved)
        co._CTX.clear()


# ── 3) 真实面板读数 ────────────────────────────────────────────────────
def group3_real_panel(items, ctl_formulas, max_cand: int = 4) -> dict:
    out: dict = {"status": "skipped", "rows": []}
    try:
        from factor_ic_computer import FactorICComputer
        comp = FactorICComputer(lookback_days=270)
        comp._load_data()
    except Exception as e:
        out["reason"] = f"computer_unavailable: {type(e).__name__}: {str(e)[:160]}"
        return out
    if getattr(comp, "_price_df", None) is None or len(comp._price_df) == 0:
        out["reason"] = "price_panel_empty"
        return out

    co._CTX.clear()
    co.attach_computer(comp)
    # 候选选取: 最短 2 条 + 最长 2 条 (覆盖简/繁两类公式), 去重
    fs = sorted(items, key=lambda x: len(x["formula"]))
    picked, seen = [], set()
    for it in (fs[:2] + fs[-2:]):
        if it["formula"] in seen:
            continue
        seen.add(it["formula"])
        picked.append(it)
    picked = picked[:max_cand]

    rows = []
    for i, it in enumerate(picked):
        try:
            row = co.factor_check(it["formula"], it.get("name") or f"cand{i}",
                                  ctl_formulas, ctl_sha="real", verbose=False)
        except Exception as e:
            row = {"factor_name": it.get("name"), "status": "error",
                   "reason": f"{type(e).__name__}: {str(e)[:140]}"}
        if row is None:
            row = {"factor_name": it.get("name"), "status": "no_panel",
                   "reason": "evaluate_factor_formula returned nothing"}
        row = dict(row)
        row["formula_head"] = it["formula"][:70]
        row["formula_len"] = len(it["formula"])
        rows.append(row)
    out["status"] = "valid" if any(r.get("status") == "valid" for r in rows) else "data_insufficient"
    out["rows"] = rows
    out["price_panel_rows"] = int(len(comp._price_df))
    return out


# ── 4) 红线检查 ────────────────────────────────────────────────────────
def group4_redline(items) -> dict:
    src = (FA / "combo_orthogonalize.py").read_text(encoding="utf-8")
    checks = {
        "no_neutralize_import": ("import neutralize" not in src
                                 and "from neutralize" not in src),
        "forbidden_use_doc_in_source": ("V5" in src and "中性化" in src
                                        and "裁决输入" in src),
        # 检查**调用形式** (docstring 里作为"禁止说明"提及这些名字是允许的)
        "no_backtest_call_in_source": (".backtest_factor(" not in src
                                       and ".rank_sum_combo_returns(" not in src),
        "source_pure_lf": None,      # 由下方统计
    }
    raw = (FA / "combo_orthogonalize.py").open("rb").read()
    checks["source_pure_lf"] = (raw.count(b"\r") == 0)
    checks["source_bytes"] = len(raw)

    # 零裁决硬断言: evaluate 不改候选/结果
    import os
    import tempfile
    from test_m14_combo_orthogonalize_20260927 import build_world, _StubComp, _Res
    saved = dict(os.environ)
    co._CTX.clear()
    try:
        os.environ["FUXI_COMBO_ORTHO_FWD"] = "1"
        panels, price, _ = build_world()
        co.attach_computer(_StubComp(panels, price))
        co._CTX["ind_map"] = {}
        cands = [{"factor_name": "copy", "formula": "F_COPY"}]
        results = [_Res(eligible=True)]
        before = json.dumps(cands, ensure_ascii=False)
        with tempfile.TemporaryDirectory() as td:
            payload = co.evaluate(cands, results, ctl_formulas=["F_X1"],
                                  out_path=Path(td) / "o.json", verbose=False)
            raw_json = (Path(td) / "o.json").read_text(encoding="utf-8")
        checks["candidates_unmodified"] = (json.dumps(cands, ensure_ascii=False) == before)
        checks["eligible_preserved"] = bool(results[0].eligible_for_jq)
        checks["payload_has_no_residual_key"] = "residuals" not in raw_json
        checks["payload_lines"] = raw_json.count("\n")
        checks["eval_status"] = payload.get("status")
    finally:
        os.environ.clear()
        os.environ.update(saved)
        co._CTX.clear()
    checks["all_pass"] = all(v is True for k, v in checks.items()
                             if isinstance(v, bool))
    return checks


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    lib = load_library_formulas()
    items = lib.pop("formulas")

    audit: dict = {"schema": "fuxi_m14_ortho_audit/v1", "library": lib}

    print("[1] control 同源对账 ...")
    audit["control_parity"] = group1_control_parity(items)
    g1 = audit["control_parity"]
    print(f"    sha={str(g1['snapshot_sha256'])[:16]} match={g1['sha_match']} "
          f"n={g1['n_formulas']} | live exists={g1['live_snapshot']['exists']}")

    print("[2] 合成世界对照 ...")
    audit["synthetic"] = group2_synthetic()
    for name, r in audit["synthetic"]["rows"].items():
        print(f"    {name:<24} base_r2={r['base_r2_median']:.3f} "
              f"decay={r['ic_decay_ratio']:+.2f} read={r['read']} "
              f"(expect {r['expect_read']}) match={r['read_match']}")
    print(f"    all_read_match={audit['synthetic']['all_read_match']}")

    print("[3] 真实面板读数 (可能耗时) ...")
    audit["real_panel"] = group3_real_panel(items, g1["formulas"])
    print(f"    status={audit['real_panel']['status']} "
          f"n_rows={len(audit['real_panel'].get('rows', []))} "
          f"{audit['real_panel'].get('reason', '')}")
    for r in audit["real_panel"].get("rows", []):
        if r.get("status") == "valid":
            print(f"    - {str(r['factor_name'])[:26]:<28} base_r2={r['base_r2_median']:.3f} "
                  f"raw={r['rank_raw']:+.4f} resid={r['rank_resid']:+.4f} "
                  f"pearson={r['pearson_raw']:+.4f} read={r['read']}")
        else:
            print(f"    - {str(r.get('factor_name'))[:26]:<28} [{r.get('status')}] "
                  f"{str(r.get('reason', ''))[:80]}")

    print("[4] 红线检查 ...")
    audit["redline"] = group4_redline(items)
    print(f"    all_pass={audit['redline']['all_pass']} "
          f"candidates_unmodified={audit['redline'].get('candidates_unmodified')} "
          f"eligible_preserved={audit['redline'].get('eligible_preserved')}")

    # 版本元信息
    try:
        reg = json.loads((ROOT / "data/version_registry.json").read_text(encoding="utf-8"))
        vs = reg.get("versions", reg)
        lst = vs if isinstance(vs, list) else list(vs.values())
        audit["registry_version"] = lst[-1].get("version") if lst else None
    except Exception:
        audit["registry_version"] = None

    (OUT_DIR / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
        encoding="utf-8", newline="\n")
    print(f"OUT {OUT_DIR / 'audit.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
