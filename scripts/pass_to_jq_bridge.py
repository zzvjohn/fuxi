# -*- coding: utf-8 -*-
"""
PASS→S5/JQ 自动桥接 (P-20260908-006 落地)
==========================================
修复"Stage2 PASS 入池后桥接悬空": 池内 status=candidate 因子自动走
S5 验证 (复刻 RalphLoop._phase_evaluate 路径, 复用 s135_candidate_check 组件)
→ eligible_for_jq 且公式可翻译的批量生成 JQ 单因子代码 (gen_jq_generic 路径 A)
→ 写入 jq_codegen_queue.json (status=pending_jq_run)。

用法:
  python scripts/pass_to_jq_bridge.py             # 盘点模式 (默认, 零副作用)
  python scripts/pass_to_jq_bridge.py --execute   # 桥接模式: S5 验证 + JQ codegen
  env FUXI_PASS_JQ_BRIDGE=on 等价 --execute (钩子由 daily_factor_hypothesis.py 调用)

安全设计:
  - 盘点模式只打印不写任何文件
  - 桥接幂等: 已在 jq_codegen_queue 的因子跳过; 同名 JQ 文件跳过
  - JQ 唯一真相源不变: 生成文件仍须用户手动复制到聚宽回测
  - 结果落盘 data/pass_to_jq_bridge_result.json 供看板/日报消费
"""
import json
import os
import sys
import argparse
import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FA_DIR = ROOT / "research" / "factor_alchemy"
DATA_DIR = ROOT / "data"
sys.path.insert(0, str(FA_DIR))
sys.path.insert(0, str(ROOT / "scripts"))

from s135_candidate_check import _is_pandas_compatible, load_lib_factors  # noqa: E402
from gen_jq_generic import prepare_factor, generate_standalone  # noqa: E402

QUEUE_PATH = DATA_DIR / "jq_codegen_queue.json"
RESULT_PATH = DATA_DIR / "pass_to_jq_bridge_result.json"


def load_pool_candidates():
    """池内 status=candidate → 候选 dict 列表 (含池内 IC/ICIR 预计算值)"""
    import csv
    with open(DATA_DIR / "passed_factor_pool.csv", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    out = []
    for r in rows:
        if r.get("status") != "candidate":
            continue
        formula = str(r.get("formula", "") or "").strip()
        out.append({
            "factor_name": str(r["name"]),
            "formula": formula,
            "hypothesis": str(r.get("hypothesis", "") or "")[:300],
            "paradigm": str(r.get("category", "") or ""),
            "ic": float(r.get("ic_mean") or 0),
            "icir": float(r.get("icir") or 0),
            "direction": str(r.get("direction", "long") or "long"),
        })
    return out


def load_exclusions():
    """排除集: JQ 队列已有 + 冷冻库"""
    excl = {"queue": set(), "frozen": set()}
    if QUEUE_PATH.exists():
        q = json.load(open(QUEUE_PATH, encoding="utf-8"))
        excl["queue"] = set(q.keys())
    fz_path = DATA_DIR / "library_freeze.json"
    if fz_path.exists():
        fz = json.load(open(fz_path, encoding="utf-8"))
        frozen = fz.get("frozen", [])
        for x in frozen:
            if isinstance(x, str):
                excl["frozen"].add(x)
            elif isinstance(x, dict):
                excl["frozen"].add(x.get("name", ""))
    return excl


def triage(candidates, excl):
    """三级分诊: pandas 兼容(S5 可跑) / JQ 可翻译 / 双通道均可"""
    tri = {"s5_capable": [], "jq_capable": [], "skip_frozen": [], "skip_queue": [],
           "skip_formula": [], "skip_jq_translate": {}}
    for c in candidates:
        name = c["factor_name"]
        if name in excl["frozen"]:
            tri["skip_frozen"].append(name)
            continue
        if name in excl["queue"]:
            tri["skip_queue"].append(name)
            continue
        if not c["formula"] or str(c["formula"]).startswith("# TODO"):
            tri["skip_formula"].append(name)
            continue
        if not _is_pandas_compatible(c["formula"]):
            tri["skip_formula"].append(name)  # forge 风格/虚构列 → S5 复刻不可跑
            continue
        tri["s5_capable"].append(c)
        m = prepare_factor(name, c["formula"])
        if m["ok"]:
            tri["jq_capable"].append(c)
        else:
            reason = m.get("reason", "?")[:60]
            tri["skip_jq_translate"][name] = reason
    return tri


def run_s135_validation(candidates):
    """复刻 s135_candidate_check 主流程: S2 相关性 + S5JointFilter + MultiStageValidator"""
    from factor_ic_computer import FactorICComputer
    from multi_stage_validator import MultiStageValidator

    ic_comp = FactorICComputer()
    lib_factors = load_lib_factors()
    print(f"[Bridge] FactorICComputer ready={ic_comp.is_ready} | 库因子 {len(lib_factors)} 个")

    # S2 预计算
    for cand in candidates:
        if not cand["formula"]:
            cand["max_corr"] = 0.0
            cand["max_corr_factor"] = ""
            continue
        mc, mf = ic_comp.compute_max_corr_vs_library(
            cand["formula"], lib_factors, candidate_name=cand["factor_name"])
        cand["max_corr"] = mc if mc >= 0 else 0.0
        cand["max_corr_factor"] = mf if mc >= 0 else ""

    # S5 真实回测
    s5_filter = None
    s5_light = False
    try:
        from s5_joint_filter import S5JointFilter
        sf = S5JointFilter(top_n=80)  # P-20260909-A: 默认全量池, FUXI_S5_UNIVERSE 可回退
        if sf.is_ready:
            s5_filter = sf
            print("[Bridge][S5] S5JointFilter 真实回测可用")
        else:
            print("[Bridge][S5] S5JointFilter 数据未就绪 → 轻量回退")
    except Exception as e:
        print(f"[Bridge][S5] S5JointFilter 不可用: {e} → 轻量回退")

    if s5_filter is None:
        s5_light = True
        for cand in candidates:
            if not cand["formula"]:
                continue
            yearly = ic_comp.compute_yearly_ic(cand["formula"], cand["factor_name"])
            if yearly:
                y25 = yearly.get("2025") or {}
                y26 = yearly.get("2026") or {}
                cand["excess_2025"] = abs(y25.get("ic", 0))
                cand["excess_2026"] = abs(y26.get("ic", 0))
                cand["calmar"] = abs(yearly.get("all", {}).get("icir", 0))

    validator = MultiStageValidator(data_dir=DATA_DIR)
    results, summary = validator.validate(
        candidates=candidates,
        library_signals=None,
        library_factors=lib_factors,
        s5_filter=s5_filter,
        s5_lightweight=s5_light,
    )
    return candidates, results, summary, s5_light


def codegen_and_enqueue(passed, max_codegen=15):
    """对 S5 通过且 JQ 可翻译的因子生成代码 + 登记 queue"""
    queue = json.load(open(QUEUE_PATH, encoding="utf-8")) if QUEUE_PATH.exists() else {}
    now = datetime.datetime.now().isoformat(timespec="seconds")
    generated, skipped = [], []
    for c in passed:
        name = c["factor_name"]
        if len(generated) >= max_codegen:
            skipped.append({"name": name, "reason": "超过本批上限 %d" % max_codegen})
            continue
        out_path = FA_DIR / f"jq_s5_pass_{name}_standalone.py"
        if out_path.exists():
            skipped.append({"name": name, "reason": "JQ 文件已存在 (幂等跳过)"})
            if name not in queue:
                queue[name] = {"status": "pending_jq_run", "generated_at": now,
                               "source": "pass_to_jq_bridge"}
            continue
        m = prepare_factor(name, c["formula"])
        if not m["ok"]:
            skipped.append({"name": name, "reason": "翻译失败: " + m.get("reason", "?")[:50]})
            continue
        path, meta = generate_standalone(
            name, c["formula"],
            meaning=c.get("hypothesis", "")[:200] or "(池内 PASS 因子, 桥接自动生成)",
            local_note="P-20260908-006 桥接: Stage2 PASS → S5 验证通过 → 自动 codegen",
        )
        if path:
            generated.append({"name": name, "file": str(path),
                              "lookback": meta.get("lookback"),
                              "fields": meta.get("fields")})
            queue[name] = {
                "status": "pending_jq_run",
                "generated_at": now,
                "source": "pass_to_jq_bridge",
                "formula": c["formula"][:200],
                "icir": c.get("icir"),
                "s5_note": "池内 candidate 经 s135 复刻 S1-S5 全链验证 eligible",
            }
        else:
            skipped.append({"name": name, "reason": "生成失败: " + meta.get("reason", "?")[:50]})
    with open(QUEUE_PATH, "w", encoding="utf-8") as f:
        json.dump(queue, f, ensure_ascii=False, indent=1)
    return generated, skipped


def main():
    ap = argparse.ArgumentParser(description="PASS→S5/JQ 自动桥接 (P-20260908-006)")
    ap.add_argument("--execute", action="store_true",
                    help="执行桥接 (默认仅盘点); env FUXI_PASS_JQ_BRIDGE=on 等价")
    ap.add_argument("--max-codegen", type=int, default=15, help="本批 JQ 代码生成上限")
    args = ap.parse_args()

    execute = args.execute or os.environ.get("FUXI_PASS_JQ_BRIDGE") == "on"
    mode = "桥接" if execute else "盘点"
    print(f"[Bridge] P-20260908-006 PASS→S5/JQ 桥接 | 模式: {mode}")

    candidates = load_pool_candidates()
    excl = load_exclusions()
    tri = triage(candidates, excl)

    print(f"[Bridge] 池内 candidate {len(candidates)} | 冷冻跳过 {len(tri['skip_frozen'])} "
          f"| 已在队列 {len(tri['skip_queue'])} | 公式不可跑 {len(tri['skip_formula'])} "
          f"| S5 可跑 {len(tri['s5_capable'])} | 其中 JQ 可翻译 {len(tri['jq_capable'])}")
    if tri["skip_jq_translate"]:
        print("[Bridge] JQ 翻译不可行 (S5 仍验证, 只出报告):")
        for n, r in tri["skip_jq_translate"].items():
            print(f"    - {n[:40]}: {r}")
    # ICIR 降序展示 S5 可跑候选
    s5_sorted = sorted(tri["s5_capable"], key=lambda c: -c.get("icir", 0))
    print(f"[Bridge] S5 待验候选 (ICIR 降序):")
    for c in s5_sorted[:20]:
        print(f"    {c['factor_name'][:42]:42} ICIR={c['icir']:.3f} IC={c['ic']:+.4f}")

    if not execute:
        print("[Bridge] 盘点模式结束 (未写任何文件; --execute 或 FUXI_PASS_JQ_BRIDGE=on 执行)")
        return

    if not tri["s5_capable"]:
        print("[Bridge] 无 S5 可跑候选, 结束")
        return

    # S1-S5 复刻验证
    tested, results, summary, s5_light = run_s135_validation(tri["s5_capable"])

    eligible = []
    detail = []
    for c, r in zip(tested, results):
        rec = {
            "factor_name": c["factor_name"], "icir": c.get("icir"),
            "s1": r.s1_passed, "s2": r.s2_passed, "s3": r.s3_passed,
            "s4": r.s4_passed, "s5": r.s5_passed,
            "eligible": bool(r.eligible_for_jq),
            "s5_reason": (r.s5_reason or "") if not r.s5_passed else "",
        }
        detail.append(rec)
        # 🔴 codegen 判定收紧 (2026-09-09 回滚教训): 必须 s5_passed。
        # validator 真实 S5 模式下 eligible=4/5 历史口径 (S5 未过也 eligible),
        # 但 bridge 的意义是把池内因子接进与 Ralph 同标准的 S5 组合超额终裁,
        # 不容绕过。实证: 17 candidate 真实 S5 全拒 (excess_25/26 全负),
        # 旧判定生成了 8 条 S5=0 代码 → 已全部回滚。
        if r.s5_passed and r.eligible_for_jq:
            eligible.append(c)
        print(f"  {c['factor_name'][:40]:40} "
              f"{'✅' if r.eligible_for_jq else '❌'} "
              f"S1{int(r.s1_passed)} S2{int(r.s2_passed)} S3{int(r.s3_passed)} "
              f"S4{int(r.s4_passed)} S5{int(r.s5_passed)}"
              + (f" | {r.s5_reason}" if not r.s5_passed and r.s1_passed and r.s2_passed else ""))

    # JQ codegen (仅 eligible 且可翻译)
    gen, skip = codegen_and_enqueue(eligible, max_codegen=args.max_codegen)

    # 落盘
    out = {
        "run_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "mode": "execute",
        "pool_candidates": len(candidates),
        "s5_tested": len(tested),
        "s5_mode": "real" if not s5_light else "lightweight",
        "eligible": len(eligible),
        "generated": gen,
        "skipped_codegen": skip,
        "detail": detail,
    }
    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

    print(f"\n[Bridge] 漏斗: candidate {len(candidates)} → S5 可跑 {len(tested)} "
          f"→ S5 通过 {len(eligible)} → JQ 生成 {len(gen)} (上限 {args.max_codegen})")
    if s5_light:
        print("[Bridge] ⚠️ 本轮为轻量 S5 (IC 年拆分代理), 生成代码建议谨慎对待")
    for g in gen:
        print(f"  [OK] {g['name']} (lookback={g['lookback']}, fields={','.join(g['fields'])})")
    for s in skip:
        print(f"  [SKIP] {s['name']}: {s['reason']}")
    print(f"[Bridge] 结果落盘: {RESULT_PATH}")
    print("[Bridge] ⚠️ pending_jq_run 文件需手动复制到聚宽研究环境回测, "
          "结果反馈后触发 D+ 蒸馏")


if __name__ == "__main__":
    main()
