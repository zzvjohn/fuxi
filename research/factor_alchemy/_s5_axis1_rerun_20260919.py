# -*- coding: utf-8 -*-
"""2026-09-19 送 S5 全量池裁决: 31 条 axis=1 因子 (首次真正评估)
================================================================================
为何是「首次」: 9-18 审计确认 `_needs_cross_section()` 用字面子串白名单 ⇒
  写法变体被漏判 ⇒ 候选落到 `_evaluate_per_stock` (逐股 Series) ⇒
  `Series.rank(axis=1)` ValueError ⇒ L1594 except 静默 return None ⇒ **从未评估**。
  ⇒ 修复后的 `needs=True` 已由 `_axis1_preflight_20260919.py` 逐条实测 (31/31)。

与 s5_fullpool_rerun.py 的关系:
  * **复用** `apply_caliber` / `CALIBERS` / `write_attempt` (口径语义必须逐字一致);
  * **另存**结果文件 —— 🔴 绝不写 `s5_fullpool_rerun_result*.json` (那是 60 条 A/B
    基线, 覆写即毁掉零回归对照);
  * 候选源 = `data/_axis1_31_candidates_20260919.json` (源码常量库导出)。

用法:
    python _s5_axis1_rerun_20260919.py --caliber default
    python _s5_axis1_rerun_20260919.py --caliber legacy --limit 5   # 廉价守卫
"""
import argparse
import io
import json
import os
import re
import sys
import time
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

BASE = Path(r'e:/quant')
sys.path.insert(0, str(BASE / "scripts"))
sys.path.insert(0, str(BASE / "research" / "factor_alchemy"))

DATA_DIR = BASE / "data"
CAND_JSON = DATA_DIR / "_axis1_31_candidates_20260919.json"

import s5_fullpool_rerun as R  # noqa: E402
from s5_fullpool_rerun import apply_caliber, write_attempt, CALIBERS  # noqa: E402


def out_path(caliber, idx_align, limit=0):
    sfx = "" if idx_align in ("off", "default") else "__ia" + idx_align
    lim = "" if not limit else "__limit%d" % limit
    return DATA_DIR / f"s5_axis1_rerun_result_{caliber}{sfx}{lim}.json"


# ── 结构退化静态预筛 (2026-09-19 新增) ────────────────────────────────────────
# 锚在「实现」而非「命名」: 检测 `{...: EXPR for <v> in <expr>.columns}` 且 <v>
# 在**值表达式**中从未被引用 ⇒ 每列算出同一条值 ⇒ 跨截面零离散 (必然常数因子),
# 且代价是 O(列数) 次全截面重算。实测 novel_041 `dispersion_anomaly` 以此为算力
# 黑洞 (单条 >25min 未收敛, 而结论必然是常数拦截) ⇒ 未评估即为正确处置, 不记 FAIL。
# 🔴 键必须剥离: `{c: ... for c in X.columns}` 中 `c` 仅作列名, 不构成逐列语义
#    (v1 版因未剥离键而漏检该条, 属「判据锚错位置」的同族复现)。
_COMP_REs = [
    re.compile(r"\{([^{}]*?)\bfor\s+(\w+)\s+in\s+[A-Za-z_]\w*\.columns\b", re.S),
    re.compile(r"\[([^\[\]]*?)\bfor\s+(\w+)\s+in\s+[A-Za-z_]\w*\.columns\b", re.S),
    re.compile(r"\(([^()]*?)\bfor\s+(\w+)\s+in\s+[A-Za-z_]\w*\.columns\b", re.S),
]


def degenerate_col_loop(formula):
    """返回未被引用的循环变量名; 非退化返回 None。"""
    for rx in _COMP_REs:
        m = rx.search(formula or "")
        if not m:
            continue
        body, var = m.group(1), m.group(2)
        # 字典推导: 元组 (键, 值) 间的顶层冒号 ⇒ 只对「值」判定
        depth, cut = 0, None
        for i, ch in enumerate(body):
            if ch in "([{":
                depth += 1
            elif ch in ")]}":
                depth -= 1
            elif ch == ":" and depth == 0:
                cut = i
                break
        value = body[cut + 1:] if cut is not None else body
        if re.search(r"(?<![\w.])%s(?![\w])" % re.escape(var), value):
            return None            # 循环体确实用到 ⇒ 正常逐列写法
        return var
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--caliber", choices=sorted(CALIBERS.keys()), default="default")
    ap.add_argument("--idx-align", choices=["default", "off", "on"], default="default")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 条 (按清单序)")
    ap.add_argument("--skip", action="append", default=[],
                    help="按 factor_name 跳过 (可重复); 记 not_evaluated, 不记为 FAIL")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    applied = apply_caliber(args.caliber)
    _CODE_FP = R.code_fingerprint()
    print("[CODE_FP] %s" % json.dumps(_CODE_FP, ensure_ascii=False))
    if args.idx_align != "default":
        os.environ["FUXI_S5_IDX_ALIGN"] = args.idx_align
    _ia_fp = (os.environ.get("FUXI_S5_IDX_ALIGN", "off") or "off").strip().lower()
    if _ia_fp not in ("on", "off"):
        _ia_fp = "off"
    _ceil_fp = (os.environ.get("FUXI_S5_DATE_CEIL", "auto") or "auto").strip()
    _fp = {"caliber": args.caliber, "date_ceil": _ceil_fp, "idx_align": _ia_fp}
    op = out_path(args.caliber, _ia_fp, args.limit)

    # ── 候选载入 + 行数对账 (铁律 #2: 载入后必做对账) ──
    raw = json.loads(CAND_JSON.read_text(encoding="utf-8"))
    cands = raw["candidates"]
    n_file = len(cands)
    print(f"[LOAD-CAND] {CAND_JSON.name}: n_axis1={n_file} "
          f"n_needs_true={raw['n_needs_true']} gate_ok={raw['gate_ok']}")
    assert n_file == raw["n_axis1"], "候选数对账失败"
    assert raw["gate_ok"], "前置路由自证未通过 ⇒ 拒绝送检 (路由未生效则读数无意义)"
    print(f"[CALIBER] {args.caliber} | idx_align={_ia_fp} | date_ceil={_ceil_fp} "
          f"| env={applied if applied else '(代码默认)'} | out={op.name}")

    if args.limit and args.limit > 0:
        cands = cands[:args.limit]

    if args.dry_run:
        for i, c in enumerate(cands, 1):
            print(f"  {i:>2}. {c['factor_name'][:50]:<50} len={c['len']}")
        return

    # ── 断点续跑 (指纹含 caliber + date_ceil + idx_align; 铁律: 窗口必须入指纹) ──
    done = {}
    if args.limit and args.limit > 0:
        print("[RESUME] --limit>0 ⇒ 禁用续跑")
    elif op.exists():
        try:
            prev = json.loads(op.read_text(encoding="utf-8"))
            pfp = {"caliber": prev.get("caliber"), "date_ceil": prev.get("date_ceil"),
                   "idx_align": prev.get("idx_align")}
            if pfp == _fp:
                done = {r["factor_name"]: r for r in prev.get("results", [])}
                print(f"[RESUME] 指纹相符 ⇒ 装载已完成 {len(done)} 条")
            else:
                print(f"[RESUME] 🔴 指纹不符 旧={pfp} 新={_fp} ⇒ 从零开始")
        except Exception as e:
            print(f"[RESUME] 读取失败, 从零开始: {type(e).__name__}: {e}")

    _run_id = "axis1__%s__%s__ceil%s__ia%s" % (
        time.strftime("%Y%m%d_%H%M%S"), args.caliber,
        re.sub(r"[^0-9A-Za-z]+", "", _ceil_fp or "auto"), _ia_fp)
    print(f"[ATTEMPT] run_id={_run_id}")

    os.environ.pop("FUXI_S5_UNIVERSE", None)
    from s5_joint_filter import S5JointFilter
    s5 = S5JointFilter(top_n=80)
    print(f"[CFG] sample_stocks={s5.sample_stocks} index={s5.index_path.name}")

    t0 = time.time()
    s5._load_data()
    print(f"[LOAD] {time.time()-t0:.0f}s universe={len(s5._universe)} "
          f"rows={len(s5._price_df)}")
    # 铁律 #3(技能 §5): 窗口落点必须落盘可查
    try:
        pmax = str(s5._price_df["trade_date"].max())[:10]
        imax = (str(s5._index_df["trade_date"].max())[:10]
                if getattr(s5, "_index_df", None) is not None else "n/a")
        print(f"[WINDOW] price_end={pmax} index_end={imax} date_ceil_env={_ceil_fp}")
    except Exception as e:
        print(f"[WINDOW] 取末端失败: {type(e).__name__}: {e}")

    results = []
    n_pass = 0
    n_ran = 0
    n_not_eval = 0
    attempt_rows = []
    for i, c in enumerate(cands):
        name = c["factor_name"]
        if name in done:
            rec = done[name]
            results.append(rec)
            if rec.get("new_passed"):
                n_pass += 1
            print(f"  [{i+1}/{len(cands)}] SKIP(resume) {name[:46]}")
            continue
        # ── 结构退化 / 手工排除 ⇒ 记 not_evaluated (非 FAIL, 不入比较分母) ──
        _var = degenerate_col_loop(c["formula"])
        _blk = "degenerate_col_loop" if _var else (
            "manual_skip" if name in set(args.skip or []) else None)
        if _blk:
            rec = {
                "factor_name": name, "novel_id": c.get("novel_id"),
                "formula_len": c.get("len"),
                "has_rank_ax1": c.get("has_rank_ax1"),
                "axis1_calls": c.get("axis1_calls"),
                "needs_cross_section": c.get("needs_cross_section"),
                "formula": c["formula"][:600],
                "eval_status": "not_evaluated", "eval_block": _blk,
                "eval_block_proof": {
                    "unused_loop_var": _var,
                    "reason": ("循环变量未在循环体中被引用 ⇒ 所有列算出同一条值 ⇒ "
                               "跨截面零离散 (必然常数), 且单条代价为 O(列数) 次"
                               "全截面重算 ⇒ 送检无信息增量"
                               if _var else "人工排除"),
                },
                "new_passed": None,
            }
            results.append(rec)
            n_not_eval += 1
            try:
                with open(op, "w", encoding="utf-8") as fh:
                    json.dump({
                        "run_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                        "caliber": args.caliber, "caliber_env": applied,
                        "date_ceil": _ceil_fp, "idx_align": _ia_fp,
                        "limit": args.limit,
                        "universe": "full_pool_5763", "benchmark": "000985",
                        "candidate_source": "NOVEL_FACTOR_LIBRARY axis=1 (31)",
                        "progress": f"{i+1}/{len(cands)}",
                        "code_fingerprint": _CODE_FP, "attempt_run_id": _run_id,
                        "results": results,
                    }, fh, ensure_ascii=False, indent=1, default=R._json_safe)
            except Exception as we:
                print(f"    [WARN] incremental save failed: {we}")
            print(f"  [{i+1}/{len(cands)}] SKIP({_blk}) {name[:40]} "
                  f"unused_loop_var={_var}")
            continue
        t1 = time.time()
        try:
            r = s5.validate_factor(name, c["formula"])
            _sd = getattr(r, "sparse_diag", None) or {}
            rec = {
                "factor_name": name,
                "novel_id": c.get("novel_id"),
                "formula_len": c.get("len"),
                "has_rank_ax1": c.get("has_rank_ax1"),
                "axis1_calls": c.get("axis1_calls"),
                "needs_cross_section": c.get("needs_cross_section"),
                "formula": c["formula"][:600],
                "new_passed": r.passed,
                "new_excess_25": round(r.excess_2025, 4),
                "new_excess_26": round(r.excess_2026, 4),
                "new_calmar_25": round(r.calmar_2025, 3),
                "new_calmar_26": round(r.calmar_2026, 3),
                "new_reason": (r.reason or "")[:400],
                # 结构化字段 (S5 技能 §8b: 披露必须走结构化, 不靠字符串尾部)
                "sparse_flags": list(_sd.get("flags") or []),
                "sparse_argmax_recent": _sd.get("decile_argmax_recent"),
                "sparse_monotone_recent": _sd.get("decile_monotone_pairs_recent"),
                "sparse_sel_zero_frac_recent": _sd.get("sel_zero_frac_mean_recent"),
                "sparse_n_pos_median_recent":
                    (_sd.get("n_pos_recent") or {}).get("median"),
                "sparse_pct_pos_below_topn_recent":
                    _sd.get("pct_periods_pos_below_topn_recent"),
                "sparse_warning": _sd.get("warning"),
            }
        except Exception as e:
            rec = {
                "factor_name": name, "novel_id": c.get("novel_id"),
                "formula": c["formula"][:600],
                "new_passed": False, "error": f"{type(e).__name__}: {e}"[:200],
            }
        results.append(rec)
        n_ran += 1
        if rec.get("new_passed"):
            n_pass += 1
        try:
            ap_ = write_attempt(_run_id, i + 1, name, rec, c["formula"],
                                args.caliber, _ceil_fp, _ia_fp)
        except Exception as e:
            ap_ = "n/a(%s)" % type(e).__name__
        attempt_rows.append({"seq": i + 1, "factor_name": name,
                             "path": ap_, "passed": bool(rec.get("new_passed"))})
        try:
            with open(op, "w", encoding="utf-8") as fh:
                json.dump({
                    "run_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "caliber": args.caliber, "caliber_env": applied,
                    "date_ceil": _ceil_fp, "idx_align": _ia_fp,
                    "limit": args.limit,
                    "universe": "full_pool_5763", "benchmark": "000985",
                    "candidate_source": "NOVEL_FACTOR_LIBRARY axis=1 (31)",
                    "progress": f"{i+1}/{len(cands)}",
                    "code_fingerprint": _CODE_FP, "attempt_run_id": _run_id,
                    "results": results,
                }, fh, ensure_ascii=False, indent=1, default=R._json_safe)
        except Exception as we:
            print(f"    [WARN] incremental save failed: {we}")
        dt = time.time() - t1
        tag = "PASS" if rec.get("new_passed") else ("ERR " if rec.get("error") else "rej ")
        print(f"  [{i+1}/{len(cands)}] {tag} {name[:46]:<46} "
              f"ex25={rec.get('new_excess_25')} ex26={rec.get('new_excess_26')} "
              f"({dt:.0f}s)")

    # ── 汇总 + 分母审计 (每个比较都断言比较数 > 0) ──
    n_err = sum(1 for r in results if r.get("error"))
    n_cmp = len(results) - n_err - n_not_eval
    print()
    print("=" * 74)
    print(f"[SUMMARY] caliber={args.caliber} ceil={_ceil_fp} ia={_ia_fp}")
    print(f"  候选 {len(cands)} | 本轮真跑 {n_ran} | 比较数 n_compared = {n_cmp} "
          f"| error {n_err} | 未评估 {n_not_eval}")
    assert n_cmp > 0, "🔴 n_compared == 0 ⇒ 本轮无任何有效比较, 不得产出裁决"
    for r in results:
        if r.get("eval_status") == "not_evaluated":
            print(f"     未评估 {r['factor_name'][:46]:<46} "
                  f"block={r.get('eval_block')} "
                  f"unused_loop_var={(r.get('eval_block_proof') or {}).get('unused_loop_var')}")
    print(f"  🔴 PASS = {n_pass}")
    for r in results:
        if r.get("new_passed"):
            print(f"     PASS  {r['factor_name'][:46]:<46} "
                  f"ex25={r.get('new_excess_25')} ex26={r.get('new_excess_26')} "
                  f"c25={r.get('new_calmar_25')} c26={r.get('new_calmar_26')}")
    zeros = [r for r in results
             if r.get("new_excess_25") == 0.0 and r.get("new_excess_26") == 0.0]
    print(f"  两腿超额同时恰为 0 的条目 = {len(zeros)}  (前置短路嫌疑, 须逐条归因)")
    for r in zeros:
        print(f"     - {r['factor_name'][:46]:<46} {str(r.get('new_reason'))[:70]}")
    print(f"  输出: {op}")
    print("=" * 74)


if __name__ == "__main__":
    main()
