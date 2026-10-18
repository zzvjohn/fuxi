"""P-20260915-015 (v0.7.20): 周频通道裁决干净源复评。

背景 (2026-09-15 v0.7.17 实证):
  weekly_prices.parquet / weekly_ohlc.parquet 曾冻结在 2026-06-21 且 vol/amount
  字段失真 (55.2% 行偏低 >10%), 污染三处: ① WeeklyLane 周频裁决 ② τ_w 校准集
  ③ Forge 周频输入。v0.7.17 已重建缓存 (291 周 → 09-13, 与日频 sum 比值 1.0000)。
  本脚本在**干净缓存**上重放历史周频裁决, 产出「判反表」+ τ_w 双口径报告。

复评宇宙 (两条):
  A) 轨迹库: data/trajectory_log.json 的 _eval_log 中走过 weekly lane 的评估记录
     (标记: rejection_reason 或 stage 含「周频/weekly」), 窗口 >= --since (默认 06-21)
  B) 因子池: data/unified_factor_pool.csv 中 weekly_icir 非空的行

复评手段: weekly_lane.get_weekly_judge().judge() — 与流水线完全同口径
  (icir>=τ_w 且 |mean_ic|>=ic_min 且活性门), 缓存即当前干净源。
τ_w 报告: lane_calibration.recalibrate_tau_w(dry=True) — 只读, 不写 tau_w_effective。

用法:
  python scripts/weekly_verdict_reeval.py                # 全量复评
  python scripts/weekly_verdict_reeval.py --limit 10     # 限流冒烟
  python scripts/weekly_verdict_reeval.py --since 2026-08-01
输出:
  reports/weekly_verdict_reeval_YYYYMMDD.md
  data/weekly_verdict_reeval.json
"""
import argparse
import importlib.util
import json
import sys
import time
from collections import OrderedDict
from pathlib import Path

_PROJECT = Path(__file__).resolve().parent.parent
_ALCHEMY = _PROJECT / "research" / "factor_alchemy"

# sys.path 布置 (复刻 scripts/style_exposure_guard.py 的防同名坑模式:
# 核心模块用 importlib 绝对路径加载注册独立名, 不依赖 sys.path 顺序)
for _p in (str(_ALCHEMY), str(_PROJECT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _load_core(rel: str, alias: str):
    path = _ALCHEMY / rel
    if alias in sys.modules:
        return sys.modules[alias]
    spec = importlib.util.spec_from_file_location(alias, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[alias] = mod
    spec.loader.exec_module(mod)
    return mod


# ── 复评宇宙 A: 轨迹库 _eval_log ──────────────────────────────
def collect_from_trajectories(since: str):
    """返回 [{name, formula, old_passed, old_reason, first_seen, n_seen}] (按公式去重)。"""
    traj_path = _PROJECT / "data" / "trajectory_log.json"
    items = []
    if traj_path.exists():
        data = json.loads(traj_path.read_text(encoding="utf-8", errors="replace"))
        items = data.get("trajectories") or []
    uniq = OrderedDict()  # formula_key -> record
    n_raw = 0
    for it in items:
        created = str(it.get("created_at") or "")[:10]
        if created and created < since:
            continue
        for e in (it.get("_eval_log") or []):
            reason = str(e.get("rejection_reason") or "")
            stage = str(e.get("stage") or "")
            blob = (reason + " " + stage)
            if ("周频" not in blob) and ("weekly" not in blob.lower()):
                continue
            n_raw += 1
            formula = str(e.get("formula") or "").strip()
            if not formula:
                continue
            key = formula
            rec = {
                "name": str(e.get("factor_name") or "?"),
                "formula": formula,
                "old_passed": bool(e.get("passed")),
                "old_reason": reason[:120],
                "old_icir": e.get("icir"),
                "first_seen": str(e.get("timestamp") or created),
            }
            if key not in uniq:
                uniq[key] = rec
            else:
                # 同公式多次出现: 保守取历史任一次 passed=True 为老判定 (放行过)
                if rec["old_passed"]:
                    uniq[key]["old_passed"] = True
                uniq[key].setdefault("n_seen", 0)
                uniq[key]["n_seen"] = uniq[key].get("n_seen", 1) + 1
    out = list(uniq.values())
    for r in out:
        r.setdefault("n_seen", 1)
    return out, n_raw


# ── 复评宇宙 B: 因子池 weekly 行 ──────────────────────────────
def collect_from_pool():
    """unified_factor_pool.csv 中 weekly_icir 非空且有公式的行。"""
    import pandas as pd
    pool_path = _PROJECT / "data" / "unified_factor_pool.csv"
    if not pool_path.exists():
        return []
    df = pd.read_csv(pool_path, low_memory=False)
    if "weekly_icir" not in df.columns:
        return []
    m = df["weekly_icir"].notna() & df.get("formula", pd.Series(dtype=str)).notna()
    sub = df[m]
    out = []
    for _, row in sub.iterrows():
        out.append({
            "name": str(row.get("name", "?")),
            "formula": str(row.get("formula", "")).strip(),
            "old_passed": None,  # 池行是登记态, 无 pass/fail 语义
            "old_reason": f"pool weekly_icir={row.get('weekly_icir')}",
            "old_icir": (None if pd.isna(row.get("weekly_icir")) else float(row["weekly_icir"])),
            "first_seen": str(row.get("date", "")),
        })
    return out


# ── 复评执行 ──────────────────────────────────────────────────
def reevaluate(records, limit=None):
    wl = _load_core("weekly_lane.py", "_weekly_lane_core")
    judge = wl.get_weekly_judge(force_reload=True)
    if judge is None:
        raise RuntimeError("weekly judge 构造失败 (V07_DUAL_LANE 关闭?) — 复评无意义")
    rows = []
    for i, r in enumerate(records):
        if limit and i >= limit:
            break
        f = r["formula"]
        try:
            res = judge.judge(f, r["name"])
        except Exception as ex:  # noqa: BLE001
            rows.append({**r, "new_eval_ok": False, "new_reason": f"{type(ex).__name__}: {ex}"[:100],
                         "new_icir": None, "new_passed": None})
            continue
        if not res.get("eval_ok"):
            rows.append({**r, "new_eval_ok": False, "new_reason": res.get("reason", "")[:100],
                         "new_icir": None, "new_passed": None})
            continue
        rows.append({**r, "new_eval_ok": True,
                     "new_ic": res.get("weekly_ic"),
                     "new_icir": res.get("weekly_icir"),
                     "new_icir_recent": res.get("weekly_icir_recent"),
                     "new_n": res.get("weekly_n"),
                     "new_activity_ok": res.get("activity_ok"),
                     "new_passed": bool(res.get("passed")),
                     "new_reason": res.get("reason", "")[:120]})
    return rows


# ── τ_w 双口径 ────────────────────────────────────────────────
def tau_w_report():
    lc = _load_core("lane_calibration.py", "_lane_calibration_core")
    cal = lc.load_calibration()
    dry = lc.recalibrate_tau_w(dry=True)
    return {
        "current_tau_w": lc._config_tau_w(),
        "effective_tau_w": lc.get_effective_tau_w(),
        "calib_points": cal.get("n_points", 0),
        "tau_w_effective": cal.get("tau_w_effective"),
        "dry_recommend": dry.get("recommended"),
        "dry_reason": (dry.get("reason") or "")[:160],
        "votes": cal.get("tau_w_votes", []),
    }


# ── 报告 ──────────────────────────────────────────────────────
def build_report(rows, tw, since):
    today = time.strftime("%Y-%m-%d")
    ok = [r for r in rows if r["new_eval_ok"]]
    fail = [r for r in rows if not r["new_eval_ok"]]
    # 判反表: 轨迹宇宙才有 old_passed 语义
    judged = [r for r in ok if r.get("old_passed") is not None]
    resurrect = [r for r in judged if (not r["old_passed"]) and r["new_passed"]]   # 原拒→今过 (误杀)
    invalidated = [r for r in judged if r["old_passed"] and (not r["new_passed"])]  # 原过→今拒 (误放)
    confirmed = [r for r in judged if r["old_passed"] == r["new_passed"]]

    lines = [f"# 周频通道裁决干净源复评 ({today})", "",
             f"- 复评窗口: >= {since} ｜ 缓存: 干净源 (v0.7.17 重建, 至 2026-09-13)",
             f"- τ_w: config={tw['current_tau_w']} ｜ effective={tw['effective_tau_w']} ｜ "
             f"校准点 n={tw['calib_points']} ｜ dry 推荐={tw['dry_recommend']}",
             f"  - dry 理由: {tw['dry_reason'] or '—'}",
             "",
             "## 漏斗", "",
             f"| 宇宙 | 记录数 | 新口径 eval 成功 | eval 失败 |", "|---|---|---|---|",
             f"| 轨迹库+因子池 合计 | {len(rows)} | {len(ok)} | {len(fail)} |", "",
             f"## 判反表 (仅轨迹宇宙 n={len(judged)})", "",
             f"- 🟢 **原拒→今过 (误杀, 候选池可复活)**: {len(resurrect)}",
             f"- 🔴 **原过→今拒 (误放, 需清库决策)**: {len(invalidated)}",
             f"- ⚪ 判定一致: {len(confirmed)}", ""]
    if resurrect:
        lines += ["### 误杀明细 (top 15)", "",
                  "| 名称 | 新ICIR | 新n | 老理由 |", "|---|---|---|---|"]
        for r in resurrect[:15]:
            lines.append(f"| {r['name'][:40]} | {r['new_icir']} | {r.get('new_n','')} | {r['old_reason'][:60]} |")
        lines.append("")
    if invalidated:
        lines += ["### 误放明细 (top 15)", "",
                  "| 名称 | 新ICIR | 新n | 老理由 |", "|---|---|---|---|"]
        for r in invalidated[:15]:
            lines.append(f"| {r['name'][:40]} | {r['new_icir']} | {r.get('new_n','')} | {r['old_reason'][:60]} |")
        lines.append("")
    lines += ["## eval 失败原因分布 (无法复评, 非因子无效)", ""]
    from collections import Counter
    _fc = Counter(str(r.get("new_reason", ""))[:46] for r in fail)
    for k, v in _fc.most_common(6):
        lines.append(f"- {v} 条: {k}")
    if any("SyntaxError" in k for k in _fc):
        lines.append("  > 注: SyntaxError 主因是轨迹日志落盘时公式被截断 (历史 _eval_log "
                     "截断长度), 该部分无法重放, 属数据限制而非复评器缺陷。")
    lines += ["", "## 因子池 weekly 行复评", ""]
    pool_rows = [r for r in ok if str(r.get("old_reason", "")).startswith("pool")]
    lines.append(f"- 池内 weekly_icir 非空且复评成功: {len(pool_rows)}")
    drift = [r for r in pool_rows if r.get("old_icir") is not None and r.get("new_icir") is not None
             and abs(r["old_icir"] - r["new_icir"]) > 0.10]
    lines.append(f"- 新旧 ICIR 偏差 >0.10 (缓存污染实锤): {len(drift)} 条")
    if drift[:10]:
        lines += ["", "| 名称 | 老ICIR | 新ICIR | Δ |", "|---|---|---|---|"]
        for r in drift[:10]:
            lines.append(f"| {r['name'][:40]} | {r['old_icir']:.3f} | {r['new_icir']:.3f} | "
                         f"{r['new_icir'] - r['old_icir']:+.3f} |")
    lines += ["", "---", "",
              "> 判定口径: `judge()` 与流水线完全一致 (icir>=τ_w 且 |mean_ic|>=ic_min 且活性门)。",
              "> τ_w 为 dry 只读报告, 未写 tau_w_effective (hysteresis 规则不变)。"]
    return "\n".join(lines), {"resurrect": len(resurrect), "invalidated": len(invalidated),
                              "confirmed": len(confirmed), "pool_drift": len(drift)}


def main() -> int:
    ap = argparse.ArgumentParser(description="周频通道裁决干净源复评 (P-20260915-015)")
    ap.add_argument("--since", default="2026-06-21", help="轨迹复评窗口起点 YYYY-MM-DD")
    ap.add_argument("--limit", type=int, default=None, help="限流 (冒烟用)")
    ap.add_argument("--no-pool", action="store_true", help="跳过因子池宇宙")
    args = ap.parse_args()

    t0 = time.time()
    traj_recs, n_raw = collect_from_trajectories(args.since)
    print(f"[collect] 轨迹库: 原始 weekly 记录 {n_raw} 条 → 去重公式 {len(traj_recs)} 个")
    pool_recs = [] if args.no_pool else collect_from_pool()
    print(f"[collect] 因子池 weekly 行: {len(pool_recs)} 条")
    records = traj_recs + pool_recs
    if not records:
        print("[collect] 复评宇宙为空 — 无事可做")
        return 1

    print(f"[re-eval] 复评 {min(len(records), args.limit or len(records))} 条 (同口径 judge, 干净缓存)...")
    rows = reevaluate(records, limit=args.limit)

    tw = tau_w_report()
    print(f"[tau_w] config={tw['current_tau_w']} effective={tw['effective_tau_w']} "
          f"points={tw['calib_points']} dry_rec={tw['dry_recommend']}")

    report_md, summary = build_report(rows, tw, args.since)
    stamp = time.strftime("%Y%m%d")
    out_md = _PROJECT / "reports" / f"weekly_verdict_reeval_{stamp}.md"
    out_json = _PROJECT / "data" / "weekly_verdict_reeval.json"
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(report_md, encoding="utf-8")
    out_json.write_text(json.dumps({
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "since": args.since,
        "tau_w": tw,
        "summary": summary,
        "rows": rows,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[done] {summary} ｜ 用时 {time.time() - t0:.0f}s")
    print(f"[done] 报告: {out_md}")
    print(f"[done] 明细: {out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
