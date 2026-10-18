# -*- coding: utf-8 -*-
"""Gate C 跨年稳健性只读诊断 (P-20260920-004)。

🔴 零裁决声明
    本模块 **不写任何 PASS/FAIL**, **不改 passed_factor_pool**, **不影响 ralph_loop /
    s5_joint_filter / daily_factor_hypothesis 的任何路径**。它只把「跨年稳健性」
    表达为可读的结构化读数。诊断结论仅用于观测与归因, 不构成任何门禁。

为什么需要它 (锚在实测, 非文档转述)
    P-20260920-002 审计识别 Gate C(稳健性) 无独立门: 伏羲只有 S5 的 cal25/cal26 两个点。
    而 2026-09-20 实测发现既有 compute_yearly_ic 存在结构性死角:
      - 它硬编码年份 [2025, 2026];
      - 唯一主链调用点在 ralph_loop.py L4101 的 **S5 不可用 fallback 分支**;
      - 其 FactorICComputer 为默认 lookback_days=270 => 数据窗口实测仅 182 个交易日
        (2025-12-22 ~ 2026-09-18), 「2025」腿只剩约 7 天;
      - 消费端 `float(y2025.get("ic", 0) or 0)` 把 None 静默记 0 => fail-open。
    故跨年诊断必须走 lookback_days=None (daily_prices.csv 全量 6 年 / 1386 交易日)。

口径 (与 S1 判决层对齐)
    forward_period=5 (fwd5) / min_stocks_per_day=30 / 逐交易日截面 Spearman Rank IC /
    单年有效 IC 天数 >= 5 才计该年。逐字复用 compute_yearly_ic_range (本批新增)。

开关
    FUXI_GATE_C_PROBE   默认 off。未开启时仅打印提示并退出 0 (不消耗算力)。
                        置 on 或传 --force 才执行完整诊断。

用法
    python scripts/gate_c_stability_probe.py --force
    python scripts/gate_c_stability_probe.py --force --max-factors 10
    python scripts/gate_c_stability_probe.py --force --factors cold_growth_rank_sum
    python scripts/gate_c_stability_probe.py --force --param-neighborhood
    python scripts/gate_c_stability_probe.py --force --emit-daily-line
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
FA_DIR = ROOT / "research" / "factor_alchemy"
sys.path.insert(0, str(FA_DIR))

POOL = ROOT / "data" / "passed_factor_pool.csv"
OUT = ROOT / "data" / "gate_c_stability_report.json"

# 公式内可调窗口参数的匹配 (pandas 风格 .rolling(N) / .shift(N) / .ewm(...) 等)
WIN_PAT = re.compile(r"\.(?:rolling|shift|diff|pct_change|ewm)\(\s*(\d+)\s*[,)]")

# 描述性分档 (advisory only —— 明确不参与任何裁决)
GRADE_CUT = {"A": 0.80, "B": 0.60}
# 有效年份数下限: 低于此值不给 A/B/C (分母缩水会造「假稳健」——
# 2026-09-20 全量实测: 14/52 条仅 2 年有效, 若照判则 13 条假 A)。
MIN_YEARS_FOR_GRADE = 4


def _env_flag_on() -> bool:
    return str(os.environ.get("FUXI_GATE_C_PROBE", "off")).strip().lower() in (
        "on", "1", "true", "enforce", "yes")


def load_candidates(limit: Optional[int] = None,
                    only: Optional[List[str]] = None) -> List[Dict]:
    """从 passed_factor_pool.csv 取 status=candidate 的因子 (有公式的)。"""
    raw = io.open(POOL, "r", encoding="utf-8-sig", newline="").read()
    rows = list(csv.DictReader(io.StringIO(raw)))
    cands = []
    for r in rows:
        st = str(r.get("status") or "").strip()
        f = (r.get("formula") or "").strip()
        if not f:
            continue
        # --factors 指定时不受 status 限制 (可诊断任意因子, 含已 s5_failed / jq_done);
        # 未指定时默认只取 candidate (待送 S5 的未裁决候选)。
        if only is None and st != "candidate":
            continue
        cands.append({"name": r.get("name"), "formula": f, "status": st})
    if only:
        want = set(only)
        cands = [c for c in cands if c["name"] in want]
    if limit:
        cands = cands[:limit]
    return cands


def aggregate_metrics(yearly: Dict, all_ic: Optional[Dict] = None) -> Dict:
    """把逐年读数汇总为跨年稳健性指标 (纯描述, 无裁决语义)。

    指标定义
      n_valid_years       有效年份数 (该年 IC 天数 >= 5)
      positive_years_ratio  IC>0 的年份数 / n_valid_years
      sign_consistency    max(正IC年数, 负IC年数) / n_valid_years  (符号一致度)
      icir_min/max/range  年际 ICIR 的最小/最大/极差
      worst_year / worst_year_ic   最差年份及其 IC
      ic_full / icir_full 全区间 IC / ICIR
    """
    ys = [(k, v) for k, v in yearly.items()
          if k.isdigit() and isinstance(v, dict)]
    out: Dict = {
        "n_valid_years": len(ys),
        "years": [int(k) for k, _ in ys],
    }
    if not ys:
        out["note"] = "no_valid_year"
        return out

    ics = [v["ic"] for _, v in ys]
    icirs = [v["icir"] for _, v in ys]
    pos = sum(1 for x in ics if x > 0)
    neg = sum(1 for x in ics if x < 0)
    worst_k, worst_v = min(ys, key=lambda kv: kv[1]["ic"])
    best_k, best_v = max(ys, key=lambda kv: kv[1]["ic"])

    out.update({
        "positive_years": pos,
        "positive_years_ratio": round(pos / len(ys), 4),
        "sign_consistency": round(max(pos, neg) / len(ys), 4),
        "icir_min": round(min(icirs), 4),
        "icir_max": round(max(icirs), 4),
        "icir_range": round(max(icirs) - min(icirs), 4),
        "ic_min_year": round(min(ics), 6),
        "ic_max_year": round(max(ics), 6),
        "worst_year": int(worst_k),
        "worst_year_ic": round(worst_v["ic"], 6),
        "best_year": int(best_k),
        "best_year_ic": round(best_v["ic"], 6),
        "ic_mean_across_years": round(sum(ics) / len(ics), 6),
    })
    if isinstance(all_ic, dict):
        out["ic_full"] = all_ic.get("ic")
        out["icir_full"] = all_ic.get("icir")
        out["n_days_full"] = all_ic.get("n_days")

    # advisory 分档 (仅描述, 不改任何裁决)
    # 🔴 分母守卫: 有效年份不足时**不给分档** —— 2 年「全正」既非稳健证据,
    #    其窗口也与 S1 判决的 270 天窗口高度重叠 (无跨年信息量)。
    if out["n_valid_years"] < MIN_YEARS_FOR_GRADE:
        out["advisory_grade"] = "insufficient_years"
        out["grade_reason"] = ("n_valid_years=%d < MIN_YEARS_FOR_GRADE=%d"
                               % (out["n_valid_years"], MIN_YEARS_FOR_GRADE))
        out["grade_kind"] = "advisory_readonly_not_a_verdict"
        return out
    r = out["positive_years_ratio"]
    if r >= GRADE_CUT["A"]:
        out["advisory_grade"] = "A_consistent"
    elif r >= GRADE_CUT["B"]:
        out["advisory_grade"] = "B_mixed"
    else:
        out["advisory_grade"] = "C_unstable"
    out["grade_kind"] = "advisory_readonly_not_a_verdict"
    return out


def param_variants(formula: str) -> Dict[str, str]:
    """从公式提取窗口参数, 生成 ±20% 变体公式 (每个唯一窗口值独立变体)。

    若无任何可调窗口参数 => 返回空 dict (由调用方记 null, **不臆造**参数)。
    """
    wins = sorted({int(m.group(1)) for m in WIN_PAT.finditer(formula)})
    vs: Dict[str, str] = {}
    for n in wins:
        for tag, mult in (("lo", 0.8), ("hi", 1.2)):
            nv = max(2, int(round(n * mult)))
            if nv == n:
                continue
            new_f = re.sub(
                r"\.(rolling|shift|diff|pct_change|ewm)\(\s*%d\s*([,)])" % n,
                lambda mm, _nv=nv: ".%s(%d%s" % (mm.group(1), _nv, mm.group(2)),
                formula)
            if new_f != formula:
                vs["w%d_%s_%d" % (n, tag, nv)] = new_f
    return vs


def main() -> int:
    ap = argparse.ArgumentParser(description="Gate C 跨年稳健性只读诊断 (P-20260920-004)")
    ap.add_argument("--force", action="store_true", help="忽略 FUXI_GATE_C_PROBE 开关")
    ap.add_argument("--factors", type=str, default="",
                    help="逗号分隔的因子名 (默认取 passed_factor_pool 的 candidate)")
    ap.add_argument("--max-factors", type=int, default=0, help="上限 (0=不限)")
    ap.add_argument("--param-neighborhood", action="store_true",
                    help="附加参数邻域平坦度 (窗口 ±20% 变体, 额外算力)")
    ap.add_argument("--emit-daily-line", action="store_true", help="只打印日报行")
    ap.add_argument("--out", type=str, default=str(OUT))
    args = ap.parse_args()

    if not args.force and not _env_flag_on():
        print("[GateC] FUXI_GATE_C_PROBE=off (默认) => 跳过诊断, 不消耗算力。")
        print("        开启方式: export FUXI_GATE_C_PROBE=on  或  加 --force")
        return 0

    t0 = time.time()
    only = [x.strip() for x in args.factors.split(",") if x.strip()] or None
    cands = load_candidates(limit=args.max_factors or None, only=only)
    if not cands:
        print("[GateC] 无可诊断因子 (candidate 池为空或 --factors 未命中)")
        return 0

    print("=" * 78)
    print("Gate C 跨年稳健性只读诊断 (P-20260920-004) —— 零裁决")
    print("=" * 78)
    print("诊断因子数: %d" % len(cands))
    for c in cands[:8]:
        print("   -", c["name"])
    if len(cands) > 8:
        print("   ... 其余 %d 条" % (len(cands) - 8))

    from factor_ic_computer import FactorICComputer

    print()
    print("[GateC] 加载全量价格 (lookback_days=None => 6 年跨年窗口)...")
    ic = FactorICComputer(lookback_days=None)
    ic._load_data()
    if ic._all_dates is not None and len(ic._all_dates) > 0:
        dr = [str(ic._all_dates.min().date()), str(ic._all_dates.max().date())]
        print("[GateC] 数据窗口: %s → %s | 交易日 %d"
              % (dr[0], dr[1], len(ic._all_dates)))
    else:
        dr = []

    results = []
    for i, c in enumerate(cands, 1):
        t1 = time.time()
        rec: Dict = {"name": c["name"], "formula": c["formula"],
                     "status_in_pool": c["status"]}
        try:
            yr = ic.compute_yearly_ic_range(c["formula"], c["name"])
            yearly = {k: v for k, v in yr.items() if k != "_meta"}
            rec["yearly"] = yearly
            rec["metrics"] = aggregate_metrics(
                {k: v for k, v in yearly.items() if k.isdigit()},
                yearly.get("all"))
            rec["meta"] = yr.get("_meta")
            if args.param_neighborhood:
                vs = param_variants(c["formula"])
                if vs:
                    base = (rec["metrics"].get("icir_full") or 0.0)
                    pv = {}
                    for tag, f2 in vs.items():
                        r2 = ic.compute(f2, "%s__%s" % (c["name"], tag))
                        pv[tag] = {"formula": f2, "icir": r2.get("icir"),
                                   "ic": r2.get("ic"), "n_days": r2.get("n_days")}
                    vals = [v["icir"] for v in pv.values() if v.get("icir") is not None]
                    allv = [base] + vals
                    rec["param_neighborhood"] = {
                        "variants": pv,
                        "icir_base": base,
                        "icir_spread": round(max(allv) - min(allv), 4) if allv else None,
                        "rel_spread": round((max(allv) - min(allv)) / (abs(base) + 1e-9), 4)
                                      if allv and base else None,
                    }
                else:
                    rec["param_neighborhood"] = None  # 无可调窗口 => 不臆造
        except Exception as e:  # noqa: BLE001
            rec["error"] = "%s: %s" % (type(e).__name__, e)
        rec["elapsed_s"] = round(time.time() - t1, 2)
        results.append(rec)
        m = rec.get("metrics") or {}
        print("[GateC] %2d/%d %-42s grade=%-13s pos_yrs=%s/%s icir_range=%s (%.1fs)"
              % (i, len(cands), c["name"], m.get("advisory_grade", "ERR"),
                 m.get("positive_years", "-"), m.get("n_valid_years", "-"),
                 m.get("icir_range", "-"), rec["elapsed_s"]))

    # 汇总 (仅描述)
    ok = [r for r in results if r.get("metrics") and r["metrics"].get("n_valid_years")]
    def _med(vals):
        vals = [v for v in vals if v is not None]
        if not vals:
            return None
        vals = sorted(vals)
        n = len(vals)
        return round(vals[n // 2] if n % 2 else (vals[n // 2 - 1] + vals[n // 2]) / 2, 4)
    summary = {
        "n_factors": len(results),
        "n_ok": len(ok),
        "n_error": len(results) - len(ok),
        "positive_years_ratio_median": _med([r["metrics"]["positive_years_ratio"] for r in ok]),
        "icir_range_median": _med([r["metrics"]["icir_range"] for r in ok]),
        "sign_consistency_median": _med([r["metrics"]["sign_consistency"] for r in ok]),
        "n_grade_A_consistent": sum(1 for r in ok if r["metrics"]["advisory_grade"] == "A_consistent"),
        "n_grade_B_mixed": sum(1 for r in ok if r["metrics"]["advisory_grade"] == "B_mixed"),
        "n_grade_C_unstable": sum(1 for r in ok if r["metrics"]["advisory_grade"] == "C_unstable"),
        "n_grade_insufficient_years": sum(1 for r in ok if r["metrics"]["advisory_grade"] == "insufficient_years"),
        "n_fully_covered_years": sum(1 for r in ok if r["metrics"]["n_valid_years"] >= 6),
        "median_valid_years": _med([r["metrics"]["n_valid_years"] for r in ok]),
        "n_valid_years_hist": dict(sorted((
            __import__("collections").Counter(
                r["metrics"]["n_valid_years"] for r in ok).items())), ) if ok else {},
    }

    report = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "probe_version": "P-20260920-004",
        "probe_kind": "readonly_diagnostic",
        "claim": "本报告为只读诊断, 零裁决; 不参与任何 PASS/FAIL, 不改因子池状态。",
        "caliber": {
            "lookback_days": None,
            "forward_period": ic.forward_period,
            "min_stocks_per_day": ic.min_stocks_per_day,
            "min_days_per_year": 5,
            "date_range": dr,
            "n_trading_days": int(len(ic._all_dates)) if ic._all_dates is not None else None,
            "param_neighborhood": bool(args.param_neighborhood),
        },
        "summary": summary,
        "factors": results,
    }
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    s = summary
    print()
    print("─" * 78)
    print("[GateC] 汇总 (%d 条, 成功 %d / 失败 %d)"
          % (s["n_factors"], s["n_ok"], s["n_error"]))
    print("[GateC] 正增量年份比例中位 = %s | 年际 ICIR 极差中位 = %s | 符号一致度中位 = %s"
          % (s["positive_years_ratio_median"], s["icir_range_median"],
             s["sign_consistency_median"]))
    print("[GateC] A_consistent=%d / B_mixed=%d / C_unstable=%d"
          % (s["n_grade_A_consistent"], s["n_grade_B_mixed"], s["n_grade_C_unstable"]))
    print("[GateC] 报告已落盘: %s" % outp)
    print("[GateC] 总耗时: %.1fs" % (time.time() - t0))

    if args.emit_daily_line:
        print()
        print("[GateC] 日报行: [GateC:readonly] 诊断 %d 条 | 正增量年份比例中位 %.2f | "
              "年际 ICIR 极差中位 %.2f | A/B/C = %d/%d/%d | 零裁决"
              % (s["n_ok"], s["positive_years_ratio_median"] or 0,
                 s["icir_range_median"] or 0, s["n_grade_A_consistent"],
                 s["n_grade_B_mixed"], s["n_grade_C_unstable"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
