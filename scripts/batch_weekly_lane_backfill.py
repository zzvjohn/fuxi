# -*- coding: utf-8 -*-
"""
P-20260830-002: τ_w 校准集负样本定向扩充 (批量复评 + 中等区间 JQ 补测)
================================================================================
背景: v0.7 校准集 JQ 锚点仅 13 个, FAILED 锚点仅 1 个 (volume_dual_rank 0.509)。
      τ_w 无上调判别力; 方向翻转 (本地 0.333 → JQ -0.303) 无本地预警。
      → 对轨迹库中所有 weekly 候选按 pandas 口径批量复评周频 ICIR,
        挑选 0.15~0.40 中等区间样本 (τ_w 附近最不确定带, 判别力最大边际)
        生成 JQ 单因子代码入队列分批回测 (默认每天 5 条)。

铁则遵守:
  - 只增校准集数据点, 不改 τ_w 生效值 (hysteresis 机制本就要求连续 2 批同值)
  - JQ 代码执行层骨架 = gen_jq_generic (v0.9 官方生成器, 已对齐 lhb v2 修复版)
  - 不碰裁决逻辑 (S1 分频 XOR 与 weekly_lane.judge 判定零改动)

用法:
  python scripts/batch_weekly_lane_backfill.py                # 复评 + 生成 5 条入队
  python scripts/batch_weekly_lane_backfill.py --dry-run      # 只复评报告, 不生成
  python scripts/batch_weekly_lane_backfill.py --n 10         # 自定义批大小
"""
import argparse
import hashlib
import json
import re
import sys
from datetime import datetime
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DATA_DIR = _PROJECT_ROOT / "data"
_FA_DIR = _PROJECT_ROOT / "research" / "factor_alchemy"

ICIR_LO = 0.15   # τ_w 附近不确定带下界
ICIR_HI = 0.40   # 上界 (>0.40 的样本多数会 PASS, 判别力低)


def _norm_formula(s: str) -> str:
    return re.sub(r"\s+", "", str(s or ""))


def _load_trajectories():
    p = _DATA_DIR / "trajectory_log.json"
    d = json.load(open(p, encoding="utf-8"))
    return d if isinstance(d, list) else d.get("trajectories", [])


def _load_calibration():
    p = _DATA_DIR / "lane_calibration.json"
    if not p.exists():
        return {}
    return json.load(open(p, encoding="utf-8"))


def _load_queue():
    p = _DATA_DIR / "jq_codegen_queue.json"
    if not p.exists():
        return {}
    d = json.load(open(p, encoding="utf-8"))
    return d if isinstance(d, dict) else {}


def _save_queue(queue: dict) -> None:
    p = _DATA_DIR / "jq_codegen_queue.json"
    json.dump(queue, open(p, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2, default=str)


def _load_jq_generator():
    import importlib.util
    p = _PROJECT_ROOT / "scripts" / "gen_jq_generic.py"
    spec = importlib.util.spec_from_file_location("gen_jq_generic", str(p))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def collect_weekly_candidates():
    """候选两来源:
    ① 轨迹库 weekly 候选 (expression.content 公式);
    ② 校准集内无 jq_outcome 且 pandas_icir ∈ [0.15,0.40] 的点 (未 JQ 验证的中等区间样本,
       连复评都省, 是提案最直接的存量目标)。
    返回 [{source, key, formula, hypothesis, outcome, jq_validated, pandas_icir?}]"""
    cands = []
    seen = set()

    # 来源 ①: 轨迹库
    trajs = _load_trajectories()
    for t in trajs:
        if str(t.get("natural_freq", "")).strip().lower() != "weekly":
            continue
        expr = t.get("expression") or {}
        if isinstance(expr, dict):
            formula = str(expr.get("content", "")).strip()
        else:
            formula = str(expr or "").strip()
        if not formula or formula.startswith("# TODO"):
            continue
        if "\n" in formula:
            lines = [ln.strip() for ln in formula.split("\n")
                     if ln.strip() and not ln.strip().startswith("#")]
            if not lines:
                continue
            formula = lines[-1]
            m = re.match(r'^\s*[a-zA-Z_][a-zA-Z0-9_]*\s*=\s*(.+)$', formula)
            if m:
                formula = m.group(1).strip()
        key = _norm_formula(formula)
        if key in seen:
            continue
        seen.add(key)
        cands.append({
            "source": "trajectory",
            "trajectory_id": t.get("trajectory_id", ""),
            "formula": formula,
            "hypothesis": str((t.get("hypothesis") or ""))[:200],
            "outcome": t.get("final_outcome", ""),
            "jq_validated": bool(t.get("jq_validated")),
            "pandas_icir": None,
        })

    # 来源 ②: 校准集存量 (无 JQ 锚点 + 中等区间)
    cal = _load_calibration()
    for p in cal.get("points", []):
        if p.get("jq_outcome"):
            continue
        pic = p.get("pandas_icir")
        if pic is None or not (ICIR_LO <= pic <= ICIR_HI):
            continue
        formula = str(p.get("formula") or "").strip()
        if not formula or formula.startswith("# TODO"):
            continue
        key = _norm_formula(formula)
        if key in seen:
            continue
        seen.add(key)
        cands.append({
            "source": "calibration",
            "trajectory_id": f"calib_{str(p.get('name') or '')[:16] or 'pt'}",
            "formula": formula,
            "hypothesis": str(p.get("name") or "")[:200],
            "outcome": "",
            "jq_validated": False,
            "pandas_icir": round(float(pic), 4),
        })
    return cands


def main():
    ap = argparse.ArgumentParser(description="τ_w 校准集负样本定向扩充 (P-20260830-002)")
    ap.add_argument("--dry-run", action="store_true", help="只复评报告, 不生成 JQ 代码")
    ap.add_argument("--n", type=int, default=5, help="本批生成 JQ 代码条数 (默认 5)")
    args = ap.parse_args()

    sys.path.insert(0, str(_FA_DIR))
    from lane_calibration import review_pandas_icir

    cal = _load_calibration()
    cal_formulas = {_norm_formula(p.get("formula")) for p in cal.get("points", [])
                    if p.get("formula")}
    queue = _load_queue()
    queued_formulas = {_norm_formula(v.get("formula")) for v in queue.values()
                       if isinstance(v, dict) and v.get("formula")}
    queued_names = set(queue.keys())

    cands = collect_weekly_candidates()
    print(f"[Backfill] 轨迹库 weekly 候选 (去重): {len(cands)}")
    print(f"[Backfill] 已在校准集: {len(cal_formulas)} 公式 | 已在 JQ 队列: {len(queued_formulas)}")

    # 复评 (来源②校准集点自带 pandas_icir, 直接采信)
    reviewed = []
    for c in cands:
        if _norm_formula(c["formula"]) in cal_formulas and c["source"] != "calibration":
            c["skip_reason"] = "已在校准集"
            continue
        if _norm_formula(c["formula"]) in queued_formulas:
            c["skip_reason"] = "已在 JQ 队列"
            continue
        if c["jq_validated"]:
            c["skip_reason"] = "已 JQ 验证"
            continue
        if c["pandas_icir"] is None:
            icir, note = review_pandas_icir(c["formula"], c["trajectory_id"])
            if icir is None:
                c["skip_reason"] = f"复评失败: {note}"
                continue
            c["pandas_icir"] = round(icir, 4)
        icir = c["pandas_icir"]
        if ICIR_LO <= icir <= ICIR_HI:
            c["band"] = "medium"
            reviewed.append(c)
        else:
            c["band"] = "out"
            c["skip_reason"] = f"ICIR={icir:.3f} 不在 [{ICIR_LO},{ICIR_HI}]"

    skipped = [c for c in cands if c.get("skip_reason")]
    print(f"[Backfill] 复评完成: 中等区间 {len(reviewed)} 条 | 跳过 {len(skipped)} 条")
    for c in skipped:
        print(f"    SKIP {c['trajectory_id'][:20]:20s} {c['skip_reason'][:60]}")

    # 排序: 距 τ_w=0.15 最近优先 (最不确定带, FAILED 锚点判别力最大边际)
    reviewed.sort(key=lambda c: abs(c["pandas_icir"] - 0.15))
    print(f"[Backfill] 中等区间样本 (按 |ICIR-0.15| 升序):")
    for c in reviewed[:20]:
        print(f"    {c['pandas_icir']:.4f}  {c['trajectory_id'][:24]:24s} "
              f"{c['formula'][:50]}")

    if args.dry_run or not reviewed:
        print("[Backfill] dry-run 或无候选, 不生成 JQ 代码")
        return

    gen = _load_jq_generator()
    n_gen = 0
    for c in reviewed[: args.n]:
        _h = hashlib.md5(_norm_formula(c["formula"]).encode("utf-8")).hexdigest()[:8]
        name = f"weekly_backfill_{c['trajectory_id'][:12]}_{_h}"
        if name in queued_names:
            continue
        local_note = ("P-20260830-002 校准集回填: 本地周频 ICIR=%.4f (中等区间) | "
                      "轨迹 outcome=%s" % (c["pandas_icir"], c["outcome"]))
        try:
            out_path, meta = gen.generate_standalone(
                name, c["formula"], meaning=c["hypothesis"],
                local_note=local_note, window_freq="weekly",
            )
        except Exception as e:
            out_path, meta = None, {"reason": f"生成异常: {type(e).__name__}: {str(e)[:80]}"}
        if out_path:
            queue[name] = {
                "file": str(out_path),
                "formula": c["formula"][:300],
                "meaning": c["hypothesis"][:200],
                "generated_at": datetime.now().isoformat(),
                "status": "pending_jq_run",
                "seed_recheck": False,
                "lookback": (meta or {}).get("lookback"),
                "natural_freq": "weekly",
                "backfill": "P-20260830-002",
                "local_icir": c["pandas_icir"],
            }
            n_gen += 1
            print(f"    [OK] {name} → {Path(out_path).name}")
        else:
            print(f"    [SKIP] {name}: {(meta or {}).get('reason', '未知错误')}")

    _save_queue(queue)
    pending = sum(1 for v in queue.values()
                  if isinstance(v, dict) and v.get("status") == "pending_jq_run")
    print(f"[Backfill] 本批生成 {n_gen}/{min(len(reviewed), args.n)} 条; "
          f"队列 pending_jq_run 共 {pending} 条")
    print("[Backfill] 提醒: 将上述 .py 复制到聚宽研究环境回测, "
          "结果反馈后 D+ 蒸馏 + lane_calibration 自动滚动更新")


if __name__ == "__main__":
    main()
