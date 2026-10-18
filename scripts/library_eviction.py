# -*- coding: utf-8 -*-
"""
P-20260904-001 落地: 因子库退役/冷冻机制 (library eviction + freeze zone)

背景:
- 因子库 corr 矩阵硬上限 200 (run_v4_pipeline.compute_correlation_matrix metadata 路径),
  库内带公式因子 194 即将触顶, 但只有 decay/crowding 监控, 没有 eviction 机制。
- P0-2 目标: 建立"退役评分 → 冷冻区 → 回捞审计"闭环, 解除容量约束, 让 500 位置
  装独立信息, 而不是无限膨胀同质因子。

设计原则 (与四铁则一致):
1. JQ 唯一真相源: 已有 JQ 裁决的因子绝不自动退役 (只报告, 需人工);
2. 本地 IC 只否决不排序: 本地复评证伪 (FLIP_AND_DEAD) 是退役最强证据;
3. 只自动冷冻 `reserve` 状态因子 (不在验证链上); champion_seed/candidate/recheck
   一律保护, 至多给出"待人工复核"建议;
4. 冷冻 ≠ 删除: 因子进 data/library_freeze.json 冷冻区, 带 recall_conditions,
   满足条件可回捞 (每轮回捞 ≤1 个, 回捞后重新走 S1, 不进 MMR 主键)。

用法:
  python scripts/library_eviction.py                 # dry-run, 输出候选清单
  python scripts/library_eviction.py --commit        # 落盘: 备份CSV → 更新status → 写freeze json
  python scripts/library_eviction.py --score-factor NAME  # 单因子评分明细

输出:
- data/library_freeze.json (冷冻区, 增量)
- unified_factor_pool.csv 的 status 列 reserve→frozen (--commit)
- data/library_eviction_report_YYYYMMDD.json (每次运行的全量评分快照)
"""
import argparse
import json
import re
import shutil
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "research" / "factor_alchemy"))

POOL_CSV = ROOT / "data" / "unified_factor_pool.csv"
FREEZE_JSON = ROOT / "data" / "library_freeze.json"
REEVAL_JSON = ROOT / "data" / "p001_full_clean_reeval_20260905.json"
DECAY_LOG = ROOT / "data" / "decay_monitor_log.json"
CROWDING_JSON = ROOT / "data" / "factor_level_crowding.json"
REPORT_JSON = ROOT / "data" / "library_eviction_report_{}.json"

# ── 评分权重 ──
W_FLIP_DEAD = 45      # 干净源复评: 符号翻转且无信号 (裁决作废)
W_FLIP_VERDICT = 35   # 干净源复评: 符号翻转但有信号 (原方向作废, 需反向复评)
W_DOWNGRADE = 30      # 干净源复评: 不达标 (污染可能虚增; 本地证伪=第二强退役证据)
W_DECAY = 20          # decay 监控 IC_UNSTABLE danger
W_CROWD = 10          # 族级拥挤极端
W_STALE = 10          # reserve 且无任何验证痕迹 (久龄闲置)
PROTECT_CHAMPION = -40
THRESHOLD_FREEZE = 40
THRESHOLD_WATCH = 25
IC_MEAN_WEAK = 0.02   # 历史 |ic_mean| 低于此值视为无信号占位

# 不自动退役的状态 (在验证链上或已有 JQ 裁决)
NO_AUTO_FREEZE_STATUS = {
    "candidate", "champion_seed", "recheck_s1_passed", "s5_passed",
    "jq_done", "jq_marginal", "jq_passed", "jq_failed_direction", "jq_run_done",
    "jq_run_done_v2", "frozen",
}
ONLY_STATUS = {"reserve"}  # 允许自动冷冻的唯一状态


def _load_decay_alerts() -> dict:
    """decay_monitor_log 最新 scan 的 linear/structural alerts → {factor_key: {type, severity}}"""
    if not DECAY_LOG.exists():
        return {}
    with open(DECAY_LOG, encoding="utf-8") as f:
        d = json.load(f)
    scans = d.get("scans") or []
    if not scans:
        return {}
    sc = scans[-1]
    out = {}
    for key in ("linear_alerts", "structural_alerts"):
        rows = sc.get(key) or []
        if isinstance(rows, str):
            import ast
            try:
                rows = ast.literal_eval(rows)
            except Exception:
                rows = []
        for r in rows:
            if not isinstance(r, dict):
                continue
            fn = r.get("factor_name") or ""
            out[fn] = {
                "alert_type": r.get("alert_type"),
                "severity": r.get("severity"),
                "ic_trend": r.get("ic_trend"),
            }
    return out


def _match_decay_alert(name: str, formula: str, alerts: dict) -> dict:
    """decay 记录里 factor_name 可能是公式截断, 按 name 或公式前缀匹配"""
    if name in alerts:
        return alerts[name]
    fx = str(formula or "")
    for k, v in alerts.items():
        if len(k) >= 20 and fx.startswith(k):
            return v
    return None


def _load_crowding() -> dict:
    if not CROWDING_JSON.exists():
        return {}
    with open(CROWDING_JSON, encoding="utf-8") as f:
        d = json.load(f)
    fs = d.get("factors") or []
    out = {}
    for r in fs:
        if isinstance(r, dict) and r.get("name"):
            out[r["name"]] = r
    return out


def _load_reeval() -> dict:
    if not REEVAL_JSON.exists():
        return {}
    with open(REEVAL_JSON, encoding="utf-8") as f:
        d = json.load(f)
    return {r.get("name"): r for r in d.get("results") or []}


def _load_freeze() -> dict:
    if FREEZE_JSON.exists():
        with open(FREEZE_JSON, encoding="utf-8") as f:
            return json.load(f)
    return {"version": 1, "frozen": [], "updated_at": ""}


def score_factor(row: pd.Series, reeval: dict, alerts: dict, crowding: dict,
                 freeze_names: set) -> dict:
    """返回 {score, breakdown: [...], protect: bool, protect_reason}"""
    name = row["name"]
    status = str(row.get("status") or "reserve")
    breakdown = []
    score = 0

    if name in freeze_names:
        return {"score": -999, "breakdown": ["already_frozen"], "protect": True,
                "protect_reason": "已在冷冻区"}

    protect = status not in ONLY_STATUS
    protect_reason = f"状态 {status} 不在自动冷冻范围" if protect else ""

    # 1. 干净源复评
    rv = reeval.get(name)
    if rv:
        v = rv.get("verdict")
        # 数据不可用 (上游停更/字段缺失 → n_days=0) 与"无信号"区分:
        # 数据不可用时不加分 (不是因子本身的错, 走数据源修复而非退役)
        nd = rv.get("n_days") or 0
        data_dead = (nd == 0 and (rv.get("ic_clean") or 0) == 0)
        if not data_dead:
            if v == "FLIP_AND_DEAD":
                score += W_FLIP_DEAD
                breakdown.append(f"复评FLIP_AND_DEAD(+{W_FLIP_DEAD})")
            elif v == "FLIP_VERDICT":
                score += W_FLIP_VERDICT
                breakdown.append(f"复评FLIP_VERDICT(+{W_FLIP_VERDICT})")
            elif v == "DOWNGRADE":
                score += W_DOWNGRADE
                breakdown.append(f"复评DOWNGRADE(+{W_DOWNGRADE})")
        else:
            breakdown.append(f"复评数据不可用(停更/缺失, 不计分)")

    # 2. decay 监控
    al = _match_decay_alert(name, row.get("formula"), alerts)
    if al and al.get("severity") == "danger" and al.get("alert_type") == "IC_UNSTABLE":
        score += W_DECAY
        breakdown.append(f"decay IC_UNSTABLE danger(+{W_DECAY})")

    # 3. 族级拥挤
    cr = crowding.get(name)
    if cr:
        beta = cr.get("comomentum_beta")
        ls = cr.get("ls_vol_ratio")
        if (beta is not None and abs(float(beta)) >= 0.7) or \
           (ls is not None and float(ls) >= 0.5):
            score += W_CROWD
            breakdown.append(f"拥挤极端(+{W_CROWD})")

    # 4. 久龄闲置 (reserve 且无任何"裁决验证"痕迹)
    #    裁决验证 = daily_icir(S1) / weekly_icir(周频) / jq_return(JQ)。
    #    注意: icir / ic_mean 是因子生成时的一次性历史统计, 不算裁决验证。
    def _has(v):
        try:
            return v is not None and not pd.isna(v)
        except (TypeError, ValueError):
            return True
    trace_cols = ["daily_icir", "weekly_icir", "jq_return"]
    has_trace = any(_has(row.get(c)) for c in trace_cols if c in row.index)
    if status == "reserve" and not has_trace:
        score += W_STALE
        breakdown.append(f"reserve久龄无裁决验证(+{W_STALE})")

    # 5. champion 保护
    if status == "champion_seed":
        score += PROTECT_CHAMPION
        breakdown.append(f"champion_seed保护({PROTECT_CHAMPION})")

    return {"score": score, "breakdown": breakdown,
            "protect": protect, "protect_reason": protect_reason}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--commit", action="store_true", help="落盘: 备份CSV→更新status→写freeze")
    ap.add_argument("--min-score", type=float, default=THRESHOLD_FREEZE,
                    help="自动冷冻评分阈值 (默认 40)")
    ap.add_argument("--score-factor", type=str, default=None, help="单因子评分明细")
    args = ap.parse_args()

    pool = pd.read_csv(POOL_CSV)
    pool = pool.drop_duplicates(subset=["name"], keep="last").copy()
    reeval = _load_reeval()
    alerts = _load_decay_alerts()
    crowding = _load_crowding()
    freeze = _load_freeze()
    freeze_names = {f["name"] for f in freeze.get("frozen", [])}

    print(f"[退役评分] 池 {len(pool)} 因子 | 复评 {len(reeval)} 条 | decay alerts "
          f"{len(alerts)} | 拥挤 {len(crowding)} | 已冷冻 {len(freeze_names)}")

    rows = []
    for _, r in pool.iterrows():
        s = score_factor(r, reeval, alerts, crowding, freeze_names)
        rows.append({
            "name": r["name"], "status": str(r.get("status") or "reserve"),
            "score": s["score"], "breakdown": s["breakdown"],
            "protect": s["protect"], "protect_reason": s["protect_reason"],
            "ic_mean": None if pd.isna(r.get("ic_mean")) else float(r["ic_mean"]),
            "daily_icir": None if pd.isna(r.get("daily_icir")) else float(r["daily_icir"]),
            "jq_return": None if pd.isna(r.get("jq_return")) else float(r["jq_return"]),
            "reeval_verdict": (reeval.get(r["name"]) or {}).get("verdict"),
        })

    if args.score_factor:
        for r in rows:
            if r["name"] == args.score_factor:
                print(json.dumps(r, ensure_ascii=False, indent=2))
        return

    # 候选清单
    auto_candidates = [r for r in rows if r["score"] >= args.min_score and not r["protect"]]
    watch = [r for r in rows
             if THRESHOLD_WATCH <= r["score"] < args.min_score and not r["protect"]]
    manual_review = [r for r in rows if r["score"] >= args.min_score and r["protect"]]

    auto_candidates.sort(key=lambda x: -x["score"])
    manual_review.sort(key=lambda x: -x["score"])
    watch.sort(key=lambda x: -x["score"])

    print("\n" + "=" * 78)
    print(f"自动冷冻候选 (reserve 且 score>={args.min_score}): {len(auto_candidates)}")
    for r in auto_candidates[:30]:
        print(f"  [{r['score']:>3}] {r['name'][:46]:<46} {r['status']} "
              f"reeval={r['reeval_verdict']} | {'; '.join(r['breakdown'])}")

    print(f"\n人工复核建议 (有保护但高分): {len(manual_review)}")
    for r in manual_review[:15]:
        print(f"  [{r['score']:>3}] {r['name'][:46]:<46} {r['status']} "
              f"({r['protect_reason']})")

    print(f"\n观察清单 (score {THRESHOLD_WATCH}-{args.min_score-1}): {len(watch)}")
    for r in watch[:15]:
        print(f"  [{r['score']:>3}] {r['name'][:46]:<46} {r['status']}")

    # 低价值库存报告 (不入评分, 供批量人工退役决策):
    # reserve + 无裁决验证 + 历史 |ic_mean|<0.02 → 纯占位
    low_value = [
        r for r in rows
        if r["status"] == "reserve" and not r["protect"]
        and r["ic_mean"] is not None and abs(r["ic_mean"]) < IC_MEAN_WEAK
        and "reserve久龄无裁决验证" in "; ".join(r["breakdown"])
        and r["reeval_verdict"] is None
    ]
    print(f"\n低价值库存 (reserve + 无裁决验证 + |ic_mean|<{IC_MEAN_WEAK} + 未复评): "
          f"{len(low_value)} 个, 可批量人工退役")
    for r in low_value[:15]:
        print(f"  {r['name'][:50]:<50} ic_mean={r['ic_mean']:.4f}")

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "thresholds": {"freeze": args.min_score, "watch": THRESHOLD_WATCH},
        "auto_freeze_candidates": auto_candidates,
        "manual_review": manual_review,
        "watch": watch,
        "low_value_inventory": low_value,
    }
    rep_path = Path(str(REPORT_JSON).format(datetime.now().strftime("%Y%m%d")))
    with open(rep_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print(f"\n评分快照已写 {rep_path}")

    if not args.commit:
        print("\n[dry-run] 未执行任何修改。加 --commit 落盘冷冻。")
        return

    # ── commit 路径 ──
    if not auto_candidates:
        print("[commit] 无自动冷冻候选, 跳过")
        return
    bak = POOL_CSV.with_suffix(".csv.bak_evict_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    shutil.copy2(POOL_CSV, bak)
    print(f"[commit] 已备份池: {bak}")

    now = datetime.now().isoformat(timespec="seconds")
    new_frozen = []
    for c in auto_candidates:
        name = c["name"]
        pool.loc[pool["name"] == name, "status"] = "frozen"
        new_frozen.append({
            "name": name,
            "frozen_at": now,
            "from_status": c["status"],
            "score": c["score"],
            "breakdown": c["breakdown"],
            "reeval_verdict": c["reeval_verdict"],
            "recall_conditions": [
                "干净源复评 verdict=MAINTAIN 或 FLIP_VERDICT 且 icir>=0.15",
                "周频 JQ 实证通过",
                "上游数据源修复并重算达标",
            ],
            "recall_audit": [],
        })
    freeze["frozen"].extend(new_frozen)
    freeze["updated_at"] = now
    with open(FREEZE_JSON, "w", encoding="utf-8") as f:
        json.dump(freeze, f, ensure_ascii=False, indent=1)
    pool.to_csv(POOL_CSV, index=False, encoding="utf-8")
    print(f"[commit] ✅ 已冷冻 {len(new_frozen)} 个因子 → {FREEZE_JSON}")
    print("[commit] status 已更新: reserve → frozen (仅统一池 status 列)")
    print("[commit] ⚠️ 注意: ralph/MMR 端读取排除 frozen 的联动尚未改造 (后续 P), "
          "冷冻因子在下次库重建前仍可能被 corr 矩阵纳入")


if __name__ == "__main__":
    main()
