# -*- coding: utf-8 -*-
"""2026-09-19 第 4 项: 主流水线跑完后的**回读**验证。

判据全部锚在实现事实 (JSON 键频 / 计数 / 集合包含), 不锚在日志文本。
零假设: 若 schema 修复无效, 则 d_plus_* 与 lessons 应再次出现「1/N 被抹签名」。
"""
import io
import json
import sys
from datetime import datetime
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

ROOT = Path(r'e:/quant')
TRAJ = ROOT / "data" / "trajectory_log.json"
BASE = ROOT / "data" / "_dplus_baseline_20260919_predround.json"
POST = ROOT / "data" / "_dplus_postround_20260919.json"

with open(BASE, 'r', encoding='utf-8') as f:
    base = json.load(f)

with open(TRAJ, 'r', encoding='utf-8') as f:
    data = json.load(f)
trajs = data["trajectories"]

from collections import Counter
keyfreq = Counter()
for t in trajs:
    keyfreq.update(t.keys())

n = len(trajs)
RICH7 = ["jq_maxdd", "jq_ic", "jq_icir", "jq_rating",
         "jq_plus_ic_pct", "jq_backtest_period", "jq_notes"]
RICH9 = RICH7 + ["d_plus_applied", "d_plus_at"]

checks = []


def chk(name, ok, detail):
    checks.append((name, bool(ok), detail))
    print(f"   [{'PASS' if ok else '*** FAIL'}] {name}: {detail}")


print("=" * 74)
print("第 4 项 · 主流水线跑完后的回读验证")
print("=" * 74)
print(f"  基线时间 {base['captured_at']}  →  本次回读 {datetime.now().isoformat()}")
print(f"  基线 n={base['n_trajectories']}  →  现在 n={n}  (+{n - base['n_trajectories']})")
print()

print("-- A. schema 是否真的经过 _save() 走了一遍 --")
chk("A1 lessons 键频 = n",
    keyfreq.get("lessons", 0) == n,
    f"lessons={keyfreq.get('lessons', 0)} / n={n}  (基线 0 ⇒ 必须变为全量)")
full32 = [k for k in keyfreq if keyfreq[k] == n]
chk("A2 库内键数 ≥ 32",
    len(keyfreq) >= 32,
    f"键并集={len(keyfreq)} (基线 {base['n_keys_in_union']})")
chk("A3 9 个原富字段全部 100% 覆盖",
    all(keyfreq.get(k, 0) == n for k in RICH9),
    "; ".join(f"{k}={keyfreq.get(k, 0)}" for k in RICH9))

print()
print("-- B. D+ 标记是否存活 (核心判据) --")
d_plus = [t for t in trajs if t.get("d_plus_applied")]
chk("B1 d_plus_applied 计数不减少",
    len(d_plus) >= base['d_plus_applied'],
    f"{len(d_plus)} ≥ 基线 {base['d_plus_applied']}")
base_ids = set(base['marked_ids'])
post_ids = {t['trajectory_id'] for t in d_plus}
lost = base_ids - post_ids
chk("B2 基线的 236 个已标记 id 全部仍在",
    not lost,
    f"丢失 {len(lost)} 条 {sorted(lost)[:5]}")

# 逐条对 source
post_src = {t['trajectory_id']: t.get('d_plus_source', '') for t in d_plus}
src_drift = {k: (base['marked_source_map'][k], post_src.get(k))
             for k in base['marked_source_map'] if post_src.get(k) != base['marked_source_map'][k]}
chk("B3 d_plus_source 逐条不变 (区分·回填 vs 真实闭环)",
    not src_drift, f"漂移 {len(src_drift)} 条 {list(src_drift.items())[:3]}")

debt = sum(1 for t in trajs if t.get("jq_validated") and not t.get("d_plus_applied"))
chk("B4 欠账仍为 0",
    debt == 0, f"欠账={debt}")

print()
print("-- C. 记录未丢失 --")
post_all = {t['trajectory_id'] for t in trajs}
missing = base_ids | {t['trajectory_id'] for t in trajs}
chk("C1 轨迹总数单调不减",
    n >= base['n_trajectories'], f"{n} ≥ {base['n_trajectories']}")

print()
print("-- D. 披露纪律: 富字段覆盖必须与「已标记」同时报 --")
rich7_cov = sum(1 for t in trajs if any(t.get(k) not in (None, "", [], {}) for k in RICH7))
print(f"   d_plus_applied = {len(d_plus)} / {n}")
print(f"   富 7 字段任一非空 = {rich7_cov} / {n}   <-- 二者必须同时披露")
chk("D1 富字段覆盖未被回填伪造",
    rich7_cov <= base['rich7_nonempty'] + 5,
    f"{rich7_cov} ≈ 基线 {base['rich7_nonempty']} (Tier-2 默认关 ⇒ 不应暴增)")

print()
print("-- E. jq_feedback 新打标的条目 (source='jq_feedback') --")
newfb = [t['trajectory_id'] for t in d_plus
         if t.get('d_plus_source') == 'jq_feedback' and t['trajectory_id'] not in base_ids]
print(f"   本次轮次新增 jq_feedback 打标 = {len(newfb)} 条: {newfb[:10]}")

print()
n_pass = sum(1 for _, ok, _ in checks if ok)
n_fail = len(checks) - n_pass
print("=" * 74)
print(f"OVERALL: {n_pass}/{len(checks)} PASS, {n_fail} FAIL")
print("=" * 74)

out = {
    "verified_at": datetime.now().isoformat(),
    "n_trajectories": n,
    "n_keys_in_union": len(keyfreq),
    "keyfreq": dict(keyfreq),
    "d_plus_applied": len(d_plus),
    "d_plus_debt": debt,
    "d_plus_source_dist": dict(Counter(t.get("d_plus_source", "") for t in d_plus)),
    "rich7_nonempty": rich7_cov,
    "new_jq_feedback_marks": newfb,
    "checks": [{"name": a, "pass": b, "detail": c} for a, b, c in checks],
    "n_pass": n_pass, "n_fail": n_fail,
    "baseline": {k: base[k] for k in ("n_trajectories", "d_plus_applied",
                                      "d_plus_debt", "rich7_nonempty",
                                      "n_keys_in_union", "lessons_key")},
}
with open(POST, 'w', encoding='utf-8', newline='') as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print(f"written: {POST}")
