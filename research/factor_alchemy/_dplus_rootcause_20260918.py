# -*- coding: utf-8 -*-
"""D+ 欠账 235 条的成因实证: 记录结构 vs 白名单 to_dict() 的键位差。
只读, 不写任何文件。"""
import io, json, os, datetime
from collections import Counter, defaultdict

TRAJ = r"e:\quant\data\trajectory_log.json"

st = os.stat(TRAJ)
print("=" * 78)
print("trajectory_log.json  last_write =", datetime.datetime.fromtimestamp(st.st_mtime))
print("size = %.1f MB" % (st.st_size / 1e6))
print("=" * 78)

d = json.load(io.open(TRAJ, encoding="utf-8"))
trajs = d["trajectories"]
print("总轨迹数 =", len(trajs))

# ── 1. 白名单 to_dict() 的 22 键 ──
WHITELIST = {"trajectory_id","paradigm","seed_factor","generation","phase","hypothesis",
             "expression","code","backtest","overall_score","final_ic","final_icir",
             "final_outcome","jq_validated","jq_return","jq_sharpe","natural_freq",
             "interpretation","created_at","completed_at","high_reward_segments","_eval_log"}

allkeys = Counter()
for t in trajs:
    allkeys.update(t.keys())

print("\n--- 1. 全库实际出现的键 (按出现次数) ---")
for k, v in allkeys.most_common():
    flag = "白名单" if k in WHITELIST else "🔴不在白名单(会被 _save() 丢弃)"
    print("  %-26s %5d  %s" % (k, v, flag))

# ── 2. D+ 标记现状 ──
n_jqv = sum(1 for t in trajs if t.get("jq_validated"))
n_dpa = sum(1 for t in trajs if t.get("d_plus_applied"))
n_dpat = sum(1 for t in trajs if t.get("d_plus_at"))
backlog = [t for t in trajs if t.get("jq_validated") and not t.get("d_plus_applied")]
print("\n--- 2. D+ 标记现状 ---")
print("  jq_validated=True            :", n_jqv)
print("  含 d_plus_applied 键         :", n_dpa)
print("  含 d_plus_at 键              :", n_dpat)
print("  欠账 (jqv & not dpa)         :", len(backlog))

# ── 3. 欠账构成 ──
print("\n--- 3. 欠账 %d 条构成 ---" % len(backlog))
print("  final_outcome :", dict(Counter(t.get("final_outcome") or "(empty)" for t in backlog)))
print("  phase         :", dict(Counter(t.get("phase") or "(empty)" for t in backlog)))

# 有无因子名 (决定是否真的进得来 D+ 蒸馏的 factor 列表)
n_name  = sum(1 for t in backlog if t.get("seed_factor"))
n_formu = sum(1 for t in backlog if isinstance(t.get("expression"), dict)
              and (t["expression"].get("content") or "").strip())
n_ret0  = sum(1 for t in backlog if not t.get("jq_return"))
print("  有 seed_factor:", n_name, "| expression.content 非空:", n_formu, "| jq_return 缺失/0:", n_ret0)

rets = [t.get("jq_return") for t in backlog if isinstance(t.get("jq_return"), (int, float))]
if rets:
    rets_sorted = sorted(rets)
    print("  jq_return: n=%d  min=%.2f  p25=%.2f  med=%.2f  p75=%.2f  max=%.2f  非零=%d"
          % (len(rets), rets_sorted[0], rets_sorted[len(rets)//4],
             rets_sorted[len(rets)//2], rets_sorted[len(rets)*3//4],
             rets_sorted[-1], sum(1 for r in rets if r)))

# ── 4. 时间分布 (created_at / completed_at) ──
print("\n--- 4. 欠账按 completed_at 日期分布 ---")
byd = Counter((t.get("completed_at") or t.get("created_at") or "?")[:10] for t in backlog)
for k in sorted(byd):
    print("   ", k, byd[k])
print("  最早 :", min(byd) if byd else "-", "  最晚 :", max(byd) if byd else "-")
print("  跨天数:", len(byd))

# ── 5. 全库 d_plus_at 是否曾落盘过 ──
print("\n--- 5. 结构性结论 ---")
print("  d_plus_applied 出现次数 = %d / %d 条" % (n_dpa, len(trajs)))
print("  d_plus_at      出现次数 = %d / %d 条" % (n_dpat, len(trajs)))
print("  白名单键数 = %d ; 全库最大键集 = %d" % (len(WHITELIST), len(allkeys)))
extra = [k for k in allkeys if k not in WHITELIST]
print("  存在于数据但不在白名单的键 (%d 个): %s" % (len(extra), extra))
