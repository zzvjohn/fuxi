"""定向 D+ 蒸馏: 仅本次 3 个 JQ 结果 → jq_feedback 闭环 (2026-08-29)
- 不触发 trigger_d_plus 全量 (当前有 152 条历史 pending 欠账, 不在本次范围)
- 防丢字段: jq_feedback 内部 TrajectoryLogger._save 用 to_dict 会清空 JSON 扩展字段
  (jq_rating/jq_maxdd/d_plus_applied/factor_name/jq_notes/jq_ic/jq_icir/jq_window),
  因此: 快照全库扩展字段 → jq_feedback → 恢复快照 → 补 3 条扩展字段 → mark d_plus_applied
"""
import json
import sys
from datetime import datetime

sys.path.insert(0, "research/factor_alchemy")

TRAJ_PATH = "data/trajectory_log.json"
NOW = datetime.now().isoformat()

# ── 3 因子 jq_backtest_result (与已写回轨迹一致) ──
FEEDBACK = {
    "batch_id": f"dplus_3factors_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    "timestamp": NOW,
    "composite_return": (91.91 - 79.46 + 38.65) / 3,
    "composite_sharpe": (0.411 - 1.083 + 0.105) / 3,
    "composite_maxdd": (30.85 + 83.2 + 34.03) / 3,
    "factors": [
        {
            "factor_name": "forge_gen1_ts_std_square_inv_returns_30",
            "formula": ("((1.0 / (((close_p - close_p.shift(1)) / close_p.shift(1)) + 1e-6)) * "
                        "(1.0 / (((close_p - close_p.shift(1)) / close_p.shift(1)) + 1e-6))).rolling(30).std()"),
            "hypothesis": "尾部风险: 日收益倒数平方的30日波动 (Forge GP Gen1)",
            "paradigm": "均值回复",
            "category": "volatility",
            "operators_used": ["close", "shift", "division", "rolling_std"],
            "local_ic": 0.02562, "local_icir": 0.2126,
            "jq_return": 91.91, "jq_sharpe": 0.411, "jq_maxdd": 30.85,
            "jq_ic": 0.0259, "jq_icir": 0.2653,
            "jq_composite_contribution": "positive",
            "root_cause": ("JQ全市场5.5年正向通过: 尾部风险做多有效; JQ per-factor IC +0.0259/ICIR +0.2653 "
                           "与 S5 局部(IC 0.0256)一致; 周频训练ICIR(+0.587)→日频IC折扣实证"),
        },
        {
            "factor_name": "v3_lhb_freq_20d",
            "formula": "lhb_flag.rolling(20).sum()",
            "hypothesis": "近20交易日龙虎榜上榜次数(正向)",
            "paradigm": "资金流",
            "category": "money_flow",
            "operators_used": ["rolling_sum", "lhb"],
            "local_ic": 0.051097, "local_icir": 0.4974,
            "jq_return": -79.46, "jq_sharpe": -1.083, "jq_maxdd": 83.2,
            "jq_ic": -0.0670, "jq_icir": -0.6680,
            "jq_composite_contribution": "negative",
            "root_cause": ("方向裁决: 龙虎榜上榜频次正向(高上榜=买)JQ全市场灾难; JQ per-factor IC -0.067/ICIR -0.668 "
                           "强负alpha → 与取负版(lhb_freq_20d JQ +64.59%)互证: 上榜频次是负向信号, "
                           "取负(低上榜)才是真alpha方向; S5 200股池正向PASS被JQ全市场推翻"),
        },
        {
            "factor_name": "forge_gen1_square_square_tsmin",
            "formula": ("(((1.0 / (((close_p - close_p.shift(1)) / close_p.shift(1)).rolling(30).rank(pct=True).rolling(10).mean() + 1e-6)).rolling(30).min() * "
                        "(1.0 / (((close_p - close_p.shift(1)) / close_p.shift(1)).rolling(30).rank(pct=True).rolling(10).mean() + 1e-6)).rolling(30).min()) * "
                        "((1.0 / (((close_p - close_p.shift(1)) / close_p.shift(1)).rolling(30).rank(pct=True).rolling(10).mean() + 1e-6)).rolling(30).min() * "
                        "(1.0 / (((close_p - close_p.shift(1)) / close_p.shift(1)).rolling(30).rank(pct=True).rolling(10).mean() + 1e-6)).rolling(30).min()))"),
            "hypothesis": "筹码分布: 日收益30日排名10日均值倒数的30日低点平方平方 (Forge GP Gen1)",
            "paradigm": "尾部风险",
            "category": "position_analysis",
            "operators_used": ["close", "shift", "rank", "rolling_mean", "rolling_min", "division", "multiplication"],
            "local_ic": 0.021979, "local_icir": 0.2169,
            "jq_return": 38.65, "jq_sharpe": 0.105, "jq_maxdd": 34.03,
            "jq_ic": 0.0289, "jq_icir": 0.2788,
            "jq_composite_contribution": "neutral",
            "root_cause": ("JQ全市场+38.65% MARGINAL: JQ per-factor IC +0.0289/ICIR +0.2788 不算差但组合收益弱 "
                           "→ 换手/尾部执行折损, 与 S5 弱证据(excess 5.06%/2.44%)一致; 保留观察不硬禁"),
        },
    ],
}

TARGET_IDS = {
    "forge_gen1_ts_std_square_inv_returns_30": "traj_eaca745e39b6",
    "v3_lhb_freq_20d": "traj_52c0cb80277d",
    "forge_gen1_square_square_tsmin": "traj_88fa28230cee",
}

# ── 1. 快照全库 JSON 扩展字段 ───────────────────────
# MiningTrajectory.to_dict 白名单 (类属性), 其余均为扩展字段
TO_DICT_KEYS = {
    "trajectory_id", "paradigm", "seed_factor", "generation", "phase",
    "hypothesis", "expression", "code", "backtest", "overall_score",
    "final_ic", "final_icir", "final_outcome", "jq_validated",
    "jq_return", "jq_sharpe", "interpretation", "created_at",
    "completed_at", "high_reward_segments", "_eval_log",
}

with open(TRAJ_PATH, encoding="utf-8") as f:
    traj_data = json.load(f)

ext_snapshot = {}
for t in traj_data["trajectories"]:
    ext = {k: v for k, v in t.items() if k not in TO_DICT_KEYS}
    if ext:
        ext_snapshot[t["trajectory_id"]] = ext

print(f"扩展字段快照: {len(ext_snapshot)} 条轨迹")

# ── 2. 跑 jq_feedback 蒸馏 ──────────────────────────
from ralph_loop import RalphLoop
from experience_memory import get_memory

memory = get_memory()
rl = RalphLoop(memory=memory)
result = rl.jq_feedback(jq_backtest_result=FEEDBACK)

# ── 3. 恢复扩展字段快照 (jq_feedback 的 _save 清空了它们) ──
with open(TRAJ_PATH, encoding="utf-8") as f:
    traj_data = json.load(f)

n_restored = 0
for t in traj_data["trajectories"]:
    tid = t["trajectory_id"]
    snap = ext_snapshot.get(tid)
    if snap:
        for k, v in snap.items():
            if k not in t:  # 只恢复被清空的键, 不覆盖已有
                t[k] = v
                n_restored += 1

# ── 4. 补 3 条扩展字段 (final_outcome/jq_return 已是类属性, jq_feedback 已更新) ──
for name, tid in TARGET_IDS.items():
    f = next(x for x in FEEDBACK["factors"] if x["factor_name"] == name)
    for t in traj_data["trajectories"]:
        if t["trajectory_id"] == tid:
            t["factor_name"] = name
            t["jq_rating"] = {"JQ_PASSED": "PASS", "JQ_FAILED": "FAIL",
                              "JQ_MARGINAL": "MARGINAL"}[
                t.get("final_outcome", "")]
            t["jq_maxdd"] = f["jq_maxdd"]
            t["jq_ic"] = f["jq_ic"]
            t["jq_icir"] = f["jq_icir"]
            t["jq_notes"] = f["root_cause"]
            t["jq_window"] = "2021-01-01~2026-07-30"
            t["jq_feedback_at"] = NOW
            t["d_plus_applied"] = True      # mark 完成
            t["d_plus_at"] = NOW
            print(f"  [D+] {tid} <- {name}: {t['final_outcome']} d_plus_applied=True")
            break

with open(TRAJ_PATH, "w", encoding="utf-8") as f:
    json.dump(traj_data, f, ensure_ascii=False, indent=2)

print(f"\n扩展字段恢复: {n_restored} 键")
print("D+ 定向蒸馏完成")
print(f"  hard_forbidden_added: {result.get('hard_forbidden_added')}")
print(f"  jq_success_confirmed: {result.get('jq_success_confirmed')}")
print(f"  trajectories_updated: {result.get('trajectories_updated')}")
