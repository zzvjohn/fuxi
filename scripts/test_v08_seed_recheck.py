# -*- coding: utf-8 -*-
"""v0.8 种子主体重检 + verification_level 单元测试"""
import sys, json, tempfile, os
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path("e:/quant/research/factor_alchemy")))

import warnings
warnings.filterwarnings("ignore")

# ═══ 1. verification_level 排序 (retrieve 主键) ═══
from experience_memory import ExperienceMemory, VERIFICATION_RANK

mem = ExperienceMemory.__new__(ExperienceMemory)
# 伪造 data 结构 (只测排序)
mem.data = {
    "success_templates": [
        {"pattern_id": "X::stage2", "occurrence_count": 9999, "success_rate": 1.0,
         "verification_level": "stage2_only", "formula": "a", "typical_operators": [], "typical_windows": []},
        {"pattern_id": "X::s5", "occurrence_count": 100, "success_rate": 0.9,
         "verification_level": "s5_passed", "formula": "b", "typical_operators": [], "typical_windows": []},
        {"pattern_id": "X::comp", "occurrence_count": 500, "success_rate": 0.9,
         "verification_level": "jq_composite", "formula": "c", "typical_operators": [], "typical_windows": []},
        {"pattern_id": "X::single", "occurrence_count": 10, "success_rate": 0.5,
         "verification_level": "jq_single", "formula": "d", "typical_operators": [], "typical_windows": []},
    ],
    "forbidden_directions": [],
    "forbidden_regions": [],
}
# 调用内部排序逻辑片段 (retrieve 的核心排序)
def sorted_ids(templates):
    return [t["pattern_id"] for t in sorted(
        templates,
        key=lambda t: (VERIFICATION_RANK.get(t.get("verification_level", "stage2_only"), 0),
                       t.get("occurrence_count", 0), t.get("success_rate", 0)),
        reverse=True)]

ids = sorted_ids(mem.data["success_templates"])
print("1. 排序结果:", ids)
assert ids == ["X::single", "X::comp", "X::s5", "X::stage2"], f"排序错误: {ids}"
print("   ✅ jq_single(occ=10) > jq_composite(occ=500) > s5_passed(occ=100) > stage2_only(occ=9999)")

# ═══ 2. _prepare_seed_recheck_candidates 逻辑 ═══
from ralph_loop import RalphLoop

rl = RalphLoop.__new__(RalphLoop)
rl._seed_recheck_cooloff_days = 30

# 伪造 state 文件路径 → 临时文件
tmp_state = Path(tempfile.gettempdir()) / "seed_recheck_test_state.json"
rl._seed_recheck_state_path = lambda: tmp_state
rl._load_seed_recheck_state = lambda: {}

candidates = [
    {"factor_name": "champ_a", "formula": "rank(sub(ts_max(close, 20), ts_min(close, 20)))", "status": "champion_seed", "icir": 0.3},
    {"factor_name": "champ_b", "formula": "rank(sub(ts_max(close, 20), ts_min(close, 20)))", "status": "champion_seed", "icir": 0.9},
    {"factor_name": "reserve_c", "formula": "close_p.pct_change(5).rolling(20).mean()", "status": "reserve", "icir": 0.7},
    {"factor_name": "jq_done", "formula": "close_p.pct_change(5).rolling(20).mean()", "status": "candidate", "icir": 1.5},
    {"factor_name": "recent_e", "formula": "close_p.pct_change(5).rolling(20).mean()", "status": "reserve", "icir": 0.4},
]
# jq_done 已单因子 JQ 验证 → 跳过; recent_e 30 天内已重检 → 跳过
rl._load_seed_recheck_state = lambda: {
    "jq_done": {"jq_single_verified": True, "last_check": datetime.now().isoformat()},
    "recent_e": {"last_check": (datetime.now() - timedelta(days=5)).isoformat()},
}
seeds = rl._prepare_seed_recheck_candidates(candidates, max_seeds=3)
names = [s["factor_name"] for s in seeds]
print("2. 挑选的种子:", names)
assert "jq_done" not in names, "已 JQ 验证的不应重检"
assert "recent_e" not in names, "冷却期内不应重检"
assert names == ["champ_b", "champ_a", "reserve_c"], f"champion 优先+icir降序错误: {names}"
assert all(s["_seed_recheck"] for s in seeds)
assert all(s["_source"] == "seed_recheck" for s in seeds)
# 翻译验证: Forge 风格 champion 公式应被翻译成 pandas
assert ".rolling" in seeds[0]["formula"], f"Forge 公式未翻译: {seeds[0]['formula'][:60]}"
print("   ✅ 过滤规则正确: 已JQ跳过/冷却跳过/champion优先/icir降序/标记注入/Forge翻译")

# ═══ 3. 排序稳定性: 同等级按 occ ═══
mem2 = {
    "success_templates": [
        {"pattern_id": "A", "occurrence_count": 10, "verification_level": "s5_passed", "success_rate": 0.5},
        {"pattern_id": "B", "occurrence_count": 50, "verification_level": "s5_passed", "success_rate": 0.5},
    ]
}
ids2 = sorted_ids(mem2["success_templates"])
assert ids2 == ["B", "A"], f"同等级应按 occ 降序: {ids2}"
print("3. ✅ 同等级 occ 次键排序正确")

# ═══ 4. 默认等级回退 ═══
mem3 = {"success_templates": [{"pattern_id": "C", "occurrence_count": 5, "success_rate": 0.5}]}
assert sorted_ids(mem3["success_templates"])[0] == "C"
print("4. ✅ 缺失 verification_level 回退 stage2_only (rank=0) 不崩溃")

print("\n全部通过 ✅")
