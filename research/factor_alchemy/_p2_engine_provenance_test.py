"""P2 (2026-09-05) 引擎溯源+跨引擎降权 功能单测 — 用临时内存, 不碰真实 data/experience_memory.json"""
import os
import sys
import tempfile
import json
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from experience_memory import (ExperienceMemory, _norm_engine, _cross_engine_enabled,
                               CROSS_ENGINE_WEIGHT)

FAILS = []


def check(name, cond, detail=""):
    if cond:
        print(f"  [PASS] {name}")
    else:
        print(f"  [FAIL] {name} {detail}")
        FAILS.append(name)


# ── 1. _norm_engine 映射 ──
print("== 1. _norm_engine 映射 ==")
check("cross_breed→gp_breed", _norm_engine("cross_breed") == "gp_breed")
check("gp_breed→gp_breed", _norm_engine("gp_breed") == "gp_breed")
check("factorforge→forge", _norm_engine("factorforge") == "forge")
check("forge→forge", _norm_engine("forge") == "forge")
check("llm→llm", _norm_engine("llm") == "llm")
check("stage2_daily→llm", _norm_engine("stage2_daily") == "llm")
check("generated→''", _norm_engine("generated") == "")
check("空→''", _norm_engine("") == "")
check("大小写", _norm_engine("GP_BREED") == "gp_breed")

# ── 2. 临时内存: motif_stats 跨引擎降权 ──
print("== 2. get_motif_forbidden 跨引擎降权 ==")
tmp = Path(tempfile.mkdtemp()) / "mem_test.json"
mem = ExperienceMemory(memory_path=tmp)

# m1: gp_breed 4 fail 0 pass
for _ in range(4):
    mem._update_motif_stats(["m1"], passed=False, engine="gp_breed")
# m2: gp_breed 3 fail + llm 3 fail
for _ in range(3):
    mem._update_motif_stats(["m2"], passed=False, engine="gp_breed")
    mem._update_motif_stats(["m2"], passed=False, engine="llm")
# m3: legacy 3 fail (无 by_engine)
for _ in range(3):
    mem._update_motif_stats(["m3"], passed=False)
# m4: 10 legacy fail + 1 gp_breed fail (by_engine 有 legacy 剩余)
for _ in range(10):
    mem._update_motif_stats(["m4"], passed=False)  # legacy (无 engine)
mem._update_motif_stats(["m4"], passed=False, engine="gp_breed")

# consumer=gp_breed: m1/m2/m3 全禁 (m1 同引擎 4>=3)
fb_gp = set(mem.get_motif_forbidden(consumer_engine="gp_breed"))
check("gp_breed 消费 m1 禁", "m1" in fb_gp)
check("gp_breed 消费 m2 禁", "m2" in fb_gp)
check("gp_breed 消费 m3 禁 (legacy)", "m3" in fb_gp)

# consumer=llm: m1 跨引擎加权 4*0.5=2<3 → 不禁 (核心救援场景)
fb_llm = set(mem.get_motif_forbidden(consumer_engine="llm"))
check("llm 消费 m1 不禁 (跨引擎降权)", "m1" not in fb_llm, f"got {fb_llm}")
check("llm 消费 m2 禁 (自身证据 3>=3)", "m2" in fb_llm)
check("llm 消费 m3 禁 (legacy 全权重)", "m3" in fb_llm)
check("llm 消费 m4 禁 (legacy 剩余全权重)", "m4" in fb_llm, f"got {fb_llm}")

# consumer=None (legacy): m1 也禁 (无消费引擎=全权重)
fb_none = set(mem.get_motif_forbidden())
check("无 consumer m1 禁 (旧行为不变)", "m1" in fb_none)

# env shadow 回退: 全权重
os.environ["FUXI_CROSS_ENGINE"] = "shadow"
check("shadow 模式 _cross_engine_enabled=False", not _cross_engine_enabled())
fb_shadow = set(mem.get_motif_forbidden(consumer_engine="llm"))
check("shadow 下 llm 消费 m1 恢复禁", "m1" in fb_shadow)
del os.environ["FUXI_CROSS_ENGINE"]
check("默认 enforce", _cross_engine_enabled())

# ── 3. motif_rules 引擎拆分 ──
print("== 3. get_motif_rules 跨引擎拆分 ==")
mem._add_motif_rule("forbid", "r_gp", "gp失败", engine="gp_breed")
mem._add_motif_rule("forbid", "r_mix", "多引擎失败", engine="mixed")
mem._add_motif_rule("forbid", "r_legacy", "legacy失败")  # 无 engine
mem._add_motif_rule("prefer", "p_gp", "gp成功", engine="gp_breed")

r_llm = mem.get_motif_rules(consumer_engine="llm")
check("llm 硬 forbid=2 (mixed+legacy)", len(r_llm["forbid"]) == 2,
      f"got {[r['motif_key'] for r in r_llm['forbid']]}")
check("llm forbid_cross=1 (r_gp)", len(r_llm["forbid_cross"]) == 1,
      f"got {[r['motif_key'] for r in r_llm['forbid_cross']]}")
check("llm prefer_cross=1 (p_gp)", len(r_llm["prefer_cross"]) == 1)
check("llm prefer=0", len(r_llm["prefer"]) == 0)

r_gp = mem.get_motif_rules(consumer_engine="gp_breed")
check("gp_breed 硬 forbid=3 (同引擎 r_gp 不降)", len(r_gp["forbid"]) == 3)
check("gp_breed forbid_cross=0", len(r_gp["forbid_cross"]) == 0)

r_none = mem.get_motif_rules()
check("无 consumer forbid=3 (旧行为)", len(r_none["forbid"]) == 3)
check("无 consumer 无 cross 键", "forbid_cross" not in r_none)

os.environ["FUXI_CROSS_ENGINE"] = "shadow"
r_shadow = mem.get_motif_rules(consumer_engine="llm")
check("shadow 下 llm forbid=3 无拆分", len(r_shadow["forbid"]) == 3)
del os.environ["FUXI_CROSS_ENGINE"]

# ── 4. 规则溯源升级 (去重时 legacy→具体→mixed) ──
print("== 4. 规则 engine 溯源升级 ==")
mem._add_motif_rule("forbid", "r_up", "v1")            # legacy
mem._add_motif_rule("forbid", "r_up", "v2", engine="llm")   # 升级 llm
mem._add_motif_rule("forbid", "r_up", "v3", engine="forge")  # 冲突 → mixed
rules = mem.data["motif_rules"]
r_up = [r for r in rules if r["motif_key"] == "r_up"][0]
check("溯源升级→mixed", r_up.get("engine") == "mixed", f"got {r_up.get('engine')}")

# ── 5. _motif_jq_engine 判定 ──
print("== 5. _motif_jq_engine ==")
s1 = {"jq_pass": 0, "jq_fail": 3, "by_engine": {"gp_breed": {"jq_pass": 0, "jq_fail": 3}}}
check("单引擎→gp_breed", mem._motif_jq_engine(s1) == "gp_breed")
s2 = {"jq_pass": 0, "jq_fail": 4,
      "by_engine": {"gp_breed": {"jq_pass": 0, "jq_fail": 2}, "llm": {"jq_pass": 0, "jq_fail": 2}}}
check("双引擎→mixed", mem._motif_jq_engine(s2) == "mixed")
s3 = {"jq_pass": 0, "jq_fail": 5,
      "by_engine": {"gp_breed": {"jq_pass": 0, "jq_fail": 2}}}  # legacy 剩余 3
check("legacy 剩余→mixed", mem._motif_jq_engine(s3) == "mixed")
s4 = {"jq_pass": 0, "jq_fail": 3}
check("无 by_engine→''", mem._motif_jq_engine(s4) == "")

# ── 6. record() engine 落盘 ──
print("== 6. record engine 落盘 ==")
mem.record(factor_name="t_eng", formula="close.pct_change()", paradigm="测试",
           outcome="PASS", engine="forge")
a = [x for x in mem.data["attempts"] if x["factor_name"] == "t_eng"][0]
check("attempt engine=forge", a.get("engine") == "forge")
# 更新不覆盖: engine="" 不覆盖旧值
mem.record(factor_name="t_eng", formula="close.pct_change()", paradigm="测试", outcome="REJECT")
a = [x for x in mem.data["attempts"] if x["factor_name"] == "t_eng"][0]
check("空 engine 更新不覆盖旧值", a.get("engine") == "forge")

# ── 7. get_engine_of 回溯 ──
print("== 7. get_engine_of ==")
check("回溯 forge", mem.get_engine_of("t_eng") == "forge")
check("无记录→''", mem.get_engine_of("no_such") == "")

print()
if FAILS:
    print(f"❌ {len(FAILS)} FAILS: {FAILS}")
    sys.exit(1)
print(f"✅ 全部通过 (降权系数 CROSS_ENGINE_WEIGHT={CROSS_ENGINE_WEIGHT})")
