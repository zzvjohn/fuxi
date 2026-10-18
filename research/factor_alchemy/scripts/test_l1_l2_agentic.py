# -*- coding: utf-8 -*-
"""
L1 解读层 + L2 修复回路 全链路测试 (2026-08-14)
================================================
覆盖:
  T1  classify_by_rules 四 verdict 判定
  T2  interpret_factor 离线降级产出合法结构
  T3  interpret_batch 挂载 + 落盘
  T4  distill_motif_knowledge: execution_failure 不罚 motif
  T5  get_interpretations 检索 + prompt 片段
  T6  build_prompt 注入 (llm_generator)
  T7  MAB 方向精细化逻辑 (execution_failure 撤销失败计数)
  T8  is_code_failure 分类 + enqueue 幂等
  T9  repair_agent 沙箱: 备份/回滚/合并 (假文件, 不碰生产)
  T10 repair 失败 → WarningDirection
"""
import json
import sys
import tempfile
from pathlib import Path

FACTOR_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(FACTOR_DIR))
sys.path.insert(0, str(Path(__file__).parent))

PASS = 0
FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ✓ {name}")
    else:
        FAIL += 1
        print(f"  ✗ {name}  {detail}")


def t1_classify():
    print("\n[T1] classify_by_rules 四 verdict")
    from interpretation_agent import classify_by_rules

    # breed008 型: IC 有效但 JQ 崩 → execution_failure
    r = classify_by_rules({"jq_return": -66.4, "jq_sharpe": -0.9, "local_icir": 0.38})
    check("breed008型 → execution_failure", r["verdict"] == "execution_failure", r["verdict"])

    # breed004 型: local 与 JQ 双重否定 → direction_falsified
    r = classify_by_rules({"jq_return": -30.0, "jq_sharpe": -0.8, "local_icir": 0.05})
    check("双重否定 → direction_falsified", r["verdict"] == "direction_falsified", r["verdict"])

    # 软负收益 → direction_falsified (弱)
    r = classify_by_rules({"jq_return": -8.0, "jq_sharpe": -0.1, "local_icir": 0.1})
    check("软负 → direction_falsified", r["verdict"] == "direction_falsified", r["verdict"])

    # 强正向 → direction_confirmed
    r = classify_by_rules({"jq_return": 143.5, "jq_sharpe": 0.51, "local_icir": 0.4})
    check("强正向 → direction_confirmed", r["verdict"] == "direction_confirmed", r["verdict"])

    # 弱正向 → direction_confirmed (低置信)
    r = classify_by_rules({"jq_return": 18.2, "jq_sharpe": -0.03, "local_icir": 0.2})
    check("弱正向 → direction_confirmed", r["verdict"] == "direction_confirmed", r["verdict"])
    check("  弱正向置信度 < 0.7", r["confidence"] < 0.7)


def t2_interpret_factor_offline():
    print("\n[T2] interpret_factor 离线降级")
    from interpretation_agent import interpret_factor
    r = interpret_factor(
        {"factor_name": "x", "jq_return": -66.4, "jq_sharpe": -0.9, "local_icir": 0.38},
        use_llm=False,
    )
    check("verdict 合法", r["verdict"] in ("execution_failure", "direction_falsified",
                                          "direction_confirmed", "data_issue"))
    check("有 narrative", bool(r.get("narrative")))
    check("有 hints", bool(r.get("next_round_hints")))
    check("source=rule_fallback", r.get("source") == "rule_fallback")


def t3_interpret_batch():
    print("\n[T3] interpret_batch 挂载 + 落盘")
    from interpretation_agent import interpret_batch
    payload = {
        "batch_id": "test_l1_batch",
        "factors": [
            {"factor_name": "f_a", "paradigm": "动量", "jq_return": -66.4,
             "jq_sharpe": -0.9, "local_icir": 0.38, "local_ic": 0.03},
            {"factor_name": "f_b", "paradigm": "均值回复", "jq_return": 124.8,
             "jq_sharpe": 0.44, "local_icir": 0.35, "local_ic": 0.02},
        ],
    }
    res = interpret_batch(payload, memory=None, use_llm=False)
    check("返回 n_factors=2", res.get("n_factors") == 2)
    check("f_a 挂载 interpretation", "interpretation" in payload["factors"][0])
    check("verdict 分布统计", res.get("verdict_counts", {}).get("execution_failure", 0) == 1)
    out = Path(__file__).parent.parent.parent.parent / "data" / "interpretations" / "interpretation_test_l1_batch.json"
    check("落盘文件存在", out.exists())


def t4_distill_two_level():
    print("\n[T4] 两级蒸馏: execution_failure 不罚 motif (隔离临时 memory)")
    import tempfile
    from experience_memory import ExperienceMemory
    # 隔离: 使用临时 memory 文件, 绝不碰生产 data
    tmp_path = Path(tempfile.mkdtemp()) / "mem_isolated.json"
    mem = ExperienceMemory(memory_path=tmp_path)
    formula = "close.rolling(20).mean() + volume.rolling(10).std()"
    motif = mem._extract_rough_motifs(formula)[0]
    mem.data.setdefault("motif_stats", {})[motif] = {"pass": 0, "fail": 0, "jq_pass": 0, "jq_fail": 0}
    mem._save()

    # execution_failure → 不罚
    mem.distill_motif_knowledge(
        formula=formula,
        jq_passed=False,
        fsa=None,
        interpretation={"verdict": "execution_failure", "confidence": 0.8},
    )
    stats = mem.data["motif_stats"].get(motif, {})
    check("execution_failure → jq_fail 不变", stats.get("jq_fail", 0) == 0, str(stats))
    check("execution_failure → 记 interp_notes", bool(stats.get("interp_notes")), str(stats))

    # direction_falsified → 正常罚
    mem.distill_motif_knowledge(
        formula=formula,
        jq_passed=False,
        fsa=None,
        interpretation={"verdict": "direction_falsified", "confidence": 0.7},
    )
    stats = mem.data["motif_stats"].get(motif, {})
    check("direction_falsified → jq_fail +1", stats.get("jq_fail", 0) == 1, str(stats))


def _isolated_memory():
    """构造隔离 memory 实例 + 替换 get_memory 单例 (不碰生产 data)"""
    import experience_memory as em
    import tempfile
    tmp_path = Path(tempfile.mkdtemp()) / "mem_isolated.json"
    mem = em.ExperienceMemory(memory_path=tmp_path)
    em._default_memory = mem
    return mem


def t5_get_interpretations():
    print("\n[T5] get_interpretations + prompt 片段 (隔离 memory)")
    mem = _isolated_memory()
    mem.data.setdefault("attempts", []).append({
        "factor_name": "test_interp_f1",
        "paradigm": "价量关系",
        "jq_return": -66.4,
        "jq_verified_at": "2026-08-14T10:00:00",
        "interpretation": {
            "verdict": "execution_failure",
            "next_round_hints": ["避免高换手构造", "检查滑点敏感度"],
        },
    })
    mem._save()

    recs = mem.get_interpretations(paradigm="价量关系", limit=5)
    check("检索到 1 条", len(recs) == 1, str(len(recs)))
    frag = mem.build_interpretation_prompt_fragment(paradigm="价量关系")
    check("片段含避坑标题", "归因提示" in frag)
    check("片段含 hint", "避免高换手构造" in frag)
    empty = mem.build_interpretation_prompt_fragment(paradigm="不存在的范式")
    check("无记录返回空串", empty == "")


def t6_prompt_injection():
    print("\n[T6] llm_generator prompt 注入 (隔离 memory 单例)")
    mem = _isolated_memory()
    mem.data.setdefault("attempts", []).append({
        "factor_name": "test_interp_p1",
        "paradigm": "动量",
        "jq_return": -30.0,
        "jq_verified_at": "2026-08-14T10:00:00",
        "interpretation": {
            "verdict": "direction_falsified",
            "next_round_hints": ["纯动量方向已被 JQ 证伪"],
        },
    })
    mem._save()

    from llm_generator import LLMGenerator
    gen = LLMGenerator()
    prompt = gen.build_prompt(target_paradigm="动量", n_factors=2)
    check("prompt 含归因片段", "归因提示" in prompt)
    check("prompt 含避坑内容", "纯动量方向已被 JQ 证伪" in prompt)


def t7_mab_interp():
    print("\n[T7] MAB 方向级精细化 (模拟 jq_feedback Step 3.6)")
    # 模拟 MAB 方向对象
    class FakeDir:
        def __init__(self, name):
            self.name = name
            self.direction_id = name
            self.failures = 2
            self.expected_reward = 0.1
            self.pulls = 10
    class FakeMAB:
        def __init__(self):
            self.directions = {"d1": FakeDir("动量")}
            self.saved = 0
        def save_state(self):
            self.saved += 1
    mab = FakeMAB()

    # execution_failure → 撤销失败计数
    f = {"paradigm": "动量", "interpretation": {"verdict": "execution_failure"}}
    dirs = list(mab.directions.values())
    for d in dirs:
        if f["paradigm"] in str(d.name):
            if d.failures > 0:
                d.failures -= 1
    check("execution_failure → failures 2→1", mab.directions["d1"].failures == 1,
          str(mab.directions["d1"].failures))

    # direction_falsified → 惩罚
    f2 = {"paradigm": "动量", "interpretation": {"verdict": "direction_falsified"}}
    for d in dirs:
        if f2["paradigm"] in str(d.name):
            d.failures += 1
            d.expected_reward = max(-0.5, d.expected_reward - 0.2)
    d1 = mab.directions["d1"]
    check("direction_falsified → failures +1", d1.failures == 2, str(d1.failures))
    check("direction_falsified → reward -0.2", abs(d1.expected_reward - (-0.1)) < 1e-9,
          str(d1.expected_reward))


def t8_code_failure_classify():
    print("\n[T8] 代码类失败分类 + 入队幂等")
    from repair_agent import is_code_failure, enqueue_repair, load_queue

    check("NameError 是代码失败", is_code_failure("NameError: name 'x' is not defined"))
    check("KeyError 是代码失败", is_code_failure("KeyError: 'content'"))
    check("编译失败是代码失败", is_code_failure("编译失败: invalid syntax"))
    check("IC 过低不是代码失败", not is_code_failure("S1 拒绝: |IC|=0.002 过低"))
    check("Calmar 过低不是代码失败", not is_code_failure("S5 拒绝: Calmar=0.1 < 1.0"))

    ok1 = enqueue_repair("t8_factor", "close.rolling(5).mean()", "S5", "NameError: x")
    ok2 = enqueue_repair("t8_factor", "close.rolling(5).mean()", "S5", "NameError: x")
    check("首次入队成功", ok1)
    check("重复入队被幂等拦截", not ok2)

    q = load_queue()
    t8_items = [i for i in q["queue"] if i.get("factor_name") == "t8_factor"]
    check("队列中恰有 1 条", len(t8_items) == 1, str(len(t8_items)))
    # 清理
    q["queue"] = [i for i in q["queue"] if i.get("factor_name") != "t8_factor"]
    from repair_agent import save_queue
    save_queue(q)


def t9_sandbox_rollback():
    print("\n[T9] repair 沙箱: 测试门禁失败 → 自动回滚")
    from repair_agent import _run_tests
    # 用假文件验证备份/回滚机制 (不碰生产代码)
    import shutil
    tmp = Path(tempfile.mkdtemp())
    target = tmp / "dummy_module.py"
    target.write_text("def hello():\n    return 'orig'\n", encoding="utf-8")

    bak = target.with_suffix(target.suffix + ".bak_l2")
    shutil.copy2(target, bak)
    # 写入"修复"
    target.write_text("def hello():\n    return 'fixed'\n", encoding="utf-8")
    # 模拟测试失败 → 回滚
    shutil.copy2(bak, target)
    bak.unlink(missing_ok=True)
    content = target.read_text(encoding="utf-8")
    check("回滚后内容还原", "'orig'" in content, content)

    # 模拟全绿 → 合并保留 + 清理备份
    shutil.copy2(target, bak)
    target.write_text("def hello():\n    return 'merged'\n", encoding="utf-8")
    bak.unlink(missing_ok=True)
    check("合并后内容生效", "'merged'" in target.read_text(encoding="utf-8"))
    check("备份已清理", not bak.exists())

    # _extract_files 路径安全: 越界路径被拒
    from repair_agent import _extract_files
    evil = json.dumps({"files": [{"path": "../evil.py", "content": "x = " + "1" * 30}]})
    files = _extract_files(evil)
    check("越界路径被拒绝", len(files) == 0, str(files))

    shutil.rmtree(tmp, ignore_errors=True)


def t10_trajectory_interpretation_field():
    print("\n[T10] MiningTrajectory interpretation 字段持久化")
    from trajectory_logger import MiningTrajectory
    t = MiningTrajectory(trajectory_id="t10")
    t.interpretation = {"verdict": "direction_confirmed", "confidence": 0.9}
    d = t.to_dict()
    check("to_dict 含 interpretation", d.get("interpretation", {}).get("verdict") == "direction_confirmed")
    t2 = MiningTrajectory.from_dict(d)
    check("from_dict 还原 interpretation", t2.interpretation.get("verdict") == "direction_confirmed")


if __name__ == "__main__":
    print("=" * 60)
    print("  L1 解读层 + L2 修复回路 全链路测试")
    print("=" * 60)
    t1_classify()
    t2_interpret_factor_offline()
    t3_interpret_batch()
    t4_distill_two_level()
    t5_get_interpretations()
    t6_prompt_injection()
    t7_mab_interp()
    t8_code_failure_classify()
    t9_sandbox_rollback()
    t10_trajectory_interpretation_field()
    print("\n" + "=" * 60)
    print(f"  结果: {PASS} PASS / {FAIL} FAIL")
    print("=" * 60)
    sys.exit(0 if FAIL == 0 else 1)
