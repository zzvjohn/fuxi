# -*- coding: utf-8 -*-
"""Register the 2026-09-22 R0-R3 remediation in atomic, versionable batches."""

import argparse
import datetime
import json
import os
import time
from pathlib import Path

NOW = datetime.datetime.now().isoformat(timespec="seconds")
ROOT = Path(__file__).resolve().parents[1]
PROPOSALS_PATH = ROOT / "data" / "stage1_optimization_proposals.json"
LOCK_PATH = PROPOSALS_PATH.with_suffix(PROPOSALS_PATH.suffix + ".lock")


PROPOSALS = {
    "r0": [{
        "proposal_id": "P-20260922-002",
        "title": "R0 单写者运行契约: 活锁拒绝、原子清单、不可复用 run_id 与全公式指纹",
        "category": "管线基础设施",
        "domains": ["T3模块/通道"],
        "risk_level": "high",
        "status": "landed",
        "proposal_change_kind": "behavior",
        "trigger": "多入口均可改写 MAB/轨迹/候选结果，但此前没有跨入口互斥、不可变运行清单或完整公式身份，无法排除并发污染与截断哈希碰撞。",
        "current_state": "直接入口和 CLI 可绕过彼此，结果文件不能证明实际代码、数据口径、开关和候选身份。",
        "proposed_change": "所有 run_v4_pipeline Ralph 写路径统一进入 FuxiWriterLock；运行清单原子落盘，记录代码/数据指纹、公开 FUXI 开关、阶段状态、完整公式 SHA256 和 research_action_id。",
        "expected_impact": "任何并发写入 fail-closed；每轮可复现、可追责，终态 run_id 不可静默重用。",
        "implementation_scope": "research/factor_alchemy/run_contract.py + run_v4_pipeline.py + 契约测试。",
        "verification": "test_run_contract: 活锁拒绝、原子清单、终态 run_id 不可复用、全公式哈希。",
        "rollback_plan": "恢复 reports/remediation_20260922/r0_baseline 中 run_v4_pipeline.py，并移除 run_contract.py；历史 manifest 保留只读。",
        "landed_detail": "R0 已落地。统一锁 data/fuxi_runtime.lock；stale 锁只改名留证，不静默删除；manifest 写临时文件+fsync+replace+读回；只记录 FUXI_* 公共开关，不记录密钥。",
        "landed_at": "2026-09-22",
        "landed_artifact": "research/factor_alchemy/run_contract.py; research/factor_alchemy/run_v4_pipeline.py; research/factor_alchemy/tests/test_run_contract.py",
        "code_changed": True,
        "jq_validation_required": False,
        "cross_verified": True,
    }],
    "r1": [{
        "proposal_id": "P-20260922-003",
        "title": "R1 Forge 截面编译器语义同构: 修复 Series 列对齐全 NaN 与 ddof/tie/常数漂移",
        "category": "编译器修复",
        "domains": ["T1闸门/裁决", "T2数据口径"],
        "risk_level": "high",
        "status": "landed",
        "proposal_change_kind": "behavior",
        "trigger": "Forge numpy 基因型可得分，但 pandas 表型 zscore/scale 按列对齐，80x3 输入扩成 80x83 且全 NaN；rank tie、zscore ddof、常数 scale 亦不等价。",
        "current_state": "生成端精选与 S1/S5 实际求值不是同一个因子，导致候选假失败或不可求值。",
        "proposed_change": "FUXI_FORGE_CS_COMPILE=enforce 使用严格轴对齐、ddof=0、stable tie 与常数规则；legacy 保留原表达式用于反事实。",
        "expected_impact": "同一表达式在 numpy Forge 和 pandas 下游逐元素等价，不再因编译器漂移制造干旱。",
        "implementation_scope": "forge/expression.py + 精确差分测试。",
        "verification": "ties/NaN/常数矩阵逐元素 allclose(equal_nan=True, atol=1e-12)。",
        "rollback_plan": "FUXI_FORGE_CS_COMPILE=legacy；或恢复 r0_baseline/forge/expression.py。",
        "landed_detail": "R1 编译器修复已落地，默认 enforce，legacy 可一键复现旧路径；确定性语义测试通过。",
        "landed_at": "2026-09-22",
        "landed_artifact": "research/factor_alchemy/forge/expression.py; research/factor_alchemy/tests/test_r1_semantics.py",
        "code_changed": True,
        "jq_validation_required": True,
        "cross_verified": True,
    }, {
        "proposal_id": "P-20260922-004",
        "title": "R1 LLM 上下文事务合并: 消除多次 receive_context 静默覆盖长期记忆",
        "category": "生成器修复",
        "domains": ["T3模块/通道"],
        "risk_level": "medium",
        "status": "landed",
        "proposal_change_kind": "behavior",
        "trigger": "ralph_loop 顺序注入成功模板、禁区、前沿卡、warning、motif、RAG；旧 receive_context 每次重建整个字典，最终 prompt 只剩最后一次注入。",
        "current_state": "日志显示已加载多个记忆源，但真正送给 LLM 的提示词丢失先前上下文，属于观测与执行不一致。",
        "proposed_change": "默认 merge，仅更新显式传入字段；显式空值可清单字段；replace=True 或 FUXI_LLM_CONTEXT_MODE=legacy 可回退。",
        "expected_impact": "成功模式、禁止方向、前沿与失败反馈可同时存在于最终 prompt。",
        "implementation_scope": "llm_generator.py + 最终 prompt marker 契约测试。",
        "verification": "四次顺序注入后最终 prompt 同时含六类 marker；legacy 测试复现覆盖。",
        "rollback_plan": "FUXI_LLM_CONTEXT_MODE=legacy；或恢复 r0_baseline/llm_generator.py。",
        "landed_detail": "R1 上下文事务合并已落地；省略参数不再清空已有字段，显式空值仍可精准清除。",
        "landed_at": "2026-09-22",
        "landed_artifact": "research/factor_alchemy/llm_generator.py; research/factor_alchemy/tests/test_r1_semantics.py",
        "code_changed": True,
        "jq_validation_required": False,
        "cross_verified": True,
    }],
    "r2": [{
        "proposal_id": "P-20260922-005",
        "title": "R2 防假评估契约: S1/S4/S5 显式状态、S4 伪 OOS 揭示与封存 holdout",
        "category": "评估治理",
        "domains": ["T1闸门/裁决", "T2数据口径"],
        "risk_level": "high",
        "status": "landed",
        "proposal_change_kind": "behavior",
        "trigger": "S4 缺独立 OOS 时复用 S1 并表述为通过；holdout 从已观测 S5 return_series 抽取；无数据与真实失败均被布尔 false/0 混写。",
        "current_state": "系统无法区分 valid fail、not_evaluated、data_insufficient、error，存在把未执行说成失败或通过的假整改空间。",
        "proposed_change": "各阶段输出 eval_status；S4 audit 明示 legacy_s1_proxy，strict 可拒绝；holdout 只接收 _sealed_holdout_returns；S5 缺测不计因子失败。",
        "expected_impact": "报告和统计不再用缺测填零，也不再把 S1 代理冒充独立 OOS。",
        "implementation_scope": "multi_stage_validator.py + ralph_loop.py + 契约测试。",
        "verification": "audit/strict 双轨、S5 缺测状态、holdout 输入来源代码契约。",
        "rollback_plan": "FUXI_S4_OOS_MODE=legacy 可恢复兼容表述；完整代码回退使用 r0_baseline 中对应文件。",
        "landed_detail": "R2 评估状态契约已落地。默认 audit 保持兼容决策但禁止称 OOS 通过；strict 可 fail-closed；未提供封存 holdout 时只记 not_evaluated。",
        "landed_at": "2026-09-22",
        "landed_artifact": "research/factor_alchemy/multi_stage_validator.py; research/factor_alchemy/ralph_loop.py; research/factor_alchemy/tests/test_r2_evaluation_contract.py",
        "code_changed": True,
        "jq_validation_required": True,
        "cross_verified": True,
    }],
    "r3": [{
        "proposal_id": "P-20260922-006",
        "title": "R3 搜索动作归因闭环: MAB 选定范式全种群约束与稳定性/换手 challenger",
        "category": "搜索治理",
        "domains": ["T1闸门/裁决", "T3模块/通道"],
        "risk_level": "high",
        "status": "landed",
        "proposal_change_kind": "behavior",
        "trigger": "MAB 选择一个范式，Forge 却遍历所有成熟 profile，奖励仍归到所选臂；旧目标仅 |ICIR|-复杂度-NaN，未约束时序稳定与换手代理。",
        "current_state": "动作与搜索空间不一致，奖励归因污染；单一总体 ICIR 容易选择阶段性和高换手表达式。",
        "proposed_change": "target_paradigm 约束整个种群，未知 profile fail-closed；候选记录 research_action_id；新增 stability_turnover challenger，生产默认暂保 legacy，待 A/B 后决定翻转。",
        "expected_impact": "MAB 奖励归因到真实动作；在不放宽 S5 的前提下提高候选稳定性并降低换手代理。",
        "implementation_scope": "forge/__init__.py + ralph_loop.py + 搜索契约测试。",
        "verification": "全种群范式一致、未知范式拒绝、稳定 IC 路径胜过符号反转路径；真实数据多 seed A/B。",
        "rollback_plan": "FUXI_FORGE_FITNESS=legacy（当前即默认）；完整回退恢复 r0_baseline/forge/__init__.py 与 ralph_loop.py。",
        "landed_detail": "R3 搜索动作身份与 challenger 已落地。新适应度未越过证据门槛前不翻 production 默认；换手仅声明为截面 rank turnover proxy，不冒充实际交易成本。",
        "landed_at": "2026-09-22",
        "landed_artifact": "research/factor_alchemy/forge/__init__.py; research/factor_alchemy/ralph_loop.py; research/factor_alchemy/tests/test_r3_search_contract.py",
        "code_changed": True,
        "jq_validation_required": True,
        "cross_verified": True,
    }],
}


def register(phase):
    wanted = PROPOSALS[phase]

    def modify(items):
        by_id = {item.get("proposal_id"): item for item in items}
        for proposal in wanted:
            if proposal["proposal_id"] in by_id:
                raise RuntimeError("proposal already exists: " + proposal["proposal_id"])
            proposal = dict(proposal)
            proposal["created_at"] = NOW
            items.append(proposal)
        if phase == "r2":
            existing = by_id.get("P-20260921-003")
            if not existing:
                raise RuntimeError("P-20260921-003 missing")
            existing.update({
                "status": "landed",
                "proposal_change_kind": "behavior",
                "domains": ["T1闸门/裁决", "T2数据口径"],
                "landed_at": "2026-09-22",
                "landed_detail": "R2 已按计算窗/评估窗分离修复：先加载至少 252 交易日 warmup，再仅在 2025-01-01 后统计；FUXI_YEARLY_IC_WINDOW=legacy 可做逐候选反事实。消费端缺年腿显式 data_insufficient，不再以 or 0 冒充观测值。",
                "landed_artifact": "research/factor_alchemy/factor_ic_computer.py; research/factor_alchemy/ralph_loop.py; research/factor_alchemy/tests/test_r2_evaluation_contract.py",
                "code_changed": True,
            })
        return items

    deadline = time.time() + 60
    fd = None
    while fd is None:
        try:
            fd = os.open(str(LOCK_PATH), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            if time.time() >= deadline:
                raise TimeoutError("proposal registry lock timeout: " + str(LOCK_PATH))
            time.sleep(0.5)
    try:
        os.write(fd, f"{os.getpid()} {NOW}".encode("ascii"))
        os.close(fd)
        fd = None
        with PROPOSALS_PATH.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        result = modify(data)
        temp = PROPOSALS_PATH.with_suffix(PROPOSALS_PATH.suffix + f".tmp.{os.getpid()}")
        with temp.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, PROPOSALS_PATH)
        with PROPOSALS_PATH.open("r", encoding="utf-8") as handle:
            json.load(handle)
    finally:
        if fd is not None:
            os.close(fd)
        try:
            LOCK_PATH.unlink()
        except FileNotFoundError:
            pass
    print("registered", phase, "total", len(result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=sorted(PROPOSALS), required=True)
    register(parser.parse_args().phase)
