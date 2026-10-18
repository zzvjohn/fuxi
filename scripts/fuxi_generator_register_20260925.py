"""Locked, idempotent registration of the generator repair and blocked study."""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from ledger_io import append_proposals  # noqa: E402

STAMP = datetime.now().strftime("%Y-%m-%d %H:%M")
REPORT = "reports/fuxi_generator_rebuild_20260925.md"
COMMON = {
    "date": "2026-09-25",
    "created_at": STAMP,
    "category": "管线修复",
    "domains": ["T3模块/通道", "T4数据/口径"],
    "proposal_change_kind": "behavior",
    "generator_hint": "manual",
    "jq_validation_required": False,
    "source": "SystemAudit",
}
ROWS = [
    {
        **COMMON,
        "proposal_id": "P-20260925-014",
        "title": "生成链确定性修复、可校验语义计划与研究特征晋级下锁",
        "trigger": "实路径审计发现 2/20 motif 被零成功公式误判 hard、GP 检索高等级头模板空公式且丢元数据、LLM 日/周 prompt 契约自相矛盾、退避混淆数据不足与经济失败。",
        "current_state": "生成链故障可造成零产出或误导搜索；现有特征/策略双层登记未携带新语义计划的目标用途。",
        "proposed_change": "精确二项上界；GP 父本公式解析与质量元数据保留；LLM 频率契约修复；S5 退避分状态；显式语义计划校验、注入和 JQ 单向下锁。",
        "expected_impact": "恢复可解析 GP 父本并减少假禁令；让计划实验可复现且研究特征不能冒充交易候选，不声称收益改善。",
        "implementation_scope": "research/factor_alchemy 的反馈、生成契约、LLM、Ralph、validator、语义计划与测试；不翻 S5/Forge 默认。",
        "risk_level": "medium",
        "rollback_plan": "旧文件字节快照见 reports/fuxi_generator_rebuild_20260925/baseline/；新模块可移除，语义计划默认 off；恢复须重跑测试及版本审计。",
        "cross_verified": True,
        "status": "implemented",
        "landed_detail": "反馈实时 hard 3→2；GP 父本 20 可解析、受控请求4产出4；语义计划显式开关并下锁研究特征；目标测试37/37，完整测试224中1项工作区外临时写失败；无 S5/JQ 新晋级。",
        "implementation_evidence": [
            REPORT,
            "research/factor_alchemy/tests/test_feedback_evidence.py",
            "research/factor_alchemy/tests/test_generation_contract.py",
            "research/factor_alchemy/tests/test_semantic_plan.py",
        ],
    },
    {
        **COMMON,
        "proposal_id": "P-20260925-015",
        "title": "同预算语义生成与单独目标函数检验的经济验收",
        "trigger": "修复搜索机制不等于找到成本后有效因子；需固定预算与候选集隔离检验。",
        "current_state": "两臂 prompt 与预算已冻结；沙盒 DeepSeek 连接拒绝，外发审批未通过；固定表达式目标函数复核仅3/6可评且无 S5经济读数。",
        "proposed_change": "在明确授权的模型通道下完成两臂等预算生成；锁定候选后按同窗同成本检验目标函数、特征增量与可交易策略，并走可信 S5/R5。",
        "expected_impact": "只能由有效净增量、可靠 S5 和新样本裁决是否脱旱，不以 prompt/IC 代理替代。",
        "implementation_scope": "scripts/fuxi_equal_budget_generation_20260925.py、scripts/fuxi_objective_isolation_20260925.py 与后续独立经济验收。",
        "risk_level": "medium",
        "rollback_plan": "实验产物隔离在 reports/fuxi_generator_rebuild_20260925/；停止调用并保留未完成状态，不动生产默认。",
        "cross_verified": False,
        "status": "partially_implemented",
        "landed_detail": "协议及目标函数固定集读数已落盘；真实两臂无模型响应、S5经济评价未运行，不能做晋级结论。",
        "remaining_scope": "具体 prompt 的外发授权或批准的本地模型；两臂真实候选；可信 S5 净收益与R5新样本。",
        "implementation_evidence": [
            REPORT,
            "reports/fuxi_generator_rebuild_20260925/equal_budget/protocol.json",
            "reports/fuxi_generator_rebuild_20260925/objective_isolation.json",
        ],
    },
]

if __name__ == "__main__":
    outcome = append_proposals(ROWS)
    print({key: outcome[key] for key in ("n_before", "n_after", "added",
                                          "skipped_existing", "lost_ids", "dup_ids", "eol")})
