"""Register verified drought-remediation batches using the existing atomic writer."""

import argparse

import register_remediation_20260922 as registry


def proposal(number, title, domains, scope, detail, rollback):
    return {
        'proposal_id': f'P-20260923-{number:03}', 'title': title,
        'category': '干旱根因整改', 'domains': domains, 'risk_level': 'high',
        'status': 'landed', 'proposal_change_kind': 'behavior',
        'trigger': 'v0.7.52 后因子干旱未解除，差分审计仍发现计算、字段、频率和证据链缺口。',
        'current_state': '旧实现不能保证被搜索、被评估和被报告的是同一个因子。',
        'proposed_change': detail, 'implementation_scope': scope,
        'expected_impact': '先恢复正确性与可归因性；不承诺产生 alpha，不降低 S5 门槛。',
        'verification': 'research/factor_alchemy/tests；reports/drought_20260923 结构化证据；详见主报告。',
        'rollback_plan': rollback, 'landed_detail': detail,
        'landed_at': '2026-09-23', 'landed_artifact': scope,
        'code_changed': True, 'jq_validation_required': number != 3,
        'cross_verified': True,
    }


BATCHES = {
    'numeric': [proposal(
        1, 'R1 全原语数值契约与真实字段约束：TS/CS 嵌套同义，禁用 vwap 成交量代理',
        ['T1闸门/裁决', 'T2数据口径'],
        'research/factor_alchemy/forge/primitives.py; research/factor_alchemy/forge/expression.py; research/factor_alchemy/ralph_loop.py',
        '固定满窗缺失、ddof、并列排名、负分母收益率、EMA 初始化与逐节点非有限值规则；常数按面板广播。'
        '常数同时适配真实 Series 消费端；数据依赖窗口在两端均拒绝，禁止以全样本均值决定窗口。'
        '默认数值/字段契约 enforce；新生成不再把 vwap 当 volume，缺真实 OHLC 不再用 close 代替。'
        '72 个算子操作数组合、嵌套和随机表达式差分以及因果扰动覆盖，不把这些测试称为收益验证。',
        'FUXI_FORGE_NUMERIC_CONTRACT=legacy 与 FUXI_FORGE_FIELD_CONTRACT=legacy 提供兼容路径；'
        '精确回退恢复 reports/drought_20260923/baseline 下同名代码。不得恢复历史数据/候选状态。')],
    'routing': [proposal(
        2, 'R2 S5 周频路由、缺失信号、基准实际日期与预检 AST 契约',
        ['T1闸门/裁决', 'T2数据口径'],
        'research/factor_alchemy/s5_joint_filter.py; research/factor_alchemy/weekly_panel.py; research/factor_alchemy/llm_generator.py; scripts/s5_fullpool_rerun.py',
        '周频 enforce 失败不得改成日频；周缺失不得无限延用旧信号，全缺失成交量保留 NaN。'
        '价格代理检查按 AST 判断，科学记数小常数不误报价格减一；pandas 方法不误报裸 DSL。'
        'S5 组合与基准按实际收益日期一致裁剪，缺基准 fail-closed。'
        '普通单表达式 rank 默认明确为截面轴，保留 rolling/expanding rank，禁止全时序 axis=0 排名。'
        '周频加载失败不得持久修改后续候选频率；新语义开关进入续跑指纹和 A/B 文件名。'
        '默认推广状态以四关证据和主报告为准。',
        '精确回退使用 baseline 与 caliber_baseline 快照；诊断兼容开关 SANITY_AST=legacy、'
        'WEEKLY_CHANNEL=shadow、RETURN_ALIGN=off、RANK_AXIS=legacy、WEEKLY_MISSING_CONTRACT=legacy'
        '（均加 FUXI_ 或 FUXI_S5_ 前缀）；精确回退以快照为准。')],
    'experiment': [proposal(
        3, 'R4/R5 隔离实验账本：失败分层重放、真实 LLM、固定组合与前瞻冻结协议',
        ['T3模块/通道'],
        'scripts/fuxi_drought_campaign_20260923.py; scripts/version_bump.py; reports/drought_20260923/protocol.json; reports/drought_20260923/r4_protocol.json',
        '冻结 8 条历史公式、3 次真实 LLM 请求和 6 个组合/消融臂，总候选预算 20。'
        '原式/广播修复/频率和 0/15/30bps 串行诊断，逐成本断点、数据与核心代码指纹。'
        'ridge 仅使用 2024 年底前可得标签；2025/2026 标注研究暴露；无自动候选入库或 JQ 晋级。'
        'R5 前瞻样本尚待积累，协议冻结不是独立验证通过。版本生成器保留 non_bump 研究补记。',
        '停止调用本批实验脚本即可停用；恢复 baseline/scripts/version_bump.py 可回退账本展示。'
        '保留所有报告和实验结果，不改写已发生的研究事实。')],
}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=sorted(BATCHES), required=True)
    phase = parser.parse_args().phase
    registry.PROPOSALS = BATCHES
    registry.register(phase)
