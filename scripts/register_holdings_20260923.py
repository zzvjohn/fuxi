"""Register the verified holdings-accounting repair, without promoting factors."""

import register_remediation_20260922 as registry


PROPOSAL = {
    'proposal_id': 'P-20260923-004',
    'title': 'S5 持仓记账独立于信号：独立日历、缺报价固定分母与错误状态优先',
    'category': '持仓记账修复', 'domains': ['T1闸门/裁决', 'T2数据口径'],
    'risk_level': 'high', 'status': 'landed', 'proposal_change_kind': 'behavior',
    'trigger': '用户要求先修持仓收益记账，再审慎审理因子干旱原因。',
    'current_state': '旧信号价格内连接会删除持仓收益、移动调仓日历；缺报价会静默缩小分母。',
    'proposed_change': 'FUXI_S5_HOLDINGS_MARK 默认 enforce：市场价格日历决定调仓及计价日期，'
                       '信号仅决定选股；调仓缺信号继续记录已有持仓但覆盖率不冒充通过；'
                       '无法观测的持仓收益保留 NaN，不删股重加权或补零；裁决年存在缺口则 data_insufficient。'
                       '上游状态传递错误优先于耗时/样本量，保留 legacy 反事实和指纹。',
    'implementation_scope': 'research/factor_alchemy/s5_joint_filter.py; '
                            'research/factor_alchemy/multi_stage_validator.py; scripts/s5_fullpool_rerun.py; '
                            'research/factor_alchemy/tests/test_holdings_mark.py',
    'expected_impact': '恢复可归因性，防止把不可计价当作有效业绩；不保证 alpha，不放宽阈值。',
    'verification': '53 项测试；四个预冻结信号、12 臂串行 A/B；四条 legacy 收益精确对齐 v55；'
                    '正式周频入口 1354 日（含明确 NaN）对账；独立复建 165 条裁决窗异常持仓记录。',
    'rollback_plan': '诊断可显式 FUXI_S5_HOLDINGS_MARK=legacy；精确回退三处生产文件使用 '
                     'reports/holdings_20260923/baseline 与 consumer_baseline 同名文件。'
                     '不得恢复旧数据/运行状态，不得把旧记账路径用于新的正式晋级。',
    'landed_detail': '仅关闭信号/行情/收益错误耦合，不是完整固定股数及次日成交资金账本。'
                     '四个代表信号均 data_insufficient，新 JQ 资格 0；不能称四个经济有效失败。'
                     '2025/2026 合计 165 候选股日缺口：140 已知全天停牌、23 未分类缺报价、2 异常守卫；'
                     '停复牌表仅至 2026-04-30。工程缺陷与预测/组合目标不匹配优先于无约束换模型。',
    'landed_at': '2026-09-23',
    'landed_artifact': 'reports/fuxi_holdings_and_drought_review_20260923.md; reports/holdings_20260923',
    'code_changed': True, 'jq_validation_required': True, 'cross_verified': True,
}


if __name__ == '__main__':
    registry.PROPOSALS = {'holdings': [PROPOSAL]}
    registry.register('holdings')
