"""Seal the three preserved research phases and write the decision-facing report."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from fuxi_v1.contracts import load_json,require
from fuxi_v1.datasets import file_hash
from fuxi_v1.artifacts import code_manifest
from fuxi_v1.runtime import controlled_writer
OUT=ROOT/'reports/fuxi_v1_concentration_20261004'
DELIVERY=ROOT/'deliverables/fuxi_concentration_20261004'


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2)


def main():
    plan=load_json(OUT/'PLAN.json');tests=load_json(OUT/'tests/TEST_RESULT.json');phases={}
    runtime=load_json(ROOT/'codex_automations/fuxi_runtime_policy.json')
    require(runtime['current_mode']=='controlled_manual' and not runtime['autonomous_writes_enabled'],'runtime policy changed')
    require(code_manifest()==plan['code_hashes']==tests['code_hashes'],'core differs from tested/frozen code')
    for name in ('attempt1','fallback_diagnostic','closure_diagnostic'):
        result=load_json(OUT/name/'OUTCOME.json');check=load_json(OUT/name/'INDEPENDENT_VERIFICATION.json')
        require(check['verified'] and check['outcome_sha256']==file_hash(OUT/name/'OUTCOME.json'),'phase verification missing')
        phases[name]=result
    bundle=load_json(DELIVERY/'EXCLUSIONS.json');delivery=load_json(OUT/'EXCLUSION_DELIVERY_CHECK.json')
    require(delivery['verified'] and delivery['bundle_sha256']==file_hash(DELIVERY/'EXCLUSIONS.json'),'exclusion check missing')
    rules=load_json(OUT/'FRACTIONAL_RULES_v2.json');retry=load_json(OUT/'source_revision1/RESULT.json')
    require(rules['rules']==load_json(OUT/'FRACTIONAL_RULES_v3.json')['rules'],'unexpected unused source rules')
    initial=load_json(OUT/'fractional_sources/RESULT.json')
    status={'schema_version':'fuxi.concentration_completion.1','completed_unique_factor_concentration_hypotheses':8,
        'phase_count':3,'artifact_configurations':36,'pressure_scenario_comparisons_attempted':108,
        'phases':{name:{'arms':r['arms_completed'],'identified_arm_scenarios':r['identified_arm_scenarios'],
            'robust_positive_arms':r['robust_positive_arms'],'remaining_gap_instruments':r['root_instruments'],
            'posthoc_diagnostic_only':r['posthoc_diagnostic_only']} for name,r in phases.items()},
        'source_supported_conditional_rules':len(rules['rules']),'core_tests':tests['tests'],
        'new_prospective_exclusions':bundle['new_blocked_instruments'],'total_prospective_exclusions':len(bundle['exclusions']),
        'actual_account_allocation_receipts':0,'formal_JQ':0,'actual_platform_runs':0,'live':0,
        'software_ready':False,'paper_release_ready':False,'drought_broken':False,
        'legacy_S5_defaults_and_thresholds_changed':False,'core_hashes':code_manifest(),
        'runtime_policy_sha256':file_hash(ROOT/'codex_automations/fuxi_runtime_policy.json'),
        'research_exposed_years':[2025,2026],'source_rule_revision_did_not_change_rules':True}
    lines=['# 伏羲 V1.0：逐事件账户分支与八臂净增量检验（2026-10-04）','',
        '本批完成八个冻结的因子/集中度假设、四个 top10 对照、固定排除13只的诊断及排除20只的最后闭合诊断。两次排除修改的是历史研究范围，均为事后配置，不能算独立新试验或改善正式显著性；八个唯一公式/集中度身份保持不变。没有新增正式 JQ 资格，也未突破因子干旱。历史账户、原始数据、首次失败与首轮实验均保留。', '',
        '## 样本、成本与检验合同','',
        '- 原始全市场 cohort 5621 只；输入 2025-01-02..2026-09-30，研究从 2025-06-11 开始；321 个研究净值收益配对。2025/2026 均已研究暴露。',
        '- 四个原 E1 通过候选：low_price_cv20、intraday_reversal、low_range、low_turnover_surprise。公式、方向、原始截面评分、momentum20 基准与 50/50 排名混合冻结。',
        '- 新轴只有 top30（单股上限 1/30）和 top50（1/50）。候选与基准使用相同集中度；每五个交易日调仓。保留原 top10 对照，不扩充候选或调用 LLM。',
        '- 初始资金 100 万元；佣金 0.03%、最低 5 元，滑点 5bps，卖出印花税 0.05%，前一交易日成交量 1% 容量，分红税固定 20%。正常、双成本、半容量三情景不变。',
        '- 原固定 nominal 95% 区块 bootstrap、3999 次、block=12，净增量日收益下界须严格大于 0；不降低成本或资格阈值。表中下界为日收益基点，不是年化收益。',
        '- E1 数值和全池消融只保留为原假设依据；本批不重做剔除池 E1/E2，不把历史研究暴露变成独立验证。分支 CI 未做多重检验修正，也不授正式资格。', '',
        '## 账户处理','',
        '每遇到原文明确支持的零碎股事件，分别计算 floor/ceil。每条分支从原初始账本重放；之后的股份、成交、费用、现金、下一笔权益和 NAV 全部重新计算，不能只对最终净值加减一股。每个角色/压力情景最多 128 条完整路径，超预算或仍有缺证据即保留 unknown。候选×基准使用所有完整分支的笛卡尔外界，不假定概率或实际分配；全体其他股东约束仍未知。', '',
        '已验证 22 条源绑定普通转增规则。001211 的 2026-06-04 与 688215 的 2026-06-10 可做条件分支；688215 的 2025-07-11、688557 的 2026-07-09、920699 的 2026-07-10 仍没有建立明确分配条款。实际投资者到账凭证为 0，因此条件分支通过也不等于精确账户认证。', '',
        '未实际持有股票的假设研究，不要求用户提供个人结算单。可靠规则的路径边界可以用于稳健研究；当前边界模块尚未授予独立投资数据认证。真实已有持仓则仍需实际券商股份/现金对账。', '',
        f"原四股票五事件补采：{initial['physical_calls']} 次成功供应商调用、{sum(len(r['documents']) for r in initial['rows'])} 份下载记录；首轮实际新增缺口补采：{retry['physical_calls']} 次调用、{sum(len(r['documents']) for r in retry['rows'])} 份下载记录，新增规则 0。官方上海登记结算指南读取未返回 PDF，失败凭证保留；不把缺源改写成已验证算法。", '',
        '## 实际读数','',
        '| 批次 | 新臂 | 可辨识压力情景 | 三压力下界均正 | 正式 JQ |',
        '| --- | ---: | ---: | ---: | ---: |']
    for name,r in phases.items():lines.append(f"| {name} | 8 | {r['identified_arm_scenarios']}/24 | {r['robust_positive_arms']}/8 | 0 |")
    lines+=['','八个唯一公式/集中度假设，三个数据处理批次；合计 36 个配置产物（每批 8 臂+4 对照）、108 个压力比较尝试。历史池的两次修改均额外披露，不计独立发现或增加显著性证据。','']
    for name,r in phases.items():
        lines+=['### '+name,'','| 候选 | TopN | 正常下界 bps/日 | 双成本 | 半容量 | 可辨识情景 |','| --- | ---: | ---: | ---: | ---: | ---: |']
        for row in r['rows']:
            values=['unknown' if s['comparison']['lower'] is None else f"{s['comparison']['lower']*10000:.4f}" for s in row['scenarios']]
            identified=sum(s['candidate_identifiable'] and s['baseline_identifiable'] for s in row['scenarios'])
            lines.append(f"| {row['name']} | {row['top_n']} | {' | '.join(values)} | {identified}/3 |")
        lines+=['','实际仍阻断的股票：'+('、'.join(r['root_instruments']) or '无')+'。','']
    excluded=phases['fallback_diagnostic']['excluded']
    lines+=['## 剔除边界与前向入口','',
        f"fallback_diagnostic 固定排除首轮 {len(excluded)} 只："+'、'.join(excluded)+'。这是从空初始账户另开的假设账本；原始持仓、原账户、原市场事件和历史结果不改。已混合的原始截面评分只过滤禁入股票，不重新按剔除池排名。', '',
        'closure_diagnostic 按前两批纯数据缺口并集固定20只，达到冻结计划的历史诊断排除上限；结果出炉前已冻结，不再扩大历史池排除范围。此批同样为另一个事后假设账本，不把20只清单当原始事前股票池。', '',
        '排除依据来自已观察的历史缺口，具有选择偏差；任何读数都只能诊断，不能晋级。剩余新缺口也保留 unknown，不继续改变历史股票池追求通过。未知账户不当成经济失败。', '',
        f"前向清单新增 {bundle['new_blocked_instruments']} 只，含既有 600077 共 {len(bundle['exclusions'])} 只；仅在 observed/effective 时间后阻止新建仓，已有持仓/权益保留，七日后复核。复核日不会自动解除。", '',
        '具体消费者为 deliverables/fuxi_concentration_20261004/decision_with_exclusions.py，调用原生 daily_decision(exclusions=...)。已检查观察时间前决策不变、生效后禁入且保留已有权重、拒绝策略绕过清单。仅此交付入口消费清单，旧 S5 默认不改；这是原生研究入口，不是聚宽/QMT平台运行证据。', '',
        '## 验证、问题及回退','',
        '236 项核心回归通过（旧 231 项+5 项分支/真实执行器/源绑定检查）；每条完成分支独立重放现金、股份、应收、待到账与 NAV。三批产物 SHA/冻结哈希/路径数量/源归属/所有外界和代表固定种子 CI 复核均通过。前向消费者另有三项实际调用合同检查，全部使用合成行情，不冒称平台实测。', '',
        '首次运行因 accounts 子目录未创建而未能写出第一个对照产物；修复前置目录并记录 RECOVERY 后重放，冻结代码不改。后续使用 fuxi_concentration_resume_20261004.py 创建目录并调用原封不动的冻结 runner。', '',
        'core 1.0.0-rc12，版本账本用项目 --bump 登记；controlled_manual、单一 runtime writer，自动写入未开启。回退时停止分支回调/研究缓存和新清单入口，按 before_core 恢复 __init__.py/action_accounting.py/simulation.py；保留所有事件、补采、失败和账本，再独立对账。', '',
        '本批仍不能宣称软件/模拟发布/实盘工程化验收通过：存在账户主证据与历史公司行动缺口，净增量未得到稳健支持，独立数据认证、真实 JQ/QMT 平台与未来 E5 尚未完成。继续扩充同窗历史参数不能补足这些条件。', '',
        '下一步保持这八臂结论冻结：先用正式结算/券商权益单或可验证原文补足保留账户，再从前向禁入生效后另立 PIT 合同和未知样本；若研究新机制，必须另行事前注册并验证单通道归因、成本后增量，不能降低资格门槛。', '',
        '事实入口：PLAN.json、三批 OUTCOME/INDEPENDENT_VERIFICATION、FRACTIONAL_RULES_v2/v3、tests/TEST_RESULT、EXCLUSION_DELIVERY_CHECK，以及交付 EXCLUSIONS.json。']
    report='\n'.join(lines)+'\n';(OUT/'REPORT.md').write_text(report,encoding='utf-8',newline='\n')
    mirror=ROOT/'research/fuxi_v1_20261004/CONCENTRATION_AND_FRACTIONAL_REVIEW.md';mirror.parent.mkdir(parents=True,exist_ok=True);mirror.write_text(report,encoding='utf-8',newline='\n')
    save(OUT/'COMPLETION.json',status)
    manifest={'core_hashes':code_manifest(),'files':{str(path.relative_to(ROOT)):file_hash(path) for path in [OUT/'PLAN.json',OUT/'REPORT.md',OUT/'COMPLETION.json',mirror,DELIVERY/'EXCLUSIONS.json',DELIVERY/'decision_with_exclusions.py',DELIVERY/'README.md',OUT/'EXCLUSION_DELIVERY_CHECK.json']},
        'formal_JQ':0,'actual_platform_runs':0,'live':0}
    save(OUT/'FINAL_MANIFEST.json',manifest)
    print(json.dumps({k:v for k,v in status.items() if k!='core_hashes'},ensure_ascii=False),flush=True)


if __name__=='__main__':
    with controlled_writer(OUT):main()
