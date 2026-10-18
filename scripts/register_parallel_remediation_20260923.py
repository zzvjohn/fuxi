"""Register the parallel evidence/diagnostics batch without promoting signals."""
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from fuxi_execution_lab_20260923 import read, write, digest
from register_execution_followup_20260923 import append_preserving
import register_remediation_20260922 as registry

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'research/factor_alchemy'))
from run_contract import FuxiWriterLock
OUT=ROOT/'reports/window_evidence_20260923'


def main():
    assert not (OUT/'release_audit.json').exists()
    p=read(OUT/'protocol.json')
    audit=read(OUT/'request_audit.json')
    assembly=read(OUT/'issuer_overlay_assembly.json')
    market=read(ROOT/'reports/window_execution_20260923/followup_v3/market.json')
    previous_tests=read(OUT/'regression.json')
    second_tests=read(OUT/'followup_regression.json')
    tests=read(OUT/'issuer_regression.json')
    assert previous_tests['successful'] and second_tests['successful']
    tests['code_sha256']={**previous_tests['code_sha256'],**second_tests['code_sha256'],**tests['code_sha256']}
    assert tests['successful'] and tests['tests_run']>0
    assert audit['responses_checked']==p['max_base_requests'] and not audit['invalid_requests']
    assert len(market['rows'])==4 and not market['formal_eligibility']
    assert all(r['reference']['status']=='matched' and r['reference']['days']>0 for r in market['rows'])
    comparison=read(ROOT/'reports/window_execution_20260923/followup_v3/followup_comparison.json')
    assert len(comparison['rows'])==4 and all(r['v2_unchanged'] for r in comparison['rows'])
    assert sum(r['v2_compared_days'] for r in comparison['rows'])==320
    second_assembly=read(OUT/'followup_assembly.json')
    original_assembly=read(OUT/'assembly_distribution.json')
    assert assembly['parent_assembly_sha256']==digest(OUT/'followup_assembly.json')
    assert second_assembly['prior_assembly_sha256']==digest(OUT/'assembly_distribution.json')
    assert original_assembly['source_assembly_sha256']==digest(OUT/'assembly.json')
    for name,expected in {**second_assembly['proof_sha256'],**assembly['additional_proof_sha256']}.items():
        assert digest(OUT/name)==expected,name
    for name,expected in tests['code_sha256'].items():
        assert digest(ROOT/name)==expected,name
    for name,expected in p['protected'].items():
        assert digest(ROOT/name)==expected,name
    for entry in audit['rows']:
        assert digest(OUT/'raw'/(entry['id']+'.json'))==entry['sha256']
    report=ROOT/'reports/fuxi_parallel_remediation_20260923.md'
    assert report.exists()
    for folder in [OUT,ROOT/'reports/window_execution_20260923',ROOT/'reports/target_diagnostics_20260923']:
        for path in folder.rglob('manifest.json'):
            assert read(path)['status']!='running',str(path)
    independent=[{'signal':r['signal'],**r['reference']} for r in market['rows']]
    matched=sum(r.get('days',0) for r in independent if r['status']=='matched')
    assert matched>0
    states='；'.join(r['signal']+'='+r['status'] for r in market['rows'])
    note=('\n\n## M20260923.7 持仓证据补采与目标错配并行诊断\n\n'
          '用户明确要求缺数据主动补采、第一/第二项并行，不以局部缺口暂停整个工作；'
          '五阶段协议追加本次优先指令。完整验收仍依赖足够证据，隔离诊断可先行。\n\n'
          f"冻结{audit['responses_checked']}请求，审计{audit['positive_response_rows']}原始响应行；"
          f"批准现金{len(assembly['approved_cash_actions'])}笔及送转{len(assembly['approved_share_actions'])}笔。"
          '六笔发行人公告只用于精确事件缺失送转组成规范化，另核宁德时代及江南奕帆回购股差异化除息/除权；'
          '231组仅方案公告日不同的唯一合同安全折叠，冲突不合并、共同空值不补零；重整转增不推断为普通持有人权益；'
          '空响应不是无事件证明，原始缓存不覆盖。\n\n'
          f"{tests['tests_run']}项回归通过；新独立Decimal参考认证{matched}有限日期位置；{states}。"
          '600898.SH摘牌后确认为400257、74条未复权转让行情，摘牌至首笔转让之间仍无可信估值，'
          '首日全市场693手小于所持124000股，不虚构全额卖出。'
          '839680.BJ至920680.BJ已核2025-10-09切换，仅应用这一证明映射。\n\n'
          '第二项完成4信号×0/15/30bps真实账本重算，2024-12-30至2025-01-17同窗14日，'
          '15bps 56位置对账；三条零费仍负。87个历史周截面平均IC均正，LLM2/R4顶部减池标签均值'
          '分别-35.85/-19.18bps，提示预测层顶部目标错配；标签不等于可成交净收益。'
          '424222重复规模键均为有限正同值，证明后安全折叠，规模87/87截面恢复；'
          '行业PIT及规模历史版本仍未认证。诊断4测试、规模1测试，重复测试不累加。\n\n'
          '2025/2026历史暴露不变，未完成五项全验收、未新增JQ、未开展R5未知样本裁决。'
          '生产S5、默认、反馈、候选、原始数据保持不变；Codex仅controlled_manual。'
          '第一轮195有限日期/5019成交、第二轮320日期/7745成交对拍保留；本段引用followup_v3结果。'
          '派特尔公告已取得，但旧代码身份时点及792.1股理论转增的账户整数分配仍未证明，不默认取整。'
          '详见reports/fuxi_parallel_remediation_20260923.md及reports/fuxi_target_diagnostics_20260923.md。\n')
    proposal={'proposal_id':'P-20260923-007','title':'持仓全窗事件补采与固定基线目标错配并行诊断',
      'category':'执行证据与目标诊断','domains':['T2数据口径','T3模块/通道'],
      'risk_level':'high','status':'landed','proposal_change_kind':'shadow',
      'trigger':'用户明确要求主动补采，并行开展第二项，局部数据缺口不阻断其他工作。',
      'current_state':'四个冻结信号账本存在事件与证券生命周期缺口，完整第一阶段尚未验收。',
      'proposed_change':'日期批采事件/停复牌/涨跌停补缺，逐事件审批并在隔离账本与独立参考对拍，同时定位预测顶部/费用/换手/规模。',
      'implementation_scope':'scripts/fuxi_window_evidence_20260923.py; scripts/fuxi_distribution_evidence_20260923.py; scripts/fuxi_window_replay_20260923.py; scripts/fuxi_window_reference_20260923.py; scripts/fuxi_target_diagnostics_20260923.py',
      'expected_impact':'补充可审计证据并定位目标错配，不接管生产或改变候选资格。',
      'verification':f"{tests['tests_run']}回归；{audit['responses_checked']}响应身份/内容/预算审计；{matched}独立参考有限日期；12成本臂、87预测截面、87规模截面；诊断4+1测试。",
      'rollback_plan':'停用本批隔离入口；保留原始数据与历史证据；baseline及release保存版本/报告快照。',
      'landed_detail':states+'；整窗、行业PIT、R5仍未验收；第二项隔离结果不作经济晋级。',
      'landed_at':'2026-09-23','landed_artifact':'reports/fuxi_parallel_remediation_20260923.md',
      'code_changed':True,'jq_validation_required':True,'cross_verified':True}
    memory=ROOT/'.workbuddy/memory/MEMORY.md'
    memory_before=memory.read_bytes()
    memory_backup=OUT/'baseline/.workbuddy/memory/MEMORY.md'
    assert not memory_backup.exists()
    memory_backup.write_bytes(memory_before)
    with FuxiWriterLock(run_id='parallel_register_'+datetime.now().strftime('%H%M%S_%f')):
        registry.NOW=datetime.now().astimezone().isoformat()
        registry.PROPOSALS={'parallel_remediation':[proposal]}
        registry.register('parallel_remediation')
        append_preserving(ROOT/'.workbuddy/memory/2026-09-23.md',note)
        append_preserving(ROOT/'.workbuddy/memory/_fuxi_landing_details.md',note.replace('## M20260923.7','## 附录 W — 2026-09-23 第七批'))
        subprocess.run([sys.executable,'-X','utf8','scripts/version_bump.py','--bump'],cwd=ROOT,check=True)
        subprocess.run([sys.executable,'-X','utf8','scripts/version_bump.py','--check'],cwd=ROOT,check=True)
        current_version=read(ROOT/'data/version_registry.json')['versions'][-1]['version']
        old_line=next(line for line in memory_before.split(b'\r\n') if line.startswith(b'- v0.7.57/'))
        new_line=(f'- {current_version}：持仓补采与基线并行；100测试，隔离账本未接管S5。附录W及'
                  'reports/fuxi_parallel_remediation_20260923.md；局部缺口不阻断其他诊断。仅手动受控、定时写入关。').encode('utf-8')
        memory_after=memory_before.replace(old_line,new_line)
        assert sum(not c.isspace() for c in memory_after.decode('utf-8'))<=3000
        memory.write_bytes(memory_after)
    allowed={'data/version_registry.json','data/stage1_optimization_proposals.json'}
    for name,expected in p['protected'].items():
        if name not in allowed:assert digest(ROOT/name)==expected,name
    old=read(OUT/'baseline/data/stage1_optimization_proposals.json')
    new=read(ROOT/'data/stage1_optimization_proposals.json')
    assert new[:len(old)]==old and len(new)==len(old)+1
    for name in ['.workbuddy/memory/2026-09-23.md','.workbuddy/memory/_fuxi_landing_details.md']:
        before=(OUT/'baseline'/name).read_bytes();after=(ROOT/name).read_bytes()
        assert after.startswith(before)
        if b'\r\n' in before:
            assert after.replace(b'\r\n',b'').count(b'\n')==before.replace(b'\r\n',b'').count(b'\n')
        else:assert b'\r\n' not in after
    version=read(ROOT/'data/version_registry.json')['versions'][-1]['version']
    names=['scripts/register_parallel_remediation_20260923.py','scripts/version_bump.py','CHANGELOG.md',
           'scripts/fuxi_window_evidence_20260923.py','scripts/fuxi_target_diagnostics_20260923.py',
           'reports/fuxi_parallel_remediation_20260923.md','reports/fuxi_target_diagnostics_20260923.md',
           'reports/delisting_evidence_20260923.md','research/fuxi_five_stage_acceptance_20260923.md',
           '.workbuddy/memory/2026-09-23.md','.workbuddy/memory/_fuxi_landing_details.md',*allowed,
           '.workbuddy/memory/MEMORY.md',
           *tests['code_sha256']]
    for name in set(names):
        target=OUT/'release'/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/name,target)
    folders=[OUT,ROOT/'reports/window_execution_20260923',ROOT/'reports/target_diagnostics_20260923',ROOT/'reports/delisting_evidence_20260923']
    hashes={p.relative_to(ROOT).as_posix():digest(p) for folder in folders for p in folder.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    write(OUT/'release_audit.json',{'version':version,'proposal_id':proposal['proposal_id'],
          'stage1_complete':False,'stage2_complete':False,'five_stages_complete':False,
          'formal_eligibility':False,'protected_inputs_unchanged':True,
          'regression_tests':tests['tests_run'],'independent_reference':independent,'evidence_sha256':hashes})
    print(json.dumps({'version':version,'sealed':True,'files':len(hashes),'independent_days':matched}))


if __name__=='__main__':main()
