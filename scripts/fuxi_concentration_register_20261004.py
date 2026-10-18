"""Append the completed batch to research memory and the project version ledger."""
import json,sys,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from fuxi_v1.contracts import load_json,require
from fuxi_v1.datasets import file_hash
from fuxi_v1.runtime import controlled_writer
OUT=ROOT/'reports/fuxi_v1_concentration_20261004'


def append(path,text):
    original=path.read_bytes();addition=('\r\n\r\n'+text.rstrip()+'\r\n').replace('\r\n','\n').replace('\n','\r\n').encode('utf-8')
    with path.open('ab') as f:f.write(addition)
    require(path.read_bytes().startswith(original),'historical memory changed')


def main():
    final=load_json(OUT/'COMPLETION.json');manifest=load_json(OUT/'FINAL_MANIFEST.json')
    for name,expected in manifest['files'].items():require(file_hash(ROOT/name)==expected,'sealed artifact changed')
    pid='P-20261004-001';ledger=ROOT/'data/stage1_optimization_proposals.json';proposals=load_json(ledger)
    require(not any(p['proposal_id']==pid for p in proposals),'proposal already registered')
    counts='、'.join(f"{name} {v['identified_arm_scenarios']}/24" for name,v in final['phases'].items())
    proposal={'proposal_id':pid,'title':'伏羲V1.0 rc12逐事件零碎股分支与冻结八臂净增量诊断',
        'category':'账户证据与前向禁入工程','trigger':'用户授权保留持仓、逐事件分支，无法解决时临时剔除；四候选top30/top50最多八个新公式/集中度假设。',
        'current_state':'零碎股阻断、分支后续成交/权益未重算；分散持仓净增量机制仅记录未运行。',
        'proposed_change':'源绑定floor/ceil树、预算溢出保留未知、真实执行器独立现金股份NAV重放；仅研究缓存；四top10对照和八新集中度臂；原始/排除13/排除20三批保留；按实际观察时间新增前向禁入并验证原生消费入口。',
        'expected_impact':'区分账户不可辨识与经济不稳健，保留原账户和权益；不降低阈值，不授资格或实盘权限。',
        'implementation_scope':'fuxi_v1 action_accounting/simulation/fractional_sources/fractional_branches; dated scripts, reports/research and native exclusion deliverable; 旧S5默认不改。',
        'landed_detail':f"236测试、三项清单消费者检查；22条条件规则；八唯一公式/集中度身份、36配置产物、108压力比较尝试；{counts}；净增量稳健新臂0，正式JQ0；新增前向禁入{final['new_prospective_exclusions']}只。历史池修改均为事后诊断。",
        'generator_hint':'manual','risk_level':'controlled_research_only',
        'rollback_plan':'停止分支/缓存/新清单入口，以reports/fuxi_v1_concentration_20261004/before_core恢复__init__.py/action_accounting.py/simulation.py；新增模块不引用，所有原始数据、源、失败、账户及权益保留，恢复前独立对账。',
        'jq_validation_required':True,'jq_validation_status':'actual_platform_not_observed_no_formal_admission',
        'references':['reports/fuxi_v1_concentration_20261004/REPORT.md','reports/fuxi_v1_concentration_20261004/COMPLETION.json',
            'research/fuxi_v1_20261004/CONCENTRATION_AND_FRACTIONAL_REVIEW.md','deliverables/fuxi_concentration_20261004/README.md'],
        'cross_verified':False,'verification_scope':'native path/account arithmetic, hash-bound public sources and synthetic causal consumer; no independently certified whole data/account or actual platform',
        'status':'landed','source':'UserAuthorized','proposal_change_kind':'behavior','domains':['T1闸门/裁决','T2数据口径','T3模块/通道']}
    proposals.append(proposal);raw=json.dumps(proposals,ensure_ascii=False,indent=2)+'\n';ledger.write_bytes(raw.replace('\n','\r\n').encode('utf-8'))
    text='\n'.join(['## AX. 伏羲rc12逐事件零碎股账户与八臂净增量（2026-10-04）','',
        '- 保留历史账户，源绑定逐事件floor/ceil、后续股份/成交/现金/权益/NAV全部重算；只研究缓存，预算128/角色情景，未知或溢出不授读数。22规则；001211及688215的2026事件可条件分支，2025的688215/688557/920699仍缺明确分配条款。',
        '- 236测试及三项前向消费者合同检查。四E1候选×top30/50=八唯一公式/集中度身份；四top10对照。原始、排除13、排除20三批均保留，36配置/108压力比较尝试；历史池改变额外披露，不是独立新发现。可辨识读数：'+counts+'。稳健新臂0、正式JQ/实际平台/实盘0。',
        '- 样本2025-01-02..2026-09-30、研究2025-06-11起、321净收益配对，2025/2026均暴露；原成本和资格门槛不降，基准与候选匹配集中度。未知账户不当经济失败；剔除不改原持仓，固定原始混合排名只过滤不重排。',
        f"- 前向禁入新增{final['new_prospective_exclusions']}只，含600077共{final['total_prospective_exclusions']}只；实际观察时间生效、七日复核不自动解禁、已有权益保留。仅交付原生decision_with_exclusions入口消费，旧S5默认/阈值不改；真实券商已有持仓需对账，未持有假设股票不要求用户个人结算单。",
        '- source补采五事件6调用/8下载记录，新增缺口9调用/7记录、规则净新增0；指南非PDF失败、首次缺accounts目录失败及恢复均保留。最后达到20只历史排除上限，剩余未知保留，不扩大池追求通过。',
        '- reports/fuxi_v1_concentration_20261004/REPORT.md及三批OUTCOME/INDEPENDENT_VERIFICATION、COMPLETION、FINAL_MANIFEST为事实。software_ready/paper_release_ready均False；净增量无稳健支持、独立主证据、真实平台与未来E5仍缺。controlled_manual/自动写入关闭。登记P-20261004-001，版本由--bump实际返回。'])
    for name in ('.workbuddy/memory/_fuxi_landing_details.md','.workbuddy/memory/2026-10-04.md'):
        path=ROOT/name;require('## AX. 伏羲rc12' not in path.read_text(encoding='utf-8'),'batch already logged');append(path,text)
    outputs=[]
    for flag in ('--bump','--render','--check'):
        result=subprocess.run([sys.executable,'-X','utf8',str(ROOT/'scripts/version_bump.py'),flag],cwd=ROOT,capture_output=True,text=True,encoding='utf-8')
        outputs.append({'flag':flag,'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr})
        require(result.returncode==0,'version tool failed')
    registry=load_json(ROOT/'data/version_registry.json');version=registry['versions'][-1]['version']
    require(any(p['proposal_id']==pid for p in registry['versions'][-1]['proposals']),'new version lacks proposal')
    value={'registered':True,'version':version,'proposal_id':pid,'commands':outputs,
        'ledger_sha256':file_hash(ledger),'registry_sha256':file_hash(ROOT/'data/version_registry.json'),
        'changelog_sha256':file_hash(ROOT/'CHANGELOG.md'),'historical_memory_prefix_preserved':True,
        'final_manifest_sha256':file_hash(OUT/'FINAL_MANIFEST.json')}
    with (OUT/'REGISTRATION.json').open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2)
    print(json.dumps({'registered':True,'version':version,'proposal_id':pid},ensure_ascii=False),flush=True)


if __name__=='__main__':
    with controlled_writer(OUT):main()
