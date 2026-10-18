"""Executable operator requests and honest receipt states for the manual lifecycle."""
from pathlib import Path
from .contracts import digest, require, load_json, ContractError, canonical
from .campaign import write_new
from .forward_archive import clock


RECEIPT_KINDS={'research_receipt':'research_acceptance','parity_receipt':'jq_parity',
    'protocol_receipt':'confirmatory_protocol','pending_receipt':'forward_pending',
    'forward_receipt':'forward_acceptance','bundle_receipt':'release_bundle',
    'activation_receipt':'paper_activation'}


def evidence_state(references=None, store=None, software_checks=None):
    references=references or {};states={};bodies={}
    for name,kind in RECEIPT_KINDS.items():
        identity=references.get(name)
        if not identity:states[name]={'state':'missing','kind':kind};continue
        if store is None:states[name]={'state':'authority_not_opened','kind':kind,'id':identity};continue
        try:
            body=store.consume_receipt(identity,kind)
            if name=='research_receipt':require(body.get('jq_candidate_allowed') is True, 'research not admitted')
            if name=='parity_receipt':require(body.get('accepted') is True and body.get('observed_on_jq') is True
                and body.get('n_pairs',0)>0, 'actual JQ parity missing')
            if name=='forward_receipt':require(body.get('accepted') is True, 'forward not accepted')
            states[name]={'state':'active','kind':kind,'id':identity};bodies[name]=body
        except ContractError as exc:states[name]={'state':'inactive_or_invalid','kind':kind,'id':identity,'reason':str(exc)}
    identities={b.get('ir_hash') for b in bodies.values() if b.get('ir_hash')}
    identity_ok=len(identities)<=1
    next_action='admit-research'
    ordered=[('research_receipt','prepare-jq'),('parity_receipt','register-protocol'),
             ('protocol_receipt','freeze-forward'),('pending_receipt','capture-forward-daily'),
             ('forward_receipt','build-release'),('bundle_receipt','activate-paper'),
             ('activation_receipt','monitor')]
    for name,action in ordered:
        if states[name]['state']!='active':break
        next_action=action
    checks=software_checks or {}
    require(all(type(v)is bool for v in checks.values()), 'explicit software check booleans required')
    software_ready=bool(checks) and all(checks.values())
    return {'schema_version':'fuxi.lifecycle_status.1','software_checks':checks,'software_ready':software_ready,
        'status':'ready_awaiting_evidence' if software_ready and next_action!='monitor' else 'evidence_pending',
        'receipts':states,'same_candidate_identity':identity_ok,
        'next_action':next_action if identity_ok else 'reconcile_candidate_identity',
        'JQ_backtest_observed':states['parity_receipt']['state']=='active',
        'forward_accepted':states['forward_receipt']['state']=='active',
        'paper_active':states['activation_receipt']['state']=='active','live_execution_allowed':False}


def build_requests(spec, root, store=None):
    require(spec.get('schema_version')=='fuxi.lifecycle_requests.1', 'lifecycle request schema invalid')
    root=Path(root).resolve();index_path=root/'REQUEST_INDEX.json'
    if index_path.is_file():
        index=load_json(index_path)
        require(index['request_hash']==digest(spec), 'cannot revise lifecycle request package')
        for name,expected in index['files'].items():
            from hashlib import sha256
            require(sha256((root/name).read_bytes()).hexdigest()==expected, 'lifecycle request file changed')
        return {**index,'idempotent':True}
    references={k:spec.get(k) for k in RECEIPT_KINDS}
    state=evidence_state(references,store,spec.get('software_checks'))
    prepared={'research_receipt':references['research_receipt'],'bundle_path':spec.get('bundle_path'),
        'output_root':str(root/'jq_packages')}
    returned={'schema_version':'fuxi.manual_jq_return.1','handoff_path':None,'reviewer':None,'reviewed_at':None,
        'evidence_files':{},'observed_path':None,'expected_path':None,'tolerance':1e-8,
        'notes':'Fill from an actual manually uploaded JQ run; summaries cannot substitute full account parity.'}
    freeze={'research_receipt':references['research_receipt'],'protocol_receipt':references['protocol_receipt'],
            'parity_receipt':references['parity_receipt']}
    observe={'pending_receipt':references['pending_receipt'],'observation_clock':None,
             'observation':None,'reviewer':None,'reviewed_at':None,'evidence_files':{}}
    register={'reviewer':None,'reviewed_at':None,'evidence_files':{},
        'calibration':{'kind':'confirmatory_calibration','verified':False,'policy_hash':spec.get('policy_hash')},
        'protocol':{'schema_version':'fuxi.confirmatory_protocol.1','policy_hash':spec.get('policy_hash'),
            'candidate_ids':[spec['candidate_id']] if spec.get('candidate_id') else [],
            'baseline_hash':spec.get('baseline_hash'),'scope':None,'frozen_at':None,'minimum_sections':60,
            'alpha':.05,'block_length':12,'bootstrap_replicates':3999,'look_schedule':[],
            'cost_profile_hash':spec.get('cost_profile_hash'),'minimum_net_increment_daily':0}}
    release={'forward_receipt':references['forward_receipt'],
        'artifacts':{k:None for k in ('code','model','strategy','data_contract','policy','state_schema')},
        'rollback_receipt':None,'reviewer':None,'reviewed_at':None,'evidence_files':{}}
    files={'LIFECYCLE_STATUS.json':state,'PREPARE_JQ_REQUEST.json':prepared,
        'RETURN_JQ_REQUEST_TEMPLATE.json':returned,'REGISTER_PROTOCOL_TEMPLATE.json':register,
        'FREEZE_FORWARD_REQUEST.json':freeze,'OBSERVE_FORWARD_REQUEST_TEMPLATE.json':observe,
        'BUILD_RELEASE_REQUEST_TEMPLATE.json':release,
        'ACTIVATE_PAPER_REQUEST.json':{'bundle_receipt':references['bundle_receipt']},
        'ROLLBACK_PAPER_REQUEST_TEMPLATE.json':{'activation_receipt':references['activation_receipt'],
            'reconciliation':{'unknown_orders':None,'account_matched':False},
            'reviewer':None,'reviewed_at':None,'evidence_files':{}}}
    if spec.get('forward_archive_request'):
        files['FORWARD_ARCHIVE_REQUEST.json']=spec['forward_archive_request']
    instructions=('所有动作由手动入口执行；当前状态和下一动作见 LIFECYCLE_STATUS.json。\n'
        '已存在有效研究回执后执行 work-packages prepare-jq，上传生成脚本到 JQ 历史回测。\n'
        '保留实际 JQ run_id、原日志及逐日 signal/target_weights/orders/holdings/cash/receivables/nav，'
        '复核原件 SHA 后执行 work-packages return-jq。收益摘要只形成研究反馈。\n'
        '独立校准并冻结真实未来日历/协议后执行 operate register-protocol 与 freeze-forward；'
        '前向 archive 每日保留真实截止输入、冻结模型双决策与完整现金股数对账，成熟注册观察才可提交 E5。\n'
        '这些请求的空值和 verified=false 表示实际证据尚未发生，不能用模板或合成数据填成验收。\n'
        '当前运行方式为手动研究及 JQ 历史回测回传。\n').encode('utf8')
    files['RUNBOOK.md']=instructions
    from hashlib import sha256
    hashes={}
    for name,value in files.items():
        destination=root/name
        raw=value if isinstance(value,bytes) else canonical(value)+b'\n'
        if destination.exists():require(destination.read_bytes()==raw, 'partial lifecycle package differs from retry')
        else:write_new(destination,raw)
        hashes[name]=sha256(destination.read_bytes()).hexdigest()
    index={'schema_version':'fuxi.lifecycle_request_index.1','created_at':clock(),'request_hash':digest(spec),
        'files':hashes,'next_action':state['next_action'],'actual_JQ_returns':0,'future_acceptance_granted':False,
        'qualification_granted':False,'live_execution_allowed':False}
    write_new(index_path,index)
    return {**index,'root':str(root),'idempotent':False}
