"""Manual JQ handoff and reviewed returns; first JQ requires E0-E4, never E5.

Investment-level data assurance is NOT required for JQ candidacy. It is deferred to the
live/QMT signal stage: ``investment_signoff`` below is the single gate that must be passed
before a candidate's simulated-trading / TraderLink signal is authorized, and its reviewer
defaults to the account owner (single-owner personal system)."""
from pathlib import Path
from .contracts import require, digest, load_json
from .evidence import now
from .campaign import write_new
from .artifacts import verify
from .operations import operate, reviewed_files

ACCOUNT_FIELDS=['signal','target_weights','orders','holdings','cash','receivables','nav']


def prepare(store, spec):
    require(spec.get('research_receipt') and spec.get('bundle_path') and spec.get('output_root'),
            'research acceptance, native bundle and new output required')
    receipt=store.consume_receipt(spec['research_receipt'],'research_acceptance')
    require(receipt['jq_candidate_allowed'] is True,'current formal research acceptance required')
    source=Path(spec['bundle_path']).resolve()
    source_manifest=verify(source)
    result=load_json(source/'evaluation.json')
    require(result['strategy_ir']['ir_hash']==receipt['ir_hash'],'handoff strategy identity changed')
    generated=operate(store,'compile-jq',spec)
    root=Path(generated['bundle_path'])
    metadata={'schema_version':'fuxi.manual_jq_handoff.1','research_receipt':spec['research_receipt'],
              'source_bundle':str(source),'source_manifest_hash':source_manifest['manifest_hash'],
              'source_evaluation_hash':digest(result),'ir_hash':receipt['ir_hash'],
              'candidate_id':receipt['candidate_id'],'policy_hash':receipt['policy_hash'],
              'snapshot_hash':receipt['snapshot_hash'],'script_sha256':generated['script_sha256'],
              'upload_mode':'user_manual_upload_to_JQ_backtest','returned_result_required':True,
              'account_fields':ACCOUNT_FIELDS,'first_JQ_requires_E5':False,'observed_on_jq':False,
              'investment_certification':receipt.get('investment_certification'),
              'QMT_orders_enabled':False,'live_execution_allowed':False}
    # Keep new operator files outside the sealed compiler bundle.
    handoff=Path(spec['output_root']).resolve()/('handoff_'+digest(metadata))
    write_new(handoff/'HANDOFF.json',metadata)
    write_new(handoff/'LOCAL_ACCOUNT.json',{'ir_hash':receipt['ir_hash'],
              'account':result['candidate_account'],'strategy_ir':result['strategy_ir'],
              'normalization':'raw local records; field mapping reviewed explicitly before parity'})
    template={'schema_version':'fuxi.manual_jq_return.1','handoff_path':str(handoff/'HANDOFF.json'),
              'reviewer':None,'reviewed_at':None,'evidence_files':{},
              'observed_path':None,'expected_path':None,'tolerance':1e-8,
              'notes':'Preserve actual JQ run ID, raw logs, session keys, signal/order/account fields; unknown NAV cannot pass.'}
    write_new(handoff/'RETURN_TEMPLATE.json',template)
    write_new(handoff/'UPLOAD.md',('上传 '+str(root/'jq_strategy.py')+' 到聚宽历史回测。保留原始日志和回测导出；'
       '填写 RETURN_TEMPLATE.json 后运行 work-packages return-jq。收益摘要用于研究反馈，'
       '正式对拍还需逐日 signal/target_weights/orders/holdings/cash/receivables/nav。'
       '首次JQ无需E5；本包没有QMT或实盘权限。\n').encode('utf8'))
    return {**generated,'handoff_path':str(handoff),'new_qualification_granted':False,
            'observed_on_jq':False,'first_JQ_requires_E5':False}


def ingest(store, spec):
    require(spec.get('schema_version')=='fuxi.manual_jq_return.1','JQ return schema invalid')
    review=reviewed_files(spec)
    for key in ('handoff_path','observed_path'):
        require(spec.get(key) in review['evidence_files'],key+' must be reviewed and hash-bound')
    handoff=load_json(spec['handoff_path']);observed=load_json(spec['observed_path'])
    current=store.consume_receipt(handoff['research_receipt'],'research_acceptance')
    require(current['ir_hash']==handoff['ir_hash'],'current research/handoff changed')
    require(observed.get('ir_hash')==handoff['ir_hash'],'returned strategy mismatch')
    require(observed.get('observed_on_jq') is True and observed.get('jq_run_id'),'actual JQ run required')
    for field in ('candidate_id','policy_hash','snapshot_hash'):
        require(observed.get(field)==handoff[field],'returned '+field+' mismatch')
    if not spec.get('expected_path') or not observed.get('rows'):
        record=store.append('manual_JQ_summary',{'handoff_hash':digest(handoff),
                'raw_observation_hash':digest(observed),'summary':observed.get('summary'),**review,
                'jq_run_id':observed['jq_run_id'],'qualification_transfer_allowed':False})
        return {'feedback_record_id':record,'receipt_id':None,'accepted':False,'n_pairs':0,
                'blockers':['full_signal_order_account_parity_rows_required'],
                'summary_recorded':True,'automatic_candidate_revision':False,'live_execution_allowed':False}
    require(spec['expected_path'] in review['evidence_files'],'expected_path must be reviewed and hash-bound')
    expected=load_json(spec['expected_path'])
    require(expected.get('source_evaluation_hash')==handoff['source_evaluation_hash'],'local expected evaluation changed')
    require(all(expected.get(k)==handoff[k] for k in ('ir_hash','candidate_id','policy_hash','snapshot_hash')),
            'local expected identity mismatch')
    result=operate(store,'import-jq-parity',{'reviewer':review['reviewer'],'reviewed_at':review['reviewed_at'],
           'evidence_files':review['evidence_files'],'research_receipt':handoff['research_receipt'],
           'observed_on_jq':True,'jq_run_id':observed['jq_run_id'],'ir_hash':handoff['ir_hash'],
           'fields':ACCOUNT_FIELDS,'expected_rows':expected['rows'],'observed_rows':observed['rows'],
           'tolerance':spec.get('tolerance',1e-8)})
    return {**result,'feedback_record_id':store.append('manual_JQ_return',{
             'handoff_hash':digest(handoff),'raw_observation_hash':digest(observed),
             'jq_run_id':observed['jq_run_id'],'comparison':result,'summary':observed.get('summary'),
             'summary_is_not_parity':True,'automatic_candidate_revision':False}),
             'live_execution_allowed':False}


def investment_signoff(store, spec):
    """Single gate before a candidate's simulated-trading / TraderLink signal is authorized.

    Who certifies? By default the ACCOUNT OWNER (single-owner personal system, sole
    risk-bearer). The reviewer is explicit: ``reviewer_role='account_owner_manual_signoff'``
    requires the owner's explicit confirmation only. Any other role (e.g. an independent
    third party) requires a real ``data_contract`` receipt with
    ``investment_eligibility_certified=True`` — mirroring the independent-review principle
    in ``data_assurance.import_assurance``. This is deliberately NOT part of research
    acceptance: JQ backtest candidacy needs only E0-E4, never an investment data contract.
    """
    require(spec.get('research_receipt'), 'research receipt required for live signoff')
    research = store.consume_receipt(spec['research_receipt'], 'research_acceptance')
    require(research.get('jq_candidate_allowed') is True, 'only JQ candidates can reach live signoff')
    reviewer_role = spec.get('reviewer_role') or 'account_owner_manual_signoff'
    reviewer = spec.get('reviewer') or 'account_owner'
    if reviewer_role != 'account_owner_manual_signoff':
        require(spec.get('data_receipt_id'), 'independent reviewer requires a data_contract receipt id')
        receipt = store.consume_receipt(spec['data_receipt_id'], 'data_contract')
        require(receipt.get('investment_eligibility_certified') is True,
                'independent review requires an investment-certified data contract')
        reviewer = spec.get('reviewer') or receipt.get('reviewer')
    require(spec.get('signoff_confirmed') is True, 'explicit signoff confirmation required')
    body = {'research_receipt': spec['research_receipt'],
            'candidate_id': research['candidate_id'], 'ir_hash': research['ir_hash'],
            'reviewer': reviewer, 'reviewer_role': reviewer_role,
            'signoff_confirmed': True, 'signoff_at': now(),
            'scoped_to': 'manual_simulated_trading_and_TraderLink_signal_only',
            'live_execution_allowed': False}
    return {'receipt_id': store.receipt('investment_signoff', body), **body}
