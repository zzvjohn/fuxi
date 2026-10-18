"""Prospective, immutable daily decisions and paired native account observations.

The wall clock is read here, not supplied by the caller. Research observation is
available before admission; only native receipts can create formal acceptance.
"""
from copy import deepcopy
from datetime import datetime, timezone, timedelta
import csv
import hashlib
import json
from pathlib import Path
from .contracts import digest, require, timestamp, load_json, finite
from .artifacts import publish, verify, code_manifest
from .data import snapshot_from_payload
from .online import daily_decision, daily_model_decision
from .portfolio import compile_strategy, validate_strategy
from .simulation import run_account
from .account_reference import reconcile


def clock():
    return datetime.now(timezone.utc).isoformat()


def _source(path, expected):
    p=Path(path).resolve();require(p.is_file(), 'forward source missing')
    raw=p.read_bytes();actual=hashlib.sha256(raw).hexdigest()
    require(actual==expected, 'forward source hash mismatch')
    return raw, {'path':str(p),'sha256':actual}


def _bundles(root, category):
    result=[]
    for path in sorted((Path(root)/category).glob('run_*')):
        manifest=verify(path)
        result.append((path,manifest))
    return result


def _freeze(root, store=None):
    rows=_bundles(root,'protocol')
    require(len(rows)==1, 'exactly one published forward protocol required')
    p=load_json(rows[0][0]/'FREEZE.json')
    require(p['freeze_hash']==digest({k:v for k,v in p.items() if k!='freeze_hash'}),
            'forward freeze changed')
    current=code_manifest()
    require(all(current.get(k)==v for k,v in p['code'].items()), 'forward code version changed; new freeze required')
    if p['request']['mode']=='formal_acceptance':
        require(store is not None, 'forward receipt authority required')
        pending=store.consume_receipt(p['request']['pending_receipt'],'forward_pending')
        protocol=store.consume_receipt(pending['protocol_receipt'],'confirmatory_protocol')
        require(digest(protocol)==p['native_protocol_hash'], 'formal forward protocol changed')
    return p


def _calendar_days(raw):
    try:
        data=json.loads(raw.decode('utf8'))
        if isinstance(data,list):return set(data)
        return set(data['sessions'])
    except (ValueError,KeyError,TypeError):
        rows=list(csv.DictReader(raw.decode('utf-8-sig').splitlines()))
        require(rows, 'calendar evidence empty')
        key=next((k for k in ('cal_date','trade_date','date','session') if k in rows[0]),None)
        require(key is not None, 'calendar date column missing')
        return {str(r[key])[:4]+'-'+str(r[key])[4:6]+'-'+str(r[key])[6:8]
                if len(str(r[key]))==8 else str(r[key]) for r in rows
                if str(r.get('is_open','1'))=='1'}


def freeze_archive(spec, root, store=None):
    require(spec.get('schema_version')=='fuxi.forward_archive_request.1', 'forward archive request schema invalid')
    require('frozen_at' not in spec and 'observation_clock' not in spec, 'forward clock is recorded by runtime')
    require(spec.get('mode') in ('research_observation','formal_acceptance'), 'forward observation mode required')
    require(all(isinstance(spec.get(k),str) and spec[k] for k in
        ('candidate_id','ir_hash','baseline_hash','cost_profile_hash')), 'forward identities required')
    for side in ('candidate','baseline'):
        contract=spec[side];validate_strategy(contract['strategy'])
        require(isinstance(contract.get('support_asts',()),(list,tuple)), 'forward support AST sequence required')
        require(('ast' in contract) != ('model' in contract), 'exactly one frozen predictor required')
        if 'model' in contract:
            state=contract['model']
            from .models import validate_frozen_model
            validate_frozen_model(state)
            require(contract.get('feature_asts') and len(contract['feature_asts'])==len(state['center']),
                    'forward model feature identity required')
            if 'features' in state:
                require(sorted(contract['feature_asts'])==state['features'], 'forward model feature order changed')
    existing=_bundles(root,'protocol')
    if existing:
        value=_freeze(root,store)
        require(value['request']==spec, 'cannot revise an existing forward freeze')
        return {**value,'idempotent':True}
    cal=spec['calendar'];days=cal['sessions'];times=cal['decision_times']
    require(days and days==sorted(set(days)) and set(times)==set(days), 'forward calendar shape invalid')
    require(all(timestamp(times[d]).date().isoformat()==d for d in days), 'forward calendar clocks invalid')
    raw,source=_source(cal['evidence_path'],cal['evidence_sha256'])
    require(set(days)<=_calendar_days(raw), 'forward sessions absent from real calendar evidence')
    now=clock();future=[d for d in days if timestamp(times[d])>timestamp(now)]
    require(all('model' not in spec[s] or 'training_cutoff' not in spec[s]['model']
                or timestamp(spec[s]['model']['training_cutoff'])<=timestamp(now) for s in ('candidate','baseline')),
            'model training cutoff is in the future')
    require(future, 'future calendar exhausted')
    horizons=spec.get('horizon_sessions',5);looks=spec.get('look_sessions',[60,120])
    minimum=spec.get('minimum_sections',60)
    grace=spec.get('capture_grace_seconds',21600)
    require(type(grace)is int and 0<grace<=86400, 'forward capture grace invalid')
    require(type(horizons)is int and horizons>0 and type(minimum)is int and minimum>0
            and looks==sorted(set(looks)) and looks and all(type(n)is int and n>=minimum for n in looks),
            'forward horizon/look/sections design invalid')
    planned=[]
    for n in looks:
        complete=len(future)>=n+horizons
        planned.append({'sections':n,'decision_end':future[n-1] if len(future)>=n else None,
            'mature_at':times[future[n-1+horizons]] if complete else None,
            'status':'calendar_resolved' if complete else 'pending_calendar'})
    protocol_hash=None
    if spec['mode']=='formal_acceptance':
        require(store is not None and spec.get('pending_receipt'), 'formal pending receipt required')
        pending=store.consume_receipt(spec['pending_receipt'],'forward_pending')
        protocol=store.consume_receipt(pending['protocol_receipt'],'confirmatory_protocol')
        research=store.consume_receipt(pending['research_receipt'],'research_acceptance')
        require(pending['ir_hash']==spec['ir_hash'] and pending['candidate_id']==spec['candidate_id']
                and protocol['baseline_hash']==spec['baseline_hash']
                and protocol['cost_profile_hash']==spec['cost_profile_hash'], 'formal forward identity mismatch')
        require(research['jq_candidate_allowed'] is True and minimum>=protocol['minimum_sections'],
                'current formal research acceptance/design required')
        require(all(p['mature_at'] in protocol['look_schedule'] for p in planned), 'archive/native fixed looks mismatch')
        protocol_hash=digest(protocol)
    files=('forward_archive.py','online.py','models.py','data.py','historical_adapter.py',
           'dsl.py','portfolio.py','simulation.py','account_reference.py','contracts.py',
           'a_share.py','action_accounting.py','account.py','evidence.py','artifacts.py')
    code=code_manifest()
    body={'schema_version':'fuxi.forward_archive_freeze.1','request':deepcopy(spec),'frozen_at':now,
          'future_sessions':future,'looks':planned,'calendar_source':source,
          'native_protocol_hash':protocol_hash,'code':{k:code[k] for k in files},
          'minimum_sections':minimum,'horizon_sessions':horizons,
          'live_execution_allowed':False,'qualification_granted':False}
    body['freeze_hash']=digest(body)
    path=publish(Path(root)/'protocol',{'FREEZE.json':body,'CALENDAR_SOURCE':raw},
                 {'kind':'prospective_forward_freeze','freeze_hash':body['freeze_hash']})
    return {**body,'bundle_path':str(path),'idempotent':False}


def _captures(root, freeze):
    result={}
    for path,manifest in _bundles(root,'decisions'):
        row=load_json(path/'CAPTURE.json')
        require(row['freeze_hash']==freeze['freeze_hash'], 'forward decision belongs to another freeze')
        require(row['session'] not in result, 'duplicate published forward day')
        result[row['session']]={**row,'bundle_path':str(path),'manifest_hash':manifest['manifest_hash']}
    return result


def capture_day(root, spec, store=None):
    freeze=_freeze(root,store);request=freeze['request'];day=spec['session']
    require(set(spec)<= {'session','snapshot_path','snapshot_sha256','candidate_prior','baseline_prior'},
            'caller-supplied future clock/decision is forbidden')
    require(day in freeze['future_sessions'], 'decision outside prospective calendar')
    now=clock();decision=request['calendar']['decision_times'][day]
    require(timestamp(freeze['frozen_at'])<timestamp(decision)<=timestamp(now), 'future decision not yet due')
    raw,source=_source(spec['snapshot_path'],spec['snapshot_sha256'])
    source_payload=json.loads(raw.decode('utf8'));snapshot_payload=source_payload.get('snapshot',source_payload)
    snapshot=snapshot_from_payload(snapshot_payload)
    require(snapshot.sessions[-1]==day and snapshot.payload['decision_times'][day]==decision,
            'forward snapshot must end at this recorded decision')
    if request['mode']=='formal_acceptance':
        require(snapshot_payload.get('exposure')=='forward_sealed', 'formal archive needs protected prospective snapshot')
        pending=store.consume_receipt(request['pending_receipt'],'forward_pending')
        require(snapshot_payload.get('quality',{}).get('protection_receipt') in
                (request['pending_receipt'],pending['cohort_id']), 'snapshot has another forward protection reference')
    captured=_captures(root,freeze)
    identity=digest({'source':source,'candidate_prior':spec.get('candidate_prior'),
                     'baseline_prior':spec.get('baseline_prior')})
    if day in captured:
        require(captured[day]['request_identity']==identity, 'cannot revise a captured forward day')
        return {**captured[day],'idempotent':True}
    grace=request.get('capture_grace_seconds',21600)
    require(timestamp(now)<=timestamp(decision)+timedelta(seconds=grace),
            'stale decision cannot be retrospectively captured')
    require(all(timestamp(r['recorded_at'])<=timestamp(now) for r in captured.values()),
            'forward capture clock cannot move backwards')
    decisions={}
    for side in ('candidate','baseline'):
        contract=request[side];prior=spec.get(side+'_prior')
        if 'model' in contract:
            decisions[side]=daily_model_decision(contract['feature_asts'],contract['model'],snapshot,day,now,
                                               contract['strategy'],prior=prior,support_asts=contract.get('support_asts',()))
        else:
            decisions[side]=daily_decision(contract['ast'],snapshot,day,now,contract['strategy'],prior=prior,
                                          support_asts=contract.get('support_asts',()))
    body={'schema_version':'fuxi.forward_capture.1','freeze_hash':freeze['freeze_hash'],
          'session':day,'decision_time':decision,'recorded_at':now,'source':source,
          'snapshot_hash':snapshot.hash,'source_exposure':snapshot_payload['exposure'],
          'request_identity':identity,'decisions':decisions,'formal_eligibility':False}
    path=publish(Path(root)/'decisions',{'CAPTURE.json':body,'SNAPSHOT_SOURCE.json':raw},
                 {'kind':'prospective_daily_capture','freeze_hash':freeze['freeze_hash'],'session':day})
    return {**body,'bundle_path':str(path),'idempotent':False}


def record_account_pair(root, spec, store=None):
    """Recompute full cash/share accounts from preserved decisions and actual input."""
    freeze=_freeze(root,store);request=freeze['request'];day=spec['through_session']
    require(set(spec)== {'input_path','input_sha256','through_session'}, 'native paired input request fields invalid')
    captures=_captures(root,freeze)
    require(captures and day in freeze['future_sessions'], 'nonempty prospective captures required')
    raw,source=_source(spec['input_path'],spec['input_sha256']);payload=json.loads(raw.decode('utf8'))
    snapshot=snapshot_from_payload(payload['snapshot']);market=payload['market'];now=clock()
    require(snapshot.sessions[-1]==day and timestamp(snapshot.payload['decision_times'][day])<=timestamp(now),
            'future market data cannot be recorded')
    require(digest(market['execution'])==request['cost_profile_hash'], 'forward account cost profile changed')
    require(not any(d>day for d in market['sessions']) and not any(b['session']>day for b in market['bars']),
            'market contains future sessions')
    first=min(captures);expected=[d for d in freeze['future_sessions'] if first<=d<=day]
    missing=[d for d in expected if d not in captures]
    require(not missing, 'forward daily captures missing: '+str(missing))
    for d in expected:
        require(snapshot.payload['decision_times'][d]==captures[d]['decision_time'], 'captured decision clock changed')
        for side in ('candidate','baseline'):
            require(set(captures[d]['decisions'][side]['scores'])==set(snapshot.universe[d]),
                    'captured universe changed')
    pairs=_bundles(root,'accounts')
    prior_markets=[]
    for path,_ in pairs:
        row=load_json(path/'PAIR.json')
        require(row['freeze_hash']==freeze['freeze_hash'], 'paired accounts belong to another freeze')
        if row['through_session']==day:
            require(row['source']==source, 'cannot replace a forward account observation')
            return {**row,'bundle_path':str(path),'idempotent':True}
        prior_markets.append(load_json(path/'INPUT_SOURCE.json')['market'])
    current_bars={(r['session'],r['instrument']):r for r in market['bars']}
    for old in prior_markets:
        require(all(current_bars.get((r['session'],r['instrument']))==r for r in old['bars']),
                'previously recorded market observations changed')
    sp=deepcopy(snapshot.payload);sp['evaluation_start']=first;snapshot=snapshot_from_payload(sp)
    accounts={};irs={};checks={}
    for side in ('candidate','baseline'):
        scores={d:captures[d]['decisions'][side]['scores'] for d in expected}
        state=request[side].get('model')
        irs[side]=compile_strategy(request[side]['strategy'],scores,snapshot,digest(market['execution']),
                                  score_origin='forward',model_hash=state.get('bundle_hash',state['model_hash']) if state else None)
        accounts[side]=run_account(irs[side],market,start=first,end=day)
        checks[side]=reconcile(accounts[side],market)
        require(checks[side]['n_pairs']>0 and checks[side]['eval_status']=='valid'
                and not checks[side]['mismatches'], 'forward native account reference mismatch')
    body={'schema_version':'fuxi.forward_account_pair.1','freeze_hash':freeze['freeze_hash'],
        'through_session':day,'recorded_at':now,'source':source,
        'captured_days':expected,'decision_ids':{d:{s:captures[d]['decisions'][s]['decision_id']
            for s in ('candidate','baseline')} for d in expected},'reference':checks,
        'account_identifiable':all(a['eval_status']=='valid' for a in accounts.values()),
        'formal_eligibility':False}
    path=publish(Path(root)/'accounts',{'PAIR.json':body,'INPUT_SOURCE.json':raw,
        'CANDIDATE_ACCOUNT.json':accounts['candidate'],'BASELINE_ACCOUNT.json':accounts['baseline'],
        'CANDIDATE_IR.json':irs['candidate'],'BASELINE_IR.json':irs['baseline']},
        {'kind':'prospective_native_paired_accounts','freeze_hash':freeze['freeze_hash']})
    return {**body,'bundle_path':str(path),'idempotent':False}


def status(root, store=None):
    freeze=_freeze(root,store);captures=_captures(root,freeze);now=clock()
    pairs=_bundles(root,'accounts')
    return {'schema_version':'fuxi.forward_archive_status.1','freeze_hash':freeze['freeze_hash'],
        'mode':freeze['request']['mode'],'frozen_at':freeze['frozen_at'],'checked_at':now,
        'captured_sections':len(captures),'paired_account_observations':len(pairs),
        'looks':[{**r,'due':bool(r['mature_at'] and timestamp(r['mature_at'])<=timestamp(now))} for r in freeze['looks']],
        'status':'ready_awaiting_observations' if not captures else 'collecting',
        'actual_JQ_granted':False,'future_acceptance_granted':False,'live_execution_allowed':False}


def build_observation(root, spec, store=None):
    freeze=_freeze(root,store);request=freeze['request'];now=clock()
    require(set(spec)== {'look_sessions'}, 'registered look only; runtime supplies observation clock')
    looks=[r for r in freeze['looks'] if r['sections']==spec['look_sessions']]
    require(len(looks)==1, 'unregistered forward look')
    look=looks[0]
    if not look['mature_at'] or timestamp(look['mature_at'])>timestamp(now):
        return {'schema_version':'fuxi.forward_observation_status.1','status':'pending_calendar' if not look['mature_at'] else 'not_due',
                'n_pairs':0,'look':look,'checked_at':now,'future_acceptance_granted':False}
    for path,_ in _bundles(root,'looks'):
        old=load_json(path/'OBSERVATION.json')
        if old.get('registered_look')==look['sections']:
            require(old['freeze_hash']==freeze['freeze_hash'], 'forward observation freeze changed')
            return {**old,'idempotent':True}
    captures=_captures(root,freeze);days=freeze['future_sessions'][:look['sections']]
    missing=[d for d in days if d not in captures]
    pairs=_bundles(root,'accounts')
    if not pairs or missing:
        return {'schema_version':'fuxi.forward_observation_status.1','status':'data_insufficient',
                'n_pairs':0,'missing_captures':missing,'account_pairs_missing':not bool(pairs),
                'future_acceptance_granted':False}
    latest=max(pairs,key=lambda x:load_json(x[0]/'PAIR.json')['through_session']);path=latest[0]
    pair=load_json(path/'PAIR.json');candidate=load_json(path/'CANDIDATE_ACCOUNT.json')
    baseline=load_json(path/'BASELINE_ACCOUNT.json')
    require([r['session'] for r in candidate['daily']]==[r['session'] for r in baseline['daily']],
            'forward paired dates changed')
    returns={}
    for i in range(1,len(candidate['daily'])):
        a0,a=candidate['daily'][i-1:i+1];b0,b=baseline['daily'][i-1:i+1]
        vals=[r.get('nav') for r in (a0,a,b0,b)]
        identifiable=all(r.get('valuation_identifiable') is True for r in (a0,a,b0,b))
        if identifiable and all(v is not None and float(v)>0 for v in vals):
            delta=float(a['nav'])/float(a0['nav'])-float(b['nav'])/float(b0['nav'])
            require(finite(delta), 'nonfinite forward account increment')
            returns[a['session']]=delta
    calendar=freeze['future_sessions'];times=request['calendar']['decision_times'];rows=[]
    for d in days:
        index=calendar.index(d);mature=calendar[index+freeze['horizon_sessions']]
        execution=calendar[index+1]
        rows.append({'decision_time':times[d],'mature_at':times[mature],
            'paired_net_increment':returns.get(execution) if mature<=pair['through_session'] else None,
            'return_session':execution,'decision_ids':pair['decision_ids'].get(d)})
    n=sum(r['paired_net_increment'] is not None for r in rows)
    scope=({'sources':[freeze['calendar_source']['sha256']],'instruments':['*'],'resolved':True,
        'intervals':[[times[days[0]],times[days[-1]]]],'label_intervals':[]})
    if request['mode']=='formal_acceptance':
        pending=store.consume_receipt(request['pending_receipt'],'forward_pending')
        scope=store.consume_receipt(pending['protocol_receipt'],'confirmatory_protocol')['scope']
    observation={'ir_hash':request['ir_hash'],'baseline_hash':request['baseline_hash'],
        'cost_profile_hash':request['cost_profile_hash'],'scope':scope,'rows':rows,
        'mature_at':look['mature_at'],'account_identifiable':pair['account_identifiable'] and n==len(days)}
    result={'schema_version':'fuxi.forward_observation_build.1',
        'status':'ready_for_independent_review' if observation['account_identifiable'] else 'data_insufficient',
        'observation_clock':look['mature_at'],'checked_at':now,'n_pairs':n,'observation':observation,
        'evidence_bundle':str(path),'pending_receipt':request.get('pending_receipt'),
        'future_acceptance_granted':False,'live_execution_allowed':False,
        'freeze_hash':freeze['freeze_hash'],'registered_look':look['sections']}
    if observation['account_identifiable']:
        publish(Path(root)/'looks',{'OBSERVATION.json':result},{'kind':'unadmitted_forward_observation',
            'freeze_hash':freeze['freeze_hash'],'look':look['sections']})
    return result
