"""Serial, resumable historical reevaluation followed by matched memory controls."""
from collections import Counter
from copy import deepcopy
import gzip
import json
from pathlib import Path
import random
from statistics import mean
import time
from .contracts import digest,require,load_json,canonical,timestamp
from .artifacts import code_manifest
from .datasets import file_hash,scope_of
from .data import Labels
from .historical_adapter import HistoricalSnapshot
from .dsl import evaluate_ast
from .evaluation import factor_statistics,DEFAULT_POLICY
from .search import node,generate
from .research_evaluation import evaluate_candidate,paired_accounts
from .research_protocol import policy_from_plan,block_inference_fast
from .statistics import holm
from .learning import feedback_card,paired_parent,behavioral_signature,structural_family,compatibility
from .campaign import write_new
from .evidence import EvidenceStore,now
from .runtime import authority_location
from .acceptance import AcceptanceService


class StrictDriftLabels:
    """Sensitivity comparator only: reject proxy-label spans touching any factor drift."""
    def __init__(self,labels,snapshot,market):
        self.base=labels;self.payload=labels.payload;self.blocked=set()
        days={d:i for i,d in enumerate(snapshot.sessions)};h=labels.payload['label_spec']['horizon_sessions']
        for bar in market['bars']:
            if bar.get('event_coverage')=='tiny_drift_consistent_reference_research_only':
                i=days[bar['session']]
                for j in range(max(0,i-h-1),i):self.blocked.add((snapshot.sessions[j],bar['instrument']))

    def value(self,day,instrument,clock):
        return None if (day,instrument) in self.blocked else self.base.value(day,instrument,clock)


def freeze(plan_path,candidates_path,input_path,output):
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    plan=load_json(plan_path);specs=load_json(candidates_path)
    from .remediation import require_preflight
    preflight_receipt=require_preflight(plan,input_path,code_manifest())
    require(sorted(c['candidate_id'] for c in specs)==plan['reevaluate_candidate_ids'],'candidate family changed')
    # Reconstruct missing ancestry without using any ancestor performance reading.
    import sqlite3
    workspace=next(p for p in Path(plan_path).resolve().parents if (p/'fuxi_v1').is_dir())
    ancestors={};ancestry_sources=[]
    for old_root in ('fuxi_v1_full_20260929','fuxi_v1_deepseek_20260929'):
        for source in (workspace/'reports'/old_root).rglob('evidence.sqlite'):
            db=sqlite3.connect(source.resolve().as_uri()+'?mode=ro',uri=True)
            for raw,expected in db.execute("SELECT payload,hash FROM facts WHERE kind='candidate'"):
                c=json.loads(raw);require(digest(c)==expected,'ancestor evidence corrupt');ancestors[c['candidate_id']]=c
            db.close();ancestry_sources.append(str(source))
    fields=load_json(input_path)['snapshot']['fields']
    for c in generate(fields,arm='random',count=100,seed=0):ancestors[c['candidate_id']]=c
    complete={c['candidate_id']:c for c in specs};added=[]
    while True:
        missing={pid for c in complete.values() for pid in c.get('parent_ids',[]) if pid not in complete}
        if not missing:break
        require(missing<=ancestors.keys(),'missing reproducible ancestor AST')
        for pid in sorted(missing):complete[pid]=ancestors[pid];added.append(ancestors[pid])
    extra_formulas={c['formula_hash'] for c in added}-{c['formula_hash'] for c in specs}
    require(len(extra_formulas)<=8,'ancestry exceeds pre-evaluation comparator cap')
    write_new(out/'PARENT_COMPARATORS.json',added)
    body={'schema_version':'fuxi.history_run.1','frozen_at':now(),'plan_path':str(Path(plan_path).resolve()),
          'plan_hash':digest(plan),'candidate_path':str(Path(candidates_path).resolve()),'candidate_hash':digest(specs),
          'input_path':str(Path(input_path).resolve()),'input_sha256':file_hash(input_path),
          'parent_comparators_hash':digest(added),'additional_parent_formulas':len(extra_formulas),
          'protocol_notes_sha256':file_hash(Path(plan_path).parent/'PROTOCOL.md'),
          'data_amendment_sha256':file_hash(Path(plan_path).parent/'DATA_AMENDMENT.json'),
          'price_semantic_audit_sha256':file_hash(Path(plan_path).parent/'PRICE_LAYOUT_AUDIT.json'),
          'parent_sources':ancestry_sources,'parent_use':'paired-comparison controls; not additional winner-selected hypotheses',
          'code_hashes':code_manifest(),'preflight_receipt':preflight_receipt,
          'collection_transport_failures':'retained separately, no formula/feedback sent',
          'implementation_clarifications':{'historical_SME_codes':'002/003 excluded before 2021-04-06 merger',
               'snapshot_availability':'18:30 daily research proxy; archive revision PIT certification remains false',
               'bootstrap_missing_dates':'preserve calendar positions; ratio of valid sums/counts, no missing-as-zero return',
               'screen_inference':'nominal research sensitivity; fixed-family Holm descriptive only, no adaptive FWER claim'}}
    write_new(out/'RUN_FREEZE.json',body)
    code=out/'code';code.mkdir()
    for name in body['code_hashes']:(code/name).write_bytes((Path(__file__).parent/name).read_bytes())
    return body


def _write_gzip(path,value):
    with Path(path).open('xb') as stream:
        with gzip.GzipFile(fileobj=stream,mode='wb',mtime=0) as gz:gz.write(canonical(value))


def _load_gzip(path):
    with gzip.open(path,'rt',encoding='utf-8') as f:return json.load(f)


def _summary(candidate,result,admission,scores,request):
    e1=result['evaluations'][1];e3=result['evaluations'][3];d=e1['diagnostics']
    account=result['candidate_account'];baseline=result['baseline_account']
    return {'candidate_id':candidate['candidate_id'],'formula_hash':candidate['formula_hash'],
            'mechanism':candidate['hypothesis']['mechanism'],'e1_status':e1['eval_status'],'e1_decision':e1['decision'],
            'rank_ic':d['spearman']['mean'],'valid_sections':d['valid_section_count'],'pair_coverage':d.get('research_pair_coverage'),
            'yearly':d.get('yearly'),'exchanges':d.get('exchanges'),'block_sensitivity':d.get('block_sensitivity'),
            'e3_status':e3['eval_status'],'account_path_identified':account['eval_status']=='valid' and baseline['eval_status']=='valid',
            'formal_account_valid':e3['eval_status']=='valid','economic_reward_eligible':e3['economic_reward_eligible'],
            'net_increment':(e3.get('metrics') or {}).get('paired_mean_increment'),
            'descriptive_net_increment':e3['diagnostics'].get('paired_mean_increment'),
            'account_gaps':dict(Counter(r['reason'] for r in account['gaps'])),
            'baseline_gaps':dict(Counter(r['reason'] for r in baseline['gaps'])),
            'blockers':admission['blockers'],'jq_candidate_allowed':admission['jq_candidate_allowed'],
            'behavior_hash':behavioral_signature(scores,request['research_start'],request['research_end']),
            'structural_family':structural_family(candidate['canonical_ast']),
            'parent_comparisons':result.get('learning_parent_comparisons',[])}


def memory_state_usable(store,candidate,cards,generation_states):
    """A reusable formula result does not erase a particular generation's ancestry."""
    card=cards.get(candidate['formula_hash'])
    edge=generation_states.get(candidate['candidate_id'])
    return bool(card and store.state_usable(card['state_id']) and (edge is None or store.state_usable(edge)))


def admit_evaluation(store, evaluation_id, data_receipt_id=None):
    """Task #59 formal admission seam: an independent data-contract receipt is
    required for investment eligibility. Without it the admission stays
    descriptive-only (jq_candidate_allowed=False with a named blocker)."""
    return AcceptanceService(store).research(evaluation_id, data_receipt_id)


def run(output, data_receipt_id=None):
    out=Path(output);frozen=load_json(out/'RUN_FREEZE.json')
    require(file_hash(frozen['input_path'])==frozen['input_sha256'],'frozen data changed')
    plan=load_json(frozen['plan_path']);all_specs=load_json(frozen['candidate_path'])
    require(digest(plan)==frozen['plan_hash'] and digest(all_specs)==frozen['candidate_hash'],'frozen protocol or hypotheses changed')
    if (out/'OUTCOME.json').exists():
        require(all(file_hash(out/'code'/n)==h for n,h in frozen['code_hashes'].items()),'archived trial code changed')
        completed=load_json(out/'OUTCOME.json')
        require(completed['freeze_hash']==digest(frozen),'completed run freeze changed')
        return completed
    require(code_manifest()==frozen['code_hashes'],'code changed after run freeze; create a new batch')
    if frozen.get('preflight_receipt'):
        from .remediation import require_preflight
        require(require_preflight(plan,frozen['input_path'],frozen['code_hashes'])==frozen['preflight_receipt'],'preflight receipt changed')
    payload=load_json(frozen['input_path']);snapshot=HistoricalSnapshot(payload['snapshot'])
    labels=Labels(payload['labels'],snapshot);market=payload['market'];clock=payload['research_clock']
    strict_labels=StrictDriftLabels(labels,snapshot,market)
    base_ast=node('rank',node('return',{'field':'close'},window=20));base_scores=evaluate_ast(base_ast,snapshot)
    cal_end=plan['window']['calibration_end']
    calibration=factor_statistics({d:v for d,v in base_scores.items() if d<=cal_end},snapshot,labels,DEFAULT_POLICY,cal_end+'T23:59:59+08:00')
    policy=policy_from_plan(plan,snapshot,calibration);request=policy['request']
    if (out/'POLICY.json').exists():require(load_json(out/'POLICY.json')==policy,'policy recalibration changed')
    else:write_new(out/'POLICY.json',policy)
    store=EvidenceStore(out/'evidence.sqlite');authority=EvidenceStore(authority_location(out))
    scope=scope_of(snapshot);authority.expose('history_'+digest(frozen),scope,'research');authority.close()
    store.append('historical_protocol',{'freeze':frozen,'policy':policy})
    store.append('exposure_audit',{'scope':scope,'classification':'research_exposed','unknown_earlier_exposure':'not_an_independent_holdout',
                                  'old_performance_imported':False,'old_hypotheses_recomputed':True})
    ancestors=load_json(out/'PARENT_COMPARATORS.json')
    require(digest(ancestors)==frozen['parent_comparators_hash'],'ancestor formulas changed')
    baseline_cache={};spec_map={s['candidate_id']:s for s in all_specs+ancestors};by_formula={}
    for spec in sorted(all_specs,key=lambda x:x['candidate_id']):by_formula.setdefault(spec['formula_hash'],spec)
    results_dir=out/'results';results_dir.mkdir(exist_ok=True)
    summaries={};cards={};started=time.monotonic()
    def compare_parents(candidate, result=None):
        scores=evaluate_ast(candidate['canonical_ast'],snapshot)
        if result is None:result=_load_gzip(results_dir/(candidate['formula_hash']+'.json.gz'))
        comparisons=[]
        for pid in candidate.get('parent_ids',[])[:2]:
            if pid not in spec_map:continue
            parent=spec_map[pid];ps=evaluate_ast(parent['canonical_ast'],snapshot)
            paired=paired_parent(scores,ps,snapshot,labels,request['research_start'],request['research_end'],timestamp(clock))
            paired['parent_id']=pid;parent_path=results_dir/(parent['formula_hash']+'.json.gz')
            if parent_path.exists():
                parent_result=_load_gzip(parent_path)
                paired['account_comparison']=paired_accounts(result['candidate_account'],parent_result['candidate_account'],request)
            else:paired['account_comparison']={'eval_status':'not_evaluated','reason':'parent_account_not_yet_computed'}
            comparisons.append(paired)
        return comparisons
    def assess(candidate):
        key=candidate['formula_hash'];path=results_dir/(key+'.json.gz');meta=results_dir/(key+'.receipt.json')
        scores=evaluate_ast(candidate['canonical_ast'],snapshot)
        if meta.exists():
            receipt=load_json(meta);require(file_hash(path)==receipt['sha256'],'cached evaluation corrupted')
            result=_load_gzip(path);summary=receipt['summary'];card=receipt['card']
            require(result['policy_hash']==policy['policy_hash'] and result['snapshot_hash']==snapshot.hash,'cached scope mismatch')
            require(store.state_usable(card['state_id']),'cached feedback revoked; rebuild requires a new run')
            summaries[key]=summary;cards[key]=card
            return summary
        require(not path.exists(),'interrupted result requires separate audited recovery')
        store.append('candidate',candidate)
        t=time.monotonic()
        result=evaluate_candidate(candidate,scores,base_scores,snapshot,labels,market,policy,clock,top_n=10,baseline_cache=baseline_cache)
        strict=factor_statistics({d:v for d,v in scores.items() if request['research_start']<=d<=request['research_end']},
                                snapshot,strict_labels,DEFAULT_POLICY,clock)
        strict_values=[r['rank_ic'] for r in strict['diagnostics']['sections']]
        result['strict_adjustment_sensitivity']={'status':strict['eval_status'],'rank_ic':strict['diagnostics']['spearman']['mean'],
                   'valid_sections':strict['diagnostics']['valid_section_count'],'n_pairs':strict['n_pairs'],
                   'inference':block_inference_fast(strict_values,block_length=12,replicates=request['bootstrap_replicates'],seed=1729),
                   'interpretation':'same scores; labels touching tiny factor drift unavailable; not a second selection protocol'}
        result['learning_parent_comparisons']=compare_parents(candidate,result)
        # E1 on 2021 remains explicitly calibration/descriptive, never a test holdout.
        result['calibration_year_candidate_description']=factor_statistics({d:v for d,v in scores.items() if d<=cal_end},
                                                        snapshot,labels,DEFAULT_POLICY,cal_end+'T23:59:59+08:00')
        evaluation_id=store.append('evaluation',result)
        admission=admit_evaluation(store,evaluation_id,data_receipt_id)
        state='history_feedback_'+digest([evaluation_id,policy['policy_hash']])
        deps=[evaluation_id]+[cards[spec_map[p]['formula_hash']]['state_id'] for p in candidate.get('parent_ids',[])
                              if p in spec_map and spec_map[p]['formula_hash'] in cards]
        store.register_consumer(state,deps,'reevaluate_on_data_protocol_change')
        card=feedback_card(candidate,result,policy,scope,evaluation_id,state);store.append('feedback_v2',card)
        summary=_summary(candidate,result,admission,scores,request)
        summary['strict_adjustment_sensitivity']=result['strict_adjustment_sensitivity']
        summary['elapsed_seconds']=time.monotonic()-t
        _write_gzip(path,result);write_new(meta,{'sha256':file_hash(path),'summary':summary,'card':card})
        summaries[key]=summary;cards[key]=card
        print(json.dumps({'evaluated':len(summaries),'formula':key[:12],'IC':summary['rank_ic'],
                          'screen':summary['e1_decision'],'account':summary['e3_status'],'seconds':summary['elapsed_seconds']}),flush=True)
        return summary

    # Parents before children where possible, without selecting by observed performance.
    pending=list({s['formula_hash']:s for s in ancestors+list(by_formula.values())}.values());done=set()
    while pending:
        ready=[s for s in pending if all(p not in spec_map or spec_map[p]['formula_hash'] in done for p in s.get('parent_ids',[]))]
        require(ready,'cyclic legacy lineage')
        for spec in ready:assess(spec);done.add(spec['formula_hash']);pending.remove(spec)
    reevaluated=[summaries[k] for k in by_formula]
    pvalues={r['formula_hash']:max((b.get('p_one_sided') or 1.) for b in r['block_sensitivity']) for r in reevaluated}
    family=holm(pvalues,plan['protocol']['alpha'])
    if not (out/'REEVALUATION.json').exists():
        write_new(out/'REEVALUATION.json',{'formulas':reevaluated,'holm_fixed_family_descriptive':family,
                    'global_adaptive_error_control':False,'planning':policy['planning']})
    trial_records=[]
    for seed in plan['search']['seeds']:
        histories={arm:[] for arm in plan['search']['arms']};anchors=list(by_formula.values())
        generation_states={arm:{} for arm in histories}
        seen={arm:set() for arm in histories};local={arm:list(anchors) for arm in histories};previous={arm:None for arm in histories}
        for round_index,action in enumerate(plan['search']['schedule']):
            for arm in (('memory','no_memory') if round_index%2==0 else ('no_memory','memory')):
                decision=out/(f'decision_{seed}_{arm}_{round_index:02}.json')
                if decision.exists():choice=load_json(decision);candidate=choice['candidate']
                else:
                    if arm=='memory':
                        pool=[s for s in local[arm] if memory_state_usable(store,s,cards,generation_states[arm])
                              and summaries[s['formula_hash']]['e1_status']=='valid']
                        pool.sort(key=lambda s:(-(summaries[s['formula_hash']]['rank_ic'] or 0),s['formula_hash']))
                        parents=[];families=set()
                        for s in pool:
                            family_id=structural_family(s['canonical_ast'])
                            if family_id not in families:parents.append(s);families.add(family_id)
                            if len(parents)==3:break
                        rationale='new_scope_valid_feedback_rank_then_structural_diversity; no economic reward'
                    else:
                        # Control selection has no data dependency on evaluator readouts.
                        parents=random.Random(seed*1000+round_index).sample(anchors,3)
                        rationale='hash_frozen_anchor_set_seeded_random; evaluator_feedback_hidden'
                    generation_rejections=[]
                    proposed=generate(snapshot.payload['fields'],arm=action,count=1,seed=seed*1000+round_index,
                                      parents=parents,seen=seen[arm],diagnostics=generation_rejections)
                    require(len(proposed)==1,'generator exhausted fixed candidate budget')
                    candidate=proposed[0]
                    choice={'seed':seed,'arm':arm,'round':round_index,'action':action,'candidate':candidate,
                            'generation_rejections':generation_rejections,
                            'parent_selection':rationale,'parent_ids':[p['candidate_id'] for p in parents],
                            'memory_evidence_ids':[cards[p['formula_hash']]['evaluation_id'] for p in parents] if arm=='memory' else [],
                            'policy_hash':policy['policy_hash'],'snapshot_hash':snapshot.hash,'frozen_at':now()}
                    write_new(decision,choice)
                spec_map[candidate['candidate_id']]=candidate;summary=deepcopy(assess(candidate))
                summary['parent_comparisons']=compare_parents(candidate)
                if summary['candidate_id']!=candidate['candidate_id']:
                    summary['evaluation_source_candidate_id']=summary['candidate_id']
                    summary['candidate_id']=candidate['candidate_id']
                    summary['jq_candidate_allowed']=False
                    summary['blockers']=sorted(set(summary['blockers']+['generation_identity_requires_own_admission']))
                edge_id=store.append('generation_feedback',{'decision_hash':digest(choice),'readout':summary,
                                       'policy_hash':policy['policy_hash'],'snapshot_hash':snapshot.hash})
                edge_state='generation_'+digest(choice)
                dependencies=[cards[candidate['formula_hash']]['state_id'],edge_id]
                dependencies += [cards[spec_map[p]['formula_hash']]['state_id'] for p in candidate['parent_ids']
                                 if p in spec_map and spec_map[p]['formula_hash'] in cards]
                dependencies += [generation_states[arm][p] for p in candidate['parent_ids'] if p in generation_states[arm]]
                if not store.db.execute('SELECT 1 FROM consumers WHERE id=?',(edge_state,)).fetchone():
                    store.register_consumer(edge_state,dependencies,'invalidate_on_parent_or_evaluation_retraction')
                require(store.state_usable(edge_state),'generation feedback dependency revoked')
                generation_states[arm][candidate['candidate_id']]=edge_state
                seen[arm].add(candidate['formula_hash']);local[arm].append(candidate)
                record={**choice,'result':summary,'actual_parent_ids':candidate['parent_ids']};histories[arm].append(record);trial_records.append(record)
            if (round_index+1)%plan['search']['checkpoint_every']==0:
                checks={};stop=False
                for arm,rows in histories.items():
                    recent=rows[-5:];older={r['result']['behavior_hash'] for r in rows[:-5]} | {r['behavior_hash'] for r in reevaluated}
                    new=len({r['result']['behavior_hash'] for r in recent}-older)
                    parents=[p for r in recent for p in r['result']['parent_comparisons'] if p['status']=='valid']
                    positive=sum(p['mean_ic_increment']>0 for p in parents);sterile=new==0 and positive==0
                    checks[arm]={'new_behaviors':new,'positive_parent_comparisons':positive,'parent_comparisons':len(parents),
                                 'formal_account_valid':sum(r['result']['formal_account_valid'] for r in rows),
                                 'blockers':dict(Counter(b for r in rows for b in r['result']['blockers'])),
                                 'sterile':sterile,'continue':not(sterile and previous[arm] and previous[arm]['sterile'])}
                    stop=stop or not checks[arm]['continue'];previous[arm]=checks[arm]
                target=out/(f'checkpoint_{seed}_{round_index+1}.json')
                if not target.exists():write_new(target,{'seed':seed,'round':round_index+1,'checks':checks,'stop_both_arms':stop})
                print(json.dumps({'checkpoint':round_index+1,'seed':seed,'stop':stop}),flush=True)
                if stop:break
    paired=[]
    for seed in plan['search']['seeds']:
        arms={}
        for arm in plan['search']['arms']:
            rows=[r for r in trial_records if r['seed']==seed and r['arm']==arm];values=[r['result']['rank_ic'] for r in rows if r['result']['rank_ic'] is not None]
            parents=[p for r in rows for p in r['result']['parent_comparisons'] if p['status']=='valid']
            arms[arm]={'candidates':len(rows),'mean_rank_ic':mean(values) if values else None,
                       'unique_behaviors':len({r['result']['behavior_hash'] for r in rows}),
                       'unique_structural_families':len({r['result']['structural_family'] for r in rows}),
                       'new_structural_families_vs_legacy':len({r['result']['structural_family'] for r in rows}-{r['structural_family'] for r in reevaluated}),
                       'new_behaviors_vs_legacy':len({r['result']['behavior_hash'] for r in rows}-{r['behavior_hash'] for r in reevaluated}),
                       'positive_parent_comparisons':sum(p['mean_ic_increment']>0 for p in parents),'parent_comparisons':len(parents),
                       'formal_account_valid':sum(r['result']['formal_account_valid'] for r in rows),
                       'JQ':sum(r['result']['jq_candidate_allowed'] for r in rows)}
        paired.append({'seed':seed,'arms':arms,'memory_minus_control_mean_IC':arms['memory']['mean_rank_ic']-arms['no_memory']['mean_rank_ic']})
    outcome={'freeze_hash':digest(frozen),'snapshot_hash':snapshot.hash,'policy_hash':policy['policy_hash'],
             'legacy_unique_formulas':len(reevaluated),'legacy_research_screen_pass':sum(r['e1_decision']=='pass' for r in reevaluated),
             'legacy_identified_account_paths':sum(r['account_path_identified'] for r in reevaluated),
             'legacy_formal_account_valid':sum(r['formal_account_valid'] for r in reevaluated),
             'legacy_JQ':sum(r['jq_candidate_allowed'] for r in reevaluated),'search_pairs':paired,
             'search_attempts':len(trial_records),'unique_evaluations':len(summaries),'external_LLM_calls':0,
             'compiler_rejected_proposals':sum(len(t.get('generation_rejections',[])) for t in trial_records),
             'elapsed_seconds_this_invocation':time.monotonic()-started,'completed_at':now(),
             'conclusion_scope':'selected historical cohort; exposed samples; local generator; no independent or market-wide claim'}
    write_new(out/'TRIALS.json',trial_records);write_new(out/'OUTCOME.json',outcome);store.close();return outcome
