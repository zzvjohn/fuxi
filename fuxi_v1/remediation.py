"""Frozen account preflight, before any expanded search budget is consumed."""
from pathlib import Path
from copy import deepcopy
from collections import Counter
from .contracts import load_json,digest,require
from .datasets import file_hash
from .historical_adapter import HistoricalSnapshot
from .dsl import evaluate_ast
from .search import node
from .portfolio import strategy_spec,compile_strategy,mix_scores
from .simulation import run_account
from .action_accounting import reconcile_account
from .artifacts import code_manifest
from .campaign import write_new
from .historical_study import _write_gzip
from .evidence import now


def preflight(input_path,plan_path,output):
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    p=load_json(input_path);plan=load_json(plan_path);s=HistoricalSnapshot(p['snapshot']);market=p['market']
    close={'field':'close'}
    asts={'momentum20':node('rank',node('return',close,window=20)),
          'reversal5':node('rank',node('neg',node('return',close,window=5))),
          'reversal20':node('rank',node('neg',node('return',close,window=20))),
          'price_cv20_low':node('rank',node('neg',node('div',node('std',close,window=20),node('mean',close,window=20))))}
    freeze={'created_at':now(),'plan_hash':digest(plan),'input_sha256':file_hash(input_path),'input_path':str(Path(input_path).resolve()),
        'code_hashes':code_manifest(),'asts':asts,'window':plan['window'],'top_n':10,
        'gate':'all four probes x normal/double_cost/half_capacity must be identifiable and independently reconciled',
        'probe_strategy':'50% candidate rank + 50% frozen momentum rank; baseline standalone momentum',
        'independent_confirmation':False}
    if (out/'FREEZE.json').exists():
        old=load_json(out/'FREEZE.json')
        require(all(old[k]==v for k,v in freeze.items() if k!='created_at'),'preflight scope changed');freeze=old
    else:write_new(out/'FREEZE.json',freeze)
    base=evaluate_ast(asts['momentum20'],s);readouts=[]
    for name,ast in asts.items():
        scores=base if name=='momentum20' else mix_scores(base,evaluate_ast(ast,s),.5)
        spec=strategy_spec('preflight_'+name,top_n=10,max_weight=.1,rebalance_sessions=5)
        ir=compile_strategy(spec,scores,s,digest(market['execution']))
        for scenario,cost,cap in [('normal',1.,1.),('double_cost',2.,1.),('half_capacity',1.,.5)]:
            path=out/(name+'_'+scenario+'.json.gz');receipt=out/(name+'_'+scenario+'.receipt.json')
            if receipt.exists():
                row=load_json(receipt);require(file_hash(path)==row['sha256'],'probe cached result changed');readouts.append(row);continue
            result=run_account(ir,market,start=plan['window']['research_start'],end=plan['window']['end'],cost_multiplier=cost,participation_multiplier=cap)
            reconciled=reconcile_account(result,market['execution']['initial_cash'],market)
            _write_gzip(path,result)
            unknown=[r for r in result['gaps'] if r['reason']!='pretrade_NAV_unidentifiable']
            row={'name':name,'scenario':scenario,'status':result['eval_status'],'formal_evidence':result['formal_evidence'],
                'journal_kinds':dict(Counter(e['kind'] for e in result['journal'])),'reconciled':reconciled,'sha256':file_hash(path),
                'gaps':dict(Counter(r['reason'] for r in result['gaps'])),'root_gaps':unknown,'freeze_hash':digest(freeze),
                'identified_days':sum(d['valuation_identifiable'] for d in result['daily'][1:]),
                'nav_final':result['daily'][-1]['nav'] if result['eval_status']=='valid' else None}
            write_new(receipt,row);readouts.append(row)
            print(name,scenario,row['status'],row['gaps'],flush=True)
    decision={'freeze_hash':digest(freeze),'plan_hash':digest(plan),'input_sha256':freeze['input_sha256'],'code_hashes':freeze['code_hashes'],
        'probe_count':len(readouts),'search_allowed':len(readouts)==12 and all(r['status']=='valid' and r['reconciled']['cash_and_share_reconciled'] and r['reconciled']['nav_reconciled'] for r in readouts),
        'readouts':readouts,'formal_account_evidence_certified':all(r['formal_evidence'] for r in readouts),
        'certification_hash':((p.get('audit') or {}).get('data_certification') or {}).get('certification_hash'),
        'certification_flags':((p.get('audit') or {}).get('data_certification') or {}).get('flags'),
        'scope':'engineering/research preflight; no JQ or independent confirmation certificate'}
    write_new(out/'DECISION.json',decision);return decision


def require_preflight(plan,input_path,code_hashes):
    if 'remediation' not in plan:return None
    path=Path(input_path).parent/'preflight'/'DECISION.json'
    receipt=load_json(path)
    require(receipt['plan_hash']==digest(plan) and receipt['input_sha256']==file_hash(input_path),'preflight data/protocol mismatch')
    require(receipt['code_hashes']==code_hashes and receipt['probe_count']==12,'preflight code/probe mismatch')
    expected={(n,s) for n in ('momentum20','reversal5','reversal20','price_cv20_low') for s in ('normal','double_cost','half_capacity')}
    require(len(receipt['readouts'])==12 and {(r['name'],r['scenario']) for r in receipt['readouts']}==expected,'preflight probes incomplete')
    require(receipt['search_allowed'] is True and all(r['status']=='valid' and r['reconciled']['cash_and_share_reconciled'] is True and
            r['reconciled']['nav_reconciled'] is True and r['reconciled']['sessions']>0 for r in receipt['readouts']),'preflight blocked expanded search')
    frozen=load_json(path.parent/'FREEZE.json');require(digest(frozen)==receipt['freeze_hash'],'preflight freeze changed')
    for row in receipt['readouts']:
        name=row['name']+'_'+row['scenario']
        require(load_json(path.parent/(name+'.receipt.json'))==row and file_hash(path.parent/(name+'.json.gz'))==row['sha256'],
                'preflight probe artifact changed')
    return {'path':str(path.resolve()),'sha256':file_hash(path)}
