"""Opt-in research subledger and conditional total-account loss scenarios.

Strict Account marks, events, ownership uncertainty and cash remain authoritative.
These estimates are neither executable quotes nor investment-grade valuations.
"""
from copy import deepcopy
from datetime import date
from decimal import Decimal
from pathlib import Path
from .account import money
from .contracts import digest, require, timestamp
from .datasets import file_hash
from .action_accounting import pending_shares


def freeze_restricted_policy(market, entries, *, horizon_end, fractions=('0','0.5','1')):
    """Freeze explicitly declared restrictions, evidenced past anchors and sources."""
    require(entries and len({e['instrument'] for e in entries})==len(entries),'empty/duplicate restrictions')
    require(date.fromisoformat(horizon_end).isoformat()==horizon_end,'invalid horizon')
    require(len(fractions)>=2 and len(set(fractions))==len(fractions) and
            all(isinstance(f,str) and 0<=money(f)<=1 for f in fractions),'invalid loss scenarios')
    rows=[]
    for entry in entries:
        require(set(entry)=={'instrument','restrict_from','known_at','known_at_basis','source_paths'},'restriction fields')
        require(entry['instrument'] and entry['source_paths'] and entry['known_at_basis'],'missing restriction evidence')
        start=entry['restrict_from'];date.fromisoformat(start);timestamp(entry['known_at'])
        require(start<=horizon_end and entry['known_at'][:10]<=horizon_end,'restriction beyond horizon')
        source_bars = (market.bar_index[(day,entry['instrument'])] for day in market.sessions if day<start) if getattr(market,"_columnar_market",False) else market['bars']
        candidates=[b for b in source_bars if b['instrument']==entry['instrument'] and b['session']<start
                    and b.get('close') is not None and money(str(b['close']))>0 and b.get('evidence_id')
                    and b.get('paused') is not True and not str(b.get('mark_source','')).startswith('declared_')]
        require(candidates,'no actually observed pre-restriction anchor')
        anchor=max(candidates,key=lambda b:b['session'])
        sources=[{'path':str(Path(p).resolve()),'sha256':file_hash(p)} for p in entry['source_paths']]
        rows.append({k:v for k,v in entry.items() if k!='source_paths'} | {'sources':sources,
          'anchor':{'session':anchor['session'],'price':str(anchor['close']),'evidence_id':anchor['evidence_id'],
                    'bar_hash':digest(anchor)}})
    policy={'schema_version':'fuxi.restricted_research.1','market_digest':digest(market),
      'entries':rows,'horizon_end':horizon_end,'fractions':list(fractions),
      'interpretation':'conditional_loss_stress_not_fair_value_or_bounds',
      'cash_policy':'evidenced_cash_only','ownership_policy':'known_nominal_quantities_unresolved_rights_retained',
      'formal_eligibility':False,'live_execution':False}
    policy['policy_hash']=digest(policy)
    return policy


class RestrictedResearch:
    def __init__(self,market,policy,fraction):
        require(policy.get('policy_hash')==digest({k:v for k,v in policy.items() if k!='policy_hash'}),'restriction policy changed')
        require(policy.get('schema_version')=='fuxi.restricted_research.1' and
                policy.get('formal_eligibility') is False and policy.get('live_execution') is False,
                'restricted mode cannot grant qualification')
        require(policy['market_digest']==digest(market),'restricted market changed')
        require(fraction in policy['fractions'],'undeclared scenario')
        for entry in policy['entries']:
            require(all(file_hash(r['path'])==r['sha256'] for r in entry['sources']),'restriction source changed')
            anchor=entry['anchor']
            source_bars = [market.bar_index.get((anchor['session'],entry['instrument']),{})] if getattr(market,"_columnar_market",False) else market['bars']
            require(any(b.get('instrument')==entry['instrument'] and b.get('session')==anchor['session'] and
                        digest(b)==anchor['bar_hash'] for b in source_bars),'anchor source mismatch')
            require(anchor['session']<entry['restrict_from'],'future anchor')
        self.policy=deepcopy(policy);self.fraction=money(fraction);self.scenario=fraction

    def active(self,when,account):
        require(when[:10]<=self.policy['horizon_end'],'restricted scenario horizon exceeded')
        clock=timestamp(when)
        active={e['instrument']:e for e in self.policy['entries']
                if when[:10]>=e['restrict_from'] and clock>=timestamp(e['known_at'])}
        # A quantity-preserving, evidenced transfer does not release trading restrictions.
        for event in account.journal:
            if event['kind']=='security_transfer' and timestamp(event['timestamp'])<=clock:
                p=event['payload']
                if p['from_instrument'] in active:active[p['to_instrument']]=active[p['from_instrument']]
        return active

    def project(self,account,when,quotes,*,quote_cutoff=None):
        active=self.active(when,account)
        cutoff=timestamp(quote_cutoff or when)
        require(cutoff<=timestamp(when),'future quote cutoff')
        held={s:sum(l['quantity'] for l in lots) for s,lots in account.positions.items() if lots}
        pending=pending_shares(account);codes=set(held)|set(pending)
        claims=sum((v for v in account.receivables.values() if v is not None),Decimal(0))
        liquid=account.cash+claims;restricted=Decimal(0);values={};rows={};missing=[]
        unsupported=[];liquid_missing=[]
        for code in sorted(codes):  # P-20261015-004 deterministic missing/liquid_missing order
            q=held.get(code,0)+pending.get(code,0);quote=quotes.get(code)
            observed=(quote is not None and quote.get('price') is not None and money(str(quote['price']))>0
                      and quote.get('evidence_id') and timestamp(quote['asof'])==cutoff)
            if code not in active:
                if not observed:missing.append(code);liquid_missing.append(code);continue
                price=money(str(quote['price']));values[code]=price;liquid+=price*q;continue
            entry=active[code];anchor=entry['anchor']
            require(timestamp(anchor['session']+'T18:00:00+08:00')<=timestamp(when),'anchor unavailable at cutoff')
            # Genuine quotes after restriction supersede the estimate for that cutoff.
            real=observed and quote['asof'][:10]>=entry['restrict_from']
            incompatible=[e['event_id'] for e in account.journal if e['kind'] in {'split','share_entitlement','share_credit'}
                          and e['payload'].get('instrument')==code and e['timestamp'][:10]>anchor['session']]
            # A stale price on a changed share basis is not a valid estimate.
            if incompatible and not real:unsupported.extend(incompatible);missing.append(code);price=None
            else:price=money(str(quote['price'])) if real else money(anchor['price'])*self.fraction
            if price is not None:values[code]=price;restricted+=price*q
            rows[code]={'settled_and_unsettled_quantity':held.get(code,0),'pending_share_quantity':pending.get(code,0),
              'nominal_quantity':q,'unit_scenario_value':str(price) if price is not None else None,
              'scenario_value':str(price*q) if price is not None else None,'spendable_cash':'0','tradable':False,
              'mark_kind':'observed_quote_still_restricted' if real else 'conditional_anchor_loss_scenario',
              'anchor':anchor,'anchor_age_calendar_days':(date.fromisoformat(when[:10])-date.fromisoformat(anchor['session'])).days,
              'restriction_source_hashes':[s['sha256'] for s in entry['sources']],
              'rights_verified':False,'known_at':entry['known_at'],'known_at_basis':entry['known_at_basis']}
        unresolved={e['event_id']:e['payload'].get('instrument') for e in account.journal if e['event_id'] in account.unresolved}
        other_unknown=[e for e,c in unresolved.items() if c not in active]
        # All restricted ownership uncertainties remain visible, even when nominal sizing can continue.
        available=not missing and not other_unknown and not unsupported
        nav=liquid+restricted if available else None
        return {'schema_version':'fuxi.restricted_mark.1','policy_hash':self.policy['policy_hash'],
          'scenario':self.scenario,'observation_cutoff':when,'restricted_assets':rows,
          'cash':str(account.cash),'cash_receivables':str(claims),'liquid_nav':str(liquid) if not liquid_missing and not other_unknown else None,
          'restricted_scenario_value':str(restricted),'conditional_total_nav':str(nav) if nav is not None else None,
          'conditional_total_return':float(nav/account.initial_cash-1) if nav is not None and account.initial_cash>0 else None,
          'initial_capital':str(account.initial_cash),'nominal_sizing_available':available and nav>0,
          'liquid_nav_to_initial_capital':float(liquid/account.initial_cash) if not liquid_missing and not other_unknown and account.initial_cash>0 else None,
          'unresolved_events':sorted(account.unresolved),'unrestricted_unresolved_events':other_unknown,
          'missing_marks':missing,'unsupported_anchor_basis_events':unsupported,
          'formal_eligibility':False,'rights_verified':False,'value_is_spendable':False,
          '_prices':values,'_active':set(active)}


def public_mark(mark):
    return {k:v for k,v in mark.items() if not k.startswith('_')}


def replay_restricted_pair(baseline_ir,candidate_ir,market,policy,**kwargs):
    """Every declared scenario gets two full, separate native cash/share replays."""
    from .simulation import run_account
    from .account_reference import reconcile
    require(not {'restricted_policy','restricted_scenario'} & set(kwargs),'scenario override forbidden')
    rows=[]
    for scenario in policy['fractions']:
        accounts={name:run_account(ir,market,restricted_policy=policy,restricted_scenario=scenario,**kwargs)
                  for name,ir in [('baseline',baseline_ir),('candidate',candidate_ir)]}
        refs={name:reconcile(a,market) for name,a in accounts.items()}
        require(all(r['n_pairs']>0 and not r['mismatches'] for r in refs.values()),'strict account reference mismatch')
        maps={name:{r['session']:r['restricted_research']['conditional_total_nav']
                    for r in a['daily'] if 'restricted_research' in r} for name,a in accounts.items()}
        days=sorted(set(maps['baseline']) & set(maps['candidate']));increments=[]
        prior=(money(accounts['candidate']['daily'][0]['nav']),money(accounts['baseline']['daily'][0]['nav']))
        for day in days:
            a,b=maps['candidate'][day],maps['baseline'][day]
            if a is None or b is None:prior=None;continue
            if prior and min(prior)>0:increments.append(float(money(a)/prior[0]-money(b)/prior[1]))
            prior=(money(a),money(b))
        require(increments,'no adjacent same-scenario account pairs')
        rows.append({'scenario':scenario,'accounts':accounts,'strict_references':refs,
          'paired_return_count':len(increments),'mean_net_daily_increment':sum(increments)/len(increments),
          'formal_eligibility':False})
    effects=[r['mean_net_daily_increment'] for r in rows]
    return {'schema_version':'fuxi.restricted_pair.1','policy_hash':policy['policy_hash'],'rows':rows,
      'scenario_count':len(rows),'mean_increment_span':[min(effects),max(effects)],
      'direction_sensitive':min(effects)<0<max(effects),'formal_eligibility':False,
      'interpretation':'research_conditional_account_sensitivity_not_independent_economic_confirmation'}


def run_restricted_study(input_path,baseline_path,candidate_path,policy_path,output,*,start,end):
    """Manual operational consumer: freeze input/code identities, replay, then verify."""
    from .contracts import load_json
    from .data import snapshot_from_payload
    from .artifacts import code_manifest
    from .campaign import write_new
    from .evidence import now
    output=Path(output)
    require(not output.exists() or not any(output.iterdir()),'restricted study requires an empty new output')
    inputs={name:{'path':str(Path(p).resolve()),'sha256':file_hash(p)} for name,p in
            [('market',input_path),('baseline',baseline_path),('candidate',candidate_path),('policy',policy_path)]}
    payload=load_json(input_path);baseline=load_json(baseline_path);candidate=load_json(candidate_path);policy=load_json(policy_path)
    snapshot=snapshot_from_payload(payload['snapshot'])
    require(baseline['snapshot_hash']==candidate['snapshot_hash']==snapshot.hash,'restricted IR/snapshot mismatch')
    require(start<=end and start in payload['market']['sessions'] and end in payload['market']['sessions'],'restricted study window')
    protocol={'schema_version':'fuxi.restricted_study.1','frozen_at':now(),'inputs':inputs,'code_hashes':code_manifest(),
              'start':start,'end':end,'policy_hash':policy['policy_hash'],'qualification_allowed':False,
              'scope':'conditional_research_only_all_scenarios_full_account_replay'}
    write_new(output/'PROTOCOL.json',protocol)
    result=replay_restricted_pair(baseline,candidate,payload['market'],policy,start=start,end=end)
    require(code_manifest()==protocol['code_hashes'] and all(file_hash(r['path'])==r['sha256'] for r in inputs.values()),
            'restricted study source/code drift')
    write_new(output/'RESULT.json',result)
    write_new(output/'COMPLETE.json',{'result_hash':digest(result),'protocol_hash':digest(protocol),
              'scenario_count':result['scenario_count'],'formal_eligibility':False})
    return {'output':str(output.resolve()),'scenario_count':result['scenario_count'],
            'paired_counts':[r['paired_return_count'] for r in result['rows']],'formal_eligibility':False}


if __name__=='__main__':
    import argparse,json
    from .runtime import controlled_writer
    parser=argparse.ArgumentParser(description='Manual restricted-asset research; no qualification or live orders')
    for name in ['input','baseline','candidate','policy','output','start','end']:parser.add_argument('--'+name,required=True)
    args=parser.parse_args()
    with controlled_writer(args.output):
        print(json.dumps(run_restricted_study(args.input,args.baseline,args.candidate,args.policy,args.output,
                         start=args.start,end=args.end),ensure_ascii=False))
