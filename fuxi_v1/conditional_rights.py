"""Source-supported fractional rights bounds with complete native account replays."""
from .contracts import digest,require
from .fractional_sources import verified_rule
from .fractional_branches import enumerate_accounts
from .account_reference import reconcile


def freeze_rules(market,max_paths=8):
    require(type(max_paths)is int and 1<=max_paths<=128,'fractional path budget invalid')
    rules={};sources=[]
    for term in market.get('action_terms',[]):
        if term.get('status')!='ordinary_distribution_supported' or not term.get('share_distribution'):continue
        for source in term.get('issuer_evidence',{}).get('sources',[]):
            checked=verified_rule(term,source)
            if checked is not None:
                rules[term['evidence_id']]=checked
                sources.append({'instrument':term['instrument'],'ex_date':term['ex_date'],
                    'path':source['path'],'sha256':source['sha256'],'text_sha256':source['text_sha256']})
                break
    body={'schema_version':'fuxi.conditional_rights_rules.1','market_digest':digest(market),
          'rules':rules,'sources':sources,'max_paths_per_account_per_scenario':max_paths,
          'probabilities_assigned':False,'actual_holder_allocation_observed':False,'formal_eligibility':False}
    body['rules_hash']=digest(body);return body


def replay(baseline_ir,candidate_ir,market,policy,rules,*,start,end,styles=None,benchmark=None,style_provenance=None):
    require(rules['rules_hash']==digest({k:v for k,v in rules.items() if k!='rules_hash'})
            and rules['market_digest']==digest(market),'conditional rights rules changed')
    require(rules['rules'],'no source-supported fractional rule')
    rows=[]
    for scenario in policy['fractions']:
        accounts={side:enumerate_accounts(ir,market,rules['rules'],max_paths=rules['max_paths_per_account_per_scenario'],
            start=start,end=end,restricted_policy=policy,restricted_scenario=scenario)
            for side,ir in [('baseline',baseline_ir),('candidate',candidate_ir)]}
        references={side:[reconcile(p['account'],market) for p in branch['paths']] for side,branch in accounts.items()}
        require(all(r['n_pairs']>0 and not r['mismatches'] for refs in references.values() for r in refs),
                'conditional native journal reference mismatch')
        pairs=[]
        for a in accounts['candidate']['paths']:
            for b in accounts['baseline']['paths']:
                aa,bb=a['account'],b['account'];increments=[];prior=None
                require([r['session'] for r in aa['daily']]==[r['session'] for r in bb['daily']],
                        'conditional rights windows differ')
                for i,(ar,br) in enumerate(zip(aa['daily'],bb['daily'])):
                    if i==0:
                        prior=(float(ar['nav']),float(br['nav']));continue
                    av=ar['restricted_research']['conditional_total_nav'];bv=br['restricted_research']['conditional_total_nav']
                    if av is None or bv is None:prior=None;continue
                    current=(float(av),float(bv))
                    if prior and min(prior)>0:increments.append(current[0]/prior[0]-current[1]/prior[1])
                    prior=current
                require(increments,'zero conditional rights comparisons')
                row={'candidate_path':a['path_id'],'baseline_path':b['path_id'],'n_pairs':len(increments),
                     'full_return_positions':len(aa['daily'])-1,'mean_net_increment':sum(increments)/len(increments),
                     'actual_holder_allocation_observed':False,'formal_eligibility':False}
                if styles is not None:
                    from .account_attribution import full_account_attribution
                    row['attribution']=full_account_attribution(aa,bb,market,styles,benchmark,style_provenance=style_provenance)
                pairs.append(row)
        require(pairs,'empty conditional path pairs')
        rows.append({'scenario':scenario,'accounts':accounts,'strict_references':references,'pairs':pairs,
                     'enumeration_complete':all(v['enumeration_complete'] for v in accounts.values())})
    return {'schema_version':'fuxi.conditional_rights_replay.1','rules_hash':rules['rules_hash'],
        'policy_hash':policy['policy_hash'],'scenarios':rows,'formal_eligibility':False,
        'actual_holder_allocation_observed':False,'scope':'complete source-supported conditional paths; no probability or observed entitlement claims'}
