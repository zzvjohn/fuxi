"""Compact historical input consumed by the native DSL, labels and account engine."""
from bisect import bisect_right
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import date
from pathlib import Path
from .contracts import digest, require, finite, timestamp, load_json
from .datasets import read_calendar, file_hash, materialize_labels, execution_profile
from .history import iso, jq_code, name_state, adjustment_class, normalize_price_frame
from .pit_universe import FRAME_PREDICATE


class HistoricalSnapshot:
    """Columnar, hash-bound historical daily snapshot; no silent raw/adjusted mixing.

    A raw lookback crossing a detected action is unavailable. Compact storage avoids
    millions of repeated provenance dictionaries; the manifest applies to every cell.
    """
    def __init__(self, payload):
        p=self.payload=deepcopy(payload)
        require(p['schema_version']=='fuxi.historical_snapshot.1','historical snapshot schema')
        require(p['exposure']=='research_exposed','historical snapshots cannot declare independent holdout')
        self.sessions=p['sessions']; self.start=self.sessions.index(p['evaluation_start'])
        require(self.sessions==sorted(set(self.sessions)) and self.sessions,'calendar ordering')
        self.universe=p['universe'];self.clocks={d:timestamp(p['decision_times'][d]) for d in self.sessions}
        self._clock_indices={v:i for i,v in enumerate(self.clocks.values())}
        self._codes={s:i for i,s in enumerate(p['instruments'])}
        require(len(self._codes)==len(p['instruments']),'duplicate instruments')
        require(set(self.universe)==set(self.sessions),'universe calendar mismatch')
        for d in self.sessions:
            require(self.universe[d] and len(set(self.universe[d]))==len(self.universe[d]) and
                    set(self.universe[d])<=set(self._codes),'empty/invalid historical universe')
        for field, matrix in p['columns'].items():
            require(field in p['fields'] and len(matrix)==len(self.sessions),'column shape mismatch')
            for row in matrix:
                require(len(row)==len(self._codes) and all(x is None or finite(x) for x in row),'invalid compact values')
        require(set(p['columns'])==set(p['fields']),'missing compact fields')
        require(len(p['barriers'])==len(self.sessions),'barrier calendar mismatch')
        for i,row in enumerate(p['barriers']):
            require(len(row)==len(self._codes) and all(type(x) is int and -1<=x<=i for x in row),'invalid history barrier')
        self.hash=digest(p)
        self._score_cache={}

    def value(self,index,instrument,field,asof):
        if index<0 or index>=len(self.sessions) or instrument not in self._codes:return None
        current=self._clock_indices.get(asof)
        if current is None:
            current=bisect_right(list(self.clocks.values()),asof)-1
        if current<index:return None
        column=self._codes[instrument]
        # Mask both price and share-volume histories across actions; no retrospective adjustment.
        if index<self.payload['barriers'][current][column]:return None
        return self.payload['columns'][field][index][column]


def _read_subset(path,codes,dates=None):
    import pandas as pd
    if (Path(path).parent/'CATALOG.json').is_file():
        from .market_catalog import load_table
        return load_table(Path(path).parent,Path(path).name,codes,dates)
    chunks=[]
    for frame in pd.read_csv(path,dtype=str,chunksize=200000):
        frame=frame[frame.ts_code.isin(codes)]
        if dates is not None:frame=frame[frame.trade_date.isin(dates)]
        chunks.append(frame)
    return pd.concat(chunks,ignore_index=True).astype(object).where(lambda x:x.notna(),None)


def dividend_terms(records,sessions):
    """Deduplicate implemented terms; conflicting rows and unsupported share events stay unknown."""
    groups=defaultdict(list)
    for r in records:
        ex=iso(r.get('ex_date'))
        if r.get('div_proc')=='实施' and ex and sessions[0]<=ex<=sessions[-1]:groups[(r['ts_code'],ex)].append(r)
    terms=[]
    for (code,ex),rows in sorted(groups.items()):
        from .action_evidence import canonical_dividend_row
        def number_from(row,key):
            try:
                v=float(row.get(key));return v if finite(v) else None
            except (ValueError,TypeError):return None
        resolved=[(canonical_dividend_row(row),row) for row in rows]
        unique={digest(c):(c,row) for c,row in resolved}
        audit={}
        rights=lambda c:{k:v for k,v in c.items() if k!='imp_ann_date'}
        # Rule A (P-009): rows whose holder-rights fields are identical and differ
        # only in the implementation-publication date are one event published twice;
        # keep the latest publication. Economic terms are identical either way.
        if len(unique)>1 and len({digest(rights(c)) for c,_ in unique.values()})==1:
            audit['imp_ann_date_variants']=sorted({c['imp_ann_date'] for c,_ in unique.values()})
            best=max((c for c,_ in unique.values()),key=lambda c:(c['imp_ann_date'] is not None,c['imp_ann_date'] or ''))
            unique={digest(best):next((c,row) for c,row in unique.values() if c==best)}
        # Rule C2 (P-009): a vendor row whose cash equals the sum of the remaining
        # rows duplicates the already-aggregated payment; drop it so the period sum
        # below reproduces the vendor's own total instead of double-counting it.
        if len(unique)>1:
            cs=[c for c,_ in unique.values()]
            if all((c['stk_div'] or 0)==0 for c in cs):
                from decimal import Decimal
                cash=[Decimal(str(c['cash_div_tax'])) for c in cs if c['cash_div_tax'] is not None]
                if len(cash)==len(cs) and len(cash)>=3:
                    total=sum(cash,Decimal(0))
                    for value in cash:
                        if abs(total-2*value)<Decimal('1e-9'):
                            audit['vendor_sum_row_removed']=float(value)
                            keep=[c for c in cs if abs(Decimal(str(c['cash_div_tax']))-value)>=Decimal('1e-9')]
                            unique={digest(c):next((cc,row) for cc,row in unique.values() if cc==c) for c in keep}
                            break
        # Rule C3 (P-009): within one period, differing cash amounts paired with
        # distinct plan-publication dates are a revised plan; the later publication
        # supersedes the earlier one. All variants remain visible in the audit.
        if len(unique)>1:
            cs=[c for c,_ in unique.values()]
            if len({iso(row.get('end_date')) for _,row in unique.values()})==1 and \
               len({c['cash_div_tax'] for c in cs})>=2 and all((c['stk_div'] or 0)==0 for c in cs):
                anns=[iso(row.get('ann_date')) for _,row in unique.values()]
                if len({a for a in anns if a})>=2:
                    audit['cash_plan_variants']=sorted({c['cash_div_tax'] for c in cs})
                    best=max(unique.values(),key=lambda cr:iso(cr[1].get('ann_date')) or '')
                    unique={digest(best[0]):best}
        r=next(iter(unique.values()))[1]
        def number(key):
            try:
                v=float(r.get(key));return v if finite(v) else None
            except (ValueError,TypeError):return None
        record,pay,ann=iso(r.get('record_date')),iso(r.get('pay_date')),iso(r.get('imp_ann_date'))
        shares,gross=number('stk_div'),number('cash_div_tax')
        periods=defaultdict(dict)
        for c,item in unique.values():periods[iso(item.get('end_date'))][digest(c)]=item
        components=[next(iter(group.values())) for group in periods.values()]
        same_clock=len({digest({k:iso(item.get(k)) for k in ('record_date','pay_date','div_listdate','imp_ann_date')}) for item in components})==1
        strict_quarter_aggregate=(len(periods)>1 and all(p and p[5:] in ('03-31','06-30','09-30','12-31') for p in periods) and
            all(len(group)==1 for group in periods.values()) and
            all(number_from(item,'stk_div')==0 and number_from(item,'cash_div_tax') is not None and number_from(item,'cash_div_tax')>0 for item in components) and
            same_clock)
        # Relaxed aggregation is allowed only when the vendor's own sum row
        # corroborates it (Rule C2); without corroboration non-quarter-end periods
        # stay ambiguous and the term stays conflicting.
        relaxed_aggregate=(not strict_quarter_aggregate and 'vendor_sum_row_removed' in audit and len(periods)>1 and
            all(len(group)==1 for group in periods.values()) and
            all(number_from(item,'stk_div')==0 and number_from(item,'cash_div_tax') is not None and number_from(item,'cash_div_tax')>0 for item in components) and
            same_clock)
        aggregate=strict_quarter_aggregate or relaxed_aggregate
        if aggregate:
            from decimal import Decimal
            gross=float(sum((Decimal(str(number_from(item,'cash_div_tax'))) for item in components),Decimal(0)))
        status='cash_terms_supported'
        if len(unique)>1 and not aggregate:status='conflicting_dividend_rows'
        elif shares is not None and shares!=0:status='share_distribution_requires_settlement_adapter'
        elif shares is None or gross is None or not record or not pay or not ann:status='dividend_terms_missing'
        elif not ann<=record<ex<=pay or gross<0:status='dividend_dates_or_amount_conflict'
        term={'instrument':jq_code(code),'ex_date':ex,'record_date':record,'pay_date':pay,
                      'ann_date':ann,'gross_cash_per_share':gross,'share_distribution':shares,
                      'status':status,'evidence_id':digest(rows if aggregate else [row for _,row in unique.values()]),
                      'tax_fraction':'0.20','tax_model':'flat_20_percent_research_sensitivity_not_investor_tax_receipt'}
        if aggregate:
            term.update(cash_component_count=len(periods),cash_aggregation=
                ('distinct implemented periods with identical entitlement/payment/publication clocks corroborated by the vendor sum row; vendor research proxy'
                 if relaxed_aggregate else
                 'distinct implemented quarter/annual periods with identical entitlement/payment/publication clocks; vendor research proxy'))
            if relaxed_aggregate:term['aggregation_corroborated_by_vendor_sum_row']=True
        if audit:term['vendor_row_resolution_audit']=audit
        terms.append(term)
    return terms


def apply_action_term_exemptions(terms,spec):
    """P-009 (a3): versioned, hash-bound registry declaring why a specific
    unresolved vendor event stays unresolved. Certification then treats the
    event as a declared residual gap, never as an unobserved contradiction."""
    require(file_hash(Path(spec['path']))==spec['sha256'],'action term exemption registry changed')
    registry=load_json(spec['path'])
    require(registry['schema_version']=='fuxi.action_term_exemptions.1','action term exemption schema')
    entries={(e['instrument'],e['ex_date']):e for e in registry['entries']}
    require(len(entries)==len(registry['entries']),'duplicate action term exemption')
    for term in terms:
        entry=entries.get((term['instrument'],term['ex_date']))
        if entry is not None:
            require(entry['status']==term['status'],'exemption status mismatch for '+term['instrument']+' '+term['ex_date'])
            term['exemption']={'reason':entry['reason'],'declared_at':entry.get('declared_at',registry['declared_at']),
                               'registry_hash':spec['sha256']}
    bound={(t['instrument'],t['ex_date']) for t in terms if 'exemption' in t}
    require(bound==set(entries),'action term exemption registry entries not all bound')
    return terms


def build_history_input(raw_root,plan,supplement_root,output,*,remediation=None,certification=None):
    import pandas as pd
    require('remediation' not in plan or remediation is not None,'remediation plan requires explicit versioned evidence configuration')
    root,supp,out=Path(raw_root),Path(supplement_root),Path(output)
    out.mkdir(parents=True,exist_ok=True)
    catalog_manifest=None;source_audit=None
    if (root/'CATALOG.json').exists():
        catalog_manifest=load_json(root/'CATALOG.json')
        require(catalog_manifest['manifest_hash']==digest({k:v for k,v in catalog_manifest.items() if k!='manifest_hash'}),'history catalog changed')
        require(plan.get('catalog_manifest_hash')==catalog_manifest['manifest_hash'],'history plan/catalog identity mismatch')
        metadata=load_json(root/'CATALOG_METADATA.json')
        require(metadata['metadata_hash']==digest({k:v for k,v in metadata.items() if k!='metadata_hash'}) and
            plan.get('catalog_metadata_hash')==metadata['metadata_hash'],'history metadata changed')
        for filename,expected in metadata['files'].items():
            actual=supp/filename if filename in ('namechange.csv','dividend.csv') else root/filename
            require(file_hash(actual)==expected,'history catalog metadata file changed: '+filename)
        if plan.get('catalog_source_mode')=='verified_immutable_partitions':
            from .market_catalog import verify_frozen_sources
            source_audit=verify_frozen_sources(root,catalog_manifest)
    start,end=plan['window']['start'],plan['window']['end']
    sessions,calendar_audit=read_calendar(root/'trade_calendar.csv',start,end)
    require(plan['window']['calibration_end'] in sessions and plan['window']['research_start'] in sessions,'explicit window not in calendar')
    require(sessions.index(plan['window']['research_start'])-sessions.index(plan['window']['calibration_end'])>plan['window']['horizon_sessions'],'calibration/research embargo too short')
    codes=[r['ts_code'] for r in plan['cohort']['members']];dates={d.replace('-','') for d in sessions}
    tables={name:_read_subset(root/name,codes,dates) for name in ['daily_prices.csv','adj_factor_daily.csv','stk_limit.csv','suspend_d.csv']}
    census=plan['cohort'].get('rule_kind')=='all_a_share_census'
    # An explicitly frozen diagnostic cohort can consume source-bound events and
    # daily fundamentals without falsely declaring that it is the all-A census.
    source_bound=census or (plan['cohort'].get('rule_kind')=='frozen_research_cohort' and
                           plan.get('account_evidence_mode')=='source_bound_research')
    if plan['cohort'].get('rule_kind')=='full_population_hash_sample' and plan.get('account_evidence_mode')=='source_bound_research':
        from .cohort_input import verify_declared_cohort
        verify_declared_cohort(plan,root,sessions)
        source_bound=True
    if plan.get('halted_distribution_policy') is not None:
        policy=plan['halted_distribution_policy']
        require(source_bound and policy.get('mode')=='declared_supported_distribution_reference' and
                policy.get('rounding') in ('half_up','floor','ceil'),'halt distribution policy invalid')
    if source_bound and catalog_manifest is not None:
        from .panel_audit import numeric_contract
        basic,basic_fields,good=numeric_contract('daily_basic',_read_subset(root/'daily_basic.csv',codes,dates))
        basic.loc[~good,basic_fields]=None
        tables['daily_basic.csv']=basic.astype(object).where(lambda f:f.notna(),None)
    tables['daily_prices.csv']=normalize_price_frame(tables['daily_prices.csv'])
    layout_audit=tables['daily_prices.csv'][['ts_code','trade_date','field_contract_status']]
    layout_audit.to_csv(out/'price_field_contract.csv',index=False,lineterminator='\n')
    maps={}
    for name,frame in tables.items():
        if name!='suspend_d.csv':require(not frame.duplicated(['ts_code','trade_date']).any(),'duplicate market key: '+name)
        maps[name]={(r['ts_code'],r['trade_date']):r for r in frame.to_dict('records')}
    overlay_sources={};crosscheck={};crosscheck_sources={};corroboration=[]
    if catalog_manifest is not None:
        from .market_catalog import factor_corroboration
        crosscheck,crosscheck_sources=factor_corroboration(root,catalog_manifest,codes,dates)
    independent_quotes={};independent_events={};independent_sources={}
    if remediation is not None:
        from .action_evidence import verified_adjustment_overlay
        maps['adj_factor_daily.csv'],overlay_sources=verified_adjustment_overlay(remediation['adjustment_directory'],codes,dates)
        if remediation.get('factor_crosscheck_directory'):
            from .action_evidence import load_factor_crosscheck
            crosscheck,crosscheck_sources=load_factor_crosscheck(remediation['factor_crosscheck_directory'],codes)
        if remediation.get('independent_event_directory'):
            from .action_evidence import load_independent_events
            independent_quotes,independent_events,independent_sources=load_independent_events(remediation['independent_event_directory'])
    elif source_bound and plan.get('independent_event_directory'):
        from .action_evidence import load_independent_events
        independent_quotes,independent_events,independent_sources=load_independent_events(plan['independent_event_directory'])
    halts=defaultdict(list)
    for r in tables['suspend_d.csv'].to_dict('records'):halts[(r['ts_code'],r['trade_date'])].append(r)
    names=pd.read_csv(supp/'namechange.csv',dtype=str).astype(object);names=names.where(names.notna(),None)
    dividends=pd.read_csv(supp/'dividend.csv',dtype=str).astype(object);dividends=dividends.where(dividends.notna(),None)
    byname=defaultdict(list)
    for row in names.to_dict('records'):byname[row['ts_code']].append(row)
    terms=dividend_terms(dividends.to_dict('records'),sessions)
    if remediation is not None:
        from .action_evidence import resolve_action_terms
        terms=resolve_action_terms(terms,dividends.to_dict('records'),remediation['issuer_adjudications'])
    elif source_bound:
        from .action_evidence import resolve_action_terms
        evidence=plan.get('issuer_action_evidence')
        adjudications=[]
        if evidence:
            require(file_hash(evidence['path'])==evidence['sha256'], 'issuer action evidence changed')
            bundle=load_json(evidence['path'])
            require(bundle['schema_version']=='fuxi.public_implementation_bundle.1' and
                    bundle['independent_assurance'] is False, 'issuer action bundle contract')
            selected={jq_code(code) for code in codes}
            adjudications=[j for j in bundle['adjudications'] if j['instrument'] in selected and
                           sessions[0]<=j['ex_date']<=sessions[-1]]
        terms=resolve_action_terms(terms,dividends.to_dict('records'),adjudications)
    selected_codes={jq_code(code) for code in codes}
    from .halt_evidence import load_halt_starts
    issuer_halts=load_halt_starts(plan.get('issuer_halt_evidence'),selected_codes,sessions)
    terms=[term for term in terms if term['instrument'] in selected_codes]
    if plan.get('action_term_exemptions'):
        terms=apply_action_term_exemptions(terms,plan['action_term_exemptions'])
    from .action_accounting import SUPPORTED
    byex={(t['instrument'],t['ex_date']):t for t in terms}
    fields=['open','high','low','close','volume','amount']
    basic_features=['turnover_rate','total_shares','float_shares','free_shares'] if 'daily_basic.csv' in tables else []
    fields+=basic_features
    units={f:'shares' if f=='volume' or f.endswith('_shares') else 'currency' if f=='amount' else 'ratio' if f=='turnover_rate' else 'price' for f in fields}
    columns={f:[] for f in fields}
    master={r['ts_code']:r for r in plan['cohort']['members']}
    members,universe_rows,bars,barriers=[],[],[],[];class_counts=Counter();state_counts=Counter();eligible_classes=Counter()
    reasons_by_session={}
    previous_barrier=[-1]*len(codes);last_quotes={};last_adj={};last_prices={};last_days={}
    drift_audit=[];halt_reference_audit=[];last_halt_distribution={};last_observed_quotes={}
    for index,day in enumerate(sessions):
        today=[];barrier_row=previous_barrier.copy();values={f:[] for f in fields};day_reasons=Counter()
        for col,code in enumerate(codes):
            instrument=jq_code(code);key=(code,day.replace('-',''));row=maps['daily_prices.csv'].get(key,{})
            def num(source,name,multiple=1):
                try:
                    v=float(source.get(name))*multiple;return v if finite(v) else None
                except (ValueError,TypeError):return None
            raw={f:num(row,'vol',100) if f=='volume' else num(row,f,1000 if f=='amount' else 1) for f in fields if f not in basic_features}
            if basic_features:
                basic=maps['daily_basic.csv'].get(key,{})
                raw.update(turnover_rate=num(basic,'turnover_rate',.01),
                    **{f:num(basic,f.replace('_shares','_share'),10000) for f in basic_features if f!='turnover_rate'})
            adj=num(maps['adj_factor_daily.csv'].get(key,{}),'adj_factor');prior=last_adj.get(code)
            prior_close=last_prices.get(code);pre_close=num(row,'pre_close')
            term=byex.get((instrument,day))
            classification=adjustment_class(prior,adj,prior_close,pre_close,term is not None)
            if classification=='factor_or_reference_conflict' and term is None and code in last_halt_distribution:
                from .cohort_input import resumed_distribution_reference
                if resumed_distribution_reference(last_halt_distribution[code],prior,adj,pre_close):
                    classification='resume_reference_corroborates_supported_halted_distribution'
            if classification=='factor_or_reference_conflict' and term is None and crosscheck:
                from .action_evidence import constant_factor_corroborated
                before=crosscheck.get((code,last_days.get(code)));after=crosscheck.get(key)
                agreed=constant_factor_corroborated(before,after,prior_close,pre_close,raw['close'])
                corroboration.append({'instrument':instrument,'session':day,'previous_adj':prior,'current_adj':adj,
                    'other_table_before':before,'other_table_after':after,'accepted':agreed,'formal_certified':False})
                if agreed:classification='constant_factor_corroborated_by_stk_factor'
            if classification=='factor_or_reference_conflict' and term is None and independent_quotes:
                from .action_evidence import independent_no_event
                previous_day=iso(last_days.get(code))
                before=independent_quotes.get((code,previous_day));after=independent_quotes.get((code,day))
                agreed=independent_no_event(before,after,independent_events.get(code),day,prior_close,pre_close,raw['close'])
                corroboration.append({'instrument':instrument,'session':day,'source':'baostock','previous_adj':prior,'current_adj':adj,
                    'independent_before':before,'independent_after':after,'event_present':day in independent_events.get(code,set()),
                    'accepted':agreed,'formal_certified':False})
                if agreed:classification='no_event_corroborated_by_independent_source'
            listed=iso(master[code]['list_date']);delisted=iso(master[code].get('delist_date'))
            if census and code.endswith('.BJ'):listed=max(listed,'2021-11-15')
            alive=listed<=day and (not delisted or day<delisted)
            if census and not alive and day<listed:
                # NEEQ quotations are not a prelisting A-share feature history.
                raw={f:None for f in fields}
            records=halts.get(key,[])
            issuer_halt=issuer_halts.get((instrument,day))
            full=bool(issuer_halt) or any(r['suspend_type']=='S' and not r.get('suspend_timing') for r in records)
            partial=any(r['suspend_type']=='S' and r.get('suspend_timing') for r in records)
            conflict=full and raw['volume'] is not None and raw['volume']>0
            paused=None if conflict else bool(full or partial) if records or issuer_halt else (False if raw['volume'] and raw['volume']>0 else None)
            # On a full-day recorded halt retain a declared last-close valuation, not a fabricated raw observation.
            mark=raw['close'];mark_source='observed_close';halt_reference=None
            if mark is None and full and not conflict and code in last_quotes:
                mark=last_quotes[code];mark_source='declared_frozen_last_close_on_issuer_halt' if issuer_halt else 'declared_frozen_last_close_on_vendor_halt'
                if term is not None:
                    # A halted price is a declared valuation reference, never a raw
                    # observation. Cash paid during the halt must leave that price
                    # before cash/receivables enter NAV; otherwise income is doubled.
                    # Noncash/differentiated rights require their own reference model.
                    from decimal import Decimal
                    distribution_reference=None
                    if plan.get('halted_distribution_policy') is not None:
                        from .cohort_input import halted_distribution_reference
                        distribution_reference=halted_distribution_reference(term,mark,plan['halted_distribution_policy'],
                            last_observed_quotes.get(code),prior_factor=prior,current_factor=adj)
                    supported_cash=(term['status']=='cash_terms_supported' and
                        term.get('share_distribution')==0 and mark is not None and
                        finite(term.get('gross_cash_per_share')) and term['gross_cash_per_share']>=0)
                    adjusted=(Decimal(str(mark))-Decimal(str(term['gross_cash_per_share']))) if supported_cash else None
                    halt_reference={'session':day,'instrument':instrument,'previous_declared_mark':mark,
                        'gross_cash_per_share':term.get('gross_cash_per_share'),'action_evidence_id':term['evidence_id'],
                        'scope':'declared halted cash-ex valuation; not an observed quote or independently certified price'}
                    if distribution_reference is not None:
                        mark=distribution_reference['reference_mark'];last_quotes[code]=mark;last_prices[code]=mark
                        mark_source='declared_ex_distribution_reference_on_vendor_halt'
                        halt_reference.update(distribution_reference,session=day,instrument=instrument)
                        last_halt_distribution[code]=distribution_reference
                    elif adjusted is not None and adjusted>0:
                        mark=float(adjusted);last_quotes[code]=mark;last_prices[code]=mark
                        mark_source='declared_ex_cash_reference_on_vendor_halt'
                        halt_reference['status']='declared_cash_reference'
                    else:
                        mark=None;last_quotes[code]=None
                        mark_source='halt_action_reference_not_identifiable'
                        halt_reference['status']='data_insufficient'
                    halt_reference['reference_mark']=mark;halt_reference_audit.append(halt_reference)
                if adj is not None and prior is not None and adj==prior and term is None:
                    classification='unchanged_factor_on_recorded_full_halt'
            if classification=='missing_evidence' and full and term is None and independent_quotes and mark is not None:
                # A full-halt day with a declared frozen mark can be corroborated
                # by the independent source's own halt record (tradestatus 0,
                # frozen reference price, no event calendar entry). Vendor single-day
                # factor noise on halt days is then not mistaken for an action.
                from .action_evidence import independent_halt_no_event
                previous_day=iso(last_days.get(code))
                before=independent_quotes.get((code,previous_day));after=independent_quotes.get((code,day))
                agreed=independent_halt_no_event(before,after,independent_events.get(code),day,mark)
                corroboration.append({'instrument':instrument,'session':day,'source':'baostock_halt','previous_adj':prior,'current_adj':adj,
                    'independent_before':before,'independent_after':after,'event_present':day in independent_events.get(code,set()),
                    'halt_reference_mark':mark,'accepted':agreed,'formal_certified':False})
                if agreed:classification='halted_no_event_corroborated_by_independent_source'
            if term is not None and term.get('vendor_event_row_missing') and independent_quotes:
                # A vendor-missed cash event declared from issuer text still owes its
                # ex-date price drop to the independent source: event calendar entry
                # present, both sessions trading, and the drop equals the declared
                # exchange reference deduction.
                from .action_evidence import independent_missed_event
                previous_day=iso(last_days.get(code))
                before=independent_quotes.get((code,previous_day));after=independent_quotes.get((code,day))
                agreed=independent_missed_event(before,after,independent_events.get(code),day,term)
                corroboration.append({'instrument':instrument,'session':day,'source':'baostock_missed_event',
                    'previous_adj':prior,'current_adj':adj,'independent_before':before,'independent_after':after,
                    'event_present':day in independent_events.get(code,set()),
                    'declared_cash':term.get('gross_cash_per_share'),
                    'reference_deduction':term.get('reference_cash_deduction'),
                    'accepted':agreed,'formal_certified':False})
                if agreed:classification='missed_vendor_event_corroborated_by_independent_source'
            if (index==0 or census and alive and code not in last_quotes) and adj is not None and raw['close'] is not None and term is None:
                classification='initial_observation_no_prior_holdings'
            known=classification in {'unchanged_reference_and_factor','tiny_drift_consistent_reference_research_only',
                'unchanged_factor_on_recorded_full_halt','initial_observation_no_prior_holdings','constant_factor_corroborated_by_stk_factor','no_event_corroborated_by_independent_source',
                'halted_no_event_corroborated_by_independent_source',
                'missed_vendor_event_corroborated_by_independent_source',
                'resume_reference_corroborates_supported_halted_distribution'} or bool(term and term['status'] in SUPPORTED)
            if census and not alive:known=False
            if classification not in {'unchanged_reference_and_factor','tiny_drift_consistent_reference_research_only','unchanged_factor_on_recorded_full_halt','initial_observation_no_prior_holdings','constant_factor_corroborated_by_stk_factor','no_event_corroborated_by_independent_source',
                                      'halted_no_event_corroborated_by_independent_source',
                                      'missed_vendor_event_corroborated_by_independent_source',
                                      'resume_reference_corroborates_supported_halted_distribution'}:
                barrier_row[col]=index
            if classification=='tiny_drift_consistent_reference_research_only':
                drift_audit.append({'session':day,'instrument':instrument,'previous_adj':prior,'current_adj':adj,
                                    'prior_close':prior_close,'pre_close':pre_close,'formal_certified':False})
            class_counts[classification]+=1
            state,name_row=name_state(byname[code],day);reasons=[]
            if not alive:reasons.append('not_listed_or_delisted')
            if plan['cohort'].get('rule_kind')!='all_a_share_census' and code.startswith(('002','003')) and day<'2021-04-06':reasons.append('SME_before_mainboard_merger')
            if code.endswith('.BJ') and day<'2021-11-15':reasons.append('not_listed_or_delisted')
            if (date.fromisoformat(day)-date.fromisoformat(listed)).days<plan['universe']['listing_age_calendar_days']:reasons.append('listing_age')
            if state!='normal':reasons.append('name_'+state)
            if paused is not False:reasons.append('halt_or_tradability_unknown')
            if raw['close'] is None:reasons.append('quote_missing')
            # A cell whose corporate-action status is unresolved is removed from new
            # selection before it can reach the account, so the tradeable surface is
            # never a place where the account acts on an unknown event. This only ever
            # removes candidates; the unresolved cell is still counted in the audit.
            if not known:reasons.append('corporate_action_unknown')
            if not reasons:
                today.append(instrument);eligible_classes[classification]+=1
            for reason in reasons:state_counts[reason]+=1
            if reasons:day_reasons['|'.join(reasons)]+=1
            universe_rows.append({'session':day,'instrument':instrument,'eligible':not reasons,'reasons':'|'.join(reasons),
                                 'list_date':listed,'delist_date':delisted,'name_status':state,
                                 'name_publication_date_missing':bool(name_row and not iso(name_row.get('ann_date'))),
                                 'name_evidence':digest(name_row) if name_row else None,'halt_evidence':digest(records) if records else None})
            limit=maps['stk_limit.csv'].get(key,{})
            bars.append({'session':day,'instrument':instrument,**{f:raw[f] for f in ['open','high','low','volume']},
                         'close':mark,'mark_source':mark_source,'up_limit':num(limit,'up_limit'),'down_limit':num(limit,'down_limit'),
                         'paused':paused,'corporate_action_known':known,'action_ids':[term['evidence_id']] if term else [],
                         'event_coverage':classification,'evidence_id':digest([plan['raw_sources']['daily_prices.csv']['sha256'],day,code]),
                         'sizing_reference_close':pre_close if term and term['status'] in SUPPORTED else None,
                         'sizing_reference_evidence':digest([row.get('pre_close'),term]) if term else None})
            if issuer_halt is not None:bars[-1]['issuer_full_halt']=issuer_halt
            if partial and not full and paused is True:
                from .halt_evidence import partial_halt_mark
                proof=partial_halt_mark(records,instrument,day,raw['close'],raw['volume'],
                    plan['raw_sources']['suspend_d.csv']['sha256'],plan['raw_sources']['daily_prices.csv']['sha256'])
                if proof is not None:
                    bars[-1]['partial_halt_mark']=proof
                    bars[-1]['mark_source']='observed_close_on_vendor_partial_halt'
            if halt_reference is not None:
                bars[-1]['halt_reference']=halt_reference
                if mark is not None:
                    bars[-1]['sizing_reference_close']=mark
                    bars[-1]['sizing_reference_evidence']=digest(halt_reference)
            for f in fields:values[f].append(raw[f])
            if raw['close'] is not None:
                last_quotes[code]=raw['close'];last_prices[code]=raw['close'];last_days[code]=key[1]
                last_observed_quotes[code]={'session':day,'close':raw['close'],
                    'evidence_id':bars[-1]['evidence_id']}
                last_halt_distribution.pop(code,None)
            if adj is not None and (not source_bound or alive):last_adj[code]=adj
        require(len(today)>=10,'auditable universe below 10 instruments on '+day)
        members.append(today);barriers.append(barrier_row);previous_barrier=barrier_row
        reasons_by_session[day]=dict(day_reasons)
        for f in fields:columns[f].append(values[f])
    provenance={'source_hashes':{k:v['sha256'] for k,v in plan['raw_sources'].items()},
                'supplement_hashes':{f:file_hash(supp/f) for f in ['namechange.csv','dividend.csv']},
                'plan_hash':digest(plan),'calendar_parse_audit':calendar_audit,
                'availability':'daily 18:00 conservative vendor proxy; no historical revision archive certification',
                'field_units':{'volume':'hands*100 -> shares','amount':'thousand CNY*1000 -> CNY'}}
    if issuer_halts:provenance['issuer_halt_evidence']=deepcopy(plan['issuer_halt_evidence'])
    if plan.get('halted_distribution_policy') is not None:
        provenance['halted_distribution_policy']=deepcopy(plan['halted_distribution_policy'])
    if catalog_manifest is not None:
        provenance['catalog_manifest_hash']=catalog_manifest['manifest_hash']
        provenance['catalog_file_sha256']=file_hash(root/'CATALOG.json')
        provenance['catalog_vintage']=catalog_manifest['vintage']
        provenance['catalog_metadata_hash']=metadata['metadata_hash']
        provenance['factor_crosscheck_sources']=crosscheck_sources
    if basic_features:
        provenance['field_units'].update(turnover_rate='vendor percentage * 0.01 -> ratio',
            total_shares='vendor ten-thousand shares * 10000 -> shares',
            float_shares='vendor ten-thousand shares * 10000 -> shares',
            free_shares='vendor ten-thousand shares * 10000 -> shares')
        provenance['daily_basic_availability']='conservative 18:00 vendor daily proxy; not proof of original release/revision clock'
    if source_audit is not None:provenance['catalog_source_verification']=source_audit
    if independent_sources:
        provenance['independent_event_sources']=independent_sources
        provenance['independent_event_scope']='cross-vendor public corroboration; not an independent investment assurance receipt'
    if remediation is not None:
        provenance['remediation']=remediation
        provenance['adjustment_overlay_sources']=overlay_sources
        provenance['factor_crosscheck_sources']=crosscheck_sources
        provenance['independent_event_sources']=independent_sources
    if plan.get('universe_restriction'):
        restriction=plan['universe_restriction']
        require(restriction.get('posthoc_exclusion') is True and restriction.get('diagnostic_only') is True,
                'unsupported restricted universe protocol')
        require(not ({r['ts_code'] for r in restriction['excluded']} & set(codes)), 'excluded security still in cohort')
        provenance['selection_audit']=deepcopy(restriction)
    snapshot={'schema_version':'fuxi.historical_snapshot.1','source_id':'tushare_local_history_v2',
              'exposure':'research_exposed','sessions':sessions,'evaluation_start':sessions[plan['window']['warmup_sessions']],
              'instruments':[jq_code(s) for s in codes],'universe':dict(zip(sessions,members)),
              'decision_times':{d:d+'T18:30:00+08:00' for d in sessions},'columns':columns,'barriers':barriers,
              'fields':{f:{'unit':units[f],
                           'frequency':'daily','adjustment':'raw'} for f in fields},
              'quality':{'pit_certified':False,'universe_certified':False,'account_events_certified':False,
                         'gaps':['historical_revision_archive_unverified','name_announcement_times_daily_proxy',
                                 'tiny_adjustment_drift_not_vendor_certified','noncash_actions_require_extra_evidence']},
              # The removal reasons are bound into the snapshot so certification is a
              # pure function of the frozen bytes; a side file could be edited later
              # and would leave no trace in the hashed artifact.
              'universe_audit':{'reasons_by_session':reasons_by_session,'cells':len(universe_rows),
                                'reason_vocabulary':sorted(state_counts)},
              'cohort_declaration':dict(plan['cohort'],rule_kind=plan['cohort'].get('rule_kind') or ('per_exchange_hash_cutoff'
                                        if plan['cohort'].get('hash_cutoffs') else 'unknown'),
                                        sampling_key_fields=['ts_code'],
                                        sampling_frame='full_lifecycle_table',
                                        frame_predicate=plan['cohort'].get('frame_predicate') or FRAME_PREDICATE,
                                        posthoc_exclusion=bool(plan.get('universe_restriction'))),
              'provenance':provenance}
    if remediation is not None:
        snapshot['quality']['gaps']=['historical_revision_archive_unverified','name_announcement_times_daily_proxy',
            'public_source_events_not_broker_settlement_receipts','investor_specific_tax_not_certified']
        if class_counts['factor_or_reference_conflict']:
            snapshot['quality']['gaps'].append('factor_reference_conflicts_unresolved')
        if class_counts['tiny_drift_consistent_reference_research_only']:
            snapshot['quality']['gaps'].append('tiny_adjustment_drift_declared_research_sensitivity')
    if plan.get('universe_restriction'):
        snapshot['quality']['gaps'].append('posthoc_universe_exclusion_selection_bias')
    execution=execution_profile();execution['sell_tax_schedule']=[['0001-01-01','0.001'],['2023-08-28','0.0005']]
    if source_bound:
        from .a_share import rule_for
        execution['instrument_rules']={code:rule_for(code) for code in codes}
        execution['order_type']='market'
        execution['scope']='all_a_share_quantity_contract_research_only'
        execution.update(odd_lot_exit='whole_residual',fractional_share_policy='block')
        if (root/'CATALOG.json').exists():
            execution['identity_aliases']=load_json(root/'CATALOG.json').get('identity_aliases',{})
    execution['dividend_tax_model']='flat_20_percent_declared_research_not_actual_tax_receipt'
    if remediation is not None:
        execution.update(odd_lot_exit='whole_residual',fractional_share_policy='block')
    market={'sessions':sessions,'bars':bars,'actions':[],'action_terms':terms,'execution':execution,
            'event_coverage_certified':False,'tradability_certified':False}
    if remediation is not None:market['provenance_hash']=digest(provenance)
    # All-census builds otherwise retain several million raw row dictionaries
    # while materializing labels and serializing the account hash. Their values
    # have already been copied into the frozen columns/bars and audit files.
    del maps, tables, frame, names, dividends
    if basic_features:
        del basic
    market['market_hash']=digest(market)
    snap=HistoricalSnapshot(snapshot);labels=materialize_labels(snap,market,plan['window']['horizon_sessions'])
    # Raw proxy labels additionally exclude untradeable entry opens and missing supporting closes.
    bar_map={(r['session'],r['instrument']):r for r in bars}
    for r in labels['rows']:
        entry=bar_map[(r['label_start'][:10],r['instrument'])]
        exit_bar=bar_map[(r['label_end'][:10],r['instrument'])]
        if entry['paused'] is not False or entry['open'] is None or exit_bar['mark_source']!='observed_close':
            r.update(value=None,status='data_insufficient')
    pd.DataFrame(universe_rows).to_csv(out/'universe_daily.csv',index=False,lineterminator='\n')
    pd.DataFrame(drift_audit).to_csv(out/'adjustment_drift.csv',index=False,lineterminator='\n')
    pd.DataFrame(halt_reference_audit).to_csv(out/'halt_reference_adjustments.csv',index=False,lineterminator='\n')
    if remediation is not None or crosscheck:
        from .campaign import write_new
        write_new(out/'factor_corroboration.json',corroboration)
    eligible_sets=[set(m) for m in members];coverage=[]
    for year in sorted({d[:4] for d in sessions}):
        indices=[i for i,d in enumerate(sessions) if d.startswith(year)]
        for col,code in enumerate(codes):
            eligible=[i for i in indices if jq_code(code) in eligible_sets[i]]
            for field in fields:
                coverage.append({'year':year,'instrument':jq_code(code),'field':field,'calendar_dates':len(indices),
                    'raw_value_dates':sum(columns[field][i][col] is not None for i in indices),
                    'eligible_dates':len(eligible),'valid_on_eligible_dates':sum(columns[field][i][col] is not None for i in eligible)})
    pd.DataFrame(coverage).to_csv(out/'selected_security_year_field.csv',index=False,lineterminator='\n')
    audit={'sessions':len(sessions),'selected_instruments':len(codes),'eligible_min':min(map(len,members)),
           'eligible_max':max(map(len,members)),'classification_counts':dict(class_counts),'universe_exclusions':dict(state_counts),
           'eligible_classification_counts':dict(eligible_classes),
           'price_layout_counts':{str(k):int(v) for k,v in layout_audit.field_contract_status.value_counts().items()},
           'name_publication_proxy_security_days':sum(r['name_publication_date_missing'] for r in universe_rows),
           'halt_reference_adjustments':dict(Counter(r['status'] for r in halt_reference_audit)),
           'issuer_explicit_halt_start_cells':sum(day==proof.get('interval_start',day) for (_,day),proof in issuer_halts.items()),
           'issuer_evidenced_halt_session_cells':len(issuer_halts),
           'issuer_halt_conflicting_start_cells':sum(r.get('start_date_conflict',False) for r in issuer_halts.values()),
           'action_terms':dict(Counter(t['status'] for t in terms)),'sample_scope':plan['cohort']['selection'],
           'gaps_by_kind':{'local_previously_unwired':['2021_2023_prices','suspend_d','delist_dates'],
                           'caliber_conflict':['tiny_factor_drift','raw_price_windows_crossing_actions'],
                           'newly_supplemented':['full_name_history','cash_payment_dates','share_distribution_terms'],
                           'still_unproven':['archive_PIT_revisions','noncash_action_settlements','investor_specific_dividend_tax']}}
    payload={'schema_version':'fuxi.history_input.1','snapshot':snapshot,'labels':labels,'market':market,
             'research_clock':sessions[-1]+'T20:00:00+08:00','audit':audit}
    if remediation is not None:
        audit['gaps_by_kind']['newly_supplemented']+=['coherent_adjustment_vintage','issuer_holder_rights','share_credit_dates']
        audit['gaps_by_kind']['still_unproven']=['archive_PIT_revisions','actual_broker_settlement_receipts','investor_specific_dividend_tax']
        audit['remaining_unsupported_action_terms']=sum(t['status'] not in SUPPORTED for t in terms)
        audit['factor_corroboration_counts']=dict(Counter((r.get('source','stk_factor')+('_accepted' if r['accepted'] else '_not_resolved')) for r in corroboration))
    elif crosscheck:
        audit['factor_corroboration_counts']=dict(Counter('stk_factor_accepted' if r['accepted'] else 'stk_factor_not_resolved' for r in corroboration))
    # Freeze only after confirming all raw bytes still match preregistration.
    for name,info in plan['raw_sources'].items():
        source=Path(info['path']) if catalog_manifest is not None else root/name
        if catalog_manifest is not None:
            from .market_catalog import TABLE_APIS
            require(info==catalog_manifest['sources'][TABLE_APIS[name]],'catalog source identity changed '+name)
        if catalog_manifest is None or plan.get('catalog_source_mode')!='verified_immutable_partitions':
            require(file_hash(source)==info['sha256'],'raw source changed '+name)
    if catalog_manifest is not None and plan.get('catalog_source_mode')=='verified_immutable_partitions':
        from .market_catalog import verify_frozen_sources
        source_audit=verify_frozen_sources(root,catalog_manifest)
        require(source_audit==snapshot['provenance']['catalog_source_verification'],'catalog origin changed during build')
        audit['catalog_source_verification']=source_audit
    if certification is not None:
        # ``None`` disables derivation; an empty dict means "derive with defaults".
        # Testing truthiness here would silently skip derivation for ``{}``, which is
        # exactly the kind of silent downgrade that keeps a flag asserted-False.
        from .certification import certify, apply_to_payload
        from .pit_universe import read_lifecycle
        lifecycle=read_lifecycle(root)
        declaration=dict(certification)
        declaration.pop('cohort',None)
        # The declared universe policy is what "no silent removal" is measured against,
        # so carry it into the certificate rather than letting a code default decide.
        declaration.setdefault('universe_policy',deepcopy(plan.get('universe') or {}))
        declaration['cohort']=dict(plan['cohort'],
            rule_kind=plan['cohort'].get('rule_kind') or ('per_exchange_hash_cutoff' if plan['cohort'].get('hash_cutoffs') else 'unknown'),
            sampling_key_fields=['ts_code'],sampling_frame='full_lifecycle_table',
            frame_predicate=plan['cohort'].get('frame_predicate') or FRAME_PREDICATE,
            posthoc_exclusion=bool(plan.get('universe_restriction')))
        if plan.get('universe_restriction'):
            declaration['cohort']={'posthoc_exclusion':True,
                'rule':plan['universe_restriction'].get('rule'),
                'sampling_key_fields':['ts_code'],
                'restriction_excluded':[r['ts_code'] for r in plan['universe_restriction']['excluded']],
                'parent_rule':{k:v for k,v in plan['cohort'].items()},
                'rule_kind':'per_exchange_hash_cutoff' if plan['cohort'].get('hash_cutoffs') else 'unknown',
                'salt':plan['cohort'].get('salt'),
                'per_exchange_initial':plan['cohort'].get('per_exchange_initial'),
                'hash_cutoffs':plan['cohort'].get('hash_cutoffs')}
        certificate=certify(payload,declaration=declaration,lifecycle=lifecycle)
        payload=apply_to_payload(payload,certificate)
        payload['audit']['data_certification']={
            'certification_hash':certificate['certification_hash'],
            'flags':{k:v['value'] for k,v in certificate['flags'].items()},
            'blocking_gaps':certificate['blocking_gaps'],
            'residual_gaps':certificate['residual_gaps'],
            'claim_scope':certificate['claim_scope']}
    return payload
