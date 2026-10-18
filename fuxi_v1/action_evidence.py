"""Versioned research action adapter. Corporate capital changes are not holder rights."""
from copy import deepcopy
from collections import defaultdict
from .contracts import digest,require,finite
from .history import iso,jq_code


def canonical_dividend_row(row):
    """Canonical dividend rights digest inputs shared by the vendor adapter and resolver.

    A share distribution with no cash row is economically a zero-cash row; vendors
    emit both spellings for the same event, so pure share rows normalise cash to 0.0
    before digesting. This is the single source of truth for the Rule B spelling
    (fuxi_v1/historical_adapter.py uses the same function).
    """
    def number(key):
        try:
            value=float(row.get(key))
            return value if finite(value) else None
        except (ValueError,TypeError):
            return None
    shares=number('stk_div');cash=number('cash_div_tax')
    if shares is not None and shares>0 and cash is None:
        cash=0.0
    return {'record_date':iso(row.get('record_date')),'pay_date':iso(row.get('pay_date')),
            'stk_div':shares,'cash_div_tax':cash,'div_listdate':iso(row.get('div_listdate')),
            'imp_ann_date':iso(row.get('imp_ann_date'))}


def resolve_action_terms(base,records,adjudications):
    groups=defaultdict(list)
    for r in records:
        if r.get('div_proc')=='实施':groups[(jq_code(r['ts_code']),iso(r.get('ex_date')))].append(r)
    judgments={(r['instrument'],r['ex_date']):r for r in adjudications}
    require(len(judgments)==len(adjudications),'duplicate action adjudication')
    result=[]
    for original in base:
        term=deepcopy(original);key=(term['instrument'],term['ex_date']);rows=groups[key]
        judgment=judgments.get(key)
        term['original_status']=term['status']
        if judgment:
            require(judgment.get('type')!='public_missing_vendor_event_v1',
                    'missing-vendor adjudication must not bind an existing vendor term')
            if judgment.get('type')=='public_restructuring_v1':
                from .public_notices import extract_restructuring,verified_restructuring_source
                require(judgment['vendor_evidence_id']==original['evidence_id'], 'restructuring bound to another vendor event')
                require(judgment.get('independent_assurance') is False and judgment['sources'],
                        'public restructuring parser is not independent assurance')
                fields=('instrument','ex_date','record_date','ann_date','pay_date','company_share_distribution',
                    'holder_cash','holder_shares','share_credit_date','company_new_shares','investor_new_shares',
                    'creditor_new_shares','vendor_evidence_id','available_at_assumption','publication_hour_verified',
                    'availability_scope','holder_scope','method')
                for source in judgment['sources']:
                    checked=extract_restructuring(verified_restructuring_source(source),source['publication_index'],
                        original,rows,source)
                    require(checked['status']=='restructuring_terms_proposed', 'restructuring primary terms no longer reproduce')
                    require(all(checked['adjudication'][f]==judgment[f] for f in fields),
                            'restructuring adjudication differs from archived primary terms')
                for source in judgment.get('corroborating_sources',[]):
                    checked=extract_restructuring(verified_restructuring_source(source),source['publication_index'],
                        original,rows,source,allow_late=True)
                    require(checked['status']=='restructuring_terms_proposed'
                        and checked['adjudication']['ann_date']>judgment['record_date'],
                        'later corroboration publication order invalid')
                    common=[f for f in fields if f not in ('ann_date','available_at_assumption')]
                    require(all(checked['adjudication'][f]==judgment[f] for f in common),
                            'later restructuring corroboration conflicts with primary holder terms')
                require(judgment['holder_cash']==judgment['holder_shares']==0,
                        'restructuring cannot grant unproven holder rights')
                term.update(record_date=judgment['record_date'],ann_date=judgment['ann_date'],pay_date=None,
                    gross_cash_per_share=0.,share_distribution=0.,share_credit_date=None,issuer_evidence=judgment,
                    status='restructuring_no_holder_distribution',
                    settlement_scope='public issuer restructuring terms for ordinary nominal stockholder; no broker or registration receipt')
                term['evidence_id']=digest([original['evidence_id'],term])
                result.append(term)
                continue
            if judgment.get('type')=='public_implementation_v1':
                from pathlib import Path
                from .datasets import file_hash
                require(judgment['vendor_evidence_id']==original['evidence_id'], 'issuer evidence bound to another vendor event')
                require(judgment['instrument']==term['instrument'] and judgment['ex_date']==term['ex_date'], 'issuer event identity mismatch')
                require(judgment['ann_date']<=judgment['record_date']<judgment['ex_date'], 'issuer event clock conflict')
                cash,shares=judgment['holder_cash'],judgment['holder_shares']
                require(finite(cash) and finite(shares) and cash>=0 and shares>=0, 'invalid public distribution amounts')
                require(not cash or judgment['pay_date'] and judgment['pay_date']>=term['ex_date'], 'cash settlement date missing')
                require(not shares or judgment['share_credit_date'] and judgment['share_credit_date']>=term['ex_date'], 'share settlement date missing')
                require(judgment['sources'] and judgment.get('independent_assurance') is False, 'public parser is not independent assurance')
                for source in judgment['sources']:
                    require(source['url'].startswith('https://') and source['excerpt'] and
                            Path(source['path']).is_file() and file_hash(source['path'])==source['sha256'] and
                            file_hash(source['text_path'])==source['text_sha256'], 'archived issuer source changed')
                    from .public_notices import extract_ordinary
                    checked=extract_ordinary(Path(source['text_path']).read_text(encoding='utf-8'),
                        source['publication_index'],original,rows,source)
                    require(checked['status']=='ordinary_terms_proposed', 'issuer terms no longer reproducible')
                    fields=('instrument','ex_date','record_date','ann_date','pay_date','holder_cash','holder_shares','share_credit_date','vendor_evidence_id')
                    require(all(checked['adjudication'][f]==judgment[f] for f in fields), 'issuer adjudication differs from archived terms')
                term.update(record_date=judgment['record_date'],ann_date=judgment['ann_date'],pay_date=judgment['pay_date'],
                    gross_cash_per_share=cash,share_distribution=shares,share_credit_date=judgment['share_credit_date'],
                    issuer_evidence=judgment,status='ordinary_distribution_supported' if shares else 'cash_terms_supported')
                term['settlement_scope']='public issuer implementation terms and declared hypothetical record-close ownership; fractional allocations remain unknown'
                term['evidence_id']=digest([original['evidence_id'],term])
                result.append(term)
                continue
            if judgment.get('type')=='public_adjusted_final_terms_v1':
                from pathlib import Path
                from .datasets import file_hash
                require(judgment['vendor_evidence_id']==original['evidence_id'], 'adjusted-terms evidence bound to another vendor event')
                require(judgment['instrument']==term['instrument'] and judgment['ex_date']==term['ex_date'], 'adjusted-terms event identity mismatch')
                require(judgment['ann_date']<=judgment['record_date']<judgment['ex_date'], 'adjusted-terms implementation clock conflict')
                require(judgment['ann_date']<=judgment['adjustment_ann_date']<=judgment['ex_date'], 'adjusted-terms adjustment clock conflict')
                cash,shares=judgment['holder_cash'],judgment['holder_shares']
                require(finite(cash) and finite(shares) and cash>=0 and shares>=0, 'invalid adjusted-final distribution amounts')
                require(not cash or judgment['pay_date'] and judgment['pay_date']>=term['ex_date'], 'adjusted-final cash settlement date missing')
                require(not shares or judgment['share_credit_date'] and judgment['share_credit_date']>=term['ex_date'], 'adjusted-final share settlement date missing')
                require(all(float(r['cash_div_tax'])==cash and float(r['stk_div'])==shares for r in rows), 'vendor final amounts differ from adjusted-final adjudication')
                require(abs(cash*10-judgment['final_cash_per_10'])<1e-9 and abs(shares*10-judgment['final_shares_per_10'])<1e-9,
                        'per-10 final amounts do not reproduce')
                require(len(judgment['sources'])==2 and judgment.get('independent_assurance') is False,
                        'adjusted-final terms require exactly the two archived public announcements')
                roles={}
                for source in judgment['sources']:
                    require(source['url'].startswith('https://') and source['excerpt'] and
                            Path(source['path']).is_file() and file_hash(source['path'])==source['sha256'] and
                            file_hash(source['text_path'])==source['text_sha256'], 'archived adjusted-terms source changed')
                    roles[source['role']]=source
                require(set(roles)=={'implementation','adjusted_final_terms'}, 'adjusted-final terms need implementation and adjustment archives')
                impl_text=Path(roles['implementation']['text_path']).read_text(encoding='utf-8')
                adj_text=Path(roles['adjusted_final_terms']['text_path']).read_text(encoding='utf-8')
                require(repr(judgment['final_cash_per_10']) in adj_text and repr(judgment['final_shares_per_10']) in adj_text,
                        'adjustment archive lacks final amounts')
                require(repr(judgment['final_cash_per_10']) not in impl_text and repr(judgment['final_shares_per_10']) not in impl_text,
                        'implementation archive must predate final amounts')
                term.update(record_date=judgment['record_date'],ann_date=judgment['ann_date'],pay_date=judgment['pay_date'],
                    gross_cash_per_share=cash,share_distribution=shares,share_credit_date=judgment['share_credit_date'],
                    issuer_evidence=judgment,status='ordinary_distribution_supported' if shares else 'cash_terms_supported')
                term['settlement_scope']='public issuer implementation and adjusted-final-terms announcements; vendor final amounts match the adjustment archive'
                term['evidence_id']=digest([original['evidence_id'],term])
                result.append(term)
                continue
            require(judgment['sources'] and all(s['sha256'] and s['url'].startswith('https://') for s in judgment['sources']),
                    'action adjudication requires archived public source')
            require(judgment['ann_date']<=judgment['record_date']<judgment['ex_date'],'late or conflicting issuer evidence')
            require(all(iso(r['record_date'])==judgment['record_date'] and
                        float(r['stk_div'])==judgment['company_share_distribution'] for r in rows),'issuer/vendor identity mismatch')
            require(judgment['holder_cash']==0 and judgment['holder_shares']>=0,'unsupported adjudication type')
            term.update(record_date=judgment['record_date'],ann_date=judgment['ann_date'],pay_date=None,
                        gross_cash_per_share=0.,share_distribution=judgment['holder_shares'],
                        share_credit_date=judgment['share_credit_date'],issuer_evidence=judgment,
                        status='ordinary_distribution_supported' if judgment['holder_shares'] else 'restructuring_no_holder_distribution')
        elif term['status']=='share_distribution_requires_settlement_adapter':
            canon=[(canonical_dividend_row(r),r) for r in rows]
            # P-20261015-011 (Rule A in the resolver): rows whose holder-rights
            # fields are identical and differ only in the implementation-publication
            # date are one event published twice (same digest rule as
            # historical_adapter.dividend_terms). BJSE vendors publish the same
            # event under two ann dates, which previously kept len(unique)>1 here
            # and blocked the settlement upgrade even though the vendor credit
            # (listing) date is present in the raw row.
            rights={digest({k:v for k,v in c.items() if k!='imp_ann_date'}):(c,r) for c,r in canon}
            if len(rights)==1:
                _,r=max(rights.values(),key=lambda cr:(iso(cr[1].get('imp_ann_date')) or ''))
                credit=iso(r.get('div_listdate'))
                record,ex,pay,ann=term['record_date'],term['ex_date'],term['pay_date'],term['ann_date']
                gross=term['gross_cash_per_share'];shares=term['share_distribution']
                if all((record,ann,credit,pay)) and finite(gross) and gross>0 and finite(shares) and shares>0 and ann<=record<ex<=credit and ex<=pay:
                    # Mixed share+cash event: both legs settle on the vendor dates.
                    term.update(status='ordinary_distribution_supported',share_credit_date=credit)
                elif finite(shares) and shares>0 and (gross is None or gross==0) and all((record,ann,credit)) and ann<=record<ex<=credit:
                    # Pure share distribution: no cash leg exists, so zero cash is the
                    # stated amount (not an invention); the vendor credit date settles
                    # the share leg. P-20261015-009 (a1): settlement adapter wiring.
                    term.update(status='ordinary_distribution_supported',share_credit_date=credit,gross_cash_per_share=0.0)
        if term['status'] in {'ordinary_distribution_supported','restructuring_no_holder_distribution'}:
            if term['share_distribution']:
                require(term['share_credit_date']>=term['ex_date'],'shares credited before entitlement')
            term['settlement_scope']='declared_research_from_vendor_dates_and_issuer_exceptions; no_broker_receipt'
            term['evidence_id']=digest([original['evidence_id'],term])
        result.append(term)
    from .public_notices import extract_ordinary
    from pathlib import Path
    from .datasets import file_hash
    base_keys={(t['instrument'],t['ex_date']) for t in base}
    missing=[j for j in adjudications if j.get('type')=='public_missing_vendor_event_v1']
    require(len(missing)==len({(j['instrument'],j['ex_date']) for j in missing}),
            'duplicate missing-vendor-event adjudication')
    for judgment in missing:
        key=(judgment['instrument'],judgment['ex_date'])
        require(key not in groups and key not in base_keys,
                'vendor rows or terms already cover the missing-vendor ex-date')
        require(judgment.get('vendor_evidence_id') is None,
                'missing-vendor event must not bind a vendor event')
        require(judgment.get('independent_assurance') is False and len(judgment['sources'])==1,
                'missing-vendor event requires exactly one archived public source')
        require(judgment['ann_date']<=judgment['record_date']<judgment['ex_date']<=judgment['pay_date'],
                'missing-vendor event clock conflict')
        cash,shares=judgment['holder_cash'],judgment['holder_shares']
        require(finite(cash) and cash>0 and finite(shares) and shares>=0,'invalid missing-vendor distribution amounts')
        require(not shares or judgment['share_credit_date'] and judgment['share_credit_date']>=judgment['ex_date'],
                'missing-vendor share settlement date missing')
        source=judgment['sources'][0]
        require(source['url'].startswith('https://') and source['excerpt'] and
                Path(source['path']).is_file() and file_hash(source['path'])==source['sha256'] and
                file_hash(source['text_path'])==source['text_sha256'],'archived missing-vendor source changed')
        original={'instrument':judgment['instrument'],'ex_date':judgment['ex_date'],'evidence_id':None}
        checked=extract_ordinary(Path(source['text_path']).read_text(encoding='utf-8'),
            source['publication_index'],original,[],source,allow_missing_vendor=True)
        require(checked['status']=='ordinary_terms_proposed','missing-vendor issuer terms no longer reproduce')
        fields=('instrument','ex_date','record_date','ann_date','pay_date','holder_cash','holder_shares','reference_cash_deduction')
        require(all(checked['adjudication'].get(f)==judgment.get(f) for f in fields),
                'missing-vendor adjudication differs from archived terms')
        term={'instrument':judgment['instrument'],'ex_date':judgment['ex_date'],
            'record_date':judgment['record_date'],'ann_date':judgment['ann_date'],'pay_date':judgment['pay_date'],
            'gross_cash_per_share':cash,'share_distribution':shares,'share_credit_date':judgment['share_credit_date'],
            'vendor_event_row_missing':True,'issuer_evidence':judgment,
            'status':'ordinary_distribution_supported' if shares else 'cash_terms_supported',
            'settlement_scope':'public issuer implementation terms; vendor dividend table has no row for the ex-date; declared event corroborated by independent event calendar and ex-date reference drop'}
        if judgment.get('reference_cash_deduction') is not None:
            term['reference_cash_deduction']=judgment['reference_cash_deduction']
        term['evidence_id']=digest(['missing_vendor_event',term])
        result.append(term)
    missing_keys={(j['instrument'],j['ex_date']) for j in missing}
    require(set(judgments)<=set(groups)|missing_keys,'issuer evidence outside dividend family')
    return result


def verified_adjustment_overlay(paths,codes,dates):
    """Replace a whole cohort vintage; never splice factor levels from different bases."""
    from pathlib import Path
    from .contracts import load_json
    from .datasets import file_hash
    mapping={};sources={}
    for code in codes:
        path=Path(paths)/('adj_factor_'+code+'.json');p=load_json(path)
        require(p['api']=='adj_factor' and p['params']['ts_code']==code,'factor source mismatch')
        require(file_hash(Path(paths)/('response_'+code+'.json'))==p['response_sha256'],'factor response changed')
        original=load_json(Path(paths)/('response_'+code+'.json'))['data']
        require(p['rows']==[dict(zip(original['fields'],r)) for r in original['items']],'factor parsed rows differ from wire evidence')
        sources[code]={'sha256':file_hash(path),'response_sha256':p['response_sha256'],'retrieved_at':p['retrieved_at']}
        for row in p['rows']:
            require(row['ts_code']==code and row['trade_date'] in dates,'factor source outside frozen scope')
            key=(code,row['trade_date']);require(key not in mapping,'duplicate vendor factor')
            require(finite(row['adj_factor']) and row['adj_factor']>0,'invalid vendor factor');mapping[key]=row
    return mapping,sources


def load_factor_crosscheck(directory,codes):
    from pathlib import Path
    from .contracts import load_json
    from .datasets import file_hash
    path=Path(directory);spec=load_json(path/'AMENDMENT.json');mapping={};sources={}
    require(spec['api']=='stk_factor' and spec['drift_tolerance_change'] is False,'unexpected crosscheck protocol')
    for day in spec['query_dates']:
        p=load_json(path/(day+'.json'))
        require(p['api']=='stk_factor' and p['params']=={'trade_date':day},'crosscheck identity changed')
        require(file_hash(path/(day+'.response'))==p['response_sha256'],'crosscheck response changed')
        original=load_json(path/(day+'.response'))['data']
        require(p['rows']==[dict(zip(original['fields'],r)) for r in original['items']],'crosscheck parsed rows differ from wire evidence')
        sources[day]={'sha256':file_hash(path/(day+'.json')),'response_sha256':p['response_sha256']}
        for row in p['rows']:
            if row['ts_code'] not in codes:continue
            key=(row['ts_code'],row['trade_date']);require(key not in mapping,'duplicate crosscheck key');mapping[key]=row
    return mapping,sources


def constant_factor_corroborated(before,after,prior_close,pre_close,current_close):
    """Exact same-table factors plus agreeing raw prices, no tolerance expansion."""
    if not before or not after:return False
    values=[before.get('adj_factor'),after.get('adj_factor'),before.get('close'),after.get('pre_close'),after.get('close'),
            prior_close,pre_close,current_close]
    if not all(finite(v) and v>0 for v in values):return False
    return (before['adj_factor']==after['adj_factor'] and abs(prior_close-pre_close)<.000001 and
            abs(before['close']-prior_close)<.000001 and abs(after['pre_close']-pre_close)<.000001 and
            abs(after['close']-current_close)<.000001)


def load_independent_events(directory):
    from pathlib import Path
    from .contracts import load_json
    from .datasets import file_hash
    root=Path(directory);spec=load_json(root/'AMENDMENT.json');complete=load_json(root/'COMPLETE.json')
    require(complete['codes']==len(spec['codes']),'independent data collection incomplete')
    hashes={file_hash(p):p for p in root.glob('wire_*.response')};quotes={};events={};sources={}
    for code in spec['codes']:
        action=load_json(root/(code+'.adjustment.json'));raw=load_json(root/(code+'.raw.json'))
        require(action['api']=='query_adjust_factor' and raw['api']=='query_history_k_data_plus','independent API identity')
        require(action['response_sha256'] in hashes and raw['response_sha256'] in hashes,'independent wire source changed')
        require(action['code']==raw['code']==code,'independent security identity')
        raw_window=spec.get('raw_windows',{}).get(code,['2021-01-04','2026-09-24'])
        for record in (action,raw):
            window=raw_window if record is raw else ['2021-01-04','2026-09-24']
            validate_independent_wire(record,hashes[record['response_sha256']].read_bytes(),expected_window=window)
        dates=[r['dividOperateDate'] for r in action['rows']]
        require(dates==sorted(set(dates)),'duplicate/unordered independent action dates')
        events[code]=set(dates)
        for row in raw['rows']:
            key=(code,row['date']);require(key not in quotes,'duplicate independent price');quotes[key]=row
        sources[code]={'events_sha256':file_hash(root/(code+'.adjustment.json')),'raw_sha256':file_hash(root/(code+'.raw.json')),
                       'response_sha256':[action['response_sha256'],raw['response_sha256']]}
    return quotes,events,sources


def validate_independent_wire(record,wire,*,expected_window=('2021-01-04','2026-09-24')):
    """Reparse archived public protocol responses without importing the network SDK."""
    import json,zlib
    header=wire[:21].decode().split('\x01');size=int(header[2]);body=wire[21:21+size]
    require(header[1] in {'16','96'},'unexpected independent response type')
    if header[1]=='96':body=zlib.decompress(body)
    fields=body.decode().split('\x01')
    require(fields[0]=='0' and fields[2]==record['api'] and fields[4]=='1','independent response method/status/page mismatch')
    if header[1]=='16':columns=fields[10].split(',');start,end=fields[8:10]
    else:
        columns=fields[8].split(',');start,end=fields[9:11]
        require(fields[11:13]==['d','3'],'independent prices are not daily unadjusted')
    from datetime import date
    require(len(expected_window)==2,'independent source window shape')
    left,right=expected_window
    require(date.fromisoformat(left)<=date.fromisoformat(right),'independent source window ordering')
    require('2021-01-04'<=left<=right<='2026-09-24','independent source outside frozen history')
    require(start==left and end==right,'independent source window mismatch')
    market={'XSHE':'sz','XSHG':'sh','SZ':'sz','SH':'sh'}.get(record['code'].split('.')[-1])
    require(market,'independent security suffix unknown')
    code=market+'.'+record['code'][:6]
    require(fields[7]==code,'independent wire security mismatch')
    # A successful empty result has an empty data field in the public SDK
    # protocol. It is distinct from a missing/failed response or absent source.
    data=json.loads(fields[6])['record'] if fields[6].strip() else []
    require(isinstance(data,list) and all(len(r)==len(columns) for r in data),'independent wire row shape')
    rows=[dict(zip(columns,r)) for r in data]
    require(rows==record['rows'] and len(rows)<2000,'independent parsed rows differ from source or truncated')


def independent_no_event(before,after,event_dates,day,prior_close,pre_close,current_close):
    """Cross-vendor event calendar and raw quotes, not cross-vendor factor magnitudes."""
    if not before or not after or event_dates is None or day in event_dates:return False
    if before.get('tradestatus')!='1' or after.get('tradestatus')!='1':return False
    try:
        prices=[float(before['close']),float(after['preclose']),float(after['close'])]
        volume=[float(before['volume']),float(after['volume'])]
    except (TypeError,ValueError,KeyError):return False
    if not all(finite(v) and v>0 for v in prices+volume+[prior_close,pre_close,current_close]):return False
    return (abs(prior_close-pre_close)<.000001 and all(abs(a-b)<.000001 for a,b in
            zip(prices,[prior_close,pre_close,current_close])))


def independent_halt_no_event(before,after,event_dates,day,mark):
    """Independent halt-day corroboration: no event calendar entry, the
    independent source records a full halt, and its frozen reference price
    matches the vendor-declared halt mark."""
    if after is None or event_dates is None or day in event_dates:return False
    if after.get('tradestatus')!='0':return False
    if after.get('close')!=after.get('preclose'):return False
    if mark is None:return False
    try:
        frozen=float(after['close'])
    except (TypeError,ValueError):return False
    return finite(frozen) and frozen>0 and abs(frozen-mark)<.000001


def independent_missed_event(before,after,event_dates,day,term):
    """Independent corroboration of a vendor-missed cash event: the independent
    event calendar records the ex-date, both surrounding sessions are trading
    days, and the observed ex-date price drop equals the issuer-declared
    exchange reference deduction (falling back to the per-share cash)."""
    if before is None or after is None or event_dates is None or day not in event_dates:return False
    if before.get('tradestatus')!='1' or after.get('tradestatus')!='1':return False
    try:
        shares=float(term.get('share_distribution') or 0)
        deduction=float(term.get('reference_cash_deduction')
                        if term.get('reference_cash_deduction') is not None
                        else term['gross_cash_per_share'])
        prior=float(before['close']);pre=float(after['preclose'])
    except (TypeError,ValueError,KeyError):return False
    if shares!=0:return False
    if not all(finite(v) and v>0 for v in (prior,pre,deduction)):return False
    return abs((prior-pre)-deduction)<=.010001
