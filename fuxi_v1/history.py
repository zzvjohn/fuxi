"""Auditable historical universes, coverage and append-only vendor supplements.

This adapter never infers a delisting from absence in today's security master.
Sparse event tables are inventoried as events, not as dense missing price panels.
"""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from .contracts import digest, require
from .datasets import file_hash, read_calendar


def iso(value):
    text = str(value)
    return text[:4]+'-'+text[4:6]+'-'+text[6:8] if re.fullmatch(r'\d{8}', text) else None


def jq_code(value):
    return value.replace('.SZ', '.XSHE').replace('.SH', '.XSHG')


def mainboard(code):
    return bool(re.fullmatch(r'(?:00\d{4}\.SZ|60\d{4}\.SH)', code))


def select_cohort(master, start, end, per_exchange=48, salt='fuxi-history-20260930'):
    """Fixed hash sampling, includes deceased listings; new IPOs use frozen hash cutoffs."""
    rows = [r for r in master if mainboard(r['ts_code']) and iso(r['list_date'])
            and iso(r['list_date']) <= end and (not iso(r.get('delist_date')) or iso(r['delist_date']) > start)]
    require(len({r['ts_code'] for r in rows}) == len(rows), 'duplicate security master')
    score = lambda code: hashlib.sha256((salt+':'+code).encode()).hexdigest()
    picked, cuts = [], {}
    for exchange in ('SH', 'SZ'):
        initial = sorted([r for r in rows if r['ts_code'].endswith(exchange) and iso(r['list_date']) <= start],
                         key=lambda r: score(r['ts_code']))
        require(len(initial) >= per_exchange, 'insufficient mainboard master')
        cut = score(initial[per_exchange-1]['ts_code']); cuts[exchange] = cut
        picked.extend(r for r in rows if r['ts_code'].endswith(exchange) and score(r['ts_code']) <= cut)
    return {'members': sorted(picked, key=lambda r: r['ts_code']), 'hash_cutoffs': cuts,
            'selection': 'per-exchange initial hash sample; same cutoffs admit subsequent IPOs; retain delisted names',
            'salt': salt, 'per_exchange_initial': per_exchange}


def inventory(raw_root, output, start, end):
    """Stream all securities, retaining exact unique-date masks across CSV chunk boundaries."""
    import pandas as pd
    import numpy as np
    root, out = Path(raw_root), Path(output)
    out.mkdir(parents=True, exist_ok=True)
    require(not (out/'COVERAGE.json').exists(), 'inventory already published')
    sessions, _ = read_calendar(root/'trade_calendar.csv', start, end)
    calendar = {d.replace('-', ''): i for i, d in enumerate(sessions)}
    yearly = Counter(d[:4] for d in sessions)
    summary, details = {}, []
    definitions = {'daily_prices.csv':['open','high','low','close','pre_close','vol','amount'],
                   'adj_factor_daily.csv':['adj_factor'], 'stk_limit.csv':['up_limit','down_limit'],
                   'suspend_d.csv':['suspend_type','suspend_timing']}
    for name, fields in definitions.items():
        path = root/name; before = (path.stat().st_size, path.stat().st_mtime_ns)
        counts, masks, valid_masks, rows, invalid_dates = Counter(), {}, {}, 0, 0
        for frame in pd.read_csv(path, dtype={'ts_code':str,'trade_date':str}, chunksize=250000,
                                 usecols=['ts_code','trade_date']+fields):
            good = frame.trade_date.str.fullmatch(r'\d{8}').fillna(False)
            invalid_dates += int((~good).sum())
            frame = frame[good & frame.trade_date.between(start.replace('-',''),end.replace('-',''))].copy()
            frame['year'] = frame.trade_date.str[:4]
            rows += len(frame)
            for (code, year), group in frame.groupby(['ts_code','year'], sort=False):
                key = (code,year); counts[key] += len(group)
                indexes = np.array([calendar[d] for d in group.trade_date if d in calendar],dtype=int)
                mask = masks.setdefault(key,np.zeros(len(sessions),dtype=bool)); mask[indexes] = True
                for field in fields:
                    ok = group[field].notna() if name == 'suspend_d.csv' else np.isfinite(pd.to_numeric(group[field],errors='coerce'))
                    fi = np.array([calendar[d] for d in group.loc[ok,'trade_date'] if d in calendar],dtype=int)
                    fm = valid_masks.setdefault((code,year,field),np.zeros(len(sessions),dtype=bool)); fm[fi] = True
        require((path.stat().st_size,path.stat().st_mtime_ns)==before,'source changed during inventory')
        source_rows = []
        for (code,year), mask in sorted(masks.items()):
            idx = np.flatnonzero(mask); observed = len(idx)
            for field in fields:
                n = int(valid_masks[(code,year,field)].sum())
                source_rows.append({'source':name,'year':year,'ts_code':code,'field':field,'rows':counts[(code,year)],
                    'unique_dates':observed,'valid_dates':n,'calendar_dates':yearly[year],
                    'missing_on_observed_dates':observed-n,'calendar_gap_dates':yearly[year]-observed,
                    'first_date':sessions[int(idx[0])] if len(idx) else None,'last_date':sessions[int(idx[-1])] if len(idx) else None,
                    'calendar_gap_interpretation':'not_expected_daily_for_sparse_events' if name=='suspend_d.csv' else 'includes_prelisting_postdelisting_and_halts'})
        details.extend(source_rows)
        summary[name]={'rows':rows,'invalid_date_rows':invalid_dates,'sha256':file_hash(path),
            'security_years':len(masks),'row_minus_unique_code_date':sum(counts[k]-int(v.sum()) for k,v in masks.items()),
            'yearly':{year:{'calendar_dates':n,'securities':sum(k[1]==year for k in masks),
                       'observed_dates':int(np.logical_or.reduce([v for k,v in masks.items() if k[1]==year]).sum())}
                       for year,n in sorted(yearly.items())}}
        print(json.dumps({'inventory':name,'rows':rows,'security_years':len(masks)}),flush=True)
    pd.DataFrame(details).to_csv(out/'coverage_security_year_field.csv',index=False,lineterminator='\n')
    result={'schema_version':'fuxi.coverage.1','window':[start,end],'sessions':len(sessions),'sources':summary,
            'detail_sha256':file_hash(out/'coverage_security_year_field.csv')}
    (out/'COVERAGE.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return result


def name_state(records, day):
    """Effective name plus publication date. No backward fill from a future ST removal."""
    active = [r for r in records if iso(r.get('start_date')) and iso(r['start_date']) <= day
              and (not iso(r.get('end_date')) or day <= iso(r['end_date']))]
    if not active:
        return 'unknown', None
    latest = max(iso(r['start_date']) for r in active)
    active = [r for r in active if iso(r['start_date']) == latest]
    if len({r['name'] for r in active}) != 1:
        return 'conflicting', None
    row = active[0]; ann = iso(row.get('ann_date'))
    if ann and ann > day:
        return 'not_yet_published', row
    # Missing publication time is a declared daily proxy, never archive PIT certification.
    return ('st' if re.search(r'ST|PT|退',row['name'].upper()) else 'normal'), row


def adjustment_class(previous, current, prior_close, pre_close, known_event=False):
    from .contracts import finite
    if known_event:
        return 'published_corporate_action'
    if not all(finite(x) for x in (previous,current,prior_close,pre_close)) or min(previous,current,prior_close,pre_close)<=0:
        return 'missing_evidence'
    reference_equal = abs(prior_close-pre_close) < .000001
    if previous == current and reference_equal:
        return 'unchanged_reference_and_factor'
    # Explicit research sensitivity model, not a vendor precision guarantee.
    # P-20261015-011: vendor adj revisions drift up to ~0.76% relative on
    # reference-continuous days; 1% relative bound (chain continuity required).
    if reference_equal and abs(current/previous-1)<=.01:
        return 'tiny_drift_consistent_reference_research_only'
    return 'factor_or_reference_conflict'


def normalize_price_frame(frame):
    """Recover a uniquely proven header/vendor permutation without editing raw bytes.

    Original header ends pct_chg,vol,amount,change; some appended rows instead
    contain change,pct_chg,vol,amount. Both return identities and VWAP units must
    corroborate a repair. Unresolved rows become unavailable, never guesses.
    """
    import pandas as pd
    import numpy as np
    f=frame.copy()
    fields=['open','high','low','close','pre_close','change','pct_chg','vol','amount']
    for c in fields:f[c]=pd.to_numeric(f[c],errors='coerce') if c in f else np.nan
    delta=f.close-f.pre_close;pct=delta/f.pre_close*100
    price_ok=(f.pre_close>0)&(f.low>0)&(f.high>=f.low)&(f.close>=f.low-.000001)&(f.close<=f.high+.000001)&(f.open>=f.low-.000001)&(f.open<=f.high+.000001)
    def checks(change,percent,volume,amount,allow_missing_change=False):
        vwap=amount*10/volume
        identities=(abs(change-delta)<=.00015) | (change.isna() if allow_missing_change else False)
        return price_ok & identities & (abs(percent-pct)<=.00015) & (volume>=0) & (amount>=0) & (
            ((volume>0)&(vwap>=f.low-.05)&(vwap<=f.high+.05)) | ((volume==0)&(amount==0)))
    native=checks(f.change,f.pct_chg,f.vol,f.amount,True)
    rotated=checks(f.pct_chg,f.vol,f.amount,f.change)
    equivalent=(f.change==f.pct_chg)&(f.pct_chg==f.vol)&(f.vol==f.amount)
    repair=(~native)&rotated
    status=pd.Series('unresolved_field_contract',index=f.index)
    status.loc[native&(~rotated|equivalent)]='header_layout_verified'
    status.loc[repair]='vendor_order_rotation_verified'
    f.loc[repair,['change','pct_chg','vol','amount']]=f.loc[repair,['pct_chg','vol','amount','change']].to_numpy()
    invalid=status=='unresolved_field_contract'
    f.loc[invalid,fields]=np.nan
    f['field_contract_status']=status
    return f


def supplement(plan, destination):
    """Only missing full-history name intervals and missing dividend columns, public vendor API."""
    import time
    import urllib.request
    import pandas as pd
    from credentials import get_tushare_token
    out=Path(destination);out.mkdir(parents=True,exist_ok=True)
    token=get_tushare_token()
    fields={'namechange':'ts_code,name,start_date,end_date,ann_date,change_reason',
            'dividend':'ts_code,end_date,ann_date,div_proc,stk_div,stk_bo_rate,stk_co_rate,cash_div,cash_div_tax,record_date,ex_date,pay_date,div_listdate,imp_ann_date'}
    calls=0
    for row in plan['cohort']['members']:
        code=row['ts_code']
        for api in fields:
            path=out/(api+'_'+code+'.json')
            if path.exists():
                saved=json.loads(path.read_text(encoding='utf-8'))
                require(saved['api']==api and saved['ts_code']==code,'supplement cache mismatch')
                continue
            require(calls < plan['collection']['max_requests'],'supplement request budget exhausted')
            request=urllib.request.Request('https://api.tushare.pro',data=json.dumps({'api_name':api,'token':token,
                    'params':{'ts_code':code},'fields':fields[api]}).encode(),headers={'Content-Type':'application/json'})
            calls+=1
            try:
                with urllib.request.urlopen(request,timeout=25) as response:
                    raw=response.read(2000001)
                require(len(raw)<=2000000,'vendor response too large')
                data=json.loads(raw)
                require(data.get('code')==0,'vendor rejected request, code='+str(data.get('code')))
                result=data['data'];require(len(result['items'])<2000,'potential vendor truncation')
                require(set(fields[api].split(','))<=set(result['fields']),'vendor fields incomplete')
                record={'api':api,'ts_code':code,'retrieved_at':datetime.now(timezone.utc).isoformat(),
                        'request_fields':fields[api],'rows':[dict(zip(result['fields'],x)) for x in result['items']],
                        'response_sha256':hashlib.sha256(raw).hexdigest(),'status':'complete_vendor_response'}
            except Exception as exc:
                # Credentials/response bodies never enter logs, even on transport failure.
                record={'api':api,'ts_code':code,'status':'failed','exception_type':type(exc).__name__}
                with (out/('failure_'+str(time.time_ns())+'.json')).open('x',encoding='utf-8') as f:json.dump(record,f)
                raise RuntimeError('targeted supplement failed: '+api+' '+type(exc).__name__) from None
            with path.open('x',encoding='utf-8') as f:json.dump(record,f,ensure_ascii=False,indent=2)
            time.sleep(.3)
        if calls%16==0:print(json.dumps({'supplement_requests_this_invocation':calls,'last_security':code}),flush=True)
    for api in fields:
        rows=[]
        for member in plan['cohort']['members']:
            record=json.loads((out/(api+'_'+member['ts_code']+'.json')).read_text(encoding='utf-8'))
            rows.extend(record['rows'])
        target=out/(api+'.csv')
        if not target.exists():pd.DataFrame(rows,columns=fields[api].split(',')).to_csv(target,index=False,lineterminator='\n')
    return {'new_requests':calls,'total_cached_requests':len(list(out.glob('namechange_*.json')))+len(list(out.glob('dividend_*.json')))}
