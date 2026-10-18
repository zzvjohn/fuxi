"""Full-census channel audit with bounded memory and explicit current-vintage gaps."""
import json
from pathlib import Path
from collections import Counter
from .contracts import digest, require
from .datasets import file_hash
from .history import normalize_price_frame
from .market_catalog import TABLE_APIS


def numeric_contract(api, frame):
    import numpy as np
    import pandas as pd
    if api=='daily':
        frame=normalize_price_frame(frame)
        fields=['open','high','low','close','pre_close','change','pct_chg','vol','amount']
        good=frame.field_contract_status!='unresolved_field_contract'
    else:
        fields={'adj_factor':['adj_factor'],'stk_limit':['pre_close','up_limit','down_limit'],
                'daily_basic':['turnover_rate','total_share','float_share','free_share']}[api]
        for field in fields:frame[field]=pd.to_numeric(frame.get(field),errors='coerce')
        if api=='adj_factor':good=frame.adj_factor>0
        elif api=='stk_limit':good=(frame.down_limit>0)&(frame.up_limit>=frame.down_limit)&(frame.pre_close>0)
        else:
            good=(frame.turnover_rate>=0)&(frame.total_share>0)&(frame.float_share>0)&(frame.float_share<=frame.total_share+.0001)
            # Free float can be unavailable; it cannot exceed float shares when observed.
            good &= frame.free_share.isna() | ((frame.free_share>=0)&(frame.free_share<=frame.float_share+.0001))
    if api=='stk_limit':
        # The local historical cache contains actual bounds without a pre_close column.
        good=(frame.down_limit>0)&(frame.up_limit>=frame.down_limit)&(frame.pre_close.isna()|(frame.pre_close>0))
    optional=['free_share'] if api=='daily_basic' else ['change'] if api=='daily' else ['pre_close'] if api=='stk_limit' else []
    good &= np.isfinite(frame[fields].drop(columns=optional,errors='ignore')).all(axis=1)
    return frame,fields,good.to_numpy(dtype=bool)


def accumulate_panel(api, frames, sessions, codes, *, keep=()):
    """Conflicting duplicate keys stay unavailable, including conflicts across chunks."""
    import numpy as np
    import pandas as pd
    shape=(len(sessions),len(codes));ci={c:i for i,c in enumerate(codes)};di={d:i for i,d in enumerate(sessions)}
    seen=np.zeros(shape,bool);valid=seen.copy();conflict=seen.copy();signature=np.zeros(shape,np.uint64)
    values={field:np.full(shape,np.nan) for field in keep};stats=Counter()
    basic_fields=['turnover_rate','total_share','float_share','free_share']
    observed_values={field:np.full(shape,np.nan) for field in basic_fields} if api=='daily_basic' else {}
    for frame,replace_dates in frames:
        if isinstance(replace_dates,dict):
            selected=[ci[c] for c in replace_dates['codes'] if c in ci]
            days=[d for d in sessions if replace_dates['start_date']<=d<=replace_dates['end_date']]
        else:selected=list(range(len(codes)));days=replace_dates
        for day in days:
            if day in di:
                i=di[day];seen[i,selected]=False;valid[i,selected]=False;conflict[i,selected]=False;signature[i,selected]=0
                for value in values.values():value[i,selected]=np.nan
                for value in observed_values.values():value[i,selected]=np.nan
        frame=frame[frame.ts_code.isin(ci)&frame.trade_date.isin(di)].copy()
        if frame.empty:continue
        frame,fields,good=numeric_contract(api,frame)
        frame[fields]=frame[fields].astype('float64')
        frame['_valid']=good
        stats['rows']+=len(frame);stats['invalid_rows']+=int((~good).sum())
        if api=='daily_basic':
            # Null says "unobserved", not "another value". Coalesce only actual
            # non-null observations; joint observed disagreements stay unavailable.
            groups=frame.groupby(['ts_code','trade_date'],sort=False,dropna=False)
            differing=groups[fields].nunique(dropna=True).gt(1).any(axis=1)
            merged=groups[fields].first().reset_index()
            d=merged.trade_date.map(di).to_numpy(dtype=int);c=merged.ts_code.map(ci).to_numpy(dtype=int)
            before=valid[d,c].copy();already=seen[d,c].copy();seen[d,c]=True
            conflict[d,c] |= differing.to_numpy(dtype=bool)
            for field,store in observed_values.items():
                new=merged[field].to_numpy(dtype=float);previous=store[d,c]
                disagreements=np.isfinite(previous)&np.isfinite(new)&(previous!=new)
                conflict[d[disagreements],c[disagreements]]=True
                take=~np.isfinite(previous)&np.isfinite(new);store[d[take],c[take]]=new[take]
            t,total,flt,free=(observed_values[f][d,c] for f in basic_fields)
            valid[d,c]=(np.isfinite(t)&np.isfinite(total)&np.isfinite(flt)&(t>=0)&(total>0)&(flt>0)&
                (flt<=total+.0001)&(~np.isfinite(free)|((free>=0)&(free<=flt+.0001))))
            stats['partial_duplicate_cells_completed_from_observations']+=int((already&~before&valid[d,c]&~conflict[d,c]).sum())
            for field,value in values.items():value[d,c]=observed_values[field][d,c]
            continue
        # Canonical numeric equality: "1" and "1.0" describe the same observation.
        frame=frame.drop_duplicates(subset=['ts_code','trade_date']+fields)
        local_conflicts=frame[frame.duplicated(['ts_code','trade_date'],keep=False)]
        local_conflicts=local_conflicts[['ts_code','trade_date']].drop_duplicates()
        frame=frame.drop_duplicates(['ts_code','trade_date'])
        d=frame.trade_date.map(di).to_numpy(dtype=int);c=frame.ts_code.map(ci).to_numpy(dtype=int)
        hashes=pd.util.hash_pandas_object(frame[fields],index=False).to_numpy(dtype=np.uint64)
        conflicts=seen[d,c] & (signature[d,c]!=hashes)
        conflict[d[conflicts],c[conflicts]]=True
        signature[d,c]=hashes;seen[d,c]=True;valid[d,c]=frame._valid.to_numpy(dtype=bool)
        if not local_conflicts.empty:
            ld=local_conflicts.trade_date.map(di).to_numpy(dtype=int);lc=local_conflicts.ts_code.map(ci).to_numpy(dtype=int)
            conflict[ld,lc]=True
        for field,value in values.items():value[d,c]=frame[field].to_numpy(dtype=float)
    valid &= ~conflict
    for value in values.values():value[~valid]=np.nan
    stats.update(observed_cells=int(seen.sum()),valid_cells=int(valid.sum()),conflicting_cells=int(conflict.sum()))
    return {'valid':valid,'observed':seen,'conflict':conflict,'values':values,'stats':dict(stats)}


def catalog_frames(root, manifest, api, aliases, codes):
    import pandas as pd
    import pyarrow.parquet as pq
    expanded=set(codes)|{old for old,row in aliases.items() if row['stable_code'] in codes}
    def stable(value):return aliases.get(value,{}).get('stable_code',value)
    for part in manifest['tables'].get(api,[]):
        path=root/part['file'];require(file_hash(path)==part['sha256'],'changed audit partition')
        for batch in pq.ParquetFile(path).iter_batches(batch_size=250000):
            frame=batch.to_pandas();frame=frame[frame.ts_code.isin(expanded)].copy()
            frame.ts_code=frame.ts_code.map(stable)
            yield frame,()
    for part in manifest['queries'].get(api,[]):
        path=Path(part['path']);require(file_hash(path)==part['sha256'],'changed audit supplement')
        query=json.loads(path.read_text(encoding='utf-8'))
        require(query['rows_hash']==digest(query['rows']),'changed audit rows')
        rows=[dict(row,ts_code=stable(row['ts_code'])) for row in query['rows'] if row['ts_code'] in expanded]
        frame=pd.DataFrame(rows,columns=manifest['fields'][api].split(','))
        params=part['params']
        replacement=(params['trade_date'],) if 'trade_date' in params else {
            'codes':[stable(params['ts_code'])],'start_date':params['start_date'],'end_date':params['end_date']}
        yield frame,replacement


def audit_catalog(root, coverage, output):
    import numpy as np
    import pandas as pd
    from .historical_adapter import dividend_terms
    from .action_evidence import resolve_action_terms
    from .action_accounting import SUPPORTED
    root,out=Path(root),Path(output);out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((root/'CATALOG.json').read_text(encoding='utf-8'))
    require(manifest['manifest_hash']==digest({k:v for k,v in manifest.items() if k!='manifest_hash'}),'catalog manifest mismatch')
    metadata=json.loads((root/'CATALOG_METADATA.json').read_text(encoding='utf-8'))
    require(metadata['metadata_hash']==digest({k:v for k,v in metadata.items() if k!='metadata_hash'}) and
        metadata['manifest_hash']==manifest['manifest_hash'],'catalog metadata mismatch')
    for filename,expected in metadata['files'].items():require(file_hash(root/filename)==expected,'catalog metadata file changed')
    sessions,codes=coverage['sessions'],coverage['codes'];aliases=manifest.get('identity_aliases',{})
    shape=(len(sessions),len(codes));ci={c:i for i,c in enumerate(codes)};di={d:i for i,d in enumerate(sessions)}
    masks={};numbers={};stats={}
    for api,keep in [('daily',('open','high','low','close','pre_close','vol')),('adj_factor',('adj_factor',)),
                     ('stk_limit',('up_limit','down_limit')),('daily_basic',())]:
        panel=accumulate_panel(api,catalog_frames(root,manifest,api,aliases,codes),sessions,codes,keep=keep)
        masks[api]=panel['valid'];masks[api+'_conflicts']=panel['conflict'];stats[api]=panel['stats']
        numbers.update(panel['values']);print(json.dumps({'audited_api':api,**panel['stats']}),flush=True)
    active=np.load(out/'LOCAL_MASKS.npz')['active']
    halt=np.zeros(shape,bool);partial=np.zeros(shape,bool);halt_observed_dates=set()
    for part in manifest['queries'].get('suspend_d',[]):
        path=Path(part['path']);require(file_hash(path)==part['sha256'],'halt audit query changed')
        query=json.loads(path.read_text(encoding='utf-8'));params=part['params']
        halt_observed_dates.update(d for d in sessions if params['start_date']<=d<=params['end_date'])
        for row in query['rows']:
            code=aliases.get(row['ts_code'],{}).get('stable_code',row['ts_code'])
            if code not in ci or row['trade_date'] not in di:continue
            if row['suspend_type']=='S':
                (partial if row['suspend_timing'] else halt)[di[row['trade_date']],ci[code]]=True
    require(set(sessions)<=halt_observed_dates,'unobserved sparse halt query intervals')
    halt_conflict=halt&masks['daily']&(numbers['vol']>0)
    masks.update(active=active,halt=halt,partial_halt=partial,halt_quote_conflict=halt_conflict)
    names=pd.read_csv(root/'namechange.csv',dtype=str).astype(object).where(lambda f:f.notna(),None)
    name_known=np.zeros(shape,bool);days=np.asarray(sessions,dtype='U8')
    for code,group in names.groupby('ts_code',sort=False):
        if code not in ci:continue
        latest=np.full(len(sessions),'',dtype='U8');chosen=np.full(len(sessions),None,dtype=object)
        conflict=np.zeros(len(sessions),bool);available=conflict.copy()
        for row in group.to_dict('records'):
            start=row.get('start_date');end=row.get('end_date') or '99991231';ann=row.get('ann_date')
            if not start:continue
            span=(days>=start)&(days<=end);new=span&(start>latest)
            same=span&(start==latest)&~new
            conflict[same] |= chosen[same]!=row['name']
            published=np.ones(len(sessions),bool) if not ann else days>=ann
            available[same] &= published[same]
            latest[new]=start;chosen[new]=row['name'];conflict[new]=False;available[new]=published[new]
        name_known[:,ci[code]]=(latest!='')&~conflict&available&active[:,ci[code]]
    masks['name_known_vendor_proxy']=name_known
    iso=lambda d:d[:4]+'-'+d[4:6]+'-'+d[6:]
    dividends=pd.read_csv(root/'dividend.csv',dtype=str).astype(object).where(lambda f:f.notna(),None)
    terms=dividend_terms(dividends.to_dict('records'),[iso(d) for d in sessions])
    terms=resolve_action_terms(terms,dividends.to_dict('records'),[])
    from .pit_universe import canonical_code
    terms=[t for t in terms if canonical_code(t['instrument']) in ci]
    term_by_day={}
    for term in terms:
        code=term['instrument'].replace('.XSHG','.SH').replace('.XSHE','.SZ');day=term['ex_date'].replace('-','')
        if code in ci and day in di:term_by_day[(di[day],ci[code])]=term
    from .market_catalog import factor_corroboration
    from .action_evidence import constant_factor_corroborated
    crosscheck,crosscheck_sources=factor_corroboration(root,manifest,codes,set(sessions))
    corroboration=[];corroborated=np.zeros(shape,bool)
    known=np.zeros(shape,bool);frozen_marks=np.zeros(shape,bool);marks=numbers['close'].copy()
    last_close=np.full(len(codes),np.nan);last_adj=last_close.copy();seen_price=np.zeros(len(codes),bool)
    last_quote_day=np.full(len(codes),None,dtype=object)
    for i,day in enumerate(sessions):
        current=numbers['adj_factor'][i];close=numbers['close'][i];reference=numbers['pre_close'][i]
        same=(current==last_adj)&(abs(reference-last_close)<.000001)
        tiny=(abs(current-last_adj)<=.0005000001)&(abs(current/last_adj-1)<=.00001)&(abs(reference-last_close)<.000001)
        halt_known=halt[i]&(current==last_adj)&np.isfinite(last_close)
        initial=(~seen_price)&masks['daily'][i]&masks['adj_factor'][i]
        known[i]=active[i]&(same|tiny|halt_known|initial)
        frozen_marks[i]=active[i]&halt_known&~masks['daily'][i]
        marks[i,frozen_marks[i]]=last_close[frozen_marks[i]]
        for j in np.flatnonzero(active[i]):
            term=term_by_day.get((i,int(j)))
            if term is not None:known[i,j]=term['status'] in SUPPORTED
            elif not known[i,j] and masks['daily'][i,j] and crosscheck:
                code=codes[int(j)];previous_day=last_quote_day[j]
                before=crosscheck.get((code,previous_day));after=crosscheck.get((code,day))
                if before and after:
                    accepted=constant_factor_corroborated(before,after,float(last_close[j]),float(reference[j]),float(close[j]))
                    corroboration.append({'ts_code':code,'trade_date':day,'previous_quote_date':previous_day,
                        'accepted':accepted,'before_query_sha256':crosscheck_sources['query_hashes'][previous_day],
                        'after_query_sha256':crosscheck_sources['query_hashes'][day],
                        'same_table_factor_before':before['adj_factor'],'same_table_factor_after':after['adj_factor'],
                        'independent_assurance':False})
                    known[i,j]=accepted;corroborated[i,j]=accepted
        have=np.isfinite(close)&active[i];last_close[have]=close[have];seen_price[have]=True
        last_quote_day[have]=day
        have=np.isfinite(current)&active[i];last_adj[have]=current[have]
    masks.update(event_known_vendor_proxy=known,frozen_mark_vendor_proxy=frozen_marks,factor_constant_corroborated_vendor_proxy=corroborated)
    masks['dense_core_complete']=masks['daily']&masks['adj_factor']&masks['stk_limit']&masks['daily_basic']
    numbers['mark']=marks
    dense_expected=active&~halt
    rows=[];gaps=[]
    for j,member in enumerate(coverage['master_rows']):
        code=member['ts_code'];needed=int(dense_expected[:,j].sum());lived=int(active[:,j].sum())
        missing={api:int((dense_expected[:,j]&~masks[api][:,j]).sum()) for api in ('daily','adj_factor','stk_limit','daily_basic')}
        unknown_event=int((active[:,j]&~known[:,j]).sum());unknown_mark=int((active[:,j]&~np.isfinite(marks[:,j])).sum())
        unknown_name=int((active[:,j]&~name_known[:,j]).sum())
        halt_conflicts=int((active[:,j]&halt_conflict[:,j]).sum())
        dense_complete=needed>0 and not any(missing.values())
        rows.append({'ts_code':code,'active_sessions':lived,'dense_expected_sessions':needed,**{k+'_missing':v for k,v in missing.items()},
            'unknown_event_sessions':unknown_event,'unknown_mark_sessions':unknown_mark,
            'unknown_name_sessions':unknown_name,
            'halt_quote_conflict_sessions':halt_conflicts,
            'halt_sessions':int((active[:,j]&halt[:,j]).sum()),'partial_halt_sessions':int((active[:,j]&partial[:,j]).sum()),
            'dense_core_complete':dense_complete,'data_complete_under_vendor_proxy':dense_complete and unknown_event==unknown_mark==unknown_name==halt_conflicts==0,
            'data_assured':False,'scope':'current vendor vintage; declared halt freeze and daily publication proxies; no primary ownership assurance'})
        reasons={**{api:dense_expected[:,j]&~masks[api][:,j] for api in missing},
            'corporate_action_unknown':active[:,j]&~known[:,j],'holding_mark_unknown':active[:,j]&~np.isfinite(marks[:,j]),
            'halt_quote_conflict':active[:,j]&halt_conflict[:,j],
            'name_missing_or_conflicting':active[:,j]&~name_known[:,j]}
        for reason,mask in reasons.items():
            for i in np.flatnonzero(mask):gaps.append({'ts_code':code,'trade_date':sessions[int(i)],'reason':reason})
    pd.DataFrame(rows).to_csv(out/'DATA_STATUS.csv',index=False,lineterminator='\n')
    pd.DataFrame(gaps,columns=['ts_code','trade_date','reason']).to_csv(out/'DATA_GAPS.csv',index=False,lineterminator='\n')
    pd.DataFrame(terms).to_csv(out/'ACTION_TERMS.csv',index=False,lineterminator='\n')
    pd.DataFrame(corroboration,columns=['ts_code','trade_date','previous_quote_date','accepted','before_query_sha256',
        'after_query_sha256','same_table_factor_before','same_table_factor_after','independent_assurance']).to_csv(out/'FACTOR_CORROBORATION.csv',index=False,lineterminator='\n')
    panel_values=dict(numbers);panel_values['adj_factor_value']=panel_values.pop('adj_factor')
    require(not (set(masks)&set(panel_values)),'audit mask/value name collision')
    np.savez_compressed(out/'AUDITED_PANELS.npz',**masks,**panel_values)
    summary={'schema_version':'fuxi.full_census_data_audit.1','manifest_hash':manifest['manifest_hash'],
        'code_hashes':{name:file_hash(Path(__file__).parent/name) for name in
            ('panel_audit.py','market_catalog.py','history.py','historical_adapter.py','action_evidence.py','action_accounting.py','a_share.py','pit_universe.py','contracts.py','datasets.py')},
        'metadata_hash':metadata['metadata_hash'],
        'instrument_count':len(codes),'sessions':len(sessions),'active_lifecycle_cells':int(active.sum()),'dense_expected_cells':int(dense_expected.sum()),
        'dense_core_complete_instruments':sum(r['dense_core_complete'] for r in rows),
        'data_complete_under_vendor_proxy_instruments':sum(r['data_complete_under_vendor_proxy'] for r in rows),
        'formal_data_assured_instruments':0,'stats':stats,'gap_counts':dict(Counter(r['reason'] for r in gaps)),
        'action_status_counts':dict(Counter(r['status'] for r in terms)),'bse_aliases':len(aliases),
        'artifact_hashes':{f:file_hash(out/f) for f in ('DATA_STATUS.csv','DATA_GAPS.csv','ACTION_TERMS.csv','AUDITED_PANELS.npz')},
        'factor_corroboration':{'accepted_cells':int(corroborated.sum()),'compared_cells':len(corroboration),**crosscheck_sources,
            'artifact_sha256':file_hash(out/'FACTOR_CORROBORATION.csv'),'no_threshold_change':True},
        'independent_assurance':False,'historical_exposure':'research_exposed'}
    (out/'DATA_AUDIT.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return summary
