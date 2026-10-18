"""Explicit manual full-A-share inventory and append-only Tushare acquisition."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fuxi_v1.contracts import digest, require
from fuxi_v1.datasets import file_hash
from fuxi_v1.runtime import controlled_writer
from fuxi_v1.tushare_source import TushareSource, VendorFailure

OUT = ROOT / 'reports/fuxi_v1_all_a_20261003'
DATA = ROOT / 'data/fuxi_v1/supplements/20261003'
FIELDS = {
    'bse_mapping': 'name,o_code,n_code,list_date',
    'stock_basic': 'ts_code,symbol,name,market,exchange,list_status,list_date,delist_date',
    'trade_cal': 'exchange,cal_date,is_open,pretrade_date',
    'daily': 'ts_code,trade_date,open,high,low,close,pre_close,change,pct_chg,vol,amount',
    'adj_factor': 'ts_code,trade_date,adj_factor',
    'stk_limit': 'ts_code,trade_date,pre_close,up_limit,down_limit',
    'suspend_d': 'ts_code,trade_date,suspend_timing,suspend_type',
    'daily_basic': 'ts_code,trade_date,turnover_rate,total_share,float_share,free_share',
    'dividend': 'ts_code,end_date,ann_date,div_proc,stk_div,stk_bo_rate,stk_co_rate,cash_div,cash_div_tax,record_date,ex_date,pay_date,div_listdate,imp_ann_date',
    'namechange': 'ts_code,name,start_date,end_date,ann_date,change_reason',
    'stk_factor': 'ts_code,trade_date,close,pre_close,adj_factor',
}


def save(path, data, *, immutable=True):
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(data, ensure_ascii=False, indent=2) + '\n').encode()
    if immutable and path.exists():
        require(digest(json.loads(path.read_text(encoding='utf-8'))) == digest(data), 'immutable output mismatch: ' + path.name)
        return
    if immutable:
        with path.open('xb') as handle: handle.write(raw)
    else:
        temp = path.with_suffix(path.suffix + '.next')
        temp.write_bytes(raw)
        temp.replace(path)


def initialise():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'PLAN.json').exists(): return
    before = OUT / 'before'
    before.mkdir()
    for source in (ROOT / 'fuxi_v1').glob('*.py'):
        shutil.copy2(source, before / source.name)
    save(OUT / 'PLAN.json', {
        'schema_version': 'fuxi.full_a_scope.1', 'frozen_at': datetime.now(timezone.utc).isoformat(),
        'window': ['2021-01-04', '2026-09-30'],
        'scope': 'all common A shares on SH SZ BJ, including lifecycle L D P; CDR and B shares recorded separately',
        'selection_basis': 'security type, listing lifecycle and evidence availability; no factor/performance outcomes',
        'exclusion_policy': 'instrument-specific unresolved evidence only after acquisition/review; service failure never excludes a whole population',
        'excluded_new_entries_do_not_remove_carried_holdings_or_entitlements': True,
        'history_exposure': 'research_exposed', 'posthoc_history_revalidation_cannot_grant_unbiased_qualification': True,
        'collection': {'provider': 'tushare', 'max_calls_per_invocation': 30000, 'serial': True,
                       'raw_data_overwrite': False, 'cache': str(DATA), 'observed_now_is_not_historical_revision_archive': True},
        'previous_source_hashes': {p.name: file_hash(p) for p in before.glob('*.py')},
        'outputs': ['SECURITY_MASTER.json', 'LOCAL_COVERAGE.json', 'COLLECTION_STATUS.json',
                    'DELIVERY_STATUS.csv', 'EXCLUSIONS.json', 'REPORT.md'],
        'live_authority': False})


def client():
    from credentials import get_tushare_token
    return TushareSource(DATA / 'responses', get_tushare_token())


def preflight(source):
    rows, sources, permissions = [], [], {}
    for status in ('L', 'D', 'P'):
        for exchange in ('SSE', 'SZSE', 'BSE'):
            result = source.query('stock_basic', {'exchange': exchange, 'list_status': status}, FIELDS['stock_basic'], page_size=6000)
            require(all(r['list_status'] == status and r['exchange'] == exchange for r in result['rows']), 'security master query identity')
            rows.extend(result['rows']); sources.extend(result['evidence'])
    require(rows and len({r['ts_code'] for r in rows}) == len(rows), 'empty/duplicate security master')
    save(OUT / 'SECURITY_MASTER.json', {'rows': rows, 'evidence': sources, 'rows_hash': digest(rows),
        'counts': dict(Counter(r['list_status'] for r in rows)),
        'market_counts': dict(Counter(r['market'] for r in rows)), 'retrieved_master_is_not_PIT_revision_archive': True})
    for api, params in [('trade_cal', {'exchange': 'SSE', 'start_date': '20210101', 'end_date': '20261003'}),
                        ('daily', {'trade_date': '20260930'}), ('adj_factor', {'trade_date': '20260930'}),
                        ('stk_limit', {'trade_date': '20260930'}), ('suspend_d', {'trade_date': '20260930'}),
                        ('daily_basic', {'trade_date': '20260930'}),
                        ('dividend', {'ts_code': '600519.SH'}), ('namechange', {'ts_code': '600519.SH'})]:
        try:
            result = source.query(api, params, FIELDS[api])
            permissions[api] = {'state': 'observed_response', 'rows': len(result['rows']), 'evidence': result['evidence']}
            if api == 'trade_cal': save(OUT / 'CALENDAR.json', result)
        except VendorFailure as exc:
            permissions[api] = {'state': exc.category, 'vendor_code': exc.code}
        print(json.dumps({'preflight_api': api, 'state': permissions[api]['state'],
                          'rows': permissions[api].get('rows')}), flush=True)
    save(OUT / 'PREFLIGHT.json', {'permissions': permissions, 'master_count': len(rows), 'calls': source.calls})


def inventory():
    import numpy as np
    import pandas as pd
    from fuxi_v1.a_share import board
    from fuxi_v1.history import normalize_price_frame
    if (OUT / 'LOCAL_COVERAGE.json').exists(): return
    master = json.loads((OUT / 'SECURITY_MASTER.json').read_text(encoding='utf-8'))
    calendar = json.loads((OUT / 'CALENDAR.json').read_text(encoding='utf-8'))
    sessions = sorted(r['cal_date'] for r in calendar['rows'] if r['is_open'] == 1 and '20210101' <= r['cal_date'] <= '20260930')
    require(sessions and len(sessions) == len(set(sessions)), 'calendar identity')
    rows = sorted((r for r in master['rows'] if board(r['ts_code'])), key=lambda r: r['ts_code'])
    codes = [r['ts_code'] for r in rows]
    ci, di = {c:i for i,c in enumerate(codes)}, {d:i for i,d in enumerate(sessions)}
    masks, sources = {}, {}
    for api, filename, fields in [('daily','daily_prices.csv',FIELDS['daily']),
                                  ('adj_factor','adj_factor_daily.csv',FIELDS['adj_factor']),
                                  ('stk_limit','stk_limit.csv',FIELDS['stk_limit']),
                                  ('daily_basic','daily_basic.csv',FIELDS['daily_basic'])]:
        path = ROOT / 'data/raw' / filename
        stat = (path.stat().st_size, path.stat().st_mtime_ns)
        mask = np.zeros((len(sessions), len(codes)), dtype=bool)
        observed = mask.copy(); invalid, total = 0, 0
        header = pd.read_csv(path, nrows=0).columns
        columns = [x for x in fields.split(',') if x in header]
        for frame in pd.read_csv(path, usecols=columns, dtype=str, chunksize=200000):
            frame.trade_date = frame.trade_date.str.replace('-', '', regex=False)
            frame = frame[frame.ts_code.isin(ci) & frame.trade_date.isin(di)].copy()
            if frame.empty: continue
            total += len(frame)
            if api == 'daily':
                frame = normalize_price_frame(frame)
                valid = frame.field_contract_status != 'unresolved_field_contract'
            elif api == 'adj_factor': valid = pd.to_numeric(frame.adj_factor, errors='coerce') > 0
            elif api == 'stk_limit':
                up, down = pd.to_numeric(frame.up_limit, errors='coerce'), pd.to_numeric(frame.down_limit, errors='coerce')
                valid = (down > 0) & (up >= down)
            else:
                valid = pd.to_numeric(frame.turnover_rate, errors='coerce').notna() & (pd.to_numeric(frame.float_share, errors='coerce') > 0)
            d, c = frame.trade_date.map(di).to_numpy(dtype=int), frame.ts_code.map(ci).to_numpy(dtype=int)
            observed[d,c] = True
            mask[d[valid.to_numpy()],c[valid.to_numpy()]] = True
            invalid += int((~valid).sum())
        require((path.stat().st_size,path.stat().st_mtime_ns) == stat, 'source changed during scan')
        masks[api] = mask
        sources[api] = {'path': str(path), 'sha256': file_hash(path), 'window_rows': total,
                        'invalid_rows': invalid, 'duplicate_key_rows': total-int(observed.sum()),
                        'valid_unique_cells': int(mask.sum())}
        print(json.dumps({'scanned_api':api, **{k:sources[api][k] for k in ('window_rows','invalid_rows','valid_unique_cells')}}), flush=True)
    active = np.zeros_like(masks['daily'])
    for j,row in enumerate(rows):
        listed = row['list_date']
        if board(row['ts_code']) == 'bj': listed = max(listed or '', '20211115')
        active[:,j] = [(listed and listed <= day and (not row['delist_date'] or day < row['delist_date'])) for day in sessions]
    masks['active'] = active
    np.savez_compressed(OUT / 'LOCAL_MASKS.npz', **masks)
    acquisition = {}
    for api in ('daily','adj_factor','stk_limit','daily_basic'):
        need = active & ~masks[api]
        acquisition[api] = [sessions[i] for i in range(len(sessions)) if need[i].any()]
    save(OUT / 'LOCAL_COVERAGE.json', {'sessions':sessions, 'codes':codes, 'master_rows':rows,
        'sources':sources, 'active_lifecycle_cells':int(active.sum()), 'date_acquisition':acquisition,
        'masks_sha256':file_hash(OUT/'LOCAL_MASKS.npz'),
        'dense_gap_is_not_proof_of_untradeability_or_delisting':True,
        'unknown_event_and_mark_coverage_remain_separate':True})


def collect(source):
    import numpy as np
    import calendar as cal
    from datetime import date, timedelta
    coverage = json.loads((OUT / 'LOCAL_COVERAGE.json').read_text(encoding='utf-8'))
    results = DATA / 'queries'; results.mkdir(parents=True, exist_ok=True)
    schedule = [('bse_mapping', {}, 1000)]
    # Sparse event inventory: all announced name histories, all halt intervals, and each ex-date.
    schedule += [('namechange', {'start_date':'19900101','end_date':'20261003'}, 1000)]
    cursor = date(2021,1,1)
    while cursor < date(2026,10,1):
        end = date(cursor.year, cursor.month, cal.monthrange(cursor.year,cursor.month)[1])
        schedule.append(('suspend_d', {'start_date':cursor.strftime('%Y%m%d'),'end_date':end.strftime('%Y%m%d')}, 5000))
        cursor = end + timedelta(days=1)
    # Daily ex-date requests avoid annual-report-only dividend caches and include interim distributions.
    schedule += [('dividend', {'ex_date':d}, 2000) for d in coverage['sessions']]
    masks = np.load(OUT/'LOCAL_MASKS.npz')
    halts = np.zeros_like(masks['daily']); di={d:i for i,d in enumerate(coverage['sessions'])}; ci={c:i for i,c in enumerate(coverage['codes'])}
    # All monthly queries must be observed before sparse absence can be interpreted.
    halt_queries = [q for q in schedule if q[0]=='suspend_d']
    complete_halts = True; halt_hashes=[]
    for api,params,page_size in halt_queries:
        identity={'api':api,'params':params,'fields':FIELDS[api],'page_size':page_size}
        path=results/(digest(identity)+'.json')
        if not path.exists(): complete_halts=False;break
        saved=json.loads(path.read_text(encoding='utf-8'))
        require(saved['rows_hash']==digest(saved['rows']),'halt cache mismatch')
        halt_hashes.append(file_hash(path))
        for row in saved['rows']:
            if row['suspend_type']=='S' and not row['suspend_timing'] and row['ts_code'] in ci and row['trade_date'] in di:
                halts[di[row['trade_date']],ci[row['ts_code']]]=True
    acquisition={}
    for api in ('daily','adj_factor','stk_limit','daily_basic'):
        need=masks['active'] & ~masks[api]
        if complete_halts and api in ('daily','stk_limit','daily_basic'): need &= ~halts
        acquisition[api]=[d for i,d in enumerate(coverage['sessions']) if need[i].any()]
        page_size={'daily':6000,'daily_basic':6000,'stk_limit':5800,'adj_factor':5000}[api]
        schedule += [(api, {'trade_date':d}, page_size) for d in acquisition[api]]
    schedule_hash = digest(schedule)
    plan_path=OUT/('COLLECTION_PLAN_V3.json' if complete_halts else 'COLLECTION_PLAN.json')
    save(plan_path, {'schedule_hash':schedule_hash,'queries':schedule,
         'planned_queries':len(schedule),'sparse_empty_response_requires_explicit_observation':True,
         'pagination_contract':{'daily':{'page_size':6000,'source':'https://tushare.pro/document/2?doc_id=27'},
             'daily_basic':{'page_size':6000,'source':'https://tushare.pro/document/2?doc_id=32'},
             'stk_limit':{'page_size':5800,'source':'https://tushare.pro/document/2?doc_id=183'},
             'tail_page_still_required_when_exactly_full':True,'existing_query_artifacts_preserved':True},
         **({'halt_source_hashes':halt_hashes,'amendment':'avoid requesting absent dense bars on evidenced full-day halts; preserve their valuation/equity obligations',
             'date_acquisition':acquisition} if complete_halts else {})})
    completed, counts = 0, Counter()
    for api, params, page_size in schedule:
        identity = {'api':api, 'params':params, 'fields':FIELDS[api], 'page_size':page_size}
        path = results / (digest(identity)+'.json')
        if path.exists():
            result = json.loads(path.read_text(encoding='utf-8'))
            require(result['identity']==identity and result['rows_hash']==digest(result['rows']), 'query cache mismatch')
        else:
            try:
                result = source.query(api, params, FIELDS[api], page_size=page_size)
            except VendorFailure as exc:
                save(OUT/'COLLECTION_STATUS.json', {'state':exc.category,'api':api,'params':params,
                     'completed_queries':completed,'planned_queries':len(schedule),'new_calls':source.calls,
                     'schedule_hash':schedule_hash,'resume_supported':True}, immutable=False)
                raise
            # Persist whole validated query, including empty sparse responses and every page identity.
            result.update(identity=identity, rows_hash=digest(result['rows']))
            save(path,result)
        completed += 1; counts[api] += len(result['rows'])
        if completed % 25 == 0 or completed == len(schedule):
            status = {'state':'complete' if completed==len(schedule) else 'collecting',
                      'completed_queries':completed,'planned_queries':len(schedule),'rows_by_api':dict(counts),
                      'new_calls':source.calls,'schedule_hash':schedule_hash,'resume_supported':True}
            save(OUT/'COLLECTION_STATUS.json',status,immutable=False)
            print(json.dumps(status),flush=True)


def catalog():
    import pandas as pd
    from fuxi_v1.market_catalog import build_catalog,TABLE_APIS
    status=json.loads((OUT/'COLLECTION_STATUS.json').read_text(encoding='utf-8'))
    require(status['state']=='complete','finish whole collection schedule before catalog sealing')
    destination=DATA/'catalog'
    manifest=build_catalog(ROOT/'data/raw',DATA/'queries',destination,FIELDS)
    calendar=json.loads((OUT/'CALENDAR.json').read_text(encoding='utf-8'))
    pd.DataFrame(calendar['rows']).to_csv(destination/'trade_calendar.csv',index=False,lineterminator='\n')
    master=json.loads((OUT/'SECURITY_MASTER.json').read_text(encoding='utf-8'))
    pd.DataFrame(master['rows']).to_csv(destination/'delist_dates.csv',index=False,lineterminator='\n')
    for api in ('namechange','dividend'):
        rows=[]
        for query in manifest['queries'][api]:
            rows.extend(json.loads(Path(query['path']).read_text(encoding='utf-8'))['rows'])
        aliases=manifest['identity_aliases']
        rows=[dict(r,ts_code=aliases.get(r['ts_code'],{}).get('stable_code',r['ts_code'])) for r in rows]
        pd.DataFrame(rows,columns=FIELDS[api].split(',')).drop_duplicates().to_csv(destination/(api+'.csv'),index=False,lineterminator='\n')
    metadata={'manifest_hash':manifest['manifest_hash'],
        'files':{f:file_hash(destination/f) for f in ('trade_calendar.csv','delist_dates.csv','namechange.csv','dividend.csv')}}
    metadata['metadata_hash']=digest(metadata)
    save(destination/'CATALOG_METADATA.json',metadata)
    save(OUT/'CATALOG_RESULT.json',{'manifest_hash':manifest['manifest_hash'],'metadata_hash':metadata['metadata_hash'],
        'path':str(destination),'investment_data_assured':False})
    from fuxi_v1.pit_universe import lifecycle_from_rows,build_pit_cohort
    coverage=json.loads((OUT/'LOCAL_COVERAGE.json').read_text(encoding='utf-8'))
    sessions=[d[:4]+'-'+d[4:6]+'-'+d[6:] for d in coverage['sessions']]
    codes,declaration,neutrality=build_pit_cohort(lifecycle_from_rows(master['rows']),sessions,
        salt='all-a-census-20261003',rule_kind='all_a_share_census')
    members={r['ts_code']:r for r in master['rows']}
    save(OUT/'FULL_A_RESEARCH_PLAN.json',{'schema_version':'fuxi.all_a_history_plan.1',
        'frozen_scope_at':json.loads((OUT/'PLAN.json').read_text(encoding='utf-8'))['frozen_at'],
        'window':{'start':sessions[0],'end':sessions[-1],'calibration_end':'2022-12-30','research_start':'2023-01-16',
            'horizon_sessions':5,'warmup_sessions':252},
        'cohort':dict(declaration,selection=declaration['rule'],members=[members[c] for c in codes]),'neutrality':neutrality,
        'universe':{'listing_age_calendar_days':90,'name_policy':'normal published effective name; ST/unknown unavailable for new selection',
            'halt_policy':'explicit observed monthly vendor events; missing required quotes remain unavailable'},
        'raw_sources':{filename:manifest['sources'][api] for filename,api in TABLE_APIS.items()},
        'catalog_manifest_hash':manifest['manifest_hash'],'catalog_file_sha256':file_hash(destination/'CATALOG.json'),
        'catalog_metadata_hash':metadata['metadata_hash'],
        'history_exposure':'research_exposed','formal_data_assured':False,
        'future_entry_exclusions_do_not_rewrite_this_history':True})


def active_catalog():
    pointer=OUT/'ACTIVE_CATALOG.json'
    if not pointer.exists():return DATA/'catalog',OUT/'FULL_A_RESEARCH_PLAN.json'
    record=json.loads(pointer.read_text(encoding='utf-8'))
    root=Path(record['path']);plan=Path(record['full_plan_path'])
    require(file_hash(root/'CATALOG.json')==record['catalog_sha256'] and
        file_hash(plan)==record['full_plan_sha256'],'active catalog identity changed')
    return root,plan


def repair(source):
    import pandas as pd
    status=pd.read_csv(OUT/'audit_attempt1/DATA_STATUS.csv')
    schedule=[('namechange',{'ts_code':c},1000) for c in sorted(status.loc[status.unknown_name_sessions>0,'ts_code'])]
    schedule += [('daily',{'ts_code':c,'start_date':'20210104','end_date':'20260930'},6000)
                 for c in sorted(status.loc[status.daily_missing>0,'ts_code'])]
    require(schedule,'repair schedule has no comparisons')
    save(OUT/'REPAIR_PLAN.json',{'queries':schedule,'schedule_hash':digest(schedule),
        'basis_sha256':file_hash(OUT/'audit_attempt1/DATA_STATUS.csv'),
        'selection':'missing data/name evidence only; no performance-based selection',
        'whole_stock_query_is_current_vintage_replacement':True})
    counts=Counter()
    for number,(api,params,page_size) in enumerate(schedule,1):
        identity={'api':api,'params':params,'fields':FIELDS[api],'page_size':page_size}
        path=DATA/'queries'/(digest(identity)+'.json')
        if path.exists():
            result=json.loads(path.read_text(encoding='utf-8'))
            require(result['identity']==identity and result['rows_hash']==digest(result['rows']),'repair cache mismatch')
        else:
            result=source.query(api,params,FIELDS[api],page_size=page_size)
            result.update(identity=identity,rows_hash=digest(result['rows']))
            require(all(r['ts_code']==params['ts_code'] for r in result['rows']),'repair stock identity mismatch')
            if api=='daily':require(all(params['start_date']<=r['trade_date']<=params['end_date'] for r in result['rows']),'repair date identity')
            save(path,result)
        counts[api]+=len(result['rows'])
        record={'state':'complete' if number==len(schedule) else 'collecting','completed_queries':number,
            'planned_queries':len(schedule),'rows_by_api':dict(counts),'new_calls':source.calls,'resume_supported':True}
        save(OUT/'REPAIR_STATUS.json',record,immutable=False)
        if number%10==0 or number==len(schedule):print(json.dumps(record),flush=True)


def core_plan():
    import numpy as np
    import pandas as pd
    from fuxi_v1.pit_universe import canonical_code
    coverage=json.loads((OUT/'LOCAL_COVERAGE.json').read_text(encoding='utf-8'))
    codes,days=coverage['codes'],coverage['sessions']
    keys=['active','daily','stk_limit','daily_basic','halt','event_known_vendor_proxy','close','pre_close']
    with np.load(OUT/'AUDITED_PANELS.npz') as archive:p={k:archive[k] for k in keys}
    ci={c:j for j,c in enumerate(codes)};di={d:i for i,d in enumerate(days)}
    declared=np.zeros_like(p['active'])
    for term in pd.read_csv(OUT/'ACTION_TERMS.csv').to_dict('records'):
        c=canonical_code(term['instrument']);d=term['ex_date'].replace('-','')
        if c in ci and d in di:declared[di[d],ci[c]]=True
    dates=set();last=np.full(len(codes),np.nan);last_day=np.full(len(codes),None,dtype=object);suspect=0
    for i,day in enumerate(days):
        missing=p['active'][i]&p['daily'][i]&~p['event_known_vendor_proxy'][i]&~declared[i]&np.isfinite(last)&(abs(p['pre_close'][i]-last)<.000001)
        suspect+=int(missing.sum())
        if missing.any():dates.add(day);dates.update(last_day[missing].tolist())
        have=p['active'][i]&p['daily'][i];last[have]=p['close'][i,have];last_day[have]=day
    require(None not in dates,'missing crosscheck anchor')
    schedule=[('stk_factor',{'trade_date':d},10000) for d in sorted(dates)]
    for api,size in [('daily_basic',6000),('stk_limit',5800)]:
        need=p['active']&p['daily']&~p['halt']&~p[api]
        schedule += [(api,{'ts_code':c,'start_date':days[0],'end_date':days[-1]},size) for j,c in enumerate(codes) if need[:,j].any()]
    require(schedule,'empty core repair scope')
    save(OUT/'CORE_REPAIR_PLAN.json',{'queries':schedule,'schedule_hash':digest(schedule),
        'basis_audit_sha256':file_hash(OUT/'DATA_AUDIT.json'),'basis_panels_sha256':file_hash(OUT/'AUDITED_PANELS.npz'),
        'crosscheck_suspect_cells':suspect,'crosscheck_dates':sorted(dates),
        'source':'https://tushare.pro/document/2?doc_id=296',
        'crosscheck_gate':'both dates present; exact unchanged positive factor in stk_factor; all raw close/reference quotes agree; no declared action; otherwise unknown',
        'drift_tolerance_change':False,'independent_investment_assurance':False,
        'dense_repair':'whole stock/date interval observations for missing basic/limit cells with actual quotes; no outcome selection'})
    print(json.dumps({'planned_queries':len(schedule),'api_counts':dict(Counter(q[0] for q in schedule)),
        'crosscheck_suspect_cells':suspect,'crosscheck_dates':len(dates)}),flush=True)


def core_collect(source):
    plan=json.loads((OUT/'CORE_REPAIR_PLAN.json').read_text(encoding='utf-8'));counts=Counter()
    require(plan['basis_audit_sha256']==file_hash(OUT/'DATA_AUDIT.json'),'core repair basis changed')
    schedule=plan['queries']
    for number,(api,params,size) in enumerate(schedule,1):
        identity={'api':api,'params':params,'fields':FIELDS[api],'page_size':size};path=DATA/'queries'/(digest(identity)+'.json')
        if path.exists():
            result=json.loads(path.read_text(encoding='utf-8'));require(result['identity']==identity and result['rows_hash']==digest(result['rows']),'core query changed')
        else:
            result=source.query(api,params,FIELDS[api],page_size=size)
            require(all(r['trade_date']==params['trade_date'] for r in result['rows']) if 'trade_date' in params else
                all(r['ts_code']==params['ts_code'] and params['start_date']<=r['trade_date']<=params['end_date'] for r in result['rows']),'core query identity')
            result.update(identity=identity,rows_hash=digest(result['rows']));save(path,result)
        counts[api]+=len(result['rows'])
        if number%10==0 or number==len(schedule):
            status={'state':'complete' if number==len(schedule) else 'collecting','completed_queries':number,
                'planned_queries':len(schedule),'rows_by_api':dict(counts),'new_calls':source.calls,'resume_supported':True}
            save(OUT/'CORE_REPAIR_STATUS.json',status,immutable=False);print(json.dumps(status),flush=True)


def refresh_catalog():
    import pandas as pd
    require(json.loads((OUT/'REPAIR_STATUS.json').read_text(encoding='utf-8'))['state']=='complete','repair incomplete')
    old,old_plan=active_catalog()
    revision=int(old.name.rsplit('_v',1)[1])+1 if '_v' in old.name else 2
    if revision>2:require(json.loads((OUT/'CORE_REPAIR_STATUS.json').read_text(encoding='utf-8'))['state']=='complete','core repair incomplete')
    destination=DATA/('catalog_v%d'%revision);destination.mkdir(exist_ok=True)
    manifest=json.loads((old/'CATALOG.json').read_text(encoding='utf-8'))
    require(manifest['manifest_hash']==digest({k:v for k,v in manifest.items() if k!='manifest_hash'}),'base catalog changed')
    old_hash=manifest.pop('manifest_hash');index={}
    for parts in manifest['tables'].values():
        for part in parts:part['file']=str((old/part['file']).resolve())
    for path in sorted((DATA/'queries').glob('*.json')):
        record=json.loads(path.read_text(encoding='utf-8'));require(record['rows_hash']==digest(record['rows']),'refresh query changed')
        identity=record['identity'];index.setdefault(identity['api'],[]).append({'path':str(path.resolve()),'sha256':file_hash(path),
            'params':identity['params'],'observation_clock':max(x['retrieved_at'] for x in record['evidence'])})
    for parts in index.values():parts.sort(key=lambda q:(q['observation_clock'],q['sha256']))
    manifest.update(queries=index,fields=FIELDS,parent_manifest_hash=old_hash,parent_catalog_sha256=file_hash(old/'CATALOG.json'))
    manifest['manifest_hash']=digest(manifest);save(destination/'CATALOG.json',manifest)
    for filename in ('trade_calendar.csv','delist_dates.csv'):
        if not (destination/filename).exists():shutil.copy2(old/filename,destination/filename)
    aliases=manifest['identity_aliases']
    def stable(code):return aliases.get(code,{}).get('stable_code',code)
    for api in ('namechange','dividend'):
        rows=[]
        for query in manifest['queries'][api]:
            params=query['params']
            if api=='namechange' and params.get('ts_code'):
                code=stable(params['ts_code']);rows=[r for r in rows if r['ts_code']!=code]
            incoming=json.loads(Path(query['path']).read_text(encoding='utf-8'))['rows']
            rows.extend(dict(r,ts_code=stable(r['ts_code'])) for r in incoming)
        target=destination/(api+'.csv')
        raw=pd.DataFrame(rows,columns=FIELDS[api].split(',')).drop_duplicates().to_csv(index=False,lineterminator='\n').encode('utf-8')
        if target.exists():require(target.read_bytes()==raw,'refreshed metadata resume mismatch')
        else:target.write_bytes(raw)
    metadata={'manifest_hash':manifest['manifest_hash'],
        'files':{f:file_hash(destination/f) for f in ('trade_calendar.csv','delist_dates.csv','namechange.csv','dividend.csv')}}
    metadata['metadata_hash']=digest(metadata);save(destination/'CATALOG_METADATA.json',metadata)
    full=json.loads(old_plan.read_text(encoding='utf-8'))
    full.update(catalog_manifest_hash=manifest['manifest_hash'],catalog_file_sha256=file_hash(destination/'CATALOG.json'),
        catalog_metadata_hash=metadata['metadata_hash'],parent_full_plan_sha256=file_hash(old_plan))
    plan=OUT/('FULL_A_RESEARCH_PLAN_V%d.json'%revision);save(plan,full)
    if (OUT/'ACTIVE_CATALOG.json').exists():save(OUT/('ACTIVE_CATALOG_V%d.json'%(revision-1)),json.loads((OUT/'ACTIVE_CATALOG.json').read_text(encoding='utf-8')))
    save(OUT/'ACTIVE_CATALOG.json',{'path':str(destination),'manifest_hash':manifest['manifest_hash'],
        'metadata_hash':metadata['metadata_hash'],'catalog_sha256':file_hash(destination/'CATALOG.json'),
        'full_plan_path':str(plan),'full_plan_sha256':file_hash(plan)},immutable=False)
    print(json.dumps({'catalog_path':str(destination),'queries':sum(map(len,index.values())),'manifest_hash':manifest['manifest_hash']}),flush=True)


def audit():
    from fuxi_v1.panel_audit import audit_catalog
    coverage=json.loads((OUT/'LOCAL_COVERAGE.json').read_text(encoding='utf-8'))
    require(not (OUT/'DATA_AUDIT.json').exists(),'data audit already sealed')
    result=audit_catalog(active_catalog()[0],coverage,OUT)
    print(json.dumps(result),flush=True)


def calendar_check(source):
    sse=json.loads((OUT/'CALENDAR.json').read_text(encoding='utf-8'))
    szse=source.query('trade_cal',{'exchange':'SZSE','start_date':'20210101','end_date':'20261003'},FIELDS['trade_cal'])
    def opens(rows):return sorted({r['cal_date'] for r in rows if str(r['is_open'])=='1'})
    left,right=opens(sse['rows']),opens(szse['rows'])
    require(left and right and left==right,'SH/SZ calendars differ; explicit exchange-specific schedule required')
    save(OUT/'CALENDAR_CROSSCHECK.json',{'sse_query_sha256':file_hash(OUT/'CALENDAR.json'),
        'szse_response':szse,'matched_open_sessions':len(left),
        'bse_calendar_basis':'vendor documents BSE uses SH/SZ calendar; broker/exchange observed evidence still unassured',
        'basis_url':'https://tushare.pro/document/2?doc_id=26','independent_calendar_assurance':False})


def repair_consumer_plan():
    root,old_plan_path=active_catalog();old=json.loads(old_plan_path.read_text(encoding='utf-8'))
    require(old['cohort']['rule_kind']=='all_a_share_census' and old['cohort'].get('rule'),'missing explicit census declaration')
    require('selection' not in old['cohort'],'consumer plan already repaired')
    destination=OUT/'smoke_attempt1'
    require(not destination.exists(),'consumer failure already preserved')
    source=OUT/'smoke'
    require(source.resolve().is_relative_to(OUT.resolve()) and destination.resolve().is_relative_to(OUT.resolve()),'preservation path escape')
    source.rename(destination)
    old_smoke=OUT/'FULL_A_SMOKE_PLAN.json';old_smoke.rename(destination/'FULL_A_SMOKE_PLAN.json')
    test_receipt=json.loads((OUT/'TEST_RESULT_FINAL.json').read_text(encoding='utf-8'))
    require(test_receipt['code_hashes']=={p.name:file_hash(p) for p in (ROOT/'fuxi_v1').glob('*.py')},'failed consumer code changed')
    save(destination/'FAILURE.json',{'command':'python -X utf8 -m fuxi_v1 snapshot-history',
        'execution_session':19527,'exit_code':1,'error_type':'KeyError','reason':'selection',
        'observation':'consumer completed side-file calculations but failed before history input publication',
        'plan_sha256':file_hash(destination/'FULL_A_SMOKE_PLAN.json'),'code_hashes':test_receipt['code_hashes'],
        'partial_file_hashes':{str(p.relative_to(destination)):file_hash(p) for p in destination.rglob('*') if p.is_file()}})
    fixed=dict(old);fixed['cohort']=dict(old['cohort'],selection=old['cohort']['rule'])
    fixed['consumer_schema_repair']={'parent_plan_sha256':file_hash(old_plan_path),
        'changed_field':'cohort.selection','value_basis':'exact cohort.rule; descriptive contract only',
        'cohort_window_and_data_unchanged':True}
    check=dict(fixed);check.pop('consumer_schema_repair');check['cohort']=dict(check['cohort']);check['cohort'].pop('selection')
    require(check==old,'consumer metadata repair changed research scope')
    new_plan=OUT/'FULL_A_RESEARCH_PLAN_SCHEMA_V1.json';save(new_plan,fixed)
    pointer=json.loads((OUT/'ACTIVE_CATALOG.json').read_text(encoding='utf-8'))
    save(OUT/'ACTIVE_CATALOG_V3.json',pointer)
    pointer.update(full_plan_path=str(new_plan),full_plan_sha256=file_hash(new_plan))
    save(OUT/'ACTIVE_CATALOG.json',pointer,immutable=False)
    print(json.dumps({'consumer_plan_repaired':True,'catalog_unchanged':str(root),'failure_preserved':str(destination)}),flush=True)


def smoke_plan():
    from fuxi_v1.pit_universe import lifecycle_from_rows,build_pit_cohort
    master=json.loads((OUT/'SECURITY_MASTER.json').read_text(encoding='utf-8'))
    full_path=active_catalog()[1];full=json.loads(full_path.read_text(encoding='utf-8'))
    coverage=json.loads((OUT/'LOCAL_COVERAGE.json').read_text(encoding='utf-8'))
    sessions=[d[:4]+'-'+d[4:6]+'-'+d[6:] for d in coverage['sessions'] if '20260506'<=d<='20260930']
    codes,declaration,neutrality=build_pit_cohort(lifecycle_from_rows(master['rows']),sessions,
        salt='full-census-consumer-smoke-20261003',rule_kind='all_a_share_census')
    members={r['ts_code']:r for r in master['rows']}
    plan=dict(full,window={'start':sessions[0],'end':sessions[-1],'calibration_end':'2026-07-31',
        'research_start':'2026-08-11','horizon_sessions':5,'warmup_sessions':20},
        cohort=dict(declaration,selection=declaration['rule'],members=[members[c] for c in codes]),neutrality=neutrality,
        frozen_at=datetime.now(timezone.utc).isoformat(),
        scope='full census in a bounded recent window; native consumer and account smoke only, not full-history or economic certification',
        parent_full_history_plan_sha256=file_hash(full_path))
    save(OUT/'FULL_A_SMOKE_PLAN.json',plan)
    print(json.dumps({'smoke_instruments':len(codes),'smoke_sessions':len(sessions),'formal_admission':False}),flush=True)


def registry():
    import csv
    import pandas as pd
    from datetime import timedelta
    from fuxi_v1.a_share import board,rule_for,delivery_verdict
    coverage=json.loads((OUT/'LOCAL_COVERAGE.json').read_text(encoding='utf-8'))
    checks=json.loads((OUT/'contracts_v3/TEN_ROUND_CONTRACTS.json').read_text(encoding='utf-8'))
    require(checks['rules_hash']==digest({c:rule_for(c) for c in coverage['codes']}),'stale quantity rule rounds')
    require(checks['code_hashes']=={p.name:file_hash(p) for p in (ROOT/'fuxi_v1').glob('*.py')},'stale contract source code')
    require(len(checks['rounds'])==10 and all(r['passed']==len(coverage['codes']) for r in checks['rounds']),'all-name account contracts incomplete')
    require(checks['runner_sha256']==file_hash(ROOT/'scripts/fuxi_all_a_contracts_20261003.py'),'stale contract runner')
    for number in range(1,11):
        path=OUT/'contracts_v3'/('CONTRACT_ROUND_%02d.json'%number)
        require(file_hash(path)==checks['round_artifact_hashes'][str(number)],'stale contract round')
        record=json.loads(path.read_text(encoding='utf-8'))
        require(record['summary']['rows_hash']==digest(record['rows']) and len(record['rows'])==len(coverage['codes']) and
            {r['instrument'] for r in record['rows']}==set(coverage['codes']),'contract cohort omission or changed rows')
    audit=json.loads((OUT/'DATA_AUDIT.json').read_text(encoding='utf-8'))
    require(all(file_hash(ROOT/'fuxi_v1'/name)==expected for name,expected in audit['code_hashes'].items()),'stale data audit code')
    require(file_hash(OUT/'DATA_STATUS.csv')==audit['artifact_hashes']['DATA_STATUS.csv'],'stale data status ledger')
    require(file_hash(OUT/'FACTOR_CORROBORATION.csv')==audit['factor_corroboration']['artifact_sha256'],'stale factor corroboration ledger')
    data={row['ts_code']:row for row in pd.read_csv(OUT/'DATA_STATUS.csv').to_dict('records')}
    history=json.loads((OUT/'TEN_ROUND_HISTORY_CHECKS.json').read_text(encoding='utf-8'))
    require(len(history['rounds'])==10,'real-history account checks incomplete')
    require(history['runner_sha256']==file_hash(ROOT/'scripts/fuxi_all_a_history_checks_20261003.py'),'stale history runner')
    valid_rounds=Counter();filled_rounds=Counter();windows={c:set() for c in coverage['codes']}
    for number in range(1,11):
        path=OUT/('HISTORY_ACCOUNT_ROUND_%02d.json'%number)
        require(file_hash(path)==history['round_artifact_hashes'][str(number)],'stale actual-history round')
        record=json.loads(path.read_text(encoding='utf-8'))
        require(record['rows_hash']==digest(record['rows']),'actual-history row ledger changed')
        for row in record['rows']:
            if row['status']=='valid':
                valid_rounds[row['ts_code']]+=1;windows[row['ts_code']].add(tuple(row['window']))
            if row.get('filled_orders',0)>0:filled_rounds[row['ts_code']]+=1
    window_exceptions=[]
    for member in coverage['master_rows']:
        code=member['ts_code'];state=data[code]
        if valid_rounds[code]==10:continue
        reason=('outside_declared_historical_window' if state['active_sessions']==0 else
                'fewer_than_three_active_sessions' if state['active_sessions']<3 else
                'all_declared_active_sessions_halted' if state['halt_sessions']==state['active_sessions'] else
                'no_complete_three_session_market_window')
        window_exceptions.append({**member,'reason':reason,'valid_diagnostic_rounds':valid_rounds[code],
            'active_sessions':state['active_sessions'],'halt_sessions':state['halt_sessions'],
            'daily_missing':state['daily_missing'],'daily_basic_missing':state['daily_basic_missing'],
            'unknown_event_sessions':state['unknown_event_sessions'],'unknown_mark_sessions':state['unknown_mark_sessions'],
            'temporary_exclusion':False,'formal_evidence':False})
    save(OUT/'DIAGNOSTIC_WINDOW_EXCEPTIONS.json',{'rows':window_exceptions,
        'reason_counts':dict(Counter(r['reason'] for r in window_exceptions)),
        'selection_scope':'diagnostic availability only; no retroactive universe changes',
        'history_source_hash':history['source_hash'],'data_status_sha256':file_hash(OUT/'DATA_STATUS.csv')})
    previous=ROOT/'reports/fuxi_v1_remediation_20260930/REPORT.md'
    now=datetime.now(timezone.utc)
    exclusions=[{'instrument':'600077.SH','reason':'delisting_transfer_ownership_and_36_transition_marks_unresolved',
        'observed_at':now.isoformat(),'effective_at':now.isoformat(),'review_after':(now+timedelta(days=7)).isoformat(),
        'evidence_ids':[file_hash(previous)],'evidence_files':[str(previous)],
        'reentry_condition':'independent ownership/availability/valuation evidence reviewed; replacement exclusion fact bound to reviewed hashes',
        'new_entries_only':True,'retain_existing_positions_and_entitlements':True}]
    save(OUT/'EXCLUSIONS.json',{'schema_version':'fuxi.prospective_exclusions.1','exclusions':exclusions,
        'history_not_rewritten':True,'historical_exclusion_would_be_posthoc':True,
        'whole_population_infrastructure_gaps_are_not_stock_exclusions':True})
    rows=[]
    for member in coverage['master_rows']:
        code=member['ts_code'];rule=rule_for(code)
        state=data[code]
        rows.append({**member,'board':board(code),'engineering_contract_verified':True,'account_contract_rounds':10,
            'native_history_valid_rounds':valid_rounds[code],
            'native_history_rounds_with_actual_vendor_bar_fill':filled_rounds[code],
            'distinct_history_windows':len(windows[code]),
            'min_buy':rule['min_buy'],'buy_step':rule['buy_step'],'max_limit_order':rule['max_limit_order'],
            'prospective_temporary_exclusion':code=='600077.SH',
            'eligibility_today':'listed' if member['list_status']=='L' else 'historical_only_no_new_live_orders',
            'data_status':'outside_declared_historical_window' if state['active_sessions']==0 else
                'complete_under_vendor_proxy' if state['data_complete_under_vendor_proxy'] else 'named_data_or_event_gaps',
            **{k:state[k] for k in ('active_sessions','daily_missing','adj_factor_missing','stk_limit_missing','daily_basic_missing',
                'unknown_event_sessions','unknown_mark_sessions','unknown_name_sessions','halt_quote_conflict_sessions','dense_core_complete','data_complete_under_vendor_proxy')},
            'account_status':'ten_actual_history_rounds_identified_diagnostic' if valid_rounds[code]==10 else
                'outside_declared_historical_window' if state['active_sessions']==0 else 'historical_market_window_unavailable',
            'account_identifiable_scope':'full ever-eligible holding/entitlement continuity not independently certified',
            'platform_status':'not_observed','historical_revision_archive':'not_independently_assured',
            **delivery_verdict(bool(state['data_complete_under_vendor_proxy']),False,False,False,False)})
    with (OUT/'DELIVERY_STATUS.csv').open('x',encoding='utf-8-sig',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    save(OUT/'DELIVERY_SUMMARY.json',{'instrument_count':len(rows),'boards':dict(Counter(r['board'] for r in rows)),
         'engineering_contract_verified':sum(r['engineering_contract_verified'] for r in rows),
         'formal_research_delivery_ready':sum(r['research_delivery_ready'] for r in rows),
         'live_delivery_ready':sum(r['live_delivery_ready'] for r in rows),
         'instruments_with_ten_identifiable_diagnostic_windows':sum(valid_rounds[c]==10 for c in coverage['codes']),
         'instruments_with_ten_nonzero_diagnostic_fill_rounds':sum(filled_rounds[c]==10 for c in coverage['codes']),
         'instruments_with_ten_distinct_diagnostic_windows':sum(len(windows[c])==10 for c in coverage['codes']),
         'diagnostic_window_exception_counts':dict(Counter(r['reason'] for r in window_exceptions)),
         'diagnostic_window_exceptions_sha256':file_hash(OUT/'DIAGNOSTIC_WINDOW_EXCEPTIONS.json'),
         'data_audit':audit,'ten_actual_history_rounds':history['rounds'],
         'temporary_exclusion_count':len(exclusions),'current_collection_status':json.loads((OUT/'COLLECTION_STATUS.json').read_text(encoding='utf-8')),
         'repair_collection_status':json.loads((OUT/'REPAIR_STATUS.json').read_text(encoding='utf-8')),
         'core_repair_collection_status':json.loads((OUT/'CORE_REPAIR_STATUS.json').read_text(encoding='utf-8')),
         'unresolved_global_dependencies':['independent_vintage_and_event_review','actual_platform_account_and_permissions',
            'broker_execution_backend_and_restart_unknown_order_reconciliation','future_E5'],
         'registry_sha256':file_hash(OUT/'DELIVERY_STATUS.csv')})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('preflight','inventory','collect','calendar-check','catalog','repair','core-plan','core-collect','refresh-catalog','audit','repair-consumer-plan','smoke-plan','registry'))
    args = parser.parse_args()
    with controlled_writer(OUT):
        initialise()
        if args.action == 'preflight': preflight(client())
        elif args.action == 'inventory': inventory()
        elif args.action == 'collect': collect(client())
        elif args.action == 'calendar-check': calendar_check(client())
        elif args.action == 'catalog': catalog()
        elif args.action == 'repair': repair(client())
        elif args.action == 'core-plan': core_plan()
        elif args.action == 'core-collect': core_collect(client())
        elif args.action == 'refresh-catalog': refresh_catalog()
        elif args.action == 'audit': audit()
        elif args.action == 'repair-consumer-plan': repair_consumer_plan()
        elif args.action == 'smoke-plan': smoke_plan()
        elif args.action == 'registry': registry()


if __name__ == '__main__': main()
