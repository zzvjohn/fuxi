"""Sharded read-only base panels plus immutable vendor supplements; no raw overwrites."""
import json
from pathlib import Path
from .contracts import digest, require
from .datasets import file_hash

TABLE_APIS={'daily_prices.csv':'daily','adj_factor_daily.csv':'adj_factor',
            'stk_limit.csv':'stk_limit','daily_basic.csv':'daily_basic','suspend_d.csv':'suspend_d'}


def verify_frozen_sources(root, manifest):
    """Audit immutable partitions independently of subsequently updated raw files."""
    root=Path(root)
    require(manifest['manifest_hash']==digest({k:v for k,v in manifest.items() if k!='manifest_hash'}),
            'catalog manifest mismatch')
    origins=[]
    for api,parts in manifest['tables'].items():
        origin=manifest['sources'][api]
        require(parts,'missing sealed source partitions: '+api)
        for part in parts:
            require(part['source_hash']==origin['sha256'] and file_hash(root/part['file'])==part['sha256'],
                    'sealed source partition changed: '+api)
        path=Path(origin['path'])
        current=file_hash(path) if path.exists() else None
        origins.append({'api':api,'origin_sha256':origin['sha256'],'current_raw_sha256':current,
                        'current_raw_matches':current==origin['sha256'],'partitions_verified':len(parts),
                        'raw_file_consumed':False})
    return {'source_mode':'verified_immutable_partitions','origins':origins,
            'historical_revision_archive_certified':False}


def factor_corroboration(root, manifest, codes, dates=None):
    """Same-table factors corroborate unchanged raw references, never replace raw factors."""
    import math
    aliases=manifest.get('identity_aliases',{});wanted=set(codes);rows={};conflicts=set();sources={}
    for part in manifest.get('queries',{}).get('stk_factor',[]):
        day=part['params']['trade_date']
        if dates is not None and day not in dates:continue
        path=Path(part['path']);require(file_hash(path)==part['sha256'],'factor corroboration query changed')
        query=json.loads(path.read_text(encoding='utf-8'))
        require(query['identity']['api']=='stk_factor' and query['identity']['params']==part['params'] and
            query['rows_hash']==digest(query['rows']),'factor corroboration identity changed')
        sources[day]=part['sha256']
        for row in query['rows']:
            require(row['trade_date']==day,'factor corroboration date changed')
            code=aliases.get(row['ts_code'],{}).get('stable_code',row['ts_code'])
            if code not in wanted:continue
            try:record={field:float(row[field]) for field in ('adj_factor','close','pre_close')}
            except (TypeError,ValueError,KeyError):continue
            if not all(math.isfinite(x) and x>0 for x in record.values()):continue
            key=(code,day)
            if key in rows and rows[key]!=record:conflicts.add(key)
            rows[key]=record
    for key in conflicts:rows.pop(key,None)
    return rows,{'query_hashes':sources,'conflicting_keys':len(conflicts),
        'scope':'same vendor second-table corroboration; not independent investment assurance'}


def load_table(root, filename, codes, dates=None):
    """Explicit cohort reads with whole-query overlays, duplicate conflicts left unavailable."""
    import pandas as pd
    import pyarrow.dataset as ds
    import pyarrow.parquet as pq
    from .history import normalize_price_frame
    root=Path(root);manifest=json.loads((root/'CATALOG.json').read_text(encoding='utf-8'))
    require(manifest['manifest_hash']==digest({k:v for k,v in manifest.items() if k!='manifest_hash'}),'catalog manifest mismatch')
    api=TABLE_APIS[filename];parts=manifest['tables'].get(api,[])
    aliases=manifest.get('identity_aliases',{})
    expanded=set(codes) | {old for old,row in aliases.items() if row['stable_code'] in codes}
    def stable(value):return aliases.get(value,{}).get('stable_code',value)
    frames=[]
    for part in parts:
        path=root/part['file']
        require(file_hash(path)==part['sha256'],'catalog partition changed')
        if pq.read_metadata(path).num_rows==0:continue
        dataset=ds.dataset(path,format='parquet')
        filt=ds.field('ts_code').isin(sorted(expanded))
        if dates is not None:filt=filt & ds.field('trade_date').isin(sorted(dates))
        frame=dataset.to_table(filter=filt).to_pandas()
        if not frame.empty:
            frame.ts_code=frame.ts_code.map(stable)
            frames.append(frame)
    columns=manifest['fields'][api].split(',')
    combined=pd.concat(frames,ignore_index=True) if frames else pd.DataFrame(columns=columns)
    # Successful date queries replace the entire dated base surface, including deleted vendor rows.
    # This is an explicit current-vintage research overlay, never a historical revision archive.
    for part in manifest['queries'].get(api,[]):
        params=part['params'];date=params.get('trade_date')
        if date and dates is not None and date not in dates:continue
        path=Path(part['path']);require(file_hash(path)==part['sha256'],'supplement query changed')
        record=json.loads(path.read_text(encoding='utf-8'))
        require(record['rows_hash']==digest(record['rows']),'supplement rows changed')
        if date: combined=combined[combined.trade_date!=date]
        elif params.get('ts_code') and params.get('start_date') and params.get('end_date'):
            code=stable(params['ts_code'])
            combined=combined[~((combined.ts_code==code)&combined.trade_date.between(params['start_date'],params['end_date']))]
        elif api=='suspend_d':
            combined=combined[~combined.trade_date.between(params['start_date'],params['end_date'])]
        rows=[dict(r,ts_code=stable(r['ts_code'])) for r in record['rows'] if r['ts_code'] in expanded and (dates is None or r.get('trade_date') in dates)]
        if rows:combined=pd.concat([combined,pd.DataFrame(rows)],ignore_index=True)
    combined=combined.astype(object).where(lambda x:x.notna(),None)
    if api=='daily':combined=normalize_price_frame(combined)
    if api!='suspend_d':
        # The audit and native adapter must agree that "1" and "1.0" are equal.
        for column in columns:
            if column not in ('ts_code','trade_date') and column in combined:
                combined[column]=pd.to_numeric(combined[column],errors='coerce')
    # Exact duplicate records may collapse. Conflicting same-key rows must not select first/last.
    if api!='suspend_d' and not combined.empty:
        combined=combined.drop_duplicates(subset=[c for c in columns if c in combined])
        if api=='daily_basic':
            numeric=[c for c in columns if c not in ('ts_code','trade_date')]
            groups=combined.groupby(['ts_code','trade_date'],sort=False,dropna=False)
            differing=groups[numeric].nunique(dropna=True).gt(1).any(axis=1)
            combined=groups[numeric].first().reset_index()
            bad=differing.to_numpy(dtype=bool)
            combined.loc[bad,numeric]=None
            combined.loc[bad,'field_contract_status']='conflicting_duplicate_unavailable'
        conflict=combined.duplicated(['ts_code','trade_date'],keep=False)
        if conflict.any():
            collapsed=[]
            for (code,day),group in combined[conflict].groupby(['ts_code','trade_date']):
                collapsed.append({'ts_code':code,'trade_date':day,'field_contract_status':'conflicting_duplicate_unavailable'})
            combined=pd.concat([combined[~conflict],pd.DataFrame(collapsed)],ignore_index=True)
    return combined.astype(object).where(lambda x:x.notna(),None)


def build_catalog(raw_root, queries_root, destination, fields):
    """Stream the base files into resume-safe partitions; supplements remain separate facts."""
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    root,out,queries=Path(raw_root),Path(destination),Path(queries_root)
    out.mkdir(parents=True,exist_ok=True)
    tables,sources,index={},{},{}
    for filename,api in TABLE_APIS.items():
        source=root/filename;source_hash=file_hash(source);before=(source.stat().st_size,source.stat().st_mtime_ns)
        parts=[];header=pd.read_csv(source,nrows=0).columns
        columns=[x for x in fields[api].split(',') if x in header]
        for number,frame in enumerate(pd.read_csv(source,usecols=columns,dtype=str,chunksize=250000)):
            frame.trade_date=frame.trade_date.str.replace('-','',regex=False)
            folder=out/api;folder.mkdir(exist_ok=True)
            target=folder/('part_%04d.parquet'%number);receipt=target.with_suffix('.json')
            if receipt.exists():
                saved=json.loads(receipt.read_text(encoding='utf-8'))
                require(saved['source_hash']==source_hash and saved['rows']==len(frame) and file_hash(target)==saved['sha256'],'catalog resume mismatch')
            else:
                schema=pa.schema([(c,pa.string()) for c in columns])
                expected=pa.Table.from_pandas(frame,schema=schema,preserve_index=False)
                if target.exists():
                    require(pq.read_table(target).equals(expected,check_metadata=False),'unsealed partition conflicts with unchanged source; retain for review')
                else:
                    staging=target.with_suffix('.next.parquet')
                    require(not staging.exists(),'incomplete catalog staging; retain for review')
                    pq.write_table(expected,staging,compression='zstd');staging.replace(target)
                saved={'file':str(target.relative_to(out)),'sha256':file_hash(target),'rows':len(frame),'source_hash':source_hash}
                receipt.write_text(json.dumps(saved),encoding='utf-8')
            parts.append(saved)
        require((source.stat().st_size,source.stat().st_mtime_ns)==before,'raw input changed during catalog build')
        tables[api]=parts;sources[api]={'path':str(source),'sha256':source_hash}
        print(json.dumps({'catalog_api':api,'partitions':len(parts),'rows':sum(p['rows'] for p in parts)}),flush=True)
    for path in sorted(queries.glob('*.json')):
        record=json.loads(path.read_text(encoding='utf-8'));require(record['rows_hash']==digest(record['rows']),'query cache changed')
        identity=record['identity'];index.setdefault(identity['api'],[]).append({'path':str(path.resolve()),'sha256':file_hash(path),'params':identity['params'],
                  'observation_clock':max(x['retrieved_at'] for x in record['evidence'])})
    # Apply observations in retrieval order. Request hashes/page sizes do not
    # determine which whole-date vendor surface is the declared current vintage.
    for api in index:index[api].sort(key=lambda q:(q['observation_clock'],q['sha256']))
    from .a_share import identity_aliases
    mapping=[]
    for part in index.get('bse_mapping',[]):mapping.extend(json.loads(Path(part['path']).read_text(encoding='utf-8'))['rows'])
    manifest={'schema_version':'fuxi.market_catalog.1','tables':tables,'sources':sources,'queries':index,'fields':fields,
              'identity_aliases':identity_aliases(mapping),
              'vintage':'retrieved_now_for_research; historical_revision_archive_unverified','investment_data_assured':False}
    manifest['manifest_hash']=digest(manifest)
    target=out/'CATALOG.json';require(not target.exists(),'catalog already sealed')
    target.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return manifest
