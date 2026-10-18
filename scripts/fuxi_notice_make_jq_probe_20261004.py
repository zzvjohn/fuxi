"""Create a read-only, user-submitted JQ coverage probe and verify its failure paths."""
import json,sys,hashlib,types
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from fuxi_v1.contracts import load_json,require
from fuxi_v1.datasets import file_hash
from fuxi_v1.history import iso
OUT=ROOT/'reports/fuxi_v1_announcements_20261003'
DEST=ROOT/'deliverables/fuxi_v1_20261004'
ASOF='2026-09-30'


RUNTIME='''
import json
import hashlib
import math
from collections import Counter


def initialize(context):
    g.fuxi_probe_done = False
    set_option('avoid_future_data', True)
    run_daily(capture_coverage, time='after_close')


def capture_coverage(context):
    if g.fuxi_probe_done:
        return
    if str(context.current_dt.date()) < SPEC['asof']:
        log.info('FUXI_PROBE_WAIT: configured date has not arrived')
        return
    encoded = json.dumps(SPEC, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode('utf-8')
    if hashlib.sha256(encoded).hexdigest() != SPEC_SHA256:
        raise ValueError('probe specification changed')
    listed = get_all_securities(types=['stock'], date=SPEC['asof'])
    buckets = {}
    for code in listed.index:
        buckets.setdefault(str(code).split('.')[0], []).append(str(code))
    rows = []
    for target in SPEC['targets']:
        hits = sorted(set(c for alias in target['prefix_aliases'] for c in buckets.get(alias, [])))
        status = 'unique_platform_mapping_observed' if len(hits) == 1 else 'platform_mapping_missing' if not hits else 'platform_mapping_ambiguous'
        rows.append(dict(target, platform_codes=hits, status=status))
    by_code = {r['instrument']:r for r in rows}
    quotes = []
    for instrument in SPEC['sample_instruments']:
        r = by_code[instrument]
        reading = {'instrument':instrument, 'platform_codes':r['platform_codes'], 'status':'mapping_not_unique'}
        if len(r['platform_codes']) == 1:
            try:
                frame = get_price(r['platform_codes'][0], start_date=SPEC['asof'], end_date=SPEC['asof'],
                    frequency='daily', fields=['close','volume'], skip_paused=False, fq=None)
                reading['rows'] = 0 if frame is None else len(frame)
                if frame is not None and len(frame) == 1:
                    close,volume = float(frame.iloc[0]['close']),float(frame.iloc[0]['volume'])
                    reading.update(close=close if math.isfinite(close) else None,
                                   volume=volume if math.isfinite(volume) else None,
                                   status='platform_daily_proxy_observed' if math.isfinite(close) and close>0 and math.isfinite(volume) and volume>=0 else 'platform_value_missing_or_invalid')
                else:
                    reading['status']='platform_daily_rows_missing_or_ambiguous'
            except Exception as exc:
                reading.update(status='platform_query_error',error=str(exc))
        quotes.append(reading)
    result={'schema_version':'fuxi.jq_readonly_coverage_probe.1','asof':SPEC['asof'],
        'platform_context_type':str(getattr(context.run_params,'type','unobserved')),
        'captured_at':str(context.current_dt), 'spec_sha256':SPEC_SHA256,
        'expected_live_instruments':len(rows),'mapping_counts':dict(Counter(r['status'] for r in rows)),
        'platform_stock_index_count':len(listed.index),'platform_suffix_counts':dict(Counter(str(c).split('.')[-1] for c in listed.index)),
        'rows':rows,'sample_quotes':quotes,'requested_price_calls_upper_bound':len(SPEC['sample_instruments']),
        'formal_assurance':False,'live_execution_allowed':False,
        'scope':'read-only provider coverage observations; no orders, factor admission or full price/PIT/account certification'}
    text=json.dumps(result,ensure_ascii=False,sort_keys=True)
    write_file('fuxi_jq_coverage_probe_20261004.json',text,append=False)
    log.info('FUXI_JQ_COVERAGE_PROBE '+json.dumps({k:result[k] for k in ('asof','expected_live_instruments','mapping_counts','platform_suffix_counts','spec_sha256')},ensure_ascii=False))
    g.fuxi_probe_done=True
'''


def main():
    master_path=ROOT/'reports/fuxi_v1_all_a_20261003/SECURITY_MASTER.json'
    catalog=ROOT/'data/fuxi_v1/supplements/20261003/catalog_v3/CATALOG.json'
    master=load_json(master_path)['rows'];aliases=load_json(catalog)['identity_aliases'];targets=[]
    for r in master:
        listed,ended=iso(r['list_date']),iso(r.get('delist_date'))
        if not listed or listed>ASOF or ended and ended<=ASOF:continue
        # A missing delisting clock cannot convert an already-delisted stock to live.
        if r.get('list_status')=='D' and not ended:continue
        stable=r['ts_code'];prefix={stable.split('.')[0]};dated=stable
        for old,relation in aliases.items():
            if relation['stable_code']==stable:
                prefix.add(old.split('.')[0]);dated=stable if ASOF>=relation['effective_session'] else old
        targets.append({'instrument':stable,'dated_venue_code':dated,'prefix_aliases':sorted(prefix)})
    targets.sort(key=lambda r:r['instrument']);require(len(targets)>0,'empty probe census')
    codes={r['instrument'] for r in targets};sample=[c for c in ['600000.SH','000001.SZ','300001.SZ','688001.SH','920000.BJ','920699.BJ'] if c in codes]
    require(len(sample)>=4 and any(c.endswith('.BJ') for c in sample),'probe lacks required board representatives')
    spec={'schema_version':'fuxi.jq_probe_spec.1','asof':ASOF,'targets':targets,'sample_instruments':sample,
        'master_sha256':file_hash(master_path),'mapping_catalog_sha256':file_hash(catalog),
        'selection':'all frozen ordinary A-share lifecycle members live on explicit date; unavailable provider names retained'}
    text=json.dumps(spec,sort_keys=True,separators=(',',':'),ensure_ascii=False)
    sha=hashlib.sha256(text.encode('utf-8')).hexdigest()
    source="# Read-only JQ strategy probe; user submits manually. No order APIs.\nimport json\nSPEC=json.loads("+repr(text)+")\nSPEC_SHA256="+repr(sha)+"\n"+RUNTIME
    compile(source,'jq_readonly_probe.py','exec')
    import pandas as pd
    checks=[]
    def fixture(index,quote_failure=False,day='2026-10-09'):
        exports=[];requests=[];env={'g':types.SimpleNamespace(),'set_option':lambda *a,**k:None,
            'run_daily':lambda *a,**k:None,'log':types.SimpleNamespace(info=lambda x:None),
            'get_all_securities':lambda **k:types.SimpleNamespace(index=index),'write_file':lambda p,t,**k:exports.append(json.loads(t))}
        def price(code,**kwargs):
            requests.append((code,kwargs))
            if quote_failure:raise RuntimeError('fixture unavailable')
            return pd.DataFrame([{'close':10.,'volume':10000.}])
        env['get_price']=price
        exec(compile(source,'jq_readonly_probe.py','exec'),env)
        ctx=types.SimpleNamespace(current_dt=datetime.fromisoformat(day),run_params=types.SimpleNamespace(type='fixture'))
        env['initialize'](ctx);env['capture_coverage'](ctx)
        return env,ctx,exports,requests
    env,ctx,export,calls=fixture(['600000.XSHG','920000.PROVIDER_OBSERVED_SUFFIX'])
    r=export[0];require(len(r['rows'])==len(targets) and sum(r['mapping_counts'].values())==len(targets),'missing targets silently removed')
    require(next(v for v in r['rows'] if v['instrument']=='920000.BJ')['platform_codes']==['920000.PROVIDER_OBSERVED_SUFFIX'],'provider suffix guessed instead of observed')
    count=len(calls);env['capture_coverage'](ctx);require(len(export)==1 and len(calls)==count,'probe duplicates calls')
    checks.append('nonempty full denominator, observed suffix, idempotent capture')
    _,_,export,calls=fixture(['600000.XSHG','600000.OTHER'],quote_failure=True)
    require(next(v for v in export[0]['rows'] if v['instrument']=='600000.SH')['status']=='platform_mapping_ambiguous' and not calls,'ambiguous mapping queried')
    _,_,export,_=fixture(['600000.XSHG'],quote_failure=True)
    require(export[0]['sample_quotes'][0]['status']=='platform_query_error','failed quote erased')
    checks.append('ambiguous mapping and query failure retained')
    _,_,export,calls=fixture(['600000.XSHG'],day='2026-09-29');require(not export and not calls,'future probe query')
    checks.append('pre-asof execution emits no future query')
    DEST.mkdir(parents=True,exist_ok=True);path=DEST/'jq_readonly_coverage_probe.py'
    with path.open('xb') as h:h.write(source.encode('utf-8'))
    receipt={'generated_at':datetime.now(timezone.utc).isoformat(),'path':str(path),'sha256':file_hash(path),
        'spec_sha256':sha,'expected_live_instruments':len(targets),'sample_count':len(sample),'local_fixture_checks':checks,
        'actual_platform_runs':0,'formal_assurance':False,'live_execution_allowed':False,'runner_sha256':file_hash(Path(__file__))}
    with (DEST/'PROBE_RECEIPT.json').open('x',encoding='utf-8') as h:json.dump(receipt,h,ensure_ascii=False,indent=2)
    print(json.dumps(receipt,ensure_ascii=False))


if __name__=='__main__':main()
