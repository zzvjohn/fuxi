"""Archived issuer halt dates; interval intersections are research valuations."""
from pathlib import Path
import re
from .contracts import require, load_json, digest
from .datasets import file_hash
from .history import iso, jq_code


def partial_halt_intervals(records,instrument,session):
    """Exact vendor intraday records, with some regular trading time remaining."""
    if not records:return None
    intervals=[]
    for row in records:
        if jq_code(row.get('ts_code',''))!=instrument or iso(row.get('trade_date'))!=session:return None
        if row.get('suspend_type') not in {'S','R'}:return None
        if row['suspend_type']=='R':continue
        text=row.get('suspend_timing')
        if not isinstance(text,str) or not text:return None
        for token in text.split(','):
            # Vendor records omit leading zeros on single-digit hours ('9:30-9:40');
            # range and ordering checks below reject any malformed span regardless.
            match=re.fullmatch(r'(\d{1,2}):(\d{1,2})-(\d{1,2}):(\d{1,2})',token.strip())
            if not match:return None
            h0,m0,h1,m1=map(int,match.groups())
            if max(h0,h1)>23 or max(m0,m1)>59:return None
            left,right=h0*60+m0,h1*60+m1
            if left>=right:return None
            intervals.append((left,right))
    if not intervals:return None
    trading=set(range(570,690))|set(range(780,900))
    stopped={minute for left,right in intervals for minute in range(left,right)}
    if not (trading&stopped) or not (trading-stopped):return None
    return [list(interval) for interval in sorted(intervals)]


def partial_halt_mark(records,instrument,session,close,volume,source_hash,price_hash):
    """Observed daily marks do not make a halted session eligible for new orders."""
    from .contracts import finite
    intervals=partial_halt_intervals(records,instrument,session)
    if intervals is None or not all(finite(v) and v>0 for v in (close,volume)):return None
    if not source_hash or not price_hash:return None
    return {'schema_version':'fuxi.vendor_partial_halt_mark.1','instrument':instrument,'session':session,
        'records':records,'records_hash':digest(records),'intervals':intervals,
        'suspend_source_sha256':source_hash,'price_source_sha256':price_hash,
        'observed_close':close,'observed_volume':volume,
        'scope':'source-bound daily observed mark on partial halt; conservative order refusal retained; not broker or independent PIT assurance'}


def valid_partial_halt_mark(bar,snapshot):
    """Recompute timing and bind observed marks to the frozen raw columns."""
    proof=bar.get('partial_halt_mark') or {}
    if proof.get('schema_version')!='fuxi.vendor_partial_halt_mark.1' or bar.get('paused') is not True:return False
    if bar.get('mark_source')!='observed_close_on_vendor_partial_halt':return False
    day,code=bar['session'],bar['instrument']
    if proof.get('session')!=day or proof.get('instrument')!=code:return False
    records=proof.get('records') or []
    expected=partial_halt_mark(records,code,day,bar.get('close'),bar.get('volume'),
        proof.get('suspend_source_sha256'),proof.get('price_source_sha256'))
    if expected is None or expected!=proof:return False
    sources=snapshot.get('provenance',{}).get('source_hashes',{})
    if sources.get('suspend_d.csv')!=proof['suspend_source_sha256'] or sources.get('daily_prices.csv')!=proof['price_source_sha256']:return False
    if day not in snapshot.get('sessions',[]) or code not in snapshot.get('instruments',[]):return False
    i,j=snapshot['sessions'].index(day),snapshot['instruments'].index(code)
    if snapshot.get("schema_version") == "fuxi.columnar_snapshot.1":
        from .columnar import read_array
        columns = {}
        try:
            refs = snapshot["arrays"]["fields"]
            shape=(len(snapshot["sessions"]),len(snapshot["instruments"]))
            for field in ("close", "volume"):
                columns[field]=read_array(refs[field],shape,"f")
            return float(columns["close"][i,j])==bar["close"] and float(columns["volume"][i,j])==bar["volume"]
        except (KeyError, IndexError, ValueError):
            return False
        finally:
            for values in columns.values(): values._mmap.close()
    columns=snapshot.get('columns',{})
    try:return columns['close'][i][j]==bar['close'] and columns['volume'][i][j]==bar['volume']
    except (KeyError,IndexError):return False


def extract_full_halt_start(text, index, instrument, session):
    compact=re.sub(r'\s+', '', text).replace('（','(').replace('）',')')
    require(index.get('ts_code','').endswith(('.SH','.SZ')), 'unsupported halt issuer mapping')
    require(jq_code(index['ts_code'])==instrument, 'halt issuer mismatch')
    require(re.search(r'证券代码[:：]?'+instrument.split('.')[0]+r'(?!\d)',compact[:700]), 'halt header mismatch')
    require(iso(index.get('ann_date')) and iso(index['ann_date'])<=session, 'halt publication after session')
    require(not any(w in compact[:700] for w in ('更正公告','撤销停牌','取消停牌')), 'halt correction requires separate adjudication')
    dates={f'{int(y):04d}-{int(m):02d}-{int(d):02d}' for y,m,d in re.findall(
        r'自(20\d{2})年(\d{1,2})月(\d{1,2})日(?:\(星期[一二三四五六日天]\))?开市起(?:开始)?停牌',compact)}
    require(dates=={session}, 'no unique explicit full-halt start on this session')
    return {'instrument':instrument,'session':session,'ann_date':iso(index['ann_date']),
            'scope':'one issuer-explicit open-of-session halt start; no inferred subsequent dates',
            'independent_assurance':False}


def load_halt_starts(evidence, instruments, sessions):
    if not evidence:return {}
    require(file_hash(evidence['path'])==evidence['sha256'], 'halt bundle changed')
    bundle=load_json(evidence['path'])
    require(bundle.get('schema_version') in {'fuxi.public_full_halt_bundle.1','fuxi.public_closed_halt_bundle.1'} and
            bundle.get('independent_assurance') is False, 'halt bundle contract')
    starts={}
    for row in bundle['rows']:
        if bundle['schema_version']=='fuxi.public_closed_halt_bundle.1':
            proof=extract_closed_halt(_verified_text(row['start_source']),_verified_text(row['resumption_source']),
                row['start_source']['index_record'],row['resumption_source']['index_record'],row['instrument'])
            dates=[d for d in sessions if proof['interval_start']<=d<=proof['interval_end']]
        else:
            proof=extract_full_halt_start(_verified_text(row),row['index_record'],row['instrument'],row['session'])
            dates=[row['session']] if row['session'] in sessions else []
        for day in dates:
            key=(row['instrument'],day)
            require(key not in starts, 'duplicate issuer halt evidence')
            if key[0] in instruments:
                starts[key]=dict(proof,session=day,evidence_id=digest(row),source=dict(row))
    return starts


def _verified_text(source):
    require(file_hash(source['path'])==source['sha256'] and
            file_hash(source['text_path'])==source['text_sha256'], 'archived halt source changed')
    require(source['url'].startswith('https://') and
            source['index_record']['url'].replace('http://','https://')==source['url'], 'halt source URL mismatch')
    text=Path(source['text_path']).read_text(encoding='utf-8')
    from pypdf import PdfReader
    original='\n'.join(page.extract_text() or '' for page in PdfReader(source['path']).pages)
    require(re.sub(r'\s+','',original)==re.sub(r'\s+','',text), 'halt extraction differs from original PDF')
    return text


def extract_closed_halt(start_text, resume_text, start_index, resume_index, instrument):
    """Common interval, without resolving conflicting source start dates.

    This uses a later retrospective resumption document only for declared
    historical valuation. It supplies no alpha feature or independent PIT claim.
    An expected halt duration never supplies an end date.
    """
    compact=lambda t:re.sub(r'\s+','',t).replace('（','(').replace('）',')')
    a,b=compact(start_text),compact(resume_text)
    date_pattern=r'(20\d{2})年(\d{1,2})月(\d{1,2})日(?:\(星期[一二三四五六日天]\))?'
    def dates(text,pattern):
        from datetime import date
        return {date(int(y),int(m),int(d)).isoformat() for y,m,d in re.findall(pattern,text)}
    starts=dates(a,r'自'+date_pattern+r'开市起(?:开始)?停牌')
    recalled=dates(b,r'自'+date_pattern+r'开市起(?:开始)?停牌')
    resumes=dates(b,r'将于'+date_pattern+r'开市起复牌')
    require(len(starts)==len(recalled)==len(resumes)==1,'ambiguous closed halt dates')
    start,recall,resume=next(iter(starts)),next(iter(recalled)),next(iter(resumes))
    extract_full_halt_start(start_text,start_index,instrument,start)
    require(jq_code(resume_index['ts_code'])==instrument and
            re.search(r'证券代码[:：]?'+instrument.split('.')[0]+r'(?!\d)',b[:700]),'resumption issuer mismatch')
    require(iso(resume_index.get('ann_date')) and iso(resume_index['ann_date'])<=resume,'resumption publication after effective date')
    require(not any(w in b[:700] for w in ('更正公告','取消复牌','撤销复牌')),'resumption correction requires review')
    number=re.search(r'公告编号[:：](20\d{2}-\d+)',a[:700])
    require(number and re.search(r'公告编号[:：]'+re.escape(number.group(1)),b),'resumption does not reference same halt incident')
    ends=re.findall(r'复牌(20\d{2}/\d{1,2}/\d{1,2})(?=20\d{2}/)(20\d{2}/\d{1,2}/\d{1,2})',b)
    from datetime import date
    table={(date(*(int(x) for x in p.split('/'))).isoformat(),date(*(int(x) for x in q.split('/'))).isoformat()) for p,q in ends}
    require(len(table)==1 and next(iter(table))[1]==resume,'explicit termination/resumption table missing')
    end=next(iter(table))[0];common=max(start,recall)
    require(common<=end<resume,'non-overlapping source halt intervals')
    return {'instrument':instrument,'interval_start':common,'interval_end':end,'resume_session':resume,
        'original_start':start,'recalled_start':recall,'start_date_conflict':start!=recall,
        'start_ann_date':iso(start_index['ann_date']),'resume_ann_date':iso(resume_index['ann_date']),
        'scope':'source-interval intersection for declared historical valuation; retrospective end confirmation; raw quotes absent; no alpha or independent PIT assurance',
        'independent_assurance':False}
