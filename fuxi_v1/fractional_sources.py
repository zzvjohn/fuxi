"""A narrow issuer-clause proof for conditional floor/ceil entitlements."""
import re
from decimal import Decimal
from pathlib import Path
from urllib.parse import urlparse
from .contracts import require,digest
from .datasets import file_hash


def clause_for_term(text,term,index):
    slash_dates={f'{int(y):04d}-{int(m):02d}-{int(d):02d}' for y,m,d in
                 re.findall(r'(?<!\d)(20\d{2})\s*/\s*(\d{1,2})\s*/\s*(\d{1,2})(?!\d)',text)}
    compact=re.sub(r'\s+','',text).replace('，',',').replace('（','(').replace('）',')')
    code=term['instrument'].split('.')[0]
    if not re.search(r'证券代码[:：]'+re.escape(code)+r'(?!\d)',compact):return None
    ann=index.get('ann_date','')
    if len(ann)!=8 or ann[:4]+'-'+ann[4:6]+'-'+ann[6:]!=term['ann_date']:return None
    for key in ('record_date','ex_date','share_credit_date'):
        y,m,d=map(int,term[key].split('-'))
        if term[key] not in slash_dates and not re.search(rf'{y}年0?{m}月0?{d}日',compact):return None
    rates={Decimal(x) for x in re.findall(r'每股.{0,8}?转增([0-9]+(?:\.[0-9]+)?)股',compact)}
    rates.update(Decimal(x)/10 for x in re.findall(r'每10股.{0,20}?转增([0-9]+(?:\.[0-9]+)?)股',compact))
    if Decimal(str(term['share_distribution'])) not in rates:return None
    for match in re.finditer(r'不足1股',compact):
        clause=compact[max(0,match.start()-90):match.end()+420]
        if ('排序' in clause or '顺序排列' in clause) and re.search(r'(?:派发|分配)1股',clause):
            return clause
    return None


def verified_rule(term,source):
    require(term['status']=='ordinary_distribution_supported' and term['share_distribution']>0,'not ordinary holder shares')
    host=urlparse(source['url']).hostname
    require(host and any(host==h or host.endswith('.'+h) for h in ('cninfo.com.cn','sse.com.cn','szse.cn','bse.cn')),'unrecognized primary issuer host')
    require(file_hash(source['path'])==source['sha256'] and file_hash(source['text_path'])==source['text_sha256'],'fractional archive changed')
    import fitz
    with fitz.open(source['path']) as document:extracted='\n\n'.join(p.get_text() for p in document)
    text=Path(source['text_path']).read_text(encoding='utf-8')
    require(text==extracted,'fractional archive text does not match primary PDF')
    index=source.get('index_record') or source.get('publication_index')
    require(index is not None,'primary publication index missing')
    clause=clause_for_term(text,term,index)
    if clause is None:return None
    return {'term_hash':digest(term),'outcomes':'floor_or_ceil','source_scope':'archived_primary_clause',
            'source':{k:source[k] for k in ('url','path','sha256','text_path','text_sha256')},
            'clause':clause,'instrument':term['instrument'],'ex_date':term['ex_date'],
            'actual_allocation_observed':False,'global_other_holder_constraints_observed':False}
