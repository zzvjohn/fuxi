"""Second immutable event assembly: unique contracts and issuer exceptions."""
import json
import math
import sys
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from fuxi_execution_lab_20260923 import read, write, digest
from fuxi_distribution_evidence_20260923 import normalize_issuer_event, checked_distribution
from fuxi_event_evidence_20260923 import checked_cash_action

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'research/factor_alchemy'))
from portfolio_execution_ledger import CorporateAction
OUT=ROOT/'reports/window_evidence_20260923'


def unique_contract(rows):
    """Prove one complete contract; no arbitrary choice of conflicting rows."""
    originals={json.dumps(r,sort_keys=True,ensure_ascii=False) for r in rows}
    distinct=[json.loads(s) for s in sorted(originals)]
    if len(distinct)==1:return distinct[0],[]
    projected={json.dumps({k:v for k,v in r.items() if k!='ann_date'},sort_keys=True,ensure_ascii=False) for r in distinct}
    if len(projected)!=1:raise ValueError('conflicting_same_day_contract')
    contract=json.loads(projected.pop())
    implementation=datetime.strptime(contract['imp_ann_date'],'%Y%m%d')
    announcements=sorted({r['ann_date'] for r in distinct})
    if any(datetime.strptime(a,'%Y%m%d')>implementation for a in announcements):
        raise ValueError('announcement_after_implementation')
    # This canonical row records implementation-time availability, not an arbitrary
    # earlier plan announcement. Original records remain separately retained.
    return {**contract,'ann_date':contract['imp_ann_date']},announcements


def differential_cash(item, proof, record_close, ex_reference, evidence):
    for field in ['ts_code','record_date','ex_date','pay_date','imp_ann_date']:
        if item[field]!=proof['vendor_contract'][field]:raise ValueError('issuer_identity_dates_mismatch')
    if item['div_proc']!='实施' or float(item['stk_div'])!=0:raise ValueError('not_cash_implementation')
    gross=float(item['cash_div_tax']);net=float(item['cash_div'])
    total=int(proof['total_shares']);eligible=int(proof['eligible_shares']);excluded=int(proof['repurchased_shares'])
    deduction=float(proof['ex_reference_deduction_per_share'])
    if not all(math.isfinite(v) for v in [gross,net,deduction,record_close,ex_reference]):raise ValueError('nonfinite')
    if not 0<=net<=gross or not math.isclose(gross,proof['gross_cash_per_share'],abs_tol=1e-10,rel_tol=0):
        raise ValueError('issuer_cash_mismatch')
    if not 0<eligible<total or total-eligible!=excluded:raise ValueError('issuer_share_counts_mismatch')
    if not math.isclose(eligible*gross,proof['total_cash'],abs_tol=.011,rel_tol=0):raise ValueError('issuer_total_cash_mismatch')
    if not math.isclose(eligible*gross/total,deduction,abs_tol=1e-7,rel_tol=0):raise ValueError('issuer_reference_formula_mismatch')
    if not math.isclose(record_close-deduction,ex_reference,abs_tol=.011,rel_tol=0):raise ValueError('raw_reference_mismatch')
    def iso(x):return datetime.strptime(x,'%Y%m%d').date().isoformat()
    ann,record,day,pay=[iso(item[k]) for k in ['imp_ann_date','record_date','ex_date','pay_date']]
    if not ann<=record<day<=pay:raise ValueError('invalid_dates')
    return CorporateAction('cash_'+item['ts_code']+'_'+day,item['ts_code'],record,day,pay,
                           cash_per_share_net=net,evidence=evidence)


def main():
    import pandas as pd
    target=OUT/'followup_assembly.json'
    assert not target.exists()
    audit=read(OUT/'request_audit.json')
    assert not audit['invalid_requests']
    groups=defaultdict(list)
    source_hashes={}
    for source in audit['rows']:
        path=OUT/'raw'/(source['id']+'.json')
        assert digest(path)==source['sha256']
        source_hashes[source['id']]=source['sha256']
        if source['api']=='dividend':
            for r in read(path)['records']:
                if (r['div_proc']=='实施' and '20241230'<=str(r.get('ex_date'))<='20260909'
                        and '20241230'<=str(r.get('record_date'))<='20260909'):
                    groups[(r['ts_code'],r['ex_date'])].append(r)
    proofs={};proof_hashes={}
    for name in ['issuer_distributions.json','followup_proofs/issuer_300969_20250507.json']:
        path=OUT/name;proof_hashes[name]=digest(path)
        for p in read(path)['events']:
            assert digest(ROOT/p['source_file'])==p['source_sha256'].lower()
            assert digest(ROOT/p['raw_request'])==p['raw_request_sha256'].lower()
            proofs[(p['code'],p['ex_date'].replace('-',''))]=p
    differential_path=OUT/'followup_proofs/300750_20250422_differential_cash_proof.json'
    original_proof=read(differential_path)
    for section in ['issuer_notice','raw_price_source','vendor_evidence']:
        source=original_proof[section]
        assert digest(Path(source['path']))==source['sha256'].lower()
    differential={'vendor_contract':original_proof['vendor_evidence']['record'],
        'source_sha256':original_proof['issuer_notice']['sha256'],
        'gross_cash_per_share':float(original_proof['documented_gross_cash_per_share']),
        'ex_reference_deduction_per_share':float(original_proof['documented_ex_reference_deduction_per_share']),
        'total_shares':original_proof['total_shares'],'eligible_shares':original_proof['eligible_shares'],
        'repurchased_shares':original_proof['repurchased_excluded_shares'],
        'total_cash':float(original_proof['documented_cash_total_cny'])}
    for vendor,documented in [('imp_ann_date','announcement'),('record_date','record'),('ex_date','ex'),('pay_date','payment')]:
        assert differential['vendor_contract'][vendor]==original_proof['dates'][documented].replace('-','')
    proof_hashes[differential_path.relative_to(OUT).as_posix()]=digest(differential_path)
    documentary=OUT/'followup_proofs/003035_issuer.pdf'
    proof_hashes[documentary.relative_to(OUT).as_posix()]=digest(documentary)
    codes={c for p in (ROOT/'reports/execution_followup_20260923/replay/targets').glob('*.json') for w in read(p)['targets'].values() if w for c in w}
    frames=[]
    for f in pd.read_csv(ROOT/'data/raw/daily_prices.csv',usecols=['ts_code','trade_date','close','pre_close'],dtype={'trade_date':str},chunksize=300000):
        f=f[f.ts_code.isin(codes)].copy()
        f['trade_date']=pd.to_datetime(f.trade_date,format='mixed').dt.strftime('%Y%m%d')
        frames.append(f[f.trade_date.between('20241220','20260909')])
    quotes=pd.concat(frames).set_index(['trade_date','ts_code']);assert quotes.index.is_unique
    actions=[];rejected=[];collapsed=[]
    for key,rows in sorted(groups.items()):
        code,date=key
        if code not in codes:continue
        try:
            r,anns=unique_contract(rows)
            if anns:collapsed.append({'code':code,'date':date,'source_ann_dates':anns,'canonical_ann_date':r['ann_date']})
            record_close=float(quotes.loc[(r['record_date'],code),'close'])
            ex_reference=float(quotes.loc[(date,code),'pre_close'])
            day=datetime.strptime(date,'%Y%m%d').date().isoformat()
            evidence=digest(OUT/'request_audit.json')+':unique_contract'
            if key==('300750.SZ','20250422'):
                action=differential_cash(r,differential,record_close,ex_reference,evidence+':issuer:'+differential['source_sha256'].lower())
            elif float(r['stk_div'])==0:
                action=checked_cash_action(r,code,day,record_close,ex_reference,evidence)
            else:
                if key in proofs:
                    r=normalize_issuer_event(r,proofs[key]);evidence+=':issuer:'+proofs[key]['source_sha256'].lower()
                action=checked_distribution(r,code,day,record_close,ex_reference,evidence)
            actions.append(asdict(action))
        except (ValueError,TypeError,KeyError) as exc:
            rejected.append({'code':code,'date':date,'reason':str(exc)})
    assert len(actions)>0 and len({(a['code'],a['ex_date']) for a in actions})==len(actions)
    prior=read(OUT/'assembly_distribution.json')
    by_key={(a['code'],a['ex_date']):a for a in actions}
    for a in prior['approved_actions']:
        b=by_key[(a['code'],a['ex_date'])]
        assert {k:v for k,v in a.items() if k!='evidence'}=={k:v for k,v in b.items() if k!='evidence'}
    write(target,{'approved_actions':actions,'approved_cash_actions':[a for a in actions if a['share_multiplier']==1],
          'approved_share_actions':[a for a in actions if a['share_multiplier']>1],
          'rejected_events':rejected,'request_gaps':prior['request_gaps'],
          'collapsed_unique_contracts':collapsed,'source_sha256':source_hashes,'proof_sha256':proof_hashes,
          'prior_assembly_sha256':digest(OUT/'assembly_distribution.json'),'code_sha256':digest(Path(__file__)),
          'formal_eligibility':False,'scope':'identical implementation contracts except earlier plan-announcement dates; one issuer differential cash reference; third issuer share normalization'})
    print('approved',len(actions),'collapsed',len(collapsed),'rejected',len(rejected),'prior preserved',len(prior['approved_actions']))


if __name__=='__main__':main()
