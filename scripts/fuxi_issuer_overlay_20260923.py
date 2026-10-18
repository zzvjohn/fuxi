"""Three additional issuer-proved share events; sealed prior assemblies retained."""
import math
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from fuxi_execution_lab_20260923 import read,write,digest
from fuxi_window_followup_evidence_20260923 import unique_contract

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'research/factor_alchemy'))
from portfolio_execution_ledger import CorporateAction
OUT=ROOT/'reports/window_evidence_20260923'


def issuer_share(item,proof,record_close,ex_reference,evidence):
    for field,pfield in [('ts_code','code'),('ex_date','ex_date'),('record_date','record_date'),
             ('pay_date','pay_date'),('div_listdate','div_listdate'),('imp_ann_date','imp_ann_date')]:
        if str(item.get(field,'')).replace('-','')!=proof[pfield].replace('-',''):
            raise ValueError('issuer_identity_or_date_mismatch')
    gross=float(item['cash_div_tax']);net=float(item['cash_div']);bonus=float(item['stk_div'])
    co=float(proof['capital_transfer_per_share']);bo=float(proof['bonus_share_per_share'])
    if (item['div_proc']!='实施' or not evidence or
        not all(math.isfinite(v) for v in [gross,net,bonus,co,bo,record_close,ex_reference]) or
        not 0<=net<=gross or gross<=0 or min(co,bo)<0 or bonus<=0 or record_close<=gross or ex_reference<=0):
        raise ValueError('invalid_issuer_distribution')
    if not math.isclose(gross,proof['cash_gross_per_share'],abs_tol=1e-8,rel_tol=0) or not math.isclose(bonus,co+bo,abs_tol=1e-8,rel_tol=0):
        raise ValueError('issuer_distribution_mismatch')
    missing_field=proof['normalization']['field']
    if missing_field not in ['stk_bo_rate','stk_co_rate'] or proof['normalization']['to']!=0:
        raise ValueError('unsupported_normalization')
    for field,wanted in [('stk_co_rate',co),('stk_bo_rate',bo)]:
        value=item.get(field)
        if value is None:
            if field!=missing_field or wanted!=0:raise ValueError('unproven_missing_component')
        elif not math.isfinite(float(value)) or not math.isclose(float(value),wanted,abs_tol=1e-8,rel_tol=0):
            raise ValueError('issuer_component_conflict')
    def iso(x):return datetime.strptime(x.replace('-',''),'%Y%m%d').date().isoformat()
    ann,record,day,pay,listed=[iso(proof[k]) for k in ['imp_ann_date','record_date','ex_date','pay_date','div_listdate']]
    if not ann<=record<day<=pay or listed<day:raise ValueError('invalid_dates')
    diff=proof.get('differential_ex_reference',{})
    if diff.get('required'):
        total=diff['total_shares'];eligible=diff['participating_shares'];excluded=diff['treasury_excluded_shares']
        cash=diff['reference_cash_per_share'];mult=diff['reference_share_multiplier']
        if (not 0<eligible<total or total-eligible!=excluded or
            not math.isclose(eligible*gross,diff['cash_total'],abs_tol=.011,rel_tol=0) or
            not math.isclose(eligible*bonus,diff['transfer_shares_total'],abs_tol=1e-6,rel_tol=0) or
            not math.isclose(gross*eligible/total,cash,abs_tol=1e-7,rel_tol=0) or
            not math.isclose(1+bonus*eligible/total,mult,abs_tol=1e-7,rel_tol=0)):
            raise ValueError('issuer_differential_formula_mismatch')
        expected=(record_close-cash)/mult
    else:expected=(record_close-gross)/(1+bonus)
    if not math.isclose(expected,ex_reference,abs_tol=.011,rel_tol=0):raise ValueError('raw_reference_mismatch')
    return CorporateAction('distribution_'+proof['code']+'_'+day,proof['code'],record,day,pay,
            cash_per_share_net=net,share_multiplier=1+bonus,new_shares_date=listed,evidence=evidence)


def main():
    import pandas as pd
    target=OUT/'issuer_overlay_assembly.json';assert not target.exists()
    files=['issuer_605377_20250604.json','issuer_301023_20250523.json','issuer_600694_20250610.json']
    proofs=[];hashes={}
    for name in files:
        path=OUT/'followup_proofs'/name;hashes[path.relative_to(OUT).as_posix()]=digest(path)
        for p in read(path)['events']:
            assert digest(ROOT/p['source_file'])==p['source_sha256'].lower()
            assert digest(ROOT/p['raw_request'])==p['raw_request_sha256'].lower()
            proofs.append(p)
    codes={p['code'] for p in proofs};frames=[]
    for f in pd.read_csv(ROOT/'data/raw/daily_prices.csv',usecols=['ts_code','trade_date','close','pre_close'],dtype={'trade_date':str},chunksize=300000):
        f=f[f.ts_code.isin(codes)].copy();f['trade_date']=pd.to_datetime(f.trade_date,format='mixed').dt.strftime('%Y-%m-%d');frames.append(f)
    quotes=pd.concat(frames).set_index(['trade_date','ts_code']);assert quotes.index.is_unique
    added=[]
    for p in proofs:
        rows=[r for r in read(ROOT/p['raw_request'])['records'] if r['ts_code']==p['code'] and r['ex_date']==p['ex_date'].replace('-','')]
        r,_=unique_contract(rows)
        a=issuer_share(r,p,float(quotes.loc[(p['record_date'],p['code']),'close']),
             float(quotes.loc[(p['ex_date'],p['code']),'pre_close']),'issuer:'+p['source_sha256'].lower())
        added.append(asdict(a))
    prior=read(OUT/'followup_assembly.json');actions=prior['approved_actions']+added
    assert len({(a['code'],a['ex_date']) for a in actions})==len(actions)
    write(target,{**prior,'approved_actions':actions,
        'approved_cash_actions':[a for a in actions if a['share_multiplier']==1],
        'approved_share_actions':[a for a in actions if a['share_multiplier']>1],
        'additional_issuer_actions':added,'additional_proof_sha256':hashes,
        'parent_assembly_sha256':digest(OUT/'followup_assembly.json'),
        'code_sha256':digest(Path(__file__)),
        'note':'Parent rejected_events retained as historical inventory; these three exact keys are now resolved. Tax fields are initial-credit convention, not final investor-specific net tax.'})
    print('added',len(added),'cash',len(prior['approved_cash_actions']),'share',len(prior['approved_share_actions'])+len(added))


if __name__=='__main__':main()
