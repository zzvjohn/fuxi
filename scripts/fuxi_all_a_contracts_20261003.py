"""Ten non-economic account contract rounds across every observed A-share identity."""
from collections import Counter
from decimal import Decimal
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from fuxi_v1.account import Account
from fuxi_v1.a_share import board,rule_for,buy_quantity,sell_quantity
from fuxi_v1.contracts import digest,require,ContractError
from fuxi_v1.datasets import file_hash
from fuxi_v1.runtime import controlled_writer


def immutable_json(path,body):
    if path.exists():require(digest(json.loads(path.read_text(encoding='utf-8')))==digest(body),'contract resume identity changed')
    else:
        with path.open('x',encoding='utf-8') as handle:json.dump(body,handle,ensure_ascii=False,indent=2)


def main():
    campaign=ROOT/'reports/fuxi_v1_all_a_20261003'
    output=campaign/'contracts_v3';output.mkdir(exist_ok=True)
    coverage=json.loads((campaign/'LOCAL_COVERAGE.json').read_text(encoding='utf-8'))
    codes=coverage['codes'];require(codes and len(codes)==len(set(codes)),'real nonempty master required')
    results=[]
    for round_id in range(1,11):
        passed=[]
        for code in codes:
            rule=rule_for(code);q=buy_quantity(rule['min_buy']+round_id,rule)
            require(q>=rule['min_buy'],'invalid board buy')
            require(buy_quantity(rule['max_limit_order']+1000000,rule)==rule['max_limit_order'],'maximum order not capped')
            require(sell_quantity(rule['max_limit_order']+1000000,rule['max_limit_order']+1000000,
                rule['max_limit_order']+1000000,rule['max_limit_order']+1000000,rule)==rule['max_limit_order'],'maximum sell not capped')
            require(buy_quantity(rule['max_market_order']+1000000,rule,order_type='market')==rule['max_market_order'],'market buy not capped')
            require(sell_quantity(rule['max_market_order']+1000000,rule['max_market_order']+1000000,
                rule['max_market_order']+1000000,rule['max_market_order']+1000000,rule,order_type='market')==rule['max_market_order'],'market sell not capped')
            account=Account('1000000')
            def event(kind,key,when,payload):
                return {'event_id':str(round_id)+':'+code+':'+key,'kind':kind,'timestamp':when,
                        'evidence_id':'synthetic_contract_fixture_'+code,'payload':payload}
            bought=event('fill','buy','2026-09-28T09:30:00+08:00',{'instrument':code,'side':'buy','quantity':q,
                          'price':'10','fee':'5','sellable_at':'2026-09-29T09:30:00+08:00'})
            account.apply(bought);require(account.apply(bought)=='already_applied','duplicate buy not deduped')
            require(sell_quantity(q,q,0,q,rule)==0,'unsettled shares sold')
            marked=account.mark('2026-09-28T18:00:00+08:00',{code:{'price':'10','asof':'2026-09-28T18:00:00+08:00','evidence_id':'fixture'}})
            require(marked['nav']=='999995' and marked['holdings'][code]==q,'cash/share/NAV mismatch')
            # A failed/missing quote remains unknown, without dropping the holding.
            missing=account.mark('2026-09-29T08:00:00+08:00',{})
            require(missing['nav'] is None and missing['holdings'][code]==q,'missing mark erased holding')
            account.apply(event('receivable','entitlement','2026-09-29T09:00:00+08:00',{'entitlement_id':code+':cash','amount':'10'}))
            sold=sell_quantity(q,q,q,q,rule);require(sold==q,'whole exit lost odd residual')
            account.apply(event('fill','sell','2026-09-29T09:30:00+08:00',{'instrument':code,'side':'sell','quantity':sold,
                'price':'10','fee':'5','sellable_at':'2026-09-29T09:30:00+08:00'}))
            before_payment=account.mark('2026-09-29T18:00:00+08:00',{})
            require(Decimal(before_payment['nav'])==Decimal('1000000') and before_payment['receivables']=='10','exit lost receivable')
            account.apply(event('payment','payment','2026-09-30T18:00:00+08:00',{'entitlement_id':code+':cash'}))
            last=account.mark('2026-09-30T18:00:00+08:00',{})
            require(last['holdings']=={} and last['cash']=='1000000' and last['nav']=='1000000','final cash mismatch')
            passed.append({'instrument':code,'board':board(code),'quantity':q,'rule_hash':digest(rule),'journal_hash':account.journal_hash})
        result={'round':round_id,'instruments':len(codes),'passed':len(passed),'failures':0,'rows_hash':digest(passed),
                'scope':'synthetic account contracts using real master identifiers; no market-data/account/platform certification'}
        path=output/('CONTRACT_ROUND_%02d.json'%round_id)
        immutable_json(path,{'summary':result,'rows':passed})
        results.append(result);print(json.dumps(result),flush=True)
    immutable_json(output/'TEN_ROUND_CONTRACTS.json',{'rounds':results,'master_sha256':file_hash(campaign/'SECURITY_MASTER.json'),
        'runner_sha256':file_hash(Path(__file__)),
        'code_hashes':{p.name:file_hash(p) for p in (ROOT/'fuxi_v1').glob('*.py')},
        'rules_hash':digest({c:rule_for(c) for c in codes}),
        'round_artifact_hashes':{str(n):file_hash(output/('CONTRACT_ROUND_%02d.json'%n)) for n in range(1,11)},
        'instrument_rounds':10*len(codes),'actual_market_account_certifications':0,'observed_platform_runs':0,'live_orders':0})


if __name__=='__main__':
    with controlled_writer(ROOT/'reports/fuxi_v1_all_a_20261003'):main()
