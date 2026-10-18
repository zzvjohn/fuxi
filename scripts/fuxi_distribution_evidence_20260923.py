"""Explicit share-distribution evidence contract; no fractional-share invention."""
import sys
import math
from datetime import datetime
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'research/factor_alchemy'))
from portfolio_execution_ledger import CorporateAction
from fuxi_execution_lab_20260923 import read, write, digest
from fuxi_event_evidence_20260923 import checked_cash_action
OUT = ROOT / 'reports/window_evidence_20260923'


def normalize_issuer_event(item, proof):
    """Normalize one missing component only after exact issuer terms match."""
    for field, expected in [('ts_code',proof['code']),('ex_date',proof['ex_date']),
                            ('record_date',proof['record_date']),('pay_date',proof['pay_date']),
                            ('div_listdate',proof['div_listdate']),('imp_ann_date',proof['imp_ann_date'])]:
        if str(item.get(field,'')).replace('-','') != expected.replace('-',''):
            raise ValueError('issuer_vendor_date_or_identity_mismatch')
    for field, expected in [('cash_div_tax',proof['cash_gross_per_share']),
                            ('stk_co_rate',proof['capital_transfer_per_share']),
                            ('stk_div',proof['capital_transfer_per_share']+proof['bonus_share_per_share'])]:
        if not math.isclose(float(item[field]),expected,rel_tol=0,abs_tol=1e-8):
            raise ValueError('issuer_vendor_distribution_mismatch')
    if proof['normalization']['field'] != 'stk_bo_rate' or proof['bonus_share_per_share'] != 0:
        raise ValueError('unsupported_issuer_normalization')
    value=item.get('stk_bo_rate')
    if value is not None:
        value=float(value)
        if math.isinf(value) or (not math.isnan(value) and value!=0):
            raise ValueError('issuer_vendor_bonus_conflict')
    return {**item,'stk_bo_rate':0.0}


def checked_distribution(item, code, day, record_close, ex_reference, evidence, allocation_evidence=None):
    """Only implemented cash/bonus/transfer events with an observed ex reference."""
    def iso(v):
        return datetime.strptime(str(v), '%Y%m%d').date().isoformat()
    if not evidence or item.get('ts_code') != code or item.get('div_proc') != '实施' or iso(item['ex_date']) != day:
        raise ValueError('wrong_event_identity_or_status')
    bonus = float(item['stk_div'])
    if bonus == 0:
        return checked_cash_action(item, code, day, record_close, ex_reference, evidence)
    gross, net = float(item['cash_div_tax']), float(item['cash_div'])
    if not all(math.isfinite(v) for v in [bonus, gross, net, record_close, ex_reference]):
        raise ValueError('nonfinite_distribution')
    if bonus <= 0 or not 0 <= net <= gross or record_close <= gross or ex_reference <= 0:
        raise ValueError('unsupported_distribution_values')
    # Capital increases can be allocated to creditors/investors, not record-date
    # ordinary shareholders. Vendor stk_div alone cannot establish entitlement.
    if (code, day) in {('002310.SZ', '2024-12-30'), ('002217.SZ', '2024-12-31'),
                       ('002647.SZ', '2025-03-13')}:
        raise ValueError('restructuring_allocation_requires_dedicated_accounting')
    if gross == 0 and not allocation_evidence:
        raise ValueError('pure_stock_requires_issuer_record_holder_allocation_evidence')
    stock_bonus, capital_transfer = float(item['stk_bo_rate']), float(item['stk_co_rate'])
    if (not all(math.isfinite(v) and v >= 0 for v in [stock_bonus, capital_transfer])
            or not math.isclose(stock_bonus + capital_transfer, bonus, abs_tol=1e-8, rel_tol=0)):
        raise ValueError('unproven_stock_distribution_components')
    record, ann = iso(item['record_date']), iso(item['imp_ann_date'])
    listed = iso(item['div_listdate'])
    # Pure stock events have no cash payable: ex_date is only a zero-value schedule marker.
    payment = iso(item['pay_date']) if gross > 0 else day
    if not ann <= record < day <= payment or listed < day:
        raise ValueError('invalid_distribution_dates')
    expected = (record_close - gross) / (1 + bonus)
    if not math.isclose(expected, ex_reference, rel_tol=0, abs_tol=.011):
        raise ValueError('distribution_reference_mismatch')
    return CorporateAction('distribution_' + code + '_' + day, code, record, day, payment,
                           cash_per_share_net=net, share_multiplier=1 + bonus,
                           new_shares_date=listed, evidence=evidence)


def main():
    import pandas as pd
    assert not (OUT / 'assembly_distribution.json').exists()
    assembled = read(OUT / 'assembly.json')
    evidence = digest(OUT / 'dividend_overlay.csv')
    assert assembled['approved_cash_actions']
    assert all(a['evidence']==evidence for a in assembled['approved_cash_actions'])
    for request, expected in assembled['source_sha256'].items():
        assert digest(OUT/'raw'/(request+'.json'))==expected
    proofs={}
    proof_path=OUT/'issuer_distributions.json'
    if proof_path.exists():
        for proof in read(proof_path)['events']:
            assert digest(ROOT/proof['source_file'])==proof['source_sha256'].lower()
            assert digest(ROOT/proof['raw_request'])==proof['raw_request_sha256'].lower()
            proofs[(proof['code'],proof['ex_date'])]=proof
    records = pd.read_csv(OUT / 'dividend_overlay.csv', dtype={'ex_date': str, 'record_date': str,
                  'imp_ann_date': str, 'pay_date': str, 'div_listdate': str})
    records = records[records.div_proc.eq('实施') & (records.stk_div > 0)].copy()
    records['day'] = pd.to_datetime(records.ex_date, format='mixed', errors='coerce').dt.strftime('%Y-%m-%d')
    records['record'] = pd.to_datetime(records.record_date, format='mixed', errors='coerce').dt.strftime('%Y-%m-%d')
    bad_dates = records[records.day.isna() | records.record.isna()]
    invalid_dates = [{'code':r.ts_code,'ex_date':r.ex_date,'record_date':r.record_date,
                      'reason':'invalid_or_missing_event_date'} for r in bad_dates.itertuples()]
    records = records[records.day.between('2024-12-30', '2026-09-09') & records.record.ge('2024-12-30')]
    target_dir = ROOT / 'reports/execution_followup_20260923/replay/targets'
    codes = {c for p in target_dir.glob('*.json') for w in read(p)['targets'].values() if w for c in w}
    records = records[records.ts_code.isin(codes)]
    parts = []
    for part in pd.read_csv(ROOT / 'data/raw/daily_prices.csv', usecols=['ts_code','trade_date','close','pre_close'],
                            dtype={'trade_date': str}, chunksize=300000):
        part = part[part.ts_code.isin(set(records.ts_code))].copy()
        part['trade_date'] = pd.to_datetime(part.trade_date, format='mixed').dt.strftime('%Y-%m-%d')
        parts.append(part[part.trade_date.between('2024-12-30', '2026-09-09')])
    quotes = pd.concat(parts).set_index(['trade_date','ts_code'])
    assert quotes.index.is_unique
    ambiguous = records.duplicated(['ts_code', 'day'], keep=False)
    actions, rejected, normalizations = [], [], []
    for index, row in records.iterrows():
        try:
            if ambiguous.loc[index]:
                raise ValueError('ambiguous_same_day_distribution')
            item=row.to_dict()
            proof=proofs.get((row.ts_code,row.day))
            provenance=evidence
            if proof:
                item=normalize_issuer_event(item,proof)
                provenance+=':issuer:'+proof['source_sha256'].lower()
                normalizations.append({'code':row.ts_code,'date':row.day,'field':'stk_bo_rate',
                                       'value':0.,'source_sha256':proof['source_sha256'].lower()})
            action = checked_distribution(item, row.ts_code, row.day,
                      float(quotes.loc[(row.record,row.ts_code),'close']),
                      float(quotes.loc[(row.day,row.ts_code),'pre_close']), provenance)
            actions.append(asdict(action))
        except (ValueError, KeyError, TypeError) as exc:
            rejected.append({'code':row.ts_code,'date':row.day,'reason':str(exc)})
    combined = assembled['approved_cash_actions'] + actions
    assert len({(a['code'], a['ex_date']) for a in combined}) == len(combined)
    write(OUT / 'assembly_distribution.json', {'approved_actions':combined,
          'approved_cash_actions':assembled['approved_cash_actions'], 'approved_share_actions':actions,
          'rejected_share_actions':rejected, 'invalid_share_dates':invalid_dates,
          'issuer_normalizations':normalizations,
          'issuer_proof_sha256':digest(proof_path) if proofs else None,
          'request_gaps':assembled['request_gaps'],
          'rejected_events':assembled['rejected_events'],
          'source_assembly_sha256':digest(OUT/'assembly.json'),
          'adapter_sha256':digest(Path(__file__)), 'formal_eligibility':False,
          'limits':['Missing cash/stock components remain unknown; restructuring allocations not inferred',
                    'Fractional entitlements remain blocked by the ledger; no cash-in-lieu inferred',
                    'Share evidence acceptance is not independent execution-engine certification',
                    'Missing lifecycle valuation remains independent of these corporate actions']})
    print('cash',len(assembled['approved_cash_actions']),'share',len(actions),'rejected_share',len(rejected),flush=True)


if __name__ == '__main__':
    main()
