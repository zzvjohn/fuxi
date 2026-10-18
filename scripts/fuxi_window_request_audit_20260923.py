"""Verify frozen request identities and raw responses before event consumption."""
from pathlib import Path
from collections import Counter
from fuxi_execution_lab_20260923 import read, write, digest

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/window_evidence_20260923'


def inspect_response(request, raw):
    errors=[]
    if raw.get('request') != request:
        return ['request_identity_mismatch']
    if raw.get('status') != 'response_received' or not isinstance(raw.get('records'),list):
        return ['unusable_response']
    records=raw['records']
    if raw.get('possible_truncation') != (len(records)>=request['cap']):
        errors.append('inconsistent_truncation_flag')
    if len(records)>=request['cap']:
        errors.append('possible_truncation')
    params=request['params']
    for row in records:
        if not isinstance(row,dict) or not row.get('ts_code'):
            errors.append('missing_security_identity');continue
        if 'ts_code' in params and row['ts_code']!=params['ts_code']:
            errors.append('wrong_security')
        field='ex_date' if request['api']=='dividend' else 'trade_date'
        if field in params and str(row.get(field))!=params[field]:
            errors.append('wrong_requested_date')
        if 'start_date' in params and not params['start_date']<=str(row.get(field,''))<=params['end_date']:
            errors.append('outside_requested_range')
        if request['api']=='suspend_d' and row.get('suspend_type') not in ['S','R']:
            errors.append('unknown_suspend_type')
    return sorted(set(errors))


def main():
    assert not (OUT/'request_audit.json').exists()
    p=read(OUT/'protocol.json')
    audits=[]
    for req in p['requests']:
        path=OUT/'raw'/(req['id']+'.json')
        if not path.exists():
            audits.append({'id':req['id'],'errors':['missing_response']});continue
        raw=read(path)
        errors=inspect_response(req,raw)
        attempts=list((OUT/'attempts'/req['id']).glob('*.json'))
        if not 1<=len(attempts)<=p['max_attempts_per_request']:
            errors.append('attempt_budget_violation')
        if not any(read(a)==raw for a in attempts):
            errors.append('response_not_equal_to_saved_attempt')
        audits.append({'id':req['id'],'api':req['api'],'rows':len(raw.get('records',[])),
                       'errors':errors,'sha256':digest(path),'attempts':len(attempts)})
    bad=[r for r in audits if r['errors']]
    write(OUT/'request_audit.json',{'planned':len(p['requests']),'responses_checked':len(audits),
          'positive_response_rows':sum(r.get('rows',0) for r in audits),'invalid_requests':bad,
          'endpoint_counts':dict(Counter(r.get('api','missing') for r in audits)),
          'rows':audits,'code_sha256':digest(Path(__file__)),
          'scope':'request/transport/schema integrity, not proof all market events are present',
          'empty_response_semantics':'original SDK None vs empty DataFrame not recorded; neither proves absence of an event',
          'formal_eligibility':False})
    assert len(audits)>0 and not bad,bad[:3]
    print('Verified requests:',len(audits),'raw rows:',sum(r.get('rows',0) for r in audits))


if __name__=='__main__':main()
