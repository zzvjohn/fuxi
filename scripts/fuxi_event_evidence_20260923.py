"""Bounded event evidence collection and single-owner transition checkpoints."""
import argparse
import json
import math
import shutil
import sys
import time
from datetime import datetime
from dataclasses import asdict
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'research/factor_alchemy'))
from run_contract import RunContract
from fuxi_execution_lab_20260923 import digest, write, read

OUT = ROOT / 'reports/execution_events_20260923'
PRIOR = ROOT / 'reports/execution_ledger_20260923'
FILES = ['codex_automations/fuxi_runtime_policy.json', 'codex_automations/FUXI_CODEX_RUNBOOK.md',
         'data/stage1_optimization_proposals.json', 'data/version_registry.json', 'scripts/version_bump.py',
         'CHANGELOG.md', 'AGENTS.md', '.workbuddy/memory/MEMORY.md',
         '.workbuddy/memory/2026-09-23.md', '.workbuddy/memory/_fuxi_landing_details.md']
FIELDS = 'ts_code,end_date,ann_date,div_proc,stk_div,stk_bo_rate,stk_co_rate,cash_div,cash_div_tax,record_date,ex_date,pay_date,div_listdate,imp_ann_date'


def checked_cash_action(item, code, day, record_close, ex_reference, evidence):
    from portfolio_execution_ledger import CorporateAction
    def iso(value):
        return datetime.strptime(value, '%Y%m%d').date().isoformat()
    if item.get('ts_code') != code or item.get('div_proc') != '实施' or iso(item['ex_date']) != day:
        raise ValueError('wrong event identity or implementation status')
    record, payment, announcement = (iso(item[k]) for k in ['record_date', 'pay_date', 'imp_ann_date'])
    if not announcement <= record < day <= payment:
        raise ValueError('event dates violate availability/payment contract')
    gross, net, bonus = (float(item[k]) for k in ['cash_div_tax', 'cash_div', 'stk_div'])
    if not all(math.isfinite(v) for v in [gross, net, bonus, record_close, ex_reference]):
        raise ValueError('nonfinite action data')
    if bonus != 0 or not 0 <= net <= gross or gross <= 0:
        raise ValueError('this bounded adapter accepts cash-only distributions')
    expected = record_close - gross
    if not math.isclose(expected, ex_reference, abs_tol=.011, rel_tol=0):
        raise ValueError('raw ex-reference does not reconcile with gross cash')
    return CorporateAction('cash_' + code + '_' + day, code, record, day, payment,
                           cash_per_share_net=net, evidence=evidence)


def freeze():
    if (OUT / 'protocol.json').exists():
        raise RuntimeError('protocol already exists')
    for name in FILES:
        target = OUT / 'baseline' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    requests = read(PRIOR / 'market_final/event_data_requests.json')['requests']
    assert len(requests) == 6
    write(OUT / 'protocol.json', {'requests': requests, 'fields': FIELDS, 'max_requests': 6,
          'created_at': datetime.now().astimezone().isoformat(),
          'prior_audit_sha256': digest(PRIOR / 'release_audit.json'),
          'prior_input_hashes': read(PRIOR / 'protocol.json')['sha256'],
          'original_file_hashes': {name: digest(ROOT / name) for name in FILES},
          'confirmation': {'user_message': '已经停用', 'date': '2026-09-23',
                           'scope': 'WorkBuddy Fuxi daily/weekly tasks stopped by user',
                           'thread_id': '01a0ab93-b94b-7820-a8a5-dc70f39aa8c7'},
          'formal_eligibility': False, 'raw_data_overwrite': False})
    print('Frozen six-request protocol and owner transition baseline.')


def collect():
    from credentials import get_tushare_api
    policy = read(ROOT / 'codex_automations/fuxi_runtime_policy.json')
    assert policy['single_writer_confirmed'] and policy['confirmation_source']
    protocol = read(OUT / 'protocol.json')
    if (OUT / 'collection.json').exists():
        raise RuntimeError('collection already recorded; no automatic repeat')
    rows = []
    with RunContract(parameters={'scope': 'six-event-evidence-only', 'protocol_sha256': digest(OUT / 'protocol.json'),
                                 'collector_sha256': digest(Path(__file__))},
                     run_id='event_collection_' + datetime.now().strftime('%H%M%S_%f'), runs_dir=OUT / 'runs'):
        client = get_tushare_api()
        # Existing SDK convention; cap the network wait without logging credentials.
        if hasattr(client, '_DataApi__timeout'):
            client._DataApi__timeout = 20
        host = urlsplit(getattr(client, '_DataApi__http_url', '')).hostname
        for request in protocol['requests']:
            code = request['code']
            started = datetime.now().astimezone().isoformat()
            row = {'code': code, 'date': request['date'], 'started_at': started,
                   'api': 'dividend', 'provider_host': host, 'fields': FIELDS, 'verified_event': False}
            try:
                frame = client.dividend(ts_code=code, fields=FIELDS)
                records = [] if frame is None else frame.to_dict('records')
                write(OUT / 'raw_dividend' / (code + '.json'), {'request': row, 'records': records})
                wanted = request['date'].replace('-', '')
                matches = [item for item in records if str(item.get('ex_date', '')).replace('-', '') == wanted]
                row.update(status='response_received', rows=len(records), possible_truncation=len(records) >= 2000,
                           event_matches=matches, raw_sha256=digest(OUT / 'raw_dividend' / (code + '.json')))
            except Exception as exc:
                detail = str(exc).lower()
                category = 'quota_or_permission' if any(w in detail for w in ('权限', '积分', 'token', '每分钟', 'limit')) else 'transport_or_api_error'
                row.update(status='request_failed', error_type=type(exc).__name__, error_category=category)
            rows.append(row)
            write(OUT / 'collection_progress.json', {'rows': rows, 'complete': len(rows) == 6})
            print(json.dumps({k: row.get(k) for k in ('code', 'status', 'rows', 'error_type')}, ensure_ascii=False), flush=True)
            time.sleep(1.3)
        write(OUT / 'collection.json', {'rows': rows, 'request_count': 6, 'formal_eligibility': False,
                                       'note': 'Raw evidence only; no action approved or price overwritten.'})


def normalize():
    for name in FILES:
        old = (OUT / 'baseline' / name).read_bytes()
        current = ROOT / name
        raw = current.read_bytes()
        if b'\r\n' in old and b'\n' not in old.replace(b'\r\n', b''):
            normalized = raw.replace(b'\r\n', b'\n').replace(b'\n', b'\r\n')
            if raw != normalized:
                current.write_bytes(normalized)
    for name in FILES[-2:]:
        assert (ROOT / name).read_bytes().startswith((OUT / 'baseline' / name).read_bytes())
    print('Original line endings preserved; dated evidence remains append-only.')


def validate():
    import pandas as pd
    if (OUT / 'approved_actions.json').exists():
        raise RuntimeError('action evidence already frozen')
    collected = read(OUT / 'collection.json')['rows']
    codes = {r['code'] for r in collected}
    pieces = []
    for chunk in pd.read_csv(ROOT / 'data/raw/daily_prices.csv',
                             usecols=['ts_code', 'trade_date', 'close', 'pre_close'],
                             dtype={'ts_code': str, 'trade_date': str}, chunksize=300000):
        chunk = chunk[chunk.ts_code.isin(codes)].copy()
        chunk['trade_date'] = pd.to_datetime(chunk.trade_date, format='mixed').dt.strftime('%Y-%m-%d')
        pieces.append(chunk[chunk.trade_date.between('2025-01-01', '2025-01-10')])
    quotes = pd.concat(pieces).set_index(['trade_date', 'ts_code'])
    assert quotes.index.is_unique
    actions, checks = [], []
    for row in collected:
        assert row['status'] == 'response_received' and not row['possible_truncation']
        raw = OUT / 'raw_dividend' / (row['code'] + '.json')
        assert digest(raw) == row['raw_sha256']
        implemented = [item for item in row['event_matches'] if item.get('div_proc') == '实施']
        if len(implemented) != 1:
            raise ValueError('missing/ambiguous implementation record: ' + row['code'])
        item = implemented[0]
        record = datetime.strptime(item['record_date'], '%Y%m%d').date().isoformat()
        close = float(quotes.loc[(record, row['code']), 'close'])
        ref = float(quotes.loc[(row['date'], row['code']), 'pre_close'])
        action = checked_cash_action(item, row['code'], row['date'], close, ref, digest(raw))
        actions.append(asdict(action))
        checks.append({'code': row['code'], 'ex_date': row['date'], 'record_close': close,
                       'ex_reference': ref, 'gross_per_share': item['cash_div_tax'],
                       'provider_net_per_share': item['cash_div'],
                       'absolute_reference_error': abs(close - item['cash_div_tax'] - ref),
                       'excluded_nonimplementation_records': len(row['event_matches']) - 1})
    assert len(actions) == 6
    write(OUT / 'approved_actions.json', {'actions': actions, 'checks': checks,
          'cash_policy': 'provider cash_div field (documented after-tax); not investor-specific deferred tax simulation',
          'scope': 'six cash-only events; one provider plus independent raw-price reference check; not issuer-notice double sourcing',
          'source_documentation': 'https://tushare.pro/document/2?doc_id=103', 'formal_eligibility': False})
    print(json.dumps({'cash_events_verified': len(actions), 'checks': checks}, ensure_ascii=False))


def replay():
    import fuxi_execution_lab_20260923 as lab
    from portfolio_execution_ledger import CorporateAction
    destination = OUT / 'replay'
    destination.mkdir(exist_ok=True)
    if (destination / 'protocol.json').exists():
        raise RuntimeError('replay protocol already exists')
    protocol = read(PRIOR / 'protocol.json')
    for name, expected in protocol['sha256'].items():
        if name.startswith('data/raw/'):
            assert digest(ROOT / name) == expected
    protocol['sha256'] = {name: digest(ROOT / name) for name in protocol['sha256']}
    action_file = 'reports/execution_events_20260923/approved_actions.json'
    protocol['sha256'][action_file] = digest(ROOT / action_file)
    protocol['revision'] = 'EL-20260923.2'
    protocol['parent_protocol_sha256'] = digest(PRIOR / 'protocol.json')
    protocol['actions'] = 'six explicitly checked cash-only events; original prices unchanged'
    write(destination / 'protocol.json', protocol)
    lab.OUT = destination
    lab.CODE = lab.CODE + ['scripts/fuxi_event_evidence_20260923.py',
                           'research/factor_alchemy/tests/test_cash_event_evidence.py']
    lab.tests()
    lab.market(actions=[CorporateAction(**a) for a in read(OUT / 'approved_actions.json')['actions']])


def audit():
    if (OUT / 'release_audit.json').exists():
        raise RuntimeError('already sealed')
    new_rows = read(OUT / 'replay/market.json')['rows']
    checks = []
    for row in new_rows:
        name = row['signal']
        old = read(PRIOR / 'market_final/ledgers' / (name + '.json'))
        new = read(OUT / 'replay/ledgers' / (name + '.json'))
        n = len(old['days']) - 1
        assert n > 0 and new['days'][:n] == old['days'][:n]
        assert len(new['days']) > len(old['days'])
        paid = sum(j['cash_delta'] for j in new['journal'] if j['kind'] == 'payment')
        assert paid > 0
        checks.append({'signal': name, 'unchanged_pre_event_days': n, 'old_days': len(old['days']),
                       'new_days': len(new['days']), 'dividend_cash_received': paid,
                       'new_status': new['status'], 'errors': new['errors']})
    protected = read(OUT / 'protocol.json')['prior_input_hashes']
    allowed = {'codex_automations/fuxi_runtime_policy.json', 'data/version_registry.json',
               'data/stage1_optimization_proposals.json'}
    for name, expected in protected.items():
        if name not in allowed:
            assert digest(ROOT / name) == expected, name
    old_audit = read(PRIOR / 'release_audit.json')
    for name, expected in old_audit['evidence_sha256'].items():
        assert digest(PRIOR / name) == expected, name
    tests = read(OUT / 'replay/tests.json')
    assert tests['successful']
    for name, expected in tests['code_sha256'].items():
        assert digest(ROOT / name) == expected, name
    old_p = read(OUT / 'baseline/data/stage1_optimization_proposals.json')
    proposals = read(ROOT / 'data/stage1_optimization_proposals.json')
    assert proposals[:len(old_p)] == old_p and len(proposals) == len(old_p) + 1
    versions = read(ROOT / 'data/version_registry.json')['versions']
    assert versions[-1]['version'] == 'v0.7.57'
    for path in list((OUT / 'runs').glob('*/manifest.json')) + list((OUT / 'replay/runs').glob('*/manifest.json')):
        assert read(path)['status'] != 'running'
    normalize()
    names = FILES + list(tests['code_sha256']) + ['reports/fuxi_execution_events_20260923.md']
    for name in set(names):
        target = OUT / 'release' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    write(OUT / 'release_audit.json', {'version': 'v0.7.57', 'owner_confirmed': True,
          'autonomous_writes_enabled': False, 'tests': tests['tests_run'], 'cash_events': 6,
          'comparison': checks, 'raw_prices_and_production_s5_unchanged': True,
          'formal_eligibility': False, 'old_sealed_files_unchanged': len(old_audit['evidence_sha256']),
          'evidence_sha256': {p.relative_to(OUT).as_posix(): digest(p) for p in OUT.rglob('*') if p.is_file()}})
    print(json.dumps({'sealed': True, 'tests': tests['tests_run'], 'comparisons': checks}, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('step', choices=['freeze', 'collect', 'validate', 'replay', 'normalize', 'audit'])
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    globals()[args.step]()
