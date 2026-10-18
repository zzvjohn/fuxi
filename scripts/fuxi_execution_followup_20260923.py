"""Bounded continuation: immutable event/state evidence, never production writes."""
import argparse
import sys
import shutil
import time
from datetime import datetime
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'research/factor_alchemy'))
from fuxi_execution_lab_20260923 import read, write, digest
from fuxi_event_evidence_20260923 import FIELDS, checked_cash_action
from run_contract import RunContract, FuxiWriterLock

OUT = ROOT / 'reports/execution_followup_20260923'
PRIOR = ROOT / 'reports/execution_events_20260923'


def freeze():
    assert not (OUT / 'protocol.json').exists()
    old = read(ROOT / 'reports/holdings_20260923/marking_attribution.json')
    cells = [c for r in old['rows'] for c in r['cells']
             if not c['quote_present'] and not c['explicit_full_day_halt']]
    events = read(PRIOR / 'replay/event_data_requests.json')['requests']
    requests = [{'id': 'dividend_' + r['code'], 'api': 'dividend',
                 'params': {'ts_code': r['code'], 'fields': FIELDS}, 'cap': 2000}
                for r in events]
    for code in sorted({c['ts_code'] for c in cells}):
        requests.append({'id': 'suspend_' + code, 'api': 'suspend_d', 'cap': 5000,
                         'params': {'ts_code': code, 'start_date': '20241220',
                                    'end_date': '20260909'}})
    for status in ['L', 'D', 'P']:
        requests.append({'id': 'lifecycle_' + status, 'api': 'stock_basic', 'cap': 6000,
                         'params': {'list_status': status,
                                    'fields': 'ts_code,name,list_status,list_date,delist_date,exchange'}})
    assert len(cells) == 23 and len(events) == 4 and len(requests) == 16
    names = ['data/version_registry.json', 'data/stage1_optimization_proposals.json',
             'scripts/version_bump.py', 'CHANGELOG.md', '.workbuddy/memory/2026-09-23.md',
             '.workbuddy/memory/_fuxi_landing_details.md']
    for name in names:
        dest = OUT / 'baseline' / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, dest)
    write(OUT / 'protocol.json', {'created_at': datetime.now().astimezone().isoformat(),
          'events': events, 'unknown_cells': cells, 'requests': requests, 'max_requests': 16,
          'history_exposed': True, 'formal_eligibility': False,
          'prior_protocol_sha256': digest(PRIOR / 'replay/protocol.json'),
          'prior_actions_sha256': digest(PRIOR / 'approved_actions.json'),
          'protected': {n: digest(ROOT / n) for n in read(PRIOR / 'replay/protocol.json')['sha256']},
          'code_sha256': digest(Path(__file__))})
    print('Frozen 16 bounded queries; four signals unchanged.', flush=True)


def collect():
    from credentials import get_tushare_api
    policy = read(ROOT / 'codex_automations/fuxi_runtime_policy.json')
    assert policy['single_writer_confirmed'] and policy['current_mode'] == 'controlled_manual'
    protocol = read(OUT / 'protocol.json')
    assert digest(Path(__file__)) == protocol['code_sha256']
    with RunContract(parameters={'protocol_sha256': digest(OUT / 'protocol.json')},
                     run_id='followup_collect_' + datetime.now().strftime('%H%M%S_%f'),
                     runs_dir=OUT / 'runs'):
        client = get_tushare_api()
        if hasattr(client, '_DataApi__timeout'):
            client._DataApi__timeout = 20
        for request in protocol['requests']:
            path = OUT / 'raw' / (request['id'] + '.json')
            if path.exists():
                continue  # immutable failed attempts also require explicit new batch
            row = {'request': request, 'retrieved_at': datetime.now().astimezone().isoformat()}
            try:
                frame = getattr(client, request['api'])(**request['params'])
                records = [] if frame is None else frame.to_dict('records')
                row.update(status='response_received', records=records,
                           possible_truncation=len(records) >= request['cap'])
            except Exception as exc:
                row.update(status='request_failed', error_type=type(exc).__name__)
            write(path, row)
            print(request['id'], row['status'], len(row.get('records', [])), flush=True)
            time.sleep(1.3)


def validate():
    import pandas as pd
    assert not (OUT / 'validation.json').exists()
    protocol = read(OUT / 'protocol.json')
    raws = {r['id']: read(OUT / 'raw' / (r['id'] + '.json')) for r in protocol['requests']}
    def usable(r):
        return r['status'] == 'response_received' and not r['possible_truncation']
    codes = {r['code'] for r in protocol['events']}
    pieces = []
    for part in pd.read_csv(ROOT / 'data/raw/daily_prices.csv',
                            usecols=['ts_code', 'trade_date', 'close', 'pre_close'],
                            dtype={'trade_date': str}, chunksize=300000):
        part = part[part.ts_code.isin(codes)].copy()
        part['trade_date'] = pd.to_datetime(part.trade_date, format='mixed').dt.strftime('%Y-%m-%d')
        pieces.append(part[part.trade_date.between('2024-12-20', '2025-02-26')])
    quotes = pd.concat(pieces).set_index(['trade_date', 'ts_code'])
    assert quotes.index.is_unique
    actions, checks = [], []
    for event in protocol['events']:
        code, day = event['code'], event['date']
        raw = raws['dividend_' + code]
        check = {'code': code, 'date': day, 'status': 'unresolved'}
        try:
            if not usable(raw):
                raise ValueError('unusable_response')
            matches = [r for r in raw['records'] if r.get('div_proc') == '实施'
                       and r.get('ex_date') == day.replace('-', '')]
            if len(matches) != 1:
                raise ValueError('missing_or_ambiguous_implementation')
            item = matches[0]
            record = datetime.strptime(item['record_date'], '%Y%m%d').date().isoformat()
            close = float(quotes.loc[(record, code), 'close'])
            ref = float(quotes.loc[(day, code), 'pre_close'])
            action = checked_cash_action(item, code, day, close, ref,
                                        digest(OUT / 'raw' / ('dividend_' + code + '.json')))
            actions.append(asdict(action))
            check.update(status='verified_cash_event', reference_error=abs(close-float(item['cash_div_tax'])-ref))
        except (ValueError, KeyError, TypeError) as exc:
            check.update(reason=str(exc))
        checks.append(check)
    states = []
    for cell in protocol['unknown_cells']:
        raw = raws['suspend_' + cell['ts_code']]
        matches = [r for r in raw.get('records', []) if r.get('ts_code') == cell['ts_code']
                   and r.get('trade_date') == cell['date'].replace('-', '')] if usable(raw) else []
        confirmed = any(r.get('suspend_type') == 'S' and not r.get('suspend_timing') for r in matches)
        life = [r for key, data in raws.items() if key.startswith('lifecycle_') and usable(data)
                for r in data['records'] if r.get('ts_code') == cell['ts_code']]
        states.append({**cell, 'new_state': 'confirmed_full_day_halt' if confirmed else 'unresolved',
                       'suspend_records': matches, 'lifecycle_records': life,
                       'note': 'No halt record is not proof of trading; lifecycle snapshot is not PIT industry/ST.'})
    write(OUT / 'validation.json', {'event_checks': checks, 'new_actions': actions,
          'state_checks': states, 'formal_eligibility': False,
          'raw_sha256': {p.name: digest(p) for p in (OUT / 'raw').glob('*.json')}})
    print('Validated events:', len(actions), 'confirmed halt cells:', sum(r['new_state']=='confirmed_full_day_halt' for r in states), flush=True)


def replay():
    import fuxi_execution_lab_20260923 as lab
    from portfolio_execution_ledger import CorporateAction
    destination = OUT / 'replay'
    assert not destination.exists()
    validation = read(OUT / 'validation.json')
    assert len(validation['new_actions']) > 0
    protocol = read(PRIOR / 'replay/protocol.json')
    for name in protocol['sha256']:
        if name.startswith('data/raw/'):
            assert digest(ROOT / name) == protocol['sha256'][name], name
    protocol['sha256'] = {n: digest(ROOT / n) for n in protocol['sha256']}
    name = 'reports/execution_followup_20260923/validation.json'
    protocol['sha256'][name] = digest(ROOT / name)
    protocol.update(revision='EL-20260923.3', parent_protocol_sha256=digest(PRIOR / 'replay/protocol.json'),
                    actions='prior six plus newly validated cash events; state evidence diagnostic only')
    write(destination / 'protocol.json', protocol)
    actions = read(PRIOR / 'approved_actions.json')['actions'] + validation['new_actions']
    assert len({a['event_id'] for a in actions}) == len(actions)
    lab.OUT = destination
    lab.CODE += ['scripts/fuxi_execution_followup_20260923.py', 'scripts/fuxi_event_evidence_20260923.py']
    lab.tests()
    with FuxiWriterLock(run_id='followup_market_' + datetime.now().strftime('%H%M%S_%f')):
        lab.market(actions=[CorporateAction(**a) for a in actions])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('step', choices=['freeze', 'collect', 'validate', 'replay'])
    OUT.mkdir(exist_ok=True)
    globals()[parser.parse_args().step]()
