"""Resume-safe window event collection. Outputs only to an isolated report tree."""
import argparse
import sys
import time
import shutil
from pathlib import Path
from datetime import datetime
from collections import Counter
from dataclasses import asdict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'research/factor_alchemy'))
from fuxi_execution_lab_20260923 import read, write, digest
from fuxi_event_evidence_20260923 import FIELDS, checked_cash_action
from run_contract import RunContract
OUT = ROOT / 'reports/window_evidence_20260923'


def freeze():
    import pandas as pd
    assert not (OUT / 'protocol.json').exists()
    calendar = pd.read_csv(ROOT / 'data/raw/trade_calendar.csv', dtype=str)
    days = sorted(set(pd.to_datetime(calendar.loc[(calendar.exchange == 'SSE') & (calendar.is_open == '1'), 'cal_date'], format='mixed').dt.strftime('%Y%m%d')))
    days = [d for d in days if '20241230' <= d <= '20260909']
    requests = []
    for code in ['601868.SH', '002014.SZ', '603998.SH']:
        requests.append({'id': 'priority_' + code, 'api': 'dividend', 'cap': 2000,
                         'params': {'ts_code': code, 'fields': FIELDS}})
    for month in pd.period_range('2024-12', '2026-09', freq='M'):
        start = max('20241220', month.start_time.strftime('%Y%m%d'))
        end = min('20260909', month.end_time.strftime('%Y%m%d'))
        requests.append({'id': 'suspend_' + str(month), 'api': 'suspend_d', 'cap': 5000,
                         'params': {'start_date': start, 'end_date': end}})
    for day in days:
        if day > '20260803':
            requests.append({'id': 'limits_' + day, 'api': 'stk_limit', 'cap': 5800,
                             'params': {'trade_date': day, 'fields': 'ts_code,trade_date,up_limit,down_limit'}})
    for day in days:
        requests.append({'id': 'dividend_' + day, 'api': 'dividend', 'cap': 2000,
                         'params': {'ex_date': day, 'fields': FIELDS}})
    prior = read(ROOT / 'reports/execution_followup_20260923/replay/protocol.json')
    protected = {n: digest(ROOT / n) for n in prior['sha256']}
    names = ['data/version_registry.json', 'data/stage1_optimization_proposals.json', 'scripts/version_bump.py',
             'CHANGELOG.md', '.workbuddy/memory/2026-09-23.md', '.workbuddy/memory/_fuxi_landing_details.md',
             'research/fuxi_five_stage_acceptance_20260923.md']
    for name in names:
        dest = OUT / 'baseline' / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, dest)
    write(OUT / 'protocol.json', {'created_at': datetime.now().astimezone().isoformat(),
          'requests': requests, 'max_base_requests': len(requests), 'max_attempts_per_request': 2,
          'start': '2024-12-30', 'end': '2026-09-09', 'protected': protected,
          'signals': prior['signals'], 'code_sha256': digest(Path(__file__)),
          'scope': 'event accounting evidence; no signal search or production data replacement',
          'formal_eligibility': False})
    print('Frozen', len(requests), dict(Counter(r['api'] for r in requests)), flush=True)


def collect():
    from credentials import get_tushare_api
    policy = read(ROOT / 'codex_automations/fuxi_runtime_policy.json')
    assert policy['single_writer_confirmed'] and policy['current_mode'] == 'controlled_manual'
    protocol = read(OUT / 'protocol.json')
    with RunContract(parameters={'protocol_sha256': digest(OUT / 'protocol.json'), 'code_sha256': digest(Path(__file__))},
                     run_id='window_collect_' + datetime.now().strftime('%H%M%S_%f'), runs_dir=OUT / 'runs'):
        client = get_tushare_api()
        if hasattr(client, '_DataApi__timeout'):
            client._DataApi__timeout = 20
        for i, req in enumerate(protocol['requests']):
            path = OUT / 'raw' / (req['id'] + '.json')
            if path.exists():
                continue
            attempts = OUT / 'attempts' / req['id']
            done = len(list(attempts.glob('*.json'))) if attempts.exists() else 0
            if done >= 2:
                continue
            result = {'request': req, 'retrieved_at': datetime.now().astimezone().isoformat()}
            try:
                frame = getattr(client, req['api'])(**req['params'])
                records = [] if frame is None else frame.to_dict('records')
                result.update(status='response_received', records=records, possible_truncation=len(records) >= req['cap'])
            except Exception as exc:
                result.update(status='request_failed', error_type=type(exc).__name__)
            write(attempts / (str(done + 1) + '.json'), result)
            if result['status'] == 'response_received':
                write(path, result)
            write(OUT / 'progress.json', {'last_index': i, 'planned': len(protocol['requests']),
                  'saved_responses': len(list((OUT / 'raw').glob('*.json'))) if (OUT / 'raw').exists() else 0,
                  'last_id': req['id'], 'last_status': result['status']})
            print(i + 1, '/', len(protocol['requests']), req['id'], result['status'], len(result.get('records', [])), flush=True)
            time.sleep(1.3)


def assemble():
    import pandas as pd
    assert not (OUT / 'assembly.json').exists()
    protocol = read(OUT / 'protocol.json')
    records = {'dividend': [], 'suspend_d': [], 'stk_limit': []}
    sources, gaps = {}, []
    for req in protocol['requests']:
        path = OUT / 'raw' / (req['id'] + '.json')
        if not path.exists():
            gaps.append({'request': req['id'], 'reason': 'no_response'})
            continue
        data = read(path)
        sources[req['id']] = digest(path)
        if data['possible_truncation']:
            gaps.append({'request': req['id'], 'reason': 'possible_truncation'})
            continue
        records[req['api']].extend(data['records'])
    for api in ['suspend_d', 'stk_limit']:
        frame = pd.DataFrame(records[api]).drop_duplicates()
        path = OUT / (api + '_overlay.csv')
        frame.to_csv(path, index=False)
    dividends = pd.DataFrame(records['dividend']).drop_duplicates()
    dividends.to_csv(OUT / 'dividend_overlay.csv', index=False)
    implemented = dividends[dividends.div_proc.eq('实施')].copy()
    implemented['iso_ex'] = pd.to_datetime(implemented.ex_date, format='mixed', errors='coerce').dt.strftime('%Y-%m-%d')
    implemented['iso_record'] = pd.to_datetime(implemented.record_date, format='mixed', errors='coerce').dt.strftime('%Y-%m-%d')
    implemented = implemented[implemented.iso_ex.between(protocol['start'], protocol['end']) &
                              implemented.iso_record.ge(protocol['start'])]
    targets = ROOT / 'reports/execution_followup_20260923/replay/targets'
    codes = {c for path in targets.glob('*.json') for w in read(path)['targets'].values() if w for c in w}
    implemented = implemented[implemented.ts_code.isin(codes)]
    pieces = []
    for part in pd.read_csv(ROOT / 'data/raw/daily_prices.csv',
                            usecols=['ts_code', 'trade_date', 'close', 'pre_close'], dtype={'trade_date': str}, chunksize=300000):
        part = part[part.ts_code.isin(set(implemented.ts_code))].copy()
        part['trade_date'] = pd.to_datetime(part.trade_date, format='mixed').dt.strftime('%Y-%m-%d')
        pieces.append(part[part.trade_date.between('2024-12-20', protocol['end'])])
    quotes = pd.concat(pieces).set_index(['trade_date', 'ts_code'])
    assert quotes.index.is_unique
    actions, rejected = [], []
    duplicates = implemented.duplicated(['ts_code', 'iso_ex'], keep=False)
    for index, row in implemented.iterrows():
        code, day = row.ts_code, row.iso_ex
        try:
            if duplicates.loc[index]:
                raise ValueError('ambiguous_same_day_event')
            close = float(quotes.loc[(row.iso_record, code), 'close'])
            ref = float(quotes.loc[(day, code), 'pre_close'])
            action = checked_cash_action(row.to_dict(), code, day, close, ref, digest(OUT / 'dividend_overlay.csv'))
            actions.append(asdict(action))
        except (ValueError, KeyError, TypeError) as exc:
            rejected.append({'code': code, 'date': day, 'reason': str(exc), 'stk_div': row.get('stk_div')})
    assert len({a['event_id'] for a in actions}) == len(actions)
    write(OUT / 'assembly.json', {'approved_cash_actions': actions, 'rejected_events': rejected,
          'request_gaps': gaps, 'source_sha256': sources, 'row_counts': {k: len(v) for k, v in records.items()},
          'target_codes': len(codes), 'formal_eligibility': False,
          'note': 'cash-only adapter; stock dividends and missing/ambiguous evidence remain explicit blockers'})
    print('Assembled cash events', len(actions), 'rejected', len(rejected), 'request gaps', len(gaps), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('step', choices=['freeze', 'collect', 'assemble'])
    OUT.mkdir(exist_ok=True)
    globals()[parser.parse_args().step]()
