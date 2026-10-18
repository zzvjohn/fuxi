"""Two-query identity follow-up for the single unresolved missing-quote cell."""
import sys
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'research/factor_alchemy'))
from fuxi_execution_lab_20260923 import read, write, digest
from run_contract import RunContract


def main():
    from credentials import get_tushare_api
    out = ROOT / 'reports/execution_followup_20260923/identity_probe'
    assert not out.exists()
    policy = read(ROOT / 'codex_automations/fuxi_runtime_policy.json')
    assert policy['single_writer_confirmed']
    write(out / 'protocol.json', {'max_requests': 2, 'code': '920680.BJ', 'day': '2025-06-27',
          'steps': ['bse_mapping exact new code', 'suspend_d exact mapped old code and day'],
          'code_sha256': digest(Path(__file__)), 'formal_eligibility': False})
    with RunContract(parameters={'protocol_sha256': digest(out / 'protocol.json')},
                     run_id='identity_' + datetime.now().strftime('%H%M%S_%f'), runs_dir=out / 'runs'):
        client = get_tushare_api()
        if hasattr(client, '_DataApi__timeout'):
            client._DataApi__timeout = 20
        mapping = client.bse_mapping(n_code='920680.BJ')
        records = mapping.to_dict('records')
        write(out / 'mapping.json', {'records': records, 'retrieved_at': datetime.now().astimezone().isoformat()})
        exact = [r for r in records if r.get('n_code') == '920680.BJ']
        if len(exact) != 1 or len(records) >= 1000:
            raise ValueError('ambiguous or truncated identity mapping')
        old = exact[0]['o_code']
        rows = client.suspend_d(ts_code=old, trade_date='20250627').to_dict('records')
        write(out / 'suspension.json', {'code': old, 'date': '20250627', 'records': rows,
                                      'retrieved_at': datetime.now().astimezone().isoformat()})
        found = [r for r in rows if r.get('ts_code') == old and r.get('trade_date') == '20250627'
                 and r.get('suspend_type') == 'S' and not r.get('suspend_timing')]
        write(out / 'result.json', {'mapped_code': old, 'new_code': '920680.BJ',
              'exact_old_code_halt': bool(found), 'mapping_effective_date_verified': False,
              'production_alias_rewrite': False,
              'note': 'Mapping plus old-code halt is diagnostic evidence; effective switch date still required for PIT alias adapter.',
              'formal_eligibility': False})
        print(old, 'old-code halt:', bool(found), flush=True)


if __name__ == '__main__':
    main()
