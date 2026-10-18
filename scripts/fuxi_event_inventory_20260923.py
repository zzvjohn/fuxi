"""Full-window corporate-action inventory from already-sealed daily responses.

The window-evidence batch already sealed one dividend response per ex-date across
the whole replay window. Those responses were previously consumed through a
narrower assembler that rejected any row whose report period disagreed with a
duplicate, which is why several held positions still block the ledger. This
script re-reads the sealed responses, merges duplicates that agree on every
binding field, and requires each survivor to reproduce the exchange reference
identity against the raw price and adjustment panels.

No new collection. No writes to raw data, sealed evidence, production state, or
any prior report. Output is a new, self-contained inventory directory.
"""

import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'research/factor_alchemy'))
sys.path.insert(0, str(ROOT / 'scripts'))
from corporate_action_inventory import assess_row, merge_same_implementation

EVIDENCE = ROOT / 'reports/window_evidence_20260923'
OUT = ROOT / 'reports/event_inventory_20260923'
START, END = '2024-12-27', '2026-09-21'
WINDOW = (START, END)
# Securities whose own history shows the vendor populating both component
# fields, which is the evidence required before a lone component is read as a
# zero. Computed from the sealed responses themselves.
FIELDS = ('ts_code', 'end_date', 'ann_date', 'div_proc', 'stk_div', 'stk_bo_rate',
          'stk_co_rate', 'cash_div', 'cash_div_tax', 'record_date', 'ex_date',
          'pay_date', 'div_listdate', 'imp_ann_date')


def iso(value):
    """Vendor responses mix compact and dashed dates; normalise to ISO."""
    text = str(value or '').strip()
    if len(text) == 8 and text.isdigit():
        return '%s-%s-%s' % (text[:4], text[4:6], text[6:8])
    return text


def load_sealed_rows():
    """Read every sealed per-ex-date dividend response, keeping provenance."""
    rows, files = [], []
    for path in sorted((EVIDENCE / 'raw').glob('dividend_*.json')):
        payload = json.loads(path.read_text(encoding='utf-8'))
        for record in payload.get('records') or []:
            if record.get('div_proc') != '实施':
                continue
            if not record.get('ex_date'):
                continue
            rows.append({**{k: record.get(k) for k in FIELDS},
                         'ex_date': iso(record.get('ex_date')),
                         'record_date': iso(record.get('record_date')),
                         'pay_date': iso(record.get('pay_date')),
                         'div_listdate': iso(record.get('div_listdate')),
                         'code': record.get('ts_code'), 'source': path.name,
                         'possible_truncation': bool(payload.get('possible_truncation'))})
        files.append(path.name)
    return rows, files


def proven_alternation(rows):
    """Securities whose own history proves both component fields carry values."""
    seen = {}
    for row in rows:
        def number(value):
            try:
                out = float(value)
            except (TypeError, ValueError):
                return None
            return out if out == out else None
        bonus, transfer = number(row.get('stk_bo_rate')), number(row.get('stk_co_rate'))
        if bonus is not None and bonus > 0:
            seen.setdefault(row['code'], set()).add('bonus')
        if transfer is not None and transfer > 0:
            seen.setdefault(row['code'], set()).add('transfer')
    return {code for code, kinds in seen.items() if len(kinds) >= 2}


def price_and_adjustment_panels():
    """Previous-session close and both adjustment legs keyed by (code, ex_date)."""
    def panel(path, columns, source, target):
        frame = pd.read_csv(path, usecols=columns, dtype={'ts_code': str, 'trade_date': str})
        frame['trade_date'] = pd.to_datetime(frame.trade_date, format='mixed').dt.strftime('%Y-%m-%d')
        frame = frame[(frame.trade_date >= START) & (frame.trade_date <= END)]
        frame = frame.drop_duplicates(['ts_code', 'trade_date']).sort_values(['ts_code', 'trade_date'])
        frame[target] = frame.groupby('ts_code')[source].shift()
        return frame

    prices = panel(ROOT / 'data/raw/daily_prices.csv',
                   ['ts_code', 'trade_date', 'close'], 'close', 'previous_close')
    adjustments = panel(ROOT / 'data/raw/adj_factor_daily.csv',
                        ['ts_code', 'trade_date', 'adj_factor'], 'adj_factor',
                        'previous_adj_factor')
    prev_prices = {(c, d): v for c, d, v in zip(prices.ts_code, prices.trade_date, prices.previous_close)}
    legs = {(c, d): (p, v) for c, d, p, v in zip(
        adjustments.ts_code, adjustments.trade_date,
        adjustments.previous_adj_factor, adjustments.adj_factor)}
    return prev_prices, legs, {'prices': len(prices), 'adjustments': len(adjustments)}


def main():
    rows, files = load_sealed_rows()
    if not rows:
        raise SystemExit('no sealed dividend rows found')
    proven = proven_alternation(rows)
    merged, _ = merge_same_implementation(rows)
    prev_close, legs, panels = price_and_adjustment_panels()

    approved, rejected, unresolved = [], [], []
    reasons = Counter()
    for row in merged:
        key = (row['code'], row['ex_date'])
        previous, legs_pair = prev_close.get(key), legs.get(key)
        measured = {'pre_close': previous, 'adj_prev': legs_pair[0] if legs_pair else None,
                    'adj_ex': legs_pair[1] if legs_pair else None}
        verdict = assess_row(row, identity_proven=row['code'] in proven, **measured)
        reasons[verdict.reason] += 1
        entry = {'code': row['code'], 'ex_date': row['ex_date'],
                 'record_date': row.get('record_date'), 'pay_date': row.get('pay_date'),
                 'new_shares_date': row.get('div_listdate') or row['ex_date'],
                 'status': verdict.status, 'reason': verdict.reason,
                 'detail': verdict.detail, 'source': row['source'],
                 'merged_rows': row.get('merged_rows', 1),
                 'merged_report_periods': row.get('merged_report_periods'),
                 'possible_truncation': row['possible_truncation']}
        if verdict.status == 'approved':
            detail = verdict.detail
            entry['action'] = {
                'event_id': 'inv_' + row['code'] + '_' + row['ex_date'],
                'code': row['code'], 'record_date': row['record_date'],
                'ex_date': row['ex_date'], 'pay_date': row['pay_date'] or row['ex_date'],
                'cash_per_share_net': detail['net_cash'],
                'share_multiplier': 1.0 + detail['share_rate'],
                'new_shares_date': row.get('div_listdate') or row['ex_date'],
                'evidence': ('reference_identity %s; sealed %s'
                             % (round(detail['relative_error'], 6), row['source']))}
            approved.append(entry)
        elif verdict.status == 'rejected':
            rejected.append(entry)
        else:
            unresolved.append(entry)

    OUT.mkdir(parents=True, exist_ok=True)
    summary = {
        'scope': 'sealed per-ex-date responses only; no new collection',
        'response_files': len(files), 'rows_read': len(rows),
        'unique_after_report_period_merge': len(merged),
        'approved': len(approved), 'rejected': len(rejected), 'unresolved': len(unresolved),
        'reason_counts': dict(sorted(reasons.items())),
        'securities_with_proven_component_alternation': len(proven),
        'panels': panels,
        'formal_eligibility': False,
        'limits': ['approval is a reference-identity reproduction, not an issuer document audit',
                   'a rejected row is unresolved accounting, not a proven absence of entitlement',
                   'no PIT availability certification for the price or adjustment panels'],
    }
    (OUT / 'inventory.json').write_text(
        json.dumps({'summary': summary, 'approved': approved, 'rejected': rejected[:400],
                    'unresolved': unresolved[:400]}, ensure_ascii=False, indent=1),
        encoding='utf-8', newline='\n')
    (OUT / 'approved_actions.json').write_text(
        json.dumps({'actions': [e['action'] for e in approved],
                    'formal_eligibility': False}, ensure_ascii=False, indent=1),
        encoding='utf-8', newline='\n')
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    for name in ('600898.SH', '601800.SH', '688697.SH', '920871.BJ'):
        hit = [e for e in approved + rejected + unresolved if e['code'] == name]
        for e in hit[:3]:
            print(name, e['ex_date'], e['status'], e['reason'],
                  None if not e['detail'] else round(e['detail']['relative_error'], 6))


if __name__ == '__main__':
    main()
