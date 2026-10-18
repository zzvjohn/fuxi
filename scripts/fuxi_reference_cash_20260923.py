"""Independent Decimal reference for evidenced cash-only finite ledger prefixes.

Consumes targets, raw quotes/limits/suspensions and approved events. Never uses
primary orders or journal to decide fills. Does not certify unknown terminal days.
"""
import sys
from decimal import Decimal, ROUND_FLOOR
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'research/factor_alchemy'))
from fuxi_execution_lab_20260923 import read, write, digest

D = lambda x: Decimal(str(x))


def simulate(days, targets, quotes, limits, halts, actions, initial=10000000, bps=15):
    cash, due, nav = D(initial), D(0), D(initial)
    quantities, marks, desired, entitlements = {}, {}, {}, {}
    seen, pending = set(), []
    output, fills = [], []
    fee_rate = D(bps) / D(10000)

    def lot(code, value):
        minimum, step = (200, 1) if code.startswith(('688', '689')) else (100, 1) if code.endswith('.BJ') else (100, 100)
        number = int((D(value) / step).to_integral_value(rounding=ROUND_FLOOR)) * step
        return number if number >= minimum else 0

    for day in days:
        fresh = sorted(t for t in targets if t < day and t not in seen)
        if fresh:
            stamp = fresh[-1]
            seen.update(fresh)
            weights = targets[stamp]
            desired = dict(quantities) if weights is None else {
                c: lot(c, nav * D(w) / (1 + fee_rate) / D(quotes[(stamp, c)]['close']))
                for c, w in weights.items() if w > 0}
        for a in actions:
            if a['ex_date'] == day:
                assert a['share_multiplier'] == 1.0
                amount = D(entitlements.get(a['event_id'], 0)) * D(a['cash_per_share_net'])
                due += amount
                pending.append((a['pay_date'], amount))
        remaining = []
        for when, amount in pending:
            if when <= day:
                cash += amount
                due -= amount
            else:
                remaining.append((when, amount))
        pending = remaining
        for buying in [False, True]:
            for code in sorted(set(quantities) | set(desired)):
                held = quantities.get(code, 0)
                delta = desired.get(code, 0) - held
                if not delta or (delta > 0) != buying:
                    continue
                if (day, code) in halts:
                    continue
                quote, boundary = quotes[(day, code)], limits[(day, code)]
                price = D(quote['open'])
                assert price > 0
                if buying and price >= D(boundary['up_limit']) - D('0.000001'):
                    continue
                if not buying and price <= D(boundary['down_limit']) + D('0.000001'):
                    continue
                amount = abs(delta)
                if buying:
                    budget = int(((cash + D('0.00000001')) / (price * (1 + fee_rate))).to_integral_value(rounding=ROUND_FLOOR))
                    amount = lot(code, min(amount, budget))
                elif amount != held:
                    amount = lot(code, amount)
                if amount == 0:
                    continue
                quantity = amount if buying else -amount
                fee = D(amount) * price * fee_rate
                cash -= D(quantity) * price + fee
                quantities[code] = held + quantity
                assert cash >= D('-0.000001') and quantities[code] >= 0
                fills.append({'date': day, 'code': code, 'quantity': quantity, 'price': float(price), 'fee': float(fee)})
        positions = {c: q for c, q in quantities.items() if q}
        for code in positions:
            if (day, code) not in halts:
                marks[code] = D(quotes[(day, code)]['close'])
            assert code in marks and marks[code] > 0
        nav = cash + due + sum(D(q) * marks[c] for c, q in positions.items())
        output.append({'date': day, 'cash': float(cash), 'receivable': float(due),
                       'nav': float(nav), 'positions': positions})
        for a in actions:
            if a['record_date'] == day:
                entitlements[a['event_id']] = quantities.get(a['code'], 0)
    return output, fills


def main():
    import pandas as pd
    from math import isclose
    out = ROOT / 'reports/execution_followup_20260923'
    assert not (out / 'independent_reference.json').exists()
    market = read(out / 'replay/market.json')
    actions = read(ROOT / 'reports/execution_events_20260923/approved_actions.json')['actions']
    actions += read(out / 'validation.json')['new_actions']
    expected = {r['signal']: read(out / 'replay/ledgers' / (r['signal'] + '.json')) for r in market['rows']}
    maximum = max(r['date'] for book in expected.values() for r in book['days'] if r['nav'] is not None)
    targets = {name: read(out / 'replay/targets' / (name + '.json'))['targets'] for name in expected}
    codes = {c for groups in targets.values() for day, w in groups.items() if day <= maximum and w for c in w}

    def panel(name, columns):
        pieces = []
        for part in pd.read_csv(ROOT / name, usecols=['ts_code', 'trade_date'] + columns,
                                dtype={'trade_date': str}, chunksize=300000):
            part = part[part.ts_code.isin(codes)].copy()
            part['trade_date'] = pd.to_datetime(part.trade_date, format='mixed').dt.strftime('%Y-%m-%d')
            pieces.append(part[part.trade_date.between('2024-12-20', maximum)])
        frame = pd.concat(pieces)
        assert not frame.duplicated(['trade_date', 'ts_code']).any()
        return frame.set_index(['trade_date', 'ts_code']).to_dict('index')

    quotes = panel('data/raw/daily_prices.csv', ['open', 'close'])
    limits = panel('data/raw/stk_limit.csv', ['up_limit', 'down_limit'])
    susp = pd.read_csv(ROOT / 'data/raw/suspend_d.csv', dtype=str).fillna('')
    susp['trade_date'] = pd.to_datetime(susp.trade_date, format='mixed').dt.strftime('%Y-%m-%d')
    halts = {(r.trade_date, r.ts_code) for r in susp.itertuples()
             if r.suspend_type == 'S' and not r.suspend_timing}
    comparisons = []
    for name, book in expected.items():
        finite = [r for r in book['days'] if r['nav'] is not None]
        assert len(finite) > 0
        independent, fills = simulate([r['date'] for r in finite], targets[name], quotes, limits, halts, actions)
        for actual, ref in zip(finite, independent):
            assert actual['date'] == ref['date']
            assert {c: v['shares'] for c, v in actual['positions'].items()} == ref['positions']
            for key in ['cash', 'receivable', 'nav']:
                assert isclose(actual[key], ref[key], abs_tol=1e-6, rel_tol=0), (name, actual['date'], key)
        last = finite[-1]['date']
        primary_fills = [{k: r[k] for k in ['date', 'code', 'quantity', 'price', 'fee']}
                         for r in book['journal'] if r['kind'] == 'fill' and r['date'] <= last]
        assert len(primary_fills) == len(fills) > 0
        for a, b in zip(primary_fills, fills):
            for key in ['date', 'code', 'quantity']:
                assert a[key] == b[key]
            for key in ['price', 'fee']:
                assert isclose(a[key], b[key], abs_tol=1e-6, rel_tol=0)
        comparisons.append({'signal': name, 'days': len(finite), 'fills': len(fills), 'end': last})
    write(out / 'independent_reference.json', {'comparisons': comparisons, 'code_sha256': digest(Path(__file__)),
          'execution_independent_of_primary_journal': True, 'arithmetic': 'Decimal',
          'coverage': 'finite cash-only prefixes only; excludes terminal unknown day',
          'input_contract': 'same raw quotes/limits/halt evidence and frozen targets; not independent data source',
          'full_window_acceptance': False, 'formal_eligibility': False})
    print(comparisons)


if __name__ == '__main__':
    main()
