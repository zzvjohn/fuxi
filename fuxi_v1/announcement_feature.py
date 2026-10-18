"""One existing structured information source, with coarse-date research limits."""
import csv
from collections import Counter
from pathlib import Path
from .contracts import digest, finite
from .datasets import file_hash
from .history import jq_code, iso


def channel(path, snapshot, *, lifespan_sessions=20):
    """Last implemented cash amount/known raw close. No announcement means missing.

    Dates lack historic download and exact publication clocks, so these signals
    are strictly a sensitivity diagnostic; no availability certificate is minted.
    """
    names = set().union(*(set(v) for v in snapshot.universe.values()))
    events, rejected, input_count = {}, Counter(), 0
    with Path(path).open(encoding='utf-8-sig',newline='') as stream:
        for row in csv.DictReader(stream):
            input_count += 1
            code = jq_code(row.get('ts_code',''))
            if code not in names:
                continue
            day = iso(row.get('imp_ann_date'))
            if row.get('div_proc') != '实施' or not day or not row.get('cash_div_tax'):
                rejected['implemented_cash_or_announcement_missing'] += 1
                continue
            try:
                value = float(row['cash_div_tax'])
            except ValueError:
                rejected['invalid_cash'] += 1
                continue
            if not finite(value) or value < 0:
                rejected['negative_or_nonfinite_cash'] += 1
                continue
            identity = digest(row)
            events.setdefault(code,[]).append({'published_date':day,'cash_per_share':value,'evidence_id':identity})
    panel, evidence = {}, []
    days = snapshot.sessions
    for i in range(snapshot.start,len(days)):
        day = days[i]
        cross = {}
        for code in snapshot.universe[day]:
            possible = [r for r in events.get(code,[]) if days[max(0,i-lifespan_sessions)] <= r['published_date'] < day]
            # Conflicting terms on the same date remain unknown; no first-row winner.
            possible.sort(key=lambda r:r['published_date'])
            latest = possible[-1] if possible else None
            same_day = [r for r in possible if r['published_date']==latest['published_date']] if latest else []
            ambiguous = len({r['cash_per_share'] for r in same_day}) > 1
            close = snapshot.value(i,code,'close',snapshot.clocks[day])
            cross[code] = latest['cash_per_share']/close if latest and not ambiguous and close is not None and close > 0 else None
            if cross[code] is not None:
                evidence.append({'session':day,'instrument':code, **latest,
                                 'earliest_trade':'strictly after '+day+' decision_time'})
        panel[day] = cross
    manifest = {'schema_version':'fuxi.information_source_trial.1', 'source_path':str(Path(path).resolve()),
                'source_sha256':file_hash(path),'input_rows':input_count,'cohort_event_rows':sum(map(len,events.values())),
                'usable_cells':sum(v is not None for cross in panel.values() for v in cross.values()),
                'required_cells':sum(map(len,panel.values())), 'rejections':dict(rejected),
                'field':'announcement_cash_yield','definition':'latest implemented cash per share / PIT known raw close; 20-session horizon',
                'missing_policy':'propagate; absence never zero','source_kind':'existing_structured_implementation_announcements',
                'availability':'date-only; at least one session delay; intraday and historic vendor-vintage unverified',
                'historical_availability_certified':False,'formal_use_allowed':False,'evidence':evidence,
                'selection_before_performance':'coverage/clock/mechanism/cost; no optimization or acquisition'}
    manifest['manifest_hash'] = digest(manifest)
    return panel, manifest
