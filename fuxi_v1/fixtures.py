"""Deterministic synthetic fixtures. Never market data or alpha evidence."""

from datetime import date, timedelta
import math
from .data import Snapshot
from .evaluation import DEFAULT_POLICY
from .contracts import digest


def synthetic_input():
    sessions = []
    day = date(2024, 1, 2)
    while len(sessions) < 24:
        if day.weekday() < 5:
            sessions.append(day.isoformat())
        day += timedelta(days=1)
    stocks = ["000001.XSHE", "000002.XSHE", "600000.XSHG", "600001.XSHG", "600002.XSHG", "600003.XSHG"]
    observations = []
    prices = {}
    for i, d in enumerate(sessions):
        for j, stock in enumerate(stocks):
            value = 10 + j + 0.04*i*(j-2) + math.sin(i*0.65+j)*0.3
            prices[(i, stock)] = value
            observations.append(dict(session=d, instrument=stock, field="close", value=value,
                                     available_at=d+"T15:05:00+08:00", revision_id="synthetic_r1"))
    snapshot = dict(schema_version="fuxi.snapshot.1", source_id="synthetic_fixture_not_market",
                    exposure="synthetic", sessions=sessions,
                    decision_times={d: d+"T16:00:00+08:00" for d in sessions},
                    universe={d: list(stocks) for d in sessions}, evaluation_start=sessions[5],
                    fields={"close": {"unit": "price", "adjustment": "raw", "frequency": "daily"}},
                    observations=observations)
    rows = []
    for i, d in enumerate(sessions[:-2]):
        for stock in stocks:
            rows.append(dict(session=d, instrument=stock, decision_time=snapshot["decision_times"][d],
                             first_eligible_execution_time=sessions[i+1]+"T09:30:00+08:00",
                             label_start=sessions[i+1]+"T15:00:00+08:00",
                             label_end=sessions[i+2]+"T15:00:00+08:00",
                             mature_at=sessions[i+2]+"T15:05:00+08:00",
                             value=prices[(i+2, stock)]/prices[(i+1, stock)]-1,
                             status="valid", evidence_id="synthetic_label_"+d+"_"+stock))
    labels = dict(schema_version="fuxi.labels.1", feature_snapshot_hash=Snapshot(snapshot).hash,
                  label_spec={"definition": "synthetic next-session close to following close; no costs; not an account return",
                              "horizon_sessions": 1, "execution_contract_hash": digest("synthetic_proxy_only"),
                              "kind": "prediction_proxy", "price_basis": "declared_in_definition"}, rows=rows)
    return dict(schema_version="fuxi.input.1", snapshot=snapshot, labels=labels,
                policy=dict(DEFAULT_POLICY), research_clock=sessions[-1]+"T18:00:00+08:00")
