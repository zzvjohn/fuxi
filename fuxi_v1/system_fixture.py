"""Long synthetic fixture for integration, recovery, policies and model comparison."""

from datetime import date, timedelta
import math
from .contracts import digest
from .data import Snapshot
from .datasets import execution_profile, materialize_labels


def synthetic_campaign(sessions_count=150, instruments_count=8):
    sessions, day = [], date(2023, 1, 3)
    while len(sessions) < sessions_count:
        if day.weekday() < 5:
            sessions.append(day.isoformat())
        day += timedelta(days=1)
    codes = [str(600000+j)+".XSHG" for j in range(instruments_count)]
    observations, bars = [], []
    for i, day in enumerate(sessions):
        for j, code in enumerate(codes):
            close = 20 + j + .018*i*(j-instruments_count/2) + .4*math.sin(i*.2+j)
            volume = 1000000 + 300000*(1+math.sin(i*.17+j))
            for field, value, unit in (("close", close, "price"), ("volume", volume, "shares")):
                observations.append({"session": day, "instrument": code, "field": field, "value": value,
                                     "available_at": day+"T18:00:00+08:00", "revision_id": "synthetic_v1"})
            bars.append({"session": day, "instrument": code, "open": close-.04*math.sin(i*.3+j),
                         "close": close, "high": close*1.02, "low": close*.98, "volume": volume,
                         "up_limit": close*1.1, "down_limit": close*.9, "paused": False,
                         "corporate_action_known": True, "action_ids": [], "evidence_id": "synthetic_"+day+code})
    snapshot = {"schema_version": "fuxi.snapshot.1", "source_id": "synthetic_system_fixture", "exposure": "synthetic",
                "sessions": sessions, "decision_times": {d:d+"T18:30:00+08:00" for d in sessions},
                "universe": {d:list(codes) for d in sessions}, "evaluation_start": sessions[20],
                "fields": {"close": {"unit": "price", "frequency": "daily", "adjustment": "raw"},
                           "volume": {"unit": "shares", "frequency": "daily", "adjustment": "raw"}},
                "observations": observations, "quality": {"pit_certified": False, "synthetic_fixture": True}}
    market = {"sessions": sessions, "bars": bars, "actions": [], "execution": execution_profile(),
              "event_coverage_certified": True, "tradability_certified": True}
    market["market_hash"] = digest(market)
    labels = materialize_labels(Snapshot(snapshot), market, 5)
    return {"schema_version": "fuxi.campaign_input.1", "snapshot": snapshot, "labels": labels, "market": market,
            "research_clock": sessions[-1]+"T20:00:00+08:00"}
