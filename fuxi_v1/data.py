"""Explicit session/universe/PIT inputs; no data fetching or missing-value fill."""

from copy import deepcopy
from datetime import date
from .contracts import digest, keys, require, finite, nonempty, timestamp

# C2 软收敛：合法 unit 集合真值源 = types_core.spec.VALID_UNITS（单点）。
# 本处保留硬编码是因为 jq.py.compile_diagnostic 会把 data.py 源码内联（过滤 `from .`
# 相对导入），data.py 必须内联自包含、不能依赖 types_core 包。改量纲格时两处同步改。
_VALID_UNITS = {"price", "shares", "ratio", "currency"}


def snapshot_from_payload(payload):
    """Use the declared schema; compact history retains action-lookback barriers."""
    if payload.get('schema_version') == 'fuxi.historical_snapshot.1':
        from .historical_adapter import HistoricalSnapshot
        return HistoricalSnapshot(payload)
    return Snapshot(payload)


class Snapshot:
    def __new__(cls, payload):
        if payload.get("schema_version") == "fuxi.columnar_snapshot.1":
            from .columnar import ColumnarSnapshot
            return ColumnarSnapshot(payload)
        return super().__new__(cls)

    def __init__(self, payload):
        self.payload = deepcopy(payload)
        p = self.payload
        keys(p, ("schema_version", "source_id", "exposure", "sessions", "decision_times",
                 "universe", "fields", "observations", "evaluation_start"), ("quality", "provenance"))
        require(p["schema_version"] in {"fuxi.snapshot.1", "fuxi.snapshot.2"}, "unsupported snapshot schema")
        versioned = p["schema_version"] == "fuxi.snapshot.2"
        require(nonempty(p["source_id"]), "source required")
        require(p["exposure"] in {"synthetic", "research_exposed", "forward_sealed"}, "declare exposure")
        if p["exposure"] == "forward_sealed":
            require(versioned and p.get("quality", {}).get("protection_receipt"), "forward scope receipt required")
        self.sessions = p["sessions"]
        require(isinstance(self.sessions, list) and len(self.sessions) > 0, "empty sessions")
        require(all(isinstance(d, str) and date.fromisoformat(d).isoformat() == d
                    for d in self.sessions), "invalid session date")
        require(self.sessions == sorted(set(self.sessions)), "sessions must be unique and sorted")
        require(p["evaluation_start"] in self.sessions, "evaluation_start outside calendar")
        self.start = self.sessions.index(p["evaluation_start"])
        require(set(p["decision_times"]) == set(self.sessions), "decision times/calendar mismatch")
        self.clocks = {d: timestamp(p["decision_times"][d]) for d in self.sessions}
        require(all(t.date().isoformat() == d for d, t in self.clocks.items()),
                "decision time must belong to session date")
        require(list(self.clocks.values()) == sorted(self.clocks.values()), "clock order mismatch")
        require(set(p["universe"]) == set(self.sessions), "universe/calendar mismatch")
        self.universe = p["universe"]
        for members in self.universe.values():
            require(isinstance(members, list) and len(members) > 0 and
                    all(nonempty(x) for x in members) and len(set(members)) == len(members),
                    "empty or duplicate universe members")
        require(isinstance(p["fields"], dict) and len(p["fields"]) > 0, "empty field catalog")
        for name, spec in p["fields"].items():
            require(nonempty(name), "invalid field name")
            keys(spec, ("unit", "adjustment", "frequency"))
            require(spec["unit"] in _VALID_UNITS, "unsupported unit")
            require(spec["frequency"] == "daily", "only daily fields supported")
            require(spec["adjustment"] in {"raw", "not_applicable"},
                    "adjusted prices require a future revision-aware adapter")
        self.cells = {}
        require(isinstance(p["observations"], list), "observations must be a list")
        for row in p["observations"]:
            keys(row, ("session", "instrument", "field", "value", "available_at", "revision_id"),
                 ("event_time", "published_at", "ingested_at", "quality_status", "source_id"))
            d, instrument, field = row["session"], row["instrument"], row["field"]
            require(d in self.clocks and nonempty(instrument) and field in p["fields"],
                    "observation outside calendar/catalog")
            require(row["value"] is None or finite(row["value"]), "nonfinite observed value")
            require(nonempty(row["revision_id"]), "revision lineage required")
            available = timestamp(row["available_at"])
            require(available.date().isoformat() >= d, "available before observation session")
            key = (d, instrument, field)
            if versioned:
                require(all(k in row for k in ("event_time", "published_at", "ingested_at", "quality_status", "source_id")),
                        "versioned observation requires bitemporal provenance")
                event, published, ingested = (timestamp(row[k]) for k in ("event_time", "published_at", "ingested_at"))
                require(event <= published <= available and ingested >= published, "invalid bitemporal sequence")
                require(row["quality_status"] in {"historically_certified", "research_limited", "invalid"}, "invalid quality")
                require(row["quality_status"] != "invalid" or row["value"] is None, "invalid observations must be null")
                revisions = self.cells.setdefault(key, [])
                require(not any(x[2] == row["revision_id"] or x[1] == available for x in revisions),
                        "duplicate/ambiguous revision ordering")
                revisions.append((row["value"], available, row["revision_id"]))
                revisions.sort(key=lambda x: x[1])
            else:
                require(key not in self.cells, "multiple cell revisions require a revision-aware adapter")
                self.cells[key] = (row["value"], available)
        self.hash = digest(p)

    def value(self, index, instrument, field, asof):
        if index < 0 or index >= len(self.sessions):
            return None
        item = self.cells.get((self.sessions[index], instrument, field))
        if isinstance(item, list):
            eligible = [row for row in item if row[1] <= asof]
            return eligible[-1][0] if eligible else None
        return None if item is None or item[1] > asof else item[0]


class Labels:
    def __new__(cls, payload, snapshot):
        if payload.get("schema_version") == "fuxi.columnar_labels.1":
            from .columnar import ColumnarLabels
            return ColumnarLabels(payload, snapshot)
        return super().__new__(cls)

    def __init__(self, payload, snapshot):
        self.payload = deepcopy(payload)
        p = self.payload
        keys(p, ("schema_version", "feature_snapshot_hash", "label_spec", "rows"))
        require(p["schema_version"] == "fuxi.labels.1", "unsupported labels schema")
        require(p["feature_snapshot_hash"] == snapshot.hash, "label/snapshot lineage mismatch")
        keys(p["label_spec"], ("definition", "horizon_sessions", "execution_contract_hash",
                              "kind", "price_basis"))
        spec = p["label_spec"]
        require(nonempty(spec["definition"]) and nonempty(spec["execution_contract_hash"]),
                "label definition/execution reference required")
        require(type(spec["horizon_sessions"]) is int and spec["horizon_sessions"] > 0,
                "positive label horizon required")
        require(spec["kind"] == "prediction_proxy", "account labels not supported in this adapter")
        require(spec["price_basis"] == "declared_in_definition", "explicit label price basis required")
        require(isinstance(p["rows"], list), "label rows must be a list")
        self.rows = {}
        universe_members = {day: set(codes) for day, codes in snapshot.universe.items()}
        for row in p["rows"]:
            keys(row, ("session", "instrument", "decision_time", "first_eligible_execution_time",
                       "label_start", "label_end", "mature_at", "value", "status", "evidence_id"))
            d, instrument = row["session"], row["instrument"]
            require(d in snapshot.clocks and instrument in universe_members[d], "label outside universe")
            times = [timestamp(row[k]) for k in ("decision_time", "first_eligible_execution_time",
                                                 "label_start", "label_end", "mature_at")]
            require(times[0] == snapshot.clocks[d], "label decision mismatch")
            require(times == sorted(times) and times[0] < times[1] and times[2] < times[3],
                    "label chronology violation")
            require(row["status"] in {"valid", "not_mature", "data_insufficient", "invalid"},
                    "invalid label status")
            require(finite(row["value"]) if row["status"] == "valid" else row["value"] is None,
                    "nonvalid labels must have null value; valid labels must be finite")
            require(nonempty(row["evidence_id"]), "label evidence required")
            key = (d, instrument)
            require(key not in self.rows, "duplicate label")
            self.rows[key] = row
        self.hash = digest(p)

    def value(self, session, instrument, research_clock):
        row = self.rows.get((session, instrument))
        if row is None or row["status"] != "valid" or timestamp(row["mature_at"]) > research_clock:
            return None
        return row["value"]
