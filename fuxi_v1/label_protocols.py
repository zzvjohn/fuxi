"""Label protocol registry for P0.2 (v2.0 landing).

The R2 blueprint (BLUEPRINT.md "时点和证据") requires the two label protocols to
be filed separately: the *current native default* implemented in ``datasets.py``
(t+1 raw open to t+6 raw close, mature t+6 18:00, corporate-action spans
excluded) and the *target* five-open-interval price label (t+1 open to t+6 open).
Historical IC values cannot silently carry over between them.

This module does NOT change the production default: ``datasets.materialize_labels``
stays untouched and remains the only label source on the production path. The
target materializer here is a read-only sibling used for protocol comparison and
future LabelSpec-driven runs.
"""

from .contracts import digest, keys, require

LABEL_SPEC_SCHEMA = "fuxi.label_spec.1"

# Protocol ids are content identities: changing any semantic element of a
# protocol (clock, maturity, exclusion rule) MUST change its id.
PROTOCOLS = {
    "legacy_t1_open_t6_close": {
        "protocol_id": "fuxi.label_protocol.legacy_t1_open_t6_close.1",
        "label_spec": {"target": "forward_interval_return",
                       "horizon_sessions": 5,
                       "entry_price_clock": "open",
                       "exit_price_clock": "close",
                       "maturity_at": "exit_session_18_00",
                       "return_basis": "raw_price_proxy",
                       "corporate_action_rule": "exclude_spans",
                       "definition": "next-session raw open to horizon-session raw close; "
                                     "corporate-action spans excluded; price proxy, no transaction costs"},
        "implemented_in": "fuxi_v1.datasets.materialize_labels",
        "status": "native_default",
    },
    "target_five_open_intervals": {
        "protocol_id": "fuxi.label_protocol.target_five_open_intervals.1",
        "label_spec": {"target": "forward_interval_return",
                       "horizon_sessions": 5,
                       "entry_price_clock": "open",
                       "exit_price_clock": "open",
                       "maturity_at": "exit_session_18_00",
                       "return_basis": "raw_price_proxy",
                       "corporate_action_rule": "exclude_spans",
                       "definition": "next-session raw open to horizon-session raw open "
                                     "(five open intervals); corporate-action spans excluded; "
                                     "price proxy, no transaction costs"},
        "implemented_in": "fuxi_v1.label_protocols.materialize_labels_target",
        "status": "target_candidate",
    },
}


def validate_label_spec(spec):
    keys(spec, ("target", "horizon_sessions", "entry_price_clock", "exit_price_clock",
                "maturity_at", "return_basis", "corporate_action_rule", "definition"))
    require(spec["target"] == "forward_interval_return", "only forward_interval_return supported")
    require(type(spec["horizon_sessions"]) is int and spec["horizon_sessions"] > 0,
            "horizon_sessions must be a positive int")
    require(spec["entry_price_clock"] in ("open", "close"), "invalid entry_price_clock")
    require(spec["exit_price_clock"] in ("open", "close"), "invalid exit_price_clock")
    require(spec["maturity_at"] == "exit_session_18_00", "unsupported maturity clock")
    require(spec["return_basis"] == "raw_price_proxy", "unsupported return basis")
    require(spec["corporate_action_rule"] == "exclude_spans", "unsupported corporate action rule")
    digest(spec)
    return spec


def protocol_identity(protocol):
    """Content identity of a protocol entry (label_spec + implementer + status)."""
    validate_label_spec(protocol["label_spec"])
    return digest({"protocol_id": protocol["protocol_id"],
                   "label_spec": protocol["label_spec"],
                   "implemented_in": protocol["implemented_in"],
                   "status": protocol["status"]})


def materialize_labels_target(snapshot, market, horizon=5):
    """Target five-open-interval label materializer (read-only sibling).

    Identical completeness logic to ``datasets.materialize_labels`` except the
    exit price clock is the *open* of session t+horizon instead of the close.
    """
    require(type(horizon) is int and horizon > 0, "invalid label horizon")
    protocol = PROTOCOLS["target_five_open_intervals"]
    bars = {(r["session"], r["instrument"]): r for r in market["bars"]}
    sessions, rows = snapshot.sessions, []
    for index, d in enumerate(sessions):
        if index + 1 + horizon >= len(sessions):
            continue
        start, end = sessions[index + 1], sessions[index + 1 + horizon]
        for code in snapshot.universe[d]:
            entry, exit_bar = bars.get((start, code), {}), bars.get((end, code), {})
            supporting = [bars.get((day, code), {}) for day in sessions[index + 1:index + 2 + horizon]]
            complete = (entry.get("open") is not None and exit_bar.get("open") is not None
                        and entry.get("open", 0) > 0
                        and all(r.get("corporate_action_known") is True
                                and not r.get("action_ids") for r in supporting))
            value = exit_bar["open"] / entry["open"] - 1 if complete else None
            rows.append(dict(session=d, instrument=code,
                             decision_time=snapshot.payload["decision_times"][d],
                             first_eligible_execution_time=start + "T09:30:00+08:00",
                             label_start=start + "T09:30:00+08:00",
                             label_end=end + "T09:30:00+08:00",
                             mature_at=end + "T18:00:00+08:00",
                             value=value,
                             status="valid" if value is not None else "data_insufficient",
                             evidence_id=digest({"market": market["market_hash"], "code": code,
                                                 "start": start, "end": end})))
    return dict(schema_version="fuxi.labels.1", feature_snapshot_hash=snapshot.hash,
                label_spec=dict(protocol["label_spec"], protocol_id=protocol["protocol_id"]),
                rows=rows)
