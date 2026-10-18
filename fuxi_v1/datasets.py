"""Read-only native market adapter and explicitly timed label materialization."""

from datetime import datetime, timezone
import hashlib
from pathlib import Path
from .contracts import digest, require, finite, timestamp
from .data import Snapshot


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def scope_of(snapshot, label_end=None):
    p = snapshot.payload
    # The canonical dependency survives vendor/source renaming or derived snapshot hashes.
    return {"sources": ["market_observations", p["source_id"]], "instruments": sorted({s for u in snapshot.universe.values() for s in u}),
            "intervals": [[p["sessions"][0]+"T00:00:00+08:00", p["sessions"][-1]+"T23:59:59+08:00"]],
            "label_intervals": [[p["sessions"][0]+"T00:00:00+08:00", label_end or p["sessions"][-1]+"T23:59:59+08:00"]],
            "resolved": True}


def materialize_labels(snapshot, market, horizon=5):
    require(type(horizon) is int and horizon > 0, "invalid label horizon")
    bars = {(r["session"], r["instrument"]): r for r in market["bars"]}
    sessions, rows = snapshot.sessions, []
    for index, d in enumerate(sessions):
        if index + 1 + horizon >= len(sessions):
            continue  # Absence remains explicit in requested-section/coverage denominators.
        start, end = sessions[index+1], sessions[index+1+horizon]
        for code in snapshot.universe[d]:
            entry, exit_bar = bars.get((start, code), {}), bars.get((end, code), {})
            supporting = [bars.get((day, code), {}) for day in sessions[index+1:index+2+horizon]]
            complete = (entry.get("open") is not None and exit_bar.get("close") is not None
                        and entry.get("open", 0) > 0 and all(r.get("corporate_action_known") is True
                                                          and not r.get("action_ids") for r in supporting))
            value = exit_bar["close"]/entry["open"]-1 if complete else None
            rows.append(dict(session=d, instrument=code, decision_time=snapshot.payload["decision_times"][d],
                             first_eligible_execution_time=start+"T09:30:00+08:00", label_start=start+"T09:30:00+08:00",
                             label_end=end+"T15:00:00+08:00", mature_at=end+"T18:00:00+08:00",
                             value=value, status="valid" if value is not None else "data_insufficient",
                             evidence_id=digest({"market": market["market_hash"], "code": code, "start": start, "end": end})))
    return dict(schema_version="fuxi.labels.1", feature_snapshot_hash=snapshot.hash,
                label_spec={"definition": "next-session raw open to horizon-session raw close; corporate-action spans excluded; price proxy, no transaction costs",
                            "horizon_sessions": horizon, "execution_contract_hash": digest(market["execution"]),
                            "kind": "prediction_proxy", "price_basis": "declared_in_definition"}, rows=rows)


def validate_label_calendar(labels, sessions):
    if labels.get("schema_version") == "fuxi.columnar_labels.1":
        from .columnar import validate_columnar_label_calendar
        return validate_columnar_label_calendar(labels, sessions)
    index = {d: i for i, d in enumerate(sessions)}
    horizon = labels["label_spec"]["horizon_sessions"]
    for row in labels["rows"]:
        start = timestamp(row["label_start"]).date().isoformat()
        end = timestamp(row["label_end"]).date().isoformat()
        require(start in index and end in index and index[end]-index[start] == horizon, "label/calendar horizon mismatch")
        require(index[start] == index[row["session"]]+1, "label must start on next supplied trading session")


def execution_profile():
    # Explicit research assumptions, not a claim of current broker terms or global defaults.
    return {"schema_version": "fuxi.execution.1", "initial_cash": "1000000", "buy_commission": "0.0003",
            "sell_commission": "0.0003", "sell_tax": "0.0005", "minimum_commission": "5",
            "slippage_bps": "5", "participation": "0.01", "buy_lot": 100, "sell_lot": 100,
            "settlement_sessions": 1, "cash_reallocation": "none", "price_model": "next_session_open",
            "liquidity_basis": "prior_session_volume",
            "cost_status": "declared_research_assumptions_not_broker_certified"}


def read_calendar(path, start, end):
    import pandas as pd
    from datetime import date
    frame = pd.read_csv(path, dtype=str)
    repeated = (frame["cal_date"] == "cal_date") & (frame["is_open"] == "is_open")
    removed_headers = int(repeated.sum())
    frame = frame[~repeated]
    require(frame["is_open"].isin(["0","1"]).all(), "invalid calendar open flag")
    def normalize(value):
        require(len(value) == 8 and value.isdigit(), "invalid calendar date")
        result = value[:4]+"-"+value[4:6]+"-"+value[6:]
        date.fromisoformat(result)
        return result
    days = sorted({normalize(d) for d in frame.loc[frame.is_open == "1", "cal_date"]})
    return [d for d in days if start <= d <= end], {"embedded_header_rows_ignored":removed_headers,
            "reason":"exact CSV header rows are serialization artifacts, not trading observations"}


def local_market_input(raw_root, start, end, max_instruments=24, horizon=5):
    """Bounded historical sample. Initial universe depends only on the first session.

    CSV channels lack archive-PIT/event completeness proof. Returned provenance
    preserves this limitation; no economic eligibility is granted by this loader.
    """
    import pandas as pd
    root = Path(raw_root)
    required = [root / name for name in ("daily_prices.csv", "trade_calendar.csv", "stk_limit.csv", "adj_factor_daily.csv", "suspend_d.csv")]
    require(all(p.is_file() for p in required), "required local data channel missing")
    before = {p.name: (p.stat().st_size, p.stat().st_mtime_ns) for p in required}
    sessions, calendar_audit = read_calendar(root/"trade_calendar.csv",start,end)
    require(len(sessions) > 60 and max_instruments >= 6, "sample too small for this adapter")
    compact = {d.replace("-", "") for d in sessions}
    first = sessions[0].replace("-", "")
    selected, parts = None, []
    # First-pass bounded extraction; no universe chosen from future returns/availability.
    for frame in pd.read_csv(root / "daily_prices.csv", dtype={"trade_date": str, "ts_code": str}, chunksize=250000):
        part = frame[frame.trade_date.isin(compact)]
        if len(part):
            parts.append(part)
    prices = pd.concat(parts, ignore_index=True)
    initial = prices[(prices.trade_date == first) & prices.ts_code.str.match(r"^(000|600)\d{3}\.(SZ|SH)$")]
    codes = sorted(set(initial.ts_code))
    # P-20261017（根源修复）：字符串排序抽样有两个缺陷——(1) 000*.SZ 字典序恒 < 600*.SH，
    # 取前 N 只全部落深市、沪市一只没有，E1 的 screen_positive_exchange_count=2（沪深两所
    # 都要正 IC）必然 reject；(2) 即便按交易所分半，`sz[:half]` 仍取代码序最小的老股票，
    # 引入幸存者偏差。改为分交易所 hash 抽样（固定 salt，可复现、无幸存者偏差），与
    # history.select_cohort 的抽样纪律一致。
    def _score(code):
        return hashlib.sha256(("fuxi-local-market-input-20261017:" + code).encode()).hexdigest()
    sz = sorted((c for c in codes if c.endswith(".SZ")), key=_score)
    sh = sorted((c for c in codes if c.endswith(".SH")), key=_score)
    half = max_instruments // 2
    selected = sz[:half] + sh[:half]
    if len(selected) < max_instruments:
        rest = sorted((set(codes) - set(selected)), key=_score)
        selected += rest[:max_instruments - len(selected)]
    require(len(selected) >= 6, "initial-session main-board sample unavailable")
    require(any(c.endswith(".SZ") for c in selected) and any(c.endswith(".SH") for c in selected),
            "initial-session sample must span both SZ and SH exchanges")
    prices = prices[prices.ts_code.isin(selected)]
    require(not prices.duplicated(["trade_date", "ts_code"]).any(), "duplicate raw price cell")
    def subset(name, columns):
        frames = []
        for frame in pd.read_csv(root / name, dtype={"trade_date": str, "ts_code": str}, usecols=columns, chunksize=250000):
            part = frame[frame.trade_date.isin(compact) & frame.ts_code.isin(selected)]
            if len(part):
                frames.append(part)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns)
    limits = subset("stk_limit.csv", ["ts_code", "trade_date", "up_limit", "down_limit"])
    adj = subset("adj_factor_daily.csv", ["ts_code", "trade_date", "adj_factor"])
    for frame in (limits, adj):
        require(not frame.duplicated(["trade_date", "ts_code"]).any(), "duplicate supporting channel cell")
    limit_map = {(r.trade_date, r.ts_code): r for r in limits.itertuples()}
    adj_map = {(r.trade_date, r.ts_code): r.adj_factor for r in adj.itertuples()}
    price_map = {(r.trade_date, r.ts_code): r for r in prices.itertuples()}
    hashes = {p.name: file_hash(p) for p in required}
    require(before == {p.name: (p.stat().st_size, p.stat().st_mtime_ns) for p in required}, "raw channels changed during snapshot")
    observations, bars = [], []
    ingested = datetime.now(timezone.utc).isoformat()
    def normal_code(s):
        return s.replace(".SZ", ".XSHE").replace(".SH", ".XSHG")
    instruments = [normal_code(s) for s in selected]
    for index, day in enumerate(sessions):
        d = day.replace("-", "")
        for code in selected:
            row, limit = price_map.get((d, code)), limit_map.get((d, code))
            def value(name, multiple=1):
                v = float(getattr(row, name)) * multiple if row is not None else None
                return v if finite(v) else None
            previous = sessions[max(0, index-1)].replace("-", "")
            a, b = adj_map.get((d, code)), adj_map.get((previous, code))
            unchanged = finite(a) and finite(b) and a == b
            bar = dict(session=day, instrument=normal_code(code), open=value("open"), close=value("close"),
                       high=value("high"), low=value("low"), volume=value("vol", 100),
                       up_limit=float(limit.up_limit) if limit is not None else None,
                       down_limit=float(limit.down_limit) if limit is not None else None,
                       # Positive actual volume is a research tradability proxy, not a certified suspension table.
                       paused=(value("vol") == 0) if row is not None else None,
                       corporate_action_known=bool(unchanged), action_ids=[],
                       evidence_id=digest([hashes["daily_prices.csv"], day, code]),
                       event_coverage="adj_unchanged_proxy_only")
            bars.append(bar)
            for field, v in (("close", value("close")), ("open", value("open")), ("high", value("high")),
                             ("low", value("low")), ("volume", value("vol", 100)), ("amount", value("amount", 1000))):
                observations.append(dict(session=day, instrument=normal_code(code), field=field, value=v,
                                         event_time=day+"T15:00:00+08:00", published_at=day+"T18:00:00+08:00",
                                         available_at=day+"T18:00:00+08:00", ingested_at=ingested,
                                         revision_id=hashes["daily_prices.csv"], source_id="local_daily_prices",
                                         quality_status="research_limited"))
    snapshot = dict(schema_version="fuxi.snapshot.2", source_id="legacy_raw_csv_research_exposed",
                    exposure="research_exposed", sessions=sessions,
                    decision_times={d: d+"T18:30:00+08:00" for d in sessions},
                    universe={d: list(instruments) for d in sessions}, evaluation_start=sessions[20],
                    fields={name: {"unit": "shares" if name == "volume" else "currency" if name == "amount" else "price",
                                   "adjustment": "raw", "frequency": "daily"} for name in ("open", "high", "low", "close", "volume", "amount")},
                    observations=observations,
                    quality={"pit_certified": False, "universe_certified": False, "account_events_certified": False,
                             "gaps": ["historical_revision_archive_unverified", "suspension_semantics_unverified",
                                      "corporate_action_entitlement_coverage_unverified", "ST_listing_point_in_time_unverified"]},
                    provenance={"source_hashes": hashes, "selection": "first-session main-board code order; no future selection",
                                "calendar_parse_audit":calendar_audit,
                                "availability": "conservative 18:00 assumption; not historical publication proof",
                                "units": {"vol": "hands*100 -> shares", "amount": "thousand_CNY*1000 -> CNY"}})
    market = {"sessions": sessions, "bars": bars, "actions": [], "execution": execution_profile(),
              "event_coverage_certified": False, "tradability_certified": False}
    market["market_hash"] = digest(market)
    snap = Snapshot(snapshot)
    labels = materialize_labels(snap, market, horizon)
    validate_label_calendar(labels, sessions)
    return {"schema_version": "fuxi.campaign_input.1", "snapshot": snapshot, "labels": labels, "market": market,
            "research_clock": sessions[-1]+"T20:00:00+08:00"}
