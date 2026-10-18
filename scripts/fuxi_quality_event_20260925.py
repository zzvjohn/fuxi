"""One bounded Tushare event request for the first cash-ledger blocker."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "research/factor_alchemy"))
from fuxi_event_evidence_20260923 import FIELDS, checked_cash_action
from run_contract import RunContract

OUT = ROOT / "reports/fuxi_quality_gate_20260925/event_920718"
CODE, DAY = "920718.BJ", "2025-01-15"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_once(path, value):
    if path.exists():
        raise RuntimeError("immutable event evidence exists: " + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n",
                    encoding="utf-8", newline="\n")


def prepare():
    result = ROOT / "reports/fuxi_quality_gate_20260925/economic_probe_v2.json"
    row = json.loads(result.read_text(encoding="utf-8"))
    assert row["errors"][0] == {"date": DAY, "code": CODE,
                                 "reason": "missing_corporate_action_event"}
    write_once(OUT / "protocol.json", {
        "schema": "fuxi_one_event_probe/v1",
        "created_at": datetime.now().astimezone().isoformat(),
        "source_result_sha256": sha(result), "code_sha256": sha(Path(__file__)),
        "adapter_sha256": sha(ROOT / "scripts/fuxi_event_evidence_20260923.py"),
        "request": {"api": "dividend", "ts_code": CODE, "fields": FIELDS,
                    "target_ex_date": DAY, "max_requests": 1, "row_cap": 2000},
        "price_sha256": sha(ROOT / "data/raw/daily_prices.csv"),
        "purpose": "resolve first held-position event only; no full-window eligibility",
    })
    print("Frozen one Tushare dividend request for", CODE, DAY)


def collect():
    from credentials import get_tushare_api

    p = json.loads((OUT / "protocol.json").read_text(encoding="utf-8"))
    assert sha(Path(__file__)) == p["code_sha256"]
    policy = json.loads((ROOT / "codex_automations/fuxi_runtime_policy.json").read_text(encoding="utf-8"))
    assert policy["single_writer_confirmed"] and policy["current_mode"] == "controlled_manual"
    assert not (OUT / "raw.json").exists()
    with RunContract(parameters={"event_protocol_sha256": sha(OUT / "protocol.json"),
                                 "request_count": 1},
                     run_id="quality_event_" + datetime.now().strftime("%H%M%S_%f"),
                     runs_dir=OUT / "runs"):
        client = get_tushare_api()
        if hasattr(client, "_DataApi__timeout"):
            client._DataApi__timeout = 20
        frame = client.dividend(ts_code=CODE, fields=FIELDS)
        records = [] if frame is None else frame.to_dict("records")
        write_once(OUT / "raw.json", {"status": "response_received",
                                        "retrieved_at": datetime.now().astimezone().isoformat(),
                                        "row_count": len(records),
                                        "possible_truncation": len(records) >= 2000,
                                        "records": records})
        print(json.dumps({"status": "response_received", "rows": len(records),
                          "possible_truncation": len(records) >= 2000}))


def validate():
    import pandas as pd

    p = json.loads((OUT / "protocol.json").read_text(encoding="utf-8"))
    raw = json.loads((OUT / "raw.json").read_text(encoding="utf-8"))
    assert sha(ROOT / "data/raw/daily_prices.csv") == p["price_sha256"]
    if raw["possible_truncation"]:
        write_once(OUT / "validation.json", {"status": "data_insufficient",
                                               "reason": "response_may_be_truncated"})
        return
    matches = [r for r in raw["records"] if r.get("div_proc") == "实施"
               and str(r.get("ex_date") or "").replace("-", "") == DAY.replace("-", "")]
    if len(matches) != 1:
        write_once(OUT / "validation.json", {"status": "data_insufficient",
                                               "reason": "implemented_event_not_unique",
                                               "matches": len(matches),
                                               "same_date_rows": [r for r in raw["records"]
                                                                  if str(r.get("ex_date") or "").replace("-", "")
                                                                  == DAY.replace("-", "")]})
        return
    item = matches[0]
    wanted = {DAY.replace("-", ""), str(item.get("record_date") or "")}
    pieces = []
    for part in pd.read_csv(ROOT / "data/raw/daily_prices.csv",
                            usecols=["ts_code", "trade_date", "close", "pre_close"],
                            dtype={"trade_date": str}, chunksize=300_000):
        sub = part[part.ts_code.eq(CODE)].copy()
        sub.trade_date = pd.to_datetime(sub.trade_date, format="mixed").dt.strftime("%Y%m%d")
        pieces.append(sub[sub.trade_date.isin(wanted)])
    prices = pd.concat(pieces, ignore_index=True).set_index("trade_date")
    try:
        assert prices.index.is_unique
        record_close = float(prices.loc[str(item["record_date"]), "close"])
        ex_ref = float(prices.loc[DAY.replace("-", ""), "pre_close"])
        action = checked_cash_action(item, CODE, DAY, record_close, ex_ref,
                                     sha(OUT / "raw.json"))
        output = {"status": "verified_cash_event", "action": asdict(action),
                  "record_close": record_close, "ex_reference": ex_ref,
                  "raw_sha256": sha(OUT / "raw.json"), "formal_eligibility": False}
    except (KeyError, ValueError, AssertionError) as exc:
        output = {"status": "data_insufficient", "reason": type(exc).__name__ + ": " + str(exc),
                  "raw_sha256": sha(OUT / "raw.json")}
    write_once(OUT / "validation.json", output)
    print(json.dumps({"status": output["status"], "reason": output.get("reason")}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "collect", "validate"))
    action = parser.parse_args().action
    {"prepare": prepare, "collect": collect, "validate": validate}[action]()
