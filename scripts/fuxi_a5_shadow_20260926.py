"""Controlled A5 shadow replay: verified events -> persistent decisions.

This deliberately has no production MAB/Memory writes and cannot promote a
candidate. The existing two economic probes are both data_insufficient.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research" / "factor_alchemy"))

from learning_event_store import LearningEventStore  # noqa: E402
from mechanism_search import load_catalog  # noqa: E402
from run_contract import FuxiWriterLock  # noqa: E402


def main() -> None:
    out_dir = ROOT / "reports" / "fuxi_a5_20260926"
    out_dir.mkdir(parents=True, exist_ok=True)
    result_path = out_dir / "shadow_replay_v3.json"
    if result_path.exists():
        raise FileExistsError("shadow replay is frozen; do not overwrite it")
    catalog = ROOT / "research" / "factor_alchemy" / "mechanism_catalog_20260925.json"
    evidence = ROOT / "reports" / "fuxi_mechanism_quantum_20260925" / "economic_evidence.jsonl"
    plans = load_catalog(catalog, "daily")
    store = LearningEventStore(out_dir / "shadow_events_v3.sqlite")
    with FuxiWriterLock(run_id="a5_shadow_20260926"):
        ingestion = store.ingest_evidence(evidence, ROOT, plans,
                                          observed_on="2026-09-26")
        decisions = []
        for index in range(4):
            chosen, audit = store.decide(f"a5_shadow_v3_20260926_{index}", plans,
                                         as_of="2026-09-26", seed=1701)
            decisions.append({"plan_id": chosen["plan_id"], "audit": audit})
        result = {
            "schema": "fuxi_a5_shadow_replay/v1",
            "source_catalog_sha256": hashlib.sha256(catalog.read_bytes()).hexdigest(),
            "source_evidence_sha256": hashlib.sha256(evidence.read_bytes()).hexdigest(),
            "code_sha256": {
                str(relative): hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
                for relative in (
                    Path("research/factor_alchemy/mechanism_search.py"),
                    Path("research/factor_alchemy/learning_event_store.py"),
                    Path("scripts/fuxi_a5_shadow_20260926.py"),
                )
            },
            "ingestion": ingestion,
            "snapshot": store.snapshot(),
            "decisions": decisions,
            "economic_verdict": "data_insufficient",
            "production_feedback_applied": False,
        }
        result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                               encoding="utf-8")
    print(json.dumps({"result": str(result_path), "ingestion": ingestion,
                      "valid_economic_observations": result["snapshot"]["valid_economic_observations"],
                      "decisions": len(decisions)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
