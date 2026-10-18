import sys
import subprocess
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch


BASE = Path(__file__).resolve().parents[1]
ROOT = BASE.parents[1]
sys.path.insert(0, str(BASE))

from learning_event_store import LearningEventStore  # noqa: E402
from mechanism_search import load_catalog  # noqa: E402


class ShadowLearningStoreTests(unittest.TestCase):
    def setUp(self):
        test_root = ROOT / "reports" / "fuxi_a5_20260926" / "test_tmp"
        test_root.mkdir(parents=True, exist_ok=True)
        self.path = test_root / f"events_{uuid.uuid4().hex}.sqlite"
        self.addCleanup(self._cleanup_db)
        self.store = LearningEventStore(self.path)
        self.plans = load_catalog(BASE / "mechanism_catalog_20260925.json", "daily")

    def _cleanup_db(self):
        for suffix in ("", "-wal", "-shm"):
            target = Path(str(self.path) + suffix)
            if target.exists():
                target.unlink()

    def test_real_missing_books_ingest_idempotently_and_never_exploit(self):
        evidence = ROOT / "reports/fuxi_mechanism_quantum_20260925/economic_evidence.jsonl"
        first = self.store.ingest_evidence(evidence, ROOT, self.plans, observed_on="2026-09-26")
        second = LearningEventStore(self.path).ingest_evidence(
            evidence, ROOT, self.plans, observed_on="2026-09-26")
        self.assertEqual((first["inserted"], second["duplicates"]), (2, 2))
        selected, audit = self.store.decide("d1", self.plans, as_of="2026-09-26", seed=17)
        self.assertIn(selected["plan_id"], {p["plan_id"] for p in self.plans})
        self.assertEqual(audit["mode"], "explore")
        self.assertEqual(audit["valid_economic_observations"], 0)
        self.assertEqual(audit["evidence_count"], 2)
        self.assertEqual(self.store.snapshot()["valid_economic_observations"], 0)
        self.assertEqual(self.store.trusted_memory_view(as_of="2026-09-26"), [])
        self.assertEqual(LearningEventStore(self.path).decide(
            "d1", self.plans, as_of="2026-09-26", seed=17)[1], audit)
        self.assertEqual(len(self.store.snapshot()["decisions"]), 1)
        child = subprocess.run(
            [sys.executable, "-X", "utf8", "-c",
             "import sys; from learning_event_store import LearningEventStore; "
             "from mechanism_search import load_catalog; from pathlib import Path; "
             "s=LearningEventStore(sys.argv[1]); p=load_catalog(Path(sys.argv[2]),'daily'); "
             "print(s.decide('d1',p,as_of='2026-09-26',seed=17)[1]['round_index'])",
             str(self.path), str(BASE / "mechanism_catalog_20260925.json")],
            text=True, capture_output=True, check=True,
        )
        self.assertEqual(child.stdout.strip(), "0")
        self.assertEqual(len(self.store.snapshot()["decisions"]), 1)

    def test_maturity_cutoff_and_conflicting_replay(self):
        pid = self.plans[0]["plan_id"]
        row = {"observation_id": "valid_1", "plan_id": pid,
               "status": "valid", "source_sha256": "a" * 64,
               "matured_on": "2026-10-02", "verified_trading_label": True,
               "economic_lcb_bps": 8.0}
        with patch("learning_event_store.load_evidence", return_value=[row]):
            self.assertEqual(self.store.ingest_evidence(
                "unused", ROOT, self.plans, observed_on="2026-09-26")["inserted"], 1)
        before = self.store.decide("before", self.plans, as_of="2026-10-01", seed=5)[1]
        after = self.store.decide("after", self.plans, as_of="2026-10-03", seed=5)[1]
        self.assertEqual(before["valid_economic_observations"], 0)
        self.assertEqual(after["valid_economic_observations"], 1)
        self.assertEqual(self.store.trusted_memory_view(as_of="2026-10-01"), [])
        self.assertEqual(len(self.store.trusted_memory_view(as_of="2026-10-03")), 1)
        with self.assertRaisesRegex(ValueError, "conflicting decision replay"):
            self.store.decide("after", self.plans, as_of="2026-10-04", seed=5)
        with patch("learning_event_store.load_evidence", return_value=[dict(row, source_sha256="b" * 64)]):
            with self.assertRaisesRegex(ValueError, "conflicting observation replay"):
                self.store.ingest_evidence("unused", ROOT, self.plans,
                                           observed_on="2026-09-26")


if __name__ == "__main__":
    unittest.main()
