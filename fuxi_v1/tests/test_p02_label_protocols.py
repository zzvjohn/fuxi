import unittest
from fuxi_v1.contracts import ContractError, digest
from fuxi_v1.data import Snapshot
from fuxi_v1.datasets import materialize_labels
from fuxi_v1.label_protocols import (PROTOCOLS, materialize_labels_target,
                                     protocol_identity, validate_label_spec)
from fuxi_v1.system_fixture import synthetic_campaign


class LabelProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture = synthetic_campaign(sessions_count=150, instruments_count=8)
        cls.snapshot = Snapshot(fixture["snapshot"])
        cls.market = fixture["market"]
        cls.legacy = materialize_labels(cls.snapshot, cls.market, 5)
        cls.target = materialize_labels_target(cls.snapshot, cls.market, 5)

    def test_protocol_ids_distinct(self):
        a = PROTOCOLS["legacy_t1_open_t6_close"]["protocol_id"]
        b = PROTOCOLS["target_five_open_intervals"]["protocol_id"]
        self.assertNotEqual(a, b)
        self.assertNotEqual(protocol_identity(PROTOCOLS["legacy_t1_open_t6_close"]),
                            protocol_identity(PROTOCOLS["target_five_open_intervals"]))

    def test_label_spec_validation(self):
        for proto in PROTOCOLS.values():
            validate_label_spec(proto["label_spec"])
        bad = dict(PROTOCOLS["legacy_t1_open_t6_close"]["label_spec"])
        bad["exit_price_clock"] = "vwap"
        with self.assertRaises(ContractError):
            validate_label_spec(bad)

    def test_same_row_count(self):
        # Both materializers iterate identical session/instrument surfaces.
        self.assertEqual(len(self.legacy["rows"]), len(self.target["rows"]))

    def test_row_key_surfaces_identical(self):
        legacy_keys = {(r["session"], r["instrument"]) for r in self.legacy["rows"]}
        target_keys = {(r["session"], r["instrument"]) for r in self.target["rows"]}
        self.assertEqual(legacy_keys, target_keys)

    def test_support_sets_disclosed(self):
        legacy_valid = {k for k in self._valid(self.legacy)}
        target_valid = {k for k in self._valid(self.target)}
        # Synthetic fixture has complete bars for both clocks, so support coincides;
        # the point of the check is that the two sets are computed independently and
        # any difference would be explicit rather than silently merged.
        self.assertEqual(len(legacy_valid), len(target_valid))
        self.assertGreater(len(legacy_valid), 0)

    def test_values_differ_by_exit_clock(self):
        # Where both protocols are valid, the value differs exactly because the
        # exit price is open (target) vs close (legacy); they must never be
        # silently assumed equal.
        legacy_map = {self._key(r): r["value"] for r in self.legacy["rows"] if r["status"] == "valid"}
        target_map = {self._key(r): r["value"] for r in self.target["rows"] if r["status"] == "valid"}
        common = set(legacy_map) & set(target_map)
        self.assertGreater(len(common), 0)
        differing = [k for k in common if abs(legacy_map[k] - target_map[k]) > 1e-12]
        self.assertGreater(len(differing), 0)

    def test_target_spec_carries_protocol_id(self):
        self.assertEqual(self.target["label_spec"]["protocol_id"],
                         PROTOCOLS["target_five_open_intervals"]["protocol_id"])
        self.assertEqual(self.target["label_spec"]["exit_price_clock"], "open")

    def test_protocol_digests_differ_for_same_surface(self):
        # Row-level evidence_id identifies the (market, instrument, window) data
        # support; protocol identity lives in label_spec. Whole-object digests
        # must differ so cross-protocol results can never be silently reused.
        self.assertNotEqual(digest(self.legacy), digest(self.target))

    @staticmethod
    def _valid(labels):
        return ((r["session"], r["instrument"]) for r in labels["rows"] if r["status"] == "valid")

    @staticmethod
    def _key(row):
        return (row["session"], row["instrument"])


if __name__ == "__main__":
    unittest.main()
