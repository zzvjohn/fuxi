import unittest
from fuxi_v1.contracts import ContractError, digest
from fuxi_v1.lineage import (candidate_lineage, source_card, validate_candidate_lineage,
                             validate_source_card, visible_cards)


def card(source_id, available_at="2025-06-01T10:00:00+08:00", kind="positive",
         window=("2024-01-01", "2024-12-31"), derived=None, revoked_deps=()):
    return source_card(source_id, available_at=available_at,
                       source_date="2025-06-01T00:00:00+08:00", source_type="internal_study",
                       mechanism_tags=["reversal"], evidence_kind=kind,
                       recent_status="validated", evidence_window=[list(window)],
                       derived_from=derived, revoked_dependencies=revoked_deps)


class LineageTests(unittest.TestCase):
    def test_no_future_winner_leak(self):
        # R2 counterexample: replaying a historical task must not see a winner
        # card that only became available later.
        past = card("c_past", available_at="2025-06-01T10:00:00+08:00")
        future_winner = card("c_winner", available_at="2026-02-01T10:00:00+08:00",
                             kind="positive", window=("2025-07-01", "2025-12-31"))
        view = visible_cards([past, future_winner], as_of="2025-12-31T18:00:00+08:00")
        seen = {c["source_id"] for c in view["cards"]}
        self.assertIn("c_past", seen)
        self.assertNotIn("c_winner", seen)
        self.assertIn("c_winner", view["omitted_reasons"])
        self.assertIn("not_available_at_as_of", view["omitted_reasons"]["c_winner"])

    def test_validation_segment_excluded(self):
        train = card("c_train", window=("2024-01-01", "2024-12-31"))
        val = card("c_val", window=("2025-01-01", "2025-06-30"))
        view = visible_cards([train, val], as_of="2026-01-01T00:00:00+08:00",
                             excluded_segments=[("2025-01-01", "2025-06-30")])
        seen = {c["source_id"] for c in view["cards"]}
        self.assertIn("c_train", seen)
        self.assertNotIn("c_val", seen)
        self.assertIn("segment_excluded", view["omitted_reasons"]["c_val"])

    def test_derived_ranking_inherits_window(self):
        base = card("c_base", window=("2024-01-01", "2024-12-31"))
        derived = card("c_rank", window=("2024-01-01", "2024-12-31"),
                       derived={"parent_source_id": "c_base",
                                "segment": ["2025-01-01", "2025-06-30"],
                                "ranking_kind": "rolling_rank"})
        view = visible_cards([base, derived], as_of="2026-01-01T00:00:00+08:00",
                             excluded_segments=[("2025-01-01", "2025-06-30")])
        seen = {c["source_id"] for c in view["cards"]}
        self.assertIn("c_base", seen)
        self.assertNotIn("c_rank", seen)
        self.assertIn("derived_ranking_segment_excluded", view["omitted_reasons"]["c_rank"])

    def test_revoked_evidence_kept_out_of_view(self):
        ok = card("c_ok")
        revoked = card("c_rev", kind="revoked")
        depends = card("c_dep", revoked_deps=["c_rev"])
        view = visible_cards([ok, revoked, depends], as_of="2026-01-01T00:00:00+08:00")
        seen = {c["source_id"] for c in view["cards"]}
        self.assertIn("c_ok", seen)
        self.assertNotIn("c_rev", seen)
        self.assertNotIn("c_dep", seen)
        self.assertIn("revoked", view["omitted_reasons"]["c_rev"])
        self.assertIn("revoked", view["omitted_reasons"]["c_dep"])

    def test_omission_accounting(self):
        cards = [card("c%d" % i) for i in range(5)]
        view = visible_cards(cards, as_of="2026-01-01T00:00:00+08:00", max_cards=3)
        self.assertEqual(len(view["cards"]), 3)
        self.assertEqual(view["omitted_count"], 2)
        self.assertTrue(all("over_budget" in r for r in view["omitted_reasons"].values()))

    def test_source_card_validation(self):
        with self.assertRaises(ContractError):
            source_card("x", available_at="not-a-time", source_date="2025-06-01T00:00:00+08:00",
                        source_type="paper", mechanism_tags=[], evidence_kind="positive",
                        recent_status="ok", evidence_window=[["2024-01-01", "2024-12-31"]])
        with self.assertRaises(ContractError):
            source_card("x", available_at="2025-06-01T10:00:00+08:00",
                        source_date="2025-06-01T00:00:00+08:00", source_type="bogus",
                        mechanism_tags=[], evidence_kind="positive",
                        recent_status="ok", evidence_window=[["2024-01-01", "2024-12-31"]])

    def test_candidate_lineage(self):
        lineage = candidate_lineage(hypothesis_id="h1", parent_ids=["p2", "p1"],
                                    edit_spec="swap lookback 20->10",
                                    generation_action_id="a7", source_refs=["s2", "s1"])
        validate_candidate_lineage(lineage)
        self.assertEqual(lineage["parent_ids"], ["p1", "p2"])
        self.assertEqual(lineage["source_refs"], ["s1", "s2"])
        with self.assertRaises(ContractError):
            candidate_lineage(hypothesis_id="", parent_ids=[], edit_spec=None,
                              generation_action_id="a", source_refs=[])

    def test_digest_stability(self):
        a = source_card("x", available_at="2025-06-01T10:00:00+08:00",
                        source_date="2025-06-01T00:00:00+08:00", source_type="paper",
                        mechanism_tags=["m"], evidence_kind="positive", recent_status="ok",
                        evidence_window=[["2024-01-01", "2024-12-31"]])
        b = source_card("x", available_at="2025-06-01T10:00:00+08:00",
                        source_date="2025-06-01T00:00:00+08:00", source_type="paper",
                        mechanism_tags=["m"], evidence_kind="positive", recent_status="ok",
                        evidence_window=[["2024-01-01", "2024-12-31"]])
        self.assertEqual(digest(a), digest(b))


if __name__ == "__main__":
    unittest.main()
