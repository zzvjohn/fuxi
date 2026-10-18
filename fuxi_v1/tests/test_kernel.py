from copy import deepcopy
from contextlib import contextmanager
from datetime import datetime
from decimal import Decimal
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import shutil
import uuid

from fuxi_v1.account import Account
from fuxi_v1.artifacts import publish, verify
from fuxi_v1.contracts import ContractError, digest, evaluation, load_json, timestamp
from fuxi_v1.data import Snapshot, Labels
from fuxi_v1.dsl import compile_ast, evaluate_ast, generate_candidates, ranks
from fuxi_v1.evaluation import DEFAULT_POLICY, factor_statistics, validate_policy, correlation, admission
from fuxi_v1.fixtures import synthetic_input
from fuxi_v1.jq import compile_diagnostic
from fuxi_v1.pipeline import run, prepare


@contextmanager
def scratch_directory():
    # Python 3.13's mode-0700 Windows temp ACL is unusable in this sandbox.
    # Use a project-owned, unique directory with inherited workspace permissions.
    root = (Path(__file__).resolve().parent / "_scratch").resolve()
    root.mkdir(exist_ok=True)
    child = root / ("test_" + uuid.uuid4().hex)
    child.mkdir()
    try:
        yield str(child)
    finally:
        resolved = child.resolve()
        if resolved.parent != root or not resolved.name.startswith("test_"):
            raise RuntimeError("refusing cleanup outside test scratch root")
        shutil.rmtree(resolved)


class DataAndDslTests(unittest.TestCase):
    def setUp(self):
        self.input = synthetic_input()
        self.snapshot = Snapshot(self.input["snapshot"])
        self.candidates = generate_candidates(self.snapshot.payload["fields"])
        self.ast = self.candidates[0]["canonical_ast"]

    def test_future_perturbation_cannot_change_past_signal(self):
        before = evaluate_ast(self.ast, self.snapshot)
        payload = deepcopy(self.input["snapshot"])
        for row in payload["observations"]:
            if row["session"] > payload["sessions"][10]:
                row["value"] *= 100
        after = evaluate_ast(self.ast, Snapshot(payload))
        days = [d for d in before if d <= payload["sessions"][10]]
        self.assertGreater(len(days), 0)
        self.assertEqual({d: before[d] for d in days}, {d: after[d] for d in days})

    def test_universe_and_row_permutation_invariance(self):
        payload = deepcopy(self.input["snapshot"])
        payload["observations"].reverse()
        for members in payload["universe"].values():
            members.reverse()
        self.assertEqual(evaluate_ast(self.ast, self.snapshot), evaluate_ast(self.ast, Snapshot(payload)))

    def test_late_field_is_masked_then_becomes_available(self):
        p = deepcopy(self.input["snapshot"])
        day = p["sessions"][5]
        code = p["universe"][day][0]
        row = next(r for r in p["observations"] if r["session"] == day and r["instrument"] == code)
        row["available_at"] = p["sessions"][6]+"T17:00:00+08:00"
        snap = Snapshot(p)
        self.assertIsNone(evaluate_ast({"field": "close"}, snap)[day][code])
        ast = {"op": "lag", "window": 2, "args": [{"field": "close"}]}
        self.assertEqual(evaluate_ast(ast, snap)[p["sessions"][7]][code], row["value"])

    def test_missing_history_is_not_filled(self):
        p = deepcopy(self.input["snapshot"])
        p["observations"] = [r for r in p["observations"] if r["session"] != p["sessions"][3]]
        raw = self.ast["args"][0]
        self.assertTrue(all(v is None for v in evaluate_ast(raw, Snapshot(p))[p["sessions"][5]].values()))

    def test_duplicate_revisions_fail(self):
        p = deepcopy(self.input["snapshot"])
        p["observations"].append(deepcopy(p["observations"][0]))
        with self.assertRaisesRegex(ContractError, "multiple cell revisions"):
            Snapshot(p)

    def test_nonfinite_and_unknown_fields_fail(self):
        for mutation in (lambda p: p["observations"][0].update(value=float("nan")),
                         lambda p: p.update(unregistered_key=1)):
            p = deepcopy(self.input["snapshot"])
            mutation(p)
            with self.assertRaises(ContractError):
                Snapshot(p)

    def test_labels_separate_and_maturity_enforced(self):
        labels = Labels(self.input["labels"], self.snapshot)
        row = self.input["labels"]["rows"][0]
        self.assertIsNone(labels.value(row["session"], row["instrument"], timestamp(row["label_start"])))
        self.assertEqual(labels.value(row["session"], row["instrument"], timestamp(row["mature_at"])), row["value"])
        original = deepcopy(self.input["labels"])
        evaluate_ast(self.ast, self.snapshot)
        self.assertEqual(original, self.input["labels"])

    def test_invalid_label_cannot_contain_zero(self):
        p = deepcopy(self.input["labels"])
        p["rows"][0].update(status="data_insufficient", value=0)
        with self.assertRaises(ContractError):
            Labels(p, self.snapshot)

    def test_label_clock_lineage_and_duplicate_rejected(self):
        mutations = [lambda p: p.update(feature_snapshot_hash="wrong"),
                     lambda p: p["rows"][0].update(mature_at="2020-01-01T00:00:00+08:00"),
                     lambda p: p["rows"].append(deepcopy(p["rows"][0]))]
        for mutation in mutations:
            p = deepcopy(self.input["labels"])
            mutation(p)
            with self.assertRaises(ContractError):
                Labels(p, self.snapshot)

    def test_ast_rejects_injection_units_and_unsupported_time_nesting(self):
        fields = dict(self.snapshot.payload["fields"], volume={"unit": "shares"})
        expressions = [{"op": "__import__", "args": []},
                       {"field": "close", "eval": "anything"},
                       {"op": "add", "args": [{"field": "close"}, {"field": "volume"}]},
                       {"op": "lag", "window": 2, "args": [self.ast]},
                       {"op": "return", "window": True, "args": [{"field": "close"}]}]
        for expression in expressions:
            with self.subTest(expression=expression), self.assertRaises(ContractError):
                compile_ast(expression, fields)

    def test_ties_and_constant_cross_section(self):
        self.assertEqual(ranks({"a": 2, "b": 2, "c": 4, "d": None}),
                         {"a": 1/3, "b": 1/3, "c": 5/6, "d": None})
        self.assertIsNone(correlation([1, 1, 1], [2, 3, 4]))
        self.assertIsNone(ranks({"a": 1})["a"])

    def test_zero_division_and_extreme_mean(self):
        p = deepcopy(self.input["snapshot"])
        for row in p["observations"]:
            row["value"] = 1e308
        snap = Snapshot(p)
        ast = {"op": "mean", "window": 2, "args": [{"field": "close"}]}
        self.assertTrue(all(v == 1e308 for row in evaluate_ast(ast, snap).values() for v in row.values()))
        zero = {"op": "sub", "args": [{"field": "close"}, {"field": "close"}]}
        ast = {"op": "div", "args": [{"field": "close"}, zero]}
        self.assertTrue(all(v is None for row in evaluate_ast(ast, snap).values() for v in row.values()))


class EvaluationTests(unittest.TestCase):
    def test_hand_computed_ic_and_coverage(self):
        source = synthetic_input()
        snap = Snapshot(source["snapshot"])
        labels = Labels(source["labels"], snap)
        scores = {d: {s: labels.value(d, s, timestamp(source["research_clock"])) for s in snap.universe[d]}
                  for d in snap.sessions[snap.start:]}
        result = factor_statistics(scores, snap, labels, DEFAULT_POLICY, source["research_clock"])
        self.assertEqual(result["eval_status"], "valid")
        self.assertAlmostEqual(result["metrics"]["pearson"]["mean"], 1)
        self.assertAlmostEqual(result["metrics"]["spearman"]["mean"], 1)
        self.assertEqual(result["diagnostics"]["sections"][-1]["pair_coverage"], 0)
        self.assertEqual(result["decision"], "not_applicable")

    def test_no_mature_labels_retains_denominators(self):
        source = synthetic_input()
        source["labels"]["rows"] = []
        files, summary = prepare(source)
        self.assertEqual(summary["valid_e1_count"], 0)
        result = next(iter(files["evaluations.json"].values()))[1]
        self.assertEqual(result["eval_status"], "data_insufficient")
        self.assertIsNone(result["metrics"])
        self.assertEqual(result["n_pairs"], 0)
        self.assertTrue(all(r["universe_count"] == 6 for r in result["diagnostics"]["sections"]))

    def test_nonvalid_pass_and_metrics_rejected(self):
        with self.assertRaises(ContractError):
            evaluation("E1", "data_insufficient", "test", decision="pass")
        with self.assertRaises(ContractError):
            evaluation("E1", "error", "test", metrics={"sharpe": 1})

    def test_formal_policy_cannot_be_self_certified(self):
        for change in ({"calibration_status": "calibrated"}, {"economic_thresholds": {"ic": 0.02}},
                       {"allow_formal": True}):
            policy = dict(DEFAULT_POLICY, **change)
            with self.assertRaises(ContractError):
                validate_policy(policy)

    def test_diagnostic_and_formal_routes_separate(self):
        gates = [evaluation("E0", "valid", "test", decision="pass")]
        result = admission(gates, DEFAULT_POLICY)
        self.assertTrue(result["jq_diagnostic_allowed"])
        self.assertFalse(result["jq_candidate_allowed"])
        self.assertNotIn("E5_not_accepted", result["blockers"])
        self.assertFalse(admission([], DEFAULT_POLICY)["jq_diagnostic_allowed"])


class AccountTests(unittest.TestCase):
    def setUp(self):
        self.account = Account("10000")

    def event(self, event_id, kind, payload, day="2024-01-02", time="10:00:00"):
        return dict(event_id=event_id, kind=kind, payload=payload,
                    timestamp=day+"T"+time+"+08:00", evidence_id="fixture_"+event_id)

    def buy(self):
        event = self.event("buy", "fill", dict(instrument="A", side="buy", quantity=100,
                          price="10", fee="1", sellable_at="2024-01-03T09:30:00+08:00"))
        self.account.apply(event)
        return event

    def mark(self, price="11", day="2024-01-03"):
        when = day+"T16:00:00+08:00"
        return self.account.mark(when, {"A": {"price": price, "asof": when, "evidence_id": "fixture_mark"}})

    def test_cash_and_shares_conservation(self):
        self.buy()
        self.assertEqual(self.account.cash, Decimal("8999"))
        self.assertEqual(Decimal(self.mark()["nav"]), Decimal("10099"))
        self.account.apply(self.event("sell", "fill", dict(instrument="A", side="sell", quantity=100,
                            price="11", fee="2", sellable_at="2024-01-03T09:30:00+08:00"), "2024-01-03"))
        self.assertEqual(self.account.cash, Decimal("10097"))
        self.assertEqual(self.mark()["holdings"], {})

    def test_duplicate_fill_idempotent_and_conflict_atomic(self):
        event = self.buy()
        self.assertEqual(self.account.apply(event), "already_applied")
        before = deepcopy(self.account.__dict__)
        event["payload"]["quantity"] = 200
        with self.assertRaises(ContractError):
            self.account.apply(event)
        self.assertEqual(before, self.account.__dict__)

    def test_unsettled_sale_rejected_atomically(self):
        self.buy()
        before = deepcopy(self.account.__dict__)
        event = self.event("sell", "fill", dict(instrument="A", side="sell", quantity=100,
                          price="11", fee="1", sellable_at="2024-01-02T10:00:00+08:00"), time="11:00:00")
        with self.assertRaisesRegex(ContractError, "settled shares"):
            self.account.apply(event)
        self.assertEqual(before, self.account.__dict__)

    def test_unfilled_exit_and_missing_mark_preserve_holdings(self):
        self.buy()
        self.account.apply(self.event("halt_exit", "unfilled", {"instrument": "A", "reason": "halted"}, "2024-01-03"))
        result = self.account.mark("2024-01-03T16:00:00+08:00", {})
        self.assertEqual(result["holdings"], {"A": 100})
        self.assertEqual(result["eval_status"], "data_insufficient")
        self.assertIsNone(result["nav"])
        self.assertEqual(result["cash"], "8999")

    def test_receivable_payment_and_split_nav(self):
        self.buy()
        self.account.apply(self.event("dividend", "receivable", {"entitlement_id": "d1", "amount": "20"}, "2024-01-03"))
        self.assertEqual(Decimal(self.mark()["nav"]), Decimal("10119"))
        self.account.apply(self.event("payment", "payment", {"entitlement_id": "d1"}, "2024-01-04"))
        self.account.apply(self.event("split", "split", {"instrument": "A", "multiplier": "2",
                           "sellable_at": "2024-01-04T09:30:00+08:00"}, "2024-01-04"))
        result = self.mark(price="5.5", day="2024-01-04")
        self.assertEqual(result["holdings"], {"A": 200})
        self.assertEqual(Decimal(result["nav"]), Decimal("10119"))
        self.assertEqual(result["receivables"], "0")

    def test_fractional_split_unknown_event_and_stale_mark_fail(self):
        self.buy()
        event = self.event("fraction", "split", {"instrument": "A", "multiplier": "1.001",
                           "sellable_at": "2024-01-03T09:30:00+08:00"}, "2024-01-03")
        with self.assertRaisesRegex(ContractError, "fractional"):
            self.account.apply(event)
        self.account.apply(self.event("gap", "unknown", {"instrument": "A", "reason": "unresolved corporate action"}, "2024-01-03"))
        self.assertFalse(self.mark()["valuation_identifiable"])
        with self.assertRaisesRegex(ContractError, "same-cutoff"):
            self.account.mark("2024-01-04T16:00:00+08:00", {"A": {"price": "11",
                              "asof": "2024-01-03T16:00:00+08:00", "evidence_id": "old"}})


class IntegrationTests(unittest.TestCase):
    def test_independent_process_has_no_legacy_imports(self):
        code = "from fuxi_v1.pipeline import prepare; from fuxi_v1.fixtures import synthetic_input; " \
               "import sys; prepare(synthetic_input()); " \
               "assert not any('factor_alchemy' in n or 'nuwa' in n for n in sys.modules)"
        result = subprocess.run([sys.executable, "-X", "utf8", "-c", code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_full_pipeline_immutable_bundle_and_tamper_detection(self):
        with scratch_directory() as directory:
            path, summary = run(synthetic_input(), directory)
            manifest = verify(path)
            self.assertEqual(summary["generated_count"], 4)
            self.assertEqual(summary["valid_e1_count"], 4)
            self.assertEqual(summary["jq_candidate_count"], 0)
            self.assertGreater(len(manifest["files"]), 0)
            (path / "summary.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ContractError, "hash mismatch"):
                verify(path)

    def test_repeated_run_has_new_attempt_same_experiment(self):
        with scratch_directory() as directory:
            a, _ = run(synthetic_input(), directory)
            b, _ = run(synthetic_input(), directory)
            self.assertNotEqual(a, b)
            self.assertEqual(verify(a)["provenance"]["experiment_id"], verify(b)["provenance"]["experiment_id"])

    def test_duplicate_identity_and_traversal_rejected(self):
        with scratch_directory() as directory:
            publish(directory, {"a.json": {}}, {}, run_id="run_abc")
            with self.assertRaises(ContractError):
                publish(directory, {"a.json": {}}, {}, run_id="run_abc")
            with self.assertRaises(ContractError):
                publish(directory, {"../escape.json": {}}, {}, run_id="run_xyz")

    def test_generated_runtime_replays_all_factor_values(self):
        source = synthetic_input()
        snapshot = Snapshot(source["snapshot"])
        count = 0
        for candidate in generate_candidates(snapshot.payload["fields"]):
            expected = evaluate_ast(candidate["canonical_ast"], snapshot)
            script, capability = compile_diagnostic(candidate, snapshot, expected)
            namespace = {}
            exec(compile(script, "generated_jq_diagnostic.py", "exec"), namespace)
            self.assertEqual(namespace["local_replay"](), expected)
            self.assertFalse(capability["observed_on_jq"])
            count += sum(len(row) for row in expected.values())
        self.assertEqual(count, 456)

    def test_provider_diagnostic_call_contract_parity_and_paused_mask(self):
        source = synthetic_input()
        snapshot = Snapshot(source["snapshot"])
        candidate = generate_candidates(snapshot.payload["fields"])[0]
        scores = evaluate_ast(candidate["canonical_ast"], snapshot)
        script, _ = compile_diagnostic(candidate, snapshot, scores)
        namespace = {}
        exec(compile(script, "generated_jq_diagnostic.py", "exec"), namespace)
        day = snapshot.sessions[-1]
        records = source["snapshot"]["observations"]
        class Frame:
            columns = ["close", "paused"]
            def __init__(self, rows):
                self.rows = rows
            def __len__(self):
                return len(self.rows)
            def iterrows(self):
                return iter(self.rows)
        def provider(stock, **kw):
            self.assertIsNone(kw["fq"])
            self.assertFalse(kw["skip_paused"])
            self.assertEqual(kw["fields"], ["close", "paused"])
            return Frame([(r["session"], {"close": r["value"], "paused": 0})
                          for r in records if r["instrument"] == stock and r["session"] <= kw["end_date"]])
        result = namespace["provider_diagnostic"](provider, day)
        self.assertEqual(result["n_pairs"], 6)
        self.assertEqual(result["max_abs_score_difference"], 0)
        self.assertGreater(result["raw_price_pair_count"], 0)
        self.assertEqual(result["max_abs_raw_price_difference"], 0)
        self.assertFalse(result["provider_parity_accepted"])
        def paused_provider(stock, **kw):
            frame = provider(stock, **kw)
            return Frame([(d, dict(row, paused=1)) for d, row in frame.rows])
        with self.assertRaisesRegex(ContractError, "zero matched pairs"):
            # Embedded code defines its own ContractError class.
            try:
                namespace["provider_diagnostic"](paused_provider, day)
            except ValueError as exc:
                raise ContractError(str(exc)) from exc

    def test_generated_input_tampering_is_detected(self):
        source = synthetic_input()
        snapshot = Snapshot(source["snapshot"])
        candidate = generate_candidates(snapshot.payload["fields"])[0]
        scores = evaluate_ast(candidate["canonical_ast"], snapshot)
        script, _ = compile_diagnostic(candidate, snapshot, scores)
        namespace = {}
        exec(compile(script, "generated_jq_diagnostic.py", "exec"), namespace)
        namespace["FROZEN"]["ast"] = {"field": "close"}
        with self.assertRaisesRegex(ValueError, "frozen diagnostic input mismatch"):
            namespace["local_replay"]()

    def test_json_duplicate_keys_rejected(self):
        with scratch_directory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text('{"x": 1, "x": 2}', encoding="utf-8")
            with self.assertRaises(ContractError):
                load_json(path)


if __name__ == "__main__":
    unittest.main()
