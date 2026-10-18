"""QEA decoder reachability and draw-boundary counterexamples."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
from forge import FactorForge
from forge.primitives import NON_INPUT_PRIMITIVES
from forge.quantum_challenger import QuantumForgeChallenger


def tiny_forge(max_depth=1):
    data = {name: np.ones((4, 12)) for name in
            ("open", "high", "low", "close", "volume", "returns")}
    forge = FactorForge(data=data, max_depth=max_depth, max_complexity=12,
                        seed=5, dim_prune="off", fitness_mode="legacy")
    forge._fitness = lambda tree: float(len(tree.to_string()) % 11)
    return forge


def coded_tape(challenger, code):
    tape = np.zeros(challenger.bit_count, dtype=np.uint8)
    tape[1:9] = ((code >> np.arange(8)) & 1).astype(np.uint8)
    return tape


class QEAEngineeringTests(unittest.TestCase):
    def test_every_root_operator_is_reachable_with_independent_stop_bit(self):
        q = QuantumForgeChallenger(tiny_forge(), seed=7, unique_budget=1)
        observed = {q._decode(coded_tape(q, code)).primitive.name for code in range(256)}
        self.assertEqual(observed, {primitive.name for primitive in NON_INPUT_PRIMITIVES})
        self.assertEqual(len(observed), len(NON_INPUT_PRIMITIVES))

    def test_choice_field_covers_terminal_weights_and_refuses_out_of_range(self):
        q = QuantumForgeChallenger(tiny_forge(), seed=7, unique_budget=1)
        reachable = {q._weighted_choice(q.terminals, {}, code).name
                     for code in range(256)}
        self.assertTrue({"open", "high", "low", "close", "volume", "returns"}
                        <= reachable)
        with self.assertRaises(ValueError):
            q._weighted_choice(q.ops, {}, 256)

    def test_duplicate_or_reject_at_boundary_still_rotates_elite(self):
        class FakeTree:
            def __init__(self, name, complexity):
                self.name, self.complexity = name, complexity
                self.primitive = type("PrimitiveStub", (), {"name": name})()

            def to_string(self):
                return self.name

            def to_pandas_string(self):
                return self.name

            def depth(self):
                return 1

            def terminals(self):
                return {"close"}

        for middle, expected_reject in (("A", 0), ("too_complex", 1)):
            with self.subTest(middle=middle):
                q = QuantumForgeChallenger(tiny_forge(), seed=11,
                                           unique_budget=2, population_size=2,
                                           islands=2, rotation=0.04)
                tapes = [np.zeros(q.bit_count, dtype=np.uint8) for _ in range(3)]
                for tape, marker in zip(tapes, (0, 1, 2)):
                    tape[0], tape[1] = marker & 1, (marker >> 1) & 1
                names = ["A", middle, "B"]
                sampled = iter(tapes)
                decoded = iter(FakeTree(name, 100 if name == "too_complex" else 1)
                               for name in names)
                with patch.object(q, "_sample", side_effect=lambda _island: next(sampled)), \
                     patch.object(q, "_decode", side_effect=lambda _tape: next(decoded)):
                    result = q.run()
                self.assertEqual(result["draws"], 3)
                self.assertEqual(result["unique_evaluated"], 2)
                self.assertEqual(result["rotation_batches"], 1)
                self.assertEqual(result["rotation_updates"], 1)
                self.assertEqual(result["pre_eval_rejected"], expected_reject)
                self.assertLess(result["mean_bit_entropy"], 1.0)

    def test_zero_rotation_control_remains_unrotated_at_draw_boundaries(self):
        q = QuantumForgeChallenger(tiny_forge(), seed=3, unique_budget=4,
                                   population_size=2, rotation=0)
        result = q.run()
        self.assertTrue(result["budget_met"])
        self.assertGreaterEqual(result["rotation_batches"], 1)
        self.assertEqual(result["rotation_updates"], 0)
        self.assertAlmostEqual(result["mean_bit_entropy"], 1.0)

    def test_final_boundary_has_no_ineffective_rotation(self):
        q = QuantumForgeChallenger(tiny_forge(), seed=3, unique_budget=2,
                                   population_size=2, rotation=0.04)
        result = q.run()
        self.assertEqual(result["draws"], 2)
        self.assertEqual(result["rotation_batches"], 0)
        self.assertEqual(result["rotation_updates"], 0)
        self.assertAlmostEqual(result["mean_bit_entropy"], 1.0)
        with self.assertRaises(ValueError):
            QuantumForgeChallenger(tiny_forge(), seed=3, unique_budget=2,
                                   max_draw_multiplier=0)


if __name__ == "__main__":
    unittest.main()
