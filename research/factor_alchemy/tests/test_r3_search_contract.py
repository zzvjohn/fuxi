from __future__ import annotations

import os
import unittest
from pathlib import Path

import numpy as np

from forge import FactorForge
from forge.paradigm_profiles import MATURE_PARADIGM_PROFILES
from ralph_loop import RalphLoop


class R3SearchContractTests(unittest.TestCase):
    def test_breed_fail_state_can_be_isolated_for_experiments(self):
        old = os.environ.get("FUXI_BREED_FAIL_STATE_PATH")
        try:
            os.environ["FUXI_BREED_FAIL_STATE_PATH"] = "reports/remediation_20260922/isolated.json"
            loop = object.__new__(RalphLoop)
            loop.data_dir = Path(__file__).resolve().parents[3] / "data"
            self.assertEqual(
                loop._breed_streak_path(),
                Path(__file__).resolve().parents[3]
                / "reports/remediation_20260922/isolated.json",
            )
        finally:
            if old is None:
                os.environ.pop("FUXI_BREED_FAIL_STATE_PATH", None)
            else:
                os.environ["FUXI_BREED_FAIL_STATE_PATH"] = old

    @staticmethod
    def _data(seed=7):
        rng = np.random.RandomState(seed)
        returns = rng.normal(0, 0.02, size=(60, 40))
        close = 100 * np.cumprod(1 + returns, axis=0)
        return {
            "open": close * (1 + rng.normal(0, 0.001, close.shape)),
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": np.exp(rng.normal(10, 0.5, close.shape)),
            "vwap": close,
            "returns": returns,
            "fwd_return": np.roll(returns, -1, axis=0),
        }

    def test_target_paradigm_constrains_entire_population(self):
        data = self._data()
        forge = FactorForge(
            data=data,
            forward_returns=data["fwd_return"],
            paradigm_profiles=MATURE_PARADIGM_PROFILES,
            max_depth=3,
            seed=11,
            fitness_mode="legacy",
        )
        best = forge.evolve(
            n_generations=1,
            pop_size=12,
            elite_count=2,
            workers=1,
            verbose=False,
            top_k=3,
            target_paradigm="动量反转",
        )
        self.assertGreater(len(best), 0)
        self.assertEqual({row["paradigm"] for row in best}, {"动量反转"})
        self.assertEqual(set(forge._paradigm_tags), {"动量反转"})
        self.assertEqual(forge.last_run_stats["target_paradigm"], "动量反转")

    def test_unknown_target_is_fail_closed(self):
        data = self._data()
        forge = FactorForge(
            data=data,
            forward_returns=data["fwd_return"],
            paradigm_profiles=MATURE_PARADIGM_PROFILES,
            seed=3,
        )
        with self.assertRaises(ValueError):
            forge.evolve(
                n_generations=1, pop_size=8, workers=1, verbose=False,
                target_paradigm="未注册范式",
            )

    def test_stability_turnover_objective_prefers_stable_ic_path(self):
        data = self._data()
        fwd = data["fwd_return"]
        forge = FactorForge(
            data=data,
            forward_returns=fwd,
            paradigm_profiles={},
            seed=5,
            fitness_mode="stability_turnover",
        )
        stable = fwd + np.random.RandomState(1).normal(0, 0.001, fwd.shape)
        alternating = stable * np.where(np.arange(stable.shape[0])[:, None] % 2, -1.0, 1.0)
        stable_score = forge._fitness_components_from_values(
            stable, complexity=2.0, nan_ratio=0.0
        )
        alternating_score = forge._fitness_components_from_values(
            alternating, complexity=2.0, nan_ratio=0.0
        )
        self.assertGreater(stable_score["score"], alternating_score["score"])
        self.assertGreater(stable_score["sign_consistency"], alternating_score["sign_consistency"])
        self.assertIn("rank_turnover_proxy", stable_score)


if __name__ == "__main__":
    unittest.main()
