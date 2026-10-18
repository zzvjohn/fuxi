from __future__ import annotations

import os
import numpy as np
import pandas as pd
import shutil
import unittest
import uuid
from pathlib import Path

from forge.expression import ExprNode
from forge.primitives import _cross_rank, _cross_scale, _cross_zscore
from llm_generator import LLMGenerator


def _eval_cs(name: str, frame: pd.DataFrame) -> np.ndarray:
    expression = ExprNode._cs_to_pandas(name, ["close_p"])
    out = eval(expression, {"np": np, "pd": pd}, {"close_p": frame})
    assert out.shape == frame.shape
    return out.to_numpy(dtype=float)


class R1SemanticsTests(unittest.TestCase):
    def setUp(self):
        base = Path(__file__).resolve().parents[3] / "reports" / "remediation_20260922" / "_test_tmp"
        base.mkdir(parents=True, exist_ok=True)
        self.tmp = base / uuid.uuid4().hex
        self.tmp.mkdir()
        self.old_compile = os.environ.get("FUXI_FORGE_CS_COMPILE")
        self.old_context = os.environ.get("FUXI_LLM_CONTEXT_MODE")
        os.environ["FUXI_FORGE_CS_COMPILE"] = "enforce"
        os.environ["FUXI_LLM_CONTEXT_MODE"] = "merge"

    def tearDown(self):
        for name, value in (
            ("FUXI_FORGE_CS_COMPILE", self.old_compile),
            ("FUXI_LLM_CONTEXT_MODE", self.old_context),
        ):
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_cross_section_translation_matches_numpy_contract(self):
        frame = pd.DataFrame(
            [
                [1.0, 1.0, 3.0, np.nan],
                [5.0, 5.0, 5.0, np.nan],
                [-2.0, 0.0, 4.0, 8.0],
            ],
            columns=list("abcd"),
        )
        expected = {
            "rank": _cross_rank(frame.to_numpy()),
            "zscore": _cross_zscore(frame.to_numpy()),
            "scale": _cross_scale(frame.to_numpy()),
        }
        for name, target in expected.items():
            actual = _eval_cs(name, frame)
            self.assertTrue(np.allclose(actual, target, equal_nan=True, atol=1e-12), name)

    def test_context_updates_are_transactional_and_prompt_complete(self):
        generator = LLMGenerator(output_dir=self.tmp)
        generator.receive_context(
            success_templates=[{
                "pattern_id": "KEEP_TEMPLATE",
                "description": "KEEP_TEMPLATE_DESCRIPTION",
                "formula": "rank(close)",
            }],
            forbidden_directions=[{"direction": "KEEP_FORBIDDEN", "reason": "do not repeat"}],
            frontier_cards="KEEP_FRONTIER_CARD",
            paradigm_priorities=[{
                "paradigm": "KEEP_PRIORITY_PARADIGM",
                "priority": "high",
                "reason": "KEEP_PRIORITY_REASON",
            }],
        )
        generator.receive_context(
            warning_directions=[{"description": "KEEP_WARNING", "reason": "soft warning"}],
        )
        generator.receive_context(
            motif_rules={"forbid": [{"motif_key": "KEEP_MOTIF", "reason": "blocked"}]},
        )
        generator.receive_context(
            rag_knowledge={"frontier_research": ["KEEP_RAG_TITLE rag summary"]},
        )

        prompt = generator.build_prompt(target_paradigm="动量反转", n_factors=1)
        for marker in (
            "KEEP_TEMPLATE_DESCRIPTION", "KEEP_FORBIDDEN", "KEEP_FRONTIER_CARD",
            "KEEP_WARNING", "KEEP_MOTIF", "KEEP_RAG_TITLE", "KEEP_PRIORITY_REASON",
        ):
            self.assertIn(marker, prompt)

    def test_explicit_empty_value_can_clear_one_context_field(self):
        generator = LLMGenerator(output_dir=self.tmp)
        generator.receive_context(success_templates=[{"pattern_id": "REMOVE_ME"}])
        generator.receive_context(success_templates=[])
        self.assertEqual(generator._context["success_templates"], [])

    def test_legacy_switch_reproduces_context_overwrite(self):
        os.environ["FUXI_LLM_CONTEXT_MODE"] = "legacy"
        generator = LLMGenerator(output_dir=self.tmp)
        generator.receive_context(frontier_cards="FIRST")
        generator.receive_context(warning_directions=[{"description": "SECOND"}])
        self.assertEqual(generator._context["frontier_cards"], "")
        self.assertEqual(generator._context["warning_directions"][0]["description"], "SECOND")


if __name__ == "__main__":
    unittest.main()
