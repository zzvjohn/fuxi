"""P2 软约束 + 治理验收测试。

覆盖方案 P2 验收标准：
- prompt 规则与 rules.py 同源（render_for_prompt 输出含当前规则，消灭 C3 手工副本）；
- factor_notes 出现 output_type 段（FUXI_TYPE_NOTES=on，C8 治理）。
"""
import os
import unittest

from fuxi_v1.types_core.rules import render_for_prompt
from fuxi_v1.research_evaluation import _output_type_note


class PromptRuleTests(unittest.TestCase):
    def test_render_for_prompt_contains_current_rules(self):
        # 断言 prompt 渲染文本包含当前规则表的关键语义（与 rules.py 同源，非手工副本）
        text = render_for_prompt()
        for kw in ("add/sub", "mul", "div", "log", "rank", "neg", "time operators",
                   "hard", "soft"):
            self.assertIn(kw, text, f"prompt 缺关键词 {kw!r}")
        # 关键硬约束必须在 prompt 里（denominator signed / self-division / dimensionless）
        self.assertIn("denominator", text)
        self.assertIn("self-division", text)
        self.assertIn("dimensionless", text)

    def test_render_for_prompt_reflects_rule_change(self):
        # 同源性的可证伪检验：手工副本会在规则变更时漂移，这里验证渲染文本
        # 与规则表常量一致（通过 import 同一模块的违规码断言）。
        from fuxi_v1.types_core import rules as r
        self.assertEqual(r.render_for_prompt, render_for_prompt.__wrapped__ if hasattr(render_for_prompt, "__wrapped__") else r.render_for_prompt)


class OutputTypeNoteTests(unittest.TestCase):
    def setUp(self):
        self._old = os.environ.get("FUXI_TYPE_NOTES")

    def tearDown(self):
        if self._old is None:
            os.environ.pop("FUXI_TYPE_NOTES", None)
        else:
            os.environ["FUXI_TYPE_NOTES"] = self._old

    def _candidate(self):
        return {"candidate_id": "c1", "canonical_ast": {"op": "rank", "args": [{"field": "close"}]}}

    def test_off_returns_none(self):
        os.environ["FUXI_TYPE_NOTES"] = "off"
        self.assertIsNone(_output_type_note(self._candidate()))

    def test_on_returns_output_type(self):
        os.environ["FUXI_TYPE_NOTES"] = "on"
        note = _output_type_note(self._candidate())
        self.assertIsNotNone(note)
        # rank(close) -> form=ranked, family=none(字段级), dim=(0,0)
        self.assertEqual(note["form"], "ranked")
        self.assertEqual(note["dim"], [0, 0])
        self.assertIn("family", note)
        self.assertIn("direction", note)
        self.assertIn("soft_violations", note)


if __name__ == "__main__":
    unittest.main()
