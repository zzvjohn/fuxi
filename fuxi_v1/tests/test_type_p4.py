"""P4 语义冗余诊断测试（G3）+ 误杀率统计口径。

G3：pool_redundancy 的 family 冗余诊断（note-only）——同 family + 相关>0.7 标注
redundancy_semantic，仅诊断不升门控，观察 2 周误杀率。
"""
import os
import unittest

from fuxi_v1.research_evaluation import _semantic_redundancy_note


def _rank_return(field, window):
    return {"op": "rank", "args": [{"op": "return", "window": window, "args": [{"field": field}]}]}


def _rank_volume_surprise(window):
    return {"op": "rank", "args": [{"op": "div", "args": [
        {"field": "volume"}, {"op": "mean", "window": window, "args": [{"field": "volume"}]}]}]}


class SemanticRedundancyTests(unittest.TestCase):
    def setUp(self):
        self._old = os.environ.get("FUXI_TYPE_NOTES")

    def tearDown(self):
        if self._old is None:
            os.environ.pop("FUXI_TYPE_NOTES", None)
        else:
            os.environ["FUXI_TYPE_NOTES"] = self._old

    def test_off_returns_none(self):
        os.environ["FUXI_TYPE_NOTES"] = "off"
        self.assertIsNone(_semantic_redundancy_note(_rank_return("close", 5), _rank_return("close", 10)))

    def test_on_same_family_momentum(self):
        os.environ["FUXI_TYPE_NOTES"] = "on"
        # 两个动量因子（return close 5d vs 10d）→ 同 family=momentum → redundancy_semantic=True
        note = _semantic_redundancy_note(_rank_return("close", 5), _rank_return("close", 10))
        self.assertIsNotNone(note)
        self.assertEqual(note["candidate_family"], "momentum")
        self.assertEqual(note["pool_family"], "momentum")
        self.assertTrue(note["same_family"])
        self.assertTrue(note["redundancy_semantic"])

    def test_on_cross_family(self):
        os.environ["FUXI_TYPE_NOTES"] = "on"
        # 动量 vs 流动性（volume_surprise 是 liquidity）→ 跨 family → redundancy_semantic=False
        note = _semantic_redundancy_note(_rank_return("close", 5), _rank_volume_surprise(5))
        self.assertIsNotNone(note)
        self.assertFalse(note["same_family"])
        self.assertFalse(note["redundancy_semantic"])

    def test_on_field_level_none_not_same(self):
        os.environ["FUXI_TYPE_NOTES"] = "on"
        # 字段级 family=none 不算同 family（none 是未分类，不构成语义冗余）
        note = _semantic_redundancy_note(
            {"op": "rank", "args": [{"field": "close"}]},
            {"op": "rank", "args": [{"field": "high"}]})
        self.assertIsNotNone(note)
        self.assertFalse(note["redundancy_semantic"])


if __name__ == "__main__":
    unittest.main()
