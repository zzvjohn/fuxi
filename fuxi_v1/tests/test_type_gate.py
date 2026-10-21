"""P1 类型闸门 A/A 门回归测试。

覆盖方案 P1 验收标准：
- off 时编译行为逐位不变（对合法模板，off/on 输出 formula_hash 一致）；
- on 时金样（templates_expanded 全量）零误杀；
- on 时非法样本（UNITS 合法但 form 冲突）100% 命中违规码；
- off 时非法样本不被拦（保持既有 UNITS 行为）。
"""
import os
import unittest

from fuxi_v1.dsl import compile_ast, type_gate_enabled
from fuxi_v1.search import templates_expanded
from fuxi_v1.contracts import ContractError

FIELDS = {"close": {"unit": "price"}, "open": {"unit": "price"},
          "high": {"unit": "price"}, "low": {"unit": "price"},
          "volume": {"unit": "shares"}, "amount": {"unit": "currency"},
          "turnover_rate": {"unit": "ratio"}, "float_shares": {"unit": "shares"},
          "free_shares": {"unit": "shares"}, "total_shares": {"unit": "shares"}}

# UNITS 相同（shares）但 form 不同（flow vs level）的语义垃圾：
# 现有 UNITS 检查放行，类型闸门拦截 —— 这是类型系统的增量价值样本。
_SEMANTIC_JUNK = {"op": "add", "args": [{"field": "volume"}, {"field": "float_shares"}]}


class TypeGateAATests(unittest.TestCase):
    def setUp(self):
        self._old = os.environ.get("FUXI_TYPE_GATE")

    def tearDown(self):
        if self._old is None:
            os.environ.pop("FUXI_TYPE_GATE", None)
        else:
            os.environ["FUXI_TYPE_GATE"] = self._old

    def test_default_is_off(self):
        os.environ.pop("FUXI_TYPE_GATE", None)
        self.assertFalse(type_gate_enabled())

    def test_off_allows_semantic_junk(self):
        os.environ["FUXI_TYPE_GATE"] = "off"
        # off 时沿用既有 UNITS 检查：volume/float_shares 同 shares，放行
        compile_ast(_SEMANTIC_JUNK, FIELDS)

    def test_on_rejects_semantic_junk(self):
        os.environ["FUXI_TYPE_GATE"] = "on"
        with self.assertRaisesRegex(ContractError, "V_FORM_MISMATCH"):
            compile_ast(_SEMANTIC_JUNK, FIELDS)

    def test_on_zero_false_positive_on_templates(self):
        # 金样：templates_expanded 全量 gate=on 零误杀
        os.environ["FUXI_TYPE_GATE"] = "on"
        for mechanism, ast in templates_expanded(FIELDS):
            compile_ast(ast, FIELDS)  # 不应抛异常

    def test_off_on_bitwise_identical_on_templates(self):
        # 对合法模板，off/on 编译输出逐位一致（on 不改变合法模板结果）
        tpl = templates_expanded(FIELDS)
        os.environ["FUXI_TYPE_GATE"] = "off"
        off = [compile_ast(ast, FIELDS) for _, ast in tpl]
        os.environ["FUXI_TYPE_GATE"] = "on"
        on = [compile_ast(ast, FIELDS) for _, ast in tpl]
        self.assertEqual([r["formula_hash"] for r in off],
                         [r["formula_hash"] for r in on])


if __name__ == "__main__":
    unittest.main()
