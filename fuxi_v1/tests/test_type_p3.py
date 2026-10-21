"""P3 增量类型状态 + 动作掩码测试。

覆盖：
- TypeState 逐步 step 的最终结果与 infer_prefix 一次性结果一致（双前端等价，R-D 治理）；
- mask_actions 试探式过滤：拦 hard 违规 token、放合法 token（add/sub 跨 form、log 无量纲、div 分母）。
"""
import unittest

from fuxi_v1.types_core import (
    TypeState, mask_actions, infer_prefix, resolve,
    V_FORM_MISMATCH, V_DENOM_SIGNED,
)

_OPS = {"add", "sub", "mul", "div", "rank", "neg", "ts_zscore"}


def _resolver(name):
    return resolve(name, kind="factor")


def _is_op(t):
    return t in _OPS


class TypeStateConsistencyTests(unittest.TestCase):
    """TypeState 增量结果 == infer_prefix 一次性结果（同一 rules.py，双前端等价）。"""

    def _both(self, tokens):
        spec_pre, vs_pre = infer_prefix(tokens, _resolver, _is_op)
        hard_pre = [v.code for v in vs_pre if v.severity == "hard"]
        st = TypeState(_resolver, _is_op)
        for t in tokens:
            st.step(t)
        return spec_pre, hard_pre, st.result(), st.hard_codes

    def test_add_legal(self):
        sp, hp, si, hi = self._both(["add", "ep", "roe"])  # 跨 family 仅 soft
        self.assertEqual(sp, si)
        self.assertEqual(hp, hi)  # 都无 hard

    def test_add_hard_form_mismatch(self):
        sp, hp, si, hi = self._both(["add", "f_score", "ret_1m"])
        self.assertEqual(hp, [V_FORM_MISMATCH])
        self.assertEqual(hi, [V_FORM_MISMATCH])

    def test_div_signed_denominator(self):
        sp, hp, si, hi = self._both(["div", "ep", "ret_1m"])
        self.assertIn(V_DENOM_SIGNED, hp)
        self.assertEqual(hp, hi)

    def test_nested(self):
        sp, hp, si, hi = self._both(["add", "rank", "ret_1m", "neg", "vol_1m"])
        self.assertEqual(sp, si)
        self.assertEqual(hp, hi)


class MaskActionsTests(unittest.TestCase):
    def test_add_second_slot_same_form(self):
        # add 后已选 ep（ratio），第二个槽只有 ratio 叶子合法（add 两侧同 form）
        st = TypeState(_resolver, _is_op)
        st.step("add")
        st.step("ep")
        legal = mask_actions(st, ["ret_1m", "roe", "f_score", "vol_1m"])
        self.assertIn("roe", legal)        # ratio 合法
        self.assertNotIn("ret_1m", legal)  # return 跨 form → 硬违规
        self.assertNotIn("f_score", legal)  # score 跨 form → 硬违规
        self.assertNotIn("vol_1m", legal)  # dispersion 跨 form → 硬违规

    def test_div_denominator_shape(self):
        # div 后已选分子 ep，分母只允许 non-signed（positive/bounded）
        st = TypeState(_resolver, _is_op)
        st.step("div")
        st.step("ep")
        legal = mask_actions(st, ["ret_1m", "vol_1m", "roe"])
        self.assertIn("vol_1m", legal)      # positive 合法
        self.assertNotIn("ret_1m", legal)   # return signed → 硬违规
        self.assertNotIn("roe", legal)      # ratio signed → 硬违规

    def test_root_any_allows_all(self):
        # 根槽位无约束：所有叶子都合法（类型层面）
        st = TypeState(_resolver, _is_op)
        legal = mask_actions(st, ["ep", "ln_mcap", "ret_1m", "f_score"])
        self.assertEqual(set(legal), {"ep", "ln_mcap", "ret_1m", "f_score"})

    def test_mul_open_no_mask(self):
        # mul 无 hard 约束，跨 family 交互放开：第二个槽任何叶子合法
        st = TypeState(_resolver, _is_op)
        st.step("mul")
        st.step("ep")
        legal = mask_actions(st, ["ret_1m", "f_score", "vol_1m"])
        self.assertEqual(set(legal), {"ret_1m", "f_score", "vol_1m"})


if __name__ == "__main__":
    unittest.main()
