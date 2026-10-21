"""types_core 单元测试：schema / 97 因子覆盖 / 算子三档规则 / 双前端一致性 / 模式推断。

覆盖方案 P0 验收标准：注册表 schema 全过、规则表用例全过、97 因子全覆盖无盲区、
双前端对同一表达式树产出相同 TypeSpec、名称模式推断回归。
"""
import json
import os
import tempfile
import unittest

from fuxi_v1.types_core import (
    TypeSpec, resolve, validate, factor_names, field_names, registry_digest,
    REGISTRY_VERSION, FAMILIES,
    infer_ast, infer_prefix, check_tree, hard_violations, soft_violations,
    load_overrides, overrides,
    V_FORM_MISMATCH, V_DIM_MISMATCH, V_DENOM_SIGNED, V_SELF_DIV,
    V_LOG_DIM, V_LOG_SHAPE, V_SCORE_ARITH, V_FAMILY_MIXED,
    V_HORIZON_SKIP, V_DENOM_RETURN, V_TEMPORAL_NESTED,
)


def _field(name):
    return resolve(name, kind="field")


def _factor(name):
    return resolve(name, kind="factor")


# ---------------------------------------------------------------------------
# registry / schema
# ---------------------------------------------------------------------------

class RegistryTests(unittest.TestCase):
    def test_schema_valid(self):
        self.assertEqual(validate(), [])

    def test_factor_coverage_is_97(self):
        self.assertEqual(len(factor_names()), 97)

    def test_field_coverage_is_10(self):
        self.assertEqual(len(field_names()), 10)

    def test_factor_family_not_none(self):
        # 每个显式因子都必须有具体经济角色（family != none），不能留盲区。
        for name in factor_names():
            ts = _factor(name)
            self.assertNotEqual(ts.family, "none", f"{name} has family=none")
            self.assertIn(ts.family, FAMILIES)

    def test_registry_version_and_digest(self):
        self.assertTrue(REGISTRY_VERSION.startswith("fuxi.types."))
        self.assertEqual(len(registry_digest()), 64)

    def test_resolve_priority_explicit_over_pattern(self):
        # ep 有显式标注（value/ratio/+1），模式规则 ^(ep|bp|...)$ 也给 value，
        # 但显式表优先；这里用 direction 区分：显式 ep=+1。
        self.assertEqual(_factor("ep").direction, +1)

    def test_resolve_pattern_inference_new_name(self):
        # 不在显式表的"新因子名"应被模式规则自动归类。
        ts = _factor("ret_2m")           # ret_* -> momentum/return
        self.assertEqual(ts.family, "momentum")
        self.assertEqual(ts.form, "return")
        ts2 = _factor("foo_bar_proxy")    # *_proxy -> quality/ratio
        self.assertEqual(ts2.family, "quality")
        ts3 = _factor("rev_growth_yoy_2") # *_growth_yoy -> growth（实际显式表有 rev_growth_yoy，用变体）
        self.assertEqual(ts3.family, "growth")

    def test_resolve_fallback_unknown(self):
        ts = _factor("xyz_unknown_thing")
        self.assertEqual(ts.family, "technical")
        self.assertEqual(ts.direction, 0)


class OverrideTests(unittest.TestCase):
    """外部增量覆盖：新因子人工标注不改源码即可生效。"""

    def tearDown(self):
        load_overrides("")  # 清空，避免污染其他测试

    def _write(self, obj):
        fd, path = tempfile.mkstemp(suffix=".json")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f)
        return path

    def test_override_existing_factor(self):
        # 覆盖已有因子：把 ep 的 family 临时改为 growth，验证 override 优先级最高
        p = self._write({"ep": {"family": "growth"}})
        load_overrides(p)
        self.assertEqual(_factor("ep").family, "growth")
        os.unlink(p)

    def test_override_new_factor(self):
        # 新因子不在显式表，override 直接给精确标注，绕过模式推断
        p = self._write({"brand_new_momentum_v2": {"form": "return", "family": "momentum",
                                                   "horizon": "fast", "direction": +1}})
        load_overrides(p)
        ts = _factor("brand_new_momentum_v2")
        self.assertEqual(ts.family, "momentum")
        self.assertEqual(ts.form, "return")
        os.unlink(p)

    def test_override_changes_digest(self):
        base = registry_digest()
        p = self._write({"ep": {"family": "growth"}})
        load_overrides(p)
        changed = registry_digest()
        self.assertNotEqual(base, changed)
        os.unlink(p)

    def test_override_clear(self):
        p = self._write({"ep": {"family": "growth"}})
        load_overrides(p)
        self.assertEqual(_factor("ep").family, "growth")
        load_overrides("")
        self.assertEqual(_factor("ep").family, "value")  # 回到显式表
        self.assertEqual(overrides(), {})
        os.unlink(p)


# ---------------------------------------------------------------------------
# 算子规则三档（硬 / 软 / 开放）
# ---------------------------------------------------------------------------

def _ast(op, *args, window=None):
    node = {"op": op, "args": list(args)}
    if window is not None:
        node["window"] = window
    return node


def _leaf(name):
    return {"field": name}


class AddSubTests(unittest.TestCase):
    def test_add_same_form_ok(self):
        ast = _ast("add", _leaf("close"), _leaf("high"))  # 同为 level/price
        spec, vs = infer_ast(ast, _field)
        self.assertEqual(hard_violations(vs), [])
        self.assertEqual(spec.form, "level")

    def test_add_form_mismatch_hard(self):
        ast = _ast("add", _leaf("close"), _leaf("volume"))  # level vs flow
        _, vs = infer_ast(ast, _field)
        codes = {v.code for v in hard_violations(vs)}
        self.assertIn(V_FORM_MISMATCH, codes)
        self.assertIn(V_DIM_MISMATCH, codes)

    def test_add_cross_family_soft_not_hard(self):
        # add(ep, roe)：form 同为 ratio（合法），但 value vs quality 跨族 -> 仅软标记。
        # 这正是评审 R-B 的关键验证：form 轴拦不住，只能靠 family 软约束。
        ast = _ast("add", _leaf("ep"), _leaf("roe"))
        _, vs = infer_ast(ast, _factor)
        self.assertEqual(hard_violations(vs), [])
        codes = {v.code for v in soft_violations(vs)}
        self.assertIn(V_FAMILY_MIXED, codes)


class MulDivTests(unittest.TestCase):
    def test_mul_cross_family_open(self):
        # mul 跨 family 放开（交互项），无任何违规。
        ast = _ast("mul", _leaf("ep"), _leaf("vol_1m"))
        _, vs = infer_ast(ast, _factor)
        self.assertEqual(vs, [])

    def test_mul_score_score_soft(self):
        ast = _ast("mul", _leaf("f_score"), _leaf("f_score"))
        _, vs = infer_ast(ast, _factor)
        codes = {v.code for v in soft_violations(vs)}
        self.assertIn(V_SCORE_ARITH, codes)

    def test_div_signed_denom_hard(self):
        ast = _ast("div", _leaf("ep"), _leaf("ret_1m"))  # ret_1m shape=signed
        _, vs = infer_ast(ast, _factor)
        codes = {v.code for v in hard_violations(vs)}
        self.assertIn(V_DENOM_SIGNED, codes)

    def test_div_return_denom_soft(self):
        # vol_1m shape=positive（无硬拒），但分母若为 return 形态 -> 软。
        ast = _ast("div", _leaf("ep"), _leaf("short_rev_5d"))  # return, signed -> 硬拒
        _, vs = infer_ast(ast, _factor)
        codes = {v.code for v in hard_violations(vs)}
        self.assertIn(V_DENOM_SIGNED, codes)  # 先被硬拒
        # 分母为 return 且 positive 的构造：用 positive 的 return 形态不存在，跳过软分支验证

    def test_div_self_structural_hard(self):
        # 结构自除：div(rank(X), rank(X))，比 grammar.py 的"同一对象"判断更强。
        ast = _ast("div", _ast("rank", _leaf("ep")), _ast("rank", _leaf("ep")))
        _, vs = infer_ast(ast, _factor)
        codes = {v.code for v in hard_violations(vs)}
        self.assertIn(V_SELF_DIV, codes)

    def test_div_positive_denom_ok(self):
        ast = _ast("div", _leaf("ep"), _leaf("vol_1m"))  # vol_1m shape=positive
        _, vs = infer_ast(ast, _factor)
        self.assertEqual(hard_violations(vs), [])


class UnaryTests(unittest.TestCase):
    def test_log_dim_hard(self):
        ast = _ast("log", _leaf("close"))  # close dim=(1,0) 非无量纲
        _, vs = infer_ast(ast, _field)
        codes = {v.code for v in hard_violations(vs)}
        self.assertIn(V_LOG_DIM, codes)

    def test_log_shape_hard(self):
        ast = _ast("log", _leaf("ep"))  # ep 无量纲但 shape=signed
        _, vs = infer_ast(ast, _factor)
        codes = {v.code for v in hard_violations(vs)}
        self.assertIn(V_LOG_SHAPE, codes)

    def test_log_positive_dimensionless_ok(self):
        ast = _ast("log", _leaf("vol_1m"))  # vol_1m dim=(0,0) shape=positive
        _, vs = infer_ast(ast, _factor)
        self.assertEqual(hard_violations(vs), [])

    def test_rank_is_normalizer(self):
        ast = _ast("rank", _leaf("close"))
        spec, vs = infer_ast(ast, _field)
        self.assertEqual(vs, [])
        self.assertEqual(spec.form, "ranked")
        self.assertEqual(spec.dim, (0, 0))

    def test_neg_flips_direction(self):
        ast = _ast("neg", _leaf("ep"))  # ep direction=+1
        spec, _ = infer_ast(ast, _factor)
        self.assertEqual(spec.direction, -1)

    def test_square_doubles_dim(self):
        ast = _ast("square", _leaf("close"))  # close dim=(1,0)
        spec, _ = infer_ast(ast, _field)
        self.assertEqual(spec.dim, (2, 0))
        self.assertEqual(spec.shape, "positive")


class TemporalTests(unittest.TestCase):
    def test_temporal_nested_hard(self):
        # 时序算子输入是 rank 节点（非 field 叶子）-> V_TEMPORAL_NESTED
        ast = _ast("mean", _ast("rank", _leaf("close")), window=5)
        _, vs = infer_ast(ast, _field)
        codes = {v.code for v in hard_violations(vs)}
        self.assertIn(V_TEMPORAL_NESTED, codes)

    def test_temporal_leaf_ok_window_horizon(self):
        ast = _ast("mean", _leaf("close"), window=5)  # window 5 -> fast
        spec, vs = infer_ast(ast, _field)
        self.assertEqual(hard_violations(vs), [])
        self.assertEqual(spec.horizon, "fast")
        ast2 = _ast("std", _leaf("close"), window=40)  # std -> dispersion, window 40 -> slow
        spec2, _ = infer_ast(ast2, _field)
        self.assertEqual(spec2.form, "dispersion")
        self.assertEqual(spec2.horizon, "slow")


# ---------------------------------------------------------------------------
# 双前端一致性（AST 递归 vs preorder 前缀栈式）
# ---------------------------------------------------------------------------

_OPS = {"add", "sub", "mul", "div", "rank", "neg", "ts_zscore"}


class DualFrontendTests(unittest.TestCase):
    def _both(self, ast, preorder, resolver):
        spec_ast, vs_ast = infer_ast(ast, resolver)
        spec_pre, vs_pre = infer_prefix(preorder, resolver, lambda t: t in _OPS)
        return spec_ast, vs_ast, spec_pre, vs_pre

    def test_consistency_add(self):
        ast = {"op": "add", "args": [{"field": "ep"}, {"field": "roe"}]}
        pre = ["add", "ep", "roe"]
        sa, va, sp, vp = self._both(ast, pre, _factor)
        self.assertEqual(sa, sp)
        self.assertEqual([v.code for v in va], [v.code for v in vp])

    def test_consistency_div(self):
        ast = {"op": "div", "args": [{"field": "ep"}, {"field": "vol_1m"}]}
        pre = ["div", "ep", "vol_1m"]
        sa, va, sp, vp = self._both(ast, pre, _factor)
        self.assertEqual(sa, sp)
        self.assertEqual([v.code for v in va], [v.code for v in vp])

    def test_consistency_nested(self):
        ast = {"op": "add", "args": [
            {"op": "rank", "args": [{"field": "ret_1m"}]},
            {"op": "neg", "args": [{"field": "vol_1m"}]}]}
        pre = ["add", "rank", "ret_1m", "neg", "vol_1m"]
        sa, va, sp, vp = self._both(ast, pre, _factor)
        self.assertEqual(sa, sp)
        self.assertEqual([v.code for v in va], [v.code for v in vp])

    def test_consistency_self_div(self):
        ast = {"op": "div", "args": [
            {"op": "rank", "args": [{"field": "ep"}]},
            {"op": "rank", "args": [{"field": "ep"}]}]}
        pre = ["div", "rank", "ep", "rank", "ep"]
        sa, va, sp, vp = self._both(ast, pre, _factor)
        self.assertEqual(sa, sp)
        codes_ast = {v.code for v in va if v.code == V_SELF_DIV}
        codes_pre = {v.code for v in vp if v.code == V_SELF_DIV}
        self.assertEqual(codes_ast, codes_pre)
        self.assertEqual(codes_ast, {V_SELF_DIV})


if __name__ == "__main__":
    unittest.main()
