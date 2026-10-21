# -*- coding: utf-8 -*-
"""GFlowNet 生产臂（第 6 臂）接线与量纲语法系统契合的回归测试。

覆盖（P-20261021 用户指令「真正把 GFlowNet 注册成生产第 6 臂、进 UCB」）：
1. ProdGrammar token 空间与 token 序列 -> 生产 AST 的还原正确性（to_ast/_parse）。
2. 类型掩码（types_core.TypeState + mask_actions）过滤非法动作（add 跨量纲、log 非无量纲）。
3. sample_asts 产出唯一、且逐条通过 compile_ast（UNITS 量纲 + 类型闸门兜底）。
4. search.generate(arm="gflownet") 冒烟：产出生产 AST，机制/动作元数据正确。
5. scheduler.choose（selection_v2 UCB）与 campaign.choose_arm（legacy UCB）均含 gflownet 臂，
   冷启动轮扫按序选中 gflownet。
6. landing_shadow 生产臂 -> 五臂映射完整性（含 gflownet / random_ast）。

纯 stdlib + fuxi_v1 自身，不依赖 torch（gflownet 臂的 generate 主路径零 torch）。
"""

import unittest

from fuxi_v1.contracts import digest
from fuxi_v1.dsl import compile_ast
from fuxi_v1.gflownet_arm import (
    ProdGrammar, PROD_BINARY, PROD_UNARY,
    _make_type_state, _apply_type_mask, sample_asts,
)
from fuxi_v1.search import generate

# 生产 DSL 10 字段（unit 与 dsl.UNITS / types_core.FIELD_REGISTRY 对齐）。
FIELDS = {
    "open": {"unit": "price"},
    "high": {"unit": "price"},
    "low": {"unit": "price"},
    "close": {"unit": "price"},
    "volume": {"unit": "shares"},
    "amount": {"unit": "currency"},
    "turnover_rate": {"unit": "ratio"},
    "float_shares": {"unit": "shares"},
    "free_shares": {"unit": "shares"},
    "total_shares": {"unit": "shares"},
}

FIELD_NAMES = sorted(FIELDS.keys())


class ProdGrammarTests(unittest.TestCase):
    def setUp(self):
        self.grammar = ProdGrammar(FIELD_NAMES)

    def test_token_space_matches_declared_ops(self):
        self.assertEqual(self.grammar.binary, list(PROD_BINARY))
        self.assertEqual(self.grammar.unary, list(PROD_UNARY))
        self.assertEqual(self.grammar.n_binary, 4)
        self.assertEqual(self.grammar.n_unary, 6)
        self.assertEqual(self.grammar.n_leaf, 10)
        self.assertEqual(self.grammar.n_tokens, 20)
        self.assertEqual(self.grammar.leaf_start, 10)

    def test_to_ast_leaf(self):
        ast = self.grammar.to_ast([self.grammar.token2id["close"]])
        self.assertEqual(ast, {"field": "close"})

    def test_to_ast_unary_compiles(self):
        seq = [self.grammar.token2id["zscore"], self.grammar.token2id["close"]]
        ast = self.grammar.to_ast(seq)
        self.assertEqual(ast, {"op": "zscore", "args": [{"field": "close"}]})
        compile_ast(ast, FIELDS)

    def test_to_ast_binary_compiles(self):
        seq = [self.grammar.token2id["sub"],
               self.grammar.token2id["close"], self.grammar.token2id["open"]]
        ast = self.grammar.to_ast(seq)
        self.assertEqual(ast, {"op": "sub", "args": [{"field": "close"}, {"field": "open"}]})
        compile_ast(ast, FIELDS)

    def test_to_ast_incomplete_returns_none(self):
        # 二元算子缺右子树 -> 索引越界 / 未消费完 -> None
        seq = [self.grammar.token2id["add"], self.grammar.token2id["close"]]
        self.assertIsNone(self.grammar.to_ast(seq))

    def test_to_ast_trailing_returns_none(self):
        # 两个叶子无法构成单棵树 -> 消费不完 -> None
        seq = [self.grammar.token2id["close"], self.grammar.token2id["close"]]
        self.assertIsNone(self.grammar.to_ast(seq))


class TypeMaskTests(unittest.TestCase):
    def test_add_across_dimensions_masked(self):
        grammar = ProdGrammar(FIELD_NAMES)
        ts = _make_type_state(grammar)
        ts.step("add")
        ts.step("open")  # price, dim (1,0)
        # 第二操作数候选：close（price，合法）vs volume（shares，dim 不匹配 -> hard）
        acts = [grammar.token2id["close"], grammar.token2id["volume"]]
        legal = {grammar.id2token[i] for i in _apply_type_mask(ts, grammar, acts)}
        self.assertIn("close", legal)
        self.assertNotIn("volume", legal)

    def test_log_non_dimensionless_masked(self):
        grammar = ProdGrammar(FIELD_NAMES)
        ts = _make_type_state(grammar)
        ts.step("log")
        # 候选：close（price 非无量纲 -> hard）vs turnover_rate（ratio 无量纲且正 -> 合法）
        acts = [grammar.token2id["close"], grammar.token2id["turnover_rate"]]
        legal = {grammar.id2token[i] for i in _apply_type_mask(ts, grammar, acts)}
        self.assertIn("turnover_rate", legal)
        self.assertNotIn("close", legal)

    def test_mask_never_blocks_normalizer(self):
        # rank/zscore 是无条件合法的一元归一器，任何状态都不应被掩码剔除。
        grammar = ProdGrammar(FIELD_NAMES)
        ts = _make_type_state(grammar)
        ts.step("close")
        acts = [grammar.token2id["rank"], grammar.token2id["zscore"]]
        legal = {grammar.id2token[i] for i in _apply_type_mask(ts, grammar, acts)}
        self.assertEqual(legal, {"rank", "zscore"})


class SampleAstsTests(unittest.TestCase):
    def test_sample_asts_unique_and_compilable(self):
        asts = sample_asts(FIELDS, count=8, seed=42)
        self.assertGreater(len(asts), 0)
        self.assertEqual(len(asts), len({digest(a) for a in asts}))
        for ast in asts:
            compile_ast(ast, FIELDS)  # 必须逐条通过量纲 + 类型闸门（gate 默认 off，UNITS 恒开）

    def test_sample_asts_respects_count_ceiling(self):
        asts = sample_asts(FIELDS, count=16, seed=7)
        self.assertLessEqual(len(asts), 16)


class GenerateWiringTests(unittest.TestCase):
    def test_generate_gflownet_smoke(self):
        counters = {}
        proposals = generate(FIELDS, arm="gflownet", count=5, seed=7, counters=counters)
        self.assertGreater(len(proposals), 0)
        for p in proposals:
            self.assertEqual(p["hypothesis"]["mechanism"], "gflownet_discrete_structure")
            self.assertEqual(p["action"], "gflownet_generate")
            self.assertIn("formula_hash", p)  # factor_spec -> compile_ast 已通过
        self.assertGreater(counters.get("generation_attempts", 0), 0)


class UcbWiringTests(unittest.TestCase):
    def test_scheduler_choose_has_gflownet_arm(self):
        from fuxi_v1.scheduler import choose
        arm, detail = choose(0, "adaptive", [], False, context_hash="h" * 64)
        self.assertEqual(arm, "fixed")  # 冷启动首访 = 臂序第一的 fixed
        self.assertIn("gflownet", detail["counters"])

    def test_campaign_choose_arm_cold_start_rounds_to_gflownet(self):
        from fuxi_v1.campaign import choose_arm
        arm0, _ = choose_arm(0, "adaptive", [], False)
        self.assertEqual(arm0, "fixed")
        # legacy 臂序：fixed, random_ast, refine, combine, gflownet -> round 4 = gflownet
        arm4, _ = choose_arm(4, "adaptive", [], False)
        self.assertEqual(arm4, "gflownet")

    def test_landing_shadow_mapping_complete(self):
        from fuxi_v1.landing_shadow import PRODUCTION_ARM_TO_V2, PRODUCTION_ARMS
        self.assertEqual(set(PRODUCTION_ARM_TO_V2), set(PRODUCTION_ARMS))
        self.assertIn("gflownet", PRODUCTION_ARM_TO_V2)
        self.assertIn("random_ast", PRODUCTION_ARM_TO_V2)
        self.assertEqual(PRODUCTION_ARM_TO_V2["gflownet"], "E_explore")


if __name__ == "__main__":
    unittest.main()
