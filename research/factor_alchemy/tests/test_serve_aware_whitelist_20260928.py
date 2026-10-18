# -*- coding: utf-8 -*-
"""P2-1b (2026-09-28) 单测: 范式级白名单降级 (FUXI_SERVE_AWARE_SCOPE)。

覆盖矩阵: {off, shadow, on} × {global, whitelist} × {白名单内, 白名单外,
有 profile, 空范式} × {forge, 非 forge}。
"""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import serve_aware as sa


def _dir(para, name="t"):
    return SimpleNamespace(paradigm=para, name=name)


_PROFILES = {"尾部风险": {}, "动量反转": {}}


class ScopeParseTests(unittest.TestCase):
    def test_default_is_whitelist(self):
        self.assertEqual(sa.resolve_serve_aware_scope({}), "whitelist")

    def test_explicit_global(self):
        self.assertEqual(sa.resolve_serve_aware_scope(
            {"FUXI_SERVE_AWARE_SCOPE": "global"}), "global")

    def test_explicit_whitelist(self):
        self.assertEqual(sa.resolve_serve_aware_scope(
            {"FUXI_SERVE_AWARE_SCOPE": "WHITELIST"}), "whitelist")

    def test_illegal_falls_back_to_default(self):
        self.assertEqual(sa.resolve_serve_aware_scope(
            {"FUXI_SERVE_AWARE_SCOPE": "bogus"}), "whitelist")


class WhitelistConstantTests(unittest.TestCase):
    def test_frozen_tuple(self):
        self.assertIsInstance(sa.SERVE_AWARE_WHITELIST, tuple)

    def test_contains_fund_flow(self):
        self.assertIn("资金流", sa.SERVE_AWARE_WHITELIST)

    def test_does_not_contain_opposed_paradigm(self):
        # P2-1 轮 A vs C 严格配对: 事件驱动 oppose (不达准入) ⇒ 不得入白名单。
        self.assertNotIn("事件驱动", sa.SERVE_AWARE_WHITELIST)

    def test_disjoint_from_forge_profiles(self):
        # 白名单条目若同时有 Forge profile, serve_aware_adjust 会先命中
        # profile 分支 ⇒ 该条目永不生效 (冗余/误导)。防回归。
        from forge.paradigm_profiles import MATURE_PARADIGM_PROFILES
        overlap = set(sa.SERVE_AWARE_WHITELIST) & set(MATURE_PARADIGM_PROFILES)
        self.assertFalse(overlap, "白名单与 profile 重叠: %s" % overlap)


class AdjustMatrixTests(unittest.TestCase):
    def test_off_never_acts(self):
        for para in ("资金流", "事件驱动"):
            for scope in ("global", "whitelist"):
                self.assertEqual(sa.serve_aware_adjust(
                    _dir(para), "forge", mode="off", profiles=_PROFILES,
                    scope=scope), "forge")

    def test_shadow_never_changes_return(self):
        for scope in ("global", "whitelist"):
            for para in ("资金流", "事件驱动"):
                self.assertEqual(sa.serve_aware_adjust(
                    _dir(para), "forge", mode="shadow", profiles=_PROFILES,
                    scope=scope), "forge")

    def test_on_whitelist_downgrades_only_listed(self):
        self.assertEqual(sa.serve_aware_adjust(
            _dir("资金流"), "forge", mode="on", profiles=_PROFILES,
            scope="whitelist"), "llm")
        self.assertEqual(sa.serve_aware_adjust(
            _dir("事件驱动"), "forge", mode="on", profiles=_PROFILES,
            scope="whitelist"), "forge")

    def test_on_global_keeps_p2m1_semantics(self):
        # P2-1 复现档: 全局降级照旧 (白名单外也降级)。
        for para in ("资金流", "事件驱动"):
            self.assertEqual(sa.serve_aware_adjust(
                _dir(para), "forge", mode="on", profiles=_PROFILES,
                scope="global"), "llm")

    def test_default_scope_used_when_not_passed(self):
        # 不传 scope ⇒ 读 env; 测试环境无该 env ⇒ whitelist ⇒ 拦截。
        self.assertEqual(sa.serve_aware_adjust(
            _dir("事件驱动"), "forge", mode="on", profiles=_PROFILES), "forge")

    def test_profile_covered_paradigm_never_touched(self):
        for mode in ("off", "shadow", "on"):
            for scope in ("global", "whitelist"):
                self.assertEqual(sa.serve_aware_adjust(
                    _dir("尾部风险"), "forge", mode=mode, profiles=_PROFILES,
                    scope=scope), "forge")

    def test_non_forge_generators_never_touched(self):
        for gen in ("llm", "gp_breed", "gp_evolve", "cross_breed"):
            self.assertEqual(sa.serve_aware_adjust(
                _dir("资金流"), gen, mode="on", profiles=_PROFILES,
                scope="whitelist"), gen)

    def test_empty_paradigm_no_action(self):
        self.assertEqual(sa.serve_aware_adjust(
            _dir(""), "forge", mode="on", profiles=_PROFILES,
            scope="whitelist"), "forge")

    def test_explicit_whitelist_injection(self):
        # 空白名单 = fail-safe 全拦截 (等价关闭降级)。
        self.assertEqual(sa.serve_aware_adjust(
            _dir("资金流"), "forge", mode="on", profiles=_PROFILES,
            scope="whitelist", whitelist=()), "forge")
        # 注入扩展仅限单测/隔离; 生产走预注册常量 (无 env 通道)。
        self.assertEqual(sa.serve_aware_adjust(
            _dir("事件驱动"), "forge", mode="on", profiles=_PROFILES,
            scope="whitelist", whitelist=("事件驱动",)), "llm")


if __name__ == "__main__":
    unittest.main()
