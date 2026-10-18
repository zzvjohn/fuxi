from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from incremental_contract import (
    MODEL_COMPARE_AXES,
    MODEL_COMPARE_BASELINE,
    IncrementSpecError,
    axes_of,
    check_increment,
    classify_key,
    design_fields_from_source,
    feature_delta,
    protocol_sha256,
)

ROOT = Path(__file__).resolve().parents[3]
PROTOCOL = ROOT / MODEL_COMPARE_BASELINE
MODEL_COMPARE_SCRIPT = ROOT / "scripts" / "fuxi_model_compare_20260924.py"
SPEC = {"axes": MODEL_COMPARE_AXES}


def _real():
    return json.loads(PROTOCOL.read_text(encoding="utf-8"))


def _fresh_features():
    """按生产形状取特征表（含 coverage 读数）。"""
    return [dict(f) for f in _real()["features"]]


class KeyClassificationTests(unittest.TestCase):
    def test_provenance_names_are_bookkeeping_and_everything_else_is_design(self):
        for key in ("hashes", "created_at", "revisions", "sha256", "batch", "purpose"):
            self.assertEqual(classify_key(key), "BOOKKEEPING", key)
        for key in ("features", "top_n", "label", "training", "pool", "unknown_name"):
            self.assertEqual(classify_key(key), "DESIGN", key)

    def test_a_spec_can_reclassify_a_key_but_only_by_name(self):
        spec = {"reading_keys": ("pool",), "bookkeeping_keys": ("whatever",)}
        self.assertEqual(classify_key("pool", spec), "READING")
        self.assertEqual(classify_key("whatever", spec), "BOOKKEEPING")
        self.assertEqual(classify_key("top_n", spec), "DESIGN")


class AxisDeclarationTests(unittest.TestCase):
    def test_every_declared_axis_member_must_be_a_design_field(self):
        with self.assertRaises(IncrementSpecError):
            axes_of({"axes": {"x": ("features", "not_a_field")}}, ["features"])

    def test_a_field_cannot_be_declared_by_two_axes(self):
        with self.assertRaises(IncrementSpecError):
            axes_of({"axes": {"a": ("features",), "b": ("features",)}}, ["features"])

    def test_an_empty_axis_is_refused(self):
        with self.assertRaises(IncrementSpecError):
            axes_of({"axes": {"a": ()}}, ["features"])

    def test_fields_not_covered_by_any_axis_become_their_own_axis(self):
        axes = axes_of({"axes": {"x": ("features",)}}, ["features", "top_n"])
        self.assertEqual(axes["x"], ("features",))
        self.assertEqual(axes["top_n"], ("top_n",))

    def test_the_axes_cover_every_design_field_of_the_production_script(self):
        """🔴 测试形状必须与生产一致。

        契约里声明的轴必须**逐字**覆盖生产脚本的 ``DESIGN_FIELDS`` 字面量；
        否则守卫会有盲区 —— 生产里新增一个设计字段而契约不知道，那它就永远不会
        阻止「同时改两个维度」。
        """
        production = design_fields_from_source(MODEL_COMPARE_SCRIPT)
        self.assertTrue(production)
        covered = tuple(m for members in MODEL_COMPARE_AXES.values() for m in members)
        self.assertEqual(sorted(covered), sorted(production))
        self.assertEqual(len(covered), len(set(covered)))


class FeatureDeltaTests(unittest.TestCase):
    def test_coverage_is_a_reading_not_a_design_change(self):
        """🔴 [mem:AD5]：补数据会让 coverage 变，但那不是设计变更。

        若把 coverage 算进设计比较，每一次数据延伸都会把「同一次增量」读成一次设计
        改动 —— 于是「补数据」和「改设计」在账上不可区分。
        """
        base = _fresh_features()
        cand = copy.deepcopy(base)
        cand[0]["coverage"] = {"median_rate": 0.0, "min_rate": 0.0, "sections": 137}
        delta = feature_delta(base, cand)
        self.assertEqual(delta["touched"], 0)
        self.assertEqual(delta["added"], [])
        self.assertEqual(delta["removed"], [])
        self.assertEqual(delta["modified"], [])
        self.assertEqual(delta["readings_changed"], [base[0]["name"]])

    def test_a_direction_edit_counts_as_one_touched_feature(self):
        base = _fresh_features()
        cand = copy.deepcopy(base)
        cand[2]["direction"] = -cand[2]["direction"]
        delta = feature_delta(base, cand)
        self.assertEqual(delta["touched"], 1)
        self.assertEqual(delta["modified"], [base[2]["name"]])
        self.assertEqual(delta["field_edits"][0]["field"], "direction")

    def test_a_rename_counts_as_two_touched_features(self):
        base = _fresh_features()
        cand = copy.deepcopy(base)
        cand[2]["name"] = "renamed"
        delta = feature_delta(base, cand)
        self.assertEqual(delta["touched"], 2, "改名 = 1 增 + 1 删 ⇒ 触碰 2 个特征")
        self.assertEqual(delta["added"], ["renamed"])
        self.assertEqual(delta["removed"], [base[2]["name"]])

    def test_reordering_is_reported_but_does_not_touch_a_feature(self):
        base = _fresh_features()
        cand = list(reversed(copy.deepcopy(base)))
        delta = feature_delta(base, cand)
        self.assertEqual(delta["touched"], 0)
        self.assertTrue(delta["order_changed"])


class VerdictTests(unittest.TestCase):
    def test_a_bookkeeping_only_change_is_identical_and_reported(self):
        base = _real()
        cand = copy.deepcopy(base)
        cand["hashes"] = dict(cand["hashes"])
        cand["hashes"]["data/raw/daily_prices.csv"] = "0" * 64
        out = check_increment(base, cand, spec=SPEC)
        self.assertEqual(out["verdict"], "IDENTICAL")
        self.assertTrue(out["accepted"])
        self.assertFalse(out["is_an_increment"])
        self.assertEqual(out["changed_design_fields"], [])
        self.assertEqual(out["changed_bookkeeping_fields"], ["hashes"])

    def test_one_axis_is_accepted_and_two_axes_are_refused(self):
        base = _real()
        one = copy.deepcopy(base)
        one["top_n"] = 100
        one["selection"] = one["selection"].replace("Top80", "Top100")
        out = check_increment(base, one, spec=SPEC)
        self.assertEqual(out["verdict"], "SINGLE_AXIS")
        self.assertTrue(out["accepted"])
        self.assertTrue(out["is_an_increment"])
        self.assertEqual(out["axes_changed"], ["selection"])

        two = copy.deepcopy(one)
        two["cost_bps"] = [0, 15, 30, 45]
        out = check_increment(base, two, spec=SPEC)
        self.assertEqual(out["verdict"], "MULTI_AXIS")
        self.assertTrue(out["refuses"])
        self.assertEqual(out["axes_changed"], ["cost", "selection"])

    def test_adding_one_candidate_feature_is_a_single_axis_increment(self):
        """加一个候选特征会同时动 features / dropped_by_coverage ⇒ 必须算**一个**轴。"""
        base = _real()
        cand = copy.deepcopy(base)
        pe = {"name": "pe_ttm", "kind": "snapshot", "direction": -1, "source": "pe_ttm",
              "coverage": {"median_rate": 0.7232, "min_rate": 0.0, "sections": 89}}
        cand["features"] = sorted(base["features"] + [pe], key=lambda f: f["name"])
        cand["dropped_by_coverage"] = []
        out = check_increment(base, cand, spec=SPEC)
        self.assertEqual(out["axes_changed"], ["feature_set"])
        self.assertEqual(out["verdict"], "SINGLE_AXIS")
        self.assertTrue(out["accepted"])
        self.assertEqual(out["feature_delta"]["added"], ["pe_ttm"])
        self.assertEqual(out["feature_delta"]["touched"], 1)
        self.assertIn("features", out["changed_design_fields"])
        self.assertIn("dropped_by_coverage", out["changed_design_fields"])

    def test_adding_two_features_at_once_is_refused(self):
        base = _real()
        cand = copy.deepcopy(base)
        extra = [{"name": n, "kind": "snapshot", "direction": -1, "source": n,
                  "coverage": {"median_rate": 0.9, "min_rate": 0.0, "sections": 89}}
                 for n in ("pe_ttm", "sf1")]
        cand["features"] = sorted(base["features"] + extra, key=lambda f: f["name"])
        cand["dropped_by_coverage"] = []
        out = check_increment(base, cand, spec=SPEC)
        self.assertEqual(out["verdict"], "AXIS_ITEM_BATCH")
        self.assertTrue(out["refuses"])
        self.assertEqual(out["feature_delta"]["touched"], 2)

    def test_an_unknown_key_is_refused_because_it_can_smuggle_dimensions(self):
        """🔴 一个未写下来的键名背后可以装任意多个维度。

        按「名字」看它只动了一处，于是会被读成单维增量。所以只要它没被写下来
        （未进轴、未登记为记账/读数），就必须拒绝 —— 不是洁癖，是夹带口子。
        """
        base = _real()
        cand = copy.deepcopy(base)
        cand["whatever_new_knob"] = 1
        out = check_increment(base, cand, spec=SPEC)
        self.assertEqual(out["verdict"], "UNDECLARED_KEY")
        self.assertTrue(out["refuses"])
        self.assertIn("whatever_new_knob", out["changed_design_fields"])
        self.assertEqual(out["undeclared_keys"]["not_explicitly_declared"],
                         ["whatever_new_knob"])
        self.assertEqual(out["undeclared_design_keys_that_were_not_written_down"],
                         ["whatever_new_knob"])
        self.assertEqual(out["flags"][0]["code"], "UNDECLARED_KEY")

    def test_writing_the_name_down_is_what_makes_it_legal(self):
        base = _real()
        for spec, expected in (
                ({"axes": MODEL_COMPARE_AXES, "bookkeeping_keys": ("whatever_new_knob",)},
                 "IDENTICAL"),
                ({"axes": {**MODEL_COMPARE_AXES, "misc": ("whatever_new_knob",)}},
                 "SINGLE_AXIS"),
        ):
            cand = copy.deepcopy(base)
            cand["whatever_new_knob"] = 1
            out = check_increment(base, cand, spec=spec)
            self.assertEqual(out["verdict"], expected)
            self.assertTrue(out["accepted"], spec)

    def test_the_escape_hatch_is_explicit_and_defaults_off(self):
        base = _real()
        cand = copy.deepcopy(base)
        cand["whatever_new_knob"] = 1
        default = check_increment(base, cand, spec=SPEC)
        widened = check_increment(base, cand, spec=SPEC, allow_new_design_keys=True)
        self.assertFalse(default["allow_new_design_keys"])
        self.assertTrue(default["refuses"])
        self.assertTrue(widened["accepted"])
        self.assertEqual(widened["verdict"], "SINGLE_AXIS")
        self.assertEqual(widened["flags"][0]["code"], "UNDECLARED_KEY",
                         "放宽之后仍然要报出来")

    def test_a_new_bookkeeping_key_is_not_a_design_change(self):
        base = _real()
        cand = copy.deepcopy(base)
        cand["generated_at"] = "2026-09-25T00:00:00+08:00"
        out = check_increment(base, cand, spec={"axes": MODEL_COMPARE_AXES,
                                               "bookkeeping_keys": ("generated_at",)})
        self.assertEqual(out["verdict"], "IDENTICAL")
        self.assertEqual(out["bookkeeping_keys_added"], ["generated_at"])
        self.assertEqual(out["flags"], [])

    def test_a_deep_nested_change_is_detected(self):
        """按对象身份或顶层浅比较都会漏掉嵌套改动。"""
        base = _real()
        cand = copy.deepcopy(base)
        cand["tree"]["params"]["min_samples_leaf"] = 250
        out = check_increment(base, cand, spec=SPEC)
        self.assertEqual(out["changed_design_fields"], ["tree"])
        self.assertEqual(out["axes_changed"], ["combiner"])
        self.assertEqual(out["verdict"], "SINGLE_AXIS")

    def test_without_axes_every_field_is_its_own_axis(self):
        base = _real()
        cand = copy.deepcopy(base)
        cand["top_n"] = 100
        cand["selection"] = cand["selection"].replace("Top80", "Top100")
        out = check_increment(base, cand)
        self.assertEqual(out["verdict"], "MULTI_AXIS",
                         "无轴声明时必须退化成最严：两个字段 = 两个维度")

    def test_a_json_round_trip_of_the_same_protocol_is_identical(self):
        base = _real()
        cand = json.loads(json.dumps(base, ensure_ascii=False))
        out = check_increment(base, cand, spec=SPEC)
        self.assertEqual(out["verdict"], "IDENTICAL")
        self.assertEqual(out["changed_design_fields"], [])
        self.assertEqual(out["changed_bookkeeping_fields"], [])
        self.assertEqual(out["changed_reading_fields"], [])

    def test_readings_are_reported_and_never_dropped_from_the_output(self):
        base = _real()
        cand = copy.deepcopy(base)
        cand["features"][0]["coverage"] = {"median_rate": 0.5, "min_rate": 0.0,
                                           "sections": 137}
        out = check_increment(base, cand, spec=SPEC)
        self.assertEqual(out["verdict"], "IDENTICAL")
        self.assertEqual(out["feature_delta"]["readings_changed"],
                         [base["features"][0]["name"]])
        self.assertEqual(out["feature_delta"]["touched"], 0)


class MonotonicityTests(unittest.TestCase):
    def test_acceptance_never_comes_back_once_a_change_is_added(self):
        """🔴 单向下锁：往一个已拒绝的对照里再加改动，永远不会变成可接受。"""
        base = _real()
        accepted = copy.deepcopy(base)
        accepted["top_n"] = 100
        accepted["selection"] = accepted["selection"].replace("Top80", "Top100")
        self.assertTrue(check_increment(base, accepted, spec=SPEC)["accepted"])

        extra_edits = [
            ("cost_bps", [0, 15, 30, 45]),
            ("primary_cost_bps", 30),
            ("reference_arm", "equal_rank"),
            ("segments", "halves by index over the trained sections"),
            ("arms", ["equal_rank", "ridge"]),
        ]
        # 逐个加上去：一旦被拒，后续任何超集都必须保持被拒
        seen_refusal = False
        for field, value in extra_edits:
            probe = copy.deepcopy(accepted)
            probe[field] = value
            out = check_increment(base, probe, spec=SPEC)
            if out["refuses"]:
                seen_refusal = True
            if seen_refusal:
                self.assertTrue(out["refuses"],
                                "拒绝之后再加改动不能变成可接受: %s" % field)
        self.assertTrue(seen_refusal, "探针必须至少触发一次拒绝，否则这条测试是空转")

    def test_more_features_never_helps(self):
        base = _real()
        for count in (1, 2, 3):
            cand = copy.deepcopy(base)
            extra = [{"name": "n%d" % i, "kind": "snapshot", "direction": -1,
                      "source": None, "coverage": {}} for i in range(count)]
            cand["features"] = base["features"] + extra
            out = check_increment(base, cand, spec=SPEC)
            if count == 1:
                self.assertTrue(out["accepted"])
            else:
                self.assertTrue(out["refuses"])
            self.assertEqual(out["feature_delta"]["touched"], count)


class HashTests(unittest.TestCase):
    def test_protocol_sha256_ignores_key_order(self):
        base = _real()
        a = dict(reversed(list(base.items())))
        self.assertEqual(protocol_sha256(base), protocol_sha256(a))
        self.assertEqual(len(protocol_sha256(base)), 64)

    def test_protocol_sha256_changes_when_anything_changes(self):
        base = _real()
        cand = copy.deepcopy(base)
        cand["top_n"] = 81
        self.assertNotEqual(protocol_sha256(base), protocol_sha256(cand))


if __name__ == "__main__":
    unittest.main()
