"""单调性诊断测试：分位组合收益 + PT MR test 相邻差（note-only，不裁决 pass/reject）。"""
import unittest

from fuxi_v1.system_fixture import synthetic_campaign
from fuxi_v1.data import Snapshot, Labels
from fuxi_v1.monotonicity import monotonicity_diagnostics, monotonicity_note, MONOTONICITY_DEFAULTS


def _fixture(sessions=50, instruments=20, group_values=(0.0, 0.002, 0.004, 0.006, 0.008)):
    """构造 legacy 快照 + 受控标签 + 单调递增 score（score = 股票序号 j）。

    label value 按 instrument 序号 j 的 5 等分组设为 group_values[g]，使分组收益
    恰为 group_values，便于精确断言（组内常量、组间可控）。
    """
    p = synthetic_campaign(sessions, instruments)
    snap = Snapshot(p["snapshot"])
    group_size = instruments // 5
    for row in p["labels"]["rows"]:
        j = int(row["instrument"].split(".")[0]) - 600000
        row["value"] = group_values[min(j // group_size, 4)]
    labels = Labels(p["labels"], snap)
    codes = snap.universe[snap.sessions[0]]
    scores = {d: {c: (int(c.split(".")[0]) - 600000) for c in codes} for d in snap.sessions}
    return snap, labels, scores, p["research_clock"]


def _note(snap, labels, scores, clock, group_values, **kw):
    start, end = snap.sessions[20], snap.sessions[-1]
    diag = monotonicity_diagnostics(scores, snap, labels, clock, start, end,
                                    block_length=6, alpha=0.05, replicates=199, **kw)
    return monotonicity_note(diag)


class MonotonicityTests(unittest.TestCase):
    def test_strict_monotonic_is_diagnostic(self):
        # 严格单调：spread 显著 → status=diagnostic（note-only，不再 pass/reject）。
        snap, labels, scores, clock = _fixture(group_values=(0.0, 0.002, 0.004, 0.006, 0.008))
        note = _note(snap, labels, scores, clock, (0.0, 0.002, 0.004, 0.006, 0.008))
        self.assertEqual(note["status"], "diagnostic")
        self.assertTrue(note["spread_significant"])
        self.assertEqual(note["purpose"], "note_only_not_gate")
        # MR test 相邻差全正
        mr = note["mr_adjacent_deltas"]
        self.assertIsNotNone(mr)
        self.assertTrue(mr["all_adjacent_deltas_positive"])
        self.assertAlmostEqual(note["diagnostics"]["mean_monotonic_spearman"], 1.0, 4)

    def test_monotonic_with_middle_dip_is_diagnostic(self):
        # “整体单调但中间有波动”：g2 略低于 g1（4 组中 1 组倒置）——不再有 spearman
        # 门槛，只要 spread 显著即 diagnostic，中间波动被显式报告。
        snap, labels, scores, clock = _fixture(group_values=(0.0, 0.004, 0.003, 0.006, 0.008))
        note = _note(snap, labels, scores, clock, (0.0, 0.004, 0.003, 0.006, 0.008))
        self.assertEqual(note["status"], "diagnostic")
        self.assertTrue(note["spread_significant"])
        # MR 相邻差：Δ2 = g3-g2 = -0.001 为负，all_positive 应为 False（如实报告）
        self.assertFalse(note["mr_adjacent_deltas"]["all_adjacent_deltas_positive"])
        self.assertLess(note["diagnostics"]["mean_adjacent_monotonic_fraction"], 1.0)

    def test_u_shaped_data_insufficient_not_reject(self):
        # 两极分化（两端高、中间低）：spread≈0 → data_insufficient（fail-open），
        # 不再 reject——形状信息不足不等于因子无效。
        snap, labels, scores, clock = _fixture(group_values=(0.008, 0.002, 0.0, 0.002, 0.008))
        note = _note(snap, labels, scores, clock, (0.008, 0.002, 0.0, 0.002, 0.008))
        self.assertEqual(note["status"], "data_insufficient")
        self.assertFalse(note["spread_significant"])
        self.assertEqual(note["reason_code"], "monotonicity_spread_not_significant")

    def test_insufficient_cross_section_fails_open(self):
        # 截面太小（6 < min_cross_section=15）→ data_insufficient，不影响 E1 已有裁决。
        snap, labels, scores, clock = _fixture(instruments=6)
        note = _note(snap, labels, scores, clock, (0.0, 0.002, 0.004, 0.006, 0.008))
        self.assertEqual(note["status"], "data_insufficient")

    def test_defaults_consistency(self):
        d = MONOTONICITY_DEFAULTS
        self.assertGreaterEqual(d["min_cross_section"], d["n_groups"])
        self.assertGreaterEqual(d["n_groups"], 2)
        # 已移除 spearman 绝对门槛（降级 note-only）
        self.assertNotIn("min_monotonic_spearman", d)


if __name__ == "__main__":
    unittest.main()
