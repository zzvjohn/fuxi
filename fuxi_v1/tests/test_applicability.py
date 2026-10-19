"""因子适用性备注测试：行业分组 rank IC + weak 行业标注（note-only，不参与门槛）。"""
import unittest

from fuxi_v1.system_fixture import synthetic_campaign
from fuxi_v1.data import Snapshot, Labels
from fuxi_v1.applicability import applicability_notes, conditioned_rank_ic, lookup_industry, cap_bucket


def _fixture(sessions=50, instruments=20):
    """强行业（j<10）正向、弱行业（j>=10）反向的受控标签。"""
    p = synthetic_campaign(sessions, instruments)
    snap = Snapshot(p["snapshot"])
    for row in p["labels"]["rows"]:
        j = int(row["instrument"].split(".")[0]) - 600000
        row["value"] = (j * 0.0005) if j < 10 else (-j * 0.0005)
    labels = Labels(p["labels"], snap)
    codes = snap.universe[snap.sessions[0]]
    scores = {d: {c: (int(c.split(".")[0]) - 600000) for c in codes} for d in snap.sessions}
    industry_map = {f"{600000 + j}.XSHG": ("强行业" if j < 10 else "弱行业") for j in range(instruments)}
    return snap, labels, scores, industry_map, p["research_clock"]


class ApplicabilityTests(unittest.TestCase):
    def test_industry_notes_flag_weak_industry(self):
        snap, labels, scores, industry_map, clock = _fixture()
        note = applicability_notes(scores, snap, labels, clock, snap.sessions[20], snap.sessions[-1],
                                   industry_map=industry_map, min_sections=5)
        self.assertEqual(note["purpose"], "note_only_not_gate")
        by_ind = note["industry"]["by_industry"]
        self.assertGreater(by_ind["强行业"]["mean_rank_ic"], 0.9)
        self.assertLess(by_ind["弱行业"]["mean_rank_ic"], -0.9)
        self.assertEqual(note["industry"]["weak_industries"], ["弱行业"])

    def test_regime_unavailable_without_labels(self):
        snap, labels, scores, industry_map, clock = _fixture(instruments=20)
        note = applicability_notes(scores, snap, labels, clock, snap.sessions[20], snap.sessions[-1],
                                   industry_map=industry_map)
        self.assertEqual(note["regime"]["status"], "regime_labels_unavailable")

    def test_regime_grouping_with_labels(self):
        snap, labels, scores, industry_map, clock = _fixture()
        # 前半段牛市、后半段熊市；标签在熊市整体下调，构造 regime 分化。
        sessions = snap.sessions
        split = len(sessions) // 2
        for row in labels.payload["rows"]:
            if row["session"] >= sessions[split]:
                row["value"] = -abs(row["value"])
        labels2 = Labels(labels.payload, snap)
        regime_by_session = {d: ("牛" if i < split else "熊") for i, d in enumerate(sessions)}
        note = applicability_notes(scores, snap, labels2, clock, snap.sessions[20], snap.sessions[-1],
                                   industry_map=industry_map, regime_by_session=regime_by_session)
        self.assertIn("牛", note["regime"]["by_regime"])
        self.assertIn("熊", note["regime"]["by_regime"])

    def test_lookup_industry_normalizes_codes(self):
        m = {"000019.SZ": "商贸零售"}
        self.assertEqual(lookup_industry("000019.SZ", m), "商贸零售")
        self.assertEqual(lookup_industry("SZ000019", m), "商贸零售")
        # 裸 6 位代码无交易所后缀，无法唯一解析（SZ/SH 有歧义）→ None
        self.assertIsNone(lookup_industry("000019", m))
        self.assertIsNone(lookup_industry("600000.SH", m))

    def test_conditioned_rank_ic_skips_unmapped(self):
        snap, labels, scores, industry_map, clock = _fixture()
        out = conditioned_rank_ic(scores, snap, labels, clock, snap.sessions[20], snap.sessions[-1],
                                  lambda day, code: industry_map.get(code))
        self.assertSetEqual(set(out), {"强行业", "弱行业"})

    def test_cap_bucket_thresholds(self):
        self.assertEqual(cap_bucket(500e8), "大盘")
        self.assertEqual(cap_bucket(200e8), "中盘")
        self.assertEqual(cap_bucket(50e8), "小盘")

    def test_market_cap_unavailable_without_cap(self):
        snap, labels, scores, industry_map, clock = _fixture()
        note = applicability_notes(scores, snap, labels, clock, snap.sessions[20], snap.sessions[-1],
                                   industry_map=industry_map)
        self.assertEqual(note["market_cap"]["status"], "cap_labels_unavailable")

    def test_market_cap_grouping(self):
        snap, labels, scores, industry_map, clock = _fixture(instruments=20)
        sessions = snap.sessions
        codes = snap.universe[sessions[0]]

        def cap_for(code):
            j = int(code.split(".")[0]) - 600000
            return 50e8 if j < 7 else 200e8 if j < 14 else 500e8

        cap_by_session = {d: {c: cap_for(c) for c in codes} for d in sessions}
        note = applicability_notes(scores, snap, labels, clock, snap.sessions[20], snap.sessions[-1],
                                   industry_map=industry_map, cap_by_session=cap_by_session)
        by_cap = note["market_cap"]["by_cap"]
        self.assertSetEqual(set(by_cap), {"小盘", "中盘", "大盘"})
        # 大盘(j>=14)标签为负 → weak_cap 应含"大盘"
        self.assertIn("大盘", note["market_cap"]["weak_cap"])


if __name__ == "__main__":
    unittest.main()
