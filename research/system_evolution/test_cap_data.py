"""Tests for cap_data (unit conversion + content-addressed snapshot)."""

import unittest

import pandas as pd

from fuxi_v1.cap_data import (
    fetch_cap_snapshot, snapshot_document, to_tushare_code, from_tushare_code,
    fetch_cap_series_by_codes, cap_series_document, load_cap_series)


class FakePro:
    def daily_basic(self, trade_date, fields):
        return pd.DataFrame([
            {"ts_code": "000001.SZ", "trade_date": trade_date, "close": 11.5,
             "total_mv": 2.231681e7, "circ_mv": 2.231654e7, "free_share": 816056.5553},
            {"ts_code": "000002.SZ", "trade_date": trade_date, "close": 3.66,
             "total_mv": 4.366640e6, "circ_mv": 3.555731e6, "free_share": 647230.2363},
        ])


class _SeriesPro:
    """按 ts_code 返回逐日序列的 fake（fetch_cap_series_by_codes 用）。"""

    def __init__(self, series):
        self._series = series
        self.calls = []

    def daily_basic(self, ts_code, start_date, end_date, fields):
        self.calls.append(ts_code)
        rows = self._series.get(ts_code, [])
        return pd.DataFrame(rows)


class TestCapData(unittest.TestCase):
    def setUp(self):
        self.pro = FakePro()

    def test_circ_mv_wan_to_yuan(self):
        caps = fetch_cap_snapshot(self.pro, "20261016", "circ_mv")
        self.assertAlmostEqual(caps["000001.SZ"], 2.231654e7 * 1e4, places=-3)
        self.assertAlmostEqual(caps["000002.SZ"], 3.555731e6 * 1e4, places=-3)

    def test_total_mv_wan_to_yuan(self):
        caps = fetch_cap_snapshot(self.pro, "20261016", "total_mv")
        self.assertAlmostEqual(caps["000001.SZ"], 2.231681e7 * 1e4, places=-3)

    def test_free_float_computed(self):
        caps = fetch_cap_snapshot(self.pro, "20261016", "free_float")
        self.assertAlmostEqual(caps["000001.SZ"], 816056.5553 * 1e4 * 11.5, places=-3)
        self.assertAlmostEqual(caps["000002.SZ"], 647230.2363 * 1e4 * 3.66, places=-3)

    def test_snapshot_document_content_addressed(self):
        caps = {"000001.SZ": 1.0, "000002.SZ": 2.0}
        doc = snapshot_document("20261016", "circ_mv", caps)
        self.assertEqual(doc["cap_unit"], "cny_yuan")
        self.assertEqual(doc["n_codes"], 2)
        self.assertIn("data_hash", doc)
        # same content -> same hash (order-insensitive)
        doc2 = snapshot_document("20261016", "circ_mv", {"000002.SZ": 2.0, "000001.SZ": 1.0})
        self.assertEqual(doc["data_hash"], doc2["data_hash"])

    def test_code_conversion_roundtrip(self):
        self.assertEqual(to_tushare_code("000001.XSHE"), "000001.SZ")
        self.assertEqual(to_tushare_code("600000.XSHG"), "600000.SH")
        self.assertEqual(from_tushare_code("000001.SZ"), "000001.XSHE")
        self.assertEqual(from_tushare_code("600000.SH"), "600000.XSHG")
        self.assertEqual(from_tushare_code(to_tushare_code("000001.XSHE")), "000001.XSHE")

    def test_fetch_cap_series_queries_tushare_format_and_keys_snapshot(self):
        series = {
            "000001.SZ": [
                {"ts_code": "000001.SZ", "trade_date": "20210104", "close": 11.5,
                 "total_mv": 2.0e7, "circ_mv": 2.0e7, "free_share": 1.0},
                {"ts_code": "000001.SZ", "trade_date": "20210105", "close": 11.6,
                 "total_mv": 2.1e7, "circ_mv": 2.1e7, "free_share": 1.0},
            ],
        }
        pro = _SeriesPro(series)
        out = fetch_cap_series_by_codes(pro, ["000001.XSHE"], "20210104", "20210105",
                                        cap_field="circ_mv", sleep_s=0)
        self.assertEqual(pro.calls, ["000001.SZ"])  # 查询用了 tushare 格式
        self.assertIn("2021-01-04", out)           # 日期归一化
        self.assertIn("000001.XSHE", out["2021-01-04"])  # 键用快照格式
        self.assertAlmostEqual(out["2021-01-04"]["000001.XSHE"], 2.0e7 * 1e4, places=-3)

    def test_cap_series_document_roundtrip(self):
        import tempfile
        cap_by_date = {"2021-01-04": {"000001.XSHE": 2.0e11}, "2021-01-05": {"000001.XSHE": 2.1e11}}
        doc = cap_series_document(cap_by_date, "circ_mv", ["000001.XSHE"], "2021-01-04", "2021-01-05")
        with tempfile.TemporaryDirectory() as td:
            from fuxi_v1.cap_data import save_cap_series
            save_cap_series(doc, td)
            loaded = load_cap_series(td, "circ_mv")
            self.assertEqual(loaded, cap_by_date)


if __name__ == "__main__":
    unittest.main()
