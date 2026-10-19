"""regime 模块测试：牛熊划分 + 缓存往返 + session 映射。"""
import tempfile
import unittest
from pathlib import Path

from fuxi_v1.regime import (
    classify_regime, csi300_regime, load_csi300_closes, save_csi300_closes,
    default_csi300_regime)


def _closes():
    # 严格单调上升（day i close = 100 + i），便于断言牛熊方向。
    days = [f"2020-01-{d:02d}" for d in range(1, 31)]
    closes = {days[i]: 100.0 + i for i in range(len(days))}
    return closes


class _FakePro:
    def __init__(self, closes):
        self._closes = closes

    def index_daily(self, ts_code, start_date, end_date):
        return [{"trade_date": d, "close": v} for d, v in self._closes.items()]


class RegimeTests(unittest.TestCase):
    def test_classify_warmup_boundary(self):
        closes = _closes()
        regime = classify_regime(closes, ma_window=5)
        # 前 4 天不满窗，无标签；第 5 天起才有。
        self.assertNotIn("2020-01-04", regime)
        self.assertIn("2020-01-05", regime)
        self.assertEqual(len(regime), len(closes) - 4)

    def test_classify_bull_bear_direction(self):
        # 单调上升序列：一旦 close > MA5 恒成立 → 全 bull。
        closes = _closes()
        regime = classify_regime(closes, ma_window=5)
        self.assertTrue(all(v == "bull" for v in regime.values()))
        # 单调下降序列 → 全 bear。
        down = {d: 100.0 - (0.0 if i < 5 else i - 5) for i, d in enumerate(closes)}
        regime_down = classify_regime(down, ma_window=5)
        self.assertTrue(all(v == "bear" for v in regime_down.values()))

    def test_cache_roundtrip_and_hash(self):
        closes = _closes()
        with tempfile.TemporaryDirectory() as td:
            save_csi300_closes(closes, "20200101", "20200130", td)
            doc = load_csi300_closes(td)
            self.assertEqual(doc["closes"], closes)
            self.assertEqual(doc["n_days"], len(closes))
            # 缓存缺失返回 None（不触网）
            self.assertIsNone(load_csi300_closes(str(Path(td) / "missing")))

    def test_csi300_regime_maps_sessions(self):
        closes = _closes()
        with tempfile.TemporaryDirectory() as td:
            sessions = ["2020-01-08", "2020-01-10", "2020-01-03"]  # 末者在 warmup 内
            out = csi300_regime(sessions, pro=_FakePro(closes), ma_window=5, cache_dir=td)
            self.assertIn("2020-01-08", out)
            self.assertIn("2020-01-10", out)
            self.assertNotIn("2020-01-03", out)  # 不满窗，跳过
            self.assertEqual(out["2020-01-08"], "bull")
            # 缓存已写，二次调用无需 pro
            out2 = csi300_regime(sessions, pro=None, ma_window=5, cache_dir=td)
            self.assertEqual(out, out2)

    def test_default_regime_no_cache_returns_none(self):
        with tempfile.TemporaryDirectory() as td:
            self.assertIsNone(default_csi300_regime(ma_window=5, cache_dir=str(Path(td) / "nope")))


if __name__ == "__main__":
    unittest.main()
