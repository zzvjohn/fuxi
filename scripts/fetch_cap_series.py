"""按研究池代码批量拉历史市值序列（tushare daily_basic），存单文件缓存。

用途：为 applicability 的市值分段（note-only）提供 {trade_date: {code: cap}} 历史
流通市值。只拉研究快照里的代码（union of universe），不拉全市场，比逐日全市场
快两个数量级。

用法：
  python scripts/fetch_cap_series.py --input <input.json> [--field circ_mv] [--out data/cap_series]
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fuxi_v1.contracts import load_json  # noqa: E402
from fuxi_v1.cap_data import (  # noqa: E402
    fetch_cap_series_by_codes, cap_series_document, save_cap_series)

INPUT_DEFAULT = str(
    ROOT / "data/fuxi_unified/runs/unified_153d8e0779d74d49e2f3606e"
    "/research/campaign_153d8e0779d74d49e2f3606e/input.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=INPUT_DEFAULT)
    ap.add_argument("--field", default="circ_mv")
    ap.add_argument("--out", default=str(ROOT / "data/cap_series"))
    ap.add_argument("--sleep", type=float, default=0.15)
    args = ap.parse_args()

    payload = load_json(args.input)
    snap = payload["snapshot"]
    instruments = sorted({s for members in snap["universe"].values() for s in members})
    sessions = snap["sessions"]
    start, end = sessions[0], sessions[-1]
    start8, end8 = start.replace("-", ""), end.replace("-", "")

    print(f"研究池代码: {len(instruments)} 只；session 范围 {start} -> {end}")

    from credentials import get_tushare_api
    pro = get_tushare_api()
    cap_by_date = fetch_cap_series_by_codes(pro, instruments, start8, end8,
                                            cap_field=args.field, sleep_s=args.sleep)
    doc = cap_series_document(cap_by_date, args.field, instruments, start, end)
    path = save_cap_series(doc, args.out)

    n_pairs = sum(len(cs) for cs in cap_by_date.values())
    print(f"已缓存: {path}  days={doc['n_days']}  code-day对={n_pairs}")
    if cap_by_date:
        first_day = min(cap_by_date)
        sample = list(cap_by_date[first_day].items())[:3]
        print(f"样例 {first_day}: {[(c, round(v/1e8, 1)) for c, v in sample]} 亿元")


if __name__ == "__main__":
    main()
