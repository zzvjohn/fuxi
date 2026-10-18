"""Rebuild a 1212-session native64 snapshot from local raw CSV channels.

Usage:
    python scripts/rebuild_snapshot_1212.py [--start 2021-01-04] [--end 2025-12-31]
        [--max-instruments 64] [--horizon 5] [--out <dir>] [--dry-run]

Produces data/fuxi_unified/snapshots/<tag>/campaign_input.json (fuxi.campaign_input.1
with a fuxi.snapshot.2 native64 snapshot), which satisfies unified_runner.normalize_request.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fuxi_v1.datasets import local_market_input  # noqa: E402


def file_sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", default="2021-01-04")
    ap.add_argument("--end", default="2025-12-31")
    ap.add_argument("--max-instruments", type=int, default=64)
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    raw_root = ROOT / args.raw
    t0 = time.time()
    payload = local_market_input(
        raw_root, args.start, args.end,
        max_instruments=args.max_instruments, horizon=args.horizon,
    )
    snap = payload["snapshot"]
    sessions = snap["sessions"]
    instruments = snap["universe"][sessions[0]]
    # P-20261017 防再犯：抽样必须沪深均衡，否则 E1 screen_positive_exchange_count=2 全 reject。
    n_sz = sum(1 for i in instruments if i.endswith(".XSHE"))
    n_sh = sum(1 for i in instruments if i.endswith(".XSHG"))
    assert n_sz > 0 and n_sh > 0, f"universe must span both exchanges, got XSHE={n_sz} XSHG={n_sh}"
    n_obs = len(snap["observations"])
    print(f"sessions        : {len(sessions)}  ({sessions[0]} .. {sessions[-1]})")
    print(f"universe        : {len(instruments)} instruments (XSHE={n_sz}, XSHG={n_sh})")
    print(f"observations    : {n_obs}")
    print(f"labels rows     : {len(payload['labels'].get('rows', []))}")
    print(f"schema          : {payload['schema_version']} / {snap['schema_version']} / exposure={snap['exposure']}")
    print(f"elapsed         : {time.time()-t0:.1f}s")

    # campaign-side window split preview (mirrors campaign.py cal/research index)
    horizon = payload["labels"]["label_spec"]["horizon_sessions"]
    cal_index = max(20 + 25, min(80, len(sessions) // 2))
    research_index = cal_index + horizon + 2
    research_window = len(sessions) - research_index
    print(f"cal_index       : {cal_index}  research_index={research_index}")
    print(f"research window : {research_window} sections (was 93)")
    import math
    min_icir = 3.0 / math.sqrt(research_window)
    print(f"min ICIR for t>=3.0: {min_icir:.4f} (was 0.311 at 93 sections)")

    if args.dry_run:
        print("[dry-run] not writing output")
        return

    out_dir = Path(args.out) if args.out else ROOT / "data" / "fuxi_unified" / "snapshots" / \
        f"native64_{args.start.replace('-', '')}_{args.end.replace('-', '')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "campaign_input.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    print(f"wrote           : {out_path}")
    print(f"size            : {out_path.stat().st_size/1e6:.1f} MB")
    print(f"sha256          : {file_sha(out_path)}")


if __name__ == "__main__":
    main()
