"""S0 缺口冻结：扫描 1125 截面快照的账户路径缺口，落缺口清单 CSV。

只读扫描：不改快照、不改数据。产出：
  - data/fuxi_unified/account_path_gaps_20261018.csv  （逐 (session,instrument) 缺口清单）
  - stdout 统计摘要（快照 sha256 + 各类缺口计数）

缺口分类（对应 run_account 里累积 gaps 的触发字段）：
  paused_unmarked           停牌日 paused=None
  ohlcv_missing             停牌日 open/close/volume 缺失
  corporate_action_unknown  corporate_action_known != True
  up_limit_missing          涨跌停 up_limit=None
  down_limit_missing        跌停 down_limit=None
  sizing_ref_missing        sizing_reference_close=None（除权除息日参考价）
"""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

SNAPSHOT = Path("data/fuxi_unified/snapshots/native64_20210104_20251231/campaign_input.json")
OUT = Path("data/fuxi_unified/account_path_gaps_20261018.csv")
FIELDS = ["session", "instrument", "gap_types", "is_halt_day", "is_trading_day_ca_unknown"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    d = json.load(open(SNAPSHOT, encoding="utf-8"))
    market = d["market"]
    bars = market["bars"]
    snap_hash = sha256(SNAPSHOT)

    rows = []
    for b in bars:
        session, inst = b["session"], b["instrument"]
        paused = b.get("paused")
        ohlcv_missing = any(b.get(k) is None for k in ("open", "close", "volume"))
        ca_known = b.get("corporate_action_known")
        gaps = []
        if paused is None:
            gaps.append("paused_unmarked")
        if ohlcv_missing:
            gaps.append("ohlcv_missing")
        if ca_known is not True:
            gaps.append("corporate_action_unknown")
        if b.get("up_limit") is None:
            gaps.append("up_limit_missing")
        if b.get("down_limit") is None:
            gaps.append("down_limit_missing")
        if b.get("sizing_reference_close") is None:
            gaps.append("sizing_ref_missing")
        if not gaps:
            continue
        rows.append({
            "session": session,
            "instrument": inst,
            "gap_types": "|".join(gaps),
            "is_halt_day": bool(paused is None and ohlcv_missing),
            "is_trading_day_ca_unknown": bool(ca_known is not True and not ohlcv_missing),
        })

    with open(OUT, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    halt_days = sum(1 for r in rows if r["is_halt_day"])
    trading_ca_unknown = sum(1 for r in rows if r["is_trading_day_ca_unknown"])
    gap_type_counts = Counter()
    for r in rows:
        for g in r["gap_types"].split("|"):
            gap_type_counts[g] += 1

    print(json.dumps({
        "snapshot_sha256": snap_hash,
        "total_bars": len(bars),
        "gap_rows": len(rows),
        "halt_days": halt_days,
        "trading_day_ca_unknown": trading_ca_unknown,
        "gap_type_counts": dict(sorted(gap_type_counts.items())),
        "output": str(OUT),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
