"""诊断：1125 截面账户路径认证现状 + S1 停牌标记预演。

只读 + 内存预演，不改任何文件。产出 stdout：
  1. 停牌日精确归因（退市前 vs 退市后）
  2. certify() 在 lifecycle 介入后的逐 flag 阻塞缺口
  3. S1 标记（paused=True + mark_source）后 certify_tradability 是否翻转
"""
import json
import sys
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[1]
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from fuxi_v1.pit_universe import read_lifecycle, canonical_code, jq_code
from fuxi_v1.certification import certify, certify_tradability, listed_at, account_scope, missing_account_bars
from fuxi_v1.contracts import digest

SNAPSHOT = WORKSPACE / "data/fuxi_unified/snapshots/native64_20210104_20251231/campaign_input.json"
RAW = WORKSPACE / "data/raw"

SUSPEND_SHA = "f18b03648b9185a16f7a57982d3c20aec594f3aed60e95d9d541256ff5d5bdd1"


def iso_day(s):
    if not s:
        return None
    return str(s)[:10].replace("-", "")


def main():
    d = json.load(open(SNAPSHOT, encoding="utf-8"))
    snap = d["snapshot"]
    market = d["market"]
    bars = market["bars"]
    sessions = snap["sessions"]

    life = read_lifecycle(RAW, "delist_dates.csv")

    # 1. 停牌日归因
    halt = [b for b in bars if b.get("paused") is None and b.get("close") is None]
    pre, post, nocov = 0, 0, 0
    for b in halt:
        st = listed_at(life, b["instrument"], b["session"])
        if st is True:
            pre += 1
        elif st is False:
            post += 1
        else:
            nocov += 1
    print("=== 停牌日归因 ===")
    print("halt_days(paused=None & close=None):", len(halt))
    print("  退市前停牌(listed_at=True):", pre)
    print("  退市后无价格(listed_at=False):", post)
    print("  未覆盖(listed_at=None):", nocov)

    # 2. suspend S 记录 map
    import csv
    suspend_s = {}
    with open(RAW / "suspend_d.csv", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if row.get("suspend_type") != "S":
                continue
            code = canonical_code(row.get("ts_code", ""))
            day = iso_day(row.get("trade_date"))
            if code and day:
                suspend_s[(jq_code(code), day)] = True
    # 退市前停牌日中，有多少在 suspend_d 有 S 记录
    pre_with_s = 0
    for b in halt:
        if listed_at(life, b["instrument"], b["session"]) is True:
            key = (b["instrument"], b["session"].replace("-", ""))
            if key in suspend_s:
                pre_with_s += 1
    print("退市前停牌日中 suspend_d 有 S 记录:", pre_with_s, "/", pre)

    # 3. certify() 现状（lifecycle 介入）
    # 行式快照 fuxi.snapshot.2 无 instruments 键，从 universe 派生（64 只冻结 cohort）
    instruments = sorted({c for v in snap["universe"].values() for c in v})
    snap2 = dict(snap, instruments=instruments)
    d2 = dict(d, snapshot=snap2)
    declaration = {}
    cert = certify(d2, declaration=declaration, lifecycle=life)
    print("\n=== certify() 现状（lifecycle 介入）===")
    print("blocking_gaps:", cert["blocking_gaps"])
    for name in ("pit_certified", "universe_certified", "market_representativeness_certified",
                 "account_events_certified", "event_coverage_certified",
                 "tradability_certified", "execution_rules_certified"):
        fl = cert["flags"][name]
        print(f"  {name}: value={fl['value']} blocking={fl['blocking_gaps']}")

    # 4. S1 预演：把退市前停牌日标 paused=True + mark_source
    import copy
    mkt2 = copy.deepcopy(market)
    marked = 0
    for b in mkt2["bars"]:
        if b.get("paused") is None and b.get("close") is None:
            if listed_at(life, b["instrument"], b["session"]) is True:
                b["paused"] = True
                b["mark_source"] = "declared_frozen_last_close_on_vendor_halt"
                b["suspend_source_sha256"] = SUSPEND_SHA
                marked += 1
    print("\n=== S1 预演（标 paused=%d 只退市前停牌）===" % marked)
    trad = certify_tradability(snap2, mkt2, life)
    print("certify_tradability: value=%s blocking=%s" % (trad["value"], trad["blocking_gaps"]))
    print("  basis:", {k: v for k, v in trad["basis"].items()})

    # 5. 顺便看 account_scope / missing_account_bars 在 lifecycle 介入后的规模
    scope = account_scope(snap2, life)
    miss = missing_account_bars(snap2, market, life)
    print("\n=== account_scope(lifecycle) / missing_account_bars ===")
    print("missing_account_bars count:", len(miss))
    tot = sum(len(v) for v in scope.values())
    print("account_scope total member cells:", tot)


if __name__ == "__main__":
    main()
