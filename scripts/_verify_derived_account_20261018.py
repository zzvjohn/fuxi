"""验证派生快照：run_account 是否 eval_status=valid + formal_evidence=True。

只读验证，不改派生快照。两种 IR：
  1. 空决策（纯现金）——验证认证绑定本身。
  2. 真实持仓——第 2 会话买入若干标的（含一只 2021 年有除权除息事件的），
     持有跨过除权日，验证不累积 gaps（identifiable 仍 True）。
"""
import json
import sys
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[1]
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from fuxi_v1.contracts import digest
from fuxi_v1.simulation import run_account
from fuxi_v1.portfolio import strategy_spec

DERIVED = WORKSPACE / "data/fuxi_unified/snapshots/native64_20210104_20251231_ledger_v1/campaign_input.json"


def make_ir(execution, snapshot_hash, decisions):
    ir = {
        "execution_hash": digest(execution),
        "snapshot_hash": snapshot_hash,
        "decisions": decisions,
        "strategy": strategy_spec("verify_test"),
    }
    ir["ir_hash"] = digest({k: v for k, v in ir.items() if k != "ir_hash"})
    return ir


def main():
    d = json.load(open(DERIVED, encoding="utf-8"))
    snap = d["snapshot"]
    market = d["market"]

    # 1. 空决策
    ir0 = make_ir(market["execution"], market["certification_snapshot_hash"], [])
    r0 = run_account(ir0, market)
    print("=== 空决策 IR ===")
    print("eval_status:", r0["eval_status"])
    print("formal_evidence:", r0["formal_evidence"])
    print("gaps:", len(r0["gaps"]))
    print("n daily rows:", len(r0["daily"]))

    # 2. 真实持仓：找一只在 2021 有现金分红/送转的标的 + 一只无事件标的
    terms = market["action_terms"]
    ex_instruments = {t["instrument"] for t in terms}
    print("\n=== 真实持仓 IR ===")
    print("有除权除息事件的标的数:", len(ex_instruments))

    # 选 2 只标的：一只 2021 有事件，一只尽量无事件（用第一个 universe 成员）
    session0 = snap["sessions"][0]
    session1 = snap["sessions"][1]
    universe0 = snap["universe"][session0]
    event_code = next((c for c in universe0 if c in ex_instruments), None)
    noevent_code = next((c for c in universe0 if c not in ex_instruments), None)
    held = [c for c in (event_code, noevent_code) if c]
    print("持有标的:", held)

    # 用 signal_session=session0（信号在第 0 会话），execution_session=session1（次会话下单）
    decisions = [{
        "signal_session": session0,
        "execution_session": session1,
        "decision_time": session0 + "T18:30:00+08:00",
        "first_order_time": session1 + "T09:30:00+08:00",
        "scores": {c: 1.0 for c in held},
    }]
    ir1 = make_ir(market["execution"], market["certification_snapshot_hash"], decisions)
    r1 = run_account(ir1, market)
    print("eval_status:", r1["eval_status"])
    print("formal_evidence:", r1["formal_evidence"])
    print("gaps:", len(r1["gaps"]))
    for g in r1["gaps"][:10]:
        print("  gap:", g)
    print("orders:", len(r1["orders"]), "filled:", sum(1 for o in r1["orders"] if o["filled"]))
    # 最后一日 NAV / 估值可识别
    last = r1["daily"][-1]
    print("last session:", last["session"], "valuation_identifiable:", last.get("valuation_identifiable"),
          "nav:", last.get("nav"))


if __name__ == "__main__":
    main()
