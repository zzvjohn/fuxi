"""Register the disclosure-only part of P-20260925-013 under the ledger lock."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from ledger_io import load_and_modify, load  # noqa: E402

PID = "P-20260925-013"
DETAIL = (
    "仅①③部分落地：S5 缺价格数/日期及权重契约经 S5Result→ValidationResult→手动候选摘要落盘；"
    "已算经济值标 diagnostic_only_data_insufficient，passed/status 不变。"
    "②缺价容差及④S6占位/影子通过语义未实现；没有正式经济晋级。"
)


def change(data):
    records = data if isinstance(data, list) else data["proposals"]
    found = [p for p in records if p.get("proposal_id") == PID]
    if len(found) != 1:
        raise ValueError("proposal identity not unique")
    row = found[0]
    if row.get("status") not in {"proposed", "partially_implemented"}:
        raise ValueError("proposal state changed")
    row["status"] = "partially_implemented"
    row["landed_detail"] = DETAIL
    row["implemented_at"] = "2026-09-25"
    row["remaining_scope"] = "② 缺价容差须预注册及四关 A/B；④ S6 读数/通过语义另修。"
    row["implementation_evidence"] = [
        "reports/fuxi_drought_execution_20260925.md",
        "research/factor_alchemy/tests/test_holdings_mark.py",
    ]
    return data


if __name__ == "__main__":
    load_and_modify(change)
    rows = load()
    rows = rows if isinstance(rows, list) else rows["proposals"]
    found = [p for p in rows if p.get("proposal_id") == PID]
    assert len(found) == 1 and found[0]["status"] == "partially_implemented"
    print(PID, found[0]["status"], found[0]["landed_detail"])
