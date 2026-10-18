"""Locked correction of v0.7.76's shadow-only experimental proposal metadata."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import json_store  # noqa: E402
import ledger_io  # noqa: E402

PID = "P-20260925-015"


def update_ledger(data):
    rows = data if isinstance(data, list) else data["proposals"]
    found = [row for row in rows if row.get("proposal_id") == PID]
    if len(found) != 1:
        raise ValueError("proposal identity changed")
    found[0]["proposal_change_kind"] = "shadow"
    return data


def update_registry(data):
    rows = data if isinstance(data, list) else data["versions"]
    found = [row for row in rows if row.get("version") == "v0.7.76"]
    if len(found) != 1:
        raise ValueError("version identity changed")
    version = found[0]
    proposals = [row for row in version["proposals"] if row.get("proposal_id") == PID]
    if len(proposals) != 1:
        raise ValueError("version proposal identity changed")
    proposals[0]["zero_adjudication"] = True
    version["behavior"] = "生成链确定性修复；语义计划默认 off；经济对照未完成"
    version["rollback"] = "代码字节备份见 reports/fuxi_generator_rebuild_20260925/baseline/；语义计划 off"
    return data


if __name__ == "__main__":
    ledger_io.load_and_modify(update_ledger)
    json_store.load_and_modify(str(ROOT / "data/version_registry.json"), update_registry)
    print(PID, "shadow, v0.7.76 zero_adjudication=true")
