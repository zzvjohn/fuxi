"""结构化知识卡存储（fuxi.knowledge_card.1）。

链路：原始材料 → 结构化知识卡 → 待验证假设 → 本地实验

知识卡与 stage1_factor_proposals 的分工：
  - 知识卡保存**完整元信息**：来源身份/许可、机制与公式、适用范围、
    证据与局限、本地状态，以及最关键的 ``knowledge_available_at``
    （知识何时可得，PIT 元知识）。
  - proposal 只是卡片派生出的**可消费候选**（三字段假设 + 公式），
    由 ``promote`` 写入现有 stage1_factor_proposals.json，campaign 链路零改动。

用法：
  python scripts/knowledge_card.py init
  python scripts/knowledge_card.py add --file card.json          # 单卡或 {"cards":[...]}
  python scripts/knowledge_card.py list [--state unverified] [--market CN]
  python scripts/knowledge_card.py promote --id kc_xxxxxxxx [--dry-run]
  python scripts/knowledge_card.py verify

纪律：
  - 卡片库 append-only，按 card_id 去重，不覆盖历史。
  - ``knowledge_available_at`` 必填：用 2026 年论文指导 2020-2025 回测，
    只能称「今天做的历史研究」，不能称「当年可得知识下的前瞻结果」。
  - ``promote`` 只写未被占用（factor_name / formula_pandas 不重复）的候选。
"""
import argparse
import hashlib
import json
import sys
from datetime import datetime, date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STORE = ROOT / "data" / "knowledge_cards.json"
PROPOSALS = ROOT / "data" / "stage1_factor_proposals.json"
SCHEMA = "fuxi.knowledge_card.1"

STATES = ("unverified", "data_insufficient", "replicated", "falsified")
REPLICATION = ("not_replicated", "replicated", "failed", "unknown")
# 与 stage1_factor_proposals 现有契约对齐：direction 用 1 / -1。
DIRECTIONS = (1, -1)
DIRECTION_ALIASES = {1: 1, -1: -1, "1": 1, "-1": -1,
                     "positive": 1, "negative": -1}


def _norm_direction(value):
    if isinstance(value, bool):
        return None
    try:
        return DIRECTION_ALIASES.get(value)
    except TypeError:
        return None


def _read(path, default):
    if not path.is_file():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path, payload):
    """CRLF 通道写出，避免下游按行处理时被转成 LF。"""
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    path.write_text(text.replace("\n", "\r\n"), encoding="utf-8", newline="")


def card_id(card):
    key = (str((card.get("source") or {}).get("url", ""))
           + "|" + str((card.get("mechanism") or {}).get("formula_pandas", ""))
           + "|" + str((card.get("mechanism") or {}).get("formula_latex", "")))
    return "kc_" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def _is_date(value):
    try:
        date.fromisoformat(str(value))
        return True
    except (TypeError, ValueError):
        return False


def validate(card):
    """返回问题清单；空列表表示通过。"""
    problems = []
    source = card.get("source") or {}
    mechanism = card.get("mechanism") or {}
    scope = card.get("scope") or {}
    evidence = card.get("evidence") or {}
    status = card.get("local_status") or {}

    for field in ("title", "url", "retrieved_at", "license"):
        if not str(source.get(field) or "").strip():
            problems.append(f"source.{field} 必填")
    if source.get("retrieved_at") and not _is_date(source["retrieved_at"]):
        problems.append("source.retrieved_at 必须是 YYYY-MM-DD")

    if not str(mechanism.get("economic_rationale") or "").strip():
        problems.append("mechanism.economic_rationale 必填")
    if not (mechanism.get("formula_pandas") or mechanism.get("formula_latex")):
        problems.append("mechanism.formula_pandas / formula_latex 至少填一个")
    if _norm_direction(mechanism.get("direction")) is None:
        problems.append("mechanism.direction 必须是 1 / -1（或 positive / negative）")

    for field in ("market", "frequency"):
        if not str(scope.get(field) or "").strip():
            problems.append(f"scope.{field} 必填")

    if evidence.get("replication_status") not in REPLICATION:
        problems.append(f"evidence.replication_status 必须是 {REPLICATION}")
    if status.get("state") not in STATES:
        problems.append(f"local_status.state 必须是 {STATES}")

    available = card.get("knowledge_available_at")
    if not _is_date(available):
        problems.append("knowledge_available_at 必填且为 YYYY-MM-DD（PIT 元知识）")
    return problems


def load_store():
    payload = _read(STORE, {"schema_version": SCHEMA, "cards": []})
    payload.setdefault("schema_version", SCHEMA)
    payload.setdefault("cards", [])
    return payload


def cmd_init(_args):
    if STORE.is_file():
        print(json.dumps({"status": "exists", "path": str(STORE),
                          "cards": len(load_store()["cards"])}, ensure_ascii=False))
        return 0
    _write(STORE, {"schema_version": SCHEMA, "cards": []})
    print(json.dumps({"status": "created", "path": str(STORE)}, ensure_ascii=False))
    return 0


def cmd_add(args):
    incoming = _read(Path(args.file), None)
    if incoming is None:
        print(json.dumps({"status": "error", "reason": "file not found"}, ensure_ascii=False))
        return 1
    if isinstance(incoming, dict):
        cards = incoming.get("cards")
        if cards is None and ("source" in incoming or "mechanism" in incoming):
            cards = [incoming]
    elif isinstance(incoming, list):
        cards = incoming
    else:
        cards = None
    if not isinstance(cards, list):
        print(json.dumps({"status": "error", "reason": "expect list or {cards:[...]}"},
                         ensure_ascii=False))
        return 1

    store = load_store()
    existing = {c.get("card_id") for c in store["cards"]}
    added, skipped, rejected = [], [], []
    for card in cards:
        card = dict(card)
        problems = validate(card)
        if problems:
            rejected.append({"title": (card.get("source") or {}).get("title", ""),
                             "problems": problems})
            continue
        cid = card_id(card)
        card["card_id"] = cid
        card.setdefault("derived_proposals", [])
        if cid in existing:
            skipped.append(cid)
            continue
        store["cards"].append(card)
        existing.add(cid)
        added.append(cid)

    if added:
        store["updated_at"] = datetime.now().isoformat(timespec="seconds")
        _write(STORE, store)
    print(json.dumps({"status": "ok", "added": added, "skipped_duplicate": skipped,
                      "rejected": rejected, "total": len(store["cards"])},
                     ensure_ascii=False, indent=2))
    return 0 if not rejected else 2


def cmd_list(args):
    store = load_store()
    rows = store["cards"]
    if args.state:
        rows = [c for c in rows if (c.get("local_status") or {}).get("state") == args.state]
    if args.market:
        rows = [c for c in rows if (c.get("scope") or {}).get("market") == args.market]
    if args.source:
        needle = args.source.lower()
        rows = [c for c in rows
                if needle in str((c.get("source") or {}).get("title", "")).lower()]
    summary = [{"card_id": c.get("card_id"),
                "title": (c.get("source") or {}).get("title", "")[:60],
                "market": (c.get("scope") or {}).get("market"),
                "state": (c.get("local_status") or {}).get("state"),
                "available_at": c.get("knowledge_available_at"),
                "derived": len(c.get("derived_proposals") or [])} for c in rows]
    print(json.dumps({"count": len(summary), "cards": summary},
                     ensure_ascii=False, indent=2))
    return 0


def cmd_promote(args):
    store = load_store()
    card = next((c for c in store["cards"] if c.get("card_id") == args.id), None)
    if card is None:
        print(json.dumps({"status": "error", "reason": "card not found"}, ensure_ascii=False))
        return 1
    mechanism = card.get("mechanism") or {}
    formula = mechanism.get("formula_pandas")
    if not formula:
        print(json.dumps({"status": "error",
                          "reason": "promote 需要 mechanism.formula_pandas（campaign 消费格式）"},
                         ensure_ascii=False))
        return 1

    proposals = _read(PROPOSALS, {"proposals": []})
    rows = proposals.setdefault("proposals", [])
    name = args.name or ("kc_" + card["card_id"].split("_", 1)[1])
    taken = {str(p.get("factor_name")) for p in rows} | \
            {str(p.get("formula_pandas")) for p in rows}
    if name in taken or formula in taken:
        print(json.dumps({"status": "skipped", "reason": "proposal 已存在", "factor_name": name},
                         ensure_ascii=False))
        return 0

    proposal = {
        "factor_name": name,
        "label": (card.get("source") or {}).get("title", "")[:80],
        "formula_pandas": formula,
        "economic_rationale": mechanism.get("economic_rationale", ""),
        "paradigm": args.paradigm or "external_knowledge",
        "direction": _norm_direction(mechanism.get("direction")),
        "source_proposal": "KNOWLEDGE-CARD",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "knowledge_card_id": card["card_id"],
        "knowledge_available_at": card.get("knowledge_available_at"),
    }
    if args.dry_run:
        print(json.dumps({"status": "dry_run", "would_write": proposal},
                         ensure_ascii=False, indent=2))
        return 0

    rows.append(proposal)
    _write(PROPOSALS, proposals)
    card.setdefault("derived_proposals", []).append(name)
    card.setdefault("local_status", {})["state"] = "unverified"
    _write(STORE, store)
    print(json.dumps({"status": "promoted", "factor_name": name, "card_id": card["card_id"]},
                     ensure_ascii=False))
    return 0


def cmd_verify(_args):
    store = load_store()
    bad = []
    seen = set()
    for card in store["cards"]:
        problems = validate(card)
        cid = card.get("card_id")
        if not cid:
            problems.append("card_id 缺失")
        elif cid in seen:
            problems.append("card_id 重复")
        seen.add(cid)
        if problems:
            bad.append({"card_id": cid, "problems": problems})
    print(json.dumps({"schema_version": store.get("schema_version"), "total": len(store["cards"]),
                      "invalid": len(bad), "details": bad}, ensure_ascii=False, indent=2))
    return 0 if not bad else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("init", help="初始化空卡片库").set_defaults(func=cmd_init)

    add = sub.add_parser("add", help="录入卡片（单卡或 {cards:[...]}）")
    add.add_argument("--file", required=True)
    add.set_defaults(func=cmd_add)

    ls = sub.add_parser("list", help="按状态/市场过滤")
    ls.add_argument("--state", choices=STATES)
    ls.add_argument("--market")
    ls.add_argument("--source")
    ls.set_defaults(func=cmd_list)

    promote = sub.add_parser("promote", help="卡片派生为候选写入 stage1_factor_proposals")
    promote.add_argument("--id", required=True)
    promote.add_argument("--name", help="factor_name，默认 kc_<digest>")
    promote.add_argument("--paradigm", help="覆盖 paradigm 标签")
    promote.add_argument("--dry-run", action="store_true")
    promote.set_defaults(func=cmd_promote)

    verify = sub.add_parser("verify", help="全库格式校验")
    verify.set_defaults(func=cmd_verify)

    args = parser.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
