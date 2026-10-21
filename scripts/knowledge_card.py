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
from collections import Counter
from datetime import datetime, date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fuxi_v1.archetypes import archetype_of, load_archetypes
STORE = ROOT / "data" / "knowledge_cards.json"
PROPOSALS = ROOT / "data" / "stage1_factor_proposals.json"
SCHEMA = "fuxi.knowledge_card.1"

STATES = ("unverified", "data_insufficient", "replicated", "falsified")
REPLICATION = ("not_replicated", "replicated", "failed", "unknown")
# 与 stage1_factor_proposals 现有契约对齐：direction 用 1 / -1。
DIRECTIONS = (1, -1)
DIRECTION_ALIASES = {1: 1, -1: -1, "1": 1, "-1": -1,
                     "positive": 1, "negative": -1}

# ------------------------------------------------------------------ 卡片 archetype
# 卡层把全局 24 archetype 中的「fundamental」细分为 6 个亚族（Fama-French 5 因子
# + 质量/成长），用于跨族配额与分布可视化；全局 24 archetype（campaign 层 HHI 预算
# 使用）保持不动。卡片 archetype 值域 = 24 全局族 ∪ 下面 6 个 fundamental 亚族。
FUNDAMENTAL_SUB_ZH = {
    "value": "价值(估值)", "quality": "质量(盈利质量)", "growth": "成长",
    "size": "规模", "profitability": "盈利", "investment": "投资",
}


def _refine_fundamental(formula, text):
    """fundamental 卡细分为 6 亚族（formula 字段精确优先，rationale 关键词兜底）。

    顺序敏感：先判最特异性字段（log(市值)=规模、asset_growth=投资、
    operating_profitability=盈利、shift/rd=成长、total_debt=杠杆质量），
    再判估值/质量字段。total_debt 须先于 total_equity（杠杆 vs 账面市值比）。
    """
    f = (formula or "").lower()
    if "log(market_cap)" in f or "log market_cap" in f:
        return "size"
    if "asset_growth" in f:
        return "investment"
    if "operating_profitability" in f:
        return "profitability"
    if "rd_expense" in f or ("shift" in f and ("revenue" in f or "net_income" in f)):
        return "growth"
    if "total_debt" in f:
        return "quality"  # 杠杆率：先于 total_equity 判，避免误入价值族
    if any(t in f for t in ("pb", "ps_ttm", "book_to_market", "dividend", "ncav",
                            "current_assets", "total_equity")):
        return "value"
    if any(t in f for t in ("cash", "accrual", "capex",
                            "operating_cash_flow", "total_assets", "operating_income")):
        return "quality"
    t = (text or "").lower()
    if any(k in t for k in ("价值", "估值", "便宜", "股息", "净净")):
        return "value"
    if any(k in t for k in ("成长", "增速", "增长", "研发")):
        return "growth"
    if any(k in t for k in ("盈利", "利润", "质量", "现金流", "杠杆", "应计")):
        return "quality"
    if any(k in t for k in ("规模", "市值")):
        return "size"
    return "fundamental"


def classify_card(card):
    """卡片 → archetype（卡层扩充口径：fundamental 细分为 6 亚族）。

    公式优先（字段精确，最可靠）：先 fundamental 亚族，再量价（volume/amount/corr）。
    仅当公式无信号时退回语义关键词（title + economic_rationale），处理
    反转/动量/事件等无独有字段的族。语义匹配仍可能因英文标题/中文理由碰撞，
    故不作为首选。
    """
    source = card.get("source") or {}
    mechanism = card.get("mechanism") or {}
    title = source.get("title") or ""
    rationale = mechanism.get("economic_rationale") or ""
    formula = str(mechanism.get("formula_pandas") or "")
    # 1. formula-first：fundamental 亚族
    sub = _refine_fundamental(formula, rationale)
    if sub != "fundamental":
        return sub
    # 2. formula-first：量价
    f = formula.lower()
    if any(t in f for t in ("volume", "amount", "corr")):
        return "price_volume"
    # 3. 语义兜底：24-archetype 关键词
    archetypes, unclassified = load_archetypes()
    return archetype_of({"label": title, "economic_rationale": rationale, "paradigm": ""},
                        archetypes, unclassified)


def archetype_zh(arch):
    if arch in FUNDAMENTAL_SUB_ZH:
        return FUNDAMENTAL_SUB_ZH[arch]
    archetypes, unclassified = load_archetypes()
    zh = {a["id"]: a["zh"] for a in archetypes}
    zh[unclassified] = "未分类"
    return zh.get(arch, arch)


def _family_counts(store, proposals):
    """card 派生提案按 archetype 计数（提案缺 archetype 时回查卡片）。"""
    cnt = Counter()
    card_by_id = {c.get("card_id"): c for c in store["cards"]}
    for p in proposals:
        if p.get("source_proposal") != "KNOWLEDGE-CARD":
            continue
        arch = p.get("archetype")
        if arch is None:
            card = card_by_id.get(p.get("knowledge_card_id"))
            arch = (card or {}).get("archetype")
        if arch:
            cnt[arch] += 1
    return cnt


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
        card["archetype"] = classify_card(card)
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
    if args.by_archetype:
        return _list_by_archetype(rows)
    summary = [{"card_id": c.get("card_id"),
                "title": (c.get("source") or {}).get("title", "")[:60],
                "market": (c.get("scope") or {}).get("market"),
                "state": (c.get("local_status") or {}).get("state"),
                "available_at": c.get("knowledge_available_at"),
                "archetype": c.get("archetype"),
                "archetype_zh": archetype_zh(c.get("archetype")),
                "derived": len(c.get("derived_proposals") or [])} for c in rows]
    print(json.dumps({"count": len(summary), "cards": summary},
                     ensure_ascii=False, indent=2))
    return 0


def _list_by_archetype(rows):
    cnt = Counter(c.get("archetype") or "unclassified" for c in rows)
    lowvol = cnt.get("volatility", 0)
    reversal = cnt.get("reversal", 0)
    dist = [{"archetype": a, "zh": archetype_zh(a), "count": n}
            for a, n in cnt.most_common()]
    print(json.dumps({
        "count": len(rows),
        "distribution": dist,
        "volatility_low_vol_count": lowvol,
        "reversal_count": reversal,
        "low_vol_or_reversal_total": lowvol + reversal,
        "low_vol_or_reversal_ratio": round((lowvol + reversal) / len(rows), 4) if rows else 0.0,
    }, ensure_ascii=False, indent=2))
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
    card_arch = card.get("archetype") or classify_card(card)
    # 跨族配额：card 派生候选按 archetype 计数，防止单族泛滥（低波/反转簇集中）。
    if args.max_per_family is not None and args.max_per_family > 0:
        counts = _family_counts(store, rows)
        cur = counts.get(card_arch, 0)
        if cur >= args.max_per_family:
            print(json.dumps({"status": "quota_blocked",
                              "archetype": card_arch, "archetype_zh": archetype_zh(card_arch),
                              "family_count": cur, "max_per_family": args.max_per_family,
                              "reason": "该族已派生 %d 个候选，达跨族配额上限 %d" % (cur, args.max_per_family)},
                             ensure_ascii=False, indent=2))
            return 0
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
        "archetype": card_arch,
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


def cmd_reclassify(args):
    """回填/重算所有卡片的 archetype 字段（append-only 库的字段补齐迁移）。"""
    store = load_store()
    changed = 0
    for card in store["cards"]:
        arch = classify_card(card)
        if card.get("archetype") != arch:
            card["archetype"] = arch
            changed += 1
    if changed:
        store["updated_at"] = datetime.now().isoformat(timespec="seconds")
        _write(STORE, store)
    print(json.dumps({"status": "ok", "changed": changed, "total": len(store["cards"])},
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
    ls.add_argument("--by-archetype", action="store_true",
                    help="输出 archetype 分布（含低波/反转聚焦计数）")
    ls.set_defaults(func=cmd_list)

    promote = sub.add_parser("promote", help="卡片派生为候选写入 stage1_factor_proposals")
    promote.add_argument("--id", required=True)
    promote.add_argument("--name", help="factor_name，默认 kc_<digest>")
    promote.add_argument("--paradigm", help="覆盖 paradigm 标签")
    promote.add_argument("--max-per-family", type=int, default=3,
                         help="跨族配额：同 archetype 已派生候选数上限（0=不设限，默认 3）")
    promote.add_argument("--dry-run", action="store_true")
    promote.set_defaults(func=cmd_promote)

    reclassify = sub.add_parser("reclassify", help="回填/重算卡片 archetype 字段")
    reclassify.set_defaults(func=cmd_reclassify)

    verify = sub.add_parser("verify", help="全库格式校验")
    verify.set_defaults(func=cmd_verify)

    args = parser.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
