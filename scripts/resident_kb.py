# -*- coding: utf-8 -*-
"""常驻策展知识库 (Resident Curated Knowledge Base) 工具 2026-10-12。

背景:
  40R 批实测 top-2 继承窗口只继承 22 条(20R 试点 52 条)——策展 d1a7a6c5 的 46 条长期知识
  被新 run 挤出继承窗(runs 目录 mtime 排序天然排挤历史策展知识)。
  本工具把继承窗口改为 slot 制:
    slot-1(新鲜位) = 最新 run 的策展副本(无副本则原库);
    slot-2(常驻位) = 跨批合并常驻知识库 curated_memory/resident_kb/。
  滚动: 新批完成后 -> 旧批策展合并进常驻库 -> 新批顶替 slot-1。

子命令:
  build      --cap 40 [--runs <run_id,...>]
            把全部策展副本(可选追加指定 run)合并为常驻库。
            判据: R1 neg-AST 去重 / R2 formula_hash 去重 / R3 家族配额容量预算。
  curate-run [--run <run_id>] [--cap-context resident]
            对最新 run(默认 mtime 最新)生成 slot-1 策展副本:
            R1 同库 neg-AST 去重 + R2 与常驻库 formula_hash 去重(只裁 slot-1 侧被支配卡)。
            产出 curation_manifest.json, 被 locate_memory_sources 既有 curated_map 机制接管。
  status     打印常驻库摘要 + 窗口合计预测。

容量契约(锚实测):
  normalize_request: export_memory 记录合计 <= 100 (unified_runner L114-115)。
  CAP_RESIDENT=40: 40 + slot-1 策展后 60 = 100; 40R 批实测策展后 47, 留 13 余量。
  若未来单批策展后 > 60: build --cap 下调预算。

数据库构建契约(锚生产读取面):
  export_memory 要求 campaigns 表恰好 1 行且 learning_compatibility 一致;
  facts.digest(payload)==hash 原行直拷; consumers.state='valid' + rewards 修订匹配。
  rewards.slot 实测为各库独立随机 UUID、跨库零重叠 -> union 判定语义逐位一致。
  表: 主源复制 schema+campaigns+attempts+exposure+protection+receipts;
      facts 仅保留被保留卡片; consumers 仅保留对应 state_id; rewards 全量 union;
      facts_no_update/facts_no_delete 触发器恢复。原库零改动。

判据先行文档: reports/fuxi_v1_resident_kb_20261012/RESIDENT_KB_PLAN_20261012.md
"""
import argparse
import hashlib
import json
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[1]
RUNS = WORKSPACE / "data" / "fuxi_unified" / "runs"
CURATED = WORKSPACE / "data" / "fuxi_unified" / "curated_memory"
RESIDENT_DIR = CURATED / "resident_kb"
RESIDENT_DB = RESIDENT_DIR / "research" / "evidence.sqlite"
RESIDENT_MANIFEST = RESIDENT_DIR / "resident_manifest.json"

CAP_RESIDENT_DEFAULT = 40

TRIGGERS = (
    ("facts_no_update",
     "CREATE TRIGGER facts_no_update BEFORE UPDATE ON facts BEGIN SELECT RAISE(ABORT,'immutable facts'); END;"),
    ("facts_no_delete",
     "CREATE TRIGGER facts_no_delete BEFORE DELETE ON facts BEGIN SELECT RAISE(ABORT,'immutable facts'); END;"),
)

TABLES = ("attempts", "campaigns", "consumers", "exposure", "facts",
          "protection", "receipts", "rewards")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def norm_ast(node):
    """去顶层 neg 的 canonical_ast 归一化（结构判据，非语义）。"""
    if isinstance(node, dict):
        if node.get("op") == "neg" and "args" in node:
            return norm_ast(node["args"][0])
        return {k: (norm_ast(v) if k == "args" else v) for k, v in sorted(node.items())}
    if isinstance(node, list):
        return [norm_ast(x) for x in node]
    return node


def read_cards(path: Path):
    """读证据库全部 feedback_v2 卡片(校验 hash), 返回 [dict(rowid, fact_id, card)]。"""
    db = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        cards = []
        for rowid, fid, payload, h in db.execute(
                "SELECT rowid,id,payload,hash FROM facts WHERE kind='feedback_v2' ORDER BY rowid"):
            assert hashlib.sha256(payload.encode("utf-8")).hexdigest() == h, f"corrupted fact {fid}"
            cards.append({"rowid": rowid, "fact_id": fid, "card": json.loads(payload)})
    finally:
        db.close()
    return cards


def read_rows(path: Path, table: str, columns: str, where: str = ""):
    db = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    try:
        sql = f"SELECT {columns} FROM {table}" + (f" WHERE {where}" if where else "")
        rows = [dict(r) for r in db.execute(sql)]
    finally:
        db.close()
    return rows


def ic_of(card):
    v = card.get("mean_rank_ic")
    return v if v is not None else float("-inf")


def keep_key(card, rowid=0):
    """留谁排序键: |IC| 大 > parent_comparisons 多 > rowid 大。"""
    return (abs(ic_of(card)), len(card.get("parent_comparisons") or []), rowid)


def mechanism_of(card):
    cand = card.get("candidate") or {}
    hyp = cand.get("hypothesis") or {}
    return hyp.get("mechanism") or "?"


def coarse_family(card):
    m = mechanism_of(card)
    return m if len(m) <= 25 else "verbose_hypothesis"


def compat_of(path: Path) -> str:
    db = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        cfg = json.loads(db.execute("SELECT config FROM campaigns").fetchone()[0])
        return cfg.get("learning_compatibility")
    finally:
        db.close()


def load_cards_with_meta(paths):
    """读多库卡片并附元信息: [dict(rowid, fact_id, card, src_path)]。"""
    out = []
    for p in paths:
        for c in read_cards(p):
            c["src_path"] = str(p.resolve())
            out.append(c)
    return out


def apply_r1r2(cards, extra_cards=None):
    """R1 neg-AST / R2 formula_hash 跨库全局去重。返回 (kept, cuts)。

    extra_cards: 另一侧(常驻库)卡片, 参与分组但不被裁切——用于 curate-run 与常驻库比大小。
    """
    all_cards = list(cards)
    if extra_cards:
        all_cards = all_cards + list(extra_cards)
    cuts = {}
    # R1
    by_ast = defaultdict(list)
    for c in all_cards:
        ast = ((c["card"].get("candidate") or {}).get("canonical_ast"))
        if ast is None:
            continue
        by_ast[json.dumps(norm_ast(ast), sort_keys=True)].append(c)
    for key, members in by_ast.items():
        if len(members) < 2:
            continue
        members.sort(key=lambda c: keep_key(c["card"], c["rowid"]), reverse=True)
        kept = members[0]
        for d in members[1:]:
            if d is kept:
                continue
            cuts[d["fact_id"]] = (
                f"R1 neg-AST重复: 与 {kept['fact_id'][:12]} 同结构, "
                f"|IC| {abs(ic_of(d['card'])):.4f} <= 保留 {abs(ic_of(kept['card'])):.4f}")
    # R2
    by_fh = defaultdict(list)
    for c in all_cards:
        fh = ((c["card"].get("candidate") or {}).get("formula_hash"))
        if fh:
            by_fh[fh].append(c)
    for fh, members in by_fh.items():
        if len(members) < 2:
            continue
        members.sort(key=lambda c: keep_key(c["card"], c["rowid"]), reverse=True)
        kept = members[0]
        for d in members[1:]:
            if d is kept:
                continue
            if d["fact_id"] in cuts:
                continue
            cuts[d["fact_id"]] = (
                f"R2 formula_hash重复({fh[:10]}): 与 {kept['fact_id'][:12]} 同公式, "
                f"|IC| {abs(ic_of(d['card'])):.4f} <= 保留 {abs(ic_of(kept['card'])):.4f}")
    kept = [c for c in cards if c["fact_id"] not in cuts]
    return kept, cuts


def apply_r3_quota(kept, cap):
    """R3 家族配额: 每 coarse 家族保 1 名额(keep_key 最大), 其余按 keep_key 全局排序填充。"""
    if len(kept) <= cap:
        return kept, {}
    families = defaultdict(list)
    for c in kept:
        families[coarse_family(c["card"])].append(c)
    chosen = []
    for fam, members in families.items():
        members.sort(key=lambda c: keep_key(c["card"], c["rowid"]), reverse=True)
        chosen.append(members[0])
    rest = [c for c in kept if c not in chosen]
    rest.sort(key=lambda c: keep_key(c["card"], c["rowid"]), reverse=True)
    chosen += rest[: max(0, cap - len(chosen))]
    chosen_ids = {c["fact_id"] for c in chosen}
    cuts = {}
    for c in kept:
        if c["fact_id"] not in chosen_ids:
            cuts[c["fact_id"]] = (
                f"R3 容量配额(CAP={cap}): 家族 {coarse_family(c['card'])} 内 "
                f"|IC| {abs(ic_of(c['card'])):.4f} 排名靠后")
    return chosen, cuts


def build_resident_db(kept_cards, master_path):
    """以 master_path 为 schema 主源, 构建常驻库(全量重建)。"""
    RESIDENT_DIR.mkdir(parents=True, exist_ok=True)
    (RESIDENT_DIR / "research").mkdir(parents=True, exist_ok=True)
    out_db = RESIDENT_DB
    if out_db.exists():
        out_db.unlink()
    dst = sqlite3.connect(str(out_db))
    try:
        dst.execute("ATTACH DATABASE ? AS src", (str(master_path.resolve()),))
        for tbl in TABLES:
            dst.execute(f"CREATE TABLE {tbl} AS SELECT * FROM src.{tbl}")
        dst.execute("DETACH DATABASE src")
        # 清空并重建三张合并表
        dst.execute("DELETE FROM facts")
        dst.execute("DELETE FROM consumers")
        dst.execute("DELETE FROM rewards")
        # facts: 仅保留被保留卡片(原行直拷, digest(payload)==hash 保真链不重写)
        seen_fids = set()
        for c in kept_cards:
            assert c["fact_id"] not in seen_fids, f"fact id collision {c['fact_id']}"
            seen_fids.add(c["fact_id"])
            rows = read_rows(Path(c["src_path"]), "facts",
                             "id,kind,payload,hash,created", where=f"id='{c['fact_id']}'")
            assert len(rows) == 1, f"fact {c['fact_id']} missing in {c['src_path']}"
            r = rows[0]
            assert hashlib.sha256(r["payload"].encode("utf-8")).hexdigest() == r["hash"], \
                f"corrupted fact {c['fact_id']}"
            dst.execute(
                "INSERT INTO facts(id,kind,payload,hash,created) VALUES(?,?,?,?,?)",
                (r["id"], r["kind"], r["payload"], r["hash"], r["created"]))
        # consumers: 保留卡片 state_id 对应行
        state_ids = {c["card"].get("state_id") for c in kept_cards if c["card"].get("state_id")}
        consumer_rows = []
        for c in kept_cards:
            sid = c["card"].get("state_id")
            if not sid:
                continue
            rows = read_rows(Path(c["src_path"]), "consumers", "id,dependencies,state,checkpoint",
                             where=f"id='{sid}'")
            assert len(rows) == 1, f"consumer {sid} not found in {c['src_path']}"
            consumer_rows.append(rows[0])
        for r in consumer_rows:
            dst.execute("INSERT INTO consumers(id,dependencies,state,checkpoint) VALUES(?,?,?,?)",
                        (r["id"], r["dependencies"], r["state"], r["checkpoint"]))
        # rewards: 全量 union(断言 slot 跨库无冲突)
        slot_src = {}
        for c in kept_cards:
            rows = read_rows(Path(c["src_path"]), "rewards",
                             "slot,revision,event_id,target,value,source_id,supersedes")
            for r in rows:
                if r["slot"] in slot_src and slot_src[r["slot"]] != c["src_path"]:
                    raise AssertionError(f"slot conflict: {r['slot']}")
                slot_src[r["slot"]] = c["src_path"]
                dst.execute(
                    "INSERT OR IGNORE INTO rewards(slot,revision,event_id,target,value,source_id,supersedes)"
                    " VALUES(?,?,?,?,?,?,?)",
                    (r["slot"], r["revision"], r["event_id"], r["target"], r["value"],
                     r["source_id"], r["supersedes"]))
        for tname, tsql in TRIGGERS:
            dst.execute(f"DROP TRIGGER IF EXISTS {tname}")
            dst.execute(tsql)
        dst.commit()
    finally:
        dst.close()


# ---------------------------------------------------------------- build
def cmd_build(args):
    sources = []
    for entry in sorted(CURATED.iterdir()):
        if entry.name == "resident_kb" or not entry.is_dir():
            continue
        man = entry / "curation_manifest.json"
        db = entry / "research" / "evidence.sqlite"
        if man.is_file() and db.is_file():
            sources.append(db)
    for rid in (args.runs or []):
        run_dir = next((RUNS / d for d in sorted(p.name for p in RUNS.iterdir())
                        if d.startswith(f"unified_{rid}")), None)
        if run_dir is None or not run_dir.exists():
            print(f"[warn] run {rid} not found, skip"); continue
        evs = list(run_dir.glob("research/*/evidence.sqlite"))
        if evs:
            sources.append(evs[0])
    if not sources:
        print("no curated sources found"); return 1
    # 断言: compat 一致
    compats = {compat_of(p) for p in sources}
    assert len(compats) == 1, f"compat mismatch: {compats}"
    # 读卡
    cards = load_cards_with_meta(sources)
    print(f"source cards: {len(cards)} from {len(sources)} sources")
    # R1+R2
    kept, cuts12 = apply_r1r2(cards)
    print(f"R1+R2: kept {len(kept)} / cut {len(cuts12)}")
    # 家族并集断言
    orig_families = {coarse_family(c["card"]) for c in cards}
    kept_families = {coarse_family(c["card"]) for c in kept}
    assert kept_families == orig_families, f"family lost: {orig_families - kept_families}"
    # R3 配额
    kept, cuts3 = apply_r3_quota(kept, args.cap)
    cuts = {**cuts12, **{k: v for k, v in cuts3.items() if k not in cuts12}}
    print(f"R3(CAP={args.cap}): kept {len(kept)}")
    # 构建
    master = sources[0]
    build_resident_db(kept, master)
    # manifest
    src_hashes = {str(p): sha256_file(p) for p in sources}
    manifest = {
        "schema_version": "fuxi.resident_kb.1",
        "built_at": datetime.now(timezone.utc).isoformat(),
        "cap": args.cap,
        "compatibility_hash": compats.pop(),
        "merged_sources": src_hashes,
        "master_source": str(master.resolve()),
        "kept_count": len(kept),
        "kept_cards": [
            {"fact_id": c["fact_id"], "src": c["src_path"],
             "ic": ic_of(c["card"]), "family": coarse_family(c["card"]),
             "formula_hash": (((c["card"].get("candidate") or {}).get("formula_hash")) or "")[:12]}
            for c in sorted(kept, key=lambda c: keep_key(c["card"], c["rowid"]), reverse=True)],
        "cut_reasons": cuts,
        "policy": ("slot-2 常驻位。判据 R1(neg-AST)/R2(formula_hash) 全局去重 + R3 家族配额(CAP)。"
                   "campaigns 单行 + schema 主源复制; facts/consumers/rewards 保真合并; "
                   "原证据库零改动。"),
        "evidence_sha256": sha256_file(RESIDENT_DB),
    }
    RESIDENT_MANIFEST.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    fams = defaultdict(int)
    for c in kept:
        fams[coarse_family(c["card"])] += 1
    print(f"resident_kb built: {len(kept)} cards, families: {dict(sorted(fams.items()))}")
    print(f"window budget: CAP {args.cap} + slot-1 <= 60 -> total <= 100")
    return 0


# ---------------------------------------------------------------- curate-run
def cmd_curate_run(args):
    # 最新 run
    run_dirs = sorted(
        [p for p in RUNS.iterdir() if p.is_dir() and list(p.glob("research/*/evidence.sqlite"))],
        key=lambda p: p.stat().st_mtime, reverse=True)
    if args.run:
        run_dirs = [p for p in run_dirs if p.name.startswith(f"unified_{args.run}")]
    if not run_dirs:
        print("no run found"); return 1
    run_dir = run_dirs[0]
    ev = list(run_dir.glob("research/*/evidence.sqlite"))[0]
    run8 = run_dir.name.split("_", 1)[1][:8]
    print(f"curate-run: {run_dir.name} ({len(read_cards(ev))} cards)")
    # 常驻库侧卡片(不裁, 参与比较)
    resident_cards = []
    if RESIDENT_DB.is_file():
        resident_cards = load_cards_with_meta([RESIDENT_DB])
        print(f"resident side: {len(resident_cards)} cards for comparison")
    cards = load_cards_with_meta([ev])
    kept, cuts = apply_r1r2(cards, extra_cards=resident_cards)
    # 只裁 slot-1 侧(apply_r1r2 的 cuts 只对 cards 成员生效, extra 不参与裁切)
    # 家族保护: 豁免每家族唯一代表
    fam_count = defaultdict(int)
    for c in cards:
        fam_count[coarse_family(c["card"])] += 1
    final_cuts = {}
    for c in cards:
        if c["fact_id"] in cuts and fam_count[coarse_family(c["card"])] > 1:
            final_cuts[c["fact_id"]] = cuts[c["fact_id"]]
            fam_count[coarse_family(c["card"])] -= 1
    kept = [c for c in cards if c["fact_id"] not in final_cuts]
    print(f"slot-1 curated: kept {len(kept)} / cut {len(final_cuts)}")
    # 构建策展副本(复用旧 curated 机制: manifest 带 source_evidence_path)
    out_dir = CURATED / f"curated_{datetime.now():%Y%m%d}_{run8}"
    out_research = out_dir / "research"
    out_research.mkdir(parents=True, exist_ok=True)
    out_db = out_research / "evidence.sqlite"
    if out_db.exists():
        out_db.unlink()
    dst = sqlite3.connect(str(out_db))
    try:
        dst.execute("ATTACH DATABASE ? AS src", (str(ev.resolve()),))
        for tbl in TABLES:
            dst.execute(f"CREATE TABLE {tbl} AS SELECT * FROM src.{tbl}")
        dst.execute("DETACH DATABASE src")
        for fid in sorted(final_cuts):
            dst.execute("DELETE FROM facts WHERE id=?", (fid,))
        for tname, tsql in TRIGGERS:
            dst.execute(f"DROP TRIGGER IF EXISTS {tname}")
            dst.execute(tsql)
        dst.commit()
    finally:
        dst.close()
    fams = defaultdict(int)
    for c in kept:
        fams[coarse_family(c["card"])] += 1
    manifest = {
        "schema_version": "fuxi.memory_curation.1",
        "curated_at": datetime.now(timezone.utc).isoformat(),
        "curated_name": f"curated_{datetime.now():%Y%m%d}_{run8}",
        "slot": "slot-1",
        "source_evidence_path": str(ev.resolve()),
        "source_evidence_sha256": sha256_file(ev),
        "policy": ("slot-1 新鲜位策展副本。R1 同库 neg-AST 去重 + R2 与常驻库 formula_hash 去重"
                   "(只裁本侧被支配卡, 常驻库不动); 家族代表豁免。原库零改动。"),
        "cut_fact_ids": sorted(final_cuts),
        "cut_count": len(final_cuts),
        "kept_feedback_v2_count": len(kept),
        "original_feedback_v2_count": len(cards),
        "mechanism_families_kept": dict(sorted(fams.items())),
        "cut_reasons": {k: final_cuts[k] for k in sorted(final_cuts)},
    }
    (out_dir / "curation_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"slot-1 curated copy: {out_dir}")
    return 0


# ---------------------------------------------------------------- status
def cmd_status(args):
    resident_n = 0
    if RESIDENT_DB.is_file():
        try:
            resident_n = len(export_records(RESIDENT_DB))
        except Exception as e:
            print(f"[warn] resident db unreadable: {e}")
    print(f"resident_kb: {resident_n} records (CAP {CAP_RESIDENT_DEFAULT})")
    run_dirs = sorted(
        [p for p in RUNS.iterdir() if p.is_dir() and list(p.glob("research/*/evidence.sqlite"))],
        key=lambda p: p.stat().st_mtime, reverse=True)
    if run_dirs:
        ev = list(run_dirs[0].glob("research/*/evidence.sqlite"))[0]
        n = len(read_cards(ev))
        print(f"latest run: {run_dirs[0].name[:20]}... {n} cards (raw)")
        print(f"window forecast: {n} + {resident_n} = {n + resident_n} {'<=100 OK' if n + resident_n <= 100 else '>100 OVER'}")
    return 0


def export_records(path):
    """轻量读常驻库可用记录数(不依赖 fuxi 包时兜底直接数卡)。"""
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "learning", str(WORKSPACE / "fuxi_v1" / "learning.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        man = json.loads(RESIDENT_MANIFEST.read_text(encoding="utf-8"))
        return mod.export_memory(path, man["compatibility_hash"])["records"]
    except Exception:
        return [c for c in read_cards(path)]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="合并策展副本为常驻库")
    b.add_argument("--cap", type=int, default=CAP_RESIDENT_DEFAULT)
    b.add_argument("--runs", nargs="*", default=[], help="追加指定 run_id 前 8 位")
    b.set_defaults(func=cmd_build)
    c = sub.add_parser("curate-run", help="对最新 run 生成 slot-1 策展副本")
    c.add_argument("--run", default=None, help="指定 run_id 前 8 位(默认 mtime 最新)")
    c.set_defaults(func=cmd_curate_run)
    s = sub.add_parser("status", help="常驻库摘要与窗口预测")
    s.set_defaults(func=cmd_status)
    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
