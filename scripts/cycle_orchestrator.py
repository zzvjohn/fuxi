"""伏羲 v2.0 闭环编排器（循环驾驶舱）。

把研究闭环的七环节串成一条自动流水线，形成"探索→假设→研究→积累→反馈"的闭环：

  1. source_scan  知识探索  —— 读深扫产出的因子候选（stage1_factor_proposals.json）
  2. data_refresh 数据采集  —— stage0 增量更新（可选，--refresh-data 时执行）
  3. snapshot     快照定位  —— 定位冻结的 campaign 输入快照（fuxi.campaign_input.1）
  4. hypothesis   假设提炼  —— 深扫候选 → external_hypotheses（mechanism/direction/source）
  5. research     研究内核  —— unified_runner run（LLM 生成 + 筛选 + 回测 + 记忆继承）
  6. feedback     经验积累  —— shared_feedback 导入（unified_runner 内已含）
  7. reflect      反思反馈  —— 导出记忆 + 写回下一轮证据清单（闭环）

闭环：第 7 阶段产出的"历史 feedback + 已证伪 motif"回灌第 4 阶段的假设证据，
以及作为下一轮第 1 阶段深扫的"已证伪"过滤清单。

治理：本编排器是操作员的自动化代理，以 manual 身份驱动研究，不自动改授权、不自动接触实盘。
"""
import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# 工作区根目录（脚本位于 e:/quant/scripts/）
WORKSPACE = Path(__file__).resolve().parents[1]
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))
PYTHON = r"C:/Users/Zhong/.workbuddy/binaries/python/envs/default/Scripts/python.exe"
LEDGER = WORKSPACE / "data" / "fuxi_unified" / "cycle_ledger.json"
PROPOSALS = WORKSPACE / "data" / "stage1_factor_proposals.json"
FALSIFIED_MOTIFS = WORKSPACE / "data" / "fuxi_unified" / "falsified_motifs.json"

# 1125 截面快照在内存展开后单份近 1.1GB，主进程 + worker 各载一份即超 2GiB 治理默认；
# manual 身份可显式抬升 worker 内存上限（design-sanctioned，见 scripts/ab_ga_ops.py 同款先例）。
# 八轴须全显式且与 DEFAULT_LIMITS 键集一致；仅 max_memory_mb 2048→8192，其余轴保持默认。
DAILY_LIMITS = {
    "max_rounds": 40, "max_candidates": 80, "max_worker_seconds": 420,
    "max_total_seconds": 36000, "max_memory_mb": 8192,
    "max_external_requests": 30, "max_llm_calls": 12, "max_output_tokens": 24000,
}


def parallel_environment(base=None):
    """M1/M2 speedup for the daily cycle (user-authorized 2026-10-19).

    The 300s→420s worker window plus a 2% overrun tolerance removes the
    measurement-jitter halt; M1/M2 turn the serial 4-candidate round into a
    spawn-all/collect-in-proposal-order round (bit-exact A/B per ARCHITECTURE §7).
    Set FUXI_CYCLE_PARALLEL=0 to fall back to the serial path.
    """
    env = dict(os.environ if base is None else base)
    if env.get("FUXI_CYCLE_PARALLEL", "1") != "0":
        env.setdefault("FUXI_V1_PARALLEL_CANDIDATES", "1")
        env.setdefault("FUXI_V1_PERSISTENT_POOL", "1")
        env.setdefault("FUXI_V1_POOL_SIZE", "4")
        env.setdefault("FUXI_V1_POOL_MAX_REQUESTS", "40")
    return env


def _now():
    return datetime.now(timezone.utc).isoformat()


def _load_json(path, default=None):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp." + str(int(time.time())))
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def _write_json_crlf(path, value):
    """JSON 写出并强制 CRLF 行尾（proposals.json 等 CRLF 资产专用，禁止 Edit 静默转 LF）。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp." + str(int(time.time())))
    text = json.dumps(value, ensure_ascii=False, indent=2)
    text = text.replace("\r\n", "\n").replace("\n", "\r\n")
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        f.write(text)
    tmp.replace(path)


def _append_ledger(entry):
    ledger = _load_json(LEDGER, {"cycles": []})
    ledger.setdefault("cycles", []).append(entry)
    _write_json(LEDGER, ledger)


# ---------------------------------------------------------------- 经验台账
def _load_falsified_motifs():
    """读已证伪 motif 台账（reflect 产出，闭环自进化经验的一部分）。"""
    return _load_json(FALSIFIED_MOTIFS, {"motifs": []})


def _append_falsified_motifs(motifs):
    """把新证伪 motif 追加进台账（按 structural_family+mechanism 去重）。"""
    if not motifs:
        return 0
    book = _load_falsified_motifs()
    existing = book.setdefault("motifs", [])
    seen = {(m.get("structural_family"), m.get("mechanism")) for m in existing}
    added = 0
    for m in motifs:
        key = (m.get("structural_family"), m.get("mechanism"))
        if key in seen:
            continue
        existing.append(m)
        seen.add(key)
        added += 1
    book["updated_at"] = _now()
    _write_json(FALSIFIED_MOTIFS, book)
    return added


def _tested_flag(p):
    """提案是否已测试/已被消费：tested_at 非空，或 status 已落定/在途。"""
    status = p.get("status")
    return bool(p.get("tested_at")) or bool(status)


# ---------------------------------------------------------------- Phase 1+4
def select_hypotheses(limit=12):
    """从深扫提案里，按「archetype 均衡轮转 + 自进化指引」挑选待验证假设。

    第 1 层改造（P-20261017）：原实现按 paradigm 频次降序轮转，731 组长尾下
    大 paradigm（资金流系）+ __none__ 组垄断 limit 个名额，长尾 paradigm 永远
    轮不到，机制方向多样性收窄。现改为按 24 个 archetype（原型）均衡轮转，
    每 archetype 每轮最多贡献 1 条，保证 limit=12 覆盖 12 个不同 archetype。

    自进化指引（消费两类闭环经验）：
      1. tested_at / status —— 已测试/在途提案不再喂给 LLM（每轮轮转）；
      2. falsified_motifs —— 已证伪 paradigm 聚合到 archetype 层沉底（降权不清零）。
    组内实证锚点：同 paradigm 多条提案按 icir 降序取代表（高 ICIR 优先）。

    返回 (hypotheses, consumed)：
      hypotheses: [{mechanism,direction,source}]（契约合规，仅三字段）；
      consumed:   被选中的 factor_name 列表，供闭环回写 tested_at 实现轮转。
    """
    from fuxi_v1.archetypes import load_archetypes, archetype_of

    data = _load_json(PROPOSALS)
    if not data:
        return [], []
    proposals = data.get("proposals", [])

    # 已证伪 paradigm（自进化指引：按候选数加权的 archetype 软降权）
    book = _load_falsified_motifs()
    falsified_paradigms = set()
    for m in book.get("motifs", []):
        for p in (m.get("paradigms") or []):
            if p:
                falsified_paradigms.add(p)
    archetypes, unclassified = load_archetypes()

    # 待选池：未测试且机制文案非空
    pool = []
    for p in proposals:
        if _tested_flag(p):
            continue
        mechanism = (p.get("economic_rationale") or p.get("label") or "").strip()
        if not mechanism:
            continue
        pool.append(p)

    # archetype 分组 → 组内 paradigm 分组 → paradigm 内按 icir 降序（实证锚点）
    arch_groups = {}
    arch_order = []
    for p in pool:
        aid = archetype_of(p, archetypes, unclassified)
        para = (p.get("paradigm") or "").strip() or "__none__"
        if aid not in arch_groups:
            arch_groups[aid] = {}
            arch_order.append(aid)
        arch_groups[aid].setdefault(para, []).append(p)
    for aid in arch_groups:
        for para in arch_groups[aid]:
            arch_groups[aid][para].sort(key=lambda q: -(q.get("icir") or 0))

    # archetype 排序：按 falsified 候选比例软降权（证伪比例低的优先），比例同则
    # 候选数降序。软降权避免「一个 paradigm 证伪连坐整个 archetype」——P-20261017
    # 原二值沉底在 falsified 覆盖 22/24 archetype 时退化为只探长尾小 archetype。
    arch_falsified = {}
    arch_total = {}
    for aid in arch_order:
        tot = 0
        fls = 0
        for para, items in arch_groups[aid].items():
            tot += len(items)
            if para in falsified_paradigms:
                fls += len(items)
        arch_total[aid] = tot
        arch_falsified[aid] = fls
    arch_order.sort(key=lambda a: (arch_falsified[a] / max(1, arch_total[a]),
                                    -arch_total[a], a))

    # 每个 archetype 内 paradigm 按候选数降序（代表性领域优先）
    para_lists = {aid: sorted(arch_groups[aid].items(), key=lambda kv: -len(kv[1])) for aid in arch_order}
    cursor = {aid: 0 for aid in arch_order}

    # 逐 archetype 轮转：每轮每 archetype 取 1 条（第一个未消费 paradigm 的 icir 最高代表）
    selected = []
    progressed = True
    while len(selected) < limit and progressed:
        progressed = False
        for aid in arch_order:
            if len(selected) >= limit:
                break
            pl = para_lists[aid]
            idx = cursor[aid]
            if idx >= len(pl):
                continue
            selected.append(pl[idx][1][0])
            cursor[aid] = idx + 1
            progressed = True

    hypotheses = []
    consumed = []
    for p in selected:
        mechanism = (p.get("economic_rationale") or p.get("label") or "").strip()
        hypotheses.append({
            "mechanism": mechanism[:400],
            "direction": p.get("direction"),
            "source": (p.get("source_proposal") or p.get("factor_name") or "")[:60],
        })
        consumed.append(p.get("factor_name") or "")
    return hypotheses, consumed


def extract_external_hypotheses(limit=12):
    """兼容旧签名的薄封装：只返回 hypotheses 列表（preview/status 用）。"""
    return select_hypotheses(limit)[0]


def mark_tested(consumed, date_str):
    """闭环内回写 tested_at：把本轮消费的提案标记为已测试（仅未落定的）。

    匹配 factor_name；status 已落定（PASS/RESERVE 等）的提案不覆盖，只给
    尚为 None 的提案盖 tested_at 时间戳。用 CRLF 通道写出，避免行尾被转 LF。
    """
    if not consumed:
        return 0
    data = _load_json(PROPOSALS)
    if not data:
        return 0
    proposals = data.get("proposals", [])
    consumed_set = set(consumed)
    updated = 0
    for p in proposals:
        if p.get("factor_name") in consumed_set and not p.get("status"):
            p["tested_at"] = date_str
            updated += 1
    if updated:
        data["proposals"] = proposals
        _write_json_crlf(PROPOSALS, data)
    return updated


# ---------------------------------------------------------------- Phase 2
def data_refresh(refresh):
    if not refresh:
        return {"status": "skipped", "reason": "refresh disabled"}
    cmd = [PYTHON, "-u", str(WORKSPACE / "scripts" / "stage0_daily_update_v3.py")]
    result = subprocess.run(cmd, cwd=str(WORKSPACE), capture_output=True, text=True, timeout=600)
    ok = result.returncode == 0
    return {"status": "completed" if ok else "failed", "exit_code": result.returncode,
            "tail": (result.stdout or result.stderr or "")[-400:]}


# ---------------------------------------------------------------- Phase 3
def locate_snapshot(explicit):
    if explicit:
        p = Path(explicit)
        return str(p) if p.is_file() else None
    # 优先取操作员冻结的 inbox 请求 input，其次取最近研究副本的 input.json
    inbox = WORKSPACE / "data" / "fuxi_unified" / "inbox"
    if inbox.exists():
        for req_file in sorted(inbox.glob("*.json")):
            req = _load_json(req_file)
            if req and req.get("input"):
                p = Path(req["input"])
                if p.is_file():
                    return str(p)
    runs = WORKSPACE / "data" / "fuxi_unified" / "runs"
    candidates = []
    if runs.exists():
        for run_dir in runs.iterdir():
            for inp in run_dir.rglob("input.json"):
                if inp.is_file() and "research" in inp.parts:
                    candidates.append((inp.stat().st_mtime, str(inp)))
    if candidates:
        candidates.sort(reverse=True)
        return candidates[0][1]
    return None


# ---------------------------------------------------------------- Phase 3b
def locate_memory_sources(limit=2):
    """slot 制继承窗口（2026-10-12 常驻策展知识库改造）。

    slot-1（新鲜位）: runs 目录 mtime 最新 run 的策展副本（curated_map 替换），无副本则原库；
    slot-2（常驻位）: curated_memory/resident_kb/research/evidence.sqlite（跨批合并常驻知识库）。
    回退: 常驻库缺失/损坏时，slot-2 退回旧逻辑（mtime 次新 run，策展替换不变），自愈不冻结。
    """
    runs = WORKSPACE / "data" / "fuxi_unified" / "runs"
    sources = []
    if not runs.exists():
        return sources
    curated_map = {}
    curated_root = WORKSPACE / "data" / "fuxi_unified" / "curated_memory"
    if curated_root.exists():
        for entry in curated_root.iterdir():
            manifest_path = entry / "curation_manifest.json"
            if not manifest_path.is_file():
                continue
            try:
                manifest = _load_json(manifest_path)
                original = manifest.get("source_evidence_path")
                replacement = entry / "research" / "evidence.sqlite"
                if original and replacement.is_file():
                    curated_map[str(Path(original).resolve())] = str(replacement.resolve())
            except Exception:
                continue  # 坏清单不阻塞记忆源定位
    resident = None
    resident_dir = curated_root / "resident_kb"
    resident_db = resident_dir / "research" / "evidence.sqlite"
    if resident_db.is_file() and (resident_dir / "resident_manifest.json").is_file():
        try:
            rman = _load_json(resident_dir / "resident_manifest.json")
            if rman.get("compatibility_hash") and rman.get("kept_count"):
                resident = str(resident_db.resolve())
        except Exception:
            resident = None  # 常驻清单坏 -> 回退旧逻辑
    for run_dir in sorted(runs.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        research = run_dir / "research"
        if not research.exists():
            continue
        for ev in research.rglob("evidence.sqlite"):
            if not ev.is_file():
                continue
            resolved = str(ev.resolve())
            source = curated_map.get(resolved, str(ev))
            if not sources:
                sources.append(source)  # slot-1: 最新 run
                if resident and resident != source:
                    sources.append(resident)  # slot-2: 常驻库
                    if len(sources) >= limit:
                        return sources
                continue
            if resident is None:
                sources.append(source)  # 回退: mtime 次新
                if len(sources) >= limit:
                    return sources
    return sources


# ---------------------------------------------------------------- Phase 5
def run_research(snapshot, hypotheses, memory_sources, *, rounds=20, candidates=4, llm=True,
                 limits=None):
    """构造 unified request 并驱动统一控制器研究段。"""
    from fuxi_v1.contracts import load_json as f_load, digest

    if not snapshot or not Path(snapshot).is_file():
        return {"state": "refused", "reason": "no frozen snapshot; data_refresh/snapshot must run first"}

    request = {
        "schema_version": "fuxi.unified_run.1",
        "input": snapshot,
        "rounds": rounds,
        "candidates_per_round": candidates,
        "seed": int(time.time()) % 100000,
        "mode": "adaptive",
        "scheduler_policy": "legacy",
        "include_catalog": True,
        "purpose": "cycle closed-loop research",
        "llm_enabled": llm,
        "model_algorithms": [],
        "external_hypotheses": hypotheses,
    }
    # 显式预算：manual 身份可抬升 worker 内存上限（design-sanctioned）。省略时回落 DEFAULT_LIMITS。
    if limits:
        request["limits"] = dict(limits)

    # 记忆继承：从上一轮 evidence.sqlite 构造 memory_context（仅当格式可导出时）
    if memory_sources:
        try:
            request["memory_context"] = _build_memory_context(snapshot, memory_sources)
        except Exception as exc:  # 记忆不可用不阻塞闭环，降级为无记忆
            request["memory_context"] = None
            print(f"[cycle] memory_context degraded: {type(exc).__name__}: {exc}")

    # 显式冻结输入哈希（manual 身份仍要求 input_sha256 精确匹配）
    request["input_sha256"] = _sha256(snapshot)

    req_path = WORKSPACE / "data" / "fuxi_unified" / "cycle_request.json"
    _write_json(req_path, request)
    cmd = [PYTHON, "-B", "-X", "utf8", "-m", "fuxi_v1.unified_runner", "run",
           "--workspace", str(WORKSPACE), "--request", str(req_path), "--trigger", "manual"]
    # subprocess 超时与 workflow 时间预算（max_total_seconds）对齐，另加 300s 收尾缓冲。
    # 原硬编码 3600s 无法覆盖 20×4=80 候选在 1125 截面上的串行运行（实测约 70min）。
    _limits_for_timeout = dict(limits) if limits else dict(DAILY_LIMITS)
    _timeout = int(_limits_for_timeout.get("max_total_seconds", 14400)) + 300
    result = subprocess.run(cmd, cwd=str(WORKSPACE), capture_output=True, text=True, timeout=_timeout,
                            env=parallel_environment())
    # 成功时 unified_runner 打印 stdout JSON；拒绝/失败时打印 stderr JSON（exit_code=2）
    answer = {}
    for stream in (result.stdout, result.stderr):
        if not stream or not stream.strip():
            continue
        try:
            parsed = json.loads(stream.strip().splitlines()[-1])
            if isinstance(parsed, dict) and "state" in parsed:
                answer = parsed
                break
        except (json.JSONDecodeError, IndexError):
            continue
    if not answer:
        answer = {"state": "refused", "reason": (result.stderr or result.stdout or "")[-500:]}
    answer["exit_code"] = result.returncode
    return answer


def _sha256(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _build_memory_context(snapshot, memory_sources):
    """用 learning.export_memory 构造跨轮记忆上下文（最多 30 条）。"""
    from fuxi_v1.learning import export_memory, compatibility
    from fuxi_v1.data import Snapshot
    from fuxi_v1.policy import calibration_request
    from fuxi_v1.contracts import load_json as f_load, digest

    payload = f_load(snapshot)
    snap = Snapshot(payload["snapshot"])
    horizon = payload["labels"]["label_spec"]["horizon_sessions"]
    index = max(snap.start + 25, min(80, len(snap.sessions) // 2))
    native = calibration_request(snap, calibration_end=snap.sessions[index],
                                 research_start=snap.sessions[index + horizon + 2], horizon=horizon)
    compat = compatibility(payload, native)
    sources = []
    for path in memory_sources:
        try:
            manifest = export_memory(path, compat)
            if not manifest.get("records"):
                continue
            sources.append({"path": path, "manifest_hash": digest(manifest)})
            if len(sources) >= 3:
                break
        except Exception:
            continue
    if not sources:
        return None
    return {"compatibility_hash": compat, "sources": sources}


# ---------------------------------------------------------------- Phase 7
def _mechanism_from_bundle(campaign_dir, bundle_path):
    """从 bundle candidate.json 反查机制标签（pareto_archive 未覆盖时兜底）。"""
    if not bundle_path:
        return None
    # bundle_path 形如 "campaign_xxx\\bundles\\run_xxx"，相对 research/ 目录（campaign_dir 的上一级）
    rel = str(bundle_path).replace("\\", "/")
    base = Path(campaign_dir).parent
    cand = base / rel / "candidate.json"
    if not cand.is_file():
        # 兜底：直接相对 campaign_dir 解析
        cand = Path(campaign_dir) / rel / "candidate.json"
    d = _load_json(cand)
    if not d:
        return None
    return (d.get("hypothesis") or {}).get("mechanism")


def _extract_falsified_motifs(report_path, consumed_paradigms):
    """从研究报告抽取被 E1 驳回候选的已证伪 motif（structural_family + mechanism + archetype）。"""
    if not report_path or not Path(report_path).is_file():
        return []
    report = _load_json(report_path)
    if not report:
        return []
    candidate_results = report.get("candidate_results", [])
    if not isinstance(candidate_results, list):
        return []
    pareto = report.get("pareto_archive", []) or []
    mechanism_by_cid = {}
    for e in pareto:
        if isinstance(e, dict) and e.get("candidate_id"):
            mechanism_by_cid[e["candidate_id"]] = e.get("mechanism")
    campaign_dir = Path(report_path).parent
    motifs = []
    seen = set()
    for c in candidate_results:
        if not isinstance(c, dict):
            continue
        cid = c.get("candidate_id")
        if not cid:
            continue
        # 已证伪判据：E1 明确 reject（eval_status 一并保留，供下游区分「有效但驳回」vs「数据不足」）
        if c.get("e1_decision") != "reject":
            continue
        mechanism = mechanism_by_cid.get(cid) or _mechanism_from_bundle(campaign_dir, c.get("bundle_path"))
        key = (c.get("structural_family") or "", mechanism or "")
        if key in seen:
            continue
        seen.add(key)
        motifs.append({
            "candidate_id": cid,
            "structural_family": c.get("structural_family"),
            "mechanism": mechanism,
            "archetype": _archetype_of_motif(mechanism, consumed_paradigms),
            "e1_decision": c.get("e1_decision"),
            "eval_status": c.get("eval_status"),
            "mean_rank_ic": c.get("mean_rank_ic"),
            "paradigms": sorted(consumed_paradigms) if consumed_paradigms else [],
            "falsified_at": _now(),
        })
    return motifs


def _archetype_of_motif(mechanism, paradigms):
    """把证伪 motif 归并到 archetype（第 3 层：falsified_motifs 聚合到 archetype 层）。

    优先用 campaign 层 mechanism（英文模板名）→ archetype；mechanism 缺失或 unclassified
    时回退用 round 级 paradigm（中文语义）逐条映射，取第一个命中。两者都无则 unclassified。
    """
    from fuxi_v1.archetypes import load_archetypes, mechanism_archetype, archetype_of
    archetypes, unclassified = load_archetypes()
    if mechanism:
        aid = mechanism_archetype(mechanism, archetypes, unclassified)
        if aid != unclassified:
            return aid
    for para in (paradigms or []):
        aid = archetype_of({"paradigm": para, "label": "", "economic_rationale": ""},
                           archetypes, unclassified)
        if aid != unclassified:
            return aid
    return unclassified


def external_search_queries(limit=6):
    """第 3 层：外部知识检索 query 自进化生成器（闭环自进化指引）。

    检索方向由经验动态决定，而非固定关键词：
      1. 欠探索 archetype —— 候选池里尚未测试且未被证伪的 archetype，按未测试候选数
         升序（越稀缺越优先补外部知识）；
      2. 高 ICIR 未验证 mechanism —— icir 高（>=0.3 伏羲验证线）但尚未 tested/落定的
         proposal 所在 archetype，检索外部文献验证其经济机理；
      3. 已证伪 archetype 从检索词自动剔除（避免浪费检索预算在已证伪方向）。

    返回 [{archetype, zh, query, rationale}]，query 为中英双语关键词串，可直接喂给
    arxiv-watcher / 研报库。纯只读（读 proposals + falsified_motifs），不写回。
    """
    from fuxi_v1.archetypes import load_archetypes, archetype_of
    proposals = (_load_json(PROPOSALS) or {}).get("proposals", [])
    book = _load_falsified_motifs()
    archetypes, unclassified = load_archetypes()
    zh = {a["id"]: a["zh"] for a in archetypes}

    # 已证伪 archetype（从 falsified_motifs 的 archetype 字段聚合）
    falsified_arch = set()
    for m in book.get("motifs", []):
        aid = m.get("archetype")
        if aid and aid != unclassified:
            falsified_arch.add(aid)

    untested_count = {}
    high_icir = {}
    for p in proposals:
        aid = archetype_of(p, archetypes, unclassified)
        if _tested_flag(p):
            continue
        untested_count[aid] = untested_count.get(aid, 0) + 1
        icir = p.get("icir") or 0
        if icir >= 0.3:
            high_icir.setdefault(aid, []).append((icir, p.get("label") or p.get("paradigm") or ""))

    queries = []
    # 方向 1：欠探索 archetype（未测试候选少 + 未证伪），按候选数升序
    underexplored = sorted(
        [(a, c) for a, c in untested_count.items() if a not in falsified_arch and a != unclassified],
        key=lambda t: (t[1], t[0]))
    for aid, cnt in underexplored:
        if len(queries) >= limit:
            break
        queries.append({
            "archetype": aid, "zh": zh.get(aid, aid),
            "query": f"{zh.get(aid, aid)} 因子构建 A股 {aid} alpha factor construction",
            "rationale": f"欠探索 archetype（未测试候选 {cnt} 条，未证伪），补外部知识生成新假设",
        })
    # 方向 2：高 ICIR 未验证 mechanism（实证锚点强但未落地），按最高 ICIR 降序
    ranked = sorted(high_icir.items(), key=lambda kv: -max(x[0] for x in kv[1]))
    for aid, items in ranked:
        if len(queries) >= limit:
            break
        if aid in falsified_arch or aid == unclassified:
            continue
        if any(q["archetype"] == aid for q in queries):
            continue
        top = max(items, key=lambda x: x[0])
        queries.append({
            "archetype": aid, "zh": zh.get(aid, aid),
            "query": f"{zh.get(aid, aid)} 因子 {aid} 经济机理 ICIR 稳健性 empirical validation",
            "rationale": f"高 ICIR({top[0]:.3f}) 未验证机制「{top[1][:30]}」，检索外部文献验证机理",
        })
    return queries


def _writeback_falsified_archetypes(falsified_archetypes):
    """把本轮的证伪 archetype 软标注回候选池（falsified_archetype_at，非阻塞）。

    只给「属于证伪 archetype 且尚无标注」的 proposal 打时间戳；不碰 status/tested_at，
    因此不影响 _tested_flag 的轮转判定，select_hypotheses 的降权仍以 falsified_motifs
    为准（本标注是候选池侧的持久化可观测信号）。CRLF 通道写出，幂等。
    """
    if not falsified_archetypes:
        return 0
    from fuxi_v1.archetypes import load_archetypes, archetype_of
    data = _load_json(PROPOSALS)
    if not data:
        return 0
    proposals = data.get("proposals", [])
    archetypes, unclassified = load_archetypes()
    stamp = _now()
    updated = 0
    for p in proposals:
        if p.get("falsified_archetype_at"):
            continue
        if archetype_of(p, archetypes, unclassified) in falsified_archetypes:
            p["falsified_archetype_at"] = stamp
            updated += 1
    if updated:
        data["proposals"] = proposals
        _write_json_crlf(PROPOSALS, data)
    return updated


def reflect(answer, hypotheses, consumed):
    """Phase 7：写回本轮结果 + 产出「已证伪 motif 清单」+ 外部检索 query 指引。

    从研究报告抽取被 E1 驳回候选的 structural_family/mechanism/archetype，追加进
    FALSIFIED_MOTIFS 台账（聚合到 archetype 层），供下一轮 select_hypotheses 对同
    archetype 软降权；并把本轮证伪 archetype 软标注回候选池。同时生成外部知识检索
    query（欠探索 archetype + 高 ICIR 未验证 mechanism，剔除已证伪），闭环自进化指引。
    """
    research = answer.get("research") or {}
    report_path = research.get("report_path")

    consumed_paradigms = set()
    data = _load_json(PROPOSALS)
    if data and consumed:
        by_name = {p.get("factor_name"): p.get("paradigm") for p in data.get("proposals", [])}
        for fn in consumed:
            para = by_name.get(fn)
            if para:
                consumed_paradigms.add(para.strip())

    falsified = _extract_falsified_motifs(report_path, consumed_paradigms)
    motifs_added = _append_falsified_motifs(falsified) if answer.get("state") == "completed" else 0

    # 第 3 层：证伪 archetype 写回候选池（软标注）+ 外部检索 query 自进化生成
    falsified_archetypes = sorted({m["archetype"] for m in falsified
                                   if m.get("archetype") and m["archetype"] != "unclassified"})
    written_back = _writeback_falsified_archetypes(falsified_archetypes) if answer.get("state") == "completed" else 0
    search_queries = external_search_queries()

    return {
        "cycle_at": _now(),
        "state": answer.get("state"),
        "run_id": answer.get("run_id"),
        "cached": answer.get("cached"),
        "hypotheses_fed": len(hypotheses),
        "consumed_marked": len(consumed),
        "attempted_candidates": research.get("attempted_candidates"),
        "valid_descriptive_factors": research.get("valid_descriptive_factors"),
        "engineering_errors": research.get("engineering_errors"),
        "jq_candidate_count": research.get("jq_candidate_count"),
        "economic_reward_count": research.get("economic_reward_count"),
        "falsified_motifs_found": len(falsified),
        "falsified_motifs_added": motifs_added,
        "falsified_archetypes": falsified_archetypes,
        "falsified_archetypes_written_back": written_back,
        "external_search_queries": search_queries,
        "exit_code": answer.get("exit_code"),
    }


# ---------------------------------------------------------------- 主入口
def run_cycle(*, refresh_data=False, snapshot=None, rounds=20, candidates=4, llm=True, dry_run=False,
              limits=None):
    started = time.monotonic()
    phases = {}

    # Phase 1+4: 深扫候选 → 假设证据（消费 tested_at + falsified_motifs 经验）
    hypotheses, consumed = select_hypotheses()
    phases["source_scan"] = {"hypotheses_count": len(hypotheses), "consumed": len(consumed)}

    # Phase 2: 数据采集
    phases["data_refresh"] = data_refresh(refresh_data)

    # Phase 3: 快照定位
    snap = locate_snapshot(snapshot)
    phases["snapshot"] = {"path": snap}

    # Phase 3b: 记忆源
    memory_sources = locate_memory_sources()
    phases["memory_sources"] = memory_sources

    # Phase 5-6: 研究 + 反馈
    if dry_run:
        answer = {"state": "dry_run", "reason": "dry-run requested"}
    else:
        answer = run_research(snap, hypotheses, memory_sources, rounds=rounds, candidates=candidates,
                              llm=llm, limits=limits)
    phases["research"] = {k: answer.get(k) for k in ("state", "run_id", "cached", "reason", "exit_code") if k in answer}

    # 断点①：研究完成后回写 tested_at（闭环轮转；status 已落定者不覆盖）
    marked = mark_tested(consumed, datetime.now().strftime("%Y%m%d")) \
        if answer.get("state") == "completed" else 0
    phases["tested_at_writeback"] = {"marked": marked}

    # Phase 7: 反思反馈（断点③：产出已证伪 motif 清单）
    reflection = reflect(answer, hypotheses, consumed)
    phases["reflect"] = reflection

    cycle = {"schema_version": "fuxi.cycle.1", "started_at": _now(),
             "elapsed_seconds": round(time.monotonic() - started, 2), "phases": phases}
    _append_ledger(cycle)
    return cycle


def main():
    parser = argparse.ArgumentParser(description="伏羲 v2.0 闭环编排器")
    parser.add_argument("action", choices=("run", "status", "preview"))
    parser.add_argument("--refresh-data", action="store_true", help="执行 stage0 增量采集")
    parser.add_argument("--snapshot", help="显式指定冻结快照路径")
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--candidates", type=int, default=4)
    parser.add_argument("--no-llm", action="store_true", help="关闭 LLM 假设生成")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-memory-mb", type=int, default=8192,
                        help="worker 内存上限 MiB（manual 身份可抬升，默认 8192 兼容 1125 截面）")
    args = parser.parse_args()

    if args.action == "status":
        ledger = _load_json(LEDGER, {"cycles": []})
        print(json.dumps({"cycles_run": len(ledger.get("cycles", [])),
                          "last": ledger["cycles"][-1] if ledger.get("cycles") else None},
                         ensure_ascii=False, indent=2))
        return 0

    if args.action == "preview":
        hyps, consumed = select_hypotheses()
        book = _load_falsified_motifs()
        print(json.dumps({"hypotheses_count": len(hyps),
                          "consumed": len(consumed),
                          "falsified_motifs": len(book.get("motifs", [])),
                          "snapshot": locate_snapshot(args.snapshot),
                          "memory_sources": locate_memory_sources(),
                          "sample_hypotheses": hyps[:3]}, ensure_ascii=False, indent=2))
        return 0

    limits = dict(DAILY_LIMITS)
    if args.max_memory_mb and args.max_memory_mb != DAILY_LIMITS["max_memory_mb"]:
        limits["max_memory_mb"] = args.max_memory_mb
    cycle = run_cycle(refresh_data=args.refresh_data, snapshot=args.snapshot,
                      rounds=args.rounds, candidates=args.candidates,
                      llm=not args.no_llm, dry_run=args.dry_run, limits=limits)
    print(json.dumps(cycle, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(json.dumps({"state": "error", "type": type(exc).__name__, "reason": str(exc)},
                         ensure_ascii=False), file=sys.stderr)
        raise SystemExit(2)
