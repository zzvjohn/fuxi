"""系统慢循环观测沉淀 + 假设提炼（阶段 A/B，纯规则非 LLM）。

observe —— 从快循环 cycle_ledger 聚合系统级观测快照（零副作用，只写 observations/）。
draft   —— 规则提炼候选假设（dry-run，不落盘、不 freeze）。

观测快照只读、不进 freeze、不回流本代（对齐 BLUEPRINT 第 8 节
"验证与正在冻结的未来收益不得回流本代"）。
"""
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OBS_DIR = ROOT / "data" / "system_evolution" / "observations"
OBS_SCHEMA = "fuxi.system_evolution.1.observation"
LEDGER = ROOT / "data" / "fuxi_unified" / "cycle_ledger.json"
MIN_FULL_CYCLE_ATTEMPTS = 20  # 完整日研究下限（正常日 77-80 候选；试跑/refused 远小于此）


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


def _sha1_short(text, n=4):
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:n]


def observe(workspace=ROOT, limit=None):
    """从 cycle_ledger 聚合系统级观测快照，落盘 observations/（只读，不进 freeze）。"""
    ledger = _load_json(workspace / "data" / "fuxi_unified" / "cycle_ledger.json", {"cycles": []})
    cycles = ledger.get("cycles", [])
    if limit:
        cycles = cycles[-limit:]
    snapshots = []
    for cyc in cycles:
        reflect = (cyc.get("phases") or {}).get("reflect") or {}
        attempted = reflect.get("attempted_candidates") or 0
        valid = reflect.get("valid_descriptive_factors") or 0
        errors = reflect.get("engineering_errors") or 0
        jq = reflect.get("jq_candidate_count") or 0
        econ = reflect.get("economic_reward_count") or 0
        falsified = reflect.get("falsified_motifs_found") or 0
        snapshots.append({
            "schema_version": OBS_SCHEMA,
            "observed_at": _now(),
            "cycle_started_at": cyc.get("started_at"),
            "state": reflect.get("state"),
            "run_id": reflect.get("run_id"),
            "counts": {
                "attempted_candidates": attempted,
                "valid_descriptive_factors": valid,
                "engineering_errors": errors,
                "jq_candidate_count": jq,
                "economic_reward_count": econ,
                "falsified_motifs_found": falsified,
            },
            "signals": {
                "error_rate": (errors / attempted) if attempted else 0.0,
                "valid_rate": (valid / attempted) if attempted else 0.0,
                "jq_rate": (jq / attempted) if attempted else 0.0,
                "falsified_rate": (falsified / attempted) if attempted else 0.0,
            },
        })
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = OBS_DIR / f"observe_{stamp}.json"
    _write_json(out, {"schema_version": OBS_SCHEMA + ".batch", "generated_at": _now(),
                      "cycle_count": len(snapshots), "snapshots": snapshots})
    return {"path": str(out), "cycle_count": len(snapshots)}


# 提炼规则（非 LLM，跨轮聚合观测）。每条规则产出候选假设的骨架字段；
# assets（baseline/incumbent/challenger 代码 + suite + aa_noise）须由 freeze 前补齐真实文件。
DRAFT_RULES = [
    {
        "kind": "correctness",
        "component": "candidate_eval",
        "test": lambda sigs: any(s["error_rate"] > 0 for s in sigs),
        "hypothesis": "修复连续轮次候选求值 engineering_error 的根因（同字段/依赖聚类）",
        "falsification": "修复后连续 N 轮 error_rate 不降反升，或无 error 候选恢复",
    },
    {
        "kind": "research_method",
        "component": "jq_gate",
        "test": lambda sigs: (sum(s["valid_rate"] for s in sigs) / max(1, len(sigs))) > 0.8
        and all(s["jq_rate"] == 0 for s in sigs),
        "hypothesis": "降低 JQ 资格门槛或调整 gate 通过判据以提高真实 JQ 候选产出",
        "falsification": "门槛调整后 jq_rate 无提升，或通过率被门槛本身制造（非真实 alpha）",
    },
    # efficiency / economic_logic 规则留占位：需先补 wall_seconds / 账户 drift 观测字段
    # （reflect 目前只存 attempted/valid/errors/jq/econ/falsified），M1 后按需启用。
]


def draft(workspace=ROOT, limit=None):
    """规则提炼候选假设（dry-run，不落盘、不 freeze）。"""
    obs_root = workspace / "data" / "system_evolution" / "observations"
    batch_path = sorted(obs_root.glob("observe_*.json"))[-1] if obs_root.exists() else None
    if not batch_path:
        return {"drafts": [], "note": "no observations yet; run observe first"}
    batch = _load_json(batch_path, {})
    snapshots = batch.get("snapshots", [])
    if limit:
        snapshots = snapshots[-limit:]
    # 只聚合完整研究轮次（attempted ≥ 下限；refused/试跑轮次 attempted 过小会稀释信号）
    sigs = [s.get("signals", {}) for s in snapshots
            if (s.get("counts", {}).get("attempted_candidates") or 0) >= MIN_FULL_CYCLE_ATTEMPTS]
    drafts = []
    for rule in DRAFT_RULES:
        try:
            if not rule["test"](sigs):
                continue
        except Exception:
            continue
        kind = rule["kind"]
        change_id = f"sys-{datetime.now().strftime('%Y%m%d')}-{kind}-{_sha1_short(rule['hypothesis'])}"
        drafts.append({
            "change_id": change_id,
            "kind": kind,
            "component": rule["component"],
            "hypothesis": rule["hypothesis"],
            "falsification": rule["falsification"],
            "assets_pending": ["baseline_code", "incumbent_code", "challenger_code",
                               "suite", "judge", "data", "labels", "cost", "aa_noise"],
            "note": "assets 须在 freeze 前指向真实工作区文件（含三版本代码 + 比较套件 + A/A 噪声）",
        })
    return {"drafts": drafts, "source_observation": str(batch_path)}
