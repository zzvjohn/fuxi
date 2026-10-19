"""Evidence gate shared by legacy v0.7 and native v1 research engines.

This gate compares system changes. It cannot approve a factor, change a default,
or activate a release. Source manifests and comparison policy are frozen before
the challenger observations are read.
"""

from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import json
import math
import os
import re
import shutil
import uuid

from fuxi_v1.contracts import canonical, digest, load_json, require
from fuxi_v1.governance import governed_writer


SCHEMA = "fuxi.system_evolution.1"
ENGINES = frozenset(("legacy_v07", "native_v10"))
KINDS = frozenset(("correctness", "efficiency", "research_method", "economic_logic"))
COHORTS = frozenset(("replay_exposed", "transfer_exposed", "declared_forward"))
STATES = frozenset(("valid", "not_evaluated", "data_insufficient", "error", "precheck_rejected"))
RESOURCES = ("wall_seconds", "memory_mb", "external_requests", "fee_cny")
ASSETS = ("baseline_code", "incumbent_code", "challenger_code", "suite", "judge", "data", "labels", "cost", "aa_noise")
PLAN_FIELDS = frozenset(("schema_version", "change_id", "title", "kind", "engines", "component", "hypothesis", "falsification", "assets", "thresholds", "budgets", "required_cohorts"))
THRESHOLD_FIELDS = frozenset(("noise_floor", "minimum_mean_gain", "maximum_case_loss", "quality_noninferiority_margin", "minimum_wall_saving"))


def _number(value, name, *, positive=False):
    require(type(value) in (int, float) and math.isfinite(value), name + " must be finite")
    require(value > 0 if positive else value >= 0, name + " must be nonnegative")
    return float(value)


def _finite(value, name):
    require(type(value) in (int, float) and math.isfinite(value), name + " must be finite")
    return float(value)


def _file_hash(path):
    h = sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def _workspace_file(workspace, raw):
    if isinstance(raw, Path):
        raw = str(raw)
    require(isinstance(raw, str) and raw, "asset path required")
    path = (Path(workspace) / raw).resolve()
    require(path.is_relative_to(Path(workspace).resolve()) and path.is_file(),
            "asset must be an existing workspace file")
    return path


def _atomic_once(path, value):
    content = canonical(value) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        require(path.read_bytes() == content, "immutable system-evolution artifact changed")
        return
    pending = path.with_name(path.name + ".pending_" + uuid.uuid4().hex)
    try:
        with pending.open("xb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        # The shared writer lease serializes participating writers. An unrelated
        # editor is outside this application-level guard and remains auditable.
        require(not path.exists(), "system-evolution artifact appeared concurrently")
        os.replace(pending, path)
    finally:
        if pending.exists():
            pending.unlink()


def _directory(workspace, change_id):
    require(isinstance(change_id, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", change_id),
            "invalid system change identity")
    return Path(workspace).resolve() / "data" / "system_evolution" / change_id


def _validate_plan(plan):
    require(isinstance(plan, dict) and set(plan) == PLAN_FIELDS, "system change plan fields differ")
    require(plan["schema_version"] == SCHEMA, "system change schema differs")
    _directory(".", plan["change_id"])
    require(all(isinstance(plan[key], str) and plan[key].strip()
                for key in ("title", "component", "hypothesis", "falsification")),
            "attributable component and falsifiable claim required")
    require(plan["kind"] in KINDS, "unknown system change kind")
    engines = plan["engines"]
    require(isinstance(engines, list) and engines and len(set(engines)) == len(engines)
            and set(engines) <= ENGINES, "explicit legacy/native engine scope required")
    require(isinstance(plan["assets"], dict) and set(plan["assets"]) == set(ASSETS),
            "frozen code, judge, data, label, cost, suite and A/A assets required")
    require(isinstance(plan["thresholds"], dict) and set(plan["thresholds"]) == THRESHOLD_FIELDS,
            "predeclared system improvement thresholds required")
    for key, value in plan["thresholds"].items():
        _number(value, key)
    require(isinstance(plan["budgets"], dict) and set(plan["budgets"]) == set(RESOURCES),
            "complete resource limits required")
    for key, value in plan["budgets"].items():
        _number(value, key, positive=key in ("wall_seconds", "memory_mb"))
    cohorts = plan["required_cohorts"]
    require(isinstance(cohorts, list) and cohorts and len(set(cohorts)) == len(cohorts)
            and set(cohorts) <= COHORTS, "explicit cohort requirement needed")
    if plan["kind"] == "research_method":
        require(bool(set(cohorts) & {"transfer_exposed", "declared_forward"}),
                "method change needs a transfer cohort")
    if plan["kind"] == "economic_logic":
        require("declared_forward" in cohorts, "economic claim needs declared forward observations")


def _suite_tasks(suite, engines):
    require(isinstance(suite, dict) and set(suite) == {"schema_version", "tasks"}
            and suite["schema_version"] == SCHEMA + ".suite", "suite schema differs")
    tasks = suite["tasks"]
    require(isinstance(tasks, list) and tasks, "comparison suite is empty")
    result = {}
    for task in tasks:
        require(isinstance(task, dict) and set(task) == {"engine", "task_id", "cohort", "support_hash", "window"},
                "suite task fields differ")
        require(task["engine"] in engines and task["cohort"] in COHORTS
                and all(isinstance(task[k], str) and task[k] for k in ("task_id", "support_hash", "window")),
                "invalid suite task")
        key = (task["engine"], task["task_id"])
        require(key not in result, "duplicate suite task")
        result[key] = task
    require({engine for engine, _ in result} == set(engines), "engine has no comparison tasks")
    return result


def _aa_band(noise, tasks):
    require(isinstance(noise, dict) and set(noise) == {"schema_version", "pairs"}
            and noise["schema_version"] == SCHEMA + ".aa", "A/A schema differs")
    pairs = noise["pairs"]
    require(isinstance(pairs, list) and len(pairs) >= 2, "real baseline repetitions required")
    keys = set()
    changes = []
    for row in pairs:
        require(isinstance(row, dict) and set(row) == {"engine", "task_id", "repeat_id", "first", "second"},
                "A/A pair fields differ")
        key = (row["engine"], row["task_id"], row["repeat_id"])
        require(key[:2] in tasks and key not in keys, "A/A task/repeat identity differs")
        require(isinstance(row["repeat_id"], str) and row["repeat_id"], "A/A repeat id required")
        keys.add(key)
        changes.append(abs(_finite(row["first"], "A/A first") - _finite(row["second"], "A/A second")))
    return max(changes)


def freeze(workspace, plan_path):
    """Freeze one component-level hypothesis before seeing challenger results."""
    workspace = Path(workspace).resolve()
    with governed_writer(workspace, "report", trigger="manual"):
        plan = load_json(_workspace_file(workspace, plan_path))
        _validate_plan(plan)
        directory = _directory(workspace, plan["change_id"])
        require(not directory.exists(), "change identity already frozen")
        directory.parent.mkdir(parents=True, exist_ok=True)
        staging = directory.parent / (".pending_" + uuid.uuid4().hex)
        staging.mkdir(exist_ok=False)
        try:
            assets = {}
            for name in ASSETS:
                source = _workspace_file(workspace, plan["assets"][name])
                require(source.stat().st_size <= 8 * 1024 * 1024,
                        "freeze a compact source manifest, not a raw data/cache file")
                expected = _file_hash(source)
                sealed = staging / "assets" / (name + ".source")
                sealed.parent.mkdir(exist_ok=True)
                with source.open("rb") as old, sealed.open("xb") as new:
                    shutil.copyfileobj(old, new)
                    new.flush()
                    os.fsync(new.fileno())
                require(_file_hash(sealed) == expected and _file_hash(source) == expected,
                        "source changed while sealing system-change protocol")
                assets[name] = {"path": str(source), "sealed_path": str(directory / "assets" / (name + ".source")),
                                "sha256": expected}
            require(assets["baseline_code"]["sha256"] != assets["challenger_code"]["sha256"]
                    and assets["incumbent_code"]["sha256"] != assets["challenger_code"]["sha256"],
                    "challenger is identical to the anchor or incumbent")
            suite = load_json(staging / "assets" / "suite.source")
            tasks = _suite_tasks(suite, plan["engines"])
            band = _aa_band(load_json(staging / "assets" / "aa_noise.source"), tasks)
            require(math.isclose(plan["thresholds"]["noise_floor"], band,
                                 rel_tol=1e-9, abs_tol=1e-12),
                    "noise floor must equal the frozen A/A reading")
            frozen = {"schema_version": SCHEMA + ".frozen", "plan": plan, "assets": assets,
                      "suite_hash": digest(suite), "task_count": len(tasks),
                      "aa_max_absolute_delta": band,
                      "frozen_at": datetime.now(timezone.utc).isoformat(),
                      "activation_allowed": False, "factor_qualification_transfer_allowed": False}
            frozen["receipt_hash"] = digest(frozen)
            _atomic_once(staging / "FROZEN.json", frozen)
            require(not directory.exists(), "change identity appeared concurrently")
            os.replace(staging, directory)
            return frozen
        finally:
            if staging.exists():
                require(staging.resolve().is_relative_to(directory.parent.resolve()),
                        "unsafe staging cleanup target")
                shutil.rmtree(staging)


def _verify_frozen(workspace, change_id, *, strict_live=False):
    directory = _directory(workspace, change_id)
    frozen = load_json(directory / "FROZEN.json")
    receipt_hash = frozen.pop("receipt_hash")
    require(digest(frozen) == receipt_hash and frozen["schema_version"] == SCHEMA + ".frozen",
            "frozen system change receipt drift")
    frozen["receipt_hash"] = receipt_hash
    _validate_plan(frozen["plan"])
    require(frozen["plan"]["change_id"] == change_id, "change identity drift")
    for name in ASSETS:
        item = frozen["assets"][name]
        sealed = _workspace_file(workspace, item["sealed_path"])
        require(sealed.is_relative_to(directory / "assets") and _file_hash(sealed) == item["sha256"],
                "frozen " + name + " sealed source drift")
        if strict_live:
            live = _workspace_file(workspace, item["path"])
            require(_file_hash(live) == item["sha256"], "frozen " + name + " source drift")
    suite = load_json(frozen["assets"]["suite"]["sealed_path"])
    tasks = _suite_tasks(suite, frozen["plan"]["engines"])
    require(len(tasks) == frozen["task_count"] and
            digest(suite) == frozen["suite_hash"],
            "frozen comparison suite drift")
    return directory, frozen, tasks


def _observations(workspace, path, role, frozen, tasks):
    document = load_json(_workspace_file(workspace, path))
    require(isinstance(document, dict) and set(document) == {"schema_version", "role", "suite_hash", "judge_sha256", "data_sha256", "labels_sha256", "cost_sha256", "cases"},
            "system observation fields differ")
    require(document["schema_version"] == SCHEMA + ".observations" and document["role"] == role
            and document["suite_hash"] == frozen["suite_hash"], "observation role/suite drift")
    for field, asset in (("judge_sha256", "judge"), ("data_sha256", "data"),
                         ("labels_sha256", "labels"), ("cost_sha256", "cost")):
        require(document[field] == frozen["assets"][asset]["sha256"], field + " drift")
    cases = document["cases"]
    require(isinstance(cases, list), "observation cases required")
    result = {}
    for row in cases:
        require(isinstance(row, dict) and set(row) == {"engine", "task_id", "support_hash", "eval_status", "score", "usage", "source_fact"},
                "observation case fields differ")
        key = (row["engine"], row["task_id"])
        require(key in tasks and key not in result and row["support_hash"] == tasks[key]["support_hash"],
                "paired task/support differs")
        source = row["source_fact"]
        require(row["eval_status"] in STATES and isinstance(source, dict)
                and set(source) == {"path", "sha256"},
                "explicit source and five-state evaluation required")
        source_path = _workspace_file(workspace, source["path"])
        require(source_path != Path(path).resolve() and _file_hash(source_path) == source["sha256"],
                "observation source fact drift")
        if row["eval_status"] == "valid":
            require(type(row["score"]) in (int, float) and math.isfinite(row["score"]),
                    "valid case needs a real score")
        else:
            require(row["score"] is None, "missing/error score cannot be zero-filled")
        usage = row["usage"]
        require(isinstance(usage, dict) and set(usage) == set(RESOURCES), "complete resource usage required")
        for metric in RESOURCES:
            if usage[metric] is not None:
                _number(usage[metric], metric)
        result[key] = row
    require(set(result) == set(tasks), "observation task set incomplete")
    return result


def _decision(frozen, tasks, anchor, incumbent, challenger):
    plan = frozen["plan"]
    limits = plan["budgets"]
    for role, values in (("anchor", anchor), ("incumbent", incumbent), ("challenger", challenger)):
        for key, row in values.items():
            for metric in RESOURCES:
                actual = row["usage"][metric]
                if actual is None:
                    return "data_insufficient", "unknown_" + metric + "_in_" + role, {}
                if actual > limits[metric]:
                    return "precheck_rejected", "budget_exceeded_in_" + role, {"task": key}
    if any(row["eval_status"] == "error" for row in challenger.values()):
        return "error", "challenger_execution_error", {}
    if any(row["eval_status"] != "valid" for values in (anchor, incumbent, challenger)
           for row in values.values()):
        return "data_insufficient", "nonvalid_pair_no_imputed_score", {}
    thresholds = plan["thresholds"]
    band = thresholds["noise_floor"]
    case_deltas = {key: challenger[key]["score"] - incumbent[key]["score"] for key in tasks}
    if min(case_deltas.values()) < -thresholds["maximum_case_loss"]:
        return "valid", "material_case_regression", {"worst_case_delta": min(case_deltas.values())}
    groups = {}
    for engine in plan["engines"]:
        for cohort in plan["required_cohorts"]:
            keys = [key for key, task in tasks.items() if key[0] == engine and task["cohort"] == cohort]
            if not keys:
                return "data_insufficient", "empty_required_engine_cohort", {"engine": engine, "cohort": cohort}
            delta = sum(case_deltas[key] for key in keys) / len(keys)
            challenger_score = sum(challenger[key]["score"] for key in keys) / len(keys)
            anchor_score = sum(anchor[key]["score"] for key in keys) / len(keys)
            incumbent_score = sum(incumbent[key]["score"] for key in keys) / len(keys)
            groups[engine + ":" + cohort] = {"n_unique_tasks": len(keys), "mean_delta": delta,
                                            "mean_challenger": challenger_score,
                                            "mean_incumbent": incumbent_score, "mean_release_anchor": anchor_score}
            if challenger_score < max(anchor_score, incumbent_score) - band:
                return "valid", "below_noise_adjusted_release_floor", {"groups": groups}
            if (plan["kind"] in ("research_method", "economic_logic")
                    and delta <= max(band, thresholds["minimum_mean_gain"])):
                return "valid", "no_transfer_gain_beyond_noise", {"groups": groups}
            if plan["kind"] == "efficiency" and delta < -thresholds["quality_noninferiority_margin"]:
                return "valid", "quality_noninferiority_not_met", {"groups": groups}
    before_wall = sum(row["usage"]["wall_seconds"] for row in incumbent.values())
    after_wall = sum(row["usage"]["wall_seconds"] for row in challenger.values())
    # case_deltas 用 "engine:task_id" 字符串键（tuple 不可 JSON 序列化），供转正评分卡 P4/P5 门槛
    # （方向一致性 / jackknife 稳健性）在 promote 引擎重算。纯只读追加，不改变裁决语义。
    case_delta_map = {f"{key[0]}:{key[1]}": delta for key, delta in case_deltas.items()}
    effects = {"groups": groups, "n_unique_tasks": len(tasks), "worst_case_delta": min(case_deltas.values()),
               "case_deltas": case_delta_map,
               "incumbent_wall_seconds": before_wall, "challenger_wall_seconds": after_wall,
               "wall_seconds_saved": before_wall - after_wall}
    if plan["kind"] == "correctness" and not any(delta > band for delta in case_deltas.values()):
        return "valid", "no_identified_correctness_fix", effects
    if plan["kind"] == "efficiency" and before_wall - after_wall <= thresholds["minimum_wall_saving"]:
        return "valid", "no_identified_efficiency_gain", effects
    return "valid", "eligible_for_independent_review", effects


def assess(workspace, change_id, anchor_path, incumbent_path, challenger_path):
    """Assess once under the common writer lease; never activate the challenger."""
    workspace = Path(workspace).resolve()
    with governed_writer(workspace, "report", trigger="manual"):
        directory, frozen, tasks = _verify_frozen(workspace, change_id, strict_live=True)
        paths = {"anchor": str(_workspace_file(workspace, anchor_path)),
                 "incumbent": str(_workspace_file(workspace, incumbent_path)),
                 "challenger": str(_workspace_file(workspace, challenger_path))}
        source_hashes = {role: _file_hash(path) for role, path in paths.items()}
        result = {role: _observations(workspace, path, role, frozen, tasks) for role, path in paths.items()}
        state, reason, effects = _decision(frozen, tasks, result["anchor"], result["incumbent"], result["challenger"])
        facts = {}
        for values in result.values():
            for row in values.values():
                source = row["source_fact"]
                previous = facts.setdefault(source["path"], source["sha256"])
                require(previous == source["sha256"], "source fact changed during comparison")
        for source, expected in facts.items():
            require(_file_hash(_workspace_file(workspace, source)) == expected,
                    "source fact changed before assessment publication")
        decision = ("review_shadow" if state == "valid" and reason == "eligible_for_independent_review"
                    else "retain_incumbent" if state == "valid" else "await_evidence")
        status_counts = {role: {status: sum(row["eval_status"] == status for row in values.values())
                                for status in sorted(STATES)}
                         for role, values in result.items()}
        receipt = {"schema_version": SCHEMA + ".assessment", "change_id": change_id,
                   "kind": frozen["plan"]["kind"], "component": frozen["plan"]["component"],
                   "hypothesis": frozen["plan"]["hypothesis"],
                   "falsification": frozen["plan"]["falsification"],
                   "engines": frozen["plan"]["engines"],
                   "frozen_receipt_hash": frozen["receipt_hash"], "source_hashes": source_hashes,
                   "source_paths": paths, "source_facts": facts,
                   "eval_status": state, "decision": decision, "reason_code": reason,
                   "effects": effects, "compared_tasks": len(tasks), "status_counts": status_counts,
                   "review_eligible": state == "valid" and reason == "eligible_for_independent_review",
                   "production_activation_allowed": False, "factor_qualification_transfer_allowed": False,
                   "economic_winner_confirmed": False}
        receipt["receipt_hash"] = digest(receipt)
        _atomic_once(directory / "ASSESSMENT.json", receipt)
        return receipt


def inspect(workspace, change_id):
    """Validate frozen sources and the optional completed receipt on every read."""
    directory, frozen, _ = _verify_frozen(Path(workspace).resolve(), change_id)
    path = directory / "ASSESSMENT.json"
    if not path.exists():
        return {"change_id": change_id, "state": "frozen_not_evaluated",
                "frozen_receipt_hash": frozen["receipt_hash"], "production_activation_allowed": False}
    receipt = load_json(path)
    receipt_hash = receipt.pop("receipt_hash")
    require(receipt_hash == digest(receipt) and receipt["frozen_receipt_hash"] == frozen["receipt_hash"],
            "system assessment receipt drift")
    for role, source in receipt["source_paths"].items():
        require(_file_hash(_workspace_file(workspace, source)) == receipt["source_hashes"][role],
                "system assessment source drift")
    for source, expected in receipt["source_facts"].items():
        require(_file_hash(_workspace_file(workspace, source)) == expected,
                "system assessment fact source drift")
    receipt["receipt_hash"] = receipt_hash
    return receipt


def history(workspace):
    """Return checked positive, negative and unresolved system learning records."""
    root = Path(workspace).resolve() / "data" / "system_evolution"
    if not root.exists():
        return []
    # 只列出含 FROZEN.json 的 change 目录，跳过 observations 等非档案子目录。
    return [inspect(workspace, p.name) for p in sorted(root.iterdir())
            if p.is_dir() and not p.name.startswith(".pending_") and (p / "FROZEN.json").is_file()]
