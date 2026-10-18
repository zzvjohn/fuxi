# -*- coding: utf-8 -*-
"""Fixed-seed, real-data Forge objective A/B for the R3 rollout gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
from datetime import datetime
from pathlib import Path


HERE = Path(__file__).resolve()
FA_ROOT = HERE.parents[1]
PROJECT_ROOT = HERE.parents[3]
sys.path.insert(0, str(FA_ROOT))

from forge import FactorForge
from forge.expression import parse_expression
from forge.paradigm_profiles import MATURE_PARADIGM_PROFILES
from ralph_loop import RalphLoop


def load_real_data():
    loader = object.__new__(RalphLoop)
    loader.data_dir = PROJECT_ROOT / "data"
    data = loader._load_forge_data()
    if not data or "fwd_return" not in data:
        raise RuntimeError("Forge real weekly panel or forward return is unavailable")
    return data


def file_fingerprint(path: Path):
    stat = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        digest.update(handle.read(1024 * 1024))
    return {
        "path": str(path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "first_mib_sha256": digest.hexdigest(),
    }


def run_one(data, *, mode: str, seed: int, paradigm: str, pop_size: int, generations: int):
    forge = FactorForge(
        data=data,
        max_depth=6,
        max_complexity=12.0,
        icir_threshold=0.1,
        forward_returns=data["fwd_return"],
        seed=seed,
        paradigm_profiles=MATURE_PARADIGM_PROFILES,
        fitness_mode=mode,
        dim_prune="shadow",
    )
    best = forge.evolve(
        n_generations=generations,
        pop_size=pop_size,
        tournament_size=3,
        elite_count=2,
        mutation_rate=0.3,
        crossover_rate=0.5,
        workers=1,
        verbose=False,
        top_k=2,
        icir_top_n=30,
        target_paradigm=paradigm,
    )
    leakage = sum(1 for row in best if row.get("paradigm") != paradigm)
    leakage += sum(
        1 for row in forge.final_population if row.get("paradigm") != paradigm
    )
    if leakage:
        raise AssertionError(f"target paradigm leakage: {leakage}")
    if not best:
        raise AssertionError("Forge returned no recorded candidates")
    selected = max(best, key=lambda row: float(row["fitness"]))
    tree = parse_expression(selected["expression"])
    if tree is None:
        raise AssertionError("best expression cannot be parsed")
    values = forge.evaluator.evaluate(tree)
    diag = forge._fitness_components_from_values(
        values,
        complexity=float(selected["complexity"]),
    )
    return {
        "mode": mode,
        "seed": seed,
        "target_paradigm": paradigm,
        "target_leakage": leakage,
        "expression": selected["expression"],
        "pandas_expression": selected["pandas_expression"],
        "generation": selected["generation"],
        "complexity": selected["complexity"],
        "search_fitness": selected["fitness"],
        "icir": diag["icir"],
        "abs_icir": abs(diag["icir"]),
        "sign_consistency": diag["sign_consistency"],
        "subperiod_stability": diag["subperiod_stability"],
        "rank_turnover_proxy": diag["rank_turnover_proxy"],
        "challenger_score": (
            abs(diag["icir"])
            * (0.25 + 0.50 * diag["sign_consistency"] + 0.25 * diag["subperiod_stability"])
            - 0.02 * float(selected["complexity"])
            - 0.20 * diag["rank_turnover_proxy"]
        ),
        "n_recorded": len(best),
        "n_final_population": len(forge.final_population),
    }


def median(rows, key):
    return statistics.median(float(row[key]) for row in rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="17,41,73,101,137")
    parser.add_argument("--paradigm", default="动量反转")
    parser.add_argument("--pop-size", type=int, default=60)
    parser.add_argument("--generations", type=int, default=3)
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "reports" / "remediation_20260922" / "r3_forge_shadow_ab.json"),
    )
    args = parser.parse_args()
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    if len(seeds) < 3:
        raise ValueError("at least three fixed seeds are required")
    if args.paradigm not in MATURE_PARADIGM_PROFILES:
        raise ValueError("unknown mature paradigm: " + args.paradigm)

    data = load_real_data()
    rows = []
    for seed in seeds:
        for mode in ("legacy", "stability_turnover"):
            print(f"[R3 A/B] seed={seed} mode={mode}", flush=True)
            rows.append(run_one(
                data,
                mode=mode,
                seed=seed,
                paradigm=args.paradigm,
                pop_size=args.pop_size,
                generations=args.generations,
            ))

    legacy = [row for row in rows if row["mode"] == "legacy"]
    challenger = [row for row in rows if row["mode"] == "stability_turnover"]
    summary = {}
    for name, group in (("legacy", legacy), ("stability_turnover", challenger)):
        summary[name] = {
            "n": len(group),
            "median_abs_icir": median(group, "abs_icir"),
            "median_sign_consistency": median(group, "sign_consistency"),
            "median_rank_turnover_proxy": median(group, "rank_turnover_proxy"),
            "median_challenger_score": median(group, "challenger_score"),
        }
    criteria = {
        "n_pairs_gt_zero": len(seeds) > 0,
        "target_leakage_zero": all(row["target_leakage"] == 0 for row in rows),
        "median_abs_icir_retention_ge_0_80": (
            summary["stability_turnover"]["median_abs_icir"]
            >= 0.80 * summary["legacy"]["median_abs_icir"]
        ),
        "median_sign_consistency_not_worse": (
            summary["stability_turnover"]["median_sign_consistency"]
            >= summary["legacy"]["median_sign_consistency"] - 0.02
        ),
        "median_turnover_proxy_not_worse": (
            summary["stability_turnover"]["median_rank_turnover_proxy"]
            <= summary["legacy"]["median_rank_turnover_proxy"] + 0.01
        ),
    }
    noninferiority_pass = all(criteria.values())
    superiority = {
        "median_abs_icir_not_lower": (
            summary["stability_turnover"]["median_abs_icir"]
            >= summary["legacy"]["median_abs_icir"]
        ),
        "median_sign_consistency_not_lower": (
            summary["stability_turnover"]["median_sign_consistency"]
            >= summary["legacy"]["median_sign_consistency"]
        ),
        "median_turnover_proxy_not_higher": (
            summary["stability_turnover"]["median_rank_turnover_proxy"]
            <= summary["legacy"]["median_rank_turnover_proxy"]
        ),
        "median_challenger_score_higher": (
            summary["stability_turnover"]["median_challenger_score"]
            > summary["legacy"]["median_challenger_score"]
        ),
    }
    superiority_observed = all(superiority.values())
    payload = {
        "schema": "fuxi.r3_forge_shadow_ab.v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "claim": "Search-quality shadow A/B only; not S5 evidence and not permission to flip production default.",
        "caliber": {
            "seeds": seeds,
            "target_paradigm": args.paradigm,
            "pop_size": args.pop_size,
            "generations": args.generations,
            "workers": 1,
            "data_shape": list(data["close"].shape),
            "data_fingerprint": file_fingerprint(
                PROJECT_ROOT / "output" / "ap_batch" / "cache" / "weekly_prices.parquet"
            ),
        },
        "rows": rows,
        "summary": summary,
        "criteria": criteria,
        "search_noninferiority_pass": noninferiority_pass,
        "superiority_criteria": superiority,
        "superiority_observed": superiority_observed,
        "eligible_for_production_flip": False,
        "production_default": os.environ.get("FUXI_FORGE_FITNESS", "legacy"),
        "recommendation": "keep_legacy_default" if not superiority_observed else "continue_end_to_end_shadow",
        "next_gate": "Run multiple bounded end-to-end rounds and compare S5/stage-status distributions without relaxing thresholds.",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix(output.suffix + ".tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, output)
    print(json.dumps({
        "output": str(output),
        "search_noninferiority_pass": payload["search_noninferiority_pass"],
        "superiority_observed": payload["superiority_observed"],
        "criteria": criteria,
        "summary": summary,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
