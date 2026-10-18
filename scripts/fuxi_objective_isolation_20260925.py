"""Re-score a fixed expression set under both Forge objectives, no search."""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FA = ROOT / "research/factor_alchemy"
sys.path.insert(0, str(FA))
from forge import FactorForge  # noqa: E402
from forge.expression import parse_expression  # noqa: E402
from ralph_loop import RalphLoop  # noqa: E402

SOURCE = ROOT / "reports/remediation_20260922/r3_forge_shadow_ab.json"
OUT = ROOT / "reports/fuxi_generator_rebuild_20260925/objective_isolation.json"
PANEL = ROOT / "output/ap_batch/cache/weekly_prices.parquet"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run():
    frozen = json.loads(SOURCE.read_text(encoding="utf-8"))
    assert frozen["caliber"]["seeds"] == [17, 41, 73, 101, 137]
    expressions = {}
    for old in frozen["rows"]:
        expressions.setdefault(old["expression"], old["complexity"])
    assert len(expressions) > 0
    loader = object.__new__(RalphLoop)
    loader.data_dir = ROOT / "data"
    data = loader._load_forge_data()
    if not data or "fwd_return" not in data:
        raise RuntimeError("frozen real-data evaluator unavailable")
    common = dict(data=data, max_depth=6, max_complexity=12.0,
                  icir_threshold=0.1, forward_returns=data["fwd_return"],
                  seed=17, dim_prune="shadow")
    legacy = FactorForge(**common, fitness_mode="legacy")
    challenger = FactorForge(**common, fitness_mode="stability_turnover")
    rows = []
    for expression, complexity in expressions.items():
        tree = parse_expression(expression)
        if tree is None:
            rows.append({"expression": expression, "status": "parse_error"})
            continue
        try:
            values = legacy.evaluator.evaluate(tree)
        except (ValueError, KeyError, TypeError) as exc:
            rows.append({"expression": expression, "status": "field_contract_rejected",
                         "error_type": type(exc).__name__, "reason": str(exc)[:180]})
            continue
        left = legacy._fitness_components_from_values(values, complexity=float(complexity))
        right = challenger._fitness_components_from_values(values, complexity=float(complexity))
        assert left["icir"] == right["icir"]
        rows.append({"expression": expression, "status": "valid",
                     "complexity": complexity, "legacy": left, "challenger": right})
    usable = [r for r in rows if r["status"] == "valid"]
    assert len(usable) > 0
    rank = {mode: [r["expression"] for r in sorted(
        usable, key=lambda item: -item[mode]["score"])] for mode in ("legacy", "challenger")}
    result = {
        "schema": "fuxi_objective_isolation/v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "fixed, research-exposed R3 expressions; no evolution, no S5",
        "source_sha256": digest(SOURCE),
        "panel_sha256": digest(PANEL),
        "panel_shape": list(data["close"].shape),
        "n_fixed_expressions": len(expressions),
        "n_valid_comparisons": len(usable),
        "rows": rows,
        "rank": rank,
        "top3_overlap": len(set(rank["legacy"][:3]) & set(rank["challenger"][:3])),
        "economic_verdict": "not_evaluated",
        "formal_drought_resolved": False,
        "production_fitness": "legacy",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8", newline="\n")
    return result


if __name__ == "__main__":
    result = run()
    print({key: result[key] for key in ("n_fixed_expressions", "n_valid_comparisons",
                                      "top3_overlap", "economic_verdict")})
