"""RunSpec content-identity contract (P0.1, v2.0 landing).

Lands the CONTRACTS.md "共同对象与身份" RunSpec as a strict, digestible Python
contract. Every semantic field is a content reference (sha256 hex string or a
nested, fully specified object); changing any semantic field must change the
digest, and equal specs must reuse the same experiment identity.

This module is a standalone contract artefact: it does not yet replace the
``experiment`` dict built by ``unified_runner.run``. Wiring order (per R2
P-batches): P0.1 gives the contract + digest; P1 gives the ScoreRef/caches that
consume these refs.
"""

from .contracts import digest, keys, require, nonempty

RUN_SPEC_SCHEMA = "fuxi.unified_run.design.1"

# The eleven content refs from CONTRACTS.md plus the common identity fields.
REQUIRED_KEYS = (
    "schema_version", "task_kind", "snapshot_ref", "universe_ref", "label_ref",
    "split_ref", "generator_config", "screen_policy_ref", "evaluation_policy_ref",
    "baseline_strategy_ref", "execution_cost_ref", "memory_view_ref", "budget_ref",
    "code_ref",
)
OPTIONAL_KEYS = ("effective_config_ref", "task_kind_detail")
TASK_KINDS = ("candidate_research", "risk_condition", "complement")
GENERATOR_MODES = ("legacy", "expanded")
GENERATOR_ACTIONS = ("template", "edit", "combine", "hypothesis")


def _sha256_hex(value):
    require(isinstance(value, str) and len(value) == 64
            and all(c in "0123456789abcdef" for c in value),
            "content ref must be a 64-char lowercase sha256 hex string: " + repr(value)[:40])
    return value


def validate_run_spec(spec):
    """Validate the RunSpec contract and return it (raises ContractError)."""
    keys(spec, REQUIRED_KEYS, OPTIONAL_KEYS)
    require(spec["schema_version"] == RUN_SPEC_SCHEMA, "unsupported RunSpec schema")
    require(spec["task_kind"] in TASK_KINDS, "invalid task_kind")
    require(nonempty(spec["task_kind"]) and nonempty(spec.get("task_kind_detail", "x"))
            or "task_kind_detail" not in spec, "task_kind_detail must be non-empty when present")
    for field in ("snapshot_ref", "universe_ref", "label_ref", "split_ref",
                  "screen_policy_ref", "evaluation_policy_ref", "baseline_strategy_ref",
                  "execution_cost_ref", "memory_view_ref", "budget_ref", "code_ref"):
        _sha256_hex(spec[field])
    gen = spec["generator_config"]
    keys(gen, ("mode", "operator_catalog_ref", "allowed_actions", "seed"))
    require(gen["mode"] in GENERATOR_MODES, "invalid generator mode")
    _sha256_hex(gen["operator_catalog_ref"])
    require(isinstance(gen["allowed_actions"], list)
            and gen["allowed_actions"]
            and all(a in GENERATOR_ACTIONS for a in gen["allowed_actions"]),
            "generator allowed_actions must be a non-empty subset of %s" % (GENERATOR_ACTIONS,))
    require(type(gen["seed"]) is int and gen["seed"] >= 0, "generator seed must be a non-negative int")
    if "effective_config_ref" in spec:
        _sha256_hex(spec["effective_config_ref"])
    digest(spec)
    return spec


def run_spec_digest(spec):
    """Content digest of a validated RunSpec (excluding nothing semantic)."""
    validate_run_spec(spec)
    return digest(spec)


def build_run_spec(*, task_kind, snapshot_ref, universe_ref, label_ref, split_ref,
                   generator_config, screen_policy_ref, evaluation_policy_ref,
                   baseline_strategy_ref, execution_cost_ref, memory_view_ref,
                   budget_ref, code_ref, effective_config_ref=None):
    """Assemble + validate + digest a RunSpec in one step."""
    spec = {"schema_version": RUN_SPEC_SCHEMA, "task_kind": task_kind,
            "snapshot_ref": snapshot_ref, "universe_ref": universe_ref,
            "label_ref": label_ref, "split_ref": split_ref,
            "generator_config": generator_config, "screen_policy_ref": screen_policy_ref,
            "evaluation_policy_ref": evaluation_policy_ref,
            "baseline_strategy_ref": baseline_strategy_ref,
            "execution_cost_ref": execution_cost_ref, "memory_view_ref": memory_view_ref,
            "budget_ref": budget_ref, "code_ref": code_ref}
    if effective_config_ref is not None:
        spec["effective_config_ref"] = effective_config_ref
    validate_run_spec(spec)
    return {"spec": spec, "run_spec_digest": digest(spec)}
