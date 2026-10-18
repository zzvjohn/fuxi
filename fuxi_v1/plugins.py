"""Optional research plugins are admitted by observable capability, never a marketing name."""

import importlib.util
from .contracts import require, digest


OPTIONAL_PLUGINS = {
    "typed_symbolic_search": {"dependency": "pysr", "phase": "M7", "consumer": "AST_proposal_only"},
    "cost_sensitive_bayesian_design": {"dependency": "botorch", "phase": "M7", "consumer": "action_priority_only"},
    "sparse_dynamics": {"dependency": "pysindy", "phase": "M7", "consumer": "hypothesis_only"},
    "prompt_evolution": {"dependency": "gepa", "phase": "M7", "consumer": "shadow_paired_proposal_only"},
}


def catalog():
    result = {}
    for name, spec in OPTIONAL_PLUGINS.items():
        present = importlib.util.find_spec(spec["dependency"]) is not None
        result[name] = {**spec, "dependency_present": present, "status": "unverified" if present else "unsupported",
                        "automatically_enabled": False, "core_dependency": False}
    return result


def validate_plugin_proposal(proposal, fields):
    """All external symbolic proposals re-enter the same typed compiler."""
    from .dsl import compile_ast
    require(set(proposal) == {"plugin_id", "ast", "evidence_ids", "training_scope"}, "plugin proposal contract mismatch")
    require(proposal["plugin_id"] in OPTIONAL_PLUGINS, "unknown plugin")
    compiled = compile_ast(proposal["ast"], fields)
    return {"proposal_hash": digest(proposal), "compiled": compiled, "eligibility": False}


def meta_compare(parent, challenger, paired_tasks):
    """Shadow-only method revision; changing evaluation/permission objects is forbidden."""
    mutable = {"prompt_template", "retrieval_limit", "mutation_weights", "priority_rule"}
    require(set(parent) == set(challenger), "unregistered meta configuration key")
    changed = {k for k in parent if parent[k] != challenger[k]}
    require(changed and changed <= mutable, "meta optimizer attempted to change immutable policy")
    require(paired_tasks and all(row["same_budget"] and row["same_visible_scope"] for row in paired_tasks),
            "meta evaluation needs matched tasks/budgets")
    valid = [r for r in paired_tasks if r["parent_valid"] and r["challenger_valid"]]
    improvements = [r["challenger_value"]-r["parent_value"] for r in valid]
    return {"parent_hash": digest(parent), "challenger_hash": digest(challenger), "changed": sorted(changed),
            "paired_count": len(valid), "mean_improvement": sum(improvements)/len(improvements) if improvements else None,
            "status": "shadow_evidence" if valid else "data_insufficient", "automatic_activation": False}
