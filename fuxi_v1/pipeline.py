"""Bounded independent factor research -> descriptive evidence -> diagnostic bundle."""

from copy import deepcopy
from pathlib import Path
from .contracts import keys, require, digest, evaluation
from .data import Snapshot, Labels
from .dsl import generate_candidates, evaluate_ast
from .evaluation import validate_policy, factor_statistics, admission
from .artifacts import publish, code_manifest
from .jq import compile_diagnostic


def prepare(payload):
    keys(payload, ("schema_version", "snapshot", "labels", "policy", "research_clock"))
    require(payload["schema_version"] == "fuxi.input.1", "unsupported input schema")
    source = deepcopy(payload)
    snapshot = Snapshot(source["snapshot"])
    labels = Labels(source["labels"], snapshot)
    policy_hash = validate_policy(source["policy"])
    # All candidates and policy are frozen before reading any factor statistics.
    candidates = generate_candidates(snapshot.payload["fields"])
    evaluations, all_scores, bundles, files = {}, {}, [], {"input.json": source, "candidates.json": candidates}
    for candidate in candidates:
        candidate_id = candidate["candidate_id"]
        scores = evaluate_ast(candidate["canonical_ast"], snapshot)
        stages = [evaluation("E0", "valid", "supported_AST_and_snapshot_contract", decision="pass"),
                  factor_statistics(scores, snapshot, labels, source["policy"], source["research_clock"])]
        for stage in ("E2", "E3", "E4", "E5"):
            stages.append(evaluation(stage, "not_evaluated", "not_implemented_in_this_milestone"))
        verdict = admission(stages, source["policy"])
        script, capability = compile_diagnostic(candidate, snapshot, scores)
        bundle = dict(schema_version="fuxi.research_bundle.1", purpose="jq_diagnostic",
                      candidate=candidate, data_snapshot_hash=snapshot.hash, labels_hash=labels.hash,
                      policy_hash=policy_hash, exposure=snapshot.payload["exposure"],
                      window={"warmup_start": snapshot.sessions[0], "evaluation_start": snapshot.sessions[snapshot.start],
                              "end": snapshot.sessions[-1]}, research_clock=source["research_clock"],
                      capability=capability, admission=verdict,
                      strategy_contract=None, account_evidence=None, live_permission=False)
        bundle["bundle_id"] = digest(bundle)
        prefix = "bundles/" + candidate_id
        files[prefix+"/bundle.json"] = bundle
        files[prefix+"/jq_diagnostic.py"] = script
        bundles.append(bundle)
        evaluations[candidate_id] = stages
        all_scores[candidate_id] = scores
    files["scores.json"], files["evaluations.json"] = all_scores, evaluations
    summary = dict(schema_version="fuxi.summary.1", phase="M1_M2_bootstrap_partial",
                   exposure=snapshot.payload["exposure"], generated_count=len(candidates),
                   evaluated_e0_count=len(candidates),
                   valid_e1_count=sum(stages[1]["eval_status"] == "valid" for stages in evaluations.values()),
                   jq_diagnostic_bundle_count=len(bundles), jq_candidate_count=0,
                   external_jq_runs=0, economic_reward_events=0, live_orders=0,
                   next_blockers=bundles[0]["admission"]["blockers"],
                   policy_hash=policy_hash, input_hash=digest(source),
                   note="Statistics describe synthetic/research-exposed prediction proxies, not strategy economics")
    files["summary.json"] = summary
    return files, summary


def run(payload, output_root):
    before = code_manifest()
    files, summary = prepare(payload)
    for name in before:
        files["code/" + name] = (Path(__file__).parent / name).read_bytes()
    require(code_manifest() == before, "source changed during run")
    from . import __version__
    provenance = dict(kernel_version=__version__, entrypoint="bootstrap_compatibility", code_hashes=before,
                      input_hash=digest(payload), policy_hash=summary["policy_hash"],
                      experiment_id=digest({"input": payload, "code": before}),
                      actual_external_calls=0, legacy_adapter_enabled=False,
                      writer_scope="isolated_output_directory")
    path = publish(output_root, files, provenance)
    return path, summary
