"""Explicit operator entry points for independently reviewed evidence and paper release.

Local operator files are trusted attestations, not cryptographic proof that a vendor
or reviewer told the truth. Generators have no action that calls these operations.
"""

from pathlib import Path
import hashlib
from .contracts import load_json, require, digest, timestamp
from .artifacts import verify
from .acceptance import AcceptanceService, ReleaseService
from .execution import reconcile_rows
from .policy import register_confirmatory_protocol


def reviewed_files(spec):
    require(isinstance(spec.get("reviewer"),str) and spec["reviewer"].strip(),"independent reviewer identity required")
    timestamp(spec["reviewed_at"])
    files = spec["evidence_files"]
    require(files,"review evidence files required")
    for name, expected in files.items():
        path = Path(name)
        require(path.is_absolute() and path.is_file(),"absolute evidence file missing")
        require(hashlib.sha256(path.read_bytes()).hexdigest() == expected,"review evidence file hash mismatch")
    return {"reviewer":spec["reviewer"],"reviewed_at":spec["reviewed_at"],"evidence_files":files,
            "trust_boundary":"explicit_local_operator_attestation"}


def register_data(store, spec):
    """Register a data contract whose flags are *derived*, not asserted.

    The previous shape asked the caller to pass four booleans and took their word
    for it. That is why the whole downstream chain stalled: nobody could honestly
    assert them, so every candidate stopped at ``data_insufficient`` and the
    difference between "the data is bad" and "nobody certified it" was invisible.

    Here the caller must supply the reviewed payload, the reviewed certificate and
    the reviewed lifecycle table. The certificate is recomputed from those bytes and
    must reproduce the stored hash; only then are its flags consumed.
    """
    from .certification import SCHEMA, certify, derived_data_receipt
    from .pit_universe import read_lifecycle_file
    review = reviewed_files(spec)
    for key in ("certificate_path", "snapshot_path", "lifecycle_path"):
        require(isinstance(spec.get(key), str) and Path(spec[key]).is_absolute(),
                key + " must be an absolute path")
        require(spec[key] in review["evidence_files"],
                key + " must appear in the reviewed evidence files")
    certificate = load_json(spec["certificate_path"])
    require(certificate.get("schema_version") == SCHEMA, "unsupported certification schema")
    authority = {k: v for k, v in certificate.items()
                 if k not in ("certified_at", "certification_hash")}
    require(digest(authority) == certificate.get("certification_hash"),
            "certificate content changed since it was written")
    payload = load_json(spec["snapshot_path"])
    lifecycle = read_lifecycle_file(spec["lifecycle_path"])
    require(digest({k: v for k, v in sorted(lifecycle.items())}) == certificate["lifecycle_digest"],
            "lifecycle table differs from the one the certificate was derived from")
    recomputed = certify(payload, declaration=certificate["declaration"], lifecycle=lifecycle)
    require(recomputed["certification_hash"] == certificate["certification_hash"],
            "certificate is not reproducible from the reviewed payload")
    require(recomputed["blocking_gaps"] == certificate["blocking_gaps"],
            "recomputed blocking gaps differ from the certificate")
    receipt = derived_data_receipt(recomputed)
    required = ("pit_certified", "universe_certified", "account_events_certified",
                "execution_rules_certified")
    require(receipt["snapshot_hash"] == spec["snapshot_hash"],
            "certificate snapshot hash does not match the registered snapshot hash")
    body = {**{k: v for k, v in spec.items() if k != "snapshot_path"}, **review, **receipt,
            "recomputed_from_reviewed_bytes": True}
    # This validator has neither vendor revision archives nor independently reviewed
    # settlement coverage. Its declaration checks must never mint an investment receipt.
    validation_id = store.receipt("data_validation", body)
    return {"receipt_id": None, "validation_receipt_id": validation_id,
            "blockers": receipt["blocking_gaps"] + ["independent_investment_data_assurance_required"], "derived": True,
            "flags": {k: receipt[k] for k in required},
            "market_representativeness_certified": receipt["market_representativeness_certified"],
            "residual_gaps": receipt["residual_gaps"]}


def operate(store, action, spec):
    acceptance, release = AcceptanceService(store), ReleaseService(store)
    if action == "register-data":
        return register_data(store, spec)
    if action == "import-data-assurance":
        from .data_assurance import import_assurance
        return import_assurance(store,spec,reviewed_files(spec))
    if action == "admit-research":
        root = Path(spec["bundle_path"])
        manifest = verify(root)
        result,policy = load_json(root/"evaluation.json"),load_json(root/"policy.json")
        require(policy["policy_hash"] == result["policy_hash"],"evaluation/policy mismatch")
        require(digest({k:v for k,v in policy.items() if k != "policy_hash"}) == policy["policy_hash"],"policy changed")
        if policy.get('schema_version')=='fuxi.research_policy.2':
            from .historical_handoff import validate_context
            validate_context(load_json(root/'history_freeze.json'),load_json(root/'history_plan.json'),
                             policy,load_json(root/'history_outcome.json'))
            require(result['snapshot_hash']==policy['snapshot_hash'],'historical evaluation snapshot mismatch')
        else:
            require(policy.get('status')=='calibrated_research_profile','frozen calibrated research policy required')
        evidence = store.append("evaluation",result)
        store.append("imported_research_bundle",{"evaluation_id":evidence,"manifest":manifest,"policy":policy})
        return acceptance.research(evidence,spec.get("data_receipt_id"))
    if action == "import-jq-parity":
        review = reviewed_files(spec)
        require(spec["observed_on_jq"] is True and spec["jq_run_id"],"actual JQ run identifier required")
        research = store.consume_receipt(spec["research_receipt"],"research_acceptance")
        require(research["ir_hash"] == spec["ir_hash"],"JQ research/IR mismatch")
        required = {"signal", "target_weights", "orders", "holdings", "cash", "receivables", "nav"}
        require(required <= set(spec["fields"]),"signal/order/account parity fields required")
        result = reconcile_rows(spec["expected_rows"],spec["observed_rows"],spec["fields"],spec.get("tolerance",1e-8))
        evidence_id = store.append("jq_parity_comparison",{**result,**review,"jq_run_id":spec["jq_run_id"]})
        body = {**result,**review,"ir_hash":spec["ir_hash"],"observed_on_jq":True,
                "research_receipt":spec["research_receipt"],"evidence_id":evidence_id,"jq_run_id":spec["jq_run_id"]}
        return {**result,"receipt_id":store.receipt("jq_parity",body) if result["accepted"] else None}
    if action == "compile-jq":
        from .jq_strategy import compile_jq_strategy
        from .artifacts import publish
        root = Path(spec["bundle_path"])
        manifest = verify(root)
        result = load_json(root/"evaluation.json")
        script,metadata = compile_jq_strategy(result["strategy_ir"],
             {"execution":result["candidate_account"]["assumptions"]},store,spec["research_receipt"])
        path = publish(spec["output_root"],{"jq_strategy.py":script,"compiler.json":metadata,
                       "strategy_ir.json":result["strategy_ir"]},{"source_manifest":manifest["manifest_hash"],
                       "research_receipt":spec["research_receipt"],"live_execution_allowed":False})
        return {"bundle_path":str(path),**metadata}
    if action == "register-protocol":
        review = reviewed_files(spec)
        calibration = spec["calibration"]
        require(calibration["kind"] == "confirmatory_calibration" and calibration["verified"] is True,
                "independent calibration adjudication required")
        evidence = store.append("confirmatory_calibration",{**calibration,**review})
        protocol = dict(spec["protocol"],calibration_evidence_id=evidence)
        return {"receipt_id":register_confirmatory_protocol(store,protocol)}
    if action == "freeze-forward":
        return {"receipt_id":acceptance.freeze_forward(spec["research_receipt"],spec["protocol_receipt"],spec["parity_receipt"])}
    if action == "observe-forward":
        from .evidence import now
        require(timestamp(spec['observation_clock'])<=timestamp(now()),
                'future observation clock cannot be submitted by a real operator')
        review = reviewed_files(spec)
        evidence = store.append("forward_observation",{**spec["observation"],"review":review})
        return acceptance.forward(spec["pending_receipt"],evidence,spec["observation_clock"])
    if action == "build-release":
        review = reviewed_files(spec)
        store.append("release_artifact_review",review)
        require(set(spec["artifacts"].values()) <= set(review["evidence_files"].values()),"release artifacts must be verified files")
        return {"receipt_id":release.bundle(spec["forward_receipt"],spec["artifacts"],spec.get("rollback_receipt"))}
    if action == "activate-paper":
        return {"receipt_id":release.activate_paper(spec["bundle_receipt"])}
    if action == "investment-signoff":
        from .manual_handoff import investment_signoff
        return investment_signoff(store,spec)
    if action == "rollback-paper":
        review = reviewed_files(spec)
        evidence = store.append("account_reconciliation",{**spec["reconciliation"],"review":review})
        return {"receipt_id":release.rollback(spec["activation_receipt"],evidence)}
    if action == "revoke":
        require(spec.get("reason"),"revocation reason required")
        store.revoke(spec["receipt_id"],spec["reason"])
        return {"receipt_id":spec["receipt_id"],"state":"revoked"}
    raise ValueError("unsupported operator action")


ACTIONS = ("register-data","import-data-assurance","admit-research","compile-jq","import-jq-parity","register-protocol","freeze-forward",
           "observe-forward","build-release","activate-paper","investment-signoff","rollback-paper","revoke")
