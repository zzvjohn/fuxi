"""Independent receipt-backed research, forward acceptance and revocable paper release."""

from .contracts import digest, require, timestamp, evaluation, validate_evaluation
from .evidence import now, overlaps
from .statistics import block_inference


class AcceptanceService:
    def __init__(self, store):
        self.store = store

    def research(self, evaluation_id, data_receipt_id=None):
        result = self.store.get(evaluation_id)
        stages = result["evaluations"]
        require(len({e["stage"] for e in stages}) == len(stages), "duplicate evaluation stage")
        blockers = []
        for stage in ("E0", "E1", "E2", "E3", "E4"):
            matches = [r for r in stages if r["stage"] == stage]
            if len(matches) != 1:
                blockers.append(stage+"_missing")
                continue
            validate_evaluation(matches[0])
            if matches[0]["eval_status"] != "valid" or matches[0]["decision"] != "pass":
                blockers.append(stage+"_not_accepted")
        if result["source_scope"] == "synthetic":
            blockers.append("synthetic_data_cannot_grant_investment_eligibility")
        if (result.get('selection_audit') or {}).get('posthoc_exclusion') is True:
            blockers.append('posthoc_universe_exclusion_requires_unbiased_revalidation')
        if any(e.get('diagnostics',{}).get('tail_resolution_adequate') is False for e in stages):
            blockers.append('bootstrap_tail_resolution_inadequate')
        # Investment-level data assurance no longer gates JQ candidacy. A research-level
        # factor that passes E0-E4 earns JQ backtest candidacy on scientific merit alone;
        # the independent data contract is deferred to the live/QMT signal stage, where the
        # account owner (default; reviewer_role is configurable) signs off before any real
        # order reaches TraderLink/QMT. Missing the contract is recorded as a pending
        # certification, never as a candidacy blocker.
        investment_certification = {"status": "pending", "reviewer": None,
                                    "reviewer_role": "account_owner_manual_signoff",
                                    "required_before": "live_execution_or_QMT_signal"}
        if data_receipt_id is not None:
            receipt = self.store.consume_receipt(data_receipt_id, "data_contract")
            if receipt.get('schema_version') != 'fuxi.independent_data_assurance.1' or receipt.get('investment_eligibility_certified') is not True:
                blockers.append("derived_validation_cannot_grant_investment_eligibility")
            if receipt["snapshot_hash"] != result["snapshot_hash"] or not all(receipt.get(k) is True for k in
                      ("pit_certified", "universe_certified", "account_events_certified", "execution_rules_certified")):
                blockers.append("data_contract_not_certified")
            investment_certification.update(status="certified", reviewer=receipt.get("reviewer"),
                                            reviewer_role=receipt.get("reviewer_role") or "independent_reviewer")
        body = {"candidate_id": result["candidate_id"], "evaluation_id": evaluation_id,
                "data_receipt_id": data_receipt_id,
                "policy_hash": result["policy_hash"], "ir_hash": result["strategy_ir"]["ir_hash"],
                "snapshot_hash": result["snapshot_hash"], "blockers": blockers,
                "investment_certification": investment_certification,
                "jq_candidate_allowed": not blockers, "live_execution_allowed": False,
                "checked_at": now(), "E5_required_for_first_JQ": False}
        evidence_id = self.store.append("research_admission", body)
        body["receipt_id"] = self.store.receipt("research_acceptance", {**body, "evidence_id": evidence_id}) if not blockers else None
        return body

    def freeze_forward(self, research_receipt, protocol_receipt, parity_receipt):
        research = self.store.consume_receipt(research_receipt, "research_acceptance")
        protocol = self.store.consume_receipt(protocol_receipt, "confirmatory_protocol")
        parity = self.store.consume_receipt(parity_receipt, "jq_parity")
        require(research["candidate_id"] in protocol["candidate_ids"] and research["ir_hash"] == parity["ir_hash"],
                "forward candidate/parity mismatch")
        require(protocol.get("policy_hash") == research.get("policy_hash"), "forward policy mismatch")
        require(parity["accepted"] is True and parity["n_pairs"] > 0 and parity["observed_on_jq"] is True,
                "independent JQ evidence required")
        cohort = "cohort_"+digest([research_receipt, protocol_receipt, parity_receipt])
        self.store.protect(cohort, protocol["scope"], protocol["frozen_at"])
        return self.store.receipt("forward_pending", {"cohort_id": cohort, "research_receipt": research_receipt,
                "protocol_receipt": protocol_receipt, "parity_receipt": parity_receipt,
                "candidate_id": research["candidate_id"], "frozen_at": protocol["frozen_at"], "ir_hash": research["ir_hash"]})

    def forward(self, pending_receipt, observation_id, observation_clock):
        pending = self.store.consume_receipt(pending_receipt, "forward_pending")
        protocol = self.store.consume_receipt(pending["protocol_receipt"], "confirmatory_protocol")
        # Fixed looks, each spent once, prevent repeated peeking/early cherry-picked stopping.
        require(observation_clock in protocol["look_schedule"], "unregistered acceptance look")
        look_id = pending_receipt+":look:"+observation_clock
        require(not self.store.db.execute("SELECT 1 FROM facts WHERE id=?", (look_id,)).fetchone(), "acceptance look already spent")
        observation = self.store.get(observation_id)
        rows = observation["rows"]
        require(isinstance(rows,list) and type(observation.get('account_identifiable')) is bool,
                'forward observation rows/account identification invalid')
        require(observation["ir_hash"] == pending["ir_hash"] and observation["baseline_hash"] == protocol["baseline_hash"],
                "forward comparison identity mismatch")
        require(observation["cost_profile_hash"] == protocol["cost_profile_hash"], "forward cost mismatch")
        require(observation["scope"] == protocol["scope"], "forward observation scope mismatch")
        require(timestamp(observation["mature_at"]) <= timestamp(observation_clock), "future labels not mature")
        require(all(timestamp(row["decision_time"]) > timestamp(pending["frozen_at"])
                    and timestamp(row["decision_time"]) < timestamp(row["mature_at"])
                    and timestamp(row["mature_at"]) <= timestamp(observation_clock) for row in observation["rows"]),
                "forward observations predate freeze or not mature")
        require(len({row["decision_time"] for row in rows}) == len(rows), "duplicate forward time sections")
        intervals = protocol["scope"]["intervals"]
        require(all(any(timestamp(a) <= timestamp(row["decision_time"]) <= timestamp(b) for a,b in intervals)
                    for row in rows), "forward decision outside registered scope")
        from .contracts import finite
        require(all(row.get('paired_net_increment') is None or finite(row['paired_net_increment']) for row in rows),
                'forward net increment must be finite or explicitly unknown')
        # Reject malformed/foreign observations before spending the fixed look.
        self.store.expose(pending["cohort_id"], protocol["scope"], "acceptance", acceptance=True)
        self.store.append("acceptance_look", {"observation_id": observation_id, "clock": observation_clock}, look_id)
        if (not observation.get("account_identifiable") or len(rows) < protocol["minimum_sections"]
                or any(r.get("paired_net_increment") is None for r in rows)):
            return evaluation("E5", "data_insufficient", "insufficient_identified_forward_pairs", scope="forward_acceptance")
        alpha = protocol["alpha"]/len(protocol["candidate_ids"])/len(protocol["look_schedule"])
        inference = block_inference([r["paired_net_increment"] for r in rows], block_length=protocol["block_length"],
                                    alpha=alpha, replicates=protocol["bootstrap_replicates"], seed=90210)
        passed = inference["status"] == "valid" and inference["lower"] > protocol["minimum_net_increment_daily"]
        result = evaluation("E5", "valid", "pre_registered_forward_decision", decision="pass" if passed else "reject",
                            reject_class=None if passed else "economic", scope="forward_acceptance",
                            metrics=inference, n_pairs=len(rows))
        event_id = self.store.append("forward_decision", {"pending_receipt": pending_receipt, "evaluation": result,
                                                         "observation_id": observation_id})
        if passed:
            result["acceptance_receipt"] = self.store.receipt("forward_acceptance", {"pending_receipt": pending_receipt,
                "decision_id": event_id, "ir_hash": pending["ir_hash"], "accepted": True})
        return result


class ReleaseService:
    def __init__(self, store):
        self.store = store

    def bundle(self, forward_receipt, artifacts, rollback_receipt=None):
        accepted = self.store.consume_receipt(forward_receipt, "forward_acceptance")
        require(artifacts and all(isinstance(v, str) and len(v) == 64 for v in artifacts.values()), "complete release artifact hashes required")
        require({"code", "model", "strategy", "data_contract", "policy", "state_schema"} <= set(artifacts), "release contents missing")
        return self.store.receipt("release_bundle", {"forward_receipt": forward_receipt, "ir_hash": accepted["ir_hash"],
                   "artifacts": artifacts, "rollback_receipt": rollback_receipt, "created_at": now()})

    def activate_paper(self, bundle_receipt):
        bundle = self.store.consume_receipt(bundle_receipt, "release_bundle")
        self.store.consume_receipt(bundle["forward_receipt"], "forward_acceptance")
        return self.store.receipt("paper_activation", {"bundle_receipt": bundle_receipt, "mode": "paper",
                                                       "live_authorized": False, "activated_at": now()})

    def rollback(self, activation_receipt, reconciliation_id):
        active = self.store.consume_receipt(activation_receipt, "paper_activation")
        reconciled = self.store.get(reconciliation_id)
        require(reconciled.get("unknown_orders") == 0 and reconciled.get("account_matched") is True,
                "reconcile actual account before rollback")
        bundle = self.store.consume_receipt(active["bundle_receipt"], "release_bundle")
        require(bundle["rollback_receipt"], "no frozen rollback target")
        # Verify the target before revoking the current activation.
        self.store.consume_receipt(bundle["rollback_receipt"], "release_bundle")
        self.store.revoke(activation_receipt, "reconciled rollback")
        return self.activate_paper(bundle["rollback_receipt"])
