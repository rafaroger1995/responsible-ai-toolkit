"""Shared review/retest orchestration composed with the toolkit's HITL queue.

All identities are fixture identities. Storage is in memory. Access checks protect
this API's transitions, not hostile code with access to the Python process.
"""
from __future__ import annotations

import copy
from dataclasses import asdict
from .evaluator import Assessment, assess, digest
from responsible_ai_toolkit.hitl.orchestrator import HITLOrchestrator, EscalationPolicy, CaseStatus


class ReviewSession:
    def __init__(self, initial: Assessment, reviewers: list, *, review_role: str, authority_role: str):
        self.initial = Assessment.load(initial.to_dict())
        self._assessments = [self.initial.to_dict()]
        self._events = []
        self._evidence = copy.deepcopy(self.initial.to_dict()["evidence_bundle"])
        self._retest = None
        self._disposition = "unresolved"
        self.review_role, self.authority_role = review_role, authority_role
        self._hitl = HITLOrchestrator()
        for reviewer in reviewers:
            self._hitl.add_reviewer(reviewer["id"], roles=list(reviewer["roles"]))
        self._hitl.add_escalation_policy(EscalationPolicy("decision-authority", lambda c: True, authority_role))
        data = initial.to_dict()
        self._case = self._hitl.submit_for_review(data["case_id"], review_role,
                    {"recorded_outcome": data["input_snapshot"].get("outcome")}, "Decision/evidence review")
        self._event("opened", None, {"assessment_id": data["assessment_id"]})

    def _auth(self, actor):
        if self._case.status == CaseStatus.COMPLETED:
            raise ValueError("Review is already closed")
        reviewer = self._hitl.get_reviewer(actor)
        role = self._case.required_reviewer_role or self._case.category
        if not reviewer or not reviewer.active or actor != self._case.assigned_to or role not in reviewer.roles:
            raise ValueError("Reviewer not authorized for assigned role")
        return reviewer

    def _event(self, action, actor, details):
        self._events.append({"event_id": "EV-" + str(len(self._events) + 1), "action": action, "actor": actor,
                             "assessment_id": self._assessments[-1]["assessment_id"], "details": copy.deepcopy(details)})

    def update_roles(self, actor, roles):
        """Fixture administrator action; not an authenticated production admin API."""
        reviewer = self._hitl.get_reviewer(actor)
        if not reviewer:
            raise ValueError("Unknown fixture actor")
        reviewer.roles = list(roles)
        self._event("fixture_roles_changed", "fixture-admin", {"actor": actor, "roles": roles})

    def assign(self, actor):
        """Explicit assignment; unavailable/mismatched authority never gets the case."""
        reviewer = self._hitl.get_reviewer(actor)
        role = self._case.required_reviewer_role or self._case.category
        if self._case.status == CaseStatus.COMPLETED or not reviewer or not reviewer.active or role not in reviewer.roles or reviewer.current_load >= reviewer.max_load:
            raise ValueError("Assignment not authorized or reviewer unavailable")
        old = self._hitl.get_reviewer(self._case.assigned_to)
        if old:
            old.current_load = max(0, old.current_load - 1)
        reviewer.current_load += 1
        self._case.assigned_to = actor
        self._case.status = CaseStatus.ASSIGNED
        self._event("assigned", actor, {"required_role": role})

    def request_info(self, actor, reason):
        self._auth(actor)
        if not reason.strip():
            raise ValueError("Information request needs a reason")
        self._hitl.record_decision(self._case.internal_id, actor, "request_info", reason)
        self._event("request_info", actor, {"reason": reason})

    def escalate(self, actor, reason):
        self._auth(actor)
        self._hitl.record_decision(self._case.internal_id, actor, "escalate", reason)
        self._event("escalated", actor, {"required_role": self.authority_role, "reason": reason})

    def provide_evidence(self, actor, record, profiles, authorities, source_ref):
        self._auth(actor)
        original = self.initial.to_dict()["input_snapshot"]
        if not source_ref or any(record.get(k) != original.get(k) for k in ["case_id", "decision_at", "outcome", "kind", "model_ref"]):
            raise ValueError("Evidence must reference the same historical decision and an evidence source")
        bundle = copy.deepcopy({"record": record, "profiles": profiles, "authority_registry": authorities})
        self._evidence = bundle
        self._retest = None
        self._case.status = CaseStatus.IN_REVIEW
        self._event("evidence_received", actor, {"source_ref": source_ref, "evidence_sha256": digest(bundle),
                                               "changed": digest(bundle) != self.initial.to_dict()["evidence_sha256"]})

    def retest(self, actor, *, run_id):
        self._auth(actor)
        previous = self._assessments[-1]
        e = self._evidence
        result = assess(e["record"], e["profiles"], e["authority_registry"], run_id=run_id,
                        parent_assessment_id=previous["assessment_id"])
        value = result.to_dict()
        original = self.initial.to_dict()
        same_policy = original["policy_sha256"] is None or value["policy_sha256"] == original["policy_sha256"]
        if not same_policy:
            result = assess(e["record"], e["profiles"], e["authority_registry"], run_id=run_id,
                            parent_assessment_id=previous["assessment_id"], assessment_kind="counterfactual")
            value = result.to_dict()
        # Explanation edits alone must not count as new substantive evidence.
        def substantive(bundle):
            bundle = copy.deepcopy(bundle)
            bundle["record"].pop("explanation", None)
            return digest(bundle)
        changed = substantive(e) != substantive(original["evidence_bundle"])
        explanation_repaired = original["decision_support"] == "supported_within_scope" and original["explanation_status"] == "inconsistent" and value["explanation_status"] == "consistent"
        original_concern = original["decision_support"] in {"policy_conflict", "insufficient_evidence"} or original["explanation_status"] == "inconsistent"
        eligible = (original_concern and value["decision_support"] == "supported_within_scope" and same_policy and
                    value["review_action"] != "review_required" and
                    (changed or explanation_repaired) and
                    (original["explanation_status"] != "inconsistent" or value["explanation_status"] == "consistent"))
        self._assessments.append(value)
        self._retest = {"assessment_id": value["assessment_id"], "predecessor": previous["assessment_id"],
                        "evidence_changed": changed, "historical_policy_preserved": same_policy,
                        "eligible_for_verified_correction": eligible}
        self._event("retest", actor, self._retest)
        return result

    def close(self, actor, disposition, reason):
        reviewer = self._auth(actor)
        if self.authority_role not in reviewer.roles or not reason.strip():
            raise ValueError("Designated disposition authority and reason required")
        if disposition not in {"correction_verified", "authorized_risk_acceptance"}:
            raise ValueError("Unsupported closure state")
        if disposition == "correction_verified" and (not self._retest or not self._retest["eligible_for_verified_correction"]):
            raise ValueError("No eligible changed-evidence historical retest")
        self._hitl.record_decision(self._case.internal_id, actor, "override", reason)
        self._disposition = disposition
        self._event(disposition, actor, {"reason": reason, "retest": self._retest,
                                       "customer_outcome_changed": False})

    def snapshot(self):
        case = asdict(self._case)
        case["status"] = self._case.status.value
        case["priority"] = self._case.priority.value
        case["decision"] = self._case.decision.value if self._case.decision else None
        closed = self._case.status == CaseStatus.COMPLETED
        return copy.deepcopy({"schema_version": "trustera.review/1.0", "case_id": self._case.case_id,
          "case": case, "disposition": self._disposition, "open": not closed,
          "completed_reviews": int(closed), "verified_corrections": int(self._disposition == "correction_verified"),
          "risk_acceptances": int(self._disposition == "authorized_risk_acceptance"),
          "customer_outcome_changes": 0, "sla_compliance_rate": self._hitl.sla_compliance_rate(),
          "reviewers": [asdict(x) for x in self._hitl._reviewers.values()],
          "original_assessment_id": self.initial.to_dict()["assessment_id"], "assessments": self._assessments,
          "latest_retest": self._retest, "events": self._events,
          "limitations": "Fixture authority; in-memory state; no authentication or durable workflow service"})
