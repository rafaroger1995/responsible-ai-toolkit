"""Regression cases for unresolved evidence and explicit review authority."""
from dataclasses import asdict
import copy
import pytest
from responsible_ai_toolkit.hitl.orchestrator import HITLOrchestrator, EscalationPolicy


def assigned_case():
    h = HITLOrchestrator()
    r = h.add_reviewer("analyst", roles=["lending"])
    c = h.submit_for_review("SYN-001", "lending", {}, "Missing source record")
    return h, r, c


def test_request_info_is_open_then_final_decision_completes_once():
    h, r, c = assigned_case()
    h.record_decision(c.internal_id, r.reviewer_id, "request_info", "Supply the source")
    assert c.status.value == "waiting_info"
    assert c.completed_at is None
    assert (r.current_load, r.total_reviews) == (1, 0)
    assert h.get_stats()["completed"] == 0
    assert h.sla_compliance_rate() is None
    assert h.decision_log[0]["within_sla"] is None
    h.record_decision(c.internal_id, r.reviewer_id, "override", "Source now reviewed")
    assert c.status.value == "completed"
    assert (r.current_load, r.total_reviews) == (0, 1)
    assert len(h.decision_log) == 2
    assert h.override_rate() == 1.0  # Requests are not completed reviews.


def test_empty_roles_do_not_receive_cases():
    h = HITLOrchestrator()
    r = h.add_reviewer("unscoped")
    c = h.submit_for_review("SYN-002", "lending", {}, "Review")
    assert c.assigned_to is None
    assert c.status.value == "pending"
    assert r.current_load == 0


def test_removed_roles_cannot_decide_and_rejection_is_nonmutating():
    h, r, c = assigned_case()
    r.roles.clear()
    before = copy.deepcopy((asdict(c), asdict(r), h.decision_log))
    with pytest.raises(ValueError, match="not authorized"):
        h.record_decision(c.internal_id, r.reviewer_id, "approve", "No authority")
    assert (asdict(c), asdict(r), h.decision_log) == before


def test_escalation_routes_to_policy_role_without_resetting_deadline():
    h, r, c = assigned_case()
    senior = h.add_reviewer("senior", roles=["credit_authority"])
    h.add_escalation_policy(EscalationPolicy("credit", lambda case: True, "credit_authority"))
    deadline = c.sla_deadline
    h.record_decision(c.internal_id, r.reviewer_id, "escalate", "Authority required")
    assert c.assigned_to == senior.reviewer_id
    assert c.completed_at is None
    assert c.sla_deadline == deadline
    assert (r.current_load, r.total_reviews, senior.current_load) == (0, 0, 1)
    h.record_decision(c.internal_id, senior.reviewer_id, "reject", "Reviewed")
    assert c.status.value == "completed"
    assert senior.total_reviews == 1


def test_unavailable_escalation_role_stays_pending():
    h, r, c = assigned_case()
    h.add_escalation_policy(EscalationPolicy("credit", lambda case: True, "credit_authority"))
    h.record_decision(c.internal_id, r.reviewer_id, "escalate", "Needs senior review")
    assert c.status.value == "pending"
    assert c.assigned_to is None
    assert c.completed_at is None
    assert h.get_stats()["completed"] == 0


def test_unmatched_escalation_remains_open_and_can_be_overdue():
    h, r, c = assigned_case()
    c.sla_deadline = 0
    h.record_decision(c.internal_id, r.reviewer_id, "escalate", "No matching policy")
    assert c.status.value == "escalated"
    assert c.assigned_to is None
    assert c.completed_at is None
    assert c in h.get_overdue_cases()
    assert h.sla_compliance_rate() is None


def test_zero_completed_reviews_do_not_report_perfect_sla():
    assert HITLOrchestrator().sla_compliance_rate() is None


def test_duplicate_registration_cannot_replace_live_reviewer():
    h, r, c = assigned_case()
    with pytest.raises(ValueError, match="already registered"):
        h.add_reviewer(r.reviewer_id, roles=["insurance"])
    assert h.get_reviewer(r.reviewer_id) is r
    assert r.roles == ["lending"] and r.current_load == 1
