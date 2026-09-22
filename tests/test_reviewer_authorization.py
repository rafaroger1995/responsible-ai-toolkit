import copy
from dataclasses import asdict

import pytest

from responsible_ai_toolkit.hitl import HITLOrchestrator


@pytest.mark.parametrize(
    "scenario, expected_error",
    [
        ("unknown", "not registered"),
        ("inactive", "inactive"),
        ("different_reviewer", "Only the assigned reviewer"),
        ("wrong_role", "not authorized"),
    ],
)
def test_unauthorized_decision_leaves_records_unchanged(
    scenario, expected_error
):
    hitl = HITLOrchestrator()
    assigned = hitl.add_reviewer("assigned", roles=["lending"])

    case = hitl.submit_for_review(
        case_id="CASE-001",
        category="lending",
        ai_decision={"recommendation": "approve"},
        reason="Manual review required",
    )
    other = hitl.add_reviewer("other", roles=["lending"])
    assert case.assigned_to == "assigned"

    reviewer_id = "assigned"
    if scenario == "unknown":
        reviewer_id = "unknown"
    elif scenario == "inactive":
        assigned.active = False
    elif scenario == "different_reviewer":
        reviewer_id = "other"
    elif scenario == "wrong_role":
        assigned.roles = ["insurance"]

    case_before = asdict(case)
    assigned_before = asdict(assigned)
    other_before = asdict(other)
    log_before = copy.deepcopy(hitl.decision_log)

    with pytest.raises(ValueError, match=expected_error):
        hitl.record_decision(
            case.internal_id,
            reviewer_id,
            "approve",
            "Reviewed supporting information",
        )

    assert asdict(case) == case_before
    assert asdict(assigned) == assigned_before
    assert asdict(other) == other_before
    assert hitl.decision_log == log_before


def test_active_assigned_reviewer_can_record_decision():
    hitl = HITLOrchestrator()
    reviewer = hitl.add_reviewer("assigned", roles=["lending"])

    case = hitl.submit_for_review(
        case_id="CASE-002",
        category="lending",
        ai_decision={"recommendation": "approve"},
        reason="Manual review required",
    )
    assert case.assigned_to == "assigned"
    assert reviewer.current_load == 1

    result = hitl.record_decision(
        case.internal_id,
        "assigned",
        "approve",
        "Reviewed supporting information",
    )

    assert result.status.value == "completed"
    assert result.decision.value == "approve"
    assert reviewer.current_load == 0
    assert reviewer.total_reviews == 1
    assert len(hitl.decision_log) == 1
    assert hitl.decision_log[0]["reviewer_id"] == "assigned"
