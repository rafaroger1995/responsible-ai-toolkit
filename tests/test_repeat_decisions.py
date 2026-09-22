import copy
from dataclasses import asdict

import pytest

from responsible_ai_toolkit.hitl import HITLOrchestrator


def _completed_case():
    hitl = HITLOrchestrator()
    assigned = hitl.add_reviewer("assigned", roles=["lending"])
    case = hitl.submit_for_review(
        case_id="CASE-003",
        category="lending",
        ai_decision={"recommendation": "approve"},
        reason="Manual review required",
    )
    other = hitl.add_reviewer("other", roles=["lending"])
    hitl.record_decision(
        case.internal_id,
        "assigned",
        "approve",
        "Reviewed supporting information",
    )
    return hitl, assigned, other, case


@pytest.mark.parametrize("second_reviewer", ["assigned", "other"])
def test_second_decision_on_completed_case_changes_nothing(second_reviewer):
    hitl, assigned, other, case = _completed_case()

    case_before = asdict(case)
    assigned_before = asdict(assigned)
    other_before = asdict(other)
    log_before = copy.deepcopy(hitl.decision_log)

    with pytest.raises(ValueError, match="completed case"):
        hitl.record_decision(
            case.internal_id,
            second_reviewer,
            "approve",
            "Attempted second decision",
        )

    assert asdict(case) == case_before
    assert asdict(assigned) == assigned_before
    assert asdict(other) == other_before
    assert hitl.decision_log == log_before
    assert assigned.total_reviews == 1
    assert len(hitl.decision_log) == 1
