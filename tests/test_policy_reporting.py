import pytest

from responsible_ai_toolkit.policy import PolicyEngine
from responsible_ai_toolkit.governance import GovernanceReporter


def test_empty_policy_report_has_no_pass_rate():
    report = PolicyEngine().evaluate({})

    assert report.all_passed is False
    assert report.pass_rate is None
    assert "N/A - no policies evaluated" in report.summary()


def test_no_policy_reports_are_not_evaluated():
    reporter = GovernanceReporter("test-system")

    assert reporter.create_snapshot().policy_status == "not_evaluated"
    package = reporter.generate_evidence_package()
    assert package["policy_compliance"]["current_status"] == "not_evaluated"


def test_empty_policy_evaluation_is_not_evaluated():
    reporter = GovernanceReporter("test-system")
    reporter.add_policy_report(PolicyEngine().evaluate({}))

    assert reporter.create_snapshot().policy_status == "not_evaluated"
    package = reporter.generate_evidence_package()
    policy = package["policy_compliance"]

    assert policy["current_status"] == "not_evaluated"
    assert policy["evaluations"][0]["pass_rate"] is None


@pytest.mark.parametrize(
    "context, expected_status, expected_rate",
    [
        ({}, "evaluation_error", 0.0),
        ({"psi": 0.10}, "checks_passed", 1.0),
        ({"psi": 0.25}, "checks_failed", 0.0),
    ],
)
def test_policy_status_matches_evaluation(
    context, expected_status, expected_rate
):
    engine = PolicyEngine()
    engine.add_policy(engine.drift_threshold_policy("DRIFT-001"))
    report = engine.evaluate(context)

    reporter = GovernanceReporter("test-system")
    reporter.add_policy_report(report)

    assert report.pass_rate == expected_rate
    assert reporter.create_snapshot().policy_status == expected_status
    package = reporter.generate_evidence_package()
    assert package["policy_compliance"]["current_status"] == expected_status


def test_export_does_not_add_timeline_entries():
    reporter = GovernanceReporter("test-system")
    reporter.create_snapshot()

    first = reporter.generate_evidence_package()
    second = reporter.generate_evidence_package()

    assert len(first["governance_timeline"]) == 1
    assert second["governance_timeline"] == first["governance_timeline"]
    assert second["executive_summary"]["governance_snapshots"] == 1
