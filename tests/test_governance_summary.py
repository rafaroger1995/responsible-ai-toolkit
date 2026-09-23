from responsible_ai_toolkit.drift.monitor import DriftReport
from responsible_ai_toolkit.governance import GovernanceReporter


def _drift(psi, interpretation):
    return DriftReport(
        psi=psi,
        kl_divergence=0.0,
        wasserstein_distance=0.0,
        psi_interpretation=interpretation,
    )


def test_no_significant_drift_is_not_counted_as_an_alert():
    reporter = GovernanceReporter(system_id="model")
    reporter.add_drift_report(_drift(0.02, "No significant drift"))
    reporter.add_drift_report(_drift(0.15, "Moderate drift — monitor closely"))

    summary = reporter.generate_evidence_package()["executive_summary"]

    assert summary["total_drift_evaluations"] == 2
    assert summary["drift_alerts"] == 0


def test_significant_drift_is_counted_as_an_alert():
    reporter = GovernanceReporter(system_id="model")
    reporter.add_drift_report(_drift(0.02, "No significant drift"))
    reporter.add_drift_report(_drift(0.40, "Significant drift — investigate"))

    summary = reporter.generate_evidence_package()["executive_summary"]

    assert summary["drift_alerts"] == 1


def test_snapshot_drift_status_uses_the_latest_interpretation():
    reporter = GovernanceReporter(system_id="model")
    reporter.add_drift_report(_drift(0.02, "No significant drift"))

    assert reporter.create_snapshot().drift_status == "no_significant_drift"


def test_summary_states_it_is_not_a_compliance_determination():
    package = GovernanceReporter(system_id="model").generate_evidence_package()

    assert package["metadata"]["report_type"] == "Governance summary"
    assert "Not a compliance determination" in package["metadata"]["scope_note"]
    assert package["policy_compliance"]["current_status"] == "not_evaluated"
