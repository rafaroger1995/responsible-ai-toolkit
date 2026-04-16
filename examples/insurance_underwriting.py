"""
Example: Insurance Underwriting Fairness & Governance
=====================================================

Demonstrates how a regional insurer can use the Responsible AI Toolkit
to monitor an AI-driven underwriting model for fairness across
policyholder demographics, detect premium scoring drift, and maintain
audit-ready compliance evidence.

This example simulates:
  1. An auto insurance pricing model evaluating risk
  2. Fairness monitoring across age and geographic groups
  3. Premium score drift after a catastrophic weather event
  4. Policy-as-code checks for state insurance regulations
  5. Audit trail for regulatory examination readiness

Run:
    python examples/insurance_underwriting.py
"""

import numpy as np
from responsible_ai_toolkit import (
    FairnessMetrics,
    BiasDetector,
    DriftMonitor,
    AuditLogger,
    PolicyEngine,
    GovernanceReporter,
)
from responsible_ai_toolkit.policy.engine import Policy


def simulate_underwriting_data(n=1500, seed=42, age_bias=False):
    """Simulate insurance underwriting decisions.

    Parameters
    ----------
    n : int
        Number of policy applications.
    seed : int
        Random seed.
    age_bias : bool
        If True, the model unfairly penalizes younger applicants.
    """
    rng = np.random.RandomState(seed)

    # Age groups (protected class in many states)
    age_groups = rng.choice(
        ["18-25", "26-40", "41-55", "56-70"],
        size=n,
        p=[0.20, 0.35, 0.30, 0.15],
    )

    # True risk level (1 = low risk / should approve, 0 = high risk)
    base_rates = {"18-25": 0.55, "26-40": 0.70, "41-55": 0.75, "56-70": 0.65}
    y_true = np.array([
        rng.binomial(1, base_rates[g]) for g in age_groups
    ])

    y_pred = y_true.copy()
    y_prob = np.zeros(n)

    for i in range(n):
        base_prob = 0.80 if y_true[i] == 1 else 0.20

        # Age bias: unfairly deny young applicants
        if age_bias and age_groups[i] == "18-25":
            if y_true[i] == 1 and rng.random() < 0.35:
                y_pred[i] = 0
                base_prob *= 0.5

        y_prob[i] = np.clip(base_prob + rng.normal(0, 0.12), 0.01, 0.99)
        y_pred[i] = 1 if y_prob[i] >= 0.5 else 0

    return y_true, y_pred, y_prob, age_groups


def main():
    print("=" * 60)
    print("INSURANCE UNDERWRITING GOVERNANCE DEMO")
    print("Responsible AI Toolkit — Regional Insurer Example")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Step 1: Baseline fairness across age groups
    # ------------------------------------------------------------------
    print("\n--- Step 1: Baseline Fairness (Unbiased Model) ---")
    y_true, y_pred, y_prob, groups = simulate_underwriting_data(age_bias=False)

    fm = FairnessMetrics(y_true, y_pred, groups, y_prob, threshold=0.80)
    report = fm.full_report()
    print(report.summary())

    # ------------------------------------------------------------------
    # Step 2: Detect age-based bias
    # ------------------------------------------------------------------
    print("\n--- Step 2: Age Bias Detection ---")
    y_true_b, y_pred_b, y_prob_b, groups_b = simulate_underwriting_data(age_bias=True)

    alerts = []
    detector = BiasDetector(
        threshold=0.80,
        on_alert=lambda a: alerts.append(a),
    )
    detector.observe_batch(y_true_b, y_pred_b, groups_b, y_prob_b)
    biased_report = detector.evaluate()
    print(biased_report.summary())
    print(f"\nAlerts triggered: {len(alerts)}")
    for a in alerts[:3]:
        print(f"  {a}")

    # ------------------------------------------------------------------
    # Step 3: Drift after catastrophic weather event
    # ------------------------------------------------------------------
    print("\n--- Step 3: Post-Catastrophe Drift Detection ---")
    rng = np.random.RandomState(55)

    # Normal conditions: risk scores centered around 0.65
    normal_scores = rng.normal(0.65, 0.12, 1000)

    # After hurricane: risk scores shift dramatically
    post_hurricane = np.concatenate([
        rng.normal(0.45, 0.18, 600),   # Many more high-risk scores
        rng.normal(0.70, 0.10, 400),   # Some unaffected areas
    ])

    monitor = DriftMonitor(reference=normal_scores)
    drift_report = monitor.evaluate(post_hurricane)
    print(drift_report.summary())

    # ------------------------------------------------------------------
    # Step 4: Policy compliance
    # ------------------------------------------------------------------
    print("\n--- Step 4: Regulatory Policy Checks ---")
    engine = PolicyEngine()

    engine.add_policy(PolicyEngine.fairness_threshold_policy(
        "INS-FAIR-001", "approval_ratio_min", threshold=0.80,
        frameworks=["State Insurance Reg", "NIST-AI-RMF"],
    ))
    engine.add_policy(PolicyEngine.drift_threshold_policy(
        "INS-DRIFT-001", psi_threshold=0.25,
        frameworks=["SR-11-7", "NIST-AI-RMF"],
    ))
    engine.add_policy(Policy(
        policy_id="INS-RATE-001",
        name="Rate Adequacy Check",
        description="Loss ratio must remain between 0.40 and 0.80 for rate adequacy.",
        rule=lambda ctx: 0.40 <= ctx.get("loss_ratio", 0) <= 0.80,
        severity="critical",
        frameworks=["State Insurance Reg"],
        category="rate_adequacy",
    ))
    engine.add_policy(PolicyEngine.data_completeness_policy(
        "INS-DATA-001",
        required_fields=["vehicle_year", "driver_age", "zip_code", "claims_history"],
        frameworks=["NIST-AI-RMF"],
    ))

    dp = biased_report.demographic_parity
    dp_vals = list(dp.values())
    ratio_min = min(dp_vals) / max(dp_vals) if max(dp_vals) > 0 else 0

    policy_result = engine.evaluate({
        "approval_ratio_min": ratio_min,
        "psi": drift_report.psi,
        "loss_ratio": 0.62,
        "vehicle_year": 2020,
        "driver_age": 34,
        "zip_code": "90210",
        "claims_history": 0,
    })
    print(policy_result.summary())

    # ------------------------------------------------------------------
    # Step 5: Audit trail and evidence package
    # ------------------------------------------------------------------
    print("\n--- Step 5: Audit Trail & Evidence ---")
    audit = AuditLogger(system_id="auto-underwriting-v3")

    # Log the fairness evaluation
    audit.log_fairness_evaluation(
        evaluation_id="eval-2025-q4",
        metrics=biased_report.demographic_parity,
        violations=biased_report.violations,
    )

    # Log drift alert
    audit.log_drift_alert(
        metric_name="PSI",
        value=drift_report.psi,
        threshold=0.25,
        interpretation=drift_report.psi_interpretation,
    )

    # Log a sample underwriting decision
    audit.log_prediction(
        model_id="auto-risk-v3",
        input_data={"driver_age": 22, "vehicle_year": 2019, "zip": "33139"},
        output={"risk_score": 0.42, "approved": False},
        metadata={"policy_type": "auto", "state": "FL"},
    )

    # Log human override
    audit.log_human_review(
        reviewer_id="underwriter_smith",
        decision="override",
        reasoning="Applicant has clean 5-year record; model penalizing zip code unfairly",
        original_prediction={"risk_score": 0.42, "approved": False},
        override=True,
    )

    print(f"  Audit entries:   {audit.size}")
    print(f"  Chain integrity: {'VERIFIED' if audit.verify_chain() else 'BROKEN'}")

    # Generate governance report
    reporter = GovernanceReporter(
        system_id="auto-underwriting-v3",
        organization="Regional Mutual Insurance",
    )
    reporter.add_fairness_report(biased_report)
    reporter.add_drift_report(drift_report)
    reporter.add_policy_report(policy_result)
    reporter.add_note(
        author="Chief Actuary",
        content="Post-hurricane drift requires model recalibration for FL/TX zip codes.",
        category="drift",
    )
    reporter.create_snapshot()

    evidence = reporter.generate_evidence_package()
    print(f"\n  Evidence package generated for: {evidence['metadata']['organization']}")
    print(f"  Fairness evaluations: {evidence['executive_summary']['total_fairness_evaluations']}")
    print(f"  Drift alerts:         {evidence['executive_summary']['drift_alerts']}")

    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
