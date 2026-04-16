"""
Example: Fair Lending Compliance Monitoring
============================================

Demonstrates how a community bank can use the Responsible AI Toolkit
to monitor an AI-driven loan approval model for compliance with
fair lending laws (ECOA, FHA) and regulatory guidance (OCC SR 11-7).

This example simulates:
  1. A credit decisioning model processing loan applications
  2. Continuous fairness monitoring across demographic groups
  3. Drift detection when economic conditions shift
  4. HITL escalation for flagged decisions
  5. Policy-as-code compliance checks
  6. Governance evidence package generation

Run:
    python examples/lending_fairness.py
"""

import numpy as np
from responsible_ai_toolkit import (
    FairnessMetrics,
    BiasDetector,
    DriftMonitor,
    AuditLogger,
    HITLOrchestrator,
    PolicyEngine,
    GovernanceReporter,
)
from responsible_ai_toolkit.policy.engine import Policy


def simulate_loan_data(n=2000, seed=42, bias_factor=0.0):
    """Simulate loan application data with optional demographic bias.

    Parameters
    ----------
    n : int
        Number of loan applications.
    seed : int
        Random seed for reproducibility.
    bias_factor : float
        0.0 = fair model, higher = more bias against Group B.
    """
    rng = np.random.RandomState(seed)

    # Demographic groups (e.g., race/ethnicity categories)
    groups = rng.choice(
        ["White", "Black", "Hispanic", "Asian"],
        size=n,
        p=[0.55, 0.20, 0.18, 0.07],
    )

    # True creditworthiness (would they repay?)
    y_true = rng.binomial(1, 0.65, size=n)

    # Model predictions with potential bias
    y_pred = y_true.copy()
    y_prob = np.zeros(n)

    for i in range(n):
        base_prob = 0.85 if y_true[i] == 1 else 0.15

        # Apply bias: reduce approval probability for certain groups
        if groups[i] in ["Black", "Hispanic"] and bias_factor > 0:
            if y_true[i] == 1 and rng.random() < bias_factor:
                y_pred[i] = 0  # False denial
                base_prob *= (1 - bias_factor)

        # Add noise to probability
        y_prob[i] = np.clip(base_prob + rng.normal(0, 0.1), 0.01, 0.99)
        y_pred[i] = 1 if y_prob[i] >= 0.5 else 0

    return y_true, y_pred, y_prob, groups


def main():
    print("=" * 60)
    print("FAIR LENDING COMPLIANCE MONITORING DEMO")
    print("Responsible AI Toolkit — Community Bank Example")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Step 1: Evaluate baseline model fairness
    # ------------------------------------------------------------------
    print("\n--- Step 1: Baseline Fairness Evaluation (Fair Model) ---")
    y_true, y_pred, y_prob, groups = simulate_loan_data(bias_factor=0.0)

    fm = FairnessMetrics(
        y_true=y_true,
        y_pred=y_pred,
        sensitive_attr=groups,
        y_prob=y_prob,
        threshold=0.80,  # Four-fifths rule
    )
    report = fm.full_report()
    print(report.summary())

    # ------------------------------------------------------------------
    # Step 2: Detect bias in a biased model
    # ------------------------------------------------------------------
    print("\n--- Step 2: Biased Model Detection ---")
    y_true_b, y_pred_b, y_prob_b, groups_b = simulate_loan_data(bias_factor=0.35)

    alerts = []
    detector = BiasDetector(
        threshold=0.80,
        window_size=2000,
        on_alert=lambda a: alerts.append(a),
    )
    detector.observe_batch(y_true_b, y_pred_b, groups_b, y_prob_b)
    biased_report = detector.evaluate()
    print(biased_report.summary())
    print(f"\nBias alerts triggered: {len(alerts)}")
    for a in alerts[:3]:
        print(f"  {a}")

    # ------------------------------------------------------------------
    # Step 3: Monitor for model drift
    # ------------------------------------------------------------------
    print("\n--- Step 3: Model Drift Monitoring ---")
    rng = np.random.RandomState(99)
    baseline_scores = rng.normal(0.65, 0.15, 1000)

    # Simulate drift due to economic downturn
    shifted_scores = rng.normal(0.50, 0.20, 1000)

    monitor = DriftMonitor(reference=baseline_scores)
    drift_report = monitor.evaluate(shifted_scores)
    print(drift_report.summary())

    # ------------------------------------------------------------------
    # Step 4: HITL escalation for flagged decisions
    # ------------------------------------------------------------------
    print("\n--- Step 4: HITL Workflow ---")
    hitl = HITLOrchestrator(confidence_threshold=0.65)
    hitl.add_reviewer("senior_analyst", roles=["lending"], max_load=20)
    hitl.add_reviewer("compliance_officer", roles=["lending", "compliance"], max_load=10)

    # Submit flagged low-confidence decisions for review
    flagged_count = 0
    for i in range(len(y_prob_b)):
        if hitl.should_review(y_prob_b[i]):
            case = hitl.submit_for_review(
                case_id=f"LOAN-{i:04d}",
                category="lending",
                priority="high" if y_prob_b[i] < 0.45 else "normal",
                ai_decision={"approved": bool(y_pred_b[i]), "score": float(y_prob_b[i])},
                reason=f"Confidence {y_prob_b[i]:.2f} below threshold",
            )
            flagged_count += 1
            if flagged_count >= 10:
                break

    stats = hitl.get_stats()
    print(f"  Cases submitted for review: {stats['total_cases']}")
    print(f"  Assigned to reviewers:      {stats['assigned']}")
    print(f"  Active reviewers:           {stats['active_reviewers']}")

    # ------------------------------------------------------------------
    # Step 5: Policy-as-code compliance checks
    # ------------------------------------------------------------------
    print("\n--- Step 5: Policy Compliance Evaluation ---")
    engine = PolicyEngine()

    engine.add_policy(PolicyEngine.fairness_threshold_policy(
        "ECOA-001", "dp_ratio_min", threshold=0.80,
        frameworks=["ECOA", "FHA"],
    ))
    engine.add_policy(PolicyEngine.confidence_threshold_policy(
        "MRM-001", threshold=0.60,
        frameworks=["SR-11-7"],
    ))
    engine.add_policy(PolicyEngine.drift_threshold_policy(
        "MRM-002", psi_threshold=0.25,
        frameworks=["SR-11-7", "OCC-MRM"],
    ))
    engine.add_policy(PolicyEngine.data_completeness_policy(
        "DATA-001",
        required_fields=["income", "credit_score", "dti_ratio"],
        frameworks=["NIST-AI-RMF"],
    ))

    # Evaluate against current system state
    dp = biased_report.demographic_parity
    dp_values = list(dp.values())
    dp_ratio_min = min(dp_values) / max(dp_values) if max(dp_values) > 0 else 0

    policy_result = engine.evaluate({
        "dp_ratio_min": dp_ratio_min,
        "confidence": 0.55,
        "psi": drift_report.psi,
        "income": 75000,
        "credit_score": 720,
        "dti_ratio": 0.35,
    })
    print(policy_result.summary())

    # ------------------------------------------------------------------
    # Step 6: Generate governance evidence package
    # ------------------------------------------------------------------
    print("\n--- Step 6: Governance Evidence Package ---")
    reporter = GovernanceReporter(
        system_id="community-bank-lending-v2",
        organization="First Community Bank",
    )
    reporter.add_fairness_report(biased_report)
    reporter.add_drift_report(drift_report)
    reporter.add_policy_report(policy_result)
    reporter.add_note(
        author="Compliance Officer",
        content="Bias detected in Q4 model. Remediation plan initiated.",
        category="fairness",
    )
    snapshot = reporter.create_snapshot(
        hitl_sla_rate=hitl.sla_compliance_rate(),
        hitl_override_rate=hitl.override_rate(),
    )

    evidence = reporter.generate_evidence_package()
    print(f"  System:              {evidence['metadata']['system_id']}")
    print(f"  Fairness status:     {evidence['fairness']['current_status']}")
    print(f"  Drift status:        {evidence['drift']['current_status']}")
    print(f"  Policy status:       {evidence['policy_compliance']['current_status']}")
    print(f"  Analyst notes:       {evidence['executive_summary']['analyst_notes']}")

    # ------------------------------------------------------------------
    # Step 7: Audit trail
    # ------------------------------------------------------------------
    print("\n--- Step 7: Audit Trail Verification ---")
    audit = AuditLogger(system_id="community-bank-lending-v2")
    audit.log_fairness_evaluation(
        evaluation_id="eval-q4-2025",
        metrics=biased_report.demographic_parity,
        violations=biased_report.violations,
    )
    audit.log_drift_alert(
        metric_name="PSI",
        value=drift_report.psi,
        threshold=0.25,
        interpretation=drift_report.psi_interpretation,
    )
    audit.log_policy_check(
        policy_id="ECOA-001",
        result="violation" if not policy_result.all_passed else "pass",
        details={"violations": len(policy_result.violations)},
    )

    print(f"  Audit entries:       {audit.size}")
    print(f"  Chain integrity:     {'VERIFIED' if audit.verify_chain() else 'BROKEN'}")

    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
