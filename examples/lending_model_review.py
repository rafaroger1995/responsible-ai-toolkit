"""
Example: Lending Model Review
=============================

Shows how a lender's model-risk or operations team could use the
Responsible AI Toolkit to review a credit-decision model over one
monitoring cycle. All data is synthetic.

This example walks through:
  1. Scoring synthetic loan applications with a stand-in model
  2. Checking whether the model's score distribution has drifted
  3. Applying configurable policy checks to the portfolio
  4. Routing low-confidence decisions to human reviewers
  5. Collecting the results into a governance summary
  6. Recording each step in a hash-chained audit log

The organization and data are fictional. Thresholds are illustrative
configuration values, not legal or supervisory standards, and the output
is not a model validation or a compliance determination.

Run:
    python examples/lending_model_review.py
"""

import numpy as np

from responsible_ai_toolkit import (
    AuditLogger,
    DriftMonitor,
    GovernanceReporter,
    HITLOrchestrator,
    PolicyEngine,
)

SEED = 42
CONFIDENCE_FOR_REVIEW = 0.65
MAX_CASES = 10


def simulate_scores(n, seed, shift=0.0):
    """Synthetic model confidence scores for n applications.

    ``shift`` moves the whole distribution down, standing in for a change
    in the applicant population or in upstream data.
    """
    rng = np.random.RandomState(seed)
    scores = rng.beta(5, 2, size=n) - shift
    return np.clip(scores, 0.01, 0.99)


def main():
    print("=" * 60)
    print("LENDING MODEL REVIEW DEMO (synthetic data)")
    print("Responsible AI Toolkit: Example Community Lender (fictional)")
    print("=" * 60)

    audit = AuditLogger(system_id="example-lending-model-v2")

    # ------------------------------------------------------------------
    # Step 1: Score applications
    # ------------------------------------------------------------------
    print("\n--- Step 1: Score applications ---")
    reference_scores = simulate_scores(2000, seed=SEED)          # validation period
    current_scores = simulate_scores(1000, seed=SEED + 1, shift=0.08)  # this cycle
    print(f"  Validation-period scores: {len(reference_scores)}")
    print(f"  Current-cycle scores:     {len(current_scores)}")

    # ------------------------------------------------------------------
    # Step 2: Drift monitoring
    # ------------------------------------------------------------------
    print("\n--- Step 2: Score drift ---")
    monitor = DriftMonitor(reference=reference_scores)
    drift_report = monitor.evaluate(current_scores)
    print(drift_report.summary())
    audit.log_drift_alert(
        metric_name="psi",
        value=drift_report.psi,
        threshold=0.25,
        interpretation=drift_report.psi_interpretation,
    )

    # ------------------------------------------------------------------
    # Step 3: Policy checks on the portfolio
    # ------------------------------------------------------------------
    print("\n--- Step 3: Policy checks ---")
    engine = PolicyEngine()
    engine.add_policy(PolicyEngine.drift_threshold_policy("MRM-002", psi_threshold=0.25))
    engine.add_policy(PolicyEngine.data_completeness_policy(
        "DATA-001", required_fields=["income", "credit_score", "dti_ratio"],
    ))
    policy_result = engine.evaluate({
        "psi": drift_report.psi,
        "income": 75000,
        "credit_score": 720,
        "dti_ratio": 0.35,
    })
    print(policy_result.summary())
    for result in policy_result.results:
        audit.log_policy_check(
            policy_id=result.policy_id,
            result="passed" if result.passed else "failed",
            details={"error": result.error},
        )

    # ------------------------------------------------------------------
    # Step 4: Human review of low-confidence decisions
    # ------------------------------------------------------------------
    print("\n--- Step 4: Human review ---")
    hitl = HITLOrchestrator(confidence_threshold=CONFIDENCE_FOR_REVIEW)
    hitl.add_reviewer("senior_analyst", roles=["lending"], max_load=20)
    hitl.add_reviewer("credit_officer", roles=["lending"], max_load=10)

    cases = []
    for i, score in enumerate(current_scores):
        if hitl.should_review(score):
            case = hitl.submit_for_review(
                case_id=f"LOAN-{i:04d}",
                category="lending",
                priority="high" if score < 0.45 else "normal",
                ai_decision={"approved": bool(score >= 0.5), "score": float(score)},
                reason=f"Confidence {score:.2f} below {CONFIDENCE_FOR_REVIEW}",
            )
            cases.append((case, float(score)))
            if len(cases) >= MAX_CASES:
                break

    for case, score in cases:
        # Stand-in for reviewer judgment: approve only above 0.50.
        decision = "approve" if score >= 0.50 else "reject"
        rationale = "Reviewed supporting documents."
        hitl.record_decision(case.internal_id, case.assigned_to, decision, rationale)
        audit.log_human_review(
            reviewer_id=case.assigned_to,
            decision=decision,
            reasoning=rationale,
            original_prediction={"score": score},
            override=(decision == "approve") != (score >= 0.5),
            metadata={"case_id": case.case_id},
        )

    stats = hitl.get_stats()
    print(f"  Cases submitted for review: {stats['total_cases']}")
    print(f"  Assigned to reviewers:      {stats['assigned']}")
    print(f"  Active reviewers:           {stats['active_reviewers']}")

    # ------------------------------------------------------------------
    # Step 5: Governance summary
    # ------------------------------------------------------------------
    print("\n--- Step 5: Governance summary ---")
    reporter = GovernanceReporter(
        system_id="example-lending-model-v2",
        organization="Example Community Lender (fictional)",
    )
    reporter.add_drift_report(drift_report)
    reporter.add_policy_report(policy_result)
    reporter.add_note(
        author="Model risk analyst",
        content="Score distribution shifted this cycle; investigating upstream data changes.",
        category="monitoring",
    )
    reporter.create_snapshot(
        hitl_sla_rate=hitl.sla_compliance_rate(),
        hitl_override_rate=hitl.override_rate(),
    )
    summary = reporter.generate_evidence_package()
    print(f"  System:          {summary['metadata']['system_id']}")
    print(f"  Drift status:    {summary['drift']['current_status']}")
    print(f"  Policy checks:   {summary['policy_compliance']['current_status']}")
    print(f"  Drift alerts:    {summary['executive_summary']['drift_alerts']}")
    print(f"  Analyst notes:   {summary['executive_summary']['analyst_notes']}")

    # ------------------------------------------------------------------
    # Step 6: Audit trail
    # ------------------------------------------------------------------
    print("\n--- Step 6: Audit trail ---")
    print(f"  Audit entries:   {audit.size}")
    print(f"  Chain integrity: {'VERIFIED' if audit.verify_chain() else 'BROKEN'}")
    print(f"  Chain head hash: {audit.entries[-1].entry_hash}")

    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
