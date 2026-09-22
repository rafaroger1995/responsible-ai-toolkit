"""
Model review demo on synthetic lending data.

Runs a small end-to-end workflow with the toolkit's audit, policy, and
human-review components:

  1. Records each model decision in a hash-chained audit log.
  2. Evaluates each decision against configurable policy checks.
  3. Routes decisions that fail a check to an assigned human reviewer,
     who works through the review queue.
  4. Shows the reviewer-authorization and repeat-decision controls.
  5. Verifies the audit chain, then shows that an edited copy fails.

All data is synthetic and generated with a fixed seed. The output shows
software behavior only. It is not a model validation, a deployment, or
evidence of use by any institution.

Usage:
    python examples/model_review_demo.py [output_dir]
"""

from __future__ import annotations

import copy
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

from responsible_ai_toolkit.audit import AuditLogger
from responsible_ai_toolkit.hitl import HITLOrchestrator
from responsible_ai_toolkit.hitl.orchestrator import ReviewDecision
from responsible_ai_toolkit.policy.engine import PolicyEngine

SEED = 20260922
N_APPLICATIONS = 12
MODEL_ID = "synthetic-credit-model-v1"
DTI_REVIEW_LIMIT = 0.45
INPUT_FIELDS = ("requested_amount", "debt_to_income", "credit_history_months")

AVAILABLE_DECISIONS = {d.value for d in ReviewDecision}
APPROVE = "approve"
DECLINE = next(
    (v for v in ("reject", "deny", "decline") if v in AVAILABLE_DECISIONS), None
)
if APPROVE not in AVAILABLE_DECISIONS or DECLINE is None:
    raise SystemExit(
        f"Unexpected review decision values: {sorted(AVAILABLE_DECISIONS)}"
    )


def make_applications(seed: int = SEED, n: int = N_APPLICATIONS) -> List[Dict[str, Any]]:
    """Generate synthetic loan applications with model outputs."""
    rng = random.Random(seed)
    applications = []
    for i in range(1, n + 1):
        applications.append({
            "application_ref": f"SYN-{i:03d}",
            "requested_amount": rng.choice([5000, 10000, 15000, 25000, 40000]),
            "debt_to_income": round(rng.uniform(0.10, 0.55), 2),
            "credit_history_months": rng.randint(6, 240),
            "model_recommendation": "approve" if rng.random() < 0.7 else "decline",
            "confidence": round(rng.uniform(0.40, 0.98), 2),
        })
    # One record with a missing input, to exercise the completeness check.
    applications[4]["credit_history_months"] = None
    return applications


def reviewer_rule(app: Dict[str, Any]) -> tuple:
    """Deterministic stand-in for a human reviewer's judgment."""
    if app["credit_history_months"] is None:
        return DECLINE, "Required input missing; cannot approve without it."
    if app["debt_to_income"] > DTI_REVIEW_LIMIT:
        return DECLINE, f"Debt-to-income above {DTI_REVIEW_LIMIT} on manual review."
    return APPROVE, "Inputs complete and within review guidelines."


def attempt(fn) -> str:
    try:
        fn()
        return "Accepted"
    except ValueError as exc:
        return f"Rejected: {exc}"


def run(output_dir: Path) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    audit = AuditLogger(system_id="synthetic-lending-model")
    engine = PolicyEngine()
    engine.add_policy(
        PolicyEngine.confidence_threshold_policy("min-confidence", threshold=0.60)
    )
    engine.add_policy(
        PolicyEngine.data_completeness_policy(
            "required-inputs",
            required_fields=["debt_to_income", "credit_history_months"],
        )
    )
    hitl = HITLOrchestrator()
    hitl.add_reviewer("reviewer-a", roles=["lending"])
    hitl.add_reviewer("reviewer-b", roles=["lending"])

    rows = []
    queue = []
    for app in make_applications():
        ref = app["application_ref"]
        ai_decision = {
            "recommendation": app["model_recommendation"],
            "confidence": app["confidence"],
        }
        audit.log_prediction(
            model_id=MODEL_ID,
            input_data={k: app[k] for k in INPUT_FIELDS},
            output=ai_decision,
            metadata={"application_ref": ref},
        )

        report = engine.evaluate(app)
        for result in report.results:
            audit.log_policy_check(
                policy_id=result.policy_id,
                result="passed" if result.passed else "failed",
                details={"application_ref": ref, "error": result.error},
            )
        findings = [r.policy_id for r in report.violations]

        row = {
            "application_ref": ref,
            "model_recommendation": app["model_recommendation"],
            "confidence": app["confidence"],
            "policy_findings": findings,
            "route": "automated",
            "reviewer": None,
            "review_decision": None,
            "override": False,
        }
        if findings:
            case = hitl.submit_for_review(
                case_id=ref,
                category="lending",
                ai_decision=ai_decision,
                reason="Failed policy checks: " + ", ".join(findings),
            )
            queue.append((app, ai_decision, case, row))
        rows.append(row)

    # Reviewers work through the queue after all cases are submitted.
    for app, ai_decision, case, row in queue:
        decision, rationale = reviewer_rule(app)
        hitl.record_decision(case.internal_id, case.assigned_to, decision, rationale)
        model_says_approve = app["model_recommendation"] == "approve"
        override = (decision == APPROVE) != model_says_approve
        audit.log_human_review(
            reviewer_id=case.assigned_to,
            decision=decision,
            reasoning=rationale,
            original_prediction=ai_decision,
            override=override,
            metadata={"application_ref": app["application_ref"]},
        )
        row.update(
            route="human review",
            reviewer=case.assigned_to,
            review_decision=decision,
            override=override,
        )

    # Review controls on a separate case.
    ctrl = hitl.submit_for_review(
        case_id="SYN-CONTROL",
        category="lending",
        ai_decision={"recommendation": "approve", "confidence": 0.51},
        reason="Control demonstration",
    )
    other = "reviewer-b" if ctrl.assigned_to == "reviewer-a" else "reviewer-a"
    controls = [
        ("Decision by an unregistered reviewer",
         attempt(lambda: hitl.record_decision(ctrl.internal_id, "reviewer-z", APPROVE, "n/a"))),
        ("Decision by a registered reviewer who is not assigned",
         attempt(lambda: hitl.record_decision(ctrl.internal_id, other, APPROVE, "n/a"))),
        ("Decision by the assigned reviewer",
         attempt(lambda: hitl.record_decision(ctrl.internal_id, ctrl.assigned_to, APPROVE, "Reviewed"))),
        ("Second decision on the completed case",
         attempt(lambda: hitl.record_decision(ctrl.internal_id, ctrl.assigned_to, DECLINE, "Changed mind"))),
    ]

    # Audit-chain verification and a tampered copy.
    chain_ok = audit.verify_chain()
    tampered = copy.deepcopy(audit)
    first_prediction = next(e for e in tampered._entries if e.event_type == "prediction")
    first_prediction.payload["output"]["confidence"] = 0.99
    tampered_ok = tampered.verify_chain()

    results = {
        "rows": rows,
        "controls": controls,
        "audit_entries": audit.size,
        "chain_verified": chain_ok,
        "tampered_copy_verified": tampered_ok,
    }

    expected = ["Rejected", "Rejected", "Accepted", "Rejected"]
    checks_ok = (
        chain_ok
        and not tampered_ok
        and [c[1].split(":")[0] for c in controls] == expected
    )
    results["all_checks_as_expected"] = checks_ok

    (output_dir / "audit_log.json").write_text(audit.export_json())
    (output_dir / "decisions.json").write_text(json.dumps(rows, indent=2))
    (output_dir / "summary.md").write_text(render_summary(results))
    return results


def render_summary(results: Dict[str, Any]) -> str:
    rows = results["rows"]
    sha = os.environ.get("GITHUB_SHA")
    commit = sha[:7] if sha else "local run"
    generated = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())
    reviewed = [r for r in rows if r["route"] == "human review"]
    lines = [
        "# Model review demo (synthetic data)",
        "",
        f"Generated by `examples/model_review_demo.py` at commit `{commit}` on {generated}.",
        "",
        f"All data is synthetic (fixed seed {SEED}). This output shows software "
        "behavior only. It is not a model validation, a deployment, or evidence "
        "of use by any institution.",
        "",
        "## Decisions",
        "",
        "| Application | Model recommendation | Confidence | Policy findings | Route | Reviewer | Review decision |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        lines.append(
            f"| {r['application_ref']} | {r['model_recommendation']} | {r['confidence']:.2f} | "
            f"{', '.join(r['policy_findings']) or 'none'} | {r['route']} | "
            f"{r['reviewer'] or '-'} | "
            f"{(r['review_decision'] or '-') + (' (override)' if r['override'] else '')} |"
        )
    lines += [
        "",
        f"Applications: {len(rows)}. Automated: {len(rows) - len(reviewed)}. "
        f"Routed to human review: {len(reviewed)}. "
        f"Reviewer overrides of the model: {sum(r['override'] for r in reviewed)}.",
        "",
        "Policy checks: minimum model confidence 0.60; required inputs present "
        "(debt-to-income, credit history). Thresholds are illustrative configuration values.",
        "",
        "## Review controls",
        "",
        "| Attempt | Result |",
        "| --- | --- |",
    ]
    for label, outcome in results["controls"]:
        lines.append(f"| {label} | {outcome} |")
    lines += [
        "",
        "## Audit log",
        "",
        f"Entries recorded: {results['audit_entries']}. "
        f"Chain verification: {'passed' if results['chain_verified'] else 'FAILED'}.",
        "",
        "Copy of the log with one prediction's confidence edited: verification "
        f"{'passed (unexpected)' if results['tampered_copy_verified'] else 'failed, as expected'}.",
        "",
        f"All demo checks as expected: {'yes' if results['all_checks_as_expected'] else 'NO'}.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    output_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("demo_output")
    results = run(output_dir)
    print((output_dir / "summary.md").read_text())
    return 0 if results["all_checks_as_expected"] else 1


if __name__ == "__main__":
    sys.exit(main())
