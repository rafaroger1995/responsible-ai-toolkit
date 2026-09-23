"""
Model review demo on synthetic lending, insurance, and fraud-alert data,
with month-by-month drift monitoring of the lending model.

Runs the same toolkit components, unchanged, on three synthetic scenarios:
consumer lending decisions, small-business property insurance quotes, and
transaction fraud alerts at a credit union. For each scenario it:

  1. Records each model decision in a hash-chained audit log.
  2. Evaluates each decision against configurable policy checks.
  3. Routes decisions that fail a check to an assigned human reviewer,
     who works through the review queue.
  4. Shows the reviewer-authorization and repeat-decision controls.
  5. Verifies the audit chain, then shows that an edited copy fails.
  6. Traces one decision through every audit entry that concerns it.

A separate monitoring run compares the lending model's scores each month
with its validation-period scores, applies a drift policy, and escalates a
failed check to a model-risk reviewer, all recorded in an audit log.

Only the scenario configuration differs between the runs: the data,
the rule settings, the reviewer roles, and a stand-in for reviewer
judgment. The toolkit code is the same for both.

All data is synthetic and generated with fixed seeds. The output shows
software behavior only. It is not a model validation, a deployment, or
evidence of use by any institution.

Writes to the output directory: index.html (comparison page), one HTML
review record per scenario, the raw decisions, audit logs, and audit
evidence packages as JSON, and summary.md.

Usage:
    python examples/model_review_demo.py [output_dir]
"""

from __future__ import annotations

import copy
import html
import json
import os
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import numpy as np

from responsible_ai_toolkit.audit import AuditLogger
from responsible_ai_toolkit.drift import DriftMonitor
from responsible_ai_toolkit.hitl import HITLOrchestrator
from responsible_ai_toolkit.hitl.orchestrator import ReviewDecision
from responsible_ai_toolkit.policy.engine import Policy, PolicyEngine

AVAILABLE_DECISIONS = {d.value for d in ReviewDecision}
APPROVE = "approve"
DECLINE = next(
    (v for v in ("reject", "deny", "decline") if v in AVAILABLE_DECISIONS), None
)
if APPROVE not in AVAILABLE_DECISIONS or DECLINE is None:
    raise SystemExit(
        f"Unexpected review decision values: {sorted(AVAILABLE_DECISIONS)}"
    )


# ----------------------------------------------------------------------
# Scenario configuration
# ----------------------------------------------------------------------

@dataclass
class Scenario:
    """Everything that differs between runs. The toolkit code does not."""

    key: str
    title: str
    short: str
    record_label: str
    category: str
    system_id: str
    model_id: str
    seed: int
    reviewers: List[str]
    input_fields: Tuple[str, ...]
    required_fields: List[str]
    confidence_threshold: float
    positive_recommendation: str
    make_records: Callable[[int], List[Dict[str, Any]]]
    reviewer_rule: Callable[[Dict[str, Any]], Tuple[str, str]]
    tamper_target: str
    rules_text: str
    extra_policies: List[Policy] = field(default_factory=list)
    decision_labels: Dict[str, str] = field(default_factory=dict)

    def label(self, decision: Any) -> str:
        """Display name for a reviewer decision in this scenario's vocabulary."""
        if not decision:
            return "-"
        return self.decision_labels.get(decision, decision)


def lending_records(seed: int) -> List[Dict[str, Any]]:
    """Synthetic consumer loan applications with model outputs."""
    rng = random.Random(seed)
    records = []
    for i in range(1, 13):
        records.append({
            "ref": f"SYN-{i:03d}",
            "requested_amount": rng.choice([5000, 10000, 15000, 25000, 40000]),
            "debt_to_income": round(rng.uniform(0.10, 0.55), 2),
            "credit_history_months": rng.randint(6, 240),
            "model_recommendation": "approve" if rng.random() < 0.7 else "decline",
            "confidence": round(rng.uniform(0.40, 0.98), 2),
        })
    # One record with a missing input, to exercise the completeness check.
    records[4]["credit_history_months"] = None
    return records


def lending_reviewer(record: Dict[str, Any]) -> Tuple[str, str]:
    """Deterministic stand-in for a loan reviewer's judgment."""
    if record["credit_history_months"] is None:
        return DECLINE, "Required input missing; cannot approve without it."
    if record["debt_to_income"] > 0.45:
        return DECLINE, "Debt-to-income above 0.45 on manual review."
    return APPROVE, "Inputs complete and within review guidelines."


def insurance_records(seed: int) -> List[Dict[str, Any]]:
    """Synthetic small-business property insurance quotes with model outputs."""
    rng = random.Random(seed)
    records = []
    for i in range(1, 13):
        records.append({
            "ref": f"INS-{i:03d}",
            "coverage_amount": rng.choice([250000, 500000, 750000, 1200000, 2000000]),
            "prior_claims": rng.randint(0, 4),
            "building_age_years": rng.randint(1, 80),
            "last_inspection_year": rng.randint(2016, 2026),
            "model_recommendation": "offer" if rng.random() < 0.75 else "decline",
            "confidence": round(rng.uniform(0.45, 0.99), 2),
        })
    # One record with a missing input, to exercise the completeness check.
    records[2]["last_inspection_year"] = None
    return records


def insurance_reviewer(record: Dict[str, Any]) -> Tuple[str, str]:
    """Deterministic stand-in for an underwriter's judgment."""
    if record["last_inspection_year"] is None:
        return DECLINE, "No inspection on file; cannot offer without it."
    if record["prior_claims"] >= 3:
        return DECLINE, "Three or more prior claims on manual review."
    return APPROVE, "Inputs complete and within underwriting guidelines."


def fraud_records(seed: int) -> List[Dict[str, Any]]:
    """Synthetic transaction alerts at a credit union with model outputs."""
    rng = random.Random(seed)
    records = []
    for i in range(1, 13):
        records.append({
            "ref": f"TXN-{i:03d}",
            "amount": rng.choice([180, 950, 2400, 6500, 12500, 18000]),
            "channel": rng.choice(["card", "online transfer", "wire"]),
            "account_age_days": rng.randint(5, 3000),
            "device_known": rng.random() < 0.7,
            "model_recommendation": "clear" if rng.random() < 0.65 else "hold",
            "confidence": round(rng.uniform(0.50, 0.99), 2),
        })
    # One record with a missing input, to exercise the completeness check.
    records[7]["account_age_days"] = None
    return records


def fraud_reviewer(record: Dict[str, Any]) -> Tuple[str, str]:
    """Deterministic stand-in for a fraud analyst's judgment."""
    if record["account_age_days"] is None:
        return DECLINE, "Account details missing; hold until verified."
    if not record["device_known"] and record["amount"] > 5000:
        return DECLINE, "Large amount from an unrecognized device; hold and contact member."
    return APPROVE, "Activity consistent with the member's history; release."


SCENARIOS = [
    Scenario(
        key="lending",
        title="Consumer lending decisions",
        short="consumer lending decisions at a community bank",
        record_label="Application",
        category="lending",
        system_id="synthetic-lending-model",
        model_id="synthetic-credit-model-v1",
        seed=20260922,
        reviewers=["reviewer-a", "reviewer-b"],
        input_fields=("requested_amount", "debt_to_income", "credit_history_months"),
        required_fields=["debt_to_income", "credit_history_months"],
        confidence_threshold=0.60,
        positive_recommendation="approve",
        make_records=lending_records,
        reviewer_rule=lending_reviewer,
        tamper_target="SYN-006",
        rules_text=(
            "model confidence of at least 0.60, and required inputs present "
            "(debt-to-income, credit history)"
        ),
    ),
    Scenario(
        key="insurance",
        title="Small-business property insurance quotes",
        short="property insurance quotes at a small insurer",
        record_label="Quote",
        category="insurance",
        system_id="synthetic-underwriting-model",
        model_id="synthetic-property-model-v1",
        seed=20260923,
        reviewers=["underwriter-a", "underwriter-b"],
        input_fields=(
            "coverage_amount", "prior_claims", "building_age_years", "last_inspection_year",
        ),
        required_fields=["prior_claims", "last_inspection_year"],
        confidence_threshold=0.70,
        positive_recommendation="offer",
        make_records=insurance_records,
        reviewer_rule=insurance_reviewer,
        tamper_target="INS-006",
        rules_text=(
            "model confidence of at least 0.70, required inputs present "
            "(prior claims, last inspection year), and coverage above "
            "1,000,000 referred for review"
        ),
        extra_policies=[
            Policy(
                policy_id="high-coverage-referral",
                name="High coverage referral",
                description="Quotes with coverage above 1,000,000 are referred for review.",
                rule=lambda ctx: ctx.get("coverage_amount", 0) <= 1_000_000,
                severity="warning",
                category="underwriting",
            )
        ],
        decision_labels={APPROVE: "offer", DECLINE: "decline"},
    ),
    Scenario(
        key="fraud",
        title="Transaction fraud alerts",
        short="fraud alerts at a credit union",
        record_label="Alert",
        category="fraud",
        system_id="synthetic-fraud-model",
        model_id="synthetic-transaction-model-v1",
        seed=20260924,
        reviewers=["analyst-a", "analyst-b"],
        input_fields=("amount", "channel", "account_age_days", "device_known"),
        required_fields=["account_age_days", "device_known"],
        confidence_threshold=0.75,
        positive_recommendation="clear",
        make_records=fraud_records,
        reviewer_rule=fraud_reviewer,
        tamper_target="TXN-006",
        rules_text=(
            "model confidence of at least 0.75, required inputs present "
            "(account age, device recognition), and transactions above "
            "10,000 held for analyst review"
        ),
        extra_policies=[
            Policy(
                policy_id="large-transaction-review",
                name="Large transaction review",
                description="Transactions above 10,000 are held for analyst review.",
                rule=lambda ctx: ctx.get("amount", 0) <= 10_000,
                severity="warning",
                category="fraud",
            )
        ],
        decision_labels={APPROVE: "release", DECLINE: "hold"},
    ),
]


# ----------------------------------------------------------------------
# Running a scenario (identical toolkit calls for every scenario)
# ----------------------------------------------------------------------

def attempt(fn) -> str:
    try:
        fn()
        return "Accepted"
    except ValueError as exc:
        return f"Rejected: {exc}"


def run_scenario(sc: Scenario) -> Dict[str, Any]:
    audit = AuditLogger(system_id=sc.system_id)
    engine = PolicyEngine()
    engine.add_policy(
        PolicyEngine.confidence_threshold_policy(
            "min-confidence", threshold=sc.confidence_threshold
        )
    )
    engine.add_policy(
        PolicyEngine.data_completeness_policy(
            "required-inputs", required_fields=sc.required_fields
        )
    )
    for policy in sc.extra_policies:
        engine.add_policy(policy)

    hitl = HITLOrchestrator()
    for reviewer in sc.reviewers:
        hitl.add_reviewer(reviewer, roles=[sc.category])

    rows = []
    queue = []
    for record in sc.make_records(sc.seed):
        ref = record["ref"]
        ai_decision = {
            "recommendation": record["model_recommendation"],
            "confidence": record["confidence"],
        }
        audit.log_prediction(
            model_id=sc.model_id,
            input_data={k: record[k] for k in sc.input_fields},
            output=ai_decision,
            metadata={"record_ref": ref},
        )

        report = engine.evaluate(record)
        for result in report.results:
            audit.log_policy_check(
                policy_id=result.policy_id,
                result="passed" if result.passed else "failed",
                details={"record_ref": ref, "error": result.error},
            )
        findings = [r.policy_id for r in report.violations]

        row = {
            "record_ref": ref,
            "model_recommendation": record["model_recommendation"],
            "confidence": record["confidence"],
            "policy_findings": findings,
            "route": "automated",
            "reviewer": None,
            "review_decision": None,
            "override": False,
        }
        if findings:
            case = hitl.submit_for_review(
                case_id=ref,
                category=sc.category,
                ai_decision=ai_decision,
                reason="Failed policy checks: " + ", ".join(findings),
            )
            queue.append((record, ai_decision, case, row))
        rows.append(row)

    # Reviewers work through the queue after all cases are submitted.
    for record, ai_decision, case, row in queue:
        decision, rationale = sc.reviewer_rule(record)
        hitl.record_decision(case.internal_id, case.assigned_to, decision, rationale)
        model_positive = record["model_recommendation"] == sc.positive_recommendation
        override = (decision == APPROVE) != model_positive
        audit.log_human_review(
            reviewer_id=case.assigned_to,
            decision=decision,
            reasoning=rationale,
            original_prediction=ai_decision,
            override=override,
            metadata={"record_ref": record["ref"]},
        )
        row.update(
            route="human review",
            reviewer=case.assigned_to,
            review_decision=decision,
            override=override,
        )

    # Review controls on a separate case.
    ctrl = hitl.submit_for_review(
        case_id=f"{sc.key.upper()}-CONTROL",
        category=sc.category,
        ai_decision={"recommendation": sc.positive_recommendation, "confidence": 0.51},
        reason="Control demonstration",
    )
    other = next(r for r in sc.reviewers if r != ctrl.assigned_to)
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
    tamper_index = next(
        i for i, e in enumerate(tampered._entries)
        if e.event_type == "prediction"
        and e.payload["metadata"].get("record_ref") == sc.tamper_target
    )
    original = tampered._entries[tamper_index].payload["output"]["confidence"]
    tamper_value = 0.99 if original != 0.99 else 0.10
    tampered._entries[tamper_index].payload["output"]["confidence"] = tamper_value
    tampered_ok = tampered.verify_chain()

    expected = ["Rejected", "Rejected", "Accepted", "Rejected"]
    checks_ok = (
        chain_ok
        and not tampered_ok
        and [c[1].split(":")[0] for c in controls] == expected
    )
    # Trace one decision: prefer the first reviewer override, else the first review.
    reviewed_refs = [r["record_ref"] for r in rows if r["route"] == "human review"]
    override_refs = [r["record_ref"] for r in rows if r["override"]]
    trace_ref = (override_refs or reviewed_refs or [rows[0]["record_ref"]])[0]
    trace = []
    for i, entry in enumerate(audit.entries):
        meta = entry.payload.get("metadata", {}) if entry.event_type != "policy_check" else {}
        details = entry.payload.get("details", {}) if entry.event_type == "policy_check" else {}
        if trace_ref in (meta.get("record_ref"), details.get("record_ref")):
            trace.append((i, entry))

    return {
        "scenario": sc,
        "rows": rows,
        "trace_ref": trace_ref,
        "trace": trace,
        "evidence_package": audit.export_evidence_package(),
        "controls": controls,
        "rule_count": len(engine.list_policies()),
        "audit": audit,
        "audit_entries": audit.size,
        "chain_verified": chain_ok,
        "tampered_copy_verified": tampered_ok,
        "tamper_index": tamper_index,
        "tamper_from": original,
        "tamper_value": tamper_value,
        "event_types": [e.event_type for e in audit.entries],
        "head_hash": audit.entries[-1].entry_hash,
        "all_checks_as_expected": checks_ok,
    }


# ----------------------------------------------------------------------
# Ongoing monitoring: drift in the lending model's scores
# ----------------------------------------------------------------------

MONITORING_SEED = 20260925
MONITORING_PERIODS = ["Month 1", "Month 2", "Month 3", "Month 4", "Month 5", "Month 6"]
# Synthetic downward shift in score distribution, per month.
MONITORING_SHIFTS = [0.0, 0.01, 0.02, 0.04, 0.06, 0.075]
PSI_LOW, PSI_HIGH = 0.10, 0.25
MONITORING_LABELS = {APPROVE: "continue automation", DECLINE: "pause automated approvals"}


def run_monitoring() -> Dict[str, Any]:
    rng = np.random.default_rng(MONITORING_SEED)
    reference = rng.beta(5, 3, 4000)  # model scores from the validation period
    monitor = DriftMonitor(reference=reference, psi_thresholds=(PSI_LOW, PSI_HIGH))

    engine = PolicyEngine()
    engine.add_policy(PolicyEngine.drift_threshold_policy("score-drift", psi_threshold=PSI_HIGH))

    audit = AuditLogger(system_id="synthetic-lending-model-monitoring")
    hitl = HITLOrchestrator()
    for reviewer in ("model-risk-a", "model-risk-b"):
        hitl.add_reviewer(reviewer, roles=["model-risk"])

    periods = []
    for period, shift in zip(MONITORING_PERIODS, MONITORING_SHIFTS):
        scores = np.clip(rng.beta(5, 3, 1500) - shift, 0.0, 1.0)
        report = monitor.evaluate(scores)
        row = {
            "period": period,
            "scored": int(scores.size),
            "psi": report.psi,
            "kl": report.kl_divergence,
            "wasserstein": report.wasserstein_distance,
            "interpretation": report.psi_interpretation,
            "policy": "passed",
            "action": "Routine monitoring",
            "reviewer": None,
            "decision": None,
        }

        alert = None
        if report.psi >= PSI_LOW:
            alert = audit.log_drift_alert(
                metric_name="psi",
                value=round(report.psi, 4),
                threshold=PSI_HIGH,
                interpretation=report.psi_interpretation,
                metadata={"period": period},
            )
            row["action"] = "Monitor closely"

        evaluation = engine.evaluate({"psi": report.psi, "period": period})
        for result in evaluation.results:
            audit.log_policy_check(
                policy_id=result.policy_id,
                result="passed" if result.passed else "failed",
                details={"period": period, "psi": round(report.psi, 4), "error": result.error},
            )

        if evaluation.violations:
            row["policy"] = "failed"
            case = hitl.submit_for_review(
                case_id=f"DRIFT-{period.replace(' ', '-').upper()}",
                category="model-risk",
                ai_decision={"psi": round(report.psi, 4), "interpretation": report.psi_interpretation},
                reason=f"Score PSI at or above {PSI_HIGH}",
            )
            audit.log_escalation(
                reason=f"Score PSI {report.psi:.3f} at or above {PSI_HIGH}",
                source_entry_id=alert.entry_id if alert else "",
                escalated_to=case.assigned_to,
                priority="high",
                metadata={"period": period},
            )
            rationale = ("Route applications to manual review until the score "
                         "shift is investigated.")
            hitl.record_decision(case.internal_id, case.assigned_to, DECLINE, rationale)
            audit.log_human_review(
                reviewer_id=case.assigned_to,
                decision=DECLINE,
                reasoning=rationale,
                original_prediction={"psi": round(report.psi, 4)},
                override=False,
                metadata={"period": period},
            )
            row.update(action="Escalated", reviewer=case.assigned_to, decision=DECLINE)
        periods.append(row)

    chain_ok = audit.verify_chain()
    tampered = copy.deepcopy(audit)
    tamper_index = next(i for i, e in enumerate(tampered._entries) if e.event_type == "drift_alert")
    original = tampered._entries[tamper_index].payload["value"]
    tamper_value = 0.05
    tampered._entries[tamper_index].payload["value"] = tamper_value
    tampered_ok = tampered.verify_chain()

    escalated = [r for r in periods if r["action"] == "Escalated"]
    trace_period = escalated[0]["period"] if escalated else periods[-1]["period"]
    trace = []
    for i, entry in enumerate(audit.entries):
        meta = entry.payload.get("metadata", {}) or {}
        details = entry.payload.get("details", {}) or {}
        if trace_period in (meta.get("period"), details.get("period")):
            trace.append((i, entry))

    checks_ok = (
        chain_ok
        and not tampered_ok
        and all(r["psi"] < PSI_LOW for r in periods[:3])
        and periods[-1]["psi"] >= PSI_HIGH
        and periods[-1]["action"] == "Escalated"
        and all(r["action"] != "Escalated" for r in periods[:-1])
    )
    return {
        "periods": periods,
        "audit": audit,
        "audit_entries": audit.size,
        "event_types": [e.event_type for e in audit.entries],
        "head_hash": audit.entries[-1].entry_hash,
        "chain_verified": chain_ok,
        "tampered_copy_verified": tampered_ok,
        "tamper_index": tamper_index,
        "tamper_from": original,
        "tamper_value": tamper_value,
        "trace_period": trace_period,
        "trace": trace,
        "evidence_package": audit.export_evidence_package(),
        "all_checks_as_expected": checks_ok,
    }


def provenance() -> Dict[str, Any]:
    """Describe where this run came from, using GitHub Actions variables if present."""
    sha = os.environ.get("GITHUB_SHA")
    server = os.environ.get("GITHUB_SERVER_URL", "https://github.com")
    repo = os.environ.get("GITHUB_REPOSITORY")
    run_id = os.environ.get("GITHUB_RUN_ID")
    run_number = os.environ.get("GITHUB_RUN_NUMBER")
    return {
        "commit": sha[:7] if sha else "local run",
        "commit_url": f"{server}/{repo}/commit/{sha}" if sha and repo else None,
        "run_label": f"GitHub Actions run #{run_number}" if run_number else None,
        "run_url": f"{server}/{repo}/actions/runs/{run_id}" if run_id and repo else None,
        "repo_url": f"{server}/{repo}" if repo else None,
        "generated": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()),
    }


# ----------------------------------------------------------------------
# Output
# ----------------------------------------------------------------------

REPORT_CSS = """
:root {
  --paper: #f3f6f5; --surface: #ffffff; --ink: #18212f; --muted: #566171;
  --rule: #d5dce2; --verified: #127a5b; --broken: #b42318; --unchecked: #b9c2cc;
  --prediction: #2f4b7c; --policy: #8a9ab0; --human: #a86a00;
  --drift: #6d5bb3; --escalation: #c2410c;
  --serif: Charter, "Bitstream Charter", "Iowan Old Style", "Sitka Text", Cambria, Georgia, serif;
  --sans: system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
  color-scheme: light dark;
}
@media (prefers-color-scheme: dark) {
  :root {
    --paper: #0f141b; --surface: #161d27; --ink: #e5e9ef; --muted: #9ba6b5;
    --rule: #2a3442; --verified: #3cbf8e; --broken: #f07268; --unchecked: #3a4452;
    --prediction: #86a3dc; --policy: #5d6c81; --human: #e3a73e;
    --drift: #a898e8; --escalation: #f08a4b;
  }
}
* { box-sizing: border-box; }
html { -webkit-text-size-adjust: 100%; }
body { margin: 0; background: var(--paper); color: var(--ink); font: 16px/1.55 var(--sans); }
main { max-width: 72rem; margin: 0 auto; padding: 2.5rem 1.25rem 4rem; }
h1, h2 { font-family: var(--serif); font-weight: 600; line-height: 1.2; }
h1 { font-size: clamp(2rem, 5vw, 2.75rem); margin: 0 0 .5rem; letter-spacing: -.01em; }
h2 { font-size: 1.45rem; margin: 3rem 0 .75rem; padding-top: 1.25rem; border-top: 1px solid var(--rule); }
p { max-width: 42rem; margin: .5rem 0; }
a { color: inherit; text-underline-offset: 3px; }
a:focus-visible { outline: 2px solid var(--prediction); outline-offset: 2px; }
code { font-size: .92em; word-break: break-all; }
.provenance { color: var(--muted); margin-bottom: 1.25rem; }
.notice { border-left: 3px solid var(--human); padding: .25rem 0 .25rem .9rem; color: var(--muted); }
.lede { font-family: var(--serif); font-size: 1.3rem; line-height: 1.45; max-width: 44rem; margin-top: 1.75rem; }
.chain { display: flex; flex-wrap: wrap; gap: 3px; margin: 1rem 0 .5rem; padding: 0; list-style: none; }
.chain li { width: 16px; height: 22px; border-radius: 2px; position: relative; }
.chain li.prediction { background: var(--prediction); }
.chain li.policy_check { background: var(--policy); }
.chain li.human_review { background: var(--human); }
.chain li.drift_alert { background: var(--drift); }
.chain li.escalation { background: var(--escalation); }
.psi-chart { width: 100%; max-width: 44rem; height: auto; display: block; margin: 1rem 0 .25rem; }
.psi-chart text { fill: var(--muted); font: 13px var(--sans); }
.psi-chart .value { fill: var(--ink); font-weight: 600; }
.psi-chart .limit { stroke: var(--muted); stroke-dasharray: 4 4; }
.psi-chart .axis { stroke: var(--rule); }
.monitor td:last-child { white-space: normal; min-width: 13rem; }
.psi-chart + .legend { margin-bottom: 1.25rem; }
.chain li.unchecked { background: repeating-linear-gradient(135deg, var(--unchecked) 0 3px, transparent 3px 6px); }
.chain li.broken { background: var(--broken); outline: 2px solid var(--broken); outline-offset: 2px; }
.chain-row { margin: 1.5rem 0; }
.chain-row .status { font-weight: 600; }
.status.ok { color: var(--verified); }
.status.bad { color: var(--broken); }
.legend { display: flex; flex-wrap: wrap; gap: .4rem 1.25rem; color: var(--muted); font-size: .9rem; margin: .25rem 0 0; padding: 0; list-style: none; }
.legend span { display: inline-block; width: 10px; height: 14px; border-radius: 2px; margin-right: .4rem; vertical-align: -2px; }
.table-wrap { overflow-x: auto; border: 1px solid var(--rule); border-radius: 6px; background: var(--surface); }
table { border-collapse: collapse; width: 100%; font-size: .95rem; font-variant-numeric: tabular-nums; }
th, td { text-align: left; padding: .55rem .7rem; border-bottom: 1px solid var(--rule); white-space: nowrap; }
th { font-weight: 600; color: var(--muted); }
tr:last-child td { border-bottom: 0; }
td.num { text-align: right; }
tr.reviewed td:first-child { box-shadow: inset 3px 0 0 var(--human); }
.override { color: var(--human); font-weight: 600; }
.result-ok { color: var(--verified); font-weight: 600; }
.result-no { color: var(--broken); font-weight: 600; }
.controls td { white-space: normal; }
.controls td code { word-break: normal; white-space: nowrap; }
pre { background: var(--surface); border: 1px solid var(--rule); border-radius: 6px; padding: .9rem 1rem; overflow-x: auto; font-size: .9rem; }
ul.plain { padding-left: 1.2rem; max-width: 42rem; }
footer { margin-top: 3rem; color: var(--muted); font-size: .9rem; }
nav { margin-bottom: 1.5rem; font-size: .95rem; }
.scenarios { display: grid; grid-template-columns: repeat(auto-fit, minmax(18rem, 1fr)); gap: 1rem; margin: 1.25rem 0; }
.scenario { background: var(--surface); border: 1px solid var(--rule); border-radius: 6px; padding: 1rem 1.1rem; }
.scenario h3 { font-family: var(--serif); font-size: 1.2rem; margin: 0 0 .35rem; }
.scenario p { margin: .25rem 0; }
.compare td { white-space: normal; }
.table-wrap + p { margin-top: .9rem; }
.table-wrap.narrow { display: inline-block; max-width: 100%; }
.table-wrap.narrow td.num { padding-left: 2.5rem; }
.compare th[scope=row] { color: var(--ink); white-space: nowrap; }
"""


def overrides_phrase(n: int) -> str:
    return {0: "in no cases", 1: "once", 2: "twice"}.get(n, f"{n} times")


def counts(res: Dict[str, Any]) -> Dict[str, int]:
    rows = res["rows"]
    reviewed = [r for r in rows if r["route"] == "human review"]
    return {
        "total": len(rows),
        "automated": len(rows) - len(reviewed),
        "reviewed": len(reviewed),
        "overrides": sum(r["override"] for r in reviewed),
    }


def provenance_line(prov: Dict[str, Any]) -> str:
    e = html.escape

    def link(text: str, url: Any) -> str:
        return f'<a href="{e(url)}">{e(text)}</a>' if url else e(text)

    source = [f"commit {link(prov['commit'], prov['commit_url'])}"]
    if prov["run_label"]:
        source.append(link(prov["run_label"], prov["run_url"]))
    return f"Generated from {' in '.join(source)} on {e(prov['generated'])}."


def page(title: str, body: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>{REPORT_CSS}</style>
</head>
<body>
<main>
{body}
<footer>Responsible AI Toolkit is experimental open-source software under the Apache License 2.0.</footer>
</main>
</body>
</html>
"""


SYNTHETIC_NOTICE = (
    '<p class="notice">All data here is synthetic, generated with fixed seeds. '
    "These pages show what the software did on that data. They are not a model "
    "validation, a deployment, or evidence of use by any institution.</p>"
)


def render_scenario_html(res: Dict[str, Any], prov: Dict[str, Any]) -> str:
    """Render a self-contained HTML review record for one scenario."""
    e = html.escape
    sc = res["scenario"]
    rows = res["rows"]
    c = counts(res)
    types = res["event_types"]
    k = res["tamper_index"]

    chain = "".join(f'<li class="{e(t)}" title="Entry {i + 1}: {e(t)}"></li>'
                    for i, t in enumerate(types))
    edited = []
    for i, t in enumerate(types):
        cls = t if i < k else ("broken" if i == k else "unchecked")
        edited.append(f'<li class="{e(cls)}" title="Entry {i + 1}"></li>')

    decision_rows = []
    for r in rows:
        decision = e(sc.label(r["review_decision"]))
        if r["override"]:
            decision += ' <span class="override">(override)</span>'
        decision_rows.append(
            f'<tr class="{"reviewed" if r["route"] == "human review" else ""}">'
            f"<td>{e(r['record_ref'])}</td>"
            f"<td>{e(r['model_recommendation'])}</td>"
            f'<td class="num">{r["confidence"]:.2f}</td>'
            f"<td>{e(', '.join(r['policy_findings']) or 'none')}</td>"
            f"<td>{e(r['route'])}</td>"
            f"<td>{e(r['reviewer'] or '-')}</td>"
            f"<td>{decision}</td></tr>"
        )

    control_rows = []
    for label, outcome in res["controls"]:
        verdict, _, reason = outcome.partition(": ")
        css = "result-ok" if outcome == "Accepted" else "result-no"
        detail = f"<br>{e(reason)}" if reason else ""
        control_rows.append(
            f'<tr><td>{e(label)}</td><td><span class="{css}">{e(verdict)}</span>{detail}</td></tr>'
        )

    type_names = {"prediction": "Model decision", "policy_check": "Policy check",
                  "human_review": "Human review"}
    trace_rows = []
    for i, entry in res["trace"]:
        pl = entry.payload
        if entry.event_type == "prediction":
            what = (f"Model recommended {pl['output']['recommendation']} "
                    f"with confidence {pl['output']['confidence']:.2f}.")
        elif entry.event_type == "policy_check":
            what = f"Rule {pl['policy_id']}: {pl['result']}."
        elif entry.event_type == "human_review":
            what = (f"{pl['reviewer_id']} decided {sc.label(pl['decision'])}"
                    f"{' (override)' if pl['override'] else ''}. {pl['reasoning']}")
        else:
            what = entry.event_type
        trace_rows.append(
            f'<tr><td class="num">{i + 1}</td><td>{e(type_names.get(entry.event_type, entry.event_type))}</td>'
            f'<td>{e(what)}</td><td><code>{e(entry.entry_hash[:12])}</code></td></tr>'
        )

    workload_rows = []
    for reviewer in sc.reviewers:
        mine = [r for r in rows if r["reviewer"] == reviewer]
        workload_rows.append(
            f'<tr><td>{e(reviewer)}</td><td class="num">{len(mine)}</td>'
            f'<td class="num">{sum(r["override"] for r in mine)}</td></tr>'
        )
    review_rate = c["reviewed"] / c["total"] if c["total"] else 0
    override_rate = c["overrides"] / c["reviewed"] if c["reviewed"] else 0

    chain_status = (
        f'<span class="status ok">All {len(types)} entries verified.</span>'
        if res["chain_verified"]
        else '<span class="status bad">Verification failed.</span>'
    )
    edited_status = (
        f'<span class="status bad">Verification failed at entry {k + 1}.</span>'
        if not res["tampered_copy_verified"]
        else '<span class="status bad">Edited copy was not detected.</span>'
    )
    rules_word = {2: "two", 3: "three"}.get(res["rule_count"], str(res["rule_count"]))

    body = f"""<nav><a href="index.html">All demo scenarios</a></nav>
<h1>Model review record</h1>
<p class="provenance">{e(sc.title)}, a synthetic demo of the Responsible AI Toolkit. {provenance_line(prov)}</p>
{SYNTHETIC_NOTICE}

<p class="lede">{c['total']} model decisions were checked against {rules_word} rules. {c['automated']} passed and were automated. {c['reviewed']} failed a rule and went to a human reviewer, who overrode the model {overrides_phrase(c['overrides'])}. Every step was written to a hash-chained audit log.</p>

<h2>Audit record</h2>
<p>Each block is one log entry, in order. Each entry's hash covers its contents and the entry before it, so an edit anywhere breaks verification from that point.</p>
<div class="chain-row">
<p>{chain_status}</p>
<ul class="chain" aria-label="Audit log entries">{chain}</ul>
<ul class="legend">
<li><span style="background:var(--prediction)"></span>Model decision</li>
<li><span style="background:var(--policy)"></span>Policy check</li>
<li><span style="background:var(--human)"></span>Human review</li>
</ul>
</div>
<div class="chain-row">
<p>A copy of the same log with one value changed: the model confidence for {e(sc.tamper_target)}, edited from {res["tamper_from"]:.2f} to {res["tamper_value"]:.2f}. {edited_status} Entries after that point are not verified.</p>
<ul class="chain" aria-label="Edited copy of the audit log">{"".join(edited)}</ul>
<ul class="legend">
<li><span style="background:var(--broken)"></span>Edited entry, hash mismatch</li>
<li><span style="background:repeating-linear-gradient(135deg, var(--unchecked) 0 3px, transparent 3px 6px)"></span>Not verified</li>
</ul>
</div>
<p>Chain head hash: <code>{e(res["head_hash"])}</code>. Keeping this value somewhere separate from the log makes later removal of entries detectable.</p>

<h2>Decisions</h2>
<p>Rules: {e(sc.rules_text)}. Thresholds are illustrative configuration values, not legal or supervisory standards.</p>
<div class="table-wrap">
<table>
<thead><tr><th>{e(sc.record_label)}</th><th>Model</th><th>Confidence</th><th>Policy findings</th><th>Route</th><th>Reviewer</th><th>Reviewer decision</th></tr></thead>
<tbody>{"".join(decision_rows)}</tbody>
</table>
</div>

<h2>Trace one decision</h2>
<p>Every audit entry that concerns {e(res["trace_ref"])}, in the order it was written. This is the record a reviewer or examiner would pull to see why the final decision was made. Each entry's hash links it to the entry before it in the full log.</p>
<div class="table-wrap">
<table class="controls">
<thead><tr><th>Entry</th><th>Type</th><th>What was recorded</th><th>Entry hash</th></tr></thead>
<tbody>{"".join(trace_rows)}</tbody>
</table>
</div>

<h2>Reviewer workload</h2>
<p>{review_rate:.0%} of decisions went to a reviewer. Reviewers overrode the model in {override_rate:.0%} of the cases they decided.</p>
<div class="table-wrap narrow">
<table>
<thead><tr><th>Reviewer</th><th>Cases decided</th><th>Overrides</th></tr></thead>
<tbody>{"".join(workload_rows)}</tbody>
</table>
</div>

<h2>Reviewer controls</h2>
<p>Four attempts to record a decision on one review case, testing who may decide and whether a finished decision can be changed.</p>
<div class="table-wrap">
<table class="controls">
<thead><tr><th>Attempt</th><th>Result</th></tr></thead>
<tbody>{"".join(control_rows)}</tbody>
</table>
</div>

<h2>Limits of this record</h2>
<ul class="plain">
<li>The reviewer decisions come from a fixed rule standing in for human judgment.</li>
<li>The audit log is held in memory for the run. Persistence, access control, and retention belong to the system that uses the toolkit.</li>
<li>The hash chain has no secret key. Anyone who can change the log can recompute its hashes, which is why the head hash should be kept separately.</li>
<li>Passing these checks does not establish compliance with any law, regulation, or guidance.</li>
</ul>

<p>Raw output: <a href="{sc.key}_decisions.json">{sc.key}_decisions.json</a>, <a href="{sc.key}_audit_log.json">{sc.key}_audit_log.json</a>, and <a href="{sc.key}_evidence_package.json">{sc.key}_evidence_package.json</a> (the audit logger's built-in export: every entry plus the verification result).</p>
"""
    return page(f"Model review record: {sc.title} (synthetic)", body)


def render_monitoring_html(mon: Dict[str, Any], prov: Dict[str, Any]) -> str:
    """Render the month-by-month drift monitoring record."""
    e = html.escape
    periods = mon["periods"]
    types = mon["event_types"]
    k = mon["tamper_index"]

    # PSI chart (inline SVG): one bar per period, dashed lines at both limits.
    w, h, left, bottom, top = 640, 260, 48, 36, 16
    plot_h = h - bottom - top
    y_max = max(0.35, max(r["psi"] for r in periods) * 1.15)
    band = lambda v: "var(--verified)" if v < PSI_LOW else ("var(--human)" if v < PSI_HIGH else "var(--broken)")
    y = lambda v: top + plot_h * (1 - v / y_max)
    step = (w - left - 8) / len(periods)
    bars = []
    for i, r in enumerate(periods):
        x = left + i * step + step * 0.2
        bw = step * 0.6
        bars.append(
            f'<rect x="{x:.1f}" y="{y(r["psi"]):.1f}" width="{bw:.1f}" '
            f'height="{top + plot_h - y(r["psi"]):.1f}" rx="2" fill="{band(r["psi"])}"></rect>'
            f'<text class="value" x="{x + bw / 2:.1f}" y="{y(r["psi"]) - 6:.1f}" text-anchor="middle">{r["psi"]:.3f}</text>'
            f'<text x="{x + bw / 2:.1f}" y="{h - 12}" text-anchor="middle">{e(r["period"])}</text>'
        )
    limits = "".join(
        f'<line class="limit" x1="{left}" x2="{w - 8}" y1="{y(v):.1f}" y2="{y(v):.1f}"></line>'
        f'<text x="{left - 6}" y="{y(v) + 4:.1f}" text-anchor="end">{v:.2f}</text>'
        for v in (PSI_LOW, PSI_HIGH)
    )
    chart = (
        f'<svg class="psi-chart" viewBox="0 0 {w} {h}" role="img" '
        f'aria-label="Population stability index by month">'
        f'<line class="axis" x1="{left}" x2="{w - 8}" y1="{top + plot_h}" y2="{top + plot_h}"></line>'
        f"{limits}{''.join(bars)}</svg>"
    )

    table_rows = []
    for r in periods:
        action = e(r["action"])
        if r["decision"]:
            action += f"<br>{e(r['reviewer'])}: {e(MONITORING_LABELS.get(r['decision'], r['decision']))}"
        css = "result-no" if r["policy"] == "failed" else "result-ok"
        table_rows.append(
            f"<tr><td>{e(r['period'])}</td><td class=\"num\">{r['scored']:,}</td>"
            f"<td class=\"num\">{r['psi']:.3f}</td><td class=\"num\">{r['kl']:.3f}</td>"
            f"<td class=\"num\">{r['wasserstein']:.3f}</td><td>{e(r['interpretation'])}</td>"
            f'<td><span class="{css}">{e(r["policy"])}</span></td><td>{action}</td></tr>'
        )

    type_names = {"policy_check": "Policy check", "drift_alert": "Drift alert",
                  "escalation": "Escalation", "human_review": "Human review"}
    trace_rows = []
    for i, entry in mon["trace"]:
        pl = entry.payload
        if entry.event_type == "drift_alert":
            what = f"PSI {pl['value']:.3f} against limit {pl['threshold']:.2f}: {pl['interpretation']}."
        elif entry.event_type == "policy_check":
            what = f"Rule {pl['policy_id']}: {pl['result']}."
        elif entry.event_type == "escalation":
            what = f"Escalated to {pl['escalated_to']} ({pl['priority']} priority). {pl['reason']}."
        elif entry.event_type == "human_review":
            what = (f"{pl['reviewer_id']} decided to "
                    f"{MONITORING_LABELS.get(pl['decision'], pl['decision'])}. {pl['reasoning']}")
        else:
            what = entry.event_type
        trace_rows.append(
            f'<tr><td class="num">{i + 1}</td><td>{e(type_names.get(entry.event_type, entry.event_type))}</td>'
            f'<td>{e(what)}</td><td><code>{e(entry.entry_hash[:12])}</code></td></tr>'
        )

    chain = "".join(f'<li class="{e(t)}" title="Entry {i + 1}: {e(t)}"></li>' for i, t in enumerate(types))
    edited = "".join(
        f'<li class="{e(t if i < k else ("broken" if i == k else "unchecked"))}" title="Entry {i + 1}"></li>'
        for i, t in enumerate(types)
    )
    chain_status = (
        f'<span class="status ok">All {len(types)} entries verified.</span>'
        if mon["chain_verified"] else '<span class="status bad">Verification failed.</span>'
    )
    edited_status = (
        f'<span class="status bad">Verification failed at entry {k + 1}.</span>'
        if not mon["tampered_copy_verified"]
        else '<span class="status bad">Edited copy was not detected.</span>'
    )

    quiet = sum(1 for r in periods if r["psi"] < PSI_LOW)
    moderate = [r["period"] for r in periods if PSI_LOW <= r["psi"] < PSI_HIGH]
    escalated = [r for r in periods if r["action"] == "Escalated"]
    quiet_word = {1: "one month", 2: "two months", 3: "three months", 4: "four months"}.get(quiet, f"{quiet} months")
    story = f"PSI stayed below {PSI_LOW:.2f} for {quiet_word}"
    if moderate:
        story += f", rose into the monitoring band in {' and '.join(moderate)}"
    if escalated:
        first = escalated[0]
        story += (f", and reached {first['psi']:.3f} in {first['period']}. That month's drift "
                  f"check failed, the case was escalated to {first['reviewer']}, who "
                  f"decided to {MONITORING_LABELS[first['decision']]} pending investigation")
    story += "."

    body = f"""<nav><a href="index.html">All demo scenarios</a></nav>
<h1>Model monitoring record</h1>
<p class="provenance">Month-by-month drift monitoring of the synthetic lending model, a demo of the Responsible AI Toolkit. {provenance_line(prov)}</p>
{SYNTHETIC_NOTICE}

<p class="lede">Each month, the lending model's scores were compared with its scores from the validation period. {e(story)}</p>

<h2>Score drift by month</h2>
<p>Population stability index (PSI) of each month's scores against the validation-period scores. Dashed lines mark the monitoring band at {PSI_LOW:.2f} and the escalation limit at {PSI_HIGH:.2f}.</p>
{chart}
<ul class="legend">
<li><span style="background:var(--verified)"></span>Below {PSI_LOW:.2f}</li>
<li><span style="background:var(--human)"></span>{PSI_LOW:.2f} to {PSI_HIGH:.2f}: monitor closely</li>
<li><span style="background:var(--broken)"></span>{PSI_HIGH:.2f} or above: escalate</li>
</ul>

<div class="table-wrap">
<table class="monitor">
<thead><tr><th>Period</th><th>Scores</th><th>PSI</th><th>KL divergence</th><th>Wasserstein</th><th>Interpretation</th><th>Drift policy</th><th>Action</th></tr></thead>
<tbody>{"".join(table_rows)}</tbody>
</table>
</div>
<p>The three measures come from the toolkit's drift monitor. The drift policy is the toolkit's built-in drift rule with a limit of {PSI_HIGH:.2f}. Both limits are common rules of thumb, not regulatory or supervisory standards.</p>

<h2>Trace the escalation</h2>
<p>Every audit entry for {e(mon["trace_period"])}, in the order it was written: the alert, the failed check, the escalation, and the reviewer's decision.</p>
<div class="table-wrap">
<table class="controls">
<thead><tr><th>Entry</th><th>Type</th><th>What was recorded</th><th>Entry hash</th></tr></thead>
<tbody>{"".join(trace_rows)}</tbody>
</table>
</div>

<h2>Audit record</h2>
<p>Each block is one log entry, in order. Months with no drift produce only a policy check; drift adds an alert, and a failed check adds an escalation and a review.</p>
<div class="chain-row">
<p>{chain_status}</p>
<ul class="chain" aria-label="Audit log entries">{chain}</ul>
<ul class="legend">
<li><span style="background:var(--policy)"></span>Policy check</li>
<li><span style="background:var(--drift)"></span>Drift alert</li>
<li><span style="background:var(--escalation)"></span>Escalation</li>
<li><span style="background:var(--human)"></span>Human review</li>
</ul>
</div>
<div class="chain-row">
<p>A copy of the same log with the first drift alert's PSI edited from {mon["tamper_from"]:.3f} to {mon["tamper_value"]:.3f}, as if to hide it. {edited_status} Entries after that point are not verified.</p>
<ul class="chain" aria-label="Edited copy of the audit log">{edited}</ul>
</div>
<p>Chain head hash: <code>{e(mon["head_hash"])}</code>. Keeping this value somewhere separate from the log makes later removal of entries detectable.</p>

<h2>Limits of this record</h2>
<ul class="plain">
<li>The score shift is synthetic and was built into the data. Real drift has causes that need investigation, such as changes in applicants, data pipelines, or economic conditions.</li>
<li>PSI measures a change in the score distribution, not a change in accuracy. Checking accuracy requires outcomes, which arrive later.</li>
<li>The reviewer decision comes from a fixed rule standing in for human judgment.</li>
<li>Scores are generated with NumPy; exact figures can differ slightly across NumPy versions, while the monthly pattern and the escalation stay the same.</li>
<li>Passing these checks does not establish compliance with any law, regulation, or guidance.</li>
</ul>

<p>Raw output: <a href="monitoring_audit_log.json">monitoring_audit_log.json</a> and <a href="monitoring_evidence_package.json">monitoring_evidence_package.json</a>.</p>
"""
    return page("Model monitoring record: synthetic lending model", body)


def comparison_rows(results: List[Dict[str, Any]]) -> List[Tuple[str, List[str]]]:
    """Rows comparing the scenarios, built from each run's configuration and results."""
    def rules(res: Dict[str, Any]) -> str:
        sc = res["scenario"]
        extra = len(sc.extra_policies)
        text = "2 built-in rules"
        return text + (f" and {extra} custom rule{'s' if extra > 1 else ''}" if extra else "")

    rows = [
        ("Audit logging", ["AuditLogger, unchanged" for _ in results]),
        ("Policy checks", [f"PolicyEngine, unchanged; {rules(r)}" for r in results]),
        ("Human review", [f"HITLOrchestrator, unchanged; reviewers with role \u201c{r['scenario'].category}\u201d" for r in results]),
        ("Confidence threshold", [f"{r['scenario'].confidence_threshold:.2f}" for r in results]),
        ("Required inputs", [", ".join(r["scenario"].required_fields) for r in results]),
        ("Outcome", [
            f"{counts(r)['automated']} automated, {counts(r)['reviewed']} reviewed, "
            f"{counts(r)['overrides']} overrides" for r in results
        ]),
        ("Reviewer controls", [
            f"{sum(1 for (lbl, out), exp in zip(r['controls'], ['Rejected', 'Rejected', 'Accepted', 'Rejected']) if out.split(':')[0] == exp)} of 4 as expected"
            for r in results
        ]),
        ("Audit chain", [
            f"{r['audit_entries']} entries verified; edit "
            f"{'detected' if not r['tampered_copy_verified'] else 'NOT detected'}"
            for r in results
        ]),
    ]
    return rows


def render_index_html(results: List[Dict[str, Any]], prov: Dict[str, Any],
                      mon: Dict[str, Any]) -> str:
    e = html.escape
    cards = []
    for res in results:
        sc = res["scenario"]
        c = counts(res)
        cards.append(
            f'<div class="scenario"><h3><a href="{sc.key}.html">{e(sc.title)}</a></h3>'
            f"<p>{c['total']} model decisions. {c['automated']} automated, {c['reviewed']} sent to a "
            f"reviewer, who overrode the model {overrides_phrase(c['overrides'])}.</p>"
            f'<p><a href="{sc.key}.html">Open the review record</a></p></div>'
        )
    head = "".join(f"<th scope=\"col\">{e(r['scenario'].title)}</th>" for r in results)
    body_rows = "".join(
        f'<tr><th scope="row">{e(label)}</th>' + "".join(f"<td>{e(v)}</td>" for v in values) + "</tr>"
        for label, values in comparison_rows(results)
    )
    repo = (f'<p>Source: <a href="{e(prov["repo_url"])}">Responsible AI Toolkit on GitHub</a>.</p>'
            if prov["repo_url"] else "")
    count_word = {2: "two", 3: "three", 4: "four"}.get(len(results), str(len(results)))
    shorts = [r["scenario"].short for r in results]
    listed = ", ".join(shorts[:-1]) + (", and " if len(shorts) > 2 else " and ") + shorts[-1]
    body = f"""<h1>One toolkit, {count_word} review workflows</h1>
<p class="provenance">Synthetic demo of the Responsible AI Toolkit. {provenance_line(prov)}</p>
{SYNTHETIC_NOTICE}

<p class="lede">The same audit, policy, and human-review components handle {e(listed)}. Only the configuration differs: the data, rule settings, and reviewer roles.</p>

<div class="scenarios">{"".join(cards)}</div>

<h2>Ongoing model monitoring</h2>
<div class="scenarios">
<div class="scenario"><h3><a href="monitoring.html">Month-by-month drift monitoring</a></h3>
<p>The lending model's scores over six months, compared with its validation period. {sum(1 for r in mon["periods"] if r["action"] == "Escalated")} month crossed the escalation limit and went to a model-risk reviewer.</p>
<p><a href="monitoring.html">Open the monitoring record</a></p></div>
</div>

<h2>What stayed the same and what changed</h2>
<p>Built from each run's configuration and results. All runs use the same toolkit code.</p>
<div class="table-wrap">
<table class="compare">
<thead><tr><th></th>{head}</tr></thead>
<tbody>{body_rows}</tbody>
</table>
</div>
<p>Scenario-specific code in this demo is limited to the synthetic data, the rule settings, one custom rule per scenario where needed (written with the toolkit's <code>Policy</code> class), the reviewer roles, and a fixed rule standing in for reviewer judgment.</p>

<h2>Reproduce these runs</h2>
<pre>python -m pip install -e ".[dev]"
python examples/model_review_demo.py demo_output</pre>
<p>The decisions, controls, and audit results match on every run. Timestamps, entry IDs, and hashes differ.</p>
{repo}
"""
    return page("Responsible AI Toolkit: synthetic review demos", body)


def render_summary(results: List[Dict[str, Any]], prov: Dict[str, Any],
                   mon: Dict[str, Any]) -> str:
    lines = [
        "# Model review demo (synthetic data)",
        "",
        f"Generated by `examples/model_review_demo.py` at commit `{prov['commit']}` on {prov['generated']}.",
        "",
        "All data is synthetic (fixed seeds). This output shows software behavior only. "
        "It is not a model validation, a deployment, or evidence of use by any institution.",
        "",
        "## Same toolkit, two workflows",
        "",
        "| | " + " | ".join(r["scenario"].title for r in results) + " |",
        "| --- |" + " --- |" * len(results),
    ]
    for label, values in comparison_rows(results):
        lines.append(f"| {label} | " + " | ".join(values) + " |")
    lines += ["", "All runs use the same toolkit code.", ""]

    for res in results:
        sc = res["scenario"]
        c = counts(res)
        lines += [
            f"## {sc.title}",
            "",
            f"Rules: {sc.rules_text}.",
            "",
            f"| {sc.record_label} | Model recommendation | Confidence | Policy findings | Route | Reviewer | Review decision |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
        for r in res["rows"]:
            lines.append(
                f"| {r['record_ref']} | {r['model_recommendation']} | {r['confidence']:.2f} | "
                f"{', '.join(r['policy_findings']) or 'none'} | {r['route']} | "
                f"{r['reviewer'] or '-'} | "
                f"{sc.label(r['review_decision']) + (' (override)' if r['override'] else '')} |"
            )
        lines += [
            "",
            f"{sc.record_label}s: {c['total']}. Automated: {c['automated']}. "
            f"Routed to human review: {c['reviewed']}. Reviewer overrides of the model: {c['overrides']}.",
            "",
            "| Review control attempt | Result |",
            "| --- | --- |",
        ]
        lines += [f"| {label} | {outcome} |" for label, outcome in res["controls"]]
        lines += [
            "",
            f"Audit entries: {res['audit_entries']}. Chain verification: "
            f"{'passed' if res['chain_verified'] else 'FAILED'}. Chain head hash: `{res['head_hash']}`.",
            "",
            f"Copy with {sc.tamper_target}'s confidence edited: verification "
            f"{'passed (unexpected)' if res['tampered_copy_verified'] else 'failed, as expected'}.",
            "",
        ]
    lines += [
        "## Month-by-month drift monitoring (lending model)",
        "",
        "| Period | PSI | KL divergence | Wasserstein | Interpretation | Drift policy | Action |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in mon["periods"]:
        action = r["action"] + (
            f" ({r['reviewer']}: {MONITORING_LABELS[r['decision']]})" if r["decision"] else ""
        )
        lines.append(
            f"| {r['period']} | {r['psi']:.3f} | {r['kl']:.3f} | {r['wasserstein']:.3f} | "
            f"{r['interpretation']} | {r['policy']} | {action} |"
        )
    lines += [
        "",
        f"Audit entries: {mon['audit_entries']}. Chain verification: "
        f"{'passed' if mon['chain_verified'] else 'FAILED'}. Edited copy: verification "
        f"{'passed (unexpected)' if mon['tampered_copy_verified'] else 'failed, as expected'}.",
        "",
    ]
    all_ok = all(r["all_checks_as_expected"] for r in results) and mon["all_checks_as_expected"]
    lines.append(f"All demo checks as expected: {'yes' if all_ok else 'NO'}.")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    output_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("demo_output")
    output_dir.mkdir(parents=True, exist_ok=True)
    prov = provenance()
    results = [run_scenario(sc) for sc in SCENARIOS]
    mon = run_monitoring()

    for res in results:
        key = res["scenario"].key
        (output_dir / f"{key}.html").write_text(render_scenario_html(res, prov), encoding="utf-8")
        (output_dir / f"{key}_decisions.json").write_text(json.dumps(res["rows"], indent=2))
        (output_dir / f"{key}_audit_log.json").write_text(res["audit"].export_json())
        (output_dir / f"{key}_evidence_package.json").write_text(
            json.dumps(res["evidence_package"], indent=2, default=str)
        )
    (output_dir / "monitoring.html").write_text(render_monitoring_html(mon, prov), encoding="utf-8")
    (output_dir / "monitoring_audit_log.json").write_text(mon["audit"].export_json())
    (output_dir / "monitoring_evidence_package.json").write_text(
        json.dumps(mon["evidence_package"], indent=2, default=str)
    )
    (output_dir / "index.html").write_text(render_index_html(results, prov, mon), encoding="utf-8")
    summary = render_summary(results, prov, mon)
    (output_dir / "summary.md").write_text(summary)

    print(summary)
    ok = all(r["all_checks_as_expected"] for r in results) and mon["all_checks_as_expected"]
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
