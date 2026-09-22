"""
Model review demo on synthetic lending and insurance data.

Runs the same toolkit components, unchanged, on two synthetic scenarios:
consumer lending decisions and small-business property insurance quotes.
For each scenario it:

  1. Records each model decision in a hash-chained audit log.
  2. Evaluates each decision against configurable policy checks.
  3. Routes decisions that fail a check to an assigned human reviewer,
     who works through the review queue.
  4. Shows the reviewer-authorization and repeat-decision controls.
  5. Verifies the audit chain, then shows that an edited copy fails.

Only the scenario configuration differs between the two runs: the data,
the rule settings, the reviewer roles, and a stand-in for reviewer
judgment. The toolkit code is the same for both.

All data is synthetic and generated with fixed seeds. The output shows
software behavior only. It is not a model validation, a deployment, or
evidence of use by any institution.

Writes to the output directory: index.html (comparison page), one HTML
review record per scenario, the raw decisions and audit logs as JSON,
and summary.md.

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

from responsible_ai_toolkit.audit import AuditLogger
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


SCENARIOS = [
    Scenario(
        key="lending",
        title="Consumer lending decisions",
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
    tampered._entries[tamper_index].payload["output"]["confidence"] = 0.99
    tampered_ok = tampered.verify_chain()

    expected = ["Rejected", "Rejected", "Accepted", "Rejected"]
    checks_ok = (
        chain_ok
        and not tampered_ok
        and [c[1].split(":")[0] for c in controls] == expected
    )
    return {
        "scenario": sc,
        "rows": rows,
        "controls": controls,
        "rule_count": len(engine.list_policies()),
        "audit": audit,
        "audit_entries": audit.size,
        "chain_verified": chain_ok,
        "tampered_copy_verified": tampered_ok,
        "tamper_index": tamper_index,
        "event_types": [e.event_type for e in audit.entries],
        "head_hash": audit.entries[-1].entry_hash,
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
  --serif: Charter, "Bitstream Charter", "Iowan Old Style", "Sitka Text", Cambria, Georgia, serif;
  --sans: system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
  color-scheme: light dark;
}
@media (prefers-color-scheme: dark) {
  :root {
    --paper: #0f141b; --surface: #161d27; --ink: #e5e9ef; --muted: #9ba6b5;
    --rule: #2a3442; --verified: #3cbf8e; --broken: #f07268; --unchecked: #3a4452;
    --prediction: #86a3dc; --policy: #5d6c81; --human: #e3a73e;
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
        decision = e(r["review_decision"] or "-")
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
<p>A copy of the same log with one value changed: the model confidence for {e(sc.tamper_target)}, edited to 0.99. {edited_status} Entries after that point are not verified.</p>
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

<p>Raw output: <a href="{sc.key}_decisions.json">{sc.key}_decisions.json</a> and <a href="{sc.key}_audit_log.json">{sc.key}_audit_log.json</a>.</p>
"""
    return page(f"Model review record: {sc.title} (synthetic)", body)


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


def render_index_html(results: List[Dict[str, Any]], prov: Dict[str, Any]) -> str:
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
    body = f"""<h1>One toolkit, two review workflows</h1>
<p class="provenance">Synthetic demo of the Responsible AI Toolkit. {provenance_line(prov)}</p>
{SYNTHETIC_NOTICE}

<p class="lede">The same audit, policy, and human-review components run a consumer lending workflow and a small-business insurance workflow. Only the configuration differs: the data, rule settings, and reviewer roles.</p>

<div class="scenarios">{"".join(cards)}</div>

<h2>What stayed the same and what changed</h2>
<p>Built from each run's configuration and results. Both runs use the same toolkit code.</p>
<div class="table-wrap">
<table class="compare">
<thead><tr><th></th>{head}</tr></thead>
<tbody>{body_rows}</tbody>
</table>
</div>
<p>Scenario-specific code in this demo is limited to the synthetic data, the rule settings, one custom insurance rule written with the toolkit's <code>Policy</code> class, the reviewer roles, and a fixed rule standing in for reviewer judgment.</p>

<h2>Reproduce these runs</h2>
<pre>python -m pip install -e ".[dev]"
python examples/model_review_demo.py demo_output</pre>
<p>The decisions, controls, and audit results match on every run. Timestamps, entry IDs, and hashes differ.</p>
{repo}
"""
    return page("Responsible AI Toolkit: synthetic review demos", body)


def render_summary(results: List[Dict[str, Any]], prov: Dict[str, Any]) -> str:
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
    lines += ["", "Both runs use the same toolkit code.", ""]

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
                f"{(r['review_decision'] or '-') + (' (override)' if r['override'] else '')} |"
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
    all_ok = all(r["all_checks_as_expected"] for r in results)
    lines.append(f"All demo checks as expected: {'yes' if all_ok else 'NO'}.")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    output_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("demo_output")
    output_dir.mkdir(parents=True, exist_ok=True)
    prov = provenance()
    results = [run_scenario(sc) for sc in SCENARIOS]

    for res in results:
        key = res["scenario"].key
        (output_dir / f"{key}.html").write_text(render_scenario_html(res, prov), encoding="utf-8")
        (output_dir / f"{key}_decisions.json").write_text(json.dumps(res["rows"], indent=2))
        (output_dir / f"{key}_audit_log.json").write_text(res["audit"].export_json())
    (output_dir / "index.html").write_text(render_index_html(results, prov), encoding="utf-8")
    summary = render_summary(results, prov)
    (output_dir / "summary.md").write_text(summary)

    print(summary)
    return 0 if all(r["all_checks_as_expected"] for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
