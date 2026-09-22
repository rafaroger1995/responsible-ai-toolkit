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

Writes summary.md, report.html, decisions.json, and audit_log.json to
the output directory.

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
TAMPER_TARGET = "SYN-006"

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
    tamper_index = next(
        i for i, e in enumerate(tampered._entries)
        if e.event_type == "prediction"
        and e.payload["metadata"].get("application_ref") == TAMPER_TARGET
    )
    tampered._entries[tamper_index].payload["output"]["confidence"] = 0.99
    tampered_ok = tampered.verify_chain()

    results = {
        "rows": rows,
        "controls": controls,
        "audit_entries": audit.size,
        "chain_verified": chain_ok,
        "tampered_copy_verified": tampered_ok,
        "tamper_index": tamper_index,
        "event_types": [e.event_type for e in audit.entries],
        "head_hash": audit.entries[-1].entry_hash,
        "provenance": provenance(),
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
    (output_dir / "report.html").write_text(render_html(results), encoding="utf-8")
    return results


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


def render_summary(results: Dict[str, Any]) -> str:
    rows = results["rows"]
    commit = results["provenance"]["commit"]
    generated = results["provenance"]["generated"]
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
        f"Chain head hash: `{results['head_hash']}`. Recording this value separately "
        "makes later removal of entries detectable.",
        "",
        "Copy of the log with one prediction's confidence edited: verification "
        f"{'passed (unexpected)' if results['tampered_copy_verified'] else 'failed, as expected'}.",
        "",
        f"All demo checks as expected: {'yes' if results['all_checks_as_expected'] else 'NO'}.",
        "",
    ]
    return "\n".join(lines)


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
"""


def render_html(results: Dict[str, Any]) -> str:
    """Render a self-contained HTML review record for this run."""
    e = html.escape
    rows = results["rows"]
    prov = results["provenance"]
    reviewed = [r for r in rows if r["route"] == "human review"]
    overrides = sum(r["override"] for r in reviewed)
    times = {0: "in no cases", 1: "once", 2: "twice"}.get(overrides, f"{overrides} times")
    types = results["event_types"]
    k = results["tamper_index"]

    def link(text: str, url: Any) -> str:
        return f'<a href="{e(url)}">{e(text)}</a>' if url else e(text)

    source = [f"commit {link(prov['commit'], prov['commit_url'])}"]
    if prov["run_label"]:
        source.append(link(prov["run_label"], prov["run_url"]))
    provenance_line = (
        f"Generated from {' in '.join(source)} on {e(prov['generated'])}."
    )

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
            f"<td>{e(r['application_ref'])}</td>"
            f"<td>{e(r['model_recommendation'])}</td>"
            f'<td class="num">{r["confidence"]:.2f}</td>'
            f"<td>{e(', '.join(r['policy_findings']) or 'none')}</td>"
            f"<td>{e(r['route'])}</td>"
            f"<td>{e(r['reviewer'] or '-')}</td>"
            f"<td>{decision}</td></tr>"
        )

    control_rows = []
    for label, outcome in results["controls"]:
        accepted = outcome == "Accepted"
        verdict, _, reason = outcome.partition(": ")
        css = "result-ok" if accepted else "result-no"
        detail = f"<br>{e(reason)}" if reason else ""
        control_rows.append(
            f'<tr><td>{e(label)}</td><td><span class="{css}">{e(verdict)}</span>{detail}</td></tr>'
        )

    chain_status = (
        f'<span class="status ok">All {len(types)} entries verified.</span>'
        if results["chain_verified"]
        else '<span class="status bad">Verification failed.</span>'
    )
    edited_status = (
        f'<span class="status bad">Verification failed at entry {k + 1}.</span>'
        if not results["tampered_copy_verified"]
        else '<span class="status bad">Edited copy was not detected.</span>'
    )

    repo_link = (
        f'<p>Source: {link("Responsible AI Toolkit on GitHub", prov["repo_url"])}.</p>'
        if prov["repo_url"] else ""
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Model review record: synthetic lending demo</title>
<style>{REPORT_CSS}</style>
</head>
<body>
<main>
<h1>Model review record</h1>
<p class="provenance">Synthetic lending demo of the Responsible AI Toolkit. {provenance_line}</p>
<p class="notice">All data here is synthetic, generated with a fixed seed ({SEED}). This page shows what the software did on that data. It is not a model validation, a deployment, or evidence of use by any institution.</p>

<p class="lede">{len(rows)} model decisions were checked against two rules. {len(rows) - len(reviewed)} passed and were automated. {len(reviewed)} failed a rule and went to a human reviewer, who overrode the model {times}. Every step was written to a hash-chained audit log.</p>

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
<p>A copy of the same log with one value changed: the model confidence for {e(TAMPER_TARGET)}, edited to 0.99. {edited_status} Entries after that point are not verified.</p>
<ul class="chain" aria-label="Edited copy of the audit log">{"".join(edited)}</ul>
<ul class="legend">
<li><span style="background:var(--broken)"></span>Edited entry, hash mismatch</li>
<li><span style="background:repeating-linear-gradient(135deg, var(--unchecked) 0 3px, transparent 3px 6px)"></span>Not verified</li>
</ul>
</div>
<p>Chain head hash: <code>{e(results["head_hash"])}</code>. Keeping this value somewhere separate from the log makes later removal of entries detectable.</p>

<h2>Decisions</h2>
<p>Rules: model confidence of at least 0.60, and required inputs present (debt-to-income, credit history). Thresholds are illustrative configuration values, not legal or supervisory standards.</p>
<div class="table-wrap">
<table>
<thead><tr><th>Application</th><th>Model</th><th>Confidence</th><th>Policy findings</th><th>Route</th><th>Reviewer</th><th>Reviewer decision</th></tr></thead>
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

<h2>Reproduce this run</h2>
<pre>python -m pip install -e ".[dev]"
python examples/model_review_demo.py demo_output</pre>
<p>The decisions, controls, and audit results match on every run. Timestamps, entry IDs, and hashes differ.</p>
<p>Raw output: <a href="decisions.json">decisions.json</a> and <a href="audit_log.json">audit_log.json</a>.</p>
{repo_link}
<footer>Responsible AI Toolkit is experimental open-source software under the Apache License 2.0.</footer>
</main>
</body>
</html>
"""


def main() -> int:
    output_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("demo_output")
    results = run(output_dir)
    print((output_dir / "summary.md").read_text())
    return 0 if results["all_checks_as_expected"] else 1


if __name__ == "__main__":
    sys.exit(main())
