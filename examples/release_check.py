"""Generate and verify the finite synthetic release demonstration.

Added September 2026. Run from an identified checkout. This is an assisted
reproduction/reuse record; it is not an independent review or a field study.
"""
from __future__ import annotations
import copy
import hashlib
import html
import json
import platform
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from responsible_ai_toolkit.decision_support import assess, ReviewSession
from responsible_ai_toolkit.decision_support.fixtures import profiles, challenge
from model_review_demo import page

ROOT = Path(__file__).resolve().parents[1]
CORE = ["responsible_ai_toolkit/decision_support/evaluator.py", "responsible_ai_toolkit/decision_support/review.py",
        "responsible_ai_toolkit/hitl/orchestrator.py", "responsible_ai_toolkit/policy/engine.py"]
ARTIFACTS = CORE + ["responsible_ai_toolkit/decision_support/fixtures.py", "responsible_ai_toolkit/decision_support/profiles.json"]


def hashes(paths):
    return {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in paths}


def adaptation_probe(domain, run_id):
    """Time only the automated configuration transformation and execution."""
    started = datetime.now(timezone.utc).isoformat()
    tick = time.perf_counter()
    config = profiles()
    p = next(x for x in config if x["domain"] == domain)
    record = challenge(domain, "ADAPT-" + domain.upper())
    field = "debt_to_income" if domain == "lending" else "prior_claims"
    record["outcome"] = "approve" if domain == "lending" else "offer"
    record["inputs"][field] = .43 if domain == "lending" else 2
    record["source_record"].update(record["inputs"], model_recommendation=record["outcome"])
    record["fixture_family"] = "new_release_configuration_probe"
    baseline = assess(record, config, run_id=run_id + "/control").to_dict()
    before_threshold = p["rules"][0]["threshold"]
    p["rules"][0]["threshold"] = .40 if domain == "lending" else 2
    p["version"] = "2026.1-new-configuration-probe"
    record["policy_version"] = p["version"]
    changed = assess(record, config, run_id=run_id + "/changed").to_dict()
    elapsed = time.perf_counter() - tick
    # Vendor/model identity withholding is a separate evidence-gap variant.
    missing = copy.deepcopy(record)
    missing["model_ref"] = None
    # Restore a supporting policy so the missing-identity finding is visible as
    # insufficiency rather than being masked by an independent conflict label.
    p["rules"][0]["threshold"] = before_threshold
    unknown = assess(missing, config, run_id=run_id + "/missing-vendor-identity")
    session = ReviewSession(unknown, [{"id": "fixture-analyst", "roles": [p["review_role"]]}],
                            review_role=p["review_role"], authority_role=p["authority_role"])
    session.request_info("fixture-analyst", "Vendor/model reference not supplied")
    return {"domain": domain, "started_at_utc": started, "automated_transform_and_run_seconds": elapsed,
            "measurement_scope": "Script transform and evaluation only; excludes engineering, research, domain review and elapsed authoring time. No savings or human-hour claim.",
            "configuration_changes": {"field": field, "threshold_before": before_threshold,
                                      "threshold_after": .40 if domain == "lending" else 2,
                                      "version": record["policy_version"]},
            "expected": ["supported_within_scope", "policy_conflict", "insufficient_evidence"],
            "actual": [baseline["decision_support"], changed["decision_support"], unknown.to_dict()["decision_support"]],
            "assessments": [baseline, changed, unknown.to_dict()], "unresolved_review": session.snapshot()}


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "demo_output"
    out.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip())
    frozen = hashes(ARTIFACTS)
    for script in ("model_review_demo.py", "decision_support_demo.py"):
        result = subprocess.run([sys.executable, str(ROOT / "examples" / script), str(out)], cwd=ROOT,
                                text=True, capture_output=True)
        (out / (script + ".txt")).write_text(result.stdout + result.stderr)
        if result.returncode:
            raise RuntimeError(script + " failed; see retained output")
    data = json.loads((out / "decision-support-results.json").read_text())
    checks = []
    def check(name, expected, actual):
        checks.append({"check": name, "expected": expected, "actual": actual, "passed": expected == actual})
    byid = {x["assessment"]["case_id"]: x["assessment"] for x in data["records"]}
    expected_family_counts = {"toolkit": {"supported_within_scope": 13, "policy_conflict": 4, "insufficient_evidence": 7},
        "challenge": {"supported_within_scope": 6, "policy_conflict": 3, "insufficient_evidence": 2},
        "b1_original": {"insufficient_evidence": 120}}
    for family, expected in expected_family_counts.items():
        check(family + " outcomes", expected, dict(Counter(x["assessment"]["decision_support"] for x in data["records"] if x["family"] == family)))
    old = json.loads((ROOT / "fixtures/decision_support/original_toolkit_records.json").read_text())
    for domain, originals in old["records"].items():
        check(domain + " all original records retained", originals,
              [x["assessment"]["input_snapshot"]["source_record"] for x in data["records"] if x["family"] == "toolkit" and x["domain"] == domain])
    for case, field, value in (("SYN-003", "debt_to_income", .48), ("INS-012", "prior_claims", 4)):
        check(case + " finding", "policy_conflict", byid[case]["decision_support"])
        check(case + " original input", value, byid[case]["input_snapshot"]["inputs"][field])
        row = next(x for x in data["records"] if x["assessment"]["case_id"] == case)
        check(case + " stays open", "open", row["routing"]["review_status"])
    check("retired policy and matching explanation", ["policy_conflict", "consistent"],
          [byid["B1-CH-RETIRED-POLICY"]["decision_support"], byid["B1-CH-RETIRED-POLICY"]["explanation_status"]])
    for domain, prefix in (("lending", "B1"), ("insurance", "INS")):
        check(domain + " supported decision with wrong reason", ["supported_within_scope", "inconsistent"],
              [byid[prefix+"-CH-WRONG-REASON"]["decision_support"], byid[prefix+"-CH-WRONG-REASON"]["explanation_status"]])
    for sequence in data["review_sequences"]:
        f = sequence["frames"]
        check(sequence["domain"] + " review stays open until disposition", [True]*5+[False], [x["review"]["open"] for x in f])
        check(sequence["domain"] + " information request completion", None, f[1]["review"]["case"]["completed_at"])
        check(sequence["domain"] + " original retained", f[0]["review"]["assessments"][0], f[-1]["review"]["assessments"][0])
        check(sequence["domain"] + " predecessor link", f[0]["review"]["original_assessment_id"], f[-1]["review"]["assessments"][1]["parent_assessment_id"])
    for neg in data["negative_reviews"]:
        if neg["variant"] == "risk_acceptance":
            check(neg["domain"] + " risk acceptance is not correction", 0, neg["review"]["verified_corrections"])
        else:
            check(neg["domain"] + " rejects " + neg["variant"], [True, True, True],
                  [neg["closure_rejected"], neg["mutation_rejected_atomically"], neg["review"]["open"]])
    probes = [adaptation_probe(d, data["run_id"] + "/adapt/" + d) for d in ("lending", "insurance")]
    for probe in probes:
        check(probe["domain"] + " configured change and missing vendor identity", probe["expected"], probe["actual"])
        check(probe["domain"] + " unresolved vendor review", [True, "waiting_info", 0],
              [probe["unresolved_review"]["open"], probe["unresolved_review"]["case"]["status"], probe["unresolved_review"]["completed_reviews"]])
    check("shared evaluator across 155 records", [frozen[CORE[0]]], sorted({x["assessment"]["implementation"]["source_sha256"] for x in data["records"]}))
    check("frozen artifacts unchanged through both domains", frozen, hashes(ARTIFACTS))
    report = {"schema_version": "trustera.release-checks/1.0", "source_commit": commit, "working_tree_dirty": dirty,
              "started_at_utc": started, "finished_at_utc": datetime.now(timezone.utc).isoformat(),
              "python": platform.python_version(), "run_id": data["run_id"], "frozen_sha256": frozen,
              "checks": checks, "configuration_probes": probes,
              "summary": {"passed": sum(x["passed"] for x in checks), "failed": sum(not x["passed"] for x in checks)},
              "boundary": "Finite two-domain synthetic reuse check; no full insurer A/B/bank C study, independent validation, institutional adoption or efficiency measurement."}
    (out / "release-checks.json").write_text(json.dumps(report, indent=2))
    table = "| Reused function | Lending | Insurance |\n|---|---|---|\n| Decision evaluator | Same frozen assess() | Same frozen assess() |\n| Review/retest | Same ReviewSession and HITL | Same ReviewSession and HITL |\n| Export | Same assessment/review schemas | Same assessment/review schemas |\n| Adapter | Same adapt_original; configured field/unit map | Same adapt_original; configured field/unit map |\n| Domain changes | Lending seed/generator, DTI/history rules, roles, challenges | Insurance seed/generator, claims/inspection rules, roles, challenges |\n"
    md = f"# Reuse and release checks\n\nSource: `{commit}`. Dirty: `{dirty}`. Run: `{data['run_id']}`.\n\n" + table
    md += f"\n{report['summary']['passed']} expected-result checks passed; {report['summary']['failed']} failed. See `release-checks.json` for values and retained snapshots. These check totals are not the proposed 90-execution study.\n\n"
    md += "The original 24 toolkit records and 120 B1 reason records are retained. Eleven challenge fixtures are separately authored. Two additional configuration probes disclose changed thresholds/version and missing vendor identity. The common core is hashed before and after all runs. Automated configuration/run durations exclude authoring and specialist work; no historical hours or savings are reconstructed.\n\n" + report["boundary"] + "\n"
    (out / "reuse-report.md").write_text(md)
    e = html.escape
    body = f"<nav><a href='index.html'>Scenarios</a></nav><h1>Reproducible toolkit candidate</h1><p>Experimental synthetic-data results. Source <code>{commit}</code>. Working tree changed: {dirty}.</p><p>Run <code>{e(data['run_id'])}</code></p>"
    body += "<p><a href='decision-support-results.json'>Complete assessments and review replays</a> · <a href='release-checks.json'>Expected/actual checks</a> · <a href='reuse-report.md'>Reuse report</a></p>"
    body += f"<p>{report['summary']['passed']} checks passed; {report['summary']['failed']} failed.</p><h2>What the same code demonstrates</h2><p>{e(report['boundary'])}</p><p>Decision support is independent of explanation consistency. Missing evidence stays open; correction and authorized risk acceptance have distinct counts. Profiles and authority records are fictional.</p><h2>Capabilities</h2><p>Distribution drift and fairness metric functions remain available. Trace consistency is implemented; model-explanation fidelity remains future work. Governance and linked review are in-memory demonstrations.</p>"
    body += "<h2>Trustera Workflow Lab relationship</h2><p>The separate website inspects precomputed Python decision-support exports and uses separate JavaScript for other workflows. Its deployed data can represent an earlier version until the website owner replaces and verifies the export. This report does not establish that deployment.</p>"
    (out / "release.html").write_text(page("Toolkit source and reuse checks", body))
    manifest = {str(p.relative_to(out)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.rglob("*")) if p.is_file() and p.name != "SHA256SUMS.json"}
    (out / "SHA256SUMS.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"commit": commit, "dirty": dirty, **report["summary"], "run_id": data["run_id"]}, indent=2))
    return 0 if not report["summary"]["failed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
