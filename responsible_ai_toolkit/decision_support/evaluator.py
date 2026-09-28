"""One declarative evaluator for lending and insurance synthetic profiles.

Snapshots and SHA-256 bind exports to supplied values, not to the authenticity of
external facts. No identity service, customer action, or compliance conclusion.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import operator
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "trustera.decision-support/1.0"
OPS = {"<=": operator.le, "<": operator.lt, ">=": operator.ge, ">": operator.gt, "==": operator.eq}


def _safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return {"invalid_numeric": str(value)}
    if isinstance(value, dict):
        return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(v) for v in value]
    return value


def canonical(value: Any) -> str:
    return json.dumps(_safe(value), sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def implementation() -> dict:
    path = Path(__file__)
    return {"module": "responsible_ai_toolkit.decision_support.evaluator.assess",
            "version": "1.0.0", "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


@dataclass(frozen=True)
class Assessment:
    """Canonical immutable snapshot. All public exports are independent copies."""
    _json: str

    def to_dict(self):
        return json.loads(self._json)

    @classmethod
    def load(cls, value):
        value = copy.deepcopy(value)
        checksum = value.pop("integrity_sha256", None)
        if value.get("schema_version") != SCHEMA_VERSION or checksum != digest(value):
            raise ValueError("Assessment schema or integrity mismatch")
        value["integrity_sha256"] = checksum
        return cls(canonical(value))


def _date(value):
    if not isinstance(value, str):
        raise ValueError("Missing ISO decision date")
    return date.fromisoformat(value)


def _period(item, at):
    return _date(item["effective_from"]) <= at < _date(item["effective_to"])


def _number(value, rule):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("Expected a finite numeric input")
    if rule["type"] == "integer" and not isinstance(value, int):
        raise ValueError("Expected an integer count")
    if not rule["min"] <= value <= rule["max"]:
        raise ValueError("Input outside declared units/range")
    return value


def _exception(record, rule, registry, at):
    claimed = record.get("exception")
    if not claimed:
        return {"status": "none", "authority": None}
    if isinstance(claimed, dict) and claimed.get("rule_id") and claimed["rule_id"] != rule["id"]:
        return {"status": "none", "authority": None}
    if not isinstance(claimed, dict) or not claimed.get("authority_id"):
        return {"status": "missing", "authority": None, "detail": "Text is not authority evidence"}
    matches = [x for x in registry if isinstance(x, dict) and x.get("authority_id") == claimed["authority_id"]]
    if len(matches) != 1:
        return {"status": "missing" if not matches else "ambiguous", "authority": matches,
                "detail": "Expected exactly one synthetic authority record"}
    authority = matches[0]
    required = ["actor", "role", "case_ids", "rule_ids", "valid_from", "valid_to", "status", "source_ref"]
    if any(not authority.get(k) for k in required):
        return {"status": "missing", "authority": authority, "detail": "Incomplete authority record"}
    if any(not isinstance(authority[k], list) or not all(isinstance(v, str) for v in authority[k]) for k in ["case_ids", "rule_ids"]):
        return {"status": "missing", "authority": authority, "detail": "Authority scope must be explicit lists"}
    try:
        valid = (bool(claimed.get("basis")) and claimed.get("rule_id") == rule["id"] and authority["status"] == "active" and authority["actor"] == claimed.get("actor")
                 and authority["role"] in rule.get("exception_roles", [])
                 and record["case_id"] in authority["case_ids"]
                 and rule["id"] in authority["rule_ids"]
                 and _date(authority["valid_from"]) <= at < _date(authority["valid_to"]))
    except (ValueError, TypeError):
        return {"status": "missing", "authority": authority, "detail": "Malformed authority period"}
    return {"status": "valid" if valid else "invalid", "authority": authority,
            "detail": "Synthetic registry checked: actor, role, case, rule, period and status"}


def assess(record: dict, profiles: list, authorities: list | None = None, *, run_id="local",
           parent_assessment_id=None, assessment_kind="historical") -> Assessment:
    """Assess every record independently of model confidence and explanation.

    Effective periods are [from, to). A passing favorable restriction is necessary
    only. A decline needs a recorded basis that the selected profile permits.
    """
    record = copy.deepcopy(record)
    profiles = copy.deepcopy(profiles)
    authorities = copy.deepcopy(authorities or [])
    findings, errors, selected = [], [], None
    trace = record.get("trace") if isinstance(record.get("trace"), dict) else None
    explanation = record.get("explanation")
    if record.get("kind") == "recommendation" and explanation is None:
        explanation_status = "not_applicable"
    elif not trace or not trace.get("reason_code") or not isinstance(explanation, dict) or not explanation.get("reason_code"):
        explanation_status = "unknown"
    else:
        explanation_status = "consistent" if trace["reason_code"] == explanation["reason_code"] else "inconsistent"

    def issue(rule_id, status, detail, **more):
        findings.append({"rule_id": rule_id, "status": status, "detail": detail, **more})

    try:
        at = _date(record.get("decision_at"))
        if not record.get("case_id") or not record.get("source_ref") or not record.get("model_ref") or not record.get("adapter_version"):
            issue("evidence-identity", "insufficient_evidence", "Case, source, model and adapter references required")
        if record.get("joined_record_count") != 1:
            issue("record-join", "insufficient_evidence", "Exactly one source record must join to this decision")
        raw = record.get("source_record") or {}
        if (raw.get("ref") or raw.get("id")) and (raw.get("ref") or raw.get("id")) != record.get("case_id"):
            issue("source-identity", "insufficient_evidence", "Source record identifier does not match case identifier")
        candidates = []
        for profile in profiles:
            try:
                if profile.get("id") == record.get("policy_id") and profile.get("domain") == record.get("domain") and _period(profile, at):
                    candidates.append(profile)
            except (KeyError, ValueError, TypeError, AttributeError) as exc:
                errors.append({"scope": "policy-selection", "error": str(exc)})
        if len(candidates) != 1:
            issue("policy-selection", "insufficient_evidence", "Missing or ambiguous applicable historical policy")
        else:
            selected = candidates[0]
            if not selected.get("version") or not selected.get("rules"):
                issue("policy-definition", "insufficient_evidence", "Version and nonempty required checks are mandatory")
            rules = selected.get("rules", [])
            rule_ids = [r.get("id") for r in rules if isinstance(r, dict)]
            if (len(rule_ids) != len(rules) or not all(isinstance(r, str) and r.strip() for r in rule_ids)
                    or len(set(r for r in rule_ids if isinstance(r, str))) != len(rule_ids)):
                issue("rule-identity", "insufficient_evidence", "Rule IDs must be present and unique within a profile")
            if not selected.get("decision_types") or not selected.get("favorable_outcomes"):
                issue("policy-scope", "insufficient_evidence", "Explicit outcome scope is required")
            if not record.get("policy_version"):
                issue("recorded-policy", "insufficient_evidence", "Applied policy version is absent")
            elif selected.get("version") and record["policy_version"] != selected["version"]:
                issue("recorded-policy", "policy_conflict", "Recorded policy differs from the policy effective on the decision date",
                      recorded=record["policy_version"], applicable=selected.get("version"))
            if record.get("kind") == "final_decision" and (not trace or not trace.get("source_ref")):
                issue("decision-trace", "insufficient_evidence", "Contemporaneous final-decision trace required")
            if record.get("kind") not in {"recommendation", "final_decision"}:
                issue("decision-kind", "insufficient_evidence", "Recommendation or final decision must be explicit")
            outcome = record.get("outcome")
            if not outcome:
                issue("decision-outcome", "insufficient_evidence", "Recorded outcome is missing")
            elif outcome not in selected.get("decision_types", []):
                issue("outcome-scope", "not_applicable", "Recorded outcome is outside this profile's decision types")
            else:
                favorable = outcome in selected.get("favorable_outcomes", [])
                basis = trace.get("basis_rule_id") if trace else None
                applicable = [r for r in selected.get("rules", []) if r.get("purpose") == "eligibility" and
                              (outcome in r.get("applies_to", []) if favorable else r.get("id") == basis and r.get("supports_adverse"))]
                if not applicable:
                    issue("applicable-basis", "insufficient_evidence", "No required check or permitted documented adverse basis; no approval inference")
                for rule in applicable:
                    try:
                        inputs = record.get("inputs") or {}
                        value = _number(inputs.get(rule["field"]), rule)
                        if record.get("input_units", {}).get(rule["field"]) != rule["unit"]:
                            raise ValueError("Input unit missing or different from rule unit")
                        threshold = _number(rule["threshold"], rule)
                        satisfies = OPS[rule["operator"]](value, threshold)
                        exception = _exception(record, rule, authorities, at) if favorable else {"status": "not_applicable", "authority": None}
                        if exception["status"] in {"missing", "ambiguous"}:
                            status = "insufficient_evidence"
                        elif exception["status"] == "invalid":
                            status = "policy_conflict"
                        elif favorable:
                            status = "pass" if satisfies or exception["status"] == "valid" else "policy_conflict"
                        else:
                            status = "pass" if not satisfies else "policy_conflict"
                        issue(rule["id"], status, "Necessary favorable restriction" if favorable else "Test of recorded adverse basis",
                              field=rule["field"], value=value, unit=rule["unit"], operator=rule["operator"],
                              threshold=threshold, exception=exception, rule_snapshot=rule)
                    except (KeyError, ValueError, TypeError, AttributeError) as exc:
                        errors.append({"scope": rule.get("id", "unknown-rule"), "error": str(exc)})
                        issue(rule.get("id", "unknown-rule"), "insufficient_evidence", str(exc))
    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        errors.append({"scope": "record", "error": str(exc)})
        issue("record-evidence", "insufficient_evidence", str(exc))

    states = {f["status"] for f in findings}
    status = ("policy_conflict" if "policy_conflict" in states else
              "insufficient_evidence" if "insufficient_evidence" in states or errors or not findings else
              "not_applicable" if "not_applicable" in states else "supported_within_scope")
    evidence = {"record": record, "profiles": profiles, "authority_registry": authorities}
    value = {"schema_version": SCHEMA_VERSION, "case_id": record.get("case_id"), "run_id": run_id,
             "assessment_id": "AS-" + digest([run_id, evidence, parent_assessment_id, assessment_kind])[:20],
             "parent_assessment_id": parent_assessment_id, "assessment_kind": assessment_kind,
             "implementation": implementation(), "input_snapshot": record,
             "record_sha256": digest(record), "source_record_sha256": digest(record.get("source_record", record)),
             "evidence_sha256": digest(evidence), "evidence_bundle": evidence,
             "policy_snapshot": selected, "policy_sha256": digest(selected) if selected else None,
             "decision_support": status, "explanation_status": explanation_status,
             "explanation_evidence": {"trace": trace, "explanation": explanation},
             "rule_findings": findings, "evaluation_errors": errors,
             "review_action": "review_required" if status in {"policy_conflict", "insufficient_evidence"} or explanation_status == "inconsistent" else "no_concern_in_evaluated_scope",
             "reviewer_disposition": None, "linked_retests": [],
             "limitations": ["Synthetic policy and authority fixtures", "Necessary checks only, not complete eligibility",
                             "Reason matching does not prove model explanation fidelity", "No customer outcome changed",
                             "Hashes bind values, not source authenticity or immutable storage"]}
    value["integrity_sha256"] = digest(value)
    return Assessment(canonical(value))
