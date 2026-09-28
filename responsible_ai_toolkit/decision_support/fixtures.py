"""Declared adapters and NEW challenge fixtures, separate from the original 24 inputs."""
from __future__ import annotations
import copy
import json
from pathlib import Path
from .evaluator import assess
from .review import ReviewSession


def profiles():
    return json.loads(Path(__file__).with_name("profiles.json").read_text())


def adapt_original(source, domain):
    profile = next(p for p in profiles() if p["domain"] == domain)
    return {"case_id": source["ref"], "domain": domain, "source_record": copy.deepcopy(source),
            "fixture_family": "original_toolkit_inputs_with_new_scope_metadata",
            "source_ref": "synthetic/original/" + source["ref"], "decision_at": "2026-09-22",
            "kind": "recommendation", "outcome": source["model_recommendation"],
            "model_ref": "synthetic-" + domain + "-model/original-seeded-prediction", "confidence": source["confidence"],
            "adapter_version": "original-toolkit/1.0", "policy_id": profile["id"], "policy_version": profile["version"],
            "joined_record_count": 1, "inputs": {k: v for k, v in source.items() if k not in {"ref", "model_recommendation", "confidence"}},
            "input_units": {r["field"]: r["unit"] for r in profile["rules"]}, "trace": None,
            "metadata_declaration": "Decision date, units and common policy scope are NEW demo specifications; original values and prediction unchanged."}


def challenge(domain="lending", case_id=None):
    lending = domain == "lending"
    source = {"ref": case_id or ("B1-CH-CONTROL" if lending else "INS-CH-CONTROL"),
              "model_recommendation": "decline", "confidence": 0.98}
    source.update({"debt_to_income": .5, "credit_history_months": 60} if lending else {"prior_claims": 4, "last_inspection_year": 2025})
    record = adapt_original(source, domain)
    rule = "DTI-LIMIT" if lending else "CLAIMS-LIMIT"
    record.update(kind="final_decision", fixture_family="new_decision_support_challenge", adapter_version="challenge/1.0",
                  trace={"source_ref": "synthetic/trace/" + source["ref"], "basis_rule_id": rule, "reason_code": rule},
                  explanation={"reason_code": rule, "source_ref": "synthetic/notice/" + source["ref"]})
    return record


def authority(record, *, status="active"):
    lending = record["domain"] == "lending"
    return {"authority_id": "AUTH-" + record["case_id"], "actor": "fixture-senior",
            "role": "credit_authority" if lending else "underwriting_authority", "case_ids": [record["case_id"]],
            "rule_ids": ["DTI-LIMIT" if lending else "CLAIMS-LIMIT"], "valid_from": "2026-01-01", "valid_to": "2027-01-01",
            "status": status, "source_ref": "synthetic/signed-exception/" + record["case_id"],
            "basis": "Fictional compensating-factor exception; registry fixture, not verified identity"}


def exception_challenge(domain="lending", case_id=None):
    record = challenge(domain, case_id or ("B1-CH-AUTHORITY" if domain == "lending" else "INS-CH-AUTHORITY"))
    record["outcome"] = "approve" if domain == "lending" else "offer"
    record["source_record"]["model_recommendation"] = record["outcome"]
    auth = authority(record)
    record["exception"] = {"authority_id": auth["authority_id"], "actor": auth["actor"],
                           "rule_id": auth["rule_ids"][0], "basis": "Documented compensating factor"}
    record["trace"]["reason_code"] = "PERMITTED_EXCEPTION"
    record["explanation"]["reason_code"] = "PERMITTED_EXCEPTION"
    return record, auth


def new_challenges():
    output = []
    for domain, prefix in [("lending", "B1"), ("insurance", "INS")]:
        control = challenge(domain)
        output.append(("Supported documented decline", control, []))
        mismatch = copy.deepcopy(control)
        mismatch["case_id"] = prefix + "-CH-WRONG-REASON"
        mismatch["explanation"]["reason_code"] = "OTHER-FACTOR"
        output.append(("Supported decision, wrong explanation", mismatch, []))
        missing, auth = exception_challenge(domain)
        output.append(("Claimed exception, authority missing", missing, []))
        valid, valid_auth = exception_challenge(domain, prefix + "-CH-VALID-EXCEPTION")
        output.append(("Legitimate permitted exception", valid, [valid_auth]))
        revoked, revoked_auth = exception_challenge(domain, prefix + "-CH-REVOKED")
        revoked_auth["status"] = "revoked"
        output.append(("Revoked exception cannot clear concern", revoked, [revoked_auth]))
    wrong = challenge("lending", "B1-CH-RETIRED-POLICY")
    wrong["inputs"]["debt_to_income"] = .43
    wrong["policy_version"] = "2025.4-retired-40pct"
    wrong["trace"]["recorded_threshold"] = .40
    output.insert(0, ("Matching reason, unsupported policy basis", wrong, []))
    for _, record, _ in output:
        record["source_record"].update(record["inputs"], ref=record["case_id"])
        record["source_ref"] = "synthetic/challenge/" + record["case_id"]
        record["trace"]["source_ref"] = "synthetic/trace/" + record["case_id"]
        record["explanation"]["source_ref"] = "synthetic/notice/" + record["case_id"]
    return output


def review_sequence(domain, run_id):
    record, auth = exception_challenge(domain)
    profile = next(p for p in profiles() if p["domain"] == domain)
    initial = assess(record, profiles(), [], run_id=run_id)
    session = ReviewSession(initial, [{"id": "analyst", "roles": [profile["review_role"]]},
                              {"id": "senior", "roles": [profile["authority_role"]]}],
                              review_role=profile["review_role"], authority_role=profile["authority_role"])
    frames = [{"label": "Open concern", "review": session.snapshot()}]
    session.request_info("analyst", "Obtain the case-specific exception authorization and its effective period.")
    frames.append({"label": "Information requested · still open", "review": session.snapshot()})
    session.escalate("analyst", "Disposition requires designated authority")
    frames.append({"label": "Escalated to designated authority", "review": session.snapshot()})
    session.provide_evidence("senior", record, profiles(), [auth], auth["source_ref"])
    frames.append({"label": "New authority evidence received", "review": session.snapshot()})
    session.retest("senior", run_id=run_id + "/retest")
    frames.append({"label": "Historical retest passed · awaiting disposition", "review": session.snapshot()})
    session.close("senior", "correction_verified", "Missing authority evidence supplied and evaluated; historical customer outcome unchanged.")
    frames.append({"label": "Evidence gap corrected · authorized closure", "review": session.snapshot()})
    return {"domain": domain, "case_id": record["case_id"], "frames": frames}
