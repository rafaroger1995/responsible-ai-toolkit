"""B1 review regressions, specified after inspecting the recovered implementation."""
import copy

import pytest

from responsible_ai_toolkit.decision_support import assess, ReviewSession
from responsible_ai_toolkit.decision_support.fixtures import profiles, exception_challenge, challenge
from responsible_ai_toolkit.policy.engine import Policy, PolicyEngine


@pytest.mark.parametrize("domain", ["lending", "insurance"])
def test_new_explanation_conflict_cannot_earn_verified_closure(domain):
    record, authority = exception_challenge(domain)
    config = profiles()
    p = next(p for p in config if p["domain"] == domain)
    session = ReviewSession(assess(record, config),
        [{"id": "senior", "roles": [p["review_role"], p["authority_role"]]}],
        review_role=p["review_role"], authority_role=p["authority_role"])
    record["explanation"]["reason_code"] = "NEW-UNRESOLVED-MISMATCH"
    session.provide_evidence("senior", record, config, [authority], "synthetic:new-authority-with-notice-conflict")
    result = session.retest("senior", run_id="closure-boundary").to_dict()
    assert result["decision_support"] == "supported_within_scope"
    assert result["explanation_status"] == "inconsistent"
    before = session.snapshot()
    with pytest.raises(ValueError, match="eligible"):
        session.close("senior", "correction_verified", "Decision evidence recovered but notice conflict remains")
    assert session.snapshot() == before
    assert session.snapshot()["open"]


def test_policy_callables_cannot_change_evaluated_input_for_other_rules_or_export():
    engine = PolicyEngine()
    def mutating_rule(context):
        context["nested"]["value"] = .99
        return False
    engine.add_policy(Policy("mutating", "Mutating test", "Challenge only", mutating_rule))
    engine.add_policy(Policy("limit", "Original input", "Should see original input", lambda c: c["nested"]["value"] < .45))
    original = {"nested": {"value": .4}}
    report = engine.evaluate(original)
    assert original == {"nested": {"value": .4}}
    assert report.results[1].passed
    assert report.context_snapshot == original


@pytest.mark.parametrize("domain", ["lending", "insurance"])
def test_duplicate_rule_identity_is_unknown_not_a_clean_pass(domain):
    record = challenge(domain)
    config = profiles()
    p = next(p for p in config if p["domain"] == domain)
    p["rules"].append(copy.deepcopy(p["rules"][0]))
    result = assess(record, config).to_dict()
    assert result["decision_support"] == "insufficient_evidence"
    assert any(f["rule_id"] == "rule-identity" for f in result["rule_findings"])
