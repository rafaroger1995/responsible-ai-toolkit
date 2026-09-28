"""Expected outcomes stated independently from the evaluator's implementation."""
import copy
import json
import pytest
from responsible_ai_toolkit.decision_support import Assessment, ReviewSession, assess, canonical, digest
from responsible_ai_toolkit.decision_support.fixtures import profiles, adapt_original, challenge, exception_challenge, authority, review_sequence
from responsible_ai_toolkit.policy.engine import Policy, PolicyEngine
from examples.model_review_demo import SCENARIOS, run_scenario


def status(record, config=None, registry=None):
    return assess(record, profiles() if config is None else config, registry).to_dict()


def favorable(domain):
    r = challenge(domain)
    r["outcome"] = "approve" if domain == "lending" else "offer"
    r["inputs"]["debt_to_income" if domain == "lending" else "prior_claims"] = .45 if domain == "lending" else 2
    return r


@pytest.mark.parametrize("key,case_id,field,expected,recommendation,confidence", [
    ("lending", "SYN-003", "debt_to_income", .48, "approve", .75),
    ("insurance", "INS-012", "prior_claims", 4, "offer", .84)])
def test_original_inputs_predictions_preserved_and_routed(key, case_id, field, expected, recommendation, confidence):
    sc = next(s for s in SCENARIOS if s.key == key)
    original = next(r for r in sc.make_records(sc.seed) if r["ref"] == case_id)
    before = copy.deepcopy(original)
    assert original[field] == expected and original["model_recommendation"] == recommendation and original["confidence"] == confidence
    output = next(r for r in run_scenario(sc)["rows"] if r["record_ref"] == case_id)
    result = output["decision_support_assessment"]
    assert result["input_snapshot"]["source_record"] == before == original
    assert result["decision_support"] == "policy_conflict"
    assert output["model_recommendation"] == recommendation and output["route"] == "review required"
    assert output["review_status"] == "open" and output["review_decision"] is None
    assert result["explanation_status"] == "not_applicable"


@pytest.mark.parametrize("domain,value,expected", [("lending", .45, "supported_within_scope"), ("lending", .450001, "policy_conflict"),
    ("insurance", 2, "supported_within_scope"), ("insurance", 3, "policy_conflict")])
def test_exact_boundaries(domain, value, expected):
    r = favorable(domain)
    r["inputs"]["debt_to_income" if domain == "lending" else "prior_claims"] = value
    assert status(r)["decision_support"] == expected


@pytest.mark.parametrize("domain", ["lending", "insurance"])
@pytest.mark.parametrize("value", [None, "0.45", True, float("nan"), float("inf"), -1, 2000])
def test_bad_inputs_are_uncertainty_not_pass(domain, value):
    r = favorable(domain)
    r["inputs"]["debt_to_income" if domain == "lending" else "prior_claims"] = value
    a = status(r)
    assert a["decision_support"] == "insufficient_evidence" and a["evaluation_errors"]
    assert Assessment.load(json.loads(canonical(a))).to_dict() == a


@pytest.mark.parametrize("domain", ["lending", "insurance"])
def test_units_and_count_types(domain):
    r = favorable(domain)
    field = "debt_to_income" if domain == "lending" else "prior_claims"
    r["input_units"][field] = "percent"
    assert status(r)["decision_support"] == "insufficient_evidence"
    if domain == "insurance":
        r["input_units"][field] = "count";r["inputs"][field] = 2.5
        assert status(r)["decision_support"] == "insufficient_evidence"


@pytest.mark.parametrize("domain", ["lending", "insurance"])
def test_adverse_basis_necessary_not_sufficient_and_confidence_independent(domain):
    r = challenge(domain)
    assert status(r)["decision_support"] == "supported_within_scope"
    r["inputs"]["debt_to_income" if domain == "lending" else "prior_claims"] = .2 if domain == "lending" else 1
    assert status(r)["decision_support"] == "policy_conflict"  # decline has no support in its recorded basis
    r["trace"]["basis_rule_id"] = "OTHER-UNDOCUMENTED-BASIS"
    assert status(r)["decision_support"] == "insufficient_evidence"  # no automatic approval
    r = favorable(domain)
    for confidence in [.01, .9999, None]:
        r["confidence"] = confidence
        assert status(r)["decision_support"] == "supported_within_scope"


@pytest.mark.parametrize("domain", ["lending", "insurance"])
@pytest.mark.parametrize("gap", ["policy", "version", "trace", "input", "join-zero", "join-two", "checks", "policy-version"])
def test_missing_evidence_and_joins(domain, gap):
    r = favorable(domain); p = profiles()
    if gap == "policy": p = []
    if gap == "version": r.pop("policy_version")
    if gap == "trace": r["trace"] = None
    if gap == "input": r["inputs"] = {}
    if gap.startswith("join"): r["joined_record_count"] = 0 if gap == "join-zero" else 2
    if gap in {"checks", "policy-version"}:
        target = next(x for x in p if x["domain"] == domain)
        if gap == "checks": target["rules"] = []
        else: target.pop("version")
    result = status(r, p)
    assert result["decision_support"] != "supported_within_scope"
    assert any(f["status"] == "insufficient_evidence" for f in result["rule_findings"])


def test_empty_or_ambiguous_profiles_never_pass():
    r = favorable("lending")
    p = profiles();p.append(copy.deepcopy(p[0]))
    assert status(r, p)["decision_support"] == "insufficient_evidence"
    assert status(r, [{}])["decision_support"] == "insufficient_evidence"


@pytest.mark.parametrize("domain", ["lending", "insurance"])
@pytest.mark.parametrize("variant,expected", [("valid", "supported_within_scope"), ("missing", "insufficient_evidence"),
    ("text", "insufficient_evidence"), ("expired", "policy_conflict"), ("revoked", "policy_conflict"),
    ("case", "policy_conflict"), ("rule", "policy_conflict"), ("role", "policy_conflict"),
    ("actor", "policy_conflict"), ("source", "insufficient_evidence"), ("ambiguous", "insufficient_evidence")])
def test_exception_authority_is_checked(domain, variant, expected):
    r, a = exception_challenge(domain); registry = [a]
    if variant == "missing": registry = []
    if variant == "text": r["exception"] = "Authorized income-document override"
    if variant == "expired": a["valid_to"] = "2026-01-02"
    if variant == "revoked": a["status"] = "revoked"
    if variant == "case": a["case_ids"] = ["unrelated"]
    if variant == "rule": a["rule_ids"] = ["unrelated"]
    if variant == "role": a["role"] = "reader"
    if variant == "actor": a["actor"] = "someone-else"
    if variant == "source": a["source_ref"] = None
    if variant == "ambiguous": registry.append(copy.deepcopy(a))
    assert status(r, registry=registry)["decision_support"] == expected


def test_matching_explanation_can_have_wrong_policy_and_reverse():
    r = challenge();r["inputs"]["debt_to_income"] = .43;r["policy_version"] = "retired-40"
    a = status(r)
    assert (a["decision_support"], a["explanation_status"]) == ("policy_conflict", "consistent")
    r = challenge();r["explanation"]["reason_code"] = "wrong"
    a = status(r)
    assert (a["decision_support"], a["explanation_status"]) == ("supported_within_scope", "inconsistent")


def test_select_historical_policy_not_latest_and_preserve_evidence():
    r = favorable("lending");r["inputs"]["debt_to_income"] = .43
    p = profiles(); later = copy.deepcopy(p[0]);later.update(version="2027.1", effective_from="2027-01-01", effective_to="2028-01-01");later["rules"][0]["threshold"] = .40;p.append(later)
    a = assess(r, p)
    assert a.to_dict()["policy_snapshot"]["version"] == "2026.1"
    assert a.to_dict()["decision_support"] == "supported_within_scope"
    r["policy_version"] = "2027.1"
    assert status(r,p)["decision_support"] == "policy_conflict"
    p[0]["rules"][0]["threshold"] = .1
    assert a.to_dict()["policy_snapshot"]["rules"][0]["threshold"] == .45


def session(domain, senior=True):
    r, a = exception_challenge(domain);p = next(x for x in profiles() if x["domain"] == domain)
    reviewers = [{"id": "analyst", "roles": [p["review_role"]]}]
    if senior: reviewers.append({"id": "senior", "roles": [p["authority_role"]]})
    s = ReviewSession(assess(r, profiles()), reviewers, review_role=p["review_role"], authority_role=p["authority_role"])
    return s,r,a


@pytest.mark.parametrize("domain", ["lending", "insurance"])
def test_request_info_escalation_unavailable_and_deadline(domain):
    s,r,a = session(domain, False);deadline=s.snapshot()["case"]["sla_deadline"]
    s.request_info("analyst", "Need source")
    v=s.snapshot();assert v["open"] and v["case"]["status"] == "waiting_info" and v["case"]["completed_at"] is None
    assert v["completed_reviews"] == v["verified_corrections"] == 0 and v["sla_compliance_rate"] is None
    assert v["reviewers"][0]["total_reviews"] == 0
    s.escalate("analyst", "Need designated authority")
    v=s.snapshot();assert v["open"] and v["case"]["assigned_to"] is None and v["case"]["sla_deadline"] == deadline
    before=s.snapshot()
    with pytest.raises(ValueError): s.assign("analyst")
    assert s.snapshot() == before


@pytest.mark.parametrize("domain", ["lending", "insurance"])
@pytest.mark.parametrize("action", ["request", "evidence", "retest", "escalate", "close", "assign"])
def test_removed_or_empty_roles_reject_atomically(domain, action):
    s,r,a=session(domain);s.update_roles("analyst", [])
    before=s.snapshot()
    operations = {"request":lambda:s.request_info("analyst", "Need source"), "evidence":lambda:s.provide_evidence("analyst",r,profiles(),[a],"source"),
                  "retest":lambda:s.retest("analyst",run_id="test"), "escalate":lambda:s.escalate("analyst","reason"),
                  "close":lambda:s.close("analyst","authorized_risk_acceptance","reason"), "assign":lambda:s.assign("analyst")}
    with pytest.raises(ValueError): operations[action]()
    assert s.snapshot() == before


@pytest.mark.parametrize("domain", ["lending", "insurance"])
@pytest.mark.parametrize("variant", ["unchanged", "reason_only", "invalid_authority", "missing", "relaxed_policy"])
def test_failed_or_ineligible_retest_cannot_close(domain, variant):
    s,r,a=session(domain);s.escalate("analyst","Need authority")
    p=profiles();registry=[]
    if variant == "reason_only": r["explanation"]["source_ref"] = "corrected-notice"
    if variant == "invalid_authority": a["status"]="revoked";registry=[a]
    if variant == "missing": r["inputs"]={}
    if variant == "relaxed_policy":
        target=next(x for x in p if x["domain"]==domain)
        target["rules"][0]["threshold"] = .9 if domain=="lending" else 9
        r.pop("exception")
    s.provide_evidence("senior",r,p,registry,"synthetic/change")
    result=s.retest("senior",run_id="negative").to_dict()
    if variant=="relaxed_policy": assert result["assessment_kind"]=="counterfactual"
    before=s.snapshot()
    with pytest.raises(ValueError):s.close("senior","correction_verified","attempt")
    assert s.snapshot()==before and s.snapshot()["open"]


@pytest.mark.parametrize("domain", ["lending", "insurance"])
def test_changed_evidence_links_then_authorized_closure_and_risk_acceptance(domain):
    frames=review_sequence(domain,"test")["frames"]
    assert all(f["review"]["open"] for f in frames[:-1])
    before=frames[0]["review"]["assessments"][0]
    final=frames[-1]["review"]
    assert final["assessments"][0]==before
    assert final["assessments"][1]["parent_assessment_id"]==before["assessment_id"]
    assert final["verified_corrections"]==1 and final["completed_reviews"]==1 and final["customer_outcome_changes"]==0
    assert frames[-2]["review"]["completed_reviews"]==0
    s,r,a=session(domain);s.escalate("analyst","Need authority")
    s.close("senior","authorized_risk_acceptance","Accept this unresolved evidence risk in the synthetic exercise")
    assert s.snapshot()["risk_acceptances"]==1 and s.snapshot()["verified_corrections"]==0
    assert s.snapshot()["assessments"][0]["decision_support"]=="insufficient_evidence"


@pytest.mark.parametrize("domain", ["lending", "insurance"])
def test_unauthorized_closure_after_role_change_and_changed_identity(domain):
    s,r,a=session(domain);s.escalate("analyst","Authority")
    s.provide_evidence("senior",r,profiles(),[a],"source");s.retest("senior",run_id="valid")
    s.update_roles("senior",[]);before=s.snapshot()
    with pytest.raises(ValueError): s.close("senior","correction_verified","reason")
    assert s.snapshot()==before
    s,r,a=session(domain);r["outcome"]="decline";before=s.snapshot()
    with pytest.raises(ValueError):s.provide_evidence("analyst",r,profiles(),[a],"source")
    assert s.snapshot()==before


def test_export_roundtrip_snapshots_tampering_and_shared_implementation():
    identities=[]
    for domain in ["lending","insurance"]:
        r,a=exception_challenge(domain);v=assess(r,profiles(),[a]);export=v.to_dict()
        assert Assessment.load(json.loads(json.dumps(export))).to_dict()==export
        identities.append(export["implementation"])
        export["input_snapshot"]["inputs"].clear()
        assert v.to_dict()["input_snapshot"]["inputs"]
        with pytest.raises(ValueError):Assessment.load(export)
        rv=review_sequence(domain,"test")["frames"][-1]["review"]
        assert json.loads(json.dumps(rv))==rv
    assert identities[0]==identities[1]


def test_general_policy_export_version_parameters_scope_and_typed_inputs():
    engine=PolicyEngine();p=Policy("p","p","test",lambda x: True,version="2",parameters={"limit":.45},scope={"paths":["manual","automated"]})
    engine.add_policy(p);context={"count":3,"values":[1,2],"flag":True};report=engine.evaluate(context)
    p.parameters["limit"]=.99;context["values"].append(3)
    result=report.results[0].to_dict()
    assert result["policy_version"]=="2" and result["parameters"]["limit"]==.45 and result["scope"]["paths"]==["manual","automated"]
    assert report.context_snapshot=={"count":3,"values":[1,2],"flag":True}
    report.context_snapshot.clear();assert engine.evaluation_history[0].context_snapshot


def test_outside_scope_is_not_applicable_not_success():
    r=challenge();r["outcome"]="withdrawn"
    assert status(r)["decision_support"]=="not_applicable"
