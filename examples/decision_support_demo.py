"""Generate actual Python evidence for the Lab; Lab inspection is not live Python.

Run: python examples/decision_support_demo.py OUTPUT_DIRECTORY
Original B1 records are retained without invented substantive inputs or policy.
"""
from __future__ import annotations
import copy
import hashlib
import json
import platform
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from model_review_demo import SCENARIOS, run_scenario
from responsible_ai_toolkit.decision_support import assess, canonical, digest, implementation, ReviewSession
from responsible_ai_toolkit.decision_support.fixtures import profiles, new_challenges, exception_challenge, review_sequence

ROOT = Path(__file__).resolve().parents[1]


def original_b1():
    source = json.loads((ROOT / "fixtures/decision_support/b1_original_before.json").read_text())
    records = []
    for raw in source["result"]["records"]:
        record = {"case_id": raw["id"], "domain": "lending", "fixture_family": "original_b1_no_substantive_enrichment",
                  "source_record": raw, "source_ref": "b1_original_before.json#" + raw["id"],
                  "kind": "final_decision", "outcome": "decline", "inputs": {}, "joined_record_count": 1,
                  "adapter_version": "b1-reason-only/1.0", "model_ref": None, "decision_at": None,
                  "trace": {"reason_code": raw["final_reason"], "source_ref": "b1_original_before.json#" + raw["id"]} if raw["final_reason"] else None,
                  "explanation": {"reason_code": raw["notice_reason"]},
                  "metadata_declaration": "No substantive input, policy, authority or decision date supplied by original fixture."}
        records.append(record)
    return source, records


def negative_reviews(domain, run_id):
    output=[]
    for variant in ["unchanged_evidence", "reason_only", "invalid_authority", "later_relaxed_policy", "risk_acceptance"]:
        record, authority = exception_challenge(domain)
        config=profiles();p=next(x for x in config if x["domain"]==domain)
        session=ReviewSession(assess(record, config, run_id=run_id + "/" + variant),
                    [{"id":"senior","roles":[p["review_role"],p["authority_role"]]}],review_role=p["review_role"],authority_role=p["authority_role"])
        registry=[]
        if variant=="risk_acceptance":
            session.close("senior","authorized_risk_acceptance","Synthetic administrative acceptance; evidence issue not corrected.")
            output.append({"domain":domain,"variant":variant,"review":session.snapshot(),"closure_rejected":False})
            continue
        if variant=="reason_only": record["explanation"]["source_ref"]="synthetic/revised-notice"
        if variant=="invalid_authority": authority["status"]="revoked";registry=[authority]
        if variant=="later_relaxed_policy":
            p["rules"][0]["threshold"] = .9 if domain=="lending" else 9
            p["version"]="2026.2-counterfactual-relaxation";record["policy_version"]=p["version"];record.pop("exception")
        session.provide_evidence("senior",record,config,registry,"synthetic/change/"+variant)
        session.retest("senior",run_id=run_id + "/" + variant + "/retest")
        before=session.snapshot()
        try:
            session.close("senior","correction_verified","Attempt closure")
        except ValueError as exc:
            output.append({"domain":domain,"variant":variant,"closure_rejected":True,"rejection":str(exc),
                           "mutation_rejected_atomically":session.snapshot()==before,"review":session.snapshot()})
        else:
            raise AssertionError("Unsafe closure accepted: " + variant)
    return output


def build():
    now=datetime.now(timezone.utc).isoformat();run_id="decision-support/"+now
    try:
        commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
        dirty=bool(subprocess.check_output(["git","status","--porcelain"],cwd=ROOT,text=True).strip())
    except (OSError,subprocess.CalledProcessError):
        commit="source archive (see package manifest)";dirty=None
    records=[];original_outputs={}
    for sc in SCENARIOS:
        result=run_scenario(sc);original_outputs[sc.key]=result["rows"]
        for row in result["rows"]:
            records.append({"title":"Original toolkit recommendation", "family":"toolkit", "domain":sc.key,
                            "assessment":row["decision_support_assessment"],"routing":{k:row[k] for k in ["route","review_status","reviewer","review_decision","policy_findings"]}})
    for title, record, registry in new_challenges():
        records.append({"title":title,"family":"challenge","domain":record["domain"],
                        "assessment":assess(record,profiles(),registry,run_id=run_id).to_dict()})
    b1_source,b1_records=original_b1()
    for record in b1_records:
        records.append({"title":"Original reason-only B1 record","family":"b1_original","domain":"lending",
                        "assessment":assess(record,[],run_id=run_id).to_dict()})
    result={"schema_version":"trustera.lab-evidence/1.0","generated_at_utc":now,"run_id":run_id,
            "mode":"Inspection and replay of actual precomputed Python results; no live Python or institution actions",
            "provenance":{"toolkit_commit":commit,"working_tree_dirty":dirty,"python":platform.python_version(),"evaluator":implementation(),
                          "review_module":"responsible_ai_toolkit.decision_support.review.ReviewSession",
                          "review_sha256":hashlib.sha256((ROOT/"responsible_ai_toolkit/decision_support/review.py").read_bytes()).hexdigest(),
                          "baseline_toolkit":"86c7aef9af63f2dba4536056ae559b9a3282cff5","baseline_lab":"e2574aae902fc9778e72c653e3fdf7815c43aaa3"},
            "profiles":profiles(),"records":records,"original_toolkit_outputs":original_outputs,
            "b1_baseline":{"source_commit":b1_source["commit"],"source_sha256":digest(b1_source),"records":b1_source["result"]["records"],
                           "reason_consistent_before":96,"reason_consistent_after_mapping":114,"unknown_traces":6,"recorded_declines":120,
                           "substantive_insufficient":120,"customer_outcomes_changed":0},
            "review_sequences":[review_sequence(d,run_id+"/"+d) for d in ["lending","insurance"]],
            "negative_reviews":[x for d in ["lending","insurance"] for x in negative_reviews(d,run_id)],
            "metrics":{"assessed_records":len(records),"decision_support":dict(Counter(x["assessment"]["decision_support"] for x in records)),
                       "customer_outcomes_changed":0,"measured_staff_hours_saved":None,"independent_validation":"pending"},
            "limitations":["Synthetic retrospective component demonstration, not BP repeatability study",
                           "No real institutional policies, identity verification, source authentication or production integrations",
                           "No measured business savings, customer remedies or model-accuracy improvement",
                           "Independent practitioner assessment pending"]}
    result["integrity_sha256"]=digest(result)
    return result


def main():
    target=Path(sys.argv[1]) if len(sys.argv)>1 else ROOT/"decision_support_output"
    target.mkdir(parents=True,exist_ok=True)
    result=build()
    (target/"decision-support-results.json").write_text(json.dumps(result,indent=2))
    # This asset is generated from the identical JSON payload; no JS evaluator.
    (target/"decision-support-results.js").write_text("window.TrusteraDecisionEvidence="+canonical(result)+";\n")
    (target/"run-summary.json").write_text(json.dumps({"provenance":result["provenance"],"metrics":result["metrics"],"run_id":result["run_id"]},indent=2))
    print(json.dumps({"output":str(target),"run_id":result["run_id"],"metrics":result["metrics"],"toolkit_commit":result["provenance"]["toolkit_commit"],"working_tree_dirty":result["provenance"]["working_tree_dirty"]},indent=2))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
