# Retrospective decision support

This component evaluates whether a recorded recommendation or final decision has
support under supplied fictional policy and evidence. It never executes a loan or
insurance decision. It keeps explanation consistency separate from decision support.

## The demonstrated gap and new specification

The original seeded `SYN-003` has DTI 0.48, recommendation `approve`, confidence
0.75; `INS-012` has four prior claims, recommendation `offer`, confidence 0.84.
At baseline 86c7aef, both use the automated route without a configured finding.
The separate manual stand-ins reject above 0.45 and at three claims respectively.

The new profiles explicitly make those restrictions necessary conditions across
both automated and manual paths. This common scope is a NEW synthetic specification,
not an institutional policy discovered in the old source. Under it, the two records
require review. Original input values and model predictions are retained. A favorable
restriction is not sufficient eligibility; a below-threshold record need not be
approved. A decline needs its own documented, permitted basis. High coverage remains
a referral check, not a claims eligibility restriction.

## One executable across domains

`responsible_ai_toolkit.decision_support.evaluator.assess` evaluates both profiles in
`profiles.json`; `ReviewSession` composes the existing `HITLOrchestrator` for both.
There is no domain-specific branch in the evaluator or review class.

| Difference | Lending | Insurance |
| --- | --- | --- |
| Favorable outcome | approve | offer |
| Necessary limit | DTI ratio <= 0.45 | prior-claims integer count < 3 |
| Additional required evidence | credit-history months | inspection year |
| Review / disposition roles | lending_review / credit_authority | insurance_review / underwriting_authority |
| Original adapter | same `adapt_original`; configured field/unit mappings | same function; configured field/unit mappings |
| Executable evaluator and review | identical functions and source hashes | identical functions and source hashes |

The original record generators, manual judgment stand-ins, and challenge-fixture
construction have domain-specific code. The adapters and fixtures are disclosed;
no adaptation-effort percentage or institution repeatability result is inferred.

## Findings and evidence

Decision support is `supported_within_scope`, `policy_conflict`,
`insufficient_evidence`, or `not_applicable`. Known contradictions take precedence
in the overall label; any concurrent missing evidence remains in rule findings.
Empty checks, missing policy/version, missing inputs, malformed numbers, nonfinite
values, unit mismatches and ambiguous joins never produce a clean pass. All records
are evaluated independently of confidence. Explanation is `consistent`,
`inconsistent`, `unknown`, or `not_applicable`. Reason matching is not an attribution
method and does not establish model explanation fidelity.

Profiles declare effective dates as inclusive start/exclusive end. Selection uses
the historical decision date, not the newest version label. The matching-reason B1
challenge has DTI 0.43 and a recorded decline citing a retired 0.40 basis while the
applicable supplied limit is 0.45. It is flagged for review; no approval is invented.

The original 120 B1 records retain 96 consistent reasons before mapping replay,
114 after, six unknowns and 120 unchanged declines. They do not contain substantive
policy/input/authority evidence: all 120 receive insufficient decision evidence.
The eleven new challenges are separately labeled and do not enrich those originals.

Exception checks require the claimed actor, rule and basis, a unique matching
authority record, permitted role, case/rule scope, valid period, active status and
source reference. Missing and invalid authority are different findings. This is a
synthetic registry, not identity verification. Revocation status is supplied by the
fixture; a production registry would need point-in-time authority history.

`Assessment` stores a canonical JSON snapshot and returns independent copies.
Exports include typed inputs, original source record, model/adapter references,
policy and authority snapshots, rule findings/errors, source hashes and parent IDs.
NaN/Infinity are tagged as invalid numeric input in strict JSON rather than silently
serialized as valid numbers. Hashes detect changed exported values; they do not
authenticate external facts or provide immutable storage.

## Review and retest

Information requests remain open with no completion timestamp or credit. Escalation
uses a designated role; missing authority leaves the case open. Assignment and
disposition recheck explicit roles, including role removal. Rejected API transitions
are nonmutating. `update_roles` is an explicit fixture-admin helper, not a production
access-management endpoint. Review objects and identities are in memory.

Evidence receipt alone does not complete a review. A retest creates a new assessment
linked to its predecessor and records the evidence change. Unchanged evidence,
missing evidence, invalid authority and reason-only edits cannot verify a decision
concern. A changed historical policy is counterfactual and cannot clear the old
finding. Only the designated reviewer can record eligible `correction_verified` or
distinct `authorized_risk_acceptance`. Risk acceptance never earns correction credit.
The example correction supplies missing authority evidence; it does not remediate
a customer's loan or insurance outcome. Original assessments are retained, and
review events join them by case and assessment IDs rather than rewriting them.

## Run and inspect

```sh
python -m pip install -e '.[dev]'
python -m pytest -q
python examples/model_review_demo.py demo_output
python examples/decision_support_demo.py decision_support_output
python examples/lending_model_review.py
python examples/insurance_underwriting.py
```

The generated JSON and JS asset contain identical actual Python computations.
The separate Trustera Workflow Lab reads those exports; its controls inspect/replay
precomputed assessments and transitions. There is no JavaScript policy evaluator or
live Python API. Run IDs, review UUIDs, timestamps and hashes that include them vary
on rerun; decision statuses and bounded workflow invariants should agree.

Existing drift, fairness, audit-chain, general policy and model-review capabilities
remain available. This finite component demonstration is not the proposed BP
90-execution repeatability study, external validation, production security,
institution adoption, a measure of customer remedies or a time-savings study.
