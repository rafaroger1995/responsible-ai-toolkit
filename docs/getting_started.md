# Getting started

This is an experimental Python toolkit and synthetic demonstration. It has no
production authentication, persistent review store or independent validation.

## Install the inspected source

```bash
git clone https://github.com/rafaroger1995/responsible-ai-toolkit.git
cd responsible-ai-toolkit
python -m pip install -e ".[dev]"
```

Record `git rev-parse HEAD` and your Python/dependency versions with any results.
An installation from a package index has not been verified for this guide.

## Interpret the components

- Fairness metrics are descriptive diagnostics. A default ratio or threshold is
  not a fair-lending standard or a determination of unlawful discrimination.
- Drift measures compare supplied populations. Configured thresholds are
  illustrative, and a signal does not establish a cause or the correct remedy.
- Policy checks execute configured rules. Passing them does not establish
  compliance with a statute, regulation, supervisory guidance or audit standard.
- The in-memory audit hash chain can expose some changes when checked against
  trusted prior records. A writer can recompute the chain; independent anchors,
  access controls, persistence and retention are not provided by this module.

## Human review

Register explicit roles matching a case category; an empty role list confers no
authority. Duplicate reviewer IDs are rejected so registration cannot silently
replace authority or reset workload. These are application checks, not verified
identities or a secure authorization service.

`request_info` keeps the case open in `waiting_info`, retaining its reviewer and
workload. An assigned authorized reviewer can later record a final disposition.
`escalate` releases the original assignment and routes to the matching policy's
`escalate_to_role`. If no qualified reviewer is available it remains pending; if
no policy matches it remains escalated and unassigned. Such cases stay eligible
for overdue reporting. There is currently no public reassignment/resumption API
for these unassigned cases; that remains a development limitation.

Only approve, reject and override complete a case. Requests and escalations are
events, not completed reviews. The original SLA deadline is retained. The SLA
rate is `None` when there are no completed reviews; callers must handle missing
data. The override rate uses completed reviews as its denominator.

## Examples and checks

```bash
python examples/lending_model_review.py
python examples/insurance_underwriting.py
python examples/model_review_demo.py demo_output
python -m pytest tests/ -q
```

The insurance example contains legacy diagnostic assumptions that have not
received a domain review. Running without errors is not content validation.
See `CHANGELOG.md` for dated changes and remaining limitations.
