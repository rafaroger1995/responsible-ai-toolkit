# Sample run: model review demo (synthetic data)

This page records the output of the first automated run of [`examples/model_review_demo.py`](../examples/model_review_demo.py).

- **Run:** [Demo run #1](https://github.com/rafaroger1995/responsible-ai-toolkit/actions/runs/35784026445), GitHub Actions, status: success
- **Commit:** [`84e016f`](https://github.com/rafaroger1995/responsible-ai-toolkit/commit/84e016f)
- **Workflow:** [Demo run](https://github.com/rafaroger1995/responsible-ai-toolkit/actions/workflows/demo.yml), which reruns the demo on every commit to `main`
- **Generated:** 2026-09-22 21:01 UTC

All data is synthetic (fixed seed 20260922). This output shows software behavior only. It is not a model validation, a deployment, or evidence of use by any institution.

## What the demo does

1. Records each model decision in a hash-chained audit log.
2. Evaluates each decision against two configurable policy checks: minimum model confidence of 0.60, and required inputs present (debt-to-income, credit history).
3. Routes decisions that fail a check to an assigned human reviewer.
4. Tests the reviewer controls: unregistered reviewers, non-assigned reviewers, and repeat decisions on completed cases.
5. Verifies the audit chain, then confirms that an edited copy fails verification.

Thresholds are illustrative configuration values, not legal or supervisory standards.

## Decisions

| Application | Model recommendation | Confidence | Policy findings | Route | Reviewer | Review decision |
| --- | --- | --- | --- | --- | --- | --- |
| SYN-001 | approve | 0.44 | min-confidence | human review | reviewer-a | approve |
| SYN-002 | approve | 0.67 | none | automated | - | - |
| SYN-003 | approve | 0.75 | none | automated | - | - |
| SYN-004 | decline | 0.71 | none | automated | - | - |
| SYN-005 | decline | 0.95 | required-inputs | human review | reviewer-b | reject |
| SYN-006 | decline | 0.44 | min-confidence | human review | reviewer-a | approve (override) |
| SYN-007 | approve | 0.44 | min-confidence | human review | reviewer-b | approve |
| SYN-008 | approve | 0.49 | min-confidence | human review | reviewer-a | approve |
| SYN-009 | decline | 0.50 | min-confidence | human review | reviewer-b | approve (override) |
| SYN-010 | approve | 0.94 | none | automated | - | - |
| SYN-011 | decline | 0.82 | none | automated | - | - |
| SYN-012 | approve | 0.60 | none | automated | - | - |

Applications: 12. Automated: 6. Routed to human review: 6. Reviewer overrides of the model: 2.

## Review controls

| Attempt | Result |
| --- | --- |
| Decision by an unregistered reviewer | Rejected: Reviewer 'reviewer-z' is not registered. |
| Decision by a registered reviewer who is not assigned | Rejected: Only the assigned reviewer may record a decision. |
| Decision by the assigned reviewer | Accepted |
| Second decision on the completed case | Rejected: Cannot record another decision on a completed case. |

## Audit log

Entries recorded: 42. Chain verification: passed.

Copy of the log with one prediction's confidence edited: verification failed, as expected.

## Reproduce this run

From a copy of the repository:

    python -m pip install -e ".[dev]"
    python examples/model_review_demo.py demo_output

The decision table, review controls, and audit results will match this page. Timestamps and entry IDs differ on each run.

GitHub keeps the run page's full report for a limited time; this page and the commit link above are the permanent record.
