# Changelog

Dates are commit dates on `main`. Each commit runs the test suite (the **CI** workflow) and the synthetic demo and example scripts (the **Demo run** workflow); results are listed under the repository's Actions tab. Planned work appears in the README under "Planned development" and is added here only once it exists.

## Unreleased (planned tag: v0.2.0)

All changes below were made on September 22, 2026.

### Corrections

- **Drift policy:** a missing PSI value now fails the drift check instead of passing it.
- **Policy evaluation:** an evaluation with no policies no longer reports that all checks passed; the pass rate is reported as unavailable; results distinguish checks that were not evaluated, errored, passed, or failed.
- **Governance exports:** generating an export no longer creates extra snapshots.
- **Governance summary:** months with no significant drift are no longer counted as drift alerts.
- **Human review:** a decision is accepted only from a registered, active reviewer who is assigned to the case and authorized for its category; a second decision on a completed case is rejected and leaves the original decision, reviewer statistics, and decision log unchanged.
- **Audit log:** each entry's hash now covers its system ID and displayed timestamp; logged data is copied when recorded, so later changes by the calling code cannot alter it; hash algorithms other than SHA-256 are rejected instead of silently ignored.
- **Drift measures:** production values outside the reference range are counted in the end bins instead of being dropped; the Wasserstein distance calculation is corrected; empty or non-finite inputs are rejected.

### Documentation and claims

- Removed statements that components satisfy, align with, or produce evidence required by specific regulations, supervisory guidance, or audit standards. Built-in policy rules no longer attach regulatory labels by default.
- README rewritten with scope, limitations, and current governance references, including a dated project history.
- The lending example was replaced by `examples/lending_model_review.py`, which uses no demographic attributes.

### Tests

Regression tests added for missing PSI and drift thresholds; empty policy reporting, evaluation statuses, and stable exports; deterministic human-review behavior; reviewer authorization; repeat decisions on completed cases; audit-log tampering (field edits, payload edits, deletion, and reordering); drift measures; and the governance summary.

### Examples and published output

- `examples/model_review_demo.py` runs two illustrative synthetic configurations (consumer lending decisions and property insurance quotes) and month-by-month drift monitoring, and publishes the results as review records to the project's GitHub Pages site on each commit.

The demo uses synthetic data and fixed rules standing in for reviewer judgment. It shows software behavior only. It is not a model validation, a pilot, an institutional evaluation, or evidence of use by any institution.

## v0.1.0: initial public version

Commits through April 15, 2026 (last commit `54ce201`). The automated test run on that commit failed. The test suite has passed on `main` since the September 2026 changes.

## Known limitations

This register lists known limitations of the current code. It will be updated as items are fixed or found.

| Area | Limitation | Status |
| --- | --- | --- |
| Human review | A reviewer registered with no roles can decide cases in any category. | Open |
| Audit log | Entries are held in memory; persistence, access control, and retention must be provided by the system using the toolkit. | By design; documented |
| Audit log | The hash chain uses no secret key. Anyone able to change the log can recompute its hashes. Removal of entries from the end is detectable only against an independently kept head hash. | By design; documented |
| Fairness module | Not reviewed in the September 2026 corrections. Group-level metrics are diagnostic and do not establish the presence or absence of unlawful discrimination. | Review pending |
| Insurance example | `examples/insurance_underwriting.py` was checked only for running without error, not reviewed for content. | Review pending |
| Thresholds | Default thresholds (for example PSI 0.10 and 0.25, confidence 0.60) are illustrative configuration values, not legal or supervisory standards. | By design; documented |
| Governance summary | The `toolkit_version` field is a fixed value rather than read from the installed package. | Open |
| Security | No independent security review has been performed. Not intended for production or real customer data. | By design; documented |
