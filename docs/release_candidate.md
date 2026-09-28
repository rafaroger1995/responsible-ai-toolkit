# Candidate reproduction and reuse

This is a source candidate until an identified remote commit/tag, release, CI run
and Pages publication are separately verified. Public results can lag the source.
No release version is assumed by this document.

## Reproduction from an identified checkout

```sh
git rev-parse HEAD
python -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pytest tests/ -q
python examples/lending_model_review.py
python examples/insurance_underwriting.py
python examples/release_check.py /tmp/toolkit-release-checks
```

The final command runs both example generators, checks the actual output, writes
`release-checks.json`, `reuse-report.md`, `release.html`, complete assessment JSON,
review frames and a SHA-256 manifest. A failed check leaves its observed values
in the output and returns failure. The output directory should be outside the
checkout so generated files do not make the source dirty.

## What is actually reused

| Layer | Shared executable | Differences to disclose |
|---|---|---|
| Decision support | `decision_support.evaluator.assess` | Profile IDs/versions, DTI versus claims restrictions, input units and required evidence |
| Human review | `decision_support.review.ReviewSession` with `HITLOrchestrator` | Review and disposition role strings and supplied authority fixtures |
| Assessment export | `Assessment`, canonical snapshots and integrity hashes | Actual records, policy/authority snapshots and observed results |
| Original-record adaptation | `fixtures.adapt_original` | Domain selection supplies field/unit mappings and model reference labels |
| Domain-specific work | Disclosed separately from shared core | Two original generators, manual reviewer stand-ins and challenge-fixture construction |

The check records hashes before and after both domains and the new configuration
probes. Original inputs are compared with the retained 24-record baseline fixture.
There are also 11 explicitly new challenges and 120 retained reason-only B1 records.
All original B1 substantive assessments remain insufficient. Their reason-mapping
replay never establishes decision support.

The two extra configuration probes vary a declared threshold/version and withhold
a vendor/model reference, using the frozen evaluator and review functions. The
reported duration measures automated transformation and execution only. It does
not measure programmer or domain-specialist time, historical adaptation costs,
customer savings or institutional transfer. The larger three-profile study and its
90 predefined executions remain planned.

## Interpreting success and failure

Exact expected agreement is required for source IDs, counts, states and selected
input values. Run timestamps, review UUIDs, elapsed durations and hashes containing
those values differ on rerun. Drift output may vary with NumPy version; no numeric
tolerance excuses a different review state. Record the dependency versions used.

The new candidate tests also check that a corrected decision-evidence gap cannot
close a newly introduced explanation mismatch, that a custom policy callable
cannot alter the other rules' inputs, and that ambiguous duplicate rule IDs do not
produce a clean assessment. The standard synthetic examples are unchanged in
their substantive expected results by these boundary corrections.

## Website relationship

The Python source and Workflow Lab are distinct applications. The website's
decision-support view reads/replays exported Python results; other workflow
demonstrations calculate separately in JavaScript. The generated JSON/JS here
preserves `trustera.lab-evidence/1.0`. A website owner must replace both assets,
identify the new source and verify affected behavior before claiming synchronization.

## Licensing and limits

LICENSE contains the complete Apache 2.0 text from
https://www.apache.org/licenses/LICENSE-2.0.txt . The original project attribution
is retained in NOTICE. The license restoration does not establish employer or
third-party rights to privately supplied material; publish only reviewed source
and permitted synthetic outputs. Dependencies are installed separately and their
licenses remain applicable; no dependency code is vendored in this repository.

All source references, policies, identities and outcomes in these demonstrations
are synthetic assertions. Hashes detect alterations relative to retained values;
they do not establish authenticity or immutable storage. Identity, durable storage,
concurrency, domain validation, institutional adoption and independent review remain
outside this finite release demonstration.
