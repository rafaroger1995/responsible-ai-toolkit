# Responsible AI Toolkit

**Reusable components for evaluating model behavior, organizing human review, and documenting AI governance in financial services.**

Responsible AI Toolkit is an open-source reference implementation bringing together audit logging, human review workflows, drift monitoring, configurable policy checks, governance reporting, and group-level outcome metrics.

The project focuses on workflows relevant to U.S. community banks and small and mid-sized insurers. Its objective is to make selected governance practices easier to implement, inspect, and adapt through shared software components and documented examples.

**Development status:** Experimental. Intended for technical evaluation and further development. The toolkit supports practitioner review; it does not establish regulatory compliance or certify a model’s fairness, safety, or suitability.

## Purpose

Evaluating an AI system involves connecting technical findings with accountable decisions: what was evaluated, which inputs and policies were used, who reviewed the results, and what action followed.

This project explores a reusable approach to that workflow. It brings evaluation, review, and documentation components into a common codebase so practitioners can examine how they work together and assess their suitability for different institutional settings.

The intended benefit is to reduce duplicated implementation effort while preserving institution-specific judgment, controls, and accountability. The extent of that benefit remains a question for practical evaluation.

## Components

| Component | Role |
| --- | --- |
| [Audit](responsible_ai_toolkit/audit/) | Event logging with hash-chain verification of stored entries. |
| [Human review](responsible_ai_toolkit/hitl/) | Review cases, reviewer assignments, authorization checks, and escalation workflows. |
| [Drift](responsible_ai_toolkit/drift/) | Distribution comparisons using measures such as population stability index, KL divergence, and Wasserstein distance. |
| [Policy](responsible_ai_toolkit/policy/) | Configurable checks evaluated against supplied metrics and inputs. |
| [Governance](responsible_ai_toolkit/governance/) | Reporting functions that bring available findings together for practitioner review. |
| [Fairness](responsible_ai_toolkit/fairness/) | Group-level outcome comparison metrics. |

These descriptions identify the components’ roles. Their effectiveness and suitability depend on the implementation, configuration, data, and operating environment.

## Getting started

### Install from source

Use a separate Python environment. Review [pyproject.toml](pyproject.toml) for Python version and dependency requirements.

```bash
git clone https://github.com/rafaroger1995/responsible-ai-toolkit.git
cd responsible-ai-toolkit
python -m pip install -e ".[dev]"
```

### Explore the examples

* [Lending model evaluation](examples/lending_fairness.py): an illustrative lending workflow.
* [Insurance underwriting](examples/insurance_underwriting.py): an illustrative insurance workflow.

```bash
python examples/lending_fairness.py
python examples/insurance_underwriting.py
```

Review each script’s inputs, configuration, and assumptions before execution. Use synthetic or appropriately authorized data.

The examples provide starting points for examining the components. They do not represent institutional deployments or independently validated outcomes.

### Run the tests

```bash
python -m pytest tests/ -v
```

To make an evaluation reproducible, record:

* The commit identifier or release.
* Python version and installed dependencies.
* Input data and configuration.
* Commands executed.
* Observed results, failures, and limitations.

Passing tests demonstrates the behavior covered by those tests in the recorded environment.

## Interpreting results

### Audit records

Hash-chain verification checks record consistency within the mechanism’s limits. It does not establish the truth or completeness of recorded information, and hash chaining alone does not provide immutable storage.

Operational assurance requires appropriate access controls, persistence, retention, and independently protected integrity references.

### Human review and policy checks

Configured rules and recorded review decisions support a governance workflow. Their reliability depends on appropriate inputs, reviewer authorization, relevant expertise, and documented follow-up.

A policy failure identifies a finding under the configured rule. A passing result does not constitute a complete compliance assessment. Missing or insufficient evidence requires separate attention.

### Drift monitoring

A monitoring threshold such as `PSI > 0.25` is an example configuration value. Exceeding it should prompt investigation in the context of model purpose, performance, materiality, and institutional policy. It does not automatically establish that a model is invalid or that revalidation is legally required.

### Fairness metrics

Group-level metrics describe particular relationships in the evaluated data. Their interpretation depends on the decision context, metric definition, group sizes, data quality, and available outcomes.

Values such as `0.80` in examples are illustrative thresholds, not universal legal standards for lending or insurance. A metric result alone does not establish the presence or absence of unlawful discrimination.

## Governance references

The following sources inform the project’s governance context. References do not imply complete implementation, certification, or endorsement.

| Source | Scope |
| --- | --- |
| [NIST AI Risk Management Framework](https://www.nist.gov/itl/ai-risk-management-framework) | A voluntary framework that can inform AI risk evaluation, oversight, and documentation. |
| [Revised interagency model-risk guidance, SR 26-2](https://www.federalreserve.gov/supervisionreg/srletters/SR2602.htm) | Nonbinding supervisory guidance issued April 17, 2026, superseding SR 11-7 and SR 21-8. Most relevant to banking organizations above $30 billion in assets, with potential relevance to certain smaller banks with significant model-risk exposure. |

The [SR 26-2 attachment](https://www.federalreserve.gov/supervisionreg/srletters/SR2602a1.pdf) excludes generative and agentic AI from its scope. It is not a universal requirement for community banks, credit unions, or insurers.

Applicable obligations depend on the institution, jurisdiction, activity, and model use. A framework label attached to a software rule does not establish that the rule fully implements the referenced source.

## Evaluation and limitations

Operational use requires institution-specific validation, security and privacy controls, appropriate reviewer authorization, and assessment of applicable requirements.

Evaluation should cover input validation, missing-data behavior, metric suitability, policy handling, review permissions, record integrity, persistence, and failure handling.

Reuse across institutions is an intended design objective. Demonstrating it requires documenting which components remain unchanged, which need configuration or code changes, and what manual work is required. The presence of multiple examples alone does not establish successful transfer or adoption.

## Project history

* **Through April 15, 2026:** Initial components, examples, and CI workflow.
* **September 22, 2026:** Corrections and regression tests for missing drift inputs, empty policy evaluations, policy evaluation statuses, export stability, reviewer authorization, repeat review decisions, and audit-chain integrity; removal of compliance and framework-alignment claims from code documentation.

Dates reflect the repository's commit history.

## Planned development

The following are development priorities, not completed capabilities:

* A versioned, project-defined assurance-record schema linking model identity, evaluation inputs, policy versions, findings, review decisions, exceptions, and remediation.
* Reproducible synthetic profiles for evaluating shared components across lending and insurance workflows.
* An adaptation report documenting shared code, configuration changes, implementation effort, and limitations.
* A practitioner review guide with version-specific reproduction instructions.
* Versioned releases documenting tested behavior and known limitations.

Longer-term possibilities include integration connectors and a user interface, subject to practical evaluation and demonstrated demand.

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md).

Useful contributions include reproducible bug reports, tests, documentation improvements, and practitioner feedback on specific workflows.

For technical findings, include the version or commit, environment, reproduction steps, expected behavior, and observed results. For practitioner feedback, identify the material reviewed, its potential usefulness, and any limitations or adaptation requirements.

Do not include confidential institutional information or personal data in public issues.

## Author

**Yash Lundia** — [GitHub](https://github.com/rafaroger1995)

M.S. in Management Science and Engineering, Stanford University.

This project focuses on reusable AI governance workflows for financial services. References to institutions, frameworks, or professional affiliations do not imply sponsorship or endorsement.

## License

[Apache License 2.0](LICENSE).
