# Getting Started

## Installation

```bash
pip install responsible-ai-toolkit
```

Or from source:

```bash
git clone https://github.com/rafaroger1995/responsible-ai-toolkit.git
cd responsible-ai-toolkit
pip install -e ".[dev]"
```

## Core Concepts

### Fairness Metrics

The toolkit evaluates AI systems against four fairness criteria commonly referenced by financial regulators:

- **Demographic Parity**: Are positive outcomes distributed proportionally across groups?
- **Equal Opportunity**: Among truly qualified applicants, are approval rates equal across groups?
- **Equalized Odds**: Are both true-positive and false-positive rates equal across groups?
- **Calibration**: When the model says "80% likely," does each group actually see 80% positive outcomes?

The default threshold is 0.80 (the "four-fifths rule"), consistent with EEOC adverse-impact guidance and commonly applied in fair lending examinations.

### Drift Monitoring

Models degrade over time as economic conditions, customer behavior, and data distributions shift. The toolkit monitors for drift using:

- **PSI (Population Stability Index)**: Industry standard in credit risk (PSI < 0.10 = stable, 0.10-0.25 = monitor, > 0.25 = review required)
- **KL Divergence**: Sensitive to tail behavior
- **Wasserstein Distance**: Robust to non-overlapping distributions

### Audit Logging

Every AI decision is recorded in a tamper-evident log using SHA-256 hash chaining. Each entry links to the previous entry's hash, making it computationally infeasible to alter historical records without detection. This satisfies audit trail requirements for SOC 2, FDIC examination, and model risk management.

### HITL Orchestration

The human-in-the-loop module routes flagged AI decisions to qualified reviewers based on configurable rules: confidence thresholds, risk scores, bias alerts, or random active-learning sampling. It tracks review SLAs, logs all decisions, and computes compliance statistics.

### Policy-as-Code

Regulatory requirements are encoded as executable rules that run against AI system outputs at runtime. Pre-built policy factories cover common requirements (fairness thresholds, drift limits, data completeness), and custom policies can be defined for institution-specific rules.

## Running the Examples

```bash
# Fair lending compliance for a community bank
python examples/lending_fairness.py

# Insurance underwriting governance
python examples/insurance_underwriting.py
```

## Running Tests

```bash
pytest tests/ -v
pytest tests/ -v --cov=responsible_ai_toolkit  # With coverage
```
