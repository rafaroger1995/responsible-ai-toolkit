# Responsible AI Toolkit

**An open-source governance framework for deploying AI responsibly in regulated industries.**

[![CI](https://github.com/rafaroger1995/responsible-ai-toolkit/actions/workflows/ci.yml/badge.svg)](https://github.com/rafaroger1995/responsible-ai-toolkit/actions)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

---

## The Problem

Small and mid-sized financial institutions — community banks, regional insurers, credit unions — are increasingly adopting AI for lending decisions, claims processing, fraud detection, and underwriting. But they lack the specialized compliance teams and technical infrastructure to govern these systems responsibly.

Large enterprises spend millions on internal AI governance. Community banks and regional insurers? They're expected to meet the same regulatory standards with a fraction of the resources.

The result: a widening **AI governance gap** that leaves smaller institutions choosing between adopting AI without adequate oversight or forgoing AI entirely and falling behind.

## The Solution

The **Responsible AI Toolkit** provides production-ready, open-source modules that operationalize responsible AI governance for regulated industries — no specialized compliance team required.

Built around the five pillars of AI governance identified by federal regulators (NIST AI RMF, OCC/Fed SR 11-7, Treasury AI guidance):

| Module | What It Does | Regulatory Alignment |
|--------|-------------|---------------------|
| **Fairness** | Continuous bias detection using demographic parity, equal opportunity, equalized odds, and calibration metrics | ECOA, FHA, CFPB guidance |
| **Drift** | Model stability monitoring via PSI, KL divergence, and Wasserstein distance | OCC SR 11-7, Fed MRM |
| **Audit** | Tamper-evident, append-only logging with SHA-256 hash chaining | SOC 2, FDIC examination |
| **HITL** | Human-in-the-loop orchestration with escalation policies and SLA enforcement | NIST AI RMF, Treasury AI |
| **Policy** | Policy-as-code engine translating regulatory requirements into runtime checks | NIST AI RMF, interagency guidance |
| **Governance** | Compliance evidence generation for regulatory examinations | All of the above |

## Quick Start

### Installation

```bash
pip install responsible-ai-toolkit
```

Or install from source:

```bash
git clone https://github.com/rafaroger1995/responsible-ai-toolkit.git
cd responsible-ai-toolkit
pip install -e ".[dev]"
```

### Fairness Evaluation in 5 Lines

```python
from responsible_ai_toolkit import FairnessMetrics

fm = FairnessMetrics(
    y_true=labels,           # Ground truth (0/1)
    y_pred=predictions,      # Model predictions (0/1)
    sensitive_attr=groups,   # Demographic groups
    y_prob=probabilities,    # Predicted probabilities
    threshold=0.80,          # Four-fifths rule
)
report = fm.full_report()
print(report.summary())
```

### Continuous Bias Monitoring

```python
from responsible_ai_toolkit import BiasDetector

detector = BiasDetector(
    threshold=0.80,
    window_size=1000,
    on_alert=lambda alert: notify_compliance_team(alert),
)

# Stream predictions as they happen
for prediction in production_stream:
    detector.observe(
        y_true=prediction.actual,
        y_pred=prediction.predicted,
        group=prediction.demographic_group,
    )

# Evaluate periodically
report = detector.evaluate()
```

### Model Drift Detection

```python
from responsible_ai_toolkit import DriftMonitor

monitor = DriftMonitor(reference=training_scores)
report = monitor.evaluate(production_scores)

if report.psi > 0.25:
    print(f"Significant drift detected: PSI = {report.psi:.4f}")
    print("Model revalidation required per SR 11-7 guidance")
```

### Tamper-Evident Audit Trail

```python
from responsible_ai_toolkit import AuditLogger

logger = AuditLogger(system_id="lending-model-v2")

# Every AI decision is logged with hash chaining
logger.log_prediction(
    model_id="credit-score-v2",
    input_data={"income": 75000, "credit_score": 720},
    output={"approved": True, "score": 0.87},
)

# Verify chain integrity at any time
assert logger.verify_chain()  # True if no tampering detected

# Export evidence for regulatory examination
evidence = logger.export_evidence_package()
```

### Human-in-the-Loop Orchestration

```python
from responsible_ai_toolkit import HITLOrchestrator

hitl = HITLOrchestrator(confidence_threshold=0.65)
hitl.add_reviewer("senior_analyst", roles=["lending"])

# Low-confidence decisions are automatically flagged
if hitl.should_review(confidence=0.52):
    case = hitl.submit_for_review(
        case_id="LOAN-2024-001",
        category="lending",
        priority="high",
        ai_decision={"approved": True, "score": 0.52},
        reason="Confidence below threshold",
    )
```

### Policy-as-Code Compliance

```python
from responsible_ai_toolkit import PolicyEngine

engine = PolicyEngine()

# Define compliance rules as executable policies
engine.add_policy(PolicyEngine.fairness_threshold_policy(
    "ECOA-001", "dp_ratio", threshold=0.80,
    frameworks=["ECOA", "FHA"],
))
engine.add_policy(PolicyEngine.drift_threshold_policy(
    "MRM-001", psi_threshold=0.25,
    frameworks=["SR-11-7"],
))

# Evaluate at runtime
result = engine.evaluate({"dp_ratio": 0.72, "psi": 0.18})
if result.critical_violations:
    print("Compliance violations detected — escalating to review")
```

## Examples

The `examples/` directory contains end-to-end demonstrations:

- **[Fair Lending Compliance](examples/lending_fairness.py)** — Complete workflow for a community bank monitoring a loan approval model
- **[Insurance Underwriting](examples/insurance_underwriting.py)** — Bias and drift monitoring for an insurance pricing model

Run any example:

```bash
python examples/lending_fairness.py
```

## Architecture

```
responsible_ai_toolkit/
├── fairness/          # Bias detection & fairness metrics
│   ├── metrics.py     # Demographic parity, equal opportunity, equalized odds, calibration
│   └── bias_detector.py  # Continuous monitoring with streaming & batch support
├── drift/             # Model stability monitoring
│   └── monitor.py     # PSI, KL divergence, Wasserstein distance
├── audit/             # Immutable audit logging
│   └── logger.py      # SHA-256 hash-chained append-only logs
├── hitl/              # Human-in-the-loop workflows
│   └── orchestrator.py  # Reviewer queues, escalation, SLA enforcement
├── policy/            # Regulatory compliance automation
│   └── engine.py      # Policy-as-code with framework alignment
└── governance/        # Compliance evidence generation
    └── report.py      # Regulator-ready evidence packages
```

## Regulatory Framework Alignment

This toolkit is designed to operationalize requirements from:

| Framework | Modules Used |
|-----------|-------------|
| **NIST AI Risk Management Framework** | Fairness, HITL, Policy, Governance |
| **OCC/Fed SR 11-7** (Model Risk Management) | Drift, Audit, Governance |
| **ECOA / Fair Housing Act** | Fairness, Policy |
| **Treasury AI in Financial Services Report** | All modules |
| **FDIC AI Compliance Plan** | Audit, Policy, Governance |
| **Interagency TPRM Guidance** | Audit, Policy |
| **Executive Order 14179** | Fairness, HITL, Policy |

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run tests with coverage
pytest tests/ -v --cov=responsible_ai_toolkit
```

## Roadmap

- [ ] Integration connectors for common financial systems (LOS, claims platforms)
- [ ] SaaS dashboard for non-technical compliance officers
- [ ] Pre-built policy packs for specific regulations (ECOA, SR 11-7, NIST AI RMF)
- [ ] Certification mechanisms for validated AI systems
- [ ] Multi-language support for accessibility
- [ ] Regional data trust capabilities for disaster-prone areas

## Contributing

Contributions are welcome. Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

Apache License 2.0 — see [LICENSE](LICENSE) for details.

## Author

**Yash Lundia** — [GitHub](https://github.com/rafaroger1995)

Staff Product Manager specializing in responsible AI systems for regulated industries. Stanford MS in Management Science & Engineering. Building governance infrastructure to democratize responsible AI adoption for institutions that need it most.
