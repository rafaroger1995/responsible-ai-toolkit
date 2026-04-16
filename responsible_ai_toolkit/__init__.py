"""
Responsible AI Toolkit
======================

An open-source governance framework for deploying AI responsibly in
regulated industries — financial services, insurance, healthcare, and beyond.

Provides production-ready modules for:
  - **Fairness**: Continuous bias detection using demographic parity,
    equal opportunity, equalized odds, and calibration metrics.
  - **Drift**: Model stability monitoring via PSI, KL divergence,
    and Wasserstein distance.
  - **Audit**: Tamper-evident, append-only audit logging with SHA-256
    hash chaining and timestamp anchoring.
  - **HITL**: Human-in-the-loop orchestration with escalation policies,
    role-based reviewer queues, and SLA enforcement.
  - **Policy**: Policy-as-code engine for translating regulatory
    requirements into automated runtime checks.
  - **Governance**: Compliance evidence generation and reporting for
    regulatory examinations.

Designed for community banks, regional insurers, and other regulated
institutions that need enterprise-grade AI governance without
enterprise-scale compliance teams.

License: Apache 2.0
"""

__version__ = "0.1.0"
__author__ = "Yash Lundia"

from responsible_ai_toolkit.fairness.metrics import FairnessMetrics
from responsible_ai_toolkit.fairness.bias_detector import BiasDetector
from responsible_ai_toolkit.drift.monitor import DriftMonitor
from responsible_ai_toolkit.audit.logger import AuditLogger
from responsible_ai_toolkit.hitl.orchestrator import HITLOrchestrator
from responsible_ai_toolkit.policy.engine import PolicyEngine
from responsible_ai_toolkit.governance.report import GovernanceReporter

__all__ = [
    "FairnessMetrics",
    "BiasDetector",
    "DriftMonitor",
    "AuditLogger",
    "HITLOrchestrator",
    "PolicyEngine",
    "GovernanceReporter",
]
