"""
Responsible AI Toolkit
======================

An open-source governance framework for deploying AI responsibly in
regulated industries — financial services, insurance, healthcare, and beyond.

Provides experimental modules for:
  - **Fairness**: Continuous bias detection using demographic parity,
    equal opportunity, equalized odds, and calibration metrics.
  - **Drift**: Model stability monitoring via PSI, KL divergence,
    and Wasserstein distance.
  - **Audit**: In-memory event logging with SHA-256 hash chaining;
    external anchoring, persistence and access controls are not provided.
  - **HITL**: Human-in-the-loop orchestration with escalation policies,
    role-based reviewer queues, and SLA enforcement.
  - **Policy**: Configurable checks and versioned synthetic decision support.
  - **Governance**: Experimental summaries and review records.

Intended for technical evaluation in U.S. banking and insurance workflows,
initially prioritizing smaller institutions. No production-readiness,
compliance, adoption or institutional-validation claim is made.

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
