"""
Governance reporting and compliance evidence generation.

Aggregates outputs from fairness evaluations, drift monitoring,
audit logs, HITL workflows, and policy checks into structured
compliance reports suitable for regulatory examinations.

Designed to generate the evidence packages required by:
  - FDIC/OCC model risk management examinations
  - Fair lending regulatory reviews (ECOA, FHA)
  - SOC 2 Type II audit evidence requirements
  - NIST AI RMF compliance documentation

Usage:
    >>> from responsible_ai_toolkit.governance import GovernanceReporter
    >>> reporter = GovernanceReporter(system_id="lending-model-v2")
    >>> reporter.add_fairness_report(fairness_report)
    >>> reporter.add_drift_report(drift_report)
    >>> evidence = reporter.generate_evidence_package()
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from responsible_ai_toolkit.fairness.metrics import FairnessReport
from responsible_ai_toolkit.drift.monitor import DriftReport
from responsible_ai_toolkit.policy.engine import EvaluationReport


@dataclass
class GovernanceSnapshot:
    """A point-in-time snapshot of governance status."""

    timestamp: float
    timestamp_iso: str
    fairness_status: str   # "pass", "warning", "violation"
    drift_status: str      # "stable", "moderate", "significant"
    policy_status: str     # "compliant", "violations_detected"
    hitl_sla_rate: Optional[float] = None
    hitl_override_rate: Optional[float] = None
    details: Dict[str, Any] = field(default_factory=dict)


class GovernanceReporter:
    """Aggregate governance data into compliance evidence packages.

    Parameters
    ----------
    system_id : str
        Identifier for the AI system being governed.
    organization : str, optional
        Organization name for report headers.
    """

    def __init__(
        self,
        system_id: str,
        organization: Optional[str] = None,
    ) -> None:
        self.system_id = system_id
        self.organization = organization or "Organization"

        self._fairness_reports: List[FairnessReport] = []
        self._drift_reports: List[DriftReport] = []
        self._policy_reports: List[EvaluationReport] = []
        self._snapshots: List[GovernanceSnapshot] = []
        self._notes: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Data ingestion
    # ------------------------------------------------------------------

    def add_fairness_report(self, report: FairnessReport) -> None:
        """Add a fairness evaluation report."""
        self._fairness_reports.append(report)

    def add_drift_report(self, report: DriftReport) -> None:
        """Add a drift monitoring report."""
        self._drift_reports.append(report)

    def add_policy_report(self, report: EvaluationReport) -> None:
        """Add a policy evaluation report."""
        self._policy_reports.append(report)

    def add_note(self, author: str, content: str, category: str = "general") -> None:
        """Add an analyst note or observation to the governance record."""
        self._notes.append({
            "author": author,
            "content": content,
            "category": category,
            "timestamp": time.time(),
            "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })

    # ------------------------------------------------------------------
    # Snapshot creation
    # ------------------------------------------------------------------

    def create_snapshot(
        self,
        hitl_sla_rate: Optional[float] = None,
        hitl_override_rate: Optional[float] = None,
    ) -> GovernanceSnapshot:
        """Create a point-in-time governance status snapshot."""
        now = time.time()

        # Determine fairness status
        if self._fairness_reports:
            latest_fair = self._fairness_reports[-1]
            if latest_fair.overall_fair:
                fairness_status = "pass"
            elif len(latest_fair.violations) <= 2:
                fairness_status = "warning"
            else:
                fairness_status = "violation"
        else:
            fairness_status = "no_data"

        # Determine drift status
        if self._drift_reports:
            latest_drift = self._drift_reports[-1]
            drift_status = latest_drift.psi_interpretation.split(" —")[0].lower().replace(" ", "_")
        else:
            drift_status = "no_data"

        # Determine policy status
        if not self._policy_reports or not self._policy_reports[-1].results:
            policy_status = "not_evaluated"
        else:
            latest_policy = self._policy_reports[-1]
            policy_status = (
                "evaluation_error"
                if any(result.error is not None for result in latest_policy.results)
                else "checks_passed"
                if latest_policy.all_passed
                else "checks_failed"
            )

        snapshot = GovernanceSnapshot(
            timestamp=now,
            timestamp_iso=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)),
            fairness_status=fairness_status,
            drift_status=drift_status,
            policy_status=policy_status,
            hitl_sla_rate=hitl_sla_rate,
            hitl_override_rate=hitl_override_rate,
            details={
                "fairness_violations": (
                    len(self._fairness_reports[-1].violations)
                    if self._fairness_reports else 0
                ),
                "drift_psi": (
                    self._drift_reports[-1].psi
                    if self._drift_reports else None
                ),
                "policy_pass_rate": (
                    self._policy_reports[-1].pass_rate
                    if self._policy_reports else None
                ),
            },
        )
        self._snapshots.append(snapshot)
        return snapshot

    # ------------------------------------------------------------------
    # Evidence package generation
    # ------------------------------------------------------------------

    def generate_evidence_package(self) -> Dict[str, Any]:
        """Generate a complete compliance evidence package.

        This produces a structured document suitable for regulatory
        examination, containing:
          - System identification and metadata
          - Fairness evaluation history and current status
          - Drift monitoring history and current status
          - Policy compliance history and current status
          - Governance snapshots over time
          - Analyst notes and observations
        """
        now = time.time()

        package = {
            "metadata": {
                "system_id": self.system_id,
                "organization": self.organization,
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)),
                "toolkit_version": "0.1.0",
                "report_type": "Governance Evidence Package",
            },
            "executive_summary": self._generate_executive_summary(),
            "fairness": {
                "total_evaluations": len(self._fairness_reports),
                "current_status": (
                    "pass" if self._fairness_reports and self._fairness_reports[-1].overall_fair
                    else "violations_detected" if self._fairness_reports
                    else "no_evaluations"
                ),
                "evaluations": [
                    {
                        "demographic_parity": r.demographic_parity,
                        "equal_opportunity": r.equal_opportunity,
                        "equalized_odds": r.equalized_odds,
                        "calibration": r.calibration,
                        "violations": r.violations,
                        "overall_fair": r.overall_fair,
                    }
                    for r in self._fairness_reports
                ],
            },
            "drift": {
                "total_evaluations": len(self._drift_reports),
                "current_status": (
                    self._drift_reports[-1].psi_interpretation
                    if self._drift_reports else "no_evaluations"
                ),
                "evaluations": [
                    {
                        "psi": r.psi,
                        "kl_divergence": r.kl_divergence,
                        "wasserstein_distance": r.wasserstein_distance,
                        "interpretation": r.psi_interpretation,
                    }
                    for r in self._drift_reports
                ],
            },
            "policy_compliance": {
                "total_evaluations": len(self._policy_reports),
                "current_status": self.create_snapshot().policy_status,
                "evaluations": [
                    {
                        "pass_rate": r.pass_rate,
                        "total_policies": len(r.results),
                        "violations": len(r.violations),
                        "critical_violations": len(r.critical_violations),
                    }
                    for r in self._policy_reports
                ],
            },
            "governance_timeline": [
                {
                    "timestamp_iso": s.timestamp_iso,
                    "fairness": s.fairness_status,
                    "drift": s.drift_status,
                    "policy": s.policy_status,
                    "hitl_sla_rate": s.hitl_sla_rate,
                    "hitl_override_rate": s.hitl_override_rate,
                }
                for s in self._snapshots
            ],
            "notes": self._notes,
        }

        return package

    def _generate_executive_summary(self) -> Dict[str, Any]:
        """Generate an executive summary for the evidence package."""
        total_violations = sum(
            len(r.violations) for r in self._fairness_reports
        )
        drift_alerts = sum(
            1 for r in self._drift_reports
            if "significant" in r.psi_interpretation.lower()
        )
        policy_failures = sum(
            len(r.violations) for r in self._policy_reports
        )

        return {
            "total_fairness_evaluations": len(self._fairness_reports),
            "total_fairness_violations": total_violations,
            "total_drift_evaluations": len(self._drift_reports),
            "drift_alerts": drift_alerts,
            "total_policy_evaluations": len(self._policy_reports),
            "policy_failures": policy_failures,
            "governance_snapshots": len(self._snapshots),
            "analyst_notes": len(self._notes),
        }

    def export_json(self, indent: int = 2) -> str:
        """Export the evidence package as formatted JSON."""
        return json.dumps(
            self.generate_evidence_package(),
            indent=indent,
            default=str,
        )
