"""
Policy-as-code engine for automated regulatory compliance.

Translates regulatory requirements, governance frameworks, and
institutional policies into executable runtime checks that can be
applied to AI model inputs, outputs, and operational contexts.

Supports:
  - Declarative policy definitions (JSON/dict-based rules)
  - Composite policies with AND/OR/NOT logic
  - Contextual evaluation with dynamic data binding
  - Policy versioning and audit-ready evidence generation
  - Framework alignment tags (NIST AI RMF, SR 11-7, ECOA, etc.)

This approach aligns with the NIST AI RMF's Govern and Map functions,
enabling institutions to encode their compliance obligations as
testable, reproducible, and auditable controls.

Usage:
    >>> from responsible_ai_toolkit.policy import PolicyEngine
    >>> engine = PolicyEngine()
    >>> engine.add_policy(Policy(
    ...     policy_id="fair-lending-001",
    ...     name="Adverse Action Threshold",
    ...     description="Reject AI decisions with confidence below 0.60",
    ...     rule=lambda ctx: ctx["confidence"] >= 0.60,
    ...     frameworks=["ECOA", "NIST-AI-RMF"],
    ...     severity="critical",
    ... ))
    >>> result = engine.evaluate({"confidence": 0.45, "model_id": "credit-v2"})
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class Policy:
    """A single policy rule for automated compliance checking.

    Parameters
    ----------
    policy_id : str
        Unique identifier for the policy.
    name : str
        Human-readable policy name.
    description : str
        Explanation of what the policy checks and why.
    rule : callable
        A function that takes a context dict and returns True (pass)
        or False (violation).
    severity : str
        "info", "warning", or "critical".
    frameworks : list of str
        Regulatory frameworks this policy supports
        (e.g., ["NIST-AI-RMF", "ECOA", "SR-11-7"]).
    category : str
        Policy category (e.g., "fairness", "transparency", "security").
    version : str
        Policy version string.
    enabled : bool
        Whether the policy is currently active.
    """

    policy_id: str
    name: str
    description: str
    rule: Callable[[Dict[str, Any]], bool]
    severity: str = "warning"
    frameworks: List[str] = field(default_factory=list)
    category: str = "general"
    version: str = "1.0"
    enabled: bool = True


@dataclass
class PolicyResult:
    """Result of evaluating a single policy."""

    policy_id: str
    policy_name: str
    passed: bool
    severity: str
    category: str
    frameworks: List[str]
    description: str
    timestamp: float = field(default_factory=time.time)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "policy_name": self.policy_name,
            "passed": self.passed,
            "severity": self.severity,
            "category": self.category,
            "frameworks": self.frameworks,
            "description": self.description,
            "timestamp": self.timestamp,
            "error": self.error,
        }


@dataclass
class EvaluationReport:
    """Complete policy evaluation report."""

    results: List[PolicyResult] = field(default_factory=list)
    context_snapshot: Dict[str, Any] = field(default_factory=dict)
    evaluated_at: float = field(default_factory=time.time)

    @property
    def all_passed(self) -> bool:
        return bool(self.results) and all(r.passed for r in self.results)

    @property
    def critical_violations(self) -> List[PolicyResult]:
        return [r for r in self.results if not r.passed and r.severity == "critical"]

    @property
    def warnings(self) -> List[PolicyResult]:
        return [r for r in self.results if not r.passed and r.severity == "warning"]

    @property
    def violations(self) -> List[PolicyResult]:
        return [r for r in self.results if not r.passed]

    @property
    def pass_rate(self) -> float:
        if not self.results:
            return 1.0
        return sum(1 for r in self.results if r.passed) / len(self.results)

    def by_framework(self, framework: str) -> List[PolicyResult]:
        """Filter results by regulatory framework."""
        return [r for r in self.results if framework in r.frameworks]

    def summary(self) -> str:
        lines = [
            "=" * 55,
            "POLICY EVALUATION REPORT",
            "=" * 55,
            f"  Total Policies Evaluated: {len(self.results)}",
            f"  Passed:                   {sum(1 for r in self.results if r.passed)}",
            f"  Violations:               {len(self.violations)}",
            f"    Critical:               {len(self.critical_violations)}",
            f"    Warnings:               {len(self.warnings)}",
            f"  Pass Rate:                {self.pass_rate:.1%}",
        ]
        if self.violations:
            lines.append("\n--- Violations ---")
            for v in self.violations:
                icon = "!!" if v.severity == "critical" else " !"
                lines.append(f"  {icon} [{v.policy_id}] {v.policy_name}")
                lines.append(f"       {v.description}")
        lines.append("=" * 55)
        return "\n".join(lines)


class PolicyEngine:
    """Policy-as-code evaluation engine.

    Manages a registry of policies and evaluates them against
    runtime contexts to produce compliance evidence.
    """

    def __init__(self) -> None:
        self._policies: Dict[str, Policy] = {}
        self._evaluation_history: List[EvaluationReport] = []

    # ------------------------------------------------------------------
    # Policy management
    # ------------------------------------------------------------------

    def add_policy(self, policy: Policy) -> None:
        """Register a policy in the engine."""
        self._policies[policy.policy_id] = policy

    def remove_policy(self, policy_id: str) -> None:
        """Remove a policy from the engine."""
        self._policies.pop(policy_id, None)

    def get_policy(self, policy_id: str) -> Optional[Policy]:
        return self._policies.get(policy_id)

    def list_policies(
        self,
        category: Optional[str] = None,
        framework: Optional[str] = None,
        enabled_only: bool = True,
    ) -> List[Policy]:
        """List registered policies with optional filtering."""
        policies = list(self._policies.values())
        if enabled_only:
            policies = [p for p in policies if p.enabled]
        if category:
            policies = [p for p in policies if p.category == category]
        if framework:
            policies = [p for p in policies if framework in p.frameworks]
        return policies

    # ------------------------------------------------------------------
    # Pre-built policy factories
    # ------------------------------------------------------------------

    @staticmethod
    def fairness_threshold_policy(
        policy_id: str,
        metric_key: str,
        threshold: float = 0.80,
        frameworks: Optional[List[str]] = None,
    ) -> Policy:
        """Create a policy that checks a fairness metric against a threshold."""
        return Policy(
            policy_id=policy_id,
            name=f"Fairness Threshold: {metric_key} >= {threshold}",
            description=(
                f"Requires {metric_key} ratio to be at least {threshold} "
                f"(four-fifths rule). Violations indicate potential adverse "
                f"impact requiring review."
            ),
            rule=lambda ctx, k=metric_key, t=threshold: ctx.get(k, 0) >= t,
            severity="critical",
            frameworks=frameworks or ["ECOA", "NIST-AI-RMF"],
            category="fairness",
        )

    @staticmethod
    def confidence_threshold_policy(
        policy_id: str,
        threshold: float = 0.60,
        frameworks: Optional[List[str]] = None,
    ) -> Policy:
        """Create a policy requiring minimum model confidence for auto-decisions."""
        return Policy(
            policy_id=policy_id,
            name=f"Minimum Confidence: >= {threshold}",
            description=(
                f"AI decisions with confidence below {threshold} must be "
                f"routed to human review. Ensures adequate model certainty "
                f"before automated processing."
            ),
            rule=lambda ctx, t=threshold: ctx.get("confidence", 0) >= t,
            severity="critical",
            frameworks=frameworks or ["SR-11-7", "NIST-AI-RMF"],
            category="transparency",
        )

    @staticmethod
    def drift_threshold_policy(
        policy_id: str,
        psi_threshold: float = 0.25,
        frameworks: Optional[List[str]] = None,
    ) -> Policy:
        """Create a policy flagging significant model drift."""
        return Policy(
            policy_id=policy_id,
            name=f"Model Drift: PSI < {psi_threshold}",
            description=(
                f"Population Stability Index must remain below {psi_threshold}. "
                f"Exceeding this threshold indicates significant distribution "
                f"shift requiring model revalidation."
            ),
            rule=lambda ctx, t=psi_threshold: ctx["psi"] < t,
            severity="critical",
            frameworks=frameworks or ["SR-11-7", "OCC-MRM"],
            category="stability",
        )

    @staticmethod
    def data_completeness_policy(
        policy_id: str,
        required_fields: List[str],
        frameworks: Optional[List[str]] = None,
    ) -> Policy:
        """Create a policy ensuring required data fields are present."""
        return Policy(
            policy_id=policy_id,
            name="Data Completeness Check",
            description=(
                f"All required fields must be present: {', '.join(required_fields)}. "
                f"Missing data may compromise model reliability."
            ),
            rule=lambda ctx, fields=required_fields: all(
                ctx.get(f) is not None for f in fields
            ),
            severity="warning",
            frameworks=frameworks or ["NIST-AI-RMF"],
            category="data_quality",
        )

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        context: Dict[str, Any],
        categories: Optional[List[str]] = None,
        frameworks: Optional[List[str]] = None,
    ) -> EvaluationReport:
        """Evaluate all applicable policies against the given context.

        Parameters
        ----------
        context : dict
            Runtime context containing the data to evaluate
            (model outputs, scores, metadata, etc.).
        categories : list of str, optional
            Only evaluate policies in these categories.
        frameworks : list of str, optional
            Only evaluate policies tagged with these frameworks.

        Returns
        -------
        EvaluationReport
        """
        policies = self.list_policies(enabled_only=True)

        if categories:
            policies = [p for p in policies if p.category in categories]
        if frameworks:
            policies = [
                p for p in policies
                if any(f in p.frameworks for f in frameworks)
            ]

        results: List[PolicyResult] = []
        for policy in policies:
            try:
                passed = bool(policy.rule(context))
                results.append(PolicyResult(
                    policy_id=policy.policy_id,
                    policy_name=policy.name,
                    passed=passed,
                    severity=policy.severity,
                    category=policy.category,
                    frameworks=policy.frameworks,
                    description=policy.description,
                ))
            except Exception as e:
                results.append(PolicyResult(
                    policy_id=policy.policy_id,
                    policy_name=policy.name,
                    passed=False,
                    severity=policy.severity,
                    category=policy.category,
                    frameworks=policy.frameworks,
                    description=policy.description,
                    error=str(e),
                ))

        report = EvaluationReport(
            results=results,
            context_snapshot={k: str(v) for k, v in context.items()},
        )
        self._evaluation_history.append(report)
        return report

    @property
    def evaluation_history(self) -> List[EvaluationReport]:
        return list(self._evaluation_history)
