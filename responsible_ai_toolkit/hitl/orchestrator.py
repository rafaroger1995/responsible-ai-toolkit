"""
Human-in-the-loop orchestration for regulated AI systems.

Provides structured workflows for human oversight of AI decisions,
including role-based reviewer queues, configurable escalation policies,
SLA enforcement, disposition logging, and active-learning sampling
for edge cases.

Designed to meet the human oversight requirements emphasized by the
NIST AI Risk Management Framework, FDIC model risk guidance, and
Treasury AI recommendations for financial institutions.

Key capabilities:
  - Route flagged AI decisions to appropriate reviewer queues
  - Enforce escalation policies based on confidence, risk, or bias alerts
  - Track review SLAs and generate compliance evidence
  - Support active-learning sampling to surface edge cases
  - Log all review decisions in the audit trail

Usage:
    >>> from responsible_ai_toolkit.hitl import HITLOrchestrator
    >>> hitl = HITLOrchestrator()
    >>> hitl.add_reviewer("analyst_1", roles=["lending", "fraud"])
    >>> case = hitl.submit_for_review(
    ...     case_id="LOAN-2024-001",
    ...     category="lending",
    ...     priority="high",
    ...     ai_decision={"approved": True, "score": 0.62},
    ...     reason="Low confidence score below threshold",
    ... )
"""

from __future__ import annotations

import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


class CasePriority(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class CaseStatus(str, Enum):
    PENDING = "pending"
    ASSIGNED = "assigned"
    IN_REVIEW = "in_review"
    COMPLETED = "completed"
    ESCALATED = "escalated"
    EXPIRED = "expired"


class ReviewDecision(str, Enum):
    APPROVE = "approve"
    REJECT = "reject"
    OVERRIDE = "override"
    ESCALATE = "escalate"
    REQUEST_INFO = "request_info"


@dataclass
class Reviewer:
    """A human reviewer in the HITL system."""

    reviewer_id: str
    roles: List[str] = field(default_factory=list)
    active: bool = True
    current_load: int = 0
    max_load: int = 50
    total_reviews: int = 0


@dataclass
class ReviewCase:
    """A case submitted for human review."""

    case_id: str
    internal_id: str
    category: str
    priority: CasePriority
    status: CaseStatus
    ai_decision: Dict[str, Any]
    reason: str
    created_at: float
    sla_deadline: float
    assigned_to: Optional[str] = None
    assigned_at: Optional[float] = None
    completed_at: Optional[float] = None
    decision: Optional[ReviewDecision] = None
    reviewer_reasoning: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_overdue(self) -> bool:
        if self.status in (CaseStatus.COMPLETED, CaseStatus.ESCALATED):
            return False
        return time.time() > self.sla_deadline

    @property
    def time_to_sla_hours(self) -> float:
        return (self.sla_deadline - time.time()) / 3600


@dataclass
class EscalationPolicy:
    """Defines when and how cases should be escalated."""

    name: str
    condition: Callable[[ReviewCase], bool]
    escalate_to_role: str
    new_priority: CasePriority = CasePriority.HIGH
    reason_template: str = "Escalated by policy: {name}"


class HITLOrchestrator:
    """Orchestrate human-in-the-loop review workflows.

    Parameters
    ----------
    default_sla_hours : dict, optional
        SLA deadlines by priority level (in hours).
        Defaults: low=72, normal=24, high=8, critical=2.
    confidence_threshold : float, default 0.70
        AI decisions with confidence below this threshold are
        automatically routed for human review.
    active_learning_rate : float, default 0.05
        Fraction of above-threshold decisions randomly sampled for
        human review (active learning for model improvement).
    """

    DEFAULT_SLAS = {
        CasePriority.LOW: 72.0,
        CasePriority.NORMAL: 24.0,
        CasePriority.HIGH: 8.0,
        CasePriority.CRITICAL: 2.0,
    }

    def __init__(
        self,
        default_sla_hours: Optional[Dict[CasePriority, float]] = None,
        confidence_threshold: float = 0.70,
        active_learning_rate: float = 0.05,
    ) -> None:
        self.sla_hours = default_sla_hours or self.DEFAULT_SLAS
        self.confidence_threshold = confidence_threshold
        self.active_learning_rate = active_learning_rate

        self._reviewers: Dict[str, Reviewer] = {}
        self._cases: Dict[str, ReviewCase] = {}
        self._escalation_policies: List[EscalationPolicy] = []
        self._decision_log: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Reviewer management
    # ------------------------------------------------------------------

    def add_reviewer(
        self,
        reviewer_id: str,
        roles: Optional[List[str]] = None,
        max_load: int = 50,
    ) -> Reviewer:
        """Register a human reviewer with their roles and capacity."""
        reviewer = Reviewer(
            reviewer_id=reviewer_id,
            roles=roles or [],
            max_load=max_load,
        )
        self._reviewers[reviewer_id] = reviewer
        return reviewer

    def get_reviewer(self, reviewer_id: str) -> Optional[Reviewer]:
        return self._reviewers.get(reviewer_id)

    # ------------------------------------------------------------------
    # Escalation policies
    # ------------------------------------------------------------------

    def add_escalation_policy(self, policy: EscalationPolicy) -> None:
        """Register an escalation policy."""
        self._escalation_policies.append(policy)

    # ------------------------------------------------------------------
    # Case submission and routing
    # ------------------------------------------------------------------

    def should_review(self, confidence: float) -> bool:
        """Determine whether an AI decision requires human review.

        Returns True if confidence is below threshold or if the
        decision is selected for active-learning sampling.
        """
        if confidence < self.confidence_threshold:
            return True

        # Active learning: random sampling of high-confidence decisions
        import random
        return random.random() < self.active_learning_rate

    def submit_for_review(
        self,
        case_id: str,
        category: str,
        ai_decision: Dict[str, Any],
        reason: str,
        priority: str = "normal",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ReviewCase:
        """Submit a case for human review.

        The case is placed in the appropriate queue and optionally
        auto-assigned to an available reviewer.
        """
        pri = CasePriority(priority)
        sla_seconds = self.sla_hours.get(pri, 24.0) * 3600
        now = time.time()

        case = ReviewCase(
            case_id=case_id,
            internal_id=str(uuid.uuid4()),
            category=category,
            priority=pri,
            status=CaseStatus.PENDING,
            ai_decision=ai_decision,
            reason=reason,
            created_at=now,
            sla_deadline=now + sla_seconds,
            metadata=metadata or {},
        )

        self._cases[case.internal_id] = case

        # Auto-assign if a qualified reviewer is available
        self._try_assign(case)

        return case

    def _try_assign(self, case: ReviewCase) -> bool:
        """Try to assign a case to the best available reviewer."""
        candidates = [
            r for r in self._reviewers.values()
            if r.active
            and r.current_load < r.max_load
            and (not r.roles or case.category in r.roles)
        ]

        if not candidates:
            return False

        # Assign to reviewer with lowest current load
        best = min(candidates, key=lambda r: r.current_load)
        case.assigned_to = best.reviewer_id
        case.assigned_at = time.time()
        case.status = CaseStatus.ASSIGNED
        best.current_load += 1
        return True

    # ------------------------------------------------------------------
    # Review decisions
    # ------------------------------------------------------------------

    def record_decision(
        self,
        internal_id: str,
        reviewer_id: str,
        decision: str,
        reasoning: str,
    ) -> ReviewCase:
        """Record a reviewer's decision on a case."""
        if internal_id not in self._cases:
            raise ValueError(f"Case '{internal_id}' not found.")

        case = self._cases[internal_id]
        dec = ReviewDecision(decision)

        case.decision = dec
        case.reviewer_reasoning = reasoning
        case.completed_at = time.time()
        case.status = CaseStatus.COMPLETED

        # Update reviewer stats
        if case.assigned_to and case.assigned_to in self._reviewers:
            reviewer = self._reviewers[case.assigned_to]
            reviewer.current_load = max(0, reviewer.current_load - 1)
            reviewer.total_reviews += 1

        # Log the decision
        self._decision_log.append({
            "case_id": case.case_id,
            "internal_id": internal_id,
            "reviewer_id": reviewer_id,
            "decision": dec.value,
            "reasoning": reasoning,
            "ai_decision": case.ai_decision,
            "is_override": dec == ReviewDecision.OVERRIDE,
            "review_time_seconds": (
                (case.completed_at - case.assigned_at)
                if case.assigned_at else None
            ),
            "within_sla": case.completed_at <= case.sla_deadline,
            "timestamp": case.completed_at,
        })

        # Handle escalation decision
        if dec == ReviewDecision.ESCALATE:
            case.status = CaseStatus.ESCALATED
            self._apply_escalation_policies(case)

        return case

    def _apply_escalation_policies(self, case: ReviewCase) -> None:
        """Apply escalation policies to a case."""
        for policy in self._escalation_policies:
            if policy.condition(case):
                case.priority = policy.new_priority
                case.status = CaseStatus.PENDING
                case.assigned_to = None
                self._try_assign(case)
                break

    # ------------------------------------------------------------------
    # Queue inspection
    # ------------------------------------------------------------------

    def get_queue(
        self, category: Optional[str] = None, status: Optional[str] = None
    ) -> List[ReviewCase]:
        """Get cases in the review queue, optionally filtered."""
        cases = list(self._cases.values())
        if category:
            cases = [c for c in cases if c.category == category]
        if status:
            st = CaseStatus(status)
            cases = [c for c in cases if c.status == st]
        return sorted(cases, key=lambda c: (
            list(CasePriority).index(c.priority),
            c.created_at,
        ))

    def get_overdue_cases(self) -> List[ReviewCase]:
        """Get all cases that have exceeded their SLA deadline."""
        return [c for c in self._cases.values() if c.is_overdue]

    @property
    def decision_log(self) -> List[Dict[str, Any]]:
        return list(self._decision_log)

    def sla_compliance_rate(self) -> float:
        """Compute the fraction of completed reviews within SLA."""
        completed = [d for d in self._decision_log if d.get("within_sla") is not None]
        if not completed:
            return 1.0
        return sum(1 for d in completed if d["within_sla"]) / len(completed)

    def override_rate(self) -> float:
        """Compute the fraction of reviews that overrode the AI decision."""
        if not self._decision_log:
            return 0.0
        return sum(1 for d in self._decision_log if d["is_override"]) / len(self._decision_log)

    def get_stats(self) -> Dict[str, Any]:
        """Return summary statistics for the HITL system."""
        return {
            "total_cases": len(self._cases),
            "pending": len(self.get_queue(status="pending")),
            "assigned": len(self.get_queue(status="assigned")),
            "completed": len(self.get_queue(status="completed")),
            "escalated": len(self.get_queue(status="escalated")),
            "overdue": len(self.get_overdue_cases()),
            "sla_compliance_rate": self.sla_compliance_rate(),
            "override_rate": self.override_rate(),
            "total_reviewers": len(self._reviewers),
            "active_reviewers": sum(1 for r in self._reviewers.values() if r.active),
        }
