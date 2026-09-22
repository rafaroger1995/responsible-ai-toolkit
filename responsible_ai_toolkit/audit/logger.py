"""
Tamper-evident, append-only audit logging for AI governance.

Provides an immutable record of every AI decision — model inputs,
outputs, policy checks, human interventions, and configuration
changes — using SHA-256 hash chaining and timestamp anchoring.

Designed to satisfy regulatory audit requirements in financial services:
  - FDIC/OCC model risk management (SR 11-7)
  - Interagency third-party risk management guidance
  - Fair lending examination evidence requirements
  - SOC 2 Type II audit trail controls

Every log entry is chained to the previous entry via a cryptographic
hash, making it computationally infeasible to tamper with historical
records without detection.

Usage:
    >>> from responsible_ai_toolkit.audit import AuditLogger
    >>> logger = AuditLogger(system_id="lending-model-v2")
    >>> logger.log_prediction(
    ...     model_id="credit-score-v2",
    ...     input_data={"income": 75000, "credit_score": 720},
    ...     output={"approved": True, "score": 0.87},
    ...     metadata={"applicant_id": "A-12345"},
    ... )
    >>> assert logger.verify_chain()
"""

from __future__ import annotations

import copy
import hashlib
import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class EventType(str, Enum):
    """Types of auditable events."""

    PREDICTION = "prediction"
    HUMAN_REVIEW = "human_review"
    POLICY_CHECK = "policy_check"
    MODEL_UPDATE = "model_update"
    CONFIG_CHANGE = "config_change"
    FAIRNESS_EVALUATION = "fairness_evaluation"
    DRIFT_ALERT = "drift_alert"
    ESCALATION = "escalation"
    OVERRIDE = "override"
    SYSTEM_EVENT = "system_event"


@dataclass
class AuditEntry:
    """A single immutable audit log entry with hash chain linkage."""

    entry_id: str
    sequence_number: int
    timestamp: float
    timestamp_iso: str
    system_id: str
    event_type: str
    payload: Dict[str, Any]
    previous_hash: str
    entry_hash: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=str, sort_keys=True)


class AuditLogger:
    """Append-only audit logger with SHA-256 hash chaining.

    Parameters
    ----------
    system_id : str
        Identifier for the AI system being audited (e.g., "lending-model-v2").
    hash_algorithm : str, default "sha256"
        Hash algorithm for chain integrity.
    """

    GENESIS_HASH = "0" * 64  # Genesis block previous hash

    def __init__(self, system_id: str, hash_algorithm: str = "sha256") -> None:
        self.system_id = system_id
        self.hash_algorithm = hash_algorithm
        self._entries: List[AuditEntry] = []
        self._sequence: int = 0

    # ------------------------------------------------------------------
    # Core hashing
    # ------------------------------------------------------------------

    def _compute_hash(self, data: str) -> str:
        """Compute SHA-256 hash of a string."""
        return hashlib.sha256(data.encode("utf-8")).hexdigest()

    def _get_previous_hash(self) -> str:
        """Get the hash of the most recent entry, or genesis hash."""
        if not self._entries:
            return self.GENESIS_HASH
        return self._entries[-1].entry_hash

    def _create_hashable_content(
        self,
        entry_id: str,
        sequence_number: int,
        timestamp: float,
        timestamp_iso: str,
        system_id: str,
        event_type: str,
        payload: Dict[str, Any],
        previous_hash: str,
    ) -> str:
        """Create a deterministic string for hashing.

        Every stored field of an entry except ``entry_hash`` is included,
        so changing any of them is detected by ``verify_chain``.
        """
        content = {
            "entry_id": entry_id,
            "sequence_number": sequence_number,
            "timestamp": timestamp,
            "timestamp_iso": timestamp_iso,
            "system_id": system_id,
            "event_type": event_type,
            "payload": payload,
            "previous_hash": previous_hash,
        }
        return json.dumps(content, sort_keys=True, default=str)

    # ------------------------------------------------------------------
    # Logging methods
    # ------------------------------------------------------------------

    def _append(self, event_type: EventType, payload: Dict[str, Any]) -> AuditEntry:
        """Create and append a new audit entry."""
        entry_id = str(uuid.uuid4())
        self._sequence += 1
        now = time.time()
        timestamp_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now))
        previous_hash = self._get_previous_hash()

        # Store a private copy so later changes to the caller's data
        # cannot alter the logged record.
        payload = copy.deepcopy(payload)

        hashable = self._create_hashable_content(
            entry_id=entry_id,
            sequence_number=self._sequence,
            timestamp=now,
            timestamp_iso=timestamp_iso,
            system_id=self.system_id,
            event_type=event_type.value,
            payload=payload,
            previous_hash=previous_hash,
        )
        entry_hash = self._compute_hash(hashable)

        entry = AuditEntry(
            entry_id=entry_id,
            sequence_number=self._sequence,
            timestamp=now,
            timestamp_iso=timestamp_iso,
            system_id=self.system_id,
            event_type=event_type.value,
            payload=payload,
            previous_hash=previous_hash,
            entry_hash=entry_hash,
        )
        self._entries.append(entry)
        return entry

    def log_prediction(
        self,
        model_id: str,
        input_data: Dict[str, Any],
        output: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AuditEntry:
        """Log an AI model prediction with full input/output traceability."""
        return self._append(
            EventType.PREDICTION,
            {
                "model_id": model_id,
                "input": input_data,
                "output": output,
                "metadata": metadata or {},
            },
        )

    def log_human_review(
        self,
        reviewer_id: str,
        decision: str,
        reasoning: str,
        original_prediction: Dict[str, Any],
        override: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AuditEntry:
        """Log a human review decision in the HITL workflow."""
        return self._append(
            EventType.HUMAN_REVIEW,
            {
                "reviewer_id": reviewer_id,
                "decision": decision,
                "reasoning": reasoning,
                "original_prediction": original_prediction,
                "override": override,
                "metadata": metadata or {},
            },
        )

    def log_policy_check(
        self,
        policy_id: str,
        result: str,
        details: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AuditEntry:
        """Log a policy-as-code evaluation result."""
        return self._append(
            EventType.POLICY_CHECK,
            {
                "policy_id": policy_id,
                "result": result,
                "details": details,
                "metadata": metadata or {},
            },
        )

    def log_fairness_evaluation(
        self,
        evaluation_id: str,
        metrics: Dict[str, Any],
        violations: List[str],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AuditEntry:
        """Log a fairness evaluation result."""
        return self._append(
            EventType.FAIRNESS_EVALUATION,
            {
                "evaluation_id": evaluation_id,
                "metrics": metrics,
                "violations": violations,
                "overall_fair": len(violations) == 0,
                "metadata": metadata or {},
            },
        )

    def log_drift_alert(
        self,
        metric_name: str,
        value: float,
        threshold: float,
        interpretation: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AuditEntry:
        """Log a model drift detection alert."""
        return self._append(
            EventType.DRIFT_ALERT,
            {
                "metric_name": metric_name,
                "value": value,
                "threshold": threshold,
                "interpretation": interpretation,
                "metadata": metadata or {},
            },
        )

    def log_escalation(
        self,
        reason: str,
        source_entry_id: str,
        escalated_to: str,
        priority: str = "normal",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AuditEntry:
        """Log an escalation event in the HITL workflow."""
        return self._append(
            EventType.ESCALATION,
            {
                "reason": reason,
                "source_entry_id": source_entry_id,
                "escalated_to": escalated_to,
                "priority": priority,
                "metadata": metadata or {},
            },
        )

    # ------------------------------------------------------------------
    # Chain verification
    # ------------------------------------------------------------------

    def verify_chain(self) -> bool:
        """Verify the integrity of the entire audit log chain.

        Returns True if all entries are correctly chained and no
        tampering is detected.  Returns False if any hash mismatch
        is found.
        """
        if not self._entries:
            return True

        for i, entry in enumerate(self._entries):
            # Verify previous hash linkage
            if i == 0:
                expected_prev = self.GENESIS_HASH
            else:
                expected_prev = self._entries[i - 1].entry_hash

            if entry.previous_hash != expected_prev:
                return False

            # Every entry must belong to this logger's system
            if entry.system_id != self.system_id:
                return False

            # Recompute and verify entry hash from the entry's own stored values
            hashable = self._create_hashable_content(
                entry_id=entry.entry_id,
                sequence_number=entry.sequence_number,
                timestamp=entry.timestamp,
                timestamp_iso=entry.timestamp_iso,
                system_id=entry.system_id,
                event_type=entry.event_type,
                payload=entry.payload,
                previous_hash=entry.previous_hash,
            )
            recomputed = self._compute_hash(hashable)
            if entry.entry_hash != recomputed:
                return False

        return True

    # ------------------------------------------------------------------
    # Query & export
    # ------------------------------------------------------------------

    @property
    def entries(self) -> List[AuditEntry]:
        return list(self._entries)

    @property
    def size(self) -> int:
        return len(self._entries)

    def get_entries_by_type(self, event_type: EventType) -> List[AuditEntry]:
        """Filter log entries by event type."""
        return [e for e in self._entries if e.event_type == event_type.value]

    def export_json(self) -> str:
        """Export the full audit log as a JSON array."""
        return json.dumps(
            [e.to_dict() for e in self._entries], default=str, indent=2
        )

    def export_evidence_package(self) -> Dict[str, Any]:
        """Generate a compliance evidence package for regulatory examination.

        Returns a structured dict containing:
          - System identification
          - Chain integrity verification result
          - Summary statistics
          - Full log entries
        """
        return {
            "system_id": self.system_id,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "chain_integrity": self.verify_chain(),
            "total_entries": self.size,
            "entry_type_counts": {
                et.value: len(self.get_entries_by_type(et))
                for et in EventType
            },
            "entries": [e.to_dict() for e in self._entries],
        }
