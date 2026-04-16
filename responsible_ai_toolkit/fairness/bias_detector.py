"""
Continuous bias detection service for production AI systems.

Monitors model predictions over time, detects emerging fairness
violations, and triggers alerts or HITL escalation when bias thresholds
are breached.  Designed for deployment alongside real-time AI pipelines
in financial services — lending, underwriting, claims, fraud detection.

The detector operates in two modes:
  - **Batch**: Evaluate a fixed dataset (e.g., nightly or weekly runs).
  - **Streaming**: Accumulate predictions in a sliding window and
    evaluate periodically.

Usage:
    >>> from responsible_ai_toolkit.fairness import BiasDetector
    >>> detector = BiasDetector(threshold=0.80, window_size=1000)
    >>> detector.observe(y_true=1, y_pred=1, y_prob=0.87, group="GroupA")
    >>> alert = detector.evaluate()
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, List, Optional

import numpy as np

from responsible_ai_toolkit.fairness.metrics import FairnessMetrics, FairnessReport


@dataclass
class BiasAlert:
    """An alert raised when a fairness violation is detected."""

    timestamp: float
    metric_name: str
    group: str
    value: float
    threshold: float
    reference_group: str
    reference_value: float
    severity: str  # "warning" or "critical"
    message: str

    def __str__(self) -> str:
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.timestamp))
        return f"[{self.severity.upper()}] {ts} — {self.message}"


@dataclass
class Observation:
    """A single model prediction observation for bias monitoring."""

    y_true: int
    y_pred: int
    y_prob: Optional[float]
    group: str
    timestamp: float = field(default_factory=time.time)


class BiasDetector:
    """Continuous bias detection engine.

    Parameters
    ----------
    threshold : float, default 0.80
        Four-fifths (80%) rule threshold for disparity ratios.
    critical_threshold : float, default 0.60
        Below this ratio, alerts are elevated to "critical" severity.
    window_size : int, default 1000
        Maximum observations in the sliding window.
    min_group_size : int, default 30
        Minimum samples per group required before evaluation.
        Statistical reliability requires sufficient observations.
    on_alert : callable, optional
        Callback invoked with a ``BiasAlert`` when violations are found.
        Use this to integrate with HITL escalation, logging, or
        notification systems.
    """

    def __init__(
        self,
        threshold: float = 0.80,
        critical_threshold: float = 0.60,
        window_size: int = 1000,
        min_group_size: int = 30,
        on_alert: Optional[Callable[[BiasAlert], None]] = None,
    ) -> None:
        self.threshold = threshold
        self.critical_threshold = critical_threshold
        self.window_size = window_size
        self.min_group_size = min_group_size
        self.on_alert = on_alert

        self._window: Deque[Observation] = deque(maxlen=window_size)
        self._alert_history: List[BiasAlert] = []
        self._evaluation_count: int = 0

    # ------------------------------------------------------------------
    # Observation ingestion
    # ------------------------------------------------------------------

    def observe(
        self,
        y_true: int,
        y_pred: int,
        group: str,
        y_prob: Optional[float] = None,
    ) -> None:
        """Record a single prediction observation."""
        self._window.append(
            Observation(y_true=y_true, y_pred=y_pred, y_prob=y_prob, group=group)
        )

    def observe_batch(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        groups: np.ndarray,
        y_prob: Optional[np.ndarray] = None,
    ) -> None:
        """Record a batch of prediction observations."""
        for i in range(len(y_true)):
            self.observe(
                y_true=int(y_true[i]),
                y_pred=int(y_pred[i]),
                group=str(groups[i]),
                y_prob=float(y_prob[i]) if y_prob is not None else None,
            )

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self) -> FairnessReport:
        """Evaluate current window for fairness violations.

        Returns a ``FairnessReport`` and triggers alerts for any
        violations detected.
        """
        if len(self._window) == 0:
            raise ValueError("No observations in the window to evaluate.")

        obs = list(self._window)
        y_true = np.array([o.y_true for o in obs])
        y_pred = np.array([o.y_pred for o in obs])
        groups = np.array([o.group for o in obs])
        has_prob = all(o.y_prob is not None for o in obs)
        y_prob = np.array([o.y_prob for o in obs]) if has_prob else None

        # Check minimum group sizes
        unique, counts = np.unique(groups, return_counts=True)
        small_groups = [g for g, c in zip(unique, counts) if c < self.min_group_size]
        if small_groups:
            # Filter out groups that are too small for reliable evaluation
            valid_mask = np.isin(groups, [g for g, c in zip(unique, counts) if c >= self.min_group_size])
            if valid_mask.sum() == 0:
                raise ValueError(
                    f"No group has >= {self.min_group_size} observations. "
                    f"Cannot produce reliable fairness evaluation."
                )
            y_true = y_true[valid_mask]
            y_pred = y_pred[valid_mask]
            groups = groups[valid_mask]
            if y_prob is not None:
                y_prob = y_prob[valid_mask]

        fm = FairnessMetrics(
            y_true=y_true,
            y_pred=y_pred,
            sensitive_attr=groups,
            y_prob=y_prob,
            threshold=self.threshold,
        )

        report = fm.full_report()
        self._evaluation_count += 1

        # Generate alerts for violations
        self._process_violations(report)

        return report

    def _process_violations(self, report: FairnessReport) -> None:
        """Generate BiasAlert objects from detected violations."""
        now = time.time()

        # Check demographic parity
        self._check_metric_alerts(
            metric_name="Demographic Parity",
            values=report.demographic_parity,
            timestamp=now,
        )

        # Check equal opportunity
        self._check_metric_alerts(
            metric_name="Equal Opportunity",
            values=report.equal_opportunity,
            timestamp=now,
        )

        # Check equalized odds (TPR)
        tpr_vals = {g: v["tpr"] for g, v in report.equalized_odds.items()}
        self._check_metric_alerts(
            metric_name="Equalized Odds (TPR)",
            values=tpr_vals,
            timestamp=now,
        )

    def _check_metric_alerts(
        self,
        metric_name: str,
        values: Dict[str, float],
        timestamp: float,
    ) -> None:
        """Check a single metric for threshold violations and emit alerts."""
        valid = {k: v for k, v in values.items() if not np.isnan(v) and v is not None}
        if not valid:
            return

        ref_group = max(valid, key=valid.get)
        ref_value = valid[ref_group]

        if ref_value == 0:
            return

        for group, value in valid.items():
            ratio = value / ref_value
            if ratio < self.threshold:
                severity = "critical" if ratio < self.critical_threshold else "warning"
                alert = BiasAlert(
                    timestamp=timestamp,
                    metric_name=metric_name,
                    group=group,
                    value=value,
                    threshold=self.threshold,
                    reference_group=ref_group,
                    reference_value=ref_value,
                    severity=severity,
                    message=(
                        f"{metric_name} disparity: Group '{group}' = {value:.4f} "
                        f"vs reference '{ref_group}' = {ref_value:.4f} "
                        f"(ratio = {ratio:.3f}, threshold = {self.threshold:.2f})"
                    ),
                )
                self._alert_history.append(alert)
                if self.on_alert:
                    self.on_alert(alert)

    # ------------------------------------------------------------------
    # State inspection
    # ------------------------------------------------------------------

    @property
    def alert_history(self) -> List[BiasAlert]:
        return list(self._alert_history)

    @property
    def window_size_current(self) -> int:
        return len(self._window)

    @property
    def evaluation_count(self) -> int:
        return self._evaluation_count

    def clear(self) -> None:
        """Clear the observation window and alert history."""
        self._window.clear()
        self._alert_history.clear()
        self._evaluation_count = 0
