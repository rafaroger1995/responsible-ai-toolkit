"""
Fairness metrics for evaluating AI model outputs across demographic groups.

Implements the four primary fairness criteria referenced by federal
regulators and the NIST AI Risk Management Framework:
  - Demographic Parity (Statistical Parity)
  - Equal Opportunity (True Positive Rate Parity)
  - Equalized Odds (TPR + FPR Parity)
  - Calibration (Predictive Value Parity)

These metrics are designed for production deployment in regulated
financial services environments — lending decisions, insurance
underwriting, claims processing, and fraud detection — where
algorithmic bias can violate fair lending laws (ECOA, FHA) and
consumer protection requirements.

Usage:
    >>> from responsible_ai_toolkit.fairness import FairnessMetrics
    >>> fm = FairnessMetrics(
    ...     y_true=labels,
    ...     y_pred=predictions,
    ...     y_prob=probabilities,
    ...     sensitive_attr=demographic_groups,
    ... )
    >>> report = fm.full_report()
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class GroupMetrics:
    """Metrics computed for a single demographic group."""

    group_name: str
    size: int
    positive_rate: float
    true_positive_rate: Optional[float] = None
    false_positive_rate: Optional[float] = None
    positive_predictive_value: Optional[float] = None
    calibration_score: Optional[float] = None


@dataclass
class FairnessReport:
    """Complete fairness evaluation across all groups and criteria."""

    demographic_parity: Dict[str, float] = field(default_factory=dict)
    equal_opportunity: Dict[str, float] = field(default_factory=dict)
    equalized_odds: Dict[str, Dict[str, float]] = field(default_factory=dict)
    calibration: Dict[str, float] = field(default_factory=dict)
    group_metrics: Dict[str, GroupMetrics] = field(default_factory=dict)
    violations: List[str] = field(default_factory=list)
    overall_fair: bool = True

    def summary(self) -> str:
        lines = ["=" * 60, "FAIRNESS EVALUATION REPORT", "=" * 60]
        lines.append(f"\nOverall Assessment: {'PASS' if self.overall_fair else 'VIOLATIONS DETECTED'}")
        lines.append(f"Violations Found:   {len(self.violations)}\n")

        lines.append("--- Demographic Parity (Selection Rate by Group) ---")
        for g, v in self.demographic_parity.items():
            lines.append(f"  {g:>20s}: {v:.4f}")

        lines.append("\n--- Equal Opportunity (True Positive Rate by Group) ---")
        for g, v in self.equal_opportunity.items():
            lines.append(f"  {g:>20s}: {v:.4f}")

        lines.append("\n--- Equalized Odds (TPR / FPR by Group) ---")
        for g, rates in self.equalized_odds.items():
            tpr = rates.get("tpr", float("nan"))
            fpr = rates.get("fpr", float("nan"))
            lines.append(f"  {g:>20s}: TPR={tpr:.4f}  FPR={fpr:.4f}")

        lines.append("\n--- Calibration (Avg Predicted Prob vs Actual Rate) ---")
        for g, v in self.calibration.items():
            lines.append(f"  {g:>20s}: {v:.4f}")

        if self.violations:
            lines.append("\n--- Violations ---")
            for v in self.violations:
                lines.append(f"  ⚠  {v}")

        lines.append("=" * 60)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core engine
# ---------------------------------------------------------------------------

class FairnessMetrics:
    """Compute fairness metrics across demographic groups.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground-truth binary labels (0 or 1).
    y_pred : array-like of shape (n_samples,)
        Predicted binary labels (0 or 1).
    sensitive_attr : array-like of shape (n_samples,)
        Categorical group membership for each sample (e.g., race,
        gender, age bracket).
    y_prob : array-like of shape (n_samples,), optional
        Predicted probabilities (required for calibration metric).
    threshold : float, default 0.80
        Minimum ratio between the least-favored and most-favored group
        for a metric to be considered fair.  The 80% (four-fifths) rule
        is the default, consistent with EEOC adverse-impact guidance and
        commonly applied in fair-lending examinations.
    reference_group : str, optional
        Group to use as the reference for ratio comparisons.  If *None*,
        the group with the highest metric value is used automatically.
    """

    def __init__(
        self,
        y_true: Sequence,
        y_pred: Sequence,
        sensitive_attr: Sequence,
        y_prob: Optional[Sequence] = None,
        threshold: float = 0.80,
        reference_group: Optional[str] = None,
    ) -> None:
        self.y_true = np.asarray(y_true, dtype=int)
        self.y_pred = np.asarray(y_pred, dtype=int)
        self.sensitive_attr = np.asarray(sensitive_attr)
        self.y_prob = np.asarray(y_prob, dtype=float) if y_prob is not None else None
        self.threshold = threshold
        self.reference_group = reference_group

        self._validate()
        self.groups = np.unique(self.sensitive_attr)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate(self) -> None:
        n = len(self.y_true)
        if len(self.y_pred) != n or len(self.sensitive_attr) != n:
            raise ValueError(
                "y_true, y_pred, and sensitive_attr must have the same length."
            )
        if self.y_prob is not None and len(self.y_prob) != n:
            raise ValueError("y_prob must have the same length as y_true.")

        unique_labels = set(np.unique(self.y_true)) | set(np.unique(self.y_pred))
        if not unique_labels.issubset({0, 1}):
            raise ValueError("y_true and y_pred must be binary (0 or 1).")

        if len(np.unique(self.sensitive_attr)) < 2:
            raise ValueError("sensitive_attr must contain at least two groups.")

    # ------------------------------------------------------------------
    # Per-group helpers
    # ------------------------------------------------------------------

    def _group_mask(self, group: str) -> np.ndarray:
        return self.sensitive_attr == group

    def _safe_divide(self, numerator: float, denominator: float) -> float:
        if denominator == 0:
            return float("nan")
        return numerator / denominator

    # ------------------------------------------------------------------
    # Demographic Parity  (P(ŷ=1 | G=g))
    # ------------------------------------------------------------------

    def demographic_parity(self) -> Dict[str, float]:
        """Compute selection (positive prediction) rate per group."""
        result: Dict[str, float] = {}
        for g in self.groups:
            mask = self._group_mask(g)
            rate = self.y_pred[mask].mean()
            result[str(g)] = float(rate)
        return result

    # ------------------------------------------------------------------
    # Equal Opportunity  (P(ŷ=1 | y=1, G=g))
    # ------------------------------------------------------------------

    def equal_opportunity(self) -> Dict[str, float]:
        """Compute true-positive rate (recall) per group."""
        result: Dict[str, float] = {}
        for g in self.groups:
            mask = self._group_mask(g)
            positives = self.y_true[mask] == 1
            if positives.sum() == 0:
                result[str(g)] = float("nan")
            else:
                tpr = self.y_pred[mask][positives].mean()
                result[str(g)] = float(tpr)
        return result

    # ------------------------------------------------------------------
    # Equalized Odds  (TPR + FPR parity)
    # ------------------------------------------------------------------

    def equalized_odds(self) -> Dict[str, Dict[str, float]]:
        """Compute TPR and FPR per group."""
        result: Dict[str, Dict[str, float]] = {}
        for g in self.groups:
            mask = self._group_mask(g)
            y_t = self.y_true[mask]
            y_p = self.y_pred[mask]

            tp = ((y_p == 1) & (y_t == 1)).sum()
            fn = ((y_p == 0) & (y_t == 1)).sum()
            fp = ((y_p == 1) & (y_t == 0)).sum()
            tn = ((y_p == 0) & (y_t == 0)).sum()

            tpr = self._safe_divide(tp, tp + fn)
            fpr = self._safe_divide(fp, fp + tn)
            result[str(g)] = {"tpr": float(tpr), "fpr": float(fpr)}
        return result

    # ------------------------------------------------------------------
    # Calibration  (E[y | ŷ_prob, G=g])
    # ------------------------------------------------------------------

    def calibration(self, n_bins: int = 10) -> Dict[str, float]:
        """Compute calibration score (mean absolute calibration error)
        per group using binned predicted probabilities.

        A perfectly calibrated model has a score of 0.0 for every group.
        """
        if self.y_prob is None:
            raise ValueError("y_prob is required for calibration metric.")

        result: Dict[str, float] = {}
        for g in self.groups:
            mask = self._group_mask(g)
            probs = self.y_prob[mask]
            labels = self.y_true[mask]

            if len(probs) == 0:
                result[str(g)] = float("nan")
                continue

            bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
            calibration_errors = []

            for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
                if lo == bin_edges[-2]:
                    bin_mask = (probs >= lo) & (probs <= hi)
                else:
                    bin_mask = (probs >= lo) & (probs < hi)

                if bin_mask.sum() == 0:
                    continue

                avg_pred = probs[bin_mask].mean()
                avg_actual = labels[bin_mask].mean()
                calibration_errors.append(abs(avg_pred - avg_actual))

            result[str(g)] = float(np.mean(calibration_errors)) if calibration_errors else float("nan")

        return result

    # ------------------------------------------------------------------
    # Disparity ratios + violation detection
    # ------------------------------------------------------------------

    def _compute_disparity_ratios(
        self, metric_values: Dict[str, float]
    ) -> Tuple[Dict[str, float], str]:
        """Compute ratio of each group's metric to the reference group."""
        valid = {k: v for k, v in metric_values.items() if not np.isnan(v)}
        if not valid:
            return {}, ""

        if self.reference_group and self.reference_group in valid:
            ref = self.reference_group
        else:
            ref = max(valid, key=valid.get)

        ref_val = valid[ref]
        ratios = {}
        for g, v in valid.items():
            ratios[g] = self._safe_divide(v, ref_val) if ref_val != 0 else float("nan")
        return ratios, ref

    def _check_violations(
        self, metric_name: str, metric_values: Dict[str, float]
    ) -> List[str]:
        """Check whether any group falls below the fairness threshold."""
        ratios, ref = self._compute_disparity_ratios(metric_values)
        violations = []
        for g, ratio in ratios.items():
            if not np.isnan(ratio) and ratio < self.threshold:
                violations.append(
                    f"{metric_name}: Group '{g}' ratio = {ratio:.3f} "
                    f"(< {self.threshold:.2f} threshold vs reference '{ref}')"
                )
        return violations

    # ------------------------------------------------------------------
    # Group-level metrics
    # ------------------------------------------------------------------

    def compute_group_metrics(self) -> Dict[str, GroupMetrics]:
        """Compute detailed metrics for each demographic group."""
        dp = self.demographic_parity()
        eo = self.equal_opportunity()
        eq = self.equalized_odds()
        cal = self.calibration() if self.y_prob is not None else {}

        result = {}
        for g in self.groups:
            gs = str(g)
            mask = self._group_mask(g)
            result[gs] = GroupMetrics(
                group_name=gs,
                size=int(mask.sum()),
                positive_rate=dp.get(gs, float("nan")),
                true_positive_rate=eo.get(gs, float("nan")),
                false_positive_rate=eq.get(gs, {}).get("fpr", float("nan")),
                positive_predictive_value=None,
                calibration_score=cal.get(gs, float("nan")),
            )
        return result

    # ------------------------------------------------------------------
    # Full report
    # ------------------------------------------------------------------

    def full_report(self) -> FairnessReport:
        """Run all fairness evaluations and return a complete report."""
        dp = self.demographic_parity()
        eo = self.equal_opportunity()
        eq = self.equalized_odds()
        cal = self.calibration() if self.y_prob is not None else {}

        violations: List[str] = []
        violations.extend(self._check_violations("Demographic Parity", dp))
        violations.extend(self._check_violations("Equal Opportunity", eo))

        # Check equalized odds (both TPR and FPR)
        tpr_vals = {g: v["tpr"] for g, v in eq.items()}
        fpr_vals = {g: v["fpr"] for g, v in eq.items()}
        violations.extend(self._check_violations("Equalized Odds (TPR)", tpr_vals))
        violations.extend(self._check_violations("Equalized Odds (FPR)", fpr_vals))

        return FairnessReport(
            demographic_parity=dp,
            equal_opportunity=eo,
            equalized_odds=eq,
            calibration=cal,
            group_metrics=self.compute_group_metrics(),
            violations=violations,
            overall_fair=len(violations) == 0,
        )
