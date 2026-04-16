"""Tests for fairness metrics and bias detection."""

import numpy as np
import pytest

from responsible_ai_toolkit.fairness.metrics import FairnessMetrics, FairnessReport
from responsible_ai_toolkit.fairness.bias_detector import BiasDetector, BiasAlert


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_fair_data(n=1000, seed=42):
    """Generate data where both groups have similar outcomes."""
    rng = np.random.RandomState(seed)
    groups = rng.choice(["GroupA", "GroupB"], size=n)
    y_true = rng.binomial(1, 0.5, size=n)
    # Fair model: similar accuracy for both groups
    y_pred = y_true.copy()
    flip = rng.random(n) < 0.1
    y_pred[flip] = 1 - y_pred[flip]
    y_prob = np.where(y_pred == 1, rng.uniform(0.6, 0.95, n), rng.uniform(0.05, 0.4, n))
    return y_true, y_pred, y_prob, groups


def _make_biased_data(n=1000, seed=42):
    """Generate data where GroupB is systematically disadvantaged."""
    rng = np.random.RandomState(seed)
    groups = rng.choice(["GroupA", "GroupB"], size=n)
    y_true = rng.binomial(1, 0.5, size=n)
    y_pred = y_true.copy()

    # Introduce bias: GroupB gets more false negatives
    for i in range(n):
        if groups[i] == "GroupB" and y_true[i] == 1:
            if rng.random() < 0.4:
                y_pred[i] = 0  # Systematically deny GroupB positives

    y_prob = np.where(y_pred == 1, rng.uniform(0.6, 0.95, n), rng.uniform(0.05, 0.4, n))
    return y_true, y_pred, y_prob, groups


# ---------------------------------------------------------------------------
# FairnessMetrics tests
# ---------------------------------------------------------------------------

class TestFairnessMetrics:

    def test_demographic_parity_fair_data(self):
        y_true, y_pred, y_prob, groups = _make_fair_data()
        fm = FairnessMetrics(y_true, y_pred, groups, y_prob)
        dp = fm.demographic_parity()
        assert "GroupA" in dp
        assert "GroupB" in dp
        # Fair data should have similar rates
        assert abs(dp["GroupA"] - dp["GroupB"]) < 0.15

    def test_demographic_parity_biased_data(self):
        y_true, y_pred, y_prob, groups = _make_biased_data()
        fm = FairnessMetrics(y_true, y_pred, groups, y_prob)
        dp = fm.demographic_parity()
        # GroupB should have lower positive rate
        assert dp["GroupB"] < dp["GroupA"]

    def test_equal_opportunity(self):
        y_true, y_pred, y_prob, groups = _make_biased_data()
        fm = FairnessMetrics(y_true, y_pred, groups, y_prob)
        eo = fm.equal_opportunity()
        assert "GroupA" in eo and "GroupB" in eo
        # GroupB should have lower TPR due to bias
        assert eo["GroupB"] < eo["GroupA"]

    def test_equalized_odds(self):
        y_true, y_pred, y_prob, groups = _make_fair_data()
        fm = FairnessMetrics(y_true, y_pred, groups, y_prob)
        eq = fm.equalized_odds()
        for g in ["GroupA", "GroupB"]:
            assert "tpr" in eq[g]
            assert "fpr" in eq[g]
            assert 0 <= eq[g]["tpr"] <= 1
            assert 0 <= eq[g]["fpr"] <= 1

    def test_calibration(self):
        y_true, y_pred, y_prob, groups = _make_fair_data()
        fm = FairnessMetrics(y_true, y_pred, groups, y_prob)
        cal = fm.calibration()
        for g in ["GroupA", "GroupB"]:
            assert g in cal
            assert cal[g] >= 0  # Calibration error is non-negative

    def test_calibration_requires_probs(self):
        y_true, y_pred, _, groups = _make_fair_data()
        fm = FairnessMetrics(y_true, y_pred, groups)
        with pytest.raises(ValueError, match="y_prob is required"):
            fm.calibration()

    def test_full_report_fair(self):
        y_true, y_pred, y_prob, groups = _make_fair_data()
        fm = FairnessMetrics(y_true, y_pred, groups, y_prob)
        report = fm.full_report()
        assert isinstance(report, FairnessReport)
        assert report.overall_fair  # Fair data should pass

    def test_full_report_biased(self):
        y_true, y_pred, y_prob, groups = _make_biased_data()
        fm = FairnessMetrics(y_true, y_pred, groups, y_prob)
        report = fm.full_report()
        assert not report.overall_fair
        assert len(report.violations) > 0

    def test_report_summary_string(self):
        y_true, y_pred, y_prob, groups = _make_biased_data()
        fm = FairnessMetrics(y_true, y_pred, groups, y_prob)
        report = fm.full_report()
        summary = report.summary()
        assert "FAIRNESS EVALUATION REPORT" in summary
        assert "VIOLATIONS DETECTED" in summary

    def test_validation_mismatched_lengths(self):
        with pytest.raises(ValueError):
            FairnessMetrics([0, 1], [0], ["A", "B"])

    def test_validation_non_binary(self):
        with pytest.raises(ValueError, match="binary"):
            FairnessMetrics([0, 1, 2], [0, 1, 0], ["A", "B", "A"])

    def test_validation_single_group(self):
        with pytest.raises(ValueError, match="at least two groups"):
            FairnessMetrics([0, 1], [0, 1], ["A", "A"])

    def test_custom_threshold(self):
        y_true, y_pred, y_prob, groups = _make_biased_data()
        # Very lenient threshold
        fm = FairnessMetrics(y_true, y_pred, groups, y_prob, threshold=0.30)
        report = fm.full_report()
        # With 30% threshold, fewer violations
        lenient_violations = len(report.violations)

        # Strict threshold
        fm2 = FairnessMetrics(y_true, y_pred, groups, y_prob, threshold=0.95)
        report2 = fm2.full_report()
        assert len(report2.violations) >= lenient_violations


# ---------------------------------------------------------------------------
# BiasDetector tests
# ---------------------------------------------------------------------------

class TestBiasDetector:

    def test_streaming_observation(self):
        detector = BiasDetector(window_size=100)
        for i in range(50):
            detector.observe(y_true=1, y_pred=1, group="A")
            detector.observe(y_true=0, y_pred=0, group="B")
        assert detector.window_size_current == 100

    def test_batch_observation(self):
        detector = BiasDetector(window_size=500)
        y_true, y_pred, y_prob, groups = _make_fair_data(n=200)
        detector.observe_batch(y_true, y_pred, groups, y_prob)
        assert detector.window_size_current == 200

    def test_evaluate_returns_report(self):
        detector = BiasDetector(window_size=500)
        y_true, y_pred, y_prob, groups = _make_fair_data(n=200)
        detector.observe_batch(y_true, y_pred, groups, y_prob)
        report = detector.evaluate()
        assert isinstance(report, FairnessReport)

    def test_alert_on_bias(self):
        alerts_received = []
        detector = BiasDetector(
            window_size=500,
            on_alert=lambda a: alerts_received.append(a),
        )
        y_true, y_pred, y_prob, groups = _make_biased_data(n=500)
        detector.observe_batch(y_true, y_pred, groups, y_prob)
        detector.evaluate()
        assert len(alerts_received) > 0
        assert all(isinstance(a, BiasAlert) for a in alerts_received)

    def test_no_alert_on_fair_data(self):
        alerts_received = []
        detector = BiasDetector(
            window_size=500,
            on_alert=lambda a: alerts_received.append(a),
        )
        y_true, y_pred, y_prob, groups = _make_fair_data(n=500)
        detector.observe_batch(y_true, y_pred, groups, y_prob)
        detector.evaluate()
        assert len(alerts_received) == 0

    def test_clear_resets_state(self):
        detector = BiasDetector()
        detector.observe(y_true=1, y_pred=1, group="A")
        detector.clear()
        assert detector.window_size_current == 0
        assert detector.evaluation_count == 0

    def test_empty_window_raises(self):
        detector = BiasDetector()
        with pytest.raises(ValueError, match="No observations"):
            detector.evaluate()
