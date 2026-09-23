import numpy as np
import pytest

from responsible_ai_toolkit.drift.monitor import DriftMonitor


def test_identical_data_shows_no_drift():
    data = np.random.default_rng(0).normal(0.0, 1.0, 2000)
    report = DriftMonitor(reference=data).evaluate(data)
    assert report.psi < 0.01
    assert report.wasserstein_distance == pytest.approx(0.0)
    assert report.psi_interpretation == "No significant drift"


def test_values_outside_reference_range_raise_psi():
    rng = np.random.default_rng(1)
    monitor = DriftMonitor(reference=rng.normal(0.0, 1.0, 5000))
    production = np.concatenate([rng.normal(0.0, 1.0, 500), np.full(500, 10.0)])

    report = monitor.evaluate(production)

    assert report.psi >= 0.25
    assert report.psi_interpretation == "Significant drift — investigate"


def test_production_entirely_outside_reference_range_gives_finite_psi():
    monitor = DriftMonitor(reference=np.linspace(0.0, 1.0, 100))

    report = monitor.evaluate(np.full(50, 5.0))

    assert np.isfinite(report.psi)
    assert np.isfinite(report.kl_divergence)
    assert report.psi >= 0.25


def test_wasserstein_equals_constant_shift():
    data = np.random.default_rng(2).normal(0.0, 1.0, 1000)
    monitor = DriftMonitor(reference=data)

    assert monitor.compute_wasserstein(data + 0.5) == pytest.approx(0.5)
    assert DriftMonitor(reference=np.zeros(10)).compute_wasserstein(np.ones(10)) == pytest.approx(1.0)


@pytest.mark.parametrize("bad", [np.array([]), np.array([0.1, np.nan, 0.3])])
def test_empty_or_missing_production_data_is_rejected(bad):
    monitor = DriftMonitor(reference=np.linspace(0.0, 1.0, 100))
    with pytest.raises(ValueError):
        monitor.evaluate(bad)


def test_missing_values_in_reference_are_rejected():
    reference = np.linspace(0.0, 1.0, 100)
    reference[3] = np.nan
    with pytest.raises(ValueError):
        DriftMonitor(reference=reference)
