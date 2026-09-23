"""
Distribution drift measures for monitoring model inputs and outputs.

Compares a reference dataset (for example, scores from the period a model
was validated on) with recent production data, and reports how far the
distribution has shifted. A shift is a signal to investigate; it does not
by itself show that a model's predictions have become unreliable.

Implements three complementary measures:
  - **Population Stability Index (PSI)**: a binned measure widely used
    in credit-risk model monitoring.
  - **KL Divergence**: an information-theoretic measure of distribution
    shift, sensitive to tail behavior.
  - **Wasserstein Distance**: the earth mover's distance between the two
    empirical distributions, in the units of the data.

Production values outside the reference range are counted in the lowest
or highest bin, so a shift beyond the reference range raises PSI and KL
rather than being ignored.

The default PSI thresholds (0.10 and 0.25) are common rules of thumb, not
regulatory or supervisory standards. Set them to fit the model and the
institution's own monitoring policy.

Usage:
    >>> import numpy as np
    >>> from responsible_ai_toolkit.drift import DriftMonitor
    >>> rng = np.random.default_rng(0)
    >>> monitor = DriftMonitor(reference=rng.normal(0.0, 1.0, 5000))
    >>> report = monitor.evaluate(rng.normal(1.0, 1.0, 5000))
    >>> report.psi > 0.25
    True
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class DriftReport:
    """Results from a drift evaluation."""

    psi: float
    kl_divergence: float
    wasserstein_distance: float
    psi_interpretation: str
    bin_details: List[Dict[str, float]] = field(default_factory=list)
    timestamp: Optional[float] = None

    def summary(self) -> str:
        lines = [
            "=" * 50,
            "MODEL DRIFT REPORT",
            "=" * 50,
            f"  PSI:                  {self.psi:.6f}  ({self.psi_interpretation})",
            f"  KL Divergence:        {self.kl_divergence:.6f}",
            f"  Wasserstein Distance: {self.wasserstein_distance:.6f}",
            "=" * 50,
        ]
        return "\n".join(lines)


class DriftMonitor:
    """Monitor distribution drift between reference and production data.

    Parameters
    ----------
    reference : array-like
        Reference distribution (e.g., training or validation scores).
    n_bins : int, default 10
        Number of bins for histogram-based metrics (PSI, KL).
    epsilon : float, default 1e-6
        Small constant added to prevent division by zero and log(0)
        in PSI and KL computations.
    psi_thresholds : tuple of (float, float), default (0.10, 0.25)
        (low, high) thresholds for PSI interpretation:
          - PSI < low: "No significant drift"
          - low <= PSI < high: "Moderate drift — monitor closely"
          - PSI >= high: "Significant drift — investigate"
        These defaults are common rules of thumb. They are configuration
        values, not regulatory or supervisory standards.
    """

    def __init__(
        self,
        reference: np.ndarray,
        n_bins: int = 10,
        epsilon: float = 1e-6,
        psi_thresholds: Tuple[float, float] = (0.10, 0.25),
    ) -> None:
        self.reference = np.asarray(reference, dtype=float)
        self.n_bins = n_bins
        self.epsilon = epsilon
        self.psi_low, self.psi_high = psi_thresholds

        if len(self.reference) < self.n_bins:
            raise ValueError(
                f"Reference data has {len(self.reference)} samples, "
                f"but at least {self.n_bins} are required for {self.n_bins} bins."
            )
        if not np.all(np.isfinite(self.reference)):
            raise ValueError("Reference data contains missing or non-finite values.")

        # Pre-compute reference bin edges and proportions
        self._bin_edges = np.histogram_bin_edges(self.reference, bins=self.n_bins)
        ref_counts, _ = np.histogram(self.reference, bins=self._bin_edges)
        self._ref_proportions = ref_counts / ref_counts.sum()

    # ------------------------------------------------------------------
    # Input handling
    # ------------------------------------------------------------------

    def _validate_production(self, production: np.ndarray) -> np.ndarray:
        production = np.asarray(production, dtype=float)
        if production.size == 0:
            raise ValueError("Production data is empty.")
        if not np.all(np.isfinite(production)):
            raise ValueError("Production data contains missing or non-finite values.")
        return production

    def _production_proportions(self, production: np.ndarray) -> np.ndarray:
        """Bin production data, counting out-of-range values in the end bins."""
        clipped = np.clip(production, self._bin_edges[0], self._bin_edges[-1])
        prod_counts, _ = np.histogram(clipped, bins=self._bin_edges)
        return prod_counts / prod_counts.sum()

    # ------------------------------------------------------------------
    # Population Stability Index
    # ------------------------------------------------------------------

    def compute_psi(self, production: np.ndarray) -> Tuple[float, List[Dict[str, float]]]:
        """Compute Population Stability Index (PSI).

        PSI = Σ (P_i - Q_i) * ln(P_i / Q_i)

        where P_i is the production proportion and Q_i is the reference
        proportion for bin i. Production values below or above the
        reference range are counted in the first or last bin.

        Returns
        -------
        psi_value : float
        bin_details : list of dict
            Per-bin breakdown showing reference proportion, production
            proportion, and contribution to overall PSI.
        """
        production = self._validate_production(production)
        prod_proportions = self._production_proportions(production)

        # Add epsilon to avoid division by zero
        ref_adj = self._ref_proportions + self.epsilon
        prod_adj = prod_proportions + self.epsilon

        # Normalize after adding epsilon
        ref_adj = ref_adj / ref_adj.sum()
        prod_adj = prod_adj / prod_adj.sum()

        psi_bins = (prod_adj - ref_adj) * np.log(prod_adj / ref_adj)
        psi_value = float(psi_bins.sum())

        bin_details = []
        for i in range(len(psi_bins)):
            bin_details.append({
                "bin_low": float(self._bin_edges[i]),
                "bin_high": float(self._bin_edges[i + 1]),
                "ref_proportion": float(self._ref_proportions[i]),
                "prod_proportion": float(prod_proportions[i]),
                "psi_contribution": float(psi_bins[i]),
            })

        return psi_value, bin_details

    # ------------------------------------------------------------------
    # KL Divergence
    # ------------------------------------------------------------------

    def compute_kl_divergence(self, production: np.ndarray) -> float:
        """Compute KL divergence: D_KL(Production || Reference).

        KL(P || Q) = Σ P_i * ln(P_i / Q_i)

        Production values outside the reference range are counted in the
        first or last bin.
        """
        production = self._validate_production(production)
        prod_proportions = self._production_proportions(production)

        ref_adj = self._ref_proportions + self.epsilon
        prod_adj = prod_proportions + self.epsilon

        ref_adj = ref_adj / ref_adj.sum()
        prod_adj = prod_adj / prod_adj.sum()

        kl = float(np.sum(prod_adj * np.log(prod_adj / ref_adj)))
        return kl

    # ------------------------------------------------------------------
    # Wasserstein Distance (Earth Mover's Distance)
    # ------------------------------------------------------------------

    def compute_wasserstein(self, production: np.ndarray) -> float:
        """Compute 1D Wasserstein (earth mover's) distance.

        W_1(P, Q) = integral |F_P(x) - F_Q(x)| dx

        Both empirical CDFs are step functions that are constant between
        consecutive observed values, so the integral is the sum of
        |F_P - F_Q| on each interval times the interval's width.
        """
        production = self._validate_production(production)
        ref_sorted = np.sort(self.reference)
        prod_sorted = np.sort(production)

        all_values = np.sort(np.concatenate([ref_sorted, prod_sorted]))

        ref_cdf = np.searchsorted(ref_sorted, all_values, side="right") / len(ref_sorted)
        prod_cdf = np.searchsorted(prod_sorted, all_values, side="right") / len(prod_sorted)

        # The CDF value at each point holds until the next point.
        diffs = np.abs(ref_cdf - prod_cdf)[:-1]
        widths = np.diff(all_values)
        distance = float(np.sum(diffs * widths))

        return distance

    # ------------------------------------------------------------------
    # PSI interpretation
    # ------------------------------------------------------------------

    def _interpret_psi(self, psi: float) -> str:
        if psi < self.psi_low:
            return "No significant drift"
        elif psi < self.psi_high:
            return "Moderate drift — monitor closely"
        else:
            return "Significant drift — investigate"

    # ------------------------------------------------------------------
    # Full evaluation
    # ------------------------------------------------------------------

    def evaluate(self, production: np.ndarray) -> DriftReport:
        """Run all drift metrics against production data.

        Parameters
        ----------
        production : array-like
            Production data (e.g., current model scores or predictions).

        Returns
        -------
        DriftReport
        """
        production = self._validate_production(production)

        psi, bin_details = self.compute_psi(production)
        kl = self.compute_kl_divergence(production)
        wasserstein = self.compute_wasserstein(production)

        return DriftReport(
            psi=psi,
            kl_divergence=kl,
            wasserstein_distance=wasserstein,
            psi_interpretation=self._interpret_psi(psi),
            bin_details=bin_details,
        )
