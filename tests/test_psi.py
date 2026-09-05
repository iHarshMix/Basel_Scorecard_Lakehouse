"""Unit tests for Population Stability Index (PSI) drift monitoring."""

import numpy as np

from basel_scorecard_lakehouse.psi_drift_monitor import PSIDriftMonitor


def test_psi_identical_distributions():
    """Verify PSI is approximately 0.0 when actual distribution matches baseline."""
    rng = np.random.default_rng(42)
    baseline_scores = rng.normal(680, 45, size=5000)
    actual_scores = rng.normal(680, 45, size=5000)

    monitor = PSIDriftMonitor(n_bins=10)
    psi_val, breakdown, status = monitor.compute_psi(actual_scores, baseline_scores)

    assert psi_val < 0.02
    assert status == "STABLE"
    assert len(breakdown) == 10


def test_psi_significant_drift_detection():
    """Verify PSI exceeds 0.25 and triggers SIGNIFICANT_DRIFT alert on population deterioration."""
    rng = np.random.default_rng(101)
    baseline_scores = rng.normal(700, 40, size=5000)  # Healthy baseline
    drifted_scores = rng.normal(610, 50, size=5000)   # Severe macro deterioration

    monitor = PSIDriftMonitor(n_bins=10)
    psi_val, _, status = monitor.compute_psi(drifted_scores, baseline_scores)

    assert psi_val >= 0.25
    assert status == "SIGNIFICANT_DRIFT"


def test_psi_breakdown_structure():
    """Verify breakdown table sums to 100% and contains correct column headers."""
    baseline = np.linspace(300, 850, 1000)
    actual = np.linspace(350, 800, 1000)

    monitor = PSIDriftMonitor(n_bins=10)
    _, breakdown, _ = monitor.compute_psi(actual, baseline)

    assert "bin_index" in breakdown.columns
    assert "expected_pct" in breakdown.columns
    assert "actual_pct" in breakdown.columns
    assert "psi_contribution" in breakdown.columns
    assert np.isclose(breakdown["expected_pct"].sum(), 100.0, atol=0.1)
    assert np.isclose(breakdown["actual_pct"].sum(), 100.0, atol=0.1)
