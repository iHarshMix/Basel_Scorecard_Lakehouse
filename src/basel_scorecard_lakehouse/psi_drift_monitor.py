"""Population Stability Index (PSI) & Production Drift Monitor.

Implements:
1. Multi-bin Population Stability Index (PSI) computation between baseline and live batches.
2. Drift classification according to Basel / Federal Reserve guidelines:
      PSI < 0.10  -> STABLE (No action needed)
      0.10 <= PSI < 0.25 -> MODERATE DRIFT (Warning)
      PSI >= 0.25 -> SIGNIFICANT DRIFT (Trigger Retraining)
3. Diagnostic distribution shift comparison plotting.
"""

from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class PSIDriftMonitor:
    """Monitors credit score distribution stability across chronological production epochs."""

    def __init__(self, n_bins: int = 10, epsilon: float = 1e-4):
        self.n_bins = n_bins
        self.epsilon = epsilon
        self.baseline_edges: List[float] = []

    def fit_baseline(self, baseline_scores: np.ndarray) -> None:
        """Establish baseline decile bin edges from training distribution (Epoch 1)."""
        quantiles = np.linspace(0, 1, self.n_bins + 1)
        edges = np.percentile(baseline_scores, quantiles * 100)
        edges = sorted(list(set(edges)))
        edges[0] = -np.inf
        edges[-1] = np.inf
        self.baseline_edges = edges

    def compute_psi(
        self, actual_scores: np.ndarray, expected_scores: np.ndarray = None
    ) -> Tuple[float, pd.DataFrame, str]:
        """Compute PSI across established decile bins against actual batch scores."""
        if not self.baseline_edges:
            if expected_scores is None:
                raise ValueError("Baseline edges not fitted and no expected_scores provided.")
            self.fit_baseline(expected_scores)

        # Expected counts (baseline)
        if expected_scores is not None:
            exp_bins = pd.cut(expected_scores, bins=self.baseline_edges, include_lowest=True)
            exp_counts = pd.Series(exp_bins).value_counts(sort=False).values.astype(float)
        else:
            exp_counts = np.ones(len(self.baseline_edges) - 1)

        # Actual counts (new batch)
        act_bins = pd.cut(actual_scores, bins=self.baseline_edges, include_lowest=True)
        act_counts = pd.Series(act_bins).value_counts(sort=False).values.astype(float)

        # Calculate percentages with smoothing epsilon
        pct_exp = (exp_counts / exp_counts.sum()).clip(min=self.epsilon)
        pct_act = (act_counts / act_counts.sum()).clip(min=self.epsilon)

        # Re-normalize after clipping
        pct_exp /= pct_exp.sum()
        pct_act /= pct_act.sum()

        # PSI = sum((Actual - Expected) * ln(Actual / Expected))
        psi_contributions = (pct_act - pct_exp) * np.log(pct_act / pct_exp)
        total_psi = float(np.sum(psi_contributions))

        # Status Assessment
        if total_psi < 0.10:
            status = "STABLE"
        elif total_psi < 0.25:
            status = "MODERATE_DRIFT"
        else:
            status = "SIGNIFICANT_DRIFT"

        breakdown_df = pd.DataFrame(
            {
                "bin_index": np.arange(1, len(pct_exp) + 1),
                "expected_pct": np.round(pct_exp * 100.0, 2),
                "actual_pct": np.round(pct_act * 100.0, 2),
                "psi_contribution": np.round(psi_contributions, 4),
            }
        )

        return round(total_psi, 4), breakdown_df, status

    def plot_psi_comparison(
        self,
        baseline_scores: np.ndarray,
        drifted_scores: np.ndarray,
        psi_val: float,
        status: str,
        output_path: Path,
        baseline_label: str = "2013-2015 Baseline",
        target_label: str = "2017 Live Batch",
    ) -> None:
        """Plot side-by-side decile distribution shift comparison."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        _, breakdown, _ = self.compute_psi(drifted_scores, baseline_scores)

        fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
        x = np.arange(len(breakdown))
        width = 0.38

        color_alert = (
            "#d9534f" if status == "SIGNIFICANT_DRIFT" else ("#f0ad4e" if status == "MODERATE_DRIFT" else "#5cb85c")
        )

        ax.bar(x - width / 2, breakdown["expected_pct"], width, label=baseline_label, color="#428bca")
        ax.bar(x + width / 2, breakdown["actual_pct"], width, label=target_label, color=color_alert)

        ax.set_title(f"Score Decile PSI Drift Analysis: PSI = {psi_val:.3f} [{status}]", fontsize=12, fontweight="bold")
        ax.set_xlabel("Score Decile (1 = Highest Risk, 10 = Lowest Risk)", fontsize=10)
        ax.set_ylabel("Population Percentage (%)", fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(breakdown["bin_index"])
        ax.set_ylim(0, max(breakdown["actual_pct"].max(), breakdown["expected_pct"].max()) * 1.25)
        ax.legend(loc="upper right", frameon=True)

        plt.tight_layout()
        plt.savefig(output_path)
        plt.close(fig)
