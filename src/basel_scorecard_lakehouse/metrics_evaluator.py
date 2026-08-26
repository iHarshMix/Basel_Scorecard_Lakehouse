"""Quantitative Risk Metrics & Decile Validation Engine.

Implements:
1. Kolmogorov-Smirnov (KS) Decile Separation Statistic.
2. Gini Coefficient (Gini = 2 * ROC_AUC - 1).
3. Brier Score Probability Calibration.
4. Asymmetric Cost-Optimal Decision Threshold (tau* = C_FP / (C_FP + C_FN)).
5. Visualizations: KS Curve, ROC Curve, and Precision-Recall Curve.
"""

from pathlib import Path
from typing import Any, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    auc,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


class MetricsEvaluator:
    """Computes Basel II/III regulatory scorecard metrics and generates diagnostic charts."""

    def __init__(self, cost_fp: float = 300.0, cost_fn: float = 10_000.0):
        self.cost_fp = cost_fp  # Lost margin / interest opportunity cost
        self.cost_fn = cost_fn  # Principal write-off / severe default loss
        self.tau_star = self.cost_fp / (self.cost_fp + self.cost_fn)

    def compute_ks_deciles(self, y_true: np.ndarray, scores: np.ndarray) -> Tuple[pd.DataFrame, float]:
        """Compute 10-decile cumulative separation and the Kolmogorov-Smirnov (KS) statistic."""
        df = pd.DataFrame({"target": y_true, "score": scores})
        # Higher credit score = Lower default risk (Decile 1 = lowest scores / highest risk)
        df["decile"] = pd.qcut(df["score"], q=10, labels=False, duplicates="drop") + 1

        grouped = (
            df.groupby("decile")
            .agg(
                total=("target", "count"),
                bads=("target", lambda y: (y == 1).sum()),
                goods=("target", lambda y: (y == 0).sum()),
                min_score=("score", "min"),
                max_score=("score", "max"),
            )
            .reset_index()
        )

        grouped["bad_rate"] = grouped["bads"] / grouped["total"]
        grouped["pct_bads"] = grouped["bads"] / grouped["bads"].sum()
        grouped["pct_goods"] = grouped["goods"] / grouped["goods"].sum()

        grouped["cum_pct_bads"] = grouped["pct_bads"].cumsum()
        grouped["cum_pct_goods"] = grouped["pct_goods"].cumsum()

        # KS Separation per decile
        grouped["ks_separation"] = np.abs(grouped["cum_pct_bads"] - grouped["cum_pct_goods"])
        ks_statistic = float(grouped["ks_separation"].max())

        return grouped, ks_statistic

    def evaluate_all(self, y_true: np.ndarray, y_prob: np.ndarray, scores: np.ndarray) -> Dict[str, Any]:
        """Compute full regulatory metric suite: KS, Gini, ROC-AUC, Brier, and Asymmetric Confusion Matrix."""
        roc_auc = float(roc_auc_score(y_true, y_prob))
        gini = float(2.0 * roc_auc - 1.0)
        brier = float(brier_score_loss(y_true, y_prob))

        decile_df, ks_stat = self.compute_ks_deciles(y_true, scores)

        # Classifications at asymmetric cost-optimal threshold (tau*)
        y_pred_optimal = (y_prob >= self.tau_star).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred_optimal).ravel()

        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        total_loss = float(fp * self.cost_fp + fn * self.cost_fn)

        results = {
            "roc_auc": round(roc_auc, 4),
            "gini": round(gini, 4),
            "ks_statistic": round(ks_stat * 100.0, 2),  # percentage
            "brier_score": round(brier, 4),
            "cost_optimal_tau": round(self.tau_star, 4),
            "precision_at_tau": round(precision, 4),
            "recall_at_tau": round(recall, 4),
            "total_financial_loss": round(total_loss, 2),
            "decile_table": decile_df,
        }
        return results

    @staticmethod
    def plot_ks_curve(decile_df: pd.DataFrame, ks_value: float, output_path: Path) -> None:
        """Plot and save Kolmogorov-Smirnov (KS) Decile Separation Curve."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
        fig, ax = plt.subplots(figsize=(8, 5), dpi=300)

        deciles = decile_df["decile"]
        cum_bads = decile_df["cum_pct_bads"] * 100.0
        cum_goods = decile_df["cum_pct_goods"] * 100.0

        ax.plot(deciles, cum_bads, marker="o", color="#d9534f", linewidth=2.2, label="Cumulative % Defaults (Bads)")
        ax.plot(
            deciles, cum_goods, marker="s", color="#5cb85c", linewidth=2.2, label="Cumulative % Non-Defaults (Goods)"
        )

        # Find max KS decile
        max_idx = decile_df["ks_separation"].argmax()
        max_decile = decile_df.loc[max_idx, "decile"]
        y_bottom = decile_df.loc[max_idx, "cum_pct_goods"] * 100.0
        y_top = decile_df.loc[max_idx, "cum_pct_bads"] * 100.0

        ax.vlines(
            x=max_decile,
            ymin=min(y_bottom, y_top),
            ymax=max(y_bottom, y_top),
            color="#2e6da4",
            linestyle="--",
            linewidth=2.0,
            label=f"Max KS = {ks_value:.1f}% (Decile {max_decile})",
        )

        ax.set_title(
            f"Kolmogorov-Smirnov (KS) Decile Separation Curve (KS = {ks_value:.1f}%)", fontsize=12, fontweight="bold"
        )
        ax.set_xlabel("Score Decile (1 = Highest Risk, 10 = Lowest Risk)", fontsize=10)
        ax.set_ylabel("Cumulative Percentage (%)", fontsize=10)
        ax.set_xticks(deciles)
        ax.set_ylim(0, 105)
        ax.legend(loc="lower right", frameon=True)
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close(fig)

    @staticmethod
    def plot_roc_curve_with_gini(
        y_true: np.ndarray, y_prob: np.ndarray, gini: float, roc_auc: float, output_path: Path
    ) -> None:
        """Plot and save ROC Curve with annotated Gini coefficient."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(7, 6), dpi=300)

        fpr, tpr, _ = roc_curve(y_true, y_prob)
        ax.plot(
            fpr, tpr, color="#0275d8", linewidth=2.5, label=f"Basel Scorecard (AUC = {roc_auc:.3f}, Gini = {gini:.3f})"
        )
        ax.plot(
            [0, 1],
            [0, 1],
            color="#888888",
            linestyle="--",
            linewidth=1.5,
            label="Random Guess (AUC = 0.50, Gini = 0.00)",
        )

        ax.set_title(f"ROC Curve & Gini Discrimination (Gini = {gini:.3f})", fontsize=12, fontweight="bold")
        ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=10)
        ax.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=10)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.05)
        ax.legend(loc="lower right", frameon=True)
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close(fig)

    @staticmethod
    def plot_pr_curve(y_true: np.ndarray, y_prob: np.ndarray, output_path: Path) -> None:
        """Plot Precision-Recall curve with dynamic prior baseline."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(7, 6), dpi=300)

        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = auc(recall, precision)
        prior_rate = float(np.mean(y_true))

        ax.plot(recall, precision, color="#5bc0de", linewidth=2.5, label=f"Scorecard PR Curve (PR-AUC = {pr_auc:.3f})")
        ax.axhline(y=prior_rate, color="#d9534f", linestyle="--", label=f"Prior Prevalence Baseline ({prior_rate:.1%})")

        ax.set_title(f"Precision-Recall Curve (PR-AUC = {pr_auc:.3f})", fontsize=12, fontweight="bold")
        ax.set_xlabel("Recall (Defaults Captured)", fontsize=10)
        ax.set_ylabel("Precision (True Default Accuracy)", fontsize=10)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.05)
        ax.legend(loc="upper right", frameon=True)
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close(fig)
