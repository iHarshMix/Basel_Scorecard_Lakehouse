"""Weight of Evidence (WoE) and Information Value (IV) Engine.

Implements:
1. Continuous feature binning with monotonic Bad Rate / WoE enforcement.
2. Information Value (IV) calculation for feature screening.
3. WoE transformation mapping for both training and production inference batches.
"""

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


class WoEIVEngine:
    """Computes monotonic Weight of Evidence (WoE) bins and Information Value (IV)."""

    def __init__(self, n_bins: int = 5, min_bin_pct: float = 0.05):
        self.n_bins = n_bins
        self.min_bin_pct = min_bin_pct
        self.woe_maps: Dict[str, Dict[str, Any]] = {}
        self.iv_summary: pd.DataFrame = pd.DataFrame()

    @staticmethod
    def _calculate_woe_iv_table(df_bin: pd.DataFrame, feature_name: str) -> Tuple[pd.DataFrame, float]:
        """Compute WoE and IV values across all bins for a single feature."""
        total_goods = max(int((df_bin["target"] == 0).sum()), 1)
        total_bads = max(int((df_bin["target"] == 1).sum()), 1)

        grouped = (
            df_bin.groupby("bin", observed=False)
            .agg(
                total=("target", "count"),
                bads=("target", lambda y: (y == 1).sum()),
                goods=("target", lambda y: (y == 0).sum()),
            )
            .reset_index()
        )

        # Smooth zero counts using Laplace epsilon smoothing
        grouped["goods"] = grouped["goods"].clip(lower=0.5)
        grouped["bads"] = grouped["bads"].clip(lower=0.5)

        grouped["pct_goods"] = grouped["goods"] / total_goods
        grouped["pct_bads"] = grouped["bads"] / total_bads

        # WoE = ln(%Goods / %Bads)  (Higher WoE = Lower Default Risk)
        grouped["woe"] = np.log(grouped["pct_goods"] / grouped["pct_bads"])

        # IV contribution = (%Goods - %Bads) * WoE
        grouped["iv_contrib"] = (grouped["pct_goods"] - grouped["pct_bads"]) * grouped["woe"]
        total_iv = float(grouped["iv_contrib"].sum())
        grouped["feature"] = feature_name

        return grouped, total_iv

    def _create_monotonic_bins(self, s: pd.Series, y: pd.Series) -> List[float]:
        """Generate quantile bin edges and iteratively merge non-monotonic adjacent bins."""
        # Initial quantile splits
        quantiles = np.linspace(0, 1, self.n_bins + 1)
        raw_edges = np.percentile(s.dropna(), quantiles * 100)
        edges = sorted(list(set(raw_edges)))

        if len(edges) < 3:
            return [-np.inf, np.inf]

        edges[0] = -np.inf
        edges[-1] = np.inf

        # Monotonicity adjustment: evaluate bad rates per bin and merge if monotonicity is violated
        max_merges = 10
        for _ in range(max_merges):
            bins = pd.cut(s, bins=edges, include_lowest=True)
            bad_rates = y.groupby(bins, observed=False).mean().values

            # If NaN exists in any bin, merge with adjacent
            if np.isnan(bad_rates).any():
                nan_idx = np.where(np.isnan(bad_rates))[0][0]
                if 0 < nan_idx < len(edges) - 1:
                    edges.pop(nan_idx)
                    continue

            # Check for monotonicity (either strictly increasing or strictly decreasing)
            diffs = np.diff(bad_rates)
            is_increasing = np.all(diffs >= -1e-4)
            is_decreasing = np.all(diffs <= 1e-4)

            if is_increasing or is_decreasing or len(edges) <= 3:
                break

            # Find largest non-monotonic reversal and merge
            violating_idx = np.argmin(np.abs(diffs)) + 1
            if 0 < violating_idx < len(edges) - 1:
                edges.pop(violating_idx)
            else:
                break

        return edges

    def fit(self, df: pd.DataFrame, features: List[str], target_col: str = "target") -> pd.DataFrame:
        """Fit WoE bins and calculate Information Value for all candidate features."""
        iv_records = []
        y = df[target_col]

        for feature in features:
            s = df[feature]

            if pd.api.types.is_numeric_dtype(s):
                edges = self._create_monotonic_bins(s, y)
                bins = pd.cut(s, bins=edges, include_lowest=True)
                bin_labels = [f"[{edges[i]:.2f}, {edges[i + 1]:.2f})" for i in range(len(edges) - 1)]
                df_temp = pd.DataFrame({"bin": bins, "target": y})
                table, iv_val = self._calculate_woe_iv_table(df_temp, feature)

                # Store mapping dictionary
                bin_to_woe = dict(zip(table["bin"], table["woe"]))
                self.woe_maps[feature] = {
                    "type": "numeric",
                    "edges": edges,
                    "bin_labels": bin_labels,
                    "bin_to_woe": bin_to_woe,
                    "table": table,
                    "iv": iv_val,
                }
            else:
                # Categorical variable WoE
                df_temp = pd.DataFrame({"bin": s.astype(str), "target": y})
                table, iv_val = self._calculate_woe_iv_table(df_temp, feature)
                bin_to_woe = dict(zip(table["bin"], table["woe"]))
                self.woe_maps[feature] = {
                    "type": "categorical",
                    "bin_to_woe": bin_to_woe,
                    "table": table,
                    "iv": iv_val,
                }

            # Regulatory assessment rating
            if iv_val < 0.02:
                rating = "Unpredictive (<0.02) - DROP"
            elif iv_val < 0.10:
                rating = "Weak (0.02-0.10)"
            elif iv_val <= 0.30:
                rating = "Medium / Strong (0.10-0.30) - PRIME"
            elif iv_val <= 0.50:
                rating = "Very Strong (0.30-0.50)"
            else:
                rating = "Suspiciously High (>0.50) - LEAKAGE CHECK"

            iv_records.append({"feature": feature, "information_value": round(iv_val, 4), "strength_rating": rating})

        self.iv_summary = pd.DataFrame(iv_records).sort_values(by="information_value", ascending=False)
        return self.iv_summary

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply learned WoE mappings to replace raw features with continuous WoE values."""
        df_woe = pd.DataFrame(index=df.index)

        for feature, meta in self.woe_maps.items():
            if feature not in df.columns:
                continue

            if meta["type"] == "numeric":
                edges = meta["edges"]
                bins = pd.cut(df[feature], bins=edges, include_lowest=True)
                woe_series = bins.map(meta["bin_to_woe"]).astype(float)
                # Fill any unmapped edge values with 0.0 (neutral log-odds)
                df_woe[f"{feature}_woe"] = woe_series.fillna(0.0)
            else:
                cat_series = df[feature].astype(str)
                woe_series = cat_series.map(meta["bin_to_woe"]).astype(float)
                df_woe[f"{feature}_woe"] = woe_series.fillna(0.0)

        return df_woe
