"""Feature Engineering and Multicollinearity (VIF) Filter Engine.

Implements:
1. Derived feature calculations (FICO midpoint, Credit Utilization Ratio, Payment-to-Income).
2. Missing value imputation (Median for numeric, Mode for categorical).
3. Variance Inflation Factor (VIF) diagnostics to drop collinear features (VIF >= 10.0).
"""

from typing import List, Tuple

import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor


class FeatureEngineer:
    """Handles cleaning, feature derivation, and VIF multicollinearity screening."""

    def __init__(self, vif_threshold: float = 10.0):
        self.vif_threshold = vif_threshold
        self.numeric_imputers = {}
        self.categorical_imputers = {}
        self.dropped_vif_features: List[str] = []
        self.selected_features: List[str] = []

    def fit_imputers(self, df: pd.DataFrame, numeric_cols: List[str], categorical_cols: List[str]) -> None:
        """Fit median and mode imputers on training data."""
        for col in numeric_cols:
            if col in df.columns:
                self.numeric_imputers[col] = float(df[col].median(skipna=True))

        for col in categorical_cols:
            if col in df.columns:
                mode_vals = df[col].mode(dropna=True)
                self.categorical_imputers[col] = mode_vals.iloc[0] if len(mode_vals) > 0 else "MISSING"

    def transform_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply missing value imputation and compute derived banking features."""
        df_out = df.copy()

        # Apply imputers
        for col, median_val in self.numeric_imputers.items():
            if col in df_out.columns:
                df_out[col] = df_out[col].fillna(median_val)

        for col, mode_val in self.categorical_imputers.items():
            if col in df_out.columns:
                df_out[col] = df_out[col].fillna(mode_val)

        # Derived Feature 1: FICO Midpoint
        if "fico_range_low" in df_out.columns and "fico_range_high" in df_out.columns:
            df_out["fico_mid"] = (df_out["fico_range_low"] + df_out["fico_range_high"]) / 2.0
        elif "fico_range_low" in df_out.columns:
            df_out["fico_mid"] = df_out["fico_range_low"] + 2.0

        # Derived Feature 2: Installment to Monthly Income Ratio
        if "installment" in df_out.columns and "annual_inc" in df_out.columns:
            monthly_inc = np.maximum(df_out["annual_inc"] / 12.0, 1.0)
            df_out["installment_to_inc"] = np.clip((df_out["installment"] / monthly_inc) * 100.0, 0.0, 100.0)

        # Derived Feature 3: Clean Revolving Utilization
        if "revol_util" in df_out.columns:
            df_out["revol_util"] = np.clip(df_out["revol_util"], 0.0, 100.0)

        # Derived Feature 4: Delinquency Flag
        if "delinq_2yrs" in df_out.columns:
            df_out["has_delinq"] = (df_out["delinq_2yrs"] > 0).astype(int)

        return df_out

    def compute_vif(self, df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
        """Compute Variance Inflation Factor (VIF) for all candidate numerical features."""
        from statsmodels.tools.tools import add_constant

        df_numeric = df[features].select_dtypes(include=[np.number]).dropna()
        if len(df_numeric) == 0:
            return pd.DataFrame(columns=["feature", "vif"])

        df_with_const = add_constant(df_numeric, has_constant="add")
        vif_data = []

        for i, col in enumerate(df_numeric.columns):
            try:
                # Column in df_with_const is offset by 1 due to const
                col_idx = list(df_with_const.columns).index(col)
                vif_val = variance_inflation_factor(df_with_const.values, col_idx)
            except Exception:
                vif_val = 1.0
            vif_data.append({"feature": col, "vif": round(float(vif_val), 2)})

        vif_df = pd.DataFrame(vif_data).sort_values(by="vif", ascending=False)
        return vif_df

    def filter_multicollinearity(
        self, df: pd.DataFrame, candidate_features: List[str]
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Iteratively drop highest VIF features exceeding threshold until all VIF < vif_threshold."""
        current_features = list(candidate_features)
        df_clean = df[current_features].select_dtypes(include=[np.number]).copy()

        while True:
            vif_df = self.compute_vif(df_clean, current_features)
            max_vif = vif_df["vif"].max()

            if max_vif >= self.vif_threshold and len(current_features) > 2:
                drop_col = vif_df.iloc[0]["feature"]
                print(f"   [VIF Filter] Dropping '{drop_col}' (VIF = {max_vif:.2f} >= {self.vif_threshold})")
                self.dropped_vif_features.append(drop_col)
                current_features.remove(drop_col)
                df_clean = df_clean[current_features]
            else:
                break

        self.selected_features = current_features
        final_vif_df = self.compute_vif(df_clean, self.selected_features)
        return final_vif_df, self.selected_features
