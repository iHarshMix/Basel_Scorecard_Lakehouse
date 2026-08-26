"""Regulatory Basel Credit Scorecard Trainer & PDO Scaling Engine.

Implements:
1. Balanced Logistic Regression training on WoE-transformed features.
2. Extraction of Logistic coefficients and Odds Ratios (e^beta).
3. Points to Double the Odds (PDO) scorecard scaling equation:
      Factor = PDO / ln(2)
      Offset = BaseScore + Factor * ln(BaseOdds)
      Score = Offset - Factor * (beta_0 + sum(beta_j * WoE_j))
4. Scorecard lookup points table generation for production interpretability.
"""

from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


class ScorecardTrainer:
    """Trains Logistic Regression Scorecard and converts log-odds into FICO 300-850 scores."""

    def __init__(
        self,
        base_score: float = 600.0,
        base_odds: float = 50.0,  # 50 Good to 1 Bad
        pdo: float = 20.0,  # 20 Points to Double the Odds
        c_penalty: float = 1.0,
    ):
        self.base_score = base_score
        self.base_odds = base_odds
        self.pdo = pdo
        self.c_penalty = c_penalty

        # PDO Calibration Constants: Score = Offset - Factor * ln(Odds_default)
        self.factor = self.pdo / np.log(2.0)
        self.offset = self.base_score - self.factor * np.log(self.base_odds)

        self.model = LogisticRegression(
            penalty="l2",
            C=self.c_penalty,
            class_weight="balanced",
            solver="lbfgs",
            max_iter=1000,
            random_state=42,
        )
        self.feature_names: List[str] = []
        self.coefficients_df: pd.DataFrame = pd.DataFrame()
        self.scorecard_points_table: pd.DataFrame = pd.DataFrame()

    def fit(self, X_woe: pd.DataFrame, y: pd.Series) -> "ScorecardTrainer":
        """Train the Logistic Regression Scorecard on WoE features."""
        self.feature_names = list(X_woe.columns)
        self.model.fit(X_woe, y)

        coefs = self.model.coef_[0]
        intercept = float(self.model.intercept_[0])

        records = [{"feature": "Intercept", "coefficient": intercept, "odds_ratio": float(np.exp(intercept))}]
        for feat, coef in zip(self.feature_names, coefs):
            records.append({"feature": feat, "coefficient": float(coef), "odds_ratio": float(np.exp(coef))})

        self.coefficients_df = pd.DataFrame(records)
        return self

    def predict_proba(self, X_woe: pd.DataFrame) -> np.ndarray:
        """Predict Probability of Default (PD = P(y=1))."""
        return self.model.predict_proba(X_woe[self.feature_names])[:, 1]

    def predict_score(self, X_woe: pd.DataFrame) -> np.ndarray:
        """Convert WoE features into standard FICO Credit Scores (300 - 850) via PDO equation."""
        # Calculate raw log-odds (logit = beta_0 + sum(beta_j * WoE_j))
        log_odds = self.model.decision_function(X_woe[self.feature_names])

        # Score = Offset - Factor * log_odds
        scores = self.offset - self.factor * log_odds

        # Clip scores to regulatory FICO bounds [300, 850]
        scores = np.clip(np.round(scores), 300, 850).astype(int)
        return scores

    def build_points_table(self, woe_maps: Dict) -> pd.DataFrame:
        """Build additive scorecard lookup table with explicit points per bin."""
        points_records = []
        n_features = len(self.feature_names)
        intercept = float(self.model.intercept_[0])

        # Base intercept points distributed equally
        base_points_per_feature = (self.offset - self.factor * intercept) / n_features
        coef_dict = dict(zip(self.feature_names, self.model.coef_[0]))

        for feat_woe, coef in coef_dict.items():
            raw_feat = feat_woe.replace("_woe", "")
            if raw_feat in woe_maps:
                table = woe_maps[raw_feat]["table"]
                for _, row in table.iterrows():
                    woe_val = float(row["woe"])
                    # Bin points contribution
                    bin_points = round(base_points_per_feature - (self.factor * coef * woe_val))
                    points_records.append(
                        {
                            "feature": raw_feat,
                            "bin": str(row["bin"]),
                            "goods": int(row["goods"]),
                            "bads": int(row["bads"]),
                            "woe": round(woe_val, 4),
                            "scorecard_points": int(bin_points),
                        }
                    )

        self.scorecard_points_table = pd.DataFrame(points_records)
        return self.scorecard_points_table

    def get_calibration_info(self) -> Dict[str, float]:
        """Return PDO scaling calibration parameters."""
        return {
            "base_score": self.base_score,
            "base_odds": self.base_odds,
            "pdo": self.pdo,
            "factor": round(self.factor, 4),
            "offset": round(self.offset, 4),
        }
