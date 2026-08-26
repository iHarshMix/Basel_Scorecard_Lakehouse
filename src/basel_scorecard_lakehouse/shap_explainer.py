"""Model Explainability & Adverse Action Reason Code Generator.

Compliant with Federal Reserve SR 11-7 & Equal Credit Opportunity Act (ECOA).
Uses SHAP LinearExplainer to decompose model log-odds and extract the Top-4 risk-driving
features for rejected loan applicants.
"""

import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

# Regulatory Reason Code Templates (ECOA Compliant)
REASON_CODE_MAP = {
    "dti": {
        "code": "RC01",
        "description": "Debt-to-Income (DTI) ratio is excessive relative to requested loan terms.",
    },
    "revol_util": {
        "code": "RC02",
        "description": "Proportion of revolving credit balances to total credit limits is too high.",
    },
    "fico_mid": {
        "code": "RC03",
        "description": "Credit bureau score does not meet minimum risk tier eligibility standards.",
    },
    "inq_last_6mths": {
        "code": "RC04",
        "description": "Number of recent inquiries on credit bureau file indicates credit-seeking behavior.",
    },
    "delinq_2yrs": {
        "code": "RC05",
        "description": "History of past-due payment delinquencies recorded within the last 24 months.",
    },
    "annual_inc": {
        "code": "RC06",
        "description": "Verified annual income is insufficient for total debt service obligations.",
    },
    "open_acc": {
        "code": "RC07",
        "description": "Total number of established active credit lines is insufficient.",
    },
}


class SHAPExplainer:
    """Computes SHAP values and formats Adverse Action Notices for loan rejections."""

    def __init__(self, model_trainer, X_sample: pd.DataFrame):
        self.trainer = model_trainer
        self.feature_names = model_trainer.feature_names
        # SHAP LinearExplainer for Logistic Regression
        self.explainer = shap.LinearExplainer(model_trainer.model, X_sample[self.feature_names])

    def explain_applicant(self, applicant_woe: pd.DataFrame, applicant_raw: pd.Series) -> Dict[str, Any]:
        """Compute SHAP feature attributions and generate top 4 regulatory denial reason codes."""
        X_vec = applicant_woe[self.feature_names]
        shap_values = self.explainer.shap_values(X_vec)[0]

        # In Logistic Regression for Default: Positive SHAP pushes towards Default (Risk Escalation)
        impact_df = pd.DataFrame(
            {
                "feature_woe": self.feature_names,
                "raw_feature": [f.replace("_woe", "") for f in self.feature_names],
                "shap_value": shap_values,
                "abs_shap": np.abs(shap_values),
            }
        ).sort_values(by="shap_value", ascending=False)

        # Select top positive risk drivers
        top_risk_drivers = impact_df.head(4)
        reasons: List[Dict[str, str]] = []

        for _, row in top_risk_drivers.iterrows():
            feat = row["raw_feature"]
            reason_info = REASON_CODE_MAP.get(
                feat,
                {
                    "code": "RC99",
                    "description": f"Credit profile indicator '{feat}' does not satisfy underwriting criteria.",
                },
            )
            reasons.append(
                {
                    "reason_code": reason_info["code"],
                    "feature_name": feat,
                    "attribution_impact": round(float(row["shap_value"]), 4),
                    "statement": reason_info["description"],
                }
            )

        loan_id = int(applicant_raw.get("loan_id", 999999))
        score = int(self.trainer.predict_score(applicant_woe)[0])
        prob_default = float(self.trainer.predict_proba(applicant_woe)[0])

        notice = {
            "application_id": loan_id,
            "decision": "DECLINED",
            "regulatory_framework": "Federal Reserve SR 11-7 / ECOA Notice of Adverse Action",
            "calculated_fico_score": score,
            "predicted_default_probability": round(prob_default, 4),
            "decision_threshold_tau_star": round(self.trainer.base_odds, 4),
            "top_adverse_action_reasons": reasons,
        }
        return notice

    def plot_waterfall(self, applicant_woe: pd.DataFrame, applicant_raw: pd.Series, output_path: Path) -> None:
        """Plot and save SHAP feature attribution waterfall for rejected applicant."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        X_vec = applicant_woe[self.feature_names]
        shap_values = self.explainer.shap_values(X_vec)[0]

        clean_names = [f.replace("_woe", "").upper() for f in self.feature_names]
        indices = np.argsort(shap_values)

        fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
        colors = ["#d9534f" if shap_values[i] > 0 else "#5cb85c" for i in indices]

        ax.barh(np.arange(len(indices)), shap_values[indices], color=colors, height=0.55)
        ax.set_yticks(np.arange(len(indices)))
        ax.set_yticklabels([clean_names[i] for i in indices], fontsize=9)
        ax.set_title(
            f"Adverse Action SHAP Risk Attribution (Applicant #{applicant_raw.get('loan_id', 101)})",
            fontsize=11,
            fontweight="bold",
        )
        ax.set_xlabel("SHAP Value (Log-Odds Impact: Red = Increased Default Risk, Green = Credit Strength)", fontsize=9)
        ax.axvline(x=0, color="#333333", linestyle="--", linewidth=1.0)

        plt.tight_layout()
        plt.savefig(output_path)
        plt.close(fig)

    @staticmethod
    def save_notice_json(notice: Dict[str, Any], output_path: Path) -> None:
        """Save Adverse Action Notice JSON artifact."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(notice, f, indent=2)
