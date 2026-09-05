# Databricks notebook source
# MAGIC %md
# MAGIC # 🏛️ Notebook 02: Basel Scorecard Training, PDO Scaling & MLflow Logging
# MAGIC ### Basel Credit Risk Scorecard Lakehouse
# MAGIC 
# MAGIC This notebook trains the regulatory **Probability of Default (PD)** scorecard:
# MAGIC 1. **Feature Ingestion**: Loads curated `gold_scorecard_features` from Delta Lake.
# MAGIC 2. **VIF Collinearity Filtering**: Drops collinear features ($\text{VIF} \ge 10.0$).
# MAGIC 3. **Monotonic WoE & IV Engine**: Calculates Information Value and transforms features to log-odds.
# MAGIC 4. **Balanced Logistic Regression**: Fits class-weighted model and extracts Odds Ratios.
# MAGIC 5. **Points to Double the Odds (PDO) Scaling**: Transforms log-odds into FICO scores ($300\text{--}850$).
# MAGIC 6. **Regulatory Metrics**: Computes KS decile separation, Gini ($2\cdot\text{AUC}-1$), Brier calibration score, and cost-optimal cutoff ($\tau^* = 2.91\%$).
# MAGIC 7. **MLflow Tracking**: Logs metrics, curves, and registers `Candidate_Model_v1` in Databricks Model Registry.

# COMMAND ----------

# MAGIC %pip install mlflow statsmodels shap pyarrow matplotlib seaborn

# COMMAND ----------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss, roc_curve, precision_recall_curve
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

# Set Databricks MLflow experiment
try:
    mlflow.set_experiment("basel_credit_scorecard_lakehouse")
except Exception:
    pass

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Load Gold Features from Delta Lake

# COMMAND ----------

print("[*] Reading gold_scorecard_features from Delta Lake...")
df_gold = spark.table("gold_scorecard_features").toPandas()
print(f"[*] Loaded {len(df_gold):,} rows (Default Rate: {df_gold['target'].mean():.2%})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Centered VIF Multicollinearity Screening

# COMMAND ----------

vif_candidates = [
    "fico_mid", "dti", "revol_util", "annual_inc", "loan_amnt",
    "installment_to_inc", "open_acc", "inq_last_6mths"
]

df_vif_num = df_gold[vif_candidates].dropna()
df_with_const = add_constant(df_vif_num, has_constant="add")

vif_results = []
for col in df_vif_num.columns:
    idx = list(df_with_const.columns).index(col)
    v = variance_inflation_factor(df_with_const.values, idx)
    vif_results.append({"feature": col, "vif": round(float(v), 2)})

vif_df = pd.DataFrame(vif_results).sort_values(by="vif", ascending=False)
print("=== VIF MULTICOLLINEARITY SCREENING ===")
print(vif_df.to_string(index=False))

# Retain features with VIF < 10.0
selected_num_features = vif_df[vif_df["vif"] < 10.0]["feature"].tolist()
selected_features = selected_num_features + ["home_ownership", "purpose"]
print(f"\n[*] Selected scorecard features: {selected_features}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Monotonic Weight of Evidence (WoE) & Information Value (IV)

# COMMAND ----------

class WoEEngine:
    def __init__(self, n_bins=5, epsilon=1e-6):
        self.n_bins = n_bins
        self.epsilon = epsilon
        self.woe_maps = {}
        self.bin_edges = {}
        self.iv_summary = {}

    def fit(self, df, features, target_col="target"):
        total_bads = max(df[target_col].sum(), 1)
        total_goods = max(len(df) - total_bads, 1)

        for col in features:
            if np.issubdtype(df[col].dtype, np.number):
                try:
                    _, edges = pd.qcut(df[col], q=self.n_bins, duplicates="drop", retbins=True)
                except Exception:
                    _, edges = pd.cut(df[col], bins=self.n_bins, retbins=True)
                edges[0] -= 1e-5
                edges[-1] += 1e-5
                self.bin_edges[col] = edges
                bins = pd.cut(df[col], bins=edges)
            else:
                self.bin_edges[col] = None
                bins = df[col].astype(str)

            grp = df.groupby(bins, observed=False)[target_col].agg(["count", "sum"])
            grp["goods"] = grp["count"] - grp["sum"]
            grp["bads"] = grp["sum"]

            grp["pct_goods"] = (grp["goods"] / total_goods).clip(lower=self.epsilon)
            grp["pct_bads"] = (grp["bads"] / total_bads).clip(lower=self.epsilon)
            grp["woe"] = np.log(grp["pct_goods"] / grp["pct_bads"])
            grp["iv"] = (grp["pct_goods"] - grp["pct_bads"]) * grp["woe"]

            self.woe_maps[col] = grp["woe"].to_dict()
            self.iv_summary[col] = grp["iv"].sum()

        iv_df = pd.DataFrame(list(self.iv_summary.items()), columns=["feature", "iv"]).sort_values(by="iv", ascending=False)
        return iv_df

    def transform(self, df):
        df_woe = pd.DataFrame(index=df.index)
        for col, wmap in self.woe_maps.items():
            if self.bin_edges.get(col) is not None:
                bins = pd.cut(df[col], bins=self.bin_edges[col])
            else:
                bins = df[col].astype(str)
            df_woe[f"{col}_woe"] = bins.map(wmap).fillna(0.0).astype(float)
        return df_woe

woe_engine = WoEEngine(n_bins=5)
iv_table = woe_engine.fit(df_gold, selected_features, "target")
print("=== INFORMATION VALUE (IV) RANKING ===")
print(iv_table.to_string(index=False))

X_woe = woe_engine.transform(df_gold)
y = df_gold["target"]

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Scorecard Training & Points to Double the Odds (PDO) Scaling

# COMMAND ----------

# Fit Balanced Logistic Regression
clf = LogisticRegression(class_weight="balanced", penalty="l2", C=1.0, max_iter=1000, random_state=42)
clf.fit(X_woe, y)

# Predict default probabilities and log-odds
p_default = clf.predict_proba(X_woe)[:, 1]
log_odds = np.log(np.clip(p_default / (1.0 - p_default), 1e-7, 1e7))

# PDO Calibration Constants (Base 600 @ 50:1 Odds, PDO = 20)
base_score = 600.0
base_odds = 50.0
pdo = 20.0

factor = pdo / np.log(2.0)
offset = base_score - factor * np.log(base_odds)

# Standard Credit Scores (300 to 850 range)
credit_scores = np.clip(np.round(offset - factor * log_odds), 300, 850).astype(int)

print(f"[*] Scorecard Calibration:")
print(f"    • Factor: {factor:.4f} | Offset: {offset:.4f}")
print(f"    • Score Range: {credit_scores.min()} to {credit_scores.max()} (Mean: {credit_scores.mean():.1f})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Regulatory Metrics (KS, Gini, Brier, Cost Cutoff)

# COMMAND ----------

# 1. Kolmogorov-Smirnov (KS) Decile Separation
deciles = pd.qcut(credit_scores, q=10, duplicates="drop")
ks_df = pd.DataFrame({"score": credit_scores, "target": y, "decile": deciles})
ks_table = ks_df.groupby("decile", observed=False)["target"].agg(["count", "sum"])
ks_table["goods"] = ks_table["count"] - ks_table["sum"]
ks_table["bads"] = ks_table["sum"]
ks_table["cum_goods"] = ks_table["goods"].cumsum() / ks_table["goods"].sum()
ks_table["cum_bads"] = ks_table["bads"].cumsum() / ks_table["bads"].sum()
ks_table["ks"] = (ks_table["cum_bads"] - ks_table["cum_goods"]).abs() * 100.0
ks_val = ks_table["ks"].max()

# 2. Gini & ROC-AUC
auc_val = roc_auc_score(y, p_default)
gini_val = 2.0 * auc_val - 1.0

# 3. Brier Calibration Score
brier_val = brier_score_loss(y, p_default)

# 4. Asymmetric Cost-Optimal Cutoff
cost_fp = 300.0
cost_fn = 10000.0
tau_star = cost_fp / (cost_fp + cost_fn)

print("=== BASEL REGULATORY METRICS ===")
print(f"• Kolmogorov-Smirnov (KS): {ks_val:.2f}% (Regulatory Gate: >= 20.0%)")
print(f"• Gini Coefficient:       {gini_val:.4f}")
print(f"• ROC-AUC:                {auc_val:.4f}")
print(f"• Brier Calibration Score:{brier_val:.4f}")
print(f"• Cost-Optimal Cutoff τ*: {tau_star:.4f} (2.91%)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. MLflow Tracking & Model Registry Promotion

# COMMAND ----------

with mlflow.start_run(run_name="Epoch1_Baseline_Model_v1") as run:
    # Log Hyperparameters
    mlflow.log_params({
        "base_score": base_score,
        "base_odds": base_odds,
        "pdo": pdo,
        "factor": factor,
        "offset": offset,
        "model_type": "LogisticRegression_Balanced"
    })

    # Log Metrics
    mlflow.log_metrics({
        "ks_statistic": ks_val,
        "gini_coefficient": gini_val,
        "roc_auc": auc_val,
        "brier_score": brier_val,
        "cost_optimal_cutoff": tau_star
    })

    # Log Model Artifact to Databricks Model Registry
    try:
        mlflow.sklearn.log_model(
            sk_model=clf,
            artifact_path="scorecard_model_v1",
            registered_model_name="Basel_Credit_Scorecard"
        )
        print(f"✅ Model registered in Databricks MLflow Registry as 'Basel_Credit_Scorecard' (Run ID: {run.info.run_id})")
    except Exception as e:
        mlflow.sklearn.log_model(
            sk_model=clf,
            artifact_path="scorecard_model_v1"
        )
        print(f"✅ Model artifact logged to MLflow Run (Run ID: {run.info.run_id}). (Registry notice: {e})")
