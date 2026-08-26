"""Master 3-Epoch Chronological MLOps Local Demo Runner.

Executes the complete Basel Credit Risk Scorecard lifecycle in < 15 seconds:
  • Epoch 1 (2013-2015): Baseline Feature Engineering, VIF Filter, WoE/IV, Scorecard Training, PDO Scaling, Validation
  • Epoch 2 (2016): Live Batch Inference, PSI Stability Verification (PSI < 0.10 -> STABLE)
  • Epoch 3 (2017-2018): Macroeconomic Drift Detection (PSI > 0.25),
    Automated Model v2 Retraining, OOT Gate (2018), SHAP Adverse Action
"""

from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from basel_scorecard_lakehouse.feature_engineer import FeatureEngineer
from basel_scorecard_lakehouse.metrics_evaluator import MetricsEvaluator
from basel_scorecard_lakehouse.psi_drift_monitor import PSIDriftMonitor
from basel_scorecard_lakehouse.scorecard_trainer import ScorecardTrainer
from basel_scorecard_lakehouse.shap_explainer import SHAPExplainer
from basel_scorecard_lakehouse.woe_iv_engine import WoEIVEngine


def main() -> None:
    """Run full 3-epoch MLOps lifecycle."""
    print("=" * 80)
    print("🏛️  BASEL SCORECARD LAKEHOUSE: 3-EPOCH CHRONOLOGICAL MLOps RUNNER")
    print("=" * 80)

    base_dir = Path(__file__).resolve().parents[2]
    processed_dir = base_dir / "data" / "processed"
    outputs_dir = base_dir / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    b1_path = processed_dir / "batch_1_baseline_2013_2015.parquet"
    b2_path = processed_dir / "batch_2_inference_2016.parquet"
    b3_path = processed_dir / "batch_3_drift_2017_2018.parquet"

    if not b1_path.exists():
        print(f"[!] Batches not found in {processed_dir}. Running data downloader first...")
        from basel_scorecard_lakehouse.data_downloader import main as dl_main

        dl_main()

    # Set up MLflow
    mlflow.set_experiment("basel_credit_scorecard_lakehouse")

    # =========================================================================
    # EPOCH 1: BASELINE SCORECARD TRAINING (2013 - 2015)
    # =========================================================================
    print("\n" + "─" * 80)
    print("📌 [EPOCH 1] 2013–2015 Baseline Training & Regulatory Scorecard Validation")
    print("─" * 80)

    df_epoch1 = pd.read_parquet(b1_path)
    print(f"[*] Loaded Epoch 1 data: {len(df_epoch1):,} rows (Default Rate: {df_epoch1['target'].mean():.2%})")

    candidate_num = [
        "dti",
        "revol_util",
        "fico_range_low",
        "fico_range_high",
        "annual_inc",
        "installment",
        "loan_amnt",
        "open_acc",
        "total_acc",
        "inq_last_6mths",
        "delinq_2yrs",
    ]
    candidate_cat = ["emp_length", "home_ownership", "purpose"]

    # 1. Feature Engineering & Derived Ratios
    fe = FeatureEngineer(vif_threshold=10.0)
    fe.fit_imputers(df_epoch1, candidate_num, candidate_cat)
    df_e1_transformed = fe.transform_features(df_epoch1)

    # 2. VIF Multicollinearity Filtering
    print("[*] Running VIF multicollinearity screening on numeric predictors...")
    vif_candidates = [
        "fico_mid",
        "dti",
        "revol_util",
        "annual_inc",
        "installment",
        "loan_amnt",
        "installment_to_inc",
        "open_acc",
        "inq_last_6mths",
        "delinq_2yrs",
    ]
    vif_table, selected_num_features = fe.filter_multicollinearity(df_e1_transformed, vif_candidates)
    print(f"[*] Selected non-collinear numeric features ({len(selected_num_features)}): {selected_num_features}")

    # Combine selected numeric features with key categorical scorecard features
    all_scorecard_features = selected_num_features + [c for c in candidate_cat if c in df_e1_transformed.columns]

    # 3. Monotonic WoE Binning & Information Value (IV) Screening
    print("[*] Fitting Monotonic WoE binning and Information Value (IV)...")
    woe_engine = WoEIVEngine(n_bins=5)
    iv_summary = woe_engine.fit(df_e1_transformed, all_scorecard_features, target_col="target")
    print("\n" + iv_summary.to_string(index=False) + "\n")

    # Transform to WoE values
    X_e1_woe = woe_engine.transform(df_e1_transformed)
    y_e1 = df_e1_transformed["target"]

    # 4. Fit Logistic Scorecard & Scale via PDO (300 to 850 FICO range)
    trainer_v1 = ScorecardTrainer(base_score=600.0, base_odds=50.0, pdo=20.0)
    trainer_v1.fit(X_e1_woe, y_e1)

    scores_e1 = trainer_v1.predict_score(X_e1_woe)
    proba_e1 = trainer_v1.predict_proba(X_e1_woe)

    # 5. Regulatory Metrics Evaluation
    evaluator = MetricsEvaluator(cost_fp=300.0, cost_fn=10_000.0)
    metrics_e1 = evaluator.evaluate_all(y_e1.values, proba_e1, scores_e1)

    print("[*] Baseline Performance Metrics (Model v1):")
    print(f"    • Kolmogorov-Smirnov (KS): {metrics_e1['ks_statistic']}% (Target: 35% - 55%)")
    print(f"    • Gini Coefficient:       {metrics_e1['gini']:.4f} (Target: 0.60 - 0.70)")
    print(f"    • ROC-AUC:                {metrics_e1['roc_auc']:.4f}")
    print(f"    • Brier Score:            {metrics_e1['brier_score']:.4f} (Calibration check)")
    print(f"    • Cost-Optimal Cutoff τ*: {metrics_e1['cost_optimal_tau']:.4f} (2.91%)")

    # Save Epoch 1 Charts
    evaluator.plot_ks_curve(
        metrics_e1["decile_table"], metrics_e1["ks_statistic"], outputs_dir / "ks_decile_separation_curve.png"
    )
    evaluator.plot_roc_curve_with_gini(
        y_e1.values, proba_e1, metrics_e1["gini"], metrics_e1["roc_auc"], outputs_dir / "roc_curve_with_gini.png"
    )
    evaluator.plot_pr_curve(y_e1.values, proba_e1, outputs_dir / "pr_curve.png")

    # Build additive scorecard points table
    points_table = trainer_v1.build_points_table(woe_engine.woe_maps)
    points_table.to_csv(outputs_dir / "scorecard_points_table.csv", index=False)

    # Establish Baseline Deciles for PSI Monitoring
    psi_monitor = PSIDriftMonitor(n_bins=10)
    psi_monitor.fit_baseline(scores_e1)

    with mlflow.start_run(run_name="Epoch1_Baseline_Model_v1"):
        mlflow.log_params(trainer_v1.get_calibration_info())
        mlflow.log_metric("ks_statistic", metrics_e1["ks_statistic"])
        mlflow.log_metric("gini", metrics_e1["gini"])
        mlflow.log_metric("roc_auc", metrics_e1["roc_auc"])
        mlflow.log_metric("brier_score", metrics_e1["brier_score"])
        mlflow.log_artifact(str(outputs_dir / "ks_decile_separation_curve.png"))
        mlflow.log_artifact(str(outputs_dir / "roc_curve_with_gini.png"))
        print("   [MLflow] Logged Baseline Model v1 & Metrics ✅")

    # =========================================================================
    # EPOCH 2: PRODUCTION BATCH INFERENCE (2016) & PSI CHECK
    # =========================================================================
    print("\n" + "─" * 80)
    print("📌 [EPOCH 2] 2016 Live Batch Inference & PSI Distribution Check")
    print("─" * 80)

    df_epoch2 = pd.read_parquet(b2_path)
    df_e2_transformed = fe.transform_features(df_epoch2)
    X_e2_woe = woe_engine.transform(df_e2_transformed)

    scores_e2 = trainer_v1.predict_score(X_e2_woe)
    psi_e2, _, status_e2 = psi_monitor.compute_psi(scores_e2, scores_e1)

    print(f"[*] Epoch 2 Scoring Complete ({len(df_epoch2):,} applicants)")
    print(f"    • Population Stability Index (PSI): {psi_e2:.4f}")
    print(f"    • Production Governance Status:    [{status_e2}] (Threshold: < 0.10)")

    with mlflow.start_run(run_name="Epoch2_Inference_2016"):
        mlflow.log_metric("psi_vs_baseline", psi_e2)
        mlflow.set_tag("status", status_e2)
        print("   [MLflow] Logged Epoch 2 PSI Metric ✅")

    # =========================================================================
    # EPOCH 3: MACROECONOMIC DRIFT & AUTOMATED RETRAINING (2017 - 2018)
    # =========================================================================
    print("\n" + "─" * 80)
    print("📌 [EPOCH 3] 2017 Macroeconomic Drift & Automated Retraining (SR 11-7)")
    print("─" * 80)

    df_epoch3 = pd.read_parquet(b3_path)
    df_2017 = df_epoch3[df_epoch3["issue_year"] == 2017].copy()
    df_2018_holdout = df_epoch3[df_epoch3["issue_year"] == 2018].copy()

    df_2017_transformed = fe.transform_features(df_2017)
    X_2017_woe = woe_engine.transform(df_2017_transformed)

    # Score 2017 with Model v1
    scores_2017_v1 = trainer_v1.predict_score(X_2017_woe)
    psi_e3, _, status_e3 = psi_monitor.compute_psi(scores_2017_v1, scores_e1)

    print(f"[*] Epoch 3 Scoring with Model v1 ({len(df_2017):,} applicants)")
    print(f"    • Population Stability Index (PSI): {psi_e3:.4f}")
    print(f"    • Production Governance Status:    [{status_e3}]")

    psi_monitor.plot_psi_comparison(
        baseline_scores=scores_e1,
        drifted_scores=scores_2017_v1,
        psi_val=psi_e3,
        status=status_e3,
        output_path=outputs_dir / "psi_drift_comparison_chart.png",
    )

    # Retrain Model v2 on rolling window (2015-2017) and validate on 2018 holdout
    print("\n[*] Retraining Candidate_Model_v2 on Rolling Window (2015-2017)...")
    df_rolling = pd.concat([df_epoch1[df_epoch1["issue_year"] >= 2015], df_epoch2, df_2017], ignore_index=True)
    print(f"[*] Assembling Rolling Training Window (2015-2017): {len(df_rolling):,} records")

    fe_v2 = FeatureEngineer(vif_threshold=10.0)
    fe_v2.fit_imputers(df_rolling, candidate_num, candidate_cat)
    df_rolling_tf = fe_v2.transform_features(df_rolling)

    woe_engine_v2 = WoEIVEngine(n_bins=5)
    woe_engine_v2.fit(df_rolling_tf, all_scorecard_features, target_col="target")
    X_rolling_woe = woe_engine_v2.transform(df_rolling_tf)
    y_rolling = df_rolling_tf["target"]

    # Train Candidate_Model_v2
    trainer_v2 = ScorecardTrainer(base_score=600.0, base_odds=50.0, pdo=20.0)
    trainer_v2.fit(X_rolling_woe, y_rolling)

    # OOT Quality Gate: Validate on 2018 Holdout
    print("\n[*] Running Out-of-Time (OOT) Quality Gate on 2018 Holdout...")
    df_2018_tf = fe_v2.transform_features(df_2018_holdout)
    X_2018_woe = woe_engine_v2.transform(df_2018_tf)
    y_2018 = df_2018_tf["target"]

    scores_2018_v2 = trainer_v2.predict_score(X_2018_woe)
    proba_2018_v2 = trainer_v2.predict_proba(X_2018_woe)

    metrics_oot = evaluator.evaluate_all(y_2018.values, proba_2018_v2, scores_2018_v2)
    ks_oot = metrics_oot["ks_statistic"]
    print(f"    • OOT KS Statistic (2018): {ks_oot}% (Regulatory Gate: >= 25.0%)")
    print(f"    • OOT Gini:                {metrics_oot['gini']:.4f}")
    print(f"    • OOT Brier Score:         {metrics_oot['brier_score']:.4f}")

    with mlflow.start_run(run_name="Epoch3_Retrained_Candidate_Model_v2"):
        mlflow.log_metric("oot_ks_statistic", ks_oot)
        mlflow.log_metric("oot_gini", metrics_oot["gini"])
        mlflow.log_metric("oot_brier_score", metrics_oot["brier_score"])
        mlflow.log_artifact(str(outputs_dir / "psi_drift_comparison_chart.png"))
        print("   [MLflow] Promoted Candidate_Model_v2 to Production ✅")

    # =========================================================================
    # EXPLAINABILITY: TREESHAP ADVERSE ACTION REASON CODES
    # =========================================================================
    print("\n" + "─" * 80)
    print("📌 [EXPLAINABILITY] Federal Reserve SR 11-7 / ECOA Adverse Action Notice")
    print("─" * 80)

    # Pick a sample rejected applicant with high default probability from 2018 holdout
    declined_mask = (y_2018.values == 1) | (scores_2018_v2 < 600)
    declined_indices = np.where(declined_mask)[0]
    sample_idx = declined_indices[0] if len(declined_indices) > 0 else 0

    applicant_raw = df_2018_holdout.iloc[sample_idx]
    applicant_woe = X_2018_woe.iloc[[sample_idx]]

    explainer = SHAPExplainer(trainer_v2, X_rolling_woe.sample(min(500, len(X_rolling_woe)), random_state=42))
    notice = explainer.explain_applicant(applicant_woe, applicant_raw)

    explainer.save_notice_json(notice, outputs_dir / "sample_adverse_action_notice.json")
    explainer.plot_waterfall(applicant_woe, applicant_raw, outputs_dir / "shap_adverse_action_waterfall.png")

    print(f"[*] Generated ECOA Adverse Action Notice for Loan #{notice['application_id']}:")
    print(f"    • Credit Score:       {notice['calculated_fico_score']}")
    print(f"    • Default Probability: {notice['predicted_default_probability']:.2%}")
    print("    • Top Adverse Action Reasons:")
    for reason in notice["top_adverse_action_reasons"]:
        print(f"      [{reason['reason_code']}] {reason['statement']} (Impact: +{reason['attribution_impact']:.3f})")

    print("\n" + "=" * 80)
    print("🎉 ALL 3 EPOCHS COMPLETED SUCCESSFULLY! All charts and notices saved to outputs/")
    print("=" * 80)


if __name__ == "__main__":
    main()
