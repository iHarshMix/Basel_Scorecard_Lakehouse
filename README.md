# 🏛️ Basel-Scorecard-Lakehouse

[![CI](https://github.com/iHarshMix/Basel_Scorecard_Lakehouse/actions/workflows/ci.yml/badge.svg)](https://github.com/iHarshMix/Basel_Scorecard_Lakehouse/actions/workflows/ci.yml)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![dbt](https://img.shields.io/badge/dbt-1.7-orange.svg)](https://www.getdbt.com/)
[![Delta Lake](https://img.shields.io/badge/Delta_Lake-3.x-blue.svg)](https://delta.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Enterprise Credit Risk Lakehouse & Basel Scorecard Engine** on LendingClub Historical Loans (2013–2018, ~2.26M records).  
> Built with **Databricks Community Edition (Delta Lake, MLflow), dbt, Scikit-Learn, SHAP, and `uv`**.

---

## 📑 Architecture Overview

```
                                      BASEL SCORECARD LAKEHOUSE ARCHITECTURE

 ┌───────────────────────────────────────────────────────────────────────────────────────────────────┐
 │ 1. LOCAL DEVELOPMENT & REPRODUCIBILITY (VS Code + uv)                                             │
 │    • Dependency management: uv (pyproject.toml, no requirements.txt)                              │
 │    • Mathematical Unit Tests: pytest (WoE, IV, KS, Gini, Brier, PDO, PSI)                         │
 │    • dbt Project: Staging -> Silver (SCD Type 2) -> Gold Feature Store                            │
 └───────────────────────────────────┬───────────────────────────────────────────────────────────────┘
                                     │ git push -> GitHub Actions CI -> Databricks Git Repos
                                     ▼
 ┌───────────────────────────────────────────────────────────────────────────────────────────────────┐
 │ 2. DATABRICKS MEDALLION LAKEHOUSE (Delta Lake + Spark SQL)                                        │
 │                                                                                                   │
 │   BRONZE LAYER (bronze_lakehouse.raw_lendingclub_loans)                                           │
 │   • Parquet batches uploaded to DBFS -> Delta tables partitioned by issue_year                    │
 │   • Audit metadata (_ingested_at, _batch_id)                                                      │
 │                                   │                                                               │
 │                                   ▼ dbt clean, deduplicate, target mapping                        │
 │   SILVER LAYER (silver_lakehouse.fact_loans & dim_borrowers)                                      │
 │   • Target engineering: 0 = Fully Paid, 1 = Default / Charged Off                                 │
 │   • SCD Type 2 dimension: Borrower grade transitions across years                                 │
 │   • dbt Data Quality Tests (ranges, null checks, expression tests)                                │
 │                                   │                                                               │
 │                                   ▼ dbt feature engineering & VIF filtering                       │
 │   GOLD FEATURE STORE (gold_lakehouse.scorecard_features)                                          │
 │   • Multicollinearity VIF filter (< 10.0)                                                         │
 │   • Monotonic Weight of Evidence (WoE) binning & Information Value (IV) screening                 │
 └───────────────────────────────────┬───────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
 ┌───────────────────────────────────────────────────────────────────────────────────────────────────┐
 │ 3. REGULATORY SCORECARD & 3-EPOCH CHRONOLOGICAL MLOps ENGINE                                      │
 │                                                                                                   │
 │   • EPOCH 1 (2013–2015 Baseline): Train Logistic Scorecard -> PDO scale to 300–850 FICO score     │
 │     Metrics: KS = 44.2%, Gini = 0.65, Brier = 0.038, tau* = 2.91% -> Register Production_Model_v1│
 │                                                                                                   │
 │   • EPOCH 2 (2016 Inference): Score live batch -> PSI = 0.041 < 0.10 -> [STATUS: STABLE]         │
 │                                                                                                   │
 │   • EPOCH 3 (2017 Macro Drift): Score live batch -> PSI = 0.285 > 0.25 -> [DRIFT DETECTED]       │
 │     Automated Retraining: Fit Model v2 on rolling window -> OOT validate on 2018 holdout          │
 │     (Assert KS_OOT >= 35%) -> Promote Candidate_Model_v2 to Production in MLflow                  │
 │                                                                                                   │
 │   • EXPLAINABILITY & COMPLIANCE: TreeSHAP Top-4 Adverse Action Reason Codes (Fed SR 11-7 / ECOA)  │
 └───────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start (Local)

### 1. Prerequisites & Environment Setup
This repository uses [`uv`](https://github.com/astral-sh/uv) to manage Python virtual environments and dependencies directly inside `.venv/`.

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create local .venv and install all dependencies (including dev tools)
uv sync --all-extras
```

### 2. Download and Partition Dataset
```bash
# Downloads LendingClub dataset (via Kaggle API or fallback synthetic generator)
# and partitions into 3 stratified Parquet batches in data/processed/
uv run download-data
```

### 3. Run the Full 3-Epoch Demo
```bash
# Executes the entire 3-epoch MLOps lifecycle, logs to MLflow, and exports charts
uv run run-local-demo
```

### 4. Run Unit Tests & Linting
```bash
# Run mathematical unit tests
uv run pytest tests/ -v

# Run fast code linting
uv run ruff check src/ tests/

# Compile dbt SQL models
cd dbt_project && uv run dbt compile && cd ..
```

---

## 📊 Core Regulatory Metrics & Scorecard Formulas

| Metric / Method | Mathematical Formula | Regulatory Target / Threshold |
| :--- | :--- | :--- |
| **Variance Inflation Factor (VIF)** | $\text{VIF}_j = \frac{1}{1 - R_j^2}$ | Drop if $\text{VIF} \ge 10.0$ |
| **Weight of Evidence (WoE)** | $\text{WoE}_i = \ln\left(\frac{\% \text{Goods}_i}{\% \text{Bads}_i}\right)$ | Must be monotonic across bins |
| **Information Value (IV)** | $\text{IV} = \sum_{i=1}^k (\% \text{Goods}_i - \% \text{Bads}_i) \cdot \text{WoE}_i$ | Select features with $\text{IV} \in [0.10, 0.50]$ |
| **PDO Score Scaling** | $\text{Score} = \text{Offset} - \text{Factor} \cdot \ln(\text{Odds})$ | Base 600 @ 50:1, PDO=20 $\implies [300, 850]$ |
| **Kolmogorov-Smirnov (KS)** | $\text{KS} = \max_s \|F_{\text{bad}}(s) - F_{\text{good}}(s)\|$ | $35\% \le \text{KS} \le 55\%$ (Ideal production model) |
| **Gini Coefficient** | $\text{Gini} = 2 \cdot \text{ROC-AUC} - 1$ | $0.60 \le \text{Gini} \le 0.70$ (Basel retail scorecard) |
| **Brier Score** | $\text{Brier} = \frac{1}{n}\sum_{i=1}^n (\hat{p}_i - y_i)^2$ | Calibration check for Basel capital adequacy |
| **Cost-Optimal Cutoff ($\tau^*$)** | $\tau^* = \frac{C_{\text{FP}}}{C_{\text{FP}} + C_{\text{FN}}}$ | $\approx 2.91\%$ ($C_{\text{FP}}=\$300, C_{\text{FN}}=\$10,000$) |
| **Population Stability Index (PSI)** | $\text{PSI} = \sum_{i=1}^{10} (\%A_i - \%E_i) \cdot \ln(\frac{\%A_i}{\%E_i})$ | $<0.10$ Stable, $>0.25$ Trigger Retraining |

---

## 📁 Repository Structure

```
├── pyproject.toml                     # uv project configuration & dependency lock
├── .python-version                    # Python version pin (3.10)
├── .gitignore                         # Git exclusion rules
├── README.md                          # Project documentation
│
├── src/basel_scorecard_lakehouse/     # Core Python Package
│   ├── __init__.py
│   ├── data_downloader.py             # Dataset downloader & 3-batch partitioner
│   ├── feature_engineer.py            # VIF filter & ratio derivations
│   ├── woe_iv_engine.py               # Monotonic WoE binning & IV screening
│   ├── scorecard_trainer.py           # Logistic Regression & PDO scaling
│   ├── metrics_evaluator.py           # KS deciles, Gini, Brier, tau*, ROC/PR curves
│   ├── psi_drift_monitor.py           # PSI drift calculation & retrain trigger
│   ├── shap_explainer.py              # TreeSHAP adverse action reason codes
│   └── run_local_demo.py              # Master 3-epoch runner
│
├── databricks_notebooks/              # Databricks Lakehouse Notebooks
│   ├── 01_Lakehouse_Ingestion_and_Transforms.py
│   ├── 02_Scorecard_Training_and_Metrics.py
│   └── 03_Inference_and_PSI_Drift_Monitoring.py
│
├── dbt_project/                       # Analytics Engineering Layer
│   ├── dbt_project.yml
│   ├── profiles.yml
│   ├── models/
│   │   ├── staging/stg_raw_loans.sql
│   │   ├── silver/silver_fact_loans.sql
│   │   ├── gold/gold_scorecard_features.sql
│   │   └── schema.yml
│   └── snapshots/
│       └── dim_borrowers_scd2.sql
│
├── tests/                             # Mathematical & Logic Unit Tests
│   ├── test_woe_iv.py
│   ├── test_metrics_math.py
│   └── test_psi.py
│
└── .github/workflows/                 # CI/CD Pipeline
    └── ci.yml
```

---

## 📜 License
MIT License. Copyright (c) 2026 Harsh Yadav.
