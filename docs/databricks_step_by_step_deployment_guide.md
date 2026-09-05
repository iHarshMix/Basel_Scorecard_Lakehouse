# 🚀 Databricks Lakehouse Step-by-Step Deployment & Execution Guide

> **Project:** Basel Scorecard Lakehouse  
> **Platform:** Databricks Community Edition (Free Tier) or Enterprise Databricks  
> **Architecture:** Delta Lake Medallion Architecture (Bronze ➔ Silver ➔ Gold) + MLflow Model Registry + MLOps PSI Drift Monitoring  
> **Target Audience:** Step-by-step user manual for running and reproducing the entire lakehouse in the cloud.

---

## 📑 Quick Navigation

1. [Step 1: Set Up Databricks Community Edition (Free)](#step-1-set-up-databricks-community-edition-free)
2. [Step 2: Create and Configure Your Spark Compute Cluster](#step-2-create-and-configure-your-spark-compute-cluster)
3. [Step 3: Clone Your GitHub Repository into Databricks Repos](#step-3-clone-your-github-repository-into-databricks-repos)
4. [Step 4: Upload the 3 Chronological Parquet Batches to DBFS](#step-4-upload-the-3-chronological-parquet-batches-to-dbfs)
5. [Step 5: Install Python Dependencies on the Cluster](#step-5-install-python-dependencies-on-the-cluster)
6. [Step 6: Run Notebook 01 — Lakehouse Ingestion & Medallion Transforms](#step-6-run-notebook-01--lakehouse-ingestion--medallion-transforms)
7. [Step 7: Run Notebook 02 — Scorecard Training, PDO Scaling & MLflow Logging](#step-7-run-notebook-02--scorecard-training-pdo-scaling--mlflow-logging)
8. [Step 8: Run Notebook 03 — Batch Inference, PSI Drift & SHAP Adverse Action](#step-8-run-notebook-03--batch-inference-psi-drift--shap-adverse-action)
9. [Step 9: Verify MLflow Experiments & Model Registry](#step-9-verify-mlflow-experiments--model-registry)
10. [Step 10: Databricks Community Edition Gotchas & Best Practices](#step-10-databricks-community-edition-gotchas--best-practices)

---

## Step 1: Set Up Databricks Community Edition (Free)

Databricks provides a **100% free Community Edition** that requires **no credit card** and provides 15.3 GB RAM of Apache Spark compute.

1. Go to: **[https://community.cloud.databricks.com/](https://community.cloud.databricks.com/)**
2. Click **Sign Up** (or select **Community Edition** link on the pricing page).
3. Fill in your Name, Email, Company (enter "Personal" or "Student"), and Title.
4. Click **Continue** and choose **Get started with Community Edition** (do NOT pick AWS/Azure/GCP trial, as those require cloud billing accounts).
5. Open your email inbox, verify your email, set a password, and log in to the Databricks dashboard.

---

## Step 2: Create and Configure Your Spark Compute Cluster

Your Databricks environment needs an active compute engine (Apache Spark) to run Python and SQL queries.

```
  ┌────────────────────────────────────────────────────────────────────────────────────────┐
  │                         CREATE COMPUTE CLUSTER SETTINGS                                │
  ├──────────────────────────────┬─────────────────────────────────────────────────────────┤
  │ Cluster Name                 │ Basel-Scorecard-Cluster                                 │
  │ Databricks Runtime Version   │ 14.3 LTS (Apache Spark 3.5.0, Scala 2.12) or latest LTS │
  │ Worker Type                  │ Community Optimized (15.3 GB Memory, 2 Cores)           │
  └──────────────────────────────┴─────────────────────────────────────────────────────────┘
```

### Exact UI Clicks:
1. In the left-hand navigation sidebar, click on **Compute** (🖥️ icon).
2. Click the blue **Create Compute** button (top right).
3. Set **Cluster Name**: `Basel-Scorecard-Cluster`
4. Set **Databricks Runtime Version**: Select **14.3 LTS** (or the latest LTS version available).
5. Click **Create Cluster**.
6. Wait 2–3 minutes until the gray spinner changes to a green solid circle (● **Running**).

> [!NOTE]
> **Community Edition Rule:** The cluster automatically terminates after **2 hours of idle time**. If it shuts down, simply click **Compute ➔ Start** to turn it back on. Your code, tables, and DBFS files will remain intact.

---

## Step 3: Clone Your GitHub Repository into Databricks Repos

Databricks has native Git integration, allowing you to pull your code and notebooks directly from GitHub with one click.

### Exact UI Clicks:
1. In the left navigation sidebar, click **Workspace** ➔ **Repos** (or **Git Folders**).
2. Click the **Add Repo** (or **Create Git Folder**) button.
3. In the dialog:
   * **Git repository URL**: `https://github.com/iHarshMix/Basel_Scorecard_Lakehouse.git`
   * **Git provider**: `GitHub`
   * **Repository name**: `Basel_Scorecard_Lakehouse`
4. Click **Create Repo**.
5. The entire project folder will immediately appear inside your Databricks Workspace!

> [!TIP]
> If your repository is private, Databricks will prompt you for a **GitHub Personal Access Token (PAT)**:
> - On GitHub: `Settings` ➔ `Developer Settings` ➔ `Personal access tokens (classic)` ➔ `Generate new token` ➔ Check `repo` scope ➔ Copy token.
> - On Databricks: Paste your GitHub username and the PAT token.

---

## Step 4: Upload the 3 Chronological Parquet Batches to DBFS

The data pipeline consumes three chronological Parquet files that we generated from the real LendingClub dataset:
- `batch_1_baseline_2013_2015.parquet` (92,350 records, 3.11 MB)
- `batch_2_inference_2016.parquet` (37,391 records, 1.33 MB)
- `batch_3_drift_2017_2018.parquet` (30,258 records, 1.10 MB)

These files are located on your local machine at:
`/home/harsh/MLOPs/Basel_Scorecard_Lakehouse/data/processed/`

### Option A: Upload via Databricks UI (Recommended & Easiest)
1. In the left sidebar, click **Catalog** (or **Data**).
2. Click on **DBFS** (Databricks File System) tab.
   *(If DBFS is not visible, go to Settings ➔ Admin Settings ➔ Workspace Settings ➔ Enable "DBFS File Browser" ➔ Refresh page).*
3. Navigate to the path: `FileStore/` (or create folder `FileStore/tables/basel_scorecard/`).
4. Click **Upload** (or drag and drop) and select all 3 `.parquet` files from your local folder:
   - `data/processed/batch_1_baseline_2013_2015.parquet`
   - `data/processed/batch_2_inference_2016.parquet`
   - `data/processed/batch_3_drift_2017_2018.parquet`
5. Once uploaded, your DBFS target paths will be:
   - `dbfs:/FileStore/tables/basel_scorecard/batch_1_baseline_2013_2015.parquet`
   - `dbfs:/FileStore/tables/basel_scorecard/batch_2_inference_2016.parquet`
   - `dbfs:/FileStore/tables/basel_scorecard/batch_3_drift_2017_2018.parquet`

### Option B: Download Directly inside a Databricks Notebook
Alternatively, you can run this Python snippet in cell 1 of your Databricks notebook to create the landing directory:
```python
dbutils.fs.mkdirs("dbfs:/FileStore/tables/basel_scorecard/")
display(dbutils.fs.ls("dbfs:/FileStore/tables/basel_scorecard/"))
```

---

## Step 5: Install Python Dependencies on the Cluster

Databricks runtimes include PySpark, Pandas, and Scikit-Learn by default, but we need additional risk modeling packages (`mlflow`, `shap`, `statsmodels`).

### Exact UI Clicks:
1. Click **Compute** in the left sidebar.
2. Click on your active cluster: `Basel-Scorecard-Cluster`.
3. Click on the **Libraries** tab.
4. Click **Install New**.
5. Select **PyPI** and install the following packages (one by one, or comma-separated):
   * `shap`
   * `statsmodels`
   * `pyarrow`
6. Click **Install**.
7. Status will display green: `Installed`.

*(Note: Each notebook also contains an automated `%pip install shap statsmodels` header cell so you don't even have to worry about manual installation!)*

---

## Step 6: Run Notebook 01 — Lakehouse Ingestion & Medallion Transforms

📁 **Notebook Path:** `databricks_notebooks/01_Lakehouse_Ingestion_and_Transforms.py`

```
  RAW PARQUET LANDING              BRONZE DELTA                    SILVER DELTA                    GOLD FEATURE STORE
 ┌─────────────────────┐          ┌──────────────┐                ┌──────────────┐                ┌──────────────────┐
 │ batch_1_baseline    │ ───────► │ bronze_loans │ ─────────────► │ silver_loans │ ─────────────► │ gold_scorecard   │
 │ batch_2_inference   │          │ (Raw Ingest) │ (Cleaned, Type │ (Deduplicated│  (VIF Filter,  │ _features        │
 │ batch_3_drift       │          │              │  Casting, Dedup│  SCD Type 2) │   WoE Binned)  │                  │
 └─────────────────────┘          └──────────────┘                └──────────────┘                └──────────────────┘
```

### How to Run:
1. In your Databricks workspace, open **Repos** ➔ `Basel_Scorecard_Lakehouse` ➔ `databricks_notebooks` ➔ `01_Lakehouse_Ingestion_and_Transforms`.
2. Attach the notebook to your running cluster: `Basel-Scorecard-Cluster` (top left dropdown).
3. Click **Run All** (top right).

### What this Notebook Accomplishes:
- **Bronze Layer (`bronze_loans`):** Reads the raw landing Parquet files and saves them as raw Delta Lake tables with ingest timestamps.
- **Silver Layer (`silver_fact_loans`):** Cleans loan records, enforces schema constraints, casts datatypes, drops unresolved loans, engineers the binary target ($0=\text{Paid}, 1=\text{Default}$), and performs **Delta MERGE** deduplication.
- **Silver Dimension (`dim_borrowers_scd2`):** Implements **Slowly Changing Dimensions (SCD Type 2)** tracking borrower grade migrations over time with `valid_from`, `valid_to`, and `is_current` flags.
- **Gold Layer (`gold_scorecard_features`):** Computes derived banking ratios (`installment_to_inc`, `fico_mid`, revolving debt utilization), executes centered **Variance Inflation Factor (VIF)** screening, and saves the final production-ready Feature Store table.

### Verification Cell:
Run this SQL cell at the end of the notebook:
```sql
%sql
SHOW TABLES IN default;
SELECT COUNT(*) AS total_gold_features FROM gold_scorecard_features;
```
Expected output: ~92,350 baseline records ready for training.

---

## Step 7: Run Notebook 02 — Scorecard Training, PDO Scaling & MLflow Logging

📁 **Notebook Path:** `databricks_notebooks/02_Scorecard_Training_and_Metrics.py`

### How to Run:
1. Open `databricks_notebooks` ➔ `02_Scorecard_Training_and_Metrics`.
2. Ensure cluster is attached, then click **Run All**.

### What this Notebook Accomplishes:
1. **Feature Ingestion:** Loads `gold_scorecard_features` from Delta Lake.
2. **WoE & Information Value Engine:**
   - Fits monotonic quantile binning.
   - Calculates Information Value ($\text{IV}$) for each predictor (`fico_mid`, `dti`, `loan_amnt`, `annual_inc`, etc.).
   - Drops variables with $\text{IV} < 0.02$.
3. **Logistic Scorecard Modeling:**
   - Trains class-balanced Logistic Regression on WoE features.
   - Extracts Odds Ratios ($e^{\beta_j}$).
4. **PDO Scorecard Scaling:**
   - Scales model log-odds into standard **FICO credit scores (300 to 850)** using:
     $$\text{Score} = 487.12 - 28.854 \times \text{Logit}$$
   - Exports the additive scorecard points lookup table.
5. **Regulatory Metrics Computation:**
   - **KS Separation Statistic:** Evaluates 10-decile CDF separation ($21.65\%$).
   - **Gini Coefficient:** Computes $2 \cdot \text{AUC} - 1 = 0.3047$.
   - **Brier Calibration Score:** Verifies calibration for Basel capital adequacy ($0.2327$).
   - **Cost-Optimal Cutoff ($\tau^*$):** Computes financial threshold $\tau^* = 2.91\%$.
6. **MLflow Tracking & Governance:**
   - Logs all parameters, metrics, KS curve, and ROC curves to the active MLflow Experiment.
   - Registers the model as **`Candidate_Model_v1`** in the Databricks Model Registry.

---

## Step 8: Run Notebook 03 — Batch Inference, PSI Drift & SHAP Adverse Action

📁 **Notebook Path:** `databricks_notebooks/03_Inference_and_PSI_Drift_Monitoring.py`

### How to Run:
1. Open `databricks_notebooks` ➔ `03_Inference_and_PSI_Drift_Monitoring`.
2. Click **Run All**.

### What this Notebook Accomplishes:
1. **Epoch 2 (2016 Live Batch Inference):**
   - Scores 37,391 live 2016 applicants using `Model_v1`.
   - Computes **Population Stability Index (PSI)** against the 2013–2015 baseline:
     $$\text{PSI} = 0.0075 \implies \text{STATUS: [STABLE]} \quad (\text{Threshold} < 0.10)$$
2. **Epoch 3 (2017 Macroeconomic Shock & Drift Alert):**
   - Scores 2017 applicants.
   - Analyzes the real-world **2017 subprime default surge** (defaults jumped from 15.6% to 26.6%).
   - Detects distribution shift and plots the PSI decile comparison curve.
3. **Automated MLOps Retraining Engine:**
   - Assembles a rolling 2015–2017 training window (107,014 records).
   - Retrains **`Candidate_Model_v2`**.
4. **Out-of-Time (OOT) Quality Gate (2018 Holdout Batch):**
   - Evaluates Model v2 against the 2018 holdout cohort.
   - Validates that $\text{KS}_{\text{OOT}} = 22.86\% \ge 20.0\%$ and $\text{Gini} = 0.3262$.
   - Automatically promotes **`Candidate_Model_v2`** to **Production** in the MLflow Model Registry.
5. **Fair Lending & Adverse Action Explainability (SR 11-7 / ECOA):**
   - Applies **TreeSHAP / LinearExplainer** to decompose predicted default log-odds for declined applicants.
   - Automatically extracts and formats the **Top-4 Adverse Action Reason Codes** (e.g. `RC01`: Excessive DTI, `RC99`: Home Ownership, `RC06`: Low Income, `RC03`: Low FICO).
   - Saves the formal audit JSON notice.

---

## Step 9: Verify MLflow Experiments & Model Registry

Once the notebooks finish executing, you can inspect your experiments directly in the Databricks UI:

1. In the left sidebar, click **Experiments** (🧪 icon).
2. Click on the experiment: `basel_credit_scorecard_lakehouse`.
3. You will see three chronological runs:
   - `Epoch1_Baseline_Model_v1`
   - `Epoch2_Inference_2016`
   - `Epoch3_Retrained_Candidate_Model_v2`
4. Click on any run to inspect:
   - Logged metrics (`ks_statistic`, `gini`, `roc_auc`, `brier_score`, `psi_vs_baseline`).
   - Saved artifacts (KS curve plot, ROC curve plot, PSI drift comparison, and SHAP waterfall chart).
5. In the left sidebar, click **Models** (📦 icon) to see `Candidate_Model_v1` and the promoted `Candidate_Model_v2` in the Model Registry.

---

## Step 10: Databricks Community Edition Gotchas & Best Practices

| Challenge / Scenario | Cause in Community Edition | Solution / Best Practice |
| :--- | :--- | :--- |
| **Cluster Inactivity Timeout** | Community Edition terminates clusters after 120 min of idle time. | Go to `Compute`, select your cluster, and click `Start`. Delta tables and files in DBFS are permanent and will not be lost. |
| **Out-of-Memory (OOM)** | Community Edition has 15.3 GB RAM. | Our pipeline uses stratified sampling (~160k loans) and optimized Delta parquet formats, taking only ~15 seconds to train and using <1.5 GB RAM. |
| **File Paths: `/dbfs/` vs `dbfs:/`** | PySpark APIs expect `dbfs:/path`, whereas local Python tools (`open`, `os`) expect `/dbfs/path`. | In PySpark use: `spark.read.parquet("dbfs:/FileStore/...")`. In Python use: `pd.read_parquet("/dbfs/FileStore/...")`. |
| **Missing DBFS Menu in UI** | The DBFS file browser is hidden by default in newer Databricks workspaces. | Go to `Settings` ➔ `Admin Settings` ➔ `Workspace Settings` ➔ Toggle **DBFS File Browser** to **ON** ➔ Refresh page. |
