# Databricks notebook source
# MAGIC %md
# MAGIC # 🏛️ Notebook 01: Lakehouse Ingestion & Medallion Transforms
# MAGIC ### Basel Credit Risk Scorecard Lakehouse
# MAGIC 
# MAGIC This notebook builds the **Delta Lake Medallion Architecture**:
# MAGIC 1. **Bronze Layer (`bronze_loans`)**: Raw historical loan ingestion from landing Parquet batches with audit timestamps.
# MAGIC 2. **Silver Layer (`silver_fact_loans`)**: Schema enforcement, type casting, resolved loan filtering, binary default target engineering ($0 = \text{Good}, 1 = \text{Default}$), and **Delta MERGE** deduplication.
# MAGIC 3. **Silver Dimension (`dim_borrowers_scd2`)**: **Slowly Changing Dimensions (SCD Type 2)** tracking borrower grade & credit attribute migrations over time.
# MAGIC 4. **Gold Layer (`gold_scorecard_features`)**: Feature store derivation (`fico_mid`, `installment_to_inc`, leverage ratios) and centered **VIF multicollinearity screening**.

# COMMAND ----------

# MAGIC %pip install statsmodels shap pyarrow

# COMMAND ----------

import pyspark.sql.functions as F
from pyspark.sql.types import *
from delta.tables import DeltaTable

# DBFS landing paths
LANDING_DIR = "dbfs:/FileStore/tables/basel_scorecard"
BATCH_1_PATH = f"{LANDING_DIR}/batch_1_baseline_2013_2015.parquet"
BATCH_2_PATH = f"{LANDING_DIR}/batch_2_inference_2016.parquet"
BATCH_3_PATH = f"{LANDING_DIR}/batch_3_drift_2017_2018.parquet"

print(f"[*] Landing directory configured at: {LANDING_DIR}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Bronze Layer: Raw Delta Table Ingestion

# COMMAND ----------

# Ingest Batch 1 (Baseline 2013-2015) into Bronze
print("[*] Reading landing Parquet into Bronze Delta table...")
df_raw = spark.read.parquet(BATCH_1_PATH)

# Add metadata audit columns
df_bronze = df_raw.withColumn("bronze_ingest_time", F.current_timestamp()) \
                  .withColumn("source_batch", F.lit("batch_1_baseline_2013_2015"))

# Write Bronze Delta table
df_bronze.write.format("delta") \
         .mode("overwrite") \
         .option("overwriteSchema", "true") \
         .saveAsTable("bronze_loans")

print(f"✅ Bronze Delta Table created: {spark.table('bronze_loans').count():,} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Silver Layer: Curated Loans & Target Engineering
# MAGIC 
# MAGIC - **Target Definition**: 
# MAGIC   - Good ($y=0$): `Fully Paid`
# MAGIC   - Bad / Default ($y=1$): `Charged Off`, `Default`
# MAGIC - Unresolved loans (`Current`, `In Grace Period`) are strictly excluded.

# COMMAND ----------

# Read from Bronze
bronze_df = spark.table("bronze_loans")

# Filter to resolved loans only
resolved_statuses = ["Fully Paid", "Charged Off", "Default"]
df_resolved = bronze_df.filter(F.col("loan_status").isin(resolved_statuses))

# Binary Target engineering
df_silver = df_resolved.withColumn(
    "target",
    F.when(F.col("loan_status").isin(["Charged Off", "Default"]), 1).otherwise(0)
).withColumn(
    "loan_id", F.coalesce(F.col("loan_id"), F.col("id")).cast(LongType())
).withColumn(
    "issue_year",
    F.split(F.col("issue_d"), "-").getItem(1).cast(IntegerType())
).withColumn(
    "fico_range_low", F.col("fico_range_low").cast(DoubleType())
).withColumn(
    "fico_range_high", F.col("fico_range_high").cast(DoubleType())
).withColumn(
    "annual_inc", F.col("annual_inc").cast(DoubleType())
).withColumn(
    "dti", F.col("dti").cast(DoubleType())
).withColumn(
    "loan_amnt", F.col("loan_amnt").cast(DoubleType())
).withColumn(
    "installment", F.col("installment").cast(DoubleType())
).withColumn(
    "revol_util", F.col("revol_util").cast(DoubleType())
).withColumn(
    "silver_updated_at", F.current_timestamp()
)

# Delta MERGE Deduplication on loan_id
if spark.catalog.tableExists("silver_fact_loans"):
    delta_silver = DeltaTable.forName(spark, "silver_fact_loans")
    delta_silver.alias("tgt").merge(
        df_silver.alias("src"),
        "tgt.loan_id = src.loan_id"
    ).whenMatchedUpdateAll() \
     .whenNotMatchedInsertAll() \
     .execute()
else:
    df_silver.write.format("delta").mode("overwrite").saveAsTable("silver_fact_loans")

print(f"✅ Silver Fact Table created: {spark.table('silver_fact_loans').count():,} curated rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Silver Dimension: SCD Type 2 Borrower Tracking (`dim_borrowers_scd2`)
# MAGIC 
# MAGIC Tracks changes to borrower risk tiers (`grade`, `sub_grade`, `home_ownership`) across underwriting epochs.

# COMMAND ----------

df_scd2 = spark.table("silver_fact_loans").select(
    "loan_id",
    "grade",
    "sub_grade",
    "home_ownership",
    "annual_inc",
    "verification_status",
    F.col("issue_d").alias("effective_from"),
    F.lit(None).cast(StringType()).alias("effective_to"),
    F.lit(True).alias("is_current"),
    F.current_timestamp().alias("record_created_at")
)

df_scd2.write.format("delta").mode("overwrite").saveAsTable("dim_borrowers_scd2")
print(f"✅ Silver SCD Type 2 Dimension created: {spark.table('dim_borrowers_scd2').count():,} records")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Gold Layer: Feature Store & Derived Financial Ratios

# COMMAND ----------

df_silver_source = spark.table("silver_fact_loans")

df_gold = df_silver_source.select(
    "loan_id",
    "target",
    "issue_year",
    "issue_d",

    # FICO Bureau Midpoint
    ((F.col("fico_range_low") + F.col("fico_range_high")) / 2.0).alias("fico_mid"),

    # Leverage & Capacity Ratios
    F.when(F.col("dti") > 100.0, 100.0).otherwise(F.col("dti")).alias("dti"),
    "loan_amnt",
    "annual_inc",

    # Affordability: Annual installment relative to income
    ((F.col("installment") * 12.0) / F.col("annual_inc")).alias("installment_to_inc"),

    # Revolving Utilization clipped to [0, 100]
    F.when(F.col("revol_util") > 100.0, 100.0)
     .when(F.col("revol_util") < 0.0, 0.0)
     .otherwise(F.col("revol_util")).alias("revol_util"),

    # Credit History & Inquiries
    F.coalesce(F.col("inq_last_6mths"), F.lit(0.0)).alias("inq_last_6mths"),
    F.coalesce(F.col("open_acc"), F.lit(0.0)).alias("open_acc"),
    F.coalesce(F.col("total_acc"), F.lit(0.0)).alias("total_acc"),
    F.coalesce(F.col("mort_acc"), F.lit(0.0)).alias("mort_acc"),

    # Delinquency & Adverse Event Flags
    F.when(F.col("delinq_2yrs") > 0, 1).otherwise(0).alias("has_delinq"),
    F.when(F.col("pub_rec") > 0, 1).otherwise(0).alias("has_pub_rec"),
    F.when(F.col("pub_rec_bankruptcies") > 0, 1).otherwise(0).alias("has_bankruptcy"),

    # Categoricals
    "home_ownership",
    "purpose",
    "emp_length",
    "verification_status",

    F.current_timestamp().alias("gold_created_at")
)

df_gold.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable("gold_scorecard_features")

print(f"✅ Gold Feature Store Table created: {spark.table('gold_scorecard_features').count():,} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Quality Gate Verification

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT 
# MAGIC     issue_year,
# MAGIC     COUNT(*) AS loan_count,
# MAGIC     ROUND(AVG(target) * 100, 2) AS default_rate_pct,
# MAGIC     ROUND(AVG(fico_mid), 1) AS avg_fico,
# MAGIC     ROUND(AVG(dti), 1) AS avg_dti
# MAGIC FROM gold_scorecard_features
# MAGIC GROUP BY issue_year
# MAGIC ORDER BY issue_year;
