"""Dataset Downloader, Cleaner, and 3-Batch Chronological Partitioner.

This module handles:
1. Downloading the LendingClub loans dataset via Kaggle API or local CSV fallback.
2. Generating realistic synthetic LendingClub consumer loan data if no Kaggle key is found.
3. Filtering unresolved loans and engineering the binary target (0 = Repaid, 1 = Default).
4. Stratified temporal sampling into 3 Parquet batches (2013-2015, 2016, 2017-2018).
"""

from pathlib import Path

import numpy as np
import pandas as pd

# Schema definition matching real LendingClub fields used in the scorecard
LENDING_CLUB_COLUMNS = [
    "loan_id",
    "loan_amnt",
    "funded_amnt",
    "term",
    "int_rate",
    "installment",
    "grade",
    "sub_grade",
    "emp_length",
    "home_ownership",
    "annual_inc",
    "verification_status",
    "issue_d",
    "loan_status",
    "purpose",
    "dti",
    "delinq_2yrs",
    "fico_range_low",
    "fico_range_high",
    "inq_last_6mths",
    "open_acc",
    "pub_rec",
    "revol_bal",
    "revol_util",
    "total_acc",
    "mort_acc",
    "pub_rec_bankruptcies",
]

GOOD_STATUSES = ["Fully Paid"]
BAD_STATUSES = ["Charged Off", "Default", "Late (31-120 days)"]
EXCLUDE_STATUSES = ["Current", "In Grace Period", "Late (16-30 days)", "Issued"]


def generate_realistic_lendingclub_data(n_samples: int = 160_000, random_state: int = 42) -> pd.DataFrame:
    """Generate realistic LendingClub loan records across 2013-2018 for local Lakehouse execution."""
    rng = np.random.default_rng(random_state)
    print(f"[*] Generating high-fidelity synthetic LendingClub dataset ({n_samples:,} records across 2013-2018)...")

    # Sample issue dates across 2013-2018 with increasing volume
    years = rng.choice([2013, 2014, 2015, 2016, 2017, 2018], size=n_samples, p=[0.12, 0.18, 0.20, 0.22, 0.16, 0.12])
    months = rng.choice(
        ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], size=n_samples
    )
    issue_d = [f"{m}-{y}" for m, y in zip(months, years)]

    # Financial and Credit features
    annual_inc = np.exp(rng.normal(11.0, 0.55, size=n_samples))  # Log-normal income ($25k - $250k)
    annual_inc = np.clip(annual_inc, 15_000, 450_000)

    loan_amnt = rng.choice(np.arange(1000, 40001, 500), size=n_samples)
    funded_amnt = loan_amnt.copy()
    term = rng.choice([" 36 months", " 60 months"], size=n_samples, p=[0.72, 0.28])

    # FICO score distribution (Mean ~695, Std ~35)
    fico_low = rng.normal(695, 35, size=n_samples).astype(int)
    fico_low = np.clip(fico_low // 5 * 5, 620, 845)
    fico_high = fico_low + 4

    # DTI (Debt to Income ratio)
    dti = rng.gamma(shape=3.5, scale=4.8, size=n_samples)
    dti = np.clip(dti, 0.0, 45.0)

    # Revolving credit utilization (%)
    revol_util = rng.beta(2.5, 2.5, size=n_samples) * 100.0
    revol_bal = np.clip(rng.exponential(12_000, size=n_samples), 0, 150_000)

    # Credit line depth & inquiries
    open_acc = rng.poisson(11, size=n_samples).clip(1, 45)
    total_acc = open_acc + rng.poisson(12, size=n_samples).clip(1, 60)
    inq_last_6mths = rng.choice([0, 1, 2, 3, 4], size=n_samples, p=[0.55, 0.26, 0.12, 0.05, 0.02])
    delinq_2yrs = rng.choice([0, 1, 2, 3], size=n_samples, p=[0.82, 0.12, 0.04, 0.02])
    pub_rec = rng.choice([0, 1, 2], size=n_samples, p=[0.87, 0.11, 0.02])
    pub_rec_bankruptcies = (pub_rec > 0).astype(int) * rng.choice([0, 1], size=n_samples, p=[0.3, 0.7])
    mort_acc = rng.poisson(1.5, size=n_samples).clip(0, 10)

    # Grades, subgrades, and interest rates linked to FICO and DTI
    risk_latent = (
        -0.035 * (fico_low - 690)
        + 0.075 * (dti - 18)
        + 0.025 * (revol_util - 50)
        + 0.40 * inq_last_6mths
        + 0.35 * delinq_2yrs
        + rng.normal(0, 0.7, size=n_samples)
    )

    grades = []
    sub_grades = []
    int_rates = []
    grade_letters = ["A", "B", "C", "D", "E", "F", "G"]

    for score in risk_latent:
        if score < -1.8:
            g_idx = 0
            base_rate = 6.8
        elif score < -0.8:
            g_idx = 1
            base_rate = 10.2
        elif score < 0.2:
            g_idx = 2
            base_rate = 13.8
        elif score < 1.2:
            g_idx = 3
            base_rate = 17.5
        elif score < 2.2:
            g_idx = 4
            base_rate = 21.5
        elif score < 3.2:
            g_idx = 5
            base_rate = 25.5
        else:
            g_idx = 6
            base_rate = 29.5

        sub_idx = int(rng.integers(1, 6))
        grades.append(grade_letters[g_idx])
        sub_grades.append(f"{grade_letters[g_idx]}{sub_idx}")
        int_rates.append(round(base_rate + sub_idx * 0.65 + rng.uniform(-0.3, 0.3), 2))

    int_rates = np.array(int_rates)
    grades = np.array(grades)
    sub_grades = np.array(sub_grades)

    # Monthly installment calculation: P * (r*(1+r)^N) / ((1+r)^N - 1)
    rate_monthly = (int_rates / 100.0) / 12.0
    n_months = np.where(term == " 36 months", 36, 60)
    installment = loan_amnt * (rate_monthly * (1 + rate_monthly) ** n_months) / ((1 + rate_monthly) ** n_months - 1)
    installment = np.round(installment, 2)

    # Categorical fields
    emp_length = rng.choice(
        ["< 1 year", "1 year", "2 years", "3 years", "5 years", "7 years", "10+ years"],
        size=n_samples,
        p=[0.08, 0.07, 0.09, 0.08, 0.12, 0.08, 0.48],
    )
    home_ownership = rng.choice(["MORTGAGE", "RENT", "OWN", "OTHER"], size=n_samples, p=[0.50, 0.39, 0.108, 0.002])
    verification_status = rng.choice(
        ["Source Verified", "Verified", "Not Verified"], size=n_samples, p=[0.38, 0.33, 0.29]
    )
    purpose = rng.choice(
        ["debt_consolidation", "credit_card", "home_improvement", "major_purchase", "small_business", "medical"],
        size=n_samples,
        p=[0.58, 0.22, 0.07, 0.05, 0.04, 0.04],
    )

    # True Default Probability with Macroeconomic Shift in 2017 (for PSI drift demonstration)
    macro_shock = np.where(years >= 2017, 0.65, 0.0)  # Macro credit tightening / subprime deterioration
    default_logit = (
        -2.60
        + 0.065 * (dti - 18.0)
        + 0.020 * (revol_util - 50.0)
        - 0.028 * (fico_low - 700.0)
        + 0.18 * inq_last_6mths
        + 0.22 * delinq_2yrs
        + macro_shock
    )
    prob_default = 1.0 / (1.0 + np.exp(-default_logit))
    is_default = rng.binomial(1, prob_default)

    loan_status = np.where(is_default == 1, "Charged Off", "Fully Paid")

    df = pd.DataFrame(
        {
            "loan_id": np.arange(100_000, 100_000 + n_samples),
            "loan_amnt": loan_amnt.astype(float),
            "funded_amnt": funded_amnt.astype(float),
            "term": term,
            "int_rate": int_rates,
            "installment": installment,
            "grade": grades,
            "sub_grade": sub_grades,
            "emp_length": emp_length,
            "home_ownership": home_ownership,
            "annual_inc": np.round(annual_inc, 2),
            "verification_status": verification_status,
            "issue_d": issue_d,
            "loan_status": loan_status,
            "purpose": purpose,
            "dti": np.round(dti, 2),
            "delinq_2yrs": delinq_2yrs,
            "fico_range_low": fico_low,
            "fico_range_high": fico_high,
            "inq_last_6mths": inq_last_6mths,
            "open_acc": open_acc,
            "pub_rec": pub_rec,
            "revol_bal": np.round(revol_bal, 2),
            "revol_util": np.round(revol_util, 2),
            "total_acc": total_acc,
            "mort_acc": mort_acc,
            "pub_rec_bankruptcies": pub_rec_bankruptcies,
        }
    )

    return df


def download_or_load_dataset(raw_dir: Path) -> pd.DataFrame:
    """Download dataset from Kaggle, read local CSV, or generate realistic fallback."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_csv = raw_dir / "accepted_2007_to_2018Q4.csv"

    if raw_csv.exists():
        print(f"[*] Found local raw CSV at: {raw_csv}")
        df = pd.read_csv(raw_csv, usecols=lambda c: c in LENDING_CLUB_COLUMNS, low_memory=False)
        return df

    # Try Kaggle CLI download if credentials exist
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if kaggle_json.exists():
        print("[*] Kaggle credentials found. Attempting download via Kaggle API...")
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi

            api = KaggleApi()
            api.authenticate()
            api.dataset_download_files("wordsforthewise/lending-club", path=str(raw_dir), unzip=True)
            if raw_csv.exists():
                df = pd.read_csv(raw_csv, usecols=lambda c: c in LENDING_CLUB_COLUMNS, low_memory=False)
                return df
        except Exception as e:
            print(f"[!] Kaggle API download failed ({e}). Proceeding to high-fidelity generator fallback.")

    # High-fidelity generator fallback
    df = generate_realistic_lendingclub_data(n_samples=160_000)
    return df


def preprocess_and_partition(df: pd.DataFrame, processed_dir: Path) -> None:
    """Filter resolved loans, engineer binary target, and partition into 3 chronological Parquet files."""
    processed_dir.mkdir(parents=True, exist_ok=True)
    print("[*] Preprocessing raw loans data...")

    # Filter to resolved loans only
    resolved_mask = df["loan_status"].isin(GOOD_STATUSES + BAD_STATUSES)
    df_clean = df[resolved_mask].copy()

    # Binary target engineering (0 = Good/Repaid, 1 = Bad/Default)
    df_clean["target"] = df_clean["loan_status"].isin(BAD_STATUSES).astype(int)

    # Extract issue_year from issue_d (e.g. "Dec-2015" -> 2015)
    df_clean["issue_year"] = df_clean["issue_d"].apply(lambda d: int(str(d).split("-")[-1]))

    print(f"[*] Cleaned dataset total rows: {len(df_clean):,}")
    print(f"[*] Overall Default Rate: {df_clean['target'].mean():.2%}")

    # 3-Epoch Chronological Partitioning
    batch_1 = df_clean[(df_clean["issue_year"] >= 2013) & (df_clean["issue_year"] <= 2015)].copy()
    batch_2 = df_clean[df_clean["issue_year"] == 2016].copy()
    batch_3 = df_clean[(df_clean["issue_year"] >= 2017) & (df_clean["issue_year"] <= 2018)].copy()

    b1_path = processed_dir / "batch_1_baseline_2013_2015.parquet"
    b2_path = processed_dir / "batch_2_inference_2016.parquet"
    b3_path = processed_dir / "batch_3_drift_2017_2018.parquet"

    batch_1.to_parquet(b1_path, index=False)
    batch_2.to_parquet(b2_path, index=False)
    batch_3.to_parquet(b3_path, index=False)

    print("\n✅ Successfully created 3 Chronological Parquet Batches:")
    print(f"   • Batch 1 (Baseline 2013-2015): {len(batch_1):,} rows ({b1_path.stat().st_size / 1e6:.2f} MB)")
    print(f"   • Batch 2 (Inference 2016):     {len(batch_2):,} rows ({b2_path.stat().st_size / 1e6:.2f} MB)")
    print(f"   • Batch 3 (Drift 2017-2018):    {len(batch_3):,} rows ({b3_path.stat().st_size / 1e6:.2f} MB)")


def main() -> None:
    """CLI Entrypoint for downloading, preprocessing, and partitioning."""
    base_dir = Path(__file__).resolve().parents[2]
    raw_dir = base_dir / "data" / "raw"
    processed_dir = base_dir / "data" / "processed"

    df = download_or_load_dataset(raw_dir)
    preprocess_and_partition(df, processed_dir)


if __name__ == "__main__":
    main()
