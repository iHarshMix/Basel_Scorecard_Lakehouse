{{ config(
    materialized='table',
    schema='gold',
    file_format='delta'
) }}

WITH silver_loans AS (
    SELECT *
    FROM {{ ref('silver_fact_loans') }}
    WHERE annual_inc > 0
      AND dti >= 0
      AND fico_range_low IS NOT NULL
      AND fico_range_high IS NOT NULL
)

SELECT
    loan_id,
    target,
    issue_year,
    issue_d,

    -- Primary Credit Bureau Risk Feature (FICO Midpoint)
    (fico_range_low + fico_range_high) / 2.0 AS fico_mid,

    -- Capacity and Leverage Ratios
    CASE
        WHEN dti > 100.0 THEN 100.0
        ELSE dti
    END AS dti,

    loan_amnt,
    annual_inc,

    -- Affordability Ratio: Annualized installment obligation relative to gross income
    (installment * 12.0) / annual_inc AS installment_to_inc,

    -- Revolving Credit Depth and Line Utilization
    revol_bal,
    CASE
        WHEN revol_util > 100.0 THEN 100.0
        WHEN revol_util < 0.0 THEN 0.0
        ELSE revol_util
    END AS revol_util,

    -- Credit Seeking & Inquiries
    inq_last_6mths,
    open_acc,
    total_acc,
    mort_acc,

    -- Adverse Derogatory Event Flags
    CASE WHEN delinq_2yrs > 0 THEN 1 ELSE 0 END AS has_delinq,
    CASE WHEN pub_rec > 0 THEN 1 ELSE 0 END AS has_pub_rec,
    CASE WHEN pub_rec_bankruptcies > 0 THEN 1 ELSE 0 END AS has_bankruptcy,

    -- Categorical Profile Attributes
    home_ownership,
    purpose,
    emp_length,
    verification_status,

    -- Model Governance Audit Stamp
    CURRENT_TIMESTAMP() AS feature_engineered_at

FROM silver_loans
