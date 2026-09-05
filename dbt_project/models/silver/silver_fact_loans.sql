{{ config(
    materialized='incremental',
    unique_key='loan_id',
    schema='silver',
    file_format='delta'
) }}

WITH staged AS (
    SELECT
        *,
        ROW_NUMBER() OVER (PARTITION BY loan_id ORDER BY staging_ingest_time DESC) AS rn
    FROM {{ ref('stg_raw_loans') }}
    {% if is_incremental() %}
        WHERE staging_ingest_time > (SELECT COALESCE(MAX(silver_updated_at), '1970-01-01') FROM {{ this }})
    {% endif %}
)

SELECT
    loan_id,
    loan_amnt,
    funded_amnt,
    term,
    int_rate,
    installment,
    grade,
    sub_grade,
    emp_length,
    home_ownership,
    annual_inc,
    verification_status,
    issue_d,
    issue_year,
    loan_status,
    target,
    purpose,
    dti,
    delinq_2yrs,
    fico_range_low,
    fico_range_high,
    inq_last_6mths,
    open_acc,
    pub_rec,
    revol_bal,
    revol_util,
    total_acc,
    mort_acc,
    pub_rec_bankruptcies,
    CURRENT_TIMESTAMP() AS silver_updated_at
FROM staged
WHERE rn = 1
