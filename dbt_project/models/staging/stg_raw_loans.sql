{{ config(
    materialized='view',
    schema='staging'
) }}

WITH source_data AS (
    SELECT
        CAST(id AS BIGINT) AS loan_id,
        CAST(loan_amnt AS DOUBLE) AS loan_amnt,
        CAST(funded_amnt AS DOUBLE) AS funded_amnt,
        CAST(term AS STRING) AS term,
        CAST(REPLACE(REPLACE(int_rate, '%', ''), ' ', '') AS DOUBLE) AS int_rate,
        CAST(installment AS DOUBLE) AS installment,
        CAST(grade AS STRING) AS grade,
        CAST(sub_grade AS STRING) AS sub_grade,
        CAST(emp_length AS STRING) AS emp_length,
        CAST(home_ownership AS STRING) AS home_ownership,
        CAST(annual_inc AS DOUBLE) AS annual_inc,
        CAST(verification_status AS STRING) AS verification_status,
        CAST(issue_d AS STRING) AS issue_d,
        CAST(loan_status AS STRING) AS loan_status,
        CAST(purpose AS STRING) AS purpose,
        CAST(dti AS DOUBLE) AS dti,
        CAST(delinq_2yrs AS DOUBLE) AS delinq_2yrs,
        CAST(fico_range_low AS DOUBLE) AS fico_range_low,
        CAST(fico_range_high AS DOUBLE) AS fico_range_high,
        CAST(inq_last_6mths AS DOUBLE) AS inq_last_6mths,
        CAST(open_acc AS DOUBLE) AS open_acc,
        CAST(pub_rec AS DOUBLE) AS pub_rec,
        CAST(revol_bal AS DOUBLE) AS revol_bal,
        CAST(REPLACE(REPLACE(revol_util, '%', ''), ' ', '') AS DOUBLE) AS revol_util,
        CAST(total_acc AS DOUBLE) AS total_acc,
        CAST(mort_acc AS DOUBLE) AS mort_acc,
        CAST(pub_rec_bankruptcies AS DOUBLE) AS pub_rec_bankruptcies
    FROM {{ source('lakehouse_landing', 'bronze_loans') }}
    WHERE id IS NOT NULL
      AND loan_status IS NOT NULL
      AND issue_d IS NOT NULL
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
    COALESCE(emp_length, 'Unknown') AS emp_length,
    COALESCE(home_ownership, 'OTHER') AS home_ownership,
    annual_inc,
    verification_status,
    issue_d,
    CAST(SUBSTRING(issue_d, INSTR(issue_d, '-') + 1) AS INT) AS issue_year,
    loan_status,
    -- Target: 0 = Repaid / Good, 1 = Default / Bad (Basel PD definition)
    CASE
        WHEN loan_status IN ('Charged Off', 'Default', 'Does not meet the credit policy. Status:Charged Off') THEN 1
        ELSE 0
    END AS target,
    purpose,
    dti,
    COALESCE(delinq_2yrs, 0.0) AS delinq_2yrs,
    fico_range_low,
    fico_range_high,
    COALESCE(inq_last_6mths, 0.0) AS inq_last_6mths,
    COALESCE(open_acc, 0.0) AS open_acc,
    COALESCE(pub_rec, 0.0) AS pub_rec,
    COALESCE(revol_bal, 0.0) AS revol_bal,
    COALESCE(revol_util, 0.0) AS revol_util,
    COALESCE(total_acc, 0.0) AS total_acc,
    COALESCE(mort_acc, 0.0) AS mort_acc,
    COALESCE(pub_rec_bankruptcies, 0.0) AS pub_rec_bankruptcies,
    CURRENT_TIMESTAMP() AS staging_ingest_time
FROM source_data
WHERE loan_status IN (
    'Fully Paid',
    'Charged Off',
    'Default',
    'Does not meet the credit policy. Status:Fully Paid',
    'Does not meet the credit policy. Status:Charged Off'
)
