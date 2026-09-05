{% snapshot dim_borrowers_scd2 %}

{{
    config(
        target_schema='silver',
        unique_key='loan_id',
        strategy='check',
        check_cols=['grade', 'sub_grade', 'home_ownership', 'annual_inc', 'verification_status'],
        file_format='delta'
    )
}}

SELECT
    loan_id,
    grade,
    sub_grade,
    home_ownership,
    annual_inc,
    verification_status,
    issue_d,
    staging_ingest_time
FROM {{ ref('stg_raw_loans') }}

{% endsnapshot %}
