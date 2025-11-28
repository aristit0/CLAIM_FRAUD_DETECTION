CREATE DATABASE IF NOT EXISTS iceberg_raw;
CREATE DATABASE IF NOT EXISTS iceberg_curated;

DROP TABLE IF EXISTS iceberg_raw.claim_header_raw;

CREATE EXTERNAL TABLE iceberg_raw.claim_header_raw (
    claim_id BIGINT,
    patient_nik STRING,
    patient_name STRING,
    patient_gender STRING,
    patient_dob DATE,
    patient_address STRING,
    patient_phone STRING,

    visit_date DATE,
    visit_type STRING,
    doctor_name STRING,
    department STRING,

    total_procedure_cost DOUBLE,
    total_drug_cost DOUBLE,
    total_vitamin_cost DOUBLE,
    total_claim_amount DOUBLE,

    status STRING,
    created_at TIMESTAMP,
    updated_at TIMESTAMP
)
STORED BY ICEBERG
TBLPROPERTIES (
    'format-version'='2'
);


DROP TABLE IF EXISTS iceberg_raw.claim_procedure_raw;

CREATE EXTERNAL TABLE iceberg_raw.claim_procedure_raw (
    id BIGINT,
    claim_id BIGINT,
    icd9_code STRING,
    icd9_description STRING,
    quantity INT,
    procedure_date DATE
)
STORED BY ICEBERG
TBLPROPERTIES ('format-version'='2');



DROP TABLE IF EXISTS iceberg_raw.claim_drug_raw;

CREATE EXTERNAL TABLE iceberg_raw.claim_drug_raw (
    id BIGINT,
    claim_id BIGINT,
    drug_code STRING,
    drug_name STRING,
    dosage STRING,
    frequency STRING,
    route STRING,
    days INT,
    cost DOUBLE
)
STORED BY ICEBERG
TBLPROPERTIES ('format-version'='2');




DROP TABLE IF EXISTS iceberg_raw.claim_vitamin_raw;

CREATE EXTERNAL TABLE iceberg_raw.claim_vitamin_raw (
    id BIGINT,
    claim_id BIGINT,
    vitamin_name STRING,
    dosage STRING,
    days INT,
    cost DOUBLE
)
STORED BY ICEBERG
TBLPROPERTIES ('format-version'='2');


CREATE EXTERNAL TABLE iceberg_curated.claim_facts (
    claim_id BIGINT,
    
    -- patient base info
    patient_nik STRING,
    patient_name STRING,
    patient_gender STRING,
    patient_dob DATE,
    patient_age INT,
    patient_address STRING,
    patient_phone STRING,

    -- visit info
    visit_date DATE,
    visit_day INT,
    visit_type STRING,
    doctor_name STRING,
    department STRING,

    -- diagnosis flattened
    icd10_primary_code STRING,
    icd10_primary_desc STRING,
    icd10_secondary_code STRING,
    icd10_secondary_desc STRING,

    -- procedure flattened
    icd9_code STRING,
    icd9_description STRING,

    -- drug info flattened
    drug_name STRING,
    drug_code STRING,

    -- vitamin
    vitamin_name STRING,

    -- cost info
    total_cost DOUBLE,
    procedure_cost DOUBLE,
    drug_cost DOUBLE,
    vitamin_cost DOUBLE,

    -- claim status
    created_at TIMESTAMP
)
PARTITIONED BY (
    visit_year INT,
    visit_month INT,
    status STRING
)
STORED BY ICEBERG;