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



DROP TABLE IF EXISTS iceberg_raw.claim_diagnosis_raw;

CREATE EXTERNAL TABLE iceberg_raw.claim_diagnosis_raw (
    id BIGINT,
    claim_id BIGINT,
    icd10_code STRING,
    icd10_description STRING,
    is_primary INT
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
TBLPROPERTIES (
  'format-version'='2'
);



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
TBLPROPERTIES (
  'format-version'='2'
);


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
TBLPROPERTIES (
  'format-version'='2'
);





CREATE EXTERNAL TABLE `iceberg_curated`.`claim_feature_set`(
`claim_id` bigint,
`patient_nik` string,
`patient_name` string,
`patient_gender` string,
`patient_dob` date,
`patient_age` int,
`visit_date` date,
`visit_year` int,
`visit_month` int,
`visit_day` int,
`visit_type` string,
`doctor_name` string,
`department` string,
`icd10_primary_code` string,
`icd10_primary_desc` string,
  `procedures_icd9_codes` array<string>, 
  `procedures_icd9_descs` array<string>, 
  `drug_codes` array<string>, 
  `drug_names` array<string>, 
  `vitamin_names` array<string>, 
`total_procedure_cost` double,
`total_drug_cost` double,
`total_vitamin_cost` double,
`total_claim_amount` double,
`biaya_anomaly_score` double,
`rule_violation_flag` int,
`rule_violation_reason` string,
`created_at` timestamp with local time zone,
`severity_score` int,
`diagnosis_procedure_mismatch` int,
`drug_mismatch_score` int,
`cost_per_procedure` double,
`cost_procedure_anomaly` int,
`patient_claim_count` int,
`patient_frequency_risk` int)
PARTITIONED BY SPEC (
visit_year,
visit_month)
STORED BY ICEBERG
TBLPROPERTIES ('format-version'='2');
