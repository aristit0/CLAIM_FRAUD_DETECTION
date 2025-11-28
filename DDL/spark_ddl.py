from pyspark.sql import SparkSession

spark = (
    SparkSession.builder
    .appName("CreateIcebergV2FeatureSet")
    .config("spark.sql.catalog.spark_catalog", "org.apache.iceberg.spark.SparkSessionCatalog")
    .config("spark.sql.catalog.spark_catalog.type", "hive")
    .getOrCreate()
)

spark.sql("DROP TABLE IF EXISTS iceberg_curated.claim_feature_set")

spark.sql("""
CREATE TABLE iceberg_curated.claim_feature_set (
    claim_id BIGINT,
    patient_nik STRING,
    patient_name STRING,
    patient_gender STRING,
    patient_dob DATE,
    patient_age INT,
    visit_date DATE,
    visit_day INT,
    visit_type STRING,
    doctor_name STRING,
    department STRING,
    icd10_primary_code STRING,
    icd10_primary_desc STRING,
    procedures_icd9_codes ARRAY<STRING>,
    procedures_icd9_descs ARRAY<STRING>,
    drug_codes ARRAY<STRING>,
    drug_names ARRAY<STRING>,
    vitamin_names ARRAY<STRING>,
    total_procedure_cost DOUBLE,
    total_drug_cost DOUBLE,
    total_vitamin_cost DOUBLE,
    total_claim_amount DOUBLE,
    tindakan_validity_score DOUBLE,
    obat_validity_score DOUBLE,
    vitamin_relevance_score DOUBLE,
    biaya_anomaly_score DOUBLE,
    rule_violation_flag INT,
    rule_violation_reason STRING,
    created_at TIMESTAMP
)
USING iceberg
PARTITIONED BY (visit_year, visit_month)
TBLPROPERTIES ('format-version'='2')
""")

print("Iceberg v2 table claim_feature_set CREATED.")