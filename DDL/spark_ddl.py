from pyspark.sql import SparkSession
import subprocess

# ===============================================================
# 1. SparkSession dengan Iceberg
# ===============================================================
spark = (
    SparkSession.builder
    .appName("recreate_claim_feature_set")
    .config("spark.sql.catalog.spark_catalog", "org.apache.iceberg.spark.SparkSessionCatalog")
    .config("spark.sql.catalog.spark_catalog.type", "hive")
    .getOrCreate()
)

print("=== START CLEAN RESET TABLE ===")

# ===============================================================
# 2. DROP TABLE dari Hive Metastore
# ===============================================================
spark.sql("USE iceberg_curated")

spark.sql("""
DROP TABLE IF EXISTS claim_feature_set PURGE
""")

print("[OK] Hive table dropped.")

# ===============================================================
# 3. DELETE HDFS TABLE DIRECTORY (WAJIB)
# ===============================================================
hdfs_path = "/warehouse/tablespace/external/hive/iceberg_curated.db/claim_feature_set"

try:
    subprocess.run(["hdfs", "dfs", "-rm", "-r", "-f", hdfs_path], check=False)
    print(f"[OK] HDFS folder deleted: {hdfs_path}")
except Exception as e:
    print(f"[WARN] Failed to delete HDFS directory: {e}")

# ===============================================================
# 4. RECREATE ICEBERG TABLE V2 (FINAL SCHEMA)
# ===============================================================

spark.sql("""
CREATE TABLE iceberg_curated.claim_feature_set (

    claim_id BIGINT,
    patient_nik STRING,
    patient_name STRING,
    patient_gender STRING,
    patient_dob DATE,
    patient_age INT,

    visit_date DATE,
    visit_year INT,
    visit_month INT,
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

    -------------------------------------------------------------
    -- LEGACY VALIDITY SCORES
    -------------------------------------------------------------
    tindakan_validity_score DOUBLE,
    obat_validity_score DOUBLE,
    vitamin_relevance_score DOUBLE,

    -------------------------------------------------------------
    -- CLINICAL COMPATIBILITY
    -------------------------------------------------------------
    diagnosis_procedure_score DOUBLE,
    diagnosis_drug_score DOUBLE,
    diagnosis_vitamin_score DOUBLE,
    treatment_consistency_score DOUBLE,

    -------------------------------------------------------------
    -- RULE FEATURES
    -------------------------------------------------------------
    severity_score INT,
    diagnosis_procedure_mismatch INT,
    drug_mismatch_score INT,
    cost_per_procedure DOUBLE,
    cost_procedure_anomaly INT,
    patient_claim_count INT,
    patient_frequency_risk INT,
    biaya_anomaly_score DOUBLE,

    -------------------------------------------------------------
    -- FINAL RISK + HUMAN LABEL
    -------------------------------------------------------------
    rule_violation_flag INT,
    rule_violation_reason STRING,

    -- ⭐ ADDED (baru)
    human_label INT,
    final_label INT,

    created_at TIMESTAMP
)
USING iceberg
PARTITIONED BY (visit_year, visit_month)
TBLPROPERTIES ('format-version'='2');
""")

print("[OK] Iceberg v2 table created.")

# ===============================================================
# 5. Print schema untuk memastikan semuanya clean
# ===============================================================
spark.table("iceberg_curated.claim_feature_set").printSchema()

print("=== CLEAN RESET COMPLETED ===")
spark.stop()