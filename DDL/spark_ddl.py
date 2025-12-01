from pyspark.sql import SparkSession
import subprocess

# ===============================================================
# 1. SparkSession dengan Iceberg
# ===============================================================
spark = (
    SparkSession.builder
    .appName("recreate_claim_feature_set_v5")
    .config("spark.sql.catalog.spark_catalog", "org.apache.iceberg.spark.SparkSessionCatalog")
    .config("spark.sql.catalog.spark_catalog.type", "hive")
    .getOrCreate()
)

print("=== START RESET TABLE v5 ===")

# ===============================================================
# 2. DROP TABLE
# ===============================================================
spark.sql("USE iceberg_curated")

spark.sql("DROP TABLE IF EXISTS claim_feature_set PURGE")
print("[OK] Hive metadata table dropped.")

# ===============================================================
# 3. DELETE FOLDER DI HDFS
# ===============================================================
hdfs_path = "/warehouse/tablespace/external/hive/iceberg_curated.db/claim_feature_set"

try:
    subprocess.run(["hdfs", "dfs", "-rm", "-r", "-f", hdfs_path], check=False)
    print(f"[OK] HDFS folder removed: {hdfs_path}")
except Exception as e:
    print(f"[WARN] Could not remove HDFS folder: {e}")

# ===============================================================
# 4. CREATE Iceberg TABLE v5
# ===============================================================
spark.sql("""
CREATE TABLE iceberg_curated.claim_feature_set (

    -------------------------------------------------------------
    -- CLAIM + PATIENT INFORMATION
    -------------------------------------------------------------
    claim_id BIGINT,
    patient_nik STRING,
    patient_name STRING,
    patient_gender STRING,
    patient_dob DATE,
    patient_age INT,

    -------------------------------------------------------------
    -- VISIT INFO
    -------------------------------------------------------------
    visit_date DATE,
    visit_year INT,
    visit_month INT,
    visit_day INT,
    visit_type STRING,
    doctor_name STRING,
    department STRING,

    -------------------------------------------------------------
    -- DIAGNOSIS
    -------------------------------------------------------------
    icd10_primary_code STRING,
    icd10_primary_desc STRING,

    -------------------------------------------------------------
    -- RAW LISTS
    -------------------------------------------------------------
    procedures_icd9_codes ARRAY<STRING>,
    procedures_icd9_descs ARRAY<STRING>,
    drug_codes ARRAY<STRING>,
    drug_names ARRAY<STRING>,
    vitamin_names ARRAY<STRING>,

    -------------------------------------------------------------
    -- COSTS
    -------------------------------------------------------------
    total_procedure_cost DOUBLE,
    total_drug_cost DOUBLE,
    total_vitamin_cost DOUBLE,
    total_claim_amount DOUBLE,

    -------------------------------------------------------------
    -- CLINICAL COMPATIBILITY SCORES
    -------------------------------------------------------------
    diagnosis_procedure_score DOUBLE,
    diagnosis_drug_score DOUBLE,
    diagnosis_vitamin_score DOUBLE,
    treatment_consistency_score DOUBLE,

    -------------------------------------------------------------
    -- *** NEW: EXPLICIT MISMATCH FLAGS ***
    -------------------------------------------------------------
    procedure_mismatch_flag INT,
    drug_mismatch_flag INT,
    vitamin_mismatch_flag INT,
    mismatch_count INT,

    -------------------------------------------------------------
    -- RISK FEATURES (Z-SCORE, SEVERITY, FREQUENCY)
    -------------------------------------------------------------
    severity_score INT,
    cost_per_procedure DOUBLE,
    cost_procedure_anomaly INT,
    patient_claim_count INT,
    patient_frequency_risk INT,
    biaya_anomaly_score DOUBLE,

    -------------------------------------------------------------
    -- LEGACY RULE FIELDS (MASIH DIPERTAHANKAN)
    -------------------------------------------------------------
    tindakan_validity_score DOUBLE,
    obat_validity_score DOUBLE,
    vitamin_relevance_score DOUBLE,
    diagnosis_procedure_mismatch INT,
    drug_mismatch_score INT,

    -------------------------------------------------------------
    -- LABELING
    -------------------------------------------------------------
    rule_violation_flag INT,
    rule_violation_reason STRING,
    human_label INT,
    final_label INT,

    -------------------------------------------------------------
    -- METADATA
    -------------------------------------------------------------
    created_at TIMESTAMP
)
USING iceberg
PARTITIONED BY (visit_year, visit_month)
TBLPROPERTIES ('format-version'='2');
""")

print("[OK] Iceberg v5 schema created successfully.")

# ===============================================================
# 5. PRINT SCHEMA
# ===============================================================
spark.table("iceberg_curated.claim_feature_set").printSchema()

print("=== RESET v5 COMPLETED ===")

spark.stop()