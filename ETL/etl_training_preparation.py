#!/usr/bin/env python3

import sys
import os
import cml.data_v1 as cmldata
from pyspark.sql import functions as F
from pyspark.sql.functions import (
    col, lit, when, collect_list, first, year, month, dayofmonth,
    current_timestamp, size, array, count, sum as spark_sum, avg, stddev
)
from pyspark.sql.window import Window
from pyspark.sql.types import DoubleType, IntegerType
from pyspark.sql import udf

# ==============================================================================
# 1. SETUP PATH & IMPORT CONFIG (ROBUST VERSION)
# ==============================================================================
try:
    # Coba deteksi path script saat ini
    current_working_dir = os.getcwd()
    print(f"Current Working Directory: {current_working_dir}")
    
    # Logika pencarian config.py yang fleksibel
    project_root = None
    
    # 1. Cek di direktori saat ini
    if os.path.exists(os.path.join(current_working_dir, "config.py")):
        project_root = current_working_dir
    # 2. Cek di parent directory (jika script di dalam folder ETL)
    elif os.path.exists(os.path.join(os.path.dirname(current_working_dir), "config.py")):
        project_root = os.path.dirname(current_working_dir)
    # 3. Fallback ke root default CML
    else:
        project_root = "/home/cdsw"

    # Tambahkan ke sys.path jika belum ada
    if project_root and project_root not in sys.path:
        sys.path.insert(0, project_root)
        
    print(f"Project root set to: {project_root}")

    from config import COMPAT_RULES, COST_THRESHOLDS
    print(f"✓ Successfully loaded {len(COMPAT_RULES)} rules from config.py")

except ImportError:
    print("⚠ WARNING: config.py not found. Using empty rules. Logic might be impacted.")
    COMPAT_RULES = {}
    COST_THRESHOLDS = {}

print("=" * 80)
print("FRAUD DETECTION ETL - FEATURE ENGINEERING PIPELINE")
print("=" * 80)

# ==============================================================================
# 2. CONNECT TO SPARK
# ==============================================================================
print("\n[1/10] Connecting to Spark...")
spark = None
try:
    conn = cmldata.get_connection("CDP-MSI")
    spark = conn.get_spark_session()
    print(f"✓ Spark Application ID: {spark.sparkContext.applicationId}")
except Exception as e:
    print(f"✗ Error connecting to Spark: {e}")
    # Exit jika dijalankan sebagai script
    if __name__ == "__main__" and not hasattr(sys, 'ps1'):
        sys.exit(1)

if spark is None:
    raise RuntimeError("Spark Initialization Failed")

# ==============================================================================
# 3. LOAD RAW TABLES
# ==============================================================================
print("\n[2/10] Loading raw tables from Iceberg...")
try:
    # Load Tables
    hdr = spark.table("iceberg_raw.claim_header_raw")
    diag = spark.table("iceberg_raw.claim_diagnosis_raw")
    proc = spark.table("iceberg_raw.claim_procedure_raw")
    drug = spark.table("iceberg_raw.claim_drug_raw")
    vit = spark.table("iceberg_raw.claim_vitamin_raw")

    row_count = hdr.count()
    if row_count == 0:
        print("⚠ WARNING: Table iceberg_raw.claim_header_raw is empty!")
    else:
        print(f"✓ Loaded {row_count:,} claims")
        
except Exception as e:
    print(f"✗ Error loading tables: {e}")
    spark.stop()
    raise e

# ==============================================================================
# 4. PRIMARY DIAGNOSIS
# ==============================================================================
print("\n[3/10] Extracting primary diagnosis...")
diag_primary = (
    diag.where(col("is_primary") == 1)
        .groupBy("claim_id")
        .agg(
            first("icd10_code").alias("icd10_primary_code"),
            first("icd10_description").alias("icd10_primary_desc")
        )
)

# ==============================================================================
# 5. AGGREGATIONS
# ==============================================================================
print("\n[4/10] Aggregating procedures, drugs, vitamins...")
proc_agg = proc.groupBy("claim_id").agg(
    collect_list("icd9_code").alias("procedures_icd9_codes"),
    collect_list("icd9_description").alias("procedures_icd9_descs"),
    collect_list("cost").alias("procedures_costs")
)

drug_agg = drug.groupBy("claim_id").agg(
    collect_list("drug_code").alias("drug_codes"),
    collect_list("drug_name").alias("drug_names"),
    collect_list("cost").alias("drug_costs")
)

vit_agg = vit.groupBy("claim_id").agg(
    collect_list("vitamin_name").alias("vitamin_names"),
    collect_list("cost").alias("vitamin_costs")
)

# ==============================================================================
# 6. JOIN ALL DATA
# ==============================================================================
print("\n[5/10] Joining all tables...")
base = (
    hdr.join(diag_primary, "claim_id", "left")
       .join(proc_agg, "claim_id", "left")
       .join(drug_agg, "claim_id", "left")
       .join(vit_agg, "claim_id", "left")
)

# ==============================================================================
# 7. BASIC FEATURES (DATE, AGE, FLAGS)
# ==============================================================================
print("\n[6/10] Creating basic features...")
base = (
    base.withColumn("patient_age",
        when(col("patient_dob").isNull(), None)
        .otherwise(year(col("visit_date")) - year(col("patient_dob")))
    )
    .withColumn("visit_year", year("visit_date"))
    .withColumn("visit_month", month("visit_date"))
    .withColumn("visit_day", dayofmonth("visit_date"))
    .withColumn("has_procedure", when(size("procedures_icd9_codes") > 0, 1).otherwise(0))
    .withColumn("has_drug", when(size("drug_codes") > 0, 1).otherwise(0))
    .withColumn("has_vitamin", when(size("vitamin_names") > 0, 1).otherwise(0))
)

# ==============================================================================
# 8. CLINICAL COMPATIBILITY CHECKING (KEY FEATURE!)
# ==============================================================================
print("\n[7/10] Checking clinical compatibility...")

# Create broadcast variable for COMPAT_RULES
compat_broadcast = spark.sparkContext.broadcast(COMPAT_RULES)

# --- Pure Python Logic Functions ---
def compute_procedure_compatibility_logic(icd10, procedures):
    if not icd10 or not procedures: return 0.0
    rules = compat_broadcast.value.get(icd10)
    if not rules: return 0.5
    allowed = rules.get("procedures", [])
    if not allowed: return 0.5
    matches = sum(1 for p in procedures if p in allowed)
    return float(matches) / len(procedures)

def compute_drug_compatibility_logic(icd10, drugs):
    if not icd10 or not drugs: return 0.0
    rules = compat_broadcast.value.get(icd10)
    if not rules: return 0.5
    allowed = rules.get("drugs", [])
    if not allowed: return 0.5
    matches = sum(1 for d in drugs if d in allowed)
    return float(matches) / len(drugs)

def compute_vitamin_compatibility_logic(icd10, vitamins):
    if not icd10 or not vitamins: return 0.0
    rules = compat_broadcast.value.get(icd10)
    if not rules: return 0.5
    allowed = rules.get("vitamins", [])
    if not allowed: return 0.5
    matches = sum(1 for v in vitamins if v in allowed)
    return float(matches) / len(vitamins)

# --- Define UDFs (Using F.udf to avoid Decorator TypeError) ---
compute_proc_udf = F.udf(lambda d, i: compute_procedure_compatibility_logic(d, i), DoubleType())
compute_drug_udf = F.udf(lambda d, i: compute_drug_compatibility_logic(d, i), DoubleType())
compute_vit_udf  = F.udf(lambda d, i: compute_vitamin_compatibility_logic(d, i), DoubleType())

# Apply compatibility checks
base = base.withColumn("diagnosis_procedure_score", compute_proc_udf(col("icd10_primary_code"), col("procedures_icd9_codes")))
base = base.withColumn("diagnosis_drug_score", compute_drug_udf(col("icd10_primary_code"), col("drug_codes")))
base = base.withColumn("diagnosis_vitamin_score", compute_vit_udf(col("icd10_primary_code"), col("vitamin_names")))

# ==============================================================================
# 9. MISMATCH FLAGS (BINARY INDICATORS)
# ==============================================================================
print("\n[8/10] Creating mismatch flags...")
base = base.withColumn("procedure_mismatch_flag", when(col("diagnosis_procedure_score") < 0.5, 1).otherwise(0)) \
           .withColumn("drug_mismatch_flag", when(col("diagnosis_drug_score") < 0.5, 1).otherwise(0)) \
           .withColumn("vitamin_mismatch_flag", when(col("diagnosis_vitamin_score") < 0.5, 1).otherwise(0))

base = base.withColumn("mismatch_count", col("procedure_mismatch_flag") + col("drug_mismatch_flag") + col("vitamin_mismatch_flag"))

# ==============================================================================
# 10. COST ANOMALY DETECTION (STATISTICAL)
# ==============================================================================
print("\n[9/10] Detecting cost anomalies...")

# Calculate z-score for costs per diagnosis
diagnosis_window = Window.partitionBy("icd10_primary_code")

base = base.withColumn("diagnosis_avg_cost", avg("total_claim_amount").over(diagnosis_window)) \
           .withColumn("diagnosis_stddev_cost", stddev("total_claim_amount").over(diagnosis_window)) \
           .withColumn("cost_zscore", 
               (col("total_claim_amount") - col("diagnosis_avg_cost")) / 
               when(col("diagnosis_stddev_cost") == 0, 1).otherwise(col("diagnosis_stddev_cost"))
           ) \
           .withColumn("biaya_anomaly_score",
               when(col("cost_zscore") > 3, 4)
               .when(col("cost_zscore") > 2, 3)
               .when(col("cost_zscore") > 1, 2)
               .otherwise(1)
           )

# Drop intermediate columns
base = base.drop("diagnosis_avg_cost", "diagnosis_stddev_cost", "cost_zscore")

# ==============================================================================
# 11. PATIENT FREQUENCY RISK
# ==============================================================================
print("\n[10/10] Calculating patient frequency...")
patient_freq = base.groupBy("patient_nik").agg(count("claim_id").alias("patient_frequency_risk"))
base = base.join(patient_freq, "patient_nik", "left")

# ==============================================================================
# 12. FINAL LABEL (GROUND TRUTH)
# ==============================================================================
print("\nCreating final labels...")

base = base.withColumn("human_label",
    when(col("status") == "declined", 1)
    .when(col("status") == "approved", 0)
    .otherwise(None)
)

base = base.withColumn("rule_violation_flag",
    when(col("mismatch_count") > 0, 1)
    .when(col("biaya_anomaly_score") >= 3, 1)
    .when(col("patient_frequency_risk") > 10, 1)
    .otherwise(0)
)

base = base.withColumn("final_label",
    when(col("human_label").isNotNull(), col("human_label"))
    .otherwise(col("rule_violation_flag"))
)

# ==============================================================================
# 13. SELECT FINAL FEATURES
# ==============================================================================
base = base.withColumn("created_at", current_timestamp())

final_columns = [
    "claim_id", "patient_nik", "patient_name", "patient_gender", "patient_dob", "patient_age",
    "visit_date", "visit_year", "visit_month", "visit_day", "visit_type", "doctor_name", "department",
    "icd10_primary_code", "icd10_primary_desc",
    "procedures_icd9_codes", "procedures_icd9_descs", "drug_codes", "drug_names", "vitamin_names",
    "total_procedure_cost", "total_drug_cost", "total_vitamin_cost", "total_claim_amount",
    "diagnosis_procedure_score", "diagnosis_drug_score", "diagnosis_vitamin_score",
    "procedure_mismatch_flag", "drug_mismatch_flag", "vitamin_mismatch_flag", "mismatch_count",
    "patient_frequency_risk", "biaya_anomaly_score",
    "rule_violation_flag", "human_label", "final_label", "created_at"
]

feature_df = base.select(*final_columns)

# ==============================================================================
# 14. SAVE TO ICEBERG
# ==============================================================================
target_table = "iceberg_curated.claim_feature_set"
print(f"\nSaving to Iceberg curated table: {target_table}")

try:
    if feature_df.count() > 0:
        feature_df.write.format("iceberg") \
            .partitionBy("visit_year", "visit_month") \
            .mode("overwrite") \
            .saveAsTable(target_table)
        print(f"✓ Data successfully saved to {target_table}")
    else:
        print("⚠ WARNING: Result DataFrame is empty. Nothing written.")

except Exception as e:
    print(f"✗ Error writing to Iceberg: {e}")

# ==============================================================================
# 15. DATA QUALITY REPORT
# ==============================================================================
print("\n" + "=" * 80)
print("ETL COMPLETE - DATA QUALITY REPORT")
print("=" * 80)

total_claims = feature_df.count()
if total_claims > 0:
    fraud_count = feature_df.filter(col("final_label") == 1).count()
    non_fraud_count = total_claims - fraud_count

    print(f"Total claims processed: {total_claims:,}")
    print(f"Fraud claims: {fraud_count:,} ({fraud_count/total_claims*100:.1f}%)")
    print(f"Non-fraud claims: {non_fraud_count:,} ({non_fraud_count/total_claims*100:.1f}%)")

    print("\nMismatch distribution:")
    mismatch_dist = feature_df.groupBy("mismatch_count").count().orderBy("mismatch_count").collect()
    for row in mismatch_dist:
        print(f"  {row['mismatch_count']} mismatches: {row['count']:,} claims")

    print("\nCost anomaly distribution:")
    anomaly_dist = feature_df.groupBy("biaya_anomaly_score").count().orderBy("biaya_anomaly_score").collect()
    for row in anomaly_dist:
        print(f"  Score {row['biaya_anomaly_score']}: {row['count']:,} claims")
else:
    print("No data processed.")

print("\n" + "=" * 80)
print(f"Feature set ready for training at: {target_table}")
print("=" * 80)

spark.stop()