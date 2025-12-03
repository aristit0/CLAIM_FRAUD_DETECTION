#!/usr/bin/env python3
"""
Production ETL Pipeline for Fraud Detection
Learns from historical approved/declined claims
Extracts features for model training and inference
"""

import sys
import os
import cml.data_v1 as cmldata
from pyspark.sql import functions as F
from pyspark.sql.functions import (
    col, lit, when, collect_list, first, year, month, dayofmonth,
    current_timestamp, size, array, count, sum as spark_sum, avg, stddev,
    datediff, lag, max as spark_max, udf as spark_udf  # FIX: Import udf correctly
)
from pyspark.sql.window import Window
from pyspark.sql.types import DoubleType, IntegerType, StringType, ArrayType

# Import centralized config
base_path = os.path.dirname(os.path.dirname(os.getcwd()))
sys.path.insert(0, base_path)
from config import COMPAT_RULES, COST_THRESHOLDS, NUMERIC_FEATURES, CATEGORICAL_FEATURES

print("=" * 80)
print("FRAUD DETECTION ETL - PRODUCTION PIPELINE")
print("Learning from Historical Claims (Approved/Declined)")
print("=" * 80)

# ================================================================
# 1. CONNECT TO SPARK
# ================================================================
print("\n[1/12] Connecting to Spark...")
conn = cmldata.get_connection("CDP-MSI")
spark = conn.get_spark_session()
print(f"✓ Spark Application ID: {spark.sparkContext.applicationId}")

# ================================================================
# 2. LOAD RAW TABLES
# ================================================================
print("\n[2/12] Loading raw tables from Iceberg...")
hdr = spark.sql("SELECT * FROM iceberg_raw.claim_header_raw")
diag = spark.sql("SELECT * FROM iceberg_raw.claim_diagnosis_raw")
proc = spark.sql("SELECT * FROM iceberg_raw.claim_procedure_raw")
drug = spark.sql("SELECT * FROM iceberg_raw.claim_drug_raw")
vit = spark.sql("SELECT * FROM iceberg_raw.claim_vitamin_raw")

total_claims = hdr.count()
print(f"✓ Loaded {total_claims:,} claims")

# ================================================================
# 3. EXTRACT PRIMARY DIAGNOSIS + VALIDATION
# ================================================================
print("\n[3/12] Extracting primary diagnosis...")
diag_primary = (
    diag.where(col("is_primary") == 1)
        .groupBy("claim_id")
        .agg(
            first("icd10_code").alias("icd10_primary_code"),
            first("icd10_description").alias("icd10_primary_desc")
        )
)

# After join, fill missing diagnosis
base = base.withColumn(
    "icd10_primary_code",
    when(col("icd10_primary_code").isNull(), lit("UNKNOWN"))
    .otherwise(col("icd10_primary_code"))
).withColumn(
    "icd10_primary_desc",
    when(col("icd10_primary_desc").isNull(), lit("Unknown diagnosis"))
    .otherwise(col("icd10_primary_desc"))
)

# ================================================================
# 4. AGGREGATE PROCEDURES, DRUGS, VITAMINS
# ================================================================
print("\n[4/12] Aggregating medical items...")
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


# ================================================================
# 5. JOIN ALL DATA + DEDUPLICATION
# ================================================================
print("\n[5/12] Joining all tables...")
base = (
    hdr.join(diag_primary, "claim_id", "left")
       .join(proc_agg, "claim_id", "left")
       .join(drug_agg, "claim_id", "left")
       .join(vit_agg, "claim_id", "left")
)

# CRITICAL: Remove duplicates by keeping first occurrence per claim_id
print("  Removing duplicate claims...")
from pyspark.sql.window import Window
from pyspark.sql.functions import row_number

window_spec = Window.partitionBy("claim_id").orderBy("visit_date")
base = base.withColumn("row_num", row_number().over(window_spec)) \
           .filter(col("row_num") == 1) \
           .drop("row_num")

print("✓ Duplicates removed")


# ================================================================
# 5.5. DATA QUALITY FIXES
# ================================================================
print("\n[5.5/12] Applying data quality fixes...")

# 1. Remove duplicates
from pyspark.sql.window import Window
from pyspark.sql.functions import row_number

window_spec = Window.partitionBy("claim_id").orderBy("visit_date")
base = base.withColumn("row_num", row_number().over(window_spec)) \
           .filter(col("row_num") == 1) \
           .drop("row_num")

claims_after_dedup = base.count()
print(f"  After deduplication: {claims_after_dedup:,} claims")

# 2. Filter claims without diagnosis
claims_before_filter = base.count()
base = base.filter(col("icd10_primary_code").isNotNull())
claims_after_filter = base.count()
removed = claims_before_filter - claims_after_filter

print(f"  Removed {removed:,} claims without diagnosis ({removed/claims_before_filter*100:.1f}%)")
print(f"  Remaining: {claims_after_filter:,} valid claims")

# 3. Ensure all arrays are not null
base = base.fillna({
    "procedures_icd9_codes": [],
    "drug_codes": [],
    "vitamin_names": []
})

print("✓ Data quality fixes applied")


# ================================================================
# 6. BASIC FEATURES (DATE, AGE, FLAGS)
# ================================================================
print("\n[6/12] Creating basic features...")
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

# ================================================================
# 7. CLINICAL COMPATIBILITY CHECKING (CRITICAL FEATURE!)
# ================================================================
print("\n[7/12] Checking clinical compatibility (CRITICAL FEATURE)...")

# Broadcast COMPAT_RULES for efficient UDF execution
compat_broadcast = spark.sparkContext.broadcast(COMPAT_RULES)

# FIX: Use spark_udf instead of udf
@spark_udf(returnType=DoubleType())
def compute_procedure_compatibility(icd10, procedures):
    """
    Check if procedures are clinically compatible with diagnosis.
    Returns: 0.0 (no match) to 1.0 (perfect match)
    """
    if not icd10 or not procedures:
        return 0.0
    
    rules = compat_broadcast.value.get(icd10)
    if not rules:
        return 0.5  # Unknown diagnosis - neutral score
    
    allowed_procedures = rules.get("procedures", [])
    if not allowed_procedures:
        return 0.5
    
    # Calculate match ratio
    matches = sum(1 for p in procedures if p in allowed_procedures)
    return float(matches) / len(procedures)

@spark_udf(returnType=DoubleType())
def compute_drug_compatibility(icd10, drugs):
    """Check if drugs are clinically appropriate for diagnosis"""
    if not icd10 or not drugs:
        return 0.0
    
    rules = compat_broadcast.value.get(icd10)
    if not rules:
        return 0.5
    
    allowed_drugs = rules.get("drugs", [])
    if not allowed_drugs:
        return 0.5
    
    matches = sum(1 for d in drugs if d in allowed_drugs)
    return float(matches) / len(drugs)

@spark_udf(returnType=DoubleType())
def compute_vitamin_compatibility(icd10, vitamins):
    """Check if vitamins are appropriate for diagnosis"""
    if not icd10 or not vitamins:
        return 0.0
    
    rules = compat_broadcast.value.get(icd10)
    if not rules:
        return 0.5
    
    allowed_vitamins = rules.get("vitamins", [])
    if not allowed_vitamins:
        return 0.5
    
    matches = sum(1 for v in vitamins if v in allowed_vitamins)
    return float(matches) / len(vitamins)

# Apply compatibility checks
base = base.withColumn(
    "diagnosis_procedure_score",
    compute_procedure_compatibility(col("icd10_primary_code"), col("procedures_icd9_codes"))
)

base = base.withColumn(
    "diagnosis_drug_score",
    compute_drug_compatibility(col("icd10_primary_code"), col("drug_codes"))
)

base = base.withColumn(
    "diagnosis_vitamin_score",
    compute_vitamin_compatibility(col("icd10_primary_code"), col("vitamin_names"))
)

print("✓ Clinical compatibility scores computed")

# ================================================================
# 8. MISMATCH FLAGS (BINARY FRAUD INDICATORS)
# ================================================================
print("\n[8/12] Creating mismatch flags...")

# A score < 0.5 indicates incompatibility (potential fraud)
base = base.withColumn(
    "procedure_mismatch_flag",
    when(col("diagnosis_procedure_score") < 0.5, 1).otherwise(0)
)

base = base.withColumn(
    "drug_mismatch_flag",
    when(col("diagnosis_drug_score") < 0.5, 1).otherwise(0)
)

base = base.withColumn(
    "vitamin_mismatch_flag",
    when(col("diagnosis_vitamin_score") < 0.5, 1).otherwise(0)
)

base = base.withColumn(
    "mismatch_count",
    col("procedure_mismatch_flag") + 
    col("drug_mismatch_flag") + 
    col("vitamin_mismatch_flag")
)

print("✓ Mismatch flags created")

# ================================================================
# 9. COST ANOMALY DETECTION (STATISTICAL)
# ================================================================
print("\n[9/12] Detecting cost anomalies...")

# Calculate statistical z-score per diagnosis
diagnosis_window = Window.partitionBy("icd10_primary_code")

base = base.withColumn(
    "diagnosis_avg_cost",
    avg("total_claim_amount").over(diagnosis_window)
).withColumn(
    "diagnosis_stddev_cost",
    stddev("total_claim_amount").over(diagnosis_window)
).withColumn(
    "cost_zscore",
    (col("total_claim_amount") - col("diagnosis_avg_cost")) /
    when(col("diagnosis_stddev_cost") == 0, 1).otherwise(col("diagnosis_stddev_cost"))
).withColumn(
    "biaya_anomaly_score",
    when(col("cost_zscore") > 3, 4)      # Extreme outlier (>3 SD)
    .when(col("cost_zscore") > 2, 3)     # High outlier (2-3 SD)
    .when(col("cost_zscore") > 1, 2)     # Moderate (1-2 SD)
    .otherwise(1)                         # Normal (<1 SD)
)

# Drop intermediate columns
base = base.drop("diagnosis_avg_cost", "diagnosis_stddev_cost", "cost_zscore")

print("✓ Cost anomaly scores computed")

# ================================================================
# 10. PATIENT FREQUENCY RISK
# ================================================================
print("\n[10/12] Calculating patient claim frequency...")

# Count claims per patient
patient_freq = base.groupBy("patient_nik").agg(
    count("claim_id").alias("patient_frequency_risk")
)

base = base.join(patient_freq, "patient_nik", "left")

# Calculate days between claims (fraud indicator)
patient_window = Window.partitionBy("patient_nik").orderBy("visit_date")

base = base.withColumn(
    "days_since_last_claim",
    datediff(col("visit_date"), lag("visit_date", 1).over(patient_window))
)

# Flag suspicious frequency (claims too close together)
base = base.withColumn(
    "suspicious_frequency_flag",
    when(col("days_since_last_claim") < 7, 1)  # Less than 1 week
    .otherwise(0)
)

print("✓ Patient frequency features created")

# ================================================================
# 11. GROUND TRUTH LABELS (LEARN FROM HISTORY)
# ================================================================
print("\n[11/12] Creating ground truth labels from historical data...")

# Human reviewer decision (from claim status)
base = base.withColumn(
    "human_label",
    when(col("status") == "declined", 1)    # Declined = Fraud
    .when(col("status") == "approved", 0)   # Approved = Legitimate
    .otherwise(None)                         # Pending/Unknown = No label
)

# Rule-based fraud flag (for validation)
base = base.withColumn(
    "rule_violation_flag",
    when(col("mismatch_count") > 0, 1)              # Clinical mismatch
    .when(col("biaya_anomaly_score") >= 3, 1)       # Cost anomaly
    .when(col("patient_frequency_risk") > 15, 1)    # High frequency
    .when(col("suspicious_frequency_flag") == 1, 1) # Claims too close
    .otherwise(0)
)

# Final label: Prioritize human decision, fallback to rules
base = base.withColumn(
    "final_label",
    when(col("human_label").isNotNull(), col("human_label"))
    .otherwise(col("rule_violation_flag"))
)

print("✓ Ground truth labels created")

# ================================================================
# 12. SELECT FINAL FEATURES WITH EXPLICIT CASTING
# ================================================================
print("\n[12/12] Selecting final feature set with data type casting...")

base = base.withColumn("created_at", current_timestamp())

# Cast all columns explicitly to avoid Iceberg data type errors
feature_df = base.select(
    col("claim_id").cast("bigint"),
    col("patient_nik").cast("string"),
    col("patient_name").cast("string"),
    col("patient_gender").cast("string"),
    col("patient_dob").cast("date"),
    col("patient_age").cast("int"),
    col("visit_date").cast("date"),
    col("visit_year").cast("int"),
    col("visit_month").cast("int"),
    col("visit_day").cast("int"),
    col("visit_type").cast("string"),
    col("doctor_name").cast("string"),
    col("department").cast("string"),
    col("icd10_primary_code").cast("string"),
    col("icd10_primary_desc").cast("string"),
    col("procedures_icd9_codes"),  # Arrays don't need casting
    col("procedures_icd9_descs"),
    col("drug_codes"),
    col("drug_names"),
    col("vitamin_names"),
    col("total_procedure_cost").cast("double"),
    col("total_drug_cost").cast("double"),
    col("total_vitamin_cost").cast("double"),
    col("total_claim_amount").cast("double"),
    col("diagnosis_procedure_score").cast("double"),
    col("diagnosis_drug_score").cast("double"),
    col("diagnosis_vitamin_score").cast("double"),
    col("procedure_mismatch_flag").cast("int"),
    col("drug_mismatch_flag").cast("int"),
    col("vitamin_mismatch_flag").cast("int"),
    col("mismatch_count").cast("int"),
    col("patient_frequency_risk").cast("bigint"),
    when(col("days_since_last_claim").isNull(), lit(None))
        .otherwise(col("days_since_last_claim")).cast("int").alias("days_since_last_claim"),
    col("suspicious_frequency_flag").cast("int"),
    col("biaya_anomaly_score").cast("int"),
    col("rule_violation_flag").cast("int"),
    col("human_label").cast("int"),
    col("final_label").cast("int"),
    col("status").cast("string"),
    col("created_at").cast("timestamp")
)

print("✓ Data types cast successfully")

# Save to Iceberg with overwrite mode and schema enforcement
print("\nSaving to Iceberg curated table...")

# Drop table if exists to avoid schema conflicts
spark.sql("DROP TABLE IF EXISTS iceberg_curated.claim_feature_set")

# Write with explicit schema
feature_df.write.format("iceberg") \
    .partitionBy("visit_year", "visit_month") \
    .mode("overwrite") \
    .option("write.format.default", "parquet") \
    .option("write.parquet.compression-codec", "snappy") \
    .saveAsTable("iceberg_curated.claim_feature_set")

print("✓ Feature set saved to: iceberg_curated.claim_feature_set")

# ================================================================
# 13. DATA QUALITY REPORT
# ================================================================
print("\n" + "=" * 80)
print("ETL COMPLETE - DATA QUALITY REPORT")
print("=" * 80)

# Cache untuk efisiensi
feature_df.cache()

# Overall statistics
total_processed = feature_df.count()
fraud_count = feature_df.filter(col("final_label") == 1).count()
non_fraud_count = total_processed - fraud_count

print(f"\n📊 Dataset Statistics:")
print(f"  Total claims processed: {total_processed:,}")
print(f"  Fraud claims: {fraud_count:,} ({fraud_count/total_processed*100:.1f}%)")
print(f"  Legitimate claims: {non_fraud_count:,} ({non_fraud_count/total_processed*100:.1f}%)")

# Label source breakdown - FIXED
print(f"\n📋 Label Source Distribution:")
human_labeled = feature_df.filter(col("human_label").isNotNull()).count()
rule_labeled = total_processed - human_labeled  # FIX: Calculate correctly

print(f"  Human reviewed: {human_labeled:,} ({human_labeled/total_processed*100:.1f}%)")
print(f"  Rule-based only: {rule_labeled:,} ({rule_labeled/total_processed*100:.1f}%)")

# Breakdown by status
print(f"\n📝 Breakdown by Review Status:")
status_dist = feature_df.groupBy("status").count().collect()
for row in status_dist:
    pct = row['count'] / total_processed * 100
    print(f"  {row['status']}: {row['count']:,} ({pct:.1f}%)")

print("\n" + "=" * 80)
print("✓ Feature engineering complete - Ready for model training")
print("=" * 80)

spark.stop()