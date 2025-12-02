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
from pyspark.sql.functions import udf


# Import centralized config
project_root = os.getcwd()
sys.path.insert(0, os.path.join(os.getcwd(), "ETL"))
from config import COMPAT_RULES, COST_THRESHOLDS

print("=" * 80)
print("FRAUD DETECTION ETL - FEATURE ENGINEERING PIPELINE")
print("=" * 80)

# ================================================================
# 1. CONNECT TO SPARK
# ================================================================
print("\n[1/10] Connecting to Spark...")
conn = cmldata.get_connection("CDP-MSI")
spark = conn.get_spark_session()
print(f"✓ Spark Application ID: {spark.sparkContext.applicationId}")

# ================================================================
# 2. LOAD RAW TABLES
# ================================================================
print("\n[2/10] Loading raw tables from Iceberg...")
hdr = spark.sql("SELECT * FROM iceberg_raw.claim_header_raw")
diag = spark.sql("SELECT * FROM iceberg_raw.claim_diagnosis_raw")
proc = spark.sql("SELECT * FROM iceberg_raw.claim_procedure_raw")
drug = spark.sql("SELECT * FROM iceberg_raw.claim_drug_raw")
vit = spark.sql("SELECT * FROM iceberg_raw.claim_vitamin_raw")

print(f"✓ Loaded {hdr.count():,} claims")

# ================================================================
# 3. PRIMARY DIAGNOSIS
# ================================================================
print("\n[3/10] Extracting primary diagnosis...")
diag_primary = (
    diag.where(col("is_primary") == 1)
        .groupBy("claim_id")
        .agg(
            first("icd10_code").alias("icd10_primary_code"),
            first("icd10_description").alias("icd10_primary_desc")
        )
)

# ================================================================
# 4. AGGREGATIONS
# ================================================================
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

# ================================================================
# 5. JOIN ALL DATA
# ================================================================
print("\n[5/10] Joining all tables...")
base = (
    hdr.join(diag_primary, "claim_id", "left")
       .join(proc_agg, "claim_id", "left")
       .join(drug_agg, "claim_id", "left")
       .join(vit_agg, "claim_id", "left")
)

# ================================================================
# 6. BASIC FEATURES (DATE, AGE, FLAGS)
# ================================================================
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

# ================================================================
# 7. CLINICAL COMPATIBILITY CHECKING (KEY FEATURE!)
# ================================================================
print("\n[7/10] Checking clinical compatibility...")

# Create broadcast variable for COMPAT_RULES
compat_broadcast = spark.sparkContext.broadcast(COMPAT_RULES)

@udf(returnType=DoubleType())
def compute_procedure_compatibility(icd10, procedures):
    """Check if procedures are compatible with diagnosis"""
    if not icd10 or not procedures:
        return 0.0
    
    rules = compat_broadcast.value.get(icd10)
    if not rules:
        return 0.5  # Unknown diagnosis
    
    allowed_procedures = rules.get("procedures", [])
    if not allowed_procedures:
        return 0.5
    
    # Calculate match ratio
    matches = sum(1 for p in procedures if p in allowed_procedures)
    return float(matches) / len(procedures)

@udf(returnType=DoubleType())
def compute_drug_compatibility(icd10, drugs):
    """Check if drugs are compatible with diagnosis"""
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

@udf(returnType=DoubleType())
def compute_vitamin_compatibility(icd10, vitamins):
    """Check if vitamins are compatible with diagnosis"""
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

# ================================================================
# 8. MISMATCH FLAGS (BINARY INDICATORS)
# ================================================================
print("\n[8/10] Creating mismatch flags...")
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
    col("procedure_mismatch_flag") + col("drug_mismatch_flag") + col("vitamin_mismatch_flag")
)

# ================================================================
# 9. COST ANOMALY DETECTION (STATISTICAL)
# ================================================================
print("\n[9/10] Detecting cost anomalies...")

# Calculate z-score for costs per diagnosis
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
    when(col("cost_zscore") > 3, 4)      # Extreme outlier
    .when(col("cost_zscore") > 2, 3)     # High outlier
    .when(col("cost_zscore") > 1, 2)     # Moderate
    .otherwise(1)                         # Normal
)

# Drop intermediate columns
base = base.drop("diagnosis_avg_cost", "diagnosis_stddev_cost", "cost_zscore")

# ================================================================
# 10. PATIENT FREQUENCY RISK
# ================================================================
print("\n[10/10] Calculating patient frequency...")
patient_freq = base.groupBy("patient_nik").agg(
    count("claim_id").alias("patient_frequency_risk")
)

base = base.join(patient_freq, "patient_nik", "left")

# ================================================================
# 11. FINAL LABEL (GROUND TRUTH)
# ================================================================
print("\nCreating final labels...")

# Human label from approval status
base = base.withColumn(
    "human_label",
    when(col("status") == "declined", 1)
    .when(col("status") == "approved", 0)
    .otherwise(None)
)

# Rule-based flag
base = base.withColumn(
    "rule_violation_flag",
    when(col("mismatch_count") > 0, 1)
    .when(col("biaya_anomaly_score") >= 3, 1)
    .when(col("patient_frequency_risk") > 10, 1)
    .otherwise(0)
)

# Final label: prioritize human label, fallback to rules
base = base.withColumn(
    "final_label",
    when(col("human_label").isNotNull(), col("human_label"))
    .otherwise(col("rule_violation_flag"))
)

# ================================================================
# 12. SELECT FINAL FEATURES
# ================================================================
base = base.withColumn("created_at", current_timestamp())

final_columns = [
    # Identifiers
    "claim_id",
    "patient_nik",
    "patient_name",
    "patient_gender",
    "patient_dob",
    "patient_age",
    
    # Visit info
    "visit_date",
    "visit_year",
    "visit_month",
    "visit_day",
    "visit_type",
    "doctor_name",
    "department",
    
    # Diagnosis
    "icd10_primary_code",
    "icd10_primary_desc",
    
    # Raw arrays (for reference)
    "procedures_icd9_codes",
    "procedures_icd9_descs",
    "drug_codes",
    "drug_names",
    "vitamin_names",
    
    # Costs
    "total_procedure_cost",
    "total_drug_cost",
    "total_vitamin_cost",
    "total_claim_amount",
    
    # Clinical compatibility scores
    "diagnosis_procedure_score",
    "diagnosis_drug_score",
    "diagnosis_vitamin_score",
    
    # Mismatch flags
    "procedure_mismatch_flag",
    "drug_mismatch_flag",
    "vitamin_mismatch_flag",
    "mismatch_count",
    
    # Risk features
    "patient_frequency_risk",
    "biaya_anomaly_score",
    
    # Labels
    "rule_violation_flag",
    "human_label",
    "final_label",
    "created_at"
]

feature_df = base.select(*final_columns)

# ================================================================
# 13. SAVE TO ICEBERG
# ================================================================
print("\nSaving to Iceberg curated table...")
feature_df.write.format("iceberg") \
    .partitionBy("visit_year", "visit_month") \
    .mode("overwrite") \
    .saveAsTable("iceberg_curated.claim_feature_set")

# ================================================================
# 14. DATA QUALITY REPORT
# ================================================================
print("\n" + "=" * 80)
print("ETL COMPLETE - DATA QUALITY REPORT")
print("=" * 80)

total_claims = feature_df.count()
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

print("\n" + "=" * 80)
print("Feature set ready for training at: iceberg_curated.claim_feature_set")
print("=" * 80)

spark.stop()