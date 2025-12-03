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
    datediff, lag, max as spark_max
)
from pyspark.sql.window import Window
from pyspark.sql.types import DoubleType, IntegerType, StringType
from pyspark.sql import udf

# Import centralized config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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
# 3. EXTRACT PRIMARY DIAGNOSIS
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
# 5. JOIN ALL DATA
# ================================================================
print("\n[5/12] Joining all tables...")
base = (
    hdr.join(diag_primary, "claim_id", "left")
       .join(proc_agg, "claim_id", "left")
       .join(drug_agg, "claim_id", "left")
       .join(vit_agg, "claim_id", "left")
)

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
# 7. CLINICAL COMPATIBILITY CHECKING
# This is the CORE FEATURE for fraud detection
# ================================================================
print("\n[7/12] Checking clinical compatibility (CRITICAL FEATURE)...")

# Broadcast COMPAT_RULES for efficient UDF execution
compat_broadcast = spark.sparkContext.broadcast(COMPAT_RULES)

@udf(returnType=DoubleType())
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

@udf(returnType=DoubleType())
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

@udf(returnType=DoubleType())
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
# 12. SELECT FINAL FEATURES
# ================================================================
print("\n[12/12] Selecting final feature set...")

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
    
    # Raw arrays (for reference/audit)
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
    
    # Clinical compatibility scores (KEY FEATURES)
    "diagnosis_procedure_score",
    "diagnosis_drug_score",
    "diagnosis_vitamin_score",
    
    # Mismatch flags (FRAUD INDICATORS)
    "procedure_mismatch_flag",
    "drug_mismatch_flag",
    "vitamin_mismatch_flag",
    "mismatch_count",
    
    # Risk features
    "patient_frequency_risk",
    "days_since_last_claim",
    "suspicious_frequency_flag",
    "biaya_anomaly_score",
    
    # Labels (GROUND TRUTH)
    "rule_violation_flag",
    "human_label",
    "final_label",
    
    # Metadata
    "status",
    "created_at"
]

feature_df = base.select(*final_columns)

# ================================================================
# 13. SAVE TO ICEBERG (PARTITIONED FOR PERFORMANCE)
# ================================================================
print("\nSaving to Iceberg curated table...")

feature_df.write.format("iceberg") \
    .partitionBy("visit_year", "visit_month") \
    .mode("overwrite") \
    .saveAsTable("iceberg_curated.claim_feature_set")

print("✓ Feature set saved to: iceberg_curated.claim_feature_set")

# ================================================================
# 14. DATA QUALITY REPORT
# ================================================================
print("\n" + "=" * 80)
print("ETL COMPLETE - DATA QUALITY REPORT")
print("=" * 80)

# Overall statistics
total_processed = feature_df.count()
fraud_count = feature_df.filter(col("final_label") == 1).count()
non_fraud_count = total_processed - fraud_count

print(f"\n📊 Dataset Statistics:")
print(f"  Total claims processed: {total_processed:,}")
print(f"  Fraud claims: {fraud_count:,} ({fraud_count/total_processed*100:.1f}%)")
print(f"  Legitimate claims: {non_fraud_count:,} ({non_fraud_count/total_processed*100:.1f}%)")

# Label source breakdown
label_breakdown = feature_df.groupBy("human_label", "rule_violation_flag").count().collect()
print(f"\n📋 Label Source Distribution:")
human_labeled = feature_df.filter(col("human_label").isNotNull()).count()
rule_labeled = feature_df.filter(col("human_label").isNull()).count()
print(f"  Human reviewed: {human_labeled:,} ({human_labeled/total_processed*100:.1f}%)")
print(f"  Rule-based: {rule_labeled:,} ({rule_labeled/total_processed*100:.1f}%)")

# Mismatch distribution
print(f"\n🚨 Clinical Mismatch Distribution:")
mismatch_dist = feature_df.groupBy("mismatch_count").count().orderBy("mismatch_count").collect()
for row in mismatch_dist:
    pct = row['count'] / total_processed * 100
    print(f"  {row['mismatch_count']} mismatches: {row['count']:,} ({pct:.1f}%)")

# Cost anomaly distribution
print(f"\n💰 Cost Anomaly Distribution:")
anomaly_dist = feature_df.groupBy("biaya_anomaly_score").count().orderBy("biaya_anomaly_score").collect()
for row in anomaly_dist:
    pct = row['count'] / total_processed * 100
    severity = ["Normal", "Normal", "Moderate", "High", "Extreme"][row['biaya_anomaly_score'] - 1]
    print(f"  Level {row['biaya_anomaly_score']} ({severity}): {row['count']:,} ({pct:.1f}%)")

# Top diagnoses
print(f"\n🏥 Top 10 Diagnoses:")
top_dx = feature_df.groupBy("icd10_primary_code", "icd10_primary_desc") \
                   .count() \
                   .orderBy(col("count").desc()) \
                   .limit(10) \
                   .collect()
for row in top_dx:
    print(f"  {row['icd10_primary_code']}: {row['icd10_primary_desc']} - {row['count']:,} claims")

# Fraud rate by department
print(f"\n🏢 Fraud Rate by Department:")
dept_fraud = feature_df.groupBy("department") \
                       .agg(
                           count("*").alias("total"),
                           spark_sum(col("final_label")).alias("fraud")
                       ) \
                       .withColumn("fraud_rate", col("fraud") / col("total") * 100) \
                       .orderBy(col("fraud_rate").desc()) \
                       .collect()
for row in dept_fraud:
    print(f"  {row['department']}: {row['fraud_rate']:.1f}% ({row['fraud']}/{row['total']})")

print("\n" + "=" * 80)
print("✓ Feature engineering complete - Ready for model training")
print("=" * 80)

spark.stop()