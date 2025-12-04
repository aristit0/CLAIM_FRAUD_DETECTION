#!/usr/bin/env python3
"""
Production ETL Pipeline for Fraud Detection - IMPROVED VERSION
Features:
- Dynamic clinical rules loading from Iceberg
- Statistical cost anomaly detection with minimum sample size
- Suspicious duplicate detection
- Temporal data integrity checks
- Comprehensive data quality reporting

Version: 2.0
Date: December 2024
"""

import sys
import os
import cml.data_v1 as cmldata
from pyspark.sql import functions as F
from pyspark.sql.functions import (
    col, lit, when, collect_list, first, year, month, dayofmonth,
    current_timestamp, size, array, count, sum as spark_sum, avg, stddev,
    datediff, lag, max as spark_max, min as spark_min, coalesce, explode, 
    array_intersect, row_number, countDistinct, expr
)
from pyspark.sql.window import Window
from pyspark.sql.types import DoubleType, IntegerType, StringType, ArrayType
from datetime import datetime

print("=" * 80)
print("FRAUD DETECTION ETL - PRODUCTION PIPELINE v2.0")
print("Features: Temporal Validation + Dynamic Rules + Quality Checks")
print("=" * 80)

# ================================================================
# 1. CONNECT TO SPARK
# ================================================================
print("\n[Step 1/15] Connecting to Spark...")
conn = cmldata.get_connection("CDP-MSI")
spark = conn.get_spark_session()

# Enable adaptive query execution for better performance
spark.conf.set("spark.sql.adaptive.enabled", "true")
spark.conf.set("spark.sql.adaptive.coalescePartitions.enabled", "true")

print(f"✓ Spark Application ID: {spark.sparkContext.applicationId}")
print(f"✓ Spark version: {spark.version}")

# ================================================================
# 2. LOAD REFERENCE TABLES (CLINICAL RULES)
# ================================================================
print("\n[Step 2/15] Loading clinical reference tables from Iceberg...")

try:
    # Load clinical rules
    ref_dx_drug = spark.sql("SELECT * FROM iceberg_ref.clinical_rule_dx_drug").cache()
    ref_dx_proc = spark.sql("SELECT * FROM iceberg_ref.clinical_rule_dx_procedure").cache()
    ref_dx_vit = spark.sql("SELECT * FROM iceberg_ref.clinical_rule_dx_vitamin").cache()
    
    # Load master tables
    master_icd10 = spark.sql("SELECT * FROM iceberg_ref.master_icd10").cache()
    master_icd9 = spark.sql("SELECT * FROM iceberg_ref.master_icd9").cache()
    master_drug = spark.sql("SELECT * FROM iceberg_ref.master_drug").cache()
    master_vitamin = spark.sql("SELECT * FROM iceberg_ref.master_vitamin").cache()
    
    print(f"✓ Clinical rules loaded:")
    print(f"  - Diagnosis → Drug rules: {ref_dx_drug.count():,}")
    print(f"  - Diagnosis → Procedure rules: {ref_dx_proc.count():,}")
    print(f"  - Diagnosis → Vitamin rules: {ref_dx_vit.count():,}")
    
    print(f"✓ Master data loaded:")
    print(f"  - ICD-10 codes: {master_icd10.count()}")
    print(f"  - ICD-9 codes: {master_icd9.count()}")
    print(f"  - Drug codes: {master_drug.count()}")
    print(f"  - Vitamins: {master_vitamin.count()}")

except Exception as e:
    print(f"✗ Error loading reference tables: {e}")
    print("  Falling back to empty rules...")
    ref_dx_drug = spark.createDataFrame([], "icd10_code STRING, drug_code STRING")
    ref_dx_proc = spark.createDataFrame([], "icd10_code STRING, icd9_code STRING")
    ref_dx_vit = spark.createDataFrame([], "icd10_code STRING, vitamin_name STRING")

# Create lookup dictionaries for compatibility checking
ref_proc_allowed = (
    ref_dx_proc
    .groupBy("icd10_code")
    .agg(collect_list("icd9_code").alias("allowed_procedures"))
)

ref_drug_allowed = (
    ref_dx_drug
    .groupBy("icd10_code")
    .agg(collect_list("drug_code").alias("allowed_drugs"))
)

ref_vit_allowed = (
    ref_dx_vit
    .groupBy("icd10_code")
    .agg(collect_list("vitamin_name").alias("allowed_vitamins"))
)

print("✓ Clinical compatibility lookup tables prepared")

# ================================================================
# 3. LOAD RAW TABLES
# ================================================================
print("\n[Step 3/15] Loading raw claim tables from Iceberg...")

hdr = spark.sql("SELECT * FROM iceberg_raw.claim_header_raw")
diag = spark.sql("SELECT * FROM iceberg_raw.claim_diagnosis_raw")
proc = spark.sql("SELECT * FROM iceberg_raw.claim_procedure_raw")
drug = spark.sql("SELECT * FROM iceberg_raw.claim_drug_raw")
vit = spark.sql("SELECT * FROM iceberg_raw.claim_vitamin_raw")

total_claims = hdr.count()
print(f"✓ Loaded {total_claims:,} claims from raw tables")

# Check date range
date_stats = hdr.agg(
    spark_min("visit_date").alias("min_date"),
    spark_max("visit_date").alias("max_date")
).collect()[0]

print(f"  Date range: {date_stats['min_date']} to {date_stats['max_date']}")

# ================================================================
# 4. EXTRACT PRIMARY DIAGNOSIS
# ================================================================
print("\n[Step 4/15] Extracting primary diagnosis...")

diag_primary = (
    diag.where(col("is_primary") == 1)
        .groupBy("claim_id")
        .agg(
            first("icd10_code").alias("icd10_primary_code"),
            first("icd10_description").alias("icd10_primary_desc")
        )
)

# Enrich with master ICD-10 description if missing
diag_primary = diag_primary.join(
    master_icd10.select(
        col("code").alias("icd10_master_code"),
        col("description").alias("icd10_master_desc")
    ),
    diag_primary.icd10_primary_code == col("icd10_master_code"),
    "left"
).withColumn(
    "icd10_primary_desc",
    when(col("icd10_primary_desc").isNull(), col("icd10_master_desc"))
    .otherwise(col("icd10_primary_desc"))
).drop("icd10_master_code", "icd10_master_desc")

diag_count = diag_primary.count()
print(f"✓ Extracted primary diagnosis for {diag_count:,} claims")

# ================================================================
# 5. AGGREGATE PROCEDURES, DRUGS, VITAMINS
# ================================================================
print("\n[Step 5/15] Aggregating medical items...")

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

print(f"✓ Aggregated procedures: {proc_agg.count():,}")
print(f"✓ Aggregated drugs: {drug_agg.count():,}")
print(f"✓ Aggregated vitamins: {vit_agg.count():,}")

# ================================================================
# 6. JOIN ALL DATA
# ================================================================
print("\n[Step 6/15] Joining all tables...")

base = (
    hdr.join(diag_primary, "claim_id", "left")
       .join(proc_agg, "claim_id", "left")
       .join(drug_agg, "claim_id", "left")
       .join(vit_agg, "claim_id", "left")
)

print(f"✓ Joined data: {base.count():,} claims")

# ================================================================
# 7. DATA QUALITY FIXES
# ================================================================
print("\n[Step 7/15] Applying data quality fixes...")

# 7.1 Remove duplicates
print("  Removing duplicates by claim_id...")
window_spec = Window.partitionBy("claim_id").orderBy("visit_date")
base = base.withColumn("row_num", row_number().over(window_spec)) \
           .filter(col("row_num") == 1) \
           .drop("row_num")

claims_after_dedup = base.count()
duplicates_removed = total_claims - claims_after_dedup
print(f"  ✓ Removed {duplicates_removed:,} duplicate claims")
print(f"  ✓ Remaining: {claims_after_dedup:,} claims")

# 7.2 Fill missing diagnosis with "UNKNOWN"
print("  Handling missing diagnosis...")
missing_diagnosis = base.filter(col("icd10_primary_code").isNull()).count()
if missing_diagnosis > 0:
    print(f"  ⚠ Found {missing_diagnosis:,} claims without primary diagnosis")
    base = base.withColumn(
        "icd10_primary_code",
        when(col("icd10_primary_code").isNull(), lit("UNKNOWN"))
        .otherwise(col("icd10_primary_code"))
    ).withColumn(
        "icd10_primary_desc",
        when(col("icd10_primary_desc").isNull(), lit("Unknown diagnosis"))
        .otherwise(col("icd10_primary_desc"))
    )
    print(f"  ✓ Filled with 'UNKNOWN'")
else:
    print(f"  ✓ No missing diagnosis found")

# 7.3 Ensure all arrays are not null
print("  Fixing null arrays...")
base = base.withColumn(
    "procedures_icd9_codes",
    coalesce(col("procedures_icd9_codes"), array())
).withColumn(
    "procedures_icd9_descs",
    coalesce(col("procedures_icd9_descs"), array())
).withColumn(
    "drug_codes",
    coalesce(col("drug_codes"), array())
).withColumn(
    "drug_names",
    coalesce(col("drug_names"), array())
).withColumn(
    "vitamin_names",
    coalesce(col("vitamin_names"), array())
)

print("✓ Data quality fixes applied")

# ================================================================
# 8. DETECT SUSPICIOUS DUPLICATES
# ================================================================
print("\n[Step 8/15] Detecting suspicious duplicates...")

# Duplicates = same patient, same date, same diagnosis, same cost
duplicate_window = Window.partitionBy(
    "patient_nik", "visit_date", "icd10_primary_code", "total_claim_amount"
)

base = base.withColumn(
    "suspicious_duplicate_count",
    count("*").over(duplicate_window)
).withColumn(
    "suspicious_duplicate_flag",
    when(col("suspicious_duplicate_count") > 1, 1).otherwise(0)
)

suspicious_dups = base.filter(col("suspicious_duplicate_flag") == 1).count()
print(f"  ⚠ Found {suspicious_dups:,} suspicious duplicate claims")
print(f"  ✓ Flagged for review")

# ================================================================
# 9. BASIC FEATURES (DATE, AGE, FLAGS)
# ================================================================
print("\n[Step 9/15] Creating basic features...")

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

print("✓ Basic features created")

# ================================================================
# 10. CLINICAL COMPATIBILITY CHECKING (DYNAMIC RULES)
# ================================================================
print("\n[Step 10/15] Checking clinical compatibility using dynamic rules...")

# Join with procedure rules
base_with_proc_rules = base.join(
    ref_proc_allowed,
    base.icd10_primary_code == ref_proc_allowed.icd10_code,
    "left"
)

# Calculate procedure compatibility score
base_with_proc_rules = base_with_proc_rules.withColumn(
    "diagnosis_procedure_score",
    when(col("allowed_procedures").isNull(), lit(0.5))  # Unknown diagnosis
    .when(size("procedures_icd9_codes") == 0, lit(0.0))  # No procedures
    .otherwise(
        size(array_intersect(col("procedures_icd9_codes"), col("allowed_procedures"))) / 
        size("procedures_icd9_codes").cast(DoubleType())
    )
).drop("icd10_code", "allowed_procedures")

# Join with drug rules
base_with_drug_rules = base_with_proc_rules.join(
    ref_drug_allowed,
    base_with_proc_rules.icd10_primary_code == ref_drug_allowed.icd10_code,
    "left"
)

# Calculate drug compatibility score
base_with_drug_rules = base_with_drug_rules.withColumn(
    "diagnosis_drug_score",
    when(col("allowed_drugs").isNull(), lit(0.5))  # Unknown diagnosis
    .when(size("drug_codes") == 0, lit(0.0))  # No drugs
    .otherwise(
        size(array_intersect(col("drug_codes"), col("allowed_drugs"))) / 
        size("drug_codes").cast(DoubleType())
    )
).drop("icd10_code", "allowed_drugs")

# Join with vitamin rules
base_with_vit_rules = base_with_drug_rules.join(
    ref_vit_allowed,
    base_with_drug_rules.icd10_primary_code == ref_vit_allowed.icd10_code,
    "left"
)

# Calculate vitamin compatibility score
base = base_with_vit_rules.withColumn(
    "diagnosis_vitamin_score",
    when(col("allowed_vitamins").isNull(), lit(0.5))  # Unknown diagnosis
    .when(size("vitamin_names") == 0, lit(0.0))  # No vitamins
    .otherwise(
        size(array_intersect(col("vitamin_names"), col("allowed_vitamins"))) / 
        size("vitamin_names").cast(DoubleType())
    )
).drop("icd10_code", "allowed_vitamins")

print("✓ Clinical compatibility scores computed from dynamic rules")

# ================================================================
# 11. MISMATCH FLAGS
# ================================================================
print("\n[Step 11/15] Creating mismatch flags...")

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

mismatch_stats = base.groupBy("mismatch_count").count().orderBy("mismatch_count").collect()
print("  Mismatch distribution:")
for row in mismatch_stats:
    print(f"    {row['mismatch_count']} mismatches: {row['count']:,} claims")

print("✓ Mismatch flags created")

# ================================================================
# 12. COST ANOMALY DETECTION (IMPROVED)
# ================================================================
print("\n[Step 12/15] Detecting cost anomalies with statistical validation...")

# Count claims per diagnosis
diagnosis_counts = base.groupBy("icd10_primary_code").agg(
    count("*").alias("dx_sample_size")
)

# Join with base
base = base.join(diagnosis_counts, "icd10_primary_code", "left")

# Calculate z-score only if sample size >= 30
diagnosis_window = Window.partitionBy("icd10_primary_code")

base = base.withColumn(
    "diagnosis_avg_cost",
    avg("total_claim_amount").over(diagnosis_window)
).withColumn(
    "diagnosis_stddev_cost",
    stddev("total_claim_amount").over(diagnosis_window)
).withColumn(
    "cost_zscore",
    when(col("dx_sample_size") < 30, lit(0))  # Not enough data
    .otherwise(
        (col("total_claim_amount") - col("diagnosis_avg_cost")) /
        when(
            (col("diagnosis_stddev_cost").isNull()) | (col("diagnosis_stddev_cost") == 0), 
            lit(1)
        ).otherwise(col("diagnosis_stddev_cost"))
    )
).withColumn(
    "biaya_anomaly_score",
    when(col("cost_zscore") > 3, 4)  # Extreme
    .when(col("cost_zscore") > 2, 3)  # High
    .when(col("cost_zscore") > 1, 2)  # Moderate
    .otherwise(1)  # Normal
)

# Clean up intermediate columns
base = base.drop("diagnosis_avg_cost", "diagnosis_stddev_cost", "cost_zscore")

anomaly_stats = base.groupBy("biaya_anomaly_score").count().orderBy("biaya_anomaly_score").collect()
print("  Cost anomaly distribution:")
anomaly_labels = ["", "Normal", "Moderate", "High", "Extreme"]
for row in anomaly_stats:
    label = anomaly_labels[int(row['biaya_anomaly_score'])]
    print(f"    Level {row['biaya_anomaly_score']} ({label}): {row['count']:,} claims")

print("✓ Cost anomaly scores computed with sample size validation")

# ================================================================
# 13. PATIENT FREQUENCY RISK
# ================================================================
print("\n[Step 13/15] Calculating patient claim frequency...")

# Count total claims per patient
patient_freq = base.groupBy("patient_nik").agg(
    count("claim_id").alias("patient_frequency_risk")
)

base = base.join(patient_freq, "patient_nik", "left")

# Days since last claim (temporal feature)
patient_window = Window.partitionBy("patient_nik").orderBy("visit_date")

base = base.withColumn(
    "days_since_last_claim",
    datediff(col("visit_date"), lag("visit_date", 1).over(patient_window))
)

# Suspicious frequency flag (< 7 days between claims)
base = base.withColumn(
    "suspicious_frequency_flag",
    when(
        (col("days_since_last_claim").isNotNull()) & 
        (col("days_since_last_claim") < 7), 
        1
    ).otherwise(0)
)

freq_stats = base.groupBy("suspicious_frequency_flag").count().collect()
print(f"  Suspicious frequency claims: {[r['count'] for r in freq_stats if r['suspicious_frequency_flag']==1][0]:,}")

print("✓ Patient frequency features created")

# ================================================================
# 14. GROUND TRUTH LABELS
# ================================================================
print("\n[Step 14/15] Creating ground truth labels from historical data...")

base = base.withColumn(
    "human_label",
    when(col("status") == "declined", 1)
    .when(col("status") == "approved", 0)
    .otherwise(None)
)

base = base.withColumn(
    "rule_violation_flag",
    when(col("mismatch_count") > 0, 1)
    .when(col("biaya_anomaly_score") >= 3, 1)
    .when(col("patient_frequency_risk") > 15, 1)
    .when(col("suspicious_frequency_flag") == 1, 1)
    .when(col("suspicious_duplicate_flag") == 1, 1)
    .otherwise(0)
)

base = base.withColumn(
    "final_label",
    when(col("human_label").isNotNull(), col("human_label"))
    .otherwise(col("rule_violation_flag"))
)

label_dist = base.groupBy("final_label").count().collect()
print("  Label distribution:")
for row in label_dist:
    label = "Fraud" if row['final_label'] == 1 else "Legitimate"
    print(f"    {label}: {row['count']:,} claims")

print("✓ Ground truth labels created")

# ================================================================
# 15. SELECT FINAL FEATURES WITH EXPLICIT CASTING
# ================================================================
print("\n[Step 15/15] Selecting final feature set with data type casting...")

base = base.withColumn("created_at", current_timestamp())

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
    col("procedures_icd9_codes"),
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
    col("suspicious_duplicate_flag").cast("int"),
    col("suspicious_duplicate_count").cast("int"),
    col("dx_sample_size").cast("int"),
    col("biaya_anomaly_score").cast("int"),
    col("rule_violation_flag").cast("int"),
    col("human_label").cast("int"),
    col("final_label").cast("int"),
    col("status").cast("string"),
    col("created_at").cast("timestamp")
)

print("✓ Data types cast successfully")

## ================================================================
# 16. TEMPORAL VALIDATION SPLIT
# ================================================================
print("\n[Step 16/15 BONUS] Creating temporal train/test split...")

from datetime import timedelta

# Get date range for split
date_range = feature_df.agg(
    spark_min("visit_date").alias("min_date"),
    spark_max("visit_date").alias("max_date")
).collect()[0]

min_date = date_range['min_date']
max_date = date_range['max_date']

print(f"  Data date range: {min_date} to {max_date}")
print(f"  Total span: {(max_date - min_date).days} days")

# Calculate 80/20 split date (temporal, no data leakage)
total_days = (max_date - min_date).days
split_days = int(total_days * 0.8)
split_date = min_date + timedelta(days=split_days)

print(f"  Calculated split date: {split_date}")
print(f"  Train period: {min_date} to {split_date}")
print(f"  Test period: {split_date} to {max_date}")

# Add split indicator
feature_df = feature_df.withColumn(
    "temporal_split",
    when(col("visit_date") <= lit(split_date), "train").otherwise("test")
)

# Validate split
split_stats = feature_df.groupBy("temporal_split").agg(
    count("*").alias("total"),
    spark_min("visit_date").alias("min_visit"),
    spark_max("visit_date").alias("max_visit"),
    spark_sum(col("final_label")).alias("fraud_count")
).collect()

print(f"\n  ✓ Split validation:")
for row in split_stats:
    split_name = row['temporal_split'].upper()
    total = row['total']
    fraud_count_split = row['fraud_count']
    fraud_pct = (fraud_count_split / total * 100) if total > 0 else 0
    min_visit = row['min_visit']
    max_visit = row['max_visit']
    
    print(f"    {split_name}:")
    print(f"      Records: {total:,}")
    print(f"      Fraud: {fraud_count_split:,} ({fraud_pct:.1f}%)")
    print(f"      Date range: {min_visit} to {max_visit}")

# Verify no data leakage
train_max = feature_df.filter(col("temporal_split") == "train").agg(spark_max("visit_date")).collect()[0][0]
test_min = feature_df.filter(col("temporal_split") == "test").agg(spark_min("visit_date")).collect()[0][0]

if train_max < test_min:
    print(f"\n  ✓ NO DATA LEAKAGE: Train max ({train_max}) < Test min ({test_min})")
    print(f"    Gap between splits: {(test_min - train_max).days} days")
else:
    print(f"\n  ✗ WARNING: DATA LEAKAGE DETECTED!")
    print(f"    Train max ({train_max}) >= Test min ({test_min})")

print("✓ Temporal split indicator added with validation")

# ================================================================
# 17. SAVE TO ICEBERG
# ================================================================
print("\n[Step 17/15 BONUS] Saving to Iceberg curated table...")

spark.sql("DROP TABLE IF EXISTS iceberg_curated.claim_feature_set")

feature_df.write.format("iceberg") \
    .partitionBy("visit_year", "visit_month") \
    .mode("overwrite") \
    .option("write.format.default", "parquet") \
    .option("write.parquet.compression-codec", "snappy") \
    .saveAsTable("iceberg_curated.claim_feature_set")

print("✓ Feature set saved to: iceberg_curated.claim_feature_set")

# ================================================================
# 18. DATA QUALITY REPORT
# ================================================================
print("\n" + "=" * 80)
print("ETL COMPLETE - COMPREHENSIVE DATA QUALITY REPORT")
print("=" * 80)

feature_df.cache()

total_processed = feature_df.count()
fraud_count = feature_df.filter(col("final_label") == 1).count()
non_fraud_count = total_processed - fraud_count

print(f"\n📊 Dataset Statistics:")
print(f"  Total claims processed: {total_processed:,}")
print(f"  Fraud claims: {fraud_count:,} ({fraud_count/total_processed*100:.1f}%)")
print(f"  Legitimate claims: {non_fraud_count:,} ({non_fraud_count/total_processed*100:.1f}%)")

# Label source distribution
human_labeled = feature_df.filter(col("human_label").isNotNull()).count()
rule_labeled = total_processed - human_labeled

print(f"\n📋 Label Source Distribution:")
print(f"  Human reviewed: {human_labeled:,} ({human_labeled/total_processed*100:.1f}%)")
print(f"  Rule-based only: {rule_labeled:,} ({rule_labeled/total_processed*100:.1f}%)")

# Temporal split distribution
print(f"\n📅 Temporal Split Distribution:")
split_stats = feature_df.groupBy("temporal_split").agg(
    count("*").alias("total"),
    spark_sum(col("final_label")).alias("fraud")
).collect()

for row in split_stats:
    split_name = row['temporal_split'].upper()
    total = row['total']
    fraud = row['fraud']
    fraud_pct = fraud / total * 100 if total > 0 else 0
    print(f"  {split_name}: {total:,} claims, {fraud:,} fraud ({fraud_pct:.1f}%)")

# Clinical mismatch distribution
print(f"\n🚨 Clinical Mismatch Distribution:")
mismatch_dist = feature_df.groupBy("mismatch_count").count().orderBy("mismatch_count").collect()
for row in mismatch_dist:
    pct = row['count'] / total_processed * 100
    print(f"  {row['mismatch_count']} mismatches: {row['count']:,} ({pct:.1f}%)")

# Cost anomaly distribution
print(f"\n💰 Cost Anomaly Distribution:")
anomaly_dist = feature_df.groupBy("biaya_anomaly_score").count().orderBy("biaya_anomaly_score").collect()
anomaly_labels = ["", "Normal", "Moderate", "High", "Extreme"]
for row in anomaly_dist:
    pct = row['count'] / total_processed * 100
    severity = anomaly_labels[int(row['biaya_anomaly_score'])]
    print(f"  Level {row['biaya_anomaly_score']} ({severity}): {row['count']:,} ({pct:.1f}%)")

# Top 10 diagnoses
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

# Data quality issues
print(f"\n⚠️  Data Quality Issues Detected:")
print(f"  Suspicious duplicates: {feature_df.filter(col('suspicious_duplicate_flag')==1).count():,}")
print(f"  Suspicious frequency: {feature_df.filter(col('suspicious_frequency_flag')==1).count():,}")
print(f"  Missing diagnosis (filled): {missing_diagnosis:,}")
print(f"  Duplicate claims (removed): {duplicates_removed:,}")

# Sample size validation
print(f"\n📈 Cost Anomaly Sample Size Validation:")
small

# ================================================================
# 18. DATA QUALITY REPORT
# ================================================================
print("\n" + "=" * 80)
print("ETL COMPLETE - COMPREHENSIVE DATA QUALITY REPORT")
print("=" * 80)

feature_df.cache()

total_processed = feature_df.count()
fraud_count = feature_df.filter(col("final_label") == 1).count()
non_fraud_count = total_processed - fraud_count

print(f"\n📊 Dataset Statistics:")
print(f"  Total claims processed: {total_processed:,}")
print(f"  Fraud claims: {fraud_count:,} ({fraud_count/total_processed*100:.1f}%)")
print(f"  Legitimate claims: {non_fraud_count:,} ({non_fraud_count/total_processed*100:.1f}%)")

# Label source distribution
human_labeled = feature_df.filter(col("human_label").isNotNull()).count()
rule_labeled = total_processed - human_labeled

print(f"\n📋 Label Source Distribution:")
print(f"  Human reviewed: {human_labeled:,} ({human_labeled/total_processed*100:.1f}%)")
print(f"  Rule-based only: {rule_labeled:,} ({rule_labeled/total_processed*100:.1f}%)")

# Temporal split distribution
print(f"\n📅 Temporal Split Distribution:")
split_stats = feature_df.groupBy("temporal_split").agg(
    count("*").alias("total"),
    spark_sum(col("final_label")).alias("fraud")
).collect()

for row in split_stats:
    split_name = row['temporal_split'].upper()
    total = row['total']
    fraud = row['fraud']
    fraud_pct = fraud / total * 100 if total > 0 else 0
    print(f"  {split_name}: {total:,} claims, {fraud:,} fraud ({fraud_pct:.1f}%)")

# Clinical mismatch distribution
print(f"\n🚨 Clinical Mismatch Distribution:")
mismatch_dist = feature_df.groupBy("mismatch_count").count().orderBy("mismatch_count").collect()
for row in mismatch_dist:
    pct = row['count'] / total_processed * 100
    print(f"  {row['mismatch_count']} mismatches: {row['count']:,} ({pct:.1f}%)")

# Cost anomaly distribution
print(f"\n💰 Cost Anomaly Distribution:")
anomaly_dist = feature_df.groupBy("biaya_anomaly_score").count().orderBy("biaya_anomaly_score").collect()
anomaly_labels = ["", "Normal", "Moderate", "High", "Extreme"]
for row in anomaly_dist:
    pct = row['count'] / total_processed * 100
    severity = anomaly_labels[int(row['biaya_anomaly_score'])]
    print(f"  Level {row['biaya_anomaly_score']} ({severity}): {row['count']:,} ({pct:.1f}%)")

# Top 10 diagnoses
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

# Data quality issues & Sample size validation (fixed duplicated blocks & NameError)
print(f"\n⚠️  Data Quality Issues Detected:")
suspicious_dup_count = feature_df.filter(col("suspicious_duplicate_flag") == 1).count()
suspicious_freq_count = feature_df.filter(col("suspicious_frequency_flag") == 1).count()
print(f"  Suspicious duplicates: {suspicious_dup_count:,}")
print(f"  Suspicious frequency: {suspicious_freq_count:,}")
print(f"  Missing diagnosis (filled): {missing_diagnosis:,}")
print(f"  Duplicate claims (removed): {duplicates_removed:,}")

# Sample size validation (fixed NameError 'small')
print(f"\n📈 Cost Anomaly Sample Size Validation:")
small_sample = feature_df.filter(col("dx_sample_size") < 30).count()
large_sample = feature_df.filter(col("dx_sample_size") >= 30).count()
print(f"  Claims with sample size < 30 (no z-score): {small_sample:,}")
print(f"  Claims with sample size >= 30 (z-score computed): {large_sample:,}")

print("\n" + "=" * 80)
print("✓ Feature engineering complete - Ready for temporal model training")
print("✓ Clinical rules loaded from Iceberg reference tables")
print("✓ Temporal validation split prepared (80% train / 20% test)")
print("✓ Data quality validated and documented")
print("=" * 80)

# Ensure data-quality counters exist and compute safely
try:
    suspicious_dup_count = int(feature_df.filter(col('suspicious_duplicate_flag') == 1).count())
except Exception:
    suspicious_dup_count = 0

try:
    suspicious_freq_count = int(feature_df.filter(col('suspicious_frequency_flag') == 1).count())
except Exception:
    suspicious_freq_count = 0

# fallback for previously computed vars if missing
missing_diagnosis = globals().get("missing_diagnosis", 0)
duplicates_removed = globals().get("duplicates_removed", 0)
total_processed = globals().get("total_processed", feature_df.count() if 'feature_df' in globals() else 0)
fraud_count = globals().get("fraud_count", feature_df.filter(col("final_label") == 1).count() if 'feature_df' in globals() else 0)
split_date = globals().get("split_date", None)

print(f"\n⚠️  Data Quality Issues Detected:")
print(f"  Suspicious duplicates: {suspicious_dup_count:,}")
print(f"  Suspicious frequency: {suspicious_freq_count:,}")
print(f"  Missing diagnosis (filled): {missing_diagnosis:,}")
print(f"  Duplicate claims (removed): {duplicates_removed:,}")

# Sample size validation
print(f"\n📈 Cost Anomaly Sample Size Validation:")
small_sample = feature_df.filter(col("dx_sample_size") < 30).count() if 'feature_df' in globals() else 0
large_sample = feature_df.filter(col("dx_sample_size") >= 30).count() if 'feature_df' in globals() else 0
print(f"  Claims with sample size < 30 (no z-score): {small_sample:,}")
print(f"  Claims with sample size >= 30 (z-score computed): {large_sample:,}")

print("\n" + "=" * 80)
print("✓ Feature engineering complete - Ready for temporal model training")
print("✓ Clinical rules loaded from Iceberg reference tables")
print("✓ Temporal validation split prepared (80% train / 20% test)")
print("✓ Data quality validated and documented")
print("=" * 80)

# ================================================================
# 19. SAVE METADATA
# ================================================================
print("\n[Step 19/15 BONUS] Saving ETL metadata...")

metadata = {
    "etl_timestamp": datetime.now().isoformat(),
    "total_claims": int(total_processed),
    "fraud_ratio": float(fraud_count / total_processed) if total_processed > 0 else 0.0,
    "temporal_split_date": str(split_date) if split_date is not None else None,
    "data_quality": {
        "duplicates_removed": int(duplicates_removed),
        "missing_diagnosis": int(missing_diagnosis),
        "suspicious_duplicates": int(suspicious_dup_count),
        "suspicious_frequency": int(suspicious_freq_count),
    },
    "feature_counts": {
        "numeric": 16,
        "categorical": 3,
        "arrays": 5,
    }
}

# Save metadata to Iceberg (as JSON string in a single-row table)
metadata_df = spark.createDataFrame(
    [(datetime.now(), str(metadata))],
    ["etl_timestamp", "metadata_json"]
)

spark.sql("DROP TABLE IF EXISTS iceberg_curated.etl_metadata")
metadata_df.write.format("iceberg").mode("overwrite").saveAsTable("iceberg_curated.etl_metadata")

print("✓ ETL metadata saved to: iceberg_curated.etl_metadata")

spark.stop()

print("\n✅ ETL PIPELINE COMPLETED SUCCESSFULLY")