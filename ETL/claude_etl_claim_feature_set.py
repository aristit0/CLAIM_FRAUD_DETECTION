#!/usr/bin/env python3
"""
Production ETL Pipeline for Fraud Detection
Uses Iceberg reference tables for clinical rules
Learns from historical approved/declined claims
"""

import sys
import os
import cml.data_v1 as cmldata
from pyspark.sql import functions as F
from pyspark.sql.functions import (
    col, lit, when, collect_list, first, year, month, dayofmonth,
    current_timestamp, size, array, count, sum as spark_sum, avg, stddev,
    datediff, lag, max as spark_max, coalesce, explode, array_intersect
)
from pyspark.sql.window import Window
from pyspark.sql.types import DoubleType, IntegerType, StringType, ArrayType

print("=" * 80)
print("FRAUD DETECTION ETL - PRODUCTION PIPELINE")
print("Using Iceberg Reference Tables for Clinical Rules")
print("=" * 80)

# ================================================================
# 1. CONNECT TO SPARK
# ================================================================
print("\n[1/13] Connecting to Spark...")
conn = cmldata.get_connection("CDP-MSI")
spark = conn.get_spark_session()
print(f"✓ Spark Application ID: {spark.sparkContext.applicationId}")

# ================================================================
# 2. LOAD REFERENCE TABLES (CLINICAL RULES)
# ================================================================
print("\n[2/13] Loading clinical reference tables...")

# Load clinical rules from Iceberg
ref_dx_drug = spark.sql("SELECT * FROM iceberg_ref.clinical_rule_dx_drug")
ref_dx_proc = spark.sql("SELECT * FROM iceberg_ref.clinical_rule_dx_procedure")
ref_dx_vit = spark.sql("SELECT * FROM iceberg_ref.clinical_rule_dx_vitamin")

# Load master tables
master_icd10 = spark.sql("SELECT * FROM iceberg_ref.master_icd10")
master_icd9 = spark.sql("SELECT * FROM iceberg_ref.master_icd9")
master_drug = spark.sql("SELECT * FROM iceberg_ref.master_drug")
master_vitamin = spark.sql("SELECT * FROM iceberg_ref.master_vitamin")

print(f"✓ Loaded reference tables:")
print(f"  - Clinical rules: {ref_dx_drug.count()} drug rules, {ref_dx_proc.count()} procedure rules, {ref_dx_vit.count()} vitamin rules")
print(f"  - Master data: {master_icd10.count()} ICD-10, {master_icd9.count()} ICD-9, {master_drug.count()} drugs, {master_vitamin.count()} vitamins")

# Create lookup dictionaries as broadcast variables
# For procedures
ref_proc_allowed = (
    ref_dx_proc
    .groupBy("icd10_code")
    .agg(collect_list("icd9_code").alias("allowed_procedures"))
)

# For drugs
ref_drug_allowed = (
    ref_dx_drug
    .groupBy("icd10_code")
    .agg(collect_list("drug_code").alias("allowed_drugs"))
)

# For vitamins
ref_vit_allowed = (
    ref_dx_vit
    .groupBy("icd10_code")
    .agg(collect_list("vitamin_name").alias("allowed_vitamins"))
)

print("✓ Clinical compatibility rules prepared")

# ================================================================
# 3. LOAD RAW TABLES
# ================================================================
print("\n[3/13] Loading raw claim tables from Iceberg...")
hdr = spark.sql("SELECT * FROM iceberg_raw.claim_header_raw")
diag = spark.sql("SELECT * FROM iceberg_raw.claim_diagnosis_raw")
proc = spark.sql("SELECT * FROM iceberg_raw.claim_procedure_raw")
drug = spark.sql("SELECT * FROM iceberg_raw.claim_drug_raw")
vit = spark.sql("SELECT * FROM iceberg_raw.claim_vitamin_raw")

total_claims = hdr.count()
print(f"✓ Loaded {total_claims:,} claims")

# ================================================================
# 4. EXTRACT PRIMARY DIAGNOSIS
# ================================================================
print("\n[4/13] Extracting primary diagnosis...")
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

# ================================================================
# 5. AGGREGATE PROCEDURES, DRUGS, VITAMINS
# ================================================================
print("\n[5/13] Aggregating medical items...")
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
# 6. JOIN ALL DATA
# ================================================================
print("\n[6/13] Joining all tables...")
base = (
    hdr.join(diag_primary, "claim_id", "left")
       .join(proc_agg, "claim_id", "left")
       .join(drug_agg, "claim_id", "left")
       .join(vit_agg, "claim_id", "left")
)

# ================================================================
# 6.5. DATA QUALITY FIXES
# ================================================================
print("\n[6.5/13] Applying data quality fixes...")

# 1. Remove duplicates
from pyspark.sql.functions import row_number

window_spec = Window.partitionBy("claim_id").orderBy("visit_date")
base = base.withColumn("row_num", row_number().over(window_spec)) \
           .filter(col("row_num") == 1) \
           .drop("row_num")

claims_after_dedup = base.count()
print(f"  After deduplication: {claims_after_dedup:,} claims")

# 2. Fill missing diagnosis with "UNKNOWN"
print("  Filling missing diagnosis...")
base = base.withColumn(
    "icd10_primary_code",
    when(col("icd10_primary_code").isNull(), lit("UNKNOWN"))
    .otherwise(col("icd10_primary_code"))
).withColumn(
    "icd10_primary_desc",
    when(col("icd10_primary_desc").isNull(), lit("Unknown diagnosis"))
    .otherwise(col("icd10_primary_desc"))
)

# 3. Ensure all arrays are not null
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
# 7. BASIC FEATURES (DATE, AGE, FLAGS)
# ================================================================
print("\n[7/13] Creating basic features...")
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
# 8. CLINICAL COMPATIBILITY CHECKING (USING REF TABLES!)
# ================================================================
print("\n[8/13] Checking clinical compatibility using reference tables...")

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
        size("procedures_icd9_codes")
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
        size("drug_codes")
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
        size("vitamin_names")
    )
).drop("icd10_code", "allowed_vitamins")

print("✓ Clinical compatibility scores computed from reference tables")

# ================================================================
# 9. MISMATCH FLAGS
# ================================================================
print("\n[9/13] Creating mismatch flags...")

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
# 10. COST ANOMALY DETECTION
# ================================================================
print("\n[10/13] Detecting cost anomalies...")

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
    when(col("cost_zscore") > 3, 4)
    .when(col("cost_zscore") > 2, 3)
    .when(col("cost_zscore") > 1, 2)
    .otherwise(1)
)

base = base.drop("diagnosis_avg_cost", "diagnosis_stddev_cost", "cost_zscore")

print("✓ Cost anomaly scores computed")

# ================================================================
# 11. PATIENT FREQUENCY RISK
# ================================================================
print("\n[11/13] Calculating patient claim frequency...")

patient_freq = base.groupBy("patient_nik").agg(
    count("claim_id").alias("patient_frequency_risk")
)

base = base.join(patient_freq, "patient_nik", "left")

patient_window = Window.partitionBy("patient_nik").orderBy("visit_date")

base = base.withColumn(
    "days_since_last_claim",
    datediff(col("visit_date"), lag("visit_date", 1).over(patient_window))
)

base = base.withColumn(
    "suspicious_frequency_flag",
    when(col("days_since_last_claim") < 7, 1).otherwise(0)
)

print("✓ Patient frequency features created")

# ================================================================
# 12. GROUND TRUTH LABELS
# ================================================================
print("\n[12/13] Creating ground truth labels from historical data...")

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
    .otherwise(0)
)

base = base.withColumn(
    "final_label",
    when(col("human_label").isNotNull(), col("human_label"))
    .otherwise(col("rule_violation_flag"))
)

print("✓ Ground truth labels created")

# ================================================================
# 13. SELECT FINAL FEATURES WITH EXPLICIT CASTING
# ================================================================
print("\n[13/13] Selecting final feature set with data type casting...")

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
    col("biaya_anomaly_score").cast("int"),
    col("rule_violation_flag").cast("int"),
    col("human_label").cast("int"),
    col("final_label").cast("int"),
    col("status").cast("string"),
    col("created_at").cast("timestamp")
)

print("✓ Data types cast successfully")

# Save to Iceberg
print("\nSaving to Iceberg curated table...")

spark.sql("DROP TABLE IF EXISTS iceberg_curated.claim_feature_set")

feature_df.write.format("iceberg") \
    .partitionBy("visit_year", "visit_month") \
    .mode("overwrite") \
    .option("write.format.default", "parquet") \
    .option("write.parquet.compression-codec", "snappy") \
    .saveAsTable("iceberg_curated.claim_feature_set")

print("✓ Feature set saved to: iceberg_curated.claim_feature_set")

# ================================================================
# 14. DATA QUALITY REPORT
# ================================================================
print("\n" + "=" * 80)
print("ETL COMPLETE - DATA QUALITY REPORT")
print("=" * 80)

feature_df.cache()

total_processed = feature_df.count()
fraud_count = feature_df.filter(col("final_label") == 1).count()
non_fraud_count = total_processed - fraud_count

print(f"\n📊 Dataset Statistics:")
print(f"  Total claims processed: {total_processed:,}")
print(f"  Fraud claims: {fraud_count:,} ({fraud_count/total_processed*100:.1f}%)")
print(f"  Legitimate claims: {non_fraud_count:,} ({non_fraud_count/total_processed*100:.1f}%)")

print(f"\n📋 Label Source Distribution:")
human_labeled = feature_df.filter(col("human_label").isNotNull()).count()
rule_labeled = total_processed - human_labeled

print(f"  Human reviewed: {human_labeled:,} ({human_labeled/total_processed*100:.1f}%)")
print(f"  Rule-based only: {rule_labeled:,} ({rule_labeled/total_processed*100:.1f}%)")

print(f"\n🚨 Clinical Mismatch Distribution:")
mismatch_dist = feature_df.groupBy("mismatch_count").count().orderBy("mismatch_count").collect()
for row in mismatch_dist:
    pct = row['count'] / total_processed * 100
    print(f"  {row['mismatch_count']} mismatches: {row['count']:,} ({pct:.1f}%)")

print(f"\n💰 Cost Anomaly Distribution:")
anomaly_dist = feature_df.groupBy("biaya_anomaly_score").count().orderBy("biaya_anomaly_score").collect()
for row in anomaly_dist:
    pct = row['count'] / total_processed * 100
    severity = ["", "Normal", "Moderate", "High", "Extreme"][int(row['biaya_anomaly_score'])]
    print(f"  Level {row['biaya_anomaly_score']} ({severity}): {row['count']:,} ({pct:.1f}%)")

print(f"\n🏥 Top 10 Diagnoses:")
top_dx = feature_df.groupBy("icd10_primary_code", "icd10_primary_desc") \
                   .count() \
                   .orderBy(col("count").desc()) \
                   .limit(10) \
                   .collect()
for row in top_dx:
    print(f"  {row['icd10_primary_code']}: {row['icd10_primary_desc']} - {row['count']:,} claims")

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
print("✓ Clinical rules loaded from Iceberg reference tables")
print("=" * 80)

spark.stop()