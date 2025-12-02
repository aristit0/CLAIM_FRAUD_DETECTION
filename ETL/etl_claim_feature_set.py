#!/usr/bin/env python3

import cml.data_v1 as cmldata
from pyspark.sql import functions as F
from pyspark.sql.functions import (
    col, lit, when, collect_list, first, year, month, dayofmonth, 
    current_timestamp, size, array, count
)
from pyspark.sql.window import Window

print("=== START FRAUD-ENHANCED ETL v6 WITH EXPLICIT MISMATCH FLAGS ===")
print("================================================================")

# CONNECT TO SPARK
conn = cmldata.get_connection("CDP-MSI")
spark = conn.get_spark_session()
print(f"Spark Application Id: {spark.sparkContext.applicationId}")
print("================================================================")

# 2. LOAD RAW TABLES
hdr = spark.sql("SELECT * FROM iceberg_raw.claim_header_raw")
diag = spark.sql("SELECT * FROM iceberg_raw.claim_diagnosis_raw")
proc = spark.sql("SELECT * FROM iceberg_raw.claim_procedure_raw")
drug = spark.sql("SELECT * FROM iceberg_raw.claim_drug_raw")
vit = spark.sql("SELECT * FROM iceberg_raw.claim_vitamin_raw")

# 3. PRIMARY DIAGNOSIS
diag_primary = (
    diag.where(col("is_primary") == 1)
        .groupBy("claim_id")
        .agg(
            first("icd10_code").alias("icd10_primary_code"),
            first("icd10_description").alias("icd10_primary_desc")
        )
)

# 4. AGGREGATIONS (PROCEDURES, DRUGS, VITAMINS)
proc_agg = proc.groupBy("claim_id").agg(
    collect_list("icd9_code").alias("procedures_icd9_codes"),
    collect_list("icd9_description").alias("procedures_icd9_descs"),
    collect_list("cost").alias("procedures_cost")
)
drug_agg = drug.groupBy("claim_id").agg(
    collect_list("drug_code").alias("drug_codes"),
    collect_list("drug_name").alias("drug_names"),
    collect_list("cost").alias("drug_cost")
)
vit_agg = vit.groupBy("claim_id").agg(
    collect_list("vitamin_name").alias("vitamin_names"),
    collect_list("cost").alias("vitamin_cost")
)

# 5. JOIN ALL RAW DATA
base = (
    hdr.join(diag_primary, "claim_id", "left")
       .join(proc_agg, "claim_id", "left")
       .join(drug_agg, "claim_id", "left")
       .join(vit_agg, "claim_id", "left")
)

# 6. DATE + AGE
base = (
    base.withColumn("patient_age",
        when(col("patient_dob").isNull(), None)
        .otherwise(year(col("visit_date")) - year(col("patient_dob")))
    )
    .withColumn("visit_year",  year("visit_date"))
    .withColumn("visit_month", month("visit_date"))
    .withColumn("visit_day",   dayofmonth("visit_date"))
    .withColumn("has_procedure", when(size("procedures_icd9_codes") > 0, 1).otherwise(0))
    .withColumn("has_drug",      when(size("drug_codes") > 0, 1).otherwise(0))
    .withColumn("has_vitamin",   when(size("vitamin_names") > 0, 1).otherwise(0))
)

# 7. CALCULATE PROCEDURE, DRUG, AND VITAMIN MATCHING SCORES
allowed_icd9 = ["A01", "B02", "C03"]  # Sample ICD9 codes
allowed_drug = ["D001", "D002", "D003"]  # Sample drug codes
allowed_vit = ["Vitamin A", "Vitamin B", "Vitamin C"]  # Sample vitamin names

base = base.withColumn(
    "diagnosis_procedure_score",
    when(size(F.array_intersect("procedures_icd9_codes", array(*[lit(code) for code in allowed_icd9]))) > 0, 1.0).otherwise(0.0)
)
base = base.withColumn(
    "diagnosis_drug_score",
    when(size(F.array_intersect("drug_codes", array(*[lit(code) for code in allowed_drug]))) > 0, 1.0).otherwise(0.0)
)
base = base.withColumn(
    "diagnosis_vitamin_score",
    when(size(F.array_intersect("vitamin_names", array(*[lit(name) for name in allowed_vit]))) > 0, 1.0).otherwise(0.0)
)

# 8. EXPLICIT MISMATCH FLAGS (for fraud detection)
base = base.withColumn(
    "procedure_mismatch_flag",
    when(col("diagnosis_procedure_score") == 0, 1).otherwise(0)
)
base = base.withColumn(
    "drug_mismatch_flag",
    when(col("diagnosis_drug_score") == 0, 1).otherwise(0)
)
base = base.withColumn(
    "vitamin_mismatch_flag",
    when(col("diagnosis_vitamin_score") == 0, 1).otherwise(0)
)

# Explicitly calculate mismatch_count after flag columns are created
base = base.withColumn(
    "mismatch_count",
    col("procedure_mismatch_flag") +
    col("drug_mismatch_flag") +
    col("vitamin_mismatch_flag")
)

# 9. CALCULATE BIAYA ANOMALY SCORE (based on claim amount)
base = base.withColumn(
    "biaya_anomaly_score",
    when(col("total_claim_amount") > 5000, 3)  # Example: High claim amount indicates anomaly
    .when(col("total_claim_amount") > 2000, 2)
    .otherwise(1)
)

# 10. CALCULATE PATIENT FREQUENCY RISK (frequency of claims per patient)
patient_freq = base.groupBy("patient_nik").agg(
    count("claim_id").alias("patient_frequency_risk")
)

# Join the frequency back into the base DataFrame
base = base.join(patient_freq, "patient_nik", "left")

# 11. RULE FLAG + FINAL LABEL
base = base.withColumn(
    "rule_violation_flag",
    when(col("mismatch_count") > 0, 1)
    .when(col("biaya_anomaly_score") > 2.5, 1)
    .when(col("patient_frequency_risk") == 1, 1)
    .otherwise(0)
)

base = base.withColumn(
    "human_label",
    when(col("status") == "declined", 1)
    .when(col("status") == "approved", 0)
)

base = base.withColumn(
    "final_label",
    when(col("human_label").isNotNull(), col("human_label"))
    .otherwise(col("rule_violation_flag"))
)

# 12. FINAL SELECT (including cost_per_procedure)
base = base.withColumn("created_at", current_timestamp())

final_cols = [
    # CLAIM + PATIENT
    "claim_id",
    "patient_nik",
    "patient_name",
    "patient_gender",
    "patient_dob",
    "patient_age",
    "visit_date",
    "visit_year",
    "visit_month",
    "visit_day",
    "visit_type",
    "doctor_name",
    "department",
    # DIAGNOSIS
    "icd10_primary_code",
    "icd10_primary_desc",
    # RAW ARRAYS
    "procedures_icd9_codes",
    "procedures_icd9_descs",
    "drug_codes",
    "drug_names",
    "vitamin_names",
    # COSTS
    "total_procedure_cost",
    "total_drug_cost",
    "total_vitamin_cost",
    "total_claim_amount",
    # CLINICAL SCORES
    "diagnosis_procedure_score",
    "diagnosis_drug_score",
    "diagnosis_vitamin_score",
    # EXPLICIT MISMATCH FLAGS
    "procedure_mismatch_flag",
    "drug_mismatch_flag",
    "vitamin_mismatch_flag",
    "mismatch_count",
    # RISK FEATURES
    "patient_frequency_risk",
    "biaya_anomaly_score",
    # LABELING
    "rule_violation_flag",
    "human_label",
    "final_label",
    "created_at"
]

# Select final features for model training
feature_df = base.select(*final_cols)

# You can now save `feature_df` to a storage location or pass it for model training
# For example:
# feature_df.write.format("parquet").save("s3://your-bucket/fraud_feature_df.parquet")

feature_df.write.format("iceberg").partitionBy("visit_year", "visit_month").mode("overwrite").saveAsTable("iceberg_curated.claim_feature_set")

print("ETL process completed successfully.")