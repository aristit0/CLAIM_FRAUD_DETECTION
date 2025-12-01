#!/usr/bin/env python3
import cml.data_v1 as cmldata
from pyspark.sql.functions import (
    col, lit, when, collect_list, first, year, month, dayofmonth, avg, stddev_pop,
    count, size, substring, current_timestamp, concat_ws,
    abs as spark_abs
)
from pyspark.sql.window import Window
from pyspark.sql import functions as F

# ================================================================
# 1. CONNECT TO SPARK
# ================================================================
CONNECTION_NAME = "CDP-MSI"
conn = cmldata.get_connection(CONNECTION_NAME)
spark = conn.get_spark_session()

print("=== START FRAUD-ENHANCED ETL ===")

# ================================================================
# 2. LOAD RAW TABLES
# ================================================================
hdr  = spark.sql("SELECT * FROM iceberg_raw.claim_header_raw")
diag = spark.sql("SELECT * FROM iceberg_raw.claim_diagnosis_raw")
proc = spark.sql("SELECT * FROM iceberg_raw.claim_procedure_raw")
drug = spark.sql("SELECT * FROM iceberg_raw.claim_drug_raw")
vit  = spark.sql("SELECT * FROM iceberg_raw.claim_vitamin_raw")

print("Loaded raw tables.")

# ================================================================
# 3. PRIMARY DIAG
# ================================================================
diag_primary = (
    diag.where(col("is_primary") == 1)
        .groupBy("claim_id")
        .agg(
            first("icd10_code").alias("icd10_primary_code"),
            first("icd10_description").alias("icd10_primary_desc")
        )
)

# ================================================================
# 4. AGGREGATE ICD9 / DRUG / VITAMIN
# ================================================================
proc_agg = proc.groupBy("claim_id").agg(
    collect_list("icd9_code").alias("procedures_icd9_codes"),
    collect_list("icd9_description").alias("procedures_icd9_descs")
)

drug_agg = drug.groupBy("claim_id").agg(
    collect_list("drug_code").alias("drug_codes"),
    collect_list("drug_name").alias("drug_names")
)

vit_agg = vit.groupBy("claim_id").agg(
    collect_list("vitamin_name").alias("vitamin_names")
)

# ================================================================
# 5. JOIN ALL
# ================================================================
base = (
    hdr.alias("h")
       .join(diag_primary, "claim_id", "left")
       .join(proc_agg, "claim_id", "left")
       .join(drug_agg, "claim_id", "left")
       .join(vit_agg,  "claim_id", "left")
)

# ================================================================
# 6. AGE & DATE DERIVATIVES
# ================================================================
base = (
    base.withColumn(
        "patient_age",
        when(col("patient_dob").isNull(), lit(None))
        .otherwise(year(col("visit_date")) - year(col("patient_dob")))
    )
    .withColumn("visit_year",  year(col("visit_date")))
    .withColumn("visit_month", month(col("visit_date")))
    .withColumn("visit_day",   dayofmonth(col("visit_date")))
)

# Presence flags
base = (
    base.withColumn("has_procedure", when(size("procedures_icd9_codes") > 0, 1).otherwise(0))
        .withColumn("has_drug",      when(size("drug_codes") > 0, 1).otherwise(0))
        .withColumn("has_vitamin",   when(size("vitamin_names") > 0, 1).otherwise(0))
)

# ================================================================
# 7. ICD10 SEVERITY SCORE
# ================================================================
severity_map = {
    "A": 3, "B": 3, "C": 4, "D": 4,
    "E": 2, "F": 1, "G": 2, "H": 1,
    "I": 3, "J": 1, "K": 2
}

mapping_expr = F.create_map([lit(x) for p in severity_map.items() for x in p])

base = (
    base.withColumn("icd10_first_letter", substring("icd10_primary_code", 1, 1))
        .withColumn("severity_score", mapping_expr[col("icd10_first_letter")])
)

# ================================================================
# 8. OLD RULES (Masih dipakai partial)
# ================================================================
base = base.withColumn(
    "diagnosis_procedure_mismatch",
    when((col("severity_score") <= 1) & (col("has_procedure") == 1), 1).otherwise(0)
)

drug_names_str = concat_ws(" ", col("drug_names"))

base = base.withColumn(
    "drug_mismatch_score",
    when((col("icd10_primary_code").startswith("J")) & (col("has_drug") == 0), 1).otherwise(0)
)

# cost per procedure
base = base.withColumn("cost_per_procedure", col("total_claim_amount") / (col("has_procedure") + lit(1)))

base = base.withColumn(
    "cost_procedure_anomaly",
    when(col("cost_per_procedure") > 75_000_000, 1).otherwise(0)
)

# ================================================================
# 9. PATIENT CLAIM HISTORY
# ================================================================
w_pid = Window.partitionBy("patient_nik")

base = base.withColumn("patient_claim_count", count("*").over(w_pid))

base = base.withColumn(
    "patient_frequency_risk",
    when(col("patient_claim_count") > 5, 1).otherwise(0)
)

# ================================================================
# 10. COST Z-SCORE
# ================================================================
w_cost = Window.partitionBy("icd10_primary_code", "visit_type")

base = (
    base.withColumn("mean_cost", avg("total_claim_amount").over(w_cost))
        .withColumn("std_cost",  stddev_pop("total_claim_amount").over(w_cost))
        .withColumn(
            "biaya_anomaly_score",
            when(col("std_cost").isNull() | (col("std_cost") == 0), 0.0)
            .otherwise(spark_abs((col("total_claim_amount") - col("mean_cost")) / col("std_cost")))
        )
)

# ================================================================
# 11. LEGACY VALIDITY SCORE (Keep)
# ================================================================
base = (
    base.withColumn(
        "tindakan_validity_score",
        when(col("has_procedure") == 0, 0.3)
        .when(col("diagnosis_procedure_mismatch") == 1, 0.2)
        .otherwise(1.0)
    )
    .withColumn(
        "obat_validity_score",
        when(col("drug_mismatch_score") == 1, 0.3)
        .when((col("has_drug") == 0) & (col("severity_score") >= 2), 0.5)
        .otherwise(1.0)
    )
    .withColumn(
        "vitamin_relevance_score",
        when((col("has_vitamin") == 1) & (col("has_drug") == 0) & (col("severity_score") <= 1), 0.3)
        .otherwise(1.0)
    )
)

# ================================================================
# 12. CLINICAL COMPATIBILITY MATRIX (PRODUCTION RULES ENHANCED)
# ================================================================

MATRIX_VERSION = "clinical_matrix_v2025_01"

# ---------------------------------------------------------------
# LOAD MATRIX WITH VERSION FILTER
# ---------------------------------------------------------------
icd9_map = (
    spark.table("iceberg_ref.icd10_icd9_map")
         .where(col("is_active") == True)
         .where(col("source") == MATRIX_VERSION)
)

drug_map = (
    spark.table("iceberg_ref.icd10_drug_map")
         .where(col("is_active") == True)
         .where(col("source") == MATRIX_VERSION)
)

vit_map = (
    spark.table("iceberg_ref.icd10_vitamin_map")
         .where(col("is_active") == True)
         .where(col("source") == MATRIX_VERSION)
)

# ---------------------------------------------------------------
# BUILD ALLOWED CODES (AS LISTS)
# ---------------------------------------------------------------
map_proc = (
    icd9_map.groupBy("icd10_code")
            .agg(F.collect_set("icd9_code").alias("allowed_icd9_codes"))
)

map_drug = (
    drug_map.groupBy("icd10_code")
            .agg(F.collect_set("drug_code").alias("allowed_drug_codes"))
)

map_vit = (
    vit_map.groupBy("icd10_code")
           .agg(F.collect_set("vitamin_name").alias("allowed_vitamins"))
)

# ---------------------------------------------------------------
# JOIN MATRIX
# ---------------------------------------------------------------
base = (
    base.join(map_proc, base.icd10_primary_code == map_proc.icd10_code, "left")
        .join(map_drug, base.icd10_primary_code == map_drug.icd10_code, "left")
        .join(map_vit,  base.icd10_primary_code == map_vit.icd10_code, "left")
)

# ---------------------------------------------------------------
# AGE RULE → PRODUCES AGE_COMPATIBILITY_SCORE (0 to 1)
# ---------------------------------------------------------------
base = base.withColumn(
    "age_compatible",
    when(
        (
            (col("patient_age").isNotNull()) &
            (
                # age >= min_age OR min_age null
                (col("patient_age") >= F.coalesce(drug_map.min_age, lit(0))) &
                # age <= max_age OR max_age null
                (col("patient_age") <= F.coalesce(drug_map.max_age, lit(200)))
            )
        ),
        1.0
    ).otherwise(0.0)
)

# ---------------------------------------------------------------
# SCORING RULE: PROCEDURE MATCH (0 or 1)
# ---------------------------------------------------------------
base = base.withColumn(
    "diagnosis_procedure_score",
    when(
        (col("allowed_icd9_codes").isNotNull()) &
        (size(F.array_intersect(col("procedures_icd9_codes"), col("allowed_icd9_codes"))) > 0),
        1.0
    ).otherwise(0.0)
)

# ---------------------------------------------------------------
# SCORING RULE: DRUG MATCH (soft score)
# ---------------------------------------------------------------
base = base.withColumn(
    "diagnosis_drug_score",
    when(
        (col("allowed_drug_codes").isNotNull()) &
        (size(F.array_intersect(col("drug_codes"), col("allowed_drug_codes"))) > 0),
        1.0
    ).otherwise(0.0)
)

# ---------------------------------------------------------------
# PENALTY 1: ANTIBIOTIC INJECTION FOR COMMON COLD (J06)
# ---------------------------------------------------------------
antibiotic_misuse = (
    (col("icd10_primary_code") == "J06") &
    (F.array_contains(col("drug_codes"), "KFA003")) &  # ceftriaxone injeksi (contoh)
    (col("severity_score") <= 1)
)

base = base.withColumn(
    "antibiotic_injection_penalty",
    when(antibiotic_misuse, 1).otherwise(0)
)

# Apply penalty to drug score
base = base.withColumn(
    "diagnosis_drug_score",
    when(
        col("antibiotic_injection_penalty") == 1,
        0.2   # very low score
    ).otherwise(col("diagnosis_drug_score"))
)

# ---------------------------------------------------------------
# SCORING RULE: VITAMIN MATCH (0 or 1)
# ---------------------------------------------------------------
base = base.withColumn(
    "diagnosis_vitamin_score",
    when(
        (col("allowed_vitamins").isNotNull()) &
        (size(F.array_intersect(col("vitamin_names"), col("allowed_vitamins"))) > 0),
        1.0
    ).otherwise(0.0)
)

# ---------------------------------------------------------------
# FINAL CONSISTENCY SCORE (weighted)
# ---------------------------------------------------------------
base = base.withColumn(
    "treatment_consistency_score",
    (
        col("diagnosis_procedure_score") * 0.4 +
        col("diagnosis_drug_score") * 0.4 +
        col("diagnosis_vitamin_score") * 0.2
    )
)

base = base.drop("allowed_icd9_codes", "allowed_drug_codes", "allowed_vitamins")

# ================================================================
# 13. FINAL RULE FLAG (UPDATED)
# ================================================================
base = base.withColumn(
    "rule_violation_flag",
    when(
        (col("diagnosis_procedure_score") == 0) |
        (col("diagnosis_drug_score") == 0) |
        (col("diagnosis_vitamin_score") == 0) |
        (col("cost_procedure_anomaly") == 1) |
        (col("patient_frequency_risk") == 1) |
        (col("biaya_anomaly_score") > 2.5),
        1
    ).otherwise(0)
)

base = base.withColumn(
    "rule_violation_reason",
    when(col("rule_violation_flag") == 0, None)
    .otherwise(
        when(col("diagnosis_procedure_score") == 0, "Procedure mismatch")
        .when(col("diagnosis_drug_score") == 0, "Drug mismatch")
        .when(col("diagnosis_vitamin_score") == 0, "Vitamin mismatch")
        .when(col("cost_procedure_anomaly") == 1, "Procedure cost anomaly")
        .when(col("patient_frequency_risk") == 1, "High patient claim frequency")
        .when(col("biaya_anomaly_score") > 2.5, "Z-score cost anomaly")
    )
)

# ================================================================
# 14. FINAL SCHEMA SELECT
# ================================================================
feature_df = base.select(
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
    "icd10_primary_code",
    "icd10_primary_desc",
    "procedures_icd9_codes",
    "procedures_icd9_descs",
    "drug_codes",
    "drug_names",
    "vitamin_names",
    "total_procedure_cost",
    "total_drug_cost",
    "total_vitamin_cost",
    "total_claim_amount",

    # OLD validity
    "tindakan_validity_score",
    "obat_validity_score",
    "vitamin_relevance_score",

    # NEW compatibility scores
    "diagnosis_procedure_score",
    "diagnosis_drug_score",
    "diagnosis_vitamin_score",
    "treatment_consistency_score",

    # Frauds
    "severity_score",
    "diagnosis_procedure_mismatch",
    "drug_mismatch_score",
    "cost_per_procedure",
    "cost_procedure_anomaly",
    "patient_claim_count",
    "patient_frequency_risk",
    "biaya_anomaly_score",

    "rule_violation_flag",
    "rule_violation_reason",

    current_timestamp().alias("created_at")
)

print("Final schema:")
feature_df.printSchema()

# ================================================================
# 15. WRITE TO ICEBERG
# ================================================================
(
    feature_df.writeTo("iceberg_curated.claim_feature_set")
               .overwritePartitions()
)

print("=== ETL COMPLETED SUCCESSFULLY ===")
spark.stop()