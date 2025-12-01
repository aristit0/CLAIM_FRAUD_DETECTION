#!/usr/bin/env python3
import cml.data_v1 as cmldata
from pyspark.sql.functions import (
    col, lit, when, collect_list, first, year, month, dayofmonth, avg, stddev_pop,
    count, size, substring, current_timestamp,
    abs as spark_abs
)
from pyspark.sql.window import Window
from pyspark.sql import functions as F

print("=== START FRAUD-ENHANCED ETL v4 FIXED FULL ===")

# ================================================================
# 1. CONNECT TO SPARK
# ================================================================
conn = cmldata.get_connection("CDP-MSI")
spark = conn.get_spark_session()

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
# 3. PRIMARY DIAGNOSIS
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
# 4. AGGREGATIONS
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
# 5. JOIN ALL RAW DATA
# ================================================================
base = (
    hdr
    .join(diag_primary, "claim_id", "left")
    .join(proc_agg, "claim_id", "left")
    .join(drug_agg, "claim_id", "left")
    .join(vit_agg, "claim_id", "left")
)

# ================================================================
# 6. DATE + AGE FEATURES
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
    .withColumn("has_procedure", when(size("procedures_icd9_codes") > 0, 1).otherwise(0))
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
# 8. RULE-BASED FEATURES (legacy)
# ================================================================
base = (
    base.withColumn(
        "diagnosis_procedure_mismatch",
        when((col("severity_score") <= 1) & (col("has_procedure") == 1), 1).otherwise(0)
    )
    .withColumn(
        "drug_mismatch_score",
        when((col("icd10_primary_code").startswith("J")) & (col("has_drug") == 0), 1).otherwise(0)
    )
    .withColumn("cost_per_procedure", col("total_claim_amount") / (col("has_procedure") + lit(1)))
    .withColumn(
        "cost_procedure_anomaly",
        when(col("cost_per_procedure") > 75_000_000, 1).otherwise(0)
    )
)

# ================================================================
# 9. PATIENT HISTORY
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
# 11. CLINICAL COMPATIBILITY MATRIX
# ================================================================
MATRIX_SOURCE = "internal_guideline_v1"

icd9_map  = spark.table("iceberg_ref.icd10_icd9_map").where(col("source") == MATRIX_SOURCE)
drug_map  = spark.table("iceberg_ref.icd10_drug_map").where(col("source") == MATRIX_SOURCE)
vit_map   = spark.table("iceberg_ref.icd10_vitamin_map").where(col("source") == MATRIX_SOURCE)

map_proc = icd9_map.groupBy("icd10_code").agg(F.collect_set("icd9_code").alias("allowed_icd9_codes"))
map_drug = drug_map.groupBy("icd10_code").agg(F.collect_set("drug_code").alias("allowed_drug_codes"))
map_vit  = vit_map.groupBy("icd10_code").agg(F.collect_set("vitamin_name").alias("allowed_vitamins"))

base = (
    base.join(map_proc, base.icd10_primary_code == map_proc.icd10_code, "left")
        .join(map_drug, base.icd10_primary_code == map_drug.icd10_code, "left")
        .join(map_vit,  base.icd10_primary_code == map_vit.icd10_code, "left")
)

# ================================================================
# 12. COMPUTE COMPATIBILITY SCORES
# ================================================================
base = base.withColumn(
    "diagnosis_procedure_score",
    when(
        (col("allowed_icd9_codes").isNotNull()) &
        (size(F.array_intersect(col("procedures_icd9_codes"), col("allowed_icd9_codes"))) > 0),
        1.0
    ).otherwise(0.0)
)

base = base.withColumn(
    "diagnosis_drug_score",
    when(
        (col("allowed_drug_codes").isNotNull()) &
        (size(F.array_intersect(col("drug_codes"), col("allowed_drug_codes"))) > 0),
        1.0
    ).otherwise(0.0)
)

base = base.withColumn(
    "diagnosis_vitamin_score",
    when(
        (col("allowed_vitamins").isNotNull()) &
        (size(F.array_intersect(col("vitamin_names"), col("allowed_vitamins"))) > 0),
        1.0
    ).otherwise(0.0)
)

base = base.withColumn(
    "treatment_consistency_score",
    col("diagnosis_procedure_score")*0.4 +
    col("diagnosis_drug_score")*0.4 +
    col("diagnosis_vitamin_score")*0.2
)

# ================================================================
# 12b. RULE VIOLATION FLAG (BARU DITAMBAH LAGI)
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

# ================================================================
# 13. HUMAN LABEL + FINAL LABEL
# ================================================================
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

# ================================================================
# 13b. LEGACY SCORES + REASON (SET NULL DULU)
# ================================================================
base = (
    base.withColumn("tindakan_validity_score", lit(None).cast("double"))
        .withColumn("obat_validity_score", lit(None).cast("double"))
        .withColumn("vitamin_relevance_score", lit(None).cast("double"))
        .withColumn("rule_violation_reason", lit(None).cast("string"))
)

# ================================================================
# 14. FINAL SELECT SESUAI DDL
# ================================================================
EXPECTED_COLS = [
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
    "tindakan_validity_score",
    "obat_validity_score",
    "vitamin_relevance_score",
    "diagnosis_procedure_score",
    "diagnosis_drug_score",
    "diagnosis_vitamin_score",
    "treatment_consistency_score",
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
    "human_label",
    "final_label",
    "created_at"
]

base = base.withColumn("created_at", current_timestamp())
feature_df = base.select(*EXPECTED_COLS)

# ================================================================
# 15. WRITE TO ICEBERG
# ================================================================
(
    feature_df.writeTo("iceberg_curated.claim_feature_set")
              .overwritePartitions()
)

print("=== ETL v4 FIXED FULL COMPLETED ===")
spark.stop()