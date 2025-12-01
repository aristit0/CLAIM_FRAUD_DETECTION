#!/usr/bin/env python3
import cml.data_v1 as cmldata
from pyspark.sql.functions import (
    col, lit, when, collect_list, first, year, month, dayofmonth,
    avg, stddev_pop, count, size, substring, current_timestamp,
    abs as spark_abs
)
from pyspark.sql import functions as F
from pyspark.sql.window import Window

print("=== START FRAUD-ENHANCED ETL v5 WITH EXPLICIT MISMATCH FLAGS ===")

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
    hdr.join(diag_primary, "claim_id", "left")
       .join(proc_agg, "claim_id", "left")
       .join(drug_agg, "claim_id", "left")
       .join(vit_agg, "claim_id", "left")
)

# ================================================================
# 6. DATE + AGE
# ================================================================
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

# ================================================================
# 7. ICD10 Severity
# ================================================================
severity_map = {
    "A": 3, "B": 3, "C": 4, "D": 4,
    "E": 2, "F": 1, "G": 2, "H": 1,
    "I": 3, "J": 1, "K": 2
}
mapping_expr = F.create_map([lit(x) for p in severity_map.items() for x in p])

base = base.withColumn("icd10_first_letter", substring("icd10_primary_code", 1, 1))
base = base.withColumn("severity_score", mapping_expr[col("icd10_first_letter")])

# ================================================================
# 8. PATIENT HISTORY
# ================================================================
w_pid = Window.partitionBy("patient_nik")
base = base.withColumn("patient_claim_count", count("*").over(w_pid))
base = base.withColumn("patient_frequency_risk", when(col("patient_claim_count") > 5, 1).otherwise(0))

# ================================================================
# 9. COST Z-SCORE
# ================================================================
w_cost = Window.partitionBy("icd10_primary_code", "visit_type")

base = (
    base.withColumn("mean_cost", avg("total_claim_amount").over(w_cost))
         .withColumn("std_cost", stddev_pop("total_claim_amount").over(w_cost))
         .withColumn(
            "biaya_anomaly_score",
            when(col("std_cost").isNull() | (col("std_cost") == 0), 0.0)
            .otherwise(spark_abs((col("total_claim_amount") - col("mean_cost")) / col("std_cost")))
         )
)

# ================================================================
# 10. LOAD CLINICAL COMPATIBILITY MAPS (reference)
# ================================================================
MATRIX_SOURCE = "internal_guideline_v1"

map_proc = spark.table("iceberg_ref.icd10_icd9_map") \
    .where(col("source") == MATRIX_SOURCE) \
    .groupBy("icd10_code").agg(F.collect_set("icd9_code").alias("allowed_icd9"))

map_drug = spark.table("iceberg_ref.icd10_drug_map") \
    .where(col("source") == MATRIX_SOURCE) \
    .groupBy("icd10_code").agg(F.collect_set("drug_code").alias("allowed_drug"))

map_vit = spark.table("iceberg_ref.icd10_vitamin_map") \
    .where(col("source") == MATRIX_SOURCE) \
    .groupBy("icd10_code").agg(F.collect_set("vitamin_name").alias("allowed_vit"))

base = (
    base.join(map_proc, base.icd10_primary_code == map_proc.icd10_code, "left")
        .join(map_drug, base.icd10_primary_code == map_drug.icd10_code, "left")
        .join(map_vit,  base.icd10_primary_code == map_vit.icd10_code, "left")
)

# ================================================================
# 11. CLINICAL SCORES
# ================================================================
base = base.withColumn(
    "diagnosis_procedure_score",
    when(size(F.array_intersect("procedures_icd9_codes","allowed_icd9")) > 0, 1.0).otherwise(0.0)
)

base = base.withColumn(
    "diagnosis_drug_score",
    when(size(F.array_intersect("drug_codes","allowed_drug")) > 0, 1.0).otherwise(0.0)
)

base = base.withColumn(
    "diagnosis_vitamin_score",
    when(size(F.array_intersect("vitamin_names","allowed_vit")) > 0, 1.0).otherwise(0.0)
)

base = base.withColumn(
    "treatment_consistency_score",
    col("diagnosis_procedure_score")*0.4 +
    col("diagnosis_drug_score")*0.4 +
    col("diagnosis_vitamin_score")*0.2
)

# ================================================================
# 11b. COST PER PROCEDURE (harus ada, sesuai DDL)
# ================================================================
base = base.withColumn(
    "cost_per_procedure",
    when(col("has_procedure") == 0, col("total_claim_amount"))
    .otherwise(col("total_claim_amount") / col("has_procedure"))
)

# ================================================================
# 11c. COST PROCEDURE ANOMALY FLAG
# ================================================================
base = base.withColumn(
    "cost_procedure_anomaly",
    when(
        (col("cost_per_procedure") > col("mean_cost") + 2.0 * col("std_cost")),
        1
    ).otherwise(0)
)

# ================================================================
# 12. *** NEW: EXPLICIT MISMATCH FLAGS ***
# ================================================================
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

# how many mismatches exist for this claim
base = base.withColumn(
    "mismatch_count",
    col("procedure_mismatch_flag") +
    col("drug_mismatch_flag") +
    col("vitamin_mismatch_flag")
)

# ================================================================
# 13. RULE FLAG + FINAL LABEL
# ================================================================
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

# ================================================================
# 13b. LEGACY RULE FIELDS (BACKWARD COMPAT)
# ================================================================
base = (
    base
    # wariskan langsung dari skor baru
    .withColumn("tindakan_validity_score", col("diagnosis_procedure_score"))
    .withColumn("obat_validity_score", col("diagnosis_drug_score"))
    .withColumn("vitamin_relevance_score", col("diagnosis_vitamin_score"))

    # legacy mismatch fields berbasis flag baru
    .withColumn(
        "diagnosis_procedure_mismatch",
        when(col("procedure_mismatch_flag") == 1, lit(1.0)).otherwise(lit(0.0))
    )
    .withColumn(
        "drug_mismatch_score",
        when(col("drug_mismatch_flag") == 1, lit(1.0)).otherwise(lit(0.0))
    )

    # reason sederhana (boleh dimodif nanti)
    .withColumn(
        "rule_violation_reason",
        F.when(col("rule_violation_flag") == 0, lit(None).cast("string"))
         .otherwise(
             F.concat_ws(
                 "; ",
                 F.when(col("mismatch_count") > 0, lit("CLINICAL_MISMATCH")),
                 F.when(col("biaya_anomaly_score") > 2.5, lit("COST_OUTLIER")),
                 F.when(col("patient_frequency_risk") == 1, lit("HIGH_FREQUENCY_PATIENT"))
             )
         )
    )
)

# ================================================================
# 14. FINAL SELECT
# ================================================================
base = base.withColumn("created_at", current_timestamp())

final_cols = [

    # CLAIM + PATIENT
    "claim_id",
    "patient_nik",
    "patient_name",
    "patient_gender",
    "patient_dob",
    "patient_age",

    # VISIT
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
    "treatment_consistency_score",

    # EXPLICIT MISMATCH FLAGS (v5)
    "procedure_mismatch_flag",
    "drug_mismatch_flag",
    "vitamin_mismatch_flag",
    "mismatch_count",

    # RISK FEATURES
    "severity_score",
    "cost_per_procedure",
    "cost_procedure_anomaly",
    "patient_claim_count",
    "patient_frequency_risk",
    "biaya_anomaly_score",

    # LEGACY RULE FIELDS
    "tindakan_validity_score",
    "obat_validity_score",
    "vitamin_relevance_score",
    "diagnosis_procedure_mismatch",
    "drug_mismatch_score",

    # LABELING
    "rule_violation_flag",
    "rule_violation_reason",
    "human_label",
    "final_label",

    # META
    "created_at"
]

feature_df = base.select(*final_cols)

# ================================================================
# 15. WRITE TO ICEBERG
# ================================================================
(
    feature_df.writeTo("iceberg_curated.claim_feature_set")
              .overwritePartitions()
)

print("=== ETL v5 WITH MISMATCH FEATURES COMPLETED ===")
spark.stop()