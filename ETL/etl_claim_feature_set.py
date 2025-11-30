import cml.data_v1 as cmldata
from pyspark.sql.functions import (
    col, lit, when, collect_list, first, year, month, dayofmonth,
    avg, stddev_pop, count, size, array_contains, substring, current_timestamp,
    abs as spark_abs
)
from pyspark.sql.window import Window
from pyspark.sql.types import IntegerType, StringType
from pyspark.sql import functions as F

# ================================================================
# 1. CONNECT TO SPARK
# ================================================================
CONNECTION_NAME = "CDP-MSI"
conn = cmldata.get_connection(CONNECTION_NAME)
spark = conn.get_spark_session()

print("=== START FRAUD-ENHANCED ETL ===")

# ================================================================
# 2. LOAD RAW TABLES (ICEBERG RAW)
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
proc_agg = (
    proc.groupBy("claim_id")
        .agg(
            collect_list("icd9_code").alias("procedures_icd9_codes"),
            collect_list("icd9_description").alias("procedures_icd9_descs")
        )
)

drug_agg = (
    drug.groupBy("claim_id")
        .agg(
            collect_list("drug_code").alias("drug_codes"),
            collect_list("drug_name").alias("drug_names")
        )
)

vit_agg = (
    vit.groupBy("claim_id")
        .agg(collect_list("vitamin_name").alias("vitamin_names"))
)


# ================================================================
# 5. JOIN ALL FEATURES
# ================================================================
base = (
    hdr.alias("h")
        .join(diag_primary.alias("d"), "claim_id", "left")
        .join(proc_agg.alias("p"), "claim_id", "left")
        .join(drug_agg.alias("dr"), "claim_id", "left")
        .join(vit_agg.alias("v"), "claim_id", "left")
)


# ================================================================
# 6. DERIVED COLUMNS (AGE, DATE PARTS)
# ================================================================
base = (
    base
    .withColumn(
        "patient_age",
        when(col("patient_dob").isNull(), lit(None))
        .otherwise(year(col("visit_date")) - year(col("patient_dob")))
    )
    .withColumn("visit_year", year(col("visit_date")).cast("int"))
    .withColumn("visit_month", month(col("visit_date")).cast("int"))
    .withColumn("visit_day", dayofmonth(col("visit_date")).cast("int"))
)


# ================================================================
# 7. ICD10 SEVERITY SCORE
# ================================================================
severity_map = {
    "A": 3, "B": 3,
    "C": 4, "D": 4,
    "E": 2, 
    "F": 1, "G": 2,
    "H": 1, "I": 3,
    "J": 1,  # flu
    "K": 2
}

severity_mapping_expr = F.create_map([lit(x) for pair in severity_map.items() for x in pair])

base = (
    base
    .withColumn("icd10_first_letter", substring(col("icd10_primary_code"), 1, 1))
    .withColumn("severity_score", severity_mapping_expr[col("icd10_first_letter")])
)


# ================================================================
# 8. MISMATCH: Diagnosis vs Procedure
# ================================================================
base = (
    base.withColumn(
        "diagnosis_procedure_mismatch",
        when(
            (col("severity_score") <= 1) &
            (size(col("procedures_icd9_codes")) > 0),
            lit(1)
        ).otherwise(lit(0))
    )
)


# ================================================================
# 9. DRUG MISMATCH — simple rule (example)
# ================================================================
base = (
    base.withColumn(
        "drug_mismatch_score",
        when(
            (col("icd10_primary_code").startswith("J")) & 
            (~array_contains(col("drug_names"), "Antibiotik")),
            lit(1)
        ).otherwise(0)
    )
)


# ================================================================
# 10. COST PER PROCEDURE ANOMALY
# ================================================================
base = (
    base.withColumn(
        "cost_per_procedure",
        col("total_claim_amount") / (size(col("procedures_icd9_codes")) + lit(1))
    )
)

base = (
    base.withColumn(
        "cost_procedure_anomaly",
        when(col("cost_per_procedure") > 75_000_000, lit(1))
        .otherwise(0)
    )
)


# ================================================================
# 11. PATIENT HISTORY RISK
# ================================================================
w_pid = Window.partitionBy("patient_nik")

base = (
    base.withColumn(
        "patient_claim_count",
        count("*").over(w_pid)
    )
)

base = (
    base.withColumn(
        "patient_frequency_risk",
        when(col("patient_claim_count") > 5, lit(1)).otherwise(0)
    )
)


# ================================================================
# 12. Z-SCORE ANOMALY USING PEER GROUP
# ================================================================
w_cost = Window.partitionBy("icd10_primary_code", "visit_type")

base = (
    base
    .withColumn("mean_cost", avg(col("total_claim_amount")).over(w_cost))
    .withColumn("std_cost", stddev_pop(col("total_claim_amount")).over(w_cost))
    .withColumn(
        "biaya_anomaly_score",
        when(col("std_cost").isNull() | (col("std_cost") == 0), lit(0.0))
        .otherwise(
            spark_abs((col("total_claim_amount") - col("mean_cost")) / col("std_cost"))
        )
    )
)


# ================================================================
# 13. STRONG RULE VIOLATION FLAG
# ================================================================
base = (
    base
    .withColumn(
        "rule_violation_flag",
        when(
            (col("diagnosis_procedure_mismatch") == 1) |
            (col("drug_mismatch_score") == 1) |
            (col("cost_procedure_anomaly") == 1) |
            (col("patient_frequency_risk") == 1) |
            (col("biaya_anomaly_score") > 2.5),
            lit(1)
        ).otherwise(0)
    )
)

base = (
    base
    .withColumn(
        "rule_violation_reason",
        when(col("rule_violation_flag") == 0, lit(None))
        .otherwise(
            when(col("diagnosis_procedure_mismatch") == 1, lit("Diagnosis-procedure mismatch"))
            .when(col("drug_mismatch_score") == 1, lit("Drug mismatch"))
            .when(col("cost_procedure_anomaly") == 1, lit("Cost-per-procedure anomaly"))
            .when(col("patient_frequency_risk") == 1, lit("Patient claim frequency high"))
            .when(col("biaya_anomaly_score") > 2.5, lit("Z-score anomaly high"))
        )
    )
)


# ================================================================
# 14. FINAL SELECT
# ================================================================
feature_df = base.select(
    "claim_id", "patient_nik", "patient_name", "patient_gender",
    "patient_dob", "patient_age", "visit_date",
    "visit_year", "visit_month", "visit_day",
    "visit_type", "doctor_name", "department",
    "icd10_primary_code", "icd10_primary_desc",
    "procedures_icd9_codes", "procedures_icd9_descs",
    "drug_codes", "drug_names", "vitamin_names",
    "total_procedure_cost", "total_drug_cost",
    "total_vitamin_cost", "total_claim_amount",
    "severity_score",
    "diagnosis_procedure_mismatch",
    "drug_mismatch_score",
    "cost_per_procedure",
    "cost_procedure_anomaly",
    "patient_claim_count",
    "patient_frequency_risk",
    "biaya_anomaly_score",
    "rule_violation_flag", "rule_violation_reason",
    current_timestamp().alias("created_at")
)


# ================================================================
# 15. WRITE TO ICEBERG (CURATED)
# ================================================================
(
    feature_df
        .writeTo("iceberg_curated.claim_feature_set")
        .overwritePartitions()
)

print("=== FRAUD ETL COMPLETED SUCCESSFULLY ===")

spark.stop()