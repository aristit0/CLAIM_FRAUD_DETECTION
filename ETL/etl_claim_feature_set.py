import cml.data_v1 as cmldata
from pyspark.sql.functions import (
    col, lit, when, collect_list, first, year, month, dayofmonth,
    avg, stddev_pop, count, size, array_contains, substring,
    current_timestamp, lower, concat_ws,
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

print("=== START FRAUD-ENHANCED ETL (RULE-BASED) ===")

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
# 3. PRIMARY DIAGNOSIS (ICD-10 UTAMA)
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
# 4. AGGREGATIONS: PROCEDURE, DRUG, VITAMIN
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
        .agg(
            collect_list("vitamin_name").alias("vitamin_names")
        )
)

# ================================================================
# 5. JOIN ALL FEATURES KE HEADER
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
# Usia simple: selisih tahun, cukup untuk feature awal
base = (
    base
    .withColumn(
        "patient_age",
        when(col("patient_dob").isNull(), lit(None))
        .otherwise(year(col("visit_date")) - year(col("patient_dob")))
    )
    .withColumn("visit_year",  year(col("visit_date")).cast("int"))
    .withColumn("visit_month", month(col("visit_date")).cast("int"))
    .withColumn("visit_day",   dayofmonth(col("visit_date")).cast("int"))
)

# Helper flags
base = (
    base
    .withColumn(
        "has_procedure",
        when(size(col("procedures_icd9_codes")) > 0, lit(1)).otherwise(lit(0))
    )
    .withColumn(
        "has_drug",
        when(size(col("drug_codes")) > 0, lit(1)).otherwise(lit(0))
    )
    .withColumn(
        "has_vitamin",
        when(size(col("vitamin_names")) > 0, lit(1)).otherwise(lit(0))
    )
)

# ================================================================
# 7. ICD10 SEVERITY SCORE (KASAR)
# ================================================================
# Mapping kasar: makin tinggi = makin berat
severity_map = {
    "A": 3, "B": 3,           # Infectious / parasitic
    "C": 4, "D": 4,           # Neoplasms & blood
    "E": 2,                   # Endocrine
    "F": 1, "G": 2,           # Mental / Neuro
    "H": 1, "I": 3,           # Eye/ear / circulatory
    "J": 1,                   # Respiratory (flu dsb)
    "K": 2                    # Digestive
}

severity_mapping_expr = F.create_map([lit(x) for pair in severity_map.items() for x in pair])

base = (
    base
    .withColumn("icd10_first_letter", substring(col("icd10_primary_code"), 1, 1))
    .withColumn("severity_score", severity_mapping_expr[col("icd10_first_letter")])
)

# ================================================================
# 8. DIAGNOSIS vs PROCEDURE MISMATCH
# ================================================================
# Contoh rule:
# - Diagnosis severity rendah (1) tapi ada banyak prosedur → mencurigakan
base = (
    base.withColumn(
        "diagnosis_procedure_mismatch",
        when(
            (col("severity_score") <= 1) &  # penyakit ringan
            (col("has_procedure") == 1),    # tapi banyak tindakan
            lit(1)
        ).otherwise(lit(0))
    )
)

# ================================================================
# 9. DRUG MISMATCH (CONTOH SEDERHANA)
# ================================================================
# Contoh rule:
# - Diagnosis Jxx (penyakit napas) tapi tidak ada obat yang relevan (sangat kasar)
drug_names_str = concat_ws(" ", col("drug_names"))

base = (
    base.withColumn(
        "drug_mismatch_score",
        when(
            (col("icd10_primary_code").startswith("J")) &  # batuk/flu dsb
            (col("has_drug") == 0),                       # tidak ada obat sama sekali
            lit(1)
        ).otherwise(lit(0))
    )
)

# ================================================================
# 10. COST PER PROCEDURE ANOMALY
# ================================================================
base = (
    base.withColumn(
        "cost_per_procedure",
        col("total_claim_amount") / (col("has_procedure") + lit(1))  # +1 biar tidak div 0
    )
)

# Rule kasar: jika cost per procedure sangat tinggi
base = (
    base.withColumn(
        "cost_procedure_anomaly",
        when(col("cost_per_procedure") > lit(75_000_000), lit(1)).otherwise(lit(0))
    )
)

# ================================================================
# 11. PATIENT HISTORY RISK (FREQUENCY)
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
        when(col("patient_claim_count") > 5, lit(1)).otherwise(lit(0))
    )
)

# ================================================================
# 12. Z-SCORE ANOMALY BY PEER GROUP (ICD10 + VISIT TYPE)
# ================================================================
w_cost = Window.partitionBy("icd10_primary_code", "visit_type")

base = (
    base
    .withColumn("mean_cost", avg(col("total_claim_amount")).over(w_cost))
    .withColumn("std_cost",  stddev_pop(col("total_claim_amount")).over(w_cost))
    .withColumn(
        "biaya_anomaly_score",
        when(col("std_cost").isNull() | (col("std_cost") == 0), lit(0.0))
        .otherwise(
            spark_abs((col("total_claim_amount") - col("mean_cost")) / col("std_cost"))
        )
    )
)

# ================================================================
# 13. STRONG RULE VIOLATION FLAG (KOMBINASI RULE)
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
        ).otherwise(lit(0))
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
# 13B. BACKWARD COMPAT — LEGACY RULE SCORES
# ================================================================
# Kita isi tiga kolom lama (tindakan/obat/vitamin validity) dari rule di atas,
# supaya:
#   - schema tetap cocok dengan tabel Iceberg existing
#   - model & backend lama masih bisa pakai field yang sama

base = (
    base
    # TINDAKAN: penalti jika tidak ada prosedur atau mismatch
    .withColumn(
        "tindakan_validity_score",
        when(col("has_procedure") == 0, lit(0.3))                       # tidak ada tindakan sama sekali
        .when(col("diagnosis_procedure_mismatch") == 1, lit(0.2))       # tindakan tidak sesuai severity
        .otherwise(lit(1.0))                                            # aman
    )
    # OBAT: penalti jika drug mismatch / tidak ada obat untuk diagnosis yang butuh obat
    .withColumn(
        "obat_validity_score",
        when(col("drug_mismatch_score") == 1, lit(0.3))
        .when((col("has_drug") == 0) & (col("severity_score") >= 2), lit(0.5))
        .otherwise(lit(1.0))
    )
    # VITAMIN: penalti jika vitamin tanpa obat di kasus ringan → cenderung "upcoding"
    .withColumn(
        "vitamin_relevance_score",
        when(
            (col("has_vitamin") == 1) &
            (col("has_drug") == 0) &
            (col("severity_score") <= 1),
            lit(0.3)
        )
        .otherwise(lit(1.0))
    )
)

# ================================================================
# 14. FINAL SELECT — MATCHING TABLE SCHEMA
# ================================================================
# Schema EXISTING di iceberg_curated.claim_feature_set (dari error sebelumnya):
# `claim_id`, `patient_nik`, `patient_name`, `patient_gender`,
# `patient_dob`, `patient_age`, `visit_date`, `visit_year`, `visit_month`,
# `visit_day`, `visit_type`, `doctor_name`, `department`, `icd10_primary_code`,
# `icd10_primary_desc`, `procedures_icd9_codes`, `procedures_icd9_descs`,
# `drug_codes`, `drug_names`, `vitamin_names`, `total_procedure_cost`,
# `total_drug_cost`, `total_vitamin_cost`, `total_claim_amount`,
# `tindakan_validity_score`, `obat_validity_score`, `vitamin_relevance_score`,
# `biaya_anomaly_score`, `rule_violation_flag`, `rule_violation_reason`,
# `created_at`.

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

    # --- OLD VALIDITY COLUMNS (HARUS ADA)
    "tindakan_validity_score",
    "obat_validity_score",
    "vitamin_relevance_score",

    # --- NEW FRAUD RULES (MATCH EXACT DDL)
    "severity_score",
    "diagnosis_procedure_mismatch",
    "drug_mismatch_score",
    "cost_per_procedure",
    "cost_procedure_anomaly",
    "patient_claim_count",
    "patient_frequency_risk",

    # --- ANOMALY
    "biaya_anomaly_score",

    # --- FINAL FLAG
    "rule_violation_flag",
    "rule_violation_reason",

    current_timestamp().alias("created_at")
)

print("Final feature_df schema:")
feature_df.printSchema()

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