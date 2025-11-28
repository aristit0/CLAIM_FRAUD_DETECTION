from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, lit, when, collect_list, first, year, month, dayofmonth,
    avg, stddev_pop, abs as spark_abs, current_timestamp
)
from pyspark.sql.window import Window

# ----------------------------------------------------------------------------------
# SPARK SESSION
# ----------------------------------------------------------------------------------

spark = SparkSession.builder.appName("claim_feature_set_etl").getOrCreate()
print("=== START ETL FEATURE SET ===")


# ----------------------------------------------------------------------------------
# 1. LOAD RAW ICEBERG TABLES
# ----------------------------------------------------------------------------------

hdr  = spark.table("iceberg_raw.claim_header_raw")
diag = spark.table("iceberg_raw.claim_diagnosis_raw")
proc = spark.table("iceberg_raw.claim_procedure_raw")
drug = spark.table("iceberg_raw.claim_drug_raw")
vit  = spark.table("iceberg_raw.claim_vitamin_raw")

print("Loaded raw tables.")


# ----------------------------------------------------------------------------------
# 2. PRIMARY DIAGNOSIS PER CLAIM
# ----------------------------------------------------------------------------------

diag_primary = (
    diag.where(col("is_primary") == 1)
        .groupBy("claim_id")
        .agg(
            first("icd10_code").alias("icd10_primary_code"),
            first("icd10_description").alias("icd10_primary_desc")
        )
)


# ----------------------------------------------------------------------------------
# 3. AGGREGATE PROCEDURE, DRUG, VITAMIN
# ----------------------------------------------------------------------------------

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

print("Aggregated proc/drug/vitamin.")


# ----------------------------------------------------------------------------------
# 4. JOIN SEMUA KE HEADER
# ----------------------------------------------------------------------------------

base = (
    hdr.alias("h")
    .join(diag_primary.alias("d"), "claim_id", "left")
    .join(proc_agg.alias("p"), "claim_id", "left")
    .join(drug_agg.alias("dr"), "claim_id", "left")
    .join(vit_agg.alias("v"), "claim_id", "left")
)


# ----------------------------------------------------------------------------------
# 5. DERIVE TANGGAL & USIA
# ----------------------------------------------------------------------------------

base = (
    base
    .withColumn(
        "patient_age",
        when(
            col("patient_dob").isNotNull(),
            year(col("visit_date")) - year(col("patient_dob"))
        ).otherwise(lit(None))
    )
    .withColumn("visit_year", year(col("visit_date")).cast("int"))
    .withColumn("visit_month", month(col("visit_date")).cast("int"))
    .withColumn("visit_day", dayofmonth(col("visit_date")).cast("int"))
)


# ----------------------------------------------------------------------------------
# 6. tindakan_validity_score
# ----------------------------------------------------------------------------------

base = base.withColumn(
    "tindakan_validity_score",
    when(col("procedures_icd9_codes").isNull(), lit(0.3)).otherwise(lit(1.0))
)


# ----------------------------------------------------------------------------------
# 7. obat_validity_score
# ----------------------------------------------------------------------------------

base = base.withColumn(
    "has_drug",
    when(col("drug_codes").isNull(), lit(0)).otherwise(lit(1))
)

base = base.withColumn(
    "obat_validity_score",
    when((col("icd10_primary_code").isNotNull()) & (col("has_drug") == 1), lit(1.0))
    .when((col("icd10_primary_code").isNotNull()) & (col("has_drug") == 0), lit(0.4))
    .otherwise(lit(0.8))
)


# ----------------------------------------------------------------------------------
# 8. vitamin_relevance_score
# ----------------------------------------------------------------------------------

base = base.withColumn(
    "has_vitamin",
    when(col("vitamin_names").isNull(), lit(0)).otherwise(lit(1))
)

base = base.withColumn(
    "vitamin_relevance_score",
    when((col("has_vitamin") == 1) & (col("has_drug") == 0), lit(0.2))
    .when((col("has_vitamin") == 1) & (col("has_drug") == 1), lit(0.7))
    .otherwise(lit(1.0))
)


# ----------------------------------------------------------------------------------
# 9. biaya_anomaly_score (Z-score)
# ----------------------------------------------------------------------------------

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


# ----------------------------------------------------------------------------------
# 10. RULE VIOLATION
# ----------------------------------------------------------------------------------

base = base.withColumn(
    "rule_violation_flag",
    when(
        (col("tindakan_validity_score") < 0.5) |
        (col("obat_validity_score") < 0.5) |
        (col("vitamin_relevance_score") < 0.5) |
        (col("biaya_anomaly_score") > 2.5),
        lit(1)
    ).otherwise(lit(0))
)

base = base.withColumn(
    "rule_violation_reason",
    when(col("rule_violation_flag") == 0, lit(None))
    .otherwise(
        when(col("tindakan_validity_score") < 0.5, lit("Tindakan minim atau tidak ada"))
        .when(col("obat_validity_score") < 0.5, lit("Obat tidak sesuai diagnosis"))
        .when(col("vitamin_relevance_score") < 0.5, lit("Vitamin tidak relevan"))
        .when(col("biaya_anomaly_score") > 2.5, lit("Biaya anomali tinggi"))
    )
)


# ----------------------------------------------------------------------------------
# 11. FINAL SELECT (URUTAN WAJIB MATCH DDL!)
# ----------------------------------------------------------------------------------

feature_df = base.select(
    "claim_id",
    "patient_nik",
    "patient_name",
    "patient_gender",
    "patient_dob",
    "patient_age",
    "visit_date",
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
    "biaya_anomaly_score",
    "rule_violation_flag",
    "rule_violation_reason",
    current_timestamp().alias("created_at"),

    col("visit_year").cast("int").alias("visit_year"),
    col("visit_month").cast("int").alias("visit_month")
)

print("Final DF ready.")

feature_df.createOrReplaceTempView("feature_tmp")


# ----------------------------------------------------------------------------------
# 12. WRITE USING SQL (PALING STABIL DI CDP)
# ----------------------------------------------------------------------------------

spark.sql("""
INSERT OVERWRITE TABLE iceberg_curated.claim_feature_set
SELECT
    claim_id,
    patient_nik,
    patient_name,
    patient_gender,
    patient_dob,
    patient_age,
    visit_date,
    visit_day,
    visit_type,
    doctor_name,
    department,
    icd10_primary_code,
    icd10_primary_desc,
    procedures_icd9_codes,
    procedures_icd9_descs,
    drug_codes,
    drug_names,
    vitamin_names,
    total_procedure_cost,
    total_drug_cost,
    total_vitamin_cost,
    total_claim_amount,
    tindakan_validity_score,
    obat_validity_score,
    vitamin_relevance_score,
    biaya_anomaly_score,
    rule_violation_flag,
    rule_violation_reason,
    created_at,
    visit_year,
    visit_month
FROM feature_tmp
""")

print("=== ETL COMPLETED SUCCESSFULLY ===")
spark.stop()