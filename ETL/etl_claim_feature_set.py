import cml.data_v1 as cmldata
from pyspark.sql.functions import (
    col, lit, when, collect_list, first, year, month, dayofmonth,
    avg, stddev_pop, abs as spark_abs, current_timestamp
)
from pyspark.sql.window import Window

# ====================================================================================
# 1. CONNECT TO SPARK VIA CML
# ====================================================================================

CONNECTION_NAME = "CDP-MSI"
conn = cmldata.get_connection(CONNECTION_NAME)
spark = conn.get_spark_session()

print("=== START ETL FEATURE SET (CML + HadoopCatalog) ===")


# ====================================================================================
# 2. LOAD RAW TABLES (HadoopCatalog via 'ice.')
# ====================================================================================

hdr  = spark.sql("SELECT * FROM iceberg_raw.claim_header_raw")
diag = spark.sql("SELECT * FROM iceberg_raw.claim_diagnosis_raw")
proc = spark.sql("SELECT * FROM iceberg_raw.claim_procedure_raw")
drug = spark.sql("SELECT * FROM iceberg_raw.claim_drug_raw")
vit  = spark.sql("SELECT * FROM iceberg_raw.claim_vitamin_raw")

print("Loaded raw tables using ice. catalog")


# ====================================================================================
# 3. PRIMARY DIAGNOSIS
# ====================================================================================

diag_primary = (
    diag.where(col("is_primary") == 1)
        .groupBy("claim_id")
        .agg(
            first("icd10_code").alias("icd10_primary_code"),
            first("icd10_description").alias("icd10_primary_desc")
        )
)


# ====================================================================================
# 4. AGGREGATIONS
# ====================================================================================

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


# ====================================================================================
# 5. JOIN KE HEADER
# ====================================================================================

base = (
    hdr.alias("h")
    .join(diag_primary.alias("d"), "claim_id", "left")
    .join(proc_agg.alias("p"), "claim_id", "left")
    .join(drug_agg.alias("dr"), "claim_id", "left")
    .join(vit_agg.alias("v"), "claim_id", "left")
)


# ====================================================================================
# 6. DERIVED COLUMNS (TANGGAL, USIA, dll)
# ====================================================================================

base = (
    base
    .withColumn(
        "patient_age",
        when(
            col("patient_dob").isNotNull(),
            year(col("visit_date")) - year(col("patient_dob"))
        )
    )
    .withColumn("visit_year", year(col("visit_date")).cast("int"))
    .withColumn("visit_month", month(col("visit_date")).cast("int"))
    .withColumn("visit_day", dayofmonth(col("visit_date")).cast("int"))
)


# tindakan_validity_score
base = base.withColumn(
    "tindakan_validity_score",
    when(col("procedures_icd9_codes").isNull(), lit(0.3)).otherwise(lit(1.0))
)

# obat_validity_score
base = (
    base
    .withColumn("has_drug", when(col("drug_codes").isNull(), lit(0)).otherwise(lit(1)))
    .withColumn(
        "obat_validity_score",
        when((col("icd10_primary_code").isNotNull()) & (col("has_drug") == 1), lit(1.0))
        .when((col("icd10_primary_code").isNotNull()) & (col("has_drug") == 0), lit(0.4))
        .otherwise(lit(0.8))
    )
)

# vitamin_relevance_score
base = (
    base
    .withColumn("has_vitamin", when(col("vitamin_names").isNull(), lit(0)).otherwise(lit(1)))
    .withColumn(
        "vitamin_relevance_score",
        when((col("has_vitamin") == 1) & (col("has_drug") == 0), lit(0.2))
        .when((col("has_vitamin") == 1) & (col("has_drug") == 1), lit(0.7))
        .otherwise(lit(1.0))
    )
)

# biaya_anomaly_score via Z-score
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

# rule violation
base = (
    base
    .withColumn(
        "rule_violation_flag",
        when(
            (col("tindakan_validity_score") < 0.5) |
            (col("obat_validity_score") < 0.5) |
            (col("vitamin_relevance_score") < 0.5) |
            (col("biaya_anomaly_score") > 2.5),
            lit(1)
        ).otherwise(lit(0))
    )
    .withColumn(
        "rule_violation_reason",
        when(col("rule_violation_flag") == 0, lit(None))
        .otherwise(
            when(col("tindakan_validity_score") < 0.5, lit("Tindakan minim atau tidak ada"))
            .when(col("obat_validity_score") < 0.5, lit("Obat tidak sesuai diagnosis"))
            .when(col("vitamin_relevance_score") < 0.5, lit("Vitamin tidak relevan"))
            .when(col("biaya_anomaly_score") > 2.5, lit("Biaya anomali tinggi"))
        )
    )
)


# ====================================================================================
# 7. FINAL DATAFRAME
# ====================================================================================

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
    "tindakan_validity_score", "obat_validity_score",
    "vitamin_relevance_score", "biaya_anomaly_score",
    "rule_violation_flag", "rule_violation_reason",
    current_timestamp().alias("created_at")
)

print("Final DataFrame ready.")


# ====================================================================================
# 8. WRITE USING ICEBERG V2 (HadoopCatalog)
# ====================================================================================

(
    feature_df
        .writeTo("iceberg_curated.claim_feature_set")
        .overwritePartitions()   # works for Iceberg v2
)

print("=== ETL COMPLETED SUCCESSFULLY ===")