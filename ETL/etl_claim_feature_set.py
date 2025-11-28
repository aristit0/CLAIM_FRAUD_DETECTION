from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, lit, when, array, collect_list, first, year, month, dayofmonth,
    avg, stddev_pop, abs as spark_abs, current_timestamp, countDistinct
)
from pyspark.sql.window import Window

spark = SparkSession.builder.appName("claim_feature_set_etl").getOrCreate()

# 1. Baca tabel raw dari Iceberg
hdr = spark.table("iceberg_raw.claim_header_raw")
diag = spark.table("iceberg_raw.claim_diagnosis_raw")
proc = spark.table("iceberg_raw.claim_procedure_raw")
drug = spark.table("iceberg_raw.claim_drug_raw")
vit = spark.table("iceberg_raw.claim_vitamin_raw")

# 2. Ambil diagnosis utama per claim
diag_primary = (
    diag
    .where(col("is_primary") == 1)
    .groupBy("claim_id")
    .agg(
        first("icd10_code").alias("icd10_primary_code"),
        first("icd10_description").alias("icd10_primary_desc")
    )
)

# 3. Agregasi procedure per claim
proc_agg = (
    proc
    .groupBy("claim_id")
    .agg(
        collect_list("icd9_code").alias("procedures_icd9_codes"),
        collect_list("icd9_description").alias("procedures_icd9_descs")
    )
)

# 4. Agregasi drug per claim
drug_agg = (
    drug
    .groupBy("claim_id")
    .agg(
        collect_list("drug_code").alias("drug_codes"),
        collect_list("drug_name").alias("drug_names")
    )
)

# 5. Agregasi vitamin per claim
vit_agg = (
    vit
    .groupBy("claim_id")
    .agg(
        collect_list("vitamin_name").alias("vitamin_names")
    )
)

# 6. Join semua ke header
base = (
    hdr.alias("h")
    .join(diag_primary.alias("d"), "claim_id", "left")
    .join(proc_agg.alias("p"), "claim_id", "left")
    .join(drug_agg.alias("dr"), "claim_id", "left")
    .join(vit_agg.alias("v"), "claim_id", "left")
)

# 7. Derive umur dan tanggal
base = (
    base
    .withColumn("patient_age",
                when(col("patient_dob").isNotNull(),
                     year(col("visit_date")) - year(col("patient_dob"))
                ).otherwise(lit(None)))
    .withColumn("visit_year", year(col("visit_date")))
    .withColumn("visit_month", month(col("visit_date")))
    .withColumn("visit_day", dayofmonth(col("visit_date")))
)

# 8. Fitur tindakan_validity_score
base = base.withColumn(
    "tindakan_validity_score",
    when(col("procedures_icd9_codes").isNull() | (countDistinct("procedures_icd9_codes").over(Window.partitionBy("claim_id")) == 0), lit(0.3))
    .when((col("procedures_icd9_codes").isNotNull()) &
          (countDistinct("procedures_icd9_codes").over(Window.partitionBy("claim_id")) == 1), lit(0.7))
    .otherwise(lit(1.0))
)

# Catatan:
# countDistinct over array tidak langsung support, jadi kita ganti dengan indikator sederhana:
# ada procedure atau tidak. Versi lebih robust:
base = base.drop("tindakan_validity_score")  # hapus dulu

base = base.withColumn(
    "has_procedure",
    when(col("procedures_icd9_codes").isNotNull(), lit(1)).otherwise(lit(0))
)

base = base.withColumn(
    "tindakan_validity_score",
    when(col("has_procedure") == 0, lit(0.3))
    .otherwise(lit(1.0))
)

# 9. Fitur obat_validity_score
base = base.withColumn(
    "has_drug",
    when(col("drug_codes").isNotNull(), lit(1)).otherwise(lit(0))
)

base = base.withColumn(
    "obat_validity_score",
    when((col("icd10_primary_code").isNotNull()) & (col("has_drug") == 1), lit(1.0))
    .when((col("icd10_primary_code").isNotNull()) & (col("has_drug") == 0), lit(0.4))
    .otherwise(lit(0.8))
)

# 10. Fitur vitamin_relevance_score
base = base.withColumn(
    "has_vitamin",
    when(col("vitamin_names").isNotNull(), lit(1)).otherwise(lit(0))
)

base = base.withColumn(
    "vitamin_relevance_score",
    when((col("has_vitamin") == 1) & (col("has_drug") == 0), lit(0.2))
    .when((col("has_vitamin") == 1) & (col("has_drug") == 1), lit(0.7))
    .otherwise(lit(1.0))
)

# 11. Biaya anomaly score (z-score per (icd10_primary_code, visit_type))
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

# 12. Rule violation flag dan reason
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
)

# Reason string sederhana, bisa kamu kembangkan nanti
base = base.withColumn(
    "rule_violation_reason",
    when(col("rule_violation_flag") == 0, lit(None))
    .otherwise(
        when(col("tindakan_validity_score") < 0.5, lit("Tindakan minim atau tidak ada"))
        .when(col("obat_validity_score") < 0.5, lit("Obat tidak sesuai pola diagnosis"))
        .when(col("vitamin_relevance_score") < 0.5, lit("Vitamin tidak relevan tanpa obat utama"))
        .when(col("biaya_anomaly_score") > 2.5, lit("Biaya anomali tinggi dibanding grup sejenis"))
    )
)

# 13. Pilih kolom untuk feature set
feature_df = base.select(
    col("claim_id"),
    col("patient_nik"),
    col("patient_name"),
    col("patient_gender"),
    col("patient_dob"),
    col("patient_age"),

    col("visit_date"),
    col("visit_year"),
    col("visit_month"),
    col("visit_day"),
    col("visit_type"),
    col("doctor_name"),
    col("department"),

    col("icd10_primary_code"),
    col("icd10_primary_desc"),

    col("procedures_icd9_codes"),
    col("procedures_icd9_descs"),
    col("drug_codes"),
    col("drug_names"),
    col("vitamin_names"),

    col("total_procedure_cost"),
    col("total_drug_cost"),
    col("total_vitamin_cost"),
    col("total_claim_amount"),

    col("tindakan_validity_score"),
    col("obat_validity_score"),
    col("vitamin_relevance_score"),
    col("biaya_anomaly_score"),
    col("rule_violation_flag"),
    col("rule_violation_reason"),
    current_timestamp().alias("created_at")
)

# 14. Tulis ke Iceberg curated table dengan overwrite per partition (simple full overwrite dulu)
feature_df.writeTo("iceberg_curated.claim_feature_set").overwritePartitions()

spark.stop()