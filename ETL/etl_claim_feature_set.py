#!/usr/bin/env python3
from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, IntegerType

spark = (
    SparkSession.builder
    .appName("ETL_Claim_Feature_Set_v3")
    .config("spark.sql.extensions", "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions")
    .config("spark.sql.catalog.iceberg", "org.apache.iceberg.spark.SparkCatalog")
    .config("spark.sql.catalog.iceberg.type", "hadoop")
    .config("spark.sql.catalog.iceberg.warehouse", "hdfs:///warehouse/iceberg")
    .getOrCreate()
)

print("=== LOAD RAW DATA ===")

hdr  = spark.read.format("iceberg").load("iceberg_raw.claim_header_raw")
dx   = spark.read.format("iceberg").load("iceberg_raw.claim_diagnosis_raw")
proc = spark.read.format("iceberg").load("iceberg_raw.claim_procedure_raw")
drug = spark.read.format("iceberg").load("iceberg_raw.claim_drug_raw")
vit  = spark.read.format("iceberg").load("iceberg_raw.claim_vitamin_raw")

print("=== LOAD REFERENCE TABLES ===")
ref_icd10 = spark.read.format("iceberg").load("iceberg_ref.master_icd10")
ref_icd9  = spark.read.format("iceberg").load("iceberg_ref.master_icd9")
ref_drug  = spark.read.format("iceberg").load("iceberg_ref.master_drug")
ref_vit   = spark.read.format("iceberg").load("iceberg_ref.master_vitamin")

# ensure visit_date is date type
hdr = hdr.withColumn("visit_date", F.to_date("visit_date"))

# =========================================================================
# PRIMARY DIAGNOSIS
# =========================================================================
w_dx = Window.partitionBy("claim_id").orderBy(F.desc("is_primary"))

primary_dx = (
    dx.withColumn("rn", F.row_number().over(w_dx))
      .where("rn = 1")
      .select(
          "claim_id",
          F.col("icd10_code").alias("primary_dx_code")
      )
      .join(ref_icd10, ref_icd10.code == F.col("primary_dx_code"), "left")
      .select("claim_id", "primary_dx_code", F.col("description").alias("primary_dx_desc"))
)

# =========================================================================
# AGGREGATE PROCEDURE, DRUG, VITAMIN
# =========================================================================
proc_agg = (
    proc.groupBy("claim_id")
        .agg(F.collect_set("icd9_code").alias("procedures"))
)

drug_agg = (
    drug.groupBy("claim_id")
        .agg(F.collect_set("drug_code").alias("drugs"))
)

vit_agg = (
    vit.groupBy("claim_id")
       .agg(F.collect_set("vitamin_name").alias("vitamins"))
)

# =========================================================================
# JOIN INTO BASE DF
# =========================================================================
base = (
    hdr.join(primary_dx, "claim_id", "left")
       .join(proc_agg, "claim_id", "left")
       .join(drug_agg, "claim_id", "left")
       .join(vit_agg,  "claim_id", "left")
)

# default empty arrays
base = (
    base.withColumn("procedures", F.coalesce("procedures", F.array().cast("array<string>")))
        .withColumn("drugs",      F.coalesce("drugs",      F.array().cast("array<string>")))
        .withColumn("vitamins",   F.coalesce("vitamins",   F.array().cast("array<string>")))
)

# =========================================================================
# COST CLEANUP
# =========================================================================
for c in ["total_procedure_cost", "total_drug_cost", "total_vitamin_cost", "total_claim_amount"]:
    base = base.withColumn(c, F.coalesce(F.col(c).cast(DoubleType()), F.lit(0.0)))

# =========================================================================
# FRAUD LABEL (from status)
# =========================================================================
base = base.withColumn(
    "fraud_label",
    F.when(F.col("status") == "declined", 1)
     .when(F.col("status") == "approved", 0)
     .otherwise(0)
)

# =========================================================================
# COST ANOMALY SCORE (per diagnosis)
# =========================================================================
stats = (
    base.groupBy("primary_dx_code")
        .agg(
            F.expr("percentile_approx(total_claim_amount, 0.5)").alias("median_cost"),
            F.stddev("total_claim_amount").alias("std_cost")
        )
)

base = base.join(stats, "primary_dx_code", "left")

base = base.withColumn(
    "cost_z",
    F.when(F.col("std_cost") > 0,
           (F.col("total_claim_amount") - F.col("median_cost")) / F.col("std_cost"))
     .otherwise(F.lit(0.0))
)

base = (
    base.withColumn(
        "cost_anomaly_score",
        F.when(F.col("cost_z") >= 3, 3)
         .when(F.col("cost_z") >= 2, 2)
         .when(F.col("cost_z") >= 1, 1)
         .otherwise(0)
    )
)

# =========================================================================
# FREQUENCY RISK – 60 day rolling window
# =========================================================================
w_freq = (
    Window.partitionBy("patient_nik")
          .orderBy(F.col("visit_date").cast("timestamp"))
          .rangeBetween(-60 * 86400, 0)
)

base = base.withColumn("claims_60d", F.count("*").over(w_freq))

base = (
    base.withColumn(
        "frequency_risk",
        F.when(F.col("claims_60d") >= 10, 3)
         .when(F.col("claims_60d") >= 5, 2)
         .when(F.col("claims_60d") >= 2, 1)
         .otherwise(0)
    )
)

# =========================================================================
# SAVE TO CURATED V3
# =========================================================================
final_cols = [
    "claim_id",
    "patient_nik",
    "patient_name",
    "patient_gender",
    "patient_dob",
    "visit_date",
    "visit_type",
    "doctor_name",
    "department",

    "primary_dx_code",
    "primary_dx_desc",

    "procedures",
    "drugs",
    "vitamins",

    "total_procedure_cost",
    "total_drug_cost",
    "total_vitamin_cost",
    "total_claim_amount",

    "cost_anomaly_score",
    "frequency_risk",
    "fraud_label"
]

final_df = base.select(*final_cols)

(
    final_df.write
    .format("iceberg")
    .mode("overwrite")
    .save("iceberg_curated.claim_feature_set_v3")
)

print("=== ETL v3 Completed ===")

spark.stop()