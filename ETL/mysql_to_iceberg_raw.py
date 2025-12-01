#!/usr/bin/env python3
"""
ETL: Read from MySQL (claimdb) -> write to Iceberg raw tables (iceberg_raw.*)
Usage: run in CML / Spark environment where cmldata is available, or adjust spark creation accordingly.
"""

import logging
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, IntegerType, DateType, TimestampType, StringType

# Jika kamu pakai CML connection helper (seperti di contoh), uncomment bagian cmldata
try:
    import cml.data_v1 as cmldata
    USE_CML = True
except Exception:
    USE_CML = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mysql-to-iceberg-etl")

# ================================================================
# KONFIG JDBC MySQL (pakai credentials yang kamu berikan)
# ================================================================
JDBC_HOST = "cdpmsi.tomodachis.org"
JDBC_PORT = 3306
JDBC_DB   = "claimdb"
JDBC_USER = "cloudera"
JDBC_PASS = "T1ku$H1t4m"

jdbc_url = f"jdbc:mysql://{JDBC_HOST}:{JDBC_PORT}/{JDBC_DB}?useSSL=false&serverTimezone=UTC"

jdbc_props = {
    "user": JDBC_USER,
    "password": JDBC_PASS,
    "driver": "com.mysql.jdbc.Driver"
}

# ================================================================
# 1. INIT SPARK SESSION
# ================================================================
if USE_CML:
    # Use CML connection (reuses config from your environment)
    CONN_NAME = "CDP-MSI"
    conn = cmldata.get_connection(CONN_NAME)
    spark = conn.get_spark_session()
    logger.info("Spark session obtained from CML connection '%s'", CONN_NAME)
else:
    # Fallback: create local SparkSession (adjust master & packages as needed)
    spark = SparkSession.builder \
        .appName("mysql-to-iceberg-etl") \
        .config("spark.sql.catalog.spark_catalog", "org.apache.iceberg.spark.SparkSessionCatalog") \
        .getOrCreate()
    logger.info("Spark session created (non-CML mode).")

# ================================================================
# 2. HELPER: read table via JDBC
# ================================================================
def read_mysql_table(table_name):
    logger.info("Reading MySQL table: %s", table_name)
    return spark.read.format("jdbc") \
        .option("url", jdbc_url) \
        .option("dbtable", table_name) \
        .option("user", jdbc_props["user"]) \
        .option("password", jdbc_props["password"]) \
        .option("driver", jdbc_props["driver"]) \
        .load()

# ================================================================
# 3. LOAD SOURCE TABLES
# ================================================================
hdr_raw  = read_mysql_table("claim_header")
diag_raw = read_mysql_table("claim_diagnosis")
proc_raw = read_mysql_table("claim_procedure")
drug_raw = read_mysql_table("claim_drug")
vit_raw  = read_mysql_table("claim_vitamin")

logger.info("Selesai load semua tabel MySQL")

# ================================================================
# 4. CAST / NORMALIZE SCHEMAS supaya cocok dengan Iceberg DDL
#    (Iceberg raw uses DOUBLE for costs, DATE/TIMESTAMP mapping, STRING for varchars)
# ================================================================
# claim_header -> iceberg_raw.claim_header_raw
hdr = (
    hdr_raw
    .withColumnRenamed("claim_id", "claim_id")
    # cast numeric costs to double
    .withColumn("total_procedure_cost", F.col("total_procedure_cost").cast(DoubleType()))
    .withColumn("total_drug_cost",      F.col("total_drug_cost").cast(DoubleType()))
    .withColumn("total_vitamin_cost",   F.col("total_vitamin_cost").cast(DoubleType()))
    .withColumn("total_claim_amount",   F.col("total_claim_amount").cast(DoubleType()))
    # dates & timestamps
    .withColumn("patient_dob",   F.col("patient_dob").cast(DateType()))
    .withColumn("visit_date",    F.col("visit_date").cast(DateType()))
    .withColumn("created_at",    F.col("created_at").cast(TimestampType()))
    .withColumn("updated_at",    F.col("updated_at").cast(TimestampType()))
)

# claim_diagnosis -> iceberg_raw.claim_diagnosis_raw
diag = (
    diag_raw
    .withColumn("id", F.col("id").cast("long"))
    .withColumn("claim_id", F.col("claim_id").cast("long"))
    .withColumn("icd10_code", F.col("icd10_code").cast(StringType()))
    .withColumn("icd10_description", F.col("icd10_description").cast(StringType()))
    .withColumn("is_primary", F.col("is_primary").cast(IntegerType()))
)

# claim_procedure -> iceberg_raw.claim_procedure_raw
proc = (
    proc_raw
    .withColumn("id", F.col("id").cast("long"))
    .withColumn("claim_id", F.col("claim_id").cast("long"))
    .withColumn("icd9_code", F.col("icd9_code").cast(StringType()))
    .withColumn("icd9_description", F.col("icd9_description").cast(StringType()))
    .withColumn("quantity", F.col("quantity").cast(IntegerType()))
    .withColumn("procedure_date", F.col("procedure_date").cast(DateType()))
)

# claim_drug -> iceberg_raw.claim_drug_raw
drug = (
    drug_raw
    .withColumn("id", F.col("id").cast("long"))
    .withColumn("claim_id", F.col("claim_id").cast("long"))
    .withColumn("drug_code", F.col("drug_code").cast(StringType()))
    .withColumn("drug_name", F.col("drug_name").cast(StringType()))
    .withColumn("dosage", F.col("dosage").cast(StringType()))
    .withColumn("frequency", F.col("frequency").cast(StringType()))
    .withColumn("route", F.col("route").cast(StringType()))
    .withColumn("days", F.col("days").cast(IntegerType()))
    .withColumn("cost", F.col("cost").cast(DoubleType()))
)

# claim_vitamin -> iceberg_raw.claim_vitamin_raw
vit = (
    vit_raw
    .withColumn("id", F.col("id").cast("long"))
    .withColumn("claim_id", F.col("claim_id").cast("long"))
    .withColumn("vitamin_name", F.col("vitamin_name").cast(StringType()))
    .withColumn("dosage", F.col("dosage").cast(StringType()))
    .withColumn("days", F.col("days").cast(IntegerType()))
    .withColumn("cost", F.col("cost").cast(DoubleType()))
)

logger.info("Schemas telah discast / dinormalisasi")

# ================================================================
# 5. WRITE TO ICEBERG (RAW) - Overwrite initial partitions (sesuaikan jika mau append)
# NOTE: writeTo API digunakan agar kompatibel dengan Iceberg catalog
# ================================================================
def write_to_iceberg(df, table_name, mode="overwrite_partitions"):
    logger.info("Menulis ke Iceberg table: %s (mode=%s)", table_name, mode)
    # Gunakan overwritePartitions() untuk menggantikan data per partition (sesuai referensimu)
    if mode == "overwrite_partitions":
        df.writeTo(table_name).overwritePartitions()
    elif mode == "append":
        df.writeTo(table_name).append()
    else:
        # fallback: save as overwrite (full)
        df.writeTo(table_name).overwrite()

# Tulis setiap table; ganti mode jika kamu ingin append incremental
write_to_iceberg(hdr, "iceberg_raw.claim_header_raw", mode="overwrite_partitions")
write_to_iceberg(diag, "iceberg_raw.claim_diagnosis_raw", mode="overwrite_partitions")
write_to_iceberg(proc, "iceberg_raw.claim_procedure_raw", mode="overwrite_partitions")
write_to_iceberg(drug, "iceberg_raw.claim_drug_raw", mode="overwrite_partitions")
write_to_iceberg(vit, "iceberg_raw.claim_vitamin_raw", mode="overwrite_partitions")

logger.info("Semua tabel berhasil ditulis ke iceberg_raw.*")

# ================================================================
# 6. DONE
# ================================================================
spark.stop()
logger.info("ETL selesai, Spark session stopped")
