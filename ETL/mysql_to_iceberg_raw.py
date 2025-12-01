#!/usr/bin/env python3
import logging
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import *
import os

# Try load CML connection
try:
    import cml.data_v1 as cmldata
    USE_CML = True
except Exception:
    USE_CML = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mysql-to-iceberg-etl")

# ================================================================
# JDBC CONFIG
# ================================================================
JDBC_HOST = "cdpmsi.tomodachis.org"
JDBC_PORT = 3306
JDBC_DB   = "claimdb"
JDBC_USER = "cloudera"
JDBC_PASS = "T1ku$H1t4m"

jdbc_url = (
    f"jdbc:mysql://{JDBC_HOST}:{JDBC_PORT}/{JDBC_DB}"
    "?useSSL=false&serverTimezone=UTC&zeroDateTimeBehavior=convertToNull"
)

jdbc_driver = "com.mysql.cj.jdbc.Driver"

jdbc_props = {
    "user": JDBC_USER,
    "password": JDBC_PASS,
    "driver": jdbc_driver
}

# ================================================================
# 1. Spark Session + Load JAR (FROM HDFS)
# ================================================================
HDFS_JAR = "hdfs:///home/aris/mysql-connector-java-8.0.13.jar"

if USE_CML:
    conn = cmldata.get_connection("CDP-MSI")
    spark = conn.get_spark_session()

    # Register JAR to driver + all executors
    spark.sparkContext.addFile(HDFS_JAR)
    spark._jsc.addJar(HDFS_JAR)

    print("Loaded MySQL JDBC from HDFS:", HDFS_JAR)

    # Test driver
    try:
        spark._sc._jvm.java.lang.Class.forName(jdbc_driver)
        print("DRIVER OK!")
    except Exception as e:
        print("DRIVER NOT FOUND:", e)

    logger.info("Spark from CML with MySQL driver from HDFS")
else:
    # Local mode (if not using CML)
    spark = (
        SparkSession.builder
        .appName("mysql-to-iceberg-etl")
        .config("spark.sql.catalog.spark_catalog", "org.apache.iceberg.spark.SparkSessionCatalog")
        .config("spark.sql.catalog.spark_catalog.type", "hive")
        .config("spark.jars", HDFS_JAR)
        .getOrCreate()
    )

# ================================================================
# 2. READ TABLE FROM MYSQL
# ================================================================
def read_mysql_table(table_name):
    logger.info(f"Reading MySQL table: {table_name}")
    return (
        spark.read
        .format("jdbc")
        .option("url", jdbc_url)
        .option("dbtable", table_name)
        .option("user", JDBC_USER)
        .option("password", JDBC_PASS)
        .option("driver", jdbc_driver)
        .load()
    )

hdr_raw  = read_mysql_table("claim_header")
diag_raw = read_mysql_table("claim_diagnosis")
proc_raw = read_mysql_table("claim_procedure")
drug_raw = read_mysql_table("claim_drug")
vit_raw  = read_mysql_table("claim_vitamin")

logger.info("All MySQL tables loaded.")

# ================================================================
# 3. CASTS → Iceberg Raw Schema
# ================================================================
hdr = (
    hdr_raw
    .withColumn("total_procedure_cost", F.col("total_procedure_cost").cast(DoubleType()))
    .withColumn("total_drug_cost",      F.col("total_drug_cost").cast(DoubleType()))
    .withColumn("total_vitamin_cost",   F.col("total_vitamin_cost").cast(DoubleType()))
    .withColumn("total_claim_amount",   F.col("total_claim_amount").cast(DoubleType()))
    .withColumn("patient_dob", F.col("patient_dob").cast(DateType()))
    .withColumn("visit_date", F.col("visit_date").cast(DateType()))
    .withColumn("created_at", F.col("created_at").cast(TimestampType()))
    .withColumn("updated_at", F.col("updated_at").cast(TimestampType()))
)

diag = (
    diag_raw
    .withColumn("id", F.col("id").cast("long"))
    .withColumn("claim_id", F.col("claim_id").cast("long"))
    .withColumn("icd10_code", F.col("icd10_code").cast(StringType()))
    .withColumn("icd10_description", F.col("icd10_description").cast(StringType()))
    .withColumn("is_primary", F.col("is_primary").cast(IntegerType()))
)

proc = (
    proc_raw
    .withColumn("id", F.col("id").cast("long"))
    .withColumn("claim_id", F.col("claim_id").cast("long"))
    .withColumn("icd9_code", F.col("icd9_code").cast(StringType()))
    .withColumn("icd9_description", F.col("icd9_description").cast(StringType()))
    .withColumn("quantity", F.col("quantity").cast(IntegerType()))
    .withColumn("procedure_date", F.col("procedure_date").cast(DateType()))
)

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

vit = (
    vit_raw
    .withColumn("id", F.col("id").cast("long"))
    .withColumn("claim_id", F.col("claim_id").cast("long"))
    .withColumn("vitamin_name", F.col("vitamin_name").cast(StringType()))
    .withColumn("dosage", F.col("dosage").cast(StringType()))
    .withColumn("days", F.col("days").cast(IntegerType()))
    .withColumn("cost", F.col("cost").cast(DoubleType()))
)

# ================================================================
# 4. WRITE TO ICEBERG RAW
# ================================================================
def write_to_iceberg(df, table):
    logger.info(f"Writing to Iceberg: {table}")
    df.writeTo(table).overwritePartitions()

write_to_iceberg(hdr,  "iceberg_raw.claim_header_raw")
write_to_iceberg(diag, "iceberg_raw.claim_diagnosis_raw")
write_to_iceberg(proc, "iceberg_raw.claim_procedure_raw")
write_to_iceberg(drug, "iceberg_raw.claim_drug_raw")
write_to_iceberg(vit,  "iceberg_raw.claim_vitamin_raw")

logger.info("All raw tables written to Iceberg.")

spark.stop()
logger.info("DONE")