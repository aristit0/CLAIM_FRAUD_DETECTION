#!/usr/bin/env python3
import logging
import mysql.connector
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import *

try:
    import cml.data_v1 as cmldata
    USE_CML = True
except:
    USE_CML = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mysql-to-iceberg-etl")

# ================================================================
# MySQL CONFIG
# ================================================================
MYSQL_CONFIG = {
    "host": "cdpmsi.tomodachis.org",
    "user": "cloudera",
    "password": "T1ku$H1t4m",
    "database": "claimdb"
}

TABLES = [
    "claim_header",
    "claim_diagnosis",
    "claim_procedure",
    "claim_drug",
    "claim_vitamin"
]

# ================================================================
# Spark Session CML
# ================================================================
if USE_CML:
    conn = cmldata.get_connection("CDP-MSI")
    spark = conn.get_spark_session()
else:
    spark = SparkSession.builder.appName("mysql-no-jdbc").getOrCreate()

# ================================================================
# Helper: read MySQL as Pandas → Spark DataFrame
# ================================================================
def read_mysql_table(table):
    logger.info(f"Reading MySQL table: {table}")

    sql = f"SELECT * FROM {table}"

    conn = mysql.connector.connect(**MYSQL_CONFIG)
    df_pd = pd.read_sql(sql, conn)
    conn.close()

    logger.info(f"MySQL → Pandas rows: {len(df_pd)}")

    df_spark = spark.createDataFrame(df_pd)

    logger.info(f"Pandas → Spark DataFrame rows: {df_spark.count()}")

    return df_spark


# ================================================================
# 1. READ TABLES
# ================================================================
hdr_raw  = read_mysql_table("claim_header")
diag_raw = read_mysql_table("claim_diagnosis")
proc_raw = read_mysql_table("claim_procedure")
drug_raw = read_mysql_table("claim_drug")
vit_raw  = read_mysql_table("claim_vitamin")

logger.info("All MySQL tables loaded WITHOUT JDBC driver.")

# ================================================================
# 2. CASTS → match Iceberg raw schema
# ================================================================
hdr = (
    hdr_raw
    .withColumn("total_procedure_cost", F.col("total_procedure_cost").cast(DoubleType()))
    .withColumn("total_drug_cost",      F.col("total_drug_cost").cast(DoubleType()))
    .withColumn("total_vitamin_cost",   F.col("total_vitamin_cost").cast(DoubleType()))
    .withColumn("total_claim_amount",   F.col("total_claim_amount").cast(DoubleType()))
    .withColumn("patient_dob", F.col("patient_dob").cast(DateType()))
    .withColumn("visit_date", F.col("visit_date").cast(DateType()))
)

diag = diag_raw.withColumn("is_primary", F.col("is_primary").cast(IntegerType()))

proc = proc_raw.withColumn("quantity", F.col("quantity").cast(IntegerType()))

drug = drug_raw.withColumn("cost", F.col("cost").cast(DoubleType()))

vit = vit_raw.withColumn("cost", F.col("cost").cast(DoubleType()))

# ================================================================
# 3. WRITE TO ICEBERG
# ================================================================
def write_to_iceberg(df, table):
    logger.info(f"Writing to Iceberg table: {table}")
    df.writeTo(table).overwritePartitions()

write_to_iceberg(hdr, "iceberg_raw.claim_header_raw")
write_to_iceberg(diag, "iceberg_raw.claim_diagnosis_raw")
write_to_iceberg(proc, "iceberg_raw.claim_procedure_raw")
write_to_iceberg(drug, "iceberg_raw.claim_drug_raw")
write_to_iceberg(vit, "iceberg_raw.claim_vitamin_raw")

logger.info("DONE — no JDBC driver needed")

spark.stop()