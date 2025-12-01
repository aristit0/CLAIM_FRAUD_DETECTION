#!/usr/bin/env python3
import logging
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

LOCAL_JAR = "/home/cdsw/mysql-connector-java-8.0.13.jar"

# ================================================================
# 1. Spark Session + Load JDBC JAR LOCALLY
# ================================================================
if USE_CML:
    conn = cmldata.get_connection("CDP-MSI")
    spark = conn.get_spark_session()

    # IMPORTANT: REGISTER JAR TO DRIVER + EXECUTORS
    spark._jsc.addJar("file://" + LOCAL_JAR)
    spark.sparkContext.addPyFile(LOCAL_JAR)

    print("\nLoaded MySQL JDBC:", LOCAL_JAR)

    # Check driver exists
    try:
        spark._sc._jvm.java.lang.Class.forName(jdbc_driver)
        print("DRIVER OK")
    except Exception as e:
        print("DRIVER NOT FOUND:", e)
else:
    spark = (
        SparkSession.builder
        .appName("mysql-etl")
        .config("spark.jars", LOCAL_JAR)
        .getOrCreate()
    )

# ================================================================
# READ TABLE
# ================================================================
def read_mysql(table):
    return (
        spark.read.format("jdbc")
        .option("url", jdbc_url)
        .option("dbtable", table)
        .option("user", JDBC_USER)
        .option("password", JDBC_PASS)
        .option("driver", jdbc_driver)
        .load()
    )

hdr_raw = read_mysql("claim_header")