#!/usr/bin/env python3

import sys
import os
import cml.data_v1 as cmldata
from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, ArrayType, StringType
from pyspark.sql.functions import (
    col, lit, when, year, month, dayofmonth, 
    count, sum as spark_sum, avg, lag, stddev, 
    collect_list, size, unix_timestamp, current_timestamp
)

# ==============================================================================
# 1. SETUP PATH & IMPORT CONFIG (SIMPLIFIED & ROBUST)
# ==============================================================================
# Deteksi apakah kita di root project atau di folder ETL
current_working_dir = os.getcwd()
print(f"Current Working Directory: {current_working_dir}")

# Logika sederhana: Jika 'config.py' tidak ada di folder ini, coba naik satu level
if not os.path.exists(os.path.join(current_working_dir, "config.py")):
    # Coba cek di parent directory
    parent_dir = os.path.dirname(current_working_dir)
    if os.path.exists(os.path.join(parent_dir, "config.py")):
        print(f"Found config.py in parent directory: {parent_dir}")
        sys.path.insert(0, parent_dir)
    else:
        # Fallback terakhir: Coba tambahkan folder '/home/cdsw' (default CML)
        print("config.py not found in CWD or parent. Trying /home/cdsw...")
        sys.path.insert(0, "/home/cdsw")
else:
    print("Found config.py in current directory.")
    sys.path.insert(0, current_working_dir)

try:
    from config import COMPAT_RULES
    print(f"✓ Successfully loaded {len(COMPAT_RULES)} rules from config.py")
except ImportError:
    print("✗ CRITICAL ERROR: config.py not found anywhere. Please ensure it exists.")
    # Agar script tidak crash total saat interactive run, kita set empty rules
    COMPAT_RULES = {}

# ==============================================================================
# 2. SETUP SPARK SESSION
# ==============================================================================
print("Initializing Spark Session...")
spark = None # Initialize as None to handle clean exit
try:
    conn = cmldata.get_connection("CDP-MSI")
    spark = conn.get_spark_session()
    print(f"✓ Spark Session Created. App ID: {spark.sparkContext.applicationId}")
except Exception as e:
    print(f"✗ Error connecting to Spark: {e}")
    # If in script mode, exit. If interactive, print error.
    if __name__ == "__main__" and not hasattr(sys, 'ps1'): 
        sys.exit(1)

if spark is None:
    print("Stopping execution because Spark Session failed to initialize.")
    # Stop execution here for interactive mode
    raise RuntimeError("Spark Initialization Failed")

# ==============================================================================
# 3. LOAD RAW DATA (FROM ICEBERG RAW)
# ==============================================================================
print("Loading raw data from Iceberg...")

try:
    df_header = spark.table("iceberg_raw.claim_header_raw")
    df_diag   = spark.table("iceberg_raw.claim_diagnosis_raw")
    df_proc   = spark.table("iceberg_raw.claim_procedure_raw")
    df_drug   = spark.table("iceberg_raw.claim_drug_raw")
    df_vit    = spark.table("iceberg_raw.claim_vitamin_raw")
    
    # Check if data exists
    if df_header.count() == 0:
        print("⚠ WARNING: Table iceberg_raw.claim_header_raw is empty!")
    else:
        print(f"✓ Loaded {df_header.count()} rows from header.")
        
except Exception as e:
    print(f"✗ Error loading tables: {e}")
    if spark: spark.stop()
    raise e

# Filter: Hanya ambil klaim yang statusnya sudah final
df_header = df_header.filter(col("status").isin("approved", "declined"))

# ==============================================================================
# 4. DATA FLATTENING & AGGREGATION
# ==============================================================================
print("Aggregating items per claim...")

# 4.1 Diagnosis Utama
df_diag_primary = df_diag.filter(col("is_primary") == 1).select(
    col("claim_id"), 
    col("icd10_code").alias("icd10_primary")
)

# 4.2 Prosedur
df_proc_agg = df_proc.groupBy("claim_id").agg(
    collect_list("icd9_code").alias("proc_codes"),
    spark_sum("cost").alias("calc_proc_cost")
)

# 4.3 Obat
df_drug_agg = df_drug.groupBy("claim_id").agg(
    collect_list("drug_code").alias("drug_codes"),
    spark_sum("cost").alias("calc_drug_cost")
)

# 4.4 Vitamin
df_vit_agg = df_vit.groupBy("claim_id").agg(
    collect_list("vitamin_name").alias("vit_names"),
    spark_sum("cost").alias("calc_vit_cost")
)

# 4.5 Join Semua
base_df = df_header.join(df_diag_primary, "claim_id", "left") \
                   .join(df_proc_agg, "claim_id", "left") \
                   .join(df_drug_agg, "claim_id", "left") \
                   .join(df_vit_agg, "claim_id", "left")

# 4.6 Fill Null & Casting
base_df = base_df.fillna(0, subset=["calc_proc_cost", "calc_drug_cost", "calc_vit_cost"]) \
                 .withColumn("total_claim_amount", col("total_claim_amount").cast("double"))

# ==============================================================================
# 5. FEATURE ENGINEERING: CLINICAL COMPATIBILITY (Fixed UDF)
# ==============================================================================
print("Calculating clinical compatibility scores...")

# Broadcast rules
rules_bc = spark.sparkContext.broadcast(COMPAT_RULES)

# Pure Python function logic
def calculate_clinical_score_logic(diagnosis, items, item_type):
    if not diagnosis or not items:
        return 0.5
    
    rule = rules_bc.value.get(diagnosis)
    if not rule:
        return 0.5 
    
    valid_list = rule.get(item_type, [])
    if not valid_list:
        return 0.5
        
    match_count = sum(1 for x in items if x in valid_list)
    
    if len(items) == 0:
        return 1.0
        
    return float(match_count) / len(items)

# Define UDFs explicitly using F.udf
# NOTE: Removed the @udf decorator which caused TypeError
score_proc_udf = F.udf(lambda d, i: calculate_clinical_score_logic(d, i, "procedures"), DoubleType())
score_drug_udf = F.udf(lambda d, i: calculate_clinical_score_logic(d, i, "drugs"), DoubleType())
score_vit_udf  = F.udf(lambda d, i: calculate_clinical_score_logic(d, i, "vitamins"), DoubleType())

# Apply UDFs
base_df = base_df.withColumn("score_proc", score_proc_udf("icd10_primary", "proc_codes")) \
                 .withColumn("score_drug", score_drug_udf("icd10_primary", "drug_codes")) \
                 .withColumn("score_vit",  score_vit_udf("icd10_primary", "vit_names"))

# ==============================================================================
# 6. FEATURE ENGINEERING: PATIENT HISTORY (Window Functions)
# ==============================================================================
print("Generating historical features (Windowing)...")

w_patient = Window.partitionBy("patient_nik").orderBy(col("visit_date").cast("timestamp").cast("long"))
w_30d = w_patient.rangeBetween(-30 * 86400, -1)

base_df = base_df.withColumn("visit_ts", col("visit_date").cast("timestamp").cast("long"))

base_df = base_df.withColumn(
    "patient_visit_last_30d", 
    count("claim_id").over(w_30d)
).withColumn(
    "patient_amount_last_30d", 
    spark_sum("total_claim_amount").over(w_30d)
).withColumn(
    "prev_visit_ts", 
    lag("visit_ts", 1).over(w_patient)
).withColumn(
    "days_since_last_visit",
    (col("visit_ts") - col("prev_visit_ts")) / 86400
)

base_df = base_df.fillna(0, subset=["patient_visit_last_30d", "patient_amount_last_30d"]) \
                 .fillna(999, subset=["days_since_last_visit"])

# ==============================================================================
# 7. FEATURE ENGINEERING: PROVIDER PROFILING (Statistical Z-Score)
# ==============================================================================
print("Generating cost anomaly features (Z-Score)...")

w_diagnosis = Window.partitionBy("icd10_primary")

base_df = base_df.withColumn("avg_cost_diagnosis", avg("total_claim_amount").over(w_diagnosis)) \
                 .withColumn("std_cost_diagnosis", stddev("total_claim_amount").over(w_diagnosis))

base_df = base_df.withColumn(
    "cost_z_score", 
    (col("total_claim_amount") - col("avg_cost_diagnosis")) / (col("std_cost_diagnosis") + 0.01)
)

base_df = base_df.fillna(0, subset=["cost_z_score"])

# ==============================================================================
# 8. LABELING & CLEANUP
# ==============================================================================
print("Finalizing dataset...")

base_df = base_df.withColumn("label", when(col("status") == "declined", 1).otherwise(0))

base_df = base_df.withColumn("visit_year", year("visit_date")) \
                 .withColumn("visit_month", month("visit_date")) \
                 .withColumn("visit_day", dayofmonth("visit_date"))

final_columns = [
    "claim_id", "visit_date", "patient_nik", "doctor_name", "department", 
    "icd10_primary", "status",
    "patient_age", "total_claim_amount", 
    "calc_proc_cost", "calc_drug_cost", "calc_vit_cost",
    "score_proc", "score_drug", "score_vit",
    "patient_visit_last_30d", "patient_amount_last_30d", "days_since_last_visit",
    "cost_z_score",
    "label",
    "visit_year", "visit_month"
]

df_final = base_df.fillna("UNKNOWN", subset=["doctor_name", "department", "icd10_primary"]) \
                  .select(final_columns)

# ==============================================================================
# 9. WRITE TO ICEBERG CURATED
# ==============================================================================
target_table = "iceberg_curated.claim_training_set"
print(f"Writing result to {target_table}...")

spark.sql("CREATE DATABASE IF NOT EXISTS iceberg_curated")

try:
    df_final.write \
        .format("iceberg") \
        .mode("overwrite") \
        .partitionBy("visit_year", "visit_month") \
        .saveAsTable(target_table)
    
    print(f"✓ ETL Selesai. Data siap digunakan untuk training di: {target_table}")
    
    # Tampilkan sample data jika berhasil
    print("Sample Data:")
    df_final.show(5)
    
except Exception as e:
    print(f"✗ Error writing to Iceberg: {e}")

if spark:
    spark.stop()