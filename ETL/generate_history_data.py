#!/usr/bin/env python3

import sys
import os
import cml.data_v1 as cmldata
from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, IntegerType, StringType
from pyspark.sql.functions import (
    col, lit, when, year, month, dayofmonth, 
    count, sum as spark_sum, avg, lag, stddev, 
    collect_list, size, date_sub, current_timestamp
)

# Load Config Rules (Pastikan file config.py ada di path yang sama/bisa diimport)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from config import COMPAT_RULES
except ImportError:
    # Fallback jika config.py tidak ditemukan, gunakan dummy rule untuk menghindari error
    COMPAT_RULES = {} 
    print("Warning: config.py not found, using empty rules.")

print("=" * 80)
print("ADVANCED FRAUD ETL - HISTORY & PROFILING PIPELINE")
print("=" * 80)

# ==============================================================================
# 1. SETUP SPARK
# ==============================================================================
conn = cmldata.get_connection("CDP-MSI")
spark = conn.get_spark_session()

# ==============================================================================
# 2. LOAD RAW DATA (FROM ICEBERG)
# ==============================================================================
# Asumsi data CDC sudah masuk ke tabel _raw ini
df_header = spark.table("iceberg_raw.claim_header_raw")
df_diag   = spark.table("iceberg_raw.claim_diagnosis_raw")
df_proc   = spark.table("iceberg_raw.claim_procedure_raw")
df_drug   = spark.table("iceberg_raw.claim_drug_raw")
df_vit    = spark.table("iceberg_raw.claim_vitamin_raw")

# Filter hanya klaim yang sudah selesai (approved/declined) untuk training
# Untuk inference nanti, filter status = 'pending'
df_header = df_header.filter(col("status").isin("approved", "declined"))

# ==============================================================================
# 3. PRE-PROCESSING & FLATTENING
# ==============================================================================
# Kita butuh agregat item medis per klaim
proc_agg = df_proc.groupBy("claim_id").agg(
    collect_list("icd9_code").alias("proc_codes"),
    spark_sum("cost").alias("total_proc_cost")
)
drug_agg = df_drug.groupBy("claim_id").agg(
    collect_list("drug_code").alias("drug_codes"),
    spark_sum("cost").alias("total_drug_cost")
)
vit_agg = df_vit.groupBy("claim_id").agg(
    collect_list("vitamin_name").alias("vit_names"),
    spark_sum("cost").alias("total_vit_cost")
)
# Diagnosis Utama
diag_primary = df_diag.filter(col("is_primary") == 1).select(
    "claim_id", col("icd10_code").alias("icd10_primary")
)

# Join semua ke Header
base_df = df_header.join(diag_primary, "claim_id", "left") \
                   .join(proc_agg, "claim_id", "left") \
                   .join(drug_agg, "claim_id", "left") \
                   .join(vit_agg, "claim_id", "left") \
                   .fillna(0, subset=["total_proc_cost", "total_drug_cost", "total_vit_cost"])

# Convert costs to double
for c in ["total_proc_cost", "total_drug_cost", "total_vit_cost", "total_claim_amount"]:
    base_df = base_df.withColumn(c, col(c).cast("double"))

# ==============================================================================
# 4. FEATURE ENGINEERING: CLINICAL RULES (Base Features)
# ==============================================================================
# Broadcast rules untuk efisiensi
rules_bc = spark.sparkContext.broadcast(COMPAT_RULES)

# UDF untuk hitung skor kecocokan
def calc_match_score(dx, items, item_type):
    if not dx or not items: return 0.5 # Neutral
    rule = rules_bc.value.get(dx)
    if not rule: return 0.5 # Unknown diagnosis
    
    valid_items = rule.get(item_type, [])
    if not valid_items: return 0.5
    
    # Hitung rasio item yang valid
    match_count = sum(1 for x in items if x in valid_items)
    return float(match_count) / len(items)

calc_proc_udf = F.udf(lambda d, i: calc_match_score(d, i, "procedures"), DoubleType())
calc_drug_udf = F.udf(lambda d, i: calc_match_score(d, i, "drugs"), DoubleType())
calc_vit_udf  = F.udf(lambda d, i: calc_match_score(d, i, "vitamins"), DoubleType())

base_df = base_df.withColumn("score_proc", calc_proc_udf("icd10_primary", "proc_codes")) \
                 .withColumn("score_drug", calc_drug_udf("icd10_primary", "drug_codes")) \
                 .withColumn("score_vit",  calc_vit_udf("icd10_primary", "vit_names"))

# ==============================================================================
# 5. FEATURE ENGINEERING: PATIENT HISTORY (Window Functions)
# ==============================================================================
# Penting: Urutkan berdasarkan visit_date untuk melihat pola waktu
w_patient = Window.partitionBy("patient_nik").orderBy(col("visit_date").cast("long"))

# 30 hari ke belakang (86400 detik * 30)
w_patient_30d = w_patient.rangeBetween(-30 * 86400, -1) 

base_df = base_df.withColumn(
    "patient_visit_last_30d", 
    count("claim_id").over(w_patient_30d)
).withColumn(
    "patient_amount_last_30d", 
    spark_sum("total_claim_amount").over(w_patient_30d)
).withColumn(
    "days_since_last_visit",
    (col("visit_date").cast("long") - lag("visit_date", 1).over(w_patient).cast("long")) / 86400
)

# Fill nulls (kunjungan pertama)
base_df = base_df.fillna(0, subset=["patient_visit_last_30d", "patient_amount_last_30d"]) \
                 .fillna(999, subset=["days_since_last_visit"]) # 999 artinya belum pernah datang

# ==============================================================================
# 6. FEATURE ENGINEERING: PROVIDER PROFILING (Z-Score)
# ==============================================================================
# Membandingkan klaim dokter ini dengan rata-rata klaim dokter lain di diagnosis yang sama
w_dx = Window.partitionBy("icd10_primary")

base_df = base_df.withColumn("avg_cost_for_dx", avg("total_claim_amount").over(w_dx)) \
                 .withColumn("std_cost_for_dx", stddev("total_claim_amount").over(w_dx)) \
                 .withColumn("cost_z_score", 
                             (col("total_claim_amount") - col("avg_cost_for_dx")) / 
                             (col("std_cost_for_dx") + 1.0)) # +1 biar ga divide by zero

# ==============================================================================
# 7. LABELING (TARGET VARIABLE)
# ==============================================================================
# Kita anggap 'declined' = Fraud/Abuse (1), 'approved' = Normal (0)
# Dalam real world, mungkin ada label khusus dari tim investigator
base_df = base_df.withColumn("label", when(col("status") == "declined", 1).otherwise(0))

# ==============================================================================
# 8. FINAL SELECTION & WRITE TO ICEBERG
# ==============================================================================
final_cols = [
    "claim_id", "visit_date", "patient_nik", "doctor_name", "department",
    "icd10_primary", 
    
    # Numeric Features
    "total_claim_amount", "total_proc_cost", "total_drug_cost", "total_vit_cost",
    
    # Clinical Scores
    "score_proc", "score_drug", "score_vit",
    
    # History Features
    "patient_visit_last_30d", "patient_amount_last_30d", "days_since_last_visit",
    
    # Statistical Features
    "cost_z_score",
    
    # Label
    "label",
    
    # Metadata
    current_timestamp().alias("etl_timestamp")
]

df_final = base_df.select(final_cols)

print("Sample Data:")
df_final.show(5)

# Tulis ke Iceberg Curated
# Mode 'overwrite' partisi atau 'append' tergantung kebutuhan
print("Writing to iceberg_curated.claim_training_set...")

# Pastikan database ada
spark.sql("CREATE DATABASE IF NOT EXISTS iceberg_curated")

# Tulis (Menggunakan partitioning bulanan untuk performa)
df_final.withColumn("visit_month", month("visit_date")) \
        .write \
        .format("iceberg") \
        .mode("overwrite") \
        .partitionBy("visit_month") \
        .saveAsTable("iceberg_curated.claim_training_set")

print("ETL Completed successfully.")
spark.stop()