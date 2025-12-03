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
    collect_list, size, unix_timestamp, current_timestamp,
    udf
)

# ==============================================================================
# 1. SETUP PATH & IMPORT CONFIG
# ==============================================================================
# Menambahkan path project root agar bisa import config.py
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir) # Naik 1 level dari folder ETL
except NameError:
    project_root = os.getcwd() # Fallback untuk interactive mode

if project_root not in sys.path:
    sys.path.insert(0, project_root)

print(f"Loading config from: {project_root}")

try:
    from config import COMPAT_RULES
    print(f"✓ Berhasil memuat {len(COMPAT_RULES)} aturan diagnosis dari config.py")
except ImportError:
    print("✗ CRITICAL: config.py tidak ditemukan. Pastikan file ada di root project.")
    sys.exit(1)

# ==============================================================================
# 2. SETUP SPARK SESSION
# ==============================================================================
print("Initializing Spark Session...")
conn = cmldata.get_connection("CDP-MSI")
spark = conn.get_spark_session()

# ==============================================================================
# 3. LOAD RAW DATA (FROM ICEBERG RAW)
# ==============================================================================
# Asumsi: Data dari MySQL sudah masuk ke tabel-tabel ini via CDC
print("Loading raw data from Iceberg...")

df_header = spark.table("iceberg_raw.claim_header_raw")
df_diag   = spark.table("iceberg_raw.claim_diagnosis_raw")
df_proc   = spark.table("iceberg_raw.claim_procedure_raw")
df_drug   = spark.table("iceberg_raw.claim_drug_raw")
df_vit    = spark.table("iceberg_raw.claim_vitamin_raw")

# Filter: Hanya ambil klaim yang statusnya sudah final (approved/declined) untuk training
# Klaim 'pending' tidak punya label ground truth
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

# 4.2 Prosedur (Array Code & Total Cost)
df_proc_agg = df_proc.groupBy("claim_id").agg(
    collect_list("icd9_code").alias("proc_codes"),
    spark_sum("cost").alias("calc_proc_cost")
)

# 4.3 Obat (Array Code & Total Cost)
df_drug_agg = df_drug.groupBy("claim_id").agg(
    collect_list("drug_code").alias("drug_codes"),
    spark_sum("cost").alias("calc_drug_cost")
)

# 4.4 Vitamin (Array Name & Total Cost)
df_vit_agg = df_vit.groupBy("claim_id").agg(
    collect_list("vitamin_name").alias("vit_names"),
    spark_sum("cost").alias("calc_vit_cost")
)

# 4.5 Join Semua ke Header
base_df = df_header.join(df_diag_primary, "claim_id", "left") \
                   .join(df_proc_agg, "claim_id", "left") \
                   .join(df_drug_agg, "claim_id", "left") \
                   .join(df_vit_agg, "claim_id", "left")

# 4.6 Fill Null & Casting
base_df = base_df.fillna(0, subset=["calc_proc_cost", "calc_drug_cost", "calc_vit_cost"]) \
                 .withColumn("total_claim_amount", col("total_claim_amount").cast("double"))

# ==============================================================================
# 5. FEATURE ENGINEERING: CLINICAL COMPATIBILITY (Rule-Based)
# ==============================================================================
print("Calculating clinical compatibility scores...")

# Broadcast rules agar worker nodes punya akses cepat
rules_bc = spark.sparkContext.broadcast(COMPAT_RULES)

# UDF (User Defined Function) untuk menghitung skor
def calculate_clinical_score(diagnosis, items, item_type):
    """
    Mengembalikan skor 0.0 (bad) sampai 1.0 (good).
    0.5 jika diagnosis tidak diketahui atau tidak ada item (netral).
    """
    if not diagnosis or not items:
        return 0.5
    
    # Ambil rule untuk diagnosis ini
    rule = rules_bc.value.get(diagnosis)
    if not rule:
        return 0.5 # Unknown diagnosis
    
    valid_list = rule.get(item_type, [])
    if not valid_list:
        return 0.5
        
    # Hitung berapa item yang ada di whitelist
    match_count = sum(1 for x in items if x in valid_list)
    
    # Hindari pembagian nol
    if len(items) == 0:
        return 1.0
        
    return float(match_count) / len(items)

# Register UDFs
udf_score_proc = udf(lambda d, i: calculate_clinical_score(d, i, "procedures"), DoubleType())
udf_score_drug = udf(lambda d, i: calculate_clinical_score(d, i, "drugs"), DoubleType())
udf_score_vit  = udf(lambda d, i: calculate_clinical_score(d, i, "vitamins"), DoubleType())

# Apply UDFs
base_df = base_df.withColumn("score_proc", udf_score_proc("icd10_primary", "proc_codes")) \
                 .withColumn("score_drug", udf_score_drug("icd10_primary", "drug_codes")) \
                 .withColumn("score_vit",  udf_score_vit("icd10_primary", "vit_names"))

# ==============================================================================
# 6. FEATURE ENGINEERING: PATIENT HISTORY (Window Functions)
# ==============================================================================
print("Generating historical features (Windowing)...")

# Window: Partisi per Pasien, Urutkan berdasarkan Waktu Kunjungan
# Penting untuk mendeteksi 'Doctor Shopping' atau frekuensi tidak wajar
w_patient = Window.partitionBy("patient_nik").orderBy(col("visit_date").cast("timestamp").cast("long"))

# Window 30 Hari (86400 detik * 30)
# Range antara -30 hari sampai -1 detik (tidak termasuk baris saat ini agar tidak bocor/leakage)
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

# Bersihkan null values hasil windowing
base_df = base_df.fillna(0, subset=["patient_visit_last_30d", "patient_amount_last_30d"]) \
                 .fillna(999, subset=["days_since_last_visit"]) # 999 artinya kunjungan pertama

# ==============================================================================
# 7. FEATURE ENGINEERING: PROVIDER PROFILING (Statistical Z-Score)
# ==============================================================================
print("Generating cost anomaly features (Z-Score)...")

# Kita ingin tahu: "Apakah klaim ini jauh lebih mahal dari rata-rata klaim 
# dengan diagnosis yang sama?"
w_diagnosis = Window.partitionBy("icd10_primary")

base_df = base_df.withColumn("avg_cost_diagnosis", avg("total_claim_amount").over(w_diagnosis)) \
                 .withColumn("std_cost_diagnosis", stddev("total_claim_amount").over(w_diagnosis))

# Hitung Z-Score: (Value - Mean) / StdDev
# Tambahkan epsilon (0.01) untuk menghindari pembagian dengan nol
base_df = base_df.withColumn(
    "cost_z_score", 
    (col("total_claim_amount") - col("avg_cost_diagnosis")) / (col("std_cost_diagnosis") + 0.01)
)

# Handle null Z-score (jika hanya ada 1 data untuk diagnosis tsb)
base_df = base_df.fillna(0, subset=["cost_z_score"])

# ==============================================================================
# 8. LABELING & CLEANUP
# ==============================================================================
print("Finalizing dataset...")

# Buat Label Target: 1 = Fraud (Declined), 0 = Normal (Approved)
base_df = base_df.withColumn("label", when(col("status") == "declined", 1).otherwise(0))

# Ekstrak fitur waktu tambahan
base_df = base_df.withColumn("visit_year", year("visit_date")) \
                 .withColumn("visit_month", month("visit_date")) \
                 .withColumn("visit_day", dayofmonth("visit_date"))

# Pilih kolom final untuk Training
final_columns = [
    # ID & Metadata
    "claim_id", "visit_date", "patient_nik", "doctor_name", "department", 
    "icd10_primary", "status",
    
    # Numeric Features (Cost & Age)
    "patient_age", "total_claim_amount", 
    "calc_proc_cost", "calc_drug_cost", "calc_vit_cost",
    
    # Clinical Scores (Feature Utama)
    "score_proc", "score_drug", "score_vit",
    
    # History Features (Advanced)
    "patient_visit_last_30d", "patient_amount_last_30d", "days_since_last_visit",
    
    # Profiling Features (Statistical)
    "cost_z_score",
    
    # Target Variable
    "label",
    
    # Partitioning Columns
    "visit_year", "visit_month"
]

# Pastikan kolom doctor_name dan department tidak null (penting untuk categorical encoding nanti)
df_final = base_df.fillna("UNKNOWN", subset=["doctor_name", "department", "icd10_primary"]) \
                  .select(final_columns)

# ==============================================================================
# 9. WRITE TO ICEBERG CURATED
# ==============================================================================
target_table = "iceberg_curated.claim_training_set"
print(f"Writing result to {target_table}...")

# Pastikan database ada
spark.sql("CREATE DATABASE IF NOT EXISTS iceberg_curated")

# Tulis data
# Menggunakan dynamic partitioning berdasarkan tahun dan bulan agar efisien
df_final.write \
    .format("iceberg") \
    .mode("overwrite") \
    .partitionBy("visit_year", "visit_month") \
    .saveAsTable(target_table)

print(f"✓ ETL Selesai. Data siap digunakan untuk training di: {target_table}")
print(f"✓ Total data processed: {df_final.count()}")

spark.stop()