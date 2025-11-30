#!/usr/bin/env python3
import cml.data_v1 as cmldata
from pyspark.sql import functions as F
from pyspark.sql import SparkSession

# ================================================================
# Connect to Spark via CML Data Connection
# ================================================================
CONNECTION_NAME = "CDP-MSI"
conn = cmldata.get_connection(CONNECTION_NAME)
spark = conn.get_spark_session()

print("=== Connected to Spark ===")

# ================================================================
# CREATE DATABASE
# ================================================================
spark.sql("CREATE DATABASE IF NOT EXISTS iceberg_ref")
print("Database iceberg_ref created.")

# ================================================================
# 1. ICD10 ↔ ICD9 (PROCEDURES)
# ================================================================
icd10_icd9_data = [
    ("I10", "03.31", "Hipertensi -> Pemeriksaan darah"),
    ("J06", "96.70", "Common cold -> Injeksi obat sederhana"),
    ("A09", "03.31", "Diare -> Pemeriksaan darah"),
    ("A09", "96.70", "Diare -> Injeksi obat"),
    ("K29", "45.13", "Gastritis -> Endoskopi"),
    ("E11", "03.31", "DM2 -> Cek darah"),
    ("E11", "96.70", "DM2 -> Injeksi insulin"),
    ("J45", "96.70", "Asma -> Nebulizer / injeksi"),
]

df_icd10_icd9 = spark.createDataFrame(
    icd10_icd9_data,
    ["icd10_code", "icd9_code", "relationship_desc"]
)

df_icd10_icd9.write.format("iceberg") \
    .mode("overwrite") \
    .saveAsTable("iceberg_ref.icd10_icd9_map")

print("Table iceberg_ref.icd10_icd9_map created.")

# ================================================================
# 2. ICD10 ↔ DRUGS
# ================================================================
icd10_drug_data = [
    ("I10", "KFA004", "Hipertensi -> Omeprazole (contoh)"),
    ("J06", "KFA001", "Common cold -> Paracetamol"),
    ("J06", "KFA002", "Common cold -> Amoxicillin"),
    ("A09", "KFA005", "Diare -> ORS / Oralit"),
    ("K29", "KFA004", "Gastritis -> Omeprazole"),
    ("E11", "KFA003", "DM2 -> Ceftriaxone (contoh)"),
    ("J45", "KFA003", "Asma -> Ceftriaxone (contoh)"),
]

df_icd10_drug = spark.createDataFrame(
    icd10_drug_data,
    ["icd10_code", "drug_code", "relationship_desc"]
)

df_icd10_drug.write.format("iceberg") \
    .mode("overwrite") \
    .saveAsTable("iceberg_ref.icd10_drug_map")

print("Table iceberg_ref.icd10_drug_map created.")

# ================================================================
# 3. ICD10 ↔ VITAMINS
# ================================================================
icd10_vitamin_data = [
    ("I10", "Vitamin D 1000 IU", "Hipertensi -> Vitamin D"),
    ("I10", "Vitamin B Complex", "Hipertensi -> Vitamin B Complex"),
    ("J06", "Vitamin C 500 mg", "Common cold -> Vitamin C"),
    ("A09", "Vitamin D 1000 IU", "Diare -> Vitamin D"),
    ("K29", "Vitamin E 400 IU", "Gastritis -> Vitamin E"),
    ("E11", "Vitamin B Complex", "DM2 -> Vitamin B Complex"),
    ("J45", "Vitamin D 1000 IU", "Asma -> Vitamin D"),
]

df_icd10_vitamin = spark.createDataFrame(
    icd10_vitamin_data,
    ["icd10_code", "vitamin_name", "relationship_desc"]
)

df_icd10_vitamin.write.format("iceberg") \
    .mode("overwrite") \
    .saveAsTable("iceberg_ref.icd10_vitamin_map")

print("Table iceberg_ref.icd10_vitamin_map created.")

# ================================================================
# SHOW RESULT
# ================================================================
print("\n=== Result Preview ===")
spark.sql("SELECT * FROM iceberg_ref.icd10_icd9_map").show(truncate=False)
spark.sql("SELECT * FROM iceberg_ref.icd10_drug_map").show(truncate=False)
spark.sql("SELECT * FROM iceberg_ref.icd10_vitamin_map").show(truncate=False)

print("\n=== DONE: Clinical Compatibility Matrix Created ===")