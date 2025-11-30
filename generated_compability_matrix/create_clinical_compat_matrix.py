#!/usr/bin/env python3
from pyspark.sql import SparkSession

spark = (
    SparkSession.builder
    .appName("create_clinical_compat_matrix")
    .config("spark.sql.catalog.spark_catalog", "org.apache.iceberg.spark.SparkSessionCatalog")
    .config("spark.sql.catalog.spark_catalog.type", "hive")
    .getOrCreate()
)

print("=== CREATE DATABASE iceberg_ref ===")
spark.sql("CREATE DATABASE IF NOT EXISTS iceberg_ref")

# ==========================================
# 1. ICD10 ↔ ICD9 (tindakan)
# ==========================================

icd10_icd9_data = [
    # Hipertensi
    ("I10", "03.31", "Hipertensi -> Pemeriksaan darah"),
    # Common cold
    ("J06", "96.70", "Common cold -> Injeksi obat sederhana"),
    # Diare
    ("A09", "03.31", "Diare -> Pemeriksaan darah"),
    ("A09", "96.70", "Diare -> Injeksi obat / cairan"),
    # Gastritis
    ("K29", "45.13", "Gastritis -> Endoskopi"),
    # Diabetes tipe 2
    ("E11", "03.31", "DM2 -> Pemeriksaan darah"),
    ("E11", "96.70", "DM2 -> Injeksi obat"),
    # Asma
    ("J45", "96.70", "Asma -> Nebulizer / injeksi obat"),
]

df_icd10_icd9 = spark.createDataFrame(
    icd10_icd9_data,
    ["icd10_code", "icd9_code", "relationship_desc"]
)

df_icd10_icd9.write.format("iceberg") \
    .mode("overwrite") \
    .saveAsTable("iceberg_ref.icd10_icd9_map")

print("icd10_icd9_map:")
spark.table("iceberg_ref.icd10_icd9_map").show(truncate=False)

# ==========================================
# 2. ICD10 ↔ DRUG (kode KFA)
# ==========================================

icd10_drug_data = [
    # Hipertensi
    ("I10", "KFA004", "Hipertensi -> Omeprazole (contoh)"),
    # Common cold
    ("J06", "KFA001", "Common cold -> Paracetamol"),
    ("J06", "KFA002", "Common cold -> Amoxicillin"),
    # Diare
    ("A09", "KFA005", "Diare -> ORS / Oralit"),
    # Gastritis
    ("K29", "KFA004", "Gastritis -> Omeprazole"),
    # Diabetes mellitus tipe 2
    ("E11", "KFA003", "DM2 -> Ceftriaxone (contoh terapi)"),
    # Asma
    ("J45", "KFA003", "Asma -> Ceftriaxone (contoh, untuk demo)")
]

df_icd10_drug = spark.createDataFrame(
    icd10_drug_data,
    ["icd10_code", "drug_code", "relationship_desc"]
)

df_icd10_drug.write.format("iceberg") \
    .mode("overwrite") \
    .saveAsTable("iceberg_ref.icd10_drug_map")

print("icd10_drug_map:")
spark.table("iceberg_ref.icd10_drug_map").show(truncate=False)

# ==========================================
# 3. ICD10 ↔ VITAMIN
# ==========================================

icd10_vitamin_data = [
    ("I10", "Vitamin D 1000 IU", "Hipertensi -> Vitamin D"),
    ("I10", "Vitamin B Complex", "Hipertensi -> Vitamin B Complex"),
    ("J06", "Vitamin C 500 mg", "Common cold -> Vitamin C"),
    ("A09", "Vitamin D 1000 IU", "Diare -> Vitamin D untuk pemulihan"),
    ("K29", "Vitamin E 400 IU", "Gastritis -> Vitamin E"),
    ("E11", "Vitamin B Complex", "DM2 -> Vitamin B Complex"),
    ("J45", "Vitamin D 1000 IU", "Asma -> Vitamin D")
]

df_icd10_vitamin = spark.createDataFrame(
    icd10_vitamin_data,
    ["icd10_code", "vitamin_name", "relationship_desc"]
)

df_icd10_vitamin.write.format("iceberg") \
    .mode("overwrite") \
    .saveAsTable("iceberg_ref.icd10_vitamin_map")

print("icd10_vitamin_map:")
spark.table("iceberg_ref.icd10_vitamin_map").show(truncate=False)

print("=== DONE CREATING CLINICAL COMPATIBILITY MATRIX TABLES ===")
spark.stop()