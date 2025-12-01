#!/usr/bin/env python3
import cml.data_v1 as cmldata
from pyspark.sql import functions as F, types as T

# ================================================================
# 0. CONNECT TO SPARK
# ================================================================
CONNECTION_NAME = "CDP-MSI"
conn = cmldata.get_connection(CONNECTION_NAME)
spark = conn.get_spark_session()

print("=== CONNECTED TO SPARK ===")

spark.sql("CREATE DATABASE IF NOT EXISTS iceberg_ref")
print("Database iceberg_ref ensured.")


# ================================================================
# 1. DROP + RECREATE TABLES
# ================================================================
print("=== DROPPING OLD TABLES ===")

spark.sql("DROP TABLE IF EXISTS iceberg_ref.icd10_icd9_map PURGE")
spark.sql("DROP TABLE IF EXISTS iceberg_ref.icd10_drug_map PURGE")
spark.sql("DROP TABLE IF EXISTS iceberg_ref.icd10_vitamin_map PURGE")

print("=== RECREATING TABLES ===")

# ------------------------------------------------------------
# ICD10 ↔ ICD9 TABLE
# ------------------------------------------------------------
spark.sql("""
CREATE TABLE iceberg_ref.icd10_icd9_map (
    icd10_code           STRING,
    icd9_code            STRING,
    relationship_type    STRING,
    is_first_line        BOOLEAN,
    min_age              INT,
    max_age              INT,
    min_severity         INT,
    max_severity         INT,
    notes                STRING,
    source               STRING,
    is_active            BOOLEAN,
    created_at           TIMESTAMP,
    updated_at           TIMESTAMP
)
USING iceberg
TBLPROPERTIES ('format-version'='2');
""")

# ------------------------------------------------------------
# ICD10 ↔ DRUG TABLE
# ------------------------------------------------------------
spark.sql("""
CREATE TABLE iceberg_ref.icd10_drug_map (
    icd10_code           STRING,
    drug_code            STRING,
    relationship_type    STRING,
    is_first_line        BOOLEAN,
    min_age              INT,
    max_age              INT,
    min_severity         INT,
    max_severity         INT,
    max_duration_days    INT,
    notes                STRING,
    source               STRING,
    is_antibiotic        BOOLEAN,
    is_injectable        BOOLEAN,
    is_active            BOOLEAN,
    created_at           TIMESTAMP,
    updated_at           TIMESTAMP
)
USING iceberg
TBLPROPERTIES ('format-version'='2');
""")

# ------------------------------------------------------------
# ICD10 ↔ VITAMIN TABLE
# ------------------------------------------------------------
spark.sql("""
CREATE TABLE iceberg_ref.icd10_vitamin_map (
    icd10_code           STRING,
    vitamin_name         STRING,
    relationship_type    STRING,
    is_first_line        BOOLEAN,
    min_age              INT,
    max_age              INT,
    min_severity         INT,
    max_severity         INT,
    max_duration_days    INT,
    notes                STRING,
    source               STRING,
    is_active            BOOLEAN,
    created_at           TIMESTAMP,
    updated_at           TIMESTAMP
)
USING iceberg
TBLPROPERTIES ('format-version'='2');
""")

print("=== TABLES CREATED ===")


# ================================================================
# 2. INSERT DATA (sample realistic mapping)
# ================================================================
now_expr = F.current_timestamp()

# ------------------------------------------------------------
# 2A. ICD10 ↔ ICD9
# ------------------------------------------------------------
icd10_icd9_rows = [
    ("I10", "03.31", "PRIMARY",  True,  18, None, 2, 4,
     "Hipertensi: lab basic / cek darah rutin",
     "internal_guideline_v1", True),

    ("I10", "96.70", "OPTIONAL", False, 18, None, 2, 4,
     "Injeksi obat antihipertensi (bila diperlukan)",
     "internal_guideline_v1", True),

    ("J06", "03.31", "OPTIONAL", False, 0, None, 1, 2,
     "Common cold: cek darah bila dicurigai infeksi berat",
     "internal_guideline_v1", True),

    ("A09", "03.31", "PRIMARY", True, 0, None, 1, 3,
     "Diare: cek darah / elektrolit",
     "internal_guideline_v1", True),

    ("A09", "96.70", "OPTIONAL", False, 0, None, 2, 4,
     "Injeksi obat bila dehydration / severe case",
     "internal_guideline_v1", True),

    ("K29", "03.31", "PRIMARY", True, 12, None, 1, 3,
     "Gastritis: cek darah dasar",
     "internal_guideline_v1", True),

    ("K29", "45.13", "SECONDARY", False, 12, None, 2, 4,
     "Endoskopi bila indikasi",
     "internal_guideline_v1", True),

    ("E11", "03.31", "PRIMARY", True, 18, None, 1, 4,
     "DM2: lab rutin",
     "internal_guideline_v1", True),

    ("E11", "96.70", "OPTIONAL", False, 18, None, 2, 4,
     "Injeksi insulin / terapi injeksi lainnya",
     "internal_guideline_v1", True),

    ("J45", "96.70", "PRIMARY", True, 0, None, 2, 4,
     "Asma: nebulizer / bronchodilator injeksi",
     "internal_guideline_v1", True),
]

schema_icd9 = T.StructType([
    T.StructField("icd10_code",        T.StringType()),
    T.StructField("icd9_code",         T.StringType()),
    T.StructField("relationship_type", T.StringType()),
    T.StructField("is_first_line",     T.BooleanType()),
    T.StructField("min_age",           T.IntegerType()),
    T.StructField("max_age",           T.IntegerType()),
    T.StructField("min_severity",      T.IntegerType()),
    T.StructField("max_severity",      T.IntegerType()),
    T.StructField("notes",             T.StringType()),
    T.StructField("source",            T.StringType()),
    T.StructField("is_active",         T.BooleanType()),
])

df_icd10_icd9 = spark.createDataFrame(icd10_icd9_rows, schema=schema_icd9) \
    .withColumn("created_at", now_expr) \
    .withColumn("updated_at", now_expr)

df_icd10_icd9.writeTo("iceberg_ref.icd10_icd9_map").overwritePartitions()
print("✓ Loaded ICD10–ICD9 compatibility")


# ------------------------------------------------------------
# 2B. ICD10 ↔ DRUG
# ------------------------------------------------------------
icd10_drug_rows = [
    ("I10", "KFA004", "SECOND_LINE", False, 18, None, 2, 3, 30,
     "Contoh saja; real antihipertensi harus disesuaikan",
     "internal_guideline_v1", False, False, True),

    ("J06", "KFA001", "FIRST_LINE", True, 0, None, 1, 2, 7,
     "Paracetamol untuk common cold",
     "internal_guideline_v1", False, False, True),

    ("J06", "KFA002", "SECOND_LINE", False, 0, None, 1, 2, 7,
     "Amoxicillin bila bacterial infection dicurigai",
     "internal_guideline_v1", False, False, True),

    ("A09", "KFA005", "FIRST_LINE", True, 0, None, 1, 3, 5,
     "ORS untuk diare",
     "internal_guideline_v1", False, False, True),

    ("K29", "KFA004", "FIRST_LINE", True, 12, None, 1, 3, 30,
     "Gastritis: omeprazole",
     "internal_guideline_v1", False, False, True),

    ("E11", "KFA003", "SECOND_LINE", False, 18, None, 2, 4, 14,
     "Ceftriaxone injeksi contoh high-cost",
     "internal_guideline_v1", True, True, True),

    ("J45", "KFA003", "SECOND_LINE", False, 0, None, 2, 4, 14,
     "Asma akut + infeksi bakteri",
     "internal_guideline_v1", True, True, True),
]

schema_drug = T.StructType([
    T.StructField("icd10_code",        T.StringType()),
    T.StructField("drug_code",         T.StringType()),
    T.StructField("relationship_type", T.StringType()),
    T.StructField("is_first_line",     T.BooleanType()),
    T.StructField("min_age",           T.IntegerType()),
    T.StructField("max_age",           T.IntegerType()),
    T.StructField("min_severity",      T.IntegerType()),
    T.StructField("max_severity",      T.IntegerType()),
    T.StructField("max_duration_days", T.IntegerType()),
    T.StructField("notes",             T.StringType()),
    T.StructField("source",            T.StringType()),
    T.StructField("is_antibiotic",     T.BooleanType()),
    T.StructField("is_injectable",     T.BooleanType()),
    T.StructField("is_active",         T.BooleanType()),
])

df_icd10_drug = spark.createDataFrame(icd10_drug_rows, schema=schema_drug) \
    .withColumn("created_at", now_expr) \
    .withColumn("updated_at", now_expr)

df_icd10_drug.writeTo("iceberg_ref.icd10_drug_map").overwritePartitions()
print("✓ Loaded ICD10–DRUG compatibility")


# ------------------------------------------------------------
# 2C. ICD10 ↔ VITAMINS
# ------------------------------------------------------------
icd10_vit_rows = [
    ("I10", "Vitamin D 1000 IU", "SUPPORTIVE", False, 18, None, 1, 3, 90,
     "Vitamin D suportif untuk hipertensi",
     "internal_guideline_v1", True),

    ("I10", "Vitamin B Complex", "SUPPORTIVE", False, 18, None, 1, 3, 90,
     "Suportif neuropathy / kelelahan",
     "internal_guideline_v1", True),

    ("J06", "Vitamin C 500 mg", "SUPPORTIVE", True, 0, None, 1, 2, 14,
     "Vitamin C sering diberikan untuk common cold",
     "internal_guideline_v1", True),

    ("A09", "Vitamin D 1000 IU", "OPTIONAL", False, 0, None, 1, 3, 30,
     "Suportif bila diperlukan",
     "internal_guideline_v1", True),

    ("K29", "Vitamin E 400 IU", "SUPPORTIVE", False, 12, None, 1, 3, 30,
     "Vitamin E antioksidan contoh",
     "internal_guideline_v1", True),

    ("E11", "Vitamin B Complex", "SUPPORTIVE", True, 18, None, 1, 4, 365,
     "DM2: vitamin B untuk neuropati",
     "internal_guideline_v1", True),

    ("J45", "Vitamin D 1000 IU", "SUPPORTIVE", False, 0, None, 1, 3, 90,
     "Asma: vitamin D tambahan",
     "internal_guideline_v1", True),
]

schema_vit = T.StructType([
    T.StructField("icd10_code",        T.StringType()),
    T.StructField("vitamin_name",      T.StringType()),
    T.StructField("relationship_type", T.StringType()),
    T.StructField("is_first_line",     T.BooleanType()),
    T.StructField("min_age",           T.IntegerType()),
    T.StructField("max_age",           T.IntegerType()),
    T.StructField("min_severity",      T.IntegerType()),
    T.StructField("max_severity",      T.IntegerType()),
    T.StructField("max_duration_days", T.IntegerType()),
    T.StructField("notes",             T.StringType()),
    T.StructField("source",            T.StringType()),
    T.StructField("is_active",         T.BooleanType()),
])

df_icd10_vit = spark.createDataFrame(icd10_vit_rows, schema=schema_vit) \
    .withColumn("created_at", now_expr) \
    .withColumn("updated_at", now_expr)

df_icd10_vit.writeTo("iceberg_ref.icd10_vitamin_map").overwritePartitions()
print("✓ Loaded ICD10–VITAMIN compatibility")


# ================================================================
# 3. PREVIEW
# ================================================================
print("\n=== PREVIEW ICD10 ↔ ICD9 ===")
spark.table("iceberg_ref.icd10_icd9_map").show(truncate=False)

print("\n=== PREVIEW ICD10 ↔ DRUG ===")
spark.table("iceberg_ref.icd10_drug_map").show(truncate=False)

print("\n=== PREVIEW ICD10 ↔ VITAMIN ===")
spark.table("iceberg_ref.icd10_vitamin_map").show(truncate=False)

print("\n=== DONE: Clinical Compatibility Matrix recreated ===")