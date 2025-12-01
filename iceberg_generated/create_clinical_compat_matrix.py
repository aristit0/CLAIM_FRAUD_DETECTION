#!/usr/bin/env python3
import cml.data_v1 as cmldata
from pyspark.sql import functions as F, types as T

# ================================================================
# 0. CONNECT TO SPARK
# ================================================================
CONNECTION_NAME = "CDP-MSI"
conn = cmldata.get_connection(CONNECTION_NAME)
spark = conn.get_spark_session()

print("=== Connected to Spark ===")

spark.sql("CREATE DATABASE IF NOT EXISTS iceberg_ref")
print("Database iceberg_ref ensured.")


# ================================================================
# 1. CREATE TABLE DDL (ICEBERG, MORE COMPLETE SCHEMA)
# ================================================================
print("=== Creating / Ensuring Iceberg tables ===")

# 1A. ICD10 ↔ ICD9 (PROCEDURES)
spark.sql("""
CREATE TABLE IF NOT EXISTS iceberg_ref.icd10_icd9_map (
    icd10_code           STRING,
    icd9_code            STRING,
    relationship_type    STRING,   -- PRIMARY / SECONDARY / OPTIONAL / SCREENING
    is_first_line        BOOLEAN,
    min_age              INT,      -- nullable, if not age-specific
    max_age              INT,
    min_severity         INT,      -- 1 = ringan, 4 = sangat berat (align dengan severity_score kamu)
    max_severity         INT,
    notes                STRING,
    source               STRING,   -- e.g. 'internal_guideline_v1', 'WHO', etc.
    is_active            BOOLEAN,
    created_at           TIMESTAMP,
    updated_at           TIMESTAMP
)
USING iceberg
TBLPROPERTIES ('format-version'='2');
""")

# 1B. ICD10 ↔ DRUGS
spark.sql("""
CREATE TABLE IF NOT EXISTS iceberg_ref.icd10_drug_map (
    icd10_code           STRING,
    drug_code            STRING,
    relationship_type    STRING,   -- FIRST_LINE / SECOND_LINE / ADJUVANT / PRN
    is_first_line        BOOLEAN,
    min_age              INT,
    max_age              INT,
    min_severity         INT,
    max_severity         INT,
    max_duration_days    INT,      -- typical max recommended duration
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

# 1C. ICD10 ↔ VITAMINS
spark.sql("""
CREATE TABLE IF NOT EXISTS iceberg_ref.icd10_vitamin_map (
    icd10_code           STRING,
    vitamin_name         STRING,
    relationship_type    STRING,   -- SUPPORTIVE / ADJUVANT / OPTIONAL
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

print("=== Tables ensured (if not exist) ===")


# ================================================================
# 2. POPULATE INITIAL MAPPING (SYNTHETIC BUT MORE REALISTIC)
#    NOTE: untuk production, ini sebaiknya diganti baca dari CSV/excel
# ================================================================
now_expr = F.current_timestamp()

# -------------------------------
# 2A. ICD10 ↔ ICD9
# -------------------------------
icd10_icd9_rows = [
    # icd10, icd9,   rel_type,   is_first, min_age, max_age, min_sev, max_sev, notes, source,   is_active
    ("I10", "03.31", "PRIMARY",  True,     18,     None,    2,       4,
     "Hipertensi: lab basic / cek darah rutin",
     "internal_guideline_v1", True),

    ("I10", "96.70", "OPTIONAL", False,    18,     None,    2,       4,
     "Injeksi obat antihipertensi (bila diperlukan)",
     "internal_guideline_v1", True),

    ("J06", "03.31", "OPTIONAL", False,    0,      None,    1,       2,
     "Common cold: pemeriksaan darah hanya bila dicurigai infeksi berat",
     "internal_guideline_v1", True),

    ("A09", "03.31", "PRIMARY",  True,     0,      None,    1,       3,
     "Diare: cek darah / elektrolit",
     "internal_guideline_v1", True),

    ("A09", "96.70", "OPTIONAL", False,    0,      None,    2,       4,
     "Injeksi obat bila dehydration / severe case",
     "internal_guideline_v1", True),

    ("K29", "03.31", "PRIMARY",  True,     12,     None,    1,       3,
     "Gastritis: cek darah dasar / Hb",
     "internal_guideline_v1", True),

    ("K29", "45.13", "SECONDARY", False,   12,     None,    2,       4,
     "Endoskopi bila indikasi (nyeri berat, perdarahan, dsb)",
     "internal_guideline_v1", True),

    ("E11", "03.31", "PRIMARY",  True,     18,     None,    1,       4,
     "DM tipe 2: lab rutin",
     "internal_guideline_v1", True),

    ("E11", "96.70", "OPTIONAL", False,    18,     None,    2,       4,
     "Injeksi insulin / terapi injeksi lain",
     "internal_guideline_v1", True),

    ("J45", "96.70", "PRIMARY", True,      0,      None,    2,       4,
     "Asma: nebulizer / bronchodilator injeksi",
     "internal_guideline_v1", True),
]

schema_icd9 = T.StructType([
    T.StructField("icd10_code",        T.StringType(), True),
    T.StructField("icd9_code",         T.StringType(), True),
    T.StructField("relationship_type", T.StringType(), True),
    T.StructField("is_first_line",     T.BooleanType(), True),
    T.StructField("min_age",           T.IntegerType(), True),
    T.StructField("max_age",           T.IntegerType(), True),
    T.StructField("min_severity",      T.IntegerType(), True),
    T.StructField("max_severity",      T.IntegerType(), True),
    T.StructField("notes",             T.StringType(), True),
    T.StructField("source",            T.StringType(), True),
    T.StructField("is_active",         T.BooleanType(), True),
])

df_icd10_icd9 = spark.createDataFrame(icd10_icd9_rows, schema=schema_icd9) \
    .withColumn("created_at", now_expr) \
    .withColumn("updated_at", now_expr)

df_icd10_icd9.writeTo("iceberg_ref.icd10_icd9_map").overwritePartitions()
print("✓ Loaded ICD10–ICD9 compatibility")


# -------------------------------
# 2B. ICD10 ↔ DRUGS
# -------------------------------
icd10_drug_rows = [
    # icd10, drug_code, rel_type,      is_first, min_age, max_age, min_sev, max_sev, max_days, notes, source, is_ab, is_inj, is_active
    ("I10", "KFA004", "SECOND_LINE",   False,    18,     None,    2,       3,      30,
     "Omeprazole lebih relevan untuk gastritis, contoh saja; untuk real data sebaiknya antihipertensi",
     "internal_example", False, False, True),

    ("J06", "KFA001", "FIRST_LINE",    True,     0,      None,    1,       2,      7,
     "Common cold: paracetamol",
     "internal_guideline_v1", False, False, True),

    ("J06", "KFA002", "SECOND_LINE",   False,    0,      None,    1,       2,      7,
     "Amoxicillin: hanya bila dicurigai bacterial infection",
     "internal_guideline_v1", False, False, True),

    ("A09", "KFA005", "FIRST_LINE",    True,     0,      None,    1,       3,      5,
     "Diare: ORS / oralit",
     "internal_guideline_v1", False, False, True),

    ("K29", "KFA004", "FIRST_LINE",    True,     12,     None,    1,       3,      30,
     "Gastritis: omeprazole",
     "internal_guideline_v1", False, False, True),

    ("E11", "KFA003", "SECOND_LINE",   False,    18,     None,    2,       4,      14,
     "DM2 + infeksi berat: ceftriaxone injeksi (contoh high-cost, harus hati-hati)",
     "internal_guideline_v1", True,  True,  True),

    ("J45", "KFA003", "SECOND_LINE",   False,    0,      None,    2,       4,      14,
     "Asma akut berat dengan infeksi bacterial terkonfirmasi",
     "internal_guideline_v1", True,  True,  True),
]

schema_drug = T.StructType([
    T.StructField("icd10_code",        T.StringType(), True),
    T.StructField("drug_code",         T.StringType(), True),
    T.StructField("relationship_type", T.StringType(), True),
    T.StructField("is_first_line",     T.BooleanType(), True),
    T.StructField("min_age",           T.IntegerType(), True),
    T.StructField("max_age",           T.IntegerType(), True),
    T.StructField("min_severity",      T.IntegerType(), True),
    T.StructField("max_severity",      T.IntegerType(), True),
    T.StructField("max_duration_days", T.IntegerType(), True),
    T.StructField("notes",             T.StringType(), True),
    T.StructField("source",            T.StringType(), True),
    T.StructField("is_antibiotic",     T.BooleanType(), True),
    T.StructField("is_injectable",     T.BooleanType(), True),
    T.StructField("is_active",         T.BooleanType(), True),
])

df_icd10_drug = spark.createDataFrame(icd10_drug_rows, schema=schema_drug) \
    .withColumn("created_at", now_expr) \
    .withColumn("updated_at", now_expr)

df_icd10_drug.writeTo("iceberg_ref.icd10_drug_map").overwritePartitions()
print("✓ Loaded ICD10–Drug compatibility")


# -------------------------------
# 2C. ICD10 ↔ VITAMINS
# -------------------------------
icd10_vit_rows = [
    # icd10, vitamin_name,         rel_type,    is_first, min_age, max_age, min_sev, max_sev, max_days, notes, source, is_active
    ("I10", "Vitamin D 1000 IU",   "SUPPORTIVE", False,   18,      None,    1,       3,      90,
     "Hipertensi: vitamin D sebagai suportif, bukan utama",
     "internal_guideline_v1", True),

    ("I10", "Vitamin B Complex",   "SUPPORTIVE", False,   18,      None,    1,       3,      90,
     "Supporting therapy (misal neuropathy, dsb)",
     "internal_guideline_v1", True),

    ("J06", "Vitamin C 500 mg",    "SUPPORTIVE", True,    0,       None,    1,       2,      14,
     "Common cold: vitamin C umum dipakai",
     "internal_guideline_v1", True),

    ("A09", "Vitamin D 1000 IU",   "OPTIONAL",  False,    0,       None,    1,       3,      30,
     "Diare: vitamin D kadang dipakai sebagai suportif",
     "internal_guideline_v1", True),

    ("K29", "Vitamin E 400 IU",    "SUPPORTIVE", False,   12,      None,    1,       3,      30,
     "Gastritis: vitamin E sebagai antioksidan (contoh)",
     "internal_guideline_v1", True),

    ("E11", "Vitamin B Complex",   "SUPPORTIVE", True,    18,      None,    1,       4,      365,
     "DM2: vitamin B untuk neuropati, contoh suportif jangka panjang",
     "internal_guideline_v1", True),

    ("J45", "Vitamin D 1000 IU",   "SUPPORTIVE", False,   0,       None,    1,       3,      90,
     "Asma: vitamin D kadang dipertimbangkan",
     "internal_guideline_v1", True),
]

schema_vit = T.StructType([
    T.StructField("icd10_code",        T.StringType(), True),
    T.StructField("vitamin_name",      T.StringType(), True),
    T.StructField("relationship_type", T.StringType(), True),
    T.StructField("is_first_line",     T.BooleanType(), True),
    T.StructField("min_age",           T.IntegerType(), True),
    T.StructField("max_age",           T.IntegerType(), True),
    T.StructField("min_severity",      T.IntegerType(), True),
    T.StructField("max_severity",      T.IntegerType(), True),
    T.StructField("max_duration_days", T.IntegerType(), True),
    T.StructField("notes",             T.StringType(), True),
    T.StructField("source",            T.StringType(), True),
    T.StructField("is_active",         T.BooleanType(), True),
])

df_icd10_vit = spark.createDataFrame(icd10_vit_rows, schema=schema_vit) \
    .withColumn("created_at", now_expr) \
    .withColumn("updated_at", now_expr)

df_icd10_vit.writeTo("iceberg_ref.icd10_vitamin_map").overwritePartitions()
print("✓ Loaded ICD10–Vitamin compatibility")


# ================================================================
# 3. QUICK PREVIEW
# ================================================================
print("\n=== Preview icd10_icd9_map ===")
spark.table("iceberg_ref.icd10_icd9_map").show(truncate=False)

print("\n=== Preview icd10_drug_map ===")
spark.table("iceberg_ref.icd10_drug_map").show(truncate=False)

print("\n=== Preview icd10_vitamin_map ===")
spark.table("iceberg_ref.icd10_vitamin_map").show(truncate=False)

print("\n=== DONE: Clinical Compatibility Matrix (enhanced) ===")