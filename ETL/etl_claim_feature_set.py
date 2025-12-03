#!/usr/bin/env python3
import cml.data_v1 as cmldata
from pyspark.sql import functions as F
from pyspark.sql.functions import *
from pyspark.sql.window import Window

print("=== START ETL v3 — Clinical Rule–Aware Fraud Feature Builder ===")

# -------------------------------------------------------------------
# CONNECT TO SPARK
# -------------------------------------------------------------------
conn = cmldata.get_connection("CDP-MSI")
spark = conn.get_spark_session()

# -------------------------------------------------------------------
# LOAD RAW ICEBERG TABLES (CDC OUTPUT)
# -------------------------------------------------------------------
hdr  = spark.sql("SELECT * FROM iceberg_raw.claim_header_raw")
diag = spark.sql("SELECT * FROM iceberg_raw.claim_diagnosis_raw")
proc = spark.sql("SELECT * FROM iceberg_raw.claim_procedure_raw")
drug = spark.sql("SELECT * FROM iceberg_raw.claim_drug_raw")
vit  = spark.sql("SELECT * FROM iceberg_raw.claim_vitamin_raw")

# -------------------------------------------------------------------
# LOAD CLINICAL RULES (CDC FROM MYSQL MASTER)
# -------------------------------------------------------------------
dx_proc = spark.sql("SELECT * FROM iceberg_ref.clinical_rule_dx_procedure")
dx_drug = spark.sql("SELECT * FROM iceberg_ref.clinical_rule_dx_drug")
dx_vit  = spark.sql("SELECT * FROM iceberg_ref.clinical_rule_dx_vitamin")

# -------------------------------------------------------------------
# PRIMARY DIAGNOSIS
# -------------------------------------------------------------------
diag_primary = (
    diag.where(col("is_primary") == 1)
        .groupBy("claim_id")
        .agg(
            first("icd10_code").alias("primary_dx"),
            first("icd10_description").alias("primary_dx_desc")
        )
)

# -------------------------------------------------------------------
# AGGREGATE RAW ARRAYS
# -------------------------------------------------------------------
proc_agg = proc.groupBy("claim_id").agg(
    collect_list("icd9_code").alias("proc_codes"),
    collect_list("icd9_description").alias("proc_desc"),
    collect_list("cost").alias("proc_costs")
)

drug_agg = drug.groupBy("claim_id").agg(
    collect_list("drug_code").alias("drug_codes"),
    collect_list("drug_name").alias("drug_names"),
    collect_list("cost").alias("drug_costs")
)

vit_agg = vit.groupBy("claim_id").agg(
    collect_list("vitamin_name").alias("vitamin_names"),
    collect_list("cost").alias("vitamin_costs")
)

# -------------------------------------------------------------------
# JOIN RAW TABLES
# -------------------------------------------------------------------
base = (
    hdr.join(diag_primary, "claim_id", "left")
       .join(proc_agg, "claim_id", "left")
       .join(drug_agg, "claim_id", "left")
       .join(vit_agg, "claim_id", "left")
)

# Fill empty arrays
base = (
    base.withColumn("proc_codes",   when(col("proc_codes").isNull(), array()).otherwise(col("proc_codes")))
        .withColumn("drug_codes",   when(col("drug_codes").isNull(), array()).otherwise(col("drug_codes")))
        .withColumn("vitamin_names",when(col("vitamin_names").isNull(), array()).otherwise(col("vitamin_names")))
)

# -------------------------------------------------------------------
# EXPLODE RULES PER CLAIM
# -------------------------------------------------------------------
proc_rule = dx_proc.groupBy("icd10_code").agg(
    collect_list(struct("icd9_code","is_mandatory","severity_level")).alias("rule_procs")
)

drug_rule = dx_drug.groupBy("icd10_code").agg(
    collect_list(struct("drug_code","is_mandatory","severity_level")).alias("rule_drugs")
)

vit_rule = dx_vit.groupBy("icd10_code").agg(
    collect_list(struct("vitamin_name","is_mandatory","severity_level")).alias("rule_vits")
)

rules = (
    proc_rule
        .join(drug_rule, "icd10_code", "left")
        .join(vit_rule, "icd10_code", "left")
)

# -------------------------------------------------------------------
# JOIN CLAIM + CLINICAL RULES
# -------------------------------------------------------------------
df = base.join(rules, base.primary_dx == rules.icd10_code, "left")

# -------------------------------------------------------------------
# CLINICAL MATCH CHECKING
# -------------------------------------------------------------------

# mandatory procedures
df = df.withColumn(
    "mandatory_proc_missed",
    F.size(
        F.filter("rule_procs", lambda r: (r["is_mandatory"] == 1) & (~F.array_contains("proc_codes", r["icd9_code"])))
    )
)

df = df.withColumn(
    "mandatory_drug_missed",
    F.size(
        F.filter("rule_drugs", lambda r: (r["is_mandatory"] == 1) & (~F.array_contains("drug_codes", r["drug_code"])))
    )
)

df = df.withColumn(
    "mandatory_vit_missed",
    F.size(
        F.filter("rule_vits", lambda r: (r["is_mandatory"] == 1) & (~F.array_contains("vitamin_names", r["vitamin_name"])))
    )
)

df = df.withColumn(
    "mandatory_missed_total",
    col("mandatory_proc_missed") +
    col("mandatory_drug_missed") +
    col("mandatory_vit_missed")
)

# mismatch flag
df = df.withColumn("procedure_mismatch_flag", when(col("mandatory_proc_missed") > 0, 1).otherwise(0))
df = df.withColumn("drug_mismatch_flag",      when(col("mandatory_drug_missed") > 0, 1).otherwise(0))
df = df.withColumn("vitamin_mismatch_flag",   when(col("mandatory_vit_missed") > 0, 1).otherwise(0))

df = df.withColumn(
    "mismatch_count",
    col("procedure_mismatch_flag") + col("drug_mismatch_flag") + col("vitamin_mismatch_flag")
)

# -------------------------------------------------------------------
# COST-BASED FRAUD SIGNAL
# -------------------------------------------------------------------
df = df.withColumn(
    "cost_anomaly_score",
    when(col("total_claim_amount") > 5000000, 4)
    .when(col("total_claim_amount") > 2500000, 3)
    .when(col("total_claim_amount") > 1500000, 2)
    .otherwise(1)
)

# -------------------------------------------------------------------
# FREQUENCY RISK
# -------------------------------------------------------------------
freq = df.groupBy("patient_nik").agg(count("claim_id").alias("visit_count"))

df = df.join(freq, "patient_nik", "left")

df = df.withColumn(
    "frequency_risk",
    when(col("visit_count") > 12, 3)
    .when(col("visit_count") > 6, 2)
    .when(col("visit_count") > 3, 1)
    .otherwise(0)
)

# -------------------------------------------------------------------
# COMPOSITE FRAUD SCORE (RULE ENGINE)
# -------------------------------------------------------------------
df = df.withColumn(
    "rule_violation_flag",
    when(col("mandatory_missed_total") > 0, 1)
    .when(col("cost_anomaly_score") >= 3, 1)
    .when(col("frequency_risk") >= 2, 1)
    .otherwise(0)
)

# combine human_label
df = df.withColumn(
    "final_label",
    when(col("status") == "declined", 1)
    .otherwise(col("rule_violation_flag"))
)

# -------------------------------------------------------------------
# FINAL OUTPUT COLUMNS
# -------------------------------------------------------------------
final = df.select(
    "claim_id",
    "primary_dx",
    "primary_dx_desc",
    "proc_codes",
    "drug_codes",
    "vitamin_names",
    "total_procedure_cost",
    "total_drug_cost",
    "total_vitamin_cost",
    "total_claim_amount",
    "mandatory_proc_missed",
    "mandatory_drug_missed",
    "mandatory_vit_missed",
    "mandatory_missed_total",
    "procedure_mismatch_flag",
    "drug_mismatch_flag",
    "vitamin_mismatch_flag",
    "mismatch_count",
    "cost_anomaly_score",
    "frequency_risk",
    "rule_violation_flag",
    "final_label",
    current_timestamp().alias("created_at"),
    year("visit_date").alias("visit_year"),
    month("visit_date").alias("visit_month")
)

# -------------------------------------------------------------------
# SAVE TO ICEBERG
# -------------------------------------------------------------------
final.write \
    .format("iceberg") \
    .mode("overwrite") \
    .partitionBy("visit_year","visit_month") \
    .saveAsTable("iceberg_curated.claim_feature_set")

print("=== ETL COMPLETED SUCCESSFULLY ===")