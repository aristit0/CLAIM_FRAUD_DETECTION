import cml.data_v1 as cmldata
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
import random
import pandas as pd
import numpy as np

CONNECTION_NAME = "CDP-MSI"
conn = cmldata.get_connection(CONNECTION_NAME)
spark = conn.get_spark_session()
print("=== START SYNTHETIC GENERATION ===")

# ===================================================================
# 📌 1. Reference ICD Compatibility Dictionaries
# ===================================================================
ICD10_CHAPTERS = {
    "I10": {  # Hypertension
        "desc": "Hipertensi",
        "procedures": ["03.31", "89.52", "90.59"],
        "drugs": ["DRG001", "DRG002"],
        "vitamins": ["Vit-B", "Vit-C"]
    },
    "J20": {  # Acute bronchitis
        "desc": "Bronkitis Akut",
        "procedures": ["89.52"],
        "drugs": ["DRG010", "DRG011"],
        "vitamins": ["Vit-C"]
    },
    "E11": {  # Diabetes Mellitus Type-2
        "desc": "Diabetes Tipe 2",
        "procedures": ["90.59"],
        "drugs": ["DRG020", "DRG021"],
        "vitamins": ["Vit-B"]
    },
    "K29": {
        "desc": "Gastritis",
        "procedures": ["03.31"],
        "drugs": ["DRG030"],
        "vitamins": ["Vit-E"]
    },
    "M54": {
        "desc": "Low Back Pain",
        "procedures": ["99.39"],
        "drugs": ["DRG040", "DRG041"],
        "vitamins": ["Vit-D"]
    }
}

DEPARTMENTS = [
    "Poli Saraf", "Poli Jantung", "Poli Anak",
    "Poli Umum", "Poli Penyakit Dalam"
]

VISIT_TYPES = ["rawat jalan", "rawat inap", "IGD"]

# ===================================================================
# 📌 2. Helper random functions
# ===================================================================
def rand_amount(min_val, max_val):
    return round(random.uniform(min_val, max_val), 2)

def pick(list_obj, n=1):
    return random.sample(list_obj, n)

# ===================================================================
# 📌 3. Generate NORMAL claim
# ===================================================================
def generate_normal_claim(claim_id):
    icd10 = random.choice(list(ICD10_CHAPTERS.keys()))
    ref = ICD10_CHAPTERS[icd10]

    patient_name = random.choice(["Ahmad", "Budi", "Siti", "Rahma", "Aulia", "Joko"])
    gender = random.choice(["M", "F"])
    dob_year = random.randint(1960, 2010)
    visit_date = f"2025-11-{random.randint(1, 28):02d}"

    # Costs
    total_proc = rand_amount(50_000, 300_000)
    total_drug = rand_amount(20_000, 200_000)
    total_vit = rand_amount(10_000, 40_000)
    total_claim = total_proc + total_drug + total_vit

    return {
        "claim_id": claim_id,
        "patient_nik": str(3000000000000000 + claim_id),
        "patient_name": patient_name,
        "patient_gender": gender,
        "patient_dob": f"{dob_year}-01-01",
        "patient_address": "Jl. Melati No. 10",
        "patient_phone": "08123456789",
        "visit_date": visit_date,
        "visit_type": random.choice(VISIT_TYPES),
        "doctor_name": "dr. Bagus",
        "department": random.choice(DEPARTMENTS),
        "icd10_code": icd10,
        "icd10_description": ref["desc"],
        "procedures": pick(ref["procedures"], 1),
        "drugs": pick(ref["drugs"], 1),
        "vitamins": pick(ref["vitamins"], 1),
        "total_procedure_cost": total_proc,
        "total_drug_cost": total_drug,
        "total_vitamin_cost": total_vit,
        "total_claim_amount": total_claim,
        "label": 0
    }

# ===================================================================
# 📌 4. Generate FRAUD claim (several types)
# ===================================================================
def generate_fraud_claim(claim_id):
    icd10 = random.choice(list(ICD10_CHAPTERS.keys()))
    ref = ICD10_CHAPTERS[icd10]

    fraud_type = random.choice(["wrong_vitamin", "wrong_drug", "wrong_proc", "over_cost"])

    # Pick wrong items
    def wrong_item(all_items, correct):
        wrongs = [x for x in all_items if x not in correct]
        return random.choice(wrongs) if wrongs else random.choice(all_items)

    all_procs = ["03.31", "89.52", "90.59", "99.39", "00.00"]
    all_drugs = ["DRG001","DRG002","DRG010","DRG011","DRG020","DRG021","DRG030","DRG040","DRG041"]
    all_vits = ["Vit-A","Vit-B","Vit-C","Vit-D","Vit-E","Vit-K"]

    # base normal
    data = generate_normal_claim(claim_id)
    data["label"] = 1

    if fraud_type == "wrong_vitamin":
        data["vitamins"] = [wrong_item(all_vits, ref["vitamins"])]

    elif fraud_type == "wrong_drug":
        data["drugs"] = [wrong_item(all_drugs, ref["drugs"])]

    elif fraud_type == "wrong_proc":
        data["procedures"] = [wrong_item(all_procs, ref["procedures"])]

    elif fraud_type == "over_cost":
        data["total_procedure_cost"] *= random.uniform(5, 12)
        data["total_claim_amount"] = (
            data["total_procedure_cost"] +
            data["total_drug_cost"] +
            data["total_vitamin_cost"]
        )

    return data

# ===================================================================
# 📌 5. Create dataset
# ===================================================================
NORMAL_COUNT = 1000
FRAUD_COUNT = 300

all_data = []

claim_id = 1
for _ in range(NORMAL_COUNT):
    all_data.append(generate_normal_claim(claim_id))
    claim_id += 1
for _ in range(FRAUD_COUNT):
    all_data.append(generate_fraud_claim(claim_id))
    claim_id += 1

df = pd.DataFrame(all_data)
print("Generated synthetic:", df.shape)

# ===================================================================
# 6. Explode into RAW tables
# ===================================================================
header_df = df[[
    "claim_id","patient_nik","patient_name","patient_gender","patient_dob",
    "patient_address","patient_phone","visit_date","visit_type","doctor_name",
    "department","total_procedure_cost","total_drug_cost","total_vitamin_cost",
    "total_claim_amount"
]]

diag_df = df[[
    "claim_id","icd10_code","icd10_description"
]]
diag_df["is_primary"] = 1

proc_records = []
drug_records = []
vit_records = []

for idx, row in df.iterrows():
    for p in row["procedures"]:
        proc_records.append([row["claim_id"], p, "Procedure Desc", 1, row["visit_date"]])
    for d in row["drugs"]:
        drug_records.append([row["claim_id"], d, "Drug Name", "1x", "1x", "oral", 1, rand_amount(10_000, 50_000)])
    for v in row["vitamins"]:
        vit_records.append([row["claim_id"], v, "1x", 1, rand_amount(10_000, 40_000)])

proc_df = pd.DataFrame(proc_records, columns=[
    "claim_id","icd9_code","icd9_description","quantity","procedure_date"
])

drug_df = pd.DataFrame(drug_records, columns=[
    "claim_id","drug_code","drug_name","dosage","frequency","route","days","cost"
])

vit_df = pd.DataFrame(vit_records, columns=[
    "claim_id","vitamin_name","dosage","days","cost"
])

# Convert to spark
spark_header = spark.createDataFrame(header_df)
spark_diag = spark.createDataFrame(diag_df)
spark_proc = spark.createDataFrame(proc_df)
spark_drug = spark.createDataFrame(drug_df)
spark_vit = spark.createDataFrame(vit_df)

# Write to Iceberg raw tables
spark_header.writeTo("synthetic.claim_header_raw").overwritePartitions()
spark_diag.writeTo("synthetic.claim_diagnosis_raw").overwritePartitions()
spark_proc.writeTo("synthetic.claim_procedure_raw").overwritePartitions()
spark_drug.writeTo("synthetic.claim_drug_raw").overwritePartitions()
spark_vit.writeTo("synthetic.claim_vitamin_raw").overwritePartitions()

print("=== SYNTHETIC DATA GENERATED SUCCESSFULLY ===")
spark.stop()