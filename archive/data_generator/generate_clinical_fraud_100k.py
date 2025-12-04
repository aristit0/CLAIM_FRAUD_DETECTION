#!/usr/bin/env python3
import random
from faker import Faker
from datetime import date, timedelta
import mysql.connector

TOTAL = 100_000
APPROVED_RATIO = 0.60     # 60% approved, 40% declined

DB = {
    "host": "localhost",
    "user": "root",
    "password": "Admin123",
    "database": "claimdb",
}

fake = Faker("id_ID")

# ============================================================
# MASTER Rules (simple & clean)
# ============================================================
CLINICAL_RULE = {
    "J06": {
        "procedures": ["96.70"],
        "drugs": ["KFA001", "KFA002"],
        "vitamins": ["Vitamin C 500 mg", "Vitamin Zinc"]
    },
    "K29": {
        "procedures": ["45.13"],
        "drugs": ["KFA004"],
        "vitamins": ["Vitamin E 400 IU"]
    },
    "E11": {
        "procedures": ["03.31"],
        "drugs": ["KFA006"],
        "vitamins": ["Vitamin B Complex", "Vitamin D 1000 IU"]
    },
    "I10": {
        "procedures": ["03.31"],
        "drugs": ["KFA007"],
        "vitamins": ["Vitamin D 1000 IU"]
    },
    "J45": {
        "procedures": ["96.70"],
        "drugs": ["KFA003"],
        "vitamins": ["Vitamin C 500 mg", "Vitamin D 1000 IU"]
    }
}

ICD10 = list(CLINICAL_RULE.keys())

ICD9_RANDOM = ["03.31", "45.13", "96.70", "93.90", "88.38", "99.04"]
DRUG_RANDOM = ["KFA001","KFA002","KFA003","KFA004","KFA005","KFA006","KFA007"]
VIT_RANDOM  = ["Vitamin C 500 mg","Vitamin B Complex","Vitamin D 1000 IU","Vitamin E 400 IU","Vitamin Zinc"]

# ============================================================
# Helper: generate NIK
# ============================================================
def gen_nik(gender):
    dob = fake.date_of_birth(minimum_age=20, maximum_age=70)
    dd = dob.day + (40 if gender == "F" else 0)
    mm = dob.month
    yy = dob.year % 100
    region = fake.random_int(320101, 327999)
    serial = fake.random_int(1, 9999)
    return f"{region}{dd:02d}{mm:02d}{yy:02d}{serial:04d}", dob

# ============================================================
# Generate one claim
# ============================================================
def generate_claim(is_fraud):
    dx = random.choice(ICD10)
    rule = CLINICAL_RULE[dx]

    # patient info
    gender = random.choice(["M", "F"])
    nik, dob = gen_nik(gender)
    name = fake.name_male() if gender == "M" else fake.name_female()

    address = fake.address().replace("\n", " ")
    phone = fake.phone_number()
    doctor = fake.name()
    department = random.choice(["Poli Umum", "Poli Dalam", "IGD"])
    visit_date = fake.date_between(start_date="-1y", end_date="today")

    # START WITH CORRECT COMBO
    procedure = rule["procedures"][0]
    drug = rule["drugs"][0]
    vitamin = rule["vitamins"][0]

    # COST
    proc_cost = random.randint(50_000, 150_000)
    drug_cost = random.randint(20_000, 120_000)
    vit_cost  = random.randint(10_000, 40_000)

    # FRAUD INJECTION
    if is_fraud:
        fraud_type = random.choice(["wrong_procedure", "wrong_drug", "wrong_vitamin", "multi"])
        
        if fraud_type in ["wrong_procedure", "multi"]:
            procedure = random.choice(ICD9_RANDOM)
        if fraud_type in ["wrong_drug", "multi"]:
            drug = random.choice(DRUG_RANDOM)
        if fraud_type in ["wrong_vitamin", "multi"]:
            vitamin = random.choice(VIT_RANDOM)

        status = "declined"
        fraud_label = 1
    else:
        status = "approved"
        fraud_label = 0
        fraud_type = None

    total = proc_cost + drug_cost + vit_cost

    return {
        "nik": nik,
        "name": name,
        "gender": gender,
        "dob": dob,
        "address": address,
        "phone": phone,
        "visit_date": visit_date,
        "doctor": doctor,
        "department": department,
        "dx": dx,
        "procedure": procedure,
        "drug": drug,
        "vitamin": vitamin,
        "proc_cost": proc_cost,
        "drug_cost": drug_cost,
        "vit_cost": vit_cost,
        "total": total,
        "status": status,
        "fraud_label": fraud_label,
        "fraud_type": fraud_type
    }

# ============================================================
# Insert to MySQL
# ============================================================
def insert(cursor, rec):
    cursor.execute("""
        INSERT INTO claim_header (
            patient_nik, patient_name, patient_gender, patient_dob,
            patient_address, patient_phone,
            visit_date, visit_type, doctor_name, department,
            total_procedure_cost, total_drug_cost, total_vitamin_cost,
            total_claim_amount, status
        ) VALUES (%s,%s,%s,%s,%s,%s,%s,'rawat jalan',%s,%s,%s,%s,%s,%s,%s)
    """, (
        rec["nik"], rec["name"], rec["gender"], rec["dob"],
        rec["address"], rec["phone"],
        rec["visit_date"], rec["doctor"], rec["department"],
        rec["proc_cost"], rec["drug_cost"], rec["vit_cost"],
        rec["total"], rec["status"]
    ))

    claim_id = cursor.lastrowid

    cursor.execute("""
        INSERT INTO claim_diagnosis (claim_id, icd10_code, icd10_description, is_primary)
        VALUES (%s,%s,%s,1)
    """, (claim_id, rec["dx"], rec["dx"]))

    cursor.execute("""
        INSERT INTO claim_procedure (claim_id, icd9_code, icd9_description, quantity, procedure_date)
        VALUES (%s,%s,%s,1,%s)
    """, (claim_id, rec["procedure"], rec["procedure"], rec["visit_date"]))

    cursor.execute("""
        INSERT INTO claim_drug (claim_id, drug_code, drug_name, dosage, frequency, route, days, cost)
        VALUES (%s,%s,%s,'1 tablet','2x sehari','oral',3,%s)
    """, (claim_id, rec["drug"], rec["drug"], rec["drug_cost"]))

    cursor.execute("""
        INSERT INTO claim_vitamin (claim_id, vitamin_name, dosage, days, cost)
        VALUES (%s,%s,'1 tablet',3,%s)
    """, (claim_id, rec["vitamin"], rec["vit_cost"]))

    return claim_id


# ============================================================
# MAIN PROCESS
# ============================================================
def main():
    conn = mysql.connector.connect(**DB)
    cursor = conn.cursor()

    approved_target = int(TOTAL * APPROVED_RATIO)
    declined_target = TOTAL - approved_target

    print(f"Generating {TOTAL} claims (approved={approved_target}, declined={declined_target})")

    count_approved = 0
    count_declined = 0

    for i in range(TOTAL):
        is_fraud = count_declined < declined_target

        rec = generate_claim(is_fraud)
        insert(cursor, rec)
        conn.commit()

        if rec["status"] == "approved":
            count_approved += 1
        else:
            count_declined += 1

        if (i + 1) % 2000 == 0:
            print(f"{i+1}/{TOTAL} inserted...")

    print(f"DONE. Inserted Approved={count_approved}, Declined={count_declined}")

    cursor.close()
    conn.close()


if __name__ == "__main__":
    main()