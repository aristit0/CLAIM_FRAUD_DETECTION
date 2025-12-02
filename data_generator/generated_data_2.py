#!/usr/bin/env python3
import random
import csv
from faker import Faker
from datetime import datetime
import mysql.connector

# ================================================================
# CONFIG
# ================================================================
TOTAL_CLAIMS = 300_000         # Target 300.000 untuk training data
BATCH_COMMIT = 1               # Commit per 1 row
FRAUD_RATIO = 0.40             # 40% fraud lebih realistis untuk training

DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "Admin123",
    "database": "claimdb",
}

fake = Faker("id_ID")

# ================================================================
# HELPERS
# ================================================================
def generate_nik(gender: str):
    prov = random.randint(11, 99)
    kab = random.randint(1, 99)
    kec = random.randint(1, 99)

    dob = fake.date_of_birth(minimum_age=18, maximum_age=75)

    dd = dob.day
    mm = dob.month
    yy = dob.year % 100

    if gender == "F":
        dd += 40

    urut = random.randint(1, 9999)

    nik = f"{prov:02d}{kab:02d}{kec:02d}{dd:02d}{mm:02d}{yy:02d}{urut:04d}"
    return nik, dob


# ================================================================
# STATIC REF (enhanced)
# ================================================================
ICD10_LIST = [
    ("A09", "Gastroenteritis"),
    ("J06", "Common cold"),
    ("I10", "Hypertension"),
    ("E11", "Diabetes Type 2"),
    ("J45", "Asthma"),
    ("K29", "Gastritis"),
]

ICD9_LIST = [
    ("03.31", "Pemeriksaan darah"),
    ("45.13", "Endoskopi"),
    ("93.90", "Electrotherapy"),
    ("96.70", "Injeksi obat"),
    ("99.04", "Transfusi darah"),
    ("88.38", "X-Ray Dada"),
]

DRUG_LIST = [
    ("KFA001", "Paracetamol 500 mg"),
    ("KFA002", "Amoxicillin 500 mg"),
    ("KFA003", "Ceftriaxone injeksi"),
    ("KFA004", "Omeprazole 20 mg"),
    ("KFA005", "ORS / Oralit"),
    ("KFA006", "Metformin 500 mg"),
    ("KFA007", "Amlodipine 10 mg"),
]

VITAMIN_LIST = [
    "Vitamin C 500 mg",
    "Vitamin B Complex",
    "Vitamin D 1000 IU",
    "Vitamin E 400 IU",
    "Vitamin Zinc",
    "Vitamin A dosis tinggi"
]

# ================================================================
# CLINICAL COMPATIBILITY
# ================================================================
COMPAT = {
    "A09": {
        "procedures": ["03.31"],
        "drugs": ["KFA005"],
        "vitamins": ["Vitamin D 1000 IU", "Vitamin Zinc"]
    },
    "J06": {
        "procedures": ["96.70"],
        "drugs": ["KFA001", "KFA002"],
        "vitamins": ["Vitamin C 500 mg", "Vitamin Zinc"]
    },
    "I10": {
        "procedures": ["03.31"],
        "drugs": ["KFA007"],
        "vitamins": ["Vitamin D 1000 IU", "Vitamin B Complex"]
    },
    "E11": {
        "procedures": ["03.31"],
        "drugs": ["KFA006"],
        "vitamins": ["Vitamin B Complex", "Vitamin D 1000 IU"]
    },
    "J45": {
        "procedures": ["96.70"],
        "drugs": ["KFA003"],
        "vitamins": ["Vitamin D 1000 IU", "Vitamin C 500 mg"]
    },
    "K29": {
        "procedures": ["45.13"],
        "drugs": ["KFA004"],
        "vitamins": ["Vitamin E 400 IU"]
    }
}

# ================================================================
# FRAUD PATTERNS (REALISTIC)
# ================================================================
def inject_fraud(dx_code, procedure, drug, vitamin, proc_cost, drug_cost):
    fraud_type = None

    r = random.random()

    # 1. Wrong vitamin
    if r < 0.15:
        vitamin = random.choice(["Vitamin A dosis tinggi", "Vitamin B Complex", "Vitamin C 1000 mg"])
        fraud_type = "wrong_vitamin"

    # 2. Wrong high-power antibiotic
    elif r < 0.30:
        drug = random.choice(["KFA003", "KFA007", "KFA002"])
        fraud_type = "wrong_drug"

    # 3. Wrong procedure
    elif r < 0.45:
        procedure = random.choice([("99.99", "Surgical procedure"), ("88.38", "X-Ray abdomen")])
        fraud_type = "wrong_procedure"

    # 4. Over-cost with large variation
    elif r < 0.70:
        proc_cost = proc_cost * random.uniform(1.2, 5.0)  # more realistic high cost
        drug_cost = drug_cost * random.uniform(1.0, 4.0)
        fraud_type = "over_cost"

    # 5. Multi-fraud: Multiple fraudulent attributes
    else:
        procedure = random.choice([("99.99", "Surgical procedure"), ("88.38", "X-Ray abdomen")])
        drug = random.choice(["KFA003", "KFA007", "KFA002"])
        vitamin = random.choice(["Vitamin A dosis tinggi", "Vitamin C 1000 mg", "Vitamin B Complex"])
        proc_cost = proc_cost * random.uniform(2.0, 7.0)  # High multiplier
        drug_cost = drug_cost * random.uniform(2.0, 5.0)
        fraud_type = "multi_fraud"

    return procedure, drug, vitamin, proc_cost, drug_cost, fraud_type


# ================================================================
# GENERATE RECORD
# ================================================================
def generate_claim():
    gender = random.choice(["M", "F"])
    nik, dob = generate_nik(gender)
    name = fake.name_male() if gender == "M" else fake.name_female()

    address = fake.address().replace("\n", ", ")
    phone = fake.phone_number()
    doctor = fake.name()
    department = random.choice(["Poli Umum", "Poli Anak", "Poli Dalam", "IGD"])

    visit_date = fake.date_between(start_date='-3y', end_date='today')

    dx = random.choice(ICD10_LIST)
    dx_code = dx[0]

    allowed = COMPAT.get(dx_code)

    # normal (not fraud)
    if allowed:
        procedure = (allowed["procedures"][0], "AUTO_PROC")
        drug = (allowed["drugs"][0], "AUTO_DRUG")
        vitamin = allowed["vitamins"][0]
    else:
        procedure = random.choice(ICD9_LIST)
        drug = random.choice(DRUG_LIST)
        vitamin = random.choice(VITAMIN_LIST)

    # cost
    proc_cost = random.randint(50_000, 150_000)
    drug_cost = random.randint(10_000, 100_000)
    vit_cost = random.randint(5_000, 50_000)

    # FRAUD
    fraud_type = None
    if random.random() < FRAUD_RATIO:
        procedure, drug, vitamin, proc_cost, drug_cost, fraud_type = inject_fraud(
            dx_code, procedure, drug, vitamin, proc_cost, drug_cost
        )

    total_claim = proc_cost + drug_cost + vit_cost

    # evaluate approval
    approve = True
    if allowed:
        if procedure[0] not in allowed["procedures"]:
            approve = False
        if drug[0] not in allowed["drugs"]:
            approve = False
        if vitamin not in allowed["vitamins"]:
            approve = False

    # rule: common cold must NOT get ceftriaxone
    if dx_code == "J06" and drug[0] == "KFA003":
        approve = False

    if total_claim > 2_000_000:
        approve = False

    status = "approved" if approve else "declined"

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
        "total": total_claim,
        "fraud_label": 0 if approve else 1,
        "fraud_type": fraud_type,
        "status": status
    }


# ================================================================
# INSERTOR
# ================================================================
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
    """, (claim_id, rec["dx"][0], rec["dx"][1]))

    cursor.execute("""
        INSERT INTO claim_procedure (claim_id, icd9_code, icd9_description, quantity, procedure_date)
        VALUES (%s,%s,%s,1,%s)
    """, (claim_id, rec["procedure"][0], rec["procedure"][1], rec["visit_date"]))

    cursor.execute("""
        INSERT INTO claim_drug (claim_id, drug_code, drug_name, dosage, frequency, route, days, cost)
        VALUES (%s,%s,%s,'1 tablet','2x sehari','oral',3,%s)
    """, (claim_id, rec["drug"][0], rec["drug"][1], rec["drug_cost"]))

    cursor.execute("""
        INSERT INTO claim_vitamin (claim_id, vitamin_name, dosage, days, cost)
        VALUES (%s,%s,'1 tablet',3,%s)
    """, (claim_id, rec["vitamin"], rec["vit_cost"]))

    return claim_id


# ================================================================
# MAIN
# ================================================================
def main():

    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor()

    print("Generating synthetic claims...\n")

    f = open("synthetic_label_300k.csv", "w", newline="", encoding="utf-8")
    writer = csv.writer(f)
    writer.writerow(["claim_id", "fraud_label", "fraud_type", "icd10", "dept", "visit_date", "amount", "status"])

    for i in range(TOTAL_CLAIMS):
        rec = generate_claim()
        claim_id = insert(cursor, rec)

        writer.writerow([
            claim_id,
            rec["fraud_label"],
            rec["fraud_type"] if rec["fraud_type"] else "",
            rec["dx"][0],
            rec["department"],
            rec["visit_date"],
            rec["total"],
            rec["status"]
        ])

        if BATCH_COMMIT == 1:
            conn.commit()
        elif (i + 1) % BATCH_COMMIT == 0:
            conn.commit()

        if (i + 1) % 1000 == 0:
            print(f"{i + 1} rows inserted...")

    conn.commit()
    cursor.close()
    conn.close()

    print("\nDONE. Generated 300,000 realistic synthetic claim records.")


if __name__ == "__main__":
    main()
