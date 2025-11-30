#!/usr/bin/env python3
import random
from faker import Faker
from datetime import datetime
import mysql.connector
import pandas as pd

# ==============================
# CONFIG
# ==============================
TOTAL_CLAIMS = 3000          # jumlah klaim yang mau digenerate
FRAUD_RATIO  = 0.25          # 25% klaim fraud (untuk training label OFFLINE)

DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "Admin123",
    "database": "claimdb",
}

fake = Faker("id_ID")


# ==============================
# HELPER: Generate NIK
# ==============================
def generate_nik(gender: str):
    prov = random.randint(11, 99)
    kab = random.randint(1, 99)
    kec = random.randint(1, 99)

    dob = fake.date_of_birth(minimum_age=18, maximum_age=65)

    dd = dob.day
    mm = dob.month
    yy = dob.year % 100

    if gender == "F":
        dd = dd + 40

    urut = random.randint(1, 9999)

    nik = f"{prov:02d}{kab:02d}{kec:02d}{dd:02d}{mm:02d}{yy:02d}{urut:04d}"
    return nik, dob


# ==============================
# DICTIONARIES ICD / PROCEDURE / DRUG / VITAMIN
# ==============================
ICD10_LIST = [
    ("A09", "Diare dan gastroenteritis"),
    ("J06", "Common cold"),
    ("I10", "Hipertensi"),
    ("E11", "Diabetes mellitus tipe 2"),
    ("J45", "Asma"),
    ("K29", "Gastritis"),
]

ICD9_LIST = [
    ("96.70", "Injeksi obat"),
    ("99.04", "Transfusi darah"),
    ("03.31", "Pemeriksaan darah"),
    ("45.13", "Endoskopi"),
    ("93.90", "Electrotherapy"),
]

DRUG_LIST = [
    ("KFA001", "Paracetamol 500 mg"),
    ("KFA002", "Amoxicillin 500 mg"),
    ("KFA003", "Ceftriaxone injeksi"),
    ("KFA004", "Omeprazole 20 mg"),
    ("KFA005", "ORS / Oralit"),
]

VITAMIN_LIST = [
    "Vitamin C 500 mg",
    "Vitamin B Complex",
    "Vitamin D 1000 IU",
    "Vitamin E 400 IU",
]


# ==============================
# GENERATE SATU CLAIM (NORMAL/FRAUD)
# ==============================
def generate_claim():
    gender = random.choice(["M", "F"])
    nik, dob = generate_nik(gender)
    name = fake.name_male() if gender == "M" else fake.name_female()

    address = fake.address().replace("\n", ", ")
    phone = fake.phone_number()

    visit_date = fake.date_between(start_date='-2y', end_date='today')
    doctor = fake.name()
    department = random.choice(["Poli Umum", "Poli Anak", "Poli Saraf", "IGD", "Poli Penyakit Dalam"])

    # Diagnosis
    dx_primary = random.choice(ICD10_LIST)
    dx_secondary = random.choice(ICD10_LIST)

    # Procedure / Drug / Vitamin yang secara default "masuk akal"
    procedure = random.choice(ICD9_LIST)
    drug = random.choice(DRUG_LIST)
    vitamin = random.choice(VITAMIN_LIST)

    # Biaya dasar
    total_proc_cost = random.randint(50_000, 300_000)
    total_drug_cost = random.randint(20_000, 150_000)
    total_vit_cost = random.randint(10_000, 80_000)

    # Flag fraud offline (untuk training)
    is_fraud = random.random() < FRAUD_RATIO

    fraud_type = None
    if is_fraud:
        # pilih tipe fraud
        fraud_type = random.choice(["wrong_vitamin", "wrong_drug", "wrong_proc", "over_cost"])

        if fraud_type == "wrong_vitamin":
            # vitamin yang tidak relevan (random dari list yang tidak match)
            vitamin = "Vitamin A 5000 IU"

        elif fraud_type == "wrong_drug":
            # obat berat / injeksi padahal diagnosa ringan
            drug = ("KFA003", "Ceftriaxone injeksi")
            total_drug_cost += random.randint(150_000, 300_000)

        elif fraud_type == "wrong_proc":
            # tindakan transfusi yang tidak relevan
            procedure = ("99.04", "Transfusi darah")
            total_proc_cost += random.randint(200_000, 400_000)

        elif fraud_type == "over_cost":
            # biaya tindakan di-markup sangat besar
            multiplier = random.uniform(4, 10)
            total_proc_cost = int(total_proc_cost * multiplier)

    total_claim = total_proc_cost + total_drug_cost + total_vit_cost

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
        "dx_primary": dx_primary,
        "dx_secondary": dx_secondary,
        "procedure": procedure,
        "drug": drug,
        "vitamin": vitamin,
        "proc_cost": total_proc_cost,
        "drug_cost": total_drug_cost,
        "vit_cost": total_vit_cost,
        "total_claim": total_claim,
        "label_fraud": 1 if is_fraud else 0,
        "fraud_type": fraud_type,
    }


# ==============================
# INSERT KE MYSQL (TANPA fraud_label)
# ==============================
def insert_claim(cursor, conn, record, label_records):
    # 1) Header
    cursor.execute("""
        INSERT INTO claim_header (
            patient_nik, patient_name, patient_gender, patient_dob,
            patient_address, patient_phone,
            visit_date, visit_type, doctor_name, department,
            total_procedure_cost, total_drug_cost, total_vitamin_cost, total_claim_amount
        ) VALUES (%s,%s,%s,%s,%s,%s,%s,'rawat jalan',%s,%s,%s,%s,%s,%s)
    """, (
        record["nik"], record["name"], record["gender"], record["dob"],
        record["address"], record["phone"],
        record["visit_date"], record["doctor"], record["department"],
        record["proc_cost"], record["drug_cost"], record["vit_cost"], record["total_claim"],
    ))
    conn.commit()

    claim_id = cursor.lastrowid

    # 2) Diagnosis primary
    cursor.execute("""
        INSERT INTO claim_diagnosis (claim_id, icd10_code, icd10_description, is_primary)
        VALUES (%s,%s,%s,1)
    """, (claim_id, record["dx_primary"][0], record["dx_primary"][1]))

    # 3) Diagnosis secondary
    cursor.execute("""
        INSERT INTO claim_diagnosis (claim_id, icd10_code, icd10_description, is_primary)
        VALUES (%s,%s,%s,0)
    """, (claim_id, record["dx_secondary"][0], record["dx_secondary"][1]))

    # 4) Procedure
    cursor.execute("""
        INSERT INTO claim_procedure (claim_id, icd9_code, icd9_description, quantity, procedure_date)
        VALUES (%s,%s,%s,1,%s)
    """, (claim_id, record["procedure"][0], record["procedure"][1], record["visit_date"]))

    # 5) Drug
    cursor.execute("""
        INSERT INTO claim_drug (claim_id, drug_code, drug_name, dosage, frequency, route, days, cost)
        VALUES (%s,%s,%s,'1 tablet','2x sehari','oral',3,%s)
    """, (claim_id, record["drug"][0], record["drug"][1], record["drug_cost"]))

    # 6) Vitamin
    cursor.execute("""
        INSERT INTO claim_vitamin (claim_id, vitamin_name, dosage, days, cost)
        VALUES (%s,%s,'1 tablet',3,%s)
    """, (claim_id, record["vitamin"], record["vit_cost"]))

    conn.commit()

    # 7) SIMPAN LABEL OFFLINE (untuk training, join via claim_id)
    label_records.append({
        "claim_id": claim_id,
        "fraud_label": record["label_fraud"],
        "fraud_type": record["fraud_type"],
        "icd10_primary": record["dx_primary"][0],
        "department": record["department"],
        "visit_date": record["visit_date"].strftime("%Y-%m-%d"),
        "total_claim_amount": record["total_claim"],
    })


# ==============================
# MAIN
# ==============================
def main():
    print(f"Generating {TOTAL_CLAIMS} synthetic claims...")

    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor()

    label_records = []

    for i in range(TOTAL_CLAIMS):
        rec = generate_claim()
        insert_claim(cursor, conn, rec, label_records)

        if (i + 1) % 100 == 0:
            print(f"Inserted {i + 1} claims...")

    cursor.close()
    conn.close()

    print("All claims inserted into MySQL.")

    # Simpan label OFFLINE untuk training
    df_labels = pd.DataFrame(label_records)
    df_labels.to_csv("synthetic_claim_labels.csv", index=False)
    print("Saved training labels to synthetic_claim_labels.csv")


if __name__ == "__main__":
    main()