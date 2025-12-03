#!/usr/bin/env python3
import random
import mysql.connector
from faker import Faker
from datetime import datetime
import csv

fake = Faker("id_ID")

# =====================================================================
# CONFIG
# =====================================================================
TOTAL = 100_000
FRAUD_RATIO = 0.35
BATCH = 500

DB = {
    "host": "localhost",
    "user": "root",
    "password": "Admin123",
    "database": "claimdb",
}

# =====================================================================
# NIK GENERATOR
# =====================================================================
def gen_nik(gender):
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
    return f"{prov:02d}{kab:02d}{kec:02d}{dd:02d}{mm:02d}{yy:02d}{urut:04d}", dob


# =====================================================================
# LOAD LOOKUP RULES
# =====================================================================
def load_lookup(cursor):
    cursor.execute("SELECT code, description FROM master_icd10")
    rows = cursor.fetchall()
    ICD10 = {code: desc for code, desc in rows}  # FIXED → dict

    cursor.execute("SELECT icd10_code, icd9_code, is_mandatory, severity_level FROM clinical_rule_dx_procedure")
    PROC = {}
    for dx, icd9, mandatory, sev in cursor.fetchall():
        PROC.setdefault(dx, []).append((icd9, mandatory, sev))

    cursor.execute("SELECT icd10_code, drug_code, is_mandatory, severity_level FROM clinical_rule_dx_drug")
    DRUG = {}
    for dx, d, mandatory, sev in cursor.fetchall():
        DRUG.setdefault(dx, []).append((d, mandatory, sev))

    cursor.execute("SELECT icd10_code, vitamin_name, is_mandatory, severity_level FROM clinical_rule_dx_vitamin")
    VIT = {}
    for dx, vit, mandatory, sev in cursor.fetchall():
        VIT.setdefault(dx, []).append((vit, mandatory, sev))

    return ICD10, PROC, DRUG, VIT


# =====================================================================
# FRAUD ENGINE
# =====================================================================
def inject_fraud(dx, proc, drug, vit, cost_p, cost_d, cost_v, DRUG, VIT):
    fraud_type = None
    r = random.random()

    allowed_drugs = [x[0] for x in DRUG.get(dx, [])]
    allowed_vits = [x[0] for x in VIT.get(dx, [])]

    all_drugs = list({x[0] for k in DRUG for x in DRUG[k]})
    all_vits   = list({x[0] for k in VIT for x in VIT[k]})

    if r < 0.20:  # wrong drug
        wrong = [d for d in all_drugs if d not in allowed_drugs]
        if wrong:
            drug = random.choice(wrong)
        fraud_type = "wrong_drug"

    elif r < 0.40:  # wrong procedure
        proc = "96.04"
        fraud_type = "wrong_procedure"

    elif r < 0.60:  # wrong vitamin
        wrong_v = [v for v in all_vits if v not in allowed_vits]
        if wrong_v:
            vit = random.choice(wrong_v)
        fraud_type = "wrong_vitamin"

    elif r < 0.80:  # inflated cost
        cost_p *= random.randint(3, 7)
        cost_d *= random.randint(2, 4)
        cost_v *= random.randint(2, 4)
        fraud_type = "inflated_cost"

    else:  # multi fraud
        proc = "96.04"
        drug = random.choice(all_drugs)
        vit  = random.choice(all_vits)
        cost_p *= random.randint(4, 8)
        cost_d *= random.randint(3, 6)
        cost_v *= random.randint(2, 5)
        fraud_type = "multi"

    return proc, drug, vit, cost_p, cost_d, cost_v, fraud_type


# =====================================================================
# GENERATE CLAIM
# =====================================================================
def generate_claim(ICD10, PROC, DRUG, VIT):
    dx = random.choice(list(ICD10.keys()))

    proc_list = PROC.get(dx, [])
    drug_list = DRUG.get(dx, [])
    vit_list  = VIT.get(dx, [])

    proc = random.choice(proc_list)[0] if proc_list else None
    drug = random.choice(drug_list)[0] if drug_list else None
    vit  = random.choice(vit_list)[0] if vit_list else None

    cost_p = random.randint(50_000, 150_000)
    cost_d = random.randint(10_000, 80_000)
    cost_v = random.randint(5_000, 30_000)

    fraud_type = None
    fraud_label = 0

    if random.random() < FRAUD_RATIO:
        proc, drug, vit, cost_p, cost_d, cost_v, fraud_type = inject_fraud(
            dx, proc, drug, vit, cost_p, cost_d, cost_v, DRUG, VIT
        )
        fraud_label = 1

    total = cost_p + cost_d + cost_v

    gender = random.choice(["M","F"])
    nik, dob = gen_nik(gender)

    return {
        "dx": dx,
        "proc": proc,
        "drug": drug,
        "vit": vit,
        "cost_p": cost_p,
        "cost_d": cost_d,
        "cost_v": cost_v,
        "total": total,
        "fraud_type": fraud_type,
        "fraud_label": fraud_label,

        "nik": nik,
        "name": fake.name(),
        "gender": gender,
        "dob": dob,
        "visit_date": fake.date_between(start_date='-3y', end_date='today'),
        "doctor": fake.name(),
        "department": random.choice(["Poli Umum","Poli Anak","Poli Dalam","IGD"]),
        "status": "pending"
    }


# =====================================================================
# INSERT
# =====================================================================
def insert_claim(cursor, rec):
    cursor.execute("""
        INSERT INTO claim_header (
            patient_nik, patient_name, patient_gender, patient_dob,
            visit_date, visit_type, doctor_name, department,
            total_procedure_cost, total_drug_cost, total_vitamin_cost,
            total_claim_amount, status
        ) VALUES (%s,%s,%s,%s,%s,'rawat jalan',%s,%s,%s,%s,%s,%s,%s)
    """, (
        rec["nik"], rec["name"], rec["gender"], rec["dob"],
        rec["visit_date"], rec["doctor"], rec["department"],
        rec["cost_p"], rec["cost_d"], rec["cost_v"],
        rec["total"], rec["status"]
    ))

    claim_id = cursor.lastrowid

    cursor.execute("""
        INSERT INTO claim_diagnosis
        (claim_id, icd10_code, icd10_description, is_primary)
        VALUES (%s,%s,(SELECT description FROM master_icd10 WHERE code=%s),1)
    """, (claim_id, rec["dx"], rec["dx"]))

    cursor.execute("""
        INSERT INTO claim_procedure
        (claim_id, icd9_code, icd9_description, quantity, procedure_date)
        VALUES (%s,%s,(SELECT description FROM master_icd9 WHERE code=%s),1,%s)
    """, (claim_id, rec["proc"], rec["proc"], rec["visit_date"]))

    cursor.execute("""
        INSERT INTO claim_drug
        (claim_id, drug_code, drug_name, dosage, frequency, route, days, cost)
        VALUES (%s,%s,(SELECT name FROM master_drug WHERE code=%s),
                '1 tablet','2x sehari','oral',3,%s)
    """, (claim_id, rec["drug"], rec["drug"], rec["cost_d"]))

    cursor.execute("""
        INSERT INTO claim_vitamin
        (claim_id, vitamin_name, dosage, days, cost)
        VALUES (%s,%s,'1 tablet',3,%s)
    """, (claim_id, rec["vit"], rec["cost_v"]))

    return claim_id


# =====================================================================
# MAIN
# =====================================================================
def main():
    conn = mysql.connector.connect(**DB)
    cursor = conn.cursor()

    ICD10, PROC, DRUG, VIT = load_lookup(cursor)

    print("Generating 100,000 realistic + fraud claims...\n")

    f = open("claim_label_100k.csv","w",newline="",encoding="utf-8")
    writer = csv.writer(f)
    writer.writerow(["claim_id","fraud_label","fraud_type"])

    count = 0

    for _ in range(TOTAL):
        rec = generate_claim(ICD10, PROC, DRUG, VIT)
        claim_id = insert_claim(cursor, rec)
        writer.writerow([claim_id, rec["fraud_label"], rec["fraud_type"]])

        count += 1
        if count % BATCH == 0:
            conn.commit()
            print(f"{count} rows inserted...")

    conn.commit()
    cursor.close()
    conn.close()

    print("DONE: 100,000 claims inserted successfully.")


if __name__ == "__main__":
    main()