import random
from faker import Faker
from datetime import datetime, timedelta
import mysql.connector

fake = Faker("id_ID")

# -------------------------------
# Generate NIK valid Indonesia
# -------------------------------
def generate_nik(gender):
    prov = random.randint(11, 99)   # kode provinsi
    kab = random.randint(1, 99)     # kode kab/kota
    kec = random.randint(1, 99)     # kode kecamatan

    dob = fake.date_of_birth(minimum_age=18, maximum_age=65)

    dd = dob.day
    mm = dob.month
    yy = dob.year % 100

    # perempuan tanggal lahir + 40
    if gender == "F":
        dd = dd + 40

    urut = random.randint(1, 9999)

    nik = f"{prov:02d}{kab:02d}{kec:02d}{dd:02d}{mm:02d}{yy:02d}{urut:04d}"
    return nik, dob


# --------------------------------------------------------
# ICD-10, ICD-9, Drugs, Vitamin Dictionaries
# --------------------------------------------------------

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
    ("93.90", "Electrotherapy")
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
    "Vitamin E 400 IU"
]

# --------------------------------------------------------
# MySQL Connection
# --------------------------------------------------------
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="Admin123",
    database="claimdb"
)
cursor = conn.cursor()


# --------------------------------------------------------
# Generate One Claim Record
# --------------------------------------------------------
def generate_claim():
    gender = random.choice(["M", "F"])
    nik, dob = generate_nik(gender)
    name = fake.name_male() if gender == "M" else fake.name_female()

    address = fake.address().replace("\n", ", ")
    phone = fake.phone_number()

    visit_date = fake.date_between(start_date='-2y', end_date='today')
    doctor = fake.name()
    department = random.choice(["Poli Umum", "Poli Anak", "Poli Saraf", "IGD"])

    # Pick diagnosis
    dx_primary = random.choice(ICD10_LIST)
    dx_secondary = random.choice(ICD10_LIST)

    # Pick procedure
    px = random.choice(ICD9_LIST)

    # Pick drug(s)
    drug = random.choice(DRUG_LIST)

    # Vitamin
    vitamin = random.choice(VITAMIN_LIST)

    # Fraud or not?
    is_fraud = random.random() < 0.25   # 25% fraud case

    total_proc_cost = random.randint(50000, 300000)
    total_drug_cost = random.randint(20000, 150000)
    total_vit_cost = random.randint(10000, 80000)

    if is_fraud:
        # Inject mismatched treatment
        px = ("99.04", "Transfusi darah")   # not relevant for many dx
        drug = ("KFA003", "Ceftriaxone injeksi")  # "fraud" antibiotic
        total_drug_cost += 200000
        total_proc_cost += 300000
        dx_secondary = ("A09", "Diare")

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
        "procedure": px,
        "drug": drug,
        "vitamin": vitamin,
        "proc_cost": total_proc_cost,
        "drug_cost": total_drug_cost,
        "vit_cost": total_vit_cost,
        "total_claim": total_claim
    }


# --------------------------------------------------------
# Insert into MySQL
# --------------------------------------------------------
def insert_claim(record):
    # header
    cursor.execute("""
        INSERT INTO claim_header (
            patient_nik, patient_name, patient_gender, patient_dob,
            patient_address, patient_phone,
            visit_date, visit_type, doctor_name, department,
            total_procedure_cost, total_drug_cost, total_vitamin_cost, total_claim_amount
        ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
    """, (
        record["nik"], record["name"], record["gender"], record["dob"],
        record["address"], record["phone"],
        record["visit_date"], "rawat jalan", record["doctor"], record["department"],
        record["proc_cost"], record["drug_cost"], record["vit_cost"], record["total_claim"]
    ))
    conn.commit()

    claim_id = cursor.lastrowid

    # diagnosis primary
    cursor.execute("""
        INSERT INTO claim_diagnosis (claim_id, icd10_code, icd10_description, is_primary)
        VALUES (%s,%s,%s,1)
    """, (claim_id, record["dx_primary"][0], record["dx_primary"][1]))

    # diagnosis secondary
    cursor.execute("""
        INSERT INTO claim_diagnosis (claim_id, icd10_code, icd10_description, is_primary)
        VALUES (%s,%s,%s,0)
    """, (claim_id, record["dx_secondary"][0], record["dx_secondary"][1]))

    # procedure
    cursor.execute("""
        INSERT INTO claim_procedure (claim_id, icd9_code, icd9_description, quantity, procedure_date)
        VALUES (%s,%s,%s,1,%s)
    """, (claim_id, record["procedure"][0], record["procedure"][1], record["visit_date"]))

    # drug
    cursor.execute("""
        INSERT INTO claim_drug (claim_id, drug_code, drug_name, dosage, frequency, route, days, cost)
        VALUES (%s,%s,%s,'1 tablet','2x sehari','oral',3,%s)
    """, (claim_id, record["drug"][0], record["drug"][1], record["drug_cost"]))

    # vitamin
    cursor.execute("""
        INSERT INTO claim_vitamin (claim_id, vitamin_name, dosage, days, cost)
        VALUES (%s,%s,'1 tablet',3,%s)
    """, (claim_id, record["vitamin"], record["vit_cost"]))

    conn.commit()


# --------------------------------------------------------
# MAIN
# --------------------------------------------------------
print("Generating 200 synthetic claims...")
for _ in range(200):
    record = generate_claim()
    insert_claim(record)

print("DONE.")
cursor.close()
conn.close()