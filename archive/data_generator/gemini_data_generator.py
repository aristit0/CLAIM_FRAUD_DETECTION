#!/usr/bin/env python3
import random
import uuid
import sys
import os
import mysql.connector
from faker import Faker
from datetime import datetime, timedelta

# ==============================================================================
# 1. SETUP & IMPORT CONFIG
# ==============================================================================
# Add project root to path to import config.py
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from config import COMPAT_RULES, COST_THRESHOLDS
    # Master Data Helpers
    ALL_DIAGNOSES = list(COMPAT_RULES.keys())
    print(f"✓ Loaded configuration. {len(ALL_DIAGNOSES)} diagnoses available.")
except ImportError:
    print("✗ CRITICAL: config.py not found. Make sure you are in the correct directory.")
    sys.exit(1)

# ==============================================================================
# 2. CONFIGURATION
# ==============================================================================
TOTAL_CLAIMS = 5000           # Adjust as needed
START_DATE = datetime(2023, 1, 1)
END_DATE = datetime(2023, 12, 31)
FRAUD_RATIO = 0.15            # 15% fraud

DB_CONFIG = {
    "host": "cdpmsi.tomodachis.org",  # Your MySQL Host
    "user": "cloudera",               # Your MySQL User
    "password": "T1ku$H1t4m",         # Your MySQL Password
    "database": "claimdb"
}

fake = Faker("id_ID")

# Extended Master Data for "Bad" choices in fraud scenarios
BAD_PROCEDURES = ["99.04", "88.38", "93.90", "96.04"] # Transfusion, CT Scan, etc.
BAD_DRUGS = ["KFA003", "KFA040", "KFA027", "KFA035"]  # Ceftriaxone, Levofloxacin, Ketorolac, Insulin
BAD_VITAMINS = ["Vitamin A 5000 IU", "Fish Oil Omega-3", "Vitamin E 400 IU"]

# ==============================================================================
# 3. POOLS (Patients & Doctors)
# ==============================================================================
class Pool:
    def __init__(self, size, role="patient"):
        self.people = []
        for _ in range(size):
            gender = random.choice(["M", "F"])
            person = {
                "id": str(uuid.uuid4())[:8],
                "nik": self._gen_nik(gender),
                "name": fake.name_male() if gender == "M" else fake.name_female(),
                "gender": gender,
                "dob": fake.date_of_birth(minimum_age=18, maximum_age=80),
                "address": fake.address().replace("\n", ", "),
                "phone": fake.phone_number(),
                # Risk profiles
                "is_abuser": random.random() < 0.05 if role == "patient" else False,
                "is_fraudster": random.random() < 0.10 if role == "doctor" else False,
                # Chronic condition for patients (consistent history)
                "chronic_dx": random.choice(["E11", "I10"]) if role == "patient" and random.random() < 0.3 else None
            }
            self.people.append(person)

    def _gen_nik(self, gender):
        prov = random.randint(11, 99)
        dob_code = f"{random.randint(1,31):02d}"
        if gender == "F": dob_code = str(int(dob_code) + 40)
        return f"{prov}{random.randint(10,99)}01{dob_code}0190{random.randint(1000,9999)}"

    def get_random(self):
        return random.choice(self.people)

# Initialize Pools
print("Initializing pools...")
patients = Pool(1000, "patient")  # 1000 unique patients
doctors = Pool(50, "doctor")      # 50 unique doctors

# ==============================================================================
# 4. GENERATION LOGIC
# ==============================================================================
def generate_single_claim(visit_date):
    # 1. Actors
    patient = patients.get_random()
    doctor = doctors.get_random()
    
    # 2. Diagnosis
    # If patient has chronic condition, 70% chance they visit for that
    if patient["chronic_dx"] and random.random() < 0.7:
        dx_code = patient["chronic_dx"]
    else:
        dx_code = random.choice(ALL_DIAGNOSES)
    
    rule = COMPAT_RULES[dx_code]
    
    # 3. Fraud Logic
    is_fraud = False
    fraud_type = None
    
    # Base chance + modifier from actor profiles
    chance = FRAUD_RATIO
    if patient["is_abuser"]: chance += 0.2
    if doctor["is_fraudster"]: chance += 0.3
    
    if random.random() < chance:
        is_fraud = True
        fraud_type = random.choice(["mismatch", "upcoding", "phantom"])

    # 4. Select Items (Procedure, Drug, Vitamin)
    # Default: Pick valid items
    proc_code = random.choice(rule["procedures"]) if rule["procedures"] else None
    drug_code = random.choice(rule["drugs"]) if rule["drugs"] else None
    vit_name = random.choice(rule["vitamins"]) if rule["vitamins"] else None
    
    # Costs
    c_proc = random.randint(COST_THRESHOLDS["procedure"]["low"], COST_THRESHOLDS["procedure"]["medium"])
    c_drug = random.randint(COST_THRESHOLDS["drug"]["low"], COST_THRESHOLDS["drug"]["medium"])
    c_vit = random.randint(COST_THRESHOLDS["vitamin"]["low"], COST_THRESHOLDS["vitamin"]["medium"])

    # Apply Fraud
    if is_fraud:
        if fraud_type == "mismatch":
            # Swap with incompatible items
            if random.random() < 0.5: drug_code = random.choice(BAD_DRUGS)
            if random.random() < 0.5: proc_code = random.choice(BAD_PROCEDURES)
        
        elif fraud_type == "upcoding":
            # Inflate costs
            c_proc *= random.randint(3, 5)
            c_drug *= random.randint(2, 4)
            
        elif fraud_type == "phantom":
            # Add expensive, unrelated procedure
            proc_code = "96.04" # Nebulizer (example)
            c_proc = 2_000_000 # Very expensive
    
    total = c_proc + c_drug + c_vit
    
    # Status
    # Simple logic: Fraud usually declined, but some slip through
    if is_fraud:
        status = "declined" if random.random() < 0.7 else "approved"
    else:
        status = "approved"

    return {
        "patient": patient,
        "doctor": doctor,
        "visit_date": visit_date,
        "dx_code": dx_code,
        "proc": (proc_code, c_proc),
        "drug": (drug_code, c_drug),
        "vit": (vit_name, c_vit),
        "total": total,
        "status": status
    }

# ==============================================================================
# 5. DATABASE INSERTION
# ==============================================================================
def insert_claim_to_db(cursor, data):
    p = data["patient"]
    d = data["doctor"]
    
    # 1. Header
    sql_header = """
        INSERT INTO claim_header (
            patient_nik, patient_name, patient_gender, patient_dob,
            patient_address, patient_phone,
            visit_date, visit_type, doctor_name, department,
            total_procedure_cost, total_drug_cost, total_vitamin_cost,
            total_claim_amount, status
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    dept = "Poli Umum" if data["dx_code"] in ["J06", "A09"] else "Poli Penyakit Dalam"
    
    vals_header = (
        p["nik"], p["name"], p["gender"], p["dob"],
        p["address"], p["phone"],
        data["visit_date"], "rawat jalan", d["name"], dept,
        data["proc"][1], data["drug"][1], data["vit"][1],
        data["total"], data["status"]
    )
    cursor.execute(sql_header, vals_header)
    claim_id = cursor.lastrowid
    
    # 2. Diagnosis
    sql_diag = "INSERT INTO claim_diagnosis (claim_id, icd10_code, icd10_description, is_primary) VALUES (%s, %s, %s, 1)"
    cursor.execute(sql_diag, (claim_id, data["dx_code"], "Description Placeholder"))
    
    # 3. Procedure
    if data["proc"][0]:
        sql_proc = "INSERT INTO claim_procedure (claim_id, icd9_code, icd9_description, quantity, procedure_date, cost) VALUES (%s, %s, %s, 1, %s, %s)"
        cursor.execute(sql_proc, (claim_id, data["proc"][0], "Proc Desc", data["visit_date"], data["proc"][1]))
        
    # 4. Drug
    if data["drug"][0]:
        sql_drug = "INSERT INTO claim_drug (claim_id, drug_code, drug_name, cost) VALUES (%s, %s, %s, %s)"
        cursor.execute(sql_drug, (claim_id, data["drug"][0], "Drug Name", data["drug"][1]))
        
    # 5. Vitamin
    if data["vit"][0]:
        sql_vit = "INSERT INTO claim_vitamin (claim_id, vitamin_name, cost) VALUES (%s, %s, %s)"
        cursor.execute(sql_vit, (claim_id, data["vit"][0], data["vit"][1]))
        
    return claim_id

# ==============================================================================
# 6. MAIN EXECUTION
# ==============================================================================
def main():
    print(f"Connecting to MySQL ({DB_CONFIG['host']})...")
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()
        print("✓ Connected.")
    except Exception as e:
        print(f"✗ Connection Failed: {e}")
        return

    print(f"Generating {TOTAL_CLAIMS} claims...")
    
    # Generate timestamps
    timestamps = [
        START_DATE + timedelta(seconds=random.randint(0, int((END_DATE - START_DATE).total_seconds())))
        for _ in range(TOTAL_CLAIMS)
    ]
    timestamps.sort() # Important for history simulation
    
    count = 0
    for ts in timestamps:
        claim_data = generate_single_claim(ts)
        insert_claim_to_db(cursor, claim_data)
        
        count += 1
        if count % 100 == 0:
            conn.commit()
            print(f"  Inserted {count}/{TOTAL_CLAIMS} claims...", end="\r")
            
    conn.commit()
    cursor.close()
    conn.close()
    print(f"\n✓ DONE! {count} claims inserted into MySQL.")

if __name__ == "__main__":
    main()