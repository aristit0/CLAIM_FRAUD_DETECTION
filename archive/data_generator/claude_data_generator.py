#!/usr/bin/env python3
import random
import csv
import sys
import os
from faker import Faker
from datetime import datetime
import mysql.connector

# ================================================================
# INLINE CONFIG (NO EXTERNAL IMPORTS)
# ================================================================
print("Initializing fraud detection data generator...")

# CLINICAL COMPATIBILITY RULES
COMPAT_RULES = {
    "A09": {
        "procedures": ["03.31", "03.91", "99.15"],
        "drugs": ["KFA005", "KFA013", "KFA024", "KFA025", "KFA038"],
        "vitamins": ["Vitamin D 1000 IU", "Zinc 20 mg", "Probiotic Complex"]
    },
    "K29": {
        "procedures": ["45.13", "03.31", "89.02"],
        "drugs": ["KFA004", "KFA012", "KFA023", "KFA034", "KFA037"],
        "vitamins": ["Vitamin E 400 IU", "Vitamin B Complex"]
    },
    "K52": {
        "procedures": ["03.31", "03.92"],
        "drugs": ["KFA004", "KFA024", "KFA038"],
        "vitamins": ["Probiotic Complex", "Zinc 20 mg"]
    },
    "K21": {
        "procedures": ["45.13", "89.02"],
        "drugs": ["KFA004", "KFA034", "KFA023"],
        "vitamins": ["Vitamin E 200 IU"]
    },
    "J06": {
        "procedures": ["96.70", "89.02", "87.03"],
        "drugs": ["KFA001", "KFA002", "KFA009", "KFA031"],
        "vitamins": ["Vitamin C 500 mg", "Vitamin C 1000 mg", "Zinc 20 mg"]
    },
    "J06.9": {
        "procedures": ["96.70", "89.02"],
        "drugs": ["KFA001", "KFA002", "KFA009"],
        "vitamins": ["Vitamin C 500 mg", "Zinc 20 mg"]
    },
    "J02": {
        "procedures": ["89.02", "34.01"],
        "drugs": ["KFA001", "KFA002", "KFA014"],
        "vitamins": ["Vitamin C 1000 mg"]
    },
    "J20": {
        "procedures": ["87.03", "89.02", "96.04"],
        "drugs": ["KFA002", "KFA022", "KFA026"],
        "vitamins": ["Vitamin C 1000 mg", "Vitamin B Complex"]
    },
    "J45": {
        "procedures": ["96.04", "93.05", "87.03"],
        "drugs": ["KFA010", "KFA011", "KFA021"],
        "vitamins": ["Vitamin D 1000 IU", "Vitamin C 500 mg"]
    },
    "J18": {
        "procedures": ["87.03", "03.31", "99.15"],
        "drugs": ["KFA003", "KFA014", "KFA030", "KFA040"],
        "vitamins": ["Vitamin C 1000 mg", "Vitamin D3 2000 IU"]
    },
    "I10": {
        "procedures": ["03.31", "89.14", "89.02"],
        "drugs": ["KFA007", "KFA019"],
        "vitamins": ["Vitamin D 1000 IU", "Magnesium 250 mg", "Vitamin B Complex"]
    },
    "E11": {
        "procedures": ["03.31", "90.59", "90.59A"],
        "drugs": ["KFA006", "KFA035", "KFA036"],
        "vitamins": ["Vitamin B Complex", "Vitamin D 1000 IU", "Magnesium 250 mg"]
    },
    "E16": {
        "procedures": ["90.59", "03.31"],
        "drugs": ["KFA035", "KFA036"],
        "vitamins": ["Vitamin B Complex"]
    },
    "R51": {
        "procedures": ["89.02"],
        "drugs": ["KFA001", "KFA008", "KFA033"],
        "vitamins": ["Vitamin B Complex", "Magnesium 250 mg"]
    },
    "G43": {
        "procedures": ["89.02", "88.53"],
        "drugs": ["KFA001", "KFA008", "KFA033"],
        "vitamins": ["Magnesium 250 mg", "Vitamin B Complex Forte"]
    },
    "M54.5": {
        "procedures": ["89.0", "93.27", "93.94"],
        "drugs": ["KFA008", "KFA033", "KFA027"],
        "vitamins": ["Vitamin D 1000 IU", "Calcium 500 mg"]
    },
    "N39": {
        "procedures": ["03.91", "03.31"],
        "drugs": ["KFA030", "KFA040"],
        "vitamins": ["Vitamin C 1000 mg"]
    },
    "L03": {
        "procedures": ["89.02", "96.70"],
        "drugs": ["KFA003", "KFA014", "KFA039"],
        "vitamins": ["Vitamin C 1000 mg"]
    },
    "T78.4": {
        "procedures": ["89.02"],
        "drugs": ["KFA009", "KFA031", "KFA028"],
        "vitamins": ["Vitamin C 500 mg"]
    },
    "H10": {
        "procedures": ["89.02"],
        "drugs": ["KFA009", "KFA031"],
        "vitamins": ["Vitamin A 5000 IU"]
    }
}

# MASTER DATA
MASTER_ICD9 = [
    "03.31", "03.91", "03.92", "04.41", "04.43", "34.01", "34.02",
    "34.03", "45.13", "45.16", "45.23", "87.03", "88.53", "88.72",
    "89.0", "89.02", "89.14", "90.59", "90.59A", "93.05", "93.27",
    "93.90", "93.94", "96.04", "96.70", "96.71", "99.04", "99.1",
    "99.15", "99.21", "99.29", "99.89"
]

MASTER_DRUGS = [
    "KFA001", "KFA002", "KFA003", "KFA004", "KFA005", "KFA006",
    "KFA007", "KFA008", "KFA009", "KFA010", "KFA011", "KFA012",
    "KFA013", "KFA014", "KFA021", "KFA022", "KFA023", "KFA024",
    "KFA025", "KFA026", "KFA027", "KFA028", "KFA030", "KFA031",
    "KFA033", "KFA034", "KFA035", "KFA036", "KFA037", "KFA038",
    "KFA039", "KFA040"
]

# ================================================================
# CONFIG
# ================================================================
TOTAL_CLAIMS = 300_000
BATCH_COMMIT = 5000  # Commit every 5000 rows for performance
FRAUD_RATIO = 0.35

DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "Admin123",
    "database": "claimdb",
}

fake = Faker("id_ID")

print(f"Configuration:")
print(f"  - Target claims: {TOTAL_CLAIMS:,}")
print(f"  - Fraud ratio: {FRAUD_RATIO:.0%}")
print(f"  - Batch commit: {BATCH_COMMIT:,}")
print(f"  - Diagnoses with rules: {len(COMPAT_RULES)}")

# ================================================================
# DESCRIPTIONS
# ================================================================
ICD10_DESCRIPTIONS = {
    "A09": "Diare dan gastroenteritis",
    "J06": "Common cold",
    "J06.9": "Common cold - unspecified",
    "I10": "Hipertensi",
    "E11": "Diabetes tipe 2",
    "J45": "Asma",
    "K29": "Gastritis",
    "J02": "Faringitis akut",
    "J20": "Bronkitis akut",
    "J18": "Pneumonia, unspecified",
    "K21": "GERD / reflux esofagitis",
    "K52": "Gastroenteritis noninfeksi",
    "N39": "Infeksi saluran kemih",
    "R51": "Sakit kepala",
    "G43": "Migrain",
    "M54.5": "Nyeri punggung bawah",
    "L03": "Selulitis",
    "T78.4": "Alergi, unspecified",
    "H10": "Konjungtivitis",
    "E16": "Hipoglikemia",
}

ICD9_DESCRIPTIONS = {
    "03.31": "Pemeriksaan darah",
    "03.91": "Urinalisis",
    "03.92": "Tes feses",
    "45.13": "Endoskopi",
    "96.70": "Injeksi obat",
    "87.03": "X-ray dada",
    "89.02": "Pemeriksaan fisik umum",
    "89.14": "EKG",
    "90.59": "Pemeriksaan glukosa darah",
    "90.59A": "HbA1C test",
    "96.04": "Nebulizer treatment",
    "93.05": "Terapi inhalasi",
    "99.15": "IV drip",
    "93.27": "Terapi ultrasound",
    "93.94": "Terapi panas",
    "89.0": "Physical therapy",
    "34.01": "Konsultasi THT",
    "88.53": "CT Scan kepala",
    "99.89": "Prosedur lainnya",
}

DRUG_DESCRIPTIONS = {
    "KFA001": "Paracetamol 500 mg",
    "KFA002": "Amoxicillin 500 mg",
    "KFA003": "Ceftriaxone injeksi",
    "KFA004": "Omeprazole 20 mg",
    "KFA005": "ORS / Oralit",
    "KFA006": "Metformin 500 mg",
    "KFA007": "Amlodipine 10 mg",
    "KFA008": "Ibuprofen 400 mg",
    "KFA009": "Cetirizine 10 mg",
    "KFA010": "Salbutamol tablet 2 mg",
    "KFA011": "Prednisone 5 mg",
    "KFA012": "Ranitidine 150 mg",
    "KFA013": "ORS Zinc Combination",
    "KFA014": "Azithromycin 500 mg",
    "KFA021": "Salbutamol inhaler",
    "KFA022": "Bromhexine 8 mg",
    "KFA023": "Antacid syrup",
    "KFA024": "Loperamide 2 mg",
    "KFA025": "ORS pediatric sachet",
    "KFA026": "Ambroxol 30 mg",
    "KFA027": "Ketorolac injeksi 30 mg",
    "KFA028": "Dexamethasone injeksi 5 mg",
    "KFA030": "Ciprofloxacin 500 mg",
    "KFA031": "Chlorpheniramine Maleate 4 mg",
    "KFA033": "Mefenamic Acid 500 mg",
    "KFA034": "Omeprazole injeksi 40 mg",
    "KFA035": "Insulin regular injeksi",
    "KFA036": "Insulin glargine",
    "KFA037": "Magnesium Hydroxide",
    "KFA038": "ORS + Probiotic sachet",
    "KFA039": "Amoxicillin-Clavulanic Acid 625 mg",
    "KFA040": "Levofloxacin 500 mg",
}

# ================================================================
# HELPERS
# ================================================================
def generate_nik(gender: str):
    """Generate Indonesian NIK"""
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
# FRAUD INJECTION
# ================================================================
def inject_fraud(dx_code, procedure, drug, vitamin, proc_cost, drug_cost, vit_cost):
    """Inject realistic fraud patterns"""
    r = random.random()
    fraud_type = None
    is_fraud = False
    
    # 1. Procedure Mismatch (15%)
    if r < 0.15:
        wrong_procedures = [p for p in MASTER_ICD9 if p not in COMPAT_RULES.get(dx_code, {}).get("procedures", [])]
        if wrong_procedures:
            wrong_proc = random.choice(wrong_procedures)
            procedure = (wrong_proc, ICD9_DESCRIPTIONS.get(wrong_proc, "Unknown procedure"))
            fraud_type = "procedure_mismatch"
            is_fraud = True
    
    # 2. Drug Mismatch (15%)
    elif r < 0.30:
        wrong_drugs = ["KFA003", "KFA027", "KFA028", "KFA035", "KFA040"]
        wrong_drug = random.choice(wrong_drugs)
        drug = (wrong_drug, DRUG_DESCRIPTIONS.get(wrong_drug, "Unknown drug"))
        drug_cost *= random.uniform(1.5, 3.0)
        fraud_type = "drug_mismatch"
        is_fraud = True
    
    # 3. Vitamin Mismatch (12%)
    elif r < 0.42:
        wrong_vitamins = ["Vitamin C Effervescent 1000 mg", "Multivitamin Adult", "Fish Oil Omega-3"]
        vitamin = random.choice(wrong_vitamins)
        vit_cost *= random.uniform(1.2, 2.0)
        fraud_type = "vitamin_mismatch"
        is_fraud = True
    
    # 4. Upcoding (20%)
    elif r < 0.62:
        proc_cost *= random.uniform(2.0, 5.0)
        drug_cost *= random.uniform(1.5, 4.0)
        vit_cost *= random.uniform(1.3, 2.5)
        fraud_type = "upcoding"
        is_fraud = True
    
    # 5. Phantom Billing (18%)
    elif r < 0.80:
        proc_cost += random.randint(200_000, 800_000)
        drug_cost += random.randint(100_000, 400_000)
        fraud_type = "phantom_billing"
        is_fraud = True
    
    # 6. Multi-fraud (20%)
    else:
        wrong_proc = random.choice([p for p in MASTER_ICD9 if p not in COMPAT_RULES.get(dx_code, {}).get("procedures", [])])
        procedure = (wrong_proc, ICD9_DESCRIPTIONS.get(wrong_proc, "Procedure"))
        
        wrong_drug = random.choice(["KFA003", "KFA027", "KFA040"])
        drug = (wrong_drug, DRUG_DESCRIPTIONS.get(wrong_drug, "Drug"))
        
        vitamin = random.choice(["Multivitamin Adult", "Fish Oil Omega-3"])
        
        proc_cost *= random.uniform(3.0, 8.0)
        drug_cost *= random.uniform(2.0, 5.0)
        vit_cost *= random.uniform(1.5, 3.0)
        
        fraud_type = "multi_fraud"
        is_fraud = True
    
    return procedure, drug, vitamin, proc_cost, drug_cost, vit_cost, fraud_type, is_fraud

# ================================================================
# GENERATE CLAIM
# ================================================================
def generate_claim():
    """Generate single claim record"""
    gender = random.choice(["M", "F"])
    nik, dob = generate_nik(gender)
    name = fake.name_male() if gender == "M" else fake.name_female()
    address = fake.address().replace("\n", ", ")
    phone = fake.phone_number()
    doctor = fake.name()
    department = random.choice(["Poli Umum", "Poli Anak", "Poli Dalam", "IGD"])
    visit_date = fake.date_between(start_date='-2y', end_date='today')
    
    # Select diagnosis
    dx_codes = list(COMPAT_RULES.keys())
    dx_code = random.choice(dx_codes)
    dx_desc = ICD10_DESCRIPTIONS.get(dx_code, "Unknown diagnosis")
    
    # Get compatible items
    allowed = COMPAT_RULES[dx_code]
    
    # Clean claim
    procedure_code = random.choice(allowed["procedures"])
    procedure = (procedure_code, ICD9_DESCRIPTIONS.get(procedure_code, "Procedure"))
    
    drug_code = random.choice(allowed["drugs"])
    drug = (drug_code, DRUG_DESCRIPTIONS.get(drug_code, "Drug"))
    
    vitamin = random.choice(allowed["vitamins"])
    
    # Base costs
    proc_cost = random.randint(50_000, 200_000)
    drug_cost = random.randint(15_000, 120_000)
    vit_cost = random.randint(8_000, 60_000)
    
    # Inject fraud
    fraud_type = None
    is_fraud = False
    
    if random.random() < FRAUD_RATIO:
        procedure, drug, vitamin, proc_cost, drug_cost, vit_cost, fraud_type, is_fraud = inject_fraud(
            dx_code, procedure, drug, vitamin, proc_cost, drug_cost, vit_cost
        )
    
    total_claim = proc_cost + drug_cost + vit_cost
    
    # Approval logic
    approve = True
    decline_reasons = []
    
    if procedure[0] not in allowed["procedures"]:
        approve = False
        decline_reasons.append("incompatible_procedure")
    
    if drug[0] not in allowed["drugs"]:
        approve = False
        decline_reasons.append("incompatible_drug")
    
    if vitamin not in allowed["vitamins"]:
        approve = False
        decline_reasons.append("incompatible_vitamin")
    
    if total_claim > 2_000_000:
        approve = False
        decline_reasons.append("exceeds_cost_limit")
    
    if dx_code == "J06" and drug[0] == "KFA003":
        approve = False
        decline_reasons.append("inappropriate_antibiotic")
    
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
        "dx": (dx_code, dx_desc),
        "procedure": procedure,
        "drug": drug,
        "vitamin": vitamin,
        "proc_cost": proc_cost,
        "drug_cost": drug_cost,
        "vit_cost": vit_cost,
        "total": total_claim,
        "is_fraud": is_fraud,
        "fraud_type": fraud_type,
        "status": status,
        "decline_reasons": ",".join(decline_reasons) if decline_reasons else None
    }

# ================================================================
# INSERT TO DB
# ================================================================
def insert(cursor, rec):
    """Insert claim to database"""
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
    print("\n" + "=" * 70)
    print("SYNTHETIC FRAUD DETECTION DATA GENERATOR")
    print("=" * 70)
    
    # Test DB connection
    print("\nTesting database connection...")
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()
        print("✓ Database connected successfully")
    except Exception as e:
        print(f"✗ Database connection failed: {e}")
        return
    
    # Open CSV
    csv_file = open("synthetic_fraud_labels.csv", "w", newline="", encoding="utf-8")
    writer = csv.writer(csv_file)
    writer.writerow([
        "claim_id", "is_fraud", "fraud_type", "icd10", "department",
        "visit_date", "total_amount", "status", "decline_reasons"
    ])
    
    print(f"\nGenerating {TOTAL_CLAIMS:,} claims...")
    print("Progress: ", end="", flush=True)
    
    fraud_count = 0
    approved_count = 0
    
    for i in range(TOTAL_CLAIMS):
        rec = generate_claim()
        claim_id = insert(cursor, rec)
        
        if rec["is_fraud"]:
            fraud_count += 1
        if rec["status"] == "approved":
            approved_count += 1
        
        writer.writerow([
            claim_id,
            1 if rec["is_fraud"] else 0,
            rec["fraud_type"] if rec["fraud_type"] else "",
            rec["dx"][0],
            rec["department"],
            rec["visit_date"],
            rec["total"],
            rec["status"],
            rec["decline_reasons"] if rec["decline_reasons"] else ""
        ])
        
        # Commit in batches
        if (i + 1) % BATCH_COMMIT == 0:
            conn.commit()
            progress = (i + 1) / TOTAL_CLAIMS * 100
            print(f"\rProgress: {i + 1:,} / {TOTAL_CLAIMS:,} ({progress:.1f}%) - Fraud: {fraud_count:,}", end="", flush=True)
    
    # Final commit
    conn.commit()
    csv_file.close()
    cursor.close()
    conn.close()
    
    print("\n\n" + "=" * 70)
    print("GENERATION COMPLETE")
    print("=" * 70)
    print(f"Total claims: {TOTAL_CLAIMS:,}")
    print(f"Fraud claims: {fraud_count:,} ({fraud_count/TOTAL_CLAIMS*100:.1f}%)")
    print(f"Approved: {approved_count:,} ({approved_count/TOTAL_CLAIMS*100:.1f}%)")
    print(f"Declined: {TOTAL_CLAIMS - approved_count:,}")
    print(f"\nLabels saved to: synthetic_fraud_labels.csv")
    print("=" * 70)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nGeneration interrupted by user")
    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback
        traceback.print_exc()