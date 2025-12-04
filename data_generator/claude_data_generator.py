#!/usr/bin/env python3
"""
BPJS Claim Data Generator - Production Version
Generates realistic claim data with controlled fraud patterns
Based on clinical rules and master data from MySQL

Features:
- Realistic fraud patterns (15-30% fraud ratio)
- Clinical rule compliance
- Master data validation
- Chronic patient simulation
- Actor profiles (patient abusers, fraudster doctors)
- Temporal patterns
"""

import random
import uuid
import sys
import os
import mysql.connector
from faker import Faker
from datetime import datetime, timedelta
from collections import defaultdict

# ==============================================================================
# CONFIGURATION
# ==============================================================================
TOTAL_CLAIMS = 100000
START_DATE = datetime(2025, 1, 1)
END_DATE = datetime(2025, 11, 20)
BASE_FRAUD_RATIO = 0.15  # 15% base fraud rate

DB_CONFIG = {
    "host": "cdpmsi.tomodachis.org",
    "user": "cloudera",
    "password": "T1ku$H1t4m",
    "database": "claimdb"
}

fake = Faker("id_ID")

# ==============================================================================
# LOAD MASTER DATA FROM DATABASE
# ==============================================================================
print("=" * 80)
print("BPJS CLAIM DATA GENERATOR - PRODUCTION VERSION")
print("=" * 80)
print("\n[1/7] Loading master data from database...")

def load_master_data():
    """Load all master data from MySQL"""
    conn = mysql.connector.connect(**DB_CONFIG)
    cur = conn.cursor(dictionary=True)
    
    # Load ICD-10 (Diagnoses)
    cur.execute("SELECT code, description FROM master_icd10")
    master_icd10 = {row["code"]: row["description"] for row in cur.fetchall()}
    
    # Load ICD-9 (Procedures)
    cur.execute("SELECT code, description FROM master_icd9")
    master_icd9 = {row["code"]: row["description"] for row in cur.fetchall()}
    
    # Load Drugs
    cur.execute("SELECT code, name FROM master_drug")
    master_drug = {row["code"]: row["name"] for row in cur.fetchall()}
    
    # Load Vitamins
    cur.execute("SELECT name FROM master_vitamin")
    master_vitamin = [row["name"] for row in cur.fetchall()]
    
    # Load Clinical Rules
    print("  Loading clinical rules...")
    
    # Diagnosis → Procedure rules
    cur.execute("""
        SELECT icd10_code, icd9_code, is_mandatory, severity_level 
        FROM clinical_rule_dx_procedure
    """)
    dx_proc_rules = defaultdict(list)
    for row in cur.fetchall():
        dx_proc_rules[row["icd10_code"]].append({
            "code": row["icd9_code"],
            "mandatory": bool(row["is_mandatory"]),
            "severity": row["severity_level"]
        })
    
    # Diagnosis → Drug rules
    cur.execute("""
        SELECT icd10_code, drug_code, is_mandatory, severity_level 
        FROM clinical_rule_dx_drug
    """)
    dx_drug_rules = defaultdict(list)
    for row in cur.fetchall():
        dx_drug_rules[row["icd10_code"]].append({
            "code": row["drug_code"],
            "mandatory": bool(row["is_mandatory"]),
            "severity": row["severity_level"]
        })
    
    # Diagnosis → Vitamin rules
    cur.execute("""
        SELECT icd10_code, vitamin_name, is_mandatory, severity_level 
        FROM clinical_rule_dx_vitamin
    """)
    dx_vit_rules = defaultdict(list)
    for row in cur.fetchall():
        dx_vit_rules[row["icd10_code"]].append({
            "name": row["vitamin_name"],
            "mandatory": bool(row["is_mandatory"]),
            "severity": row["severity_level"]
        })
    
    cur.close()
    conn.close()
    
    # Build complete clinical rules dictionary
    clinical_rules = {}
    all_diagnoses = list(master_icd10.keys())
    
    for dx_code in all_diagnoses:
        clinical_rules[dx_code] = {
            "description": master_icd10[dx_code],
            "procedures": [r["code"] for r in dx_proc_rules.get(dx_code, [])],
            "mandatory_procedures": [r["code"] for r in dx_proc_rules.get(dx_code, []) if r["mandatory"]],
            "drugs": [r["code"] for r in dx_drug_rules.get(dx_code, [])],
            "mandatory_drugs": [r["code"] for r in dx_drug_rules.get(dx_code, []) if r["mandatory"]],
            "vitamins": [r["name"] for r in dx_vit_rules.get(dx_code, [])],
        }
    
    print(f"  ✓ Loaded {len(master_icd10)} diagnoses")
    print(f"  ✓ Loaded {len(master_icd9)} procedures")
    print(f"  ✓ Loaded {len(master_drug)} drugs")
    print(f"  ✓ Loaded {len(master_vitamin)} vitamins")
    print(f"  ✓ Built clinical rules for {len(clinical_rules)} diagnoses")
    
    return {
        "icd10": master_icd10,
        "icd9": master_icd9,
        "drug": master_drug,
        "vitamin": master_vitamin,
        "clinical_rules": clinical_rules
    }

# Load master data
MASTER_DATA = load_master_data()
CLINICAL_RULES = MASTER_DATA["clinical_rules"]
ALL_DIAGNOSES = list(MASTER_DATA["icd10"].keys())

# Build "bad" items for fraud (items that don't match diagnosis)
ALL_PROCEDURES = list(MASTER_DATA["icd9"].keys())
ALL_DRUGS = list(MASTER_DATA["drug"].keys())
ALL_VITAMINS = MASTER_DATA["vitamin"]

print("\n[2/7] Building fraud pattern library...")

# For each diagnosis, identify incompatible items
INCOMPATIBLE_ITEMS = {}
for dx_code in ALL_DIAGNOSES:
    rules = CLINICAL_RULES[dx_code]
    
    # Items that are NOT allowed for this diagnosis
    INCOMPATIBLE_ITEMS[dx_code] = {
        "procedures": [p for p in ALL_PROCEDURES if p not in rules["procedures"]],
        "drugs": [d for d in ALL_DRUGS if d not in rules["drugs"]],
        "vitamins": [v for v in ALL_VITAMINS if v not in rules["vitamins"]],
    }

print(f"  ✓ Built incompatibility matrix for {len(INCOMPATIBLE_ITEMS)} diagnoses")

# ==============================================================================
# COST THRESHOLDS (Realistic BPJS ranges)
# ==============================================================================
COST_RANGES = {
    "J06": {"proc": (50000, 150000), "drug": (20000, 80000), "vit": (10000, 30000)},  # Common cold
    "K29": {"proc": (100000, 300000), "drug": (50000, 150000), "vit": (15000, 40000)},  # Gastritis
    "E11": {"proc": (150000, 400000), "drug": (100000, 300000), "vit": (20000, 60000)},  # Diabetes
    "I10": {"proc": (100000, 350000), "drug": (80000, 250000), "vit": (15000, 50000)},  # Hypertension
    "J45": {"proc": (120000, 380000), "drug": (100000, 280000), "vit": (20000, 55000)},  # Asthma
}

# Default for unknown diagnoses
DEFAULT_COST_RANGE = {"proc": (80000, 200000), "drug": (40000, 120000), "vit": (15000, 40000)}

# ==============================================================================
# PATIENT & DOCTOR POOLS
# ==============================================================================
print("\n[3/7] Creating patient and doctor pools...")

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
                # Risk profiles (reduced from original)
                "is_abuser": random.random() < 0.03 if role == "patient" else False,  # 3% (was 5%)
                "is_fraudster": random.random() < 0.05 if role == "doctor" else False,  # 5% (was 10%)
                # Chronic conditions
                "chronic_dx": random.choice(["E11", "I10"]) if role == "patient" and random.random() < 0.25 else None,
                # Claim history
                "claim_count": 0,
                "last_visit": None,
            }
            self.people.append(person)
    
    def _gen_nik(self, gender):
        prov = random.randint(11, 99)
        day = random.randint(1, 31)
        if gender == "F":
            day += 40
        month = random.randint(1, 12)
        year = random.randint(50, 99)
        return f"{prov}{random.randint(10,99)}{day:02d}{month:02d}{year:02d}{random.randint(1000,9999)}"
    
    def get_random(self):
        return random.choice(self.people)
    
    def get_chronic_patient(self, dx_code):
        """Get a patient with specific chronic condition"""
        candidates = [p for p in self.people if p["chronic_dx"] == dx_code]
        return random.choice(candidates) if candidates else self.get_random()

# Initialize pools
patients = Pool(1000, "patient")
doctors = Pool(50, "doctor")

print(f"  ✓ Created {len(patients.people)} patients")
print(f"  ✓ Created {len(doctors.people)} doctors")
print(f"  ✓ Patient abusers: {sum(1 for p in patients.people if p['is_abuser'])}")
print(f"  ✓ Fraudster doctors: {sum(1 for d in doctors.people if d['is_fraudster'])}")

# ==============================================================================
# CLAIM GENERATION LOGIC
# ==============================================================================
print("\n[4/7] Preparing claim generation engine...")

def calculate_fraud_probability(patient, doctor, dx_code):
    """Calculate fraud probability with caps"""
    base_prob = BASE_FRAUD_RATIO  # 15%
    
    # Patient risk
    if patient["is_abuser"]:
        base_prob += 0.10  # +10% (reduced from 20%)
    
    # Doctor risk
    if doctor["is_fraudster"]:
        base_prob += 0.10  # +10% (reduced from 30%)
    
    # Frequency risk (if patient has many recent claims)
    if patient["claim_count"] > 10:
        base_prob += 0.05  # +5% for high frequency
    
    # Cap at 30% max
    return min(base_prob, 0.30)


def select_valid_items(dx_code):
    """Select clinically appropriate items for diagnosis"""
    rules = CLINICAL_RULES[dx_code]
    cost_range = COST_RANGES.get(dx_code, DEFAULT_COST_RANGE)
    
    # Procedures - must include mandatory
    procedures = []
    if rules["mandatory_procedures"]:
        procedures.extend(rules["mandatory_procedures"])
    
    # Add optional procedures (50% chance)
    optional_procs = [p for p in rules["procedures"] if p not in rules["mandatory_procedures"]]
    if optional_procs and random.random() < 0.5:
        procedures.append(random.choice(optional_procs))
    
    # Drugs - must include mandatory
    drugs = []
    if rules["mandatory_drugs"]:
        drugs.extend(rules["mandatory_drugs"])
    
    # Add optional drugs (60% chance)
    optional_drugs = [d for d in rules["drugs"] if d not in rules["mandatory_drugs"]]
    if optional_drugs and random.random() < 0.6:
        drugs.append(random.choice(optional_drugs))
    
    # Vitamins (30% chance)
    vitamins = []
    if rules["vitamins"] and random.random() < 0.3:
        vitamins.append(random.choice(rules["vitamins"]))
    
    # Calculate costs
    proc_cost = random.randint(*cost_range["proc"]) if procedures else 0
    drug_cost = random.randint(*cost_range["drug"]) if drugs else 0
    vit_cost = random.randint(*cost_range["vit"]) if vitamins else 0
    
    return {
        "procedures": procedures,
        "drugs": drugs,
        "vitamins": vitamins,
        "proc_cost": proc_cost,
        "drug_cost": drug_cost,
        "vit_cost": vit_cost,
    }


def apply_fraud_pattern(items, dx_code, fraud_type):
    """Apply specific fraud pattern"""
    incompatible = INCOMPATIBLE_ITEMS[dx_code]
    
    if fraud_type == "clinical_mismatch":
        # Replace with incompatible items
        if items["procedures"] and incompatible["procedures"]:
            if random.random() < 0.6:  # 60% chance to replace procedure
                items["procedures"][0] = random.choice(incompatible["procedures"])
        
        if items["drugs"] and incompatible["drugs"]:
            if random.random() < 0.7:  # 70% chance to replace drug
                items["drugs"][0] = random.choice(incompatible["drugs"])
        
        if random.random() < 0.4:  # 40% chance to add incompatible vitamin
            if incompatible["vitamins"]:
                items["vitamins"] = [random.choice(incompatible["vitamins"])]
    
    elif fraud_type == "upcoding":
        # Inflate costs by 2-5x
        multiplier = random.uniform(2.0, 5.0)
        items["proc_cost"] = int(items["proc_cost"] * multiplier)
        items["drug_cost"] = int(items["drug_cost"] * multiplier)
        items["vit_cost"] = int(items["vit_cost"] * multiplier)
    
    elif fraud_type == "phantom":
        # Add expensive, unnecessary procedure
        if incompatible["procedures"]:
            phantom_proc = random.choice(incompatible["procedures"])
            items["procedures"].append(phantom_proc)
            items["proc_cost"] += random.randint(1000000, 2500000)  # Very expensive
    
    elif fraud_type == "duplicate":
        # Duplicate existing items
        if items["procedures"]:
            items["procedures"].append(items["procedures"][0])
            items["proc_cost"] *= 2
        if items["drugs"]:
            items["drugs"].append(items["drugs"][0])
            items["drug_cost"] *= 2
    
    return items


def generate_single_claim(visit_date, claim_number):
    """Generate one realistic claim"""
    
    # Select patient
    # 25% chance to select chronic patient with matching diagnosis
    if random.random() < 0.25:
        dx_code = random.choice(["E11", "I10"])  # Chronic diagnoses
        patient = patients.get_chronic_patient(dx_code)
    else:
        patient = patients.get_random()
        # If patient has chronic condition, 70% chance they visit for that
        if patient["chronic_dx"] and random.random() < 0.7:
            dx_code = patient["chronic_dx"]
        else:
            dx_code = random.choice(ALL_DIAGNOSES)
    
    # Select doctor
    doctor = doctors.get_random()
    
    # Calculate fraud probability
    fraud_prob = calculate_fraud_probability(patient, doctor, dx_code)
    is_fraud = random.random() < fraud_prob
    
    # Select items (valid by default)
    items = select_valid_items(dx_code)
    
    fraud_type = None
    if is_fraud:
        # Select fraud type
        fraud_type = random.choice([
            "clinical_mismatch",  # 35%
            "upcoding",           # 35%
            "phantom",            # 20%
            "duplicate",          # 10%
        ])
        items = apply_fraud_pattern(items, dx_code, fraud_type)
    
    # Calculate total
    total_cost = items["proc_cost"] + items["drug_cost"] + items["vit_cost"]
    
    # Determine status
    # Fraud: 75% declined, 25% slip through (approved)
    # Legitimate: 95% approved, 5% incorrectly declined
    if is_fraud:
        status = "declined" if random.random() < 0.75 else "approved"
    else:
        status = "approved" if random.random() < 0.95 else "declined"
    
    # Update patient history
    patient["claim_count"] += 1
    patient["last_visit"] = visit_date
    
    return {
        "claim_number": claim_number,
        "patient": patient,
        "doctor": doctor,
        "visit_date": visit_date,
        "dx_code": dx_code,
        "dx_desc": CLINICAL_RULES[dx_code]["description"],
        "items": items,
        "total_cost": total_cost,
        "status": status,
        "is_fraud": is_fraud,
        "fraud_type": fraud_type,
    }


# ==============================================================================
# DATABASE INSERTION
# ==============================================================================
def insert_claim_to_db(cursor, claim_data):
    """Insert claim into MySQL database"""
    p = claim_data["patient"]
    d = claim_data["doctor"]
    items = claim_data["items"]
    
    # Determine department based on diagnosis
    dept_mapping = {
        "J06": "Poli Umum",
        "J45": "Poli Paru",
        "K29": "Poli Penyakit Dalam",
        "E11": "Poli Penyakit Dalam",
        "I10": "Poli Jantung",
    }
    dept = dept_mapping.get(claim_data["dx_code"], "Poli Umum")
    
    # 1. Insert header
    sql_header = """
        INSERT INTO claim_header (
            patient_nik, patient_name, patient_gender, patient_dob,
            patient_address, patient_phone,
            visit_date, visit_type, doctor_name, department,
            total_procedure_cost, total_drug_cost, total_vitamin_cost,
            total_claim_amount, status
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    

    vals_header = (
        p["nik"], p["name"], p["gender"], p["dob"],
        p["address"], p["phone"],
        claim_data["visit_date"].strftime("%Y-%m-%d"),   
        "rawat jalan", d["name"], dept,
        items["proc_cost"], items["drug_cost"], items["vit_cost"],
        claim_data["total_cost"], claim_data["status"]
    )
    
    cursor.execute(sql_header, vals_header)
    claim_id = cursor.lastrowid
    
    # 2. Insert diagnosis
    sql_diag = """
        INSERT INTO claim_diagnosis (claim_id, icd10_code, icd10_description, is_primary)
        VALUES (%s, %s, %s, 1)
    """
    cursor.execute(sql_diag, (claim_id, claim_data["dx_code"], claim_data["dx_desc"]))
    
    # 3. Insert procedures
    for proc_code in items["procedures"]:
        proc_desc = MASTER_DATA["icd9"].get(proc_code, "Unknown procedure")
        sql_proc = """
            INSERT INTO claim_procedure (claim_id, icd9_code, icd9_description, quantity, procedure_date, cost)
            VALUES (%s, %s, %s, 1, %s, %s)
        """
        # Distribute cost across procedures
        proc_unit_cost = items["proc_cost"] // len(items["procedures"]) if items["procedures"] else 0
        cursor.execute(sql_proc, (claim_id, proc_code, proc_desc, claim_data["visit_date"], proc_unit_cost))
    
    # 4. Insert drugs
    for drug_code in items["drugs"]:
        drug_name = MASTER_DATA["drug"].get(drug_code, "Unknown drug")
        sql_drug = """
            INSERT INTO claim_drug (claim_id, drug_code, drug_name, cost)
            VALUES (%s, %s, %s, %s)
        """
        # Distribute cost across drugs
        drug_unit_cost = items["drug_cost"] // len(items["drugs"]) if items["drugs"] else 0
        cursor.execute(sql_drug, (claim_id, drug_code, drug_name, drug_unit_cost))
    
    # 5. Insert vitamins
    for vit_name in items["vitamins"]:
        sql_vit = """
            INSERT INTO claim_vitamin (claim_id, vitamin_name, cost)
            VALUES (%s, %s, %s)
        """
        # Distribute cost across vitamins
        vit_unit_cost = items["vit_cost"] // len(items["vitamins"]) if items["vitamins"] else 0
        cursor.execute(sql_vit, (claim_id, vit_name, vit_unit_cost))
    
    return claim_id


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
def main():
    print("\n[5/7] Generating claim timestamps...")
    
    # Generate timestamps (sorted chronologically)
    total_seconds = int((END_DATE - START_DATE).total_seconds())
    timestamps = sorted([
        START_DATE + timedelta(seconds=random.randint(0, total_seconds))
        for _ in range(TOTAL_CLAIMS)
    ])
    
    print(f"  ✓ Generated {len(timestamps):,} timestamps")
    print(f"  ✓ Date range: {timestamps[0].date()} to {timestamps[-1].date()}")
    
    print("\n[6/7] Connecting to database...")
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()
        print(f"  ✓ Connected to {DB_CONFIG['host']}")
    except Exception as e:
        print(f"  ✗ Connection failed: {e}")
        return
    
    print(f"\n[7/7] Generating and inserting {TOTAL_CLAIMS:,} claims...")
    print("  (This may take several minutes...)\n")
    
    stats = {
        "total": 0,
        "fraud": 0,
        "legitimate": 0,
        "approved": 0,
        "declined": 0,
        "fraud_types": defaultdict(int),
    }
    
    start_time = datetime.now()
    
    for i, ts in enumerate(timestamps, 1):
        claim_data = generate_single_claim(ts, i)
        insert_claim_to_db(cursor, claim_data)
        
        # Update stats
        stats["total"] += 1
        if claim_data["is_fraud"]:
            stats["fraud"] += 1
            if claim_data["fraud_type"]:
                stats["fraud_types"][claim_data["fraud_type"]] += 1
        else:
            stats["legitimate"] += 1
        
        if claim_data["status"] == "approved":
            stats["approved"] += 1
        else:
            stats["declined"] += 1
        
        # Commit every 100 claims
        if i % 100 == 0:
            conn.commit()
            elapsed = (datetime.now() - start_time).total_seconds()
            rate = i / elapsed if elapsed > 0 else 0
            eta = (TOTAL_CLAIMS - i) / rate if rate > 0 else 0
            
            print(f"  Progress: {i:,}/{TOTAL_CLAIMS:,} ({i/TOTAL_CLAIMS*100:.1f}%) | "
                  f"Rate: {rate:.0f} claims/sec | ETA: {eta/60:.1f} min", end="\r")
    
    # Final commit
    conn.commit()
    cursor.close()
    conn.close()
    
    elapsed_total = (datetime.now() - start_time).total_seconds()
    
    # Print summary
    print("\n\n" + "=" * 80)
    print("DATA GENERATION COMPLETE")
    print("=" * 80)
    
    print(f"\n📊 Generation Statistics:")
    print(f"  Total claims:       {stats['total']:,}")
    print(f"  Legitimate:         {stats['legitimate']:,} ({stats['legitimate']/stats['total']*100:.1f}%)")
    print(f"  Fraud:              {stats['fraud']:,} ({stats['fraud']/stats['total']*100:.1f}%)")
    print(f"  Approved:           {stats['approved']:,} ({stats['approved']/stats['total']*100:.1f}%)")
    print(f"  Declined:           {stats['declined']:,} ({stats['declined']/stats['total']*100:.1f}%)")
    
    print(f"\n🚨 Fraud Type Distribution:")
    for fraud_type, count in sorted(stats['fraud_types'].items(), key=lambda x: x[1], reverse=True):
        pct = count / stats['fraud'] * 100 if stats['fraud'] > 0 else 0
        print(f"  {fraud_type:20s}: {count:,} ({pct:.1f}%)")
    
    print(f"\n⏱️  Performance:")
    print(f"  Total time:         {elapsed_total/60:.1f} minutes")
    print(f"  Average rate:       {stats['total']/elapsed_total:.1f} claims/second")
    
    print("\n✅ Data successfully generated and inserted into MySQL!")
    print("=" * 80)


if __name__ == "__main__":
    main()