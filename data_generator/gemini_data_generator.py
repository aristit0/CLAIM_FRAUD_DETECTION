#!/usr/bin/env python3
import random
import csv
import uuid
from faker import Faker
from datetime import datetime, timedelta
import numpy as np

# Konfigurasi
TOTAL_CLAIMS = 50_000         # Total data history yang mau di-generate
START_DATE = datetime(2023, 1, 1)
END_DATE = datetime(2023, 12, 31)

# Rasio Fraud Global (misal 5% murni, sisanya pola abu-abu)
GLOBAL_FRAUD_RATIO = 0.15

# Setup Faker
fake = Faker("id_ID")

# ==============================================================================
# 1. CLINICAL RULES & KNOWLEDGE BASE (Single Source of Truth)
# ==============================================================================
# Disalin dan diperluas dari config.py Anda agar standalone
COMPAT_RULES = {
    # --- PENYAKIT KRONIS (Cenderung Berulang) ---
    "E11": {  # Diabetes Tipe 2
        "desc": "Diabetes mellitus tipe 2",
        "type": "chronic",
        "procedures": ["03.31", "90.59", "90.59A"], # Lab rutin, Glukosa, HbA1c
        "drugs": ["KFA006", "KFA035", "KFA036"],    # Metformin, Insulin
        "vitamins": ["Vitamin B Complex", "Vitamin D 1000 IU", "Magnesium 250 mg"]
    },
    "I10": {  # Hipertensi
        "desc": "Hipertensi esensial",
        "type": "chronic",
        "procedures": ["03.31", "89.14", "89.02"],  # Lab, EKG, Fisik
        "drugs": ["KFA007", "KFA019", "KFA018"],    # Amlodipine, Captopril, Simvastatin
        "vitamins": ["Vitamin D 1000 IU", "Vitamin B Complex"]
    },
    
    # --- PENYAKIT AKUT (Insidental) ---
    "J06": {  # Common Cold
        "desc": "Infeksi saluran pernapasan atas akut",
        "type": "acute",
        "procedures": ["89.02", "96.70"],           # Fisik, Injeksi (jarang)
        "drugs": ["KFA001", "KFA009", "KFA031"],    # Paracetamol, Cetirizine, CTM
        "vitamins": ["Vitamin C 500 mg", "Zinc 20 mg"]
    },
    "A09": {  # Diare
        "desc": "Diare dan gastroenteritis",
        "type": "acute",
        "procedures": ["03.31", "99.15"],           # Lab, Infus (bila parah)
        "drugs": ["KFA005", "KFA024", "KFA038"],    # Oralit, Loperamide, Zinc
        "vitamins": ["Zinc 20 mg", "Probiotic Complex"]
    },
    "K29": {  # Gastritis
        "desc": "Gastritis dan duodenitis",
        "type": "acute",
        "procedures": ["45.13", "03.31"],           # Endoskopi (jarang), Lab
        "drugs": ["KFA004", "KFA023", "KFA012"],    # Omeprazole, Antacid, Ranitidine
        "vitamins": ["Vitamin E 400 IU"]
    }
}

# Master Data Dummy untuk Random Pick saat Fraud
ALL_PROCEDURES = [
    ("99.04", "Transfusi darah"), ("87.03", "X-Ray Dada"), ("88.38", "CT Scan"),
    ("93.90", "Fisioterapi"), ("96.04", "Nebulizer")
]
ALL_DRUGS = [
    ("KFA003", "Ceftriaxone injeksi"), ("KFA002", "Amoxicillin"), 
    ("KFA040", "Levofloxacin"), ("KFA027", "Ketorolac")
]
ALL_VITAMINS = [
    "Vitamin A 5000 IU", "Fish Oil Omega-3", "Multivitamin Anak", "Vitamin E 400 IU"
]

# ==============================================================================
# 2. PATIENT & DOCTOR POOL (Untuk Realisme History)
# ==============================================================================
class PersonPool:
    def __init__(self, size, role="patient"):
        self.pool = []
        print(f"Generating {role} pool ({size} people)...")
        for _ in range(size):
            gender = random.choice(["M", "F"])
            person = {
                "id": str(uuid.uuid4())[:8],
                "nik": self._gen_nik(gender),
                "name": fake.name_male() if gender == "M" else fake.name_female(),
                "gender": gender,
                "dob": fake.date_of_birth(minimum_age=18, maximum_age=80),
                "risk_profile": random.choices(["normal", "abuser"], weights=[0.95, 0.05])[0] if role == "patient" else "normal",
                "fraud_tendency": random.choices(["honest", "fraudster"], weights=[0.90, 0.10])[0] if role == "doctor" else "honest"
            }
            # Assign chronic disease to some patients
            if role == "patient" and random.random() < 0.30: # 30% populasi punya penyakit kronis
                person["chronic_condition"] = random.choice(["E11", "I10"])
            else:
                person["chronic_condition"] = None
                
            self.pool.append(person)

    def _gen_nik(self, gender):
        # Simple NIK generator
        prov = random.randint(11, 99)
        dob_code = f"{random.randint(1,31):02d}"
        if gender == "F": dob_code = str(int(dob_code) + 40)
        return f"{prov}{random.randint(10,99)}01{dob_code}0190{random.randint(1000,9999)}"

    def get_random(self):
        return random.choice(self.pool)

# Inisialisasi Pool
# Kita buat 5.000 pasien unik dan 100 dokter untuk mensimulasikan populasi nyata
patients = PersonPool(5000, "patient")
doctors = PersonPool(100, "doctor")

# ==============================================================================
# 3. LOGIKA GENERASI KLAIM
# ==============================================================================
def generate_visit(patient, doctor, date_visit):
    # 1. Tentukan Diagnosis
    # Jika pasien punya penyakit kronis, 70% kemungkinan dia datang untuk kontrol penyakit itu
    if patient["chronic_condition"] and random.random() < 0.70:
        dx_code = patient["chronic_condition"]
    else:
        # Penyakit akut random
        dx_code = random.choice(["J06", "A09", "K29"])
    
    rule = COMPAT_RULES[dx_code]
    
    # 2. Tentukan Profil Klaim (Normal vs Fraud)
    # Fraud bisa dipicu oleh Dokter yang 'nakal' atau Pasien 'abuser' atau random bad luck
    is_fraud = False
    fraud_type = None
    
    # Probabilitas fraud dasar
    fraud_chance = 0.05 
    if doctor["fraud_tendency"] == "fraudster": fraud_chance += 0.30  # Dokter nakal nambah risiko
    if patient["risk_profile"] == "abuser": fraud_chance += 0.20      # Pasien nakal nambah risiko
    
    if random.random() < fraud_chance:
        is_fraud = True
        # Pilih tipe fraud
        fraud_type = random.choice(["clinical_mismatch", "upcoding", "phantom_billing"])

    # 3. Generate Item Medis (Procedure, Drug, Vitamin)
    items = {
        "proc": [], "drug": [], "vit": []
    }
    costs = {"proc": 0, "drug": 0, "vit": 0}

    # -- Logic Normal --
    # Ambil dari rule yang valid
    proc_code = random.choice(rule["procedures"])
    drug_code = random.choice(rule["drugs"])
    vit_name  = random.choice(rule["vitamins"])
    
    # Base Cost
    c_proc = random.randint(100_000, 300_000)
    c_drug = random.randint(50_000, 150_000)
    c_vit  = random.randint(20_000, 80_000)

    # -- Logic Fraud Injection --
    if is_fraud:
        if fraud_type == "clinical_mismatch":
            # Ganti dengan obat/tindakan yang TIDAK ADA di rule (misal Antibotik berat untuk Flu)
            # Ambil dari ALL_DRUGS yang tidak ada di allowed drugs diagnosis ini
            bad_drug = random.choice(ALL_DRUGS)
            drug_code = bad_drug[0] # Inject mismatch drug
            c_drug += 200_000       # Mahal pula
            
            # Kadang tambah vitamin mahal ga perlu
            if random.random() < 0.5:
                vit_name = random.choice(ALL_VITAMINS)
                c_vit += 100_000

        elif fraud_type == "upcoding":
            # Item medis benar/sesuai, tapi harganya di-markup gila-gilaan
            c_proc *= random.randint(3, 8)
            c_drug *= random.randint(2, 5)
        
        elif fraud_type == "phantom_billing":
            # Diagnosis ringan, tapi ada tindakan besar yang aneh
            # Misal: Common Cold tapi ada Transfusi Darah atau CT Scan
            bad_proc = random.choice(ALL_PROCEDURES)
            proc_code = bad_proc[0]
            c_proc = random.randint(1_000_000, 5_000_000)

    # Finalize Items
    items["proc"] = proc_code
    items["drug"] = drug_code
    items["vit"] = vit_name
    costs["proc"] = c_proc
    costs["drug"] = c_drug
    costs["vit"] = c_vit
    
    total_cost = c_proc + c_drug + c_vit

    # Status approval (Simulasi keputusan sistem saat ini)
    # Aturan simpel: Jika mismatch klinis -> Declined. Jika kemahalan -> Declined/Review.
    status = "approved"
    if is_fraud:
        # Sistem tidak 100% sempurna menangkap fraud, misal hanya 80% tertangkap
        if random.random() < 0.80:
            status = "declined"
    
    # Return record
    return {
        "visit_date": date_visit.strftime("%Y-%m-%d"),
        "patient_id": patient["id"],
        "patient_name": patient["name"],
        "patient_nik": patient["nik"],
        "patient_gender": patient["gender"],
        "patient_age": (date_visit.date() - patient["dob"]).days // 365,
        
        "doctor_name": doctor["name"],
        "doctor_flag": doctor["fraud_tendency"], # Fitur tersembunyi untuk analisis
        
        "diagnosis_code": dx_code,
        "diagnosis_desc": rule["desc"],
        
        "proc_code": items["proc"],
        "drug_code": items["drug"],
        "vit_name": items["vit"],
        
        "cost_proc": costs["proc"],
        "cost_drug": costs["drug"],
        "cost_vit": costs["vit"],
        "total_cost": total_cost,
        
        "is_fraud": 1 if is_fraud else 0,
        "fraud_type": fraud_type if is_fraud else "none",
        "status": status
    }

# ==============================================================================
# 4. MAIN LOOP GENERATOR
# ==============================================================================
def main():
    print(f"Generating {TOTAL_CLAIMS} historical claims...")
    print(f"Period: {START_DATE.date()} to {END_DATE.date()}")
    
    # Kita urutkan berdasarkan waktu agar seperti history beneran
    # Generate random timestamps dalam rentang tahun
    timestamps = [
        START_DATE + timedelta(seconds=random.randint(0, int((END_DATE - START_DATE).total_seconds())))
        for _ in range(TOTAL_CLAIMS)
    ]
    timestamps.sort() # Sort kronologis

    filename = "historical_claims_data.csv"
    
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = [
            "claim_id", "visit_date", "patient_nik", "patient_name", "patient_gender", "patient_age",
            "doctor_name", "department", "diagnosis_code", "diagnosis_desc",
            "proc_code", "drug_code", "vitamin_name",
            "cost_proc", "cost_drug", "cost_vit", "total_cost",
            "status", "fraud_label", "fraud_type"
        ]
        writer.writerow(header)

        for i, ts in enumerate(timestamps):
            # Pick patient & doctor
            # Logika: Pasien kronis mungkin muncul beberapa kali di list timestamps
            # Kita ambil random dari pool, secara statistik pool kecil = frekuensi tinggi
            patient = patients.get_random()
            doctor = doctors.get_random()
            
            rec = generate_visit(patient, doctor, ts)
            
            writer.writerow([
                f"CLM{i+1:08d}",
                rec["visit_date"],
                rec["patient_nik"],
                rec["patient_name"],
                rec["patient_gender"],
                rec["patient_age"],
                rec["doctor_name"],
                "Poli Umum" if rec["diagnosis_code"] in ["J06", "A09"] else "Poli Penyakit Dalam", # Sederhana
                rec["diagnosis_code"],
                rec["diagnosis_desc"],
                rec["proc_code"],
                rec["drug_code"],
                rec["vit_name"],
                rec["cost_proc"],
                rec["cost_drug"],
                rec["cost_vit"],
                rec["total_cost"],
                rec["status"],
                rec["is_fraud"],
                rec["fraud_type"]
            ])
            
            if (i+1) % 5000 == 0:
                print(f"  ... generated {i+1} claims")

    print(f"\nDONE! Data saved to {filename}")

if __name__ == "__main__":
    main()