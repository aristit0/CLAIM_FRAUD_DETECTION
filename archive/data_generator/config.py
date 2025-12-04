#!/usr/bin/env python3
"""
Centralized configuration for fraud detection system.
This ensures consistency across data generation, ETL, training, and inference.
"""

# ================================================================
# CLINICAL COMPATIBILITY RULES (SINGLE SOURCE OF TRUTH)
# ================================================================
COMPAT_RULES = {
    # Gastrointestinal
    "A09": {  # Diare dan gastroenteritis
        "procedures": ["03.31", "03.91", "99.15"],
        "drugs": ["KFA005", "KFA013", "KFA024", "KFA025", "KFA038"],
        "vitamins": ["Vitamin D 1000 IU", "Zinc 20 mg", "Probiotic Complex"]
    },
    "K29": {  # Gastritis
        "procedures": ["45.13", "03.31", "89.02"],
        "drugs": ["KFA004", "KFA012", "KFA023", "KFA034", "KFA037"],
        "vitamins": ["Vitamin E 400 IU", "Vitamin B Complex"]
    },
    "K52": {  # Gastroenteritis noninfeksi
        "procedures": ["03.31", "03.92"],
        "drugs": ["KFA004", "KFA024", "KFA038"],
        "vitamins": ["Probiotic Complex", "Zinc 20 mg"]
    },
    "K21": {  # GERD
        "procedures": ["45.13", "89.02"],
        "drugs": ["KFA004", "KFA034", "KFA023"],
        "vitamins": ["Vitamin E 200 IU"]
    },
    
    # Respiratory
    "J06": {  # Common cold
        "procedures": ["96.70", "89.02", "87.03"],
        "drugs": ["KFA001", "KFA002", "KFA009", "KFA031"],
        "vitamins": ["Vitamin C 500 mg", "Vitamin C 1000 mg", "Zinc 20 mg"]
    },
    "J06.9": {  # Common cold - unspecified
        "procedures": ["96.70", "89.02"],
        "drugs": ["KFA001", "KFA002", "KFA009"],
        "vitamins": ["Vitamin C 500 mg", "Zinc 20 mg"]
    },
    "J02": {  # Faringitis akut
        "procedures": ["89.02", "34.01"],
        "drugs": ["KFA001", "KFA002", "KFA014"],
        "vitamins": ["Vitamin C 1000 mg"]
    },
    "J20": {  # Bronkitis akut
        "procedures": ["87.03", "89.02", "96.04"],
        "drugs": ["KFA002", "KFA022", "KFA026"],
        "vitamins": ["Vitamin C 1000 mg", "Vitamin B Complex"]
    },
    "J45": {  # Asma
        "procedures": ["96.04", "93.05", "87.03"],
        "drugs": ["KFA010", "KFA011", "KFA021"],
        "vitamins": ["Vitamin D 1000 IU", "Vitamin C 500 mg"]
    },
    "J18": {  # Pneumonia
        "procedures": ["87.03", "03.31", "99.15"],
        "drugs": ["KFA003", "KFA014", "KFA030", "KFA040"],
        "vitamins": ["Vitamin C 1000 mg", "Vitamin D3 2000 IU"]
    },
    
    # Cardiovascular
    "I10": {  # Hipertensi
        "procedures": ["03.31", "89.14", "89.02"],
        "drugs": ["KFA007", "KFA019"],
        "vitamins": ["Vitamin D 1000 IU", "Magnesium 250 mg", "Vitamin B Complex"]
    },
    
    # Endocrine
    "E11": {  # Diabetes Type 2
        "procedures": ["03.31", "90.59", "90.59A"],
        "drugs": ["KFA006", "KFA035", "KFA036"],
        "vitamins": ["Vitamin B Complex", "Vitamin D 1000 IU", "Magnesium 250 mg"]
    },
    "E16": {  # Hipoglikemia
        "procedures": ["90.59", "03.31"],
        "drugs": ["KFA035", "KFA036"],
        "vitamins": ["Vitamin B Complex"]
    },
    
    # Pain & Inflammation
    "R51": {  # Sakit kepala
        "procedures": ["89.02"],
        "drugs": ["KFA001", "KFA008", "KFA033"],
        "vitamins": ["Vitamin B Complex", "Magnesium 250 mg"]
    },
    "G43": {  # Migrain
        "procedures": ["89.02", "88.53"],
        "drugs": ["KFA001", "KFA008", "KFA033"],
        "vitamins": ["Magnesium 250 mg", "Vitamin B Complex Forte"]
    },
    "M54.5": {  # Nyeri punggung bawah
        "procedures": ["89.0", "93.27", "93.94"],
        "drugs": ["KFA008", "KFA033", "KFA027"],
        "vitamins": ["Vitamin D 1000 IU", "Calcium 500 mg"]
    },
    
    # Infections
    "N39": {  # Infeksi saluran kemih
        "procedures": ["03.91", "03.31"],
        "drugs": ["KFA030", "KFA040"],
        "vitamins": ["Vitamin C 1000 mg"]
    },
    "L03": {  # Selulitis
        "procedures": ["89.02", "96.70"],
        "drugs": ["KFA003", "KFA014", "KFA039"],
        "vitamins": ["Vitamin C 1000 mg"]
    },
    
    # Allergic
    "T78.4": {  # Alergi
        "procedures": ["89.02"],
        "drugs": ["KFA009", "KFA031", "KFA028"],
        "vitamins": ["Vitamin C 500 mg"]
    },
    "H10": {  # Konjungtivitis
        "procedures": ["89.02"],
        "drugs": ["KFA009", "KFA031"],
        "vitamins": ["Vitamin A 5000 IU"]
    },
}

# ================================================================
# COST THRESHOLDS (REALISTIC INDONESIAN HEALTHCARE)
# ================================================================
COST_THRESHOLDS = {
    "procedure": {
        "low": 50_000,
        "medium": 150_000,
        "high": 300_000,
        "extreme": 1_000_000
    },
    "drug": {
        "low": 10_000,
        "medium": 50_000,
        "high": 150_000,
        "extreme": 500_000
    },
    "vitamin": {
        "low": 5_000,
        "medium": 30_000,
        "high": 80_000,
        "extreme": 200_000
    },
    "total_claim": {
        "normal": 300_000,
        "suspicious": 1_000_000,
        "extreme": 2_000_000
    }
}

# ================================================================
# FEATURE DEFINITIONS (FOR TRAINING & INFERENCE CONSISTENCY)
# ================================================================
NUMERIC_FEATURES = [
    "patient_age",
    "total_procedure_cost",
    "total_drug_cost",
    "total_vitamin_cost",
    "total_claim_amount",
    "biaya_anomaly_score",
    "patient_frequency_risk",
    "visit_year",
    "visit_month",
    "visit_day",
    # Clinical Compatibility Scores
    "diagnosis_procedure_score",
    "diagnosis_drug_score",
    "diagnosis_vitamin_score",
    # Explicit Mismatch Flags
    "procedure_mismatch_flag",
    "drug_mismatch_flag",
    "vitamin_mismatch_flag",
    "mismatch_count",
]

CATEGORICAL_FEATURES = [
    "visit_type",
    "department",
    "icd10_primary_code",
]

# ================================================================
# MASTER DATA (FROM YOUR DATABASE)
# ================================================================
MASTER_ICD10 = [
    "A09", "A41", "B34.9", "E03", "E11", "E16", "E55", "E66",
    "G43", "G44", "H10", "I10", "J02", "J06", "J06.9", "J18",
    "J20", "J45", "K21", "K29", "K52", "K59", "K80", "K85",
    "L03", "L29", "M25.5", "M54.5", "M79.1", "N39", "R50",
    "R51", "R52", "R63", "S63", "S83", "T78.4"
]

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
    "KFA013", "KFA014", "KFA015", "KFA016", "KFA017", "KFA018",
    "KFA019", "KFA020", "KFA021", "KFA022", "KFA023", "KFA024",
    "KFA025", "KFA026", "KFA027", "KFA028", "KFA029", "KFA030",
    "KFA031", "KFA032", "KFA033", "KFA034", "KFA035", "KFA036",
    "KFA037", "KFA038", "KFA039", "KFA040"
]

MASTER_VITAMINS = [
    "Vitamin A 5000 IU", "Vitamin B Complex", "Vitamin B Complex Forte",
    "Vitamin B1 100 mg", "Vitamin B6 50 mg", "Vitamin B12 500 mcg",
    "Vitamin C 500 mg", "Vitamin C 1000 mg", "Vitamin C Effervescent 1000 mg",
    "Vitamin D 1000 IU", "Vitamin D3 2000 IU", "Vitamin E 200 IU",
    "Vitamin E 400 IU", "Vitamin K2 MK-7", "Calcium 500 mg",
    "Magnesium 250 mg", "Zinc 20 mg", "Iron Supplement 250 mg",
    "Folic Acid 1 mg", "Fish Oil Omega-3", "Probiotic Complex",
    "Multivitamin Adult", "Multivitamin Children", "Electrolyte Powder"
]