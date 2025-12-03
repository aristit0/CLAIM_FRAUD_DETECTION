#!/usr/bin/env python3
"""
Centralized configuration for BPJS fraud detection system.
Integrates with Iceberg reference tables for clinical rules.
Version: 2.0 - Production with Iceberg Integration
"""

import os
import sys

# ================================================================
# DATABASE CONNECTION SETTINGS
# ================================================================
ICEBERG_CONNECTION = "CDP-MSI"

# Table names
ICEBERG_RAW_TABLES = {
    "claim_header": "iceberg_raw.claim_header_raw",
    "claim_diagnosis": "iceberg_raw.claim_diagnosis_raw",
    "claim_procedure": "iceberg_raw.claim_procedure_raw",
    "claim_drug": "iceberg_raw.claim_drug_raw",
    "claim_vitamin": "iceberg_raw.claim_vitamin_raw",
}

ICEBERG_REF_TABLES = {
    "clinical_rule_dx_drug": "iceberg_ref.clinical_rule_dx_drug",
    "clinical_rule_dx_procedure": "iceberg_ref.clinical_rule_dx_procedure",
    "clinical_rule_dx_vitamin": "iceberg_ref.clinical_rule_dx_vitamin",
    "master_icd10": "iceberg_ref.master_icd10",
    "master_icd9": "iceberg_ref.master_icd9",
    "master_drug": "iceberg_ref.master_drug",
    "master_vitamin": "iceberg_ref.master_vitamin",
}

ICEBERG_CURATED_TABLES = {
    "claim_feature_set": "iceberg_curated.claim_feature_set",
}

# ================================================================
# FEATURE DEFINITIONS (MUST MATCH ETL OUTPUT!)
# ================================================================

NUMERIC_FEATURES = [
    # Patient demographics
    "patient_age",
    
    # Temporal features
    "visit_year",
    "visit_month",
    "visit_day",
    
    # Cost features
    "total_procedure_cost",
    "total_drug_cost",
    "total_vitamin_cost",
    "total_claim_amount",
    
    # Clinical compatibility scores (KEY FRAUD INDICATORS)
    "diagnosis_procedure_score",
    "diagnosis_drug_score",
    "diagnosis_vitamin_score",
    
    # Mismatch flags (BINARY FRAUD INDICATORS)
    "procedure_mismatch_flag",
    "drug_mismatch_flag",
    "vitamin_mismatch_flag",
    "mismatch_count",
    
    # Risk scores
    "biaya_anomaly_score",
    "patient_frequency_risk",
]

CATEGORICAL_FEATURES = [
    "visit_type",
    "department",
    "icd10_primary_code",
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

LABEL_COLUMN = "final_label"

# ================================================================
# CLINICAL COMPATIBILITY RULES (FALLBACK - IF ICEBERG NOT AVAILABLE)
# In production, these are loaded from Iceberg reference tables
# ================================================================
COMPAT_RULES_FALLBACK = {
    "J06": {
        "procedures": ["89.02", "03.31", "96.04"],
        "drugs": ["KFA001", "KFA009", "KFA031", "KFA022"],
        "vitamins": ["Vitamin C 500 mg", "Zinc 20 mg"],
        "description": "Common cold / ISPA ringan"
    },
    "K29": {
        "procedures": ["89.02", "45.13", "03.31"],
        "drugs": ["KFA004", "KFA012", "KFA023"],
        "vitamins": ["Multivitamin Adult", "Vitamin B1 100 mg"],
        "description": "Gastritis"
    },
    "E11": {
        "procedures": ["03.31", "90.59", "90.59A"],
        "drugs": ["KFA006", "KFA035", "KFA036"],
        "vitamins": ["Vitamin B Complex", "Folic Acid 1 mg"],
        "description": "Diabetes tipe 2"
    },
    "I10": {
        "procedures": ["89.14", "03.31", "90.59"],
        "drugs": ["KFA007", "KFA019", "KFA018"],
        "vitamins": ["Multivitamin Adult"],
        "description": "Hipertensi"
    },
    "J45": {
        "procedures": ["93.05", "96.04", "89.02"],
        "drugs": ["KFA021", "KFA010", "KFA026"],
        "vitamins": ["Vitamin C 500 mg", "Multivitamin Adult"],
        "description": "Asma"
    },
}

# ================================================================
# COST THRESHOLDS (INDONESIAN HEALTHCARE CONTEXT - BPJS)
# ================================================================
COST_THRESHOLDS = {
    "procedure": {
        "low": 50_000,        # Pemeriksaan rutin
        "medium": 150_000,    # Tindakan standar
        "high": 300_000,      # Tindakan kompleks
        "extreme": 1_000_000  # Tindakan advance/bedah
    },
    "drug": {
        "low": 10_000,        # Obat generik
        "medium": 50_000,     # Obat branded
        "high": 150_000,      # Obat mahal
        "extreme": 500_000    # Obat sangat mahal/rare
    },
    "vitamin": {
        "low": 5_000,         # Vitamin basic
        "medium": 30_000,     # Vitamin branded
        "high": 80_000,       # Suplemen mahal
        "extreme": 200_000    # Suplemen premium
    },
    "total_claim": {
        "normal": 300_000,        # Rawat jalan normal
        "suspicious": 1_000_000,  # Perlu review
        "extreme": 2_000_000      # Very high - manual review
    }
}

# ================================================================
# FRAUD PATTERNS DEFINITION (FOR EXPLAINABILITY)
# ================================================================
FRAUD_PATTERNS = {
    "procedure_mismatch": {
        "description": "Tindakan medis tidak sesuai dengan diagnosis",
        "severity": "HIGH",
        "examples": [
            "Endoskopi untuk common cold",
            "EKG untuk gastritis ringan",
            "X-Ray untuk ISPA tanpa indikasi"
        ],
        "recommendation": "Verifikasi justifikasi medis dengan dokter"
    },
    "drug_mismatch": {
        "description": "Obat yang diresepkan tidak sesuai diagnosis",
        "severity": "HIGH",
        "examples": [
            "Antibiotik kuat untuk flu biasa",
            "Insulin untuk non-diabetes",
            "Obat jantung untuk ISPA"
        ],
        "recommendation": "Cek guideline pengobatan BPJS"
    },
    "vitamin_mismatch": {
        "description": "Vitamin/suplemen tidak relevan dengan kondisi",
        "severity": "MEDIUM",
        "examples": [
            "Suplemen mahal untuk kondisi ringan",
            "Multivitamin tanpa indikasi medis",
            "Vitamin tidak sesuai guideline"
        ],
        "recommendation": "Evaluasi kebutuhan vitamin"
    },
    "upcoding": {
        "description": "Biaya tindakan dinaikkan tanpa justifikasi",
        "severity": "HIGH",
        "examples": [
            "Pemeriksaan rutin ditagih sebagai prosedur kompleks",
            "Obat generik ditagih harga branded",
            "Markup harga tidak wajar"
        ],
        "recommendation": "Bandingkan dengan tarif BPJS standar"
    },
    "unbundling": {
        "description": "Memecah satu prosedur menjadi beberapa tagihan",
        "severity": "HIGH",
        "examples": [
            "Satu pemeriksaan ditagih berkali-kali",
            "Paket tindakan dipecah per item",
            "Splitting charges tanpa justifikasi"
        ],
        "recommendation": "Review bundling sesuai ketentuan"
    },
    "unnecessary_treatment": {
        "description": "Pengobatan berlebihan yang tidak perlu",
        "severity": "MEDIUM",
        "examples": [
            "Vitamin mahal untuk kondisi ringan",
            "Obat tambahan tidak perlu",
            "Over-prescription tanpa indikasi"
        ],
        "recommendation": "Verifikasi clinical necessity"
    },
    "high_frequency": {
        "description": "Frekuensi klaim terlalu tinggi/mencurigakan",
        "severity": "MEDIUM",
        "examples": [
            "Klaim setiap hari untuk kondisi sama",
            "Multiple visits tanpa perubahan diagnosa",
            "Pola kunjungan tidak wajar"
        ],
        "recommendation": "Review pola kunjungan pasien"
    },
    "cost_anomaly": {
        "description": "Biaya sangat berbeda dari rata-rata diagnosis",
        "severity": "HIGH",
        "examples": [
            "Biaya ISPA > 1 juta rupiah",
            "Gastritis > 2 juta rupiah",
            "Outlier cost tanpa justifikasi"
        ],
        "recommendation": "Minta dokumentasi pendukung biaya"
    }
}

# ================================================================
# MODEL HYPERPARAMETERS (OPTIMIZED FOR FRAUD DETECTION)
# ================================================================
MODEL_HYPERPARAMETERS = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "eta": 0.03,                    # Learning rate
    "max_depth": 7,                 # Tree depth
    "subsample": 0.85,              # Row sampling
    "colsample_bytree": 0.85,       # Column sampling
    "min_child_weight": 3,          # Min samples in leaf
    "gamma": 0.3,                   # Min loss reduction
    "reg_alpha": 0.1,               # L1 regularization
    "reg_lambda": 1.0,              # L2 regularization
    "tree_method": "hist",          # Fast histogram method
    "seed": 42,
}

# Training parameters
TRAINING_CONFIG = {
    "test_size": 0.2,
    "random_state": 42,
    "max_samples": 300_000,         # Max training samples
    "smote_threshold": 0.25,        # Apply SMOTE if fraud < 25%
    "smote_strategy": 0.4,          # Target ratio after SMOTE
    "num_boost_round": 1000,
    "early_stopping_rounds": 50,
    "verbose_eval": 25,
}

# ================================================================
# VALIDATION RULES FOR BPJS CLAIMS
# ================================================================
VALIDATION_RULES = {
    "max_claim_amount": 5_000_000,      # Rp 5 juta
    "max_drug_items": 10,               # Max 10 jenis obat
    "max_vitamin_items": 5,             # Max 5 jenis vitamin
    "max_procedures": 5,                # Max 5 prosedur per kunjungan
    "suspicious_frequency": 10,         # Klaim per bulan
    "min_days_between_claims": 7,       # Minimal 1 minggu antar klaim
    
    # Specific diagnosis rules
    "common_cold_rules": {
        "max_cost": 300_000,
        "forbidden_procedures": ["45.13", "88.53", "89.14"],  # No endoskopi/EKG untuk flu
        "forbidden_drugs": ["KFA003", "KFA030", "KFA040"],    # No antibiotik kuat
        "max_vitamins": 2
    },
    "gastritis_rules": {
        "max_cost": 500_000,
        "max_procedures": 3,
    },
    "diabetes_rules": {
        "mandatory_procedures": ["90.59", "90.59A"],  # Harus ada cek gula darah
        "mandatory_drugs": ["KFA006"],                 # Harus ada metformin
    },
    "hypertension_rules": {
        "mandatory_procedures": ["89.14"],             # Harus ada EKG
        "mandatory_drugs": ["KFA007", "KFA019"],       # Harus ada antihipertensi
    }
}

# ================================================================
# RISK SCORING WEIGHTS (FOR FINAL FRAUD SCORE CALCULATION)
# ================================================================
RISK_WEIGHTS = {
    "clinical_mismatch": 0.40,      # 40% - Most important
    "cost_anomaly": 0.30,           # 30% - Second most important
    "frequency_risk": 0.15,         # 15% - Third
    "complexity_score": 0.15,       # 15% - Additional context
}

# ================================================================
# EXPLAINABILITY TEMPLATES (FOR REVIEWER UI)
# ================================================================
EXPLANATION_TEMPLATES = {
    "id": {
        "high_fraud": "🔴 RISIKO TINGGI: {reasons}. Rekomendasi: {recommendation}",
        "medium_fraud": "🟡 RISIKO SEDANG: {reasons}. Rekomendasi: {recommendation}",
        "low_fraud": "🟢 RISIKO RENDAH: {reasons}",
        "no_fraud": "✅ TIDAK ADA INDIKASI FRAUD: Klaim sesuai standar BPJS"
    },
    "en": {
        "high_fraud": "🔴 HIGH RISK: {reasons}. Recommendation: {recommendation}",
        "medium_fraud": "🟡 MODERATE RISK: {reasons}. Recommendation: {recommendation}",
        "low_fraud": "🟢 LOW RISK: {reasons}",
        "no_fraud": "✅ NO FRAUD INDICATORS: Claim meets BPJS standards"
    }
}

# ================================================================
# DEPLOYMENT SETTINGS
# ================================================================
DEPLOYMENT_CONFIG = {
    "model_name": "bpjs_fraud_detection",
    "model_version": "v2.0_iceberg",
    "model_description": "BPJS Fraud Detection - Integrated with Iceberg reference tables",
    "replicas": 2,
    "cpu": 2,
    "memory": 4,  # GB
    "environment_variables": {
        "ICEBERG_CONNECTION": ICEBERG_CONNECTION,
        "MODEL_VERSION": "v2.0",
        "LOG_LEVEL": "INFO"
    }
}

# Model artifacts paths
MODEL_ARTIFACTS = {
    "model_file": "model.json",
    "calibrator_file": "calibrator.pkl",
    "preprocess_file": "preprocess.pkl",
    "metadata_file": "meta.json",
    "summary_file": "training_summary.txt"
}

# ================================================================
# HELPER FUNCTIONS
# ================================================================

def get_compatible_items(icd10_code: str, item_type: str) -> list:
    """
    Get compatible items for a diagnosis from fallback rules.
    In production, this should query Iceberg reference tables.
    
    Args:
        icd10_code: ICD-10 diagnosis code
        item_type: 'procedures', 'drugs', or 'vitamins'
    
    Returns:
        List of compatible item codes/names
    """
    rules = COMPAT_RULES_FALLBACK.get(icd10_code, {})
    return rules.get(item_type, [])


def check_compatibility(icd10_code: str, item_code: str, item_type: str) -> bool:
    """
    Check if an item is compatible with diagnosis.
    
    Args:
        icd10_code: ICD-10 diagnosis code
        item_code: Item code/name to check
        item_type: 'procedures', 'drugs', or 'vitamins'
    
    Returns:
        True if compatible, False otherwise
    """
    allowed_items = get_compatible_items(icd10_code, item_type)
    return item_code in allowed_items


def get_fraud_pattern_description(pattern_type: str, language: str = "id") -> str:
    """
    Get human-readable description of fraud pattern.
    
    Args:
        pattern_type: Type of fraud pattern
        language: 'id' or 'en'
    
    Returns:
        Description string
    """
    pattern = FRAUD_PATTERNS.get(pattern_type, {})
    return pattern.get("description", "Unknown fraud pattern")


def get_diagnosis_rules(icd10_code: str) -> dict:
    """
    Get specific validation rules for a diagnosis.
    
    Args:
        icd10_code: ICD-10 diagnosis code
    
    Returns:
        Dictionary of validation rules
    """
    diagnosis_mapping = {
        "J06": VALIDATION_RULES["common_cold_rules"],
        "K29": VALIDATION_RULES["gastritis_rules"],
        "E11": VALIDATION_RULES["diabetes_rules"],
        "I10": VALIDATION_RULES["hypertension_rules"],
    }
    return diagnosis_mapping.get(icd10_code, {})


def load_reference_tables_from_iceberg(spark):
    """
    Load clinical reference tables from Iceberg.
    To be used in ETL and model inference.
    
    Args:
        spark: SparkSession
    
    Returns:
        Dictionary of DataFrames
    """
    ref_tables = {}
    
    for table_name, table_path in ICEBERG_REF_TABLES.items():
        try:
            ref_tables[table_name] = spark.sql(f"SELECT * FROM {table_path}")
            print(f"✓ Loaded {table_name}: {ref_tables[table_name].count()} records")
        except Exception as e:
            print(f"✗ Failed to load {table_name}: {e}")
            ref_tables[table_name] = None
    
    return ref_tables


def get_cost_threshold(item_type: str, level: str) -> int:
    """
    Get cost threshold for validation.
    
    Args:
        item_type: 'procedure', 'drug', 'vitamin', or 'total_claim'
        level: 'low', 'medium', 'high', 'extreme', 'normal', 'suspicious'
    
    Returns:
        Threshold amount in Rupiah
    """
    return COST_THRESHOLDS.get(item_type, {}).get(level, 0)


def calculate_fraud_score(features: dict) -> float:
    """
    Calculate weighted fraud score from features.
    
    Args:
        features: Dictionary of feature values
    
    Returns:
        Fraud score between 0 and 1
    """
    clinical_score = (
        features.get("procedure_mismatch_flag", 0) +
        features.get("drug_mismatch_flag", 0) +
        features.get("vitamin_mismatch_flag", 0)
    ) / 3.0
    
    cost_score = min(features.get("biaya_anomaly_score", 1) / 4.0, 1.0)
    
    freq_score = min(features.get("patient_frequency_risk", 0) / 20.0, 1.0)
    
    weighted_score = (
        clinical_score * RISK_WEIGHTS["clinical_mismatch"] +
        cost_score * RISK_WEIGHTS["cost_anomaly"] +
        freq_score * RISK_WEIGHTS["frequency_risk"]
    )
    
    return min(max(weighted_score, 0.0), 1.0)


# ================================================================
# RUNTIME CHECKS
# ================================================================

def validate_config():
    """Validate configuration at startup"""
    errors = []
    
    # Check feature consistency
    if len(set(ALL_FEATURES)) != len(ALL_FEATURES):
        errors.append("Duplicate features detected")
    
    # Check threshold values
    for item_type, thresholds in COST_THRESHOLDS.items():
        values = list(thresholds.values())
        if values != sorted(values):
            errors.append(f"Threshold values for {item_type} not in ascending order")
    
    # Check risk weights sum
    weight_sum = sum(RISK_WEIGHTS.values())
    if abs(weight_sum - 1.0) > 0.01:
        errors.append(f"Risk weights sum to {weight_sum}, should be 1.0")
    
    if errors:
        raise ValueError(f"Configuration validation failed: {errors}")
    
    return True


# Validate on import
try:
    validate_config()
    print("✓ Configuration validated successfully")
except Exception as e:
    print(f"⚠ Configuration validation warning: {e}")


# ================================================================
# EXPORT PUBLIC API
# ================================================================

__all__ = [
    # Tables
    'ICEBERG_CONNECTION',
    'ICEBERG_RAW_TABLES',
    'ICEBERG_REF_TABLES',
    'ICEBERG_CURATED_TABLES',
    
    # Features
    'NUMERIC_FEATURES',
    'CATEGORICAL_FEATURES',
    'ALL_FEATURES',
    'LABEL_COLUMN',
    
    # Rules
    'COMPAT_RULES_FALLBACK',
    'COST_THRESHOLDS',
    'FRAUD_PATTERNS',
    'VALIDATION_RULES',
    
    # Model
    'MODEL_HYPERPARAMETERS',
    'TRAINING_CONFIG',
    'RISK_WEIGHTS',
    
    # Deployment
    'DEPLOYMENT_CONFIG',
    'MODEL_ARTIFACTS',
    
    # Functions
    'get_compatible_items',
    'check_compatibility',
    'get_fraud_pattern_description',
    'get_diagnosis_rules',
    'load_reference_tables_from_iceberg',
    'get_cost_threshold',
    'calculate_fraud_score',
    'validate_config',
]