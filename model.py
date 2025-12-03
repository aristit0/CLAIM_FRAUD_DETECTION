#!/usr/bin/env python3
import os
import json
import pickle
import pandas as pd
import numpy as np
import xgboost as xgb
from datetime import datetime

# ==============================================================================
# 1. CONFIG & RULES (Single Source of Truth untuk Inference)
# ==============================================================================
# Disalin dari config.py agar model script mandiri (self-contained) di deployment
COMPAT_RULES = {
    "E11": { 
        "procedures": ["03.31", "90.59", "90.59A"],
        "drugs": ["KFA006", "KFA035", "KFA036"],
        "vitamins": ["Vitamin B Complex", "Vitamin D 1000 IU", "Magnesium 250 mg"]
    },
    "I10": {
        "procedures": ["03.31", "89.14", "89.02"],
        "drugs": ["KFA007", "KFA019", "KFA018"],
        "vitamins": ["Vitamin D 1000 IU", "Vitamin B Complex"]
    },
    "J06": {
        "procedures": ["89.02", "96.70"],
        "drugs": ["KFA001", "KFA009", "KFA031"],
        "vitamins": ["Vitamin C 500 mg", "Zinc 20 mg"]
    },
    "A09": {
        "procedures": ["03.31", "99.15"],
        "drugs": ["KFA005", "KFA024", "KFA038"],
        "vitamins": ["Zinc 20 mg", "Probiotic Complex"]
    },
    "K29": {
        "procedures": ["45.13", "03.31"],
        "drugs": ["KFA004", "KFA023", "KFA012"],
        "vitamins": ["Vitamin E 400 IU"]
    },
    # Fallback/Default rules untuk diagnosis umum lainnya bisa ditambahkan di sini
    "J45": {
        "procedures": ["96.04", "93.05", "87.03"],
        "drugs": ["KFA010", "KFA011", "KFA021"],
        "vitamins": ["Vitamin D 1000 IU", "Vitamin C 500 mg"]
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
    "J18": {
        "procedures": ["87.03", "03.31", "99.15"],
        "drugs": ["KFA003", "KFA014", "KFA030", "KFA040"],
        "vitamins": ["Vitamin C 1000 mg", "Vitamin D3 2000 IU"]
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

# ==============================================================================
# 2. LOAD ARTIFACTS
# ==============================================================================
ARTIFACT_PATH = "fraud_model_history_v1.pkl"
artifacts = {}

print(f"Loading model artifacts from {ARTIFACT_PATH}...")
try:
    with open(ARTIFACT_PATH, "rb") as f:
        artifacts = pickle.load(f)
    print("✓ Model artifacts loaded successfully.")
    
    # Extract components
    model = artifacts["model"]
    encoder = artifacts["encoder"]
    calibrator = artifacts["calibrator"]
    threshold = artifacts["threshold"]
    features_num = artifacts["features"]["numeric"]
    features_cat = artifacts["features"]["categorical"]
    
except Exception as e:
    print(f"CRITICAL ERROR: Failed to load model artifacts. {e}")
    # Initialize dummy objects to prevent immediate crash, though prediction will fail
    model = None

# ==============================================================================
# 3. HELPER FUNCTIONS
# ==============================================================================
def calculate_match_score(dx, items, item_type):
    """Menghitung skor kecocokan klinis (0.0 - 1.0)"""
    if not dx or not items: 
        return 0.5 # Neutral
    
    # Clean up diagnosis code (remove version/dot if necessary, depends on config)
    dx_clean = str(dx).split('.')[0] if str(dx) not in COMPAT_RULES else str(dx)
    
    rule = COMPAT_RULES.get(dx_clean)
    if not rule: 
        # Coba exact match jika split gagal
        rule = COMPAT_RULES.get(str(dx))
        if not rule:
            return 0.5 # Unknown diagnosis
    
    valid_items = rule.get(item_type, [])
    if not valid_items: 
        return 0.5
    
    # Hitung rasio item yang valid
    match_count = sum(1 for x in items if x in valid_items)
    
    if len(items) == 0:
        return 1.0
        
    return float(match_count) / len(items)

def explain_fraud(row, proba, threshold):
    """Membuat penjelasan sederhana kenapa klaim ini dianggap fraud"""
    reasons = []
    
    # Cek Konsistensi Klinis
    if row.get("diagnosis_drug_score", 1.0) < 0.5: reasons.append("Obat tidak sesuai diagnosis")
    if row.get("diagnosis_procedure_score", 1.0) < 0.5: reasons.append("Tindakan tidak sesuai diagnosis")
    if row.get("diagnosis_vitamin_score", 1.0) < 0.5: reasons.append("Vitamin tidak sesuai diagnosis")
    
    # Cek History (Frequency)
    freq = row.get("patient_frequency_risk", 0)
    if freq > 3: 
        reasons.append(f"Frekuensi kunjungan tinggi ({int(freq)}x dalam 30 hari)")
        
    # Cek Anomali Biaya
    anomaly = row.get("biaya_anomaly_score", 0)
    if anomaly >= 3:
        reasons.append("Biaya EKSTREM di atas rata-rata diagnosis (Level 3/4)")
    elif anomaly >= 2:
        reasons.append("Biaya tinggi di atas rata-rata diagnosis (Level 2)")
        
    # Cek Mismatch Count
    mismatch = row.get("mismatch_count", 0)
    if mismatch > 0:
        reasons.append(f"Terdapat {int(mismatch)} ketidakcocokan klinis")

    # Fallback explanation
    if not reasons and proba >= threshold:
        reasons.append("Pola risiko tinggi terdeteksi oleh model (kombinasi faktor)")
        
    risk_level = "HIGH RISK" if proba > 0.7 else "MODERATE RISK" if proba > threshold else "LOW RISK"
    
    explanation_str = f"[{risk_level}] " + "; ".join(reasons) if reasons else f"[{risk_level}] Tidak ada indikator fraud spesifik."
    return explanation_str, risk_level

# ==============================================================================
# 4. PREDICT FUNCTION (CML Entry Point)
# ==============================================================================
def predict(args):
    """
    Fungsi utama yang dipanggil oleh CML saat ada request API.
    Args:
        args (dict): Payload JSON input. 
                     Structure:
                     {
                        "request": {
                            "data": { ... claim features ... }
                        }
                     }
                     OR directly:
                     {
                        "data": { ... }
                     }
    """
    start_time = datetime.now()
    
    try:
        # Normalize Input (Handle different JSON structures)
        if "request" in args:
            input_data = args["request"].get("data", {})
        else:
            input_data = args.get("data", {})
            
        if not input_data:
            return {"status": "error", "message": "No data provided in request"}

        # ---------------------------------------------------------
        # A. FEATURE ENGINEERING (ON-THE-FLY)
        # ---------------------------------------------------------
        dx = str(input_data.get("icd10_primary_code", "UNKNOWN"))
        
        # 1. Hitung Clinical Scores (Real-time)
        # Note: Input diharapkan berupa list/array untuk codes/names
        proc_codes = input_data.get("procedures_icd9_codes", [])
        drug_codes = input_data.get("drug_codes", [])
        vit_names  = input_data.get("vitamin_names", [])
        
        # Handle jika input string tunggal (bukan list)
        if isinstance(proc_codes, str): proc_codes = [proc_codes]
        if isinstance(drug_codes, str): drug_codes = [drug_codes]
        if isinstance(vit_names, str): vit_names = [vit_names]
        
        score_proc = calculate_match_score(dx, proc_codes, "procedures")
        score_drug = calculate_match_score(dx, drug_codes, "drugs")
        score_vit  = calculate_match_score(dx, vit_names, "vitamins")
        
        # 2. Hitung Mismatch Flags
        flag_proc = 1 if score_proc < 0.5 else 0
        flag_drug = 1 if score_drug < 0.5 else 0
        flag_vit  = 1 if score_vit < 0.5 else 0
        mismatch_count = flag_proc + flag_drug + flag_vit
        
        # 3. Construct DataFrame Row (Sesuai Training Columns)
        # Pastikan nama kolom SAMA PERSIS dengan NUMERIC_FEATURES + CATEGORICAL_FEATURES di training
        row_dict = {
            # Numeric Features
            "total_claim_amount": float(input_data.get("total_claim_amount", 0)),
            "total_procedure_cost": float(input_data.get("total_procedure_cost", 0)),
            "total_drug_cost": float(input_data.get("total_drug_cost", 0)),
            "total_vitamin_cost": float(input_data.get("total_vitamin_cost", 0)),
            
            "diagnosis_procedure_score": score_proc,
            "diagnosis_drug_score": score_drug,
            "diagnosis_vitamin_score": score_vit,
            
            "patient_frequency_risk": float(input_data.get("patient_frequency_risk", 0)),
            "patient_amount_last_30d": float(input_data.get("patient_amount_last_30d", 0)),
            "days_since_last_visit": float(input_data.get("days_since_last_visit", 999)), # Default 999 for new patient
            
            "biaya_anomaly_score": float(input_data.get("biaya_anomaly_score", 1)),
            "mismatch_count": int(mismatch_count),
            
            # Categorical Features
            "department": str(input_data.get("department", "UNKNOWN")),
            "icd10_primary_code": dx,
            "doctor_name": str(input_data.get("doctor_name", "UNKNOWN"))
        }
        
        df_input = pd.DataFrame([row_dict])
        
        # ---------------------------------------------------------
        # B. PREPROCESSING (ENCODING)
        # ---------------------------------------------------------
        # Terapkan TargetEncoder
        df_encoded = encoder.transform(df_input)
        
        # Select columns in correct order
        # Pastikan kolom-kolom ini ada di df_encoded
        final_features = features_num + features_cat
        df_ready = df_encoded[final_features]
        
        # ---------------------------------------------------------
        # C. PREDICTION & CALIBRATION
        # ---------------------------------------------------------
        if model is None:
            raise RuntimeError("Model not loaded correctly")
            
        # Prediksi raw (probability dari XGBoost)
        raw_prob = model.predict_proba(df_ready)[0][1]
        
        # Kalibrasi (Isotonic)
        final_prob = calibrator.predict([raw_prob])[0]
        
        # Keputusan
        is_fraud = int(final_prob >= threshold)
        
        # ---------------------------------------------------------
        # D. RESPONSE FORMATTING
        # ---------------------------------------------------------
        explanation_str, risk_level = explain_fraud(row_dict, final_prob, threshold)
        
        response = {
            "status": "success",
            "claim_id": input_data.get("claim_id", "UNKNOWN"),
            "prediction": {
                "fraud_score": float(round(final_prob, 4)),
                "is_fraud": is_fraud,
                "risk_level": risk_level,
                "explanation": explanation_str
            },
            "features_used": row_dict,
            "processing_time_ms": (datetime.now() - start_time).microseconds / 1000
        }
        
        return response

    except Exception as e:
        import traceback
        return {
            "status": "error",
            "message": str(e),
            "traceback": traceback.format_exc()
        }

# Tes Lokal (Optional - For debugging purposes)
if __name__ == "__main__":
    test_payload = {
        "data": {
            "claim_id": "TEST_INVOKER",
            "total_claim_amount": 5000000,
            "total_procedure_cost": 4000000,
            "total_drug_cost": 500000,
            "total_vitamin_cost": 500000,
            "patient_frequency_risk": 5,
            "patient_amount_last_30d": 10000000,
            "days_since_last_visit": 2,
            "biaya_anomaly_score": 4,
            "department": "Poli Umum",
            "icd10_primary_code": "J06", # Flu
            "doctor_name": "Dr. Strange",
            "procedures_icd9_codes": ["99.04"], # Transfusi (Mismatch with Flu)
            "drug_codes": ["KFA003"], # Antibiotic keras (Mismatch)
            "vitamin_names": ["Vitamin A 5000 IU"] # Mismatch
        }
    }
    print(json.dumps(predict(test_payload), indent=2))