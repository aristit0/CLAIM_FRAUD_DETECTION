#!/usr/bin/env python3
import os
import sys
import json
import pickle
import pandas as pd
import numpy as np
import xgboost as xgb
from datetime import datetime

# ==============================================================================
# 1. GLOBAL INITIALIZATION
# ==============================================================================
# Global variables to hold model artifacts
model = None
encoder = None
calibrator = None
threshold = 0.5
features_num = []
features_cat = []

# Config Rules (Self-contained)
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

def init_model():
    """Load model artifacts into global variables"""
    global model, encoder, calibrator, threshold, features_num, features_cat
    
    # Try multiple paths for the artifact
    possible_paths = [
        "fraud_model_history_v1.pkl",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "fraud_model_history_v1.pkl"),
        "/home/cdsw/fraud_model_history_v1.pkl"
    ]
    
    artifact_path = None
    for p in possible_paths:
        if os.path.exists(p):
            artifact_path = p
            break
            
    if artifact_path is None:
        print("✗ CRITICAL: Model artifact 'fraud_model_history_v1.pkl' not found.")
        return False

    try:
        print(f"Loading artifacts from {artifact_path}...")
        with open(artifact_path, "rb") as f:
            artifacts = pickle.load(f)
            
        model = artifacts["model"]
        encoder = artifacts["encoder"]
        calibrator = artifacts["calibrator"]
        threshold = artifacts["threshold"]
        features_num = artifacts["features"]["numeric"]
        features_cat = artifacts["features"]["categorical"]
        
        print("✓ Model loaded successfully.")
        return True
    except Exception as e:
        print(f"✗ Failed to load artifacts: {e}")
        return False

# Initialize on startup
if not init_model():
    print("WARNING: Model initialization failed. Predictions will error out.")

# ==============================================================================
# 2. HELPER FUNCTIONS
# ==============================================================================
def calculate_match_score(dx, items, item_type):
    if not dx or not items: return 0.5
    
    dx_clean = str(dx).split('.')[0] if str(dx) not in COMPAT_RULES else str(dx)
    rule = COMPAT_RULES.get(dx_clean)
    if not rule: 
        rule = COMPAT_RULES.get(str(dx))
        if not rule: return 0.5 
    
    valid_items = rule.get(item_type, [])
    if not valid_items: return 0.5
    
    match_count = sum(1 for x in items if x in valid_items)
    if len(items) == 0: return 1.0
    return float(match_count) / len(items)

def explain_fraud(row, proba, th):
    reasons = []
    
    if row.get("diagnosis_drug_score", 1.0) < 0.5: reasons.append("Obat tidak sesuai diagnosis")
    if row.get("diagnosis_procedure_score", 1.0) < 0.5: reasons.append("Tindakan tidak sesuai diagnosis")
    if row.get("diagnosis_vitamin_score", 1.0) < 0.5: reasons.append("Vitamin tidak sesuai diagnosis")
    
    freq = row.get("patient_frequency_risk", 0)
    if freq > 3: reasons.append(f"Frekuensi kunjungan tinggi ({int(freq)}x)")
        
    anomaly = row.get("biaya_anomaly_score", 0)
    if anomaly >= 3: reasons.append("Biaya ekstrem tinggi")
    elif anomaly >= 2: reasons.append("Biaya di atas rata-rata")
        
    if not reasons and proba >= th:
        reasons.append("Pola risiko tinggi terdeteksi model")
        
    risk_level = "HIGH" if proba > 0.7 else "MODERATE" if proba > th else "LOW"
    return f"[{risk_level}] " + "; ".join(reasons), risk_level

# ==============================================================================
# 3. PREDICT FUNCTION
# ==============================================================================
def predict(args):
    start_time = datetime.now()
    
    # Global check
    if model is None:
        return {"status": "error", "message": "Model not loaded. Check deployment logs."}

    try:
        # Flexible input handling
        if "request" in args:
            input_data = args["request"].get("data", {})
        else:
            input_data = args.get("data", {})
            
        if not input_data:
            return {"status": "error", "message": "Empty input data"}

        # 1. Feature Eng (On-the-fly)
        dx = str(input_data.get("icd10_primary_code", "UNKNOWN"))
        proc_codes = input_data.get("procedures_icd9_codes", [])
        drug_codes = input_data.get("drug_codes", [])
        vit_names  = input_data.get("vitamin_names", [])
        
        # Ensure list type
        if isinstance(proc_codes, str): proc_codes = [proc_codes]
        if isinstance(drug_codes, str): drug_codes = [drug_codes]
        if isinstance(vit_names, str): vit_names = [vit_names]
        
        score_proc = calculate_match_score(dx, proc_codes, "procedures")
        score_drug = calculate_match_score(dx, drug_codes, "drugs")
        score_vit  = calculate_match_score(dx, vit_names, "vitamins")
        
        mismatch_count = (1 if score_proc < 0.5 else 0) + \
                         (1 if score_drug < 0.5 else 0) + \
                         (1 if score_vit < 0.5 else 0)
        
        # 2. Build DataFrame Row
        row_dict = {
            "total_claim_amount": float(input_data.get("total_claim_amount", 0)),
            "total_procedure_cost": float(input_data.get("total_procedure_cost", 0)),
            "total_drug_cost": float(input_data.get("total_drug_cost", 0)),
            "total_vitamin_cost": float(input_data.get("total_vitamin_cost", 0)),
            "diagnosis_procedure_score": score_proc,
            "diagnosis_drug_score": score_drug,
            "diagnosis_vitamin_score": score_vit,
            "patient_frequency_risk": float(input_data.get("patient_frequency_risk", 0)),
            "patient_amount_last_30d": float(input_data.get("patient_amount_last_30d", 0)),
            "days_since_last_visit": float(input_data.get("days_since_last_visit", 999)),
            "biaya_anomaly_score": float(input_data.get("biaya_anomaly_score", 1)),
            "mismatch_count": int(mismatch_count),
            "department": str(input_data.get("department", "UNKNOWN")),
            "icd10_primary_code": dx,
            "doctor_name": str(input_data.get("doctor_name", "UNKNOWN"))
        }
        
        df_input = pd.DataFrame([row_dict])
        
        # 3. Transform & Predict
        # Use only columns expected by the model to avoid mismatches
        final_cols = features_num + features_cat
        
        # Pre-fill missing categorical cols if any (safety net)
        for c in features_cat:
            if c not in df_input.columns: df_input[c] = "UNKNOWN"
            
        df_encoded = encoder.transform(df_input)
        
        # Ensure column order matches training
        df_ready = df_encoded[final_cols]
        
        raw_prob = model.predict_proba(df_ready)[0][1]
        final_prob = calibrator.predict([raw_prob])[0]
        is_fraud = int(final_prob >= threshold)
        
        expl_str, risk_lvl = explain_fraud(row_dict, final_prob, threshold)
        
        return {
            "status": "success",
            "claim_id": input_data.get("claim_id", "UNKNOWN"),
            "prediction": {
                "fraud_score": float(round(final_prob, 4)),
                "is_fraud": is_fraud,
                "risk_level": risk_lvl,
                "explanation": expl_str
            },
            "debug": {
                "raw_prob": float(round(raw_prob, 4)),
                "mismatches": int(mismatch_count)
            },
            "processing_time_ms": (datetime.now() - start_time).microseconds / 1000
        }

    except Exception as e:
        # Catch-all to ensure we return JSON, not crash the container
        return {"status": "error", "message": str(e)}
