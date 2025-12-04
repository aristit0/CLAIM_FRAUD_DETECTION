#!/usr/bin/env python3
"""
BPJS Fraud Detection Model - Lightweight Production Version
Optimized for CML deployment
"""

import json
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
import cml.models_v1 as models
from datetime import datetime
import sys
import os

# ================================================================
# GLOBAL VARIABLES & MODEL LOADING
# ================================================================
MODEL_VERSION = "v2.0_lite"
MODEL_NAME = "BPJS Fraud Detection"

print("=" * 80)
print(f"{MODEL_NAME} - Loading...")
print("=" * 80)

# Load artifacts
try:
    booster = xgb.Booster()
    booster.load_model("model.json")
    
    with open("calibrator.pkl", "rb") as f:
        calibrator = pickle.load(f)
    
    with open("preprocess.pkl", "rb") as f:
        preprocess = pickle.load(f)
    
    with open("meta.json", "r") as f:
        model_meta = json.load(f)
    
    numeric_cols = preprocess["numeric_cols"]
    categorical_cols = preprocess["categorical_cols"]
    encoders = preprocess["encoders"]
    best_threshold = preprocess["best_threshold"]
    feature_names = numeric_cols + categorical_cols
    
    print(f"✓ Model loaded - Threshold: {best_threshold:.3f}")
    print("=" * 80)
    
except Exception as e:
    print(f"✗ Error loading model: {e}")
    raise

# Clinical rules - compact version
RULES = {
    "J06": {"proc": ["89.02", "96.70"], "drug": ["KFA001", "KFA009", "KFA031"], "vit": ["Vitamin C 500 mg", "Zinc 20 mg"]},
    "K29": {"proc": ["45.13", "03.31", "89.02"], "drug": ["KFA004", "KFA012", "KFA023"], "vit": ["Multivitamin Adult", "Vitamin B1 100 mg"]},
    "E11": {"proc": ["03.31", "90.59", "90.59A"], "drug": ["KFA006", "KFA035", "KFA036"], "vit": ["Vitamin B Complex", "Folic Acid 1 mg"]},
    "I10": {"proc": ["89.14", "03.31", "90.59"], "drug": ["KFA007", "KFA019", "KFA018"], "vit": ["Multivitamin Adult"]},
    "J45": {"proc": ["93.05", "96.04", "89.02"], "drug": ["KFA021", "KFA010", "KFA026"], "vit": ["Vitamin C 500 mg", "Multivitamin Adult"]}
}

# ================================================================
# HELPER FUNCTIONS
# ================================================================

def compute_age(dob, visit_date):
    try:
        dob_dt = datetime.strptime(str(dob), "%Y-%m-%d").date()
        visit_dt = datetime.strptime(str(visit_date), "%Y-%m-%d").date()
        age = visit_dt.year - dob_dt.year - ((visit_dt.month, visit_dt.day) < (dob_dt.month, dob_dt.day))
        return max(age, 0)
    except:
        return 0


def calc_compat(icd10, items, item_type):
    """Calculate compatibility score"""
    rules = RULES.get(icd10)
    if not rules or not items:
        return 0.5
    
    allowed = rules.get(item_type, [])
    if not allowed:
        return 0.5
    
    matches = sum(1 for x in items if x in allowed)
    return matches / len(items) if len(items) > 0 else 0.5


def build_features(claim):
    """Build features from claim"""
    dt = datetime.strptime(claim["visit_date"], "%Y-%m-%d").date()
    
    # Ensure lists
    proc = claim.get("procedures", [])
    drug = claim.get("drugs", [])
    vit = claim.get("vitamins", [])
    
    if not isinstance(proc, list): proc = [proc] if proc else []
    if not isinstance(drug, list): drug = [drug] if drug else []
    if not isinstance(vit, list): vit = [vit] if vit else []
    
    icd10 = claim.get("icd10_primary_code", "UNKNOWN")
    
    # Compatibility scores
    proc_score = calc_compat(icd10, proc, "proc")
    drug_score = calc_compat(icd10, drug, "drug")
    vit_score = calc_compat(icd10, vit, "vit")
    
    # Mismatch flags
    proc_flag = 1 if proc_score < 0.5 else 0
    drug_flag = 1 if drug_score < 0.5 else 0
    vit_flag = 1 if vit_score < 0.5 else 0
    
    # Cost anomaly
    total = float(claim.get("total_claim_amount", 0))
    if total > 1_500_000: cost_anom = 4
    elif total > 1_000_000: cost_anom = 3
    elif total > 500_000: cost_anom = 2
    else: cost_anom = 1
    
    return {
        "claim_id": claim.get("claim_id"),
        "patient_age": compute_age(claim.get("patient_dob"), claim["visit_date"]),
        "total_procedure_cost": float(claim.get("total_procedure_cost", 0)),
        "total_drug_cost": float(claim.get("total_drug_cost", 0)),
        "total_vitamin_cost": float(claim.get("total_vitamin_cost", 0)),
        "total_claim_amount": total,
        "biaya_anomaly_score": cost_anom,
        "patient_frequency_risk": claim.get("patient_frequency_risk", 2),
        "visit_year": dt.year,
        "visit_month": dt.month,
        "visit_day": dt.day,
        "diagnosis_procedure_score": proc_score,
        "diagnosis_drug_score": drug_score,
        "diagnosis_vitamin_score": vit_score,
        "procedure_mismatch_flag": proc_flag,
        "drug_mismatch_flag": drug_flag,
        "vitamin_mismatch_flag": vit_flag,
        "mismatch_count": proc_flag + drug_flag + vit_flag,
        "visit_type": claim.get("visit_type", "UNKNOWN"),
        "department": claim.get("department", "UNKNOWN"),
        "icd10_primary_code": icd10,
    }


def explain(row, score):
    """Generate explanation"""
    reasons = []
    
    if row["mismatch_count"] > 0:
        items = []
        if row["procedure_mismatch_flag"] == 1: items.append("tindakan")
        if row["drug_mismatch_flag"] == 1: items.append("obat")
        if row["vitamin_mismatch_flag"] == 1: items.append("vitamin")
        reasons.append(f"Ketidaksesuaian: {', '.join(items)}")
    
    if row["biaya_anomaly_score"] >= 3:
        reasons.append(f"Biaya tinggi (Rp {row['total_claim_amount']:,.0f})")
    
    if row["patient_frequency_risk"] > 10:
        reasons.append(f"Frekuensi tinggi ({row['patient_frequency_risk']}x)")
    
    if score > 0.8: level = "🔴 TINGGI"
    elif score > 0.5: level = "🟡 SEDANG"
    elif score > 0.3: level = "🟠 RENDAH"
    else: level = "🟢 MINIMAL"
    
    return f"{level}: " + ("; ".join(reasons) if reasons else "Normal")


def recommend(score, mismatch):
    """Generate recommendation"""
    if score > 0.8: return "🚫 Decline atau minta dokumen lengkap"
    elif score > 0.5: return "⚠️ Verifikasi manual diperlukan"
    elif score > 0.3: return "📋 Quick review"
    else: return "✅ Approve"


# ================================================================
# MAIN PREDICTION ENDPOINT
# ================================================================

@models.cml_model
def predict(data):
    """Main prediction endpoint"""
    
    # Parse input
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except:
            return {"status": "error", "error": "Invalid JSON"}
    
    # Validate
    if "claims" not in data or not isinstance(data["claims"], list) or len(data["claims"]) == 0:
        return {"status": "error", "error": "Missing or empty 'claims' field"}
    
    try:
        results = []
        
        for claim in data["claims"]:
            # Build features
            feat = build_features(claim)
            claim_id = feat.pop("claim_id")
            
            # Create DataFrame
            df = pd.DataFrame([feat])
            
            # Encode categoricals
            for col in categorical_cols:
                if col in df.columns:
                    df[col] = df[col].astype(str).fillna("UNKNOWN")
                    df[col] = encoders[col].transform(df[[col]])[col]
            
            # Clean numerics
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
                    df[col].replace([np.inf, -np.inf], 0, inplace=True)
            
            # Predict
            X = df[feature_names]
            dmatrix = xgb.DMatrix(X, feature_names=feature_names)
            
            y_raw = booster.predict(dmatrix)
            y_cal = calibrator.predict(y_raw)
            fraud_score = float(y_cal[0])
            fraud_flag = int(fraud_score >= best_threshold)
            
            # Risk level
            if fraud_score > 0.8: risk = "HIGH"; color = "red"
            elif fraud_score > 0.5: risk = "MODERATE"; color = "orange"
            elif fraud_score > 0.3: risk = "LOW"; color = "yellow"
            else: risk = "MINIMAL"; color = "green"
            
            results.append({
                "claim_id": claim_id,
                "fraud_score": round(fraud_score, 4),
                "fraud_probability": f"{fraud_score * 100:.1f}%",
                "fraud_flag": fraud_flag,
                "risk_level": risk,
                "risk_color": color,
                "explanation": explain(feat, fraud_score),
                "recommendation": recommend(fraud_score, feat["mismatch_count"]),
                "features": {
                    "mismatch_count": int(feat["mismatch_count"]),
                    "cost_anomaly": int(feat["biaya_anomaly_score"]),
                    "total_claim": float(feat["total_claim_amount"])
                }
            })
        
        return {
            "status": "success",
            "model_version": MODEL_VERSION,
            "timestamp": datetime.now().isoformat(),
            "total_claims": len(results),
            "fraud_detected": sum(1 for r in results if r["fraud_flag"] == 1),
            "results": results
        }
    
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


@models.cml_model
def health_check(data=None):
    """Health check"""
    return {
        "status": "healthy",
        "model": MODEL_NAME,
        "version": MODEL_VERSION,
        "timestamp": datetime.now().isoformat(),
        "threshold": best_threshold
    }


if __name__ == "__main__":
    print(f"\n✓ {MODEL_NAME} Ready for CML Deployment")
    print("  Endpoints: predict(), health_check()")
    print("=" * 80)