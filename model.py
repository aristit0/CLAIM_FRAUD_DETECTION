#!/usr/bin/env python3
"""
BPJS Fraud Detection Model - Production Deployment (v2.0)
Consistent dengan training ETL features dan config.py
"""

import json
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
import cml.models_v1 as models
from datetime import datetime
from typing import Dict, List, Any
import sys
import os

# Import config
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    NUMERIC_FEATURES, CATEGORICAL_FEATURES, 
    COMPAT_RULES_FALLBACK, COST_THRESHOLDS,
    get_fraud_pattern_description, get_diagnosis_rules
)

print("=" * 80)
print("BPJS FRAUD DETECTION MODEL v2.0 - LOADING")
print("=" * 80)

# ================================================================
# LOAD MODEL ARTIFACTS
# ================================================================
MODEL_JSON = "model.json"
CALIB_FILE = "calibrator.pkl"
PREPROCESS_FILE = "preprocess.pkl"
META_FILE = "meta.json"

try:
    # 1. Load XGBoost Model
    booster = xgb.Booster()
    booster.load_model(MODEL_JSON)
    print(f"✓ Model loaded: {MODEL_JSON}")
    
    # 2. Load Calibrator
    with open(CALIB_FILE, "rb") as f:
        calibrator = pickle.load(f)
    print(f"✓ Calibrator loaded: {CALIB_FILE}")
    
    # 3. Load Preprocessing Config
    with open(PREPROCESS_FILE, "rb") as f:
        preprocess = pickle.load(f)
    print(f"✓ Preprocessing config loaded: {PREPROCESS_FILE}")
    
    # 4. Load Metadata
    with open(META_FILE, "r") as f:
        model_meta = json.load(f)
    print(f"✓ Metadata loaded: {META_FILE}")
    
    # Extract configuration
    numeric_cols = preprocess["numeric_cols"]
    categorical_cols = preprocess["categorical_cols"]
    encoders = preprocess["encoders"]
    best_threshold = preprocess["best_threshold"]
    feature_importance_map = preprocess.get("feature_importance", {})
    
    feature_names = numeric_cols + categorical_cols
    
    # Global feature importance
    GLOBAL_FEATURE_IMPORTANCE = [
        {"feature": k, "importance": float(v)}
        for k, v in sorted(feature_importance_map.items(), 
                          key=lambda kv: kv[1], reverse=True)
    ]
    
    print(f"\n📊 Model Information:")
    print(f"  Version: {model_meta.get('model_version', 'unknown')}")
    print(f"  Training Date: {model_meta.get('training_date', 'unknown')}")
    print(f"  AUC: {model_meta.get('performance', {}).get('auc', 0):.4f}")
    print(f"  Threshold: {best_threshold:.3f}")
    print(f"  Features: {len(feature_names)} ({len(numeric_cols)} numeric, {len(categorical_cols)} categorical)")
    
except Exception as e:
    print(f"✗ Error loading artifacts: {e}")
    raise

print(f"\n✓ Model ready for inference")
print("=" * 80)

# ================================================================
# FEATURE ENGINEERING FUNCTIONS (MUST MATCH TRAINING!)
# ================================================================

def compute_age(dob: str, visit_date: str) -> int:
    """Calculate patient age at visit"""
    try:
        dob_dt = datetime.strptime(str(dob), "%Y-%m-%d").date()
        visit_dt = datetime.strptime(str(visit_date), "%Y-%m-%d").date()
        age = visit_dt.year - dob_dt.year - (
            (visit_dt.month, visit_dt.day) < (dob_dt.month, dob_dt.day)
        )
        return max(age, 0)
    except:
        return 0


def compute_compatibility_scores(icd10: str, procedures: List[str], 
                                 drugs: List[str], vitamins: List[str]) -> Dict[str, float]:
    """Calculate clinical compatibility scores (from config.py COMPAT_RULES)"""
    rules = COMPAT_RULES_FALLBACK.get(icd10, {})
    
    # Default scores (0.5 means neutral/unknown)
    proc_score = 0.5
    drug_score = 0.5
    vit_score = 0.5
    
    if rules:
        allowed_procs = rules.get("procedures", [])
        allowed_drugs = rules.get("drugs", [])
        allowed_vits = rules.get("vitamins", [])
        
        if procedures and allowed_procs:
            matches = sum(1 for p in procedures if p in allowed_procs)
            proc_score = matches / len(procedures)
        
        if drugs and allowed_drugs:
            matches = sum(1 for d in drugs if d in allowed_drugs)
            drug_score = matches / len(drugs)
        
        if vitamins and allowed_vits:
            matches = sum(1 for v in vitamins if v in allowed_vits)
            vit_score = matches / len(vitamins)
    
    return {
        "diagnosis_procedure_score": float(proc_score),
        "diagnosis_drug_score": float(drug_score),
        "diagnosis_vitamin_score": float(vit_score)
    }


def compute_mismatch_flags(compatibility_scores: Dict[str, float]) -> Dict[str, int]:
    """Convert compatibility scores to mismatch flags"""
    proc_flag = 1 if compatibility_scores["diagnosis_procedure_score"] < 0.3 else 0
    drug_flag = 1 if compatibility_scores["diagnosis_drug_score"] < 0.3 else 0
    vit_flag = 1 if compatibility_scores["diagnosis_vitamin_score"] < 0.3 else 0
    
    return {
        "procedure_mismatch_flag": proc_flag,
        "drug_mismatch_flag": drug_flag,
        "vitamin_mismatch_flag": vit_flag,
        "mismatch_count": proc_flag + drug_flag + vit_flag
    }


def compute_cost_anomaly_score(total_claim: float, icd10: str = None) -> int:
    """Calculate cost anomaly score (1-4 scale)"""
    thresholds = COST_THRESHOLDS.get("total_claim", {})
    
    if total_claim > thresholds.get("extreme", 2_000_000):
        return 4
    elif total_claim > thresholds.get("suspicious", 1_000_000):
        return 3
    elif total_claim > thresholds.get("normal", 300_000):
        return 2
    else:
        return 1


def compute_patient_frequency_risk(patient_history: Dict = None) -> int:
    """Compute frequency risk (simplified for inference)"""
    # In production, this would query patient history
    # For now, return dummy value
    return 2


# ================================================================
# MAIN FEATURE BUILDER
# ================================================================

def build_features_from_raw(raw: Dict[str, Any]) -> tuple:
    """
    Transform raw claim data into model features.
    CRITICAL: MUST match training ETL exactly!
    """
    # Ensure lists exist
    procedures = raw.get("procedures", [])
    drugs = raw.get("drugs", [])
    vitamins = raw.get("vitamins", [])
    
    if not isinstance(procedures, list):
        procedures = [procedures] if procedures else []
    if not isinstance(drugs, list):
        drugs = [drugs] if drugs else []
    if not isinstance(vitamins, list):
        vitamins = [vitamins] if vitamins else []
    
    # Extract basic info
    claim_id = raw.get("claim_id", "UNKNOWN")
    icd10 = raw.get("icd10_primary_code", "UNKNOWN")
    
    # Calculate compatibility scores
    compatibility = compute_compatibility_scores(icd10, procedures, drugs, vitamins)
    
    # Calculate mismatch flags
    mismatch = compute_mismatch_flags(compatibility)
    
    # Calculate other features
    total_proc = float(raw.get("total_procedure_cost", 0))
    total_drug = float(raw.get("total_drug_cost", 0))
    total_vit = float(raw.get("total_vitamin_cost", 0))
    total_claim = float(raw.get("total_claim_amount", 0))
    
    visit_date = raw.get("visit_date", "2000-01-01")
    dt = datetime.strptime(visit_date, "%Y-%m-%d")
    
    # Build feature dictionary (MUST match NUMERIC_FEATURES + CATEGORICAL_FEATURES)
    feature_row = {
        # Patient demographics
        "patient_age": compute_age(raw.get("patient_dob", "2000-01-01"), visit_date),
        
        # Temporal features
        "visit_year": dt.year,
        "visit_month": dt.month,
        "visit_day": dt.day,
        
        # Cost features
        "total_procedure_cost": total_proc,
        "total_drug_cost": total_drug,
        "total_vitamin_cost": total_vit,
        "total_claim_amount": total_claim,
        
        # Clinical compatibility scores
        "diagnosis_procedure_score": compatibility["diagnosis_procedure_score"],
        "diagnosis_drug_score": compatibility["diagnosis_drug_score"],
        "diagnosis_vitamin_score": compatibility["diagnosis_vitamin_score"],
        
        # Mismatch flags
        "procedure_mismatch_flag": mismatch["procedure_mismatch_flag"],
        "drug_mismatch_flag": mismatch["drug_mismatch_flag"],
        "vitamin_mismatch_flag": mismatch["vitamin_mismatch_flag"],
        "mismatch_count": mismatch["mismatch_count"],
        
        # Risk scores
        "biaya_anomaly_score": compute_cost_anomaly_score(total_claim, icd10),
        "patient_frequency_risk": compute_patient_frequency_risk(),
        
        # Categorical features
        "visit_type": raw.get("visit_type", "UNKNOWN"),
        "department": raw.get("department", "UNKNOWN"),
        "icd10_primary_code": icd10,
    }
    
    return claim_id, feature_row, compatibility, mismatch


# ================================================================
# PREPROCESSING PIPELINE
# ================================================================

def preprocess_features(records: List[Dict[str, Any]]) -> xgb.DMatrix:
    """Apply preprocessing pipeline (encoding, cleaning)"""
    df = pd.DataFrame.from_records(records)
    
    # Ensure all expected columns exist
    for col in numeric_cols + categorical_cols:
        if col not in df.columns:
            df[col] = None
    
    # 1. Encode categorical features (using trained encoders)
    for col in categorical_cols:
        df[col] = df[col].astype(str).fillna("UNKNOWN")
        if col in encoders:
            enc = encoders[col]
            df[col] = enc.transform(df[[col]])[col]
        else:
            # Fallback: simple target encoding
            df[col] = df[col].apply(lambda x: hash(x) % 100 / 100.0)
    
    # 2. Clean numeric features
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        df[col].replace([np.inf, -np.inf], 0, inplace=True)
    
    # 3. Create DMatrix
    X = df[numeric_cols + categorical_cols]
    dmatrix = xgb.DMatrix(X, feature_names=feature_names)
    
    return dmatrix, df


# ================================================================
# EXPLAINABILITY FUNCTIONS
# ================================================================

def generate_explanation(feature_row: Dict, fraud_score: float, 
                        icd10: str) -> str:
    """Generate human-readable explanation for reviewer"""
    reasons = []
    
    # Clinical mismatches
    if feature_row["mismatch_count"] > 0:
        mismatches = []
        if feature_row["procedure_mismatch_flag"] == 1:
            mismatches.append("prosedur")
        if feature_row["drug_mismatch_flag"] == 1:
            mismatches.append("obat")
        if feature_row["vitamin_mismatch_flag"] == 1:
            mismatches.append("vitamin")
        
        if mismatches:
            reasons.append(f"Ketidaksesuaian klinis: {', '.join(mismatches)} tidak sesuai diagnosis {icd10}")
    
    # Cost anomaly
    if feature_row["biaya_anomaly_score"] >= 3:
        severity = "sangat tinggi" if feature_row["biaya_anomaly_score"] == 4 else "tinggi"
        reasons.append(f"Biaya klaim {severity}: Rp {feature_row['total_claim_amount']:,.0f}")
    
    # Frequency risk
    if feature_row["patient_frequency_risk"] > 10:
        reasons.append("Frekuensi klaim pasien mencurigakan")
    
    # Risk level
    if fraud_score > 0.7:
        risk_level = "RISIKO TINGGI"
        icon = "🔴"
    elif fraud_score > 0.4:
        risk_level = "RISIKO SEDANG"
        icon = "🟡"
    elif fraud_score > 0.2:
        risk_level = "RISIKO RENDAH"
        icon = "🟢"
    else:
        risk_level = "RISIKO MINIMAL"
        icon = "✅"
    
    if reasons:
        return f"{icon} {risk_level}: " + "; ".join(reasons)
    else:
        return f"{icon} {risk_level}: Tidak ada indikasi fraud yang signifikan"


def get_top_risk_factors(feature_row: Dict, top_n: int = 3) -> List[Dict]:
    """Identify top risk factors for this claim"""
    risk_factors = []
    
    # Sort features by importance
    sorted_features = sorted(
        [(f, feature_importance_map.get(f, 0)) for f in feature_names],
        key=lambda x: x[1],
        reverse=True
    )
    
    for feat_name, importance in sorted_features:
        if feat_name in feature_row:
            value = feature_row[feat_name]
            
            # Check if feature indicates risk
            is_risk = False
            interpretation = ""
            
            if feat_name.endswith("_mismatch_flag") and value == 1:
                is_risk = True
                item_type = feat_name.replace("_mismatch_flag", "").replace("_", " ")
                interpretation = f"{item_type.title()} tidak sesuai dengan diagnosis"
            
            elif feat_name == "mismatch_count" and value > 0:
                is_risk = True
                interpretation = f"{int(value)} ketidaksesuaian klinis ditemukan"
            
            elif feat_name == "biaya_anomaly_score" and value >= 2:
                is_risk = True
                levels = ["", "Normal", "Sedang", "Tinggi", "Sangat Tinggi"]
                interpretation = f"Anomali biaya level {levels[int(value)]}"
            
            elif feat_name == "diagnosis_procedure_score" and value < 0.3:
                is_risk = True
                interpretation = f"Kompatibilitas prosedur rendah ({value:.1%})"
            
            if is_risk:
                risk_factors.append({
                    "feature": feat_name,
                    "value": float(value) if isinstance(value, (int, float)) else str(value),
                    "importance": float(importance),
                    "interpretation": interpretation
                })
            
            if len(risk_factors) >= top_n:
                break
    
    return risk_factors


# ================================================================
# VALIDATION
# ================================================================

def validate_input(data: Dict[str, Any]) -> tuple:
    """Validate input data structure"""
    errors = []
    
    if not isinstance(data, dict):
        errors.append("Input must be a JSON object")
        return False, errors
    
    if "raw_records" not in data:
        errors.append("Missing required field: 'raw_records'")
        return False, errors
    
    raw_records = data["raw_records"]
    
    if not isinstance(raw_records, list):
        errors.append("'raw_records' must be a list")
        return False, errors
    
    if len(raw_records) == 0:
        errors.append("'raw_records' cannot be empty")
        return False, errors
    
    # Validate each record
    required_fields = [
        "claim_id",
        "patient_dob",
        "visit_date",
        "visit_type",
        "department",
        "icd10_primary_code",
        "total_procedure_cost",
        "total_drug_cost",
        "total_vitamin_cost",
        "total_claim_amount",
    ]
    
    for i, rec in enumerate(raw_records):
        missing = [f for f in required_fields if f not in rec]
        if missing:
            errors.append(f"Record {i} missing fields: {missing}")
    
    if errors:
        return False, errors
    
    return True, []


# ================================================================
# MAIN PREDICTION ENDPOINT
# ================================================================

@models.cml_model
def predict(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main prediction endpoint for BPJS fraud detection.
    
    Input: See example below
    Output: Fraud scores and explanations for each claim
    """
    # Handle JSON string input
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except json.JSONDecodeError as e:
            return {
                "status": "error",
                "error": f"Invalid JSON: {str(e)}"
            }
    
    # Validate input
    is_valid, errors = validate_input(data)
    if not is_valid:
        return {
            "status": "error",
            "error": "Input validation failed",
            "details": errors
        }
    
    try:
        raw_records = data["raw_records"]
        
        # Step 1: Build features for each claim
        processed_data = []
        claim_ids = []
        compatibility_data = []
        
        for raw in raw_records:
            claim_id, feature_row, compatibility, _ = build_features_from_raw(raw)
            claim_ids.append(claim_id)
            processed_data.append(feature_row)
            compatibility_data.append(compatibility)
        
        # Step 2: Preprocess and predict
        dmatrix, feature_df = preprocess_features(processed_data)
        
        # Get raw predictions
        y_raw = booster.predict(dmatrix)
        
        # Apply calibration
        y_calibrated = calibrator.predict(y_raw)
        
        # Apply threshold
        y_pred = (y_calibrated >= best_threshold).astype(int)
        
        # Step 3: Build response
        results = []
        
        for i, claim_id in enumerate(claim_ids):
            feature_row = feature_df.iloc[i].to_dict()
            fraud_score = float(y_calibrated[i])
            model_flag = int(y_pred[i])
            
            # Get explanation and risk factors
            explanation = generate_explanation(
                feature_row, 
                fraud_score,
                raw_records[i].get("icd10_primary_code", "UNKNOWN")
            )
            
            risk_factors = get_top_risk_factors(feature_row, top_n=3)
            
            # Calculate confidence
            confidence = min(abs(fraud_score - best_threshold) * 3, 1.0)
            
            # Risk level
            if fraud_score > 0.7:
                risk_level = "HIGH"
                color = "#FF0000"
            elif fraud_score > 0.4:
                risk_level = "MEDIUM"
                color = "#FFA500"
            elif fraud_score > 0.2:
                risk_level = "LOW"
                color = "#FFFF00"
            else:
                risk_level = "MINIMAL"
                color = "#00FF00"
            
            # Recommendation
            if fraud_score > 0.7:
                recommendation = "RECOMMENDED: Decline atau minta dokumen pendukung tambahan"
            elif fraud_score > 0.4:
                recommendation = "RECOMMENDED: Manual review diperlukan"
            elif feature_row["mismatch_count"] > 0:
                recommendation = "RECOMMENDED: Verifikasi ketidaksesuaian klinis"
            else:
                recommendation = "RECOMMENDED: Approve jika dokumen lengkap"
            
            results.append({
                "claim_id": claim_id,
                "fraud_score": round(fraud_score, 4),
                "fraud_probability": f"{fraud_score * 100:.1f}%",
                "model_flag": model_flag,
                "final_flag": model_flag,
                "risk_level": risk_level,
                "risk_color": color,
                "confidence": round(confidence, 3),
                "explanation": explanation,
                "recommendation": recommendation,
                "top_risk_factors": risk_factors,
                "compatibility_scores": {
                    "procedure": round(feature_row["diagnosis_procedure_score"], 3),
                    "drug": round(feature_row["diagnosis_drug_score"], 3),
                    "vitamin": round(feature_row["diagnosis_vitamin_score"], 3)
                },
                "key_indicators": {
                    "mismatch_count": int(feature_row["mismatch_count"]),
                    "cost_anomaly": int(feature_row["biaya_anomaly_score"]),
                    "total_amount": float(feature_row["total_claim_amount"])
                }
            })
        
        return {
            "status": "success",
            "model_version": model_meta.get("model_version", "v2.0"),
            "training_date": model_meta.get("training_date", ""),
            "timestamp": datetime.now().isoformat(),
            "model_performance": {
                "auc": model_meta.get("performance", {}).get("auc", 0),
                "threshold": best_threshold,
                "fraud_detection_rate": model_meta.get("performance", {}).get("fraud_detection_rate", 0)
            },
            "total_claims": len(results),
            "fraud_claims": sum(1 for r in results if r["model_flag"] == 1),
            "results": results
        }
        
    except Exception as e:
        import traceback
        return {
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc()
        }


# ================================================================
# HEALTH CHECK ENDPOINT
# ================================================================

@models.cml_model
def health_check(data: Dict[str, Any] = None) -> Dict[str, Any]:
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_version": model_meta.get("model_version", "unknown"),
        "timestamp": datetime.now().isoformat(),
        "features_count": len(feature_names),
        "threshold": best_threshold,
        "artifacts_loaded": all([
            booster is not None,
            calibrator is not None,
            preprocess is not None
        ])
    }


# ================================================================
# MODEL INFO ENDPOINT
# ================================================================

@models.cml_model
def get_model_info(data: Dict[str, Any] = None) -> Dict[str, Any]:
    """Get model metadata and configuration"""
    return {
        "status": "success",
        "model_metadata": {
            "version": model_meta.get("model_version"),
            "training_date": model_meta.get("training_date"),
            "performance": model_meta.get("performance", {})
        },
        "configuration": {
            "numeric_features": numeric_cols,
            "categorical_features": categorical_cols,
            "total_features": len(feature_names),
            "threshold": best_threshold
        },
        "top_features": GLOBAL_FEATURE_IMPORTANCE[:10]
    }


# ================================================================
# STARTUP MESSAGE
# ================================================================

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("BPJS FRAUD DETECTION MODEL - READY FOR INFERENCE")
    print("=" * 80)
    print("\n✅ Model artifacts loaded successfully")
    print(f"📊 Model version: {model_meta.get('model_version', 'v2.0')}")
    print(f"🎯 Detection threshold: {best_threshold:.3f}")
    print(f"🔢 Features: {len(feature_names)}")
    print("\n📋 Available endpoints:")
    print("  1. predict() - Main fraud detection")
    print("  2. health_check() - Model health status")
    print("  3. get_model_info() - Model metadata")
    print("\n💡 Model capabilities:")
    print("  ✓ Clinical compatibility checking")
    print("  ✓ Cost anomaly detection")
    print("  ✓ Fraud probability scoring (0-100%)")
    print("  ✓ Human-readable explanations")
    print("  ✓ Reviewer recommendations")
    print("=" * 80)