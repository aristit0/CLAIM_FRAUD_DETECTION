#!/usr/bin/env python3
"""
BPJS Fraud Detection Model - Production Deployment
Provides fraud scoring for claim reviewers
Checks clinical compatibility: diagnosis vs procedures, drugs, vitamins
"""

import json
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
import cml.models_v1 as models
from datetime import datetime
from typing import Dict, List, Any, Optional
import sys
import os

# Import centralized config
sys.path.insert(0, '/home/cdsw')  # Pastikan path ini ada
from config import COMPAT_RULES, FRAUD_PATTERNS, get_fraud_pattern_description

# ================================================================
# LOAD MODEL ARTIFACTS
# ================================================================
MODEL_JSON = "model.json"
CALIB_FILE = "calibrator.pkl"
PREPROCESS_FILE = "preprocess.pkl"
META_FILE = "meta.json"

print("=" * 80)
print("BPJS FRAUD DETECTION MODEL - LOADING")
print("=" * 80)

try:
    # Load XGBoost Booster
    booster = xgb.Booster()
    booster.load_model(MODEL_JSON)
    print(f"✓ Model loaded: {MODEL_JSON}")
    
    # Load Calibrator
    with open(CALIB_FILE, "rb") as f:
        calibrator = pickle.load(f)
    print(f"✓ Calibrator loaded: {CALIB_FILE}")
    
    # Load Preprocessing metadata
    with open(PREPROCESS_FILE, "rb") as f:
        preprocess = pickle.load(f)
    print(f"✓ Preprocessing config loaded: {PREPROCESS_FILE}")
    
    # Load metadata
    with open(META_FILE, "r") as f:
        model_meta = json.load(f)
    print(f"✓ Metadata loaded: {META_FILE}")
    
    print("\n📊 Model Information:")
    print(f"  Version: {model_meta.get('model_version', 'unknown')}")
    print(f"  Training Date: {model_meta.get('training_date', 'unknown')}")
    print(f"  AUC Score: {model_meta.get('performance', {}).get('auc', 0):.4f}")
    print(f"  Fraud Detection Rate: {model_meta.get('performance', {}).get('fraud_detection_rate', 0):.1%}")
    print(f"  Features: {model_meta.get('features', {}).get('total_count', 0)}")
    
except Exception as e:
    print(f"✗ Error loading artifacts: {e}")
    raise

# Extract preprocessing config
numeric_cols = preprocess["numeric_cols"]
categorical_cols = preprocess["categorical_cols"]
encoders = preprocess["encoders"]
best_threshold = preprocess["best_threshold"]
feature_importance_map = preprocess["feature_importance"]

feature_names = numeric_cols + categorical_cols

print(f"\n✓ Model ready for inference")
print("=" * 80)

# Global feature importance for response
GLOBAL_FEATURE_IMPORTANCE = [
    {"feature": k, "importance": float(v)}
    for k, v in sorted(feature_importance_map.items(), key=lambda kv: kv[1], reverse=True)
]

# ================================================================
# UTILITY FUNCTIONS
# ================================================================

def compute_age(dob: str, visit_date: str) -> int:
    """Calculate patient age at visit date"""
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
    """
    Calculate clinical compatibility scores.
    This is the CORE fraud detection feature.
    Checks if procedures, drugs, and vitamins are appropriate for the diagnosis.
    """
    rules = COMPAT_RULES.get(icd10)
    
    if not rules:
        # Unknown diagnosis - neutral score
        return {
            "diagnosis_procedure_score": 0.5,
            "diagnosis_drug_score": 0.5,
            "diagnosis_vitamin_score": 0.5
        }
    
    # Get allowed items
    allowed_procedures = rules.get("procedures", [])
    allowed_drugs = rules.get("drugs", [])
    allowed_vitamins = rules.get("vitamins", [])
    
    # Calculate procedure compatibility
    proc_score = 0.5
    if procedures and allowed_procedures:
        proc_matches = sum(1 for p in procedures if p in allowed_procedures)
        proc_score = proc_matches / len(procedures)
    
    # Calculate drug compatibility
    drug_score = 0.5
    if drugs and allowed_drugs:
        drug_matches = sum(1 for d in drugs if d in allowed_drugs)
        drug_score = drug_matches / len(drugs)
    
    # Calculate vitamin compatibility
    vit_score = 0.5
    if vitamins and allowed_vitamins:
        vit_matches = sum(1 for v in vitamins if v in allowed_vitamins)
        vit_score = vit_matches / len(vitamins)
    
    return {
        "diagnosis_procedure_score": float(proc_score),
        "diagnosis_drug_score": float(drug_score),
        "diagnosis_vitamin_score": float(vit_score)
    }


def compute_mismatch_flags(compatibility_scores: Dict[str, float]) -> Dict[str, int]:
    """
    Calculate mismatch flags based on compatibility scores.
    Mismatch = potential fraud indicator
    """
    proc_flag = 1 if compatibility_scores["diagnosis_procedure_score"] < 0.5 else 0
    drug_flag = 1 if compatibility_scores["diagnosis_drug_score"] < 0.5 else 0
    vit_flag = 1 if compatibility_scores["diagnosis_vitamin_score"] < 0.5 else 0
    
    return {
        "procedure_mismatch_flag": proc_flag,
        "drug_mismatch_flag": drug_flag,
        "vitamin_mismatch_flag": vit_flag,
        "mismatch_count": proc_flag + drug_flag + vit_flag
    }


def compute_cost_anomaly_score(total_claim: float, icd10: str = None) -> int:
    """
    Compute cost anomaly score.
    In production, this uses diagnosis-specific statistics.
    For inference, we use general thresholds.
    """
    if total_claim > 1_500_000:
        return 4  # Extreme
    elif total_claim > 1_000_000:
        return 3  # High
    elif total_claim > 500_000:
        return 2  # Moderate
    else:
        return 1  # Normal


def get_compatibility_details(icd10: str, procedures: List[str], 
                              drugs: List[str], vitamins: List[str]) -> Dict[str, Any]:
    """
    Get detailed compatibility analysis for UI display.
    Shows which items are compatible/incompatible.
    """
    rules = COMPAT_RULES.get(icd10)
    
    if not rules:
        return {
            "diagnosis_known": False,
            "diagnosis_description": "Unknown diagnosis - no compatibility rules defined",
            "procedure_details": [],
            "drug_details": [],
            "vitamin_details": []
        }
    
    allowed_procedures = rules.get("procedures", [])
    allowed_drugs = rules.get("drugs", [])
    allowed_vitamins = rules.get("vitamins", [])
    
    # Check each procedure
    procedure_details = []
    for proc in procedures:
        procedure_details.append({
            "code": proc,
            "compatible": proc in allowed_procedures,
            "status": "✓ Compatible" if proc in allowed_procedures else "✗ Incompatible"
        })
    
    # Check each drug
    drug_details = []
    for drug in drugs:
        drug_details.append({
            "code": drug,
            "compatible": drug in allowed_drugs,
            "status": "✓ Compatible" if drug in allowed_drugs else "✗ Incompatible"
        })
    
    # Check each vitamin
    vitamin_details = []
    for vit in vitamins:
        vitamin_details.append({
            "name": vit,
            "compatible": vit in allowed_vitamins,
            "status": "✓ Compatible" if vit in allowed_vitamins else "✗ Incompatible"
        })
    
    return {
        "diagnosis_known": True,
        "diagnosis_description": rules.get("description", ""),
        "procedure_details": procedure_details,
        "drug_details": drug_details,
        "vitamin_details": vitamin_details
    }


# ================================================================
# FEATURE ENGINEERING (MUST MATCH ETL!)
# ================================================================

def build_features_from_raw(raw: Dict[str, Any]) -> tuple:
    """
    Transform raw claim data into model features.
    CRITICAL: This MUST match ETL feature engineering exactly!
    """
    claim_id = raw.get("claim_id")
    
    # Basic fields
    visit_date = raw.get("visit_date")
    dt = datetime.strptime(visit_date, "%Y-%m-%d").date()
    
    # Arrays (convert to lists if needed)
    procedures = raw.get("procedures", [])
    drugs = raw.get("drugs", [])
    vitamins = raw.get("vitamins", [])
    
    # Ensure they are lists
    if not isinstance(procedures, list):
        procedures = [procedures] if procedures else []
    if not isinstance(drugs, list):
        drugs = [drugs] if drugs else []
    if not isinstance(vitamins, list):
        vitamins = [vitamins] if vitamins else []
    
    # Costs
    total_proc = float(raw.get("total_procedure_cost", 0))
    total_drug = float(raw.get("total_drug_cost", 0))
    total_vit = float(raw.get("total_vitamin_cost", 0))
    total_claim = float(raw.get("total_claim_amount", 0))
    
    # Patient age
    patient_age = compute_age(raw.get("patient_dob"), visit_date)
    
    # Clinical compatibility (CORE FEATURE)
    icd10 = raw.get("icd10_primary_code", "UNKNOWN")
    compatibility = compute_compatibility_scores(icd10, procedures, drugs, vitamins)
    
    # Mismatch flags (FRAUD INDICATORS)
    mismatch = compute_mismatch_flags(compatibility)
    
    # Cost anomaly
    biaya_anomaly = compute_cost_anomaly_score(total_claim, icd10)
    
    # Patient frequency (dummy for now - in production, query from DB)
    patient_freq = 2  # Default value
    
    # Build feature dictionary
    feature_row = {
        # Numeric features
        "patient_age": patient_age,
        "total_procedure_cost": total_proc,
        "total_drug_cost": total_drug,
        "total_vitamin_cost": total_vit,
        "total_claim_amount": total_claim,
        "biaya_anomaly_score": biaya_anomaly,
        "patient_frequency_risk": patient_freq,
        "visit_year": dt.year,
        "visit_month": dt.month,
        "visit_day": dt.day,
        
        # Clinical compatibility scores
        "diagnosis_procedure_score": compatibility["diagnosis_procedure_score"],
        "diagnosis_drug_score": compatibility["diagnosis_drug_score"],
        "diagnosis_vitamin_score": compatibility["diagnosis_vitamin_score"],
        
        # Mismatch flags
        "procedure_mismatch_flag": mismatch["procedure_mismatch_flag"],
        "drug_mismatch_flag": mismatch["drug_mismatch_flag"],
        "vitamin_mismatch_flag": mismatch["vitamin_mismatch_flag"],
        "mismatch_count": mismatch["mismatch_count"],
        
        # Categorical features
        "visit_type": raw.get("visit_type", "UNKNOWN"),
        "department": raw.get("department", "UNKNOWN"),
        "icd10_primary_code": icd10,
    }
    
    return claim_id, feature_row, compatibility, mismatch


# ================================================================
# FEATURE DATAFRAME BUILDER
# ================================================================

def build_feature_df(records: List[Dict[str, Any]]) -> tuple:
    """
    Build feature DataFrame from processed records.
    Apply same transformations as training.
    """
    df = pd.DataFrame.from_records(records)
    
    # Ensure all columns exist
    for col_name in numeric_cols + categorical_cols:
        if col_name not in df.columns:
            df[col_name] = None
    
    # Encode categorical features (using trained encoders)
    for col_name in categorical_cols:
        df[col_name] = df[col_name].astype(str).fillna("UNKNOWN")
        enc = encoders[col_name]
        df[col_name] = enc.transform(df[[col_name]])[col_name]
    
    # Clean numeric features
    for col_name in numeric_cols:
        df[col_name] = pd.to_numeric(df[col_name], errors="coerce").fillna(0.0)
        # Handle inf
        df[col_name].replace([np.inf, -np.inf], 0, inplace=True)
    
    # Create DMatrix for XGBoost
    X = df[numeric_cols + categorical_cols]
    dmatrix = xgb.DMatrix(X, feature_names=feature_names)
    
    return df, dmatrix


# ================================================================
# EXPLANATION GENERATOR
# ================================================================

def generate_explanation(row: Dict[str, Any], fraud_score: float, 
                        icd10: str, compatibility_details: Dict) -> str:
    """
    Generate human-readable explanation for BPJS reviewers.
    Focus on actionable insights.
    """
    reasons = []
    
    # Clinical mismatches (most important)
    if row["mismatch_count"] > 0:
        mismatch_items = []
        if row["procedure_mismatch_flag"] == 1:
            mismatch_items.append("prosedur tidak sesuai diagnosis")
        if row["drug_mismatch_flag"] == 1:
            mismatch_items.append("obat tidak sesuai diagnosis")
        if row["vitamin_mismatch_flag"] == 1:
            mismatch_items.append("vitamin tidak relevan")
        
        reasons.append(f"Ketidaksesuaian klinis: {', '.join(mismatch_items)}")
    
    # Cost anomaly
    if row["biaya_anomaly_score"] >= 3:
        severity = "sangat tinggi" if row["biaya_anomaly_score"] == 4 else "tinggi"
        reasons.append(f"Biaya klaim {severity} untuk diagnosis ini")
    
    # High frequency
    if row["patient_frequency_risk"] > 10:
        reasons.append("Frekuensi klaim pasien mencurigakan")
    
    # Risk level determination
    if fraud_score > 0.8:
        risk_level = "RISIKO TINGGI"
        color = "🔴"
    elif fraud_score > 0.5:
        risk_level = "RISIKO SEDANG"
        color = "🟡"
    elif fraud_score > 0.3:
        risk_level = "RISIKO RENDAH"
        color = "🟢"
    else:
        risk_level = "RISIKO MINIMAL"
        color = "🟢"
    
    if reasons:
        explanation = f"{color} {risk_level}: " + "; ".join(reasons)
    else:
        explanation = f"{color} {risk_level}: Tidak ada indikator fraud yang terdeteksi"
    
    return explanation


def get_top_risk_factors(row: Dict[str, Any], 
                         feature_importance: Dict[str, float],
                         top_n: int = 5) -> List[Dict[str, Any]]:
    """Identify top risk factors for this specific claim"""
    risk_factors = []
    
    # Get top N important features
    top_features = list(feature_importance.items())[:top_n * 3]  # Get extra to filter
    
    for feat_name, importance in top_features:
        if feat_name in row:
            value = row[feat_name]
            
            # Only include if value is significant
            if isinstance(value, (int, float)):
                if feat_name.endswith("_flag") and value == 1:
                    interpretation = {
                        "procedure_mismatch_flag": "Prosedur tidak sesuai diagnosis",
                        "drug_mismatch_flag": "Obat tidak sesuai diagnosis",
                        "vitamin_mismatch_flag": "Vitamin tidak relevan",
                    }.get(feat_name, feat_name.replace("_", " ").title())
                    
                    risk_factors.append({
                        "feature": feat_name,
                        "value": value,
                        "importance": float(importance),
                        "interpretation": interpretation
                    })
                    
                elif feat_name == "mismatch_count" and value > 0:
                    risk_factors.append({
                        "feature": feat_name,
                        "value": value,
                        "importance": float(importance),
                        "interpretation": f"{int(value)} ketidaksesuaian klinis terdeteksi"
                    })
                    
                elif feat_name == "biaya_anomaly_score" and value >= 2:
                    severity = ["", "Normal", "Sedang", "Tinggi", "Sangat Tinggi"][int(value)]
                    risk_factors.append({
                        "feature": feat_name,
                        "value": value,
                        "importance": float(importance),
                        "interpretation": f"Anomali biaya level {severity}"
                    })
            
            if len(risk_factors) >= top_n:
                break
    
    return risk_factors


def get_recommendation(fraud_score: float, mismatch_count: int, 
                       cost_anomaly: int) -> str:
    """Generate recommendation for reviewer"""
    if fraud_score > 0.8:
        return "RECOMMENDED: Decline atau minta dokumen pendukung tambahan"
    elif fraud_score > 0.5:
        return "RECOMMENDED: Manual review mendalam diperlukan"
    elif mismatch_count > 0:
        return "RECOMMENDED: Verifikasi ketidaksesuaian klinis dengan dokter"
    elif cost_anomaly >= 3:
        return "RECOMMENDED: Verifikasi justifikasi biaya tinggi"
    else:
        return "RECOMMENDED: Approve jika dokumen lengkap"


# ================================================================
# VALIDATION
# ================================================================

def validate_input(data: Dict[str, Any]) -> tuple:
    """Validate input data"""
    errors = []
    
    # Check if raw_records exists
    if "raw_records" not in data:
        errors.append("Missing 'raw_records' field")
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
        "claim_id", "patient_dob", "visit_date",
        "total_procedure_cost", "total_drug_cost", "total_vitamin_cost",
        "total_claim_amount", "icd10_primary_code", "department", "visit_type"
    ]
    
    for i, rec in enumerate(raw_records):
        missing = [f for f in required_fields if f not in rec]
        if missing:
            errors.append(f"Record {i}: missing required fields {missing}")
    
    if errors:
        return False, errors
    
    return True, []


# ================================================================
# MAIN INFERENCE HANDLER
# ================================================================

@models.cml_model
def predict(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main prediction endpoint for BPJS fraud detection.
    
    Input format:
    {
        "raw_records": [
            {
                "claim_id": "12345",
                "patient_dob": "1980-01-01",
                "visit_date": "2024-01-15",
                "visit_type": "rawat jalan",
                "department": "Poli Umum",
                "icd10_primary_code": "J06",
                "procedures": ["89.02", "96.70"],
                "drugs": ["KFA001", "KFA002"],
                "vitamins": ["Vitamin C 500 mg"],
                "total_procedure_cost": 150000,
                "total_drug_cost": 50000,
                "total_vitamin_cost": 25000,
                "total_claim_amount": 225000
            }
        ]
    }
    
    Output format:
    {
        "status": "success",
        "model_version": "v7_production_bpjs",
        "timestamp": "2024-01-15T10:30:00",
        "results": [
            {
                "claim_id": "12345",
                "fraud_score": 0.234,
                "fraud_probability": "23.4%",
                "model_flag": 0,
                "risk_level": "LOW RISK",
                "confidence": 0.85,
                "explanation": "...",
                "recommendation": "...",
                "clinical_compatibility": {
                    "procedure_compatible": true,
                    "drug_compatible": true,
                    "vitamin_compatible": false,
                    "details": {...}
                },
                "top_risk_factors": [...],
                "features": {...}
            }
        ]
    }
    """
    
    # Parse JSON if string
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except json.JSONDecodeError as e:
            return {
                "status": "error",
                "error": f"Invalid JSON: {str(e)}"
            }
    
    # Validate input
    is_valid, validation_errors = validate_input(data)
    if not is_valid:
        return {
            "status": "error",
            "error": "Input validation failed",
            "details": validation_errors
        }
    
    raw_records = data["raw_records"]
    
    try:
        # Process each record
        processed_records = []
        claim_ids = []
        compatibility_data = []
        mismatch_data = []
        icd10_codes = []
        
        for raw in raw_records:
            claim_id, feature_row, compatibility, mismatch = build_features_from_raw(raw)
            claim_ids.append(claim_id)
            processed_records.append(feature_row)
            compatibility_data.append(compatibility)
            mismatch_data.append(mismatch)
            icd10_codes.append(raw.get("icd10_primary_code", "UNKNOWN"))
        
        # Build feature DataFrame and predict
        df_features, dmatrix = build_feature_df(processed_records)
        
        # Get predictions
        y_raw = booster.predict(dmatrix)
        y_calibrated = calibrator.predict(y_raw)
        y_pred = (y_calibrated >= best_threshold).astype(int)
        
        # Build results
        results = []
        
        for i, claim_id in enumerate(claim_ids):
            row = df_features.iloc[i].to_dict()
            fraud_score = float(y_calibrated[i])
            model_flag = int(y_pred[i])
            
            # Confidence (distance from threshold)
            confidence = abs(fraud_score - best_threshold) * 2
            confidence = min(confidence, 1.0)
            
            # Risk level
            if fraud_score > 0.8:
                risk_level = "HIGH RISK"
            elif fraud_score > 0.5:
                risk_level = "MODERATE RISK"
            elif fraud_score > 0.3:
                risk_level = "LOW RISK"
            else:
                risk_level = "MINIMAL RISK"
            
            # Get detailed compatibility info
            raw_record = raw_records[i]
            procedures = raw_record.get("procedures", [])
            drugs = raw_record.get("drugs", [])
            vitamins = raw_record.get("vitamins", [])
            
            # Ensure lists
            if not isinstance(procedures, list):
                procedures = [procedures] if procedures else []
            if not isinstance(drugs, list):
                drugs = [drugs] if drugs else []
            if not isinstance(vitamins, list):
                vitamins = [vitamins] if vitamins else []
            
            compatibility_details = get_compatibility_details(
                icd10_codes[i], procedures, drugs, vitamins
            )
            
            # Generate explanation
            explanation = generate_explanation(row, fraud_score, icd10_codes[i], compatibility_details)
            
            # Top risk factors
            risk_factors = get_top_risk_factors(row, feature_importance_map, top_n=5)
            
            # Recommendation
            recommendation = get_recommendation(fraud_score, row["mismatch_count"], row["biaya_anomaly_score"])
            
            # Clinical compatibility summary
            clinical_compat = {
                "procedure_compatible": row["diagnosis_procedure_score"] >= 0.5,
                "drug_compatible": row["diagnosis_drug_score"] >= 0.5,
                "vitamin_compatible": row["diagnosis_vitamin_score"] >= 0.5,
                "overall_compatible": row["mismatch_count"] == 0,
                "details": compatibility_details
            }
            
            results.append({
                "claim_id": claim_id,
                "fraud_score": round(fraud_score, 4),
                "fraud_probability": f"{fraud_score * 100:.1f}%",
                "model_flag": model_flag,
                "final_flag": model_flag,
                "risk_level": risk_level,
                "confidence": round(confidence, 4),
                "explanation": explanation,
                "recommendation": recommendation,
                "top_risk_factors": risk_factors,
                "clinical_compatibility": clinical_compat,
                "features": {
                    "mismatch_count": int(row["mismatch_count"]),
                    "cost_anomaly_score": int(row["biaya_anomaly_score"]),
                    "total_claim_amount": float(row["total_claim_amount"]),
                    "diagnosis_procedure_score": round(row["diagnosis_procedure_score"], 3),
                    "diagnosis_drug_score": round(row["diagnosis_drug_score"], 3),
                    "diagnosis_vitamin_score": round(row["diagnosis_vitamin_score"], 3),
                }
            })
        
        return {
            "status": "success",
            "model_version": model_meta.get("model_version", "unknown"),
            "timestamp": datetime.now().isoformat(),
            "total_claims_processed": len(results),
            "fraud_detected": sum(1 for r in results if r["model_flag"] == 1),
            "results": results,
            "model_info": {
                "threshold": best_threshold,
                "training_auc": model_meta.get("performance", {}).get("auc", 0),
                "training_f1": model_meta.get("performance", {}).get("f1", 0),
                "fraud_detection_rate": model_meta.get("performance", {}).get("fraud_detection_rate", 0),
            }
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
def health_check(data: Dict[str, Any]) -> Dict[str, Any]:
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_version": model_meta.get("model_version", "unknown"),
        "timestamp": datetime.now().isoformat(),
        "features_count": len(feature_names),
        "threshold": best_threshold,
        "supported_diagnoses": len(COMPAT_RULES),
    }


# ================================================================
# MODEL INFO ENDPOINT
# ================================================================

@models.cml_model
def get_model_info(data: Dict[str, Any]) -> Dict[str, Any]:
    """Return model metadata and feature importance"""
    return {
        "status": "success",
        "model_metadata": model_meta,
        "feature_importance": GLOBAL_FEATURE_IMPORTANCE[:20],  # Top 20
        "compatibility_rules_count": len(COMPAT_RULES),
        "supported_diagnoses": list(COMPAT_RULES.keys()),
        "fraud_patterns": FRAUD_PATTERNS
    }


# ================================================================
# STARTUP MESSAGE
# ================================================================

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("BPJS FRAUD DETECTION MODEL - INFERENCE API READY")
    print("=" * 80)
    print("\nAvailable endpoints:")
    print("  1. predict() - Main fraud detection endpoint")
    print("  2. health_check() - Model health status")
    print("  3. get_model_info() - Model metadata and rules")
    print("\nModel capabilities:")
    print("  ✓ Fraud score prediction (0-100%)")
    print("  ✓ Clinical compatibility checking")
    print("  ✓ Diagnosis vs Procedure compatibility")
    print("  ✓ Diagnosis vs Drug compatibility")
    print("  ✓ Diagnosis vs Vitamin compatibility")
    print("  ✓ Cost anomaly detection")
    print("  ✓ Actionable recommendations for reviewers")
    print("\n" + "=" * 80)