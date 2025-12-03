#!/usr/bin/env python3
"""
Fraud Detection Model - Production Inference API
Version: v7_production
Aligned with: data_generator v2, ETL v7, training v7
"""

import json
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
import cml.models_v1 as models
from datetime import datetime
from typing import Dict, List, Any, Optional

# ================================================================
# CLINICAL COMPATIBILITY RULES (MUST MATCH DATA GENERATOR!)
# ================================================================
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

# ================================================================
# LOAD MODEL ARTIFACTS
# ================================================================
MODEL_JSON = "model.json"
CALIB_FILE = "calibrator.pkl"
PREPROCESS_FILE = "preprocess.pkl"
META_FILE = "meta.json"

print("=" * 80)
print("FRAUD DETECTION MODEL - LOADING ARTIFACTS")
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
    
    print("\nModel Information:")
    print(f"  Version: {model_meta.get('model_version', 'unknown')}")
    print(f"  Training Date: {model_meta.get('training_date', 'unknown')}")
    print(f"  AUC Score: {model_meta.get('performance', {}).get('auc', 0):.4f}")
    print(f"  F1 Score: {model_meta.get('performance', {}).get('f1', 0):.4f}")
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
    MUST MATCH ETL LOGIC EXACTLY!
    """
    rules = COMPAT_RULES.get(icd10)
    
    if not rules:
        # Unknown diagnosis - neutral score
        return {
            "diagnosis_procedure_score": 0.5,
            "diagnosis_drug_score": 0.5,
            "diagnosis_vitamin_score": 0.5
        }
    
    # Calculate match ratios
    allowed_procedures = rules.get("procedures", [])
    allowed_drugs = rules.get("drugs", [])
    allowed_vitamins = rules.get("vitamins", [])
    
    proc_score = 0.5
    if procedures and allowed_procedures:
        proc_matches = sum(1 for p in procedures if p in allowed_procedures)
        proc_score = proc_matches / len(procedures)
    
    drug_score = 0.5
    if drugs and allowed_drugs:
        drug_matches = sum(1 for d in drugs if d in allowed_drugs)
        drug_score = drug_matches / len(drugs)
    
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
    """Calculate mismatch flags based on compatibility scores"""
    proc_flag = 1 if compatibility_scores["diagnosis_procedure_score"] < 0.5 else 0
    drug_flag = 1 if compatibility_scores["diagnosis_drug_score"] < 0.5 else 0
    vit_flag = 1 if compatibility_scores["diagnosis_vitamin_score"] < 0.5 else 0
    
    return {
        "procedure_mismatch_flag": proc_flag,
        "drug_mismatch_flag": drug_flag,
        "vitamin_mismatch_flag": vit_flag,
        "mismatch_count": proc_flag + drug_flag + vit_flag
    }


def compute_cost_anomaly_score(total_claim: float) -> int:
    """
    Compute cost anomaly score.
    Note: In production, this should use diagnosis-specific statistics.
    For now, using simple thresholds.
    """
    if total_claim > 1_500_000:
        return 4  # Extreme
    elif total_claim > 1_000_000:
        return 3  # High
    elif total_claim > 500_000:
        return 2  # Moderate
    else:
        return 1  # Normal


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
    
    # Arrays
    procedures = raw.get("procedures", [])
    drugs = raw.get("drugs", [])
    vitamins = raw.get("vitamins", [])
    
    # Costs
    total_proc = float(raw.get("total_procedure_cost", 0))
    total_drug = float(raw.get("total_drug_cost", 0))
    total_vit = float(raw.get("total_vitamin_cost", 0))
    total_claim = float(raw.get("total_claim_amount", 0))
    
    # Patient age
    patient_age = compute_age(raw.get("patient_dob"), visit_date)
    
    # Clinical compatibility
    icd10 = raw.get("icd10_primary_code", "UNKNOWN")
    compatibility = compute_compatibility_scores(icd10, procedures, drugs, vitamins)
    
    # Mismatch flags
    mismatch = compute_mismatch_flags(compatibility)
    
    # Cost anomaly
    biaya_anomaly = compute_cost_anomaly_score(total_claim)
    
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
    
    return claim_id, feature_row


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

def generate_explanation(row: Dict[str, Any], fraud_score: float) -> str:
    """Generate human-readable fraud explanation"""
    reasons = []
    
    # Clinical mismatches
    if row["mismatch_count"] > 0:
        mismatch_details = []
        if row["procedure_mismatch_flag"] == 1:
            mismatch_details.append("procedure")
        if row["drug_mismatch_flag"] == 1:
            mismatch_details.append("drug")
        if row["vitamin_mismatch_flag"] == 1:
            mismatch_details.append("vitamin")
        
        reasons.append(f"Clinical incompatibility detected ({', '.join(mismatch_details)})")
    
    # Cost anomaly
    if row["biaya_anomaly_score"] >= 3:
        severity = "extreme" if row["biaya_anomaly_score"] == 4 else "high"
        reasons.append(f"Claim amount shows {severity} deviation from normal")
    
    # High frequency
    if row["patient_frequency_risk"] > 10:
        reasons.append("Unusually high claim frequency for patient")
    
    # Fraud score interpretation
    if fraud_score > 0.8:
        risk_level = "HIGH RISK"
    elif fraud_score > 0.5:
        risk_level = "MODERATE RISK"
    elif fraud_score > 0.3:
        risk_level = "LOW RISK"
    else:
        risk_level = "MINIMAL RISK"
    
    if reasons:
        explanation = f"{risk_level}: " + "; ".join(reasons)
    else:
        explanation = f"{risk_level}: No specific fraud indicators detected"
    
    return explanation


def get_top_risk_factors(row: Dict[str, Any], 
                         feature_importance: Dict[str, float],
                         top_n: int = 5) -> List[Dict[str, Any]]:
    """Identify top risk factors for this specific claim"""
    risk_factors = []
    
    # Get top N important features
    top_features = list(feature_importance.items())[:top_n * 2]  # Get extra to filter
    
    for feat_name, importance in top_features:
        if feat_name in row:
            value = row[feat_name]
            
            # Only include if value is significant
            if isinstance(value, (int, float)):
                if feat_name.endswith("_flag") and value == 1:
                    risk_factors.append({
                        "feature": feat_name,
                        "value": value,
                        "importance": float(importance),
                        "interpretation": f"{feat_name.replace('_', ' ').title()} detected"
                    })
                elif feat_name == "mismatch_count" and value > 0:
                    risk_factors.append({
                        "feature": feat_name,
                        "value": value,
                        "importance": float(importance),
                        "interpretation": f"{int(value)} clinical mismatches"
                    })
                elif feat_name == "biaya_anomaly_score" and value >= 2:
                    risk_factors.append({
                        "feature": feat_name,
                        "value": value,
                        "importance": float(importance),
                        "interpretation": f"Cost anomaly level {int(value)}"
                    })
            
            if len(risk_factors) >= top_n:
                break
    
    return risk_factors


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
    Main prediction endpoint.
    
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
        "model_version": "v7_production",
        "timestamp": "2024-01-15T10:30:00",
        "results": [
            {
                "claim_id": "12345",
                "fraud_score": 0.234,
                "fraud_probability": "23.4%",
                "model_flag": 0,
                "final_flag": 0,
                "risk_level": "LOW RISK",
                "confidence": 0.85,
                "explanation": "...",
                "top_risk_factors": [...],
                "clinical_compatibility": {...}
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
        
        for raw in raw_records:
            claim_id, feature_row = build_features_from_raw(raw)
            claim_ids.append(claim_id)
            processed_records.append(feature_row)
        
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
            
            # Generate explanation
            explanation = generate_explanation(row, fraud_score)
            
            # Top risk factors
            risk_factors = get_top_risk_factors(row, feature_importance_map, top_n=5)
            
            # Risk level
            if fraud_score > 0.8:
                risk_level = "HIGH RISK"
            elif fraud_score > 0.5:
                risk_level = "MODERATE RISK"
            elif fraud_score > 0.3:
                risk_level = "LOW RISK"
            else:
                risk_level = "MINIMAL RISK"
            
            # Clinical compatibility summary
            clinical_compat = {
                "procedure_compatible": row["diagnosis_procedure_score"] >= 0.5,
                "drug_compatible": row["diagnosis_drug_score"] >= 0.5,
                "vitamin_compatible": row["diagnosis_vitamin_score"] >= 0.5,
                "overall_compatible": row["mismatch_count"] == 0
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
                "top_risk_factors": risk_factors,
                "clinical_compatibility": clinical_compat,
                "features": {
                    "mismatch_count": int(row["mismatch_count"]),
                    "cost_anomaly_score": int(row["biaya_anomaly_score"]),
                    "total_claim_amount": float(row["total_claim_amount"]),
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
    }


# ================================================================
# BATCH PREDICTION ENDPOINT (OPTIONAL)
# ================================================================

@models.cml_model
def predict_batch(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Batch prediction endpoint for large volumes.
    Same as predict() but optimized for throughput.
    """
    return predict(data)


# ================================================================
# MODEL METADATA ENDPOINT
# ================================================================

@models.cml_model
def get_model_info(data: Dict[str, Any]) -> Dict[str, Any]:
    """Return model metadata and feature importance"""
    return {
        "status": "success",
        "model_metadata": model_meta,
        "feature_importance": GLOBAL_FEATURE_IMPORTANCE[:20],  # Top 20
        "compatibility_rules_count": len(COMPAT_RULES),
        "supported_diagnoses": list(COMPAT_RULES.keys())
    }


# ================================================================
# STARTUP MESSAGE
# ================================================================

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("FRAUD DETECTION MODEL - INFERENCE API READY")
    print("=" * 80)
    print("\nAvailable endpoints:")
    print("  1. predict() - Main fraud detection endpoint")
    print("  2. health_check() - Model health status")
    print("  3. predict_batch() - Batch prediction")
    print("  4. get_model_info() - Model metadata")
    print("\n" + "=" * 80)