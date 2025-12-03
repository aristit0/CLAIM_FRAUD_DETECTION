#!/usr/bin/env python3
"""
BPJS Fraud Detection Model - Production Deployment for CML
Detects fraud in new claim data with clinical compatibility checking
Integrated with Iceberg reference tables

Version: 2.0
Purpose: Serve prediction requests from approval UI application
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
import traceback

# Import configuration
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from config import (
        NUMERIC_FEATURES, 
        CATEGORICAL_FEATURES, 
        FRAUD_PATTERNS,
        get_fraud_pattern_description,
        calculate_fraud_score,
        COMPAT_RULES_FALLBACK
    )
    print("✓ Configuration loaded successfully")
except ImportError as e:
    print(f"⚠ Warning: Could not load config, using defaults: {e}")
    # Fallback configuration
    NUMERIC_FEATURES = [
        "patient_age", "total_procedure_cost", "total_drug_cost", "total_vitamin_cost",
        "total_claim_amount", "diagnosis_procedure_score", "diagnosis_drug_score",
        "diagnosis_vitamin_score", "procedure_mismatch_flag", "drug_mismatch_flag",
        "vitamin_mismatch_flag", "mismatch_count", "biaya_anomaly_score",
        "patient_frequency_risk", "visit_year", "visit_month", "visit_day"
    ]
    CATEGORICAL_FEATURES = ["visit_type", "department", "icd10_primary_code"]
    COMPAT_RULES_FALLBACK = {}

# ================================================================
# GLOBAL VARIABLES
# ================================================================
MODEL_VERSION = "v2.0_production"
MODEL_NAME = "BPJS Fraud Detection"

print("=" * 80)
print(f"{MODEL_NAME} - LOADING ARTIFACTS")
print("=" * 80)

# ================================================================
# LOAD MODEL ARTIFACTS
# ================================================================
MODEL_JSON = "model.json"
CALIB_FILE = "calibrator.pkl"
PREPROCESS_FILE = "preprocess.pkl"
META_FILE = "meta.json"

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
print(f"  Features: {len(feature_names)}")
print(f"  Threshold: {best_threshold:.3f}")
print("=" * 80)

# Global feature importance for response
GLOBAL_FEATURE_IMPORTANCE = [
    {"feature": k, "importance": float(v)}
    for k, v in sorted(feature_importance_map.items(), key=lambda kv: kv[1], reverse=True)
]


# ================================================================
# CLINICAL COMPATIBILITY FUNCTIONS
# ================================================================

def load_clinical_rules_from_iceberg():
    """
    Load clinical rules from Iceberg reference tables.
    Falls back to hardcoded rules if Iceberg not available.
    """
    try:
        import cml.data_v1 as cmldata
        conn = cmldata.get_connection("CDP-MSI")
        spark = conn.get_spark_session()
        
        # Load reference tables
        ref_dx_drug = spark.sql("SELECT * FROM iceberg_ref.clinical_rule_dx_drug").toPandas()
        ref_dx_proc = spark.sql("SELECT * FROM iceberg_ref.clinical_rule_dx_procedure").toPandas()
        ref_dx_vit = spark.sql("SELECT * FROM iceberg_ref.clinical_rule_dx_vitamin").toPandas()
        
        # Build compatibility dictionary
        rules = {}
        
        # Get unique diagnoses
        all_dx = set(list(ref_dx_drug['icd10_code'].unique()) + 
                     list(ref_dx_proc['icd10_code'].unique()) + 
                     list(ref_dx_vit['icd10_code'].unique()))
        
        for dx in all_dx:
            rules[dx] = {
                "procedures": ref_dx_proc[ref_dx_proc['icd10_code'] == dx]['icd9_code'].tolist(),
                "drugs": ref_dx_drug[ref_dx_drug['icd10_code'] == dx]['drug_code'].tolist(),
                "vitamins": ref_dx_vit[ref_dx_vit['icd10_code'] == dx]['vitamin_name'].tolist(),
            }
        
        print(f"✓ Loaded clinical rules from Iceberg: {len(rules)} diagnoses")
        spark.stop()
        return rules
        
    except Exception as e:
        print(f"⚠ Could not load from Iceberg, using fallback rules: {e}")
        return COMPAT_RULES_FALLBACK


# Load clinical rules at startup
try:
    CLINICAL_RULES = load_clinical_rules_from_iceberg()
except Exception as e:
    CLINICAL_RULES = COMPAT_RULES_FALLBACK

print(f"✓ Clinical rules loaded: {len(CLINICAL_RULES)} diagnoses")


def compute_age(dob: str, visit_date: str) -> int:
    """Calculate patient age at visit date"""
    try:
        dob_dt = datetime.strptime(str(dob), "%Y-%m-%d").date()
        visit_dt = datetime.strptime(str(visit_date), "%Y-%m-%d").date()
        age = visit_dt.year - dob_dt.year - (
            (visit_dt.month, visit_dt.day) < (dob_dt.month, dob_dt.day)
        )
        return max(age, 0)
    except Exception:
        return 0


def compute_compatibility_scores(
    icd10: str, 
    procedures: List[str], 
    drugs: List[str], 
    vitamins: List[str]
) -> Dict[str, float]:
    """
    Calculate clinical compatibility scores.
    Returns scores between 0.0 (no match) and 1.0 (perfect match).
    """
    rules = CLINICAL_RULES.get(icd10)
    
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
        proc_score = proc_matches / len(procedures) if len(procedures) > 0 else 0.5
    elif not procedures:
        proc_score = 0.0
    
    # Calculate drug compatibility
    drug_score = 0.5
    if drugs and allowed_drugs:
        drug_matches = sum(1 for d in drugs if d in allowed_drugs)
        drug_score = drug_matches / len(drugs) if len(drugs) > 0 else 0.5
    elif not drugs:
        drug_score = 0.0
    
    # Calculate vitamin compatibility
    vit_score = 0.5
    if vitamins and allowed_vitamins:
        vit_matches = sum(1 for v in vitamins if v in allowed_vitamins)
        vit_score = vit_matches / len(vitamins) if len(vitamins) > 0 else 0.5
    elif not vitamins:
        vit_score = 0.0
    
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


def compute_cost_anomaly_score(total_claim: float, icd10: str = None) -> int:
    """
    Compute cost anomaly score based on claim amount.
    1 = Normal, 2 = Moderate, 3 = High, 4 = Extreme
    """
    if total_claim > 1_500_000:
        return 4  # Extreme
    elif total_claim > 1_000_000:
        return 3  # High
    elif total_claim > 500_000:
        return 2  # Moderate
    else:
        return 1  # Normal


def get_compatibility_details(
    icd10: str, 
    procedures: List[str], 
    drugs: List[str], 
    vitamins: List[str]
) -> Dict[str, Any]:
    """
    Get detailed compatibility analysis for UI display.
    Shows which items are compatible/incompatible with diagnosis.
    """
    rules = CLINICAL_RULES.get(icd10)
    
    if not rules:
        return {
            "diagnosis_known": False,
            "diagnosis_description": f"Diagnosis {icd10} tidak memiliki clinical rule yang terdefinisi",
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
        is_compatible = proc in allowed_procedures
        procedure_details.append({
            "code": proc,
            "compatible": is_compatible,
            "status": "✓ Sesuai" if is_compatible else "✗ Tidak Sesuai",
            "severity": "normal" if is_compatible else "warning"
        })
    
    # Check each drug
    drug_details = []
    for drug in drugs:
        is_compatible = drug in allowed_drugs
        drug_details.append({
            "code": drug,
            "compatible": is_compatible,
            "status": "✓ Sesuai" if is_compatible else "✗ Tidak Sesuai",
            "severity": "normal" if is_compatible else "warning"
        })
    
    # Check each vitamin
    vitamin_details = []
    for vit in vitamins:
        is_compatible = vit in allowed_vitamins
        vitamin_details.append({
            "name": vit,
            "compatible": is_compatible,
            "status": "✓ Sesuai" if is_compatible else "✗ Tidak Sesuai",
            "severity": "normal" if is_compatible else "info"
        })
    
    return {
        "diagnosis_known": True,
        "diagnosis_code": icd10,
        "diagnosis_description": rules.get("description", ""),
        "procedure_details": procedure_details,
        "drug_details": drug_details,
        "vitamin_details": vitamin_details,
        "summary": {
            "total_procedures": len(procedures),
            "compatible_procedures": sum(1 for p in procedure_details if p["compatible"]),
            "total_drugs": len(drugs),
            "compatible_drugs": sum(1 for d in drug_details if d["compatible"]),
            "total_vitamins": len(vitamins),
            "compatible_vitamins": sum(1 for v in vitamin_details if v["compatible"])
        }
    }


# ================================================================
# FEATURE ENGINEERING
# ================================================================

def build_features_from_claim(claim: Dict[str, Any]) -> tuple:
    """
    Transform raw claim data into model features.
    MUST match ETL feature engineering exactly!
    """
    claim_id = claim.get("claim_id")
    
    # Basic fields
    visit_date = claim.get("visit_date")
    dt = datetime.strptime(visit_date, "%Y-%m-%d").date()
    
    # Arrays - ensure they are lists
    procedures = claim.get("procedures", [])
    drugs = claim.get("drugs", [])
    vitamins = claim.get("vitamins", [])
    
    if not isinstance(procedures, list):
        procedures = [procedures] if procedures else []
    if not isinstance(drugs, list):
        drugs = [drugs] if drugs else []
    if not isinstance(vitamins, list):
        vitamins = [vitamins] if vitamins else []
    
    # Costs
    total_proc = float(claim.get("total_procedure_cost", 0))
    total_drug = float(claim.get("total_drug_cost", 0))
    total_vit = float(claim.get("total_vitamin_cost", 0))
    total_claim = float(claim.get("total_claim_amount", 0))
    
    # Patient age
    patient_age = compute_age(claim.get("patient_dob"), visit_date)
    
    # Clinical compatibility (CORE FEATURE)
    icd10 = claim.get("icd10_primary_code", "UNKNOWN")
    compatibility = compute_compatibility_scores(icd10, procedures, drugs, vitamins)
    
    # Mismatch flags (FRAUD INDICATORS)
    mismatch = compute_mismatch_flags(compatibility)
    
    # Cost anomaly
    biaya_anomaly = compute_cost_anomaly_score(total_claim, icd10)
    
    # Patient frequency (in production, query from database/cache)
    patient_freq = claim.get("patient_frequency_risk", 2)  # Default
    
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
        "visit_type": claim.get("visit_type", "UNKNOWN"),
        "department": claim.get("department", "UNKNOWN"),
        "icd10_primary_code": icd10,
    }
    
    return claim_id, feature_row, compatibility, mismatch


def build_feature_dataframe(records: List[Dict[str, Any]]) -> tuple:
    """
    Build feature DataFrame and apply preprocessing.
    """
    df = pd.DataFrame.from_records(records)
    
    # Ensure all columns exist
    for col_name in numeric_cols + categorical_cols:
        if col_name not in df.columns:
            df[col_name] = None
    
    # Encode categorical features
    for col_name in categorical_cols:
        df[col_name] = df[col_name].astype(str).fillna("UNKNOWN")
        enc = encoders[col_name]
        df[col_name] = enc.transform(df[[col_name]])[col_name]
    
    # Clean numeric features
    for col_name in numeric_cols:
        df[col_name] = pd.to_numeric(df[col_name], errors="coerce").fillna(0.0)
        df[col_name].replace([np.inf, -np.inf], 0, inplace=True)
    
    # Create DMatrix for XGBoost
    X = df[numeric_cols + categorical_cols]
    dmatrix = xgb.DMatrix(X, feature_names=feature_names)
    
    return df, dmatrix


# ================================================================
# EXPLANATION GENERATION
# ================================================================

def generate_explanation(
    row: Dict[str, Any], 
    fraud_score: float, 
    icd10: str
) -> str:
    """
    Generate human-readable explanation for BPJS reviewers.
    """
    reasons = []
    
    # Clinical mismatches (most critical)
    if row["mismatch_count"] > 0:
        mismatch_items = []
        if row["procedure_mismatch_flag"] == 1:
            mismatch_items.append("tindakan tidak sesuai diagnosis")
        if row["drug_mismatch_flag"] == 1:
            mismatch_items.append("obat tidak sesuai diagnosis")
        if row["vitamin_mismatch_flag"] == 1:
            mismatch_items.append("vitamin tidak relevan")
        
        reasons.append(f"Ketidaksesuaian klinis: {', '.join(mismatch_items)}")
    
    # Cost anomaly
    if row["biaya_anomaly_score"] >= 3:
        severity = "sangat tinggi" if row["biaya_anomaly_score"] == 4 else "tinggi"
        reasons.append(f"Biaya klaim {severity} untuk diagnosis ini (Rp {row['total_claim_amount']:,.0f})")
    
    # High frequency
    if row["patient_frequency_risk"] > 10:
        reasons.append(f"Frekuensi klaim mencurigakan ({row['patient_frequency_risk']} klaim)")
    
    # Risk level determination
    if fraud_score > 0.8:
        risk_level = "🔴 RISIKO TINGGI"
    elif fraud_score > 0.5:
        risk_level = "🟡 RISIKO SEDANG"
    elif fraud_score > 0.3:
        risk_level = "🟠 RISIKO RENDAH"
    else:
        risk_level = "🟢 RISIKO MINIMAL"
    
    if reasons:
        explanation = f"{risk_level}: " + "; ".join(reasons)
    else:
        explanation = f"{risk_level}: Tidak ada indikator fraud yang signifikan"
    
    return explanation


def get_recommendation(
    fraud_score: float, 
    mismatch_count: int, 
    cost_anomaly: int
) -> str:
    """Generate actionable recommendation for reviewer"""
    if fraud_score > 0.8:
        return "🚫 RECOMMENDED: Decline atau minta dokumentasi tambahan lengkap"
    elif fraud_score > 0.5:
        if mismatch_count > 0:
            return "⚠️ RECOMMENDED: Verifikasi justifikasi klinis dengan dokter"
        else:
            return "⚠️ RECOMMENDED: Review manual mendalam diperlukan"
    elif fraud_score > 0.3:
        return "📋 RECOMMENDED: Quick review, approve jika dokumen lengkap"
    else:
        return "✅ RECOMMENDED: Approve, tidak ada red flag"


def get_top_risk_factors(
    row: Dict[str, Any], 
    top_n: int = 5
) -> List[Dict[str, Any]]:
    """Identify top risk factors for this claim"""
    risk_factors = []
    
    # Get top features by importance
    top_features = list(feature_importance_map.items())[:top_n * 2]
    
    for feat_name, importance in top_features:
        if feat_name in row:
            value = row[feat_name]
            
            if isinstance(value, (int, float)):
                if feat_name.endswith("_flag") and value == 1:
                    interpretation = {
                        "procedure_mismatch_flag": "Tindakan tidak sesuai diagnosis",
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
                        "interpretation": f"Anomali biaya: {severity}"
                    })
            
            if len(risk_factors) >= top_n:
                break
    
    return risk_factors


# ================================================================
# INPUT VALIDATION
# ================================================================

def validate_input(data: Dict[str, Any]) -> tuple:
    """Validate input data from approval UI"""
    errors = []
    
    # Check if claims field exists
    if "claims" not in data:
        errors.append("Missing 'claims' field")
        return False, errors
    
    claims = data["claims"]
    
    if not isinstance(claims, list):
        errors.append("'claims' must be a list")
        return False, errors
    
    if len(claims) == 0:
        errors.append("'claims' cannot be empty")
        return False, errors
    
    # Validate each claim
    required_fields = [
        "claim_id", "patient_dob", "visit_date",
        "total_procedure_cost", "total_drug_cost", "total_vitamin_cost",
        "total_claim_amount", "icd10_primary_code", "department", "visit_type"
    ]
    
    for i, claim in enumerate(claims):
        missing = [f for f in required_fields if f not in claim]
        if missing:
            errors.append(f"Claim {i} (ID: {claim.get('claim_id', 'unknown')}): missing {missing}")
    
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
    Called by approval UI application.
    
    Input format:
    {
        "claims": [
            {
                "claim_id": "CLM001",
                "patient_dob": "1980-01-01",
                "visit_date": "2024-12-04",
                "visit_type": "rawat jalan",
                "department": "Poli Umum",
                "icd10_primary_code": "J06",
                "procedures": ["89.02"],
                "drugs": ["KFA001", "KFA009"],
                "vitamins": ["Vitamin C 500 mg"],
                "total_procedure_cost": 100000,
                "total_drug_cost": 50000,
                "total_vitamin_cost": 20000,
                "total_claim_amount": 170000
            }
        ]
    }
    
    Output format:
    {
        "status": "success",
        "model_version": "v2.0_production",
        "timestamp": "2024-12-04T10:30:00",
        "total_claims_processed": 1,
        "fraud_detected": 0,
        "results": [...]
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
    
    claims = data["claims"]
    
    try:
        # Process each claim
        processed_records = []
        claim_ids = []
        compatibility_data = []
        mismatch_data = []
        icd10_codes = []
        raw_claims = []
        
        for claim in claims:
            claim_id, feature_row, compatibility, mismatch = build_features_from_claim(claim)
            claim_ids.append(claim_id)
            processed_records.append(feature_row)
            compatibility_data.append(compatibility)
            mismatch_data.append(mismatch)
            icd10_codes.append(claim.get("icd10_primary_code", "UNKNOWN"))
            raw_claims.append(claim)
        
        # Build feature DataFrame and predict
        df_features, dmatrix = build_feature_dataframe(processed_records)
        
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
                risk_level = "HIGH"
                risk_color = "red"
            elif fraud_score > 0.5:
                risk_level = "MODERATE"
                risk_color = "orange"
            elif fraud_score > 0.3:
                risk_level = "LOW"
                risk_color = "yellow"
            else:
                risk_level = "MINIMAL"
                risk_color = "green"
            
            # Get detailed compatibility info
            raw_claim = raw_claims[i]
            procedures = raw_claim.get("procedures", [])
            drugs = raw_claim.get("drugs", [])
            vitamins = raw_claim.get("vitamins", [])
            
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
            explanation = generate_explanation(row, fraud_score, icd10_codes[i])
            
            # Top risk factors
            risk_factors = get_top_risk_factors(row, top_n=5)
            
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
                "fraud_flag": model_flag,
                "risk_level": risk_level,
                "risk_color": risk_color,
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
                },
                "patient_info": {
                    "age": int(row["patient_age"]),
                    "frequency_risk": int(row["patient_frequency_risk"])
                }
            })
        
        return {
            "status": "success",
            "model_version": model_meta.get("model_version", MODEL_VERSION),
            "model_name": MODEL_NAME,
            "timestamp": datetime.now().isoformat(),
            "total_claims_processed": len(results),
            "fraud_detected": sum(1 for r in results if r["fraud_flag"] == 1),
            "high_risk_count": sum(1 for r in results if r["risk_level"] == "HIGH"),
            "results": results,
            "model_info": {
                "threshold": best_threshold,
                "training_auc": model_meta.get("performance", {}).get("auc", 0),
                "training_f1": model_meta.get("performance", {}).get("f1", 0),
                "fraud_detection_rate": model_meta.get("performance", {}).get("fraud_detection_rate", 0),
            }
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc(),
            "timestamp": datetime.now().isoformat()
        }


# ================================================================
# BATCH PREDICTION ENDPOINT
# ================================================================

@models.cml_model
def predict_batch(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Batch prediction endpoint for processing multiple claims efficiently.
    Same as predict() but optimized for larger volumes.
    """
    return predict(data)


# ================================================================
# HEALTH CHECK ENDPOINT
# ================================================================

@models.cml_model
def health_check(data: Dict[str, Any] = None) -> Dict[str, Any]:
    """Health check endpoint for monitoring"""
    return {
        "status": "healthy",
        "model_name": MODEL_NAME,
        "model_version": model_meta.get("model_version", MODEL_VERSION),
        "timestamp": datetime.now().isoformat(),
        "features_count": len(feature_names),
        "threshold": best_threshold,
        "clinical_rules_loaded": len(CLINICAL_RULES),
        "training_metrics": {
            "auc": model_meta.get("performance", {}).get("auc", 0),
            "f1": model_meta.get("performance", {}).get("f1", 0),
            "fraud_detection_rate": model_meta.get("performance", {}).get("fraud_detection_rate", 0),
        }
    }


# ================================================================
# MODEL INFO ENDPOINT
# ================================================================

@models.cml_model
def get_model_info(data: Dict[str, Any] = None) -> Dict[str, Any]:
    """Return detailed model metadata for UI"""
    return {
        "status": "success",
        "model_name": MODEL_NAME,
        "model_metadata": model_meta,
        "feature_importance": GLOBAL_FEATURE_IMPORTANCE[:20],
        "clinical_rules": {
            "total_diagnoses": len(CLINICAL_RULES),
            "supported_diagnoses": list(CLINICAL_RULES.keys()),
        },
        "capabilities": [
            "Real-time fraud detection",
            "Clinical compatibility checking",
            "Cost anomaly detection",
            "Patient frequency analysis",
            "Detailed explanations for reviewers"
        ]
    }


# ================================================================
# STARTUP MESSAGE
# ================================================================

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print(f"{MODEL_NAME} - INFERENCE API READY FOR CML DEPLOYMENT")
    print("=" * 80)
    print("\n📡 Available endpoints:")
    print("  1. predict() - Main fraud detection for single/multiple claims")
    print("  2. predict_batch() - Optimized batch processing")
    print("  3. health_check() - Model health and status")
    print("  4. get_model_info() - Model metadata and capabilities")
    print("\n🎯 Model capabilities:")
    print("  ✓ Real-time fraud score prediction (0-100%)")
    print("  ✓ Clinical compatibility checking (diagnosis vs treatment)")
    print("  ✓ Cost anomaly detection")
    print("  ✓ Patient frequency risk analysis")
    print("  ✓ Actionable recommendations for reviewers")
    print("  ✓ Detailed explanations in Bahasa Indonesia")
    print("\n🔗 Integration:")
    print("  - Approval UI application")
    print("  - Iceberg reference tables")
    print("  - BPJS claim workflow")
    print("\n" + "=" * 80)