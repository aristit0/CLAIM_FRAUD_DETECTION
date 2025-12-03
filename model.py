#!/usr/bin/env python3
"""
Production Fraud Detection Model - Inference API
Consistent with ETL output and training pipeline
Version: 2.0 - Iceberg integrated
"""
import os
import json
import pickle
import traceback
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Tuple

# ------------------------------------------------------------------
# IMPORTS & CONFIGURATION
# ------------------------------------------------------------------

from config import (
    COMPAT_RULES_FALLBACK as COMPAT_RULES,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
)

# Artifact paths
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_JSON = os.path.join(MODEL_DIR, "model.json")
CALIB_FILE = os.path.join(MODEL_DIR, "calibrator.pkl")
PREPROCESS_FILE = os.path.join(MODEL_DIR, "preprocess.pkl")
META_FILE = os.path.join(MODEL_DIR, "meta.json")

# Global state (lazy-loaded)
booster = None
calibrator = None
preprocess = None
model_meta = None
numeric_cols: List[str] = list(NUMERIC_FEATURES)
categorical_cols: List[str] = list(CATEGORICAL_FEATURES)
encoders: Dict[str, Any] = {}
best_threshold: float = 0.5
feature_importance_map: Dict[str, float] = {}
feature_names: List[str] = []
GLOBAL_FEATURE_IMPORTANCE: List[Dict[str, Any]] = []
LOAD_ERROR: str = ""
MODEL_LOADED: bool = False


def _ensure_model_loaded() -> bool:
    """
    Lazy-load model artifacts on first use.
    Returns True if loaded successfully, False otherwise.
    """
    global booster, calibrator, preprocess, model_meta
    global numeric_cols, categorical_cols, encoders
    global best_threshold, feature_importance_map, feature_names
    global GLOBAL_FEATURE_IMPORTANCE, LOAD_ERROR, MODEL_LOADED

    if MODEL_LOADED:
        return True

    try:
        # Late import of XGBoost
        try:
            import xgboost as xgb
        except ImportError:
            raise ImportError("XGBoost tidak terinstall. pip install xgboost")

        # Load XGBoost model
        booster_local = xgb.Booster()
        booster_local.load_model(MODEL_JSON)

        # Load calibrator
        with open(CALIB_FILE, "rb") as f:
            calibrator_local = pickle.load(f)

        # Load preprocessing config
        with open(PREPROCESS_FILE, "rb") as f:
            preprocess_local = pickle.load(f)

        # Load metadata
        with open(META_FILE, "r") as f:
            model_meta_local = json.load(f)

        # Extract from preprocess dict (trained)
        numeric_cols_local = preprocess_local.get("numeric_cols", NUMERIC_FEATURES)
        categorical_cols_local = preprocess_local.get("categorical_cols", CATEGORICAL_FEATURES)
        encoders_local = preprocess_local.get("encoders", {})
        best_threshold_local = preprocess_local.get("best_threshold", 0.5)
        feature_importance_map_local = preprocess_local.get("feature_importance", {})

        feature_names_local = numeric_cols_local + categorical_cols_local

        global_feature_importance = [
            {"feature": k, "importance": float(v)}
            for k, v in sorted(
                feature_importance_map_local.items(),
                key=lambda kv: kv[1],
                reverse=True,
            )
        ]

        # Assign globals
        booster = booster_local
        calibrator = calibrator_local
        preprocess = preprocess_local
        model_meta = model_meta_local
        numeric_cols = numeric_cols_local
        categorical_cols = categorical_cols_local
        encoders = encoders_local
        best_threshold = best_threshold_local
        feature_importance_map = feature_importance_map_local
        feature_names = feature_names_local
        GLOBAL_FEATURE_IMPORTANCE[:] = global_feature_importance
        LOAD_ERROR = ""
        MODEL_LOADED = True

        return True

    except Exception as e:
        LOAD_ERROR = f"{type(e).__name__}: {str(e)}"
        return False


# ================================================================
# UTILITY FUNCTIONS - MUST MATCH ETL
# ================================================================

def compute_age(dob: str, visit_date: str) -> int:
    """Calculate patient age at visit date (exact match with ETL)"""
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
    vitamins: List[str],
) -> Dict[str, float]:
    """
    Calculate clinical compatibility scores (EXACT match with ETL).
    Uses reference rules to compare diagnosis vs treatments.
    """
    rules = COMPAT_RULES.get(icd10)

    if not rules:
        # Unknown diagnosis - neutral score
        return {
            "diagnosis_procedure_score": 0.5,
            "diagnosis_drug_score": 0.5,
            "diagnosis_vitamin_score": 0.5,
        }

    allowed_procedures = rules.get("procedures", [])
    allowed_drugs = rules.get("drugs", [])
    allowed_vitamins = rules.get("vitamins", [])

    # Procedure compatibility
    proc_score = 0.5
    if procedures and allowed_procedures:
        proc_matches = sum(1 for p in procedures if p in allowed_procedures)
        proc_score = proc_matches / len(procedures)

    # Drug compatibility
    drug_score = 0.5
    if drugs and allowed_drugs:
        drug_matches = sum(1 for d in drugs if d in allowed_drugs)
        drug_score = drug_matches / len(drugs)

    # Vitamin compatibility
    vit_score = 0.5
    if vitamins and allowed_vitamins:
        vit_matches = sum(1 for v in vitamins if v in allowed_vitamins)
        vit_score = vit_matches / len(vitamins)

    return {
        "diagnosis_procedure_score": float(proc_score),
        "diagnosis_drug_score": float(drug_score),
        "diagnosis_vitamin_score": float(vit_score),
    }


def compute_mismatch_flags(compatibility_scores: Dict[str, float]) -> Dict[str, int]:
    """
    Convert compatibility scores to binary flags (EXACT match with ETL).
    Score < 0.5 = mismatch = 1
    """
    proc_flag = 1 if compatibility_scores["diagnosis_procedure_score"] < 0.5 else 0
    drug_flag = 1 if compatibility_scores["diagnosis_drug_score"] < 0.5 else 0
    vit_flag = 1 if compatibility_scores["diagnosis_vitamin_score"] < 0.5 else 0

    return {
        "procedure_mismatch_flag": proc_flag,
        "drug_mismatch_flag": drug_flag,
        "vitamin_mismatch_flag": vit_flag,
        "mismatch_count": proc_flag + drug_flag + vit_flag,
    }


def compute_cost_anomaly_score(total_claim: float, icd10: str = None) -> int:
    """
    Simple cost anomaly scoring (EXACT match with ETL).
    Returns 1-4 scale.
    """
    if total_claim > 1_500_000:
        return 4
    if total_claim > 1_000_000:
        return 3
    if total_claim > 500_000:
        return 2
    return 1


def get_compatibility_details(
    icd10: str,
    procedures: List[str],
    drugs: List[str],
    vitamins: List[str],
) -> Dict[str, Any]:
    """
    Return detailed compatibility information for UI display.
    """
    rules = COMPAT_RULES.get(icd10)

    if not rules:
        return {
            "diagnosis_known": False,
            "diagnosis_description": "Diagnosis tidak ada di clinical rules",
            "procedure_details": [],
            "drug_details": [],
            "vitamin_details": [],
        }

    allowed_procedures = rules.get("procedures", [])
    allowed_drugs = rules.get("drugs", [])
    allowed_vitamins = rules.get("vitamins", [])

    procedure_details = [
        {
            "code": proc,
            "compatible": proc in allowed_procedures,
            "status": "✓ Compatible" if proc in allowed_procedures else "✗ Incompatible",
        }
        for proc in procedures
    ]

    drug_details = [
        {
            "code": drug,
            "compatible": drug in allowed_drugs,
            "status": "✓ Compatible" if drug in allowed_drugs else "✗ Incompatible",
        }
        for drug in drugs
    ]

    vitamin_details = [
        {
            "name": vit,
            "compatible": vit in allowed_vitamins,
            "status": "✓ Compatible" if vit in allowed_vitamins else "✗ Incompatible",
        }
        for vit in vitamins
    ]

    return {
        "diagnosis_known": True,
        "diagnosis_description": rules.get("description", ""),
        "procedure_details": procedure_details,
        "drug_details": drug_details,
        "vitamin_details": vitamin_details,
    }


# ================================================================
# FEATURE ENGINEERING (MUST MATCH ETL EXACTLY)
# ================================================================

def build_features_from_raw(raw: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """
    Transform raw claim data to feature row.
    OUTPUT MUST MATCH ETL: claude_etl_claim_feature_set.py
    """
    claim_id = raw.get("claim_id")
    visit_date = raw.get("visit_date")
    dt = datetime.strptime(visit_date, "%Y-%m-%d").date()

    # Handle list inputs
    procedures = raw.get("procedures", [])
    drugs = raw.get("drugs", [])
    vitamins = raw.get("vitamins", [])

    if not isinstance(procedures, list):
        procedures = [procedures] if procedures else []
    if not isinstance(drugs, list):
        drugs = [drugs] if drugs else []
    if not isinstance(vitamins, list):
        vitamins = [vitamins] if vitamins else []

    # Cost fields
    total_proc = float(raw.get("total_procedure_cost", 0))
    total_drug = float(raw.get("total_drug_cost", 0))
    total_vit = float(raw.get("total_vitamin_cost", 0))
    total_claim = float(raw.get("total_claim_amount", 0))

    # Patient age
    patient_age = compute_age(raw.get("patient_dob"), visit_date)

    # Diagnosis
    icd10 = raw.get("icd10_primary_code", "UNKNOWN")

    # Clinical compatibility (FROM ETL)
    compatibility = compute_compatibility_scores(icd10, procedures, drugs, vitamins)
    mismatch = compute_mismatch_flags(compatibility)
    biaya_anomaly = compute_cost_anomaly_score(total_claim, icd10)

    # Patient frequency (placeholder - should come from database in production)
    patient_freq = raw.get("patient_frequency_risk", 1)

    # BUILD FEATURE ROW (EXACT ORDER MATCHING ETL)
    feature_row = {
        # ===== NUMERIC FEATURES (from NUMERIC_FEATURES config) =====
        "patient_age": patient_age,
        "visit_year": dt.year,
        "visit_month": dt.month,
        "visit_day": dt.day,
        "total_procedure_cost": total_proc,
        "total_drug_cost": total_drug,
        "total_vitamin_cost": total_vit,
        "total_claim_amount": total_claim,
        "diagnosis_procedure_score": compatibility["diagnosis_procedure_score"],
        "diagnosis_drug_score": compatibility["diagnosis_drug_score"],
        "diagnosis_vitamin_score": compatibility["diagnosis_vitamin_score"],
        "procedure_mismatch_flag": mismatch["procedure_mismatch_flag"],
        "drug_mismatch_flag": mismatch["drug_mismatch_flag"],
        "vitamin_mismatch_flag": mismatch["vitamin_mismatch_flag"],
        "mismatch_count": mismatch["mismatch_count"],
        "biaya_anomaly_score": biaya_anomaly,
        "patient_frequency_risk": patient_freq,
        
        # ===== CATEGORICAL FEATURES (from CATEGORICAL_FEATURES config) =====
        "visit_type": raw.get("visit_type", "UNKNOWN"),
        "department": raw.get("department", "UNKNOWN"),
        "icd10_primary_code": icd10,
    }

    return claim_id, feature_row


def build_feature_df(records: List[Dict[str, Any]]) -> Tuple[pd.DataFrame, Any]:
    """
    Build DataFrame and XGBoost DMatrix from feature rows.
    Matches training pipeline exactly.
    """
    _ensure_model_loaded()

    if not MODEL_LOADED:
        raise RuntimeError(f"Model tidak ter-load: {LOAD_ERROR}")

    try:
        from xgboost import DMatrix
    except ImportError:
        raise ImportError("XGBoost tidak terinstall")

    # Create DataFrame
    df = pd.DataFrame.from_records(records)

    # Ensure all columns exist (EXACT MATCH with training)
    for col_name in numeric_cols + categorical_cols:
        if col_name not in df.columns:
            if col_name in numeric_cols:
                df[col_name] = 0.0
            else:
                df[col_name] = "UNKNOWN"

    # Encode categorical using trained encoders
    for col_name in categorical_cols:
        df[col_name] = df[col_name].astype(str).fillna("UNKNOWN")
        
        if col_name in encoders:
            enc = encoders[col_name]
            try:
                # TargetEncoder transform
                df[col_name] = enc.transform(df[[col_name]])[col_name]
            except Exception as e:
                # If transform fails, use default encoding
                df[col_name] = 0.0

    # Clean numeric columns
    for col_name in numeric_cols:
        df[col_name] = pd.to_numeric(df[col_name], errors="coerce").fillna(0.0)
        df[col_name] = df[col_name].replace([np.inf, -np.inf], 0)

    # Select features in correct order (MATCHES TRAINING)
    X = df[numeric_cols + categorical_cols]
    
    # Create DMatrix
    dmatrix = DMatrix(X, feature_names=feature_names)

    return df, dmatrix


# ================================================================
# EXPLANATIONS & RECOMMENDATIONS
# ================================================================

def generate_explanation(
    row: Dict[str, Any],
    fraud_score: float,
    icd10: str,
) -> str:
    """
    Generate human-readable fraud explanation.
    """
    reasons: List[str] = []

    # Clinical mismatches
    if row.get("mismatch_count", 0) > 0:
        mismatch_items = []
        if row.get("procedure_mismatch_flag") == 1:
            mismatch_items.append("prosedur")
        if row.get("drug_mismatch_flag") == 1:
            mismatch_items.append("obat")
        if row.get("vitamin_mismatch_flag") == 1:
            mismatch_items.append("vitamin")
        reasons.append(f"Ketidaksesuaian klinis: {', '.join(mismatch_items)}")

    # Cost anomalies
    if row.get("biaya_anomaly_score", 0) >= 3:
        severity = "sangat tinggi" if row.get("biaya_anomaly_score") == 4 else "tinggi"
        reasons.append(f"Biaya klaim {severity}")

    # Frequency risk
    if row.get("patient_frequency_risk", 0) > 10:
        reasons.append("Frekuensi klaim pasien tinggi")

    # Determine risk level
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
        explanation = "{} {}: {}".format(color, risk_level, "; ".join(reasons))
    else:
        explanation = "{} {}: Tidak ada indikator fraud".format(color, risk_level)

    return explanation


def get_top_risk_factors(
    row: Dict[str, Any],
    top_n: int = 5,
) -> List[Dict[str, Any]]:
    """
    Get top N risk factors from feature importance.
    """
    risk_factors: List[Dict[str, Any]] = []

    for feat_name, importance in list(feature_importance_map.items())[:top_n * 3]:
        if feat_name not in row:
            continue

        feat_value = row[feat_name]
        risk_factors.append({
            "feature": feat_name,
            "value": float(feat_value) if isinstance(feat_value, (int, float)) else feat_value,
            "importance": float(importance),
        })

    return risk_factors[:top_n]


def get_recommendation(
    fraud_score: float,
    mismatch_count: int,
    cost_anomaly: int,
) -> str:
    """
    Get action recommendation for reviewer.
    """
    if fraud_score > 0.8:
        return "RECOMMENDED: REJECT - Tingkat fraud tinggi, perlu investigasi"
    if fraud_score > 0.5:
        return "RECOMMENDED: REVIEW - Periksa detail, ada indikator mencurigakan"
    if mismatch_count > 0:
        return "RECOMMENDED: CLARIFY - Ada ketidaksesuaian klinis"
    if cost_anomaly >= 3:
        return "RECOMMENDED: VERIFY - Biaya tinggi, verifikasi dengan provider"
    return "RECOMMENDED: APPROVE - Sesuai standar BPJS"


# ================================================================
# VALIDATION
# ================================================================

def validate_input(data: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate input data before inference.
    """
    errors: List[str] = []

    if "raw_records" not in data:
        errors.append("Field 'raw_records' tidak ditemukan")
        return False, errors

    raw_records = data.get("raw_records")

    if not isinstance(raw_records, list):
        errors.append("Field 'raw_records' harus berupa list")
        return False, errors

    if len(raw_records) == 0:
        errors.append("Field 'raw_records' tidak boleh kosong")
        return False, errors

    required_fields = [
        "claim_id",
        "patient_dob",
        "visit_date",
        "total_procedure_cost",
        "total_drug_cost",
        "total_vitamin_cost",
        "total_claim_amount",
        "icd10_primary_code",
        "department",
        "visit_type",
    ]

    for i, rec in enumerate(raw_records):
        for field in required_fields:
            if field not in rec:
                errors.append(f"Record {i}: Field '{field}' tidak ditemukan")

    if errors:
        return False, errors

    return True, []


# ================================================================
# MAIN INFERENCE
# ================================================================

def predict(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main prediction endpoint.
    Input: {"raw_records": [...]}
    Output: Fraud scores, explanations, recommendations
    """
    _ensure_model_loaded()

    if not MODEL_LOADED:
        return {
            "status": "error",
            "message": f"Model tidak ter-load: {LOAD_ERROR}",
            "predictions": [],
        }

    # Validate
    is_valid, errors = validate_input(data)
    if not is_valid:
        return {
            "status": "error",
            "message": "Input validation failed",
            "errors": errors,
            "predictions": [],
        }

    raw_records = data.get("raw_records", [])

    try:
        # Build features
        feature_rows = []
        claim_ids = []

        for raw in raw_records:
            claim_id, feature_row = build_features_from_raw(raw)
            feature_rows.append(feature_row)
            claim_ids.append(claim_id)

        # Build DMatrix
        df, dmatrix = build_feature_df(feature_rows)

        # Predict
        raw_scores = booster.predict(dmatrix)
        calibrated_scores = calibrator.predict_proba(raw_scores.reshape(-1, 1))[:, 1]

        # Build predictions
        predictions = []

        for idx, raw in enumerate(raw_records):
            claim_id = raw.get("claim_id")
            icd10 = raw.get("icd10_primary_code", "UNKNOWN")
            procedures = raw.get("procedures", [])
            drugs = raw.get("drugs", [])
            vitamins = raw.get("vitamins", [])

            fraud_score = float(calibrated_scores[idx])
            is_fraud = 1 if fraud_score > best_threshold else 0

            # Get compatibility details
            compat_details = get_compatibility_details(icd10, procedures, drugs, vitamins)

            # Generate explanation
            feature_row = feature_rows[idx]
            explanation = generate_explanation(feature_row, fraud_score, icd10)

            # Get top risk factors
            top_factors = get_top_risk_factors(feature_row, top_n=5)

            # Get recommendation
            recommendation = get_recommendation(
                fraud_score,
                feature_row.get("mismatch_count", 0),
                feature_row.get("biaya_anomaly_score", 1),
            )

            predictions.append({
                "claim_id": claim_id,
                "fraud_score": fraud_score,
                "is_fraud": is_fraud,
                "fraud_threshold": best_threshold,
                "explanation": explanation,
                "recommendation": recommendation,
                "compatibility_details": compat_details,
                "top_risk_factors": top_factors,
            })

        return {
            "status": "success",
            "message": "Prediction completed",
            "model_version": model_meta.get("model_version", "unknown") if model_meta else "unknown",
            "predictions": predictions,
        }

    except Exception as e:
        return {
            "status": "error",
            "message": f"Prediction failed: {str(e)}",
            "predictions": [],
            "error_details": traceback.format_exc(),
        }


# ================================================================
# HEALTH CHECK
# ================================================================

def health_check(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Health check endpoint for monitoring.
    """
    _ensure_model_loaded()

    if not MODEL_LOADED:
        return {
            "status": "unhealthy",
            "model_loaded": False,
            "error": LOAD_ERROR,
        }

    return {
        "status": "healthy",
        "model_loaded": True,
        "model_version": model_meta.get("model_version", "unknown") if model_meta else "unknown",
        "features_count": len(feature_names),
        "numeric_features": len(numeric_cols),
        "categorical_features": len(categorical_cols),
        "threshold": best_threshold,
    }


# ================================================================
# MODEL INFO
# ================================================================

def get_model_info(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get model information and feature importance.
    """
    _ensure_model_loaded()

    if not MODEL_LOADED:
        return {
            "status": "error",
            "message": f"Model tidak ter-load: {LOAD_ERROR}",
        }

    return {
        "status": "success",
        "model_version": model_meta.get("model_version", "unknown") if model_meta else "unknown",
        "model_type": "XGBoost",
        "features": {
            "numeric": numeric_cols,
            "categorical": categorical_cols,
            "total_count": len(feature_names),
        },
        "threshold": best_threshold,
        "feature_importance": GLOBAL_FEATURE_IMPORTANCE,
        "metadata": model_meta if model_meta else {},
    }


def create_app():
    """
    Create Flask application for model serving.
    """
    from flask import Flask, request, jsonify
    
    app = Flask(__name__)
    
    @app.route("/health", methods=["GET"])
    def health_endpoint():
        """Health check endpoint"""
        return jsonify(health_check({})), 200
    
    @app.route("/predict", methods=["POST"])
    def predict_endpoint():
        """Prediction endpoint"""
        try:
            data = request.get_json()
            if not data:
                return jsonify({
                    "status": "error",
                    "message": "Request body harus JSON"
                }), 400
            
            result = predict(data)
            status_code = 200 if result["status"] == "success" else 400
            return jsonify(result), status_code
        
        except Exception as e:
            return jsonify({
                "status": "error",
                "message": str(e),
                "error_details": traceback.format_exc()
            }), 500
    
    @app.route("/info", methods=["GET"])
    def info_endpoint():
        """Model info endpoint"""
        result = get_model_info({})
        status_code = 200 if result["status"] == "success" else 400
        return jsonify(result), status_code
    
    @app.route("/validate", methods=["POST"])
    def validate_endpoint():
        """Validate input without prediction"""
        try:
            data = request.get_json()
            if not data:
                return jsonify({
                    "status": "error",
                    "message": "Request body harus JSON"
                }), 400
            
            is_valid, errors = validate_input(data)
            return jsonify({
                "status": "success" if is_valid else "error",
                "is_valid": is_valid,
                "errors": errors
            }), 200
        
        except Exception as e:
            return jsonify({
                "status": "error",
                "message": str(e)
            }), 500
    
    return app


# ================================================================
# BATCH PREDICTION (FOR OFFLINE PROCESSING)
# ================================================================

def predict_batch(records_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Batch prediction helper function.
    Useful for processing multiple claims at once.
    
    Args:
        records_list: List of raw claim records
    
    Returns:
        Dictionary with predictions for all records
    """
    return predict({"raw_records": records_list})


def predict_from_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Predict from a pandas DataFrame.
    
    Args:
        df: DataFrame with columns matching raw record format
    
    Returns:
        DataFrame with predictions appended
    """
    records = df.to_dict(orient="records")
    result = predict_batch(records)
    
    if result["status"] != "success":
        raise RuntimeError(f"Prediction failed: {result['message']}")
    
    # Convert predictions to DataFrame
    predictions_df = pd.DataFrame(result["predictions"])
    
    # Merge dengan original data
    output_df = pd.concat([df.reset_index(drop=True), predictions_df], axis=1)
    
    return output_df


# ================================================================
# EXPLAIN SINGLE PREDICTION
# ================================================================

def explain_prediction(claim_id: str, raw_record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get detailed explanation for a single claim prediction.
    
    Args:
        claim_id: Claim ID for logging
        raw_record: Raw claim record
    
    Returns:
        Dictionary with detailed explanation
    """
    _ensure_model_loaded()
    
    if not MODEL_LOADED:
        return {
            "status": "error",
            "message": f"Model tidak ter-load: {LOAD_ERROR}"
        }
    
    try:
        # Build features
        _, feature_row = build_features_from_raw(raw_record)
        
        # Predict
        df_single = pd.DataFrame([feature_row])
        
        # Ensure all columns
        for col_name in numeric_cols + categorical_cols:
            if col_name not in df_single.columns:
                if col_name in numeric_cols:
                    df_single[col_name] = 0.0
                else:
                    df_single[col_name] = "UNKNOWN"
        
        # Encode categorical
        for col_name in categorical_cols:
            df_single[col_name] = df_single[col_name].astype(str).fillna("UNKNOWN")
            if col_name in encoders:
                enc = encoders[col_name]
                try:
                    df_single[col_name] = enc.transform(df_single[[col_name]])[col_name]
                except Exception:
                    df_single[col_name] = 0.0
        
        # Clean numeric
        for col_name in numeric_cols:
            df_single[col_name] = pd.to_numeric(df_single[col_name], errors="coerce").fillna(0.0)
        
        # Predict
        from xgboost import DMatrix
        X = df_single[numeric_cols + categorical_cols]
        dmatrix = DMatrix(X, feature_names=feature_names)
        
        raw_score = booster.predict(dmatrix)[0]
        calibrated_score = calibrator.predict_proba(np.array([[raw_score]]))[0, 1]
        
        icd10 = raw_record.get("icd10_primary_code", "UNKNOWN")
        procedures = raw_record.get("procedures", [])
        drugs = raw_record.get("drugs", [])
        vitamins = raw_record.get("vitamins", [])
        
        if not isinstance(procedures, list):
            procedures = [procedures] if procedures else []
        if not isinstance(drugs, list):
            drugs = [drugs] if drugs else []
        if not isinstance(vitamins, list):
            vitamins = [vitamins] if vitamins else []
        
        compat_details = get_compatibility_details(icd10, procedures, drugs, vitamins)
        explanation = generate_explanation(feature_row, calibrated_score, icd10)
        top_factors = get_top_risk_factors(feature_row, top_n=10)
        recommendation = get_recommendation(
            calibrated_score,
            feature_row.get("mismatch_count", 0),
            feature_row.get("biaya_anomaly_score", 1)
        )
        
        return {
            "status": "success",
            "claim_id": claim_id,
            "fraud_score": float(calibrated_score),
            "raw_score": float(raw_score),
            "is_fraud": 1 if calibrated_score > best_threshold else 0,
            "threshold": best_threshold,
            "explanation": explanation,
            "recommendation": recommendation,
            "compatibility_details": compat_details,
            "top_risk_factors": top_factors,
            "feature_values": {k: float(v) if isinstance(v, (int, float, np.number)) else v 
                              for k, v in feature_row.items()},
        }
    
    except Exception as e:
        return {
            "status": "error",
            "claim_id": claim_id,
            "message": str(e),
            "error_details": traceback.format_exc()
        }


# ================================================================
# COMPARE PREDICTIONS (FOR TESTING/MONITORING)
# ================================================================

def compare_predictions(
    claim_id: str,
    raw_record: Dict[str, Any],
    expected_fraud: int = None
) -> Dict[str, Any]:
    """
    Get prediction dengan comparison ke expected label.
    Useful untuk monitoring model performance di production.
    
    Args:
        claim_id: Claim ID
        raw_record: Raw claim record
        expected_fraud: Expected label (0 atau 1) untuk comparison
    
    Returns:
        Dictionary dengan prediction + comparison metrics
    """
    explanation = explain_prediction(claim_id, raw_record)
    
    if explanation["status"] != "success":
        return explanation
    
    fraud_score = explanation["fraud_score"]
    predicted_fraud = explanation["is_fraud"]
    
    result = {
        **explanation,
        "expected_fraud": expected_fraud,
        "prediction_correct": None,
        "metrics": {}
    }
    
    if expected_fraud is not None:
        is_correct = (predicted_fraud == expected_fraud)
        result["prediction_correct"] = is_correct
        
        result["metrics"] = {
            "true_positive": predicted_fraud == 1 and expected_fraud == 1,
            "true_negative": predicted_fraud == 0 and expected_fraud == 0,
            "false_positive": predicted_fraud == 1 and expected_fraud == 0,
            "false_negative": predicted_fraud == 0 and expected_fraud == 1,
            "confidence": max(fraud_score, 1 - fraud_score)
        }
    
    return result


# ================================================================
# MONITORING & ANALYTICS
# ================================================================

def get_prediction_stats(
    predictions_list: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Get statistics dari list of predictions.
    Useful untuk monitoring model performance.
    """
    if not predictions_list:
        return {
            "status": "error",
            "message": "Predictions list kosong"
        }
    
    fraud_scores = [p.get("fraud_score", 0) for p in predictions_list]
    is_frauds = [p.get("is_fraud", 0) for p in predictions_list]
    
    return {
        "status": "success",
        "total_predictions": len(predictions_list),
        "fraud_detected": sum(is_frauds),
        "normal_claims": len(predictions_list) - sum(is_frauds),
        "fraud_rate": sum(is_frauds) / len(predictions_list) if predictions_list else 0,
        "fraud_score_stats": {
            "mean": float(np.mean(fraud_scores)),
            "median": float(np.median(fraud_scores)),
            "min": float(np.min(fraud_scores)),
            "max": float(np.max(fraud_scores)),
            "std": float(np.std(fraud_scores)),
            "percentiles": {
                "p25": float(np.percentile(fraud_scores, 25)),
                "p50": float(np.percentile(fraud_scores, 50)),
                "p75": float(np.percentile(fraud_scores, 75)),
                "p95": float(np.percentile(fraud_scores, 95)),
                "p99": float(np.percentile(fraud_scores, 99)),
            }
        },
        "threshold": best_threshold
    }


def get_feature_contribution(
    raw_record: Dict[str, Any],
    top_n: int = 10
) -> Dict[str, Any]:
    """
    Get feature contribution untuk specific claim.
    Menggunakan SHAP values (approximation menggunakan feature importance).
    """
    _, feature_row = build_features_from_raw(raw_record)
    
    contributions = []
    for feat_name, importance in sorted(
        feature_importance_map.items(),
        key=lambda x: x[1],
        reverse=True
    )[:top_n]:
        if feat_name in feature_row:
            contributions.append({
                "feature": feat_name,
                "value": float(feature_row[feat_name]) if isinstance(feature_row[feat_name], (int, float, np.number)) else feature_row[feat_name],
                "importance": float(importance),
                "contribution": float(importance) * (feature_row[feat_name] if isinstance(feature_row[feat_name], (int, float, np.number)) else 0)
            })
    
    return {
        "status": "success",
        "feature_contributions": contributions,
        "note": "Contribution = Importance × Feature Value (approximation)"
    }


# ================================================================
# EXPORT & SERIALIZATION
# ================================================================

def export_predictions_to_csv(
    predictions: List[Dict[str, Any]],
    filepath: str
) -> Dict[str, Any]:
    """
    Export predictions ke CSV file.
    """
    try:
        df = pd.DataFrame(predictions)
        
        # Flatten nested dictionaries
        if "compatibility_details" in df.columns:
            df = df.drop("compatibility_details", axis=1)
        if "top_risk_factors" in df.columns:
            df = df.drop("top_risk_factors", axis=1)
        
        df.to_csv(filepath, index=False)
        
        return {
            "status": "success",
            "message": f"Exported {len(df)} predictions",
            "filepath": filepath,
            "records_count": len(df)
        }
    
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "error_details": traceback.format_exc()
        }


def export_predictions_to_json(
    predictions: List[Dict[str, Any]],
    filepath: str
) -> Dict[str, Any]:
    """
    Export predictions ke JSON file.
    """
    try:
        with open(filepath, "w") as f:
            json.dump(predictions, f, indent=2)
        
        return {
            "status": "success",
            "message": f"Exported {len(predictions)} predictions",
            "filepath": filepath,
            "records_count": len(predictions)
        }
    
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "error_details": traceback.format_exc()
        }


# ================================================================
# MAIN ENTRYPOINT
# ================================================================

if __name__ == "__main__":
    import sys
    
    # Parse command line arguments
    mode = sys.argv[1] if len(sys.argv) > 1 else "server"
    
    if mode == "server":
        # Start Flask server
        print("\n" + "="*80)
        print("FRAUD DETECTION MODEL - FLASK SERVER")
        print("="*80)
        
        app = create_app()
        
        port = int(sys.argv[2]) if len(sys.argv) > 2 else 5000
        debug = sys.argv[3].lower() == "true" if len(sys.argv) > 3 else False
        
        print(f"\n[INFO] Starting server on port {port}")
        print(f"[INFO] Debug mode: {debug}")
        print("\nEndpoints:")
        print("  GET  /health  - Health check")
        print("  POST /predict - Fraud prediction")
        print("  GET  /info    - Model info")
        print("  POST /validate - Validate input")
        print("\n" + "="*80 + "\n")
        
        app.run(
            host="0.0.0.0",
            port=port,
            debug=debug,
            threaded=True
        )
    
    elif mode == "test":
        # Test mode
        print("\n" + "="*80)
        print("FRAUD DETECTION MODEL - TEST MODE")
        print("="*80 + "\n")
        
        _ensure_model_loaded()
        
        if not MODEL_LOADED:
            print(f"[ERROR] Model tidak ter-load: {LOAD_ERROR}")
            sys.exit(1)
        
        print("[INFO] Model loaded successfully")
        print(f"[INFO] Features: {len(feature_names)} ({len(numeric_cols)} numeric, {len(categorical_cols)} categorical)")
        print(f"[INFO] Threshold: {best_threshold}")
        
        # Create sample test record
        sample_record = {
            "claim_id": "CLAIM-TEST-001",
            "patient_dob": "1985-05-15",
            "visit_date": "2025-02-15",
            "icd10_primary_code": "E11",
            "visit_type": "rawat jalan",
            "department": "internal medicine",
            "total_procedure_cost": 150000.0,
            "total_drug_cost": 200000.0,
            "total_vitamin_cost": 50000.0,
            "total_claim_amount": 400000.0,
            "procedures": ["90.99"],
            "drugs": ["C09AA"],
            "vitamins": ["Vitamin C"],
            "patient_frequency_risk": 2
        }
        
        print("\n[TEST] Running prediction on sample record...")
        result = predict({"raw_records": [sample_record]})
        
        print(f"\n[RESULT] Status: {result['status']}")
        if result["status"] == "success":
            pred = result["predictions"][0]
            print(f"  Fraud Score: {pred['fraud_score']:.4f}")
            print(f"  Is Fraud: {pred['is_fraud']}")
            print(f"  Explanation: {pred['explanation']}")
            print(f"  Recommendation: {pred['recommendation']}")
        else:
            print(f"  Error: {result['message']}")
        
        print("\n" + "="*80 + "\n")
    
    else:
        print(f"Usage: python model.py [server|test] [port] [debug]")
        print(f"  - server (default): Start Flask API server")
        print(f"  - test: Run test prediction")
        print(f"  - port: Port number (default: 5000)")
        print(f"  - debug: Debug mode true/false (default: false)")
        sys.exit(1)