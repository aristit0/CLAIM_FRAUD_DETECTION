#!/usr/bin/env python3
"""
Fraud Detection Model for Health Claims - Production Inference API
Provides fraud scoring for claim reviewers and clinical compatibility checks.
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
# IMPORT CONFIG & GLOBALS
# ------------------------------------------------------------------

from config import (
    COMPAT_RULES_FALLBACK as COMPAT_RULES,
    FRAUD_PATTERNS,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
)

# Lokasi artefak relatif ke file ini
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_JSON = os.path.join(MODEL_DIR, "model.json")
CALIB_FILE = os.path.join(MODEL_DIR, "calibrator.pkl")
PREPROCESS_FILE = os.path.join(MODEL_DIR, "preprocess.pkl")
META_FILE = os.path.join(MODEL_DIR, "meta.json")

print("=" * 80)
print("CLAIM FRAUD DETECTION MODEL - MODULE IMPORT")
print("=" * 80)

# Global variables di-init None, nanti di-load lazy
booster = None
calibrator = None
preprocess = None
model_meta = None
numeric_cols: List[str] = []
categorical_cols: List[str] = []
encoders: Dict[str, Any] = {}
best_threshold: float = 0.5
feature_importance_map: Dict[str, float] = {}
feature_names: List[str] = []
GLOBAL_FEATURE_IMPORTANCE: List[Dict[str, Any]] = []
LOAD_ERROR: str = ""


def _ensure_model_loaded() -> None:
    """
    Lazy-load model artefak. Kalau gagal, simpan pesan error di LOAD_ERROR,
    tapi JANGAN raise exception supaya module tetap bisa di-import.
    """
    global booster, calibrator, preprocess, model_meta
    global numeric_cols, categorical_cols, encoders
    global best_threshold, feature_importance_map, feature_names
    global GLOBAL_FEATURE_IMPORTANCE, LOAD_ERROR

    # Kalau sudah loaded, tidak usah ulang
    if booster is not None and preprocess is not None and model_meta is not None:
        return

    try:
        # Import xgboost di sini, bukan di top-level
        try:
            import xgboost as xgb
        except ImportError:
            raise ImportError("XGBoost tidak terinstall. Install dengan: pip install xgboost")

        print("\n[Loader] Loading model artefacts from:", MODEL_DIR)

        # Load XGBoost Booster
        booster_local = xgb.Booster()
        booster_local.load_model(MODEL_JSON)
        print(f"  ✓ Model loaded: {MODEL_JSON}")

        # Load Calibrator
        with open(CALIB_FILE, "rb") as f:
            calibrator_local = pickle.load(f)
        print(f"  ✓ Calibrator loaded: {CALIB_FILE}")

        # Load Preprocessing metadata
        with open(PREPROCESS_FILE, "rb") as f:
            preprocess_local = pickle.load(f)
        print(f"  ✓ Preprocessing config loaded: {PREPROCESS_FILE}")

        # Load metadata
        with open(META_FILE, "r") as f:
            model_meta_local = json.load(f)
        print(f"  ✓ Metadata loaded: {META_FILE}")

        # Extract preprocessing config
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

        # Assign ke global hanya setelah semua sukses
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

        print("\n[Loader] ✓ Model ready for inference")
        print("=" * 80)
        print(
            f"  Version: {model_meta_local.get('model_version', 'unknown')}, "
            f"Features: {model_meta_local.get('features', {}).get('total_count', 0)}"
        )
        print("=" * 80)

    except Exception as e:
        LOAD_ERROR = f"{type(e).__name__}: {e}"
        print(f"[Loader] ✗ Error loading artifacts: {LOAD_ERROR}")
        print(traceback.format_exc())


# ================================================================
# UTILITY FUNCTIONS
# ================================================================

def compute_age(dob: str, visit_date: str) -> int:
    """Calculate patient age at visit date."""
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
    Hitung skor kompatibilitas klinis diagnosis vs prosedur/obat/vitamin.
    """
    rules = COMPAT_RULES.get(icd10)

    if not rules:
        return {
            "diagnosis_procedure_score": 0.5,
            "diagnosis_drug_score": 0.5,
            "diagnosis_vitamin_score": 0.5,
        }

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
        "diagnosis_vitamin_score": float(vit_score),
    }


def compute_mismatch_flags(compatibility_scores: Dict[str, float]) -> Dict[str, int]:
    """
    Flag ketidaksesuaian klinis berdasarkan skor kompatibilitas.
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
    Skor anomali biaya secara umum (tanpa distribusi diagnosis spesifik).
    """
    if total_claim > 1_500_000:
        return 4  # Extreme
    if total_claim > 1_000_000:
        return 3  # High
    if total_claim > 500_000:
        return 2  # Moderate
    return 1  # Normal


def get_compatibility_details(
    icd10: str,
    procedures: List[str],
    drugs: List[str],
    vitamins: List[str],
) -> Dict[str, Any]:
    """
    Detail kompatibilitas klinis untuk UI.
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
# FEATURE ENGINEERING (MUST MATCH ETL!)
# ================================================================

def build_features_from_raw(raw: Dict[str, Any]) -> Tuple[str, Dict[str, Any], Dict[str, float], Dict[str, int]]:
    """
    Transform raw claim data into model features.
    """
    claim_id = raw.get("claim_id")

    visit_date = raw.get("visit_date")
    dt = datetime.strptime(visit_date, "%Y-%m-%d").date()

    procedures = raw.get("procedures", [])
    drugs = raw.get("drugs", [])
    vitamins = raw.get("vitamins", [])

    if not isinstance(procedures, list):
        procedures = [procedures] if procedures else []
    if not isinstance(drugs, list):
        drugs = [drugs] if drugs else []
    if not isinstance(vitamins, list):
        vitamins = [vitamins] if vitamins else []

    total_proc = float(raw.get("total_procedure_cost", 0))
    total_drug = float(raw.get("total_drug_cost", 0))
    total_vit = float(raw.get("total_vitamin_cost", 0))
    total_claim = float(raw.get("total_claim_amount", 0))

    patient_age = compute_age(raw.get("patient_dob"), visit_date)

    icd10 = raw.get("icd10_primary_code", "UNKNOWN")
    compatibility = compute_compatibility_scores(icd10, procedures, drugs, vitamins)
    mismatch = compute_mismatch_flags(compatibility)
    biaya_anomaly = compute_cost_anomaly_score(total_claim, icd10)

    patient_freq = 2  # dummy, nanti bisa ganti

    feature_row = {
        # Numeric
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
        # Compatibility scores
        "diagnosis_procedure_score": compatibility["diagnosis_procedure_score"],
        "diagnosis_drug_score": compatibility["diagnosis_drug_score"],
        "diagnosis_vitamin_score": compatibility["diagnosis_vitamin_score"],
        # Flags
        "procedure_mismatch_flag": mismatch["procedure_mismatch_flag"],
        "drug_mismatch_flag": mismatch["drug_mismatch_flag"],
        "vitamin_mismatch_flag": mismatch["vitamin_mismatch_flag"],
        "mismatch_count": mismatch["mismatch_count"],
        # Categoricals
        "visit_type": raw.get("visit_type", "UNKNOWN"),
        "department": raw.get("department", "UNKNOWN"),
        "icd10_primary_code": icd10,
    }

    return claim_id, feature_row, compatibility, mismatch


def build_feature_df(records: List[Dict[str, Any]]) -> Tuple[pd.DataFrame, Any]:
    """
    Bangun DataFrame fitur dan DMatrix XGBoost dari feature rows.
    """
    _ensure_model_loaded()
    
    if LOAD_ERROR:
        raise RuntimeError(f"Model tidak ter-load: {LOAD_ERROR}")

    try:
        from xgboost import DMatrix
    except ImportError:
        raise ImportError("XGBoost tidak terinstall")

    df = pd.DataFrame.from_records(records)

    # Pastikan semua kolom ada
    for col_name in numeric_cols + categorical_cols:
        if col_name not in df.columns:
            if col_name in numeric_cols:
                df[col_name] = 0.0
            else:
                df[col_name] = "UNKNOWN"

    # Encode kategorikal pakai encoder hasil training
    for col_name in categorical_cols:
        df[col_name] = df[col_name].astype(str).fillna("UNKNOWN")
        if col_name in encoders:
            enc = encoders[col_name]
            df[col_name] = enc.transform(df[[col_name]])[col_name]

    # Bersihkan kolom numerik
    for col_name in numeric_cols:
        df[col_name] = pd.to_numeric(df[col_name], errors="coerce").fillna(0.0)
        df[col_name] = df[col_name].replace([np.inf, -np.inf], 0)

    X = df[numeric_cols + categorical_cols]
    dmatrix = DMatrix(X, feature_names=feature_names)

    return df, dmatrix


# ================================================================
# EXPLANATION & RECOMMENDATION
# ================================================================

def generate_explanation(
    row: Dict[str, Any],
    fraud_score: float,
    icd10: str,
    compatibility_details: Dict[str, Any],
) -> str:
    """Generate explanation untuk fraud score."""
    reasons: List[str] = []

    if row.get("mismatch_count", 0) > 0:
        mismatch_items = []
        if row.get("procedure_mismatch_flag") == 1:
            mismatch_items.append("prosedur")
        if row.get("drug_mismatch_flag") == 1:
            mismatch_items.append("obat")
        if row.get("vitamin_mismatch_flag") == 1:
            mismatch_items.append("vitamin")

        reasons.append(f"Ketidaksesuaian klinis: {', '.join(mismatch_items)}")

    if row.get("biaya_anomaly_score", 0) >= 3:
        severity = "sangat tinggi" if row.get("biaya_anomaly_score") == 4 else "tinggi"
        reasons.append(f"Biaya klaim {severity} untuk diagnosis ini")

    if row.get("patient_frequency_risk", 0) > 10:
        reasons.append("Frekuensi klaim pasien mencurigakan")

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
        explanation = "{} {}: Tidak ada indikator fraud yang terdeteksi".format(color, risk_level)

    return explanation


def get_top_risk_factors(
    row: Dict[str, Any],
    feature_importance: Dict[str, float],
    top_n: int = 5,
) -> List[Dict[str, Any]]:
    """Get top risk factors untuk claim."""
    risk_factors: List[Dict[str, Any]] = []
    top_features = list(feature_importance.items())[: top_n * 3]

    for feat_name, importance in top_features:
        if feat_name not in row:
            continue
        
        feat_value = row[feat_name]
        risk_factors.append({
            "feature": feat_name,
            "value": feat_value,
            "importance": float(importance),
        })

    return risk_factors[:top_n]


def get_recommendation(
    fraud_score: float,
    mismatch_count: int,
    cost_anomaly: int,
) -> str:
    """Get rekomendasi tindakan berdasarkan fraud score."""
    if fraud_score > 0.8:
        return "RECOMMENDED: REJECT - Tingkat fraud tinggi, perlu investigasi mendalam"
    if fraud_score > 0.5:
        return "RECOMMENDED: REVIEW - Periksa detail klaim, ada indikator mencurigakan"
    if mismatch_count > 0:
        return "RECOMMENDED: CLARIFY - Ada ketidaksesuaian klinis, minta penjelasan dokter"
    if cost_anomaly >= 3:
        return "RECOMMENDED: VERIFY - Biaya tinggi, verifikasi dengan provider"
    return "RECOMMENDED: APPROVE - Approve jika dokumen lengkap dan valid"


# ================================================================
# VALIDATION
# ================================================================

def validate_input(data: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validate input data untuk inference."""
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
# MAIN INFERENCE HANDLER
# ================================================================

def predict(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main prediction endpoint.
    Input: {"raw_records": [list of claim records]}
    Output: Fraud scores dan explanations
    """
    _ensure_model_loaded()

    if LOAD_ERROR:
        return {
            "status": "error",
            "message": "Model tidak ter-load: {}".format(LOAD_ERROR),
            "predictions": [],
        }

    # Validate input
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
        compatibilities = {}
        mismatches = {}

        for raw in raw_records:
            claim_id, feature_row, compat, mismatch = build_features_from_raw(raw)
            feature_rows.append(feature_row)
            compatibilities[claim_id] = compat
            mismatches[claim_id] = mismatch

        # Build DMatrix
        df, dmatrix = build_feature_df(feature_rows)

        # XGBoost prediction (raw scores)
        raw_scores = booster.predict(dmatrix)

        # Calibrate scores menggunakan calibrator
        calibrated_scores = calibrator.predict_proba(raw_scores.reshape(-1, 1))[:, 1]

        # Build response
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
            explanation = generate_explanation(
                feature_row,
                fraud_score,
                icd10,
                compat_details,
            )

            # Get top risk factors
            top_factors = get_top_risk_factors(
                feature_row,
                feature_importance_map,
                top_n=5,
            )

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
                "feature_values": feature_row,
            })

        return {
            "status": "success",
            "message": "Prediction completed",
            "model_version": model_meta.get("model_version", "unknown"),
            "predictions": predictions,
        }

    except Exception as e:
        return {
            "status": "error",
            "message": "Prediction failed: {}".format(str(e)),
            "predictions": [],
            "traceback": traceback.format_exc(),
        }


# ================================================================
# HEALTH CHECK ENDPOINT
# ================================================================

def health_check(data: Dict[str, Any]) -> Dict[str, Any]:
    """Health check endpoint untuk monitoring."""
    _ensure_model_loaded()

    if LOAD_ERROR:
        return {
            "status": "unhealthy",
            "model_loaded": False,
            "error": LOAD_ERROR,
        }

    return {
        "status": "healthy",
        "model_loaded": True,
        "model_version": model_meta.get("model_version") if model_meta else "unknown",
        "features_count": len(feature_names),
        "numeric_features": len(numeric_cols),
        "categorical_features": len(categorical_cols),
        "threshold": best_threshold,
    }


# ================================================================
# MODEL INFO ENDPOINT
# ================================================================

def get_model_info(data: Dict[str, Any]) -> Dict[str, Any]:
    """Endpoint untuk mendapat informasi model dan feature importance."""
    _ensure_model_loaded()

    if LOAD_ERROR:
        return {
            "status": "error",
            "message": "Model tidak ter-load: {}".format(LOAD_ERROR),
        }

    return {
        "status": "success",
        "model_version": model_meta.get("model_version", "unknown"),
        "model_type": "XGBoost",
        "features": {
            "numeric": numeric_cols,
            "categorical": categorical_cols,
            "total_count": len(feature_names),
        },
        "threshold": best_threshold,
        "feature_importance": GLOBAL_FEATURE_IMPORTANCE,
        "metadata": model_meta,
    }


print("\nModel inference API module imported. Lazy loader will load artifacts on first call.")