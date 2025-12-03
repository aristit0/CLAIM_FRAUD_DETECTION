#!/usr/bin/env python3
"""
Fraud Detection Model for Health Claims - Production Inference API
Provides fraud scoring for claim reviewers and clinical compatibility checks.
"""

import json
import pickle
import numpy as np
import pandas as pd
import cml.models_v1 as models
from datetime import datetime
from typing import Dict, List, Any
import os

# ------------------------------------------------------------------
# IMPORT CONFIG & GLOBALS
# ------------------------------------------------------------------

from config import (
    COMPAT_RULES_FALLBACK as COMPAT_RULES,
    FRAUD_PATTERNS,
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
        except Exception as e:
            LOAD_ERROR = f"Failed to import xgboost: {type(e).__name__}: {e}"
            print(f"[Loader] ✗ {LOAD_ERROR}")
            return

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
        numeric_cols_local = preprocess_local["numeric_cols"]
        categorical_cols_local = preprocess_local["categorical_cols"]
        encoders_local = preprocess_local["encoders"]
        best_threshold_local = preprocess_local["best_threshold"]
        feature_importance_map_local = preprocess_local["feature_importance"]

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
            f"  Version: {model_meta.get('model_version', 'unknown')}, "
            f"Features: {model_meta.get('features', {}).get('total_count', 0)}"
        )
        print("=" * 80)

    except Exception as e:
        import traceback

        LOAD_ERROR = f"{type(e).__name__}: {e}"
        print(f"[Loader] ✗ Error loading artifacts: {LOAD_ERROR}")
        print(traceback.format_exc())


# ================================================================
# UTILITY FUNCTIONS
# ================================================================

def compute_age(dob: str, visit_date: str) -> int:
    """Calculate patient age at visit date."""
    try:
        from datetime import datetime as _dt

        dob_dt = _dt.strptime(str(dob), "%Y-%m-%d").date()
        visit_dt = _dt.strptime(str(visit_date), "%Y-%m-%d").date()
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

def build_features_from_raw(raw: Dict[str, Any]) -> tuple:
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


def build_feature_df(records: List[Dict[str, Any]]) -> tuple:
    """
    Bangun DataFrame fitur dan DMatrix XGBoost dari feature rows.
    """
    from xgboost import DMatrix  # import lokal, setelah dipastikan tersedia

    df = pd.DataFrame.from_records(records)

    # Pastikan semua kolom ada
    for col_name in numeric_cols + categorical_cols:
        if col_name not in df.columns:
            df[col_name] = None

    # Encode kategorikal pakai encoder hasil training
    for col_name in categorical_cols:
        df[col_name] = df[col_name].astype(str).fillna("UNKNOWN")
        enc = encoders[col_name]
        df[col_name] = enc.transform(df[[col_name]])[col_name]

    # Bersihkan kolom numerik
    for col_name in numeric_cols:
        df[col_name] = pd.to_numeric(df[col_name], errors="coerce").fillna(0.0)
        df[col_name].replace([np.inf, -np.inf], 0, inplace=True)

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
    reasons: List[str] = []

    if row["mismatch_count"] > 0:
        mismatch_items = []
        if row["procedure_mismatch_flag"] == 1:
            mismatch_items.append("prosedur tidak sesuai diagnosis")
        if row["drug_mismatch_flag"] == 1:
            mismatch_items.append("obat tidak sesuai diagnosis")
        if row["vitamin_mismatch_flag"] == 1:
            mismatch_items.append("vitamin tidak relevan")

        reasons.append(f"Ketidaksesuaian klinis: {', '.join(mismatch_items)}")

    if row["biaya_anomaly_score"] >= 3:
        severity = "sangat tinggi" if row["biaya_anomaly_score"] == 4 else "tinggi"
        reasons.append(f"Biaya klaim {severity} untuk diagnosis ini")

    if row["patient_frequency_risk"] > 10:
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
        explanation = f"{color} {risk_level}: " + "; ".join(reasons)
    else:
        explanation = f"{color} {risk_level}: Tidak ada indikator fraud yang terdeteksi"

    return explanation


def get_top_risk_factors(
    row: Dict[str, Any],
    feature_importance: Dict[str, float],
    top_n: int = 5,
) -> List[Dict[str, Any]]:
    risk_factors: List[Dict[str, Any]] = []
    top_features = list(feature_importance.items())[: top_n * 3]

    for feat_name, importance in top_features:
        if feat_name not in row:
            continue

        value = row[feat_name]

        if isinstance(value, (int, float)):
            if feat_name.endswith("_flag") and value == 1:
                interpretation = {
                    "procedure_mismatch_flag": "Prosedur tidak sesuai diagnosis",
                    "drug_mismatch_flag": "Obat tidak sesuai diagnosis",
                    "vitamin_mismatch_flag": "Vitamin tidak relevan",
                }.get(feat_name, feat_name.replace("_", " ").title())

                risk_factors.append(
                    {
                        "feature": feat_name,
                        "value": value,
                        "importance": float(importance),
                        "interpretation": interpretation,
                    }
                )

            elif feat_name == "mismatch_count" and value > 0:
                risk_factors.append(
                    {
                        "feature": feat_name,
                        "value": value,
                        "importance": float(importance),
                        "interpretation": f"{int(value)} ketidaksesuaian klinis terdeteksi",
                    }
                )

            elif feat_name == "biaya_anomaly_score" and value >= 2:
                severity = ["", "Normal", "Sedang", "Tinggi", "Sangat Tinggi"][int(value)]
                risk_factors.append(
                    {
                        "feature": feat_name,
                        "value": value,
                        "importance": float(importance),
                        "interpretation": f"Anomali biaya level {severity}",
                    }
                )

        if len(risk_factors) >= top_n:
            break

    return risk_factors


def get_recommendation(
    fraud_score: float,
    mismatch_count: int,
    cost_anomaly: int,
) -> str:
    if fraud_score > 0.8:
        return "RECOMMENDED: Decline atau minta dokumen pendukung tambahan"
    if fraud_score > 0.5:
        return "RECOMMENDED: Manual review mendalam diperlukan"
    if mismatch_count > 0:
        return "RECOMMENDED: Verifikasi ketidaksesuaian klinis dengan dokter"
    if cost_anomaly >= 3:
        return "RECOMMENDED: Verifikasi justifikasi biaya tinggi"
    return "RECOMMENDED: Approve jika dokumen lengkap"


# ================================================================
# VALIDATION
# ================================================================

def validate_input(data: Dict[str, Any]) -> tuple:
    errors: List[str] = []

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
    Endpoint utama untuk fraud scoring klaim kesehatan.
    """
    # Pastikan artefak sudah loaded
    _ensure_model_loaded()
    if LOAD_ERROR:
        return {
            "status": "error",
            "error": "Model artifacts failed to load",
            "details": LOAD_ERROR,
        }

    # Parse JSON kalau datang sebagai string
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except json.JSONDecodeError as e:
            return {"status": "error", "error": f"Invalid JSON: {e}"}

    # Validasi input
    is_valid, validation_errors = validate_input(data)
    if not is_valid:
        return {
            "status": "error",
            "error": "Input validation failed",
            "details": validation_errors,
        }

    raw_records = data["raw_records"]

    try:
        processed_records: List[Dict[str, Any]] = []
        claim_ids: List[Any] = []
        icd10_codes: List[str] = []

        for raw in raw_records:
            claim_id, feature_row, _, _ = build_features_from_raw(raw)
            claim_ids.append(claim_id)
            processed_records.append(feature_row)
            icd10_codes.append(raw.get("icd10_primary_code", "UNKNOWN"))

        df_features, dmatrix = build_feature_df(processed_records)

        # Booster dan calibrator sudah disiapkan oleh _ensure_model_loaded
        y_raw = booster.predict(dmatrix)
        y_calibrated = calibrator.predict(y_raw)
        y_pred = (y_calibrated >= best_threshold).astype(int)

        results: List[Dict[str, Any]] = []

        for i, claim_id in enumerate(claim_ids):
            row = df_features.iloc[i].to_dict()
            fraud_score = float(y_calibrated[i])
            model_flag = int(y_pred[i])

            confidence = abs(fraud_score - best_threshold) * 2
            confidence = min(confidence, 1.0)

            if fraud_score > 0.8:
                risk_level = "HIGH RISK"
            elif fraud_score > 0.5:
                risk_level = "MODERATE RISK"
            elif fraud_score > 0.3:
                risk_level = "LOW RISK"
            else:
                risk_level = "MINIMAL RISK"

            raw_record = raw_records[i]
            procedures = raw_record.get("procedures", [])
            drugs = raw_record.get("drugs", [])
            vitamins = raw_record.get("vitamins", [])

            if not isinstance(procedures, list):
                procedures = [procedures] if procedures else []
            if not isinstance(drugs, list):
                drugs = [drugs] if drugs else []
            if not isinstance(vitamins, list):
                vitamins = [vitamins] if vitamins else []

            compatibility_details = get_compatibility_details(
                icd10_codes[i],
                procedures,
                drugs,
                vitamins,
            )

            explanation = generate_explanation(
                row,
                fraud_score,
                icd10_codes[i],
                compatibility_details,
            )

            risk_factors = get_top_risk_factors(
                row,
                feature_importance_map,
                top_n=5,
            )

            recommendation = get_recommendation(
                fraud_score,
                row["mismatch_count"],
                row["biaya_anomaly_score"],
            )

            clinical_compat = {
                "procedure_compatible": row["diagnosis_procedure_score"] >= 0.5,
                "drug_compatible": row["diagnosis_drug_score"] >= 0.5,
                "vitamin_compatible": row["diagnosis_vitamin_score"] >= 0.5,
                "overall_compatible": row["mismatch_count"] == 0,
                "details": compatibility_details,
            }

            results.append(
                {
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
                        "diagnosis_procedure_score": round(
                            row["diagnosis_procedure_score"], 3
                        ),
                        "diagnosis_drug_score": round(
                            row["diagnosis_drug_score"], 3
                        ),
                        "diagnosis_vitamin_score": round(
                            row["diagnosis_vitamin_score"], 3
                        ),
                    },
                }
            )

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
                "fraud_detection_rate": model_meta.get("performance", {}).get(
                    "fraud_detection_rate", 0
                ),
            },
        }

    except Exception as e:
        import traceback

        return {
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc(),
        }


# ================================================================
# HEALTH CHECK ENDPOINT
# ================================================================

@models.cml_model
def health_check(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Health check sederhana untuk model serving.
    """
    _ensure_model_loaded()
    return {
        "status": "ok" if not LOAD_ERROR else "error",
        "load_error": LOAD_ERROR,
        "model_version": model_meta.get("model_version", "unknown") if model_meta else None,
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
    """
    Mengembalikan metadata model dan feature importance.
    """
    _ensure_model_loaded()
    if LOAD_ERROR:
        return {"status": "error", "error": LOAD_ERROR}

    return {
        "status": "success",
        "model_metadata": model_meta,
        "feature_importance": GLOBAL_FEATURE_IMPORTANCE[:20],
        "compatibility_rules_count": len(COMPAT_RULES),
        "supported_diagnoses": list(COMPAT_RULES.keys()),
        "fraud_patterns": FRAUD_PATTERNS,
    }


print("\nModel inference API module imported. Lazy loader will load artifacts on first call.")