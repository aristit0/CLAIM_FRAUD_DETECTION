#!/usr/bin/env python3
"""
Fraud Detection Model for Health Claims - Production Inference API
Simple structure: load artifacts on import, single predict() endpoint.
"""

import json
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
import cml.models_v1 as models
from datetime import datetime

# ================================================================
# LOAD ARTIFACTS
# ================================================================
MODEL_JSON = "model.json"
CALIB_FILE = "calibrator.pkl"
PREPROCESS_FILE = "preprocess.pkl"

print("=== LOADING MODEL ARTIFACTS ===")

booster = None
calibrator = None
preprocess = None
numeric_cols = []
categorical_cols = []
encoders = {}
best_threshold = 0.5
feature_importance_map = {}
feature_names = []
GLOBAL_FEATURE_IMPORTANCE = []
LOAD_ERROR = None

try:
    # XGBoost Booster
    booster = xgb.Booster()
    booster.load_model(MODEL_JSON)
    print(f"  ✓ Loaded model from {MODEL_JSON}")

    # Calibrator
    with open(CALIB_FILE, "rb") as f:
        calibrator = pickle.load(f)
    print(f"  ✓ Loaded calibrator from {CALIB_FILE}")

    # Preprocessing metadata
    with open(PREPROCESS_FILE, "rb") as f:
        preprocess = pickle.load(f)
    print(f"  ✓ Loaded preprocess config from {PREPROCESS_FILE}")

    numeric_cols = preprocess["numeric_cols"]
    categorical_cols = preprocess["categorical_cols"]
    encoders = preprocess["encoders"]
    best_threshold = preprocess["best_threshold"]
    feature_importance_map = preprocess["feature_importance"]

    feature_names = numeric_cols + categorical_cols

    GLOBAL_FEATURE_IMPORTANCE = [
        {"feature": k, "importance": float(v)}
        for k, v in sorted(feature_importance_map.items(), key=lambda kv: kv[1], reverse=True)
    ]

    LOAD_ERROR = None
    print("=== MODEL ARTIFACTS READY ===")

except Exception as e:
    LOAD_ERROR = f"{type(e).__name__}: {e}"
    print(f"!!! ERROR loading model artifacts: {LOAD_ERROR}")


# ================================================================
# UTILITY
# ================================================================
def compute_age(dob, visit_date):
    try:
        dob_dt = datetime.strptime(str(dob), "%Y-%m-%d").date()
        visit_dt = datetime.strptime(str(visit_date), "%Y-%m-%d").date()
        age = visit_dt.year - dob_dt.year - (
            (visit_dt.month, visit_dt.day) < (dob_dt.month, dob_dt.day)
        )
        return max(age, 0)
    except Exception:
        return 0


def validate_input(payload):
    """
    Very basic validation: must have raw_records (list) with minimal fields.
    """
    errors = []

    if not isinstance(payload, dict):
        return False, ["Payload must be a JSON object"]

    if "raw_records" not in payload:
        return False, ["Missing field: raw_records"]

    raw_records = payload["raw_records"]

    if not isinstance(raw_records, list):
        return False, ["raw_records must be a list"]

    if len(raw_records) == 0:
        return False, ["raw_records cannot be empty"]

    required = [
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
        missing = [f for f in required if f not in rec]
        if missing:
            errors.append(f"Record {i} missing: {missing}")

    if errors:
        return False, errors

    return True, []


# ================================================================
# RAW → FEATURE ENGINEERING (Rule Set A Revised)
# ================================================================
def build_features_from_raw(raw):
    """
    Transform satu klaim mentah menjadi satu baris fitur.
    Rule Set A (versi sederhana, self-contained).
    """
    claim_id = raw.get("claim_id")

    visit_date = raw.get("visit_date")
    dt = datetime.strptime(visit_date, "%Y-%m-%d").date()

    procedures = raw.get("procedures", []) or []
    drugs = raw.get("drugs", []) or []
    vitamins = raw.get("vitamins", []) or []

    if not isinstance(procedures, list):
        procedures = [procedures]
    if not isinstance(drugs, list):
        drugs = [drugs]
    if not isinstance(vitamins, list):
        vitamins = [vitamins]

    total_proc = float(raw.get("total_procedure_cost", 0))
    total_drug = float(raw.get("total_drug_cost", 0))
    total_vit = float(raw.get("total_vitamin_cost", 0))
    total_claim = float(raw.get("total_claim_amount", 0))

    # Basic rule-based signals (Rule Set A – Human Friendly)
    if total_proc <= 100000:
        severity_score = 1
    elif total_proc <= 300000:
        severity_score = 2
    else:
        severity_score = 3

    cost_per_procedure = total_proc / max(len(procedures), 1)
    biaya_anomaly_score = total_claim / max(total_proc, 1)

    # Frequency dummy (sementara hard-coded)
    patient_claim_count = 2
    patient_frequency_risk = 1 if patient_claim_count > 10 else 0

    # Clinical consistency (simplified / neutral)
    diagnosis_procedure_score = 1.0
    diagnosis_drug_score = 1.0
    diagnosis_vitamin_score = 1.0
    treatment_consistency_score = 1.0

    # REVISED mismatch rules (only high cost triggers mismatch)
    procedure_mismatch_flag = 1 if total_proc > 300000 else 0
    drug_mismatch_flag = 1 if total_drug > 150000 else 0
    vitamin_mismatch_flag = 1 if total_vit > 80000 else 0

    mismatch_count = (
        procedure_mismatch_flag + drug_mismatch_flag + vitamin_mismatch_flag
    )

    feature_row = {
        # Basic numeric
        "patient_age": compute_age(raw.get("patient_dob"), visit_date),
        "total_procedure_cost": total_proc,
        "total_drug_cost": total_drug,
        "total_vitamin_cost": total_vit,
        "total_claim_amount": total_claim,

        # Rule-based numeric
        "severity_score": severity_score,
        "cost_per_procedure": cost_per_procedure,
        "patient_claim_count": patient_claim_count,
        "biaya_anomaly_score": biaya_anomaly_score,
        "cost_procedure_anomaly": 1 if cost_per_procedure > 500000 else 0,
        "patient_frequency_risk": patient_frequency_risk,

        # Date breakdown
        "visit_year": dt.year,
        "visit_month": dt.month,
        "visit_day": dt.day,

        # Clinical consistency scores
        "diagnosis_procedure_score": diagnosis_procedure_score,
        "diagnosis_drug_score": diagnosis_drug_score,
        "diagnosis_vitamin_score": diagnosis_vitamin_score,
        "treatment_consistency_score": treatment_consistency_score,

        # Flags
        "procedure_mismatch_flag": procedure_mismatch_flag,
        "drug_mismatch_flag": drug_mismatch_flag,
        "vitamin_mismatch_flag": vitamin_mismatch_flag,
        "mismatch_count": mismatch_count,

        # Categoricals
        "visit_type": raw.get("visit_type"),
        "department": raw.get("department"),
        "icd10_primary_code": raw.get("icd10_primary_code"),
    }

    return claim_id, feature_row


# ================================================================
# BUILD FEATURE DF
# ================================================================
def build_feature_df(records):
    """
    Build DataFrame fitur dan DMatrix XGBoost dari list feature_row.
    """
    df = pd.DataFrame.from_records(records)

    # Pastikan semua kolom yang dipakai model ada
    for c in numeric_cols + categorical_cols:
        if c not in df.columns:
            df[c] = None

    # Encode categoricals pakai encoder dari training
    for c in categorical_cols:
        df[c] = df[c].astype(str).fillna("__MISSING__")
        enc = encoders[c]
        df[c] = enc.transform(df[[c]])[c]

    # Clean numeric
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    X = df[numeric_cols + categorical_cols]
    dmatrix = xgb.DMatrix(X, feature_names=feature_names)

    return df, dmatrix


# ================================================================
# RULE-BASED SUSPICIOUS TAGS
# ================================================================
def derive_suspicious(row):
    """
    Tambahan tag sederhana untuk jelaskan kenapa flag bisa naik.
    """
    sec = []

    if row.get("mismatch_count", 0) > 1:
        sec.append("mismatch_multiple")

    if row.get("biaya_anomaly_score", 0) > 2.5:
        sec.append("cost_suspicious")

    if row.get("cost_procedure_anomaly", 0) == 1:
        sec.append("high_cost_per_procedure")

    if row.get("patient_frequency_risk", 0) == 1:
        sec.append("high_claim_frequency")

    return sec


# ================================================================
# MAIN INFERENCE HANDLER
# ================================================================
@models.cml_model
def predict(data):
    """
    Main endpoint untuk scoring fraud klaim.

    Input:
    {
      "raw_records": [
        {
          "claim_id": "CLAIM-001",
          "patient_dob": "1980-01-01",
          "visit_date": "2024-11-01",
          "visit_type": "rawat jalan",
          "department": "Poli Umum",
          "icd10_primary_code": "J06",
          "procedures": ["89.02"],
          "drugs": ["KFA001"],
          "vitamins": ["Vitamin C 500 mg"],
          "total_procedure_cost": 150000,
          "total_drug_cost": 50000,
          "total_vitamin_cost": 25000,
          "total_claim_amount": 225000
        }
      ]
    }
    """
    # Kalau artefak gagal load, balikin error yang jelas
    if LOAD_ERROR is not None:
        return {
            "status": "error",
            "error": "Model artifacts failed to load",
            "details": LOAD_ERROR,
        }

    # Parse JSON string kalau perlu
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except json.JSONDecodeError as e:
            return {"status": "error", "error": f"Invalid JSON: {e}"}

    # Validasi input
    ok, errors = validate_input(data)
    if not ok:
        return {
            "status": "error",
            "error": "Input validation failed",
            "details": errors,
        }

    raw_records = data["raw_records"]

    processed_records = []
    claim_ids = []

    for raw in raw_records:
        cid, feature_row = build_features_from_raw(raw)
        claim_ids.append(cid)
        processed_records.append(feature_row)

    # Feature preparation and model prediction
    df_raw, dmatrix = build_feature_df(processed_records)

    y_raw = booster.predict(dmatrix)
    y_calibrated = calibrator.predict(y_raw)
    y_pred = (y_calibrated >= best_threshold).astype(int)

    results = []

    for i, cid in enumerate(claim_ids):
        row = df_raw.iloc[i].to_dict()
        suspicious = derive_suspicious(row)

        fraud_score = float(y_calibrated[i])
        fraud_flag = int(y_pred[i])

        results.append(
            {
                "claim_id": cid,
                "fraud_score": round(fraud_score, 4),
                "fraud_probability": f"{fraud_score * 100:.1f}%",
                "model_flag": fraud_flag,
                "final_flag": fraud_flag,
                "suspicious_sections": suspicious,
                "feature_importance": GLOBAL_FEATURE_IMPORTANCE[:20],
            }
        )

    return {
        "status": "success",
        "timestamp": datetime.now().isoformat(),
        "total_claims_processed": len(results),
        "fraud_detected": sum(1 for r in results if r["model_flag"] == 1),
        "results": results,
    }