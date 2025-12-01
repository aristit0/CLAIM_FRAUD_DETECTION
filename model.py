#!/usr/bin/env python3
import json
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
import cml.models_v1 as models
from datetime import datetime, date

# ================================================================
# LOAD ARTIFACTS
# ================================================================
MODEL_JSON = "model.json"
CALIB_FILE = "calibrator.pkl"
PREPROCESS_FILE = "preprocess.pkl"
META_FILE = "meta.json"

print("=== LOADING MODEL ARTIFACTS ===")

# XGBoost Booster
booster = xgb.Booster()
booster.load_model(MODEL_JSON)

# Calibrator
with open(CALIB_FILE, "rb") as f:
    calibrator = pickle.load(f)

# Preprocess metadata
with open(PREPROCESS_FILE, "rb") as f:
    preprocess = pickle.load(f)

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


# ================================================================
# UTILS
# ================================================================
def compute_age(dob, visit_date):
    try:
        dob = datetime.strptime(str(dob), "%Y-%m-%d").date()
        visit_date = datetime.strptime(str(visit_date), "%Y-%m-%d").date()
        age = visit_date.year - dob.year - (
            (visit_date.month, visit_date.day) < (dob.month, dob.day)
        )
        return max(age, 0)
    except:
        return 0


# ================================================================
# RAW → FEATURE ENGINEERING (SAMA DENGAN TRAINING)
# ================================================================
def build_features_from_raw(raw):
    claim_id = raw.get("claim_id")

    # Basic fields
    visit_date = raw.get("visit_date")
    dt = datetime.strptime(visit_date, "%Y-%m-%d").date()

    total_proc = float(raw.get("total_procedure_cost", 0))
    total_drug = float(raw.get("total_drug_cost", 0))
    total_vit = float(raw.get("total_vitamin_cost", 0))
    total_claim = float(raw.get("total_claim_amount", 0))

    # Lists
    procedures = raw.get("procedures", [])
    drugs = raw.get("drugs", [])
    vitamins = raw.get("vitamins", [])

    # Rule features
    severity_score = 3 if total_proc > 100000 else 1
    cost_per_procedure = total_proc / max(len(procedures), 1)
    patient_claim_count = 12
    biaya_anomaly_score = total_claim / max(total_proc, 1)
    cost_procedure_anomaly = 1 if cost_per_procedure > 500000 else 0
    patient_frequency_risk = 1 if patient_claim_count > 10 else 0

    # Clinical compatibility (dummy logic)
    diagnosis_procedure_score = 0.2
    diagnosis_drug_score = 0.1
    diagnosis_vitamin_score = 0.0
    treatment_consistency_score = 0.3

    # mismatch flags
    procedure_mismatch_flag = 1 if len(procedures) > 0 else 0
    drug_mismatch_flag = 1 if len(drugs) > 0 else 0
    vitamin_mismatch_flag = 1 if len(vitamins) > 0 else 0
    mismatch_count = procedure_mismatch_flag + drug_mismatch_flag + vitamin_mismatch_flag

    # FINAL FEATURE ROW
    feature_row = {
        "patient_age": compute_age(raw.get("patient_dob"), visit_date),
        "total_procedure_cost": total_proc,
        "total_drug_cost": total_drug,
        "total_vitamin_cost": total_vit,
        "total_claim_amount": total_claim,
        "severity_score": severity_score,
        "cost_per_procedure": cost_per_procedure,
        "patient_claim_count": patient_claim_count,
        "biaya_anomaly_score": biaya_anomaly_score,
        "cost_procedure_anomaly": cost_procedure_anomaly,
        "patient_frequency_risk": patient_frequency_risk,
        "visit_year": dt.year,
        "visit_month": dt.month,
        "visit_day": dt.day,
        "diagnosis_procedure_score": diagnosis_procedure_score,
        "diagnosis_drug_score": diagnosis_drug_score,
        "diagnosis_vitamin_score": diagnosis_vitamin_score,
        "treatment_consistency_score": treatment_consistency_score,
        "procedure_mismatch_flag": procedure_mismatch_flag,
        "drug_mismatch_flag": drug_mismatch_flag,
        "vitamin_mismatch_flag": vitamin_mismatch_flag,
        "mismatch_count": mismatch_count,
        "visit_type": raw.get("visit_type"),
        "department": raw.get("department"),
        "icd10_primary_code": raw.get("icd10_primary_code"),
    }

    return claim_id, feature_row


# ================================================================
# ENCODE FEATURES → DMATRIX
# ================================================================
def build_feature_df(records):
    df = pd.DataFrame.from_records(records)

    # Ensure all required columns exist
    for c in numeric_cols + categorical_cols:
        if c not in df.columns:
            df[c] = None

    # Encode categoricals
    for c in categorical_cols:
        df[c] = df[c].astype(str).fillna("__MISSING__")
        encoder = encoders[c]
        df[c] = encoder.transform(df[[c]])[c]

    # Clean numeric
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    X = df[numeric_cols + categorical_cols]
    dmatrix = xgb.DMatrix(X, feature_names=feature_names)

    return df, dmatrix


# ================================================================
# RULE-BASED SUSPICIOUS SECTIONS
# ================================================================
def derive_suspicious(row):
    sec = []
    if row.get("diagnosis_procedure_score", 1) < 1:
        sec.append("procedure_incompatible_with_diagnosis")
    if row.get("diagnosis_drug_score", 1) < 1:
        sec.append("drug_incompatible_with_diagnosis")
    if row.get("diagnosis_vitamin_score", 1) < 1:
        sec.append("vitamin_incompatible_with_diagnosis")
    if row.get("treatment_consistency_score", 1) < 0.67:
        sec.append("overall_treatment_inconsistent")
    if row.get("biaya_anomaly_score", 0) > 2.5:
        sec.append("cost_anomaly")
    if row.get("patient_frequency_risk", 0) == 1:
        sec.append("high_patient_claim_frequency")
    if row.get("procedure_mismatch_flag", 0) == 1:
        sec.append("procedure_mismatch")
    if row.get("drug_mismatch_flag", 0) == 1:
        sec.append("drug_mismatch")
    if row.get("vitamin_mismatch_flag", 0) == 1:
        sec.append("vitamin_mismatch")
    return sec


# ================================================================
# MAIN PREDICT FUNCTION
# ================================================================
@models.cml_model
def predict(data):

    # Parse JSON
    if isinstance(data, str):
        data = json.loads(data)

    raw_records = data.get("raw_records")
    if not raw_records or not isinstance(raw_records, list):
        return {"error": "raw_records must be a non-empty list"}

    processed_records = []
    claim_ids = []

    # RAW → FEATURE ENGINEERING
    for raw in raw_records:
        cid, feature_row = build_features_from_raw(raw)
        claim_ids.append(cid)
        processed_records.append(feature_row)

    # FE → encoded dataframe → DMatrix
    df_raw, dmatrix = build_feature_df(processed_records)

    # Predict (uncalibrated)
    y_raw = booster.predict(dmatrix)

    # Calibrated probability
    y_proba = calibrator.predict(y_raw)
    y_pred = (y_proba >= best_threshold).astype(int)

    results = []

    for i, cid in enumerate(claim_ids):
        row = df_raw.iloc[i].to_dict()
        suspicious = derive_suspicious(row)

        results.append({
            "claim_id": cid,
            "fraud_score": float(y_proba[i]),
            "model_flag": int(y_pred[i]),
            "final_flag": int(y_pred[i]),
            "rule_flag": None,
            "rule_reason": None,
            "suspicious_sections": suspicious,
            "feature_importance": GLOBAL_FEATURE_IMPORTANCE
        })

    return {"results": results}