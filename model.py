#!/usr/bin/env python3
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

# XGBoost Booster
booster = xgb.Booster()
booster.load_model(MODEL_JSON)

# Calibrator
with open(CALIB_FILE, "rb") as f:
    calibrator = pickle.load(f)

# Preprocessing metadata
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
# UTILITY
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
# RAW → FEATURE ENGINEERING (Rule Set A Revised)
# ================================================================
def build_features_from_raw(raw):
    claim_id = raw.get("claim_id")

    visit_date = raw.get("visit_date")
    dt = datetime.strptime(visit_date, "%Y-%m-%d").date()

    procedures = raw.get("procedures", [])
    drugs = raw.get("drugs", [])
    vitamins = raw.get("vitamins", [])

    total_proc = float(raw.get("total_procedure_cost", 0))
    total_drug = float(raw.get("total_drug_cost", 0))
    total_vit = float(raw.get("total_vitamin_cost", 0))
    total_claim = float(raw.get("total_claim_amount", 0))

    # Basic rule-based signals (Rule Set A – Human Friendly)
    severity_score = 1 if total_proc <= 100000 else 2 if total_proc <= 300000 else 3
    cost_per_procedure = total_proc / max(len(procedures), 1)
    biaya_anomaly_score = total_claim / max(total_proc, 1)

    # Frequency dummy
    patient_claim_count = 2
    patient_frequency_risk = 1 if patient_claim_count > 10 else 0

    # Clinical consistency (simplified / neutral)
    diagnosis_procedure_score = 1
    diagnosis_drug_score = 1
    diagnosis_vitamin_score = 1
    treatment_consistency_score = 1

    # REVISED mismatch rules (only high cost triggers mismatch)
    procedure_mismatch_flag = 1 if total_proc > 300000 else 0
    drug_mismatch_flag = 1 if total_drug > 150000 else 0
    vitamin_mismatch_flag = 1 if total_vit > 80000 else 0

    mismatch_count = procedure_mismatch_flag + drug_mismatch_flag + vitamin_mismatch_flag

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
        "cost_procedure_anomaly": 1 if cost_per_procedure > 500000 else 0,
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
# RULE-BASED SUSPICIOUS
# ================================================================
def derive_suspicious(row):
    sec = []

    if row["mismatch_count"] > 1:
        sec.append("mismatch_multiple")

    if row["biaya_anomaly_score"] > 2.5:
        sec.append("cost_suspicious")

    return sec

# ================================================================
# MAIN INFERENCE HANDLER
# ================================================================
@models.cml_model
def predict(data):

    if isinstance(data, str):
        data = json.loads(data)

    raw_records = data.get("raw_records")
    if not raw_records:
        return {"error": "raw_records must be provided"}

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

        results.append({
            "claim_id": cid,
            "fraud_score": float(y_calibrated[i]),
            "model_flag": int(y_pred[i]),
            "final_flag": int(y_pred[i]),
            "suspicious_sections": suspicious,
            "feature_importance": GLOBAL_FEATURE_IMPORTANCE
        })

    return {"results": results}