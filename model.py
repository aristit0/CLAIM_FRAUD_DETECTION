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

print("=== LOADING MODEL ARTIFACTS (RAW PREPROCESS + RULE SET A) ===")

# XGBoost model
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
# UTIL
# ================================================================
def compute_age(dob, visit_date):
    try:
        dob = datetime.strptime(str(dob), "%Y-%m-%d").date()
        visit_date = datetime.strptime(str(visit_date), "%Y-%m-%d").date()
        return max(visit_date.year - dob.year - ((visit_date.month, visit_date.day) < (dob.month, dob.day)), 0)
    except:
        return 0


# ================================================================
# RAW → FEATURE ENGINEERING
# ================================================================
def fe_from_raw(raw):
    claim_id = raw.get("claim_id")

    visit_date = raw.get("visit_date")
    dt = datetime.strptime(visit_date, "%Y-%m-%d").date()

    # Costs
    total_proc = float(raw.get("total_procedure_cost", 0))
    total_drug = float(raw.get("total_drug_cost", 0))
    total_vit = float(raw.get("total_vitamin_cost", 0))
    total_claim = float(raw.get("total_claim_amount", 0))

    procedures = raw.get("procedures", [])
    drugs = raw.get("drugs", [])
    vitamins = raw.get("vitamins", [])

    # Rule Set A: Minimalist
    severity_score = 3 if total_proc > 100000 else 1
    cost_per_procedure = total_proc / max(len(procedures), 1)
    biaya_anomaly_score = total_claim / max(total_proc, 1)

    # Simple frequency logic (placeholder)
    patient_claim_count = 3
    patient_frequency_risk = 1 if patient_claim_count > 8 else 0

    # Clinical dummy scoring
    diagnosis_procedure_score = 1
    diagnosis_drug_score = 1
    diagnosis_vitamin_score = 1
    treatment_consistency_score = 1

    # mismatch
    procedure_mismatch_flag = 1 if len(procedures) > 0 else 0
    drug_mismatch_flag = 1 if len(drugs) > 0 else 0
    vitamin_mismatch_flag = 1 if len(vitamins) > 0 else 0
    mismatch_count = procedure_mismatch_flag + drug_mismatch_flag + vitamin_mismatch_flag

    return claim_id, {
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


# ================================================================
# ENCODING (XGBOOST INPUT)
# ================================================================
def encode_df(records):
    df = pd.DataFrame.from_records(records)

    # Ensure required cols exist
    for c in numeric_cols + categorical_cols:
        if c not in df:
            df[c] = None

    # Encode categoricals
    for c in categorical_cols:
        df[c] = df[c].astype(str).fillna("__MISSING__")
        te = encoders[c]
        df[c] = te.transform(df[[c]])[c]

    # Cast numerics
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    # X matrix
    X = df[numeric_cols + categorical_cols]
    dmat = xgb.DMatrix(X, feature_names=feature_names)
    return df, dmat


# ================================================================
# RULE SET A – Minimalist Suspicious Logic
# ================================================================
def suspicious_A(row):
    sec = []

    if row["mismatch_count"] > 1:
        sec.append("mismatch_multiple")

    if row["biaya_anomaly_score"] > 3:
        sec.append("cost_suspicious")

    if row["severity_score"] == 3 and row["total_claim_amount"] < 100000:
        sec.append("low_cost_for_high_severity")

    if row["patient_frequency_risk"] == 1:
        sec.append("high_claim_frequency")

    return sec


# ================================================================
# PREDICT
# ================================================================
@models.cml_model
def predict(data):

    if isinstance(data, str):
        data = json.loads(data)

    raw_records = data.get("raw_records")
    if not raw_records:
        return {"error": "raw_records must be a non-empty list"}

    processed = []
    cids = []

    for raw in raw_records:
        cid, feat = fe_from_raw(raw)
        processed.append(feat)
        cids.append(cid)

    df_raw, dmatrix = encode_df(processed)

    # ML prediction
    y_raw = booster.predict(dmatrix)
    y_proba = calibrator.predict(y_raw)
    y_pred = (y_proba >= best_threshold).astype(int)

    out = []
    for i, cid in enumerate(cids):
        row = df_raw.iloc[i].to_dict()
        sus = suspicious_A(row)

        out.append({
            "claim_id": cid,
            "fraud_score": float(y_proba[i]),
            "model_flag": int(y_pred[i]),
            "final_flag": int(y_pred[i]),
            "suspicious_sections": sus,
            "feature_importance": GLOBAL_FEATURE_IMPORTANCE
        })

    return {"results": out}