#!/usr/bin/env python3
import json
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
import cml.models_v1 as models
from datetime import datetime

# ======================================================
# LOAD ARTIFACTS (MODEL + CALIBRATOR)
# ======================================================
MODEL_JSON = "model.json"
CALIB_FILE = "calibrator.pkl"

print("=== LOADING MODEL ARTIFACTS ===")
model = xgb.Booster()
model.load_model(MODEL_JSON)

with open(CALIB_FILE, "rb") as f:
    calibrator = pickle.load(f)

print("Model + calibrator loaded.")

# ======================================================
# CORE FEATURE ENGINEERING (AUTO PREPROCESSING)
# ======================================================
def compute_age(dob: str, visit_date: str):
    try:
        d1 = datetime.strptime(dob, "%Y-%m-%d").date()
        d2 = datetime.strptime(visit_date, "%Y-%m-%d").date()
        return d2.year - d1.year - ((d2.month, d2.day) < (d1.month, d1.day))
    except:
        return 0

def cost_anomaly(total_claim, proc_cost):
    if proc_cost == 0:
        return 0
    return total_claim / proc_cost

def procedure_mismatch(icd10, procedures):
    return 1 if len(procedures) > 0 else 0

def drug_mismatch(icd10, drugs):
    return 1 if len(drugs) > 0 else 0

def vitamin_mismatch(icd10, vitamins):
    return 1 if len(vitamins) > 0 else 0

def build_features_from_raw(raw):
    """
    raw input example:
    {
      "claim_id": 123,
      "patient_dob": "1988-01-10",
      "visit_date": "2025-01-15",
      "visit_type": "rawat jalan",
      "department": "Poli Umum",
      "icd10_primary_code": "J10",
      "procedures": [ {"code":"99.1","cost":50000}, ... ],
      "drugs": [ {"code":"D001","cost":12000}, ... ],
      "vitamins": [ {"code":"V001","cost":5000} ],
      "total_procedure_cost": 50000,
      "total_drug_cost": 12000,
      "total_vitamin_cost": 5000,
      "total_claim_amount": 67000
    }
    """

    proc_cost = float(raw.get("total_procedure_cost", 0))
    drug_cost = float(raw.get("total_drug_cost", 0))
    vit_cost  = float(raw.get("total_vitamin_cost", 0))
    total_claim = float(raw.get("total_claim_amount", 0))

    procedures = raw.get("procedures", [])
    drugs      = raw.get("drugs", [])
    vitamins   = raw.get("vitamins", [])

    icd10 = raw.get("icd10_primary_code")

    # AUTO FEATURE ENGINEERING
    features = {
        "claim_id": raw.get("claim_id"),
        "patient_age": compute_age(raw.get("patient_dob"), raw.get("visit_date")),
        "visit_year": int(raw.get("visit_date")[0:4]),
        "visit_month": int(raw.get("visit_date")[5:7]),
        "visit_day": int(raw.get("visit_date")[8:10]),
        "visit_type": raw.get("visit_type"),
        "department": raw.get("department"),
        "icd10_primary_code": icd10,

        "total_procedure_cost": proc_cost,
        "total_drug_cost": drug_cost,
        "total_vitamin_cost": vit_cost,
        "total_claim_amount": total_claim,

        # === AUTO RULES ===
        "severity_score": 3 if proc_cost > 100000 else 1,
        "cost_per_procedure": proc_cost / max(len(procedures), 1),
        "patient_claim_count": 1,            # can be improved later
        "biaya_anomaly_score": cost_anomaly(total_claim, proc_cost),
        "cost_procedure_anomaly": 1 if proc_cost > 500000 else 0,
        "patient_frequency_risk": 0,

        # === CLINICAL DUMMY (bisa digantikan AI) ===
        "diagnosis_procedure_score": 1.0,
        "diagnosis_drug_score": 1.0,
        "diagnosis_vitamin_score": 1.0,
        "treatment_consistency_score": 1.0,

        # === MISMATCH ===
        "procedure_mismatch_flag": procedure_mismatch(icd10, procedures),
        "drug_mismatch_flag": drug_mismatch(icd10, drugs),
        "vitamin_mismatch_flag": vitamin_mismatch(icd10, vitamins),
    }

    # auto compute mismatch_count
    features["mismatch_count"] = (
        features["procedure_mismatch_flag"] +
        features["drug_mismatch_flag"] +
        features["vitamin_mismatch_flag"]
    )

    return features

# ======================================================
# AUTO SUSPICIOUS SECTION DETECTOR
# ======================================================
def derive_suspicious_sections(feat):
    sec = []
    if feat["procedure_mismatch_flag"] == 1:
        sec.append("procedure_mismatch")
    if feat["drug_mismatch_flag"] == 1:
        sec.append("drug_mismatch")
    if feat["vitamin_mismatch_flag"] == 1:
        sec.append("vitamin_mismatch")

    if feat["biaya_anomaly_score"] > 2.0:
        sec.append("cost_anomaly")

    return sec

# ======================================================
# MAIN CML MODEL (RAW INPUT)
# ======================================================
@models.cml_model
def predict(data):

    if isinstance(data, str):
        data = json.loads(data)

    raw_records = data.get("raw_records")
    if not raw_records:
        return {"error": "raw_records is required"}

    # FEATURE ENGINEERING otomatis
    processed = [build_features_from_raw(r) for r in raw_records]

    df = pd.DataFrame(processed)

    # ensure numeric
    df = df.apply(pd.to_numeric, errors="ignore").fillna(0)

    # XGBoost requires only numeric input → drop non-numeric
    df_model = df.select_dtypes(include=[np.number])

    dmat = xgb.DMatrix(df_model)

    y_raw = model.predict(dmat)
    y_proba = calibrator.predict(y_raw)
    y_pred  = (y_proba >= 0.5).astype(int)

    results = []
    for i, row in df.iterrows():
        suspicious = derive_suspicious_sections(row.to_dict())
        results.append({
            "claim_id": row["claim_id"],
            "fraud_score": float(y_proba[i]),
            "final_flag": int(y_pred[i]),
            "suspicious_sections": suspicious,
            "features_used": row.to_dict()
        })

    return {"results": results}