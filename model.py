#!/usr/bin/env python3
import json
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
import cml.models_v1 as models

# ======================================================
# FILES
# ======================================================
MODEL_JSON = "model.json"
CALIB_FILE = "calibrator.pkl"
PREPROCESS_FILE = "preprocess.pkl"
META_FILE = "meta.json"

# ======================================================
# LOAD ARTIFACTS SEKALI SAJA
# ======================================================
print("=== LOADING ARTIFACTS (Booster + Calibrator) ===")

# Booster model
model = xgb.Booster()
model.load_model(MODEL_JSON)

# Calibrator
with open(CALIB_FILE, "rb") as f:
    calibrator = pickle.load(f)

# Preprocess metadata
with open(PREPROCESS_FILE, "rb") as f:
    preprocess = pickle.load(f)

numeric_cols      = preprocess["numeric_cols"]
categorical_cols  = preprocess["categorical_cols"]
encoders          = preprocess["encoders"]
best_threshold    = preprocess.get("best_threshold", 0.5)
feature_importance_map = preprocess.get("feature_importance", {})
label_col         = preprocess.get("label_col", "final_label")

feature_names = numeric_cols + categorical_cols

# GLOBAL FEATURE IMPORTANCE (untuk UI)
GLOBAL_FEATURE_IMPORTANCE = [
    {"feature": k, "importance": float(v)}
    for k, v in sorted(feature_importance_map.items(), key=lambda kv: kv[1], reverse=True)
]

print("Loaded Booster model + calibrator + preprocess metadata.")

# ======================================================
# FEATURE DF BUILDER
# ======================================================
def _build_feature_df(records):
    df = pd.DataFrame.from_records(records)

    # Ensure all columns exist
    for c in numeric_cols + categorical_cols:
        if c not in df.columns:
            df[c] = None

    # Apply target encoder for categorical columns
    for c in categorical_cols:
        df[c] = df[c].astype(str).fillna("__MISSING__")
        te = encoders[c]
        df[c] = te.transform(df[[c]])[c]

    # Numeric cleaning
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    # Final X matrix
    X = df[numeric_cols + categorical_cols]

    # Convert to DMatrix for Booster
    dmatrix = xgb.DMatrix(X, feature_names=feature_names)

    return df, dmatrix

# ======================================================
# RULE-BASED SUSPICIOUS SIGNALS (clinical mismatch)
# ======================================================
def _derive_suspicious_sections(row):
    sec = []

    try:
        if row.get("diagnosis_procedure_score", 1.0) < 1.0:
            sec.append("procedure_incompatible_with_diagnosis")

        if row.get("diagnosis_drug_score", 1.0) < 1.0:
            sec.append("drug_incompatible_with_diagnosis")

        if row.get("diagnosis_vitamin_score", 1.0) < 1.0:
            sec.append("vitamin_incompatible_with_diagnosis")

        if row.get("treatment_consistency_score", 1.0) < 0.67:
            sec.append("overall_treatment_inconsistent")

        if row.get("biaya_anomaly_score", 0.0) > 2.5:
            sec.append("cost_anomaly")

        if row.get("patient_frequency_risk", 0) == 1:
            sec.append("high_patient_claim_frequency")

        if row.get("cost_procedure_anomaly", 0) == 1:
            sec.append("procedure_cost_extreme")

        # mismatch flags explicitly
        if row.get("procedure_mismatch_flag", 0) == 1:
            sec.append("procedure_mismatch")

        if row.get("drug_mismatch_flag", 0) == 1:
            sec.append("drug_mismatch")

        if row.get("vitamin_mismatch_flag", 0) == 1:
            sec.append("vitamin_mismatch")

    except Exception:
        pass

    return sec

# ======================================================
# MAIN PREDICT ENDPOINT
# ======================================================
@models.cml_model
def predict(data):

    # Handle string input (manual curl)
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except Exception:
            return {"error": "invalid JSON"}

    if not isinstance(data, dict):
        return {"error": "input must be JSON object"}

    records = data.get("records")
    if not isinstance(records, list) or len(records) == 0:
        return {"error": "'records' must be a non-empty list"}

    try:
        # Build dataframe + DMatrix
        df_raw, dmatrix = _build_feature_df(records)

        # Booster prediction (uncalibrated)
        y_raw = model.predict(dmatrix)

        # Calibrate score
        y_proba = calibrator.predict(y_raw)

        # Apply learned threshold
        y_pred = (y_proba >= float(best_threshold)).astype(int)

        results = []

        for i, rec in enumerate(records):
            row = df_raw.iloc[i]

            rule_flag   = rec.get("rule_violation_flag")
            rule_reason = rec.get("rule_violation_reason")

            suspicious = _derive_suspicious_sections(row.to_dict())

            # Combine rule engine + ML
            if rule_flag is None:
                final_flag = int(y_pred[i])
                rule_flag_out = None
            else:
                rf = int(rule_flag)
                final_flag = max(rf, int(y_pred[i]))
                rule_flag_out = rf

            results.append({
                "claim_id": rec.get("claim_id"),
                "fraud_score": float(y_proba[i]),
                "model_flag": int(y_pred[i]),
                "final_flag": final_flag,
                "rule_flag": rule_flag_out,
                "rule_reason": rule_reason,
                "suspicious_sections": suspicious,
                "feature_importance": GLOBAL_FEATURE_IMPORTANCE
            })

        return {"results": results}

    except Exception as e:
        return {"error": str(e)}