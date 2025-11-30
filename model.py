#!/usr/bin/env python3
import json
import pickle
import numpy as np
import pandas as pd
import cml.models_v1 as models

MODEL_FILE = "model.pkl"
PREPROCESS_FILE = "preprocess.pkl"
META_FILE = "meta.json"

# ======================================================
# LOAD ARTIFACTS ONCE (WAJIB)
# ======================================================
with open(MODEL_FILE, "rb") as f:
    model = pickle.load(f)

with open(PREPROCESS_FILE, "rb") as f:
    preprocess = pickle.load(f)

numeric_cols = preprocess["numeric_cols"]
categorical_cols = preprocess["categorical_cols"]
encoders = preprocess["encoders"]
best_threshold = preprocess.get("best_threshold", 0.5)

# Feature names = numerik + encoded categoricals
feature_names = numeric_cols + categorical_cols


# ======================================================
# HANDLE FEATURE DF FOR INFERENCE
# ======================================================
def _build_feature_df(records):
    df = pd.DataFrame.from_records(records)

    # ensure all columns exist
    for c in numeric_cols + categorical_cols:
        if c not in df.columns:
            df[c] = None

    # categorical via TargetEncoder (safe)
    for c in categorical_cols:
        df[c] = df[c].astype(str).fillna("__MISSING__")

        te = encoders[c]
        # transform menggunakan encoder fit dari training
        df[c] = te.transform(df[[c]])

    # numeric safe conversion
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    X = df[numeric_cols + categorical_cols]
    return df, X


# ======================================================
# RULE-BASED CHECK
# ======================================================
def _derive_suspicious_sections(row):
    sec = []

    if row.get("diagnosis_procedure_mismatch", 0) == 1:
        sec.append("diagnosis-procedure mismatch")

    if row.get("drug_mismatch_score", 0) == 1:
        sec.append("drug inconsistency")

    if row.get("cost_procedure_anomaly", 0) == 1:
        sec.append("procedure cost anomaly")

    if row.get("patient_frequency_risk", 0) == 1:
        sec.append("patient high claim frequency")

    if row.get("biaya_anomaly_score", 0) >= 2.5:
        sec.append("cost z-score anomaly")

    return sec


# ======================================================
# FEATURE IMPORTANCE (GLOBAL)
# ======================================================
def _extract_feature_importance():
    try:
        imps = model.feature_importances_
        pairs = sorted(zip(feature_names, imps), key=lambda x: x[1], reverse=True)
        return [{"feature": f, "importance": float(v)} for f, v in pairs]
    except:
        return []

GLOBAL_FEATURE_IMPORTANCE = _extract_feature_importance()


# ======================================================
# MAIN PREDICT (CML SERVING)
# ======================================================
@models.cml_model
def predict(data):

    if isinstance(data, str):
        try:
            data = json.loads(data)
        except:
            return {"error": "Invalid JSON string"}

    if not isinstance(data, dict):
        return {"error": "Input must be JSON object"}

    records = data.get("records")
    if not isinstance(records, list) or len(records) == 0:
        return {"error": "'records' must be a non-empty list"}

    try:
        df_raw, X = _build_feature_df(records)

        # model outputs
        probs = model.predict_proba(X)[:, 1]
        preds = (probs >= best_threshold).astype(int)

        results = []

        for i, rec in enumerate(records):

            row = df_raw.iloc[i]
            suspicious = _derive_suspicious_sections(row)

            rule_flag = rec.get("rule_violation_flag")
            rule_reason = rec.get("rule_violation_reason")

            # combine ML + rule
            final_flag = rule_flag if rule_flag is not None else int(preds[i])

            results.append({
                "claim_id": rec.get("claim_id"),
                "fraud_score": float(probs[i]),
                "predicted_flag": int(preds[i]),
                "final_flag": final_flag,
                "rule_flag": rule_flag,
                "rule_reason": rule_reason,
                "suspicious_sections": suspicious,
                "feature_importance": GLOBAL_FEATURE_IMPORTANCE
            })

        return {"results": results}

    except Exception as e:
        return {"error": str(e)}