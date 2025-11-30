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
# LOAD ARTIFACTS SEKALI SAJA (WAJIB DI CML SERVING)
# ======================================================
with open(MODEL_FILE, "rb") as f:
    model = pickle.load(f)

with open(PREPROCESS_FILE, "rb") as f:
    preprocess = pickle.load(f)

numeric_cols      = preprocess["numeric_cols"]
categorical_cols  = preprocess["categorical_cols"]
encoders          = preprocess["encoders"]
best_threshold    = preprocess.get("best_threshold", 0.5)
feature_importance_map = preprocess.get("feature_importance", {})
label_col         = preprocess.get("label_col", "rule_violation_flag")

feature_names = numeric_cols + categorical_cols

# Build GLOBAL_FEATURE_IMPORTANCE list (untuk UI)
if feature_importance_map:
    GLOBAL_FEATURE_IMPORTANCE = [
        {"feature": k, "importance": float(v)}
        for k, v in sorted(feature_importance_map.items(), key=lambda kv: kv[1], reverse=True)
    ]
else:
    try:
        imps = model.feature_importances_
        GLOBAL_FEATURE_IMPORTANCE = [
            {"feature": n, "importance": float(v)}
            for n, v in sorted(zip(feature_names, imps), key=lambda x: x[1], reverse=True)
        ]
    except Exception:
        GLOBAL_FEATURE_IMPORTANCE = []

# ======================================================
# BUILD FEATURE DF: records (JSON) → X (NUMPY)
# ======================================================
def _build_feature_df(records):
    df = pd.DataFrame.from_records(records)

    # Pastikan semua kolom ada
    for c in numeric_cols + categorical_cols:
        if c not in df.columns:
            df[c] = None

    # TargetEncoder untuk categorical
    for c in categorical_cols:
        df[c] = df[c].astype(str).fillna("__MISSING__")
        te = encoders[c]
        # TargetEncoder expect DataFrame
        transformed = te.transform(df[[c]])
        df[c] = transformed[c]

    # Numeric
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    X = df[numeric_cols + categorical_cols]
    return df, X

# ======================================================
# RULE-BASED EXPLANATION dari FEATURE KLINIS
# ======================================================
def _derive_suspicious_sections(row):
    sec = []

    # 1. Clinical compatibility
    try:
        if row.get("diagnosis_procedure_score", 1.0) < 1.0:
            sec.append("procedure_incompatible_with_diagnosis")

        if row.get("diagnosis_drug_score", 1.0) < 1.0:
            sec.append("drug_incompatible_with_diagnosis")

        if row.get("diagnosis_vitamin_score", 1.0) < 1.0:
            sec.append("vitamin_incompatible_with_diagnosis")

        # 2. Overall treatment consistency
        if row.get("treatment_consistency_score", 1.0) < 0.67:
            sec.append("overall_treatment_inconsistent")

        # 3. Legacy / cost / frequency signals
        if row.get("biaya_anomaly_score", 0.0) > 2.5:
            sec.append("cost_anomaly")

        if row.get("patient_frequency_risk", 0) == 1:
            sec.append("high_patient_claim_frequency")

        if row.get("cost_procedure_anomaly", 0) == 1:
            sec.append("procedure_cost_extreme")

    except Exception:
        # Jangan bikin error kalau ada kolom yang hilang
        pass

    return sec

# ======================================================
# MAIN PREDICT (UNTUK CML MODEL SERVING)
# ======================================================
@models.cml_model
def predict(data):
    """
    Endpoint CML akan memanggil fungsi ini.
    Input:
      {
        "records": [
          {
            "claim_id": 123,
            "visit_type": "...",
            "department": "...",
            "icd10_primary_code": "...",
            ... semua fitur numeric & categorical ...
            "rule_violation_flag": 0/1 (opsional),
            "rule_violation_reason": "..." (opsional)
          },
          ...
        ]
      }
    """

    # Terima raw JSON string (kalau dipanggil dari HTTP manual)
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except Exception:
            return {"error": "invalid JSON string"}

    if not isinstance(data, dict):
        return {"error": "input must be JSON object"}

    records = data.get("records")
    if not isinstance(records, list) or len(records) == 0:
        return {"error": "'records' must be a non-empty list"}

    try:
        df_raw, X = _build_feature_df(records)
        proba = model.predict_proba(X)[:, 1]

        # Pakai threshold hasil training
        preds_model = (proba >= float(best_threshold)).astype(int)

        results = []

        for i, rec in enumerate(records):
            row = df_raw.iloc[i]

            rule_flag   = rec.get("rule_violation_flag")
            rule_reason = rec.get("rule_violation_reason")

            # Suspicious sections: kombinasi clinical compatibility + cost/frequency
            suspicious = _derive_suspicious_sections(row.to_dict())

            # Final flag: kombinasikan rule & model
            if rule_flag is None:
                final_flag = int(preds_model[i])
                rule_flag_out = None
            else:
                rf = int(rule_flag)
                final_flag = int(max(rf, preds_model[i]))
                rule_flag_out = rf

            results.append({
                "claim_id": rec.get("claim_id"),
                "fraud_score": float(proba[i]),
                "model_flag": int(preds_model[i]),
                "final_flag": final_flag,
                "rule_flag": rule_flag_out,
                "rule_reason": rule_reason,
                "suspicious_sections": suspicious,
                "feature_importance": GLOBAL_FEATURE_IMPORTANCE
            })

        return {"results": results}

    except Exception as e:
        return {"error": str(e)}