import json
import pickle
import numpy as np
import pandas as pd
import cml.models_v1 as models

MODEL_FILE = "model.pkl"
PREPROCESS_FILE = "preprocess.pkl"
META_FILE = "meta.json"

# ======================================================
# Load artifacts **ONCE ONLY** (WAJIB untuk CML Serving)
# ======================================================
with open(MODEL_FILE, "rb") as f:
    model = pickle.load(f)

with open(PREPROCESS_FILE, "rb") as f:
    preprocess = pickle.load(f)

numeric_cols = preprocess["numeric_cols"]
cat_cols = preprocess["cat_cols"]
encoders = preprocess["encoders"]
feature_names = preprocess["feature_names"]


# ======================================================
# Build feature DF
# ======================================================
def _build_feature_df(records):

    df = pd.DataFrame.from_records(records)

    # Ensure columns
    for c in numeric_cols + cat_cols:
        if c not in df.columns:
            df[c] = None

    # Encode categoricals
    for c in cat_cols:
        df[c] = df[c].astype(str).fillna("__MISSING__")

        le = encoders[c]
        known = set(le.classes_)

        df[c] = df[c].apply(lambda v: v if v in known else "__MISSING__")

        if "__MISSING__" not in known:
            le.classes_ = np.append(le.classes_, "__MISSING__")

        df[c] = le.transform(df[c])

    # Numeric
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    X = df[numeric_cols + cat_cols]
    return df, X


# ======================================================
# Rule Derivation
# ======================================================
def _derive_suspicious_sections(row):
    sec = []
    if row.get("tindakan_validity_score", 1) < 0.5:
        sec.append("procedures")
    if row.get("obat_validity_score", 1) < 0.5:
        sec.append("drug")
    if row.get("vitamin_relevance_score", 1) < 0.5:
        sec.append("vitamin")
    if row.get("biaya_anomaly_score", 0) > 2.5:
        sec.append("cost_anomaly")
    return sec


# ======================================================
# Feature Importance
# ======================================================
def _build_feature_importance():
    try:
        imps = model.feature_importances_
        pairs = sorted(zip(feature_names, imps), key=lambda x: x[1], reverse=True)
        return [{"feature": n, "importance": float(v)} for (n, v) in pairs]
    except:
        return []

GLOBAL_FEATURE_IMPORTANCE = _build_feature_importance()


# ======================================================
# MAIN PREDICT — Decorated for Model Serving
# ======================================================
@models.cml_model
def predict(data):
    """
    CML Model Serving automatically passes JSON → dict.
    """

    # Accept string input (raw HTTP body)
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except:
            return {"error": "Invalid JSON string"}

    if not isinstance(data, dict):
        return {"error": "Input must be JSON object"}

    records = data.get("records")
    if not isinstance(records, list) or len(records) == 0:
        return {"error": "'records' must be non-empty list"}

    try:
        df_raw, X = _build_feature_df(records)

        proba = model.predict_proba(X)[:, 1]
        preds = (proba >= 0.5).astype(int)

        out = []
        for i, rec in enumerate(records):
            row = df_raw.iloc[i]

            out.append({
                "claim_id": rec.get("claim_id"),
                "fraud_score": float(proba[i]),
                "suspicious_sections": _derive_suspicious_sections(row),
                "rule_violations": {
                    "flag": int(rec.get("rule_violation_flag", preds[i])),
                    "reason": rec.get("rule_violation_reason")
                },
                "feature_importance": GLOBAL_FEATURE_IMPORTANCE
            })

        return {"results": out}

    except Exception as e:
        return {"error": str(e)}