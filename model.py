import os
import json
import numpy as np
import pandas as pd
import pickle

# ============================================
# FIX: CML Model Serving TIDAK memiliki __file__
# Gunakan path absolut untuk Resources
# ============================================

MODEL_FILE = f"model.pkl"
PREPROCESS_FILE = f"preprocess.pkl"
META_FILE = f"meta.json"

print("Loading model & preprocess from /home/cdsw/...")

with open(MODEL_FILE, "rb") as f:
    model = pickle.load(f)

with open(PREPROCESS_FILE, "rb") as f:
    preprocess = pickle.load(f)

numeric_cols = preprocess["numeric_cols"]
cat_cols = preprocess["cat_cols"]
encoders = preprocess["encoders"]
feature_names = preprocess["feature_names"]
label_col = preprocess["label_col"]

print("Model & preprocess loaded successfully.")


# ============================================
# BUILD FEATURE DF
# ============================================

def _build_feature_df(records):
    df = pd.DataFrame.from_records(records)

    # Pastikan semua kolom tersedia
    for c in numeric_cols + cat_cols:
        if c not in df.columns:
            df[c] = None

    # Encode kategorikal
    for c in cat_cols:
        le = encoders[c]
        df[c] = df[c].fillna("__MISSING__").astype(str)

        known = set(le.classes_)
        df[c] = df[c].apply(lambda v: v if v in known else "__MISSING__")

        if "__MISSING__" not in known:
            le.classes_ = np.append(le.classes_, "__MISSING__")

        df[c] = le.transform(df[c])

    # Numeric
    for c in numeric_cols:
        df[c] = df[c].astype(float).fillna(0.0)

    X = df[numeric_cols + cat_cols]
    return df, X


# ============================================
# RULE EXPLANATION
# ============================================

def _derive_suspicious_sections(row):
    sections = []
    try:
        if row.get("tindakan_validity_score", 1) < 0.5:
            sections.append("procedures")
        if row.get("obat_validity_score", 1) < 0.5:
            sections.append("drug")
        if row.get("vitamin_relevance_score", 1) < 0.5:
            sections.append("vitamin")
        if row.get("biaya_anomaly_score", 0) > 2.5:
            sections.append("cost_anomaly")
    except:
        pass
    return sections


# ============================================
# GLOBAL FEATURE IMPORTANCE
# ============================================

def _build_feature_importance():
    imps = model.feature_importances_
    fi = [{"feature": n, "importance": float(v)} for n, v in zip(feature_names, imps)]
    return sorted(fi, key=lambda x: x["importance"], reverse=True)

GLOBAL_FEATURE_IMPORTANCE = _build_feature_importance()


# ============================================
# MAIN PREDICT FUNCTION
# ============================================

def predict(data):
    # 1. Safe JSON parsing
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except Exception as e:
            return {"error": f"Invalid JSON: {e}"}

    # 2. Validation
    if not isinstance(data, dict):
        return {"error": "Input must be a JSON object (dictionary)."}

    records = data.get("records", [])
    if not records:
        return {"error": "No records provided", "results": []}

    try:
        df_raw, X = _build_feature_df(records)

        proba = model.predict_proba(X)[:, 1]
        preds = (proba >= 0.5).astype(int)

        results = []

        for i, rec in enumerate(records):
            row = df_raw.iloc[i]

            fraud_score = float(proba[i])
            suspicious_sections = _derive_suspicious_sections(row)

            rule_flag = int(rec.get("rule_violation_flag", preds[i]))
            rule_reason = rec.get("rule_violation_reason", None)

            results.append({
                "claim_id": rec.get("claim_id"),
                "fraud_score": fraud_score,
                "suspicious_sections": suspicious_sections,
                "rule_violations": {
                    "flag": rule_flag,
                    "reason": rule_reason,
                },
                "feature_importance": GLOBAL_FEATURE_IMPORTANCE,
            })

        return {"results": results}

    except Exception as e:
        return {"error": str(e)}