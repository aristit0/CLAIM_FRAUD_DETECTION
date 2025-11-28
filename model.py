import os
import json
import numpy as np
import pandas as pd
import pickle

# ============================================
# DEBUG: Check working directory
# ============================================
print("DEBUG: CWD =", os.getcwd())
print("DEBUG: Files in CWD:", os.listdir("."))

# ============================================
# LOAD ARTIFACTS (must be in same folder)
# ============================================
MODEL_FILE = "model.pkl"
PREPROCESS_FILE = "preprocess.pkl"
META_FILE = "meta.json"

print("DEBUG: Loading MODEL_FILE =", MODEL_FILE)
with open(MODEL_FILE, "rb") as f:
    model = pickle.load(f)
print("DEBUG: Model loaded OK")

print("DEBUG: Loading PREPROCESS_FILE =", PREPROCESS_FILE)
with open(PREPROCESS_FILE, "rb") as f:
    preprocess = pickle.load(f)
print("DEBUG: Preprocess loaded OK")

numeric_cols = preprocess["numeric_cols"]
cat_cols = preprocess["cat_cols"]
encoders = preprocess["encoders"]
feature_names = preprocess["feature_names"]
label_col = preprocess["label_col"]

print("DEBUG: Preprocess keys loaded OK")


# ============================================
# BUILD FEATURE DF
# ============================================
def _build_feature_df(records):
    print("DEBUG: Building DF...")

    df = pd.DataFrame.from_records(records)
    print("DEBUG: Raw DF columns:", df.columns.tolist())

    # Ensure columns exist
    for c in numeric_cols + cat_cols:
        if c not in df.columns:
            df[c] = None

    print("DEBUG: Columns aligned")

    # Category encoding
    for c in cat_cols:
        df[c] = df[c].astype(str).fillna("__MISSING__")
        le = encoders[c]
        known = set(le.classes_)

        df[c] = df[c].apply(lambda v: v if v in known else "__MISSING__")

        if "__MISSING__" not in known:
            le.classes_ = np.append(le.classes_, "__MISSING__")

        df[c] = le.transform(df[c])

    print("DEBUG: categorical OK")

    # Numerics
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    print("DEBUG: numeric OK")

    X = df[numeric_cols + cat_cols]
    print("DEBUG: final X shape:", X.shape)

    return df, X


# ============================================
# RULES
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
# FEATURE IMPORTANCE
# ============================================
def _build_feature_importance():
    try:
        imps = model.feature_importances_
        fi = [{"feature": n, "importance": float(v)} for n, v in zip(feature_names, imps)]
        return sorted(fi, key=lambda x: x["importance"], reverse=True)
    except:
        return []

GLOBAL_FEATURE_IMPORTANCE = _build_feature_importance()


# ============================================
# MAIN
# ============================================
def predict(data):
    print("DEBUG: predict() called")

    if isinstance(data, str):
        try:
            data = json.loads(data)
        except:
            return {"error": "Invalid JSON"}

    if not isinstance(data, dict):
        return {"error": "Input must be dict"}

    records = data.get("records", [])
    if not isinstance(records, list) or len(records) == 0:
        return {"error": "records must be non-empty list"}

    try:
        df_raw, X = _build_feature_df(records)

        proba = model.predict_proba(X)[:, 1]
        preds = (proba >= 0.5).astype(int)

        outputs = []

        for i, rec in enumerate(records):
            row = df_raw.iloc[i]
            outputs.append({
                "claim_id": rec.get("claim_id"),
                "fraud_score": float(proba[i]),
                "suspicious_sections": _derive_suspicious_sections(row),
                "rule_violations": {
                    "flag": int(rec.get("rule_violation_flag", preds[i])),
                    "reason": rec.get("rule_violation_reason"),
                },
                "feature_importance": GLOBAL_FEATURE_IMPORTANCE,
            })

        return {"results": outputs}

    except Exception as e:
        print("ERROR in predict:", str(e))
        return {"error": str(e)}