import json
import pickle
import numpy as np
import pandas as pd
import os

# ---------- CEK LINGKUNGAN ----------
print("CWD:", os.getcwd())
print("Files:", os.listdir("."))

MODEL_FILE = "model.pkl"
PREPROCESS_FILE = "preprocess.pkl"
META_FILE = "meta.json"

print("\n=== Loading artifacts ===")
with open(MODEL_FILE, "rb") as f:
    model = pickle.load(f)
print("Model loaded OK:", type(model))

with open(PREPROCESS_FILE, "rb") as f:
    preprocess = pickle.load(f)
print("Preprocess loaded OK:", type(preprocess))

print("Preprocess keys:", preprocess.keys())

numeric_cols = preprocess["numeric_cols"]
cat_cols = preprocess["cat_cols"]
encoders = preprocess["encoders"]
feature_names = preprocess["feature_names"]
label_col = preprocess["label_col"]

print("numeric_cols:", numeric_cols)
print("cat_cols:", cat_cols)
print("label_col:", label_col)


# ---------- FEATURE DF BUILDER ----------
def _build_feature_df(records):
    print("\nDEBUG: Building DF...")

    df = pd.DataFrame.from_records(records)
    print("DEBUG: Raw DF columns:", df.columns.tolist())

    # Pastikan semua kolom ada
    for c in numeric_cols + cat_cols:
        if c not in df.columns:
            print(f"DEBUG: Column {c} missing, filling with None")
            df[c] = None

    print("DEBUG: Columns aligned:", df.columns.tolist())

    # Encoding kategori
    for c in cat_cols:
        print(f"DEBUG: Encoding categorical col: {c}")
        df[c] = df[c].astype(str).fillna("__MISSING__")
        le = encoders[c]

        known = set(le.classes_)
        # Replace unknown dengan "__MISSING__"
        df[c] = df[c].apply(lambda v: v if v in known else "__MISSING__")

        if "__MISSING__" not in known:
            le.classes_ = np.append(le.classes_, "__MISSING__")

        df[c] = le.transform(df[c])

    print("DEBUG: categorical OK")

    # Numerik
    for c in numeric_cols:
        print(f"DEBUG: Casting numeric col: {c}")
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    print("DEBUG: numeric OK")

    X = df[numeric_cols + cat_cols]
    print("DEBUG: final X shape:", X.shape)

    return df, X


# ---------- EXPLANATION ----------
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
    except Exception as e:
        print("WARN in _derive_suspicious_sections:", e)
    return sections


# ---------- FEATURE IMPORTANCE ----------
def _build_feature_importance():
    try:
        imps = model.feature_importances_
        fi = [{"feature": n, "importance": float(v)} for n, v in zip(feature_names, imps)]
        return sorted(fi, key=lambda x: x["importance"], reverse=True)
    except Exception as e:
        print("WARN building feature importance:", e)
        return []

GLOBAL_FEATURE_IMPORTANCE = _build_feature_importance()
print("Top 5 feature importance:", GLOBAL_FEATURE_IMPORTANCE[:5])


# ---------- MAIN PREDICT ----------
def predict(data):
    print("\n=== predict() called ===")

    # Kalau string, parse dulu
    if isinstance(data, str):
        print("Input is string, trying json.loads")
        data = json.loads(data)

    if not isinstance(data, dict):
        raise ValueError("Input must be a dict or JSON string representing an object.")

    records = data.get("records", [])
    if not isinstance(records, list) or len(records) == 0:
        raise ValueError("Input must contain non-empty 'records' list.")

    print(f"DEBUG: Got {len(records)} records")

    df_raw, X = _build_feature_df(records)

    print("DEBUG: Running model.predict_proba")
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


# ---------- TEST MANUAL ----------
if __name__ == "__main__":
    # sample yang sama kaya request kamu
    sample_input = {
        "records": [
            {
              "claim_id": 987654,
              "patient_age": 45,
              "visit_year": 2024,
              "visit_month": 7,
              "visit_day": 21,
              "visit_type": "rawat jalan",
              "department": "IGD",
              "icd10_primary_code": "J06",
              "total_claim_amount": 349996,
              "total_procedure_cost": 171496,
              "total_drug_cost": 129770,
              "total_vitamin_cost": 48730,
              "tindakan_validity_score": 1,
              "obat_validity_score": 1,
              "vitamin_relevance_score": 0.7,
              "biaya_anomaly_score": 0.54,
              "rule_violation_flag": 0,
              "rule_violation_reason": None
            }
        ]
    }

    print("\n=== Running local test ===")
    try:
        out = predict(sample_input)
        print("\nPREDICTION RESULT:")
        print(json.dumps(out, indent=2))
    except Exception as e:
        print("\nERROR during predict():", repr(e))