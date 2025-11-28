import os
import json
import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
import pickle

# ================================
# CONFIG: EDIT SESUAI LINGKUNGAN
# ================================
# Cara 1: pakai latest Production version dari Registry
MODEL_NAME = "fraud_detection_claims_xgb"
MODEL_STAGE = os.getenv("MODEL_STAGE", "Production")  # atau "Staging"

# Load model & preprocessing sekali saat container start
print("Loading model from MLflow...")
model = mlflow.xgboost.load_model(f"models:/{MODEL_NAME}/{MODEL_STAGE}")

# Load preprocess artifact dari Registry yang sama
client = mlflow.tracking.MlflowClient()
latest = client.get_latest_versions(MODEL_NAME, stages=[MODEL_STAGE])[0]
run_id = latest.run_id

artifact_uri = client.get_run(run_id).info.artifact_uri
print("Artifact URI:", artifact_uri)

# Di CML, artifact_uri biasanya local path, jadi bisa langsung
import pathlib
preprocess_file = str(pathlib.Path(artifact_uri) / "artifacts" / "preprocess.pkl")

with open(preprocess_file, "rb") as f:
    preprocess = pickle.load(f)

numeric_cols = preprocess["numeric_cols"]
cat_cols = preprocess["cat_cols"]
encoders = preprocess["encoders"]
feature_names = preprocess["feature_names"]
label_col = preprocess["label_col"]

print("Model & preprocess loaded.")


def _build_feature_df(records):
    """
    records: list of dict, each dict = satu claim record.
    Hanya pakai kolom yang dibutuhkan oleh model.
    """
    df = pd.DataFrame.from_records(records)

    # Pastikan semua kolom ada
    for c in numeric_cols + cat_cols:
        if c not in df.columns:
            df[c] = None

    # Handle kategorikal
    for c in cat_cols:
        le = encoders[c]
        df[c] = df[c].fillna("__MISSING__").astype(str)
        # Map ke label yang sudah dikenal, yang tidak dikenal set ke 0 (atau kelas "missing")
        known_classes = set(le.classes_)
        df[c] = df[c].apply(lambda v: v if v in known_classes else "__MISSING__")
        # Kalau "__MISSING__" belum ada di le.classes_, harus extend dulu
        if "__MISSING__" not in known_classes:
            le.classes_ = np.append(le.classes_, "__MISSING__")
        df[c] = le.transform(df[c])

    # Numeric null ke 0
    for c in numeric_cols:
        df[c] = df[c].astype(float).fillna(0.0)

    X = df[numeric_cols + cat_cols]
    return df, X


def _derive_suspicious_sections(row):
    """
    row: pandas Series dari feature_df original
    Simple rule-based explanation, sesuai fitur yang kamu buat.
    """
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
    except Exception:
        pass

    return sections


def _build_feature_importance():
    """
    Global feature importance dari model (bukan per-record).
    Bisa kamu ganti ke SHAP kalau mau lebih detail nanti.
    """
    importances = model.feature_importances_
    fi = [
        {"feature": fname, "importance": float(imp)}
        for fname, imp in zip(feature_names, importances)
    ]
    # Sort desc
    fi = sorted(fi, key=lambda x: x["importance"], reverse=True)
    return fi


GLOBAL_FEATURE_IMPORTANCE = _build_feature_importance()


def predict(data):
    """
    Entry point untuk CML Model Serving.
    `data` = dict dari request JSON.

    Contoh input JSON:
    {
      "records": [
        {
          "claim_id": 123,
          "patient_age": 45,
          "visit_year": 2024,
          "visit_month": 5,
          "visit_day": 10,
          "visit_type": "EMERGENCY",
          "department": "CARDIOLOGY",
          "icd10_primary_code": "I21",
          "total_claim_amount": 1500000,
          "total_procedure_cost": 900000,
          "total_drug_cost": 400000,
          "total_vitamin_cost": 200000,
          "tindakan_validity_score": 0.3,
          "obat_validity_score": 0.4,
          "vitamin_relevance_score": 0.7,
          "biaya_anomaly_score": 3.1,
          "rule_violation_flag": 1,
          "rule_violation_reason": "Biaya anomali tinggi"
        }
      ]
    }
    """

    records = data.get("records", [])
    if not records:
        return {"error": "No records provided", "results": []}

    df_raw, X = _build_feature_df(records)

    proba = model.predict_proba(X)[:, 1]  # fraud probability
    preds = (proba >= 0.5).astype(int)

    results = []
    for i, rec in enumerate(records):
        row = df_raw.iloc[i]
        fraud_score = float(proba[i])

        suspicious_sections = _derive_suspicious_sections(row)

        rule_flag = int(rec.get("rule_violation_flag", preds[i]))
        rule_reason = rec.get("rule_violation_reason", None)

        result = {
            "claim_id": rec.get("claim_id"),
            "fraud_score": fraud_score,
            "suspicious_sections": suspicious_sections,
            "rule_violations": {
                "flag": rule_flag,
                "reason": rule_reason,
            },
            "feature_importance": GLOBAL_FEATURE_IMPORTANCE,
        }
        results.append(result)

    return {"results": results}