#!/usr/bin/env python3
"""
Fraud Detection Model - Production Inference API
Version: v3_production
Aligned with: ETL v3, Training Model v3 (RandomForest + threshold)
"""

import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any
import cml.models_v1 as models

# ==========================================================
# LOAD MODEL ARTIFACTS
# ==========================================================

MODEL_BUNDLE_PATH = "model_v3/fraud_model_v3.pkl"
CONFIG_PATH       = "model_v3/feature_config_v3.json"

print("=" * 70)
print("LOADING MODEL V3...")
print("=" * 70)

try:
    bundle = joblib.load(MODEL_BUNDLE_PATH)
    model = bundle["model"]
    feature_cols = bundle["feature_columns"]
    best_threshold = bundle["best_threshold"]

    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)

    print("✓ Loaded model_v3 and config")
except Exception as e:
    print(f"✗ ERROR loading model bundle: {e}")
    raise


# ==========================================================
# INPUT → FEATURE BUILDER (MUST MATCH ETL)
# ==========================================================

def build_feature_row(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert one raw record into feature row according to ETL v3.
    Must match curated column logic exactly.
    """

    # Base numeric fields
    row = {
        "total_procedure_cost": float(record.get("total_procedure_cost", 0)),
        "total_drug_cost": float(record.get("total_drug_cost", 0)),
        "total_vitamin_cost": float(record.get("total_vitamin_cost", 0)),
        "total_claim_amount": float(record.get("total_claim_amount", 0)),
        "cost_anomaly_score": float(record.get("cost_anomaly_score", 0)),
        "frequency_risk": float(record.get("frequency_risk", 0)),
    }

    # One-hot diagnosis (ETL v3 uses dx_<code>)
    dx_code = record.get("primary_dx_code", "UNKNOWN")
    for c in config["dx_features"]:
        row[c] = 1 if c.endswith(dx_code) else 0

    # Multi-hot procedures
    procedures = record.get("procedures", [])
    for c in config["procedure_features"]:
        code = c.replace("proc_", "")
        row[c] = 1 if code in procedures else 0

    # Multi-hot drugs
    drugs = record.get("drugs", [])
    for c in config["drug_features"]:
        code = c.replace("drug_", "")
        row[c] = 1 if code in drugs else 0

    # Multi-hot vitamins
    vitamins = record.get("vitamins", [])
    for c in config["vitamin_features"]:
        code = c.replace("vit_", "")
        row[c] = 1 if code in vitamins else 0

    return row


# ==========================================================
# INPUT VALIDATOR
# ==========================================================

def validate_input(payload: Dict[str, Any]):
    if "records" not in payload:
        return False, ["Missing field: records[]"]

    errors = []
    required = [
        "claim_id",
        "primary_dx_code",
        "procedures",
        "drugs",
        "vitamins",
        "total_procedure_cost",
        "total_drug_cost",
        "total_vitamin_cost",
        "total_claim_amount",
        "cost_anomaly_score",
        "frequency_risk"
    ]

    for i, rec in enumerate(payload["records"]):
        missing = [f for f in required if f not in rec]
        if missing:
            errors.append(f"Record {i} missing: {missing}")

    if errors:
        return False, errors
    return True, []


# ==========================================================
# INFERENCE ENDPOINT
# ==========================================================

@models.cml_model
def predict(data: Dict[str, Any]) -> Dict[str, Any]:

    # Parse JSON if string
    if isinstance(data, str):
        data = json.loads(data)

    # Validate
    ok, errors = validate_input(data)
    if not ok:
        return {
            "status": "error",
            "errors": errors
        }

    raw_records = data["records"]

    # Build feature matrix
    feature_rows = []
    claim_ids = []

    for rec in raw_records:
        claim_ids.append(rec["claim_id"])
        feature_rows.append(build_feature_row(rec))

    df = pd.DataFrame(feature_rows)
    X = df[feature_cols].values

    # Predict fraud probability
    prob = model.predict_proba(X)[:, 1]
    pred = (prob >= best_threshold).astype(int)

    # Build output
    results = []
    for i, cid in enumerate(claim_ids):
        results.append({
            "claim_id": cid,
            "fraud_probability": round(float(prob[i]), 4),
            "fraud_flag": int(pred[i]),
            "threshold": best_threshold
        })

    return {
        "status": "success",
        "model_version": "v3_production",
        "timestamp": datetime.now().isoformat(),
        "results": results
    }


# ==========================================================
# HEALTH CHECK
# ==========================================================

@models.cml_model
def health_check(data: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "status": "healthy",
        "model_version": "v3_production",
        "feature_count": len(feature_cols),
        "threshold": best_threshold,
        "timestamp": datetime.now().isoformat()
    }


print("\nModel V3 inference API loaded and ready.")