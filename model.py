#!/usr/bin/env python3
"""
BPJS Fraud Detection Model - Production Inference API

Aligned with training script:
- Uses model.json, calibrator.pkl, preprocess.pkl, meta.json
- Features: NUMERIC_FEATURES + CATEGORICAL_FEATURES from config.py
- Input: raw_records (raw claim data)
"""

import json
import pickle
import os
import sys
from datetime import datetime
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import xgboost as xgb
import cml.models_v1 as models

# ============================================================================
# IMPORT CONFIG
# ============================================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CURRENT_DIR)

from config import (  # type: ignore
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    COMPAT_RULES_FALLBACK,
    FRAUD_PATTERNS,
    MODEL_ARTIFACTS,
    COST_THRESHOLDS,
)

# Alias supaya lebih pendek
COMPAT_RULES = COMPAT_RULES_FALLBACK

# ============================================================================
# ARTIFACT PATHS
# ============================================================================
MODEL_JSON = MODEL_ARTIFACTS.get("model_file", "model.json")
CALIB_FILE = MODEL_ARTIFACTS.get("calibrator_file", "calibrator.pkl")
PREPROCESS_FILE = MODEL_ARTIFACTS.get("preprocess_file", "preprocess.pkl")
META_FILE = MODEL_ARTIFACTS.get("metadata_file", "meta.json")

print("=" * 80)
print("BPJS FRAUD DETECTION MODEL - LOADING ARTIFACTS")
print("=" * 80)

booster: xgb.Booster | None = None
calibrator: Any = None
preprocess: Dict[str, Any] = {}
model_meta: Dict[str, Any] = {}

numeric_cols: List[str] = NUMERIC_FEATURES.copy()
categorical_cols: List[str] = CATEGORICAL_FEATURES.copy()
feature_names: List[str] = numeric_cols + categorical_cols
encoders: Dict[str, Any] = {}
best_threshold: float = 0.5
feature_importance_map: Dict[str, float] = {}
GLOBAL_FEATURE_IMPORTANCE: List[Dict[str, Any]] = []

LOAD_ERROR: str | None = None

# ============================================================================
# LOAD MODEL / PREPROCESSING / META
# ============================================================================
try:
    # 1. Model
    booster = xgb.Booster()
    booster.load_model(MODEL_JSON)
    print(f"✓ Model loaded: {MODEL_JSON}")

    # 2. Calibrator
    with open(CALIB_FILE, "rb") as f:
        calibrator = pickle.load(f)
    print(f"✓ Calibrator loaded: {CALIB_FILE}")

    # 3. Preprocess config
    with open(PREPROCESS_FILE, "rb") as f:
        preprocess = pickle.load(f)
    print(f"✓ Preprocess config loaded: {PREPROCESS_FILE}")

    numeric_cols = preprocess.get("numeric_cols", NUMERIC_FEATURES)
    categorical_cols = preprocess.get("categorical_cols", CATEGORICAL_FEATURES)
    encoders = preprocess.get("encoders", {})
    best_threshold = float(preprocess.get("best_threshold", 0.5))
    feature_importance_map = preprocess.get("feature_importance", {})

    feature_names = numeric_cols + categorical_cols

    GLOBAL_FEATURE_IMPORTANCE = [
        {"feature": k, "importance": float(v)}
        for k, v in sorted(
            feature_importance_map.items(), key=lambda kv: kv[1], reverse=True
        )
    ]

    # 4. Metadata (opsional tapi berguna)
    try:
        with open(META_FILE, "r") as f:
            model_meta = json.load(f)
        print(f"✓ Metadata loaded: {META_FILE}")
    except Exception as e_meta:
        model_meta = {}
        print(f"⚠ Could not load metadata ({META_FILE}): {e_meta}")

    LOAD_ERROR = None
    print("=== MODEL ARTIFACTS READY ===")

except Exception as e:
    LOAD_ERROR = f"{type(e).__name__}: {e}"
    print(f"✗ ERROR loading model artifacts: {LOAD_ERROR}")

print("=" * 80)
print("Model import finished")
print("=" * 80)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def compute_age(dob: Any, visit_date: Any) -> int:
    """Hitung umur pasien saat kunjungan."""
    try:
        dob_dt = datetime.strptime(str(dob), "%Y-%m-%d").date()
        visit_dt = datetime.strptime(str(visit_date), "%Y-%m-%d").date()
        age = visit_dt.year - dob_dt.year - (
            (visit_dt.month, visit_dt.day) < (dob_dt.month, dob_dt.day)
        )
        return max(age, 0)
    except Exception:
        return 0


def compute_compatibility_scores(
    icd10: str, procedures: List[str], drugs: List[str], vitamins: List[str]
) -> Dict[str, float]:
    """
    Hitung skor kompatibilitas klinis diagnosis vs prosedur, obat, vitamin.

    Di production penuh, ini idealnya dari Iceberg.
    Di sini kita pakai COMPAT_RULES_FALLBACK dari config.
    """
    rules = COMPAT_RULES.get(icd10)

    if not rules:
        # Diagnosis tidak punya rule → skor netral
        return {
            "diagnosis_procedure_score": 0.5,
            "diagnosis_drug_score": 0.5,
            "diagnosis_vitamin_score": 0.5,
        }

    allowed_procedures = rules.get("procedures", [])
    allowed_drugs = rules.get("drugs", [])
    allowed_vitamins = rules.get("vitamins", [])

    proc_score = 0.5
    if procedures and allowed_procedures:
        proc_matches = sum(1 for p in procedures if p in allowed_procedures)
        proc_score = proc_matches / len(procedures)

    drug_score = 0.5
    if drugs and allowed_drugs:
        drug_matches = sum(1 for d in drugs if d in allowed_drugs)
        drug_score = drug_matches / len(drugs)

    vit_score = 0.5
    if vitamins and allowed_vitamins:
        vit_matches = sum(1 for v in vitamins if v in allowed_vitamins)
        vit_score = vit_matches / len(vitamins)

    return {
        "diagnosis_procedure_score": float(proc_score),
        "diagnosis_drug_score": float(drug_score),
        "diagnosis_vitamin_score": float(vit_score),
    }


def compute_mismatch_flags(compatibility_scores: Dict[str, float]) -> Dict[str, int]:
    """Flag mismatch berdasarkan skor kompatibilitas."""
    proc_flag = 1 if compatibility_scores["diagnosis_procedure_score"] < 0.5 else 0
    drug_flag = 1 if compatibility_scores["diagnosis_drug_score"] < 0.5 else 0
    vit_flag = 1 if compatibility_scores["diagnosis_vitamin_score"] < 0.5 else 0

    return {
        "procedure_mismatch_flag": proc_flag,
        "drug_mismatch_flag": drug_flag,
        "vitamin_mismatch_flag": vit_flag,
        "mismatch_count": proc_flag + drug_flag + vit_flag,
    }


def compute_cost_anomaly_score(total_claim: float) -> int:
    """
    Skor anomali biaya (1–4) pakai COST_THRESHOLDS dari config.
    """
    thresholds = COST_THRESHOLDS.get("total_claim", {})
    normal = thresholds.get("normal", 300_000)
    suspicious = thresholds.get("suspicious", 1_000_000)
    extreme = thresholds.get("extreme", 2_000_000)

    if total_claim > extreme:
        return 4  # sangat tinggi
    elif total_claim > suspicious:
        return 3  # tinggi
    elif total_claim > normal:
        return 2  # menengah
    else:
        return 1  # normal


def get_compatibility_details(
    icd10: str, procedures: List[str], drugs: List[str], vitamins: List[str]
) -> Dict[str, Any]:
    """
    Detail kompatibilitas klinis untuk UI (prosedur/obat/vitamin mana yang cocok atau tidak).
    """
    rules = COMPAT_RULES.get(icd10)

    if not rules:
        return {
            "diagnosis_known": False,
            "diagnosis_description": "Tidak ada aturan kompatibilitas untuk diagnosis ini",
            "procedure_details": [],
            "drug_details": [],
            "vitamin_details": [],
        }

    allowed_procedures = rules.get("procedures", [])
    allowed_drugs = rules.get("drugs", [])
    allowed_vitamins = rules.get("vitamins", [])

    procedure_details = []
    for proc in procedures:
        is_ok = proc in allowed_procedures
        procedure_details.append(
            {
                "code": proc,
                "compatible": is_ok,
                "status": "✓ Compatible" if is_ok else "✗ Incompatible",
            }
        )

    drug_details = []
    for drug in drugs:
        is_ok = drug in allowed_drugs
        drug_details.append(
            {
                "code": drug,
                "compatible": is_ok,
                "status": "✓ Compatible" if is_ok else "✗ Incompatible",
            }
        )

    vitamin_details = []
    for vit in vitamins:
        is_ok = vit in allowed_vitamins
        vitamin_details.append(
            {
                "name": vit,
                "compatible": is_ok,
                "status": "✓ Compatible" if is_ok else "✗ Incompatible",
            }
        )

    return {
        "diagnosis_known": True,
        "diagnosis_description": rules.get("description", ""),
        "procedure_details": procedure_details,
        "drug_details": drug_details,
        "vitamin_details": vitamin_details,
    }


# ============================================================================
# FEATURE ENGINEERING (HARUS SELINE DENGAN ETL / TRAINING)
# ============================================================================
def build_features_from_raw(raw: Dict[str, Any]) -> tuple:
    """
    Transform klaim mentah → satu baris fitur.
    Nama kolom HARUS sama dengan yang dipakai di training:
    - NUMERIC_FEATURES + CATEGORICAL_FEATURES
    """
    claim_id = raw.get("claim_id")

    visit_date = raw.get("visit_date")
    dt = datetime.strptime(visit_date, "%Y-%m-%d").date()

    # List fields
    procedures = raw.get("procedures", []) or []
    drugs = raw.get("drugs", []) or []
    vitamins = raw.get("vitamins", []) or []

    if not isinstance(procedures, list):
        procedures = [procedures]
    if not isinstance(drugs, list):
        drugs = [drugs]
    if not isinstance(vitamins, list):
        vitamins = [vitamins]

    # Cost fields
    total_proc = float(raw.get("total_procedure_cost", 0))
    total_drug = float(raw.get("total_drug_cost", 0))
    total_vit = float(raw.get("total_vitamin_cost", 0))
    total_claim = float(raw.get("total_claim_amount", 0))

    # Age
    patient_age = compute_age(raw.get("patient_dob"), visit_date)

    # Clinical compatibility
    icd10 = raw.get("icd10_primary_code", "UNKNOWN")
    compatibility = compute_compatibility_scores(icd10, procedures, drugs, vitamins)

    # Mismatch flags
    mismatch = compute_mismatch_flags(compatibility)

    # Cost anomaly
    biaya_anomaly = compute_cost_anomaly_score(total_claim)

    # Frequency risk:
    # Di training bisa dari agregasi, di runtime kita default 0 (atau value lain kalau punya source).
    patient_freq_risk = int(raw.get("patient_frequency_risk", 0))

    feature_row = {
        # numeric (harus match NUMERIC_FEATURES)
        "patient_age": patient_age,
        "visit_year": dt.year,
        "visit_month": dt.month,
        "visit_day": dt.day,
        "total_procedure_cost": total_proc,
        "total_drug_cost": total_drug,
        "total_vitamin_cost": total_vit,
        "total_claim_amount": total_claim,
        "diagnosis_procedure_score": compatibility["diagnosis_procedure_score"],
        "diagnosis_drug_score": compatibility["diagnosis_drug_score"],
        "diagnosis_vitamin_score": compatibility["diagnosis_vitamin_score"],
        "procedure_mismatch_flag": mismatch["procedure_mismatch_flag"],
        "drug_mismatch_flag": mismatch["drug_mismatch_flag"],
        "vitamin_mismatch_flag": mismatch["vitamin_mismatch_flag"],
        "mismatch_count": mismatch["mismatch_count"],
        "biaya_anomaly_score": biaya_anomaly,
        "patient_frequency_risk": patient_freq_risk,
        # categorical (harus match CATEGORICAL_FEATURES)
        "visit_type": raw.get("visit_type", "UNKNOWN"),
        "department": raw.get("department", "UNKNOWN"),
        "icd10_primary_code": icd10,
    }

    return claim_id, feature_row, compatibility, mismatch


def build_feature_df(records: List[Dict[str, Any]]) -> tuple:
    """
    Build DataFrame + DMatrix XGBoost dari list feature_row.
    Apply encoder dan cleaning sama seperti training.
    """
    df = pd.DataFrame.from_records(records)

    # Pastikan semua kolom ada
    for col_name in numeric_cols + categorical_cols:
        if col_name not in df.columns:
            df[col_name] = None

    # Categorical encoding
    for col_name in categorical_cols:
        df[col_name] = df[col_name].astype(str).fillna("UNKNOWN")
        enc = encoders[col_name]
        df[col_name] = enc.transform(df[[col_name]])[col_name]

    # Numeric cleaning
    for col_name in numeric_cols:
        df[col_name] = pd.to_numeric(df[col_name], errors="coerce").fillna(0.0)
        df[col_name].replace([np.inf, -np.inf], 0.0, inplace=True)

    X = df[numeric_cols + categorical_cols]
    dmatrix = xgb.DMatrix(X, feature_names=feature_names)

    return df, dmatrix


# ============================================================================
# EXPLANATION & RISK FACTORS
# ============================================================================
def generate_explanation(
    row: Dict[str, Any],
    fraud_score: float,
    icd10: str,
    compatibility_details: Dict[str, Any],
) -> str:
    """Penjelasan human-readable untuk reviewer BPJS."""
    reasons: List[str] = []

    if row.get("mismatch_count", 0) > 0:
        mismatch_items = []
        if row.get("procedure_mismatch_flag", 0) == 1:
            mismatch_items.append("prosedur tidak sesuai diagnosis")
        if row.get("drug_mismatch_flag", 0) == 1:
            mismatch_items.append("obat tidak sesuai diagnosis")
        if row.get("vitamin_mismatch_flag", 0) == 1:
            mismatch_items.append("vitamin/suplemen tidak relevan")
        if mismatch_items:
            reasons.append("Ketidaksesuaian klinis: " + ", ".join(mismatch_items))

    if row.get("biaya_anomaly_score", 1) >= 3:
        severity = "sangat tinggi" if row["biaya_anomaly_score"] == 4 else "tinggi"
        reasons.append(f"Biaya klaim {severity} dibanding klaim tipikal")

    if row.get("patient_frequency_risk", 0) > 10:
        reasons.append("Frekuensi klaim pasien terlihat mencurigakan")

    # Risk level
    if fraud_score > 0.8:
        risk_level = "RISIKO TINGGI"
        icon = "🔴"
    elif fraud_score > 0.5:
        risk_level = "RISIKO SEDANG"
        icon = "🟡"
    elif fraud_score > 0.3:
        risk_level = "RISIKO RENDAH"
        icon = "🟢"
    else:
        risk_level = "RISIKO MINIMAL"
        icon = "🟢"

    if reasons:
        explanation = f"{icon} {risk_level}: " + "; ".join(reasons)
    else:
        explanation = (
            f"{icon} {risk_level}: Tidak ada indikator fraud yang menonjol dari fitur utama"
        )

    return explanation


def get_top_risk_factors(
    row: Dict[str, Any],
    feature_importance: Dict[str, float],
    top_n: int = 5,
) -> List[Dict[str, Any]]:
    """Ambil faktor risiko teratas per klaim."""
    risk_factors: List[Dict[str, Any]] = []
    top_features = list(feature_importance.items())[: top_n * 4]

    for feat_name, importance in top_features:
        if feat_name not in row:
            continue

        value = row[feat_name]
        interp = None

        if feat_name == "mismatch_count" and value > 0:
            interp = f"{int(value)} ketidaksesuaian klinis terdeteksi"
        elif feat_name.endswith("_mismatch_flag") and value == 1:
            mapping = {
                "procedure_mismatch_flag": "Prosedur tidak sesuai diagnosis",
                "drug_mismatch_flag": "Obat tidak sesuai diagnosis",
                "vitamin_mismatch_flag": "Vitamin/suplemen tidak relevan",
            }
            interp = mapping.get(
                feat_name, feat_name.replace("_", " ").title()
            )
        elif feat_name == "biaya_anomaly_score" and value >= 2:
            level_map = {
                1: "Normal",
                2: "Sedang",
                3: "Tinggi",
                4: "Sangat Tinggi",
            }
            interp = f"Anomali biaya level {level_map.get(int(value), 'Tidak diketahui')}"

        if interp:
            risk_factors.append(
                {
                    "feature": feat_name,
                    "value": float(value) if isinstance(value, (int, float)) else value,
                    "importance": float(importance),
                    "interpretation": interp,
                }
            )

        if len(risk_factors) >= top_n:
            break

    return risk_factors


def get_recommendation(
    fraud_score: float,
    mismatch_count: int,
    cost_anomaly: int,
) -> str:
    """Rekomendasi tindak lanjut untuk reviewer."""
    if fraud_score > 0.8:
        return "RECOMMENDED: Pertimbangkan decline atau minta dokumen pendukung tambahan"
    if fraud_score > 0.5:
        return "RECOMMENDED: Manual review mendalam diperlukan"
    if mismatch_count > 0:
        return "RECOMMENDED: Verifikasi ketidaksesuaian klinis dengan dokter penanggung jawab"
    if cost_anomaly >= 3:
        return "RECOMMENDED: Verifikasi justifikasi biaya klaim yang tinggi"
    return "RECOMMENDED: Dapat di-approve bila dokumen pendukung lengkap"


# ============================================================================
# INPUT VALIDATION
# ============================================================================
def validate_input(payload: Any) -> tuple[bool, List[str]]:
    errors: List[str] = []

    if not isinstance(payload, dict):
        return False, ["Payload harus berupa JSON object"]

    if "raw_records" not in payload:
        return False, ["Field 'raw_records' wajib ada"]

    raw_records = payload["raw_records"]

    if not isinstance(raw_records, list):
        return False, ["'raw_records' harus berupa list"]
    if len(raw_records) == 0:
        return False, ["'raw_records' tidak boleh kosong"]

    required = [
        "claim_id",
        "patient_dob",
        "visit_date",
        "total_procedure_cost",
        "total_drug_cost",
        "total_vitamin_cost",
        "total_claim_amount",
        "icd10_primary_code",
        "department",
        "visit_type",
    ]

    for i, rec in enumerate(raw_records):
        if not isinstance(rec, dict):
            errors.append(f"Record {i} harus berupa object")
            continue
        missing = [f for f in required if f not in rec]
        if missing:
            errors.append(f"Record {i} missing fields: {missing}")

    if errors:
        return False, errors

    return True, []


# ============================================================================
# MAIN PREDICT ENDPOINT
# ============================================================================
@models.cml_model
def predict(data: Any) -> Dict[str, Any]:
    """
    Endpoint utama untuk scoring fraud klaim BPJS.

    Input:
    {
      "raw_records": [
        {
          "claim_id": "CLAIM-2024-0001",
          "patient_dob": "1980-01-15",
          "visit_date": "2024-11-01",
          "visit_type": "rawat jalan",
          "department": "Poli Umum",
          "icd10_primary_code": "J06",
          "procedures": ["89.02"],
          "drugs": ["KFA001", "KFA009"],
          "vitamins": ["Vitamin C 500 mg"],
          "total_procedure_cost": 150000,
          "total_drug_cost": 50000,
          "total_vitamin_cost": 25000,
          "total_claim_amount": 225000
        }
      ]
    }
    """
    if LOAD_ERROR is not None:
        return {
            "status": "error",
            "error": "Model artifacts failed to load",
            "details": LOAD_ERROR,
        }

    # Parse JSON string jika perlu
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except json.JSONDecodeError as e:
            return {"status": "error", "error": f"Invalid JSON: {e}"}

    # Validasi
    ok, errors = validate_input(data)
    if not ok:
        return {
            "status": "error",
            "error": "Input validation failed",
            "details": errors,
        }

    raw_records: List[Dict[str, Any]] = data["raw_records"]

    try:
        processed_records: List[Dict[str, Any]] = []
        claim_ids: List[str] = []
        icd10_codes: List[str] = []

        for raw in raw_records:
            cid, feature_row, compatibility, mismatch = build_features_from_raw(raw)
            claim_ids.append(cid)
            processed_records.append(feature_row)
            icd10_codes.append(feature_row["icd10_primary_code"])

        df_features, dmatrix = build_feature_df(processed_records)

        y_raw = booster.predict(dmatrix)  # type: ignore[arg-type]
        y_calibrated = calibrator.predict(y_raw)  # type: ignore[call-arg]
        y_pred = (y_calibrated >= best_threshold).astype(int)

        results: List[Dict[str, Any]] = []

        for i, claim_id in enumerate(claim_ids):
            row = df_features.iloc[i].to_dict()
            fraud_score = float(y_calibrated[i])
            model_flag = int(y_pred[i])

            # confidence = seberapa jauh dari threshold
            confidence = abs(fraud_score - best_threshold) * 2.0
            confidence = float(min(confidence, 1.0))

            # Risk level
            if fraud_score > 0.8:
                risk_level = "HIGH RISK"
            elif fraud_score > 0.5:
                risk_level = "MODERATE RISK"
            elif fraud_score > 0.3:
                risk_level = "LOW RISK"
            else:
                risk_level = "MINIMAL RISK"

            raw_rec = raw_records[i]
            procedures = raw_rec.get("procedures", []) or []
            drugs = raw_rec.get("drugs", []) or []
            vitamins = raw_rec.get("vitamins", []) or []

            if not isinstance(procedures, list):
                procedures = [procedures]
            if not isinstance(drugs, list):
                drugs = [drugs]
            if not isinstance(vitamins, list):
                vitamins = [vitamins]

            compatibility_details = get_compatibility_details(
                icd10_codes[i],
                procedures,
                drugs,
                vitamins,
            )

            explanation = generate_explanation(
                row, fraud_score, icd10_codes[i], compatibility_details
            )

            risk_factors = get_top_risk_factors(
                row, feature_importance_map, top_n=5
            )

            recommendation = get_recommendation(
                fraud_score,
                int(row.get("mismatch_count", 0)),
                int(row.get("biaya_anomaly_score", 1)),
            )

            clinical_compat = {
                "procedure_compatible": row["diagnosis_procedure_score"] >= 0.5,
                "drug_compatible": row["diagnosis_drug_score"] >= 0.5,
                "vitamin_compatible": row["diagnosis_vitamin_score"] >= 0.5,
                "overall_compatible": row["mismatch_count"] == 0,
                "details": compatibility_details,
            }

            results.append(
                {
                    "claim_id": claim_id,
                    "fraud_score": round(fraud_score, 4),
                    "fraud_probability": f"{fraud_score * 100:.1f}%",
                    "model_flag": model_flag,
                    "final_flag": model_flag,
                    "risk_level": risk_level,
                    "confidence": round(confidence, 4),
                    "explanation": explanation,
                    "recommendation": recommendation,
                    "top_risk_factors": risk_factors,
                    "clinical_compatibility": clinical_compat,
                    "features": {
                        "mismatch_count": int(row.get("mismatch_count", 0)),
                        "biaya_anomaly_score": int(row.get("biaya_anomaly_score", 1)),
                        "total_claim_amount": float(row.get("total_claim_amount", 0.0)),
                        "diagnosis_procedure_score": float(
                            row.get("diagnosis_procedure_score", 0.5)
                        ),
                        "diagnosis_drug_score": float(
                            row.get("diagnosis_drug_score", 0.5)
                        ),
                        "diagnosis_vitamin_score": float(
                            row.get("diagnosis_vitamin_score", 0.5)
                        ),
                    },
                    "global_feature_importance": GLOBAL_FEATURE_IMPORTANCE[:20],
                }
            )

        return {
            "status": "success",
            "model_version": model_meta.get("model_version", "unknown"),
            "timestamp": datetime.now().isoformat(),
            "total_claims_processed": len(results),
            "fraud_detected": sum(1 for r in results if r["model_flag"] == 1),
            "results": results,
            "model_info": {
                "threshold": best_threshold,
                "training_auc": model_meta.get("performance", {}).get("auc", 0.0),
                "training_f1": model_meta.get("performance", {}).get("f1", 0.0),
                "fraud_detection_rate": model_meta.get(
                    "performance", {}
                ).get("fraud_detection_rate", 0.0),
            },
        }

    except Exception as e:
        import traceback

        return {
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc(),
        }


# ============================================================================
# HEALTH CHECK
# ============================================================================
@models.cml_model
def health_check(data: Any) -> Dict[str, Any]:
    return {
        "status": "healthy" if LOAD_ERROR is None else "degraded",
        "error": LOAD_ERROR,
        "model_version": model_meta.get("model_version", "unknown"),
        "timestamp": datetime.now().isoformat(),
        "features_count": len(feature_names),
        "threshold": best_threshold,
        "supported_diagnoses": len(COMPAT_RULES),
    }


# ============================================================================
# MODEL INFO
# ============================================================================
@models.cml_model
def get_model_info(data: Any) -> Dict[str, Any]:
    return {
        "status": "success",
        "model_metadata": model_meta,
        "feature_importance": GLOBAL_FEATURE_IMPORTANCE[:50],
        "compatibility_rules_count": len(COMPAT_RULES),
        "supported_diagnoses": list(COMPAT_RULES.keys()),
        "fraud_patterns": FRAUD_PATTERNS,
    }


if __name__ == "__main__":
    print("=" * 80)
    print("BPJS FRAUD DETECTION MODEL - LOCAL RUN INFO")
    print("=" * 80)
    print(f"LOAD_ERROR: {LOAD_ERROR}")
    print(f"Numeric features ({len(numeric_cols)}): {numeric_cols}")
    print(f"Categorical features ({len(categorical_cols)}): {categorical_cols}")
    print("Ready for CML model serving.")