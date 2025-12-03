#!/usr/bin/env python3
"""
OPERASIONAL Fraud Detection Model - Production Deployment
Integrated with Iceberg reference tables for clinical rules
Provides fraud scoring with detailed clinical compatibility checking
"""

import json
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
import cml.models_v1 as models
import cml.data_v1 as cmldata
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import sys
import os

# Import centralized config
from config import (
    COMPAT_RULES_FALLBACK, FRAUD_PATTERNS, COST_THRESHOLDS,
    get_compatible_items, check_compatibility, get_fraud_pattern_description,
    calculate_fraud_score, get_cost_threshold
)

# ================================================================
# LOAD MODEL ARTIFACTS
# ================================================================
MODEL_JSON = "model.json"
CALIB_FILE = "calibrator.pkl"
PREPROCESS_FILE = "preprocess.pkl"
META_FILE = "meta.json"

print("=" * 80)
print("OPERASIONAL FRAUD DETECTION MODEL - LOADING")
print("Iceberg Integration Version")
print("=" * 80)

try:
    # Load XGBoost Booster
    booster = xgb.Booster()
    booster.load_model(MODEL_JSON)
    print(f"✓ Model loaded: {MODEL_JSON}")
    
    # Load Calibrator
    with open(CALIB_FILE, "rb") as f:
        calibrator = pickle.load(f)
    print(f"✓ Calibrator loaded: {CALIB_FILE}")
    
    # Load Preprocessing metadata
    with open(PREPROCESS_FILE, "rb") as f:
        preprocess = pickle.load(f)
    print(f"✓ Preprocessing config loaded: {PREPROCESS_FILE}")
    
    # Load metadata
    with open(META_FILE, "r") as f:
        model_meta = json.load(f)
    print(f"✓ Metadata loaded: {META_FILE}")
    
    print("\n📊 Model Information:")
    print(f"  Version: {model_meta.get('model_version', 'unknown')}")
    print(f"  Training Date: {model_meta.get('training_date', 'unknown')}")
    print(f"  AUC Score: {model_meta.get('performance', {}).get('auc', 0):.4f}")
    print(f"  Fraud Detection Rate: {model_meta.get('performance', {}).get('fraud_detection_rate', 0):.1%}")
    print(f"  Features: {model_meta.get('features', {}).get('total_count', 0)}")
    
except Exception as e:
    print(f"✗ Error loading artifacts: {e}")
    raise

# Extract preprocessing config
numeric_cols = preprocess["numeric_cols"]
categorical_cols = preprocess["categorical_cols"]
encoders = preprocess["encoders"]
best_threshold = preprocess["best_threshold"]
feature_importance_map = preprocess["feature_importance"]
label_col = preprocess["label_col"]

feature_names = numeric_cols + categorical_cols

# Initialize Spark connection for Iceberg reference tables
print("\n🔗 Connecting to Iceberg reference tables...")
try:
    conn = cmldata.get_connection("CDP-MSI")
    spark = conn.get_spark_session()
    
    # Load reference tables
    from config import ICEBERG_REF_TABLES
    ref_tables = {}
    
    for table_name, table_path in ICEBERG_REF_TABLES.items():
        try:
            ref_tables[table_name] = spark.table(table_path)
            print(f"  ✓ Loaded: {table_name} ({ref_tables[table_name].count()} records)")
        except Exception as e:
            print(f"  ⚠ Could not load {table_name}: {e}")
            ref_tables[table_name] = None
    
except Exception as e:
    print(f"  ⚠ Spark/Iceberg connection failed: {e}")
    print("  ⚠ Using fallback rules only")
    spark = None
    ref_tables = {}

print(f"\n✓ Model ready for inference")
print("=" * 80)

# Global feature importance for response
GLOBAL_FEATURE_IMPORTANCE = [
    {"feature": k, "importance": float(v)}
    for k, v in sorted(feature_importance_map.items(), key=lambda kv: kv[1], reverse=True)
]

# ================================================================
# UTILITY FUNCTIONS
# ================================================================

def compute_age(dob: str, visit_date: str) -> int:
    """Calculate patient age at visit date"""
    try:
        dob_dt = datetime.strptime(str(dob), "%Y-%m-%d").date()
        visit_dt = datetime.strptime(str(visit_date), "%Y-%m-%d").date()
        age = visit_dt.year - dob_dt.year - (
            (visit_dt.month, visit_dt.day) < (dob_dt.month, dob_dt.day)
        )
        return max(age, 0)
    except:
        return 0


def get_compatibility_from_iceberg(icd10: str, item_type: str) -> List[str]:
    """
    Get compatible items from Iceberg reference tables.
    Falls back to local rules if Iceberg unavailable.
    
    Args:
        icd10: ICD-10 diagnosis code
        item_type: 'procedures', 'drugs', or 'vitamins'
    
    Returns:
        List of compatible item codes
    """
    # Try Iceberg first
    if spark is not None:
        try:
            if item_type == "procedures":
                table_name = "clinical_rule_dx_procedure"
            elif item_type == "drugs":
                table_name = "clinical_rule_dx_drug"
            elif item_type == "vitamins":
                table_name = "clinical_rule_dx_vitamin"
            else:
                return []
            
            table_df = ref_tables.get(table_name)
            if table_df is not None:
                # Query compatible items for this diagnosis
                query = f"SELECT item_code FROM {ICEBERG_REF_TABLES[table_name]} WHERE icd10_code = '{icd10}'"
                results = spark.sql(query).collect()
                return [row.item_code for row in results]
        except Exception as e:
            print(f"  ⚠ Iceberg query failed for {icd10}: {e}")
    
    # Fallback to local rules
    return get_compatible_items(icd10, item_type)


def compute_compatibility_scores(icd10: str, procedures: List[str], 
                                 drugs: List[str], vitamins: List[str]) -> Dict[str, float]:
    """
    Calculate clinical compatibility scores using Iceberg reference tables.
    This is the CORE fraud detection feature.
    """
    if not icd10 or icd10 == "UNKNOWN":
        return {
            "diagnosis_procedure_score": 0.5,
            "diagnosis_drug_score": 0.5,
            "diagnosis_vitamin_score": 0.5
        }
    
    # Get allowed items from Iceberg or fallback
    allowed_procedures = get_compatibility_from_iceberg(icd10, "procedures")
    allowed_drugs = get_compatibility_from_iceberg(icd10, "drugs")
    allowed_vitamins = get_compatibility_from_iceberg(icd10, "vitamins")
    
    # Calculate procedure compatibility
    proc_score = 0.5
    if procedures and allowed_procedures:
        proc_matches = sum(1 for p in procedures if check_compatibility(icd10, p, "procedures"))
        proc_score = proc_matches / len(procedures) if len(procedures) > 0 else 0.5
    
    # Calculate drug compatibility
    drug_score = 0.5
    if drugs and allowed_drugs:
        drug_matches = sum(1 for d in drugs if check_compatibility(icd10, d, "drugs"))
        drug_score = drug_matches / len(drugs) if len(drugs) > 0 else 0.5
    
    # Calculate vitamin compatibility
    vit_score = 0.5
    if vitamins and allowed_vitamins:
        vit_matches = sum(1 for v in vitamins if check_compatibility(icd10, v, "vitamins"))
        vit_score = vit_matches / len(vitamins) if len(vitamins) > 0 else 0.5
    
    return {
        "diagnosis_procedure_score": float(proc_score),
        "diagnosis_drug_score": float(drug_score),
        "diagnosis_vitamin_score": float(vit_score)
    }


def compute_mismatch_flags(compatibility_scores: Dict[str, float]) -> Dict[str, int]:
    """
    Calculate mismatch flags based on compatibility scores.
    """
    proc_flag = 1 if compatibility_scores["diagnosis_procedure_score"] < 0.5 else 0
    drug_flag = 1 if compatibility_scores["diagnosis_drug_score"] < 0.5 else 0
    vit_flag = 1 if compatibility_scores["diagnosis_vitamin_score"] < 0.5 else 0
    
    return {
        "procedure_mismatch_flag": proc_flag,
        "drug_mismatch_flag": drug_flag,
        "vitamin_mismatch_flag": vit_flag,
        "mismatch_count": proc_flag + drug_flag + vit_flag
    }


def compute_cost_anomaly_score(total_claim: float, diagnosis: str = None) -> int:
    """
    Compute cost anomaly score using config thresholds.
    """
    if total_claim > get_cost_threshold("total_claim", "extreme"):
        return 4  # Extreme
    elif total_claim > get_cost_threshold("total_claim", "suspicious"):
        return 3  # High
    elif total_claim > get_cost_threshold("total_claim", "normal"):
        return 2  # Moderate
    else:
        return 1  # Normal


def get_diagnosis_info_from_iceberg(icd10: str) -> Dict[str, Any]:
    """
    Get diagnosis information from Iceberg master table.
    """
    if spark is not None and ref_tables.get("master_icd10") is not None:
        try:
            query = f"""
                SELECT diagnosis_name, description, severity_level 
                FROM {ICEBERG_REF_TABLES["master_icd10"]} 
                WHERE icd10_code = '{icd10}'
            """
            result = spark.sql(query).first()
            if result:
                return {
                    "diagnosis_name": result.diagnosis_name,
                    "description": result.description,
                    "severity_level": result.severity_level
                }
        except Exception as e:
            print(f"  ⚠ Could not fetch diagnosis info: {e}")
    
    # Fallback
    rules = COMPAT_RULES_FALLBACK.get(icd10, {})
    return {
        "diagnosis_name": f"Unknown ICD-10: {icd10}",
        "description": rules.get("description", "No description available"),
        "severity_level": "Unknown"
    }


def get_compatibility_details(icd10: str, procedures: List[str], 
                              drugs: List[str], vitamins: List[str]) -> Dict[str, Any]:
    """
    Get detailed compatibility analysis using Iceberg reference tables.
    """
    diagnosis_info = get_diagnosis_info_from_iceberg(icd10)
    
    if not icd10 or icd10 == "UNKNOWN":
        return {
            "diagnosis_known": False,
            **diagnosis_info,
            "procedure_details": [],
            "drug_details": [],
            "vitamin_details": [],
            "iceberg_source": False
        }
    
    # Get allowed items
    allowed_procedures = get_compatibility_from_iceberg(icd10, "procedures")
    allowed_drugs = get_compatibility_from_iceberg(icd10, "drugs")
    allowed_vitamins = get_compatibility_from_iceberg(icd10, "vitamins")
    
    # Check each procedure
    procedure_details = []
    for proc in procedures:
        is_compatible = check_compatibility(icd10, proc, "procedures")
        procedure_details.append({
            "code": proc,
            "compatible": is_compatible,
            "status": "✓ Sesuai standar OPERASIONAL" if is_compatible else "✗ Perlu verifikasi medis"
        })
    
    # Check each drug
    drug_details = []
    for drug in drugs:
        is_compatible = check_compatibility(icd10, drug, "drugs")
        drug_details.append({
            "code": drug,
            "compatible": is_compatible,
            "status": "✓ Sesuai standar OPERASIONAL" if is_compatible else "✗ Perlu review resep"
        })
    
    # Check each vitamin
    vitamin_details = []
    for vit in vitamins:
        is_compatible = check_compatibility(icd10, vit, "vitamins")
        vitamin_details.append({
            "name": vit,
            "compatible": is_compatible,
            "status": "✓ Sesuai standar OPERASIONAL" if is_compatible else "✗ Perlu justifikasi medis"
        })
    
    return {
        "diagnosis_known": True,
        **diagnosis_info,
        "procedure_details": procedure_details,
        "drug_details": drug_details,
        "vitamin_details": vitamin_details,
        "iceberg_source": spark is not None,
        "total_procedures": len(procedures),
        "total_drugs": len(drugs),
        "total_vitamins": len(vitamins),
        "compatible_procedures": sum(1 for p in procedure_details if p["compatible"]),
        "compatible_drugs": sum(1 for d in drug_details if d["compatible"]),
        "compatible_vitamins": sum(1 for v in vitamin_details if v["compatible"])
    }


def calculate_claim_fraud_score(features: Dict[str, float]) -> float:
    """
    Calculate fraud score using weighted features.
    """
    return calculate_fraud_score(features)


def get_cost_assessment(total_claim: float) -> Dict[str, Any]:
    """
    Assess claim cost against OPERASIONAL thresholds.
    """
    thresholds = COST_THRESHOLDS["total_claim"]
    
    if total_claim > thresholds["extreme"]:
        level = "extreme"
        assessment = "Sangat Tinggi - Wajib review manual"
        color = "🔴"
    elif total_claim > thresholds["suspicious"]:
        level = "suspicious"
        assessment = "Tinggi - Perlu verifikasi"
        color = "🟡"
    elif total_claim > thresholds["normal"]:
        level = "normal"
        assessment = "Normal - Sesuai standar"
        color = "🟢"
    else:
        level = "low"
        assessment = "Rendah - Standar OPERASIONAL"
        color = "🟢"
    
    return {
        "total_amount": total_claim,
        "level": level,
        "assessment": assessment,
        "color": color,
        "thresholds": thresholds
    }


# ================================================================
# FEATURE ENGINEERING (MUST MATCH ETL!)
# ================================================================

def build_features_from_raw(raw: Dict[str, Any]) -> tuple:
    """
    Transform raw claim data into model features.
    CRITICAL: This MUST match ETL feature engineering exactly!
    """
    claim_id = raw.get("claim_id", "UNKNOWN")
    
    # Basic fields
    visit_date = raw.get("visit_date", datetime.now().strftime("%Y-%m-%d"))
    try:
        dt = datetime.strptime(str(visit_date), "%Y-%m-%d").date()
        visit_year = dt.year
        visit_month = dt.month
        visit_day = dt.day
    except:
        dt = datetime.now().date()
        visit_year = dt.year
        visit_month = dt.month
        visit_day = dt.day
    
    # Arrays (convert to lists if needed)
    procedures = raw.get("procedures", [])
    drugs = raw.get("drugs", [])
    vitamins = raw.get("vitamins", [])
    
    # Ensure they are lists
    if not isinstance(procedures, list):
        procedures = [procedures] if procedures else []
    if not isinstance(drugs, list):
        drugs = [drugs] if drugs else []
    if not isinstance(vitamins, list):
        vitamins = [vitamins] if vitamins else []
    
    # Costs
    total_proc = float(raw.get("total_procedure_cost", 0))
    total_drug = float(raw.get("total_drug_cost", 0))
    total_vit = float(raw.get("total_vitamin_cost", 0))
    total_claim = float(raw.get("total_claim_amount", 0))
    
    # Patient age
    patient_age = compute_age(raw.get("patient_dob", "1980-01-01"), visit_date)
    
    # Clinical compatibility (CORE FEATURE)
    icd10 = raw.get("icd10_primary_code", "UNKNOWN")
    compatibility = compute_compatibility_scores(icd10, procedures, drugs, vitamins)
    
    # Mismatch flags (FRAUD INDICATORS)
    mismatch = compute_mismatch_flags(compatibility)
    
    # Cost anomaly
    biaya_anomaly = compute_cost_anomaly_score(total_claim, icd10)
    
    # Patient frequency (simplified for inference)
    patient_freq = int(raw.get("patient_claim_frequency", 2))
    
    # Build feature dictionary
    feature_row = {
        # Numeric features
        "patient_age": patient_age,
        "total_procedure_cost": total_proc,
        "total_drug_cost": total_drug,
        "total_vitamin_cost": total_vit,
        "total_claim_amount": total_claim,
        "biaya_anomaly_score": biaya_anomaly,
        "patient_frequency_risk": patient_freq,
        "visit_year": visit_year,
        "visit_month": visit_month,
        "visit_day": visit_day,
        
        # Clinical compatibility scores
        "diagnosis_procedure_score": compatibility["diagnosis_procedure_score"],
        "diagnosis_drug_score": compatibility["diagnosis_drug_score"],
        "diagnosis_vitamin_score": compatibility["diagnosis_vitamin_score"],
        
        # Mismatch flags
        "procedure_mismatch_flag": mismatch["procedure_mismatch_flag"],
        "drug_mismatch_flag": mismatch["drug_mismatch_flag"],
        "vitamin_mismatch_flag": mismatch["vitamin_mismatch_flag"],
        "mismatch_count": mismatch["mismatch_count"],
        
        # Categorical features
        "visit_type": raw.get("visit_type", "rawat jalan"),
        "department": raw.get("department", "Poli Umum"),
        "icd10_primary_code": icd10,
    }
    
    # Calculate weighted fraud score
    manual_score = calculate_claim_fraud_score(feature_row)
    
    return claim_id, feature_row, compatibility, mismatch, manual_score


# ================================================================
# FEATURE DATAFRAME BUILDER
# ================================================================

def build_feature_df(records: List[Dict[str, Any]]) -> tuple:
    """
    Build feature DataFrame from processed records.
    Apply same transformations as training.
    """
    df = pd.DataFrame.from_records(records)
    
    # Ensure all columns exist
    for col_name in numeric_cols + categorical_cols:
        if col_name not in df.columns:
            df[col_name] = None if col_name in categorical_cols else 0.0
    
    # Encode categorical features (using trained encoders)
    for col_name in categorical_cols:
        df[col_name] = df[col_name].astype(str).fillna("UNKNOWN")
        if col_name in encoders:
            enc = encoders[col_name]
            df[col_name] = enc.transform(df[[col_name]])[col_name]
        else:
            # Simple encoding if encoder not available
            df[col_name] = pd.factorize(df[col_name])[0]
    
    # Clean numeric features
    for col_name in numeric_cols:
        df[col_name] = pd.to_numeric(df[col_name], errors="coerce").fillna(0.0)
        # Handle inf
        df[col_name].replace([np.inf, -np.inf], 0, inplace=True)
    
    # Create DMatrix for XGBoost
    X = df[numeric_cols + categorical_cols]
    dmatrix = xgb.DMatrix(X, feature_names=feature_names)
    
    return df, dmatrix


# ================================================================
# EXPLANATION GENERATOR
# ================================================================

def generate_explanation(row: Dict[str, Any], fraud_score: float, 
                        icd10: str, compatibility_details: Dict) -> str:
    """
    Generate human-readable explanation for OPERASIONAL reviewers.
    """
    reasons = []
    
    # Clinical mismatches (most important)
    if row["mismatch_count"] > 0:
        mismatch_items = []
        if row["procedure_mismatch_flag"] == 1:
            mismatch_items.append("prosedur tidak sesuai diagnosis")
        if row["drug_mismatch_flag"] == 1:
            mismatch_items.append("obat tidak sesuai diagnosis")
        if row["vitamin_mismatch_flag"] == 1:
            mismatch_items.append("vitamin tidak relevan")
        
        if mismatch_items:
            reasons.append(f"Ketidaksesuaian klinis: {', '.join(mismatch_items)}")
    
    # Cost anomaly
    if row["biaya_anomaly_score"] >= 3:
        severity = "sangat tinggi" if row["biaya_anomaly_score"] == 4 else "tinggi"
        cost_assessment = get_cost_assessment(row["total_claim_amount"])
        reasons.append(f"Biaya {cost_assessment['assessment']}")
    
    # High frequency
    if row["patient_frequency_risk"] > 10:
        reasons.append("Frekuensi klaim pasien mencurigakan")
    
    # Specific compatibility issues
    if compatibility_details.get("compatible_procedures", 0) < compatibility_details.get("total_procedures", 0):
        mismatched = compatibility_details["total_procedures"] - compatibility_details["compatible_procedures"]
        reasons.append(f"{mismatched} dari {compatibility_details['total_procedures']} prosedur tidak sesuai standar")
    
    # Risk level determination
    if fraud_score > 0.8:
        risk_level = "RISIKO TINGGI"
        color = "🔴"
    elif fraud_score > 0.5:
        risk_level = "RISIKO SEDANG"
        color = "🟡"
    elif fraud_score > 0.3:
        risk_level = "RISIKO RENDAH"
        color = "🟢"
    else:
        risk_level = "RISIKO MINIMAL"
        color = "🟢"
    
    if reasons:
        explanation = f"{color} {risk_level}: " + "; ".join(reasons)
    else:
        explanation = f"{color} {risk_level}: Tidak ada indikator fraud yang terdeteksi"
    
    return explanation


def get_top_risk_factors(row: Dict[str, Any], 
                         feature_importance: Dict[str, float],
                         top_n: int = 5) -> List[Dict[str, Any]]:
    """Identify top risk factors for this specific claim"""
    risk_factors = []
    
    # Get top N important features
    top_features = list(feature_importance.items())[:top_n * 3]
    
    for feat_name, importance in top_features:
        if feat_name in row:
            value = row[feat_name]
            
            # Only include if value indicates risk
            if isinstance(value, (int, float)):
                if feat_name.endswith("_flag") and value == 1:
                    interpretation = {
                        "procedure_mismatch_flag": "Prosedur tidak sesuai diagnosis",
                        "drug_mismatch_flag": "Obat tidak sesuai diagnosis",
                        "vitamin_mismatch_flag": "Vitamin tidak relevan",
                    }.get(feat_name, feat_name.replace("_", " ").title())
                    
                    risk_factors.append({
                        "feature": feat_name,
                        "value": value,
                        "importance": float(importance),
                        "interpretation": interpretation,
                        "severity": "HIGH"
                    })
                    
                elif feat_name == "mismatch_count" and value > 0:
                    risk_factors.append({
                        "feature": feat_name,
                        "value": value,
                        "importance": float(importance),
                        "interpretation": f"{int(value)} ketidaksesuaian klinis",
                        "severity": "HIGH" if value >= 2 else "MEDIUM"
                    })
                    
                elif feat_name == "biaya_anomaly_score" and value >= 2:
                    severity_map = ["", "Normal", "Sedang", "Tinggi", "Sangat Tinggi"]
                    severity = severity_map[int(value)] if int(value) < len(severity_map) else "Tinggi"
                    risk_factors.append({
                        "feature": feat_name,
                        "value": value,
                        "importance": float(importance),
                        "interpretation": f"Anomali biaya ({severity})",
                        "severity": "HIGH" if value >= 3 else "MEDIUM"
                    })
                    
                elif feat_name == "patient_frequency_risk" and value > 10:
                    risk_factors.append({
                        "feature": feat_name,
                        "value": value,
                        "importance": float(importance),
                        "interpretation": f"Frekuensi klaim tinggi ({int(value)}x)",
                        "severity": "MEDIUM"
                    })
            
            if len(risk_factors) >= top_n:
                break
    
    return risk_factors


def get_recommendation(fraud_score: float, mismatch_count: int, 
                       cost_anomaly: int, total_claim: float) -> List[str]:
    """Generate recommendations for reviewer"""
    recommendations = []
    
    if fraud_score > 0.8:
        recommendations.append("🚨 DECLINE atau minta dokumen pendukung lengkap")
        recommendations.append("Verifikasi semua item dengan dokter penanggung jawab")
        recommendations.append("Cek riwayat pasien untuk pola mencurigakan")
    
    elif fraud_score > 0.5:
        recommendations.append("🔍 MANUAL REVIEW mendalam diperlukan")
        recommendations.append("Verifikasi justifikasi medis untuk item tidak sesuai")
        recommendations.append("Konfirmasi biaya dengan standar OPERASIONAL")
    
    elif mismatch_count > 0:
        recommendations.append("📋 Verifikasi ketidaksesuaian klinis")
        recommendations.append("Minta penjelasan dokter untuk item tidak sesuai")
    
    elif cost_anomaly >= 3:
        recommendations.append("💰 Verifikasi justifikasi biaya tinggi")
        recommendations.append("Bandingkan dengan tarif OPERASIONAL standar")
    
    else:
        recommendations.append("✅ APPROVE jika dokumen lengkap")
        recommendations.append("Pastikan semua item sesuai standar OPERASIONAL")
    
    # Add cost-specific recommendation
    cost_assessment = get_cost_assessment(total_claim)
    if cost_assessment["level"] in ["suspicious", "extreme"]:
        recommendations.append(f"💰 Biaya {cost_assessment['assessment']}")
    
    return recommendations


# ================================================================
# VALIDATION
# ================================================================

def validate_input(data: Dict[str, Any]) -> tuple:
    """Validate input data"""
    errors = []
    
    # Check if raw_records exists
    if "raw_records" not in data:
        errors.append("Missing 'raw_records' field")
        return False, errors
    
    raw_records = data["raw_records"]
    
    if not isinstance(raw_records, list):
        errors.append("'raw_records' must be a list")
        return False, errors
    
    if len(raw_records) == 0:
        errors.append("'raw_records' cannot be empty")
        return False, errors
    
    # Validate each record
    required_fields = [
        "claim_id", "patient_dob", "visit_date",
        "total_procedure_cost", "total_drug_cost", "total_vitamin_cost",
        "total_claim_amount", "icd10_primary_code"
    ]
    
    recommended_fields = ["department", "visit_type", "procedures", "drugs", "vitamins"]
    
    for i, rec in enumerate(raw_records):
        missing_required = [f for f in required_fields if f not in rec]
        if missing_required:
            errors.append(f"Record {i}: missing required fields {missing_required}")
        
        missing_recommended = [f for f in recommended_fields if f not in rec]
        if missing_recommended:
            print(f"⚠ Record {i}: missing recommended fields {missing_recommended}")
    
    if errors:
        return False, errors
    
    return True, []


# ================================================================
# MAIN INFERENCE HANDLER
# ================================================================

@models.cml_model
def predict(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main prediction endpoint for OPERASIONAL fraud detection.
    
    Input format:
    {
        "raw_records": [
            {
                "claim_id": "12345",
                "patient_dob": "1980-01-01",
                "visit_date": "2024-01-15",
                "visit_type": "rawat jalan",
                "department": "Poli Umum",
                "icd10_primary_code": "J06",
                "procedures": ["89.02", "96.70"],
                "drugs": ["KFA001", "KFA002"],
                "vitamins": ["Vitamin C 500 mg"],
                "total_procedure_cost": 150000,
                "total_drug_cost": 50000,
                "total_vitamin_cost": 25000,
                "total_claim_amount": 225000,
                "patient_claim_frequency": 3
            }
        ]
    }
    """
    
    # Parse JSON if string
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except json.JSONDecodeError as e:
            return {
                "status": "error",
                "error": f"Invalid JSON: {str(e)}"
            }
    
    # Validate input
    is_valid, validation_errors = validate_input(data)
    if not is_valid:
        return {
            "status": "error",
            "error": "Input validation failed",
            "details": validation_errors
        }
    
    raw_records = data["raw_records"]
    
    try:
        # Process each record
        processed_records = []
        claim_ids = []
        compatibility_data = []
        mismatch_data = []
        manual_scores = []
        icd10_codes = []
        
        for raw in raw_records:
            claim_id, feature_row, compatibility, mismatch, manual_score = build_features_from_raw(raw)
            claim_ids.append(claim_id)
            processed_records.append(feature_row)
            compatibility_data.append(compatibility)
            mismatch_data.append(mismatch)
            manual_scores.append(manual_score)
            icd10_codes.append(raw.get("icd10_primary_code", "UNKNOWN"))
        
        # Build feature DataFrame and predict
        df_features, dmatrix = build_feature_df(processed_records)
        
        # Get predictions
        y_raw = booster.predict(dmatrix)
        y_calibrated = calibrator.predict(y_raw)
        y_pred = (y_calibrated >= best_threshold).astype(int)
        
        # Build results
        results = []
        
        for i, claim_id in enumerate(claim_ids):
            row = df_features.iloc[i].to_dict()
            fraud_score = float(y_calibrated[i])
            model_flag = int(y_pred[i])
            manual_score = manual_scores[i]
            
            # Combine model score with manual score (weighted)
            final_score = (fraud_score * 0.7) + (manual_score * 0.3)
            final_flag = 1 if final_score > best_threshold else 0
            
            # Confidence (distance from threshold)
            confidence = abs(fraud_score - best_threshold) * 2
            confidence = min(confidence, 1.0)
            
            # Risk level
            if final_score > 0.8:
                risk_level = "TINGGI"
                risk_color = "🔴"
            elif final_score > 0.5:
                risk_level = "SEDANG"
                risk_color = "🟡"
            elif final_score > 0.3:
                risk_level = "RENDAH"
                risk_color = "🟢"
            else:
                risk_level = "MINIMAL"
                risk_color = "🟢"
            
            # Get detailed compatibility info
            raw_record = raw_records[i]
            procedures = raw_record.get("procedures", [])
            drugs = raw_record.get("drugs", [])
            vitamins = raw_record.get("vitamins", [])
            
            # Ensure lists
            if not isinstance(procedures, list):
                procedures = [procedures] if procedures else []
            if not isinstance(drugs, list):
                drugs = [drugs] if drugs else []
            if not isinstance(vitamins, list):
                vitamins = [vitamins] if vitamins else []
            
            compatibility_details = get_compatibility_details(
                icd10_codes[i], procedures, drugs, vitamins
            )
            
            # Generate explanation
            explanation = generate_explanation(row, final_score, icd10_codes[i], compatibility_details)
            
            # Top risk factors
            risk_factors = get_top_risk_factors(row, feature_importance_map, top_n=5)
            
            # Recommendations
            recommendations = get_recommendation(
                final_score, row["mismatch_count"], 
                row["biaya_anomaly_score"], row["total_claim_amount"]
            )
            
            # Cost assessment
            cost_assessment = get_cost_assessment(row["total_claim_amount"])
            
            # Clinical compatibility summary
            clinical_compat = {
                "procedure_compatible": row["diagnosis_procedure_score"] >= 0.5,
                "drug_compatible": row["diagnosis_drug_score"] >= 0.5,
                "vitamin_compatible": row["diagnosis_vitamin_score"] >= 0.5,
                "overall_compatible": row["mismatch_count"] == 0,
                "procedure_score": round(row["diagnosis_procedure_score"], 3),
                "drug_score": round(row["diagnosis_drug_score"], 3),
                "vitamin_score": round(row["diagnosis_vitamin_score"], 3),
                "details": compatibility_details
            }
            
            results.append({
                "claim_id": claim_id,
                "fraud_score": round(final_score, 4),
                "fraud_probability": f"{final_score * 100:.1f}%",
                "model_score": round(fraud_score, 4),
                "manual_score": round(manual_score, 4),
                "model_flag": model_flag,
                "final_flag": final_flag,
                "risk_level": risk_level,
                "risk_color": risk_color,
                "confidence": round(confidence, 4),
                "explanation": explanation,
                "recommendations": recommendations,
                "top_risk_factors": risk_factors,
                "clinical_compatibility": clinical_compat,
                "cost_assessment": cost_assessment,
                "features": {
                    "mismatch_count": int(row["mismatch_count"]),
                    "cost_anomaly_score": int(row["biaya_anomaly_score"]),
                    "total_claim_amount": float(row["total_claim_amount"]),
                    "patient_age": int(row["patient_age"]),
                    "patient_frequency": int(row["patient_frequency_risk"]),
                    "diagnosis_procedure_score": round(row["diagnosis_procedure_score"], 3),
                    "diagnosis_drug_score": round(row["diagnosis_drug_score"], 3),
                    "diagnosis_vitamin_score": round(row["diagnosis_vitamin_score"], 3),
                }
            })
        
        # Summary statistics
        fraud_detected = sum(1 for r in results if r["final_flag"] == 1)
        total_amount = sum(r["features"]["total_claim_amount"] for r in results)
        avg_risk_score = np.mean([r["fraud_score"] for r in results])
        
        return {
            "status": "success",
            "model_version": model_meta.get("model_version", "unknown"),
            "timestamp": datetime.now().isoformat(),
            "iceberg_available": spark is not None,
            "summary": {
                "total_claims_processed": len(results),
                "fraud_detected": fraud_detected,
                "fraud_percentage": f"{fraud_detected/len(results)*100:.1f}%",
                "total_amount_analyzed": total_amount,
                "average_risk_score": round(avg_risk_score, 3),
                "high_risk_claims": sum(1 for r in results if r["fraud_score"] > 0.8),
                "medium_risk_claims": sum(1 for r in results if r["fraud_score"] > 0.5),
            },
            "results": results,
            "model_info": {
                "threshold": best_threshold,
                "training_auc": model_meta.get("performance", {}).get("auc", 0),
                "training_f1": model_meta.get("performance", {}).get("f1", 0),
                "fraud_detection_rate": model_meta.get("performance", {}).get("fraud_detection_rate", 0),
            }
        }
    
    except Exception as e:
        import traceback
        return {
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc()
        }


# ================================================================
# HEALTH CHECK ENDPOINT
# ================================================================

@models.cml_model
def health_check(data: Dict[str, Any]) -> Dict[str, Any]:
    """Health check endpoint"""
    iceberg_status = "connected" if spark is not None else "disconnected"
    
    # Check if reference tables are loaded
    loaded_tables = []
    if spark is not None:
        for table_name, table_df in ref_tables.items():
            if table_df is not None:
                loaded_tables.append(table_name)
    
    return {
        "status": "healthy",
        "model_version": model_meta.get("model_version", "unknown"),
        "timestamp": datetime.now().isoformat(),
        "features_count": len(feature_names),
        "threshold": best_threshold,
        "iceberg_status": iceberg_status,
        "loaded_reference_tables": loaded_tables,
        "supported_diagnoses": len(COMPAT_RULES_FALLBACK),
        "model_accuracy": model_meta.get("performance", {}).get("auc", 0),
    }


# ================================================================
# MODEL INFO ENDPOINT
# ================================================================

@models.cml_model
def get_model_info(data: Dict[str, Any]) -> Dict[str, Any]:
    """Return model metadata and feature importance"""
    return {
        "status": "success",
        "model_metadata": {
            "version": model_meta.get("model_version"),
            "training_date": model_meta.get("training_date"),
            "purpose": model_meta.get("description"),
            "dataset": model_meta.get("dataset"),
        },
        "performance": model_meta.get("performance", {}),
        "feature_importance": GLOBAL_FEATURE_IMPORTANCE[:15],
        "features": {
            "numeric": numeric_cols,
            "categorical": categorical_cols,
            "total": len(feature_names)
        },
        "compatibility_rules": {
            "source": "Iceberg" if spark is not None else "Fallback",
            "count": len(COMPAT_RULES_FALLBACK),
            "supported_diagnoses": list(COMPAT_RULES_FALLBACK.keys()),
        },
        "cost_thresholds": COST_THRESHOLDS,
        "fraud_patterns": {k: {"description": v["description"], "severity": v["severity"]} 
                          for k, v in FRAUD_PATTERNS.items()}
    }


# ================================================================
# SINGLE CLAIM PREDICTION (SIMPLIFIED)
# ================================================================

@models.cml_model
def predict_single(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Simplified endpoint for single claim prediction.
    """
    # Wrap single record in raw_records format
    if "raw_records" not in data and any(key in data for key in ["claim_id", "icd10_primary_code"]):
        data = {"raw_records": [data]}
    
    return predict(data)


# ================================================================
# BATCH PREDICTION WITH CSV UPLOAD
# ================================================================

@models.cml_model
def predict_batch(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Batch prediction endpoint for CSV data.
    
    Input format:
    {
        "csv_data": "claim_id,patient_dob,visit_date,...\n12345,1980-01-01,2024-01-15,...",
        "format": "csv"
    }
    """
    try:
        if "csv_data" not in data:
            return {
                "status": "error",
                "error": "Missing 'csv_data' field"
            }
        
        # Parse CSV
        csv_content = data["csv_data"]
        df_csv = pd.read_csv(pd.compat.StringIO(csv_content))
        
        # Convert to raw_records format
        raw_records = []
        for _, row in df_csv.iterrows():
            record = row.to_dict()
            
            # Handle list fields (comma-separated)
            for field in ["procedures", "drugs", "vitamins"]:
                if field in record and isinstance(record[field], str):
                    record[field] = [item.strip() for item in record[field].split(",") if item.strip()]
            
            raw_records.append(record)
        
        # Call main predict function
        return predict({"raw_records": raw_records})
        
    except Exception as e:
        import traceback
        return {
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc()
        }


# ================================================================
# STARTUP MESSAGE
# ================================================================

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("OPERASIONAL FRAUD DETECTION MODEL - INFERENCE API READY")
    print("Iceberg Integrated Version")
    print("=" * 80)
    print("\nAvailable endpoints:")
    print("  1. predict() - Main batch fraud detection endpoint")
    print("  2. predict_single() - Single claim prediction")
    print("  3. predict_batch() - Batch prediction from CSV")
    print("  4. health_check() - Model health status")
    print("  5. get_model_info() - Model metadata and rules")
    print("\nModel capabilities:")
    print("  ✓ Fraud score prediction (0-100%)")
    print("  ✓ Clinical compatibility checking via Iceberg")
    print("  ✓ Diagnosis vs Procedure compatibility")
    print("  ✓ Diagnosis vs Drug compatibility")
    print("  ✓ Diagnosis vs Vitamin compatibility")
    print("  ✓ Cost anomaly detection with OPERASIONAL thresholds")
    print("  ✓ Actionable recommendations for reviewers")
    print("  ✓ Manual risk score calculation")
    print(f"  ✓ Iceberg integration: {'✅ Connected' if spark is not None else '⚠ Disconnected'}")
    print("\n" + "=" * 80)