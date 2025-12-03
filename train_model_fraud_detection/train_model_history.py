#!/usr/bin/env python3

import os
import json
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
import cml.data_v1 as cmldata
from pyspark.sql.functions import col
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix, average_precision_score
)
from sklearn.isotonic import IsotonicRegression
from category_encoders import TargetEncoder
from imblearn.over_sampling import SMOTE

print("=" * 80)
print("ADVANCED FRAUD MODEL TRAINING - HISTORY & PROFILING EDITION")
print("=" * 80)

# ==============================================================================
# 1. SETUP & LOAD DATA
# ==============================================================================
print("\n[Step 1/10] Connecting to Spark & Loading Data...")
conn = cmldata.get_connection("CDP-MSI")
spark = conn.get_spark_session()

# Membaca tabel hasil ETL Advanced
df_spark = spark.table("iceberg_curated.claim_feature_set")

# [FIX] Menggunakan 'final_label' sebagai target (bukan 'label')
TARGET = "final_label"

# Filter data valid
df_spark = df_spark.filter(col(TARGET).isNotNull())

print(f"✓ Data Source: iceberg_curated.claim_feature_set")
print(f"✓ Total Rows: {df_spark.count():,}")

# Convert to Pandas for Training
df = df_spark.toPandas()
print(f"✓ Loaded into Pandas: {df.shape}")

# ==============================================================================
# 2. FEATURE DEFINITION
# ==============================================================================
print("\n[Step 2/10] Defining Features...")

# --- Fitur Numerik (Hasil Windowing & Profiling) ---
NUMERIC_FEATURES = [
    # Biaya
    "total_claim_amount", "total_procedure_cost", "total_drug_cost", "total_vitamin_cost",
    
    # Clinical Consistency Scores (0.0 - 1.0)
    "diagnosis_procedure_score", "diagnosis_drug_score", "diagnosis_vitamin_score",
    
    # History Features (Windowing)
    "patient_frequency_risk",   # patient_visit_last_30d di rename di ETL
    "patient_amount_last_30d",
    "days_since_last_visit",
    
    # Statistical Features (Profiling)
    "biaya_anomaly_score",      # cost_z_score binning di ETL
    "mismatch_count"
]

# --- Fitur Kategorikal ---
CATEGORICAL_FEATURES = [
    "department", 
    "icd10_primary_code",
    "doctor_name" 
]

print(f"✓ Numeric Features: {len(NUMERIC_FEATURES)}")
print(f"✓ Categorical Features: {len(CATEGORICAL_FEATURES)}")

# ==============================================================================
# 3. PREPROCESSING
# ==============================================================================
print("\n[Step 3/10] Cleaning & Preprocessing...")

# Fill Nulls
for col_name in NUMERIC_FEATURES:
    if col_name in df.columns:
        df[col_name] = pd.to_numeric(df[col_name], errors='coerce').fillna(0)
    else:
        print(f"⚠ Warning: Feature {col_name} not found in dataset. Filling with 0.")
        df[col_name] = 0

for col_name in CATEGORICAL_FEATURES:
    if col_name in df.columns:
        df[col_name] = df[col_name].fillna("UNKNOWN").astype(str)
    else:
        print(f"⚠ Warning: Feature {col_name} not found in dataset. Filling with UNKNOWN.")
        df[col_name] = "UNKNOWN"

# ==============================================================================
# 4. SPLITTING DATA
# ==============================================================================
print("\n[Step 4/10] Splitting Train/Test...")

X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
y = df[TARGET].astype(int)

# Stratified Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"✓ Train shape: {X_train.shape}")
print(f"✓ Test shape:  {X_test.shape}")
print(f"✓ Fraud Rate (Train): {y_train.mean():.2%}")

# ==============================================================================
# 5. ENCODING (TARGET ENCODER)
# ==============================================================================
print("\n[Step 5/10] Encoding Categorical Features...")

encoder = TargetEncoder(cols=CATEGORICAL_FEATURES, smoothing=1.0)
X_train_enc = encoder.fit_transform(X_train, y_train)
X_test_enc = encoder.transform(X_test)

print("✓ Target Encoding applied")

# ==============================================================================
# 6. HANDLING IMBALANCE (SMOTE)
# ==============================================================================
print("\n[Step 6/10] Handling Class Imbalance...")

if y_train.mean() < 0.15:
    print("  Applying SMOTE...")
    smote = SMOTE(random_state=42, sampling_strategy=0.3)
    X_train_res, y_train_res = smote.fit_resample(X_train_enc, y_train)
    print(f"  ✓ SMOTE applied. New Train Size: {X_train_res.shape}")
else:
    print("  Skipping SMOTE (Fraud ratio sufficient)")
    X_train_res, y_train_res = X_train_enc, y_train

# ==============================================================================
# 7. TRAINING XGBOOST
# ==============================================================================
print("\n[Step 7/10] Training XGBoost Model...")

scale_pos_weight = (y_train_res == 0).sum() / (y_train_res == 1).sum()

model = xgb.XGBClassifier(
    objective="binary:logistic",
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    eval_metric="aucpr",
    early_stopping_rounds=50,
    random_state=42,
    n_jobs=-1
)

model.fit(
    X_train_res, y_train_res,
    eval_set=[(X_test_enc, y_test)],
    verbose=50
)

print(f"✓ Training finished. Best Iteration: {model.best_iteration}")

# ==============================================================================
# 8. CALIBRATION
# ==============================================================================
print("\n[Step 8/10] Calibrating Probabilities...")

y_pred_raw = model.predict_proba(X_test_enc)[:, 1]

calibrator = IsotonicRegression(out_of_bounds="clip")
calibrator.fit(y_pred_raw, y_test)
y_pred_calib = calibrator.predict(y_pred_raw)

print("✓ Model output calibrated")

# ==============================================================================
# 9. EVALUATION
# ==============================================================================
print("\n" + "=" * 40)
print("MODEL EVALUATION")
print("=" * 40)

thresholds = np.arange(0.1, 0.9, 0.05)
best_th = 0.5
best_f1 = 0
for th in thresholds:
    f1 = f1_score(y_test, (y_pred_calib >= th).astype(int))
    if f1 > best_f1:
        best_f1 = f1
        best_th = th

y_final_pred = (y_pred_calib >= best_th).astype(int)

print(f"Optimal Threshold : {best_th:.2f}")
print(f"AUC-ROC           : {roc_auc_score(y_test, y_pred_calib):.4f}")
print(f"AUC-PR            : {average_precision_score(y_test, y_pred_calib):.4f}")
print(f"F1 Score          : {f1_score(y_test, y_final_pred):.4f}")
print("-" * 40)
print(classification_report(y_test, y_final_pred))
print("-" * 40)

# Feature Importance
print("\nTOP 10 RISK FACTORS (Feature Importance):")
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
for i in range(10):
    feat = X_train_enc.columns[indices[i]]
    score = importances[indices[i]]
    print(f"{i+1}. {feat:<30} : {score:.4f}")

# ==============================================================================
# 10. SAVING ARTIFACTS
# ==============================================================================
print("\n[Step 10/10] Saving Artifacts for Deployment...")

artifacts = {
    "model": model,
    "encoder": encoder,
    "calibrator": calibrator,
    "threshold": best_th,
    "features": {
        "numeric": NUMERIC_FEATURES,
        "categorical": CATEGORICAL_FEATURES
    }
}

with open("fraud_model_history_v1.pkl", "wb") as f:
    pickle.dump(artifacts, f)

print("✓ Model saved to 'fraud_model_history_v1.pkl'")
print("\nSelesai! Model ini siap digunakan di script deployment (model.py).")
spark.stop()