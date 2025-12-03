#!/usr/bin/env python3
import os
import json

import cml.data_v1 as cmldata
from pyspark.sql import functions as F

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix
)

import mlflow
import mlflow.sklearn
import joblib


print("=== START TRAINING — Fraud Model V3 (Clinical + Cost + Frequency) ===")

# -------------------------------------------------------------------
# CONNECT TO SPARK
# -------------------------------------------------------------------
conn = cmldata.get_connection("CDP-MSI")
spark = conn.get_spark_session()

# -------------------------------------------------------------------
# LOAD CURATED FEATURE TABLE V3
# -------------------------------------------------------------------
table_name = "iceberg_curated.claim_feature_set_v3"
print(f"Loading data from {table_name} ...")

df_spark = spark.read.format("iceberg").load(table_name)
print(f"Loaded {df_spark.count()} rows.")

df_spark = df_spark.select(
    "primary_dx_code",
    "procedures",
    "drugs",
    "vitamins",
    "total_procedure_cost",
    "total_drug_cost",
    "total_vitamin_cost",
    "total_claim_amount",
    "cost_anomaly_score",
    "frequency_risk",
    "fraud_label"
)

print("Converting Spark DF → pandas ...")
df = df_spark.toPandas()

# -------------------------------------------------------------------
# SAFETY CHECK → FIX MISSING LABELS
# -------------------------------------------------------------------
df["fraud_label"] = df["fraud_label"].fillna(0).astype(int)

label_counts = df["fraud_label"].value_counts()
print("\nLabel distribution:")
print(label_counts)

if len(label_counts) < 2:
    print("\n❌ ERROR: Dataset hanya punya 1 kelas. Model tidak bisa training.")
    print("   Solusi: periksa ETL atau generator untuk menghasilkan label 0/1.")
    raise SystemExit

# -------------------------------------------------------------------
# FIX NUMERICAL COLUMNS
# -------------------------------------------------------------------
num_cols = [
    "total_procedure_cost",
    "total_drug_cost",
    "total_vitamin_cost",
    "total_claim_amount",
    "cost_anomaly_score",
    "frequency_risk"
]

for c in num_cols:
    df[c] = df[c].fillna(0).astype(float)

# -------------------------------------------------------------------
# MULTI-HOT ENCODER (SAFE VERSION)
# -------------------------------------------------------------------
def build_multi_hot(df, col, prefix):
    all_codes = sorted({
        code
        for lst in df[col].dropna()
        for code in (lst if isinstance(lst, (list, tuple)) else [])
    })

    if len(all_codes) == 0:
        print(f"[WARN] No values found in {col}, skipping.")
        return df, []

    feature_cols = []
    for code in all_codes:
        colname = f"{prefix}_{code}"
        df[colname] = df[col].apply(
            lambda xs: int(isinstance(xs, (list, tuple)) and code in xs)
        )
        feature_cols.append(colname)

    print(f"Built {len(feature_cols)} features for {col}")
    return df, feature_cols


# -------------------------------------------------------------------
# DX ONE-HOT
# -------------------------------------------------------------------
dx_dummies = pd.get_dummies(df["primary_dx_code"].fillna("UNKNOWN"), prefix="dx")
dx_cols = list(dx_dummies.columns)
df = pd.concat([df, dx_dummies], axis=1)


# -------------------------------------------------------------------
# MULTI-HOT FOR PROCEDURE/DRUG/VITAMIN
# -------------------------------------------------------------------
df, proc_cols = build_multi_hot(df, "procedures", "proc")
df, drug_cols = build_multi_hot(df, "drugs", "drug")
df, vit_cols  = build_multi_hot(df, "vitamins", "vit")


# -------------------------------------------------------------------
# FINAL FEATURE LIST
# -------------------------------------------------------------------
feature_cols = (
    num_cols + dx_cols + proc_cols + drug_cols + vit_cols
)

print("\n=== FINAL FEATURES USED ===")
for c in feature_cols:
    print(" -", c)

X = df[feature_cols].values
y = df["fraud_label"].values


# -------------------------------------------------------------------
# TRAIN/VAL/TEST SPLIT (SAFE STRATIFICATION)
# -------------------------------------------------------------------
def safe_split(X, y, test_size, random_state):
    if len(np.unique(y)) < 2:
        strat = None
    else:
        strat = y
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=strat)


X_trainval, X_test, y_trainval, y_test = safe_split(X, y, 0.15, 42)
X_train, X_val, y_train, y_val = safe_split(X_trainval, y_trainval, 0.1765, 42)

print(f"\nTrain size: {len(y_train)}")
print(f"Val size  : {len(y_val)}")
print(f"Test size : {len(y_test)}")


# -------------------------------------------------------------------
# TRAIN MODEL (CLASS-WEIGHT BALANCED)
# -------------------------------------------------------------------
print("\nTraining RandomForest V3 ...")

rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_split=10,
    min_samples_leaf=5,
    class_weight="balanced",
    n_jobs=-1,
    random_state=42
)

rf.fit(X_train, y_train)

# SAFETY FIX — predict_proba
def get_proba(model, X):
    proba = model.predict_proba(X)
    if proba.shape[1] == 1:
        # Only class 0 → return proba=0 for fraud
        return np.zeros(len(X))
    return proba[:, 1]

val_proba = get_proba(rf, X_val)

# -------------------------------------------------------------------
# VALIDATION METRICS
# -------------------------------------------------------------------
val_roc = roc_auc_score(y_val, val_proba)
val_pr  = average_precision_score(y_val, val_proba)

print(f"\nValidation ROC AUC  : {val_roc:.4f}")
print(f"Validation PR AUC   : {val_pr:.4f}")

# -------------------------------------------------------------------
# THRESHOLD TUNING
# -------------------------------------------------------------------
best_thr = 0.5
best_f1 = 0

for thr in np.linspace(0.1, 0.9, 17):
    pred = (val_proba >= thr).astype(int)

    tp = np.sum((pred == 1) & (y_val == 1))
    fp = np.sum((pred == 1) & (y_val == 0))
    fn = np.sum((pred == 0) & (y_val == 1))

    prec = tp / (tp + fp + 1e-9)
    rec  = tp / (tp + fn + 1e-9)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)

    if f1 > best_f1:
        best_f1 = f1
        best_thr = thr

print(f"Best threshold : {best_thr:.4f}")
print(f"Best F1 score  : {best_f1:.4f}")

# -------------------------------------------------------------------
# TEST METRICS
# -------------------------------------------------------------------
test_proba = get_proba(rf, X_test)

test_roc = roc_auc_score(y_test, test_proba)
test_pr  = average_precision_score(y_test, test_proba)

print("\n=== TEST RESULTS ===")
print(f"ROC AUC : {test_roc:.4f}")
print(f"PR AUC  : {test_pr:.4f}")

test_pred = (test_proba >= best_thr).astype(int)

print("\nClassification Report:")
print(classification_report(y_test, test_pred, digits=4))

print("Confusion Matrix:")
print(confusion_matrix(y_test, test_pred))


# -------------------------------------------------------------------
# SAVE MODEL BUNDLE
# -------------------------------------------------------------------
os.makedirs("./model_v3", exist_ok=True)
bundle = {
    "model": rf,
    "best_threshold": float(best_thr),
    "feature_columns": feature_cols
}
joblib.dump(bundle, "./model_v3/fraud_model_v3.pkl")

print("\nModel saved → ./model_v3/fraud_model_v3.pkl")

with open("./model_v3/feature_config_v3.json", "w") as f:
    json.dump({
        "feature_cols": feature_cols,
        "best_threshold": float(best_thr),
    }, f, indent=2)


# -------------------------------------------------------------------
# MLFLOW LOGGING
# -------------------------------------------------------------------
mlflow.set_experiment("fraud_detection_v3_fixed")

with mlflow.start_run(run_name="fraud_model_v3_fixed"):
    mlflow.log_param("model", "RandomForestClassifier")
    mlflow.log_param("best_threshold", best_thr)
    mlflow.log_metric("val_roc", val_roc)
    mlflow.log_metric("val_pr", val_pr)
    mlflow.log_metric("test_roc", test_roc)
    mlflow.log_metric("test_pr", test_pr)
    mlflow.sklearn.log_model(rf, "model")

print("\n=== TRAINING COMPLETED (FIXED V3) ===")