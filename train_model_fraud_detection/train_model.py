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

n_rows = df_spark.count()
print(f"Loaded {n_rows} rows from curated feature table.")

# Only columns needed for training
df_spark = df_spark.select(
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
    "frequency_risk",
    "fraud_label"
)

# Convert to pandas
print("Converting Spark DataFrame to pandas ...")
df = df_spark.toPandas()
print(f"Pandas shape: {df.shape}")

# Ensure correct dtypes
df["fraud_label"] = df["fraud_label"].fillna(0).astype(int)

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
# HELPER: MULTI-HOT ENCODING UNTUK ARRAY KODE
# -------------------------------------------------------------------
def build_multi_hot(df, col, prefix):
    """
    df[col] berisi list kode (procedures/drugs/vitamins).
    Hasil: menambah kolom prefix_<code> = 0/1
    """
    all_codes = sorted({
        code
        for lst in df[col].dropna()
        for code in (lst if isinstance(lst, (list, tuple)) else [])
    })

    feature_cols = []
    for code in all_codes:
        fname = f"{prefix}_{code}"
        df[fname] = df[col].apply(
            lambda xs: int(isinstance(xs, (list, tuple)) and code in xs)
        )
        feature_cols.append(fname)

    print(f"Built {len(feature_cols)} multi-hot features for {col} ({prefix})")
    return df, feature_cols


# -------------------------------------------------------------------
# FEATURE ENGINEERING: DX + PROC + DRUG + VIT
# -------------------------------------------------------------------

# 1) One-hot untuk primary_dx_code (5 diagnosis utama)
dx_dummies = pd.get_dummies(df["primary_dx_code"].fillna("UNKNOWN"), prefix="dx")
dx_feature_cols = list(dx_dummies.columns)
df = pd.concat([df, dx_dummies], axis=1)
print(f"DX one-hot features: {dx_feature_cols}")

# 2) Multi-hot untuk procedures, drugs, vitamins
df, proc_feature_cols = build_multi_hot(df, "procedures", "proc")
df, drug_feature_cols = build_multi_hot(df, "drugs", "drug")
df, vit_feature_cols  = build_multi_hot(df, "vitamins", "vit")

# -------------------------------------------------------------------
# FINAL FEATURE SET
# -------------------------------------------------------------------
feature_cols = (
    num_cols
    + dx_feature_cols
    + proc_feature_cols
    + drug_feature_cols
    + vit_feature_cols
)

print("Feature columns used:")
for c in feature_cols:
    print("  -", c)

X = df[feature_cols].values
y = df["fraud_label"].values

print(f"X shape: {X.shape}, y shape: {y.shape}")

# -------------------------------------------------------------------
# TRAIN / VAL / TEST SPLIT
# -------------------------------------------------------------------
# First: train+val vs test
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)

# Split lagi train vs val
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.1765,  # kira-kira 15% total data
    random_state=42, stratify=y_trainval
)

print(f"Train size : {X_train.shape[0]}")
print(f"Val size   : {X_val.shape[0]}")
print(f"Test size  : {X_test.shape[0]}")

# -------------------------------------------------------------------
# TRAIN MODEL (RandomForest, class_weight balanced)
# -------------------------------------------------------------------
print("Training model...")

rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_split=10,
    min_samples_leaf=5,
    n_jobs=-1,
    class_weight="balanced",
    random_state=42
)

rf.fit(X_train, y_train)

# -------------------------------------------------------------------
# VALIDATION METRICS + THRESHOLD TUNING
# -------------------------------------------------------------------
val_proba = rf.predict_proba(X_val)[:, 1]

val_roc = roc_auc_score(y_val, val_proba)
val_pr  = average_precision_score(y_val, val_proba)

print(f"Validation ROC AUC     : {val_roc:.4f}")
print(f"Validation PR AUC (AP) : {val_pr:.4f}")

# Cari threshold terbaik (grid search sederhana)
best_thr = 0.5
best_f1 = -1.0

for thr in np.linspace(0.1, 0.9, 17):
    val_pred = (val_proba >= thr).astype(int)
    tp = np.sum((val_pred == 1) & (y_val == 1))
    fp = np.sum((val_pred == 1) & (y_val == 0))
    fn = np.sum((val_pred == 0) & (y_val == 1))

    precision = tp / (tp + fp + 1e-9)
    recall    = tp / (tp + fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)

    if f1 > best_f1:
        best_f1 = f1
        best_thr = thr

print(f"Best threshold (val)   : {best_thr:.4f}")
print(f"Best F1 at this thr    : {best_f1:.4f}")

# -------------------------------------------------------------------
# TEST METRICS
# -------------------------------------------------------------------
test_proba = rf.predict_proba(X_test)[:, 1]
test_roc   = roc_auc_score(y_test, test_proba)
test_pr    = average_precision_score(y_test, test_proba)

print("\n=== Test metrics (using tuned threshold) ===")
print(f"ROC AUC : {test_roc:.4f}")
print(f"PR AUC  : {test_pr:.4f}")

test_pred = (test_proba >= best_thr).astype(int)

print("\nClassification Report:")
print(classification_report(y_test, test_pred, digits=4))

print("Confusion Matrix:")
print(confusion_matrix(y_test, test_pred))

# -------------------------------------------------------------------
# SAVE MODEL + CONFIG LOCALLY
# -------------------------------------------------------------------
os.makedirs("./model_v3", exist_ok=True)
model_path = "./model_v3/fraud_model_v3.pkl"
config_path = "./model_v3/feature_config_v3.json"

# We store model + feature_cols + threshold in one dict
import joblib

model_bundle = {
    "model_type": "RandomForestClassifier",
    "model": rf,
    "feature_columns": feature_cols,
    "best_threshold": float(best_thr),
}

joblib.dump(model_bundle, model_path)
print(f"\nModel saved to: {model_path}")

config = {
    "table_source": table_name,
    "numeric_features": num_cols,
    "dx_features": dx_feature_cols,
    "procedure_features": proc_feature_cols,
    "drug_features": drug_feature_cols,
    "vitamin_features": vit_feature_cols,
    "all_features": feature_cols,
    "best_threshold": float(best_thr),
}

with open(config_path, "w") as f:
    json.dump(config, f, indent=2)

print(f"Config saved to: {config_path}")

# -------------------------------------------------------------------
# LOG TO MLFLOW (OPTIONAL, BUT NICE IN CML)
# -------------------------------------------------------------------
mlflow.set_experiment("fraud_detection_v3")

with mlflow.start_run(run_name="fraud_model_v3_clinical_cost_freq"):
    mlflow.log_param("model_type", "RandomForestClassifier")
    mlflow.log_param("n_estimators", rf.n_estimators)
    mlflow.log_param("class_weight", "balanced")
    mlflow.log_param("best_threshold", best_thr)

    mlflow.log_metric("val_roc_auc", val_roc)
    mlflow.log_metric("val_pr_auc", val_pr)
    mlflow.log_metric("test_roc_auc", test_roc)
    mlflow.log_metric("test_pr_auc", test_pr)
    mlflow.log_metric("best_f1_val", best_f1)

    mlflow.sklearn.log_model(rf, "model")
    mlflow.log_artifact(config_path, artifact_path="config")
    mlflow.log_artifact(model_path, artifact_path="model_bundle")

print("\n=== Training V3 completed successfully ===")