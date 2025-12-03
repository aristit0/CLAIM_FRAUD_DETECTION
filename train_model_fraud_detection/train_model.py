#!/usr/bin/env python3
import os
import json
import numpy as np
import pandas as pd

import cml.data_v1 as cmldata
from pyspark.sql import functions as F

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    f1_score,
)
import joblib

import mlflow
import mlflow.sklearn


# =========================================================
# CONFIG
# =========================================================
ICEBERG_TABLE = "iceberg_curated.claim_feature_set"

MODEL_DIR = os.getenv("MODEL_DIR", "./model_v2")
os.makedirs(MODEL_DIR, exist_ok=True)

EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT", "fraud_detection_v2")


# =========================================================
# LOAD DATA FROM ICEBERG
# =========================================================
def load_feature_table():
    conn = cmldata.get_connection("CDP-MSI")
    spark = conn.get_spark_session()

    print(f"Loading data from {ICEBERG_TABLE} ...")
    sdf = spark.table(ICEBERG_TABLE)

    # Pastikan kolom yang kita butuhkan tersedia
    cols_needed = [
        "claim_id",
        "primary_dx",
        "primary_dx_desc",
        "total_procedure_cost",
        "total_drug_cost",
        "total_vitamin_cost",
        "total_claim_amount",
        "mandatory_proc_missed",
        "mandatory_drug_missed",
        "mandatory_vit_missed",
        "mandatory_missed_total",
        "procedure_mismatch_flag",
        "drug_mismatch_flag",
        "vitamin_mismatch_flag",
        "mismatch_count",
        "cost_anomaly_score",
        "frequency_risk",
        "rule_violation_flag",
        "final_label",
    ]

    sdf = sdf.select(*cols_needed).where(F.col("final_label").isNotNull())

    pdf = sdf.toPandas()
    print(f"Loaded {len(pdf)} rows from curated feature table.")
    return pdf


# =========================================================
# PREPARE FEATURES
# =========================================================
def prepare_features(pdf: pd.DataFrame):
    # Drop rows with missing primary_dx or label
    pdf = pdf.dropna(subset=["primary_dx", "final_label"]).copy()

    # Target
    y = pdf["final_label"].astype(int)

    # Base numeric features
    numeric_cols = [
        "total_procedure_cost",
        "total_drug_cost",
        "total_vitamin_cost",
        "total_claim_amount",
        "mandatory_proc_missed",
        "mandatory_drug_missed",
        "mandatory_vit_missed",
        "mandatory_missed_total",
        "procedure_mismatch_flag",
        "drug_mismatch_flag",
        "vitamin_mismatch_flag",
        "mismatch_count",
        "cost_anomaly_score",
        "frequency_risk",
        "rule_violation_flag",
    ]

    # One-hot primary_dx
    base_df = pdf[["primary_dx"] + numeric_cols].copy()
    X = pd.get_dummies(base_df, columns=["primary_dx"], drop_first=False)

    feature_names = list(X.columns)

    print("Feature columns used:")
    for c in feature_names:
        print("  -", c)

    return X, y, feature_names


# =========================================================
# FIND BEST THRESHOLD
# =========================================================
def find_best_threshold(y_true, y_proba, min_precision=0.5):
    """
    Cari threshold yang punya F1 terbaik,
    dengan constraint precision >= min_precision (default 0.5).
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)

    best_thr = 0.5
    best_f1 = 0.0

    # precision_recall_curve tidak punya thr untuk titik terakhir,
    # jadi iterasi sampai len(thresholds)
    for p, r, t in zip(precision[:-1], recall[:-1], thresholds):
        if p >= min_precision:
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
            if f1 > best_f1:
                best_f1 = f1
                best_thr = t

    return best_thr, best_f1


# =========================================================
# MAIN TRAINING
# =========================================================
def main():
    # MLflow setup
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="fraud_model_v2_training"):

        # 1. Load data
        pdf = load_feature_table()

        # 2. Prepare features
        X, y, feature_names = prepare_features(pdf)

        # 3. Train / val / test split
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.30, random_state=42, stratify=y
        )

        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
        )

        print(f"Train size : {len(X_train)}")
        print(f"Val size   : {len(X_val)}")
        print(f"Test size  : {len(X_test)}")

        # 4. Define model (RandomForest balanced)
        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=6,
            min_samples_leaf=50,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )

        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_param("n_estimators", 300)
        mlflow.log_param("max_depth", 6)
        mlflow.log_param("min_samples_leaf", 50)
        mlflow.log_param("class_weight", "balanced")

        # 5. Train
        print("Training model...")
        model.fit(X_train, y_train)

        # 6. Validation metrics + threshold tuning
        val_proba = model.predict_proba(X_val)[:, 1]
        val_auc = roc_auc_score(y_val, val_proba)
        val_ap = average_precision_score(y_val, val_proba)

        best_thr, best_f1 = find_best_threshold(y_val, val_proba, min_precision=0.5)

        print(f"Validation ROC AUC        : {val_auc:.4f}")
        print(f"Validation PR AUC (AP)    : {val_ap:.4f}")
        print(f"Best threshold (val)      : {best_thr:.4f}")
        print(f"Best F1 at this threshold : {best_f1:.4f}")

        mlflow.log_metric("val_roc_auc", float(val_auc))
        mlflow.log_metric("val_pr_auc", float(val_ap))
        mlflow.log_metric("best_threshold_val", float(best_thr))
        mlflow.log_metric("best_f1_val", float(best_f1))

        # 7. Evaluate on test set
        test_proba = model.predict_proba(X_test)[:, 1]
        test_auc = roc_auc_score(y_test, test_proba)
        test_ap = average_precision_score(y_test, test_proba)

        test_pred = (test_proba >= best_thr).astype(int)

        print("\n=== Test metrics (using tuned threshold) ===")
        print("ROC AUC :", test_auc)
        print("PR AUC  :", test_ap)
        print("\nClassification Report:\n", classification_report(y_test, test_pred))
        print("Confusion Matrix:\n", confusion_matrix(y_test, test_pred))

        mlflow.log_metric("test_roc_auc", float(test_auc))
        mlflow.log_metric("test_pr_auc", float(test_ap))

        # 8. Save model + feature config
        model_path = os.path.join(MODEL_DIR, "fraud_model_v2.pkl")
        config_path = os.path.join(MODEL_DIR, "feature_config_v2.json")

        joblib.dump(model, model_path)

        feature_config = {
            "feature_names": feature_names,
            "threshold": float(best_thr),
            "label_name": "final_label",
            "primary_dx_values": sorted(list(X["primary_dx_J06"].index)) if "primary_dx_J06" in X.columns else None,
        }

        with open(config_path, "w") as f:
            json.dump(feature_config, f, indent=2)

        print(f"\nModel saved to: {model_path}")
        print(f"Config saved to: {config_path}")

        # 9. Log to MLflow
        mlflow.sklearn.log_model(model, artifact_path="model")
        mlflow.log_artifact(config_path, artifact_path="config")
        mlflow.log_artifact(model_path, artifact_path="artifacts")

        print("\n=== Training completed successfully (v2) ===")


if __name__ == "__main__":
    main()