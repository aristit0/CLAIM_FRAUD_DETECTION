#!/usr/bin/env python3

import os
import json
import pickle
import pandas as pd
import numpy as np
import cml.data_v1 as cmldata
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt
from pyspark.sql.functions import col  # <-- ADD THIS LINE
from sklearn.model_selection import StratifiedKFold, train_test_split, GridSearchCV
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix
)
from sklearn.isotonic import IsotonicRegression
from category_encoders.target_encoder import TargetEncoder
from imblearn.over_sampling import SMOTE

# =====================================================
# 0. CONNECT TO SPARK
# =====================================================
print("=== CONNECTING TO SPARK ===")
conn = cmldata.get_connection("CDP-MSI")
spark = conn.get_spark_session()

# =====================================================
# 1. DEFINISI LABEL & FEATURE LIST (ALIGNED DENGAN ETL v5)
# =====================================================

label_col = "final_label"

# numeric features (sesuai schema df_spark.printSchema yang kamu kirim)
numeric_cols = [
    "patient_age",
    "total_procedure_cost",
    "total_drug_cost",
    "total_vitamin_cost",
    "total_claim_amount",
    "severity_score",
    "cost_per_procedure",
    "patient_claim_count",
    "biaya_anomaly_score",
    "cost_procedure_anomaly",
    "patient_frequency_risk",
    "visit_year",
    "visit_month",
    "visit_day",

    # CLINICAL COMPATIBILITY SCORES
    "diagnosis_procedure_score",
    "diagnosis_drug_score",
    "diagnosis_vitamin_score",
    "treatment_consistency_score",

    # EXPLICIT MISMATCH FLAGS
    "procedure_mismatch_flag",
    "drug_mismatch_flag",
    "vitamin_mismatch_flag",
    "mismatch_count",
]

# categorical yang kuat hubungannya dengan pola fraud
categorical_cols = [
    "visit_type",
    "department",
    "icd10_primary_code",
]

all_cols = numeric_cols + categorical_cols + [label_col]

print("Numeric features:", numeric_cols)
print("Categorical features:", categorical_cols)
print("Total features:", len(all_cols) - 1, "+ label")

# =====================================================
# 2. LOAD DATA DARI ICEBERG CURATED
# =====================================================
print("=== LOADING DATA FROM iceberg_curated.claim_feature_set ===")

df_spark = (
    spark.table("iceberg_curated.claim_feature_set")
         .where(col(label_col).isNotNull())
         .select(*all_cols)
)

# BATAS MAKSIMAL ROW UNTUK toPandas (kalau resource cukup bisa dinaikkan)
MAX_ROWS = 300_000
df_spark = df_spark.limit(MAX_ROWS)

print("Spark DF schema:")
df_spark.printSchema()

print("=== CONVERT TO PANDAS ===")
df = df_spark.toPandas()
print("Loaded pandas dataset:", df.shape)

# =====================================================
# 3. LABEL
# =====================================================
print("=== PREP LABEL ===")
df[label_col] = df[label_col].astype(int)

# (optional) log distribusi label
label_counts = df[label_col].value_counts()
print("Label distribution:")
print(label_counts)


# =====================================================
# 4. ENCODING CATEGORICAL (TARGET ENCODING)
# =====================================================
print("=== TARGET ENCODING CATEGORICAL ===")

encoders = {}

for c in categorical_cols:
    df[c] = df[c].fillna("__UNKNOWN__").astype(str)
    te = TargetEncoder(cols=[c], smoothing=0.3)
    df[c] = te.fit_transform(df[c], df[label_col])
    encoders[c] = te

# =====================================================
# 5. CLEAN NUMERIC
# =====================================================
print("=== CLEAN NUMERIC FEATURES ===")

for c in numeric_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

X = df[numeric_cols + categorical_cols]
y = df[label_col]

# =====================================================
# 6. HANDLE IMBALANCE (SMOTE)
# =====================================================
print("=== HANDLING IMBALANCE WITH SMOTE ===")

smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)

print(f"Resampled dataset shape {X_res.shape}")

# =====================================================
# 7. TRAIN / TEST SPLIT
# =====================================================
print("=== TRAIN / TEST SPLIT ===")
X_train, X_test, y_train, y_test = train_test_split(
    X_res, y_res,
    test_size=0.2,
    random_state=42,
    stratify=y_res
)

positive = int((y_train == 1).sum())
negative = int((y_train == 0).sum())
scale_pos_weight = negative / positive

print(
    f"Train size: {X_train.shape}, Pos: {positive}, "
    f"Neg: {negative}, scale_pos_weight: {scale_pos_weight:.4f}"
)

# =====================================================
# 8. XGBOOST Booster (xgb.train) + EARLY STOPPING
# =====================================================
print("=== PREPARE DMatrix ===")

feature_names = X.columns.tolist()

dtrain = xgb.DMatrix(
    X_train,
    label=y_train.values,
    feature_names=feature_names
)
dtest = xgb.DMatrix(
    X_test,
    label=y_test.values,
    feature_names=feature_names
)

params = {
    "objective": "binary:logistic",
    "eval_metric": "auc",         # untuk fraud VS non-fraud separation
    "eta": 0.025,
    "max_depth": 8,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "min_child_weight": 4,
    "gamma": 0.4,
    "reg_alpha": 0.2,
    "reg_lambda": 1.2,
    "tree_method": "hist",
    "scale_pos_weight": scale_pos_weight,
    "seed": 42,
}

print("=== TRAINING XGBOOST Booster with early stopping ===")
evals = [(dtrain, "train"), (dtest, "valid")]

model = xgb.train(
    params=params,
    dtrain=dtrain,
    num_boost_round=1000,
    evals=evals,
    early_stopping_rounds=60,
    verbose_eval=30,
)

print(f"Best iteration: {model.best_iteration}")
print("Best score (AUC):", model.best_score)


# =====================================================
# 9. CALIBRATION (ISOTONIC REGRESSION)
# =====================================================
print("=== CALIBRATING FRAUD SCORE (Isotonic Regression) ===")

# raw probability dari booster (uncalibrated)
y_proba_raw = model.predict(dtest)

iso = IsotonicRegression(out_of_bounds="clip")
iso.fit(y_proba_raw, y_test)

# calibrated probability
y_proba = iso.predict(y_proba_raw)


# =====================================================
# 10. THRESHOLD OPTIMIZATION (F1)
#    - supaya suspicious flag lebih akurat
# =====================================================
print("=== THRESHOLD TUNING (F1) ===")

thresholds = np.arange(0.1, 0.9, 0.02)
best_t = 0.5
best_f1 = 0.0

for t in thresholds:
    pred = (y_proba >= t).astype(int)
    f1 = f1_score(y_test, pred)
    if f1 > best_f1:
        best_f1 = f1
        best_t = float(t)

print(f"Best threshold: {best_t:.3f} | Best F1: {best_f1:.4f}")


# =====================================================
# 11. FINAL EVALUATION
# =====================================================
print("=== FINAL EVALUATION (CALIBRATED SCORE) ===")

y_pred = (y_proba >= best_t).astype(int)

auc = roc_auc_score(y_test, y_proba)
f1 = f1_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print(f"AUC       : {auc:.4f}")
print(f"F1        : {f1:.4f}")
print(f"Precision : {prec:.4f}")
print(f"Recall    : {rec:.4f}")
print("Confusion matrix [[TN FP] [FN TP]]:")
print(cm)
print("Classification report:")
print(classification_report(y_test, y_pred))


# =====================================================
# 12. FEATURE IMPORTANCE (GAIN)
# =====================================================
print("=== FEATURE IMPORTANCE (GAIN) ===")

raw_importances = model.get_score(importance_type="gain")

# langsung dipakai, tidak perlu mapping f0→nama kolom
feature_scores = dict(
    sorted(raw_importances.items(), key=lambda x: x[1], reverse=True)
)

print("Top 20 important features:")
for i, (k, v) in enumerate(feature_scores.items()):
    if i >= 20:
        break
    print(f"{i+1:2d}. {k}: {v:.4f}")


# =====================================================
# 13. SAVE ARTIFACTS (MODEL + CALIBRATOR + PREPROCESS META)
# =====================================================
print("=== SAVING ARTIFACTS ===")

ROOT = "/home/cdsw"
os.makedirs(ROOT, exist_ok=True)

# 13.1 Simpan XGBoost Booster (bentuk JSON)
model_path = os.path.join(ROOT, "model.json")
model.save_model(model_path)

# 13.2 Simpan calibrator
calib_path = os.path.join(ROOT, "calibrator.pkl")
with open(calib_path, "wb") as f:
    pickle.dump(iso, f)

# 13.3 Simpan preprocessing config
preprocess = {
    "numeric_cols": numeric_cols,
    "categorical_cols": categorical_cols,
    "encoders": encoders,
    "best_threshold": float(best_t),
    "feature_importance": feature_scores,
    "label_col": label_col,
}

preprocess_path = os.path.join(ROOT, "preprocess.pkl")
with open(preprocess_path, "wb") as f:
    pickle.dump(preprocess, f)

# 13.4 Meta info (buat tracking versi di app)
meta_path = os.path.join(ROOT, "meta.json")
with open(meta_path, "w") as f:
    json.dump(
        {
            "description": "Fraud model v5 — mismatch-aware + calibrated",
            "label_source": "human+rules",
            "version": "v5",
            "n_rows": int(df.shape[0]),
            "n_features": int(len(feature_names)),
        },
        f,
        indent=2,
    )

print("Saved:")
print(" -", model_path)
print(" -", calib_path)
print(" -", preprocess_path)
print(" -", meta_path)

print("=== TRAINING COMPLETE ===")
spark.stop()