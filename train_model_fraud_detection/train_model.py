#!/usr/bin/env python3

import cml.data_v1 as cmldata
from pyspark.sql.functions import col
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    classification_report
)
from category_encoders.target_encoder import TargetEncoder
from sklearn.isotonic import IsotonicRegression
import xgboost as xgb
import os, json, pickle

print("=== CONNECTING TO SPARK ===")
conn = cmldata.get_connection("CDP-MSI")
spark = conn.get_spark_session()

# =====================================================
# DEFINISI LABEL & FEATURE LIST (ALIGNED DENGAN ETL v5)
# =====================================================

label_col = "final_label"

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
    "visit_year", "visit_month", "visit_day",

    # clinical compatibility scores
    "diagnosis_procedure_score",
    "diagnosis_drug_score",
    "diagnosis_vitamin_score",
    "treatment_consistency_score",

    # explicit mismatch flags
    "procedure_mismatch_flag",
    "drug_mismatch_flag",
    "vitamin_mismatch_flag",
    "mismatch_count",
]

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
# LOAD DATA DARI SPARK (HANYA KOLOM YANG DIPAKAI)
# =====================================================

print("=== LOADING DATA FROM ICEBERG (FEATURE_SET) ===")
df_spark = (
    spark.table("iceberg_curated.claim_feature_set")
         .where(col(label_col).isNotNull())
         .select(*all_cols)
)

# OPTIONAL: batasi max rows supaya aman buat toPandas
MAX_ROWS = 300_000  # bisa dinaikkan kalau resource cukup
df_spark = df_spark.limit(MAX_ROWS)

print("Spark DF schema:")
df_spark.printSchema()

print("=== CONVERT TO PANDAS ===")
df = df_spark.toPandas()
print("Loaded pandas dataset:", df.shape)

# =====================================================
# LABEL
# =====================================================
df[label_col] = df[label_col].astype(int)

# =====================================================
# ENCODING CATEGORICAL
# =====================================================
encoders = {}

for c in categorical_cols:
    df[c] = df[c].fillna("__UNKNOWN__").astype(str)
    te = TargetEncoder(cols=[c], smoothing=0.3)
    df[c] = te.fit_transform(df[c], df[label_col])
    encoders[c] = te

# =====================================================
# CLEAN NUMERIC
# =====================================================
for c in numeric_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

X = df[numeric_cols + categorical_cols]
y = df[label_col]

# =====================================================
# TRAIN/TEST SPLIT
# =====================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

positive = (y_train == 1).sum()
negative = (y_train == 0).sum()
scale_pos_weight = negative / positive
print("Train size:", X_train.shape, "Pos:", positive, "Neg:", negative,
      "scale_pos_weight:", scale_pos_weight)

# =====================================================
# XGBOOST MODEL (v5 IMPROVED)
# =====================================================
model = xgb.XGBClassifier(
    n_estimators=700,
    max_depth=8,
    learning_rate=0.025,
    subsample=0.9,
    colsample_bytree=0.9,
    min_child_weight=4,
    gamma=0.4,
    reg_alpha=0.2,
    reg_lambda=1.2,
    objective="binary:logistic",
    eval_metric="auc",
    tree_method="hist",
    scale_pos_weight=scale_pos_weight,
)

print("=== TRAINING MODEL with early stopping ===")
model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    early_stopping_rounds=60,
    verbose=30
)

# =====================================================
# CALIBRATION (ISOTONIC REGRESSION)
# =====================================================
print("=== CALIBRATING FRAUD SCORE ===")
y_proba_raw = model.predict_proba(X_test)[:, 1]

iso = IsotonicRegression(out_of_bounds="clip")
iso.fit(y_proba_raw, y_test)

y_proba = iso.predict(y_proba_raw)

# =====================================================
# THRESHOLD OPTIMIZATION (F1)
# =====================================================
thresholds = np.arange(0.1, 0.9, 0.02)
best_t = 0.5
best_f1 = 0

for t in thresholds:
    pred = (y_proba >= t).astype(int)
    f1 = f1_score(y_test, pred)
    if f1 > best_f1:
        best_f1 = f1
        best_t = t

print("Best threshold:", best_t, "F1:", best_f1)

# =====================================================
# FINAL EVALUATION
# =====================================================
y_pred = (y_proba >= best_t).astype(int)

print("AUC:", roc_auc_score(y_test, y_proba))
print("F1 :", f1_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall   :", recall_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# =====================================================
# FEATURE IMPORTANCE
# =====================================================
importances = model.feature_importances_
feature_scores = dict(
    sorted(
        zip(X.columns, importances),
        key=lambda x: x[1],
        reverse=True
    )
)

# =====================================================
# SAVE ARTIFACTS
# =====================================================
ROOT = "/home/cdsw"
os.makedirs(ROOT, exist_ok=True)

pickle.dump(model, open(f"{ROOT}/model.pkl", "wb"))
pickle.dump(iso, open(f"{ROOT}/calibrator.pkl", "wb"))

pickle.dump({
    "numeric_cols": numeric_cols,
    "categorical_cols": categorical_cols,
    "encoders": encoders,
    "best_threshold": float(best_t),
    "feature_importance": feature_scores,
    "label_col": label_col,
}, open(f"{ROOT}/preprocess.pkl", "wb"))

with open(f"{ROOT}/meta.json", "w") as f:
    json.dump({
        "description": "Fraud model v5 — mismatch-aware + calibrated",
        "label_source": "human+rules",
        "version": "v5"
    }, f, indent=2)

print("=== TRAINING COMPLETE ===")
spark.stop()