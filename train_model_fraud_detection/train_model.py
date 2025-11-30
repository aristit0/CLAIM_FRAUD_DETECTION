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
import xgboost as xgb
import os, json, pickle

# =====================================================
# 1. CONNECT TO SPARK & LOAD FEATURE TABLE
# =====================================================
print("=== CONNECTING TO SPARK ===")
CONNECTION_NAME = "CDP-MSI"
conn = cmldata.get_connection(CONNECTION_NAME)
spark = conn.get_spark_session()

df_spark = spark.sql("SELECT * FROM iceberg_curated.claim_feature_set")
df = df_spark.toPandas()
print("Loaded feature_set:", df.shape)

# =====================================================
# 2. LABEL COLUMN
# =====================================================
label_col = "rule_violation_flag"
df[label_col] = df[label_col].fillna(0).astype(int)

# =====================================================
# 3. FEATURE SELECTION
# =====================================================

# PURE NUMERIC FEATURES
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

    "visit_year",
    "visit_month",
    "visit_day",

    # Clinical compatibility scores (BARU)
    "diagnosis_procedure_score",
    "diagnosis_drug_score",
    "diagnosis_vitamin_score",
    "treatment_consistency_score",

    # Old rule flags (tetap ikut, bisa penting)
    "diagnosis_procedure_mismatch",
    "drug_mismatch_score",
    "cost_procedure_anomaly",
    "patient_frequency_risk",
]

# CATEGORICAL FEATURES
categorical_cols = [
    "visit_type",
    "department",
    "icd10_primary_code",
]

print("NUMERIC COLS:", numeric_cols)
print("CAT COLS    :", categorical_cols)

# =====================================================
# 4. PREPROCESSING (TARGET ENCODING UNTUK CATEGORICAL)
# =====================================================
print("=== TARGET ENCODING CATEGORICAL FEATURES ===")

encoders = {}
for c in categorical_cols:
    te = TargetEncoder(cols=[c], smoothing=0.3)
    df[c] = df[c].fillna("__MISSING__").astype(str)
    df[c] = te.fit_transform(df[c], df[label_col])
    encoders[c] = te

# Isi numeric NaN
for c in numeric_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

X = df[numeric_cols + categorical_cols]
y = df[label_col]

print("Final feature matrix:", X.shape, "Labels:", y.shape)

# =====================================================
# 5. TRAIN / TEST SPLIT
# =====================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

positive = (y_train == 1).sum()
negative = (y_train == 0).sum()
scale_pos_weight = negative / positive if positive > 0 else 1.0

print("Fraud (1):", positive, " Normal (0):", negative)
print("scale_pos_weight:", scale_pos_weight)

# =====================================================
# 6. XGBOOST MODEL (FRAUD-ORIENTED)
# =====================================================
model = xgb.XGBClassifier(
    n_estimators=600,
    max_depth=7,
    learning_rate=0.03,
    subsample=0.9,
    colsample_bytree=0.9,
    min_child_weight=3,
    gamma=0.5,
    reg_alpha=0.2,
    reg_lambda=1.5,
    objective="binary:logistic",
    eval_metric="auc",
    scale_pos_weight=scale_pos_weight,
    tree_method="hist",
)

print("=== TRAINING MODEL ===")
model.fit(
    X_train,
    y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    verbose=50
)

# =====================================================
# 7. THRESHOLD OPTIMIZATION (F1 MAX)
# =====================================================
print("=== THRESHOLD OPTIMIZATION ===")
y_proba = model.predict_proba(X_test)[:, 1]

thresholds = np.arange(0.1, 0.9, 0.02)
best_f1 = 0.0
best_t = 0.5

for t in thresholds:
    pred = (y_proba >= t).astype(int)
    f1 = f1_score(y_test, pred)
    if f1 > best_f1:
        best_f1 = f1
        best_t = t

print("Best threshold:", best_t, "F1:", best_f1)

# =====================================================
# 8. FINAL EVALUATION
# =====================================================
y_pred = (y_proba >= best_t).astype(int)

auc  = roc_auc_score(y_test, y_proba)
f1   = f1_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec  = recall_score(y_test, y_pred, zero_division=0)

print("\n=== FINAL PERFORMANCE @ THRESHOLD {:.2f} ===".format(best_t))
print("AUC       :", auc)
print("F1        :", f1)
print("Precision :", prec)
print("Recall    :", rec)
print("\n=== CLASSIFICATION REPORT ===")
print(classification_report(y_test, y_pred, zero_division=0))

# =====================================================
# 9. FEATURE IMPORTANCE (SIMPLE, TANPA SHAP)
# =====================================================
print("=== FEATURE IMPORTANCE (XGBoost GAIN) ===")
try:
    importances = model.feature_importances_
    feature_scores = dict(
        sorted(
            zip(X.columns, importances),
            key=lambda kv: kv[1],
            reverse=True
        )
    )
    for k, v in list(feature_scores.items())[:20]:
        print(f"{k}: {v}")
except Exception as e:
    print("Cannot compute feature_importances_:", str(e))
    feature_scores = {}

# =====================================================
# 10. SAVE ARTIFACTS (MODEL + PREPROCESS + META)
# =====================================================
ROOT = "/home/cdsw"
os.makedirs(ROOT, exist_ok=True)

print("\n=== SAVING MODEL & PREPROCESS ARTIFACTS TO", ROOT, "===")

# 1. Model
with open(os.path.join(ROOT, "model.pkl"), "wb") as f:
    pickle.dump(model, f)

# 2. Preprocess
with open(os.path.join(ROOT, "preprocess.pkl"), "wb") as f:
    pickle.dump(
        {
            "numeric_cols": numeric_cols,
            "categorical_cols": categorical_cols,
            "encoders": encoders,
            "best_threshold": float(best_t),
            "feature_importance": feature_scores,
            "label_col": label_col,
        },
        f
    )

# 3. Metadata
with open(os.path.join(ROOT, "meta.json"), "w") as f:
    json.dump(
        {
            "description": "Fraud detection model v3 (clinical compatibility + cost/frequency)",
            "version": "v3",
            "algorithm": "XGBoost",
            "features_used": numeric_cols + categorical_cols,
            "performance": {
                "auc": float(auc),
                "f1": float(f1),
                "precision": float(prec),
                "recall": float(rec),
                "best_threshold": float(best_t),
            },
        },
        f,
        indent=2
    )

print("=== TRAINING COMPLETE. ARTIFACTS SAVED. ===")
spark.stop()