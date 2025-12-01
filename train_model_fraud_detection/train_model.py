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

print("=== CONNECTING TO SPARK ===")
conn = cmldata.get_connection("CDP-MSI")
spark = conn.get_spark_session()

df_spark = spark.sql("""
    SELECT *
    FROM iceberg_curated.claim_feature_set
    WHERE final_label IS NOT NULL     -- ignore pending claims
""")

df = df_spark.toPandas()
print("Loaded dataset:", df.shape)

# =====================================================
# LABEL
# =====================================================
label_col = "final_label"
df[label_col] = df[label_col].astype(int)

# =====================================================
# FEATURES
# =====================================================
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
    "diagnosis_procedure_score",
    "diagnosis_drug_score",
    "diagnosis_vitamin_score",
    "treatment_consistency_score",
    "diagnosis_procedure_mismatch",
    "drug_mismatch_score",
    "cost_procedure_anomaly",
    "patient_frequency_risk",
]

categorical_cols = [
    "visit_type",
    "department",
    "icd10_primary_code",
]

print("Numeric:", numeric_cols)
print("Cat:", categorical_cols)

# =====================================================
# TARGET ENCODING
# =====================================================
encoders = {}
for c in categorical_cols:
    df[c] = df[c].fillna("__MISSING__").astype(str)
    te = TargetEncoder(cols=[c], smoothing=0.3)
    df[c] = te.fit_transform(df[c], df[label_col])
    encoders[c] = te

# Fill numeric NaN
for c in numeric_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

X = df[numeric_cols + categorical_cols]
y = df[label_col]

# =====================================================
# SPLIT
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

# =====================================================
# XGBOOST MODEL
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
    tree_method="hist",
    scale_pos_weight=scale_pos_weight,
)

print("=== TRAINING MODEL ===")
model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    verbose=50
)

# =====================================================
# THRESHOLD OPTIMIZATION
# =====================================================
y_proba = model.predict_proba(X_test)[:, 1]
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
# FINAL EVAL
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

with open(f"{ROOT}/model.pkl", "wb") as f:
    pickle.dump(model, f)

with open(f"{ROOT}/preprocess.pkl", "wb") as f:
    pickle.dump({
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "encoders": encoders,
        "best_threshold": float(best_t),
        "feature_importance": feature_scores,
        "label_col": label_col,
    }, f)

with open(f"{ROOT}/meta.json", "w") as f:
    json.dump({
        "description": "Fraud model v4 — using human labels",
        "label_source": "human+rules",
        "version": "v4"
    }, f, indent=2)

print("=== TRAINING COMPLETE ===")
spark.stop()