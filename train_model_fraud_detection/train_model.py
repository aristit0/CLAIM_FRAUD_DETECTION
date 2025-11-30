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
from sklearn.preprocessing import LabelEncoder
from category_encoders.target_encoder import TargetEncoder
import xgboost as xgb
import shap
import os, json, pickle


print("=== CONNECTING TO SPARK ===")
CONNECTION_NAME = "CDP-MSI"
conn = cmldata.get_connection(CONNECTION_NAME)
spark = conn.get_spark_session()

df_spark = spark.sql("SELECT * FROM iceberg_curated.claim_feature_set")
df = df_spark.toPandas()
print("Loaded:", df.shape)


# =====================================================
# LABEL COLUMN
# =====================================================
label_col = "rule_violation_flag"
df[label_col] = df[label_col].fillna(0).astype(int)


# =====================================================
# FEATURE SELECTION + EXTRA DERIVED FEATURES
# =====================================================

numeric_cols = [
    "patient_age",
    "total_procedure_cost",
    "total_drug_cost",
    "total_vitamin_cost",
    "total_claim_amount",
    "severity_score",
    "diagnosis_procedure_mismatch",
    "drug_mismatch_score",
    "cost_per_procedure",
    "cost_procedure_anomaly",
    "patient_claim_count",
    "patient_frequency_risk",
    "biaya_anomaly_score",
    "visit_year", "visit_month", "visit_day"
]

categorical_cols = ["visit_type", "department", "icd10_primary_code"]

# EXTRA FEATURE: counts
df["procedure_count"] = df["procedures_icd9_codes"].apply(lambda x: len(x) if isinstance(x, list) else 0)
df["drug_count"] = df["drug_names"].apply(lambda x: len(x) if isinstance(x, list) else 0)
df["vitamin_count"] = df["vitamin_names"].apply(lambda x: len(x) if isinstance(x, list) else 0)

numeric_cols += ["procedure_count", "drug_count", "vitamin_count"]


# =====================================================
# PREPROCESSING (TARGET ENCODING)
# =====================================================
print("=== TARGET ENCODING ===")

encoders = {}
for c in categorical_cols:
    te = TargetEncoder(cols=[c], smoothing=0.3)
    df[c] = df[c].fillna("__MISSING__").astype(str)
    df[c] = te.fit_transform(df[c], df[label_col])
    encoders[c] = te


# Fill numeric nulls
for c in numeric_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

X = df[numeric_cols + categorical_cols]
y = df[label_col]

print("Final feature matrix:", X.shape)


# =====================================================
# TRAIN/TEST SPLIT
# =====================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# class imbalance handling
positive = (y_train == 1).sum()
negative = (y_train == 0).sum()
scale_pos_weight = negative / positive

print("Fraud:", positive, " Normal:", negative)


# =====================================================
# XGBOOST IMPROVED PARAMS
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
    tree_method="hist"
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
best_f1 = 0
best_t = 0.5

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

auc = roc_auc_score(y_test, y_proba)
f1 = f1_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)

print("\n=== FINAL PERFORMANCE ===")
print("AUC:", auc)
print("F1 :", f1)
print("Precision:", prec)
print("Recall:", rec)
print(classification_report(y_test, y_pred))


# =====================================================
# SHAP
# =====================================================
print("=== SHAP (safe version) ===")

explainer = shap.Explainer(model, X_train[:200])
shap_values = explainer(X_train[:200])

# nilai absolute mean importance
feature_scores = dict(
    sorted(
        zip(X.columns, np.abs(shap_values.values).mean(axis=0)),
        key=lambda kv: kv[1],
        reverse=True
    )
)


# =====================================================
# SAVE ARTIFACTS
# =====================================================
ROOT = "/home/cdsw"

with open(os.path.join(ROOT, "model.pkl"), "wb") as f:
    pickle.dump(model, f)

with open(os.path.join(ROOT, "preprocess.pkl"), "wb") as f:
    pickle.dump(
        {
            "numeric_cols": numeric_cols,
            "categorical_cols": categorical_cols,
            "encoders": encoders,
            "best_threshold": best_t,
            "feature_importance": feature_scores
        },
        f
    )

print("\n=== DONE ===")
spark.stop()