#!/usr/bin/env python3
import cml.data_v1 as cmldata
from pyspark.sql.functions import col
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, classification_report
import shap
import os, json, pickle


# =====================================================
# 1. CONNECT TO SPARK AND LOAD FEATURE DATA
# =====================================================
CONNECTION_NAME = "CDP-MSI"
conn = cmldata.get_connection(CONNECTION_NAME)
spark = conn.get_spark_session()

print("=== LOAD FEATURE TABLE FROM ICEBERG ===")

df_spark = spark.sql("""
    SELECT *
    FROM iceberg_curated.claim_feature_set
""")

print("Schema:")
df_spark.printSchema()
rows = df_spark.count()
print("Total rows:", rows)


# =====================================================
# 2. SELECT FEATURE COLUMNS (UPDATED)
# =====================================================

label_col = "rule_violation_flag"

numeric_cols = [
    "patient_age",
    "total_procedure_cost",
    "total_drug_cost",
    "total_vitamin_cost",
    "total_claim_amount",

    # NEW FEATURES
    "severity_score",
    "diagnosis_procedure_mismatch",
    "drug_mismatch_score",
    "cost_per_procedure",
    "cost_procedure_anomaly",
    "patient_claim_count",
    "patient_frequency_risk",
    "biaya_anomaly_score",

    "visit_year",
    "visit_month",
    "visit_day",
]

categorical_cols = [
    "visit_type",
    "department",
    "icd10_primary_code",
]

print("NUMERIC COLS:", numeric_cols)
print("CAT COLS:", categorical_cols)

all_cols = numeric_cols + categorical_cols + [label_col]

df_spark_sel = df_spark.select(*[col(c) for c in all_cols])
df = df_spark_sel.dropna(subset=[label_col]).toPandas()

print("\n=== SAMPLE DATA ===")
print(df.head())


# =====================================================
# 3. PREPROCESSING (ENCODE CATEGORICAL)
# =====================================================

encoders = {}
for c in categorical_cols:
    le = LabelEncoder()
    df[c] = df[c].fillna("__MISSING__").astype(str)
    df[c] = le.fit_transform(df[c])
    encoders[c] = le

# Fill numeric NaN
for c in numeric_cols:
    df[c] = df[c].fillna(0.0)

X = df[numeric_cols + categorical_cols]
y = df[label_col].astype(int)

print("FEATURE MATRIX:", X.shape, " LABELS:", y.shape)


# =====================================================
# 4. TRAIN/TEST SPLIT
# =====================================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Train size:", len(X_train), "Test size:", len(X_test))


# =====================================================
# 5. HANDLE IMBALANCED FRAUD CASES
# =====================================================
positive = (y_train == 1).sum()
negative = (y_train == 0).sum()
scale_pos_weight = negative / positive if positive > 0 else 1

print("Positive (fraud):", positive)
print("Negative:", negative)
print("scale_pos_weight =", scale_pos_weight)


# =====================================================
# 6. TRAIN XGBOOST MODEL (FRAUD-OPTIMIZED)
# =====================================================
model = xgb.XGBClassifier(
    n_estimators=450,
    max_depth=6,
    learning_rate=0.045,
    subsample=0.85,
    colsample_bytree=0.85,
    objective="binary:logistic",
    eval_metric="auc",
    scale_pos_weight=scale_pos_weight,
    n_jobs=4,
    tree_method="hist"
)

print("\n=== TRAINING MODEL ===")
model.fit(
    X_train,
    y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    verbose=20
)


# =====================================================
# 7. EVALUATION
# =====================================================
y_proba = model.predict_proba(X_test)[:, 1]
y_pred = (y_proba >= 0.5).astype(int)

auc = roc_auc_score(y_test, y_proba)
f1 = f1_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)

print("\n=== MODEL PERFORMANCE ===")
print("AUC :", auc)
print("F1  :", f1)
print("Precision:", prec)
print("Recall   :", rec)

print("\n=== CLASSIFICATION REPORT ===")
print(classification_report(y_test, y_pred, zero_division=0))


# =====================================================
# 8. SHAP FEATURE IMPORTANCE (EXPLAINABILITY)
# =====================================================
print("=== COMPUTING SHAP VALUES ===")

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_train[:200])  # speed optimized

feature_scores = dict(
    sorted(
        zip(X.columns, np.abs(shap_values).mean(axis=0)),
        key=lambda kv: kv[1],
        reverse=True
    )
)

print("\n=== TOP FEATURES (SHAP) ===")
for k, v in list(feature_scores.items())[:15]:
    print(f"{k}: {v}")


# =====================================================
# 9. EXPORT ARTIFACTS FOR MODEL SERVING
# =====================================================
ROOT = "/home/cdsw"

print("\n=== EXPORT MODEL + ENCODERS ===")

# Model
with open(os.path.join(ROOT, "model.pkl"), "wb") as f:
    pickle.dump(model, f)

# Preprocessing artifacts
with open(os.path.join(ROOT, "preprocess.pkl"), "wb") as f:
    pickle.dump(
        {
            "numeric_cols": numeric_cols,
            "categorical_cols": categorical_cols,
            "encoders": encoders,
            "feature_importance": feature_scores,
            "label_col": label_col,
        },
        f
    )

# Metadata
with open(os.path.join(ROOT, "meta.json"), "w") as f:
    json.dump(
        {
            "description": "Fraud detection model v2 (ICD logic + anomaly + cost + mismatch)",
            "version": "v2",
            "algorithm": "XGBoost",
            "features_used": numeric_cols + categorical_cols,
            "performance": {
                "auc": float(auc),
                "f1": float(f1),
                "precision": float(prec),
                "recall": float(rec),
            },
        },
        f,
        indent=2
    )

print("=== TRAINING COMPLETE. ARTIFACTS SAVED TO /home/cdsw ===")

spark.stop()