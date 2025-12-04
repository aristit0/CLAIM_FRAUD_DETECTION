#!/usr/bin/env python3

import os
import json
import pickle
import pandas as pd
import numpy as np
import cml.data_v1 as cmldata
import xgboost as xgb
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix, roc_curve
)
from sklearn.isotonic import IsotonicRegression
from category_encoders.target_encoder import TargetEncoder
from imblearn.over_sampling import SMOTE
from pyspark.sql.functions import col
import matplotlib.pyplot as plt
import seaborn as sns

print("=" * 80)
print("FRAUD DETECTION MODEL TRAINING PIPELINE")
print("=" * 80)

# ================================================================
# 0. CONNECT TO SPARK
# ================================================================
print("\n[Step 1/15] Connecting to Spark...")
conn = cmldata.get_connection("CDP-MSI")
spark = conn.get_spark_session()
print(f"✓ Spark Application ID: {spark.sparkContext.applicationId}")

# ================================================================
# 1. FEATURE DEFINITIONS (MUST MATCH ETL OUTPUT!)
# ================================================================
print("\n[Step 2/15] Defining features...")

label_col = "final_label"

# Numeric features (EXACT match dengan ETL output)
numeric_cols = [
    "patient_age",
    "total_procedure_cost",
    "total_drug_cost",
    "total_vitamin_cost",
    "total_claim_amount",
    "biaya_anomaly_score",
    "patient_frequency_risk",
    "visit_year",
    "visit_month",
    "visit_day",
    # Clinical Compatibility Scores
    "diagnosis_procedure_score",
    "diagnosis_drug_score",
    "diagnosis_vitamin_score",
    # Explicit Mismatch Flags
    "procedure_mismatch_flag",
    "drug_mismatch_flag",
    "vitamin_mismatch_flag",
    "mismatch_count",
]

# Categorical features
categorical_cols = [
    "visit_type",
    "department",
    "icd10_primary_code",
]

all_cols = numeric_cols + categorical_cols + [label_col]

print(f"✓ Numeric features: {len(numeric_cols)}")
print(f"✓ Categorical features: {len(categorical_cols)}")
print(f"✓ Total features: {len(numeric_cols) + len(categorical_cols)}")

# ================================================================
# 2. LOAD DATA FROM ICEBERG
# ================================================================
print("\n[Step 3/15] Loading data from Iceberg curated table...")

df_spark = (
    spark.table("iceberg_curated.claim_feature_set")
         .where(col(label_col).isNotNull())
         .select(*all_cols)
)

total_records = df_spark.count()
print(f"✓ Total records in feature set: {total_records:,}")

# Sample if too large (for faster iteration)
MAX_ROWS = 300_000
if total_records > MAX_ROWS:
    print(f"  Sampling {MAX_ROWS:,} records for training...")
    df_spark = df_spark.limit(MAX_ROWS)

print("\n[Step 4/15] Converting to Pandas...")
df = df_spark.toPandas()
print(f"✓ Pandas DataFrame shape: {df.shape}")

# ================================================================
# 3. DATA QUALITY CHECKS
# ================================================================
print("\n[Step 5/15] Performing data quality checks...")

# Check for nulls in critical columns
null_counts = df[all_cols].isnull().sum()
critical_nulls = null_counts[null_counts > 0]

if len(critical_nulls) > 0:
    print("⚠ Warning: Null values detected:")
    for col_name, count in critical_nulls.items():
        print(f"  - {col_name}: {count} nulls ({count/len(df)*100:.2f}%)")
    
    # Fill nulls for numeric
    for col_name in numeric_cols:
        if df[col_name].isnull().sum() > 0:
            df[col_name].fillna(0, inplace=True)
    
    # Fill nulls for categorical
    for col_name in categorical_cols:
        if df[col_name].isnull().sum() > 0:
            df[col_name].fillna("UNKNOWN", inplace=True)
    
    print("✓ Null values handled")
else:
    print("✓ No null values detected")

# ================================================================
# 4. LABEL DISTRIBUTION
# ================================================================
print("\n[Step 6/15] Analyzing label distribution...")

df[label_col] = df[label_col].astype(int)
label_counts = df[label_col].value_counts().sort_index()

print("Label distribution:")
for label, count in label_counts.items():
    label_name = "Non-Fraud" if label == 0 else "Fraud"
    print(f"  {label_name} ({label}): {count:,} ({count/len(df)*100:.1f}%)")

fraud_ratio = label_counts[1] / len(df) if 1 in label_counts else 0
print(f"\nFraud ratio: {fraud_ratio:.2%}")

if fraud_ratio < 0.05:
    print("⚠ WARNING: Very low fraud ratio - model may struggle to learn fraud patterns")
elif fraud_ratio > 0.60:
    print("⚠ WARNING: Very high fraud ratio - may not reflect real-world distribution")

# ================================================================
# 5. ENCODING CATEGORICAL FEATURES
# ================================================================
print("\n[Step 7/15] Encoding categorical features (Target Encoding)...")

encoders = {}

for col_name in categorical_cols:
    # Ensure string type
    df[col_name] = df[col_name].fillna("UNKNOWN").astype(str)
    
    # Target encoding with smoothing
    te = TargetEncoder(cols=[col_name], smoothing=0.3, min_samples_leaf=20)
    df[col_name] = te.fit_transform(df[col_name], df[label_col])
    
    encoders[col_name] = te
    print(f"  ✓ Encoded: {col_name}")

# ================================================================
# 6. CLEAN NUMERIC FEATURES
# ================================================================
print("\n[Step 8/15] Cleaning numeric features...")

for col_name in numeric_cols:
    # Convert to numeric, coerce errors
    df[col_name] = pd.to_numeric(df[col_name], errors="coerce").fillna(0.0)
    
    # Check for inf values
    inf_count = np.isinf(df[col_name]).sum()
    if inf_count > 0:
        print(f"  ⚠ {col_name}: {inf_count} inf values replaced with 0")
        df[col_name].replace([np.inf, -np.inf], 0, inplace=True)

print("✓ Numeric features cleaned")

# ================================================================
# 7. PREPARE X, y
# ================================================================
print("\n[Step 9/15] Preparing features and labels...")

X = df[numeric_cols + categorical_cols].copy()
y = df[label_col].copy()

print(f"✓ Feature matrix shape: {X.shape}")
print(f"✓ Label vector shape: {y.shape}")

# ================================================================
# 8. HANDLE CLASS IMBALANCE (SMOTE - CONDITIONAL)
# ================================================================
print("\n[Step 10/15] Handling class imbalance...")

# Only apply SMOTE if imbalance is severe
if fraud_ratio < 0.25:
    print(f"  Fraud ratio {fraud_ratio:.2%} < 25%, applying SMOTE...")
    smote = SMOTE(sampling_strategy=0.4, random_state=42, k_neighbors=5)
    X_res, y_res = smote.fit_resample(X, y)
    
    fraud_after = (y_res == 1).sum() / len(y_res)
    print(f"  ✓ After SMOTE: {X_res.shape[0]:,} samples, fraud ratio: {fraud_after:.2%}")
else:
    print(f"  Fraud ratio {fraud_ratio:.2%} >= 25%, skipping SMOTE")
    X_res, y_res = X, y

# ================================================================
# 9. TRAIN/TEST SPLIT
# ================================================================
print("\n[Step 11/15] Splitting train/test sets...")

X_train, X_test, y_train, y_test = train_test_split(
    X_res, y_res,
    test_size=0.2,
    random_state=42,
    stratify=y_res
)

positive = int((y_train == 1).sum())
negative = int((y_train == 0).sum())
scale_pos_weight = negative / positive if positive > 0 else 1.0

print(f"✓ Train set: {X_train.shape[0]:,} samples")
print(f"✓ Test set: {X_test.shape[0]:,} samples")
print(f"✓ Train fraud: {positive:,} ({positive/len(y_train)*100:.1f}%)")
print(f"✓ Train non-fraud: {negative:,} ({negative/len(y_train)*100:.1f}%)")
print(f"✓ Scale pos weight: {scale_pos_weight:.4f}")

# ================================================================
# 10. XGBOOST TRAINING
# ================================================================
print("\n[Step 12/15] Training XGBoost model...")

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

# Hyperparameters (tuned for fraud detection)
params = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "eta": 0.03,                    # Learning rate
    "max_depth": 7,                 # Tree depth
    "subsample": 0.85,              # Row sampling
    "colsample_bytree": 0.85,       # Column sampling
    "min_child_weight": 3,          # Min samples in leaf
    "gamma": 0.3,                   # Min loss reduction
    "reg_alpha": 0.1,               # L1 regularization
    "reg_lambda": 1.0,              # L2 regularization
    "tree_method": "hist",          # Fast histogram method
    "scale_pos_weight": scale_pos_weight,
    "seed": 42,
}

print("Hyperparameters:")
for key, val in params.items():
    print(f"  {key}: {val}")

evals = [(dtrain, "train"), (dtest, "valid")]

model = xgb.train(
    params=params,
    dtrain=dtrain,
    num_boost_round=1000,
    evals=evals,
    early_stopping_rounds=50,
    verbose_eval=25,
)

print(f"\n✓ Training complete!")
print(f"  Best iteration: {model.best_iteration}")
print(f"  Best AUC: {model.best_score:.4f}")

# ================================================================
# 11. CALIBRATION (ISOTONIC REGRESSION)
# ================================================================
print("\n[Step 13/15] Calibrating model scores...")

# Raw predictions
y_proba_raw = model.predict(dtest)

# Isotonic calibration
iso = IsotonicRegression(out_of_bounds="clip")
iso.fit(y_proba_raw, y_test)

# Calibrated predictions
y_proba = iso.predict(y_proba_raw)

print("✓ Model calibrated with Isotonic Regression")

# ================================================================
# 12. THRESHOLD OPTIMIZATION
# ================================================================
print("\n[Step 14/15] Optimizing classification threshold...")

thresholds = np.arange(0.1, 0.9, 0.01)
best_threshold = 0.5
best_f1 = 0.0

f1_scores = []
for t in thresholds:
    pred = (y_proba >= t).astype(int)
    f1 = f1_score(y_test, pred)
    f1_scores.append(f1)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = float(t)

print(f"✓ Optimal threshold: {best_threshold:.3f}")
print(f"✓ Best F1 score: {best_f1:.4f}")

# ================================================================
# 13. FINAL EVALUATION
# ================================================================
print("\n[Step 15/15] Final model evaluation...")

y_pred = (y_proba >= best_threshold).astype(int)

# Metrics
auc = roc_auc_score(y_test, y_proba)
f1 = f1_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("\n" + "=" * 80)
print("MODEL PERFORMANCE METRICS")
print("=" * 80)
print(f"AUC-ROC       : {auc:.4f}")
print(f"F1 Score      : {f1:.4f}")
print(f"Precision     : {precision:.4f}")
print(f"Recall        : {recall:.4f}")
print("\nConfusion Matrix:")
print("                 Predicted")
print("                 Non-Fraud  Fraud")
print(f"Actual Non-Fraud   {cm[0][0]:6d}  {cm[0][1]:6d}")
print(f"       Fraud       {cm[1][0]:6d}  {cm[1][1]:6d}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Non-Fraud", "Fraud"]))

# Calculate fraud detection rate
fraud_detected = cm[1][1]
total_fraud = cm[1][0] + cm[1][1]
detection_rate = fraud_detected / total_fraud if total_fraud > 0 else 0

print(f"\nFraud Detection Rate: {detection_rate:.1%} ({fraud_detected}/{total_fraud})")

# ================================================================
# 14. FEATURE IMPORTANCE
# ================================================================
print("\n" + "=" * 80)
print("FEATURE IMPORTANCE (Top 20)")
print("=" * 80)

raw_importances = model.get_score(importance_type="gain")
feature_scores = dict(
    sorted(raw_importances.items(), key=lambda x: x[1], reverse=True)
)

print(f"\n{'Rank':<6} {'Feature':<35} {'Gain Score':<12}")
print("-" * 80)
for i, (feat, score) in enumerate(list(feature_scores.items())[:20]):
    print(f"{i+1:<6} {feat:<35} {score:<12.4f}")

# ================================================================
# 15. SAVE ARTIFACTS
# ================================================================
print("\n" + "=" * 80)
print("SAVING MODEL ARTIFACTS")
print("=" * 80)

ROOT = "/home/cdsw"
os.makedirs(ROOT, exist_ok=True)

# 15.1 Save XGBoost model
model_path = os.path.join(ROOT, "model.json")
model.save_model(model_path)
print(f"✓ Model saved: {model_path}")

# 15.2 Save calibrator
calib_path = os.path.join(ROOT, "calibrator.pkl")
with open(calib_path, "wb") as f:
    pickle.dump(iso, f)
print(f"✓ Calibrator saved: {calib_path}")

# 15.3 Save preprocessing metadata
preprocess = {
    "numeric_cols": numeric_cols,
    "categorical_cols": categorical_cols,
    "encoders": encoders,
    "best_threshold": float(best_threshold),
    "feature_importance": feature_scores,
    "label_col": label_col,
}

preprocess_path = os.path.join(ROOT, "preprocess.pkl")
with open(preprocess_path, "wb") as f:
    pickle.dump(preprocess, f)
print(f"✓ Preprocessing config saved: {preprocess_path}")

# 15.4 Save metadata
meta = {
    "model_version": "v7_production",
    "training_date": pd.Timestamp.now().isoformat(),
    "description": "Production fraud detection model - aligned with ETL v7",
    "dataset": {
        "total_samples": int(len(df)),
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "fraud_ratio": float(fraud_ratio),
        "fraud_after_smote": float((y_res == 1).sum() / len(y_res)) if fraud_ratio < 0.25 else float(fraud_ratio),
    },
    "features": {
        "numeric_count": len(numeric_cols),
        "categorical_count": len(categorical_cols),
        "total_count": len(numeric_cols) + len(categorical_cols),
    },
    "hyperparameters": params,
    "performance": {
        "auc": float(auc),
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "best_threshold": float(best_threshold),
        "fraud_detection_rate": float(detection_rate),
    },
    "feature_importance_top10": [
        {"feature": k, "gain": float(v)}
        for k, v in list(feature_scores.items())[:10]
    ],
}

meta_path = os.path.join(ROOT, "meta.json")
with open(meta_path, "w") as f:
    json.dump(meta, f, indent=2)
print(f"✓ Metadata saved: {meta_path}")

# 15.5 Save training summary
summary_path = os.path.join(ROOT, "training_summary.txt")
with open(summary_path, "w") as f:
    f.write("=" * 80 + "\n")
    f.write("FRAUD DETECTION MODEL - TRAINING SUMMARY\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"Training Date: {pd.Timestamp.now()}\n")
    f.write(f"Model Version: v7_production\n\n")
    f.write(f"Dataset:\n")
    f.write(f"  Total samples: {len(df):,}\n")
    f.write(f"  Train samples: {len(X_train):,}\n")
    f.write(f"  Test samples: {len(X_test):,}\n")
    f.write(f"  Fraud ratio: {fraud_ratio:.2%}\n\n")
    f.write(f"Model Performance:\n")
    f.write(f"  AUC-ROC: {auc:.4f}\n")
    f.write(f"  F1 Score: {f1:.4f}\n")
    f.write(f"  Precision: {precision:.4f}\n")
    f.write(f"  Recall: {recall:.4f}\n")
    f.write(f"  Optimal Threshold: {best_threshold:.3f}\n")
    f.write(f"  Fraud Detection Rate: {detection_rate:.1%}\n\n")
    f.write(f"Confusion Matrix:\n")
    f.write(f"  TN: {cm[0][0]:,}  FP: {cm[0][1]:,}\n")
    f.write(f"  FN: {cm[1][0]:,}  TP: {cm[1][1]:,}\n\n")
    f.write(f"Top 10 Important Features:\n")
    for i, (feat, score) in enumerate(list(feature_scores.items())[:10]):
        f.write(f"  {i+1}. {feat}: {score:.4f}\n")

print(f"✓ Training summary saved: {summary_path}")

# ================================================================
# 16. VISUALIZATION (OPTIONAL - IF MATPLOTLIB AVAILABLE)
# ================================================================
try:
    print("\nGenerating visualizations...")
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. ROC Curve
    axes[0, 0].plot(fpr, tpr, label=f'AUC = {auc:.4f}')
    axes[0, 0].plot([0, 1], [0, 1], 'k--', label='Random')
    axes[0, 0].set_xlabel('False Positive Rate')
    axes[0, 0].set_ylabel('True Positive Rate')
    axes[0, 0].set_title('ROC Curve')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Confusion Matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 1])
    axes[0, 1].set_title('Confusion Matrix')
    axes[0, 1].set_xlabel('Predicted')
    axes[0, 1].set_ylabel('Actual')
    
    # 3. Feature Importance
    top_features = list(feature_scores.items())[:15]
    feat_names = [f[0] for f in top_features]
    feat_scores = [f[1] for f in top_features]
    
    axes[1, 0].barh(range(len(feat_names)), feat_scores)
    axes[1, 0].set_yticks(range(len(feat_names)))
    axes[1, 0].set_yticklabels(feat_names, fontsize=8)
    axes[1, 0].set_xlabel('Gain Score')
    axes[1, 0].set_title('Top 15 Feature Importance')
    axes[1, 0].invert_yaxis()
    
    # 4. Threshold vs F1
    axes[1, 1].plot(thresholds, f1_scores)
    axes[1, 1].axvline(best_threshold, color='r', linestyle='--', label=f'Best: {best_threshold:.3f}')
    axes[1, 1].set_xlabel('Threshold')
    axes[1, 1].set_ylabel('F1 Score')
    axes[1, 1].set_title('Threshold Optimization')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    viz_path = os.path.join(ROOT, "training_visualization.png")
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    print(f"✓ Visualization saved: {viz_path}")
    
except Exception as e:
    print(f"⚠ Visualization skipped: {e}")

# ================================================================
# 17. CLEANUP
# ================================================================
print("\n" + "=" * 80)
print("TRAINING COMPLETE")
print("=" * 80)
print("\nArtifacts saved:")
print(f"  1. {model_path}")
print(f"  2. {calib_path}")
print(f"  3. {preprocess_path}")
print(f"  4. {meta_path}")
print(f"  5. {summary_path}")

print("\nModel is ready for deployment!")
print("=" * 80)

spark.stop()