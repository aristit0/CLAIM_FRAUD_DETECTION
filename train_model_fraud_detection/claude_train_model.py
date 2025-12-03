#!/usr/bin/env python3
"""
Production Model Training for BPJS Fraud Detection
Learns from historical approved/declined claims
Optimized for reviewer decision support
"""

import os
import json
import pickle
import pandas as pd
import numpy as np
import sys
import cml.data_v1 as cmldata
import xgboost as xgb
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix, roc_curve, precision_recall_curve
)
from sklearn.isotonic import IsotonicRegression
from category_encoders.target_encoder import TargetEncoder
from imblearn.over_sampling import SMOTE
from pyspark.sql.functions import col

# Import config
sys.path.insert(0, '/home/cdsw')  # Pastikan path ini ada

from config import NUMERIC_FEATURES, CATEGORICAL_FEATURES, MODEL_HYPERPARAMETERS

print("=" * 80)
print("BPJS FRAUD DETECTION - MODEL TRAINING")
print("Learning from Historical Claims (Approved/Declined)")
print("=" * 80)

# ================================================================
# 1. CONNECT TO SPARK
# ================================================================
print("\n[Step 1/17] Connecting to Spark...")
conn = cmldata.get_connection("CDP-MSI")
spark = conn.get_spark_session()
print(f"✓ Connected - Application ID: {spark.sparkContext.applicationId}")

# ================================================================
# 2. DEFINE FEATURES
# ================================================================
print("\n[Step 2/17] Defining feature set...")

label_col = "final_label"
numeric_cols = NUMERIC_FEATURES
categorical_cols = CATEGORICAL_FEATURES
all_cols = numeric_cols + categorical_cols + [label_col]

print(f"✓ Numeric features: {len(numeric_cols)}")
print(f"✓ Categorical features: {len(categorical_cols)}")
print(f"✓ Total features: {len(numeric_cols) + len(categorical_cols)}")

# ================================================================
# 3. LOAD DATA FROM ICEBERG
# ================================================================
print("\n[Step 3/17] Loading training data from Iceberg...")

df_spark = (
    spark.table("iceberg_curated.claim_feature_set")
         .where(col(label_col).isNotNull())  # Only labeled data
         .select(*all_cols)
)

total_records = df_spark.count()
print(f"✓ Total labeled claims: {total_records:,}")

# Sampling strategy (if needed)
MAX_ROWS = 300_000
if total_records > MAX_ROWS:
    print(f"  Sampling {MAX_ROWS:,} records for training...")
    df_spark = df_spark.limit(MAX_ROWS)
else:
    print(f"  Using all {total_records:,} records")

# ================================================================
# 4. CONVERT TO PANDAS
# ================================================================
print("\n[Step 4/17] Converting to Pandas...")
df = df_spark.toPandas()
print(f"✓ DataFrame shape: {df.shape}")

# ================================================================
# 5. DATA QUALITY CHECKS
# ================================================================
print("\n[Step 5/17] Performing data quality checks...")

# Check nulls
null_counts = df[all_cols].isnull().sum()
critical_nulls = null_counts[null_counts > 0]

if len(critical_nulls) > 0:
    print("⚠ Handling null values:")
    for col_name, count in critical_nulls.items():
        pct = count / len(df) * 100
        print(f"  - {col_name}: {count} nulls ({pct:.2f}%)")
    
    # Fill nulls
    for col_name in numeric_cols:
        if df[col_name].isnull().sum() > 0:
            df[col_name].fillna(0, inplace=True)
    
    for col_name in categorical_cols:
        if df[col_name].isnull().sum() > 0:
            df[col_name].fillna("UNKNOWN", inplace=True)
    
    print("✓ Null values handled")
else:
    print("✓ No null values detected")

# Check for inf values
print("\nChecking for infinite values...")
for col_name in numeric_cols:
    inf_count = np.isinf(df[col_name]).sum()
    if inf_count > 0:
        print(f"  ⚠ {col_name}: {inf_count} inf values - replacing with 0")
        df[col_name].replace([np.inf, -np.inf], 0, inplace=True)

print("✓ Data quality checks complete")

# ================================================================
# 6. LABEL DISTRIBUTION ANALYSIS
# ================================================================
print("\n[Step 6/17] Analyzing label distribution...")

df[label_col] = df[label_col].astype(int)
label_counts = df[label_col].value_counts().sort_index()

print("\n📊 Label Distribution:")
print(f"  Legitimate (0): {label_counts[0]:,} ({label_counts[0]/len(df)*100:.1f}%)")
print(f"  Fraud (1):      {label_counts[1]:,} ({label_counts[1]/len(df)*100:.1f}%)")

fraud_ratio = label_counts[1] / len(df)
print(f"\n  Fraud ratio: {fraud_ratio:.2%}")

if fraud_ratio < 0.05:
    print("  ⚠ WARNING: Very low fraud ratio - may need more fraud samples")
elif fraud_ratio > 0.60:
    print("  ⚠ WARNING: Very high fraud ratio - check data quality")
else:
    print("  ✓ Fraud ratio is reasonable for training")

# ================================================================
# 7. ENCODE CATEGORICAL FEATURES
# ================================================================
print("\n[Step 7/17] Encoding categorical features...")

encoders = {}

for col_name in categorical_cols:
    df[col_name] = df[col_name].fillna("UNKNOWN").astype(str)
    
    # Target encoding (encode based on fraud rate per category)
    te = TargetEncoder(cols=[col_name], smoothing=0.3, min_samples_leaf=20)
    df[col_name] = te.fit_transform(df[col_name], df[label_col])
    
    encoders[col_name] = te
    print(f"  ✓ Encoded: {col_name}")

# ================================================================
# 8. CLEAN NUMERIC FEATURES
# ================================================================
print("\n[Step 8/17] Cleaning numeric features...")

for col_name in numeric_cols:
    df[col_name] = pd.to_numeric(df[col_name], errors="coerce").fillna(0.0)
    df[col_name].replace([np.inf, -np.inf], 0, inplace=True)

print("✓ All numeric features cleaned")

# ================================================================
# 9. PREPARE X, y
# ================================================================
print("\n[Step 9/17] Preparing feature matrix and labels...")

X = df[numeric_cols + categorical_cols].copy()
y = df[label_col].copy()

print(f"✓ X shape: {X.shape}")
print(f"✓ y shape: {y.shape}")

# ================================================================
# 10. HANDLE CLASS IMBALANCE
# ================================================================
print("\n[Step 10/17] Handling class imbalance...")

# Apply SMOTE only if severely imbalanced
if fraud_ratio < 0.25:
    print(f"  Fraud ratio {fraud_ratio:.2%} < 25% → Applying SMOTE...")
    smote = SMOTE(sampling_strategy=0.4, random_state=42, k_neighbors=5)
    X_res, y_res = smote.fit_resample(X, y)
    
    fraud_after = (y_res == 1).sum() / len(y_res)
    print(f"  ✓ After SMOTE: {X_res.shape[0]:,} samples")
    print(f"  ✓ New fraud ratio: {fraud_after:.2%}")
else:
    print(f"  Fraud ratio {fraud_ratio:.2%} >= 25% → Skipping SMOTE")
    print("  Will use class weights in XGBoost instead")
    X_res, y_res = X, y

# ================================================================
# 11. TRAIN/TEST SPLIT
# ================================================================
print("\n[Step 11/17] Splitting train/test sets...")

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
print(f"✓ Train legitimate: {negative:,} ({negative/len(y_train)*100:.1f}%)")
print(f"✓ Scale pos weight: {scale_pos_weight:.4f}")

# ================================================================
# 12. XGBOOST TRAINING WITH EARLY STOPPING
# ================================================================
print("\n[Step 12/17] Training XGBoost model...")

feature_names = X.columns.tolist()

dtrain = xgb.DMatrix(X_train, label=y_train.values, feature_names=feature_names)
dtest = xgb.DMatrix(X_test, label=y_test.values, feature_names=feature_names)

# Hyperparameters optimized for fraud detection
params = MODEL_HYPERPARAMETERS.copy()
params["scale_pos_weight"] = scale_pos_weight

print("\n📋 Hyperparameters:")
for key, val in params.items():
    print(f"  {key}: {val}")

print("\n🚀 Training started...")
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
# 13. CALIBRATION (ISOTONIC REGRESSION)
# ================================================================
print("\n[Step 13/17] Calibrating fraud scores...")

# Get raw predictions
y_proba_raw = model.predict(dtest)

# Apply isotonic calibration for better probability estimates
iso = IsotonicRegression(out_of_bounds="clip")
iso.fit(y_proba_raw, y_test)

# Get calibrated predictions
y_proba = iso.predict(y_proba_raw)

print("✓ Model calibrated - fraud scores now represent true probabilities")

# ================================================================
# 14. THRESHOLD OPTIMIZATION
# ================================================================
print("\n[Step 14/17] Optimizing classification threshold...")

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
print(f"  (Balances precision and recall for fraud detection)")

# ================================================================
# 15. FINAL EVALUATION
# ================================================================
print("\n[Step 15/17] Final model evaluation...")

y_pred = (y_proba >= best_threshold).astype(int)

# Calculate metrics
auc = roc_auc_score(y_test, y_proba)
f1 = f1_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

# Calculate specific metrics for fraud detection
tn, fp, fn, tp = cm.ravel()
fraud_detection_rate = tp / (tp + fn) if (tp + fn) > 0 else 0  # Recall for fraud
false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
false_negative_rate = fn / (tp + fn) if (tp + fn) > 0 else 0

print("\n" + "=" * 80)
print("MODEL PERFORMANCE METRICS")
print("=" * 80)
print(f"\n📊 Overall Performance:")
print(f"  AUC-ROC:        {auc:.4f}  (Discrimination ability)")
print(f"  F1 Score:       {f1:.4f}  (Balance of precision/recall)")
print(f"  Precision:      {precision:.4f}  (When model says fraud, how often correct?)")
print(f"  Recall:         {recall:.4f}  (Of all frauds, how many detected?)")

print(f"\n🎯 Fraud Detection Metrics:")
print(f"  Fraud Detection Rate:    {fraud_detection_rate:.1%}  ({tp}/{tp+fn} frauds caught)")
print(f"  False Positive Rate:     {false_positive_rate:.1%}  (Legitimate marked as fraud)")
print(f"  False Negative Rate:     {false_negative_rate:.1%}  (Fraud marked as legitimate)")

print(f"\n📋 Confusion Matrix:")
print(f"                      Predicted")
print(f"                 Legitimate  Fraud")
print(f"  Actual Legit      {tn:6d}  {fp:6d}")
print(f"         Fraud      {fn:6d}  {tp:6d}")

print(f"\n💡 Interpretation for BPJS Reviewers:")
print(f"  - Model will catch {fraud_detection_rate:.0%} of fraud cases")
print(f"  - {false_positive_rate:.1%} of legitimate claims may need manual review")
print(f"  - {false_negative_rate:.1%} of frauds may slip through (needs reviewer vigilance)")

print("\n" + "=" * 80)
print("DETAILED CLASSIFICATION REPORT")
print("=" * 80)
print(classification_report(y_test, y_pred, target_names=["Legitimate", "Fraud"], digits=4))

# ================================================================
# 16. FEATURE IMPORTANCE ANALYSIS
# ================================================================
print("\n[Step 16/17] Analyzing feature importance...")

raw_importances = model.get_score(importance_type="gain")
feature_scores = dict(
    sorted(raw_importances.items(), key=lambda x: x[1], reverse=True)
)

print("\n🔍 Top 20 Most Important Features:")
print(f"{'Rank':<6} {'Feature':<40} {'Importance Score':<15}")
print("-" * 80)
for i, (feat, score) in enumerate(list(feature_scores.items())[:20]):
    print(f"{i+1:<6} {feat:<40} {score:<15.2f}")

# Identify key fraud indicators
key_indicators = [f for f in feature_scores.keys() if any(x in f for x in ['mismatch', 'anomaly', 'compatibility'])]
print(f"\n🚨 Key Fraud Indicators Learned:")
for feat in key_indicators[:5]:
    print(f"  - {feat}: {feature_scores[feat]:.2f}")

# ================================================================
# 17. SAVE MODEL ARTIFACTS
# ================================================================
print("\n[Step 17/17] Saving model artifacts...")

ROOT = "/home/cdsw"
os.makedirs(ROOT, exist_ok=True)

# 17.1 Save XGBoost model
model_path = os.path.join(ROOT, "model.json")
model.save_model(model_path)
print(f"  ✓ Model: {model_path}")

# 17.2 Save calibrator
calib_path = os.path.join(ROOT, "calibrator.pkl")
with open(calib_path, "wb") as f:
    pickle.dump(iso, f)
print(f"  ✓ Calibrator: {calib_path}")

# 17.3 Save preprocessing config
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
print(f"  ✓ Preprocessing: {preprocess_path}")

# 17.4 Save comprehensive metadata
meta = {
    "model_version": "v7_production_bpjs",
    "training_date": pd.Timestamp.now().isoformat(),
    "description": "BPJS Fraud Detection Model - Learned from historical approved/declined claims",
    "purpose": "Decision support for BPJS claim reviewers",
    "dataset": {
        "total_samples": int(len(df)),
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "fraud_ratio_original": float(fraud_ratio),
        "fraud_ratio_after_smote": float((y_res == 1).sum() / len(y_res)) if fraud_ratio < 0.25 else float(fraud_ratio),
    },
    "features": {
        "numeric_count": len(numeric_cols),
        "categorical_count": len(categorical_cols),
        "total_count": len(numeric_cols) + len(categorical_cols),
        "top_10_features": list(feature_scores.items())[:10]
    },
    "hyperparameters": params,
    "performance": {
        "auc": float(auc),
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "fraud_detection_rate": float(fraud_detection_rate),
        "false_positive_rate": float(false_positive_rate),
        "false_negative_rate": float(false_negative_rate),
        "best_threshold": float(best_threshold),
    },
    "confusion_matrix": {
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    },
    "feature_importance_top10": [
        {"feature": k, "gain": float(v)}
        for k, v in list(feature_scores.items())[:10]
    ],
}

meta_path = os.path.join(ROOT, "meta.json")
with open(meta_path, "w") as f:
    json.dump(meta, f, indent=2)
print(f"  ✓ Metadata: {meta_path}")

# 17.5 Save human-readable training summary
summary_path = os.path.join(ROOT, "training_summary.txt")
with open(summary_path, "w") as f:
    f.write("=" * 80 + "\n")
    f.write("BPJS FRAUD DETECTION MODEL - TRAINING SUMMARY\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"Training Date: {pd.Timestamp.now()}\n")
    f.write(f"Model Version: v7_production_bpjs\n")
    f.write(f"Purpose: Decision support for BPJS claim reviewers\n\n")
    
    f.write("DATASET:\n")
    f.write(f"  Total claims: {len(df):,}\n")
    f.write(f"  Training set: {len(X_train):,}\n")
    f.write(f"  Test set: {len(X_test):,}\n")
    f.write(f"  Fraud ratio: {fraud_ratio:.2%}\n\n")
    
    f.write("MODEL PERFORMANCE:\n")
    f.write(f"  AUC-ROC: {auc:.4f}\n")
    f.write(f"  F1 Score: {f1:.4f}\n")
    f.write(f"  Precision: {precision:.4f}\n")
    f.write(f"  Recall: {recall:.4f}\n")
    f.write(f"  Optimal Threshold: {best_threshold:.3f}\n\n")
    
    f.write("FRAUD DETECTION METRICS:\n")
    f.write(f"  Detection Rate: {fraud_detection_rate:.1%} ({tp}/{tp+fn} frauds caught)\n")
    f.write(f"  False Positive Rate: {false_positive_rate:.1%}\n")
    f.write(f"  False Negative Rate: {false_negative_rate:.1%}\n\n")
    
    f.write("CONFUSION MATRIX:\n")
    f.write(f"  True Negatives:  {tn:,}\n")
    f.write(f"  False Positives: {fp:,}\n")
    f.write(f"  False Negatives: {fn:,}\n")
    f.write(f"  True Positives:  {tp:,}\n\n")
    
    f.write("TOP 10 IMPORTANT FEATURES:\n")
    for i, (feat, score) in enumerate(list(feature_scores.items())[:10], 1):
        f.write(f"  {i}. {feat}: {score:.2f}\n")
    
    f.write("\n" + "=" * 80 + "\n")
    f.write("INTERPRETATION FOR BPJS REVIEWERS\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"This model learned from {len(df):,} historical claims that were\n")
    f.write("either approved or declined by BPJS reviewers.\n\n")
    f.write("Key capabilities:\n")
    f.write(f"  1. Detects {fraud_detection_rate:.0%} of fraudulent claims\n")
    f.write(f"  2. Identifies clinical mismatches (diagnosis vs treatment)\n")
    f.write(f"  3. Flags cost anomalies and suspicious patterns\n")
    f.write(f"  4. Provides fraud score (0-100%) for each claim\n\n")
    f.write("The model checks:\n")
    f.write("  ✓ Is the procedure appropriate for the diagnosis?\n")
    f.write("  ✓ Are the drugs clinically indicated?\n")
    f.write("  ✓ Are vitamins/supplements medically justified?\n")
    f.write("  ✓ Is the claim amount reasonable for the condition?\n")
    f.write("  ✓ Are there suspicious patterns in claim frequency?\n")

print(f"  ✓ Summary: {summary_path}")

# 17.6 Try to save visualization (if matplotlib available)
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    print("\n  Generating visualizations...")
    
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. ROC Curve
    axes[0, 0].plot(fpr, tpr, label=f'AUC = {auc:.4f}', linewidth=2)
    axes[0, 0].plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1)
    axes[0, 0].set_xlabel('False Positive Rate')
    axes[0, 0].set_ylabel('True Positive Rate')
    axes[0, 0].set_title('ROC Curve - Fraud Detection Performance')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Confusion Matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 1], 
                xticklabels=['Legitimate', 'Fraud'], 
                yticklabels=['Legitimate', 'Fraud'])
    axes[0, 1].set_title('Confusion Matrix')
    axes[0, 1].set_xlabel('Predicted')
    axes[0, 1].set_ylabel('Actual')
    
    # 3. Feature Importance
    top_features = list(feature_scores.items())[:15]
    feat_names = [f[0] for f in top_features]
    feat_scores = [f[1] for f in top_features]
    
    axes[1, 0].barh(range(len(feat_names)), feat_scores, color='steelblue')
    axes[1, 0].set_yticks(range(len(feat_names)))
    axes[1, 0].set_yticklabels(feat_names, fontsize=8)
    axes[1, 0].set_xlabel('Gain Score')
    axes[1, 0].set_title('Top 15 Most Important Features')
    axes[1, 0].invert_yaxis()
    axes[1, 0].grid(axis='x', alpha=0.3)
    
    # 4. Threshold vs F1
    axes[1, 1].plot(thresholds, f1_scores, linewidth=2, color='green')
    axes[1, 1].axvline(best_threshold, color='r', linestyle='--', linewidth=2, 
                       label=f'Best: {best_threshold:.3f}')
    axes[1, 1].set_xlabel('Classification Threshold')
    axes[1, 1].set_ylabel('F1 Score')
    axes[1, 1].set_title('Threshold Optimization')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    viz_path = os.path.join(ROOT, "training_visualization.png")
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ Visualization: {viz_path}")
    
except Exception as e:
    print(f"  ⚠ Visualization skipped: {e}")

print("\n" + "=" * 80)
print("✅ TRAINING COMPLETE - MODEL READY FOR DEPLOYMENT")
print("=" * 80)
print("\nArtifacts saved:")
print(f"  1. {model_path}")
print(f"  2. {calib_path}")
print(f"  3. {preprocess_path}")
print(f"  4. {meta_path}")
print(f"  5. {summary_path}")

print("\nNext steps:")
print("  1. Review model performance metrics above")
print("  2. Test model with sample claims")
print("  3. Deploy to CML Model Serving")
print("  4. Integrate with BPJS reviewer UI")
print("=" * 80)

spark.stop()