import cml.data_v1 as cmldata
from pyspark.sql.functions import col
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from sklearn.metrics import roc_auc_score, f1_score, classification_report
import os, json, pickle


# =====================================================
# 1. Connect to Spark
# =====================================================
CONNECTION_NAME = "CDP-MSI"
conn = cmldata.get_connection(CONNECTION_NAME)
spark = conn.get_spark_session()

print("=== LOAD FEATURE TABLE ===")

df_spark = spark.sql("""
    SELECT *
    FROM iceberg_curated.claim_feature_set
""")

df_spark.printSchema()
print("Total rows:", df_spark.count())


# =====================================================
# 2. Select feature columns
# =====================================================
label_col = "rule_violation_flag"

numeric_cols = [
    "patient_age",
    "total_procedure_cost",
    "total_drug_cost",
    "total_vitamin_cost",
    "total_claim_amount",
    "tindakan_validity_score",
    "obat_validity_score",
    "vitamin_relevance_score",
    "biaya_anomaly_score",
    "visit_year",
    "visit_month",
    "visit_day",
]

cat_cols = [
    "visit_type",
    "department",
    "icd10_primary_code",
]

all_cols = numeric_cols + cat_cols + [label_col]

df_spark_sel = df_spark.select(*[col(c) for c in all_cols])
df = df_spark_sel.dropna(subset=[label_col]).toPandas()

print(df.head())
print(df[label_col].value_counts())


# =====================================================
# 3. Encode Categorical + Fill NaN
# =====================================================
encoders = {}

for c in cat_cols:
    le = LabelEncoder()
    df[c] = df[c].fillna("__MISSING__").astype(str)
    df[c] = le.fit_transform(df[c])
    encoders[c] = le

# Fill numeric NaN
for c in numeric_cols:
    df[c] = df[c].fillna(0.0)

X = df[numeric_cols + cat_cols]
y = df[label_col].astype(int)

feature_names = list(X.columns)

print("FEATURE MATRIX:", X.shape, " LABELS:", y.shape)


# =====================================================
# 4. Train-test split
# =====================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Train:", len(X_train), "Test:", len(X_test))


# =====================================================
# 5. Train XGBoost Model
# =====================================================
pos_ratio = (y_train == 1).sum() / len(y_train)
neg_ratio = (y_train == 0).sum() / len(y_train)
scale_pos_weight = neg_ratio / pos_ratio if pos_ratio > 0 else 1.0

print("scale_pos_weight =", scale_pos_weight)

model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="auc",
    scale_pos_weight=scale_pos_weight,
    n_jobs=4,
    tree_method="hist"
)

model.fit(
    X_train,
    y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    verbose=20
)


# =====================================================
# 6. Evaluate
# =====================================================
y_pred_proba = model.predict_proba(X_test)[:, 1]
y_pred = (y_pred_proba >= 0.5).astype(int)

auc = roc_auc_score(y_test, y_pred_proba)
f1 = f1_score(y_test, y_pred)

print("AUC:", auc)
print("F1 :", f1)
print(classification_report(y_test, y_pred))


# =====================================================
# 7. Export model to ROOT DIRECTORY
# =====================================================
print("\n=== EXPORT ARTIFACTS TO ROOT DIRECTORY ===")

ROOT = "/home/cdsw"

# model
with open(os.path.join(ROOT, "model.pkl"), "wb") as f:
    pickle.dump(model, f)

# preprocess
with open(os.path.join(ROOT, "preprocess.pkl"), "wb") as f:
    pickle.dump(
        {
            "numeric_cols": numeric_cols,
            "cat_cols": cat_cols,
            "encoders": encoders,
            "feature_names": feature_names,
            "label_col": label_col,
        },
        f
    )

# metadata
with open(os.path.join(ROOT, "meta.json"), "w") as f:
    json.dump(
        {
            "description": "Fraud detection model for claims",
            "algorithm": "XGBoost",
            "version": "v1"
        },
        f,
        indent=2
    )

print("=== MODEL + ARTIFACTS SAVED TO /home/cdsw ===")

spark.stop()