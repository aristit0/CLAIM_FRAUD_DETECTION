import cml.data_v1 as cmldata
from pyspark.sql.functions import col
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import mlflow
import mlflow.xgboost

# ============================================
# 1. Connect ke Spark via CML
# ============================================
CONNECTION_NAME = "CDP-MSI"
conn = cmldata.get_connection(CONNECTION_NAME)
spark = conn.get_spark_session()

print("=== LOAD FEATURE TABLE ===")

df_spark = spark.sql("""
    SELECT *
    FROM ice.iceberg_curated.claim_feature_set
""")

df_spark.printSchema()
print(f"Total rows: {df_spark.count()}")

# ============================================
# 2. Pilih kolom feature & label
# ============================================

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

# Kalau datanya super besar dan kamu cuma mau sampling:
# df_spark_sel = df_spark_sel.sample(withReplacement=False, fraction=0.3)

# Convert ke pandas
df = df_spark_sel.dropna(subset=[label_col]).toPandas()
print(df.head())
print(df[label_col].value_counts())

# ============================================
# 3. Encode categorical & handle nulls
# ============================================

encoders = {}
for c in cat_cols:
    le = LabelEncoder()
    # fillna ke string spesial biar tidak error
    df[c] = df[c].fillna("__MISSING__")
    df[c] = le.fit_transform(df[c].astype(str))
    encoders[c] = le

# Numeric null -> 0 (atau bisa pakai median, terserah kamu nanti tweak)
for c in numeric_cols:
    df[c] = df[c].fillna(0.0)

X = df[numeric_cols + cat_cols]
y = df[label_col].astype(int)

print(X.shape, y.shape)


# ============================================
# 4. Train / test split (stratified)
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Train size:", X_train.shape[0])
print("Test size :", X_test.shape[0])


# ============================================
# 5. Train XGBoost model
# ============================================

pos_ratio = (y_train == 1).sum() / len(y_train)
neg_ratio = (y_train == 0).sum() / len(y_train)
scale_pos_weight = neg_ratio / pos_ratio if pos_ratio > 0 else 1.0
print("scale_pos_weight:", scale_pos_weight)

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
    tree_method="hist",
)

model.fit(
    X_train,
    y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    verbose=20
)


from sklearn.metrics import roc_auc_score, f1_score, classification_report

y_pred_proba = model.predict_proba(X_test)[:, 1]
y_pred = (y_pred_proba >= 0.5).astype(int)

auc = roc_auc_score(y_test, y_pred_proba)
f1 = f1_score(y_test, y_pred)

print("AUC:", auc)
print("F1 :", f1)
print(classification_report(y_test, y_pred))


# ============================================
# 6. Log ke MLflow & register model
# ============================================

mlflow.set_experiment("fraud_detection_claims")

feature_names = list(X.columns)

with mlflow.start_run(run_name="xgboost_fraud_detection_v1"):
    mlflow.log_param("model_type", "xgboost")
    mlflow.log_param("n_estimators", 300)
    mlflow.log_param("max_depth", 5)
    mlflow.log_param("learning_rate", 0.05)
    mlflow.log_param("scale_pos_weight", scale_pos_weight)

    mlflow.log_metric("auc", auc)
    mlflow.log_metric("f1", f1)

    # Simpan info preprocessing sebagai artifact
    import pickle, os, json
    import tempfile

    temp_dir = tempfile.mkdtemp()
    preprocess_path = os.path.join(temp_dir, "preprocess.pkl")
    meta_path = os.path.join(temp_dir, "meta.json")

    with open(preprocess_path, "wb") as f:
        pickle.dump(
            {
                "numeric_cols": numeric_cols,
                "cat_cols": cat_cols,
                "encoders": encoders,
                "feature_names": feature_names,
                "label_col": label_col,
            },
            f,
        )

    with open(meta_path, "w") as f:
        json.dump(
            {
                "description": "Fraud detection model for claims",
                "version": "v1",
            },
            f,
            indent=2
        )

    mlflow.log_artifact(preprocess_path, artifact_path="artifacts")
    mlflow.log_artifact(meta_path, artifact_path="artifacts")

    # Log XGBoost model
    mlflow.xgboost.log_model(
        model,
        artifact_path="model",
        registered_model_name="fraud_detection_claims_xgb"
    )

    run_id = mlflow.active_run().info.run_id
    print("Logged run_id:", run_id)

    