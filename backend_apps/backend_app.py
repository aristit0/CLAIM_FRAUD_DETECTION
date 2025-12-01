#!/usr/bin/env python3
import os
import json
from datetime import datetime, date

from flask import Flask, jsonify, request
import mysql.connector
import requests

# ==============================
# Config
# ==============================

MYSQL_CONFIG = {
    "host": os.getenv("MYSQL_HOST", "cdpmsi.tomodachis.org"),
    "user": os.getenv("MYSQL_USER", "cloudera"),
    "password": os.getenv("MYSQL_PASSWORD", "T1ku$H1t4m"),
    "database": os.getenv("MYSQL_DB", "claimdb"),
}

# CML Model Serving URL:
CML_MODEL_URL = os.getenv(
    "CML_MODEL_URL",
    "https://modelservice.cloudera-ai.apps.ds.tomodachis.org/model?accessKey=YOUR_ACCESS_KEY"
)
CML_MODEL_TOKEN = os.getenv("CML_MODEL_TOKEN")

# OpenAI API
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_CHAT_URL = "https://api.openai.com/v1/chat/completions"
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-mini")


app = Flask(__name__)


# ==============================
# MySQL helper
# ==============================
def get_mysql_connection():
    return mysql.connector.connect(**MYSQL_CONFIG)


# ==============================
# Age calculation
# ==============================
def compute_age(dob, visit_date):
    if not dob or not visit_date:
        return 0

    if isinstance(dob, str):
        dob = datetime.strptime(dob, "%Y-%m-%d").date()
    if isinstance(visit_date, str):
        visit_date = datetime.strptime(visit_date, "%Y-%m-%d").date()

    age = visit_date.year - dob.year - (
        (visit_date.month, visit_date.day) < (dob.month, dob.day)
    )
    return max(age, 0)


# ==============================
# Fetch data from MySQL → raw feature row
# ==============================
def get_claim_features(claim_id: int):
    conn = get_mysql_connection()
    cur = conn.cursor(dictionary=True)

    cur.execute("""
        SELECT claim_id, patient_dob, visit_date, patient_name,
               visit_type, department,
               total_procedure_cost, total_drug_cost,
               total_vitamin_cost, total_claim_amount
        FROM claim_header
        WHERE claim_id=%s
    """, (claim_id,))
    header = cur.fetchone()
    if not header:
        cur.close()
        conn.close()
        return None

    cur.execute("""
        SELECT icd10_code, icd10_description
        FROM claim_diagnosis
        WHERE claim_id=%s
        ORDER BY is_primary DESC, id ASC
        LIMIT 1
    """, (claim_id,))
    diag = cur.fetchone()

    cur.close()
    conn.close()

    visit_date = header["visit_date"]
    if isinstance(visit_date, datetime):
        dt = visit_date.date()
    else:
        dt = datetime.strptime(str(visit_date), "%Y-%m-%d").date()

    feature_row = {
        "claim_id": claim_id,
        "patient_age": compute_age(header["patient_dob"], dt),
        "visit_year": dt.year,
        "visit_month": dt.month,
        "visit_day": dt.day,
        "visit_type": header["visit_type"],
        "department": header["department"],
        "icd10_primary_code": diag["icd10_code"] if diag else None,
        "total_claim_amount": float(header["total_claim_amount"] or 0),
        "total_procedure_cost": float(header["total_procedure_cost"] or 0),
        "total_drug_cost": float(header["total_drug_cost"] or 0),
        "total_vitamin_cost": float(header["total_vitamin_cost"] or 0)
    }

    return feature_row


# ==============================
# Call CML model
# ==============================
def call_cml_model(feature_row: dict):
    payload = {
        "request": {
            "records": [feature_row]
        }
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {CML_MODEL_TOKEN}",
    }

    resp = requests.post(
        CML_MODEL_URL,
        json=payload,
        headers=headers,
        verify=False,
        timeout=30,
    )

    data = resp.json()
    return data["response"]["results"][0]


# ==============================
# Call ChatGPT
# ==============================
def generate_ai_explanation(claim_id: int, model_output: dict):
    if not OPENAI_API_KEY:
        return "AI explanation tidak tersedia."

    fraud = model_output.get("fraud_score")
    suspicious = model_output.get("suspicious_sections", [])
    final_flag = model_output.get("final_flag")

    prompt = f"""
Analisa klaim kesehatan berdasarkan output model:

- Claim ID: {claim_id}
- Fraud score: {fraud}
- Suspicious: {suspicious}
- Final Flag (0=wajar, 1=fraud): {final_flag}

Buat penjelasan dua paragraf yang mudah dipahami tim klaim.
"""

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}",
    }

    body = {
        "model": OPENAI_MODEL,
        "messages": [
            {"role": "system", "content": "Kamu adalah analis klaim asuransi kesehatan."},
            {"role": "user", "content": prompt}
        ]
    }

    resp = requests.post(OPENAI_CHAT_URL, json=body, headers=headers)
    return resp.json()["choices"][0]["message"]["content"]


# ==============================
# Main API endpoint /score/<id>
# ==============================
@app.route("/score/<int:claim_id>", methods=["GET"])
def score_claim(claim_id):
    try:
        feature_row = get_claim_features(claim_id)
        if not feature_row:
            return jsonify({"error": "Claim not found"}), 404

        # call CML model
        model_output = call_cml_model(feature_row)

        # AI explanation
        ai_explanation = generate_ai_explanation(claim_id, model_output)

        # final response
        return jsonify({
            "claim_id": claim_id,
            "features": feature_row,
            "model_output": model_output,
            "ai_explanation": ai_explanation
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ==============================
# RUN APP
# ==============================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=2222)