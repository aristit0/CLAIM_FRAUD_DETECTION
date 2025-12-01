#!/usr/bin/env python3
import os
import json
from datetime import datetime
from decimal import Decimal

from flask import Flask, jsonify
import mysql.connector
import requests

# ==============================
# CONFIG
# ==============================
MYSQL_CONFIG = {
    "host": "cdpmsi.tomodachis.org",
    "user": "cloudera",
    "password": "T1ku$H1t4m",
    "database": "claimdb",
}

CML_MODEL_URL = "https://modelservice.cloudera-ai.apps.ds.tomodachis.org/model?accessKey=YOUR_ACCESS_KEY"
CML_TOKEN = os.getenv("CML_MODEL_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

OPENAI_URL = "https://api.openai.com/v1/chat/completions"
OPENAI_MODEL = "gpt-5-mini"

app = Flask(__name__)

# ==============================
# MYSQL
# ==============================
def db():
    return mysql.connector.connect(**MYSQL_CONFIG)

def to_float(x):
    return float(x) if isinstance(x, (int, float, Decimal)) else 0.0

# ==============================
# RAW CLAIM LOADER
# ==============================
def load_raw_claim(claim_id):
    conn = db()
    cur = conn.cursor(dictionary=True)

    cur.execute("SELECT * FROM claim_header WHERE claim_id=%s", (claim_id,))
    header = cur.fetchone()
    if not header:
        return None

    # diagnosis primary
    cur.execute("""
        SELECT icd10_code
        FROM claim_diagnosis
        WHERE claim_id=%s
        ORDER BY is_primary DESC
        LIMIT 1
    """, (claim_id,))
    diag = cur.fetchone()

    # procedures
    cur.execute("SELECT code, cost FROM claim_procedure WHERE claim_id=%s", (claim_id,))
    procedures = cur.fetchall()

    # drugs
    cur.execute("SELECT code, cost FROM claim_drug WHERE claim_id=%s", (claim_id,))
    drugs = cur.fetchall()

    # vitamins
    cur.execute("SELECT code, cost FROM claim_vitamin WHERE claim_id=%s", (claim_id,))
    vitamins = cur.fetchall()

    cur.close()
    conn.close()

    return {
        "claim_id": claim_id,
        "patient_dob": str(header["patient_dob"]),
        "visit_date": str(header["visit_date"]),
        "visit_type": header["visit_type"],
        "department": header["department"],
        "icd10_primary_code": diag["icd10_code"] if diag else None,
        "procedures": procedures,
        "drugs": drugs,
        "vitamins": vitamins,
        "total_procedure_cost": to_float(header["total_procedure_cost"]),
        "total_drug_cost": to_float(header["total_drug_cost"]),
        "total_vitamin_cost": to_float(header["total_vitamin_cost"]),
        "total_claim_amount": to_float(header["total_claim_amount"]),
    }

# ==============================
# CALL CML RAW MODEL
# ==============================
def call_cml(raw):
    payload = {"raw_records": [raw]}
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {CML_TOKEN}",
    }
    resp = requests.post(CML_MODEL_URL, json=payload, headers=headers, verify=False)
    return resp.json()["response"]["results"][0]

# ==============================
# GPT EXPLANATION
# ==============================
def explain(claim_id, model_output):
    if not OPENAI_API_KEY:
        return "AI explanation disabled."

    prompt = f"""
Analisa klaim ID {claim_id}.
Fraud score: {model_output.get('fraud_score')}
Sinyal mencurigakan: {model_output.get('suspicious_sections')}

Buatkan penjelasan 2 paragraf untuk tim klaim.
"""

    resp = requests.post(
        OPENAI_URL,
        headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"},
        json={
            "model": OPENAI_MODEL,
            "messages": [
                {"role": "system", "content": "Kamu analis fraud kesehatan."},
                {"role": "user", "content": prompt},
            ],
        }
    )

    return resp.json()["choices"][0]["message"]["content"]

# ==============================
# MAIN ENDPOINT
# ==============================
@app.route("/score/<int:claim_id>")
def score(claim_id):

    raw = load_raw_claim(claim_id)
    if not raw:
        return jsonify({"error": "Claim not found"}), 404

    model_out = call_cml(raw)
    ai_exp = explain(claim_id, model_out)

    return jsonify({
        "claim_id": claim_id,
        "raw_input": raw,
        "model_output": model_out,
        "ai_explanation": ai_exp
    })

# ==============================
# RUN
# ==============================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=2222)