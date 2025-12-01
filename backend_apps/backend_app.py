#!/usr/bin/env python3
import os
import json
from datetime import datetime
from decimal import Decimal

from flask import Flask, jsonify
import mysql.connector
import requests

# ==========================================================
# CONFIG
# ==========================================================
MYSQL_CONF = {
    "host": "cdpmsi.tomodachis.org",
    "user": "cloudera",
    "password": "T1ku$H1t4m",
    "database": "claimdb",
}

CML_URL = "https://modelservice.cloudera-ai.apps.ds.tomodachis.org/model?accessKey=YOUR_ACCESS_KEY"
CML_TOKEN = os.getenv("CML_MODEL_TOKEN")

OPENAI_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_URL = "https://api.openai.com/v1/chat/completions"
OPENAI_MODEL = "gpt-5-mini"

app = Flask(__name__)

# ==========================================================
# MYSQL HELPERS
# ==========================================================
def db():
    return mysql.connector.connect(**MYSQL_CONF)

def to_float(x):
    return float(x) if isinstance(x, (int, float, Decimal)) else 0.0

# ==========================================================
# LOAD RAW CLAIM FROM DB
# ==========================================================
def load_claim_raw(claim_id):
    conn = db()
    cur = conn.cursor(dictionary=True)

    cur.execute("SELECT * FROM claim_header WHERE claim_id=%s", (claim_id,))
    header = cur.fetchone()
    if not header:
        return None

    cur.execute("""
        SELECT icd10_code
        FROM claim_diagnosis
        WHERE claim_id=%s ORDER BY is_primary DESC LIMIT 1
    """, (claim_id,))
    diag = cur.fetchone()

    cur.execute("SELECT code, cost FROM claim_procedure WHERE claim_id=%s", (claim_id,))
    procedures = cur.fetchall()

    cur.execute("SELECT code, cost FROM claim_drug WHERE claim_id=%s", (claim_id,))
    drugs = cur.fetchall()

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

# ==========================================================
# CALL CML MODEL (RAW MODE)
# ==========================================================
def model_infer(raw):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {CML_TOKEN}"
    }
    payload = {"raw_records": [raw]}

    resp = requests.post(CML_URL, json=payload, headers=headers, verify=False)

    return resp.json()["response"]["results"][0]

# ==========================================================
# GPT COMMENTARY
# ==========================================================
def generate_explanation(cid, model_out):
    if not OPENAI_KEY:
        return "AI explanation disabled."

    prompt = f"""
Analisa klaim ID {cid}.
Fraud score: {model_out.get('fraud_score')}
Suspicious signals: {model_out.get('suspicious_sections')}

Berikan penjelasan 2 paragraf dengan bahasa profesional untuk tim klaim.
"""

    headers = {
        "Authorization": f"Bearer {OPENAI_KEY}",
        "Content-Type": "application/json"
    }

    resp = requests.post(
        OPENAI_URL,
        headers=headers,
        json={
            "model": OPENAI_MODEL,
            "messages": [
                {"role": "system", "content": "Kamu analis fraud kesehatan."},
                {"role": "user", "content": prompt}
            ]
        }
    )

    return resp.json()["choices"][0]["message"]["content"]

# ==========================================================
# SCORE ENDPOINT
# ==========================================================
@app.route("/score/<int:claim_id>")
def score(claim_id):

    raw = load_claim_raw(claim_id)
    if not raw:
        return jsonify({"error": "Claim not found"}), 404

    model_out = model_infer(raw)
    explanation = generate_explanation(claim_id, model_out)

    return jsonify({
        "claim_id": claim_id,
        "raw_input": raw,
        "model_output": model_out,
        "ai_explanation": explanation
    })

# ==========================================================
# RUN SERVER
# ==========================================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=2222)