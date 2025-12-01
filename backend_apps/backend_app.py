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

CML_MODEL_URL = (
    "https://modelservice.cloudera-ai.apps.ds.tomodachis.org/model"
    "?accessKey=YOUR_ACCESS_KEY"
)
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
    if isinstance(x, Decimal):
        return float(x)
    if isinstance(x, (int, float)):
        return float(x)
    return 0.0


# ==============================
# LOAD RAW CLAIM
# ==============================
def load_raw_claim(claim_id):
    conn = db()
    cur = conn.cursor(dictionary=True)

    # HEADER
    cur.execute("SELECT * FROM claim_header WHERE claim_id=%s", (claim_id,))
    header = cur.fetchone()
    if not header:
        return None

    # DIAGNOSIS PRIMARY
    cur.execute("""
        SELECT icd10_code
        FROM claim_diagnosis
        WHERE claim_id=%s
        ORDER BY is_primary DESC
        LIMIT 1
    """, (claim_id,))
    dx = cur.fetchone()

    # PROCEDURES (icd9_code → "code", cost → cost)
    cur.execute("""
        SELECT icd9_code AS code, cost
        FROM claim_procedure
        WHERE claim_id=%s
    """, (claim_id,))
    procedures = [
        {"code": p["code"], "cost": to_float(p["cost"])}
        for p in cur.fetchall()
    ]

    # DRUGS
    cur.execute("""
        SELECT drug_code AS code, cost
        FROM claim_drug
        WHERE claim_id=%s
    """, (claim_id,))
    drugs = [
        {"code": d["code"], "cost": to_float(d["cost"])}
        for d in cur.fetchall()
    ]

    # VITAMINS
    cur.execute("""
        SELECT vitamin_name AS code, cost
        FROM claim_vitamin
        WHERE claim_id=%s
    """, (claim_id,))
    vitamins = [
        {"code": v["code"], "cost": to_float(v["cost"])}
        for v in cur.fetchall()
    ]

    cur.close()
    conn.close()

    return {
        "claim_id": claim_id,
        "patient_dob": str(header["patient_dob"]),
        "visit_date": str(header["visit_date"]),
        "visit_type": header["visit_type"],
        "department": header["department"],
        "icd10_primary_code": dx["icd10_code"] if dx else None,
        "procedures": procedures,
        "drugs": drugs,
        "vitamins": vitamins,
        "total_procedure_cost": to_float(header["total_procedure_cost"]),
        "total_drug_cost": to_float(header["total_drug_cost"]),
        "total_vitamin_cost": to_float(header["total_vitamin_cost"]),
        "total_claim_amount": to_float(header["total_claim_amount"]),
    }


# ==============================
# CALL MODEL
# ==============================
def call_cml(raw_record):
    payload = {"raw_records": [raw_record]}
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {CML_TOKEN}",
    }

    resp = requests.post(
        CML_MODEL_URL,
        json=payload,
        headers=headers,
        verify=False
    )

    return resp.json()["response"]["results"][0]


# ==============================
# GPT EXPLANATION
# ==============================
def explain(claim_id, model_out):

    if not OPENAI_API_KEY:
        return "AI explanation disabled."

    prompt = f"""
Analisa klaim ID {claim_id}.
Fraud score: {model_out.get('fraud_score')}
Sinyal mencurigakan: {model_out.get('suspicious_sections')}

Buatkan penjelasan 2 paragraf untuk tim klaim.
"""

    resp = requests.post(
        OPENAI_URL,
        headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        },
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
# ENDPOINT
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