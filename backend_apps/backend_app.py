#!/usr/bin/env python3
import os
import json
from datetime import datetime
from decimal import Decimal

from flask import Flask, jsonify
import mysql.connector
import requests

MYSQL = {
    "host": "cdpmsi.tomodachis.org",
    "user": "cloudera",
    "password": "T1ku$H1t4m",
    "database": "claimdb",
}

CML_URL = "https://modelservice.cloudera-ai.apps.ds.tomodachis.org/model?accessKey=YOUR_ACCESS_KEY"
CML_TOKEN = os.getenv("CML_MODEL_TOKEN")

OPENAI = os.getenv("OPENAI_API_KEY")
GPT_URL = "https://api.openai.com/v1/chat/completions"
GPT_MODEL = "gpt-5-mini"

app = Flask(__name__)

def db(): return mysql.connector.connect(**MYSQL)

def to_f(x):
    return float(x) if isinstance(x, (int, float, Decimal)) else 0.0

# =====================================================
# Load RAW CLAIM from MySQL
# =====================================================
def load_claim_raw(claim_id):
    conn = db()
    cur = conn.cursor(dictionary=True)

    cur.execute("SELECT * FROM claim_header WHERE claim_id=%s", (claim_id,))
    h = cur.fetchone()
    if not h: return None

    cur.execute("SELECT icd10_code FROM claim_diagnosis WHERE claim_id=%s ORDER BY is_primary DESC LIMIT 1", (claim_id,))
    d = cur.fetchone()

    cur.execute("SELECT code, cost FROM claim_procedure WHERE claim_id=%s", (claim_id,))
    proc = cur.fetchall()

    cur.execute("SELECT code, cost FROM claim_drug WHERE claim_id=%s", (claim_id,))
    drug = cur.fetchall()

    cur.execute("SELECT code, cost FROM claim_vitamin WHERE claim_id=%s", (claim_id,))
    vit = cur.fetchall()

    cur.close()
    conn.close()

    return {
        "claim_id": claim_id,
        "patient_dob": str(h["patient_dob"]),
        "visit_date": str(h["visit_date"]),
        "visit_type": h["visit_type"],
        "department": h["department"],
        "icd10_primary_code": d["icd10_code"] if d else None,
        "procedures": proc,
        "drugs": drug,
        "vitamins": vit,
        "total_procedure_cost": to_f(h["total_procedure_cost"]),
        "total_drug_cost": to_f(h["total_drug_cost"]),
        "total_vitamin_cost": to_f(h["total_vitamin_cost"]),
        "total_claim_amount": to_f(h["total_claim_amount"]),
    }

# =====================================================
# Call CML
# =====================================================
def call_model(raw):
    resp = requests.post(
        CML_URL,
        json={"raw_records": [raw]},
        headers={"Authorization": f"Bearer {CML_TOKEN}"},
        verify=False
    )
    return resp.json()["response"]["results"][0]

# =====================================================
# OpenAI Explanation
# =====================================================
def explain(cid, model):
    if not OPENAI: return "AI disabled."

    prompt = f"""
Klaim ID {cid}.
Fraud Score: {model['fraud_score']}
Sinyal mencurigakan: {model['suspicious_sections']}

Tuliskan penjelasan 2 paragraf untuk tim klaim.
"""

    resp = requests.post(
        GPT_URL,
        headers={"Authorization": f"Bearer {OPENAI}", "Content-Type": "application/json"},
        json={
            "model": GPT_MODEL,
            "messages": [
                {"role": "system", "content": "Kamu analis fraud kesehatan."},
                {"role": "user", "content": prompt}
            ]
        }
    )

    return resp.json()["choices"][0]["message"]["content"]

# =====================================================
# Endpoint
# =====================================================
@app.route("/score/<int:claim_id>")
def score(claim_id):

    raw = load_claim_raw(claim_id)
    if not raw:
        return jsonify({"error": "Claim not found"}), 404

    model_out = call_model(raw)
    ai_text = explain(claim_id, model_out)

    return jsonify({
        "claim_id": claim_id,
        "raw_input": raw,
        "model_output": model_out,
        "ai_explanation": ai_text
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=2222)