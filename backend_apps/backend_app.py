#!/usr/bin/env python3
import os
import json
from datetime import datetime
from decimal import Decimal

from flask import Flask, jsonify
import mysql.connector
import requests


# =========================================================
# ENV CONFIG
# =========================================================
MYSQL_CONFIG = {
    "host": os.getenv("MYSQL_HOST", "localhost"),
    "user": os.getenv("MYSQL_USER", "root"),
    "password": os.getenv("MYSQL_PASSWORD", ""),
    "database": os.getenv("MYSQL_DB", "claimdb"),
}

CML_MODEL_URL   = os.getenv("CML_MODEL_URL")    # sudah termasuk ?accessKey=...
CML_MODEL_TOKEN = os.getenv("CML_MODEL_TOKEN")  # Bearer token ke CML

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL   = os.getenv("OPENAI_MODEL", "gpt-5-mini")

app = Flask(__name__)


# =========================================================
# DECIMAL CLEANER
# =========================================================
def clean_decimal(obj):
    if isinstance(obj, Decimal):
        return float(obj)
    if isinstance(obj, dict):
        return {k: clean_decimal(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clean_decimal(v) for v in obj]
    return obj


# =========================================================
# MYSQL CONNECTION
# =========================================================
def db():
    return mysql.connector.connect(**MYSQL_CONFIG)


def to_float(x):
    return float(x) if isinstance(x, (int, float, Decimal)) else 0.0


# =========================================================
# LOAD RAW CLAIM
# =========================================================
def load_raw_claim(claim_id):
    conn = db()
    cur = conn.cursor(dictionary=True)

    # Header
    cur.execute("""
        SELECT claim_id, patient_dob, visit_date, visit_type, department,
               total_procedure_cost, total_drug_cost, total_vitamin_cost,
               total_claim_amount
        FROM claim_header
        WHERE claim_id=%s
    """, (claim_id,))
    header = cur.fetchone()
    if not header:
        cur.close()
        conn.close()
        return None

    # Diagnosis utama
    cur.execute("""
        SELECT icd10_code
        FROM claim_diagnosis
        WHERE claim_id=%s
        ORDER BY is_primary DESC
        LIMIT 1
    """, (claim_id,))
    diag = cur.fetchone()

    # Procedures (pakai cost)
    cur.execute("""
        SELECT icd9_code AS code, cost
        FROM claim_procedure
        WHERE claim_id=%s
    """, (claim_id,))
    procedures = cur.fetchall()

    # Drugs
    cur.execute("""
        SELECT drug_code AS code, cost
        FROM claim_drug
        WHERE claim_id=%s
    """, (claim_id,))
    drugs = cur.fetchall()

    # Vitamins
    cur.execute("""
        SELECT vitamin_name AS code, cost
        FROM claim_vitamin
        WHERE claim_id=%s
    """, (claim_id,))
    vitamins = cur.fetchall()

    cur.close()
    conn.close()

    raw = {
        "claim_id":          claim_id,
        "patient_dob":       str(header["patient_dob"]),
        "visit_date":        str(header["visit_date"]),
        "visit_type":        header["visit_type"],
        "department":        header["department"],
        "icd10_primary_code": diag["icd10_code"] if diag else None,
        "procedures":        procedures,
        "drugs":             drugs,
        "vitamins":          vitamins,
        "total_procedure_cost": to_float(header["total_procedure_cost"]),
        "total_drug_cost":      to_float(header["total_drug_cost"]),
        "total_vitamin_cost":   to_float(header["total_vitamin_cost"]),
        "total_claim_amount":   to_float(header["total_claim_amount"]),
    }

    # Pastikan tidak ada Decimal lagi
    return clean_decimal(raw)


# =========================================================
# CALL CML MODEL (PAKAI TOKEN + request WRAPPER)
# =========================================================
def call_cml(raw):
    """
    CML expected pattern (sesuai sample):

    POST {CML_MODEL_URL}?accessKey=...
    Headers:
        Authorization: Bearer <CML_MODEL_TOKEN>
        Content-Type: application/json
    Body:
        {
          "request": {
            "raw_records": [ {...} ]
          }
        }
    """

    if not CML_MODEL_URL:
        return {"error": "cml_url_not_set"}

    headers = {
        "Content-Type": "application/json",
    }

    # Kalau ada token, pakai Authorization Bearer
    if CML_MODEL_TOKEN:
        headers["Authorization"] = f"Bearer {CML_MODEL_TOKEN}"

    payload = {
        "request": {
            "raw_records": [clean_decimal(raw)]
        }
    }

    try:
        resp = requests.post(
            CML_MODEL_URL,
            json=payload,
            headers=headers,
            verify=False,
            timeout=20
        )
    except Exception as e:
        return {"error": "cml_connection_error", "details": str(e)}

    # Kalau bukan 200, return apa adanya buat debugging
    try:
        j = resp.json()
    except Exception:
        return {
            "error": "cml_invalid_json",
            "status_code": resp.status_code,
            "text": resp.text,
        }

    # Struktur standar CML: {"response": {"results": [...]}}
    if "response" in j and "results" in j["response"]:
        return clean_decimal(j["response"]["results"][0])

    # Model function saya yang pakai decorator cml_model
    if "results" in j:
        return clean_decimal(j["results"][0])

    # Error-style
    if "errors" in j:
        return {
            "error": "model_error",
            "details": clean_decimal(j["errors"])
        }

    return {
        "error": "unknown_response_format",
        "status_code": resp.status_code,
        "raw": j
    }


# =========================================================
# GPT EXPLANATION
# =========================================================
def generate_explanation(claim_id, model_output):
    if not OPENAI_API_KEY:
        return "AI explanation disabled."

    prompt = f"""
Analisis klaim ID {claim_id}:

Fraud score: {model_output.get("fraud_score")}
Model flag: {model_output.get("model_flag")}
Sinyal mencurigakan: {model_output.get("suspicious_sections")}

Buatkan penjelasan 2 paragraf untuk tim klaim.
"""

    resp = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": OPENAI_MODEL,
            "messages": [
                {"role": "system", "content": "Kamu analis fraud kesehatan."},
                {"role": "user", "content": prompt}
            ]
        }
    )

    try:
        return resp.json()["choices"][0]["message"]["content"]
    except Exception:
        return "AI explanation error."


# =========================================================
# API ENDPOINT
# =========================================================
@app.route("/score/<int:claim_id>", methods=["GET"])
def score(claim_id):
    raw = load_raw_claim(claim_id)
    if not raw:
        return jsonify({"error": "Claim not found"}), 404

    model_output = call_cml(raw)
    explanation  = generate_explanation(claim_id, model_output)

    response = {
        "claim_id":       claim_id,
        "raw_input":      raw,
        "model_output":   model_output,
        "ai_explanation": explanation
    }

    return jsonify(clean_decimal(response))


# =========================================================
# RUN
# =========================================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=2222)