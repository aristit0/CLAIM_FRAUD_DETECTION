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

# MySQL config (isi via ENV di systemd)
MYSQL_CONFIG = {
    "host": os.getenv("MYSQL_HOST", "cdpmsi.tomodachis.org"),
    "user": os.getenv("MYSQL_USER", "cloudera"),
    "password": os.getenv("MYSQL_PASSWORD"),  # WAJIB di-set via ENV
    "database": os.getenv("MYSQL_DB", "claimdb"),
}

# CML Model Serving
# Pastikan ENV CML_MODEL_URL berisi FULL URL dengan accessKey, contoh:
# https://modelservice.cloudera-ai.apps.ds.tomodachis.org/model?accessKey=xxxx
CML_MODEL_URL = os.getenv(
    "CML_MODEL_URL",
    "https://modelservice.cloudera-ai.apps.ds.tomodachis.org/model?accessKey=YOUR_ACCESS_KEY"
)
CML_MODEL_TOKEN = os.getenv("CML_MODEL_TOKEN")  # Bearer token (sama seperti test_model.py)

# OpenAI ChatGPT
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_CHAT_URL = "https://api.openai.com/v1/chat/completions"
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-mini")

# HBase REST
HBASE_REST_HOST = os.getenv("HBASE_REST_HOST", "https://cdpmsi.tomodachis.org:20550")
HBASE_TABLE = os.getenv("HBASE_TABLE", "fraud_scoring")
HBASE_CF = os.getenv("HBASE_CF", "cf")
HBASE_COL = os.getenv("HBASE_COL", "result")
HBASE_VERIFY_SSL = False  # Self-signed cert, jika sudah ada CA, bisa diganti True

app = Flask(__name__)


# ==============================
# Helper: MySQL Connection
# ==============================

def get_mysql_connection():
    return mysql.connector.connect(**MYSQL_CONFIG)


# ==============================
# Helper: MySQL -> Feature Row
# ==============================

def _compute_age(patient_dob, visit_date):
    """
    Hitung umur berdasarkan patient_dob dan visit_date.
    Jika salah satu tidak valid → umur = 0.
    """

    if not patient_dob or not visit_date:
        return 0

    # Normalisasi patient_dob
    if isinstance(patient_dob, str):
        patient_dob = datetime.strptime(patient_dob, "%Y-%m-%d").date()
    elif isinstance(patient_dob, datetime):
        patient_dob = patient_dob.date()

    # Normalisasi visit_date
    if isinstance(visit_date, str):
        visit_date = datetime.strptime(visit_date, "%Y-%m-%d").date()
    elif isinstance(visit_date, datetime):
        visit_date = visit_date.date()

    if visit_date < patient_dob:
        return 0

    # Perhitungan umur akurat
    age = visit_date.year - patient_dob.year - (
        (visit_date.month, visit_date.day) < (patient_dob.month, patient_dob.day)
    )

    return age


def get_claim_features(claim_id: int):
    """
    Ambil data klaim dari beberapa tabel MySQL
    dan bentuk satu dict fitur untuk model.
    Field disesuaikan dengan feature yang dipakai model:
      - patient_age
      - visit_year, visit_month, visit_day
      - visit_type, department
      - icd10_primary_code
      - total_claim_amount, total_procedure_cost, total_drug_cost, total_vitamin_cost
    """
    conn = get_mysql_connection()
    cursor = conn.cursor(dictionary=True)

    # Header: info utama claim (sesuai DESCRIBE claim_header)
    cursor.execute("""
    SELECT 
        claim_id,
        patient_dob,
        patient_name,
        visit_date,
        visit_type,
        department,
        total_procedure_cost,
        total_drug_cost,
        total_vitamin_cost,
        total_claim_amount,
        status
    FROM claim_header
    WHERE claim_id = %s
    """, (claim_id,))
    header = cursor.fetchone()
    if not header:
        cursor.close()
        conn.close()
        return None

    # Ambil primary ICD10: pakai is_primary = 1 atau kalau tidak ada, ambil pertama
    cursor.execute("""
        SELECT icd10_code, icd10_description
        FROM claim_diagnosis
        WHERE claim_id = %s
        ORDER BY is_primary DESC, id ASC
        LIMIT 1
    """, (claim_id,))
    diag = cursor.fetchone()
    header["icd10_primary_code"] = diag["icd10_code"] if diag else None
    header["icd10_primary_description"] = diag["icd10_description"] if diag else None
    cursor.close()
    conn.close()

    # Hitung umur dan pecah visit_date
    visit_date = header.get("visit_date")
    patient_dob = header.get("patient_dob")

    if isinstance(visit_date, datetime):
        dt = visit_date
    elif isinstance(visit_date, date):
        dt = datetime.combine(visit_date, datetime.min.time())
    else:
        dt = datetime.strptime(str(visit_date), "%Y-%m-%d")

    patient_age = _compute_age(patient_dob, dt.date())

    feature_row = {
        "claim_id": header["claim_id"],
        "patient_age": patient_age,
        "visit_year": dt.year,
        "visit_month": dt.month,
        "visit_day": dt.day,
        "visit_type": header.get("visit_type"),
        "department": header.get("department"),
        "icd10_primary_code": header.get("icd10_primary_code"),
        "total_claim_amount": float(header.get("total_claim_amount") or 0),
        "total_procedure_cost": float(header.get("total_procedure_cost") or 0),
        "total_drug_cost": float(header.get("total_drug_cost") or 0),
        "total_vitamin_cost": float(header.get("total_vitamin_cost") or 0),
    }

    return feature_row


# ==============================
# Helper: Compute Rules / Scores
# ==============================

def compute_scores(feature_row: dict):
    """
    Dummy / initial rule scoring.
    Bisa kamu refine nanti.
    """
    total_proc = float(feature_row.get("total_procedure_cost", 0) or 0)
    total_drug = float(feature_row.get("total_drug_cost", 0) or 0)
    total_vit = float(feature_row.get("total_vitamin_cost", 0) or 0)
    total_claim = float(feature_row.get("total_claim_amount", 0) or 0)

    # contoh rule sederhana
    tindakan_validity_score = 1.0 if total_proc > 0 else 0.3
    obat_validity_score = 1.0 if total_drug <= total_claim else 0.2
    vitamin_relevance_score = 1.0 if total_vit < 50000 else 0.4

    denom = total_proc + 1.0
    biaya_anomaly_score = total_claim / denom if denom > 0 else 0.0

    rule_violation_flag = 1 if biaya_anomaly_score > 3.0 else 0
    rule_violation_reason = (
        "biaya tidak wajar dibandingkan tindakan"
        if rule_violation_flag
        else None
    )

    return {
        "tindakan_validity_score": tindakan_validity_score,
        "obat_validity_score": obat_validity_score,
        "vitamin_relevance_score": vitamin_relevance_score,
        "biaya_anomaly_score": biaya_anomaly_score,
        "rule_violation_flag": rule_violation_flag,
        "rule_violation_reason": rule_violation_reason,
    }

# ==============================
# Helper: Call CML Model Serving
# ==============================

def call_cml_model(feature_row: dict):
    """
    Kirim 1 record ke CML Model Serving dan ambil hasil.

    Format DISESUIKAN dengan script /root/test_model.py yang sudah sukses:

      url = "https://modelservice.../model?accessKey=..."
      headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer <TOKEN>"
      }
      payload = { "request": { "records": [ feature_row ] } }

    Response contoh:
    {
      "success": true,
      "response": {
        "results": [ {...} ]
      },
      "ReplicaID": "...",
      "Size": 1000,
      "StatusCode": 200
    }
    """

    if not CML_MODEL_TOKEN:
        raise RuntimeError("CML_MODEL_TOKEN tidak di-set di environment")

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
        verify=False,   # sama seperti test_model.py
        timeout=30,
    )

    if resp.status_code != 200:
        raise RuntimeError(f"CML Model HTTP {resp.status_code}: {resp.text}")

    data = resp.json()

    # Pastikan "success" true
    if not data.get("success", False):
        raise RuntimeError(f"CML Model success=false: {data}")

    # Ambil "response.results"
    response_obj = data.get("response") or {}
    results = response_obj.get("results")

    if not results:
        raise RuntimeError(f"Tidak ada 'results' di response model: {data}")

    return results[0]


# ==============================
# Helper: Call OpenAI ChatGPT
# ==============================

def generate_ai_explanation(claim_id: int, model_output: dict, feature_row: dict):
    """
    Call ChatGPT untuk bikin penjelasan human-readable
    buat fraud_score + suspicious_sections + rule_violations.
    """
    if not OPENAI_API_KEY:
        # Kalau tidak ada API key, kembalikan penjelasan default
        return "AI explanation tidak tersedia (OPENAI_API_KEY belum di-set)."

    fraud_score = model_output.get("fraud_score")
    suspicious_sections = model_output.get("suspicious_sections", [])
    rule_violations = model_output.get("rule_violations", {})

    prompt = f"""
Kamu adalah asisten fraud analyst untuk klaim asuransi kesehatan.

Berikut ringkasan klaim:

- Claim ID: {claim_id}
- Fraud score (0-1): {fraud_score}
- Suspicious sections: {suspicious_sections}
- Rule violations: {rule_violations}

Berikut beberapa nilai fitur penting:
- total_claim_amount: {feature_row.get("total_claim_amount")}
- total_procedure_cost: {feature_row.get("total_procedure_cost")}
- total_drug_cost: {feature_row.get("total_drug_cost")}
- total_vitamin_cost: {feature_row.get("total_vitamin_cost")}
- biaya_anomaly_score: {feature_row.get("biaya_anomaly_score")}
- tindakan_validity_score: {feature_row.get("tindakan_validity_score")}
- obat_validity_score: {feature_row.get("obat_validity_score")}
- vitamin_relevance_score: {feature_row.get("vitamin_relevance_score")}

Buat penjelasan singkat dalam bahasa Indonesia,
maksimal 2 paragraf, tentang:
1) Apakah klaim ini tampak wajar atau mencurigakan (berdasarkan fraud_score),
2) Bagian mana yang perlu dicek lebih lanjut (kalau ada suspicious_sections),
3) Saran tindakan untuk tim klaim (misal: "boleh langsung approve", atau "perlu review manual dulu").
"""

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}",
    }

    body = {
        "model": OPENAI_MODEL,
        "messages": [
            {
                "role": "system",
                "content": "Kamu adalah analis klaim asuransi yang membantu menjelaskan skor fraud ke tim klaim."
            },
            {
                "role": "user",
                "content": prompt
            },
        ],
    }

    resp = requests.post(
        OPENAI_CHAT_URL,
        headers=headers,
        json=body,
        timeout=30,
    )

    if resp.status_code != 200:
        return f"Gagal memanggil OpenAI: {resp.status_code} {resp.text}"

    data = resp.json()
    try:
        return data["choices"][0]["message"]["content"]
    except Exception:
        return f"Format response OpenAI tidak terduga: {data}"


# ==============================
# Helper: Write to HBase via REST
# ==============================

def write_to_hbase(claim_id: int, record: dict):
    """
    Simpan hasil scoring ke HBase sebagai JSON string.
    Rowkey = claim_id, kolom cf:result
    """
    rowkey = str(claim_id)
    url = f"{HBASE_REST_HOST}/{HBASE_TABLE}/{rowkey}/{HBASE_CF}:{HBASE_COL}"

    # Pastikan semua tipe bisa di-JSON-kan
    payload_str = json.dumps(record, default=float)

    headers = {
        "Content-Type": "application/octet-stream"
    }

    resp = requests.put(
        url,
        data=payload_str.encode("utf-8"),
        headers=headers,
        verify=HBASE_VERIFY_SSL,
        timeout=30,
    )

    if resp.status_code not in (200, 201):
        raise RuntimeError(f"Gagal tulis ke HBase: {resp.status_code} {resp.text}")
    

# ==============================
# Flask Endpoints
# ==============================

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


@app.route("/score/<int:claim_id>", methods=["POST", "GET"])
def score_claim(claim_id):
    """
    Flow (sesuai TAHAP 7):

      1. Terima claim_id
      2. Ambil data dari MySQL
      3. Hitung rule-based scores
      4. Kirim fitur ke CML Model Serving
      5. Dapat fraud_score
      6. Panggil ChatGPT untuk AI Explanation
      7. Tulis hasil scoring ke HBase
      8. Return response ke Frontend Approval
    """
    try:
        # 2. Ambil raw data klaim dari MySQL
        base_row = get_claim_features(claim_id)
        if not base_row:
            return jsonify({"error": f"claim_id {claim_id} tidak ditemukan"}), 404

        # 3. Hitung rule-based scores
        scores = compute_scores(base_row)

        # Gabung feature final untuk model
        feature_row = {**base_row, **scores}

        # 4. Kirim ke CML Model Serving
        model_output = call_cml_model(feature_row)

        # 5. Panggil ChatGPT untuk explanation
        ai_explanation = generate_ai_explanation(claim_id, model_output, feature_row)

        # 6. Bentuk record akhir dan tulis ke HBase
        result_record = {
            "claim_id": claim_id,
            "features": feature_row,
            "model_output": model_output,
            "ai_explanation": ai_explanation,
        }

        write_to_hbase(claim_id, result_record)

        # 7. Return ke frontend
        return jsonify(result_record), 200

    except Exception as e:
        # Biar kelihatan di log systemd
        print(f"[ERROR] score_claim({claim_id}): {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # Listen di 0.0.0.0:2222 sesuai request
    app.run(host="0.0.0.0", port=2222, debug=False)