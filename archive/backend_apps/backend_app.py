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

CML_MODEL_URL = os.getenv("CML_MODEL_URL")
CML_MODEL_TOKEN = os.getenv("CML_MODEL_TOKEN")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

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

    # Procedures
    cur.execute("""
        SELECT icd9_code
        FROM claim_procedure
        WHERE claim_id=%s
    """, (claim_id,))
    procedures = [row["icd9_code"] for row in cur.fetchall()]

    # Drugs
    cur.execute("""
        SELECT drug_code
        FROM claim_drug
        WHERE claim_id=%s
    """, (claim_id,))
    drugs = [row["drug_code"] for row in cur.fetchall()]

    # Vitamins
    cur.execute("""
        SELECT vitamin_name
        FROM claim_vitamin
        WHERE claim_id=%s
    """, (claim_id,))
    vitamins = [row["vitamin_name"] for row in cur.fetchall()]

    cur.close()
    conn.close()

    raw_claim = {
        "claim_id": str(header["claim_id"]),
        "patient_dob": str(header["patient_dob"]),
        "visit_date": str(header["visit_date"]),
        "visit_type": str(header["visit_type"]),
        "department": str(header["department"]),
        "icd10_primary_code": diag["icd10_code"] if diag else "UNKNOWN",
        "procedures": procedures or [],
        "drugs": drugs or [],
        "vitamins": vitamins or [],
        "total_procedure_cost": int(float(header["total_procedure_cost"] or 0)),
        "total_drug_cost": int(float(header["total_drug_cost"] or 0)),
        "total_vitamin_cost": int(float(header["total_vitamin_cost"] or 0)),
        "total_claim_amount": int(float(header["total_claim_amount"] or 0)),
        "patient_frequency_risk": 2
    }

    return raw_claim


# =========================================================
# CALL CML MODEL
# =========================================================
def call_cml_model(raw_claim):
    if not CML_MODEL_URL:
        return {"error": "cml_url_not_set"}
    
    payload = {
        "request": {
            "claims": [raw_claim]
        }
    }
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {CML_MODEL_TOKEN}"
    }

    try:
        resp = requests.post(
            CML_MODEL_URL,
            json=payload,
            headers=headers,
            timeout=30,
            verify=False
        )
        
        if resp.status_code == 200:
            result = resp.json()
            
            if isinstance(result, dict):
                if "response" in result:
                    inner = result["response"]
                    if isinstance(inner, dict):
                        if inner.get("status") == "success" and "results" in inner:
                            return inner["results"][0]
                
                elif result.get("status") == "success" and "results" in result:
                    return result["results"][0]
                
                elif "fraud_flag" in result:
                    return result
            
            return {"error": "unexpected_format", "response": result}
                
        else:
            return {"error": "http_error", "status_code": resp.status_code}
            
    except Exception as e:
        return {"error": "request_failed", "details": str(e)}


# =========================================================
# SIMPLE EXPLANATION GENERATOR
# =========================================================
def generate_simple_explanation(claim_id, model_output):
    """Generate explanation tanpa OpenAI API"""
    
    fraud_score = model_output.get("fraud_score", 0)
    risk_level = model_output.get("risk_level", "UNKNOWN")
    explanation = model_output.get("explanation", "")
    recommendation = model_output.get("recommendation", "")
    
    features = model_output.get("features", {})
    total_claim = features.get("total_claim", 0)
    mismatch_count = features.get("mismatch_count", 0)
    cost_anomaly = features.get("cost_anomaly", 1)
    
    # Map emoji to text
    emoji_map = {
        "🟢": "MINIMAL",
        "🟡": "MODERATE", 
        "🟠": "MEDIUM",
        "🔴": "HIGH"
    }
    
    # Clean explanation from emoji
    clean_explanation = explanation
    for emoji, text in emoji_map.items():
        clean_explanation = clean_explanation.replace(emoji, "")
    
    # Risk level mapping
    risk_text = {
        "MINIMAL": "risiko minimal",
        "MODERATE": "risiko sedang", 
        "MEDIUM": "risiko menengah",
        "HIGH": "risiko tinggi",
        "LOW": "risiko rendah"
    }.get(risk_level, "risiko tidak diketahui")
    
    # Generate simple explanation
    explanation_parts = []
    
    explanation_parts.append(f"**Analisis Klaim ID {claim_id}**")
    explanation_parts.append(f"Skor Fraud: {fraud_score:.1%} ({risk_text})")
    explanation_parts.append(f"Total Klaim: Rp {total_claim:,}")
    
    if mismatch_count > 0:
        explanation_parts.append(f"Terdapat {mismatch_count} ketidaksesuaian klinis")
    
    if cost_anomaly > 2:
        explanation_parts.append(f"Anomali biaya: tingkat {cost_anomaly}/4")
    
    explanation_parts.append(f"**Temuan:** {clean_explanation.strip()}")
    explanation_parts.append(f"**Rekomendasi:** {recommendation}")
    
    # Add action recommendation
    if fraud_score > 0.7:
        explanation_parts.append("**Tindakan:** Klaim perlu investigasi mendalam")
    elif fraud_score > 0.4:
        explanation_parts.append("**Tindakan:** Verifikasi manual diperlukan")
    else:
        explanation_parts.append("**Tindakan:** Klaim dapat diproses normal")
    
    return "\n\n".join(explanation_parts)


# =========================================================
# GPT EXPLANATION (OPTIONAL)
# =========================================================
def generate_explanation(claim_id, model_output):
    """Generate explanation dengan OpenAI API (optional)"""
    
    # Jika OpenAI API tidak dikonfigurasi, gunakan simple explanation
    if not OPENAI_API_KEY or not OPENAI_MODEL:
        return generate_simple_explanation(claim_id, model_output)
    
    # Check if model_output is valid
    if not isinstance(model_output, dict) or "error" in model_output:
        return f"Tidak bisa generate explanation: {model_output.get('error', 'Model output tidak valid')}"
    
    # Prepare data
    fraud_score = model_output.get("fraud_score", 0)
    risk_level = model_output.get("risk_level", "UNKNOWN")
    explanation = model_output.get("explanation", "")
    recommendation = model_output.get("recommendation", "")
    
    features = model_output.get("features", {})
    total_claim = features.get("total_claim", 0)
    
    # Clean emoji from explanation
    clean_explanation = explanation
    emojis = ["🟢", "🟡", "🟠", "🔴", "✅", "⚠️", "🚫", "⛔"]
    for emoji in emojis:
        clean_explanation = clean_explanation.replace(emoji, "")
    
    prompt = f"""Analisis klaim kesehatan untuk BPJS:

ID Klaim: {claim_id}
Skor Fraud: {fraud_score:.1%}
Tingkat Risiko: {risk_level}
Total Nilai Klaim: Rp {total_claim:,}

Hasil Deteksi:
{clean_explanation.strip()}

Rekomendasi Sistem: 
{recommendation}

Buat analisis profesional dalam bahasa Indonesia untuk tim reviewer.
Fokus pada:
1. Interpretasi hasil fraud detection
2. Poin-poin kritis yang perlu diverifikasi
3. Rekomendasi tindakan (approve/reject/verifikasi lebih lanjut)

Format: 2-3 paragraf, jelas dan mudah dipahami."""

    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": OPENAI_MODEL,
                "messages": [
                    {
                        "role": "system", 
                        "content": "Anda adalah analis fraud di BPJS dengan pengalaman 10 tahun."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                "temperature": 0.7,
                "max_tokens": 350
            },
            timeout=15
        )

        if response.status_code == 200:
            result = response.json()
            return result["choices"][0]["message"]["content"]
        else:
            # Fallback to simple explanation jika OpenAI error
            error_msg = f"OpenAI API error {response.status_code}"
            print(f"OpenAI Error: {error_msg}")
            return generate_simple_explanation(claim_id, model_output)
            
    except Exception as e:
        # Fallback to simple explanation
        print(f"OpenAI Exception: {str(e)}")
        return generate_simple_explanation(claim_id, model_output)


# =========================================================
# API ENDPOINTS
# =========================================================
@app.route("/score/<int:claim_id>", methods=["GET"])
def score(claim_id):
    raw_claim = load_raw_claim(claim_id)
    if not raw_claim:
        return jsonify({"error": "Claim not found"}), 404
    
    model_output = call_cml_model(raw_claim)
    
    # Generate explanation
    if isinstance(model_output, dict) and "error" not in model_output:
        explanation = generate_explanation(claim_id, model_output)
    else:
        explanation = f"Model prediction failed: {model_output.get('error', 'Unknown error')}"
        if "details" in model_output:
            explanation += f" - {model_output['details']}"
    
    response = {
        "claim_id": claim_id,
        "raw_input": clean_decimal(raw_claim),
        "model_output": clean_decimal(model_output),
        "ai_explanation": explanation
    }
    
    return jsonify(response)


@app.route("/simple_score/<int:claim_id>", methods=["GET"])
def simple_score(claim_id):
    """Endpoint tanpa OpenAI API"""
    raw_claim = load_raw_claim(claim_id)
    if not raw_claim:
        return jsonify({"error": "Claim not found"}), 404
    
    model_output = call_cml_model(raw_claim)
    
    # Generate simple explanation (no OpenAI)
    if isinstance(model_output, dict) and "error" not in model_output:
        explanation = generate_simple_explanation(claim_id, model_output)
    else:
        explanation = f"Model prediction failed: {model_output.get('error', 'Unknown error')}"
    
    response = {
        "claim_id": claim_id,
        "raw_input": clean_decimal(raw_claim),
        "model_output": clean_decimal(model_output),
        "explanation": explanation
    }
    
    return jsonify(response)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "cml_configured": bool(CML_MODEL_URL),
        "openai_configured": bool(OPENAI_API_KEY and OPENAI_MODEL)
    })


# =========================================================
# RUN
# =========================================================
if __name__ == "__main__":
    print("=" * 60)
    print("FRAUD DETECTION BACKEND API")
    print("=" * 60)
    print(f"CML Model: {'✓ Configured' if CML_MODEL_URL else '✗ Not configured'}")
    print(f"OpenAI API: {'✓ Configured' if OPENAI_API_KEY else '✗ Not configured'}")
    print(f"OpenAI Model: {OPENAI_MODEL}")
    print("=" * 60)
    app.run(host="0.0.0.0", port=2222, debug=False)