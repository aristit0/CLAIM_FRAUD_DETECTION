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
    "host": os.getenv("MYSQL_HOST", "cdpmsi.tomodachis.org"),
    "user": os.getenv("MYSQL_USER", "cloudera"),
    "password": os.getenv("MYSQL_PASSWORD", "T1ku$H1t4m"),
    "database": os.getenv("MYSQL_DB", "claimdb"),
}

CML_MODEL_URL = os.getenv("CML_MODEL_URL", "https://modelservice.cloudera-ai.apps.ds.tomodachis.org/model")
CML_ACCESS_KEY = os.getenv("CML_MODEL_ACCESS_KEY", "m87e09mpmbn2eajnidbb34rx3wgeuyjd")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

app = Flask(__name__)


# =========================================================
# HELPERS
# =========================================================
def clean_decimal(obj):
    """Convert Decimal to float recursively"""
    if isinstance(obj, Decimal):
        return float(obj)
    if isinstance(obj, dict):
        return {k: clean_decimal(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clean_decimal(v) for v in obj]
    return obj


def to_float(x):
    """Safe conversion to float"""
    return float(x) if isinstance(x, (int, float, Decimal)) else 0.0


def db():
    """Database connection"""
    return mysql.connector.connect(**MYSQL_CONFIG)


# =========================================================
# LOAD RAW CLAIM FROM DATABASE
# =========================================================
def load_raw_claim(claim_id):
    """Load complete claim data for model inference"""
    conn = db()
    cur = conn.cursor(dictionary=True)

    # Header
    cur.execute("""
        SELECT claim_id, patient_nik, patient_name, patient_dob, 
               visit_date, visit_type, department,
               total_procedure_cost, total_drug_cost, total_vitamin_cost,
               total_claim_amount, status
        FROM claim_header
        WHERE claim_id=%s
    """, (claim_id,))
    header = cur.fetchone()
    
    if not header:
        cur.close()
        conn.close()
        return None

    # Primary diagnosis
    cur.execute("""
        SELECT icd10_code, icd10_description
        FROM claim_diagnosis
        WHERE claim_id=%s AND is_primary=1
        LIMIT 1
    """, (claim_id,))
    diag = cur.fetchone()

    # Procedures (arrays of codes)
    cur.execute("""
        SELECT icd9_code
        FROM claim_procedure
        WHERE claim_id=%s
    """, (claim_id,))
    procedures = [row["icd9_code"] for row in cur.fetchall()]

    # Drugs (arrays of codes)
    cur.execute("""
        SELECT drug_code
        FROM claim_drug
        WHERE claim_id=%s
    """, (claim_id,))
    drugs = [row["drug_code"] for row in cur.fetchall()]

    # Vitamins (arrays of names)
    cur.execute("""
        SELECT vitamin_name
        FROM claim_vitamin
        WHERE claim_id=%s
    """, (claim_id,))
    vitamins = [row["vitamin_name"] for row in cur.fetchall()]

    cur.close()
    conn.close()

    # Build raw record for model
    raw = {
        "claim_id": str(claim_id),
        "patient_dob": str(header["patient_dob"]),
        "visit_date": str(header["visit_date"]),
        "visit_type": header["visit_type"],
        "department": header["department"],
        "icd10_primary_code": diag["icd10_code"] if diag else "UNKNOWN",
        "procedures": procedures,
        "drugs": drugs,
        "vitamins": vitamins,
        "total_procedure_cost": to_float(header["total_procedure_cost"]),
        "total_drug_cost": to_float(header["total_drug_cost"]),
        "total_vitamin_cost": to_float(header["total_vitamin_cost"]),
        "total_claim_amount": to_float(header["total_claim_amount"]),
    }

    return clean_decimal(raw)


# =========================================================
# CALL CML MODEL
# =========================================================
def call_cml_model(raw_claim):
    """
    Call CML model endpoint with proper structure.
    Model expects: {"raw_records": [{...}]}
    Model returns: {"status": "success", "results": [{...}], ...}
    """
    
    if not CML_MODEL_URL:
        return {"error": "CML_MODEL_URL not configured"}

    # Build request URL with access key
    url = f"{CML_MODEL_URL}?accessKey={CML_ACCESS_KEY}"
    
    # Payload structure expected by model
    payload = {
        "raw_records": [clean_decimal(raw_claim)]
    }
    
    headers = {
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(
            url,
            json=payload,
            headers=headers,
            verify=False,
            timeout=30
        )
        
        if response.status_code != 200:
            return {
                "error": "model_http_error",
                "status_code": response.status_code,
                "message": response.text
            }
        
        result = response.json()
        
        # Check if successful response
        if result.get("status") == "success" and "results" in result:
            # Extract first result (single claim)
            model_result = result["results"][0]
            
            # Add model metadata
            model_result["model_version"] = result.get("model_version", "unknown")
            model_result["model_info"] = result.get("model_info", {})
            
            return clean_decimal(model_result)
        
        # Handle error response
        if result.get("status") == "error":
            return {
                "error": "model_error",
                "details": result.get("error", "Unknown error"),
                "traceback": result.get("traceback", "")
            }
        
        # Unknown format
        return {
            "error": "unexpected_response_format",
            "raw_response": result
        }
    
    except requests.exceptions.Timeout:
        return {"error": "model_timeout", "message": "Model request timed out after 30s"}
    
    except requests.exceptions.ConnectionError:
        return {"error": "model_connection_error", "message": "Cannot connect to model endpoint"}
    
    except Exception as e:
        return {"error": "model_exception", "message": str(e)}


# =========================================================
# GENERATE AI EXPLANATION (IMPROVED)
# =========================================================
def generate_ai_explanation(claim_id, raw_claim, model_output):
    """
    Generate concise, actionable AI explanation.
    Focus on clarity and brevity for claim reviewers.
    """
    
    if not OPENAI_API_KEY:
        return "AI explanation unavailable (OpenAI API key not configured)"
    
    # Extract key information
    fraud_score = model_output.get("fraud_score", 0)
    fraud_prob = model_output.get("fraud_probability", "0%")
    risk_level = model_output.get("risk_level", "UNKNOWN")
    explanation = model_output.get("explanation", "No explanation provided")
    
    # Clinical compatibility
    clinical = model_output.get("clinical_compatibility", {})
    compat_issues = []
    if not clinical.get("procedure_compatible", True):
        compat_issues.append("prosedur tidak sesuai diagnosis")
    if not clinical.get("drug_compatible", True):
        compat_issues.append("obat tidak sesuai diagnosis")
    if not clinical.get("vitamin_compatible", True):
        compat_issues.append("vitamin tidak sesuai diagnosis")
    
    # Features
    features = model_output.get("features", {})
    mismatch_count = features.get("mismatch_count", 0)
    cost_anomaly = features.get("cost_anomaly_score", 0)
    total_claim = features.get("total_claim_amount", 0)
    
    # Build context for GPT
    context = f"""
Klaim #{claim_id} - {raw_claim.get('department', 'N/A')}
Diagnosis: {raw_claim.get('icd10_primary_code', 'N/A')}
Total Klaim: Rp {total_claim:,.0f}

Risk Level: {risk_level}
Fraud Score: {fraud_score:.2f} ({fraud_prob})

Temuan:
- Clinical mismatch: {mismatch_count}
- Cost anomaly level: {cost_anomaly}/4
- Ketidaksesuaian: {', '.join(compat_issues) if compat_issues else 'tidak ada'}

Model explanation: {explanation}
"""

    prompt = f"""Kamu adalah AI assistant untuk tim reviewer klaim asuransi kesehatan.

Analisis klaim berikut dan berikan penjelasan SINGKAT (maksimal 3 kalimat):

{context}

Format:
1. Kesimpulan risiko (1 kalimat)
2. Alasan utama (1 kalimat)
3. Rekomendasi tindakan (1 kalimat)

Gunakan bahasa yang jelas, hindari jargon teknis."""

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
                        "content": "Kamu adalah AI assistant untuk reviewer klaim kesehatan. Berikan penjelasan singkat, jelas, dan actionable."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.3,
                "max_tokens": 200
            },
            timeout=15
        )
        
        if response.status_code == 200:
            ai_response = response.json()
            return ai_response["choices"][0]["message"]["content"].strip()
        else:
            return f"AI explanation error (HTTP {response.status_code})"
    
    except Exception as e:
        return f"AI explanation unavailable: {str(e)}"


# =========================================================
# MAIN SCORING ENDPOINT
# =========================================================
@app.route("/score/<int:claim_id>", methods=["GET"])
def score_claim(claim_id):
    """
    Main endpoint for fraud scoring.
    
    Returns:
    {
        "claim_id": 123,
        "status": "success",
        "raw_claim": {...},
        "model_output": {...},
        "ai_explanation": "...",
        "timestamp": "2024-01-20T10:30:00"
    }
    """
    
    # Load claim from database
    raw_claim = load_raw_claim(claim_id)
    
    if not raw_claim:
        return jsonify({
            "status": "error",
            "error": "claim_not_found",
            "message": f"Claim ID {claim_id} not found in database"
        }), 404
    
    # Call model
    model_output = call_cml_model(raw_claim)
    
    # Check for model errors
    if "error" in model_output:
        return jsonify({
            "status": "error",
            "claim_id": claim_id,
            "raw_claim": raw_claim,
            "model_output": model_output,
            "ai_explanation": "Model error - cannot generate explanation",
            "timestamp": datetime.now().isoformat()
        }), 500
    
    # Generate AI explanation
    ai_explanation = generate_ai_explanation(claim_id, raw_claim, model_output)
    
    # Build response
    response = {
        "status": "success",
        "claim_id": claim_id,
        "raw_claim": raw_claim,
        "model_output": model_output,
        "ai_explanation": ai_explanation,
        "timestamp": datetime.now().isoformat()
    }
    
    return jsonify(clean_decimal(response))


# =========================================================
# HEALTH CHECK
# =========================================================
@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    
    # Test database
    db_status = "healthy"
    try:
        conn = db()
        conn.close()
    except Exception as e:
        db_status = f"unhealthy: {str(e)}"
    
    # Test model
    model_status = "unknown"
    if CML_MODEL_URL:
        model_status = "configured"
    else:
        model_status = "not_configured"
    
    return jsonify({
        "status": "healthy",
        "database": db_status,
        "model": model_status,
        "openai": "configured" if OPENAI_API_KEY else "not_configured",
        "timestamp": datetime.now().isoformat()
    })


# =========================================================
# BATCH SCORING (OPTIONAL)
# =========================================================
@app.route("/score/batch", methods=["POST"])
def batch_score():
    """
    Batch scoring endpoint.
    
    Request:
    {
        "claim_ids": [123, 456, 789]
    }
    
    Returns:
    {
        "status": "success",
        "total": 3,
        "results": [...]
    }
    """
    
    data = request.get_json()
    claim_ids = data.get("claim_ids", [])
    
    if not claim_ids:
        return jsonify({
            "status": "error",
            "error": "no_claim_ids",
            "message": "Please provide claim_ids array"
        }), 400
    
    results = []
    
    for claim_id in claim_ids:
        raw_claim = load_raw_claim(claim_id)
        
        if not raw_claim:
            results.append({
                "claim_id": claim_id,
                "status": "error",
                "error": "claim_not_found"
            })
            continue
        
        model_output = call_cml_model(raw_claim)
        ai_explanation = generate_ai_explanation(claim_id, raw_claim, model_output)
        
        results.append({
            "claim_id": claim_id,
            "status": "success" if "error" not in model_output else "error",
            "model_output": model_output,
            "ai_explanation": ai_explanation
        })
    
    return jsonify({
        "status": "success",
        "total": len(claim_ids),
        "results": clean_decimal(results),
        "timestamp": datetime.now().isoformat()
    })


# =========================================================
# ERROR HANDLERS
# =========================================================
@app.errorhandler(404)
def not_found(e):
    return jsonify({
        "status": "error",
        "error": "not_found",
        "message": str(e)
    }), 404


@app.errorhandler(500)
def internal_error(e):
    return jsonify({
        "status": "error",
        "error": "internal_server_error",
        "message": str(e)
    }), 500


# =========================================================
# RUN
# =========================================================
if __name__ == "__main__":
    print("=" * 60)
    print("FRAUD DETECTION BACKEND API")
    print("=" * 60)
    print(f"Database: {MYSQL_CONFIG['host']}")
    print(f"Model URL: {CML_MODEL_URL}")
    print(f"OpenAI: {'Configured' if OPENAI_API_KEY else 'Not configured'}")
    print("=" * 60)
    
    app.run(host="0.0.0.0", port=2222, debug=False)