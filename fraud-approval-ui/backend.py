#!/usr/bin/env python3
"""
Flask Backend API for Fraud Approval UI
Connects to MySQL and Scoring Backend
Port: 2225
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import mysql.connector
from datetime import datetime, date
from decimal import Decimal
import math

# ============================================================
# CONFIG
# ============================================================
MYSQL_CONFIG = {
    "host": "cdpmsi.tomodachis.org",
    "user": "cloudera",
    "password": "T1ku$H1t4m",
    "database": "claimdb",
}

SCORING_URL = "http://localhost:2222/score/"

app = Flask(__name__)
CORS(app)

def db():
    return mysql.connector.connect(**MYSQL_CONFIG)

def serialize(obj):
    """Convert MySQL types to JSON serializable"""
    if isinstance(obj, dict):
        return {k: serialize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [serialize(i) for i in obj]
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, Decimal):
        return float(obj)
    return obj


def normalize_model_output(m):
    """Normalize model output for UI"""
    if not isinstance(m, dict):
        m = {}
    
    if "error" in m:
        return {
            "fraud_score": 0, "fraud_probability": "0%", "risk_level": "ERROR",
            "clinical_compatibility": {"procedure_compatible": False, "drug_compatible": False, "vitamin_compatible": False},
            "features": {"mismatch_count": 0, "cost_anomaly": 0},
            "explanation": f"Error: {m.get('error')}", "confidence": 0.5, "top_risk_factors": []
        }
    
    features = m.get("features", {})
    fraud_score = m.get("fraud_score", 0)
    mismatch = features.get("mismatch_count", 0)
    cost_anomaly = features.get("cost_anomaly", 0)
    explanation = m.get("explanation", "")
    
    # Calculate confidence
    confidence = 0.9 if fraud_score > 0.8 or fraud_score < 0.2 else 0.7 if fraud_score > 0.6 or fraud_score < 0.4 else 0.5
    
    # Clinical compatibility
    proc_ok = mismatch == 0 or "tindakan" not in explanation.lower()
    drug_ok = mismatch == 0 or "obat" not in explanation.lower()
    vit_ok = mismatch == 0 or "vitamin" not in explanation.lower()
    
    # Risk factors
    risk_factors = []
    if "ketidaksesuaian" in explanation.lower():
        risk_factors.append({"interpretation": "Clinical mismatch detected", "importance": 0.8})
    if cost_anomaly >= 3:
        risk_factors.append({"interpretation": "High cost anomaly", "importance": 0.9})
    elif cost_anomaly >= 2:
        risk_factors.append({"interpretation": "Medium cost anomaly", "importance": 0.6})
    if fraud_score > 0.7:
        risk_factors.append({"interpretation": "High fraud probability", "importance": 0.95})
    
    return {
        "fraud_score": fraud_score,
        "fraud_probability": m.get("fraud_probability", "0%"),
        "risk_level": m.get("risk_level", "UNKNOWN"),
        "clinical_compatibility": {"procedure_compatible": proc_ok, "drug_compatible": drug_ok, "vitamin_compatible": vit_ok},
        "features": {"mismatch_count": mismatch, "cost_anomaly": cost_anomaly, "total_claim": features.get("total_claim", 0)},
        "explanation": explanation,
        "recommendation": m.get("recommendation", "-"),
        "confidence": confidence,
        "top_risk_factors": sorted(risk_factors, key=lambda x: -x["importance"])
    }


# ============================================================
# API ENDPOINTS
# ============================================================

@app.route("/api/claims", methods=["GET"])
def list_claims():
    page = int(request.args.get("page", 1))
    limit = int(request.args.get("limit", 20))
    status = request.args.get("status", "pending")
    offset = (page - 1) * limit
    
    conn = db()
    cur = conn.cursor(dictionary=True)
    
    cur.execute("SELECT COUNT(*) AS total FROM claim_header WHERE status=%s", (status,))
    total = cur.fetchone()["total"]
    
    cur.execute("""
        SELECT claim_id, patient_name, visit_date, visit_type, department, total_claim_amount, status
        FROM claim_header WHERE status=%s ORDER BY claim_id DESC LIMIT %s OFFSET %s
    """, (status, limit, offset))
    claims = cur.fetchall()
    
    cur.close()
    conn.close()
    
    return jsonify({
        "claims": serialize(claims),
        "total": total,
        "page": page,
        "total_pages": math.ceil(total / limit)
    })


@app.route("/api/stats", methods=["GET"])
def get_stats():
    conn = db()
    cur = conn.cursor(dictionary=True)
    
    # Status counts
    cur.execute("SELECT status, COUNT(*) as count FROM claim_header GROUP BY status")
    status_counts = {r["status"]: r["count"] for r in cur.fetchall()}
    
    # By department
    cur.execute("""
        SELECT department as name, COUNT(*) as count 
        FROM claim_header WHERE status='pending' 
        GROUP BY department ORDER BY count DESC LIMIT 5
    """)
    by_dept = cur.fetchall()
    
    # Daily trend (last 7 days)
    cur.execute("""
        SELECT DATE(created_at) as date, COUNT(*) as count 
        FROM claim_header 
        WHERE created_at >= DATE_SUB(CURDATE(), INTERVAL 7 DAY)
        GROUP BY DATE(created_at) ORDER BY date
    """)
    daily = cur.fetchall()
    
    # High risk count (approximate - claims with high amount)
    cur.execute("SELECT COUNT(*) as count FROM claim_header WHERE status='pending' AND total_claim_amount > 1000000")
    high_risk = cur.fetchone()["count"]
    
    cur.close()
    conn.close()
    
    return jsonify(serialize({
        "pending": status_counts.get("pending", 0),
        "approved": status_counts.get("approved", 0),
        "declined": status_counts.get("declined", 0),
        "total": sum(status_counts.values()),
        "high_risk": high_risk,
        "by_department": by_dept,
        "daily_trend": [{"date": str(d["date"])[-5:], "count": d["count"]} for d in daily]
    }))


@app.route("/api/claim/<int:claim_id>", methods=["GET"])
def get_claim(claim_id):
    conn = db()
    cur = conn.cursor(dictionary=True)
    
    cur.execute("SELECT * FROM claim_header WHERE claim_id=%s", (claim_id,))
    header = cur.fetchone()
    if not header:
        cur.close()
        conn.close()
        return jsonify({"error": "Not found"}), 404
    
    cur.execute("SELECT * FROM claim_diagnosis WHERE claim_id=%s", (claim_id,))
    diagnosis = cur.fetchall()
    
    cur.execute("SELECT * FROM claim_procedure WHERE claim_id=%s", (claim_id,))
    procedures = cur.fetchall()
    
    cur.execute("SELECT * FROM claim_drug WHERE claim_id=%s", (claim_id,))
    drugs = cur.fetchall()
    
    cur.execute("SELECT * FROM claim_vitamin WHERE claim_id=%s", (claim_id,))
    vitamins = cur.fetchall()
    
    cur.close()
    conn.close()
    
    return jsonify(serialize({
        "header": header,
        "diagnosis": diagnosis,
        "procedures": procedures,
        "drugs": drugs,
        "vitamins": vitamins
    }))


@app.route("/api/score/<int:claim_id>", methods=["GET"])
def get_score(claim_id):
    try:
        resp = requests.get(f"{SCORING_URL}{claim_id}", timeout=10, verify=False)
        data = resp.json()
        return jsonify({
            "model_output": normalize_model_output(data.get("model_output", {})),
            "ai_explanation": data.get("ai_explanation")
        })
    except Exception as e:
        return jsonify({"error": str(e), "model_output": normalize_model_output({"error": str(e)})}), 500


@app.route("/api/update_status/<int:claim_id>", methods=["POST"])
def update_status(claim_id):
    status = request.json.get("status")
    if not status:
        return jsonify({"error": "Status required"}), 400
    
    conn = db()
    cur = conn.cursor()
    cur.execute("UPDATE claim_header SET status=%s, updated_at=NOW() WHERE claim_id=%s", (status, claim_id))
    conn.commit()
    cur.close()
    conn.close()
    
    return jsonify({"success": True, "claim_id": claim_id, "status": status})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=2225, debug=False)
