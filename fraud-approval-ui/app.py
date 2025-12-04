#!/usr/bin/env python3
"""
Updated app.py with API endpoints for React UI
Replace your existing app.py with this file
"""
from flask import Flask, render_template, request, redirect, session, jsonify
from flask_cors import CORS
import requests
import mysql.connector
import math
from datetime import timedelta

APP_SECRET = "supersecret"
BACKEND_URL = "http://127.0.0.1:2222/score/"

MYSQL_CONFIG = {
    "host": "cdpmsi.tomodachis.org",
    "user": "cloudera",
    "password": "T1ku$H1t4m",
    "database": "claimdb",
}

def db():
    return mysql.connector.connect(**MYSQL_CONFIG)

app = Flask(__name__)

# CORS for React frontend
CORS(app, origins=["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:5173", "http://127.0.0.1:5173"])

app.secret_key = "supersecretkey_approval_ui"
app.config.update(
    PERMANENT_SESSION_LIFETIME=timedelta(hours=1),
    SESSION_COOKIE_NAME="fraud_approval_session",
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SECURE=False
)

def rupiah(n):
    if n is None:
        return "-"
    return "Rp {:,.2f}".format(float(n)).replace(",", "X").replace(".", ",").replace("X", ".")
app.jinja_env.filters["rupiah"] = rupiah


def normalize_model_output(m):
    if not isinstance(m, dict):
        m = {}

    if "error" in m:
        return {
            "fraud_score": 0,
            "fraud_probability": "0%",
            "risk_level": "ERROR",
            "risk_color": "gray",
            "clinical_compatibility": {
                "procedure_compatible": False,
                "drug_compatible": False,
                "vitamin_compatible": False,
                "overall_compatible": False,
                "details": {}
            },
            "features": {
                "mismatch_count": 0,
                "cost_anomaly": 0,
                "cost_anomaly_score": 0,
                "total_claim": 0,
                "total_claim_amount": 0
            },
            "explanation": f"Model Error: {m.get('error', 'Unknown error')}",
            "recommendation": "Manual review required",
            "confidence": 0.5,
            "fraud_flag": 0,
            "top_risk_factors": [],
            "claim_id": m.get("claim_id"),
        }

    features = m.get("features", {})
    cost_anomaly = features.get("cost_anomaly", 0)
    total_claim = features.get("total_claim", 0)
    mismatch_count = features.get("mismatch_count", 0)
    
    fraud_score = m.get("fraud_score", 0)
    fraud_probability = m.get("fraud_probability", "0%")
    
    if fraud_score > 0.8 or fraud_score < 0.2:
        confidence = 0.9
    elif fraud_score > 0.6 or fraud_score < 0.4:
        confidence = 0.7
    else:
        confidence = 0.5
    
    procedure_compatible = mismatch_count == 0 or "procedure" not in m.get("explanation", "").lower()
    drug_compatible = mismatch_count == 0 or "obat" not in m.get("explanation", "").lower()
    vitamin_compatible = mismatch_count == 0 or "vitamin" not in m.get("explanation", "").lower()
    
    top_risk_factors = []
    explanation = m.get("explanation", "")
    
    if "Ketidaksesuaian" in explanation:
        top_risk_factors.append({"interpretation": "Clinical mismatch detected", "importance": 0.8})
    
    if cost_anomaly >= 3:
        top_risk_factors.append({"interpretation": "High cost anomaly", "importance": 0.9})
    elif cost_anomaly >= 2:
        top_risk_factors.append({"interpretation": "Medium cost anomaly", "importance": 0.6})
    
    if fraud_score > 0.7:
        top_risk_factors.append({"interpretation": "High fraud probability", "importance": 0.95})
    
    return {
        "fraud_score": fraud_score,
        "fraud_probability": fraud_probability,
        "risk_level": m.get("risk_level", "UNKNOWN"),
        "risk_color": m.get("risk_color", "gray"),
        "clinical_compatibility": {
            "procedure_compatible": procedure_compatible,
            "drug_compatible": drug_compatible,
            "vitamin_compatible": vitamin_compatible,
            "overall_compatible": mismatch_count == 0,
            "details": {}
        },
        "features": {
            "mismatch_count": mismatch_count,
            "cost_anomaly": cost_anomaly,
            "cost_anomaly_score": cost_anomaly,
            "total_claim": total_claim,
            "total_claim_amount": total_claim
        },
        "explanation": explanation,
        "recommendation": m.get("recommendation", "-"),
        "confidence": confidence,
        "fraud_flag": m.get("fraud_flag", 0),
        "top_risk_factors": top_risk_factors,
        "claim_id": m.get("claim_id"),
    }


# ============================================================
# API ENDPOINTS FOR REACT UI
# ============================================================

@app.route("/api/claims", methods=["GET"])
def api_list_claims():
    page = int(request.args.get("page", 1))
    limit = int(request.args.get("limit", 20))
    offset = (page - 1) * limit

    conn = db()
    cur = conn.cursor(dictionary=True)

    cur.execute("SELECT COUNT(*) AS total FROM claim_header WHERE status='pending'")
    total = cur.fetchone()["total"]
    total_pages = math.ceil(total / limit)

    cur.execute(f"""
        SELECT claim_id, patient_name, visit_date, visit_type,
               department, total_claim_amount, status
        FROM claim_header 
        WHERE status='pending'
        ORDER BY claim_id DESC
        LIMIT {limit} OFFSET {offset}
    """)
    claims = cur.fetchall()
    
    for claim in claims:
        if claim.get('visit_date'):
            claim['visit_date'] = str(claim['visit_date'])

    cur.close()
    conn.close()

    return jsonify({
        "claims": claims,
        "total": total,
        "page": page,
        "total_pages": total_pages
    })


@app.route("/api/claim/<int:claim_id>", methods=["GET"])
def api_claim_detail(claim_id):
    conn = db()
    cur = conn.cursor(dictionary=True)

    cur.execute("SELECT * FROM claim_header WHERE claim_id=%s", (claim_id,))
    header = cur.fetchone()
    
    if not header:
        cur.close()
        conn.close()
        return jsonify({"error": "Claim not found"}), 404

    for key in ['visit_date', 'patient_dob']:
        if header.get(key):
            header[key] = str(header[key])

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

    return jsonify({
        "header": header,
        "diagnosis": diagnosis,
        "procedures": procedures,
        "drugs": drugs,
        "vitamins": vitamins
    })


@app.route("/api/score/<int:claim_id>", methods=["GET"])
def api_score(claim_id):
    try:
        scoring_raw = requests.get(f"{BACKEND_URL}{claim_id}", verify=False).json()
        model_output = normalize_model_output(scoring_raw.get("model_output", {}))
        return jsonify({
            "model_output": model_output,
            "ai_explanation": scoring_raw.get("ai_explanation")
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/update_status/<int:claim_id>", methods=["POST", "OPTIONS"])
def api_update_status(claim_id):
    if request.method == "OPTIONS":
        return jsonify({"ok": True})
    
    new_status = request.json.get("status")
    
    if not new_status:
        return jsonify({"error": "Status required"}), 400

    conn = db()
    cur = conn.cursor()
    cur.execute("UPDATE claim_header SET status=%s WHERE claim_id=%s",
                (new_status, claim_id))
    conn.commit()
    cur.close()
    conn.close()

    return jsonify({"success": True, "claim_id": claim_id, "status": new_status})


# ============================================================
# ORIGINAL FLASK UI ROUTES (backward compatibility)
# ============================================================

@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        user = request.form.get("username")
        pw = request.form.get("password")
        if user == "aris" and pw == "Admin123":
            session.permanent = True
            session["user"] = user
            return redirect("/dashboard")
        return render_template("login.html", error="Invalid username or password")
    return render_template("login.html")


@app.route("/dashboard")
def dashboard():
    if "user" not in session:
        return redirect("/")
    page = int(request.args.get("page", 1))
    limit = 20
    offset = (page - 1) * limit
    conn = db()
    cur = conn.cursor(dictionary=True)
    cur.execute("SELECT COUNT(*) AS total FROM claim_header WHERE status='pending'")
    total = cur.fetchone()["total"]
    total_pages = math.ceil(total / limit)
    cur.execute(f"""
        SELECT claim_id, patient_name, visit_date, visit_type,
               department, total_claim_amount, status
        FROM claim_header 
        WHERE status='pending'
        ORDER BY claim_id DESC
        LIMIT {limit} OFFSET {offset}
    """)
    claims = cur.fetchall()
    cur.close()
    conn.close()
    return render_template("dashboard.html", claims=claims, page=page, total_pages=total_pages)


@app.route("/api/get_claim", methods=["POST"])
def api_get_claim():
    claim_id = request.json.get("claim_id")
    conn = db()
    cur = conn.cursor(dictionary=True)
    cur.execute("SELECT * FROM claim_header WHERE claim_id=%s", (claim_id,))
    header = cur.fetchone()
    cur.close()
    conn.close()
    if not header:
        return jsonify({"error": "Claim not found"}), 404
    scoring_raw = requests.get(f"{BACKEND_URL}{claim_id}", verify=False).json()
    model_output = normalize_model_output(scoring_raw.get("model_output", {}))
    return jsonify({"claim_id": claim_id, "header": header, "model_output": model_output})


@app.route("/review/<int:claim_id>")
def review(claim_id):
    if "user" not in session:
        return redirect("/")
    conn = db()
    cur = conn.cursor(dictionary=True)
    cur.execute("SELECT * FROM claim_header WHERE claim_id=%s", (claim_id,))
    header = cur.fetchone()
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
    scoring_raw = requests.get(f"{BACKEND_URL}{claim_id}", verify=False).json()
    model_output = normalize_model_output(scoring_raw.get("model_output", {}))
    scoring = {"ai_explanation": scoring_raw.get("ai_explanation"), "model_output": model_output}
    return render_template("review.html", header=header, diagnosis=diagnosis, 
                          procedures=procedures, drugs=drugs, vitamins=vitamins, scoring=scoring)


@app.route("/logout")
def logout():
    session.clear()
    return redirect("/")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=2223)
