#!/usr/bin/env python3
from flask import Flask, render_template, request, redirect, session, jsonify
import requests
import mysql.connector

APP_SECRET = "supersecret"
BACKEND_URL = "http://127.0.0.1:2222/score/"

# ---------------------------
# MySQL Connection
# ---------------------------
MYSQL_CONFIG = {
    "host": "cdpmsi.tomodachis.org",
    "user": "cloudera",
    "password": "T1ku$H1t4m",
    "database": "claimdb",
}

def db():
    return mysql.connector.connect(**MYSQL_CONFIG)


app = Flask(__name__)
app.secret_key = APP_SECRET


# ============================================================
# LOGIN PAGE
# ============================================================
@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        user = request.form.get("username")
        pw = request.form.get("password")

        if user == "aris" and pw == "Admin123":
            session["logged_in"] = True
            return redirect("/dashboard")

        return render_template("login.html", error="Invalid username or password")

    return render_template("login.html")


# ============================================================
# DASHBOARD — LIST CLAIMS
# ============================================================
@app.route("/dashboard")
def dashboard():
    if "logged_in" not in session:
        return redirect("/")

    conn = db()
    cur = conn.cursor(dictionary=True)

    cur.execute("""
        SELECT claim_id,
               patient_name,
               visit_date,
               total_claim_amount,
               status
        FROM claim_header
        ORDER BY claim_id DESC
    """)

    claims = cur.fetchall()
    cur.close()
    conn.close()

    return render_template("claims.html", claims=claims)


# ============================================================
# REVIEW PAGE — DETAIL CLAIM
# ============================================================
@app.route("/review/<int:claim_id>")
def review(claim_id):
    if "logged_in" not in session:
        return redirect("/")

    conn = db()
    cur = conn.cursor(dictionary=True)

    # --- Header
    cur.execute("SELECT * FROM claim_header WHERE claim_id = %s", (claim_id,))
    header = cur.fetchone()

    # --- Diagnosis
    cur.execute("""
        SELECT icd10_code, icd10_description
        FROM claim_diagnosis WHERE claim_id = %s
    """, (claim_id,))
    diagnosis = cur.fetchall()

    # --- Procedures
    cur.execute("""
        SELECT icd9_code, icd9_description, quantity, procedure_date
        FROM claim_procedure WHERE claim_id = %s
    """, (claim_id,))
    procedures = cur.fetchall()

    # --- Drugs
    cur.execute("""
        SELECT drug_code, drug_name, dosage, frequency, days, cost
        FROM claim_drug WHERE claim_id = %s
    """, (claim_id,))
    drugs = cur.fetchall()

    # --- Vitamins
    cur.execute("""
        SELECT vitamin_name, dosage, days, cost
        FROM claim_vitamin WHERE claim_id = %s
    """, (claim_id,))
    vitamins = cur.fetchall()

    cur.close()
    conn.close()

    # --- Fraud Scoring dari backend CML
    scoring = requests.get(f"{BACKEND_URL}{claim_id}", verify=False).json()

    return render_template(
        "review.html",
        claim=header,
        diagnosis=diagnosis,
        procedures=procedures,
        drugs=drugs,
        vitamins=vitamins,
        scoring=scoring,
    )


# ============================================================
# API — ACTION APPROVAL
# ============================================================
@app.route("/api/update_status/<int:claim_id>", methods=["POST"])
def update_status(claim_id):
    if "logged_in" not in session:
        return jsonify({"error": "not authenticated"}), 401

    data = request.json
    new_status = data.get("status")

    if not new_status:
        return jsonify({"error": "missing status"}), 400

    conn = db()
    cur = conn.cursor()

    cur.execute("""
        UPDATE claim_header
        SET status = %s
        WHERE claim_id = %s
    """, (new_status, claim_id))

    conn.commit()
    cur.close()
    conn.close()

    return jsonify({"success": True, "claim_id": claim_id, "status": new_status})


# ============================================================
# LOGOUT
# ============================================================
@app.route("/logout")
def logout():
    session.clear()
    return redirect("/")


# ============================================================
# RUN SERVER
# ============================================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=2223)