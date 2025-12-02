#!/usr/bin/env python3
from flask import Flask, render_template, request, redirect, session, jsonify
import requests
import mysql.connector
import math
from datetime import timedelta

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

# SECRET KEY (HARUS SATU, TIDAK BOLEH DOUBLE)
app.secret_key = "supersecretkey_approval_ui"

# FIX SESSION DROPPING
app.config.update(
    PERMANENT_SESSION_LIFETIME=timedelta(hours=1),
    SESSION_COOKIE_NAME="fraud_approval_session",
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SECURE=False  # <- kamu pakai HTTP, jadi FALSE
)
# ----------------------------
# Custom Filter: Currency IDR
# ----------------------------
def rupiah(n):
    if n is None:
        return "-"
    return "Rp {:,.2f}".format(float(n)).replace(",", "X").replace(".", ",").replace("X", ".")
app.jinja_env.filters["rupiah"] = rupiah


# ============================================================
# LOGIN
# ============================================================
@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        user = request.form.get("username")
        pw = request.form.get("password")

        if user == "aris" and pw == "Admin123":
            session.permanent = True
            session["user"] = user     # cukup user saja
            return redirect("/dashboard")

        return render_template("login.html", error="Invalid username or password")

    return render_template("login.html")

# ============================================================
# DASHBOARD — PAGINATION 20 ROWS
# ============================================================
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
        SELECT claim_id,
               patient_name,
               visit_date,
               visit_type,
               department,
               total_claim_amount,
               status
        FROM claim_header 
        WHERE status='pending'
        ORDER BY claim_id DESC
        LIMIT {limit} OFFSET {offset}
    """)

    claims = cur.fetchall()
    cur.close()
    conn.close()

    return render_template(
        "dashboard.html",
        claims=claims,
        page=page,
        total_pages=total_pages
    )


# ============================================================
# FETCH CLAIM DETAIL (AJAX SEARCH)
# ============================================================
@app.route("/api/get_claim", methods=["POST"])
def api_get_claim():
    data = request.json
    claim_id = data.get("claim_id")

    conn = db()
    cur = conn.cursor(dictionary=True)
    cur.execute("SELECT * FROM claim_header WHERE claim_id=%s", (claim_id,))
    header = cur.fetchone()
    cur.close()
    conn.close()

    if not header:
        return jsonify({"error": "Claim not found"}), 404

    # Fetch scoring
    scoring_raw = requests.get(f"{BACKEND_URL}{claim_id}", verify=False).json()
    model_output = scoring_raw.get("model_output", {})

    return jsonify({
        "claim_id": claim_id,
        "header": header,
        "model_output": model_output
    })


# ============================================================
# REVIEW PAGE — FULL DETAIL
# ============================================================
@app.route("/review/<int:claim_id>")
def review(claim_id):
    if "user" not in session:
        return redirect("/")

    conn = db()
    cur = conn.cursor(dictionary=True)

    cur.execute("SELECT * FROM claim_header WHERE claim_id = %s", (claim_id,))
    header = cur.fetchone()

    cur.execute("SELECT * FROM claim_diagnosis WHERE claim_id = %s", (claim_id,))
    diagnosis = cur.fetchall()

    cur.execute("SELECT * FROM claim_procedure WHERE claim_id = %s", (claim_id,))
    procedures = cur.fetchall()

    cur.execute("SELECT * FROM claim_drug WHERE claim_id = %s", (claim_id,))
    drugs = cur.fetchall()

    cur.execute("SELECT * FROM claim_vitamin WHERE claim_id = %s", (claim_id,))
    vitamins = cur.fetchall()

    cur.close()
    conn.close()

    # Fetch scoring correct model structure
    scoring_raw = requests.get(f"{BACKEND_URL}{claim_id}", verify=False).json()


    scoring = {
        "ai_explanation": scoring_raw.get("ai_explanation"),
        "model_output": scoring_raw.get("model_output", {}),
    }

    return render_template(
        "review.html",
        header=header,
        diagnosis=diagnosis,
        procedures=procedures,
        drugs=drugs,
        vitamins=vitamins,
        scoring=scoring
    )


# ============================================================
# UPDATE STATUS API
# ============================================================
@app.route("/api/update_status/<int:claim_id>", methods=["POST"])
def update_status(claim_id):
    if "user" not in session:
        return jsonify({"error": "Unauthorized"}), 401

    new_status = request.json.get("status")

    conn = db()
    cur = conn.cursor()
    cur.execute("UPDATE claim_header SET status=%s WHERE claim_id=%s", (new_status, claim_id))
    conn.commit()
    cur.close()
    conn.close()

    return jsonify({"success": True})


# ============================================================
# LOGOUT
# ============================================================
@app.route("/logout")
def logout():
    session.clear()
    return redirect("/")


# ============================================================
# RUN
# ============================================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=2223)