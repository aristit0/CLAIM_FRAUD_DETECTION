#!/usr/bin/env python3
from flask import Flask, render_template, redirect, request, url_for, session
import mysql.connector
from datetime import timedelta
import math
from datetime import timedelta


app = Flask(__name__)
app.secret_key = "supersecretkey_123"
app.permanent_session_lifetime = timedelta(hours=6)

app.config.update(
    PERMANENT_SESSION_LIFETIME=timedelta(hours=1),
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=False,   # ganti True kalau pakai https
    SESSION_COOKIE_HTTPONLY=True
)


VALID_USER = "aris"
VALID_PASS = "Admin123"

# -----------------------------------
# DB Connection
# -----------------------------------
def get_db():
    return mysql.connector.connect(
        host="cdpmsi.tomodachis.org",
        user="cloudera",
        password="T1ku$H1t4m",
        database="claimdb"
    )


# -----------------------------------
# LOGIN
# -----------------------------------
@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        if request.form.get("username") == VALID_USER and request.form.get("password") == VALID_PASS:
            session.permanent = True
            session["user"] = VALID_USER
            return redirect(url_for("submit_claim"))
        return render_template("login.html", error="Invalid credentials")
    return render_template("login.html")


# -----------------------------------
# Rupiah Formatting
# -----------------------------------
def format_rupiah(x):
    try:
        x = float(x)
        formatted = f"Rp {x:,.2f}"
        formatted = formatted.replace(",", "#").replace(".", ",").replace("#", ".")
        return formatted
    except:
        return x

app.jinja_env.filters["rupiah"] = format_rupiah


# -----------------------------------
# LOGIN REQUIRED DECORATOR
# -----------------------------------
def login_required(f):
    def wrapper(*args, **kwargs):
        if "user" not in session:
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    wrapper.__name__ = f.__name__
    return wrapper


# -----------------------------------
# SUBMISSION FORM (DYNAMIC)
# -----------------------------------
@app.route("/submit", methods=["GET", "POST"])
@login_required
def submit_claim():

    # Load master data
    db = get_db()
    cursor = db.cursor()

    cursor.execute("SELECT code, description FROM master_icd10 ORDER BY code")
    ICD10 = cursor.fetchall()

    cursor.execute("SELECT code, description FROM master_icd9 ORDER BY code")
    ICD9 = cursor.fetchall()

    cursor.execute("SELECT code, name FROM master_drug ORDER BY code")
    DRUGS = cursor.fetchall()

    cursor.execute("SELECT name FROM master_vitamin ORDER BY name")
    VITAMINS = [row[0] for row in cursor.fetchall()]

    cursor.close()
    db.close()

    # -----------------------------
    # POST — Submit Claim
    # -----------------------------
    if request.method == "POST":

        # -------- Patient data --------
        patient_name = request.form.get("patient_name")
        patient_nik = request.form.get("patient_nik")
        patient_dob = request.form.get("patient_dob")
        patient_gender = request.form.get("patient_gender")
        patient_phone = request.form.get("patient_phone")
        patient_address = request.form.get("patient_address")

        # -------- Visit data --------
        visit_date = request.form.get("visit_date")
        visit_type = request.form.get("visit_type")
        doctor_name = request.form.get("doctor_name")
        department = request.form.get("department")

        # -------- Diagnosis --------
        dx_primary = request.form.get("diagnosis_primary")
        dx_secondary = request.form.get("diagnosis_secondary")

        ICD10_MAP = {code: desc for code, desc in ICD10}
        px_desc_map = {code: desc for code, desc in ICD9}
        drug_map = {code: name for code, name in DRUGS}

        dx_primary_desc = ICD10_MAP[dx_primary]
        dx_secondary_desc = ICD10_MAP[dx_secondary] if dx_secondary else None

        # -------- Dynamic Procedures --------
        proc_codes = request.form.getlist("procedure_code[]")
        proc_costs = request.form.getlist("procedure_cost[]")

        proc_costs = [float(x or 0) for x in proc_costs]
        total_procedure_cost = sum(proc_costs)

        # -------- Dynamic Drugs --------
        drug_codes = request.form.getlist("drug_code[]")
        drug_costs = request.form.getlist("drug_cost[]")

        drug_costs = [float(x or 0) for x in drug_costs]
        total_drug_cost = sum(drug_costs)

        # -------- Dynamic Vitamins --------
        vitamin_names = request.form.getlist("vitamin_name[]")
        vitamin_costs = request.form.getlist("vitamin_cost[]")

        vitamin_costs = [float(x or 0) for x in vitamin_costs]
        total_vitamin_cost = sum(vitamin_costs)

        # -------- TOTAL CLAIM --------
        total_claim = total_procedure_cost + total_drug_cost + total_vitamin_cost

        # -------- Insert Claim Header --------
        db = get_db()
        cursor = db.cursor()

        cursor.execute("""
            INSERT INTO claim_header (
                patient_nik, patient_name, patient_gender, patient_dob,
                patient_address, patient_phone,
                visit_date, visit_type, doctor_name, department,
                total_procedure_cost, total_drug_cost, total_vitamin_cost,
                total_claim_amount, status
            )
            VALUES (%s,%s,%s,%s,%s,%s,
                    %s,%s,%s,%s,
                    %s,%s,%s,%s,'pending')
        """, (
            patient_nik, patient_name, patient_gender, patient_dob,
            patient_address, patient_phone,
            visit_date, visit_type, doctor_name, department,
            total_procedure_cost, total_drug_cost, total_vitamin_cost,
            total_claim
        ))

        db.commit()
        claim_id = cursor.lastrowid

        # -------- Insert Primary Diagnosis --------
        cursor.execute("""
            INSERT INTO claim_diagnosis (claim_id, icd10_code, icd10_description, is_primary)
            VALUES (%s,%s,%s,1)
        """, (claim_id, dx_primary, dx_primary_desc))

        # Secondary DX (optional)
        if dx_secondary:
            cursor.execute("""
                INSERT INTO claim_diagnosis (claim_id, icd10_code, icd10_description, is_primary)
                VALUES (%s,%s,%s,0)
            """, (claim_id, dx_secondary, dx_secondary_desc))

        # -------- Insert Procedures (Dynamic Rows) --------
        for code, cost in zip(proc_codes, proc_costs):
            cursor.execute("""
                INSERT INTO claim_procedure (claim_id, icd9_code, icd9_description, quantity, procedure_date, cost)
                VALUES (%s,%s,%s,1,%s,%s)
            """, (claim_id, code, px_desc_map[code], visit_date, cost))

        # -------- Insert Drugs (Dynamic Rows) --------
        for code, cost in zip(drug_codes, drug_costs):
            cursor.execute("""
                INSERT INTO claim_drug (claim_id, drug_code, drug_name, dosage, frequency, route, days, cost)
                VALUES (%s,%s,%s,'1 tablet','2x sehari','oral',1,%s)
            """, (claim_id, code, drug_map[code], cost))

        # -------- Insert Vitamins (Dynamic Rows) --------
        for name, cost in zip(vitamin_names, vitamin_costs):
            cursor.execute("""
                INSERT INTO claim_vitamin (claim_id, vitamin_name, dosage, days, cost)
                VALUES (%s,%s,'1 tablet',1,%s)
            """, (claim_id, name, cost))

        db.commit()
        cursor.close()
        db.close()

        return render_template(
            "submit_claim.html",
            msg="Claim berhasil disubmit!",
            ICD10=ICD10, ICD9=ICD9, DRUGS=DRUGS, VITAMINS=VITAMINS
        )

    # -----------------------------
    # GET — Render Form
    # -----------------------------
    return render_template(
        "submit_claim.html",
        ICD10=ICD10, ICD9=ICD9, DRUGS=DRUGS, VITAMINS=VITAMINS
    )


# -----------------------------------
# LIST CLAIMS
# -----------------------------------
@app.route("/list")
@login_required
def list_claims():

    page = int(request.args.get("page", 1))
    limit = 20
    offset = (page - 1) * limit

    db = get_db()
    cursor = db.cursor(dictionary=True)

    cursor.execute("SELECT COUNT(*) AS total FROM claim_header WHERE status='pending'")
    total = cursor.fetchone()["total"]
    total_pages = math.ceil(total / limit)

    cursor.execute(f"""
        SELECT claim_id, patient_name, patient_nik, total_claim_amount,
               status, created_at
        FROM claim_header
        WHERE status='pending'
        ORDER BY claim_id DESC
        LIMIT {limit} OFFSET {offset}
    """)
    results = cursor.fetchall()

    cursor.close()
    db.close()

    return render_template(
        "list_claims.html",
        claims=results,
        page=page,
        total_pages=total_pages
    )


# -----------------------------------
# LOGOUT
# -----------------------------------
@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("login"))


# -----------------------------------
# RUN SERVER
# -----------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=2221)