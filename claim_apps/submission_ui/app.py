#!/usr/bin/env python3
from flask import Flask, render_template, redirect, request, url_for, session
import mysql.connector
from datetime import timedelta
import math

app = Flask(__name__)
app.secret_key = "supersecretkey_123"
app.permanent_session_lifetime = timedelta(hours=6)

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

# Register filter
app.jinja_env.filters["rupiah"] = format_rupiah

def login_required(f):
    def wrapper(*args, **kwargs):
        if "user" not in session:
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    wrapper.__name__ = f.__name__
    return wrapper


# -----------------------------------
# SUBMISSION FORM V2
# -----------------------------------
@app.route("/submit", methods=["GET", "POST"])
@login_required
def submit_claim():

    # === MASTER DATA (dropdowns) ===
    ICD10 = [
        ("A09", "Diare dan gastroenteritis"),
        ("J06", "Common cold"),
        ("I10", "Hipertensi"),
        ("E11", "Diabetes mellitus tipe 2"),
        ("J45", "Asma"),
        ("K29", "Gastritis"),
    ]

    ICD9 = [
        ("96.70", "Injeksi obat"),
        ("99.04", "Transfusi darah"),
        ("03.31", "Pemeriksaan darah"),
        ("45.13", "Endoskopi"),
        ("93.90", "Electrotherapy"),
    ]

    DRUGS = [
        ("KFA001", "Paracetamol 500 mg"),
        ("KFA002", "Amoxicillin 500 mg"),
        ("KFA003", "Ceftriaxone injeksi"),
        ("KFA004", "Omeprazole 20 mg"),
        ("KFA005", "ORS / Oralit"),
    ]

    VITAMINS = [
        "Vitamin C 500 mg",
        "Vitamin B Complex",
        "Vitamin D 1000 IU",
        "Vitamin E 400 IU",
    ]

    if request.method == "POST":

        # PATIENT INFO
        patient_name = request.form.get("patient_name")
        patient_nik = request.form.get("patient_nik")
        patient_dob = request.form.get("patient_dob")
        patient_gender = request.form.get("patient_gender")
        patient_phone = request.form.get("patient_phone")
        patient_address = request.form.get("patient_address")

        # VISIT INFO
        visit_date = request.form.get("visit_date")
        visit_type = request.form.get("visit_type")
        doctor_name = request.form.get("doctor_name")
        department = request.form.get("department")

        # MEDICAL DETAILS
        dx_primary = request.form.get("diagnosis_primary")
        dx_primary_desc = dict(ICD10)[dx_primary]

        dx_secondary = request.form.get("diagnosis_secondary")
        dx_secondary_desc = dict(ICD10)[dx_secondary] if dx_secondary else None

        px_code = request.form.get("procedure")
        px_desc = dict(ICD9)[px_code]

        drug_code = request.form.get("drug_code")
        drug_name = dict(DRUGS)[drug_code]
        drug_cost = float(request.form.get("drug_cost") or 0)

        vitamin_name = request.form.get("vitamin_name")
        vitamin_cost = float(request.form.get("vitamin_cost") or 0)

        # COSTS
        proc_cost = float(request.form.get("procedure_cost") or 0)
        total_claim = proc_cost + drug_cost + vitamin_cost

        # DB
        db = get_db()
        cursor = db.cursor()

        # INSERT CLAIM HEADER
        cursor.execute("""
            INSERT INTO claim_header (
                patient_nik, patient_name, patient_gender, patient_dob,
                patient_address, patient_phone,
                visit_date, visit_type, doctor_name, department,
                total_procedure_cost, total_drug_cost, total_vitamin_cost, total_claim_amount, status
            )
            VALUES (%s,%s,%s,%s,%s,%s,
                    %s,%s,%s,%s,
                    %s,%s,%s,%s,'pending')
        """, (
            patient_nik, patient_name, patient_gender, patient_dob,
            patient_address, patient_phone,
            visit_date, visit_type, doctor_name, department,
            proc_cost, drug_cost, vitamin_cost, total_claim
        ))

        db.commit()
        claim_id = cursor.lastrowid

        # INSERT PRIMARY DX
        cursor.execute("""
            INSERT INTO claim_diagnosis (claim_id, icd10_code, icd10_description, is_primary)
            VALUES (%s,%s,%s,1)
        """, (claim_id, dx_primary, dx_primary_desc))

        # INSERT SECONDARY DX (optional)
        if dx_secondary:
            cursor.execute("""
                INSERT INTO claim_diagnosis (claim_id, icd10_code, icd10_description, is_primary)
                VALUES (%s,%s,%s,0)
            """, (claim_id, dx_secondary, dx_secondary_desc))

        # INSERT PROCEDURE
        cursor.execute("""
            INSERT INTO claim_procedure (claim_id, icd9_code, icd9_description, quantity, procedure_date)
            VALUES (%s,%s,%s,1,%s)
        """, (claim_id, px_code, px_desc, visit_date))

        # INSERT DRUG
        cursor.execute("""
            INSERT INTO claim_drug (claim_id, drug_code, drug_name, dosage, frequency, route, days, cost)
            VALUES (%s,%s,%s,'1 tablet','2x sehari','oral',1,%s)
        """, (claim_id, drug_code, drug_name, drug_cost))

        # INSERT VITAMIN
        cursor.execute("""
            INSERT INTO claim_vitamin (claim_id, vitamin_name, dosage, days, cost)
            VALUES (%s,%s,'1 tablet',1,%s)
        """, (claim_id, vitamin_name, vitamin_cost))

        db.commit()
        cursor.close()
        db.close()

        return render_template("submit_claim.html",
                               msg="Claim berhasil disubmit!",
                               ICD10=ICD10, ICD9=ICD9, DRUGS=DRUGS, VITAMINS=VITAMINS)

    return render_template("submit_claim.html",
                           ICD10=ICD10, ICD9=ICD9, DRUGS=DRUGS, VITAMINS=VITAMINS)


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

    cursor.execute("SELECT COUNT(*) AS total FROM claim_header")
    total = cursor.fetchone()["total"]
    total_pages = math.ceil(total / limit)

    cursor.execute(f"""
        SELECT claim_id, patient_name, patient_nik, total_claim_amount, status, created_at
        FROM claim_header
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
# RUN
# -----------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=2221)