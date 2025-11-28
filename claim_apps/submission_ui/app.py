from flask import Flask, render_template, redirect, request, url_for, session
import mysql.connector
from datetime import timedelta

app = Flask(__name__)
app.secret_key = "supersecretkey_123"
app.permanent_session_lifetime = timedelta(hours=6)

VALID_USER = "aris"
VALID_PASS = "Admin123"

# -----------------------------------
# MySQL Connection
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


def login_required(f):
    def wrapper(*args, **kwargs):
        if "user" not in session:
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    wrapper.__name__ = f.__name__
    return wrapper


# -----------------------------------
# SUBMIT CLAIM
# -----------------------------------
@app.route("/submit", methods=["GET", "POST"])
@login_required
def submit_claim():

    if request.method == "POST":

        patient_name = request.form.get("patient_name")
        patient_nik = request.form.get("patient_nik")

        # RAW INPUT
        diagnosis_raw = request.form.get("diagnosis")
        procedure_raw = request.form.get("procedure")

        drug = request.form.get("drug")
        vitamin = request.form.get("vitamin")
        cost = request.form.get("cost")

        # PARSE ICD CODES (ambil kode saja)
        # Example input: "J06 – Common cold" → "J06"
        diagnosis_code = diagnosis_raw.split(" ")[0]
        procedure_code = procedure_raw.split(" ")[0]

        db = get_db()
        cursor = db.cursor()

        # Insert claim_header
        cursor.execute("""
            INSERT INTO claim_header 
            (patient_nik, patient_name, patient_gender, patient_dob, patient_address, patient_phone,
             visit_date, visit_type, doctor_name, department,
             total_procedure_cost, total_drug_cost, total_vitamin_cost, total_claim_amount, status)
            VALUES (%s,%s,'M','2000-01-01','-','-', CURDATE(), 'rawat jalan','dr. Default','Poli Umum',
                    %s,0,0,%s,'pending')
        """, (patient_nik, patient_name, cost, cost))

        db.commit()
        claim_id = cursor.lastrowid

        # Insert ICD-10 Diagnosis
        cursor.execute("""
            INSERT INTO claim_diagnosis (claim_id, icd10_code, icd10_description, is_primary)
            VALUES (%s,%s,%s,1)
        """, (claim_id, diagnosis_code, diagnosis_raw))

        # Insert ICD-9 Procedure
        cursor.execute("""
            INSERT INTO claim_procedure (claim_id, icd9_code, icd9_description, quantity, procedure_date)
            VALUES (%s,%s,%s,1,CURDATE())
        """, (claim_id, procedure_code, procedure_raw))

        # Insert drug
        cursor.execute("""
            INSERT INTO claim_drug (claim_id, drug_code, drug_name, dosage, frequency, route, days, cost)
            VALUES (%s,'-',%s,'1 tablet','2x sehari','oral',1,0)
        """, (claim_id, drug))

        # Insert vitamin
        cursor.execute("""
            INSERT INTO claim_vitamin (claim_id, vitamin_name, dosage, days, cost)
            VALUES (%s,%s,'1 tablet',1,0)
        """, (claim_id, vitamin))

        db.commit()
        cursor.close()
        db.close()

        return render_template("submit_claim.html", msg="Claim berhasil disubmit!")

    return render_template("submit_claim.html")


# -----------------------------------
# LIST CLAIMS
# -----------------------------------
@app.route("/list")
@login_required
def list_claims():
    db = get_db()
    cursor = db.cursor(dictionary=True)

    cursor.execute("""
        SELECT claim_id, patient_name, patient_nik, total_claim_amount, status, created_at 
        FROM claim_header
        ORDER BY claim_id DESC
    """)

    results = cursor.fetchall()
    cursor.close()
    db.close()

    return render_template("list_claims.html", claims=results)


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