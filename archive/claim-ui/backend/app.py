#!/usr/bin/env python3
"""
ClaimFlow Backend API
Flask server with MySQL integration
"""
from flask import Flask, request, jsonify, session
from flask_cors import CORS
from datetime import timedelta
import mysql.connector
from functools import wraps
import os

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'supersecretkey_claimflow_2025')
app.permanent_session_lifetime = timedelta(hours=6)

# CORS untuk React frontend
CORS(app, supports_credentials=True, origins=['http://localhost:2221', 'http://localhost:5173'])

# Database config dari environment
DB_CONFIG = {
    'host': os.environ.get('DB_HOST', 'cdpmsi.tomodachis.org'),
    'user': os.environ.get('DB_USER', 'cloudera'),
    'password': os.environ.get('DB_PASSWORD', 'T1ku$H1t4m'),
    'database': os.environ.get('DB_NAME', 'claimdb')
}

# Valid credentials
VALID_USER = os.environ.get('ADMIN_USER', 'aris')
VALID_PASS = os.environ.get('ADMIN_PASS', 'Admin123')


def get_db():
    """Create database connection"""
    return mysql.connector.connect(**DB_CONFIG)


def login_required(f):
    """Decorator to check if user is logged in"""
    @wraps(f)
    def wrapper(*args, **kwargs):
        if 'user' not in session:
            return jsonify({'error': 'Unauthorized'}), 401
        return f(*args, **kwargs)
    return wrapper


# ============================================
# AUTH ENDPOINTS
# ============================================

@app.route('/api/login', methods=['POST'])
def login():
    data = request.json
    username = data.get('username')
    password = data.get('password')
    
    if username == VALID_USER and password == VALID_PASS:
        session.permanent = True
        session['user'] = username
        return jsonify({'success': True, 'user': username})
    
    return jsonify({'error': 'Invalid credentials'}), 401


@app.route('/api/logout', methods=['POST'])
def logout():
    session.pop('user', None)
    return jsonify({'success': True})


@app.route('/api/auth/check', methods=['GET'])
def check_auth():
    if 'user' in session:
        return jsonify({'authenticated': True, 'user': session['user']})
    return jsonify({'authenticated': False}), 401


# ============================================
# MASTER DATA ENDPOINTS
# ============================================

@app.route('/api/master-data', methods=['GET'])
@login_required
def get_master_data():
    """Get all master data for form dropdowns"""
    try:
        db = get_db()
        cursor = db.cursor()
        
        cursor.execute("SELECT code, description FROM master_icd10 ORDER BY code")
        icd10 = [{'code': r[0], 'description': r[1]} for r in cursor.fetchall()]
        
        cursor.execute("SELECT code, description FROM master_icd9 ORDER BY code")
        icd9 = [{'code': r[0], 'description': r[1]} for r in cursor.fetchall()]
        
        cursor.execute("SELECT code, name FROM master_drug ORDER BY code")
        drugs = [{'code': r[0], 'name': r[1]} for r in cursor.fetchall()]
        
        cursor.execute("SELECT name FROM master_vitamin ORDER BY name")
        vitamins = [r[0] for r in cursor.fetchall()]
        
        cursor.close()
        db.close()
        
        return jsonify({
            'icd10': icd10,
            'icd9': icd9,
            'drugs': drugs,
            'vitamins': vitamins
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============================================
# CLAIMS ENDPOINTS
# ============================================

@app.route('/api/claims', methods=['GET'])
@login_required
def list_claims():
    """Get paginated list of claims"""
    try:
        page = int(request.args.get('page', 1))
        limit = int(request.args.get('limit', 10))
        offset = (page - 1) * limit
        
        db = get_db()
        cursor = db.cursor(dictionary=True)
        
        # Get total count
        cursor.execute("SELECT COUNT(*) AS total FROM claim_header WHERE status='pending'")
        total = cursor.fetchone()['total']
        
        # Get claims
        cursor.execute("""
            SELECT claim_id, patient_name, patient_nik, total_claim_amount,
                   status, created_at
            FROM claim_header
            WHERE status='pending'
            ORDER BY claim_id DESC
            LIMIT %s OFFSET %s
        """, (limit, offset))
        claims = cursor.fetchall()
        
        # Convert datetime to string
        for c in claims:
            if c.get('created_at'):
                c['created_at'] = c['created_at'].isoformat()
        
        cursor.close()
        db.close()
        
        import math
        return jsonify({
            'claims': claims,
            'page': page,
            'limit': limit,
            'total': total,
            'total_pages': math.ceil(total / limit)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/claims', methods=['POST'])
@login_required
def submit_claim():
    """Submit a new claim"""
    try:
        data = request.json
        db = get_db()
        cursor = db.cursor()
        
        # Calculate totals
        total_procedure = sum(float(p.get('cost', 0)) for p in data.get('procedures', []))
        total_drug = sum(float(d.get('cost', 0)) for d in data.get('drugs', []))
        total_vitamin = sum(float(v.get('cost', 0)) for v in data.get('vitamins', []))
        total_claim = total_procedure + total_drug + total_vitamin
        
        # Insert claim header
        cursor.execute("""
            INSERT INTO claim_header (
                patient_nik, patient_name, patient_gender, patient_dob,
                patient_address, patient_phone,
                visit_date, visit_type, doctor_name, department,
                total_procedure_cost, total_drug_cost, total_vitamin_cost,
                total_claim_amount, status
            )
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,'pending')
        """, (
            data['patient_nik'], data['patient_name'], data['patient_gender'],
            data['patient_dob'], data.get('patient_address', ''),
            data.get('patient_phone', ''), data['visit_date'], data['visit_type'],
            data['doctor_name'], data['department'],
            total_procedure, total_drug, total_vitamin, total_claim
        ))
        db.commit()
        claim_id = cursor.lastrowid
        
        # Get ICD10 descriptions
        cursor.execute("SELECT code, description FROM master_icd10")
        icd10_map = {r[0]: r[1] for r in cursor.fetchall()}
        
        # Get ICD9 descriptions
        cursor.execute("SELECT code, description FROM master_icd9")
        icd9_map = {r[0]: r[1] for r in cursor.fetchall()}
        
        # Get drug names
        cursor.execute("SELECT code, name FROM master_drug")
        drug_map = {r[0]: r[1] for r in cursor.fetchall()}
        
        # Insert diagnosis
        dx_primary = data['diagnosis_primary']
        cursor.execute("""
            INSERT INTO claim_diagnosis (claim_id, icd10_code, icd10_description, is_primary)
            VALUES (%s,%s,%s,1)
        """, (claim_id, dx_primary, icd10_map.get(dx_primary, '')))
        
        dx_secondary = data.get('diagnosis_secondary')
        if dx_secondary:
            cursor.execute("""
                INSERT INTO claim_diagnosis (claim_id, icd10_code, icd10_description, is_primary)
                VALUES (%s,%s,%s,0)
            """, (claim_id, dx_secondary, icd10_map.get(dx_secondary, '')))
        
        # Insert procedures
        for proc in data.get('procedures', []):
            if proc.get('code'):
                cursor.execute("""
                    INSERT INTO claim_procedure (claim_id, icd9_code, icd9_description, quantity, procedure_date, cost)
                    VALUES (%s,%s,%s,1,%s,%s)
                """, (claim_id, proc['code'], icd9_map.get(proc['code'], ''), data['visit_date'], proc.get('cost', 0)))
        
        # Insert drugs
        for drug in data.get('drugs', []):
            if drug.get('code'):
                cursor.execute("""
                    INSERT INTO claim_drug (claim_id, drug_code, drug_name, dosage, frequency, route, days, cost)
                    VALUES (%s,%s,%s,'1 tablet','2x sehari','oral',1,%s)
                """, (claim_id, drug['code'], drug_map.get(drug['code'], ''), drug.get('cost', 0)))
        
        # Insert vitamins
        for vit in data.get('vitamins', []):
            if vit.get('name'):
                cursor.execute("""
                    INSERT INTO claim_vitamin (claim_id, vitamin_name, dosage, days, cost)
                    VALUES (%s,%s,'1 tablet',1,%s)
                """, (claim_id, vit['name'], vit.get('cost', 0)))
        
        db.commit()
        cursor.close()
        db.close()
        
        return jsonify({
            'success': True,
            'claim_id': claim_id,
            'total_claim': total_claim
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/claims/<int:claim_id>', methods=['GET'])
@login_required
def get_claim(claim_id):
    """Get single claim details"""
    try:
        db = get_db()
        cursor = db.cursor(dictionary=True)
        
        cursor.execute("SELECT * FROM claim_header WHERE claim_id = %s", (claim_id,))
        claim = cursor.fetchone()
        
        if not claim:
            return jsonify({'error': 'Claim not found'}), 404
        
        # Get diagnosis
        cursor.execute("SELECT * FROM claim_diagnosis WHERE claim_id = %s", (claim_id,))
        claim['diagnoses'] = cursor.fetchall()
        
        # Get procedures
        cursor.execute("SELECT * FROM claim_procedure WHERE claim_id = %s", (claim_id,))
        claim['procedures'] = cursor.fetchall()
        
        # Get drugs
        cursor.execute("SELECT * FROM claim_drug WHERE claim_id = %s", (claim_id,))
        claim['drugs'] = cursor.fetchall()
        
        # Get vitamins
        cursor.execute("SELECT * FROM claim_vitamin WHERE claim_id = %s", (claim_id,))
        claim['vitamins'] = cursor.fetchall()
        
        cursor.close()
        db.close()
        
        return jsonify(claim)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============================================
# HEALTH CHECK
# ============================================

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'service': 'claimflow-api'})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=2220, debug=True)
