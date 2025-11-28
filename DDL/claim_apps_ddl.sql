CREATE DATABASE IF NOT EXISTS claimdb;
USE claimdb;

CREATE TABLE claim_header (
    claim_id BIGINT AUTO_INCREMENT PRIMARY KEY,
    patient_nik VARCHAR(20),
    patient_name VARCHAR(255),
    patient_gender VARCHAR(1),
    patient_dob DATE,
    patient_address TEXT,
    patient_phone VARCHAR(20),

    visit_date DATE,
    visit_type VARCHAR(50),           -- rawat jalan, rawat inap, IGD
    doctor_name VARCHAR(255),
    department VARCHAR(100),

    total_procedure_cost DECIMAL(12,2),
    total_drug_cost DECIMAL(12,2),
    total_vitamin_cost DECIMAL(12,2),
    total_claim_amount DECIMAL(12,2),

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);


CREATE TABLE claim_diagnosis (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    claim_id BIGINT,
    icd10_code VARCHAR(10),
    icd10_description VARCHAR(255),
    is_primary TINYINT DEFAULT 0,
    FOREIGN KEY (claim_id) REFERENCES claim_header(claim_id)
);


CREATE TABLE claim_procedure (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    claim_id BIGINT,
    icd9_code VARCHAR(10),
    icd9_description VARCHAR(255),
    quantity INT DEFAULT 1,
    procedure_date DATE,
    FOREIGN KEY (claim_id) REFERENCES claim_header(claim_id)
);

CREATE TABLE claim_drug (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    claim_id BIGINT,
    drug_code VARCHAR(50),               -- kode KFA atau kode farmasi RS
    drug_name VARCHAR(255),
    dosage VARCHAR(100),
    frequency VARCHAR(50),
    route VARCHAR(50),
    days INT DEFAULT 1,
    cost DECIMAL(12,2),
    FOREIGN KEY (claim_id) REFERENCES claim_header(claim_id)
);


CREATE TABLE claim_vitamin (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    claim_id BIGINT,
    vitamin_name VARCHAR(255),
    dosage VARCHAR(100),
    days INT DEFAULT 1,
    cost DECIMAL(12,2),
    FOREIGN KEY (claim_id) REFERENCES claim_header(claim_id)
);


ALTER TABLE claim_header 
ADD COLUMN status VARCHAR(20) DEFAULT 'pending';