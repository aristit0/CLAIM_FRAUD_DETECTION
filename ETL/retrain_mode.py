#!/usr/bin/env python3
import os
import subprocess

# Jalankan ETL dulu
print("=== RUNNING ETL JOB ===")
code = subprocess.call(["python3", "/home/cdsw/etl_v4_fixed_full.py"])
if code != 0:
    print("ETL FAILED. STOP TRAINING.")
    exit(1)

# Training
print("=== RUNNING TRAINING JOB ===")
code = subprocess.call(["python3", "/home/cdsw/train_model.py"])
if code != 0:
    print("TRAINING FAILED.")
    exit(1)

# Reload CML Model Deployment
print("=== REFRESHING DEPLOYED MODEL ===")
cmd = """
curl -X POST \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer $CML_API_KEY" \\
  https://<CML_DOMAIN>/api/v1/models/<MODEL_ID>/versions/<VERSION_ID>/deploy
"""
print("NOTE: Replace <CML_DOMAIN>, <MODEL_ID>, <VERSION_ID>")