# deploy_model.py
import cml.models_v1 as models

project_id = "your-project-id"

# Create model
model = models.create_model(
    project_id=project_id,
    name="BPJS Fraud Detection",
    description="Real-time fraud detection for BPJS claims with clinical compatibility checking",
    disable_authentication=False
)

# Create model build
build = models.create_model_build(
    project_id=project_id,
    model_id=model.id,
    file_path="model.py",
    function_name="predict",
    kernel="python3",
    cpu=2,
    memory=4
)

# Deploy model
deployment = models.create_model_deployment(
    project_id=project_id,
    model_id=model.id,
    build_id=build.id,
    cpu=2,
    memory=4,
    replicas=2
)

print(f"✓ Model deployed successfully!")
print(f"  Model ID: {model.id}")
print(f"  Deployment ID: {deployment.id}")
print(f"  Endpoint: {deployment.access_key}")

## 📝 **EXAMPLE REQUEST FROM APPROVAL UI**
import requests
import json

# CML Model endpoint
endpoint_url = "https://your-cml-instance/models/bpjs-fraud-detection"
api_key = "your-api-key"

# Request payload
payload = {
    "claims": [
        {
            "claim_id": "CLM20241204001",
            "patient_dob": "1985-06-15",
            "visit_date": "2024-12-04",
            "visit_type": "rawat jalan",
            "department": "Poli Umum",
            "icd10_primary_code": "J06",
            "procedures": ["89.02"],
            "drugs": ["KFA001", "KFA009"],
            "vitamins": ["Vitamin C 500 mg"],
            "total_procedure_cost": 100000,
            "total_drug_cost": 50000,
            "total_vitamin_cost": 20000,
            "total_claim_amount": 170000,
            "patient_frequency_risk": 3
        }
    ]
}

# Call model
response = requests.post(
    f"{endpoint_url}/predict",
    headers={"Authorization": f"Bearer {api_key}"},
    json=payload
)

result = response.json()
print(json.dumps(result, indent=2))


## 📊 **EXPECTED RESPONSE**
{
  "status": "success",
  "model_version": "v2.0_production",
  "timestamp": "2024-12-04T10:30:00",
  "total_claims_processed": 1,
  "fraud_detected": 0,
  "results": [
    {
      "claim_id": "CLM20241204001",
      "fraud_score": 0.15,
      "fraud_probability": "15.0%",
      "fraud_flag": 0,
      "risk_level": "MINIMAL",
      "risk_color": "green",
      "confidence": 0.92,
      "explanation": "🟢 RISIKO MINIMAL: Tidak ada indikator fraud yang signifikan",
      "recommendation": "✅ RECOMMENDED: Approve, tidak ada red flag",
      "clinical_compatibility": {
        "procedure_compatible": true,
        "drug_compatible": true,
        "vitamin_compatible": true,
        "overall_compatible": true,
        "details": {
          "diagnosis_code": "J06",
          "procedure_details": [
            {"code": "89.02", "status": "✓ Sesuai", "compatible": true}
          ],
          "drug_details": [
            {"code": "KFA001", "status": "✓ Sesuai", "compatible": true},
            {"code": "KFA009", "status": "✓ Sesuai", "compatible": true}
          ],
          "vitamin_details": [
            {"name": "Vitamin C 500 mg", "status": "✓ Sesuai", "compatible": true}
          ]
        }
      }
    }
  ]
}