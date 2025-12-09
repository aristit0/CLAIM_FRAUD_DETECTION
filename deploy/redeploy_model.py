#!/usr/bin/env python3
"""
Simple Model Redeployment Script for CML
Auto-deploys trained model without manual UI interaction

Usage in CML Session:
  !python deploy/redeploy_model.py

Or run directly:
  python deploy/redeploy_model.py
"""

import cmlapi
import time
import sys
import os
import json
from datetime import datetime

# ================================================================
# CONFIGURATION - EDIT THESE VALUES
# ================================================================

MODEL_NAME = "model_fraud_detection_claim"
MODEL_DESCRIPTION = "Claim Fraud Detection Model - Auto-deployed"
MODEL_FILE = "model.py"
MODEL_FUNCTION = "predict"

# Runtime settings
KERNEL = "python3"

# ================================================================
# HELPER FUNCTIONS
# ================================================================

def get_latest_runtime(client, project_id, kernel="python3"):
    """Get the latest compatible runtime for the project"""
    try:
        # List available runtimes
        runtimes = client.list_runtimes(
            search_filter=json.dumps({
                "kernel": kernel
            })
        )
        
        if not runtimes.runtimes:
            print("⚠ No runtimes found, using default")
            return None
        
        # Find Python 3.10 runtime (Standard)
        for runtime in runtimes.runtimes:
            if "python3.10" in runtime.image_identifier.lower() and "standard" in runtime.edition.lower():
                print(f"✓ Found runtime: {runtime.image_identifier}")
                return runtime.image_identifier
        
        # Fallback to first available Python 3 runtime
        for runtime in runtimes.runtimes:
            if "python3" in runtime.image_identifier.lower():
                print(f"✓ Using runtime: {runtime.image_identifier}")
                return runtime.image_identifier
        
        # Use first runtime as last resort
        runtime = runtimes.runtimes[0]
        print(f"✓ Using default runtime: {runtime.image_identifier}")
        return runtime.image_identifier
        
    except Exception as e:
        print(f"⚠ Could not get runtime list: {e}")
        return None

# ================================================================
# DEPLOYMENT FUNCTION
# ================================================================

def deploy_model():
    """Deploy trained model to CML"""
    
    print("=" * 80)
    print("CML MODEL AUTO-DEPLOYER")
    print("=" * 80)
    
    # Initialize CML API client
    try:
        client = cmlapi.default_client()
        print("✓ Connected to CML API")
    except Exception as e:
        print(f"✗ Failed to connect to CML API: {e}")
        return False
    
    # Get current project
    try:
        project_id = os.getenv("CDSW_PROJECT_ID")
        if not project_id:
            print("✗ CDSW_PROJECT_ID not found. Run this script inside a CML project.")
            return False
        
        project = client.get_project(project_id)
        print(f"✓ Project: {project.name}")
        print(f"✓ Project ID: {project.id}")
    except Exception as e:
        print(f"✗ Failed to get project: {e}")
        return False
    
    print("=" * 80)
    
    # Get runtime identifier
    print(f"\n[0/5] Getting runtime information...")
    runtime_identifier = get_latest_runtime(client, project.id, KERNEL)
    
    if not runtime_identifier:
        print("✗ Could not determine runtime. Please check CML UI for available runtimes.")
        return False
    
    # Step 1: Find or create model
    print(f"\n[1/5] Searching for model '{MODEL_NAME}'...")
    model = None
    
    try:
        models = client.list_models(project.id)
        for m in models.models:
            if m.name == MODEL_NAME:
                model = m
                print(f"✓ Found existing model: {model.id}")
                break
        
        if not model:
            print(f"  Model not found. Creating new model...")
            model_body = cmlapi.CreateModelRequest(
                project_id=project.id,
                name=MODEL_NAME,
                description=MODEL_DESCRIPTION,
                disable_authentication=True
            )
            model = client.create_model(model_body, project.id)
            print(f"✓ Model created: {model.id}")
    
    except Exception as e:
        print(f"✗ Error with model: {e}")
        return False
    
    # Step 2: Build model
    print(f"\n[2/5] Building model from '{MODEL_FILE}'...")
    
    model_file_path = os.path.join("/home/cdsw", MODEL_FILE)
    if not os.path.exists(model_file_path):
        print(f"✗ Model file not found: {model_file_path}")
        print("  Make sure model.py exists in project root")
        return False
    
    print(f"✓ File found: {model_file_path}")
    
    try:
        # Create build request WITH runtime identifier
        model_build_body = cmlapi.CreateModelBuildRequest(
            project_id=project.id,
            model_id=model.id,
            file_path=MODEL_FILE,
            function_name=MODEL_FUNCTION,
            kernel=KERNEL,
            runtime_identifier=runtime_identifier  # ← REQUIRED!
        )
        
        model_build = client.create_model_build(
            model_build_body,
            project.id,
            model.id
        )
        
        print(f"✓ Build started: {model_build.id}")
        print(f"  Waiting for build (max 30 minutes)...")
        
        # Wait for build
        start_time = time.time()
        while model_build.status not in ["built", "build failed"]:
            if time.time() - start_time > 1800:  # 30 min timeout
                print("✗ Build timeout")
                return False
            
            elapsed = int(time.time() - start_time)
            print(f"  Building... {elapsed}s elapsed", end="\r")
            time.sleep(10)
            
            model_build = client.get_model_build(
                project.id,
                model.id,
                model_build.id
            )
        
        print()  # New line
        
        if model_build.status == "build failed":
            print("✗ Build failed. Check CML UI for details.")
            return False
        
        build_time = int(time.time() - start_time)
        print(f"✓ Build completed in {build_time}s")
    
    except Exception as e:
        print(f"✗ Error building model: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 3: Deploy model
    print(f"\n[3/5] Deploying model...")
    
    try:
        # Create deployment request
        model_deployment_body = cmlapi.CreateModelDeploymentRequest(
            project_id=project.id,
            model_id=model.id,
            build_id=model_build.id
        )
        
        model_deployment = client.create_model_deployment(
            model_deployment_body,
            project.id,
            model.id,
            model_build.id
        )
        
        print(f"✓ Deployment started: {model_deployment.id}")
        print(f"  Waiting for deployment (max 30 minutes)...")
        
        # Wait for deployment
        start_time = time.time()
        while model_deployment.status not in ["stopped", "failed", "deployed"]:
            if time.time() - start_time > 1800:  # 30 min timeout
                print("✗ Deployment timeout")
                return False
            
            elapsed = int(time.time() - start_time)
            print(f"  Deploying... {elapsed}s elapsed", end="\r")
            time.sleep(10)
            
            model_deployment = client.get_model_deployment(
                project.id,
                model.id,
                model_build.id,
                model_deployment.id
            )
        
        print()  # New line
        
        if model_deployment.status != "deployed":
            print(f"✗ Deployment failed with status: {model_deployment.status}")
            return False
        
        deploy_time = int(time.time() - start_time)
        print(f"✓ Deployed in {deploy_time}s")
    
    except Exception as e:
        print(f"✗ Error deploying model: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 4: Stop old deployments
    print(f"\n[4/5] Stopping old deployments...")
    
    try:
        builds = client.list_model_builds(project.id, model.id)
        stopped_count = 0
        
        for build in builds.model_builds:
            deployments = client.list_model_deployments(
                project.id,
                model.id,
                build.id
            )
            
            for deployment in deployments.model_deployments:
                # Skip current deployment
                if deployment.id == model_deployment.id:
                    continue
                
                # Stop if running
                if deployment.status == "deployed":
                    client.stop_model_deployment(
                        project.id,
                        model.id,
                        build.id,
                        deployment.id
                    )
                    stopped_count += 1
                    print(f"  Stopped: {deployment.id[:8]}")
        
        if stopped_count > 0:
            print(f"✓ Stopped {stopped_count} old deployment(s)")
        else:
            print("✓ No old deployments")
    
    except Exception as e:
        print(f"⚠ Warning: Could not stop old deployments: {e}")
    
    # Step 5: Test deployment
    print(f"\n[5/5] Testing deployment...")
    
    try:
        deployment_info = client.get_model_deployment(
            project.id,
            model.id,
            model_build.id,
            model_deployment.id
        )
        
        print(f"✓ Status: {deployment_info.status}")
        
        # Get access key
        if hasattr(deployment_info, 'access_key'):
            print(f"✓ Access Key: {deployment_info.access_key}")
            
            # Test prediction
            test_payload = {
                "claims": [{
                    "claim_id": "TEST001",
                    "patient_dob": "1985-06-15",
                    "visit_date": "2024-12-04",
                    "visit_type": "rawat jalan",
                    "department": "Poli Umum",
                    "icd10_primary_code": "J06",
                    "procedures": ["89.02"],
                    "drugs": ["KFA001"],
                    "vitamins": ["Vitamin C 500 mg"],
                    "total_procedure_cost": 100000,
                    "total_drug_cost": 50000,
                    "total_vitamin_cost": 20000,
                    "total_claim_amount": 170000,
                    "patient_frequency_risk": 2
                }]
            }
            
            try:
                response = client.call_model(
                    deployment_info.access_key,
                    json.dumps(test_payload)
                )
                
                result = json.loads(response)
                if result.get("status") == "success":
                    fraud_score = result['results'][0]['fraud_score']
                    print(f"✓ Test prediction successful!")
                    print(f"  Fraud score: {fraud_score:.4f}")
                    print(f"  Fraud probability: {result['results'][0]['fraud_probability']}")
                else:
                    print(f"⚠ Test returned: {result}")
            except Exception as e:
                print(f"⚠ Could not test prediction: {e}")
        else:
            print("⚠ No access key found in deployment info")
    
    except Exception as e:
        print(f"⚠ Could not verify deployment: {e}")
    
    # Success!
    print("\n" + "=" * 80)
    print("✅ DEPLOYMENT SUCCESSFUL")
    print("=" * 80)
    print(f"Model Name:     {MODEL_NAME}")
    print(f"Model ID:       {model.id}")
    print(f"Build ID:       {model_build.id}")
    print(f"Deployment ID:  {model_deployment.id}")
    print(f"Status:         {model_deployment.status}")
    print(f"Runtime:        {runtime_identifier}")
    print("=" * 80)
    
    return True

# ================================================================
# MAIN
# ================================================================

if __name__ == "__main__":
    success = deploy_model()
    sys.exit(0 if success else 1)