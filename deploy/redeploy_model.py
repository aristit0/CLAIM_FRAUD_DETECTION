#!/usr/bin/env python3
"""
Model Redeployment Script for CML
Works in both CML Jobs and Interactive Sessions

Usage:
  python deploy/redeploy_model.py  # As script/job
  !python deploy/redeploy_model.py  # In notebook
"""

import cmlapi
import time
import sys
import os
import json
from datetime import datetime

# ================================================================
# CONFIGURATION
# ================================================================

MODEL_NAME = "model_fraud_detection_claim"
MODEL_DESCRIPTION = "Claim Fraud Detection Model - Auto-deployed"
MODEL_FILE = "model.py"
MODEL_FUNCTION = "predict"
KERNEL = "python3"

# ================================================================
# HELPER FUNCTIONS
# ================================================================

def is_notebook():
    """Check if running in Jupyter/IPython"""
    try:
        get_ipython()
        return True
    except NameError:
        return False

def get_runtime_identifier(client, project_id, model_id=None):
    """Get runtime identifier - try multiple methods"""
    
    # Method 1: From existing build
    if model_id:
        try:
            print("  Getting runtime from existing build...")
            builds = client.list_model_builds(project_id, model_id)
            
            for build in builds.model_builds:
                if build.status == "built" and hasattr(build, 'runtime_identifier'):
                    print(f"  ✓ Using: {build.runtime_identifier[:60]}...")
                    return build.runtime_identifier
        except Exception as e:
            print(f"  Could not get from builds: {e}")
    
    # Method 2: List all runtimes
    try:
        print("  Listing available runtimes...")
        runtimes = client.list_runtimes()
        
        if runtimes and hasattr(runtimes, 'runtimes') and runtimes.runtimes:
            # Find Python 3.10 Standard
            for runtime in runtimes.runtimes:
                img = runtime.image_identifier.lower()
                edition = runtime.edition.lower()
                
                if "python" in img and "3.10" in img and "standard" in edition:
                    print(f"  ✓ Found: {runtime.image_identifier[:60]}...")
                    return runtime.image_identifier
            
            # Fallback
            for runtime in runtimes.runtimes:
                img = runtime.image_identifier.lower()
                edition = runtime.edition.lower()
                
                if "python" in img and "3" in img and "standard" in edition:
                    print(f"  ✓ Using: {runtime.image_identifier[:60]}...")
                    return runtime.image_identifier
            
            # Last resort
            runtime = runtimes.runtimes[0]
            print(f"  ⚠ Using first runtime: {runtime.image_identifier[:60]}...")
            return runtime.image_identifier
    
    except Exception as e:
        print(f"  Could not list runtimes: {e}")
    
    # Method 3: Default pattern
    print("  ⚠ Using default runtime")
    return "docker.repository.cloudera.com/cloudera/cdsw/ml-runtime-workbench-python3.10-standard:2024.02.1-b4"

# ================================================================
# DEPLOYMENT FUNCTION
# ================================================================

def deploy_model():
    """Deploy trained model to CML"""
    
    print("=" * 80)
    print("CML MODEL AUTO-DEPLOYER")
    print("=" * 80)
    
    # Initialize API client
    try:
        client = cmlapi.default_client()
        print("✓ Connected to CML API")
    except Exception as e:
        print(f"✗ Failed to connect: {e}")
        return False
    
    # Get project
    try:
        project_id = os.getenv("CDSW_PROJECT_ID")
        if not project_id:
            print("✗ CDSW_PROJECT_ID not found")
            return False
        
        project = client.get_project(project_id)
        print(f"✓ Project: {project.name}")
        print(f"✓ Project ID: {project.id}")
    except Exception as e:
        print(f"✗ Failed to get project: {e}")
        return False
    
    print("=" * 80)
    
    # Step 1: Find or create model
    print(f"\n[1/6] Searching for model '{MODEL_NAME}'...")
    model = None
    
    try:
        models = client.list_models(project.id)
        for m in models.models:
            if m.name == MODEL_NAME:
                model = m
                print(f"✓ Found: {model.id}")
                break
        
        if not model:
            print("  Creating new model...")
            model_body = cmlapi.CreateModelRequest(
                project_id=project.id,
                name=MODEL_NAME,
                description=MODEL_DESCRIPTION,
                disable_authentication=True
            )
            model = client.create_model(model_body, project.id)
            print(f"✓ Created: {model.id}")
    
    except Exception as e:
        print(f"✗ Error: {e}")
        return False
    
    # Step 2: Get runtime
    print(f"\n[2/6] Getting runtime...")
    runtime_identifier = get_runtime_identifier(client, project.id, model.id)
    
    if not runtime_identifier:
        print("✗ Could not determine runtime")
        return False
    
    print(f"✓ Runtime configured")
    
    # Step 3: Build model
    print(f"\n[3/6] Building model from '{MODEL_FILE}'...")
    
    model_file_path = os.path.join("/home/cdsw", MODEL_FILE)
    if not os.path.exists(model_file_path):
        print(f"✗ File not found: {model_file_path}")
        return False
    
    print(f"✓ File found: {model_file_path}")
    
    try:
        model_build_body = cmlapi.CreateModelBuildRequest(
            project_id=project.id,
            model_id=model.id,
            file_path=MODEL_FILE,
            function_name=MODEL_FUNCTION,
            kernel=KERNEL,
            runtime_identifier=runtime_identifier
        )
        
        model_build = client.create_model_build(
            model_build_body,
            project.id,
            model.id
        )
        
        print(f"✓ Build started: {model_build.id[:8]}...")
        print("  Waiting for build (max 30 min)...")
        
        # Wait for build
        start_time = time.time()
        while model_build.status not in ["built", "build failed"]:
            if time.time() - start_time > 1800:
                print("✗ Build timeout")
                return False
            
            elapsed = int(time.time() - start_time)
            print(f"  Building... {elapsed}s", end="\r")
            time.sleep(10)
            
            model_build = client.get_model_build(
                project.id,
                model.id,
                model_build.id
            )
        
        print()  # New line
        
        if model_build.status == "build failed":
            print("✗ Build failed - check CML UI")
            return False
        
        build_time = int(time.time() - start_time)
        print(f"✓ Build completed in {build_time}s")
    
    except Exception as e:
        print(f"✗ Build error: {e}")
        return False
    
    # Step 4: Deploy model
    print(f"\n[4/6] Deploying model...")
    
    try:
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
        
        print(f"✓ Deployment started: {model_deployment.id[:8]}...")
        print("  Waiting for deployment (max 30 min)...")
        
        start_time = time.time()
        while model_deployment.status not in ["stopped", "failed", "deployed"]:
            if time.time() - start_time > 1800:
                print("✗ Deployment timeout")
                return False
            
            elapsed = int(time.time() - start_time)
            print(f"  Deploying... {elapsed}s", end="\r")
            time.sleep(10)
            
            model_deployment = client.get_model_deployment(
                project.id,
                model.id,
                model_build.id,
                model_deployment.id
            )
        
        print()  # New line
        
        if model_deployment.status != "deployed":
            print(f"✗ Deployment failed: {model_deployment.status}")
            return False
        
        deploy_time = int(time.time() - start_time)
        print(f"✓ Deployed in {deploy_time}s")
    
    except Exception as e:
        print(f"✗ Deployment error: {e}")
        return False
    
    # Step 5: Stop old deployments
    print(f"\n[5/6] Stopping old deployments...")
    
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
                if deployment.id == model_deployment.id:
                    continue
                
                if deployment.status == "deployed":
                    client.stop_model_deployment(
                        project.id,
                        model.id,
                        build.id,
                        deployment.id
                    )
                    stopped_count += 1
        
        if stopped_count > 0:
            print(f"✓ Stopped {stopped_count} old deployment(s)")
        else:
            print("✓ No old deployments to stop")
    
    except Exception as e:
        print(f"⚠ Warning: {e}")
    
    # Step 6: Verify deployment
    print(f"\n[6/6] Verifying deployment...")
    
    try:
        # Refresh deployment info
        deployment_info = client.get_model_deployment(
            project.id,
            model.id,
            model_build.id,
            model_deployment.id
        )
        
        print(f"✓ Status: {deployment_info.status}")
        
        # Check for access key (may not be immediately available)
        access_key = None
        if hasattr(deployment_info, 'access_key') and deployment_info.access_key:
            access_key = deployment_info.access_key
        
        # Alternative: get from model list
        if not access_key:
            try:
                model_info = client.get_model(project.id, model.id)
                if hasattr(model_info, 'latest_deployment_details'):
                    details = model_info.latest_deployment_details
                    if hasattr(details, 'access_key'):
                        access_key = details.access_key
            except:
                pass
        
        if access_key:
            print(f"✓ Access Key: {access_key}")
            
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
                    access_key,
                    json.dumps(test_payload)
                )
                
                result = json.loads(response)
                if result.get("status") == "success":
                    print("✓ Test prediction successful!")
                    print(f"  Fraud score: {result['results'][0]['fraud_score']:.4f}")
                else:
                    print(f"⚠ Unexpected result: {result.get('status')}")
            except Exception as e:
                print(f"⚠ Test failed: {e}")
        else:
            print("⚠ Access key not available yet")
            print("  Check CML UI for deployment details")
    
    except Exception as e:
        print(f"⚠ Verification warning: {e}")
    
    # Success summary
    print("\n" + "=" * 80)
    print("✅ DEPLOYMENT SUCCESSFUL")
    print("=" * 80)
    print(f"Model:       {MODEL_NAME}")
    print(f"Build:       {model_build.id}")
    print(f"Deployment:  {model_deployment.id}")
    print(f"Status:      {model_deployment.status}")
    print("=" * 80)
    
    return True

# ================================================================
# MAIN
# ================================================================

def main():
    """Main entry point"""
    try:
        success = deploy_model()
        
        # Handle exit differently for notebook vs script
        if is_notebook():
            # In notebook, just return the result
            if success:
                print("\n✅ Script completed successfully")
            else:
                print("\n❌ Script failed")
            return success
        else:
            # In script/job, use sys.exit
            sys.exit(0 if success else 1)
    
    except Exception as e:
        print(f"\n❌ FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        
        if is_notebook():
            return False
        else:
            sys.exit(1)

if __name__ == "__main__":
    main()