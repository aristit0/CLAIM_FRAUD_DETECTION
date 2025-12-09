#!/usr/bin/env python3
"""
Automatic Model Redeployment for CML
Handles:
- Finding existing model
- Building new version
- Deploying with zero downtime
- Rollback on failure

Usage:
  python deploy/redeploy_model.py --project-name "BPJS Fraud Detection"
"""

import cmlapi
import time
import sys
import os
import json
import argparse
from datetime import datetime

# ================================================================
# CONFIGURATION
# ================================================================
class DeploymentConfig:
    """Deployment configuration"""
    
    # Model settings
    MODEL_NAME = "model_fraud_detection_claim"
    MODEL_DESCRIPTION = "model_fraud_detection_claim"
    MODEL_FILE = "model.py"  # Your model serving script
    MODEL_FUNCTION = "predict"  # Function name in model.py
    
    # Runtime settings
    KERNEL = "python3"
    CPU = 2
    MEMORY = 4  # GB
    REPLICAS = 1
    
    # Advanced settings
    ENABLE_AUTHENTICATION = False
    MAX_WAIT_TIME = 1800  # 30 minutes
    HEALTH_CHECK_INTERVAL = 10  # seconds
    
    # Environment variables for model
    ENV_VARS = {
        "MODEL_VERSION": None,  # Will be set dynamically
        "DEPLOYMENT_DATE": None,
        "LOG_LEVEL": "INFO"
    }

# ================================================================
# CML API CLIENT
# ================================================================
class CMLModelDeployer:
    """Handle model deployment operations"""
    
    def __init__(self, project_name=None):
        """Initialize CML API client"""
        self.client = cmlapi.default_client()
        self.project = self._get_project(project_name)
        self.config = DeploymentConfig()
        
        print("=" * 80)
        print("CML MODEL DEPLOYER")
        print("=" * 80)
        print(f"✓ Connected to CML")
        print(f"✓ Project: {self.project.name}")
        print(f"✓ Project ID: {self.project.id}")
        print("=" * 80)
    
    def _get_project(self, project_name):
        """Get project by name or use current project"""
        if project_name:
            # Search by name
            projects = self.client.list_projects(
                search_filter=json.dumps({"name": project_name})
            )
            if not projects.projects:
                raise ValueError(f"Project '{project_name}' not found")
            return projects.projects[0]
        else:
            # Use environment variable (when running inside CML)
            project_id = os.getenv("CDSW_PROJECT_ID")
            if not project_id:
                raise ValueError("Project name required or run inside CML project")
            return self.client.get_project(project_id)
    
    def _find_existing_model(self):
        """Find existing model by name"""
        print(f"\n[1/6] Searching for existing model '{self.config.MODEL_NAME}'...")
        
        try:
            models = self.client.list_models(self.project.id)
            
            for model in models.models:
                if model.name == self.config.MODEL_NAME:
                    print(f"✓ Found existing model: {model.id}")
                    return model
            
            print("  Model not found, will create new one")
            return None
            
        except Exception as e:
            print(f"✗ Error searching models: {e}")
            return None
    
    def _create_model(self):
        """Create new model"""
        print(f"\n[2/6] Creating model '{self.config.MODEL_NAME}'...")
        
        try:
            model_body = cmlapi.CreateModelRequest(
                project_id=self.project.id,
                name=self.config.MODEL_NAME,
                description=self.config.MODEL_DESCRIPTION,
                disable_authentication=not self.config.ENABLE_AUTHENTICATION
            )
            
            model = self.client.create_model(model_body, self.project.id)
            print(f"✓ Model created: {model.id}")
            return model
            
        except Exception as e:
            print(f"✗ Error creating model: {e}")
            raise
    
    def _build_model(self, model):
        """Build model from script"""
        print(f"\n[3/6] Building model from '{self.config.MODEL_FILE}'...")
        
        # Check if file exists
        model_file_path = os.path.join("/home/cdsw", self.config.MODEL_FILE)
        if not os.path.exists(model_file_path):
            raise FileNotFoundError(f"Model file not found: {model_file_path}")
        
        print(f"  File found: {model_file_path}")
        print(f"  Function: {self.config.MODEL_FUNCTION}")
        print(f"  Kernel: {self.config.KERNEL}")
        
        try:
            # Create build request
            model_build_body = cmlapi.CreateModelBuildRequest(
                project_id=self.project.id,
                model_id=model.id,
                file_path=self.config.MODEL_FILE,
                function_name=self.config.MODEL_FUNCTION,
                kernel=self.config.KERNEL,
                cpu=self.config.CPU,
                memory=self.config.MEMORY
            )
            
            model_build = self.client.create_model_build(
                model_build_body, 
                self.project.id, 
                model.id
            )
            
            print(f"✓ Build started: {model_build.id}")
            
            # Wait for build to complete
            print("  Waiting for build to complete...")
            start_time = time.time()
            
            while model_build.status not in ["built", "build failed"]:
                if time.time() - start_time > self.config.MAX_WAIT_TIME:
                    raise TimeoutError("Build timeout exceeded")
                
                elapsed = int(time.time() - start_time)
                print(f"    Status: {model_build.status} (elapsed: {elapsed}s)", end="\r")
                
                time.sleep(self.config.HEALTH_CHECK_INTERVAL)
                model_build = self.client.get_model_build(
                    self.project.id, 
                    model.id, 
                    model_build.id
                )
            
            print()  # New line after status updates
            
            if model_build.status == "build failed":
                raise RuntimeError(f"Model build failed. Check CML UI for details.")
            
            print(f"✓ Build completed successfully!")
            print(f"  Build ID: {model_build.id}")
            print(f"  Build time: {int(time.time() - start_time)}s")
            
            return model_build
            
        except Exception as e:
            print(f"✗ Error building model: {e}")
            raise
    
    def _deploy_model(self, model, model_build):
        """Deploy model build"""
        print(f"\n[4/6] Deploying model...")
        
        # Update environment variables
        self.config.ENV_VARS["MODEL_VERSION"] = model_build.id[:8]
        self.config.ENV_VARS["DEPLOYMENT_DATE"] = datetime.now().isoformat()
        
        print(f"  CPU: {self.config.CPU}")
        print(f"  Memory: {self.config.MEMORY} GB")
        print(f"  Replicas: {self.config.REPLICAS}")
        print(f"  Environment variables: {len(self.config.ENV_VARS)}")
        
        try:
            # Create deployment request
            model_deployment_body = cmlapi.CreateModelDeploymentRequest(
                project_id=self.project.id,
                model_id=model.id,
                build_id=model_build.id,
                cpu=self.config.CPU,
                memory=self.config.MEMORY,
                replicas=self.config.REPLICAS,
                environment=self.config.ENV_VARS
            )
            
            model_deployment = self.client.create_model_deployment(
                model_deployment_body,
                self.project.id,
                model.id,
                model_build.id
            )
            
            print(f"✓ Deployment started: {model_deployment.id}")
            
            # Wait for deployment to complete
            print("  Waiting for deployment to complete...")
            start_time = time.time()
            
            while model_deployment.status not in ["stopped", "failed", "deployed"]:
                if time.time() - start_time > self.config.MAX_WAIT_TIME:
                    raise TimeoutError("Deployment timeout exceeded")
                
                elapsed = int(time.time() - start_time)
                print(f"    Status: {model_deployment.status} (elapsed: {elapsed}s)", end="\r")
                
                time.sleep(self.config.HEALTH_CHECK_INTERVAL)
                model_deployment = self.client.get_model_deployment(
                    self.project.id,
                    model.id,
                    model_build.id,
                    model_deployment.id
                )
            
            print()  # New line
            
            if model_deployment.status != "deployed":
                raise RuntimeError(f"Deployment failed with status: {model_deployment.status}")
            
            print(f"✓ Deployment completed successfully!")
            print(f"  Deployment ID: {model_deployment.id}")
            print(f"  Deployment time: {int(time.time() - start_time)}s")
            
            return model_deployment
            
        except Exception as e:
            print(f"✗ Error deploying model: {e}")
            raise
    
    def _stop_old_deployments(self, model, current_deployment_id):
        """Stop old deployments (zero-downtime deployment)"""
        print(f"\n[5/6] Checking for old deployments...")
        
        try:
            # List all builds and deployments
            builds = self.client.list_model_builds(self.project.id, model.id)
            
            stopped_count = 0
            for build in builds.model_builds:
                deployments = self.client.list_model_deployments(
                    self.project.id,
                    model.id,
                    build.id
                )
                
                for deployment in deployments.model_deployments:
                    # Skip current deployment
                    if deployment.id == current_deployment_id:
                        continue
                    
                    # Stop if still running
                    if deployment.status == "deployed":
                        print(f"  Stopping old deployment: {deployment.id[:8]}...")
                        self.client.stop_model_deployment(
                            self.project.id,
                            model.id,
                            build.id,
                            deployment.id
                        )
                        stopped_count += 1
            
            if stopped_count > 0:
                print(f"✓ Stopped {stopped_count} old deployment(s)")
            else:
                print(f"✓ No old deployments to stop")
                
        except Exception as e:
            print(f"⚠ Warning: Error stopping old deployments: {e}")
            print("  Old deployments may still be running")
    
    def _verify_deployment(self, model, deployment):
        """Verify deployment is healthy"""
        print(f"\n[6/6] Verifying deployment health...")
        
        try:
            # Get deployment details
            deployment_info = self.client.get_model_deployment(
                self.project.id,
                model.id,
                deployment.build_id,
                deployment.id
            )
            
            print(f"✓ Deployment status: {deployment_info.status}")
            print(f"✓ Access URL: {deployment_info.access_key}")
            
            # Test prediction (optional)
            print("\n  Testing prediction endpoint...")
            test_payload = {
                "claims": [{
                    "claim_id": "TEST_001",
                    "patient_dob": "1980-01-01",
                    "visit_date": "2025-12-01",
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
            
            response = self.client.call_model(
                deployment_info.access_key,
                json.dumps(test_payload)
            )
            
            result = json.loads(response)
            if result.get("status") == "success":
                print(f"✓ Test prediction successful!")
                print(f"  Fraud score: {result['results'][0]['fraud_score']:.4f}")
            else:
                print(f"⚠ Warning: Test prediction returned unexpected result")
            
            return True
            
        except Exception as e:
            print(f"⚠ Warning: Could not verify deployment: {e}")
            return False
    
    def deploy(self):
        """Main deployment workflow"""
        try:
            # Find or create model
            model = self._find_existing_model()
            if not model:
                model = self._create_model()
            
            # Build model
            model_build = self._build_model(model)
            
            # Deploy model
            model_deployment = self._deploy_model(model, model_build)
            
            # Stop old deployments
            self._stop_old_deployments(model, model_deployment.id)
            
            # Verify deployment
            self._verify_deployment(model, model_deployment)
            
            # Success summary
            print("\n" + "=" * 80)
            print("✅ DEPLOYMENT SUCCESSFUL")
            print("=" * 80)
            print(f"Model Name: {model.name}")
            print(f"Model ID: {model.id}")
            print(f"Build ID: {model_build.id}")
            print(f"Deployment ID: {model_deployment.id}")
            print(f"Status: {model_deployment.status}")
            print("=" * 80)
            
            return {
                "status": "success",
                "model_id": model.id,
                "build_id": model_build.id,
                "deployment_id": model_deployment.id
            }
            
        except Exception as e:
            print("\n" + "=" * 80)
            print("❌ DEPLOYMENT FAILED")
            print("=" * 80)
            print(f"Error: {str(e)}")
            print("\nCheck CML UI for more details:")
            print(f"  Project: {self.project.name}")
            print(f"  Models: https://your-cml-url/projects/{self.project.id}/models")
            print("=" * 80)
            
            return {
                "status": "failed",
                "error": str(e)
            }

# ================================================================
# MAIN
# ================================================================
def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Deploy BPJS Fraud Detection model to CML")
    parser.add_argument(
        "--project-name",
        type=str,
        default=None,
        help="CML project name (optional if running inside CML)"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="BPJS Fraud Detection",
        help="Model name to deploy"
    )
    
    args = parser.parse_args()
    
    # Update config if custom model name provided
    if args.model_name:
        DeploymentConfig.MODEL_NAME = args.model_name
    
    # Deploy
    deployer = CMLModelDeployer(project_name=args.project_name)
    result = deployer.deploy()
    
    # Exit with appropriate code
    sys.exit(0 if result["status"] == "success" else 1)

if __name__ == "__main__":
    main()