"""
MLflow Model Registration Script

This script registers the best performing model from MLflow tracking server
to the MLflow Model Registry for production deployment.
"""

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import dagshub
import yaml


# Initialize DagsHub integration
dagshub.init(
    repo_owner="Rohitpatil1304",
    repo_name="Score_Predictor-Machine-Learning-and-MLops-",
    mlflow=True
)

# Set MLflow tracking URI
mlflow.set_tracking_uri(
    "https://dagshub.com/Rohitpatil1304/Score_Predictor-Machine-Learning-and-MLops-.mlflow"
)

# Set experiment name
mlflow.set_experiment("Prediction_Model")


def get_best_run(experiment_name="Prediction_Model", metric="test_r2"):
    """
    Get the best run from an MLflow experiment based on a specified metric.
    Only considers parent runs (non-nested runs) that have the model artifact.
    
    Args:
        experiment_name (str): Name of the MLflow experiment
        metric (str): Metric to optimize for (default: "test_r2")
    
    Returns:
        run: Best MLflow run object
    """
    client = MlflowClient()
    
    # Get experiment by name
    experiment = client.get_experiment_by_name(experiment_name)
    
    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' not found.")
    

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="tags.mlflow.parentRunId = ''",
        order_by=[f"metrics.{metric} DESC"],
        max_results=10  # Get top 10 to find one with artifacts
    )
    
    if not runs:
        # If no runs found with parent filter, try without filter
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="",
            order_by=[f"metrics.{metric} DESC"],
            max_results=10
        )
    
    if not runs:
        raise ValueError(f"No runs found in experiment '{experiment_name}'.")
    
  
    best_run = None
    for run in runs:
        try:
            artifacts = client.list_artifacts(run.info.run_id)
            artifact_paths = [artifact.path for artifact in artifacts]
            if "best_model" in artifact_paths:
                best_run = run
                break
        except Exception:
            continue
    
    if best_run is None:
     
        best_run = runs[0]
    
    print(f"Best Run Found:")
    print(f"   Run ID: {best_run.info.run_id}")
    print(f"   Run Name: {best_run.data.tags.get('mlflow.runName', 'N/A')}")
    print(f"   {metric}: {best_run.data.metrics.get(metric, 'N/A')}")
    
    return best_run


def register_best_model(model_name="T20-Score-Predictor", experiment_name="Prediction_Model", metric="test_r2", model_path="model/pipe.pkl"):
    """
    Register the best model from local file to Model Registry.
    Creates a new MLflow run to log and register the model.
    
    Args:
        model_name (str): Name for the registered model
        experiment_name (str): Name of the MLflow experiment
        metric (str): Metric to optimize for
        model_path (str): Path to the saved model pickle file
    
    Returns:
        registered_model_version: The registered model version object
    """
    import pickle
    import os
    import json
    
    client = MlflowClient()
    
   
    best_run = get_best_run(experiment_name, metric)
    
   
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    print(f"\nLoading Model from: {model_path}")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    

    params_path = "model/best_params.json"
    best_params = {}
    if os.path.exists(params_path):
        with open(params_path, "r") as f:
            best_params = json.load(f)
    
    print(f"Registering Model...")
    print(f"Model Name: {model_name}")
    

    with mlflow.start_run(run_name="Model_Registration") as run:
    
        if best_params:
            mlflow.log_params(best_params)
        
        
        test_r2 = best_run.data.metrics.get('test_r2', 0)
        test_mae = best_run.data.metrics.get('test_mae', 0)
        best_cv_r2 = best_run.data.metrics.get('best_cv_r2', 0)
        
        mlflow.log_metric("test_r2", test_r2)
        mlflow.log_metric("test_mae", test_mae)
        mlflow.log_metric("best_cv_r2", best_cv_r2)
        
    
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=model_name
        )
        
        model_uri = f"runs:/{run.info.run_id}/model"
    
    print(f"\nModel Registered Successfully!")
    print(f"   Name: {model_name}")
    print(f"   Model URI: {model_uri}")
    

    try:
        registered_versions = client.search_model_versions(f"name='{model_name}'")
        if registered_versions:
            latest_version = registered_versions[0]
            
          
            client.update_model_version(
                name=model_name,
                version=latest_version.version,
                description=f"XGBoost model for T20 cricket score prediction. "
                           f"Trained with best hyperparameters. "
                           f"Test R2: {test_r2:.4f}, "
                           f"Test MAE: {test_mae:.4f}"
            )
            
 
            client.set_model_version_tag(
                name=model_name,
                version=latest_version.version,
                key="model_type",
                value="XGBoost"
            )
            
            client.set_model_version_tag(
                name=model_name,
                version=latest_version.version,
                key="task",
                value="regression"
            )
            
            print(f"   Version: {latest_version.version}")
            
            return latest_version
    except Exception as e:
        print(f"Warning: Could not update model metadata: {e}")
        return None


def transition_model_stage(model_name="T20-Score-Predictor", version=None, stage="Staging"):
    
    client = MlflowClient()
    

    if version is None:
        latest_versions = client.get_latest_versions(model_name, stages=["None"])
        if not latest_versions:
            raise ValueError(f"No versions found for model '{model_name}'")
        version = latest_versions[0].version
    
    print(f"\nTransitioning Model to {stage}...")
    print(f"   Model: {model_name}")
    print(f"   Version: {version}")
    
    try:
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=stage,
            archive_existing_versions=True  
        )
        
        print(f"Model transitioned to {stage} successfully!")
        
    except Exception as e:
        print(f"Error transitioning model: {e}")
        raise


def load_registered_model(model_name="T20-Score-Predictor", stage="Production"):
    
    model_uri = f"models:/{model_name}/{stage}"
    
    print(f"\nLoading Model...")
    print(f"   Model URI: {model_uri}")
    
    try:
        model = mlflow.sklearn.load_model(model_uri)
        print(f"Model loaded successfully from {stage}!")
        return model
        
    except Exception as e:
        print(f"Error loading model: {e}")
        raise


if __name__ == "__main__":
   
    
    print("=" * 60)
    print("MLflow Model Registration Pipeline")
    print("=" * 60)
    

    MODEL_NAME = "T20-Score-Predictor"
    EXPERIMENT_NAME = "Tracker"  
    METRIC = "test_r2"  
    

    registered_model = register_best_model(
        model_name=MODEL_NAME,
        experiment_name=EXPERIMENT_NAME,
        metric=METRIC
    )
    
    if registered_model is None:
        print("\nWarning: Model registered but version info not available")
        print("Checking for latest version...")
        client = MlflowClient()
        latest_versions = client.search_model_versions(f"name='{MODEL_NAME}'")
        if latest_versions:
            registered_model = latest_versions[0]
    

    if registered_model:
        transition_model_stage(
            model_name=MODEL_NAME,
            version=registered_model.version,
            stage="Staging"
        )
    else:
        print("\nWarning: Could not transition model - version not found")
    

    
    print("\n" + "=" * 60)
    print("Model Registration Complete!")
    print("=" * 60)
    
    print(f"\nModel Summary:")
    print(f"  • Model Name: {MODEL_NAME}")
    print(f"  • Version: {registered_model.version}")
    print(f"  • Current Stage: Staging")
    print(f"\nTo transition to Production, uncomment the code in the script")
    print(f"or run: transition_model_stage('{MODEL_NAME}', {registered_model.version}, 'Production')")
