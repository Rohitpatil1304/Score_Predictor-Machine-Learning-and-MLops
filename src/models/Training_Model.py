import pandas as pd
import numpy as np
import pickle
import json
import os
import copy
import yaml

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.metrics import r2_score, mean_absolute_error

from xgboost import XGBRegressor

import mlflow
import mlflow.sklearn
import dagshub


# =========================
# DAGSHUB + MLFLOW SETUP
# =========================
dagshub.init(
    repo_owner="Rohitpatil1304",
    repo_name="Score_Predictor-Machine-Learning-and-MLops-",
    mlflow=True
)

mlflow.set_tracking_uri(
    "https://dagshub.com/Rohitpatil1304/Score_Predictor-Machine-Learning-and-MLops-.mlflow"
)

mlflow.set_experiment("Tracker")


# =========================
# LOAD PARAMS
# =========================
with open("params.yaml", "r") as f:
    params_config = yaml.safe_load(f)

train_params = params_config["train"]
model_params = params_config["model"]
param_dist_config = params_config["param_dist"]


# =========================
# LOAD DATA
# =========================
df = pd.read_pickle("data/interim/dataset_level3_feature_ready.pkl")

df.drop(columns=["batting_team", "bowling_team", "city"], inplace=True)

X = df.drop(columns=["runs_x"])
y = df["runs_x"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=train_params["test_size"],
    random_state=train_params["random_state"]
)


# =========================
# PIPELINE
# =========================
pipe = Pipeline(steps=[
    ("model", XGBRegressor(
        objective=model_params["objective"],
        random_state=model_params["random_state"],
        eval_metric=model_params["eval_metric"],
        tree_method=model_params["tree_method"]
    ))
])


# =========================
# ALL PARAMETER COMBINATIONS
# =========================
param_grid = {
    f"model__{key}": value
    for key, value in param_dist_config.items()
}

param_list = list(ParameterGrid(param_grid))
print(f"Total parameter combinations: {len(param_list)}")


# =========================
# TRAINING + NESTED MLFLOW
# =========================
with mlflow.start_run(run_name="Parent_Run"):

    mlflow.log_param("model", "XGBRegressor")
    mlflow.log_param("search_type", "GridSearch_All_Combinations")
    mlflow.log_param("total_trials", len(param_list))

    best_score = -np.inf
    best_params = None
    best_model = None

    for i, params in enumerate(param_list, start=1):

        with mlflow.start_run(run_name=f"trial_{i}", nested=True):

            # Set parameters
            pipe.set_params(**params)

            # Train
            pipe.fit(X_train, y_train)

            # Predictions
            y_test_pred = pipe.predict(X_test)

            # Metrics
    
            test_r2 = r2_score(y_test, y_test_pred)
            test_mae = mean_absolute_error(y_test, y_test_pred)

            # Log params & metrics
            mlflow.log_params(params)
            mlflow.log_metric("test_r2", test_r2)
            mlflow.log_metric("test_mae", test_mae)

            print(
                f"Trial {i} | "
                f"Test R2: {test_r2:.4f} | "
                f"MAE: {test_mae:.4f}"
            )

            # Track best model
            if test_r2 > best_score:
                best_score = test_r2
                best_params = params
                best_model = copy.deepcopy(pipe)


    # =========================
    # LOG BEST MODEL
    # =========================
    mlflow.log_metric("best_test_r2", best_score)
    mlflow.log_params(best_params)

    mlflow.sklearn.log_model(
        best_model,
        artifact_path="best_model"
    )

    os.makedirs("model", exist_ok=True)

    with open("model/pipe.pkl", "wb") as f:
        pickle.dump(best_model, f)

    with open("reports/best_params.json", "w") as f:
        json.dump(best_params, f, indent=4)

    print("\n==========================")
    print("BEST MODEL TRAINED")
    print("Best Test R2:", best_score)
    print("Best Params:", best_params)
    print("==========================")
