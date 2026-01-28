import pandas as pd
import numpy as np
import pickle
import json
import os
import copy
import yaml

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, ParameterSampler, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error
from xgboost import XGBRegressor
import mlflow
import mlflow.sklearn
import dagshub



dagshub.init(
    repo_owner="Rohitpatil1304",
    repo_name="Score_Predictor-Machine-Learning-and-MLops-",
    mlflow=True
)

mlflow.set_tracking_uri(
    "https://dagshub.com/Rohitpatil1304/Score_Predictor-Machine-Learning-and-MLops-.mlflow"
)

mlflow.set_experiment("Prediction_Model")

# Load parameters from params.yaml
with open("params.yaml", "r") as f:
    params_config = yaml.safe_load(f)

train_params = params_config["train"]
model_params = params_config["model"]
param_dist_config = params_config["param_dist"]

df = pd.read_pickle("data/interim/dataset_level3_feature_ready.pkl")

X = df.drop(columns=["runs_x"])
y = df["runs_x"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=train_params["test_size"], random_state=train_params["random_state"]
)


preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"),
            ["batting_team", "bowling_team", "city"]
        )
    ],
    remainder="passthrough"
)


pipe = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("model", XGBRegressor(
        objective=model_params["objective"],
        random_state=model_params["random_state"],
        eval_metric=model_params["eval_metric"],
        tree_method=model_params["tree_method"]
    ))
])

# Build param_dist from params.yaml
param_dist = {
    f"model__{key}": value
    for key, value in param_dist_config.items()
}


with mlflow.start_run(run_name="Parent_Run"):

    mlflow.log_param("model", "XGBRegressor")
    mlflow.log_param("search_type", "Random ParameterSampler")
    mlflow.log_param("cv", train_params["cv_folds"])

    best_score = -np.inf
    best_params = None

    param_list = list(ParameterSampler(
        param_dist,
        n_iter=train_params["n_trials"],
        random_state=train_params["random_state"]
    ))

    for i, params in enumerate(param_list, start=1):

        with mlflow.start_run(run_name=f"trial_{i}", nested=True):

            pipe.set_params(**params)

            cv_scores = cross_val_score(
                pipe,
                X_train,
                y_train,
                cv=train_params["cv_folds"],
                scoring="r2",
                n_jobs=-1
            )

            mean_cv = np.mean(cv_scores)

            mlflow.log_params(params)
            mlflow.log_metric("cv_r2_mean", mean_cv)

            print(f"Trial {i} | CV R2 = {mean_cv:.4f}")

            if mean_cv > best_score:
                best_score = mean_cv
                best_params = params


    pipe.set_params(**best_params)
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)

    test_r2 = r2_score(y_test, y_pred)
    test_mae = mean_absolute_error(y_test, y_pred)

    mlflow.log_metric("best_cv_r2", best_score)
    mlflow.log_metric("test_r2", test_r2)
    mlflow.log_metric("test_mae", test_mae)

    mlflow.log_params(best_params)


    mlflow.sklearn.log_model(pipe, artifact_path="best_model")

    os.makedirs("model", exist_ok=True)

    with open("model/pipe.pkl", "wb") as f:
        pickle.dump(pipe, f)

    with open("model/best_params.json", "w") as f:
        json.dump(best_params, f, indent=4)

    print("\n✅ BEST MODEL TRAINED")
    print("Best CV R2:", best_score)
    print("Test R2:", test_r2)
    print("Test MAE:", test_mae)
    print("Best Params:", best_params)
