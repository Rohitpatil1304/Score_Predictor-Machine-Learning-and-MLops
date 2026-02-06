#  T20 Score Predictor

> **A Machine Learning-powered application to predict the final score in T20 cricket matches based on current match situations.**

[![MLflow](https://img.shields.io/badge/MLflow-Tracking-blue)](https://dagshub.com/Rohitpatil1304/Score_Predictor-Machine-Learning-and-MLops-.mlflow)
[![DVC](https://img.shields.io/badge/DVC-Data%20Version%20Control-purple)](https://dvc.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Backend-green)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Frontend-red)](https://streamlit.io/)

---

##  Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Architecture](#project-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [MLOps Pipeline](#mlops-pipeline)
- [Model Details](#model-details)
- [API Documentation](#api-documentation)
- [Project Structure](#project-structure)

---

##  Overview

This project predicts the **final score** in a T20 cricket match based on the current match situation. It uses an **XGBoost Regressor** model trained on international T20 cricket match data. The complete MLOps pipeline includes:

- **Data versioning** with DVC
- **Experiment tracking** with MLflow & DagsHub
- **Model registry** for production deployment
- **REST API** with FastAPI
- **Interactive UI** with Streamlit

### Supported Teams
The model is trained on data from **10 international cricket teams**:
- Australia, India, Bangladesh, New Zealand, South Africa
- England, West Indies, Afghanistan, Pakistan, Sri Lanka

---

##  Features

| Feature | Description |
|---------|-------------|
|  **Score Prediction** | Predicts final T20 score with 10 run confidence range |
|  **Real-time Stats** | Shows current run rate (CRR), balls left, required rate |
|  **DVC Pipeline** | Reproducible ML pipeline with data versioning |
|  **MLflow Tracking** | Experiment tracking with DagsHub integration |
|  **FastAPI Backend** | Production-ready REST API |
|  **Streamlit UI** | Beautiful cricket-themed interactive frontend |
|  **Model Registry** | MLflow model versioning and staging |

---

##  Tech Stack

### Machine Learning
- **XGBoost** - Gradient boosting regressor
- **Scikit-learn** - ML utilities and pipelines
- **Pandas/NumPy** - Data manipulation

### MLOps
- **DVC** - Data version control & pipeline orchestration
- **MLflow** - Experiment tracking & model registry
- **DagsHub** - Remote MLflow server & collaboration

### Backend & Frontend
- **FastAPI** - REST API framework
- **Streamlit** - Interactive web application
- **Pydantic** - Data validation

---

##  Project Architecture

`

                        T20 Score Predictor                       

                                                                  
                 
    Raw Data      Processed      Feature           
    (Pickle)            Data            Engineering        
                 
                                                               
                         
                                                                 
                                                                 
                                             
                     Model Training                            
                      (XGBoost)                                
                     + Grid Search                             
                                             
                                                                 
                                    
                                                              
                    
        MLflow        Model       Model                   
       Tracking       (.pkl)     Registry                 
       (DagsHub)                 (Staging)                
                    
                                                                
                                     
                                                              
                                
        FastAPI      Streamlit                    
        Backend                 Frontend                    
       (Port 8000)             (Port 8501)                  
                                
                                                                  

`

---

##  Installation

### Prerequisites
- Python 3.8+
- Git
- pip

### Step 1: Clone the Repository
`ash
git clone https://github.com/Rohitpatil1304/Score_Predictor-Machine-Learning-and-MLops-.git
cd T20_Score_Predictor
`

### Step 2: Create Virtual Environment
`ash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
`

### Step 3: Install Dependencies
`ash
pip install -r requirements.txt
pip install -e .
`

### Step 4: Install Additional Packages
`ash
pip install fastapi uvicorn streamlit xgboost mlflow dagshub dvc
`

---

##  Usage

### Option 1: Run the Complete Pipeline

`ash
# Run DVC pipeline (data processing  feature engineering  training  register)
dvc repro
`

### Option 2: Run Individual Stages

`ash
# Data Preprocessing
python src/data/Data_Preprocessing.py

# Feature Engineering
python src/features/Feature_Engineering.py

# Model Training
python src/models/Training_Model.py

# Register Model
python src/models/register_model.py
`

### Option 3: Start the Application

**Terminal 1 - Start FastAPI Backend:**
`ash
cd fast_api
uvicorn main:app --reload --host 0.0.0.0 --port 8000
`

**Terminal 2 - Start Streamlit Frontend:**
`ash
streamlit run frontend/app.py
`

**Access the Application:**
-  **Streamlit UI**: http://localhost:8501
-  **FastAPI Docs**: http://localhost:8000/docs
-  **MLflow UI**: https://dagshub.com/Rohitpatil1304/Score_Predictor-Machine-Learning-and-MLops-.mlflow

---

##  MLOps Pipeline

### DVC Pipeline Stages

The pipeline is defined in `dvc.yaml`:

| Stage | Description | Input | Output |
|-------|-------------|-------|--------|
| **Data_Preprocessing** | Clean raw data, filter teams, extract ball-by-ball data | `data/raw/dataset_level_new.pkl` | `data/processed/dataset_level2.pkl` |
| **Feature_Engineering** | Create ML features (CRR, balls left, last 5 overs runs) | `data/processed/dataset_level2.pkl` | `data/interim/dataset_level3_feature_ready.pkl` |
| **Training_Model** | Grid search with XGBoost, track with MLflow | Feature-ready data | `model/pipe.pkl`, `reports/best_params.json` |
| **Register_Model** | Register best model to MLflow Model Registry | Trained model | Model version in registry |

### Pipeline Commands

`ash
# Reproduce entire pipeline
dvc repro

# Run specific stage
dvc repro -s Training_Model

# View pipeline DAG
dvc dag

# Check pipeline status
dvc status
`

### MLflow Experiment Tracking

All experiments are tracked on **DagsHub**:
- **Tracking URI**: `https://dagshub.com/Rohitpatil1304/Score_Predictor-Machine-Learning-and-MLops-.mlflow`
- **Experiment Name**: `Tracker`

Tracked Metrics:
- `test_r2` - R score on test set
- `test_mae` - Mean Absolute Error
- Best hyperparameters for each trial

---

##  Model Details

### Algorithm
**XGBoost Regressor** with the following configuration:

`yaml
# Base Model Parameters (params.yaml)
model:
  objective: "reg:squarederror"
  eval_metric: "rmse"
  tree_method: "hist"
  random_state: 1
`

### Best Hyperparameters

After grid search optimization:

| Parameter | Value |
|-----------|-------|
| `n_estimators` | 300 |
| `learning_rate` | 0.05 |
| `max_depth` | 8 |
| `subsample` | 0.8 |
| `colsample_bytree` | 0.8 |
| `reg_alpha` | 0 |
| `reg_lambda` | 1 |

### Input Features (5 Features)

| Feature | Description | Type |
|---------|-------------|------|
| `current_score` | Runs scored so far | Integer |
| `balls_left` | Remaining balls (out of 120) | Integer |
| `wickets_left` | Wickets remaining (0-10) | Integer |
| `crr` | Current Run Rate | Float |
| `last_five` | Runs in last 5 overs (30 balls) | Integer |

### Output
- **Predicted Score**: Final predicted total score
- **Score Range**: Confidence interval (10 runs)

---

##  API Documentation

### Base URL
`
http://localhost:8000
`

### Endpoints

#### Health Check
`http
GET /health
`
**Response:**
`json
{
  "status": "healthy",
  "model_loaded": true
}
`

#### Predict Score
`http
POST /predict
Content-Type: application/json
`

**Request Body:**
`json
{
  "current_score": 85,
  "overs": 10.3,
  "wickets_left": 8,
  "last_five_overs_runs": 45
}
`

**Response:**
`json
{
  "current_score": 85,
  "overs_completed": 10.3,
  "wickets_left": 8,
  "predicted_score": 175,
  "score_range": "165 - 185"
}
`

### Example cURL Request
`ash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "current_score": 85,
    "overs": 10.3,
    "wickets_left": 8,
    "last_five_overs_runs": 45
  }'
`

---

##  Project Structure

`
T20_Score_Predictor/

  dvc.yaml              # DVC pipeline definition
  params.yaml           # Training hyperparameters
  requirements.txt      # Python dependencies
  setup.py              # Package setup
  Makefile              # Build automation

  data/
    raw/                 # Original match data (pickle)
    processed/           # Cleaned & preprocessed data
    interim/             # Feature-engineered data

  src/
     data/
       Data_Preprocessing.py    # Data cleaning & filtering
       Data extracting.py       # Raw data extraction
   
     features/
       Feature_Engineering.py   # Feature creation
   
     models/
       Training_Model.py        # Model training with MLflow
       register_model.py        # MLflow model registration
   
     visualization/
        visualize.py             # Visualization utilities

  model/
    pipe.pkl             # Trained model pipeline
    best_params.json     # Best hyperparameters

  fast_api/
    main.py              # FastAPI application
    README.md            # API documentation

  frontend/
    app.py               # Streamlit application

  notebooks/
    Data Ingestion.ipynb
    Feature_Extraction.ipynb
    Model_training.ipynb

  mlruns/               # MLflow local tracking

  reports/
     best_params.json     # Best model parameters
     figures/             # Generated plots
`

---

##  Data Pipeline Details

### Stage 1: Data Preprocessing (`src/data/Data_Preprocessing.py`)

**What it does:**
1. Loads raw match data from pickle file
2. Drops unnecessary columns (meta data, supersubs, etc.)
3. Filters to keep male T20 matches only (20 overs)
4. Extracts ball-by-ball delivery data
5. Identifies bowling team from batting team
6. Filters to keep only top 10 international teams
7. Saves processed data to `data/processed/dataset_level2.pkl`

**Columns in output:**
- `match_id`, `batting_team`, `bowling_team`, `ball`, `runs`, `player_dismissed`, `city`, `venue`

### Stage 2: Feature Engineering (`src/features/Feature_Engineering.py`)

**Features created:**
| Feature | Calculation |
|---------|-------------|
| `current_score` | Cumulative sum of runs |
| `balls_bowled` | (overs  6) + balls |
| `balls_left` | 120 - balls_bowled |
| `wickets_left` | 10 - cumulative wickets |
| `crr` | (current_score  6) / balls_bowled |
| `last_five` | Rolling sum of last 30 balls |

**Data filtering:**
- Cities with 600+ deliveries
- Removes null values

### Stage 3: Model Training (`src/models/Training_Model.py`)

**Process:**
1. Loads feature-ready data
2. Splits into train/test (90/10)
3. Creates XGBoost pipeline
4. Performs grid search over all parameter combinations
5. Tracks each trial as nested MLflow run
6. Saves best model and parameters

**Hyperparameter Search Space:**
`yaml
param_dist:
  n_estimators: [500, 1000]
  learning_rate: [0.01, 0.1]
  max_depth: [6, 8, 10]
  subsample: [0.7]
  colsample_bytree: [0.8]
`

### Stage 4: Model Registration (`src/models/register_model.py`)

**Registers model to MLflow Model Registry with:**
- Model name: `T20-Score-Predictor`
- Metrics and parameters logged
- Model transitioned to "Staging" stage
- Tags for model type and task

---

##  Frontend Features (Streamlit)

- **Cricket-themed dark UI** with stadium lights effect
- **Real-time input validation**
- **Live stats display** (CRR, balls left, wickets)
- **API health status** in sidebar
- **Prediction insights** (accelerate, steady, etc.)
- **Animated prediction reveal** with balloons

---

##  Configuration

### Training Parameters (`params.yaml`)

`yaml
train:
  test_size: 0.1           # 10% test split
  random_state: 1          # Reproducibility
  cv_folds: 5              # Cross-validation folds
  n_trials: 10             # Number of trials

model:
  objective: "reg:squarederror"
  random_state: 1
  eval_metric: "rmse"
  tree_method: "hist"
`

### DagsHub Configuration

Set environment variables for remote tracking:
`ash
export MLFLOW_TRACKING_URI=https://dagshub.com/Rohitpatil1304/Score_Predictor-Machine-Learning-and-MLops-.mlflow
export MLFLOW_TRACKING_USERNAME=Rohitpatil1304
export MLFLOW_TRACKING_PASSWORD=<your_token>
`

---

- GitHub: [@Rohitpatil1304](https://github.com/Rohitpatil1304)
- DagsHub: [Score_Predictor-Machine-Learning-and-MLops-](https://dagshub.com/Rohitpatil1304/Score_Predictor-Machine-Learning-and-MLops)

---

##  Acknowledgments

- [Cookiecutter Data Science](https://drivendata.github.io/cookiecutter-data-science/) - Project structure
- [DagsHub](https://dagshub.com/) - MLflow hosting & collaboration
- [DVC](https://dvc.org/) - Data version control
- [MLflow](https://mlflow.org/) - Experiment tracking
- Cricket data from international T20 matches
