#  T20 Score Predictor

> **A Machine Learning-powered application to predict the final score in T20 cricket matches based on current match situations.**

![MLflow](https://img.shields.io/badge/MLflow-Tracking-blue)
[![DVC](https://img.shields.io/badge/DVC-Data%20Version%20Control-purple)](https://dvc.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Backend-green)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Frontend-red)](https://streamlit.io/)

---

##  Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Architecture](#pipeline-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [MLOps Pipeline](#mlops-pipeline)
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

##  Pipeline Architecture
![Pipeline](https://github.com/Rohitpatil1304/Score_Predictor-Machine-Learning-and-MLops/blob/master/src/visualization/Pipeline%20Structure.jpeg)
---

##  Installation

### Prerequisites
- Python 3.8+
- Git
- pip

### Step 1: Clone the Repository
`ash
git clone https://github.com/Rohitpatil1304/Score_Predictor-Machine-Learning-and-MLops.git
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

All experiments are tracked on **DagsHub** and **MLflow**:
- **Tracking URI**: `https://dagshub.com/Rohitpatil1304/Score_Predictor-Machine-Learning-and-MLops.mlflow`
- **Experiment Name**: `Tracker`

Tracked Metrics:
- `test_r2` - R score on test set
- `test_mae` - Mean Absolute Error
- Best hyperparameters for each trial

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

  model/
    pipe.pkl             # Trained model pipeline

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

- GitHub: [@Rohitpatil1304](https://github.com/Rohitpatil1304)
- DagsHub: [Score_Predictor-Machine-Learning-and-MLops-](https://dagshub.com/Rohitpatil1304/Score_Predictor-Machine-Learning-and-MLops)

---

##  Acknowledgments

- [Cookiecutter Data Science](https://drivendata.github.io/cookiecutter-data-science/) - Project structure
- [DagsHub](https://dagshub.com/) - MLflow hosting & collaboration
- [DVC](https://dvc.org/) - Data version control
- [MLflow](https://mlflow.org/) - Experiment tracking
- Cricket data from international T20 matches
