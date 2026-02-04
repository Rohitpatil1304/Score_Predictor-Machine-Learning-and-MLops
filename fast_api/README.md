# T20 Score Predictor - FastAPI

A REST API to predict the final score in a T20 cricket match based on the current match situation.

## Features

- Predict final T20 score based on current match state
- Supports 10 international cricket teams
- Validates input data for teams and cities
- Returns predicted score with confidence range

## Installation

1. Install dependencies:
```bash
pip install fastapi uvicorn pandas scikit-learn xgboost
```

2. Make sure the trained model exists at `model/pipe.pkl`

## Running the API

```bash
# From the project root directory
cd fast_api
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Or run directly:
```bash
python main.py
```

## API Endpoints

### Root
- **GET** `/` - Welcome message and available endpoints

### Health Check
- **GET** `/health` - Check if API and model are running

### Teams & Cities
- **GET** `/teams` - List of valid teams
- **GET** `/cities` - List of valid cities

### Prediction
- **POST** `/predict` - Predict final score

## Usage Example

### Request
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
           "batting_team": "India",
           "bowling_team": "Australia",
           "city": "Mumbai",
           "current_score": 85,
           "overs": 10.3,
           "wickets": 2,
           "last_five_overs_runs": 45
         }'
```

### Response
```json
{
    "batting_team": "India",
    "bowling_team": "Australia",
    "city": "Mumbai",
    "current_score": 85,
    "overs_completed": 10.3,
    "wickets_fallen": 2,
    "predicted_score": 175,
    "score_range": "165 - 185"
}
```

## Valid Teams
- Australia
- India
- Bangladesh
- New Zealand
- South Africa
- England
- West Indies
- Afghanistan
- Pakistan
- Sri Lanka

## API Documentation

Once the server is running, visit:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`
