import pickle
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import os

# Initialize FastAPI app
app = FastAPI(
    title="T20 Score Predictor API",
    description="API to predict the final score in a T20 cricket match based on current match situation",
    version="1.0.0"
)

# Load the model
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model", "pipe.pkl")

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    print(f"Model loaded successfully from {MODEL_PATH}")
except FileNotFoundError:
    raise RuntimeError(f"Model file not found at {MODEL_PATH}. Please ensure the model is trained and saved.")


# Request schema
class MatchInput(BaseModel):
    current_score: int = Field(..., ge=0, description="Current score of batting team")
    overs: float = Field(..., ge=0.1, le=20.0, description="Overs completed (e.g., 10.3 for 10 overs and 3 balls)")
    wickets_left: int = Field(..., ge=0, le=10, description="Wickets remaining (0-10)")
    last_five_overs_runs: int = Field(..., ge=0, description="Runs scored in last 5 overs (30 balls)")

    class Config:
        json_schema_extra = {
            "example": {
                "current_score": 85,
                "overs": 10.3,
                "wickets_left": 8,
                "last_five_overs_runs": 45
            }
        }


# Response schema
class PredictionResponse(BaseModel):
    current_score: int
    overs_completed: float
    wickets_left: int
    predicted_score: int
    score_range: str

    class Config:
        json_schema_extra = {
            "example": {
                "current_score": 85,
                "overs_completed": 10.3,
                "wickets_left": 8,
                "predicted_score": 175,
                "score_range": "165 - 185"
            }
        }


@app.get("/")
def root():
    """Root endpoint with API information"""
    return {
        "message": "Welcome to T20 Score Predictor API",
        "endpoints": {
            "/predict": "POST - Predict final score",
            "/health": "GET - Health check"
        }
    }


@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict_score(match_input: MatchInput):
    """
    Predict the final score in a T20 match.
    
    - **current_score**: Current score of batting team
    - **overs**: Overs completed (e.g., 10.3 means 10 overs and 3 balls)
    - **wickets_left**: Wickets remaining (0-10)
    - **last_five_overs_runs**: Runs scored in last 5 overs
    """
    try:
        # Calculate derived features
        over_part = int(match_input.overs)
        ball_part = round((match_input.overs - over_part) * 10)
        
        # Validate ball number
        if ball_part > 6:
            raise HTTPException(
                status_code=400,
                detail="Invalid overs format. Ball number cannot exceed 6 (e.g., use 10.6 not 10.7)"
            )
        
        balls_bowled = (over_part * 6) + ball_part
        balls_left = max(0, 120 - balls_bowled)
        
        # Calculate current run rate (CRR = runs per over)
        if balls_bowled > 0:
            crr = (match_input.current_score * 6) / balls_bowled
        else:
            crr = 0
        
        # Create input dataframe for prediction
        # Model features: current_score, balls_left, wickets_left, crr, last_five
        input_df = pd.DataFrame({
            'current_score': [match_input.current_score],
            'balls_left': [balls_left],
            'wickets_left': [match_input.wickets_left],
            'crr': [crr],
            'last_five': [match_input.last_five_overs_runs]
        })
        
        # Make prediction
        predicted_score = int(model.predict(input_df)[0])
        
        # Ensure predicted score is at least current score
        predicted_score = max(predicted_score, match_input.current_score)
        
        # Calculate score range (±10 runs)
        score_range = f"{predicted_score - 10} - {predicted_score + 10}"
        
        return PredictionResponse(
            current_score=match_input.current_score,
            overs_completed=match_input.overs,
            wickets_left=match_input.wickets_left,
            predicted_score=predicted_score,
            score_range=score_range
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
