from fastapi import APIRouter, HTTPException
from app.services.data_loader import load_candidates
from app.services.prediction_service import load_model, predict_candidate
from pydantic import BaseModel
import random


router = APIRouter()

# Load the pre-trained XGBoost model at the startup
xgb_model = load_model()

class PredictionRequest(BaseModel):
    candidate_id: int
    updated_features: dict  # This will contain user-modified features


@router.post("/predict/update", tags=["Prediction"])
def update_prediction(request: PredictionRequest):
    """
    Update AI prediction when a protected feature (e.g., gender, age, race) is modified.
    
    Parameters:
    - candidate_id (int): The ID of the candidate.
    - updated_features (dict): Key-value pairs of updated features.

    Returns:
    JSON: Updated prediction results.
    """
    try:
        # Load the candidate data
        candidates = load_candidates()
        print(candidates)
        # Find the candidate in the dataset
        candidate_row = candidates[candidates["Candidate_ID"] == request.candidate_id]
        if candidate_row.empty:
            raise HTTPException(status_code=404, detail="Candidate not found.")

        candidate_row = candidate_row.iloc[0].copy()  # Extract row as mutable Series

        # Apply updates to candidate row
        for feature, value in request.updated_features.items():
            candidate_row[feature] = value

        # If Age was modified, randomly pick an age within the given range
        if "Age" in request.updated_features:
            age_range = request.updated_features["Age"]
            
            if isinstance(age_range, str) and "-" in age_range:
                # If the value is a string with a range, split and pick a random age
                min_age, max_age = map(int, age_range.split("-"))
                candidate_row["Age"] = random.randint(min_age, max_age)
            elif isinstance(age_range, int):
                # If the frontend already sent an integer, use it directly
                candidate_row["Age"] = age_range
            else:
                raise HTTPException(status_code=400, detail=f"Invalid Age format: {age_range}")

        # Recalculate prediction
        prediction_result = predict_candidate(candidate_row, xgb_model)

        return {
            "candidate_id": request.candidate_id,
            "prediction_probability": round(prediction_result["prediction_probability"], 2),
            "is_good_fit": prediction_result["is_good_fit"],
            "top_features": prediction_result["top_features"]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{e.__traceback__.tb_lineno}, {str(type(e).__name__)}: {str(e)}")
    

@router.get("/predict/{candidate_id}", tags=["Prediction"])
def predict_candidate_api(candidate_id: int):
    """
    Predict if a selected candidate is a good fit.

    Parameters:
    candidate_id (int): The ID of the candidate.

    Returns:
    JSON: Prediction result for the candidate.
    """
    try:
        # Load candidate data
        candidates = load_candidates()
        # Ensure candidate_id exists in Candidate_ID column
        candidate_row = candidates[candidates["Candidate_ID"] == candidate_id]
        if candidate_row.empty:
            raise HTTPException(status_code=404, detail="Candidate not found.")
        
        # Extract the row as a Series
        candidate_row = candidate_row.iloc[0]

        # Perform the prediction
        prediction_result = predict_candidate(candidate_row, xgb_model)

        return {
            "candidate_id": candidate_id,
            **prediction_result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{e.__traceback__.tb_lineno},{str(type(e).__name__)}: {str(e)}")
