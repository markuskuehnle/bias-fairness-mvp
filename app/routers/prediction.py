from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import pandas as pd
import random

from app.services.data_loader import load_candidates
from app.services.prediction_service import load_model, predict_candidate


router = APIRouter()

# Load the pre-trained XGBoost model at the startup
xgb_model = load_model()

class PredictionRequest(BaseModel):
    candidate_id: int
    updated_features: dict  # This will contain user-modified features


def get_age_group(age: int) -> str:
    """Return the age group based on predefined bins."""
    if age < 30:
        return "<30"
    elif 30 <= age <= 50:
        return "30-50"
    else:
        return ">50"


@router.post("/predict/update", tags=["Prediction"])
def update_prediction(request: PredictionRequest):
    try:
        # Load the candidate data
        candidates = load_candidates()

        # Find the candidate in the dataset
        candidate_row = candidates[candidates["Candidate_ID"] == request.candidate_id]
        if candidate_row.empty:
            raise HTTPException(status_code=404, detail="Candidate not found.")

        candidate_row = candidate_row.iloc[0].copy()  # Extract row as mutable Series

        # Apply updates to candidate row
        for feature, value in request.updated_features.items():
            candidate_row[feature] = value

        # Validate Age if modified
        if "Age" in request.updated_features:
            age_value = request.updated_features["Age"]
            
            if isinstance(age_value, str) and "-" in age_value:
                # If the value is a string range, randomly select an age
                min_age, max_age = map(int, age_value.split("-"))
                candidate_row["Age"] = random.randint(min_age, max_age)
            elif isinstance(age_value, int):
                candidate_row["Age"] = age_value
            else:
                raise HTTPException(status_code=400, detail=f"Invalid Age format: {age_value}")

            # Ensure Age is valid before assigning AgeGroup
            if "Age" in candidate_row and not pd.isnull(candidate_row["Age"]):
                candidate_row["AgeGroup"] = get_age_group(candidate_row["Age"])
            else:
                raise HTTPException(status_code=400, detail="Missing or invalid Age value.")

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
