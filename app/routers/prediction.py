from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
import pandas as pd
import random
from functools import lru_cache

from app.services.data_loader import load_candidates
from app.services.prediction_service import load_model, predict_candidate


router = APIRouter()

# Load the pre-trained XGBoost model at the startup
xgb_model = load_model()
STATIC_PREDICTIONS_PATH = "app/data/static_predictions.parquet"

class PredictionRequest(BaseModel):
    candidate_id: int
    updated_features: dict  # e.g. {"Sex": 1} or {"Age": "50-60"} or {"RaceDesc_Black or African American": 1, "RaceDesc_White": 0, "RaceDesc_Asian": 0}


@lru_cache(maxsize=1)
def load_static_predictions() -> pd.DataFrame:
    """Load and cache the static predictions dataset."""
    df = pd.read_parquet(STATIC_PREDICTIONS_PATH)
    return df

def get_age_group(age: float) -> str:
    if age < 30:
        return "20-30"
    elif age < 40:
        return "30-40"
    elif age < 50:
        return "40-50"
    elif age < 60:
        return "50-60"
    else:
        return ">60"
    

@router.post("/predict/update", tags=["Prediction"])
def update_prediction(request: PredictionRequest, static_predictions: pd.DataFrame = Depends(load_static_predictions)):
    try:
        # Load the candidate data
        candidates = load_candidates()
        
        # Find the candidate in the dataset
        baseline_candidate = candidates[candidates["Candidate_ID"] == request.candidate_id]
        if baseline_candidate.empty:
            raise HTTPException(status_code=404, detail="Candidate not found.")

        baseline_candidate = baseline_candidate.iloc[0].copy()  # Extract row as mutable Series
        candidate_prediction_rows = static_predictions[static_predictions["Candidate_ID"] == request.candidate_id]

        # Define the set of modifiable attributes (keys expected in updated_features)
        modifiable_attributes = ["Sex", "Age", "RaceDesc_White", "RaceDesc_Black or African American", "RaceDesc_Asian"]
        
        # Compare the original values with the provided updated_features to determine which attribute changed.

        # Determine which attributes are different.
        differences = {}
        for key, new_val in request.updated_features.items():
            if key not in modifiable_attributes:
                continue
            if key == "Age":
                # Convert baseline candidate's Age (numeric) to an age group string.
                baseline_age_group = get_age_group(float(baseline_candidate.get("Age")))
                # Compare with the new value (which should be a string like "40-50").
                if baseline_age_group != str(new_val):
                    differences[key] = new_val
            else:
                # For Sex and race columns, compare as strings.
                if str(baseline_candidate.get(key)) != str(new_val):
                    differences[key] = new_val
        
        # If no difference detected, return the original prediction.
        if not differences:
            original_row = candidate_prediction_rows[candidate_prediction_rows["Modified_Attribute"].isnull()]
            if original_row.empty:
                raise HTTPException(status_code=404, detail="Original prediction not found.")
            original_row = original_row.iloc[0]
            
            # Convert numeric and NumPy types to native Python types.
            original_good_fit = bool(original_row["GoodFit"])
            original_pred_prob = float(round(original_row["Prediction_Probability"], 2))
            converted_top_features = []
            for feat in original_row["Top_Features"]:
                converted_top_features.append({
                    "Feature": feat["Feature"],
                    "SHAP Value": float(feat["SHAP Value"])
                })
    
            return {
                "candidate_id": request.candidate_id,
                "prediction_probability": original_pred_prob,
                "is_good_fit": original_good_fit,
                "top_features": converted_top_features
            }
        
        # If more than one modifiable attribute is changed, check for race-specific changes.
        if len(differences) > 1:
            race_keys = [k for k in differences if k.startswith("RaceDesc_")]
            if race_keys:
                new_race = None
                for k in race_keys:
                    if str(differences[k]) == "1":
                        new_race = k.replace("RaceDesc_", "")
                        break
                if new_race is None:
                    raise HTTPException(status_code=400, detail="Invalid race update.")
                differences = {"Race": new_race}
            else:
                raise HTTPException(status_code=400, detail="Please update only one attribute at a time.")
        else:
            mod_attr = list(differences.keys())[0]
            if mod_attr.startswith("RaceDesc_"):
                differences = {"Race": mod_attr.replace("RaceDesc_", "")}
            elif mod_attr == "Sex":
                differences["Sex"] = "Male" if str(differences["Sex"]) == "1" else "Female"
                
        mod_attribute = list(differences.keys())[0]
        new_value_str = str(differences[mod_attribute])
        if mod_attribute == "Age":
            baseline_age_group = get_age_group(float(baseline_candidate["Age"]))

        query = (
            (candidate_prediction_rows["Modified_Attribute"] == mod_attribute) &
            (candidate_prediction_rows["New_Value"] == new_value_str)
        )
        counterfactual_prediction = candidate_prediction_rows[query]
        if counterfactual_prediction.empty:
            raise HTTPException(status_code=404, detail="No precomputed counterfactual prediction found for the updated attribute.")
        counterfactual_prediction = counterfactual_prediction.iloc[0]

       # Convert prediction probability and top features to native Python types.
        pred_prob = float(round(counterfactual_prediction["Prediction_Probability"], 2))
        good_fit = bool(counterfactual_prediction["GoodFit"])
        converted_top_features = []
        for feat in counterfactual_prediction["Top_Features"]:
            converted_top_features.append({
                "Feature": feat["Feature"],
                "SHAP Value": float(feat["SHAP Value"])
            })
        
        return {
            "candidate_id": request.candidate_id,
            "prediction_probability": pred_prob,
            "is_good_fit": good_fit,
            "top_features": converted_top_features
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{e.__traceback__.tb_lineno}, {str(type(e).__name__)}: {str(e)}")


@router.get("/predict/{candidate_id}", tags=["Prediction"])
def predict_candidate_api(candidate_id: int, static_predictions: pd.DataFrame = Depends(load_static_predictions)):
    """
    Predict if a selected candidate is a good fit.

    Parameters:
    candidate_id (int): The ID of the candidate.

    Returns:
    JSON: Prediction result for the candidate.
    """
    try:
        candidate_prediction_rows = static_predictions[static_predictions["Candidate_ID"] == candidate_id]
        if candidate_prediction_rows.empty:
            raise HTTPException(status_code=404, detail="Candidate not found.")
        
        # Extract the row as a Series
        original_row = candidate_prediction_rows[candidate_prediction_rows["Modified_Attribute"].isnull()]
        if original_row.empty:
            raise HTTPException(status_code=404, detail="Original prediction not found.")
        original_row = original_row.iloc[0]
        
        # Convert numeric and NumPy types to native Python types.
        original_good_fit = bool(original_row["GoodFit"])
        original_pred_prob = float(round(original_row["Prediction_Probability"], 2))
        converted_top_features = []
        for feat in original_row["Top_Features"]:
            converted_top_features.append({
                "Feature": feat["Feature"],
                "SHAP Value": float(feat["SHAP Value"])
            })

        return {
            "candidate_id": candidate_id,
            "prediction_probability": original_pred_prob,
            "is_good_fit": original_good_fit,
            "top_features": converted_top_features
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{e.__traceback__.tb_lineno},{str(type(e).__name__)}: {str(e)}")
