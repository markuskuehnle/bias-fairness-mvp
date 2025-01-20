from fastapi import APIRouter, HTTPException
from app.services.data_loader import load_candidates
from app.services.prediction_service import load_model, predict_candidate, prepare_candidate_for_prediction
router = APIRouter()

# Load the pre-trained XGBoost model at the startup
xgb_model = load_model()

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

