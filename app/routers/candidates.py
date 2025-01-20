from fastapi import APIRouter, HTTPException
from app.services.data_loader import load_candidates, filter_candidate_columns

router = APIRouter()

@router.get("/candidates", tags=["Candidates"])
def get_candidates():
    """
    Retrieve the list of candidates with selected columns.

    Returns:
    JSON: List of filtered candidates with their attributes.
    """
    try:
        candidates = load_candidates()
        print(candidates.head(3))
        filtered_candidates = filter_candidate_columns(candidates)
        return filtered_candidates.to_dict(orient="records")
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=f"{e.__traceback__.tb_lineno},{str(type(e).__name__)}: {str(e)}")
