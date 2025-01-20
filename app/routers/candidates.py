from fastapi import APIRouter, HTTPException
from app.services.data_loader import load_candidates

router = APIRouter()

@router.get("/candidates", tags=["Candidates"])
def get_candidates():
    """
    Get the list of two candidates formatted as fact sheets.

    Returns:
    list: List of candidate fact sheets.
    """
    try:
        # Load the candidate data
        candidates = load_candidates()

        # Select only two candidates
        selected_candidates = candidates.head(2)

        # Format each candidate as a fact sheet
        fact_sheets = []
        for _, row in selected_candidates.iterrows():
            fact_sheets.append({
                "Name": row["Employee_Name"].split(", ")[0],
                "Prename": row["Employee_Name"].split(", ")[1],
                "Gender": "Female" if row["Sex"] == 0 else "Male",
                "Age": "",
                "Nationality": "",
                "Skills": {
                    "Languages": 3,
                    "Degree": 3,
                    "Work Experience": 3,
                    "Social Skills": 3,
                    "Programming": 3,
                }
            })

        return fact_sheets

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{e.__traceback__.tb_lineno},{str(type(e).__name__)}: {str(e)}")
