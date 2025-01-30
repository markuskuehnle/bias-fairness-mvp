from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import HTMLResponse
from app.services.data_loader import load_candidates
from app.services.prediction_service import load_model, predict_candidate

router = APIRouter()

# Load the pre-trained XGBoost model
xgb_model = load_model()

# Store seen and invited candidates in memory (TODO: Maybe Use Redis/DB for persistence)
seen_candidates = set()
invited_candidates = set()


@router.get("/candidates/data", tags=["Candidates"])
def get_candidates_data(exclude_ids: list[int] = Query([], alias="exclude")):
    """
    Get the list of candidates formatted as fact sheets, excluding already invited candidates.

    Parameters:
    exclude_ids (list[int]): List of candidate IDs to exclude.

    Returns:
    list: List of candidate fact sheets.
    """
    try:
        global invited_candidates, seen_candidates

        # Load candidate data
        candidates = load_candidates()

        # Add new seen candidates to the global tracking set
        seen_candidates.update(exclude_ids)
        excluded_candidates = seen_candidates.union(invited_candidates)

        # Remove already seen/invited candidates from the available pool
        available_candidates = candidates[~candidates["Candidate_ID"].isin(excluded_candidates)]

        # Select up to 6 new random candidates
        selected_candidates = available_candidates.sample(n=min(6, len(available_candidates)))

        fact_sheets = []
        for _, row in selected_candidates.iterrows():
            nationality = (
                "US Citizen" if row["CitizenDesc_US Citizen"] == 1 else
                "Eligible Non-Citizen" if row["CitizenDesc_Eligible NonCitizen"] == 1 else
                "Non-Citizen" if row["CitizenDesc_Non-Citizen"] == 1 else
                "Unknown"
            )

            prediction_result = predict_candidate(row, xgb_model)

            race_column_mapping = {
                "White": "RaceDesc_White",
                "Black or African American": "RaceDesc_Black or African American",
                "Asian": "RaceDesc_Asian",
                "American Indian or Alaska Native": "RaceDesc_American Indian or Alaska Native",
                "Hispanic": "RaceDesc_Hispanic",
            }

            def get_race(row, mapping):
                for race, column in mapping.items():
                    if row[column] == 1:
                        return "Black" if race == "Black or African American" else (
                            "American Indian" if race == "American Indian or Alaska Native" else race
                        )
                return "Unknown"

            fact_sheets.append({
                "Candidate_ID": row["Candidate_ID"],
                "Name": row["Employee_Name"].split(", ")[0],
                "Prename": row["Employee_Name"].split(", ")[1],
                "Gender": "Female" if row["Sex"] == 0 else "Male",
                "Nationality": nationality,
                "Birthplace": row["Birthplace"],
                "Skills": {
                    "Degree": row["Education"] + 1,
                    "Technical Skills": row["Technical_Skills"],
                    "Certifications": int(row["Certifications_Score"]),
                    "Social Skills": 3,
                },
                "Race": get_race(row, race_column_mapping),
                "Age": row["Age"],
                "GoodFit": prediction_result["is_good_fit"],
                "Probability": round(prediction_result["prediction_probability"], 2),
                "TopFeatures": prediction_result["top_features"]
            })

        return fact_sheets

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{e.__traceback__.tb_lineno},{str(type(e).__name__)}: {str(e)}")


@router.get("/candidates", response_class=HTMLResponse, tags=["Candidates"])
def show_candidates_frontend():
    """
    Serve the frontend HTML file for candidates.
    """
    try:
        with open("app/frontend/index.html", "r") as file:
            html_content = file.read()
        return HTMLResponse(content=html_content)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Frontend file not found.")
    

@router.post("/candidates/invite", tags=["Candidates"])
def invite_candidate(candidate_id: int):
    """
    Store an invited candidate in the memory set so they are never shown again.

    Parameters:
    candidate_id (int): The ID of the invited candidate.
    """
    try:
        global invited_candidates
        invited_candidates.add(candidate_id)
        return {"message": f"Candidate {candidate_id} invited successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
