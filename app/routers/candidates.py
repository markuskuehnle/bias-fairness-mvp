from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import pandas as pd 
import ast

from app.services.data_loader import load_candidates
from app.services.prediction_service import load_model, predict_candidate
from app.routers.prediction import load_static_predictions

router = APIRouter()

# Load the pre-trained XGBoost model
xgb_model = load_model()

# Store seen and invited candidates in memory (TODO: Maybe Use Redis/DB for persistence)
seen_candidates = set()
invited_candidates = set()


class InviteRequest(BaseModel):
    candidate_id: int


@router.get("/candidates/data", tags=["Candidates"])
def get_candidates_data(exclude_ids: list[int] = Query([], alias="exclude")):
    try:
        global invited_candidates, seen_candidates

        # Load candidate data
        candidates = load_candidates()
        static_predictions = load_static_predictions()

        # Add new seen candidates to the global tracking set
        seen_candidates.update(exclude_ids)
        excluded_candidates = seen_candidates.union(invited_candidates)

        # Remove already seen/invited candidates from the available pool
        available_candidates = candidates[~candidates["Candidate_ID"].isin(excluded_candidates)]
        candidates_with = available_candidates[
            available_candidates["Candidate_ID"].isin(
                static_predictions[
                    static_predictions["Modified_Attribute"].isnull() & (static_predictions["GoodFit"] == True)
                ]["Candidate_ID"]
            )
        ]
        candidates_without = available_candidates[
            available_candidates["Candidate_ID"].isin(
                static_predictions[
                    static_predictions["Modified_Attribute"].isnull() & (static_predictions["GoodFit"] == False)
                ]["Candidate_ID"]
            )
        ]

        if not candidates_with.empty and not candidates_without.empty:
            selected_candidates = pd.concat([candidates_with.sample(n=1), candidates_without.sample(n=1)]).sample(frac=1)
        elif len(available_candidates) >= 2:
            selected_candidates = available_candidates.sample(n=2)
        else:
            selected_candidates = available_candidates

        fact_sheets = []
        for _, row in selected_candidates.iterrows():
            nationality = (
                "US Citizen" if row["CitizenDesc_US Citizen"] == 1 else
                "Eligible Non-Citizen" if row["CitizenDesc_Eligible NonCitizen"] == 1 else
                "Non-Citizen" if row["CitizenDesc_Non-Citizen"] == 1 else
                "Unknown"
            )

            # Get the original prediction from static predictions
            pred_row = static_predictions[
                (static_predictions["Candidate_ID"] == row["Candidate_ID"]) &
                (static_predictions["Modified_Attribute"].isnull())
            ]
            if pred_row.empty:
                raise HTTPException(status_code=404, detail="Original prediction not found for candidate.")
            pred_row = pred_row.iloc[0]
            # Ensure TopFeatures is a list (parse if needed)
            top_features = pred_row["Top_Features"]
            if not isinstance(top_features, list):
                # If it's a numpy array, convert it to a list; otherwise, try literal_eval
                try:
                    import numpy as np
                    if isinstance(top_features, np.ndarray):
                        top_features = top_features.tolist()
                    else:
                        top_features = ast.literal_eval(top_features)
                except Exception:
                    top_features = []
            prediction_result = {
                "is_good_fit": bool(pred_row["GoodFit"]),
                "prediction_probability": float(round(pred_row["Prediction_Probability"], 2)),
                "top_features": top_features
            }
            
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
def show_candidates_frontend(): # TODO: modify frontend serving to show one recommended and one not-recommended candidate?
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
def invite_candidate(invite_data: InviteRequest):
    """
    Store an invited candidate in memory so they are never shown again.

    Parameters:
    invite_data (InviteRequest): The candidate ID wrapped in a Pydantic model.
    """
    try:
        global invited_candidates
        candidate_id = invite_data.candidate_id  # Extract from request body
        invited_candidates.add(candidate_id)
        return {"message": f"Candidate {candidate_id} invited successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/candidates/reset", tags=["Candidates"])
def reset_tool():
    """
    Fully reset the tool: clear seen/invited candidates and reload the full candidate pool.
    """
    try:
        global seen_candidates, invited_candidates

        # Fully clear seen and invited lists
        seen_candidates.clear()
        invited_candidates.clear()

        return {"message": "Tool has been fully reset, and the full candidate pool is available again."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
