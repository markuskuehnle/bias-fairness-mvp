from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse
from app.services.data_loader import load_candidates
from app.services.prediction_service import load_model, predict_candidate

router = APIRouter()

# Load the pre-trained XGBoost model
xgb_model = load_model()

@router.get("/candidates/data", tags=["Candidates"])
def get_candidates_data():
    """
    Get the list of two candidates formatted as fact sheets.

    Returns:
    list: List of candidate fact sheets.
    """
    try:
        # Load the candidate data
        candidates = load_candidates()

        # Select 6 random candidates (or fewer if the dataframe has less than 6 rows)
        selected_candidates = candidates.sample(n=min(6, len(candidates)))

        # Format each candidate as a fact sheet
        fact_sheets = []
        for _, row in selected_candidates.iterrows():
            # Determine Nationality
            if row["CitizenDesc_US Citizen"] == 1:
                nationality = "US Citizen"
            elif row["CitizenDesc_Eligible NonCitizen"] == 1:
                nationality = "Eligible Non-Citizen"
            elif row["CitizenDesc_Non-Citizen"] == 1:
                nationality = "Non-Citizen"
            else:
                nationality = "Unknown"

            # Perform the prediction
            prediction_result = predict_candidate(row, xgb_model)
            
            # Define a mapping of race descriptions to column names
            race_column_mapping = {
                "White": "RaceDesc_White",
                "Black or African American": "RaceDesc_Black or African American",
                "Asian": "RaceDesc_Asian",
                "American Indian or Alaska Native": "RaceDesc_American Indian or Alaska Native",
                "Hispanic": "RaceDesc_Hispanic",
            }

            # Function to get race from the available columns
            def get_race(row, race_column_mapping):
                for race, column in race_column_mapping.items():
                    if row[column] == 1:
                        if race == "Black or African American":
                            return "Black"  # Normalize to match radio value
                        if race == "American Indian or Alaska Native":
                            return "American Indian"
                        return race
                return "Unknown"  # Default fallback
            
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
                "Race": get_race(row, race_column_mapping),  # Dynamically determine race
                "Age": row["Age"],
                "GoodFit": prediction_result["is_good_fit"], # Target Value
                "Probability": round(prediction_result["prediction_probability"], 2), # Probabilistic Forecast
                "TopFeatures": prediction_result["top_features"] # SHAP values
            })
        
        # Assuming fact_sheets is a list and populated as shown
        for fact_sheet in fact_sheets:
            print(f"Candidate ID: {fact_sheet['Candidate_ID']}, Race: {fact_sheet['Race']}")
            
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
