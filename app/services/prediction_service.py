import pickle
import pandas as pd

MODEL_PATH = "app/models/xgb_model.pkl"

def load_model(model_path: str = MODEL_PATH):
    """
    Load the pre-trained XGBoost model.

    Parameters:
    model_path (str): Path to the pre-trained model.

    Returns:
    object: The loaded XGBoost model.
    """
    try:
        with open(model_path, "rb") as file:
            return pickle.load(file)
    except Exception as e:
        raise RuntimeError(f"Error loading model from {model_path}: {e}")


def prepare_candidate_for_prediction(candidate_row: pd.Series) -> pd.DataFrame:
    """
    Prepare a candidate's data for prediction by removing unnecessary columns.

    Parameters:
    candidate_row (pd.Series): Row data for the selected candidate.

    Returns:
    pd.DataFrame: Processed data suitable for prediction.
    """
    exclude_columns = ['Employee_Name', 'Candidate_ID']
    return candidate_row.drop(labels=exclude_columns).to_frame().T


def predict_candidate(candidate_row: pd.Series, model: object) -> dict:
    """
    Predict if a candidate is a good fit using the XGBoost model.

    Parameters:
    candidate_row (pd.Series): Row data for the selected candidate.
    model (object): The pre-trained XGBoost model.

    Returns:
    dict: Prediction result including the probability and fit status.
    """
    prepared_data = prepare_candidate_for_prediction(candidate_row)
    prepared_data = prepared_data.apply(pd.to_numeric, errors='coerce')

    non_int_columns = prepared_data.select_dtypes(exclude=['int64', 'int32'])
    print(non_int_columns)

    prediction_proba = float(model.predict_proba(prepared_data)[:, 1][0])  # Convert to Python float
    is_good_fit = prediction_proba >= 0.5

    return {
        "prediction_probability": prediction_proba,
        "is_good_fit": is_good_fit
    }
