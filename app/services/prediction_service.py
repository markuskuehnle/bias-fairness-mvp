import pickle
import pandas as pd
import shap

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
    
    # Prediction
    prediction_proba = float(model.predict_proba(prepared_data)[:, 1][0])
    is_good_fit = prediction_proba >= 0.5

    return {
        "prediction_probability": prediction_proba,
        "is_good_fit": is_good_fit
    }


def predict_candidate_with_shap(candidate_row: pd.Series, model: object) -> dict:
    """
    Predict if a candidate is a good fit and return SHAP values.
    """
    prepared_data = prepare_candidate_for_prediction(candidate_row)
    prepared_data = prepared_data.apply(pd.to_numeric, errors='coerce')

    # Prediction
    prediction_proba = float(model.predict_proba(prepared_data)[:, 1][0])
    is_good_fit = prediction_proba >= 0.5

    # Compute SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(prepared_data)
    feature_importance = pd.DataFrame({
        'Feature': prepared_data.columns,
        'SHAP Value': shap_values.values[0]
    }).sort_values(by='SHAP Value', ascending=False).head(10)

    return {
        "prediction_probability": prediction_proba,
        "is_good_fit": is_good_fit,
        "top_features": feature_importance.to_dict(orient="records")
    }