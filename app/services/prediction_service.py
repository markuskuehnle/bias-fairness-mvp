import pickle
import pandas as pd
import shap
import json
import numpy as np

MODEL_PATH = "models/xgb_model.pkl"
FEATURE_LIST_PATH = "app/models/features.json"


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


def _load_feature_list():
    """
    Load the feature list from JSON.

    Returns:
    list: List of features to keep.
    """
    try:
        with open(FEATURE_LIST_PATH, "r") as file:
            features = json.load(file)
            return features
    except FileNotFoundError:
        raise FileNotFoundError(f"Feature list not found at path: {FEATURE_LIST_PATH}")
    except Exception as e:
        raise ValueError(f"Error loading feature list: {e.__traceback__.tb_lineno}, {str(type(e).__name__)}: {str(e)}")


def _prepare_candidate_for_prediction(candidate_row: pd.Series) -> pd.DataFrame:
    """
    Prepare a candidate's data for prediction by filtering with the feature list.

    Parameters:
    candidate_row (pd.Series): Row data for the selected candidate.

    Returns:
    pd.DataFrame: Processed data suitable for prediction.
    """
    try:
        # Load the feature list
        feature_list = _load_feature_list()

        # Filter the candidate row to include only the features in the list
        filtered_data = candidate_row[feature_list]

        # Convert to DataFrame for model prediction
        return filtered_data.to_frame().T
    except KeyError as e:
        raise ValueError(f"KeyError in preparing data: Missing feature {str(e)}")
    except Exception as e:
        raise ValueError(f"Error preparing candidate data for prediction: {e.__traceback__.tb_lineno}, {str(type(e).__name__)}: {str(e)}")


def _filter_top_features(feature_importance: pd.DataFrame) -> pd.DataFrame:
    """
    Filter out unwanted features from the SHAP importance list.

    Parameters:
    feature_importance (pd.DataFrame): DataFrame containing feature names and SHAP values.

    Returns:
    pd.DataFrame: Filtered DataFrame with unwanted features removed.
    """
    excluded_features = {"Age", "AgeGroup", "Sex"}
    excluded_prefixes = ("RaceDesc_",)

    return feature_importance[
        ~feature_importance["Feature"].isin(excluded_features) & 
        ~feature_importance["Feature"].str.startswith(excluded_prefixes)
    ]


def predict_candidate(candidate_row: pd.Series, model: object) -> dict:
    """
    Predict if a candidate is a good fit using the XGBoost model.

    Parameters:
    candidate_row (pd.Series): Row data for the selected candidate.
    model (object): The pre-trained XGBoost model.

    Returns:
    dict: Prediction result including the probability and fit status.
    """
    try:
        prepared_data = _prepare_candidate_for_prediction(candidate_row)
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
        }).sort_values(by='SHAP Value', ascending=False)

        feature_importance = _filter_top_features(feature_importance)
        top_features = feature_importance.head(3).to_dict(orient="records")

        if np.isnan(prediction_proba):
            prediction_proba = 0.0
            is_good_fit = False

        return {
            "prediction_probability": prediction_proba,
            "is_good_fit": is_good_fit,
            "top_features": top_features
        }
    except Exception as e:
        raise ValueError(f"Error candidate data prediction: {e.__traceback__.tb_lineno},{str(type(e).__name__)}: {str(e)}")
