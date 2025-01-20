import pandas as pd
import os

def load_candidates(file_path: str = "app/data/static_data.parquet") -> pd.DataFrame:
    """
    Load candidate data from a static Parquet file.

    Parameters:
    file_path (str): Path to the Parquet file containing candidate data.

    Returns:
    pd.DataFrame: The loaded candidate data.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found at {file_path}")
    
    try:
        print(pd.read_parquet(file_path))
        return pd.read_parquet(file_path)
    except Exception as e:
        raise RuntimeError(f"Error loading candidates from {file_path}: {e}")


def filter_candidate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter specific columns from the candidate data.

    Parameters:
    df (pd.DataFrame): The full candidate data.

    Returns:
    pd.DataFrame: The filtered candidate data with specific columns.
    """
    # Select required columns
    columns_to_include = ['Candidate_ID', 'Sex', 'Employee_Name', 'YearsExperience'] + [
        col for col in df.columns if col.startswith('RaceDesc_')
    ]
    return df[columns_to_include]


def prepare_candidate_for_prediction(candidate_row: pd.Series) -> pd.DataFrame:
    """
    Prepare a candidate's data for prediction by removing unnecessary columns.

    Parameters:
    candidate_row (pd.Series): Row data for the selected candidate.

    Returns:
    pd.DataFrame: Processed data suitable for prediction.
    """
    # Exclude columns not used in prediction
    exclude_columns = ['Employee_Name']
    return candidate_row.drop(labels=exclude_columns).to_frame().T
