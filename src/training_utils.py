import os
import csv
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix


def save_experiment_metadata(file_path, model_name, parameters, comments, y_test, y_pred):
    """
    Save model/experiment metadata and metrics to an experiment tracker CSV file.

    Parameters:
        file_path (str): Path to the experiment tracker CSV file.
        model_name (str): Name of the model.
        parameters (dict): Model parameters as a dictionary.
        comments (str): Additional comments about the experiment (e.g., preprocessing steps).
        y_test (array-like): True labels for the test set.
        y_pred (array-like): Predicted labels from the model.
    """
    # Define column headers
    headers = [
        'Timestamp', 'Model Name', 'Parameters', 'Comments', 
        'Class 0 Precision', 'Class 0 Recall', 'Class 0 F1-Score', 
        'Class 0 Support', 'Confusion Matrix'
    ]

    # Generate classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    class_0_metrics = report['0']  # Metrics for class 0
    precision_0 = class_0_metrics['precision']
    recall_0 = class_0_metrics['recall']
    f1_0 = class_0_metrics['f1-score']
    support_0 = class_0_metrics['support']

    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    cm_str = str(cm.tolist())  # Convert confusion matrix to a string

    # Prepare data to save
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    parameters_str = str(parameters)  # Convert parameters dictionary to string

    # Check if the file exists
    file_exists = os.path.isfile(file_path)

    # Write to the CSV file
    with open(file_path, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)

        # Write headers if the file is new
        if not file_exists:
            writer.writerow(headers)

        # Append the experiment metadata
        writer.writerow([
            timestamp, model_name, parameters_str, comments,
            precision_0, recall_0, f1_0, support_0, cm_str
        ])
