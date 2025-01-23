# Bias and Fairness Demonstrator: Predicting Role Fit

This project demonstrates bias and fairness in machine learning decision-making systems, focusing on predicting whether a candidate is a good fit for a specific role based on their CV attributes. The project integrates bias detection and fairness interventions, allowing users to interactively upload CVs or select predefined ones to analyze decision outcomes.

This is based on my initial idea and project setup sketch. I will iterate on this design and make adjustments as needed.

## Objective
The primary goal is to:
- Train an ML model on **biased data** to simulate real-world biases.
- Analyze and visualize how biased data influences decisions.
- Implement counterfactual analysis to evaluate how changes in individual features impact model decisions.
- Demonstrate fairness interventions with user-friendly visualizations.

---

## Dataset Information

### HR Data Set
This project utilizes the **HR Data Set Based on Human Resources Data Set**:
- Source: [Kaggle Dataset](https://www.kaggle.com/datasets/davidepolizzi/hr-data-set-based-on-human-resources-data-set).
- Additional datasets for selection can be found under: [Bias-Fairness-Data Repository](https://github.com/markuskuehnle/bias-fairness-data?tab=readme-ov-file).

---

## Dataset Preparation

### Define the Target Variable
- **"Fit for the Role"**:
  - Use `PositionID` or `Department` as the target role identifier.
  - Define "fit" based on:
    - **Performance**: Use `PerformanceScore`, `Rating`, or `EngagementSurvey` as indicators of success in similar roles.
    - **Experience**: Use tenure (`DateofHire` to `DateofTermination` or current date for active employees).
    - **Job Alignment**: Match attributes like skills (`PayRate`, `Position`) to job requirements.

### Label Candidates
- Create a binary label (e.g., `Good Fit` or `Not a Good Fit`) for a specific role based on:
  - Performance in the same or similar roles.
  - Tenure above a certain threshold (e.g., >1 year in the same position).
  - Department or role alignment.
  - Attributes like `PayRate` aligning with the average for that position.

### Feature Selection
- Include:
  - **Demographics**: (`GenderID`, `RaceDesc` for bias analysis).
  - **Professional attributes**: (`DeptID`, `PositionID`, `PayRate`, `SpecialProjectsCount`).
  - **Performance metrics**: (`PerformanceScore`, `EngagementSurvey`, `DaysLateLast30`).
  - **Tenure and experience metrics**.
- Exclude attributes that won’t impact role fit (e.g., `Zip`, unless geography matters).

### Encode Categorical Features
- Encode features like `Position`, `Department`, `RecruitmentSource`, and `ManagerName`.

### Normalize Continuous Features
- Normalize attributes like `PayRate`, `EngagementSurvey`, and `Age` for consistent scaling.

### Handle Missing Data
- Impute missing values (e.g., `PayRate`, `DOB`) with domain-appropriate methods.

---

## Step-by-Step Plan to Build the MVP

### Phase 1: Initial Development in Notebooks

✅ 1. **Define Scope**
   - Identify key functionalities: predicting role fit, analyzing bias, visualizing fairness metrics, and exploring counterfactuals.

✅ 2. **Data Exploration**
   - Understand and clean the dataset.
   - Analyze distributions of sensitive features (e.g., gender, age, race).
   - Engineer features like age, tenure, and department alignment.

✅ 3. **Enhance Dataset with Simulated Data**
   - Augment the dataset with simulated information for skills, certifications, and education levels tailored to each role.
   - Introduce a scoring system to calculate fit points based on qualifications, ensuring alignment with real-world expectations for each role.
   - Adjust qualifications dynamically for employees with higher performance scores to reflect logical consistency.
   - Validate the updates by correlating features and ensuring thresholds for role fit are met realistically.

✅ 4. **Encode and Prepare Data**
   - Transform categorical features using one-hot encoding, label encoding, and multi-label binarization.  
   - Encode multi-label columns (e.g., skills, certifications) and combined them with the main dataset.  
   - Save the processed dataset and encoding models for consistency in downstream tasks.  

✅ 5. **Train Biased Models**
   - Use historical data containing biases to train a baseline model.
   - Train a biased classifier (e.g., logistic regression, decision tree).
   - Visualize biased decisions using SHAP or LIME.

✅ 6. **Demonstrate Bias**
   - Implement bias metrics in the notebook.
   - Show which features contributed most to biased decisions.

✅ 7. **Counterfactual Analysis**
   - Develop a module to simulate "what-if" scenarios:
     - Example: "What if the candidate’s gender were different?"
     - Measure changes in prediction probability and decision outcomes.

### **Phase 2: Backend Development**

✅ 8. **Build Core API Endpoints (Priority Task)**  
   - Use **Flask** or **FastAPI** to implement the following endpoints:  
     - **GET /candidates**: Retrieve a list of 3 static candidates with their attributes and predictions.  
     - **POST /modify**: Accept changes to a candidate’s attributes (e.g., gender, age, race) and return updated predictions and counterfactual results.  
     - **GET /fairness** (Optional): Provide fairness and bias insights if needed later.

✅ 9. **Static Data Integration**  
   - Pre-load the data for 3 candidates into a static CSV or JSON file for simplicity.  
   - Use a utility script (`static_data_loader.py`) to serve this data via the API.

✅ 10. **Use Pre-Trained Model**  
   - Integrate the pre-trained **XGBoost** model to serve predictions via the API.  
   - Implement logic for simulating counterfactuals in the backend using **`counterfactuals.py`**.

11. **Add Configurations**  
   - Use a configuration file (`config.yaml`) to manage:
     - Data paths (e.g., CSV/JSON for candidates).
     - Sensitive attributes (e.g., gender, race, age).
     - Thresholds for predictions.

### **Phase 3: Frontend Development**

✅ 12. **Build a Simple Interactive Frontend**  
   - Use **HTML/CSS** for rapid frontend prototyping:
     - Display the list of 3 candidates with their current attributes and predictions.  

### **Phase 4: Add Additional Backend Routes**

13. **Extend API Functionality**  
   - Add the following routes to enhance interactivity:
     - **POST /select_candidate**:  
       - Accept a candidate's ID as input and mark the candidate as selected.
       - Return a confirmation response with the selected candidate's details.  
     - **GET /selected_candidate**:  
       - Retrieve the details of the currently selected candidate, including attributes and predictions.

14. **Update Static Data Handling**  
   - Ensure the `static_data_loader.py` script supports tracking the selection state for candidates.  
   - Add logic to update and serve the selected candidate dynamically.

### **Phase 5: Deployment**

15. **Containerize the Application**  
   - Use **Docker** to containerize the backend and frontend for consistency.  

16. **Deploy the MVP**  
   - Host the MVP on **Heroku**, **Render**, or **AWS** for user testing and feedback.  

![Applicant Selection Screenshot](imgs/screenshot_applicant_selection.png)

---

## Demonstrator Workflow

### User Interaction
- **Upload CV**: Allow users to upload a CV or select a predefined one.
  - Parse CV into feature schema used for modeling.
- **Role Selection**: User selects a role (e.g., `PositionID` or `Department`).

### Model Prediction
- Model predicts whether the person is a good fit for the selected role.
- Outputs:
  - Probability score (e.g., 85% fit).
  - Explanation of key contributing factors (e.g., SHAP values).

### Bias and Fairness Insights
- Visualize:
  - How sensitive attributes influence predictions.
  - Comparisons of fit probability across demographic groups.

### Comparison of Models
- Show predictions with and without fairness interventions.

---

## Focus on Bias, Fairness, and Counterfactuals

### Bias Simulation
- Train models on biased data to simulate real-world decisions.
- Analyze which features drive biased decisions and how sensitive attributes like `GenderID` or `RaceDesc` contribute.

### Counterfactual Analysis
1. **Feature Simulations**:
   - Allow users to modify attributes (e.g., gender, age, pay rate).
   - Observe how these changes influence predictions.

2. **Fairness Metrics**:
   - Disparate Impact.
   - Equal Opportunity.
   - Demographic Parity.

3. **Fairness Interventions**:
   - **Pre-processing**: Mask sensitive attributes.
   - **In-processing**: Use fairness-aware algorithms.
   - **Post-processing**: Adjust predictions to mitigate bias.

4. **Visualizations**:
   - Display feature importance for biased and debiased models.
   - Show counterfactual outcomes for individual candidates.

---

## Simple Tech Stack Suggestion
- **Data Processing & Modeling**: Python, Pandas, Scikit-learn.
- **Backend**: Flask/FastAPI, SQLite/PostgreSQL.
- **Frontend**: Streamlit (initial MVP), React.js (optional advanced).
- **Visualization**: Matplotlib, Plotly, SHAP.
- **Deployment**: Docker, Heroku/Render.

---

## Future Improvements
- Integrate more advanced fairness-aware algorithms.
- Add real-world CV parsers for user uploads.
- Implement a recommendation system for role suggestions.


## To Discuss
- Changed "Age" in the MVP to "YearsExperience" as selection parameter
- Should it be possible to continue with the next selection round, without selecting the specified number of candidates?
- How many applicants should be contained in the data?
- How many applicants are suggested in the frontend for the user?
- XAI: Is the candidate stored with original or changed attributes?

Tech:
- How to store user decision?