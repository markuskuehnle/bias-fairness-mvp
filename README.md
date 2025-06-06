![IBA Cover](imgs/institut_fuer_business_analytics_der_universitaet_ulm_cover.jpeg)

# Bias and Fairness Demonstrator 

This repository contains the research artifact for a behavioral study on human decision-making. The project examines how users interact with candidate profiles when assisted by AI-generated recommendations—and how explainable AI (XAI) influences those decisions.

We simulate a realistic hiring context where participants evaluate job candidates based on attributes such as skills, certifications, experience, and demographic information. Across different experimental conditions, we vary the level of AI assistance—from simple badges to detailed explanations and interactive manipulation tools—to study how transparency and interactivity shape trust, fairness perception, and final decisions.

The project combines machine learning, fairness analysis, and human-computer interaction to investigate:

- The influence of AI recommendations on candidate selection behavior
- The effect of SHAP-based explainability on user trust and decision rationale
- User responses to interactive counterfactual tools that allow modifying candidate features
- Emerging patterns of bias, gaming behavior, and decision fatigue

This system serves as both a research artifact and an interactive tool to explore bias and fairness in algorithm-assisted decisions.

---

## Objective  

- Train a classifier on **historically biased employee data** to reflect real-world biases. 
- Implement counterfactual analysis to evaluate how changes in individual features impact model decisions.
- Develop an interactive system to study user decision-making when selecting candidates based on AI recommendations.  
- Implement and compare different variations of AI-assisted selection to evaluate their impact:  
  1. **AI Recommendation Only:** Candidate cards display an AI-generated **good fit** badge based on the (biased) classifier.  
  2. **AI Recommendation + XAI:** Candidate cards include the **good fit** badge and an **explainability (XAI) section**, showing the most important features influencing the decision (SHAP values).  
  3. **AI Recommendation + XAI + Feature Manipulation:** Users can **manipulate candidate attributes**, triggering a **new AI prediction** to observe how changes affect the decision.  
- Demonstrate fairness interventions with user-friendly visualizations.
- Analyze and visualize how biased data influences decisions.

The candidate pool is preselected from the dataset to maintain consistency in evaluation.  

---

## Contributors  

This project is developed as part of a research collaboration between:  

- **Chiara Schwenke** – Research lead, concept & study design  
- **Markus Kühnle** – System development & implementation & study design 
- **Institute of Business Analytics, University of Ulm** – Research support & academic supervision  

This tool serves as a research artifact to study decision-making behavior in candidate selection.  

## Interactive Candidate Selection Interface Mockup

![Mockup](imgs/mockup.png)

---

## **Project Setup**  

To set up the project, follow these steps:  

### **1. Clone the Repository**
Navigate to your desired local directory and clone the repository:  

```sh
cd /path/to/your/projects  
git clone https://github.com/markuskuehnle/bias-fairness-mvp.git  
cd bias-fairness-mvp  
```

### **2. Install `uv` (Optional, Recommended for Dependency Management)**  
[`uv`](https://github.com/astral-sh/uv) is a fast package manager that simplifies dependency handling. If you don’t have `uv` installed, you can install it with:

```sh
pip install uv
```

Then verify the installation:

```sh
uv --version
```

### **3. Create a Virtual Environment**  
It's recommended to use a virtual environment to isolate dependencies.  

Run the following command to create a `.venv` folder in the project directory:

```sh
python -m venv .venv  
```

Activate the virtual environment:  

- **macOS/Linux:**  
  ```sh
  source .venv/bin/activate  
  ```
- **Windows (PowerShell):**  
  ```sh
  .venv\Scripts\Activate  
  ```

### **4. Install Dependencies**  

#### **Using `uv` (Recommended)**
If you have `uv` installed, install dependencies from `uv.lock`:

```sh
uv venv .venv  
uv pip sync  
```

Alternatively, if you need to install from `requirements.txt`:

```sh
uv add -r requirements.txt  
```

#### **Using `pip` (Alternative)**
If you prefer `pip`, install dependencies from `requirements.txt`:

```sh
pip install -r requirements.txt  
```

### **5. Verify Installation**  

Check if the virtual environment is active and dependencies are installed correctly:

```sh
python -m pip list  
```

If using `uv`, you can also verify dependencies with:

```sh
uv pip list  
```

### **6. Start the Application**

Once the setup is complete, you can start the FastAPI application using uvicorn. Make sure you are in the project root directory:

```sh
uvicorn app.main:app --reload
```

This will launch the API with automatic reloading enabled, making it easier for development.

You're now ready to run the notebooks and start the project. 🖥️

---

![data_description](imgs/data_description_banner.png)

## Dataset Information

### HR Data Set
This project utilizes the **HR Data Set Based on Human Resources Data Set**:
- Source: [Kaggle Dataset](https://www.kaggle.com/datasets/davidepolizzi/hr-data-set-based-on-human-resources-data-set).
- Additional datasets for selection can be found under: [Bias-Fairness-Data Repository](https://github.com/markuskuehnle/bias-fairness-data?tab=readme-ov-file).

---

## Data Preparation Summary

This section outlines the key steps taken to prepare the data for classifier training and the MVP dataset.

### Classifier Training Dataset Preparation

#### 1. Data Cleaning (Notebook: `02 - Data Cleaning`)

- Loaded raw datasets (`tbl_action.csv`, `tbl_employee.csv`, `tbl_perf.csv`, `hr_data.csv`).
- Standardized missing values and handled null values for critical columns like `TermReason`, `ManagerName`, and `DateofTermination`.
- Transformed date columns to ensure proper format and consistency.
- Removed duplicates based on unique identifiers.
- Converted categorical and numeric data to appropriate types.
- Created new features, such as `Churn`, `Tenure`, `Age`, and `Churn-Yes/No`.
- Engineered features like `GoodFit` based on performance and tenure.
- Removed high-null and unnecessary columns to streamline the dataset.
- Filtered rows to remove invalid employment statuses.
- Conducted class balance analysis to understand `GoodFit` distribution.
- Dropped non-essential columns for model training.
- Calculated `YearsExperience` and grouped ages into categorical bins (`AgeGroup`).
- Saved the cleaned dataset as `hr_data.parquet`.

#### 2. Simulating Additional Information (Notebook: `03 - Simulate Additional Information`)

- Enriched dataset with **synthetic role-specific features:** `Skills`, `Certifications`, and `Education`.
- Used predefined mappings to assign skills and certifications based on experience.
- Simulated role-based education levels with probabilistic assignment.
- Improved qualifications for high performers by enhancing skills, certifications, and education.
- Calculated `GoodFit` based on a weighted scoring system of skills, education, and certifications.
- Analyzed changes in `GoodFit` distribution post-simulation.
- Conducted correlation analysis and fairness assessments across demographic groups.
- Saved the enriched dataset as `hr_data_simulated.parquet`.

#### 3. Encoding Data (Notebook: `04 - Encode Data`)

- Applied one-hot encoding for categorical features (`Position`, `CitizenDesc`, `RaceDesc`, `Department`).
- Used label encoding for `State`, `Sex`, `AgeGroup`, and `HispanicLatino`.
- Mapped `ExperienceCategory` and `Education` to numerical values.
- Applied multi-label binarization to `Skills` and `Certifications`.
- Saved label encoders and multi-label binarizers for future use.
- Saved processed dataset as `hr_data_encoded.parquet`.

#### 4. Train-Test Split (Notebook: `05 - Train-Test-Split`)

- Defined `GoodFit` as the target variable.
- Split data into training (90%) and test (10%) sets using stratified sampling.
- Saved `X_train`, `X_test`, `y_train`, and `y_test` as Parquet files.

### MVP Dataset Preparation

The MVP dataset preparation follows a similar structure but is tailored to meet the requirements of the **Bias & Fairness Demonstrator Application**.

### 1. Counterfactual Calculation (Notebook: `10 - Counterfactual Calculation`)

- Loaded the trained `XGBoost` model and the test dataset (`X_test.parquet`).
- Predicted probability scores for each row.
- Identified candidates with probabilities near the classification threshold (`0.40 - 0.60`).
- Modified demographic attributes (`Race`, `Gender`, `YearsExperience`) to assess model sensitivity.
- Calculated counterfactual probabilities to analyze the impact of single-attribute changes.
- Identified individuals whose predictions significantly changed due to these modifications.
- Extracted key candidates for further analysis in the Bias & Fairness Demonstrator.

### 2. Creating Static Data for the App MVP (Notebook: `11 - Create Static Data for App MVP`)

- Filtered the test dataset to include candidates for a specific role (`Production Technician I`).
- Merged this filtered dataset with HR metadata to retain **employee names** and additional attributes.
- Ensured all selected candidates had complete demographic and experience details.
- Assigned **synthetic birthplace information** based on race and citizenship probability distributions.
- Extracted **technical skills and certification scores** using predefined role-based mappings.
- Standardized skill and certification scoring on a `0-5` scale for consistency.
- Created a final dataset combining:
  - Candidate demographics
  - Role-related information
  - Model predictions
  - Counterfactual probabilities
- Saved the **final dataset** as `static_data.parquet` for integration into the Bias & Fairness Demonstrator.

### 3. Feature Description Generation (Notebook: `12 - Create Feature Descriptions`)

- Defined a detailed mapping of feature names to descriptions.
- Mapped skills, certifications, and job positions to meaningful business terms.
- Validated feature names against the dataset to ensure completeness.
- Identified and resolved missing feature descriptions.
- Exported the final feature descriptions as `feature_description.json` for use in the MVP app.

### 4. Reviewing Predictions & Dataset Balancing (Notebook: `13 - Review Predictions`)

- Loaded the **static dataset** (`static_data.parquet`) and pre-trained `XGBoost` model.
- Predicted `GoodFit` probabilities and classified candidates accordingly.
- Visualized prediction distributions and **SHAP feature importance**.
- Identified fairness concerns by examining demographic patterns in the model’s decisions.
- Downsampled **"Good Fit"** predictions to **balance** the dataset for fair analysis.
- Saved the **resampled dataset** as `static_data.parquet` for use in the Bias & Fairness Demonstrator.

### 5. Precomputing GoodFit Predictions (Notebook: `15 - GoodFit Prediction Static Data`)

- Loaded the **processed dataset** (`static_data.parquet`) and pre-trained `XGBoost` model.  
- Systematically **varied protected attributes** (age, sex, race) for each candidate to generate counterfactual predictions.  
- Predicted `GoodFit` probabilities and identified **top contributing features** using **SHAP**.  
- Evaluated **potential biases** by comparing original and modified predictions.  
- Filtered candidates whose `GoodFit` status **changed** due to multiple attribute modifications.  
- Saved the **precomputed dataset** as `static_predictions.parquet` for use in the Bias & Fairness Demonstrator.  

---

## Notebook Summary  

- **01_data_exploration**: Performs an initial analysis of the dataset, identifying key patterns, distributions, and potential data quality issues.  
- **02_data_cleaning**: Prepares the raw HR dataset by handling missing values, correcting inconsistencies, and engineering new features. This step ensures a clean, standardized dataset suitable for fairness and bias analysis.
- **03_simulate_additional_information**: Enhances the dataset by simulating skills, certifications, and education levels based on role-specific requirements. This improves realism and helps evaluate biases in AI-based CV screening.
- **04_encode_data**: Transforms categorical features into a machine-readable format using encoding techniques like one-hot encoding and label encoding. It also normalizes numerical features and prepares the dataset for modeling.
- **05_train_test_split**: Splits the processed dataset into training and testing sets while ensuring stratification of the target variable. This ensures balanced class distributions for model evaluation.
- **06_train_baseline_model**: Trains a simple baseline model as a reference for performance comparison.  
- **07_train_xgb_model**: Trains an XGBoost model, tuning hyperparameters for improved predictive accuracy.  
- **08_show_metrics**: Extracts and displays model evaluation metrics from the experiment tracker. It retrieves parameters from a specific experiment and provides insights into performance comparisons.
- **09_train_nn**: Trains a neural network for the prediction task, leveraging deep learning techniques.  
- **10_counterfactual_calculation**: Computes counterfactual predictions by modifying specific attributes (e.g., race, gender, age) to assess their impact on model outcomes. The notebook highlights potential biases and evaluates the model’s sensitivity to individual features.
- **11_create_static_data_for_mvp**: Generates a static dataset for the app MVP by selecting relevant candidate data from processed datasets. It includes feature enrichment, birthplace estimation, technical skills, and certification scoring before saving the final dataset as a Parquet file.
- **12_create_feature_description_json**: Creates a JSON file that documents feature descriptions, aligning them with role-specific skills and certifications. The notebook ensures consistency between feature names and role attributes while validating completeness.
- **13_review_predictions**: Applies a pre-trained XGBoost model to predict candidate suitability and visualizes the prediction distribution. It also explores SHAP-based feature importance, balances the dataset via downsampling, and prepares the data for further analysis.
- **14_run_pipeline**: Executes the full data pipeline, integrating all preprocessing, model training, and evaluation steps.  
- **15_goodfit_prediction_static_data**:  Precomputes GoodFit predictions for a static dataset by varying protected attributes such as age, sex, and race. It generates a structured dataset where each candidate's GoodFit score is recalculated under different attribute modifications. This enables the frontend to allow users to explore potential biases by adjusting one attribute at a time without requiring real-time model inference.
- **16_bias_demonstration**: Analyzes potential bias in model predictions using statistical tests, confusion matrices, calibration checks, and SHAP explainability. It examines demographic disparities and evaluates fairness across different subgroups.

---

## Counterfactual Fairness Analysis Summary

This section demonstrates our approach to evaluating bias in model predictions using counterfactual fairness metrics. By systematically altering protected attributes (sex, age, and race) for each candidate, we pre-compute counterfactual predictions. These results enable us to quantify how much a single attribute change can affect a candidate’s predicted probability of being a “Good Fit.”

### Dataset Overview

- **Number of Unique Candidate IDs:** 245  
- **Distribution of Original GoodFit**  
  - Approximately 70% labeled **True** (GoodFit)  
  - Approximately 30% labeled **False** (not a GoodFit)  

![goodfit_distribution](imgs/goodfit_distribution.png)

This indicates that the majority of candidates start out with a positive GoodFit label, although a substantial minority are labeled otherwise from the outset.

### Key Counterfactual Fairness Metrics

- **Average Maximum Change in Prediction Probability:** 0.8101998  
  _Interpretation:_ On average, when a protected attribute (such as sex, age, or race) is altered, the model’s predicted probability shifts by about **81 percentage points**. This high average change suggests that the model is extremely sensitive to these attributes, raising potential fairness concerns.

- **Flip Rate:** 1.0  
  _Interpretation:_ A flip rate of **1.0** (or **100%**) indicates that **every candidate** experienced a change in their GoodFit label under at least one counterfactual modification. In other words, for every candidate, there exists some attribute modification that causes the model’s decision to flip, underscoring the model’s volatility with respect to protected features.

### Detailed Distribution of Counterfactual Changes

- **Overall Counterfactual Changes:**  
  - **Average Change:** 0.5399411  
  - **Median Change:** 0.5970941  
  - **Variance:** 0.1098562  
  _Summary:_ The overall distribution of absolute changes in prediction probability shows that, on average, modifications lead to a change of about **54 percentage points**. The median is slightly higher, suggesting that many modifications result in substantial prediction shifts, although there is variability across candidates.

- **By Attribute:**  
  - **Sex:**  
    - **Average Change:** 0.3592  
    - **Median Change:** 0.3094  
    _Summary:_ Changes in the candidate’s sex produce the smallest average change in prediction probability, indicating that the model is somewhat less sensitive to alterations in this attribute.
  
  - **Age:**  
    - **Average Change:** 0.5934  
    - **Median Change:** 0.6570  
    _Summary:_ Age modifications lead to the largest shifts in prediction probability. This suggests that the model heavily weights age information, resulting in large swings when age is altered.
  
  - **Race:**  
    - **Average Change:** 0.5233  
    - **Median Change:** 0.5802  
    _Summary:_ Modifications in race also cause substantial changes, highlighting the model’s sensitivity to racial attributes.

### What This Means

These findings point to significant fairness concerns. The high average change and a 100% flip rate imply that the model’s decisions are highly volatile when protected attributes are modified. In practice, this means that small, seemingly innocuous changes to a candidate’s demographic profile can lead to dramatic changes in the AI’s recommendation—potentially reinforcing or even amplifying existing biases.

---

![mvp_banner](imgs/mvp_banner.png)

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

### **Phase 3: Frontend Development**

✅ 11. **Build a Simple Interactive Frontend**  
   - Use **HTML/CSS** for rapid frontend prototyping:
     - Display the list of 3 candidates with their current attributes and predictions.  
     - Show Predictions and SHAP values 

### **Phase 4: Add Additional Backend Routes**

✅ 12. **Extend API Functionality**  
   - Add the following routes to enhance interactivity:  
     - **POST /select_candidate**:  
       - Accept a candidate's ID as input and mark the candidate as selected.  
       - Return a confirmation response with the selected candidate's details.  
     - **GET /selected_candidate**:  
       - Retrieve the details of the currently selected candidate, including attributes and predictions.  
   - **Next Steps: Build the XAI Selection Tool**:  
     - Allow users to temporarily modify one attribute at a time (e.g., age, sex, gender) and review the new prediction in the frontend.  
     - Ensure invited candidates are dropped from the selection pool after being selected.  
     - Present a new set of candidates for the next round of invitations, up to **6 rounds** in total.  
     - Add a **"Next Round" button** in the frontend, prompting confirmation before proceeding.  
     - Remove all candidates from the previous round permanently when moving to the next round.  

✅ 13. **Save Candidate Selection and Flags**
   - Ensure the data about invited candidates is saved. (Consider saving the alternative candidates per round - round and user-id required)

### **Phase 5: Deployment**

✅ 15. **Setup the Database**  
   - Use **Supabase** to create a table schema
   - Connect backend with Supabase

✅ 16. **Deploy the MVP**  
   - Host the application on **Railway**, and **Supabase** for user testing and feedback.  

---

## Tool Screenshots: Candidate Selection & Bias Analysis Overview

![intro](imgs/intro.png)

![Applicant Selection Screenshot 5](imgs/applicant_selection5.png)

![Applicant Selection Screenshot 6](imgs/applicant_selection6.png)

---

## Final Research Artifact Screenshots

This tool was developed as part of a research project to explore human-AI decision-making. Below is a walkthrough of the user experience and technical setup.


### User Experience

When users access the application (hosted on **Railway**), they are greeted with an introductory screen explaining the task:

![Intro Screen](imgs/final/start_1.png)

Upon clicking **Start**, users are randomly assigned to one of four experimental groups:

- **no-xai**: A baseline group. Users see blank candidate profiles with no AI indicators.
- **badge**: Users see a yellow **"AI Recommendation"** badge on one profile per round.
- **prediction**: In addition to the badge, users also see:
  - A **Good Fit** boolean (Yes/No)
  - A **Prediction Probability** score (e.g., 0.84)
- **interactive**: This group receives the richest experience:
  - Badge, Good Fit label, and Prediction Probability
  - A detailed **XAI explanation section** (showing top influencing features)
  - A **sidebar tool** allowing users to manipulate one protected attribute at a time (e.g., gender, age, race) and observe real-time changes in the AI’s decision

Here’s an example of what users in the interactive group see before beginning the task:

![Interactive Group Explanation - 1](imgs/final/grp_interactive_1.png)
![Interactive Group Explanation - 2](imgs/final/grp_interactive_2.png)
![Interactive Group Explanation - 3](imgs/final/grp_interactive_3.png)

### Questionnaire Phase

After completing all 6 rounds of candidate evaluation, users are directed to a feedback questionnaire consisting of **25 Likert-scale questions**.

![Questionnaire Screenshot](imgs/final/questionaire_placeholder.png)

### Technical Setup

- **Frontend & Backend Deployment**: The full-stack app is deployed using [**Railway**](https://railway.app), enabling fast and easy deployment of both the frontend and backend logic.

![Railway Deployment](imgs/final/railway_deployment.png)

- **Database**: All session data (including user interactions, rounds, and feedback) is stored in **Supabase**, a hosted Postgres solution with built-in authentication and API access.

![Supabase Database](imgs/final/supabase_db.png)

---

## Tech Stack
- **Data Processing & Modeling**: Python, Pandas, Scikit-learn.
- **Backend**: FastAPI (Python) hosted on Railway.
- **Frontend**: Static HTML, CSS, JavaScript deployed on Railway.
- **Visualization**: Matplotlib, Plotly, SHAP.
- **Database**: PostgreSQL managed by Supabase.
- **User Session Tracking**: Supabase (Auth + DB).
- **Deployment**: Railway.

---

## Result Structure

Result json structure for storage:

```json

{
  "session_id": "a574ff6d-6f72-406f-b156-5b640e886ab3",
  "user_group": "features",
  "sessionTime": 16.371168,
  "rounds": [
    {
      "round_number": 1,
      "candidate_count": 2,
      "invited_count": 1,
      "round_duration": 9.963,
      "next_round_clicked": false,
      "candidates": [
        {
          "candidate_id": 207,
          "name": "Nevaeh Calhoun",
          "attributes": {
            "age": 50,
            "sex": "Male",
            "race": "White",
            "years_experience": 0,
            "technical_skills_score": 0,
            "certifications_score": 2
          },
          "good_fit": false,
          "recommended": false,
          "invited": false,
          "manipulated": false,
          "hover_events": []
        },
        {
          "candidate_id": 1219,
          "name": "Samuel Jennings",
          "attributes": {
            "age": 50,
            "sex": "Female",
            "race": "Black",
            "years_experience": 0,
            "technical_skills_score": 0,
            "certifications_score": 0
          },
          "good_fit": true,
          "recommended": true,
          "invited": true,
          "manipulated": true,
          "hover_events": [
            {
              "feature": "Basic Machinery Maintenance",
              "hover_duration": 0.21
            },
            {
              "feature": "Problem Identification",
              "hover_duration": 2.59
            }
          ],
          "manipulations": [
            {
              "changed_attribute": "race",
              "new_value": "White",
              "prediction_probability": 0.009999999776482582,
              "is_good_fit": false,
              "xai_features": [
                {
                  "Feature": "Safety Protocols",
                  "SHAP Value": 0.9836218953132629
                },
                {
                  "Feature": "Problem Identification",
                  "SHAP Value": 0.9336374402046204
                },
                {
                  "Feature": "Basic Machinery Maintenance",
                  "SHAP Value": 0.4883248209953308
                }
              ],
              "timestamp": "2025-03-10T13:15:48.503Z"
            },
            {
              "changed_attribute": "race",
              "new_value": "Black",
              "prediction_probability": 0.9800000190734863,
              "is_good_fit": true,
              "xai_features": [
                {
                  "Feature": "Safety Protocols",
                  "SHAP Value": 1.0299004316329956
                },
                {
                  "Feature": "Basic Machinery Maintenance",
                  "SHAP Value": 0.7226954698562622
                },
                {
                  "Feature": "Problem Identification",
                  "SHAP Value": 0.5744005441665649
                }
              ],
              "timestamp": "2025-03-10T13:15:50.699Z"
            }
          ]
        }
      ]
    },
    ...
  "feedbackTime": 22.282,
  "feedbackAnswers": {
      "1": "5",
      "2": "4",
      "3": "5",
      ...
      "23": "4",
      "24": "4",
      "25": "4"
    }
```

With the current **result structure**, we can analyze several **key aspects** of **user behavior, AI influence, and bias effects** in the decision-making process. Below is a breakdown of what we can analyze and potential additions to **improve the data collection for deeper analysis**.


### **What We Can Analyze with This Structure**
#### **1. AI Influence on User Decisions**
   - Compare **invited_count** across groups to see if users in the **badge** or **features** group invite more AI-recommended candidates than the **manipulation** group.
   - Track **good_fit vs. invited** candidates to check if users trust AI recommendations.
   - Analyze how often **recommended candidates** are **manipulated** before being invited (indicating skepticism toward AI).

#### **2. Manipulation Patterns (Only for `manipulation` Group)**
   - Identify which attributes users change most frequently (e.g., **sex, race, gender**).
   - Track if manipulations **increase or decrease** the AI **GoodFit** probability.
   - Compare **original GoodFit** vs. **post-manipulation GoodFit** to measure how much users "game" the system.
   - Identify manipulation behaviors: Are users consistently changing attributes like **age** or **gender** to fit a perceived AI pattern?

#### **3. Selection Trends & Decision Fatigue**
   - Track **invited_count** over rounds to check if users invite fewer candidates as they progress (suggesting fatigue).
   - Identify if users **become more reliant** on AI recommendations over time.
   - Check whether users who manipulate candidates in earlier rounds **stop manipulating** later (indicating adaptation to AI).

#### **4. User Behavior in Different Selection Groups**
   - Compare users across **badge, features, and manipulation** groups to see:
     - Does seeing **SHAP-based explanations** reduce bias?
     - Do users **trust** the AI more if they see **top features**?
     - Does the **manipulation** group create better or worse candidates?

#### **5. Bias in Decision-Making**
   - Analyze demographic distributions of invited candidates:
     - Do users **prefer** or **avoid** certain groups (e.g., inviting more men, older candidates, or specific racial groups)?
     - Does **seeing SHAP values** in the **features group** reduce bias?
   - Compare if certain attributes **affect** user decisions **more than AI predictions do**.
   - Track whether **highly qualified** but **historically underrepresented** candidates (e.g., female engineers, older candidates) are still invited.


### **What Should Be Considered to Add?**
To **strengthen** our analysis, consider adding:

#### **1. User Interaction Time per Round**
   - `"time_spent_seconds"`: Tracks how long users take to **make a decision per round**.
   - Helps analyze:
     - Does seeing **explainability (XAI)** increase or decrease decision speed?
     - Do users spend **more time** manipulating attributes?
     - Are users making **snap decisions** or **carefully considering** AI predictions?

#### **2. Which Features Were Manipulated?**
   - `"manipulated_attributes": ["age", "race", "gender"]`
   - Helps track:
     - **Which features are manipulated the most?**
     - **Are users strategically modifying certain attributes to optimize AI predictions?**
     - **Does manipulation affect invitation likelihood?**

#### **3. AI Confidence Scores (Pre/Post Manipulation)**
   - Add `"ai_score_before"` and `"ai_score_after"` for **manipulated candidates**.
   - Helps measure:
     - **How much does user input shift the AI’s decision?**
     - **Are users actively trying to game AI predictions?**

#### **4. Candidate Order on Screen**
   - `"candidate_display_order": ["C003", "C001", "C002"]`
   - Helps track:
     - **Are users more likely to pick top-listed candidates?**
     - **Does AI recommendation placement influence selection?**
     - **Do users scroll or stop at first options?**

#### **5. User Actions per Candidate**
   - `"actions": [{"candidate_id": "C003", "clicked_more_info": true, "hovered_xai": false, "expanded_skills": true}]`
   - Helps track:
     - **Are users engaging with explanations?**
     - **Do users click to expand SHAP explanations?**
     - **Do users explore candidate profiles before deciding?**

---

![key_decisions](imgs/key_decisions_banner.png)


###  Ground Truth Considerations

Establishing a reliable ground truth is essential for evaluating our AI-assisted candidate selection system and ensuring fairness in decision-making. We propose two potential metrics as the basis for a single ground truth variable that can objectively benchmark candidate quality.

#### Option 1: Aggregated Star Rating (Skill Score)

This metric aggregates individual star ratings for a candidate’s skills into a single, intuitive score. It provides a clear, easy-to-understand measure of technical proficiency based on predefined skill assessments. By using this aggregated star rating, we can objectively compare candidates’ abilities, independent of model predictions.

#### Option 2: Aggregated Composite Score

This approach combines technical skills and certification ratings into one comprehensive metric. The composite score reflects not only the candidate’s technical abilities but also the credibility provided by relevant certifications. This holistic measure offers a broader perspective on a candidate's overall qualifications and can serve as a robust ground truth for assessing both AI recommendations and user decision-making.

Both options provide objective benchmarks; however, the choice between them depends on whether a more straightforward evaluation of skills (Option 1) or a comprehensive assessment that includes certifications (Option 2) is preferred for the research context.
