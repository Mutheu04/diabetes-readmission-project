# Diabetes 30-Day Hospital Readmission Prediction

🔗 **[Live Dashboard](https://diabetes-readmission-rm.streamlit.app/)**

An end-to-end machine learning project predicting whether diabetic patients will be readmitted to hospital within 30 days of discharge.

---

## The Problem

Hospital readmissions within 30 days are a major quality and cost indicator in healthcare. In the US, hospitals face financial penalties for excessive readmission rates under the Hospital Readmissions Reduction Program. If high-risk patients can be identified before discharge, clinicians can intervene with targeted support — follow-up appointments, medication reviews, and discharge planning — to prevent avoidable readmissions.

## The Data

**Source:** [UCI ML Repository — Diabetes 130-US Hospitals (1999–2008)](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008)

- **101,766 inpatient encounters** across 130 US hospitals over 10 years
- **50 features** including demographics, diagnoses (ICD-9 codes), medications, lab indicators, and visit history
- **Target variable:** Whether the patient was readmitted within 30 days (binary classification)

### Data Challenges

This dataset is genuinely messy, which made it ideal for demonstrating real-world data preparation skills:

- `weight` column — 96.9% missing values (dropped)
- `medical_specialty` — 49.1% missing (imputed as "Unknown")
- Missing values encoded as `?` instead of NaN
- 800+ raw ICD-9 diagnosis codes requiring clinical grouping
- 16,773 patients with multiple encounters creating data leakage risk
- Heavy class imbalance — only 9% of patients readmitted within 30 days
- 16 out of 23 medication columns with near-zero variance (95%+ identical values)
- 3 rows with "Unknown/Invalid" gender
- Deceased and hospice patients included (cannot be readmitted — must be removed)

## Objective

Build a binary classification model to predict 30-day readmission, following the CRISP-DM methodology:

1. **Data Collection** — Web scraped directly from UCI
2. **Data Understanding** — Exploratory analysis of 50 features
3. **Data Preparation** — Cleaning, imputation, ICD-9 grouping, feature engineering
4. **Modelling** — Three algorithms compared with two imbalance-handling strategies
5. **Evaluation** — ROC-AUC, Precision-Recall, SHAP explainability, clinical threshold analysis
6. **Deployment** — Interactive Streamlit dashboard with patient risk prediction tool

## Key Results

| Metric | Value |
|---|---|
| Best Model | Gradient Boosting (class weighted) |
| ROC-AUC | 0.654 |
| Average Precision | 0.177 |
| Recall at threshold 0.30 | 91.4% |
| Recall at default threshold 0.50 | 51.9% |
| Final dataset | 69,990 patients × 99 features |

### Model Comparison

| Model | ROC-AUC | Avg Precision |
|---|---|---|
| Logistic Regression (weighted) | 0.628 | 0.150 |
| Random Forest (weighted) | 0.646 | 0.155 |
| **Gradient Boosting (weighted)** | **0.654** | **0.177** |
| Logistic Regression (SMOTE) | 0.577 | 0.113 |
| Random Forest (SMOTE) | 0.550 | 0.104 |
| Gradient Boosting (SMOTE) | 0.577 | 0.117 |

## Key Insights

### 1. Discharge Destination Drives Readmission Risk
Discharge disposition is by far the most important predictor (SHAP analysis). Patients transferred to rehab facilities have a 26% readmission rate compared to just 6% for patients discharged home. Where a patient goes after discharge is the strongest signal of whether they will return.

### 2. Prior Hospitalisations Create a "Frequent Flyer" Pattern
The number of prior inpatient visits is the second strongest predictor. Patients with recurring hospitalisations are significantly more likely to be readmitted, indicating a cycle of instability that a single hospital stay does not resolve.

### 3. Comorbidities Matter More Than Diabetes Itself
Only 8.2% of patients had diabetes as their primary diagnosis, despite this being a diabetic patient dataset. Most patients were admitted for circulatory, respiratory, or injury-related conditions while also having diabetes. Mental health comorbidities showed a 10.7% readmission rate, highlighting the link between mental health and diabetes management.

### 4. Age Is a Consistent Risk Factor
Readmission rates increase steadily with age — from 2% in children (0–10) to over 10% in patients aged 70–90. Older patients have more complex comorbidities and frailer health, making post-discharge recovery harder.

### 5. SMOTE Hurt Performance
SMOTE degraded all three models (ROC-AUC dropped by 0.05–0.10). The synthetic samples introduced noise rather than useful patterns because many features are one-hot encoded binary columns — interpolating between binary values creates fractional values that do not represent real patients. Class weighting achieved the same goal without altering the training data.

### 6. This Is a Hard Prediction Problem
All models achieved modest ROC-AUC scores (0.63–0.65), consistent with published research on this dataset. No single feature strongly predicts readmission — the correlation heatmap showed all individual feature correlations with the target below 0.10. The model works best as a screening tool that combines many weak signals rather than relying on any one factor.

## Clinical Implications

Even with modest predictive performance, the model has practical value as a **screening tool**:

- At **threshold 0.30**, the model catches **91.4% of readmissions** — suitable for low-cost interventions like automated follow-up calls or pharmacist medication reviews
- At **threshold 0.50**, it catches **51.9%** with fewer false alarms — suitable for higher-cost interventions like extended observation or discharge planning nurse assignments
- The threshold analysis gives hospital administrators the data to choose the right balance for their specific context and intervention costs

## Limitations

- The dataset spans 1999–2008 — clinical practices and readmission patterns may have changed
- Key predictors are missing: lab result values, medication adherence, socioeconomic factors, clinical notes
- The model has not been externally validated on data from other hospital systems
- XGBoost, LightGBM, or deep learning approaches could be explored for marginal improvements

## Feature Engineering Highlights

- **ICD-9 Code Grouping:** Mapped 800+ raw diagnosis codes into 19 clinically meaningful categories using the WHO ICD-9-CM chapter structure
- **Derived Features:** Created `num_total_visits` (healthcare utilisation), `num_active_medications` (treatment complexity), `med_change_intensity` (treatment instability), and `has_diabetes_primary`
- **Near-Zero Variance Removal:** Identified and dropped 16 medication columns where 95%+ of values were identical
- **Leakage Prevention:** Deduplicated by patient ID to prevent the same patient appearing in both train and test sets

## Project Structure
```
├── app/
│   └── app.py                          # Streamlit dashboard and prediction tool
├── data/
│   ├── raw/                            # Original scraped data (gitignored)
│   └── processed/                      # Cleaned dataset
├── notebooks/
│   └── Diabetes Re-admission.ipynb     # Full pipeline: scraping, EDA, cleaning, feature engineering, modelling
├── outputs/                            # Charts, saved model (.pkl), SHAP plots
├── .gitignore
├── README.md
└── requirements.txt
```

## Tools and Technologies

Python · pandas · NumPy · scikit-learn · matplotlib · seaborn · SHAP · Streamlit · Plotly · Git · Jupyter

## Setup and Reproduction

```bash
# Clone the repository
git clone https://github.com/Mutheu04/diabetes-readmission-project.git
cd diabetes-readmission-project

# Create virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt

# Run the full pipeline
# Open and run the notebook: notebooks/Diabetes Re-admission.ipynb

# Run the Streamlit dashboard
streamlit run app/app.py
```
## Author

**Ruth Mutheu** — Data Scientist

[GitHub](https://github.com/Mutheu04)