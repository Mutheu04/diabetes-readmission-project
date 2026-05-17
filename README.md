\# Diabetes 30-Day Readmission Prediction



Predicting hospital readmission within 30 days for diabetic patients using the UCI Diabetes 130-US Hospitals dataset (1999–2008).



🔗 \*\*\[Live Dashboard](https://diabetes-readmission-rm.streamlit.app)\*\*



\## Objective



Build an end-to-end ML pipeline to identify diabetic patients at high risk of 30-day hospital readmission, enabling targeted discharge interventions.



\## Key Results



| Metric | Value |

|---|---|

| Best Model | Gradient Boosting |

| ROC-AUC | 0.654 |

| Average Precision | 0.177 |

| Recall (at threshold 0.30) | 91.4% |

| Dataset Size | 101,766 → 69,990 (after cleaning) |



\## Key Findings



\- \*\*Discharge destination is the strongest predictor\*\* — patients transferred to rehab facilities have 2–3x the readmission rate of those discharged home

\- \*\*Prior inpatient visits\*\* are the second most important feature — patients with recurring hospitalisations are at significantly higher risk

\- \*\*Class weighting outperformed SMOTE\*\* — SMOTE degraded all models due to noise from interpolating sparse one-hot features

\- \*\*30-day readmission is inherently difficult to predict\*\* from structured clinical data alone, consistent with published research (ROC-AUC 0.60–0.68)



\## Project Structure
├── app/

│   └── app.py                  # Streamlit dashboard \& prediction tool

├── data/

│   ├── raw/                    # Original scraped data (gitignored)

│   └── processed/              # Cleaned dataset

├── notebooks/

│   └── Diabetes Re-admission.ipynb  # Full EDA, cleaning, and modelling

├── outputs/                    # Charts, saved model, SHAP plots

├── src/

│   └── scrape\_dataset.py       # Webscrape dataset from UCI

├── .gitignore

├── README.md

└── requirements.txt

## Pipeline



1\. \*\*Data Collection\*\* — Web scraped from UCI ML Repository

2\. \*\*EDA \& Cleaning\*\* — Handled 97% missing weight column, replaced `?` with NaN, removed deceased patients, deduplicated by patient ID

3\. \*\*Feature Engineering\*\* — Grouped 800+ ICD-9 codes into 19 clinical categories, created derived features (total visits, medication change intensity)

4\. \*\*Modelling\*\* — Compared Logistic Regression, Random Forest, and Gradient Boosting with class weighting and SMOTE

5\. \*\*Explainability\*\* — SHAP analysis confirming clinically meaningful feature importance

6\. \*\*Deployment\*\* — Interactive Streamlit dashboard with patient risk prediction tool



\## Tools



Python · pandas · scikit-learn · matplotlib · seaborn · SHAP · Streamlit · Plotly · Git



\## Setup



```bash

git clone https://github.com/Mutheu04/diabetes-readmission-project.git

cd diabetes-readmission-project

python -m venv venv

source venv/bin/activate   # Windows: venv\\Scripts\\activate

pip install -r requirements.txt

python src/scrape\_dataset.py

streamlit run app/app.py

```



\## Dataset



\[UCI ML Repository — Diabetes 130-US Hospitals](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008)



\## Author



\*\*Ruth Mutheu\*\* — Data Scientist

\- \[GitHub](https://github.com/Mutheu04)


