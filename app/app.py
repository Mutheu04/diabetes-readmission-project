"""
Diabetes 30-Day Readmission Prediction Dashboard
Built with Streamlit
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import os

# ── Page Config ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="Diabetes Readmission Predictor",
    page_icon="🏥",
    layout="wide"
)

# ── Load Model and Data ────────────────────────────────────────────
@st.cache_resource
def load_model():
    """Load the trained model, scaler, and feature names."""
    model_path = os.path.join(os.path.dirname(__file__), '..', 'outputs')
    model = joblib.load(os.path.join(model_path, 'gb_model.pkl'))
    scaler = joblib.load(os.path.join(model_path, 'scaler.pkl'))
    feature_names = joblib.load(os.path.join(model_path, 'feature_names.pkl'))
    return model, scaler, feature_names

@st.cache_data
def load_data():
    """Load the processed dataset for the dashboard."""
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'diabetic_processed.csv')
    return pd.read_csv(data_path)

model, scaler, feature_names = load_model()
df = load_data()

# ── Sidebar Navigation ─────────────────────────────────────────────
page = st.sidebar.radio("Navigate", ["📊 Dashboard", "🔮 Predict Readmission"])

# ════════════════════════════════════════════════════════════════════
# PAGE 1: DASHBOARD
# ════════════════════════════════════════════════════════════════════
if page == "📊 Dashboard":
    st.title("🏥 Diabetes 30-Day Readmission Dashboard")
    st.markdown("Exploring patterns in hospital readmission for diabetic patients using the "
                "UCI Diabetes 130-US Hospitals dataset (1999–2008).")

    # ── Key Metrics ─────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Patients", f"{len(df):,}")
    col2.metric("Readmission Rate", f"{df['readmitted_30'].mean()*100:.1f}%")
    col3.metric("Features Used", f"{len(feature_names)}")
    col4.metric("Best Model ROC-AUC", "0.654")

    st.divider()

    # ── Row 1: Target Distribution and Age ──────────────────────────
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Target Variable Distribution")
        target_counts = df['readmitted_30'].value_counts().reset_index()
        target_counts.columns = ['Readmitted', 'Count']
        target_counts['Readmitted'] = target_counts['Readmitted'].map(
            {0: 'Not Readmitted <30d', 1: 'Readmitted <30d'})
        fig = px.bar(target_counts, x='Readmitted', y='Count',
                     color='Readmitted',
                     color_discrete_map={'Not Readmitted <30d': '#2ecc71', 'Readmitted <30d': '#e74c3c'})
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("Readmission Rate by Age Group")
        age_map = {0: '0-10', 1: '10-20', 2: '20-30', 3: '30-40', 4: '40-50',
                   5: '50-60', 6: '60-70', 7: '70-80', 8: '80-90', 9: '90-100'}
        df['age_label'] = df['age'].map(age_map)
        age_readmit = df.groupby('age_label')['readmitted_30'].mean().reset_index()
        age_readmit.columns = ['Age Group', 'Readmission Rate']
        age_readmit['Readmission Rate'] = age_readmit['Readmission Rate'] * 100
        # Sort by age order
        age_order = ['0-10', '10-20', '20-30', '30-40', '40-50', '50-60', '60-70', '70-80', '80-90', '90-100']
        age_readmit['Age Group'] = pd.Categorical(age_readmit['Age Group'], categories=age_order, ordered=True)
        age_readmit = age_readmit.sort_values('Age Group')
        fig = px.bar(age_readmit, x='Age Group', y='Readmission Rate',
                     color='Readmission Rate', color_continuous_scale='Reds')
        fig.update_layout(height=400, yaxis_title='Readmission Rate (%)')
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ── Row 2: Feature Importance and Model Comparison ──────────────
    col_left2, col_right2 = st.columns(2)

    with col_left2:
        st.subheader("Top 10 Most Important Features")
        importances = pd.Series(model.feature_importances_, index=feature_names)
        top_10 = importances.sort_values(ascending=False).head(10).reset_index()
        top_10.columns = ['Feature', 'Importance']
        top_10 = top_10.sort_values('Importance', ascending=True)
        fig = px.bar(top_10, x='Importance', y='Feature', orientation='h',
                     color='Importance', color_continuous_scale='Reds')
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col_right2:
        st.subheader("Model Comparison")
        model_results = pd.DataFrame({
            'Model': ['Logistic Regression', 'Random Forest', 'Gradient Boosting',
                      'LR + SMOTE', 'RF + SMOTE', 'GB + SMOTE'],
            'ROC-AUC': [0.6283, 0.6461, 0.6539, 0.5766, 0.5498, 0.5770],
            'Method': ['Class Weighting', 'Class Weighting', 'Class Weighting',
                       'SMOTE', 'SMOTE', 'SMOTE']
        })
        fig = px.bar(model_results, x='Model', y='ROC-AUC', color='Method',
                     color_discrete_map={'Class Weighting': '#2ecc71', 'SMOTE': '#e74c3c'},
                     barmode='group')
        fig.update_layout(height=400, yaxis_range=[0.4, 0.75])
        fig.add_hline(y=0.5, line_dash="dash", line_color="grey",
                      annotation_text="Random Classifier")
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ── Row 3: Threshold Analysis ───────────────────────────────────
    st.subheader("Clinical Decision Threshold Analysis")
    st.markdown("Adjusting the classification threshold controls the trade-off between "
                "catching more readmissions (recall) and reducing false alarms (precision).")

    threshold_data = pd.DataFrame({
        'Threshold': [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50],
        'Recall (%)': [100.0, 99.5, 98.7, 97.9, 96.3, 91.4, 75.4, 51.9],
        'Precision (%)': [9.0, 9.0, 9.1, 9.3, 9.7, 10.1, 11.9, 14.7],
        'Patients Flagged': [13992, 13915, 13710, 13250, 12467, 11346, 7984, 4435]
    })

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=threshold_data['Threshold'], y=threshold_data['Recall (%)'],
                             name='Recall (%)', line=dict(color='#e74c3c', width=3)))
    fig.add_trace(go.Scatter(x=threshold_data['Threshold'], y=threshold_data['Precision (%)'],
                             name='Precision (%)', line=dict(color='#3498db', width=3)))
    fig.update_layout(height=400, xaxis_title='Threshold', yaxis_title='Percentage',
                      title='Recall vs Precision at Different Thresholds')
    st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════════════════
# PAGE 2: PREDICTION TOOL
# ════════════════════════════════════════════════════════════════════
elif page == "🔮 Predict Readmission":
    st.title("🔮 Patient Readmission Risk Predictor")
    st.markdown("Enter patient details below to estimate their 30-day readmission risk.")

    # ── Input Form ──────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Demographics")
        age = st.selectbox("Age Group", options=list(range(10)),
                          format_func=lambda x: ['0-10', '10-20', '20-30', '30-40', '40-50',
                                                  '50-60', '60-70', '70-80', '80-90', '90-100'][x],
                          index=6)
        gender = st.selectbox("Gender", ["Female", "Male"])
        race = st.selectbox("Race", ["Caucasian", "AfricanAmerican", "Hispanic", "Asian", "Other"])

    with col2:
        st.subheader("Clinical Details")
        time_in_hospital = st.slider("Days in Hospital", 1, 14, 4)
        num_lab_procedures = st.slider("Number of Lab Procedures", 1, 132, 43)
        num_procedures = st.slider("Number of Procedures", 0, 6, 1)
        num_medications = st.slider("Number of Medications", 1, 81, 16)
        number_diagnoses = st.slider("Number of Diagnoses", 1, 16, 7)

    with col3:
        st.subheader("Visit History")
        number_inpatient = st.slider("Prior Inpatient Visits", 0, 15, 0)
        number_emergency = st.slider("Prior Emergency Visits", 0, 20, 0)
        number_outpatient = st.slider("Prior Outpatient Visits", 0, 30, 0)
        admission_type_id = st.selectbox("Admission Type",
                                          options=[1, 2, 3, 4, 5, 6, 7, 8],
                                          format_func=lambda x: {1: 'Emergency', 2: 'Urgent',
                                                                   3: 'Elective', 4: 'Newborn',
                                                                   5: 'Not Available', 6: 'NULL',
                                                                   7: 'Trauma Center', 8: 'Not Mapped'}[x])
        discharge_disposition_id = st.selectbox("Discharge Disposition",
                                                 options=[1, 2, 3, 4, 5, 6, 7, 22, 23, 25],
                                                 format_func=lambda x: {1: 'Discharged to home',
                                                                         2: 'Transferred to short-term hospital',
                                                                         3: 'Transferred to SNF',
                                                                         4: 'Transferred to ICF',
                                                                         5: 'Transferred to other inpatient',
                                                                         6: 'Home with home health service',
                                                                         7: 'Left AMA',
                                                                         22: 'Transferred to rehab',
                                                                         23: 'Transferred to long-term care',
                                                                         25: 'Not Mapped'}[x])

    # ── Build feature vector ────────────────────────────────────────
    if st.button("🔍 Predict Readmission Risk", type="primary", use_container_width=True):

        # Start with all features set to 0
        input_data = pd.DataFrame(0, index=[0], columns=feature_names)

        # Fill in the numerical features
        input_data['age'] = age
        input_data['time_in_hospital'] = time_in_hospital
        input_data['num_lab_procedures'] = num_lab_procedures
        input_data['num_procedures'] = num_procedures
        input_data['num_medications'] = num_medications
        input_data['number_diagnoses'] = number_diagnoses
        input_data['number_inpatient'] = number_inpatient
        input_data['number_emergency'] = number_emergency
        input_data['number_outpatient'] = number_outpatient
        input_data['admission_type_id'] = admission_type_id
        input_data['discharge_disposition_id'] = discharge_disposition_id
        input_data['admission_source_id'] = 7  # Default: emergency room
        input_data['num_total_visits'] = number_inpatient + number_emergency + number_outpatient

        # Set one-hot encoded gender
        if gender == "Male":
            if 'gender_Male' in feature_names:
                input_data['gender_Male'] = 1

        # Set one-hot encoded race
        race_col = f'race_{race}'
        if race_col in feature_names:
            input_data[race_col] = 1

        # ── Predict ────────────────────────────────────────────────
        probability = model.predict_proba(input_data)[0][1]
        risk_pct = probability * 100

        # ── Display Results ─────────────────────────────────────────
        st.divider()

        if risk_pct >= 30:
            risk_level = "🔴 HIGH RISK"
            risk_color = "#e74c3c"
        elif risk_pct >= 15:
            risk_level = "🟡 MODERATE RISK"
            risk_color = "#f39c12"
        else:
            risk_level = "🟢 LOW RISK"
            risk_color = "#2ecc71"

        col_r1, col_r2 = st.columns([1, 2])

        with col_r1:
            st.markdown(f"### {risk_level}")
            st.markdown(f"# <span style='color:{risk_color}'>{risk_pct:.1f}%</span>",
                       unsafe_allow_html=True)
            st.caption("Estimated probability of 30-day readmission")

        with col_r2:
            st.markdown("### Recommended Actions")
            if risk_pct >= 30:
                st.markdown("""
                - 🔴 Schedule follow-up appointment within 7 days
                - 🔴 Arrange pharmacist medication review before discharge
                - 🔴 Assign discharge planning nurse
                - 🔴 Consider extended observation
                """)
            elif risk_pct >= 15:
                st.markdown("""
                - 🟡 Schedule follow-up appointment within 14 days
                - 🟡 Provide detailed discharge instructions
                - 🟡 Arrange follow-up phone call within 48 hours
                """)
            else:
                st.markdown("""
                - 🟢 Standard discharge procedure
                - 🟢 Provide written discharge instructions
                - 🟢 Schedule routine follow-up appointment
                """)

        # Risk gauge chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=risk_pct,
            title={'text': "Readmission Risk Score"},
            delta={'reference': 9.0, 'relative': False, 'suffix': '% vs avg'},
            gauge={
                'axis': {'range': [0, 50]},
                'bar': {'color': risk_color},
                'steps': [
                    {'range': [0, 15], 'color': '#eafaf1'},
                    {'range': [15, 30], 'color': '#fef9e7'},
                    {'range': [30, 50], 'color': '#fdedec'}
                ],
                'threshold': {
                    'line': {'color': 'black', 'width': 2},
                    'thickness': 0.75,
                    'value': 9.0
                }
            }
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

        st.caption("⚠️ This is a portfolio project demonstrating an end-to-end ML pipeline. "
                   "It is not intended for clinical decision-making. "
                   "The model was trained on historical data (1999–2008) and has a ROC-AUC of 0.654.")

# ── Footer ──────────────────────────────────────────────────────────
st.sidebar.divider()
st.sidebar.caption("Built by Ruth Mutheu | Data Scientist")
st.sidebar.caption("[GitHub](https://github.com/Mutheu04/diabetes-readmission-project)")