import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# 1. Set Page Configuration & Title
st.set_page_config(
    page_title="Jaya Jaya Institut - Student Retention System",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Student Retention Predictive Analytics Platform")
st.markdown("Early warning system powered by Machine Learning to detect student dropout risks.")
st.markdown("---")

# 2. Load the Trained Model Artifact
MODEL_PATH = "model.joblib"

@st.cache_resource
def load_prediction_model(path):
    if not os.path.exists(path):
        st.error(f"Model file not found at {path}. Please run the training notebook first.")
        return None
    return joblib.load(path)

model = load_prediction_model(MODEL_PATH)

if model is not None:
    # 3. Create Two Columns for Input Parameters
    st.subheader("Student Feature Input Form")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Demographic & Financial Profile")
        age = st.number_input("Age at Enrollment", min_value=15, max_value=60, value=20, step=1)
        gender = st.selectbox("Gender", options=[("Female", 0), ("Male", 1)], format_func=lambda x: x[0])[1]
        debtor = st.selectbox("Has Outstanding Debt?", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0])[1]
        tuition_up_to_date = st.selectbox("Tuition Fees Up to Date?", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0])[1]
        scholarship = st.selectbox("Scholarship Holder?", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0])[1]
        
    with col2:
        st.markdown("### Academic Performance")
        sem1_approved = st.number_input("1st Semester Approved Units", min_value=0, max_value=30, value=5, step=1)
        sem1_grade = st.number_input("1st Semester Average Grade (0-20)", min_value=0.0, max_value=20.0, value=12.0, step=0.1)
        sem2_approved = st.number_input("2nd Semester Approved Units", min_value=0, max_value=30, value=5, step=1)
        sem2_grade = st.number_input("2nd Semester Average Grade (0-20)", min_value=0.0, max_value=20.0, value=12.0, step=0.1)

    st.markdown("---")
    
    # 4. Prediction Execution Trigger
    if st.button("Run Risk Assessment Analysis", type="primary"):
        # Map inputs directly to the matching DataFrame structure used in training
        input_data = pd.DataFrame([{
            'Age_at_enrollment': age,
            'Gender': gender,
            'Debtor': debtor,
            'Tuition_fees_up_to_date': tuition_up_to_date,
            'Scholarship_holder': scholarship,
            'Curricular_units_1st_sem_approved': sem1_approved,
            'Curricular_units_1st_sem_grade': sem1_grade,
            'Curricular_units_2nd_sem_approved': sem2_approved,
            'Curricular_units_2nd_sem_grade': sem2_grade
        }])
        
        # Calculate raw probabilities
        dropout_probability = model.predict_proba(input_data)[0, 1]
        
        # Display Results using the custom 0.35 threshold
        st.subheader("Analysis Results")
        
        # Visual metrics layout
        m_col1, m_col2 = st.columns(2)
        with m_col1:
            st.metric(label="Calculated Dropout Probability", value=f"{dropout_probability * 100:.1f}%")
        
        with m_col2:
            BUSINESS_THRESHOLD = 0.35
            if dropout_probability >= BUSINESS_THRESHOLD:
                st.error("SYSTEM STATUS: HIGH RISK (Flagged for Early Intervention)")
                st.markdown(
                    "**Recommendation:** This student falls above the 35% business risk threshold. "
                    "Schedule an academic counseling session and verify if financial assistance or tutoring is required."
                )
            else:
                st.success("SYSTEM STATUS: LOW RISK (Stable Academic Progression)")
                st.markdown(
                    "**Recommendation:** The student displays normal progression patterns. "
                    "Continue standard institutional tracking protocols."
                )