import joblib
import streamlit as st
import numpy as np
import pandas as pd

## Load trained model
model = joblib.load("models/depression_rf_tuned_model.pkl")
kmeans = joblib.load("models/kmeans_model.pkl")

## Streamlit app
st.title("Depression Prediction")
st.subheader("2401058I Temasek Polytechnic MLDP Project - [SIMULATED]")
st.text("===== THE FOLLOWING IS A SCHOOL PROJECT. DO NOT USE FOR REAL USES =======")
st.text("Please enter the students' information to predict the likelihood of depression.")
st.text("Note that this is not authoritative - students predicted to be at risk of depression should seek help from a mental health professional for proper diagnosis and support.")

## User inputs
gender_selected = st.selectbox("Select Gender", ["Male", "Female"])
age_selected = st.slider("Select Age", 
                                min_value=12, 
                                max_value=60,
                                step = 1, 
                                value=18)
cgpa_selected = st.slider("Select CGPA (Indian System)", 
                                min_value=0.0, 
                                max_value=10.0, 
                                value=6.0)
prior_suicidal_thoughts_selected = st.selectbox("Prior Suicidal Thoughts", ["Yes", "No"])
family_history_selected = st.selectbox("Family History of Depression", ["Yes", "No"])
dietary_habits_selected = st.selectbox("Dietary Habits", ["Healthy", "Moderate", "Unhealthy"])
sleep_duration_selected = st.selectbox("Sleep Duration", ["Less than 5 hours", "5-6 hours", "7-8 hours", "More than 8 hours"])
financial_stress_selected = st.selectbox("Financial Stress (1.0 = none, 5.0 = high)", ["1.0", "2.0", "3.0", "4.0", "5.0"])
academic_pressure_selected = st.selectbox("Academic Pressure (0 = none, 5 = high)", ["0", "1", "2", "3", "4", "5"])
work_study_hours_selected = st.slider("Work/Study Hours", max_value = 12, min_value = 0, step = 1)
study_satisfaction_selected = st.slider("Study Satisfaction", max_value = 5, min_value = 0, step = 1)

## Predict button
if st.button("Predict depression"):
    input_data = {
        'Suicidal Thoughts': [prior_suicidal_thoughts_selected],
        'Academic Pressure': [academic_pressure_selected],
        'CGPA': [cgpa_selected],
        'Age': [age_selected],
        'Work/Study Hours': [work_study_hours_selected],
        'Study Satisfaction': [study_satisfaction_selected],
        "Financial Stress": [financial_stress_selected],
        "Dietary Habits": [dietary_habits_selected],
        "Gender": [gender_selected],
        "Family History": [family_history_selected],
        "Sleep Duration": [sleep_duration_selected]
    }

    df_input = pd.DataFrame(
        input_data
    )

    df_input = pd.get_dummies(df_input)
    print('ohe input as \n', df_input)

    df_input = df_input.reindex(columns = kmeans.feature_names_in_, fill_value=0)   

    cluster = kmeans.predict(df_input)
    print("successfully predicted cluster as ", cluster)

    # kmeans came before dropping job satisfaction and work pressure
    # hack it by dropping lol
    df_input = df_input.drop(labels=["Job Satisfaction", "Work Pressure"], axis=1)

    df_input["Cluster"] = cluster

    y_unseen_pred = model.predict(df_input)[0]

    st.success(f"Predicted depression: {"Yes" if y_unseen_pred else "No"}")

## Page design
st.markdown(
    f"""
    <style>
    .stApp {{
        background-color: #4D4C99;
    }}
    </style>
    """,
    unsafe_allow_html=True
)