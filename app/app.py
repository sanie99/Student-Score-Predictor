import streamlit as st
import joblib
import numpy as np

st.title("Student Progress Tracker")

study_time = st.number_input("Enter Study Time (hours per week)", min_value=0, max_value=40, step=1)
attendance = st.number_input("Enter Attendance (percentage)", min_value=0, max_value=100, step=1)
maths_score = st.number_input("Enter Math Score", min_value=0, max_value=100, step=1)
science_score = st.number_input("Enter Science Score", min_value=0, max_value=100, step=1)
english_score = st.number_input("Enter English Score", min_value=0, max_value=100, step=1)

if st.button("Predict Overall Score"):

    # Load the trained model
    model = joblib.load("../models/rf_model.pkl")

    # Scale the input features using the same scaler used during training
    scaler_model = joblib.load("../models/scaler.pkl")
    features = np.array([[study_time, attendance, maths_score, science_score, english_score]]).reshape(1, -1)
    features_scaled = scaler_model.transform(features)

    # Make a prediction based on the input study time
    predicted_score = model.predict(features_scaled)[0]
    st.success(f"Predicted Overall Score: {predicted_score}")