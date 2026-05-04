import streamlit as st
import joblib
import numpy as np
import os
import requests

MODEL_URL = "https://huggingface.co/sanie99/student-score-prediction-rf-model/resolve/main/rf_model.pkl"
SCALER_URL = "https://huggingface.co/sanie99/student-score-prediction-rf-model/resolve/main/scaler.pkl"

MODEL_PATH = "models/rf_model.pkl"
SCALER_PATH = "models/scaler.pkl"

def download_file(url, path):
    os.makedirs("models", exist_ok=True)
    response = requests.get(url)
    with open(path, "wb") as f:
        f.write(response.content)

@st.cache_resource
def load_model_and_scaler():
    # Download if missing
    if not os.path.exists(MODEL_PATH):
        download_file(MODEL_URL, MODEL_PATH)

    if not os.path.exists(SCALER_PATH):
        download_file(SCALER_URL, SCALER_PATH)

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    return model, scaler


st.title("📊 Student Progress Tracker")

study_time = st.number_input("Study Time (hours/week)", min_value=0, max_value=40, step=1)
attendance = st.number_input("Attendance (%)", min_value=0, max_value=100, step=1)
maths_score = st.number_input("Math Score", min_value=0, max_value=100, step=1)
science_score = st.number_input("Science Score", min_value=0, max_value=100, step=1)
english_score = st.number_input("English Score", min_value=0, max_value=100, step=1)

if st.button("Predict Overall Score"):
    try:
        model, scaler = load_model_and_scaler()
        features = np.array([[study_time, attendance, maths_score, science_score, english_score]])
        features_scaled = scaler.transform(features)
        predicted_score = model.predict(features_scaled)[0]
        st.success(f"🎯 Predicted Overall Score: {round(predicted_score, 2)}")

    except Exception as e:
        st.error("⚠️ Error loading model. Please check your model URLs.")
        st.exception(e)