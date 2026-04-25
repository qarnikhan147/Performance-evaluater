import joblib
import numpy as np
import streamlit as st


@st.cache_resource
def load_artifacts():
    model = joblib.load("model.joblib")
    scaler = joblib.load("scaler.joblib")
    return model, scaler


st.title("Student Performance Evaluator")

try:
    model, scaler = load_artifacts()
except Exception as exc:
    st.error("Could not load the trained model or scaler.")
    st.exception(exc)
    st.stop()

attendance = st.slider("Attendance", 0, 100)
assignment = st.slider("Assignment Marks", 0, 100)
quiz = st.slider("Quiz Marks", 0, 100)
mid = st.slider("Mid Marks", 0, 100)
study_hours = st.slider("Study Hours", 0, 12)

if st.button("Evaluate"):
    data = np.array([[attendance, assignment, quiz, mid, study_hours]])
    scaled = scaler.transform(data)
    prediction = model.predict(scaled)

    if prediction[0] == 1:
        st.success("Student Likely PASS")
    else:
        st.error("Student Likely FAIL")
