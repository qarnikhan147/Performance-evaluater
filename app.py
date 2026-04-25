import joblib
import numpy as np
import streamlit as st


@st.cache_resource
def load_artifacts():
    model = joblib.load("model.joblib")
    scaler = joblib.load("scaler.joblib")
    return model, scaler


st.title("Student Performance Evaluator")

st.write("This app uses an Artificial Neural Network (ANN) to predict if a student will PASS or FAIL based on their academic metrics. Adjust the inputs below and click Evaluate.")

try:
    model, scaler = load_artifacts()
except Exception as exc:
    st.error("Could not load the trained model or scaler.")
    st.exception(exc)
    st.stop()

# Change sliders to selectboxes
attendance_options = list(range(0, 101, 5))
assignment_options = list(range(0, 101, 5))
quiz_options = list(range(0, 101, 5))
mid_options = list(range(0, 101, 5))
study_hours_options = list(range(0, 13))

attendance = st.selectbox("Attendance (%)", attendance_options, index=10)  # default 50
assignment = st.selectbox("Assignment Marks (%)", assignment_options, index=10)
quiz = st.selectbox("Quiz Marks (%)", quiz_options, index=10)
mid = st.selectbox("Mid Marks (%)", mid_options, index=10)
study_hours = st.selectbox("Study Hours per Week", study_hours_options, index=5)  # default 5

st.subheader("Model Performance")
st.image("confusion_matrix.png", caption="Confusion Matrix of the ANN Model")

if st.button("Evaluate"):
    data = np.array([[attendance, assignment, quiz, mid, study_hours]])
    scaled = scaler.transform(data)
    prediction = model.predict(scaled)

    if prediction[0] == 1:
        st.success("🎉 Student Likely to PASS! Keep up the good work.")
    else:
        st.error("⚠️ Student Likely to FAIL. Consider additional study or support.")

st.write("---")
st.write("Developed for educational purposes. Model trained on sample data.")
