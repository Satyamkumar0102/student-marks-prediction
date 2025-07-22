import streamlit as st
import pandas as pd
import joblib

# Load model, scaler, and encoders
model = joblib.load("exam_score_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoders = joblib.load("label_encoders.pkl")

st.set_page_config(page_title="Student Exam Score Predictor", layout="centered")
st.title("📊 Student Exam Score Prediction")

# --- User Inputs ---
with st.form("exam_form"):
    st.header("📥 Enter Student Details")
    
    hours_studied = st.number_input("Hours Studied", min_value=0.0, max_value=24.0, step=0.5)
    attendance = st.slider("Attendance (%)", 0, 100)
    parental_involvement = st.slider("Parental Involvement (0-10)", 0, 10)
    access_to_resources = st.slider("Access to Resources (0-10)", 0, 10)
    extracurricular = st.slider("Extracurricular Activities (0-10)", 0, 10)
    sleep_hours = st.number_input("Sleep Hours", min_value=0.0, max_value=24.0, step=0.5)
    previous_scores = st.slider("Previous Exam Scores", 0, 100)
    motivation = st.slider("Motivation Level (0-10)", 0, 10)
    internet = st.selectbox("Internet Access", label_encoders["Internet_Access"].classes_)
    tutoring = st.slider("Tutoring Sessions", 0, 20)
    income = st.number_input("Family Income (in ₹1000s)", min_value=0.0, step=1.0)
    teacher_quality = st.slider("Teacher Quality (0-10)", 0, 10)
    peer = st.selectbox("Peer Influence", label_encoders["Peer_Influence"].classes_)
    physical = st.slider("Physical Activity (hrs/week)", 0, 20)
    parent_edu = st.selectbox("Parental Education Level", label_encoders["Parental_Education_Level"].classes_)
    distance = st.number_input("Distance from Home (in km)", min_value=0.0, step=0.5)

    submit = st.form_submit_button("Predict Exam Score")

# --- Prediction ---
if submit:
    try:
        # Encode categorical inputs
        internet_enc = label_encoders["Internet_Access"].transform([internet])[0]
        peer_enc = label_encoders["Peer_Influence"].transform([peer])[0]
        parent_edu_enc = label_encoders["Parental_Education_Level"].transform([parent_edu])[0]

        # Prepare input as dataframe
        input_df = pd.DataFrame([[
            hours_studied, attendance, parental_involvement, access_to_resources,
            extracurricular, sleep_hours, previous_scores, motivation,
            internet_enc, tutoring, income, teacher_quality, peer_enc,
            physical, parent_edu_enc, distance
        ]], columns=[
            'Hours_Studied', 'Attendance', 'Parental_Involvement', 'Access_to_Resources',
            'Extracurricular_Activities', 'Sleep_Hours', 'Previous_Scores', 'Motivation_Level',
            'Internet_Access', 'Tutoring_Sessions', 'Family_Income', 'Teacher_Quality',
            'Peer_Influence', 'Physical_Activity', 'Parental_Education_Level',
            'Distance_from_Home'
        ])

        # Scale input
        input_scaled = scaler.transform(input_df)

        # Predict
        prediction = model.predict(input_scaled)[0]
        st.success(f"🎯 Predicted Exam Score: **{prediction:.2f}** out of 100")

    except Exception as e:
        st.error(f"⚠️ Error during prediction: {e}")
