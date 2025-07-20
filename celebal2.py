# app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Load model, encoders, scaler
model = joblib.load("model.pkl")
label_encoders = joblib.load("label_encoders.pkl")
scaler = joblib.load("scaler.pkl")

# 🖼️ Custom clinic background image from URL
st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://images.unsplash.com/photo-1588776814546-ec7c12c99607?auto=format&fit=crop&w=1920&q=80");
        background-size: cover;
        background-attachment: fixed;
    }
    .main-container {
        background-color: rgba(255, 255, 255, 0.85);
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    .stButton>button {
        background-color: #3A84F5;
        color: white;
        font-weight: bold;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title section
st.markdown('<div class="main-container">', unsafe_allow_html=True)
st.title("🧠 Stroke Risk Prediction Tool")
st.write("This tool helps clinicians and patients estimate the risk of stroke using health indicators.")

st.sidebar.header("📋 Patient Info")

# --- INPUT FUNCTION ---
def user_input():
    gender = st.sidebar.selectbox("Gender", ["Male", "Female", "Other"])
    age = st.sidebar.slider("Age", 0, 100, 30)
    hypertension = st.sidebar.radio("Hypertension", [0, 1], format_func=lambda x: "Yes" if x else "No")
    heart_disease = st.sidebar.radio("Heart Disease", [0, 1], format_func=lambda x: "Yes" if x else "No")
    ever_married = st.sidebar.selectbox("Ever Married", ["Yes", "No"])
    work_type = st.sidebar.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
    residence_type = st.sidebar.selectbox("Residence Type", ["Urban", "Rural"])
    avg_glucose_level = st.sidebar.slider("Glucose Level", 50.0, 300.0, 100.0)
    bmi = st.sidebar.slider("BMI", 10.0, 60.0, 25.0)
    smoking_status = st.sidebar.selectbox("Smoking Status", ["formerly smoked", "never smoked", "smokes", "Unknown"])

    return pd.DataFrame([{
        "gender": gender,
        "age": age,
        "hypertension": hypertension,
        "heart_disease": heart_disease,
        "ever_married": ever_married,
        "work_type": work_type,
        "Residence_type": residence_type,
        "avg_glucose_level": avg_glucose_level,
        "bmi": bmi,
        "smoking_status": smoking_status
    }])

df_input = user_input()

# --- Encode and scale ---
for col in label_encoders:
    df_input[col] = label_encoders[col].transform(df_input[col])
df_input[["age", "avg_glucose_level", "bmi"]] = scaler.transform(df_input[["age", "avg_glucose_level", "bmi"]])

# --- Predict ---
if st.sidebar.button("🔮 Predict Stroke Risk"):
    prediction = model.predict(df_input)[0]
    proba = model.predict_proba(df_input)[0][1]

    result = "💥 High Stroke Risk" if prediction == 1 else "✅ Low Stroke Risk"
    color = "#FF4B4B" if prediction == 1 else "#28a745"

    st.markdown(f"""
        <div class="prediction-box" style="
            background-color: {color}10;
            padding: 1rem;
            border-radius: 10px;
            margin-top: 20px;">
            <h3 style="color:{color};">{result}</h3>
            <p><strong>Probability:</strong> {proba:.2f}</p>
        </div>
    """, unsafe_allow_html=True)

    # Feature importance
    st.subheader("📊 Feature Importance")
    feat_imp = model.feature_importances_
    features = df_input.columns
    fig, ax = plt.subplots()
    ax.barh(features, feat_imp, color="#3A84F5")
    ax.set_title("What affected the prediction?")
    ax.set_xlabel("Importance")
    st.pyplot(fig)

st.markdown("</div>", unsafe_allow_html=True)  # close main-container
st.markdown("---")
st.caption("🔒 For educational use only. Not for real medical diagnosis.")
