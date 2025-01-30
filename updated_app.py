import streamlit as st
import numpy as np
import joblib

# Load the saved Random Forest model
model = joblib.load("heart_disease_rf_model.pkl")

# Title and Description
st.title("Heart Disease Prediction")
st.write("""
This app predicts the likelihood of heart disease based on user-provided health metrics. 
Provide the inputs below to get the prediction.
""")

# Input fields for user data
age = st.number_input("Age", min_value=1, max_value=120, value=30)
sex = st.selectbox("Sex", ["Male", "Female"])
cp = st.selectbox("Chest Pain Type (cp)", ["Type 1", "Type 2", "Type 3", "Type 4"])
trestbps = st.number_input("Resting Blood Pressure (trestbps)", min_value=50, max_value=200, value=120)
chol = st.number_input("Cholesterol Level (chol)", min_value=100, max_value=400, value=200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs)", ["Yes", "No"])
restecg = st.selectbox("Resting Electrocardiographic Results (restecg)", ["Normal", "Abnormal", "Hypertrophy"])
thalach = st.number_input("Maximum Heart Rate Achieved (thalach)", min_value=50, max_value=220, value=150)
exang = st.selectbox("Exercise-Induced Angina (exang)", ["Yes", "No"])
oldpeak = st.number_input("ST Depression Induced by Exercise (oldpeak)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
slope = st.selectbox("Slope of the Peak Exercise ST Segment (slope)", ["Upsloping", "Flat", "Downsloping"])
ca = st.number_input("Number of Major Vessels Colored by Fluoroscopy (ca)", min_value=0, max_value=4, value=0)
thal = st.selectbox("Thalassemia (thal)", ["Normal", "Fixed Defect", "Reversible Defect"])

# ✅ Properly encode categorical values
sex_map = {"Male": 1, "Female": 0}
fbs_map = {"Yes": 1, "No": 0}
exang_map = {"Yes": 1, "No": 0}
thal_map = {"Normal": 1, "Fixed Defect": 2, "Reversible Defect": 3}
cp_map = {"Type 1": 1, "Type 2": 2, "Type 3": 3, "Type 4": 4}
restecg_map = {"Normal": 0, "Abnormal": 1, "Hypertrophy": 2}
slope_map = {"Upsloping": 1, "Flat": 2, "Downsloping": 3}

# Convert input values to numeric format
input_data = np.array([
    age,
    sex_map[sex],
    cp_map[cp],
    trestbps,
    chol,
    fbs_map[fbs],
    restecg_map[restecg],
    thalach,
    exang_map[exang],
    oldpeak,
    slope_map[slope],
    ca,
    thal_map[thal]
]).reshape(1, -1)

# ✅ Ensure the input is float
input_data = np.array(input_data, dtype=np.float32)

# Prediction
if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0]
    
    if prediction == 1:
        st.error(f"Prediction: High Risk of Heart Disease! (Confidence: {prob[1]*100:.2f}%)")
    else:
        st.success(f"Prediction: Low Risk of Heart Disease. (Confidence: {prob[0]*100:.2f}%)")
