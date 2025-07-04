import streamlit as st
import pandas as pd
import numpy as np
import joblib,pickle


with open('preprocessor.pkl', 'rb') as f:
    preprocessor = pickle.load(f)

model = joblib.load("RandomForestClassifier_best.pkl")

st.title("Customer Churn Prediction")

age = st.slider("Age", min_value=18, max_value=100)

gender = st.selectbox("Gender", ["Male", "Female"])
tenure = st.number_input("Tenure (in months)", min_value=0, max_value=100, value=12)
usage = st.number_input("Usage Frequency", min_value=0.0, max_value=100.0, value=3.5)
support_calls = st.number_input("Support Calls", min_value=0, max_value=100, value=1)
delay = st.number_input("Payment Delay (in days)", min_value=0, max_value=60, value=0)

subscription = st.selectbox("Subscription Type", ["Basic", "Standard", "Premium"])

contract_length = st.selectbox("Contract Length", ["Monthly", "Quarterly" "Annual"])
spend = st.number_input("Total Spend ($)", min_value=0.0, max_value=1000.0, value=100.0)

interaction = st.number_input("Days Since Last Interaction", min_value=0, max_value=365, value=10)

if st.button("Predict"):
   
    input_data = pd.DataFrame([{
        "Gender": gender,
        "Subscription Type": subscription,
        "Contract Length": contract_length,
        "Age": age,
        "Tenure": tenure,
        "Usage Frequency": usage,
        "Support Calls": support_calls,
        "Payment Delay": delay,
        "Total Spend": spend,
        "Last Interaction": interaction
    }])

    
    transformed_input = preprocessor.transform(input_data)

    
    prediction = model.predict(transformed_input)[0]
    probability = model.predict_proba(transformed_input)[0][int(prediction)]

  
    st.markdown(f"### Predicted Churn: {'Yes' if int(prediction) == 1 else 'No'}")
    st.markdown(f"**Confidence:** {probability:.2%}")
