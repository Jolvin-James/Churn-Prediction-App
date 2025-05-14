# Female --> 0 and Male --> 1
# Churn --> 1 and No Churn --> 0

import streamlit as st
import joblib
import numpy as np

scaler = joblib.load("scaler.pkl")
model = joblib.load("model.pkl")

st.title("Churn Prediction App")

st.divider()

st.write("Please enter the values and click on the button to predict if the customer will churn.")

st.divider()

age = st.number_input("Enter Age", min_value=18, max_value=100, value=30)

tenure = st.number_input("Enter Tenure", min_value=0, max_value=130, value=10)

usage = st.number_input("Enter Usage", min_value=0, max_value=30, value=15)

support_calls = st.number_input("Enter Support Calls", min_value=0, max_value=10, value=5)

payments = st.number_input("Enter Payments", min_value=0, max_value=30, value=10)

totalcharges = st.number_input("Enter Total Charges", min_value=0, max_value=1000, value=500)

interaction = st.number_input("Enter Interaction", min_value=1, max_value=30, value=15)

gender = st.selectbox("Enter your gender", ['Male', 'Female'])
if gender:
    genderselected = 0 if gender == 'Female' else 1

subscription = st.selectbox("Enter Subscription", ['Basic', 'Standard', 'Premium'])
if subscription:
    if subscription == 'Basic':
        subscriptionselected = 0
    elif subscription == 'Premium':
        subscriptionselected = 1
    else:
        subscriptionselected = 2
# 0 --> Basic, 1 --> Premium, 2 --> Standard

contract = st.selectbox("Enter Contract", ['Quarterly', 'Monthly', 'Annual'])
if contract:
    if contract == 'Annual':
        contractselected = 0
    elif contract == 'Monthly':
        contractselected = 1
    else:
        contractselected = 2
# 0 --> Annual, 1 --> Monthly, 2 --> Quarterly


st.divider()

predict_button = st.button("Predict")

if predict_button:
    # Preprocess the input data
    X = np.array([[age, genderselected, tenure, usage, support_calls, payments, subscriptionselected,
                   contractselected, totalcharges, interaction]])
    X_array = scaler.transform(X)

    # Make prediction
    prediction = model.predict(X_array)[0]  

    if prediction == 1:
        predicted = "Churn"
    else:
        predicted = "No Churn"

    st.write(f"Prediction: {predicted}")

else:
    st.warning("Please enter the values and click on the button to predict.")

