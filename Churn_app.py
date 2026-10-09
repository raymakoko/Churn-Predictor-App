

import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Define the numerical and categorical features
numerical_features = ['tenure','MonthlyCharges','TotalCharges']
categorical_features = ['gender','SeniorCitizen','Partner','Dependents','PhoneService','MultipleLines','InternetService','OnlineSecurity',
                        'OnlineBackup',	'DeviceProtection','TechSupport','StreamingTV','StreamingMovies','Contract','PaperlessBilling',
                        'PaymentMethod']

#  Load the saved model and preprocessing tools
final_rf = joblib.load("final_rf.pkl")
encoder = joblib.load("encoder.pkl")
scaler = joblib.load("scaler.pkl")
feature_names= joblib.load("feature_names.pkl")

# Streamlit page settings
st.set_page_config(page_title="Customer Churn Prediction", page_icon="📊", layout="centered")

# App title
st.title("📊Customer Churn Prediction")
st.write("Enter the Customer's information below to predict the likelihood of churn")
st.info("This application is for educational and demonstration purposes." "It requires further testing to be approved as a customer churn predictor")

st.divider()

st.subheader("Customer Informationℹ️")
# Create User input fields
gender = st.selectbox("Gender", ["Female","Male"])

senior_citizen = st.selectbox("Are you a seniour citizen?",["No", "Yes"])

partner = st.selectbox("Do you have a partner?", ["No", "Yes"])

dependents = st.selectbox("Do you have dependents?", ["No", "Yes"])

tenure = st.number_input("How many months have you been with IBM?", min_value=0, max_value=100, value=12)

monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=50.0)

total_charges = st.number_input("Total Charges", min_value=0.0, value=600.0)

# service inputs
st.subheader("Additional Services")

phone_services =  st.selectbox("Do you have a phone service?",['No', 'Yes'])   

multiple_lines = st.selectbox("Do you have multiple lines?",['No phone service', 'No', 'Yes'])

internet_service = st.selectbox("Do you have internet service?",['DSL', 'Fiber optic', 'No'])

online_security = st.selectbox("Do you have online security?", ['No', 'Yes', 'No internet service'])

online_backup = st.selectbox("Do you have online backup?", ['Yes', 'No', 'No internet service'])

device_protection = st.selectbox("Do you have device protection?",['No', 'Yes', 'No internet service'])

tech_support = st.selectbox("Do you have tech support?", ['No', 'Yes', 'No internet service'])

streaming_tv = st.selectbox("Do you have streaming tv services?",['No', 'Yes', 'No internet service'])

streaming_movies = st.selectbox("Do you have streaming movies services?", ['No', 'Yes', 'No internet service'])

# Contract and Billing information

st.subheader("Contract & Billing")
contract = st.selectbox("Which contract have you subscribeed to?",['Month-to-month', 'One year', 'Two year'])

paperless_billing = st.selectbox("Do you use paperless billing",["No", "Yes"] )

payment_method = st.selectbox("How do you pay for IBM?", ['Electronic check', 'Mailed check', 'Bank transfer (automatic)',
       'Credit card (automatic)'])

# Create input data for the model
input_data = pd.DataFrame({
    "gender": [gender],
    "SeniorCitizen": [senior_citizen],
    "Partner": [partner],
    "Dependents": [dependents],
    "tenure": [tenure],
    "MonthlyCharges": [monthly_charges],
    "TotalCharges": [total_charges],
    "PhoneService": [phone_services],
    "MultipleLines": [multiple_lines],
    "InternetService": [internet_service],
    "OnlineSecurity": [online_security],
    "OnlineBackup": [online_backup],
    "DeviceProtection": [device_protection],
    "TechSupport": [tech_support],
    "StreamingTV": [streaming_tv],
    "StreamingMovies": [streaming_movies],
    "Contract": [contract],
    "PaperlessBilling": [paperless_billing],
    "PaymentMethod": [payment_method]     
})

# include encoder and scaler
# separate categorical and numerical features

input_categorical = input_data[categorical_features]
input_numerical = input_data[numerical_features]

# encode categorical features
input_categorical_encoded = encoder.transform(input_categorical)

input_categorical_encoded = pd.DataFrame(
    input_categorical_encoded.toarray(),
    columns=encoder.get_feature_names_out(categorical_features)
)

# scale numerical features
input_numerical_scaled = scaler.transform(input_numerical)

input_numerical_scaled = pd.DataFrame(
    input_numerical_scaled,
    columns=numerical_features
)

# Combine numerical and categorical features
input_final = pd.concat([input_numerical_scaled, input_categorical_encoded], axis=1)

# Ensure features are in the  exact same order
input_final = input_final[feature_names] 

# Prediction
st.divider()
if st.button("🔮Predict Churn", use_container_width=True):
    # Make prediction 
    prediction = final_rf.predict(input_final)

    # Get Churn probability
    probability = final_rf.predict_proba(input_final)[0][1]
    churning_probability = probability * 100

    st.subheader("📊 Prediction Results")

    # Display results
    if prediction[0] == 1:
        st.error("⚠️Higher likelihood for the customer to Churn.")

        st.write("Based on the information provided, the model predicts"
                 "a higher likelihood of churning"
            
        )
        st.metric("Model Probability, f"**{churning_probability:.1f}%**")


    else:
        st.success("✅Lower Likelihood of the customer to churn.")

        st.write("Based on the information provided, the model predicts"
                 "a lower likelihood of churning"
            
        )
        st.metric("Model Probability, f"**{churning_probability:.1f}%**")
