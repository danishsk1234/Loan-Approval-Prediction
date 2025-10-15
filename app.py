import streamlit as st
import pandas as pd
import pickle

# Load trained model
model = pickle.load(open('model.pkl', 'rb'))

st.title("🏦 Loan Approval Prediction")

# --- USER INPUT ---
gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Married", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])
applicant_income = st.number_input("Applicant Income", min_value=0)
coapplicant_income = st.number_input("Coapplicant Income", min_value=0)
loan_amount = st.number_input("Loan Amount", min_value=0)
loan_amount_term = st.number_input("Loan Amount Term", min_value=0)
credit_history = st.selectbox("Credit History", [1.0, 0.0])
property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

# --- CREATE INPUT DATAFRAME ---
if st.button("Predict Loan Status"):
    input_data = pd.DataFrame({
        'Gender': [gender],
        'Married': [married],
        'Dependents': [dependents],
        'Education': [education],
        'Self_Employed': [self_employed],
        'ApplicantIncome': [applicant_income],
        'CoapplicantIncome': [coapplicant_income],
        'LoanAmount': [loan_amount],
        'Loan_Amount_Term': [loan_amount_term],
        'Credit_History': [credit_history],
        'Property_Area': [property_area]
    })

    # --- CONVERT CATEGORICAL TO NUMERIC ---
    input_data['Gender'] = input_data['Gender'].map({'Male': 1, 'Female': 0})
    input_data['Married'] = input_data['Married'].map({'Yes': 1, 'No': 0})
    input_data['Education'] = input_data['Education'].map({'Graduate': 1, 'Not Graduate': 0})
    input_data['Self_Employed'] = input_data['Self_Employed'].map({'Yes': 1, 'No': 0})
    input_data['Credit_History'] = input_data['Credit_History'].astype(float)

    # One-hot encode Property_Area and Dependents
    input_data = pd.get_dummies(input_data, columns=['Property_Area', 'Dependents'], drop_first=True)

    # --- ALIGN COLUMNS WITH TRAINING DATA ---
    missing_cols = set(model.feature_names_in_) - set(input_data.columns)
    for c in missing_cols:
        input_data[c] = 0
    input_data = input_data[model.feature_names_in_]

    # --- PREDICT ---
    prediction = model.predict(input_data)[0]

    # --- DISPLAY RESULT ---
    if prediction == 1:
        st.success("✅ Loan Approved!")
    else:
        st.error("❌ Loan Rejected!")

