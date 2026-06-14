import streamlit as st
import pickle
import numpy as np

model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("💰 Loan Approval Prediction (ML Only)")

st.write("Enter applicant details")

age = st.number_input("Age", 18, 80, 30)
income = st.number_input("Income", 0, 200000, 50000)
loan_amount = st.number_input("Loan Amount", 0, 200000, 20000)
employment = st.number_input("Employment Years", 0, 40, 5)
debt = st.number_input("Existing Debt", 0, 500000, 10000)

if st.button("Predict"):

    data = np.array([[age, income, loan_amount, employment, debt]])

    data_scaled = scaler.transform(data)

    prediction = model.predict(data_scaled)

    if prediction[0] == 1:
        st.success("Loan Approved")
    else:
        st.error("Loan Rejected")