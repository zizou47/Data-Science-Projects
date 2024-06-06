import streamlit as st
import pandas as pd
from utilities import prediction_pipeline

def main():
    st.title("Bank Customer Churn Predictor")

    # Collecting user inputs
    credit_score = st.number_input("Customer Credit Score", min_value=0, max_value=1000, format="%d")
    country = st.selectbox("Choose customer geography", ("Spain", "France", "Germany"))
    gender = st.radio("Enter Customer Gender", ("Male", "Female"))
    age = st.number_input("Customer Age", min_value=0, format="%d")
    tenure = st.number_input("Customer Tenure", min_value=0, max_value=10, format="%d")
    balance = st.number_input("Customer Balance", min_value=1.0, format="%f")
    products_number = st.selectbox("Choose which product did the customer use", (1, 2, 3, 4))
    credit_card = 1 if st.radio("Did The Customer Have a Credit Card?", ('Yes', 'No')) == 'Yes' else 0
    active_member = 1 if st.radio("Is The Customer an Active Member?", ('Yes', 'No')) == 'Yes' else 0
    estimated_salary = st.number_input("Customer Estimated Salary", min_value=1.0, format="%f")

    # Creating a data dictionary
    data = {
        'credit_score': credit_score,
        'country': country,
        'gender': gender,
        'age': age,
        'tenure': tenure,
        'balance': balance,
        'products_number': products_number,
        'credit_card': credit_card,
        'active_member': active_member,
        'estimated_salary': estimated_salary
    }

    # Converting data dictionary to DataFrame
    data_df = pd.DataFrame(data, index=[0])

    # Predicting customer churn
    if st.button('Predict!'):
        prediction = prediction_pipeline(data_df)
        if prediction[0] == 1:
            st.write('This customer is likely to churn.')
        else:
            st.write('This customer is not likely to churn.')

if __name__ == "__main__":
    main()
