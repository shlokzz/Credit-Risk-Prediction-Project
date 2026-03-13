import numpy as np
import streamlit as st
import pickle
import pandas as pd

# load the saved trained model
with open("models/trained_model.sav", "rb") as f:
    loaded_trained_model = pickle.load(f)

# loading the saved scaler
with open("models/labelencoder.sav", "rb") as f:
    loaded_label_encoder_model = pickle.load(f)

# loading the saved scaler
with open("models/standardscaler.sav", "rb") as f:
    loaded_scaler = pickle.load(f)

# loading the saved scaler
with open("models/one_hot_encoder.sav", "rb") as f:
    loaded_one_hot_encoder = pickle.load(f)

# loading the saved modal columns name
with open("models/model_columns.sav", "rb") as f:
    loaded_model_columns = pickle.load(f)

numerical_cols= ['person_age','person_income','person_emp_length','loan_amnt','loan_int_rate','loan_percent_income','cb_person_cred_hist_length']

category_cols = ['person_home_ownership','loan_intent','loan_grade']

def credit_risk_prediction(input_data):
    # convert to pandas DataFrame
    input_data_as_dataframe = pd.DataFrame([input_data]) 

    # use the saved Label Encoder
    input_data_as_dataframe['cb_person_default_on_file'] = loaded_label_encoder_model.transform(input_data_as_dataframe['cb_person_default_on_file'])

    # use the saved Standard Scaler
    std_data = loaded_scaler.transform(input_data_as_dataframe[numerical_cols])
    std_num_df = pd.DataFrame(std_data, columns= numerical_cols)

    # use the saved One-Hot Encoder
    one_encoded_data = loaded_one_hot_encoder.transform(input_data_as_dataframe[category_cols])
    one_encoded_df = pd.DataFrame(one_encoded_data, columns= loaded_one_hot_encoder.get_feature_names_out(category_cols))

    # Concatenate them together
    final_X = pd.concat([std_num_df, one_encoded_df, input_data_as_dataframe[['cb_person_default_on_file']]], axis=1)

    # Reindex to match the training column order exactly
    final_X = final_X[loaded_model_columns.columns]

    prediction = loaded_trained_model.predict(final_X)
    print(prediction)

    if (prediction[0] == 0):
        return 'Paid'

    else:
        return 'Not Paid. Default'

def main():
    # giving a title
    st.set_page_config(page_title="Credit Risk Analyzer", layout="wide")
    st.title("Credit Risk Prediction System")
    st.markdown("---")

    # use the columns the UI look organized
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Personal Information")

        # getting the input data from the user
        PersonAge = st.number_input("Enter Applicant Age", min_value=18, max_value=110, step=1)
        PersonIncome = st.number_input("Enter Annual Income ($)", min_value=0, step= 1000)
        PersonEmpLength = st.number_input("Enter Employment Length (Years)", min_value=0.0, max_value=65.0, step=0.5)
        CbPersonCreditHistoryLength = st.number_input("Enter Credit History Length (Years)", min_value=0, max_value=50)

    with col2:  
        st.subheader("Loan Details")

        # getting the input data from the user
        LoanAmount = st.number_input("Enter Requested Loan Amount ($)", min_value=0, step=500)
        LoanInterestRate = st.number_input("Enter Interest Rate (%)", min_value=0.0, max_value=30.0, step=0.1)
        LoanPercentIncome = st.number_input("Enter Loan Percent Income (0.0 - 1.0)", min_value=0.0, max_value=1.0, step=0.01)
        
        c1, c2,c3,c4 = st.columns(4)

        with c1:
            PersonHomeOwnership = st.selectbox("Home Ownership",["RENT","OWN","MORTGAGE","OTHER"])

        with c2:
            LoanIntent = st.selectbox("Loan Purpose",["PERSONAL","EDUCATION","MEDICAL","REVENUE","HOMEMPORVEMENT" ,"DEBTCONSOLIDATION"])

        with c3:
            LoanGrade = st.selectbox("Loan Grade",["A","B","C","D","E","F","G"])

        with c4:
            CbPersonDefaultOnFile = st.selectbox("Previous Default History?", ["Y","N"] )

    # code for prediction
    prediction = ''

    # creating a button for prediction
    if st.button('Credit Test Result'):

        inputs = [PersonAge, PersonIncome, PersonHomeOwnership, PersonEmpLength, LoanIntent, LoanGrade, LoanAmount, LoanInterestRate, LoanPercentIncome, CbPersonDefaultOnFile, CbPersonCreditHistoryLength]

        prediction = credit_risk_prediction(inputs)
        st.success(prediction)

if __name__ == '__main__':
    main()