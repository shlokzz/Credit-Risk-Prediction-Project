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
    st.title("Credit Risk Prediction Web App")

    # getting the input data from the user

    PersonAge = st.text_input("Enter Age")
    PersonIncome = st.text_input("Enter Person Income")
    PersonHomeOwnership = st.text_input("Enter Person Home Ownership")
    PersonEmpLength = st.text_input("Enter Person Employment Length")
    LoanIntent = st.text_input("Enter Loan Intent")
    LoanGrade = st.text_input("Enter Loan Grade")
    LoanAmount = st.text_input("Enter Loan Amount")
    LoanInterestRate = st.text_input("Enter Interest Rate")
    LoanPercentIncome = st.text_input("Enter Loan Percent Income")
    CbPersonDefaultOnFile = st.text_input("Enter Cb Person Default On File")
    CbPersonCreditHistoryLength = st.text_input("Enter Cb Person Credit History Length")


    # code for prediction
    prediction = ''

    # creating a button for prediction
    
    if st.button('Credit Test Result'):

        inputs = [PersonAge, PersonIncome, PersonHomeOwnership, PersonEmpLength, LoanIntent, LoanGrade, LoanAmount, LoanInterestRate, LoanPercentIncome, CbPersonDefaultOnFile, CbPersonCreditHistoryLength]

        #check for empty fields
        if any(value.strip() == "" for value in inputs):
            st.error("Please Fill in all the Required Fields.")
        

if __name__ == '__main__':
    main()