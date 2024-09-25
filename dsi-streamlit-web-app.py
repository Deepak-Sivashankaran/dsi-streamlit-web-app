
##############################################
########### WEB APP  -  STREAMLIT##########
##############################################

##### IMPORT LIBRARIES

import streamlit as st
import pandas as pd
import joblib

##### LOAD OUR MODEL PIPELINE

model = joblib.load("model.joblib")

#### ADD TITLE AND INSTRUCTIONS

st.title("Purchase Prediction Model")
st.header("Created by Deepak :sunglasses:")
st.subheader("Let's roll!! :balloon:")
st.subheader("Enter customer information and submit for likelihood for purchase")

#### Let's add more details

## Age input form

age = st.number_input(
    label="01. Enter the customer's age",
    min_value = 18,
    max_value=100,
    value = 35
    )

## Let's add one for gender

gender = st.radio("02. Select the gender",
                  options=["Male","Female","Can't say"])



## Credit score input

credit_score = st.number_input(
    label="03. Enter the credit score",
    min_value = 0,
    max_value=1000,
    value = 500
    )


## Submit the input to model

if st.button("Submit for Prediction"):
    
    #store our data in a dataframe (new_data) for prediction of credit score
    new_data = pd.DataFrame({"age" : [age], "gender":[gender],"credit_score":[credit_score]})
    
    #when we click the submit button, the data frame will be complied with 3 input values
    
    # apply model pipeline to the input data and extract the model predicition
    pred_proba = model.predict_proba(new_data)[0][1]
    
    
    # output prediction
    st.subheader(f"Based on the customer attributes our model predicts a purchase probability of {pred_proba:.0%}")



