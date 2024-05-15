# Import the libraries.
import numpy as np
import pandas as pd
import pickle
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
import gzip




compressed_pickle_file = 'models/rf_final.pkl.gz'

# Open the compressed pickle file and decompress it
with gzip.open(compressed_pickle_file, 'rb') as f:
    # Load the decompressed pickle file
    rf_final = pickle.load(f)

# Load Ordinal Encoder model
with open('models/cs_enc.pkl', 'rb') as f:
    cs_encode = pickle.load(f)

# Load Label Encoder model
with open('models/cs_le.pkl', 'rb') as f:
    cs_le = pickle.load(f)

cat = ['Credit_Mix']

df_final = pd.read_csv('data/df_final.csv')

# Streamlit UI
st.set_page_config(page_title= "CREDIT SCORE",
                   layout= "wide")
st.title('**Credit Score Prediction**', anchor = False)
st.divider()

# Define User input
def user_input_data():
    # Sidebar Configuration

    monthly_inhand_salary = st.sidebar.number_input("Monthly Inhand Salary", value=8000.00)
    interest_rate = st.sidebar.number_input("Interest Rate (%)",0.0,32.0,0.0)
    delay_from_due_date = st.sidebar.number_input("Delay from Due Date (days)", 0.0, 70.0, 0.0)
    changed_credit_limit = st.sidebar.number_input("Changed Credit Limit",0.0,30.0,0.0,0.0)
    credit_mix = st.sidebar.selectbox("Credit Mix",df_final['Credit_Mix'].unique())
    outstanding_debt = st.sidebar.number_input("Outstanding Debt", value = 0.00)
    credit_history_age = st.sidebar.number_input("Credit History Age (days)", 0.0, 450.0, 0.0)
    amount_invested_monthly = st.sidebar.number_input("Amount Invested Monthly", value=0.00)
    monthly_balance = st.sidebar.number_input("Monthly Balance", value=0.00)
    
    
    data = {'Monthly_Inhand_Salary': [monthly_inhand_salary],
            'Amount_invested_monthly': [amount_invested_monthly],
            'Monthly_Balance': [monthly_balance],
            'Delay_from_due_date': [delay_from_due_date],
            'Changed_Credit_Limit': [changed_credit_limit],
            'Credit_History_Age': [credit_history_age],
            'Credit_Mix': [credit_mix],
            'Interest_Rate': [interest_rate],
            'Outstanding_Debt': [outstanding_debt],      
            }
    input_data = pd.DataFrame(data, index=[0])  
    
    return input_data


# Sidebar Configuration
# Add a sidebar to the web page. 
st.sidebar.header("User input parameter")

# get input datas
col1, col2 = st.columns([4, 6])
# st.sidebar.write('Developed by ...')
# st.sidebar.write('Contact at ...')


df = user_input_data() 


with col1:
    if st.checkbox('Show User Inputs:', value=True):
        st.dataframe(df.astype(str).T.rename(columns={0:'input_data'}).style.highlight_max(axis=0))

with col2:
    for i in range(2): 
        st.markdown('#')
    if st.button('Make Prediction'):   
        try:
            
            df[cat] = cs_encode.transform(df[cat])
            # Make prediction
            prediction = rf_final.predict(df)
            # Inverse transform the prediction
            prediction = cs_le.inverse_transform(prediction)[0]
            st.success(f'Credit score probability is: {prediction}')
        except Exception as e:
            st.error(f"An error occurred: {e}")