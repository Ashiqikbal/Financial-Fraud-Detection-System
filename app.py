import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the saved model and scaler
model = joblib.load('fraud_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("🛡️ Financial Fraud Detection System")
st.markdown("Enter transaction details to check for potential fraud.")

# Sidebar for inputs
st.sidebar.header("Transaction Details")

# In a real project, we'd have inputs for V1-V28, 
# but for a demo, let's create inputs for the main ones
amount = st.sidebar.number_input("Transaction Amount", min_value=0.0, value=100.0)
time = st.sidebar.number_input("Time (Seconds since first transaction)", min_value=0, value=3600)

# Since the model expects 30 features (V1-V28 + scaled_time + scaled_amount)
# we will simulate the V1-V28 values as 0 for this simple demo input
if st.button("Analyze Transaction"):
    # Preprocess inputs
    scaled_amount = scaler.transform([[amount]])[0][0]
    # (Note: In your actual model, ensure the order of features matches X_train)
    
    # Create a dummy feature array of 30 features
    features = np.zeros((1, 30)) 
    features[0, 28] = scaled_amount # Assuming 'scaled_amount' was at this index
    features[0, 29] = time # Assuming 'scaled_time' was at this index
    
    # Predict
    prediction = model.predict(features)
    prob = model.predict_proba(features)[0][1]
    
    if prediction[0] == 1:
        st.error(f"🚨 FRAUD DETECTED! (Probability: {prob:.2%})")
    else:
        st.success(f"✅ Transaction Legitimate (Probability of Fraud: {prob:.2%})")
