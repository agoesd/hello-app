import streamlit as st
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib

# Set page title
st.set_page_config(page_title="Procurement Time Prediction App", layout="wide")

# Set sidebar
st.sidebar.title("Procurement Time Prediction")

# Create input fields for features
feature1 = st.sidebar.number_input("Feature 1")
feature2 = st.sidebar.number_input("Feature 2")
# Add more feature inputs if needed

# Load the joblib model
model = joblib.load("https://github.com/agoesd/hello-app/blob/main/modeldecisiontree_lamatender.joblib")

# Procurement time prediction
st.subheader("Procurement Time Prediction")

# Make prediction
prediction = model.predict([[feature1, feature2]])[0]

# Display prediction
st.write("Predicted Procurement Time:", prediction)
