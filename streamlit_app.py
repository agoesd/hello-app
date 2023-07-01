import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Set page title
st.set_page_config(page_title="Procurement Time Prediction App", layout="wide")

# Set sidebar
st.sidebar.title("Procurement Time Prediction")

# Upload data file
data_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

if data_file is not None:
    # Load data
    data = pd.read_csv(data_file)

    # Display the raw data
    st.subheader("Raw Data")
    st.write(data)

    # Data preprocessing
    # Add your data preprocessing steps here

    # Split data into features and target
    X = data.drop("Procurement Time", axis=1)
    y = data["Procurement Time"]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model training
    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    # Model evaluation
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    # Display model performance metrics
    st.subheader("Model Performance")
    st.write("Mean Squared Error:", mse)
    st.write("Mean Absolute Error:", mae)

    # Procurement time prediction form
    st.subheader("Procurement Time Prediction")

    # Create input fields for features
    feature1 = st.number_input("Feature 1")
    feature2 = st.number_input("Feature 2")
    # Add more feature inputs if needed

    # Make prediction
    prediction = model.predict([[feature1, feature2]])[0]

    # Display prediction
    st.write("Predicted Procurement Time:", prediction)
