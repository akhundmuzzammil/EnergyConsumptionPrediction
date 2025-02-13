import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle

# Load the trained model and scaler
model = joblib.load('linear_regression_model.pkl')
with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Set page title
st.title('Energy Consumption Prediction')
st.write('Enter the building details to predict energy consumption')

# Create input fields
building_type = st.selectbox(
    'Building Type',
    ['Residential', 'Commercial', 'Industrial']
)

square_footage = st.number_input('Square Footage', min_value=500, max_value=50000, value=1000)
number_of_occupants = st.number_input('Number of Occupants', min_value=1, max_value=100, value=10)
appliances_used = st.number_input('Appliances Used', min_value=1, max_value=50, value=5)
average_temperature = st.number_input('Average Temperature (°C)', min_value=10.0, max_value=35.0, value=20.0)
day_of_week = st.selectbox(
    'Day of Week',
    ['Weekday', 'Weekend']
)

if st.button('Predict Energy Consumption'):
    # Create a DataFrame with user inputs
    input_data = pd.DataFrame({
        'Building Type': [building_type],
        'Square Footage': [square_footage],
        'Number of Occupants': [number_of_occupants],
        'Appliances Used': [appliances_used],
        'Average Temperature': [average_temperature],
        'Day of Week': [day_of_week]
    })

    # Perform one-hot encoding
    categorical_features = ["Building Type", "Day of Week"]
    input_encoded = pd.get_dummies(input_data, columns=categorical_features, drop_first=True)

    # Ensure all columns from training are present
    expected_columns = ['Square Footage', 'Number of Occupants', 'Appliances Used',
                       'Average Temperature', 'Building Type_Industrial', 'Building Type_Residential',
                       'Day of Week_Weekend']
    
    for col in expected_columns:
        if col not in input_encoded.columns:
            input_encoded[col] = 0

    # Reorder columns to match training data
    input_encoded = input_encoded[expected_columns]

    # Scale numerical features
    numerical_features = ["Square Footage", "Number of Occupants", "Appliances Used", "Average Temperature"]
    input_encoded[numerical_features] = scaler.transform(input_encoded[numerical_features])

    # Make prediction
    prediction = model.predict(input_encoded)[0]

    # Display result
    st.success(f'Predicted Energy Consumption: {prediction:.2f} kWh')

    # Additional insights
    st.write('### Prediction Insights')
    st.write('Factors that generally increase energy consumption:')
    st.write('- Larger square footage')
    st.write('- More occupants')
    st.write('- More appliances')
    st.write('- Extreme temperatures')
    st.write('- Industrial building type')

# Add footer with instructions
st.markdown("""
---
### Instructions:
1. Enter the building details in the fields above
2. Click 'Predict Energy Consumption' to see the prediction
3. Values are automatically validated to ensure they're within reasonable ranges
4. The model uses historical data to make predictions
""")
