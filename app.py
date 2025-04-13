import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

# Set page configuration for SEO and theme
st.set_page_config(
    page_title="Energy Consumption Prediction",
    page_icon="🔋",
    layout="wide", # Changed to wide for better layout
    initial_sidebar_state="auto",
)

# Personal Branding
st.sidebar.markdown("""
<div style='padding: 5px 0; border-bottom: 1px solid #eee;'>
    <p>
        Built with Streamlit by <a href='https://akhundmuzzammil.com' target='_blank'>Muzzammil Akhund</a> 
        <br>
        Connect: <a href='https://github.com/akhundmuzzammil' target='_blank'>GitHub</a> 
         | <a href='https://linkedin.com/in/akhundmuzzammil' target='_blank'>LinkedIn</a>
    </p>
</div>
""", unsafe_allow_html=True)

# Add project information to sidebar
st.sidebar.title('About This Project')
st.sidebar.markdown("""
### [GitHub Repository](github.com/akhundmuzzammil/EnergyConsumptionPrediction)

### Overview
This application predicts energy consumption based on building parameters using a Linear Regression model.

### Features Used
- Building Type
- Square Footage
- Number of Occupants
- Appliances Used
- Average Temperature
- Day of Week

### Data Source
The model is trained on the [Energy Consumption Dataset](https://www.kaggle.com/datasets/govindaramsriram/energy-consumption-dataset-linear-regression) from Kaggle.

### Project Details
This project demonstrates:
- Data preprocessing and feature engineering
- Linear regression modeling
- Standardization of numerical features
- One-hot encoding of categorical features
""")

# Load the trained model and scaler
model = joblib.load('linear_regression_model.pkl')
with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Set page title with standard Streamlit elements
st.title('🔋 Energy Consumption Prediction')

# Create two columns for input fields
col1, col2 = st.columns(2)

with col1:
    st.markdown("### Building Information")
    building_type = st.selectbox(
        'Building Type',
        ['Residential', 'Commercial', 'Industrial']
    )
    square_footage = st.number_input('Square Footage (sq ft)', min_value=500, max_value=50000, value=1000)
    number_of_occupants = st.number_input('Number of Occupants', min_value=1, max_value=100, value=10)

with col2:
    st.markdown("### Environment Details")
    appliances_used = st.number_input('Appliances Used', min_value=1, max_value=50, value=5)
    average_temperature = st.slider('Average Temperature (°C)', min_value=10.0, max_value=35.0, value=20.0, step=0.5)
    day_of_week = st.selectbox(
        'Day of Week',
        ['Weekday', 'Weekend']
    )

# Center the prediction button
predict_button = st.button('Predict Energy Consumption', key='predict_button')

if predict_button:
    # Show a spinner during prediction
    with st.spinner('Calculating energy consumption...'):
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

    # Display result with standard Streamlit styling
    st.success(f"Predicted Energy Consumption: {prediction:.2f} kWh")
    
    # Create columns for displaying results
    results_col1, results_col2 = st.columns([2, 3])
    
    with results_col1:
        st.markdown("### Prediction Result")
        
        # Add a gauge chart for visual representation
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = prediction,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Energy Consumption (kWh)"},
            gauge = {
                'axis': {'range': [None, max(50000, prediction*1.5)]},
                'bar': {'color': "#2e7d32"},
                'steps': [
                    {'range': [0, 10000], 'color': "lightgreen"},
                    {'range': [10000, 30000], 'color': "yellow"},
                    {'range': [30000, 50000], 'color': "orange"}
                ],
            }
        ))
        fig.update_layout(height=250)
        st.plotly_chart(fig, use_container_width=True)
        
    with results_col2:
        st.markdown("### Factors Affecting Consumption")
        
        # Create a bar chart showing the importance of each input
        factor_data = {
            'Factor': ['Square Footage', 'Occupants', 'Appliances', 'Temperature', 'Building Type'],
            'Value': [square_footage/1000, number_of_occupants/10, appliances_used/5, 
                     (average_temperature-10)/25, 
                     0.5 if building_type == 'Residential' else (0.7 if building_type == 'Commercial' else 0.9)]
        }
        
        df_factors = pd.DataFrame(factor_data)
        fig = px.bar(df_factors, x='Factor', y='Value', 
                    color='Value', 
                    color_continuous_scale=['green', 'yellow', 'red'],
                    title="Relative Impact of Factors (normalized)")
        fig.update_layout(height=250)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        **Factors that increase energy consumption:**
        - Larger square footage 🏢
        - More occupants 👥
        - More appliances 🔌
        - Extreme temperatures 🌡️
        - Industrial building type 🏭
        """)

# Add footer with instructions
st.markdown("---")
st.markdown("### How to Use This Tool")
st.markdown("""
1. Enter your building details in the form above
2. Click the 'Predict Energy Consumption' button
3. Review the prediction and insights
4. Adjust input parameters to see how they affect energy consumption
""")

# Add a fun facts section at the bottom
with st.expander("💡 Energy Saving Tips"):
    st.markdown("""
    - **Upgrade to energy-efficient appliances** to reduce consumption by up to 30%
    - **Improve insulation** to lower heating and cooling costs
    - **Use smart thermostats** to optimize temperature control
    - **Install LED lighting** to reduce electricity use
    - **Reduce phantom power** by unplugging devices when not in use    """)
