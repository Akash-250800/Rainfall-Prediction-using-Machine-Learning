import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load the saved model and preprocessing artifacts
@st.cache_resource
def load_artifacts():
    # Load the imputer
    with open('imputer.pkl', 'rb') as f:
        imputer = pickle.load(f)
    
    # Load the scaler
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    # Load the PCA
    with open('pca.pkl', 'rb') as f:
        pca = pickle.load(f)
    
    # Load the model
    with open('rainfall_model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    # Load the dataset to infer feature columns
    data = pd.read_csv('Rainfall.csv')
    feature_cols = [col for col in data.columns if col != 'rainfall']  # Adjust 'rainfall' to your target column
    
    return imputer, scaler, pca, model, feature_cols

imputer, scaler, pca, model, feature_cols = load_artifacts()

# Prediction function
def predict_rainfall(input_data):
    # Convert input data to DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Apply preprocessing steps
    input_imputed = imputer.transform(input_df)
    input_scaled = scaler.transform(input_imputed)
    input_pca = pca.transform(input_scaled)
    
    # Make prediction
    prediction = model.predict(input_pca)
    return "Rain" if prediction[0] == 1 else "No Rain"

# Streamlit app
st.title("Rainfall Prediction App")
st.write("Enter weather-related details below to predict rainfall likelihood.")

# Create input form dynamically based on features
input_data = {}
with st.form(key='prediction_form'):
    st.write("### Input Features")
    for col in feature_cols:
        # Assume all features are numerical (post-imputation and scaling)
        input_data[col] = st.number_input(f"{col}", step=0.1, format="%.2f")
    
    submit_button = st.form_submit_button(label="Predict")

# Handle prediction
if submit_button:
    try:
        prediction = predict_rainfall(input_data)
        st.success("Prediction completed!")
        st.write(f"Predicted Outcome: **{prediction}**")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
