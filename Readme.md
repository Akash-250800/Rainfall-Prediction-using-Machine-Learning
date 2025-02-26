# Rainfall Prediction Using Machine Learning

This project implements a machine learning-based system to predict rainfall likelihood using a pre-trained model. It includes a pipeline with class balancing, dimensionality reduction, and model selection, along with a Streamlit web application for user-friendly predictions.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Dataset](#dataset)
- [Training the Model](#training-the-model)
- [Running the Streamlit App](#running-the-streamlit-app)
- [Usage](#usage)
- [File Structure](#file-structure)
- [Contributing](#contributing)
- [License](#license)

## Overview
The Rainfall Prediction system uses a dataset (`Rainfall.csv`) to train a machine learning model that predicts rainfall (binary: Rain/No Rain) based on weather-related features. The pipeline handles missing values, balances classes, reduces dimensionality, and tunes a model (default: Logistic Regression, with Random Forest as an alternative). A Streamlit app (`app.py`) allows users to input data and receive predictions interactively.

## Features
- **Data Preprocessing**:
  - Imputation of missing values with `SimpleImputer`.
  - Feature scaling with `StandardScaler`.
  - Class balancing with SMOTE.
  - Dimensionality reduction with PCA (95% variance retained).
- **Model Options**:
  - Logistic Regression with hyperparameter tuning (default).
  - RandomForestClassifier (alternative, can be implemented by modifying the training script).
- **Web Interface**: Streamlit app for easy prediction input and output.
- **Modularity**: Saved artifacts enable reuse without retraining.

## Requirements
- Python 3.8+
- Libraries:
  - `pandas`
  - `numpy`
  - `scikit-learn`
  - `imblearn` (for SMOTE)
  - `streamlit`
  - `pickle` (built-in)
Dataset
File: rainfall.csv
Description: Contains weather features (e.g., temperature, humidity, wind speed, pressure) and a target column (rainfall, binary: 0 = No Rain, 1 = Rain).
Source: [Specify your dataset source, e.g., "Collected from local weather station" or "Public dataset from Kaggle"].
Note: The dataset may contain missing values (NaNs), which are handled by the pipeline.
Training the Model
Ensure rainfall.csv is in the project directory.
Run the training script (example provided separately or in a Jupyter notebook):
python
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
import pickle
# ... (full preprocessing and training code)
Options:
Default: Logistic Regression with GridSearchCV for tuning C and solver.
Alternative: Replace with RandomForestClassifier (e.g., tune n_estimators, max_depth).
Outputs:
imputer.pkl: Imputer for missing values.
scaler.pkl: Feature scaler.
pca.pkl: PCA transformation.
rainfall_model.pkl: Trained model (Logistic Regression or Random Forest).
Running the Streamlit App
Ensure all .pkl files and rainfall.csv are in the same directory as app.py.
Launch the app:
bash
streamlit run app.py
Open your browser at http://localhost:8501 to access the interface.
Usage
Training: Run the training script to generate model and preprocessing files (if not already done).
Prediction:
Open the Streamlit app.
Enter numerical values for all features (e.g., temperature, humidity) in the form.
Click "Predict" to see if rainfall is likely ("Rain" or "No Rain").
Example Input (adjust to your dataset):
Temperature: 23.5
Humidity: 75.0
Wind Speed: 6.0
Pressure: 1010.0
File Structure
rainfall-prediction/
│
├── rainfall.csv           # Training dataset
├── app.py                 # Streamlit app for predictions
├── imputer.pkl            # Imputer for missing values
├── scaler.pkl             # StandardScaler for features
├── pca.pkl                # PCA transformation
├── rainfall_model.pkl     # Trained ML model (Logistic Regression or Random Forest)
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
Contributing
Contributions are welcome! To contribute:
Fork the repository.
Create a feature branch (git checkout -b feature/your-feature).
Commit changes (git commit -m "Add your feature").
Push to the branch (git push origin feature/your-feature).
Open a pull request.
License
This project is licensed under the MIT License - see the LICENSE file for details (add a LICENSE file if applicable).

### Notes on RandomForestClassifier Integration
- **Mention in README**: I’ve included `RandomForestClassifier` as an alternative model option in the [Features](#features) and [Training the Model](#training-the-model) sections.
- **How to Use It**: To train with `RandomForestClassifier` instead of Logistic Regression, modify the training script like this:
  ```python
  from sklearn.ensemble import RandomForestClassifier

  # Replace LogisticRegression with RandomForestClassifier
  model = RandomForestClassifier(random_state=42)
  param_grid = {
      'n_estimators': [50, 100, 200],
      'max_depth': [None, 10, 20],
      'min_samples_split': [2, 5, 10]
  }
  grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
  Then retrain and save the new rainfall_model.pkl. The Streamlit app will work with either model as long as the .pkl file is updated.
  