import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Check current working directory
logger.info(f"Current working directory: {os.getcwd()}")

# Load the dataset
try:
    logger.info("Loading dataset...")
    data = pd.read_csv('crop_data.csv')
    logger.info("Dataset loaded successfully.")
except FileNotFoundError:
    logger.error("'crop_data.csv' not found in the current directory.")
    logger.error("Please ensure the file is in the same directory as this script or provide the correct path.")
    exit(1)
except Exception as e:
    logger.error(f"Failed to load dataset: {str(e)}")
    exit(1)

# Select features and target
features = ['State', 'Crop', 'CostCultivation', 'Production', 'Yield', 'Temperature', 'RainFall Annual']
try:
    logger.info("Selecting features and target...")
    X = data[features].copy()  # Create an explicit copy to avoid SettingWithCopyWarning
    y = data['Price']
    logger.info("Features and target selected.")
except KeyError as e:
    logger.error(f"Missing column in dataset: {str(e)}")
    exit(1)

# Encode categorical variables
label_encoders = {}
try:
    logger.info("Encoding categorical variables...")
    for col in ['State', 'Crop']:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le
    logger.info("Categorical variables encoded.")
except Exception as e:
    logger.error(f"Failed to encode categorical variables: {str(e)}")
    exit(1)

# Create model directory if it doesn't exist
model_dir = 'model'
try:
    logger.info(f"Ensuring {model_dir} directory exists...")
    os.makedirs(model_dir, exist_ok=True)
    logger.info(f"{model_dir} directory ready.")
except Exception as e:
    logger.error(f"Failed to create model directory: {str(e)}")
    exit(1)

# Save label encoders
try:
    logger.info("Saving label encoders...")
    with open(os.path.join(model_dir, 'label_encoders.pkl'), 'wb') as f:
        pickle.dump(label_encoders, f)
    logger.info("Label encoders saved to model/label_encoders.pkl.")
except Exception as e:
    logger.error(f"Failed to save label encoders: {str(e)}")
    exit(1)

# Split data into train and test sets
try:
    logger.info("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    logger.info("Data split completed.")
except Exception as e:
    logger.error(f"Failed to split data: {str(e)}")
    exit(1)

# Train Random Forest model
try:
    logger.info("Training Random Forest model...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    logger.info("Model training completed.")
except Exception as e:
    logger.error(f"Failed to train model: {str(e)}")
    exit(1)

# Predict on test set
try:
    logger.info("Making predictions on test set...")
    y_pred = model.predict(X_test)
    logger.info("Predictions completed.")
except Exception as e:
    logger.error(f"Failed to make predictions: {str(e)}")
    exit(1)

# Evaluation metrics
try:
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print("Model Trained Successfully!")
    print(f"Mean Squared Error: {mse}")
    print(f"R² Score: {r2}")
    logger.info(f"Model evaluation: MSE={mse}, R²={r2}")
except Exception as e:
    logger.error(f"Failed to evaluate model: {str(e)}")
    exit(1)

# Feature importance plot
try:
    logger.info("Generating feature importance plot...")
    plt.figure(figsize=(10, 6))
    feature_importance = model.feature_importances_
    plt.bar(features, feature_importance, color='dodgerblue')
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.title('Feature Importance for Crop Price Prediction')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    logger.info("Feature importance plot saved to feature_importance.png.")
except Exception as e:
    logger.error(f"Failed to generate feature importance plot: {str(e)}")
    exit(1)

# Actual vs Predicted plot
try:
    logger.info("Generating actual vs predicted plot...")
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.6, color='dodgerblue')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    plt.title('Actual vs Predicted Crop Prices')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('actual_vs_predicted.png')
    logger.info("Actual vs predicted plot saved to actual_vs_predicted.png.")
except Exception as e:
    logger.error(f"Failed to generate actual vs predicted plot: {str(e)}")
    exit(1)

# Save the model
try:
    logger.info("Saving model...")
    model_path = os.path.join(model_dir, 'crop_price_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    logger.info(f"Model saved to {model_path}.")
except Exception as e:
    logger.error(f"Failed to save model: {str(e)}")
    exit(1)