import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'crop_data.csv')
MODEL_DIR = os.path.join(BASE_DIR, 'model')
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'crop_price_model.pkl')
ENCODERS_PATH = os.path.join(BASE_DIR, 'model', 'label_encoders.pkl')
METRICS_PATH = os.path.join(BASE_DIR, 'model', 'model_metrics.pkl')
FIG_DIR = os.path.join(BASE_DIR, 'static')

# Create directories if they don't exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

def train_model():
    try:
        logging.info("Loading dataset...")
        df = pd.read_csv(DATA_PATH)

        # Encode categorical variables
        label_encoders = {}
        for column in ['State', 'Crop']:
            le = LabelEncoder()
            df[column] = le.fit_transform(df[column])
            label_encoders[column] = le

        # Features and target
        X = df[['State', 'Crop', 'CostCultivation', 'CostCultivation2', 'Production', 'Yield', 'Temperature', 'RainFall Annual']]
        y = df['Price']

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train model
        logging.info("Training model...")
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Evaluate model
        score = model.score(X_test, y_test)
        logging.info(f"Model R^2 score: {score:.3f}")

        # Save model, encoders, and metrics
        with open(MODEL_PATH, 'wb') as f:
            pickle.dump(model, f)
        with open(ENCODERS_PATH, 'wb') as f:
            pickle.dump(label_encoders, f)
        with open(METRICS_PATH, 'wb') as f:
            pickle.dump({'r2_score': score}, f)
        logging.info(f"Model saved to {MODEL_PATH}")
        logging.info(f"Encoders saved to {ENCODERS_PATH}")
        logging.info(f"Metrics saved to {METRICS_PATH}")

        # Feature importance plot
        plt.figure(figsize=(10, 6))
        plt.bar(X.columns, model.feature_importances_)
        plt.xticks(rotation=45)
        plt.title('Feature Importance')
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, 'feature_importance.png'))
        plt.close()
        logging.info("Feature importance plot saved")

        # Actual vs Predicted plot
        y_pred = model.predict(X_test)
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        plt.xlabel('Actual Price')
        plt.ylabel('Predicted Price')
        plt.title('Actual vs Predicted Prices')
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, 'actual_vs_predicted.png'))
        plt.close()
        logging.info("Actual vs Predicted plot saved")

    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
        raise
    except Exception as e:
        logging.error(f"Error during training: {e}")
        raise

if __name__ == '__main__':
    train_model()