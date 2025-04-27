from flask import Flask, request, jsonify, render_template
import pandas as pd
import pickle
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Define file paths relative to app.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'crop_price_model.pkl')
ENCODERS_PATH = os.path.join(BASE_DIR, 'model', 'label_encoders.pkl')
DATA_PATH = os.path.join(BASE_DIR, 'crop_data.csv')
METRICS_PATH = os.path.join(BASE_DIR, 'model', 'model_metrics.pkl')

# Load model, encoders, data, and metrics
try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(ENCODERS_PATH, 'rb') as f:
        label_encoders = pickle.load(f)
    df = pd.read_csv(DATA_PATH)
    with open(METRICS_PATH, 'rb') as f:
        metrics = pickle.load(f)
    r2_score = metrics['r2_score']
except FileNotFoundError as e:
    print(f"Error: File not found - {e}")
    raise
except Exception as e:
    print(f"Error loading files: {e}")
    raise

@app.route('/')
def home():
    states = sorted(df['State'].unique())
    crops = sorted(df['Crop'].unique())
    return render_template('index.html', states=states, crops=crops, r2_score=r2_score)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form
        state = data['state']
        crop = data['crop']
        cost_cultivation = float(data['cost_cultivation'])
        cost_cultivation2 = float(data['cost_cultivation2'])
        production = float(data['production'])
        yield_val = float(data['yield'])
        temperature = float(data['temperature'])
        rainfall_annual = float(data['rainfall_annual'])

        # Encode categorical variables
        state_encoded = label_encoders['State'].transform([state])[0]
        crop_encoded = label_encoders['Crop'].transform([crop])[0]

        # Prepare input for model
        input_data = [[state_encoded, crop_encoded, cost_cultivation, cost_cultivation2, production, yield_val, temperature, rainfall_annual]]
        input_df = pd.DataFrame(input_data, columns=['State', 'Crop', 'CostCultivation', 'CostCultivation2', 'Production', 'Yield', 'Temperature', 'RainFall Annual'])

        # Make prediction
        prediction = model.predict(input_df)[0]

        return jsonify({
            'prediction': round(prediction, 2),
            'feature_importance_plot': '/static/feature_importance.png',
            'actual_vs_predicted_plot': '/static/actual_vs_predicted.png'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)