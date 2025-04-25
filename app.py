from flask import Flask, request, jsonify, render_template
import pandas as pd
import pickle
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

# Load model and encoders
model_path = 'model/crop_price_model.pkl'
encoders_path = 'model/label_encoders.pkl'

if not os.path.exists(model_path) or not os.path.exists(encoders_path):
    raise FileNotFoundError("Model or encoder files not found in 'model' directory.")

with open(model_path, 'rb') as f:
    model = pickle.load(f)
with open(encoders_path, 'rb') as f:
    label_encoders = pickle.load(f)

# Load dataset for visualization data
data = pd.read_csv('crop_data.csv')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data
        input_data = request.json
        state = input_data['state']
        crop = input_data['crop']
        cost_cultivation = float(input_data['costCultivation'])
        production = float(input_data['production'])
        yield_val = float(input_data['yield'])
        temperature = float(input_data['temperature'])
        rainfall = float(input_data['rainfall'])

        # Validate inputs
        if state not in label_encoders['State'].classes_:
            return jsonify({'error': f"Invalid state: {state}"}), 400
        if crop not in label_encoders['Crop'].classes_:
            return jsonify({'error': f"Invalid crop: {crop}"}), 400

        # Encode categorical variables
        state_encoded = label_encoders['State'].transform([state])[0]
        crop_encoded = label_encoders['Crop'].transform([crop])[0]

        # Prepare input for model
        features = [[state_encoded, crop_encoded, cost_cultivation, production, yield_val, temperature, rainfall]]
        prediction = model.predict(features)[0]

        # Get feature importance
        feature_names = ['State', 'Crop', 'CostCultivation', 'Production', 'Yield', 'Temperature', 'RainFall Annual']
        feature_importance = model.feature_importances_.tolist()

        # Sample actual vs predicted data (using test set)
        X = data[feature_names].copy()
        for col in ['State', 'Crop']:
            X[col] = label_encoders[col].transform(X[col])
        y_actual = data['Price'].values
        y_pred = model.predict(X)
        sample_data = [{'actual': float(actual), 'predicted': float(pred)} for actual, pred in zip(y_actual[:10], y_pred[:10])]

        return jsonify({
            'price': round(prediction, 2),
            'featureImportance': {name: imp for name, imp in zip(feature_names, feature_importance)},
            'actualVsPredicted': sample_data
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)