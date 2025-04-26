Crop Price Prediction
This project predicts crop prices using a Random Forest model and provides a web interface for user inputs and visualizations. The app is deployed on Render.
Setup

Install Dependencies (local testing):
pip install -r requirements.txt


Prepare Dataset and Model:

Ensure crop_data.csv, model/crop_price_model.pkl, and model/label_encoders.pkl are in the project directory.
Run train_crop_model.py to generate model files:python train_crop_model.py




Local Testing:

Start the Flask server:python app.py


Open http://localhost:5000 in a browser.



Deployment on Render

Push to GitHub:git init
git add .
git commit -m "Deploy to Render"
git remote add origin https://github.com/Amalin-Varsha/crop_price_prediction.git
git push -u origin main


Create a Web Service on Render:
Environment: Python
Build Command: pip install -r requirements.txt
Start Command: gunicorn app:app
Instance Type: Free


Access at https://crop-price-prediction-app.onrender.com.

Files

app.py: Flask backend.
templates/index.html: Web interface.
train_crop_model.py: Trains the model.
crop_data.csv: Dataset.
model/: Model and encoders.
Procfile, requirements.txt: Render configuration.
feature_importance.png, actual_vs_predicted.png: Visualizations.

Future Improvements

Add input range validation.
Enhance visualizations.
Optimize model with hyperparameter tuning.

