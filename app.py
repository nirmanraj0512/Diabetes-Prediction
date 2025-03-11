from flask import Flask, render_template, request
import numpy as np
import pickle
import os

app = Flask(__name__)

# Load trained model and scaler
model_path = os.path.join('Models', 'model.pkl')
scaler_path = os.path.join('Models', 'scaler.pkl')

with open(model_path, 'rb') as model_file:
    model = pickle.load(model_file)

with open(scaler_path, 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Feature means from training data (replace with your actual values)
FEATURE_MEANS = {
    'Glucose': 121.686763,
    'Insulin': 155.548223,
    'BMI': 32.457464
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data with validation
        glucose = float(request.form.get('glucose', 0))
        insulin = float(request.form.get('insulin', 0))
        bmi = float(request.form.get('bmi', 0))
        age = float(request.form.get('age', 0))

        # Handle missing values using training data means
        features = np.array([[
            glucose if glucose != 0 else FEATURE_MEANS['Glucose'],
            insulin if insulin != 0 else FEATURE_MEANS['Insulin'],
            bmi if bmi != 0 else FEATURE_MEANS['BMI'],
            age
        ]])

        # Scale features using the trained scaler
        scaled_features = scaler.transform(features)

        # Make prediction
        prediction = model.predict(scaled_features)
        result = 'Diabetic' if prediction[0] == 1 else 'Not Diabetic'

        return render_template('index.html',
                             prediction_text=f'Prediction: {result}',
                             show_result=True)

    except Exception as e:
        return render_template('index.html',
                             error_message=f'Error processing request: {str(e)}',
                             show_result=False)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)