# app.py

from flask import Flask, render_template, request, jsonify
import numpy as np
from sklearn.preprocessing import StandardScaler
from keras.models import load_model

app = Flask(__name__)

# Load the trained model
model = load_model('diabetes_cnn_model.h5')


# Define a function to preprocess input data
def preprocess_input(data):
    features = [float(x) if x else np.nan for x in data]
    if any(np.isnan(features)):
        return None
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform([features])
    return features_scaled


# Define a function to make predictions
def make_prediction(features_scaled):
    prediction = model.predict(features_scaled)
    if np.isnan(prediction).any():
        return None
    diabetes_probability = prediction[0][0]
    if diabetes_probability >= 0.5:
        result = "Diabetes"
    else:
        result = "No Diabetes"
    return {'prediction': result, 'probability': float(diabetes_probability)}


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Get the form data
    features = request.form.values()

    # Preprocess input data
    features_scaled = preprocess_input(features)
    if features_scaled is None:
        return jsonify({'error': 'Invalid input data'})

    # Make prediction
    prediction_result = make_prediction(features_scaled)
    if prediction_result is None:
        return jsonify({'error': 'Error occurred during prediction'})

    # Return prediction
    return jsonify(prediction_result)


if __name__ == '__main__':
    app.run(debug=True)
