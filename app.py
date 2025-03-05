from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
# Load model
import joblib


app = Flask(__name__)


# Load the correct model file
model = joblib.load('heart_disease_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    input_data = np.array(data['features']).reshape(1, -1)
    prediction = model.predict(input_data)
    return jsonify({'prediction': str(prediction[0][0])})

if __name__ == '__main__':
    app.run(debug=True)
