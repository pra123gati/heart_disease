from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np

app = Flask(__name__)

# Load model
model = tf.keras.models.load_model('heart_disease_model.h5')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    input_data = np.array(data['features']).reshape(1, -1)
    prediction = model.predict(input_data)
    return jsonify({'prediction': str(prediction[0][0])})

if __name__ == '__main__':
    app.run(debug=True)
