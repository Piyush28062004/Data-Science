from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
from joblib import load

# Load the trained model
model = load('breast_cancer.joblib')

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from form
    features = [int(x) for x in request.form.values()]
    final_features = [np.array(features)]

    # Predict using the model
    prediction = model.predict(final_features)
    
    # Convert prediction to a readable form (0 = Alive, 1 = Deceased)
    output = 'Alive' if prediction[0] == 0 else 'Deceased'
    
    return render_template('index.html', prediction_text=f'The patient is {output}')

if __name__ == "__main__":
    app.run(debug=True)
