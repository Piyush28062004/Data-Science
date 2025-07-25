from flask import Flask, request, jsonify, render_template
import numpy as np
from joblib import load

# Create the Flask app
app = Flask(__name__)

# Load the trained model
model = load('insurance.joblib')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data from the HTML form
    int_features = [float(x) for x in request.form.values()]
    
    # Convert to NumPy array
    final_features = np.asarray(int_features).reshape(1, -1)
    
    # Make prediction
    prediction = model.predict(final_features)
    
    # Return prediction result
    output = round(prediction[0], 2)
    
    return render_template('index.html', prediction_text=f'Predicted Insurance Charges: ${output}')

if __name__ == "__main__":
    app.run(debug=True)
