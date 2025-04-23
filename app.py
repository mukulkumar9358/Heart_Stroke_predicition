import numpy as np
from flask import Flask, request, render_template
import pandas as pd
import joblib

app = Flask(__name__)
model = joblib.load('model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

def process(form_data):
    # Define mappings and expected columns
    mappings = {
        'Sex': {'M': 1, 'F': 0},
        'ChestPainType': {'TA': 0, 'ATA': 1, 'NAP': 2, 'ASY': 3},
        'RestingECG': {'Normal': 0, 'ST': 1, 'LVH': 2},
        'ExerciseAngina': {'N': 0, 'Y': 1},
        'ST_Slope': {'Up': 0, 'Flat': 1, 'Down': 2}
    }

    expected_columns = [
        'Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol',
        'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope'
    ]

    # Initialize data dictionary
    data = {}

    # Process form data
    for key in expected_columns:
        value = form_data.get(key, '')

        # Convert continuous fields to float, default to 0 if empty
        if key in ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak']:
            data[key] = float(value) if value else 0.0
        # Map categorical fields using mappings
        elif key in mappings:
            data[key] = mappings[key].get(value, 0)
        else:
            data[key] = value  # fallback (in case)

    # Convert to DataFrame in expected column order
    df = pd.DataFrame([data], columns=expected_columns)
    return df

@app.route('/predict', methods=['POST'])
def predict():
    df = process(request.form)
    prediction = model.predict_proba(df)[0]
    return render_template('predict.html', prediction=round(prediction[1]*100, 2))

if __name__ == "__main__":
    app.run(debug=True, port=8000)
