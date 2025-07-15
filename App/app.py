from flask import Flask, request, render_template
import numpy as np
import joblib
import os

app = Flask(__name__)

# 📌 Load the trained model and scaler
model = joblib.load('model/model.pkl')
scaler = joblib.load('model/scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 📥 Collect input data from form
        features = [
            float(request.form['Age']),
            float(request.form['Gender']),
            float(request.form['BMI']),
            float(request.form['Smoker']),
            float(request.form['Number_of_Dependents']),
            float(request.form['Type_of_Claim']),
            float(request.form['Claim_Amount']),
            float(request.form['Number_of_Previous_Claims']),
            float(request.form['Hospital_Stay_Duration']),
            float(request.form['Doctor_Visits'])
        ]

        # 🧠 Reshape and scale the data
        input_data = np.array(features).reshape(1, -1)
        input_scaled = scaler.transform(input_data)

        # 🔍 Predict
        prediction = model.predict(input_scaled)[0]

        # 📌 Interpret result
        if prediction == 1:
            result = "✅ Claim Approved"
        else:
            result = "❌ Claim Rejected"

        return render_template('index.html', prediction_text=result)

    except Exception as e:
        return render_template('index.html', prediction_text=f"⚠️ Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
