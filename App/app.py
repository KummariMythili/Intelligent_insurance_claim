from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load model and scaler
model = joblib.load('model/fine_tune.pkl')
scaler = joblib.load('model/scaler.pkl')

# Encoding mappings
gender_map = {'Male': 1, 'Female': 0}
smoker_map = {'Yes': 1, 'No': 0}
claim_type_map = {'Health': 0, 'Accident': 1, 'Fire': 2, 'Theft': 3}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect inputs
        age = int(request.form['age'])
        gender_text = request.form['gender']
        bmi = float(request.form['bmi'])
        smoker_text = request.form['smoker']
        dependents = int(request.form['dependents'])
        claim_type_text = request.form['claim_type']
        claim_amount = float(request.form['claim_amount'])
        previous_claims = int(request.form['previous_claims'])
        hospital_stay = int(request.form['hospital_stay'])
        doctor_visits = int(request.form['doctor_visits'])

        # Encode inputs
        gender = gender_map[gender_text]
        smoker = smoker_map[smoker_text]
        claim_type = claim_type_map[claim_type_text]

        # Combine input
        input_data = np.array([[age, gender, bmi, smoker, dependents,
                                claim_type, claim_amount, previous_claims,
                                hospital_stay, doctor_visits]])

        # Scale input
        input_scaled = scaler.transform(input_data)

        # Predict
        prediction = model.predict(input_scaled)[0]
        result = "✅ Claim Approved" if prediction == 1 else "❌ Claim Rejected"

        # Collect user input to display
        user_input = {
            "Age": age,
            "Gender": gender_text,
            "BMI": bmi,
            "Smoker": smoker_text,
            "Number of Dependents": dependents,
            "Type of Claim": claim_type_text,
            "Claim Amount": claim_amount,
            "Number of Previous Claims": previous_claims,
            "Hospital Stay Duration": hospital_stay,
            "Doctor Visits": doctor_visits
        }

        return render_template('index.html', prediction=result, user_input=user_input)

    except Exception as e:
        return render_template('index.html', prediction=f"⚠️ Error: {e}")

if __name__ == '__main__':
    app.run(debug=True)
