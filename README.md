Project/
├── App/
│   ├── app.py
│   ├── model/
│   │   ├── model.pkl
│   │   ├── fine_tune.pkl
│   │   └── scaler.pkl
│   ├── templates/
│   │   └── index.html
│   ├── static/
│   │   ├── main_css.css
│   │   └── main.js
├── Data/
│   ├── intelligent_insurance_claim_data.csv
│   ├── preprocessed_data.csv
├── Training/
│   ├── preprocess_data.ipynb
│   ├── training_notebook.ipynb
├── Evaluation/
│   ├── best_model_saving.ipynb
│   ├── evaluation_and_tuning.ipynb
├── README.md
├── requirements.txt
└── setup.exe

 Abstract
The project aims to build an intelligent system that predicts whether an insurance claim will be approved or rejected based on applicant data and claim-related features using supervised machine learning algorithms.

🎯 Objective
Predict claim approval using historical data

Assist insurance companies in fast-tracking approvals

Reduce human bias and manual error

Improve decision-making using data

📁 Dataset
The dataset includes the following features:

Feature	Description
Age	Age of the claimant
Gender	Male or Female
BMI	Body Mass Index
Smoker	Yes or No
Number_of_Dependents	Number of dependents in the family
Type_of_Claim	Health, Accident, Fire, Theft
Claim_Amount	Total amount of claim
Number_of_Previous_Claims	Claim history
Hospital_Stay_Duration	Stay duration in days
Doctor_Visits	Number of medical visits
Claim_Status	Target variable (Approved or Rejected)

🧠 Machine Learning Models Used
Model	Status
Logistic Regression	✅ Evaluated
SVM (Support Vector Machine)	🏆 Best Model
Random Forest	✅ Evaluated
AdaBoost	✅ Evaluated
Gradient Boosting	✅ Evaluated

The SVM model gave the highest accuracy and was further fine-tuned using GridSearchCV.

🛠 Methodology
Data Preprocessing

Clean and encode categorical data

Handle missing values

Balance the dataset

Scale numeric features

Model Training

Evaluate 5 classifiers

Select best model based on accuracy

Model Tuning

Fine-tune best model using GridSearchCV

Save both original and tuned models

Flask Integration

Build a web UI with input form

Load scaler and model for prediction

Display real-time prediction results

🚀 How to Run the App
✅ Step 1: Install Requirements
bash
Copy code
pip install -r requirements.txt
✅ Step 2: Launch Flask App
bash
Copy code
cd App
python app.py
Visit: http://127.0.0.1:5000

🧪 Sample Input
Field	Value
Age	45
Gender	Male
BMI	28.7
Smoker	No
Number of Dependents	2
Type of Claim	Health
Claim Amount	8500
Previous Claims	1
Hospital Stay	5
Doctor Visits	3

✅ Output
✅ Claim Approved
or
❌ Claim Rejected

📦 Requirements
See requirements.txt.
Key libraries:

Flask

scikit-learn

pandas

numpy

joblib

