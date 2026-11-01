
import joblib
import pandas as pd

from flask import Flask, request, jsonify

app = Flask(__name__)

# Load saved model
artifact = joblib.load("churn_model.pkl")

model = artifact["model"]
scaler = artifact["scaler"]
encoders = artifact["encoders"]
features = artifact["features"]


# GET Request
@app.route("/")
def home():
    return jsonify({
        "message": "Welcome to Customer Churn Prediction API!"
    })


# POST Request
@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    gender = data["gender"]
    SeniorCitizen = data["SeniorCitizen"]
    Partner = data["Partner"]
    Dependents = data["Dependents"]
    tenure = data["tenure"]
    PhoneService = data["PhoneService"]
    MultipleLines = data["MultipleLines"]
    InternetService = data["InternetService"]
    OnlineSecurity = data["OnlineSecurity"]
    OnlineBackup = data["OnlineBackup"]
    DeviceProtection = data["DeviceProtection"]
    TechSupport = data["TechSupport"]
    StreamingTV = data["StreamingTV"]
    StreamingMovies = data["StreamingMovies"]
    Contract = data["Contract"]
    PaperlessBilling = data["PaperlessBilling"]
    PaymentMethod = data["PaymentMethod"]
    MonthlyCharges = data["MonthlyCharges"]
    TotalCharges = data["TotalCharges"]

    # Prepare input data
    input_data = pd.DataFrame([[
        gender,
        SeniorCitizen,
        Partner,
        Dependents,
        tenure,
        PhoneService,
        MultipleLines,
        InternetService,
        OnlineSecurity,
        OnlineBackup,
        DeviceProtection,
        TechSupport,
        StreamingTV,
        StreamingMovies,
        Contract,
        PaperlessBilling,
        PaymentMethod,
        MonthlyCharges,
        TotalCharges
    ]], columns=features)

    # Apply Label Encoding
    for col, le in encoders.items():
        input_data[col] = le.transform(
            input_data[col].astype(str)
        )

    # Apply Normalization
    input_data = scaler.transform(input_data)

    # Make prediction
    pred = model.predict(input_data)

    # Return result
    result = "Yes" if pred[0] == 1 else "No"

    return jsonify({
        "prediction": result,
        "message": "Customer is likely to leave"
        if result == "Yes"
        else "Customer is likely to stay"
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
