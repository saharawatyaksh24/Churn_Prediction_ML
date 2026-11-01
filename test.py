
import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# Load dataset
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Load saved model
artifact = joblib.load("churn_model.pkl")

model = artifact["model"]
scaler = artifact["scaler"]
encoders = artifact["encoders"]

# Remove unnecessary column
df = df.drop("customerID", axis=1)

# Convert TotalCharges into numeric
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"], errors="coerce"
)

df["TotalCharges"] = df["TotalCharges"].fillna(
    df["TotalCharges"].median()
)

# Separate features and target
X = df.drop("Churn", axis=1)
y = df["Churn"].map({"No": 0, "Yes": 1})

# Apply saved Label Encoders
for col, le in encoders.items():

    known = set(le.classes_)

    X[col] = X[col].astype(str).apply(
        lambda x: le.transform([x])[0] if x in known else -1
    )

# Apply saved normalization
X = scaler.transform(X)

# Make predictions
pred = model.predict(X)

# Calculate evaluation metrics
accuracy = accuracy_score(y, pred)
precision = precision_score(y, pred, zero_division=0)
recall = recall_score(y, pred, zero_division=0)
f1 = f1_score(y, pred, zero_division=0)

print("\nCUSTOMER CHURN MODEL TESTING")

print("Accuracy:", round(accuracy, 3))
print("Precision:", round(precision, 3))
print("Recall:", round(recall, 3))
print("F1-score:", round(f1, 3))

# Assertions
assert len(pred) == len(X)
assert all(p in [0, 1] for p in pred)
assert 0 <= accuracy <= 1

print("\nAll test cases passed!")
