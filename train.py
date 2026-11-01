import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# Load dataset
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Remove unnecessary column
df = df.drop("customerID", axis=1)

# Convert TotalCharges into numeric
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"], errors="coerce"
)

# Handle missing values
df["TotalCharges"] = df["TotalCharges"].fillna(
    df["TotalCharges"].median()
)

# Separate features and target
X = df.drop("Churn", axis=1)
y = df["Churn"].map({"No": 0, "Yes": 1})

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Label Encoding
label_encoders = {}

for col in X_train.select_dtypes(include="object").columns:
    le = LabelEncoder()

    X_train[col] = le.fit_transform(X_train[col])

    known = set(le.classes_)

    X_test[col] = X_test[col].apply(
        lambda x: le.transform([x])[0] if x in known else -1
    )

    label_encoders[col] = le

# Normalization
scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define models and parameters
models = {
    "Decision Tree": (
        DecisionTreeClassifier(random_state=42),
        {
            "max_depth": [3, 5, 10],
            "min_samples_split": [2, 10]
        }
    ),

    "Random Forest": (
        RandomForestClassifier(random_state=42),
        {
            "n_estimators": [50, 100],
            "max_depth": [5, 10]
        }
    ),

    "SVM": (
        SVC(random_state=42),
        {
            "C": [0.1, 1, 10],
            "kernel": ["linear", "rbf"]
        }
    )
}

results = []
best_model = None
best_f1 = -1
best_name = ""

# Model training and hyperparameter tuning
for name, (model, params) in models.items():

    print("\nTraining:", name)

    grid = GridSearchCV(
        model,
        params,
        cv=5,
        scoring="f1",
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    model = grid.best_estimator_

    # Prediction
    pred = model.predict(X_test)

    # Evaluation
    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred, zero_division=0)
    recall = recall_score(y_test, pred, zero_division=0)
    f1 = f1_score(y_test, pred, zero_division=0)

    print("Best parameters:", grid.best_params_)
    print("Accuracy:", round(accuracy, 3))
    print("Precision:", round(precision, 3))
    print("Recall:", round(recall, 3))
    print("F1-score:", round(f1, 3))

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1
    })

    # Select best model
    if f1 > best_f1:
        best_f1 = f1
        best_model = model
        best_name = name

# Compare models
print("\nMODEL COMPARISON")
print(pd.DataFrame(results).to_string(index=False))

print("\nBest Model:", best_name)

# Save model and preprocessing objects
joblib.dump({
    "model": best_model,
    "scaler": scaler,
    "encoders": label_encoders,
    "features": X.columns.tolist(),
    "model_name": best_name
}, "churn_model.pkl")

print("\nModel saved successfully!")
