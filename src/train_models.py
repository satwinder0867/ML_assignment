import os
import joblib
import pandas as pd

# Import preprocessing function
from data_preprocessing import load_and_preprocess_data

# ML models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Evaluation metrics
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)

# Load data using preprocessing module
X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data(
    "data/credit_card.xls"
)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ),
    "XGBoost": XGBClassifier(
        eval_metric="logloss",
        random_state=42
    )
}

def evaluate_model(model, X_test, y_test):
    """
    Evaluates a classification model using required metrics.
    """

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_prob),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
        "MCC": matthews_corrcoef(y_test, y_pred)
    }

# Create directory to save models
os.makedirs("models", exist_ok=True)

results = []

for model_name, model in models.items():
    print(f"Training {model_name}...")

    # Train model
    model.fit(X_train, y_train)

    # Evaluate model
    metrics = evaluate_model(model, X_test, y_test)

    # Save model
    model_path = f"models/{model_name.replace(' ', '_').lower()}.pkl"
    joblib.dump(model, model_path)

    # Store results
    metrics["Model"] = model_name
    results.append(metrics)

results_df = pd.DataFrame(results)

# Reorder columns (for README / PDF)
results_df = results_df[
    ["Model", "Accuracy", "AUC", "Precision", "Recall", "F1 Score", "MCC"]
]

print("\nModel Performance Comparison:\n")
print(results_df)
