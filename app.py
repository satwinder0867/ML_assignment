import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

# --------------------------------------------------
# Page Config
# --------------------------------------------------
st.set_page_config(
    page_title="Credit Card Default Prediction - ML Models",
    layout="wide"
)

st.title("💳 Credit Card Default Prediction – ML Models")

st.markdown(
    """
    This application allows users to evaluate multiple trained machine learning
    models on **test data only**, as per assignment guidelines.
    """
)

st.markdown("---")

# --------------------------------------------------
# Model Selection
# --------------------------------------------------
st.subheader("🤖 Model Selection")

model_option = st.selectbox(
    "Select a Machine Learning Model",
    [
        "Logistic Regression",
        "Decision Tree",
        "KNN",
        "Naive Bayes",
        "Random Forest",
        "XGBoost"
    ]
)

model_paths = {
    "Logistic Regression": "models/logistic_regression.pkl",
    "Decision Tree": "models/decision_tree.pkl",
    "KNN": "models/knn.pkl",
    "Naive Bayes": "models/naive_bayes.pkl",
    "Random Forest": "models/random_forest.pkl",
    "XGBoost": "models/xgboost.pkl"
}

model = joblib.load(model_paths[model_option])

st.markdown("---")

# --------------------------------------------------
# Dataset Source Selection
# --------------------------------------------------
st.subheader("📁 Dataset Source")

data_source = st.radio(
    "Choose how you want to provide the test dataset:",
    (
        "Use default test dataset (from repository)",
        "Upload custom test dataset (CSV)"
    )
)

data = None

if data_source == "Use default test dataset (from repository)":
    st.info("Using default test dataset available in the repository.")
    data = pd.read_csv("data/credit_card_test.csv")

elif data_source == "Upload custom test dataset (CSV)":
    uploaded_file = st.file_uploader(
        "Upload Test Dataset (CSV)",
        type=["csv"]
    )
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)

# --------------------------------------------------
# Prediction & Evaluation
# --------------------------------------------------
if data is not None:

    # Dataset preview
    with st.expander("🔍 Preview Dataset"):
        st.dataframe(data.head())

    # Feature / target split
    X = data.drop("default payment next month", axis=1)
    y = data["default payment next month"]

    # Predictions
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    # --------------------------------------------------
    # Metrics Section
    # --------------------------------------------------
    st.subheader("📊 Evaluation Metrics")

    col1, col2, col3 = st.columns(3)
    col4, col5, col6 = st.columns(3)

    col1.metric("Accuracy", f"{accuracy_score(y, y_pred):.2f}")
    col2.metric("AUC", f"{roc_auc_score(y, y_prob):.2f}")
    col3.metric("Precision", f"{precision_score(y, y_pred):.2f}")

    col4.metric("Recall", f"{recall_score(y, y_pred):.2f}")
    col5.metric("F1 Score", f"{f1_score(y, y_pred):.2f}")
    col6.metric("MCC", f"{matthews_corrcoef(y, y_pred):.2f}")

    # --------------------------------------------------
    # Model Insight
    # --------------------------------------------------
    st.markdown("### 🧠 Model Insight")

    if model_option == "Naive Bayes":
        st.info(
            "Naive Bayes often predicts the majority class due to class imbalance, "
            "leading to reasonable accuracy but low recall for defaulters."
        )
    elif model_option == "Logistic Regression":
        st.info(
            "Logistic Regression provides a strong baseline but struggles with "
            "complex non-linear patterns."
        )
    elif model_option == "Decision Tree":
        st.info(
            "Decision Trees capture non-linearity but are prone to overfitting."
        )
    elif model_option == "KNN":
        st.info(
            "KNN offers balanced performance but is sensitive to scaling and "
            "computationally expensive."
        )
    elif model_option == "Random Forest":
        st.success(
            "Random Forest benefits from ensemble learning and provides "
            "stable, robust performance."
        )
    elif model_option == "XGBoost":
        st.success(
            "XGBoost delivers the best overall trade-off between metrics due "
            "to boosting and regularization."
        )

    st.markdown("---")

    # --------------------------------------------------
    # Confusion Matrix
    # --------------------------------------------------
    st.subheader("🧩 Confusion Matrix (Actual vs Predicted)")
    st.caption("Rows represent actual labels, columns represent predicted labels.")

    cm = confusion_matrix(y, y_pred)

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    st.pyplot(fig)

    # --------------------------------------------------
    # Classification Report
    # --------------------------------------------------
    st.subheader("📄 Classification Report")
    st.text(classification_report(y, y_pred))
