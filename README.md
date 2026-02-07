
# Machine Learning Assignment 2 – Classification Models & Deployment

## 1. Problem Statement
The objective of this assignment is to implement multiple machine learning classification models on a real-world dataset, evaluate their performance using standard evaluation metrics, and deploy an interactive Streamlit web application to demonstrate the models and their results.

This assignment demonstrates the complete end-to-end machine learning workflow including data preprocessing, model training, evaluation, and deployment.

---

## 2. Dataset Description
The dataset used for this assignment is **“Default of Credit Card Clients”** from the **UCI Machine Learning Repository**.

- Total Instances: 30,000  
- Total Features: 23 (after removing the ID column)  
- Target Variable: `default payment next month`  
  - 0 → No Default  
  - 1 → Default  

This is a binary classification problem with class imbalance, making it suitable for evaluating multiple classification algorithms.

---

## 3. Models Implemented & Evaluation Metrics
The following six classification models were implemented on the same dataset:

1. Logistic Regression  
2. Decision Tree Classifier  
3. K-Nearest Neighbors (KNN)  
4. Naive Bayes (Gaussian)  
5. Random Forest (Ensemble)  
6. XGBoost (Ensemble)  

Each model was evaluated using the following metrics:
- Accuracy  
- AUC Score  
- Precision  
- Recall  
- F1 Score  
- Matthews Correlation Coefficient (MCC)  

---

## 4. Model Performance Comparison

| ML Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|--------|---------|------|-----------|--------|----------|-----|
| Logistic Regression | 0.81 | 0.71 | 0.69 | 0.24 | 0.36 | 0.32 |
| Decision Tree | 0.72 | 0.61 | 0.37 | 0.41 | 0.39 | 0.21 |
| KNN | 0.79 | 0.70 | 0.55 | 0.36 | 0.43 | 0.32 |
| Naive Bayes | 0.75 | 0.72 | 0.45 | 0.55 | 0.50 | 0.34 |
| Random Forest | 0.81 | 0.75 | 0.63 | 0.36 | 0.46 | 0.37 |
| XGBoost | 0.81 | 0.76 | 0.63 | 0.36 | 0.46 | 0.38 |

---

## 5. Observations on Model Performance

| Model | Observation |
|------|------------|
| Logistic Regression | Strong baseline model but struggles with non-linear patterns |
| Decision Tree | Captures non-linearity but prone to overfitting |
| KNN | Balanced performance but computationally expensive |
| Naive Bayes | High recall but biased towards the majority class |
| Random Forest | Robust and stable due to ensemble averaging |
| XGBoost | Best overall performance with optimal bias-variance trade-off |

---

## 6. Streamlit Application
A Streamlit web application was developed to:
- Upload test datasets (CSV)
- Select trained machine learning models
- Display evaluation metrics
- Visualize confusion matrices and classification reports

Due to Streamlit free-tier resource constraints, only a subset of test data is used for demonstration, as per assignment instructions.

---

## 7. Repository Structure

project-root/
├── app.py  
├── requirements.txt  
├── README.md  
├── data/  
│   └── credit_card.xls  
├── src/  
│   ├── data_preprocessing.py  
│   ├── train_models.py  
│   └── evaluate_models.py  
└── models/  
    └── saved model files (*.pkl)

---

## 8. Deployment
The application is deployed on **Streamlit Community Cloud** using the GitHub repository.  
The deployed application provides an interactive frontend for model evaluation.

---

## 9. Conclusion
This project demonstrates a complete machine learning pipeline from data preprocessing to deployment.  
The comparative analysis shows that ensemble methods, particularly XGBoost, perform best on the given dataset.

