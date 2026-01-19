# Employee-Attrition-Analysis-and-Prediction
📌 Project: Employee Attrition Prediction System
🔍 Problem Statement
Employee attrition is a critical challenge for organizations as it leads to increased hiring costs, loss of expertise, and reduced productivity. The goal of this project was to build a machine learning system that predicts the likelihood of an employee leaving the organization, enabling HR teams to take proactive retention actions.

🎯 Objective
Predict employee attrition (Yes/No) using historical HR data, Handle class imbalance effectively, Compare multiple machine learning models and select the best-performing one, Deploy the final model as an interactive Streamlit web application

🛠️ Data Preprocessing & Feature Engineering
1.Removed constant and non-informative features (EmployeeCount, EmployeeNumber, StandardHours, Over18)
2.Encoded categorical variables using One-Hot Encoding
3.Converted binary variables such as OverTime and Attrition into numerical format
4.Identified and removed weakly correlated features
5.Applied StandardScaler for linear models
6.Addressed class imbalance using SMOTE (Synthetic Minority Over-sampling Technique)

🤖 Machine Learning Models Used
Logistic Regression
Random Forest Classifier
XGBoost Classifier
Gradient Boosting Classifier (Final Model)
Models were evaluated both before and after SMOTE to assess improvement in minority class prediction.

📈 Model Evaluation Metrics
Due to class imbalance, the following metrics were prioritized over accuracy:
ROC-AUC Score, 
F1-Score (Attrition class)
Precision, Recall, and Confusion Matrix
The Gradient Boosting model achieved the best balance between recall and ROC-AUC and was selected for deployment.

🚀 Deployment
Built an interactive Streamlit dashboard for real-time attrition prediction where Users can input employee details and receive:
Attrition probability
Risk classification (Low / Medium / High)
Ensured production safety by saving training feature names and reindexing input data to avoid feature mismatch errors
Applied a custom probability threshold (0.4) to improve recall for high-risk employees

🧰 Tools & Technologies
Programming: Python
Libraries: Pandas, NumPy, Scikit-learn, XGBoost, Imbalanced-learn, Joblib
Visualization: Matplotlib, Seaborn
Deployment: Streamlit
Improved minority class detection through SMOTE and threshold tuning

Delivered an HR-friendly dashboard that converts machine learning predictions into actionable insights
