# Hire-o-Dynamics : Performance Prediction, Classification and Attrition Modelling

## Abstract

This project focuses on three critical HR analytics tasks: predicting employee performance, classifying performance levels, and identifying high-risk attrition cases. We employ various supervised machine learning models—including Logistic Regression, Random Forest, Decision Trees, SVM using a well-engineered feature set from employee performance and engagement data. The models are evaluated on their accuracy, F1-score, precision and recall, while confusion matrices provide insight into prediction quality.## 

## Dataset

Data Source: [https://www.kaggle.com/datasets/sanjanchaudhari/employees-performance-for-hr-analytics/data)]

Records: ~17,000 rows (employees)

Features:

Demographics: department, region, education, gender

Performance Metrics: no_of_trainings, avg_training_score, previous_year_rating, KPIs_met_more_than_80, awards_won, length_of_service

Derived Metrics: training_efficiency, experience_rating_ratio, awards_per_year, performance_score, attrition

## Prediction Objectives

Performance Prediction (Regression): Estimate a numerical score to represent overall employee performance.

Performance Classification: Predict whether an employee's performance is satisfactory or not.

Attrition Classification: Predict the likelihood of an employee leaving based on performance-related criteria.

## Practical Use Case

These models support HR decisions like personalized development, promotions, and retention strategies.

## Process Overview

We began with regression models for numeric performance prediction and moved to classification models for performance categories and attrition. Mis-steps included data leakage and overfitting with certain models. Final approaches include preprocessing pipelines, SMOTE balancing and grid search for hyperparameter tuning.

## EDA Summary

X: Employee attributes, ratings, and derived metrics

Y: performance_score (regression), performance_class (classification), attrition (binary classification)

## Distributions

education, gender and department are skewed

KPIs and awards are heavily imbalanced

## Correlation

previous_year_rating and performance_score are positively correlated

avg_training_score correlates with performance as expected

![image](https://github.com/user-attachments/assets/7ba00ebd-7d11-45a4-a402-193520d103fb)

## Feature Importance (Random Forest)

KPIs_met_more_than_80, avg_training_score, previous_year_rating are top predictors

## Feature Engineering

Created derived features:

training_efficiency = avg training score / (no of trainings + 1)

experience_rating_ratio = previous_year_rating / (length_of_service + 1)

awards_per_year = awards / service length

### Encoding:

Used OneHotEncoding for nominal features

Used Frequency Encoding for region

StandardScaler for numerical features

## Model Fitting

Train-Test Split: 80:20 using train_test_split with stratification for classification

Data Leakage Prevention: Feature engineering and leakage-prone variables (performance_score in classification) were excluded where needed

Models Used

Linear Regression (for performance prediction)

Logistic Regression

Decision Tree Classifier

Random Forest Classifier

Support Vector Machine (SVM)

Neural Network (Keras)

Hyperparameter Tuning

Used GridSearchCV for Logistic, Decision Tree, and Random Forest

Used dropout, batch norm, and Adam optimizer in Neural Network tuning

📏 Metrics

Regression: R², MSE, MAE

Classification: Accuracy, Precision, Recall, F1-Score

Used SMOTE for class imbalance

Confusion Matrix Highlights:

All models evaluated using heatmap confusion matrices

Model Accuracy (Classification):

Model

Accuracy

Logistic Regression

76.2%

Decision Tree

75.4%

Random Forest

78.1%

SVM

77.0%

Neural Network

75.2%

⚖ Overfitting & Underfitting

Decision Trees showed mild overfitting at high depth

Dropout in Neural Networks helped reduce overfitting

Ensemble models like Random Forests performed well overall

🚀 Production & Use

Recommend integrating prediction API into HR dashboards

Alert HR if an employee has high attrition probability and low predicted performance

Risk: Needs retraining on recent data periodically to remain effective

📈 Future Work

Add more features (e.g., manager ratings, attendance data)

Use explainable AI (SHAP/LIME) for transparency

Experiment with ensemble stacking for further gains
