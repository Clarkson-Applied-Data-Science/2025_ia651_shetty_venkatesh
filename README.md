# Hire-o-Dynamics : Performance Prediction, Classification and Attrition Modelling

## Abstract

This project focuses on three critical HR analytics tasks: predicting employee performance, classifying performance levels and identifying high-risk attrition cases. We employ various supervised machine learning models—including Logistic Regression, Random Forest, Decision Trees, SVM using a well-engineered feature set from employee performance and engagement data. The models are evaluated on their accuracy, F1-score, precision and recall, while confusion matrices provide insight into prediction quality.

## Dataset

Data Source: [https://www.kaggle.com/datasets/sanjanchaudhari/employees-performance-for-hr-analytics/data)]

Records: ~17,000 rows (employees)

Features:

 **Demographics:** department, region, education, gender

 **Performance Metrics:** no_of_trainings, avg_training_score, previous_year_rating, KPIs_met_more_than_80, awards_won, length_of_service

 **Derived Metrics:** training_efficiency, experience_rating_ratio, awards_per_year, performance_score, attrition

## Prediction Objectives

**Performance Prediction (Regression):** Estimate a numerical score to represent overall employee performance.

**Performance Classification:** Predict whether an employee's performance is satisfactory or not.

**Attrition Classification:** Predict the likelihood of an employee leaving based on performance-related criteria.

## Practical Use Case

These models support HR decisions like personalized development, promotions, and retention strategies.

## Process Overview

We began with regression models for numeric performance prediction and moved to classification models for performance categories and attrition. Final approaches include preprocessing pipelines, SMOTE balancing and grid search for hyperparameter tuning.

## EDA Summary

X: Employee attributes, ratings, and derived metrics

Y: performance_score (regression), performance_class (classification), attrition (binary classification)

## Distributions

![image](https://github.com/user-attachments/assets/b34a6699-41c1-49d4-a2a2-cdac4ac5b1a9)
![image](https://github.com/user-attachments/assets/c7f50f4c-dcef-4740-a65c-4b269ea8dd65)
![image](https://github.com/user-attachments/assets/fd3becec-862d-4d52-a4be-9352c2cbd181)
![image](https://github.com/user-attachments/assets/575fcd4e-651b-4c5a-b154-129cd4c27182)
![image](https://github.com/user-attachments/assets/84e32600-e6af-4dbf-a9b8-ac1de47c1929)
![image](https://github.com/user-attachments/assets/beaaf0d8-af66-4731-97c4-1eb8d1c97a6a)

## Correlation

Previous_year_rating and performance_score are positively correlated. avg_training_score correlates with performance as expected.

![image](https://github.com/user-attachments/assets/7ba00ebd-7d11-45a4-a402-193520d103fb)

## Feature Importance (Random Forest)

KPIs_met_more_than_80, avg_training_score, previous_year_rating are top predictors

## Feature Engineering

Created derived features:

attrition = (KPIs_met_more_than_80 == 0) & (avg_training_score < 60) & (length_of_service > 5) & (previous_year_rating <= 2) & (awards_won == 0)

performance_score = (avg_training_score / 100 ) + (KPIs_met_more_than_80 * 0.5)

training_efficiency = avg training score / (no of trainings + 1)

experience_rating_ratio = previous_year_rating / (length_of_service + 1)

awards_per_year = awards / service length

### Encoding:

Used OneHotEncoding for nominal features, Frequency Encoding for region and StandardScaler for numerical features

## Model Fitting

**Train-Test Split:** 80:20 using train_test_split with stratification for classification

**Data Leakage Prevention:** Feature engineering and leakage-prone variables (performance_score in classification) were excluded where needed

## Models Used

Linear Regression (for performance prediction), Logistic Regression, Decision Tree Classifier, Random Forest Classifier, Support Vector Machine (SVM), Hyperparameter Tuning, Used GridSearchCV for Logistic, Decision Tree and Random Forest, Used dropout, batch norm and Adam optimizer in Neural Network tuning

## Metrics

**Regression:** R², MSE, MAE

**Classification:** Accuracy, Precision, Recall, F1-Score

Used SMOTE for class imbalance

All models evaluated using heatmap confusion matrices

## Model Evaluation Metrics

### Performance Prediction

| Model                   | R2 Score |   MSE    |   MAE    | Accuracy (%) |
|------------------------|----------|----------|----------|---------------|
| Random Forest Regressor| 0.389813 | 0.048664 | 0.190620 | 76.537546     |
| Linear Regression      | 0.373556 | 0.049961 | 0.198158 | 75.609741     |
| SVR                    | 0.373350 | 0.049977 | 0.185287 | 77.193927     |

### Performance Classification

| Model               | Accuracy | Precision | Recall | F1 Score |
|---------------------|----------|-----------|--------|----------|
| Logistic Regression | 69.43%   | 60.62%    | 42.24% | 49.79%   |
| Random Forest       | 70.44%   | 63.92%    | 40.40% | 49.51%   |
| Decision Tree       | 70.44%   | 65.24%    | 37.68% | 47.77%   |
| SVM                 | 68.77%   | 57.42%    | 50.16% | 53.54%   |

![image](https://github.com/user-attachments/assets/6e15ad2d-218e-47e0-9ec7-8c812a3c585e)

### Attrition Classification

| Model               | Accuracy | Precision | Recall | F1 Score |
|---------------------|----------|-----------|--------|----------|
| Logistic Regression | 71.99%   | 11.64%    | 82.24% | 20.39%   |
| Random Forest       | 84.30%   | 17.57%    | 70.39% | 28.12%   |
| Decision Tree       | 87.51%   | 18.34%    | 53.95% | 27.38%   |
| SVM                 | 84.59%   | 14.94%    | 53.95% | 23.40%   |

![image](https://github.com/user-attachments/assets/f28f719d-40fd-4f46-8f02-a86ee8127b96)

## Production & Use

Recommend integrating prediction API into HR dashboards

Alert HR if an employee has high attrition probability and low predicted performance

Risk: Needs retraining on recent data periodically to remain effective

## Future Scope
1. Incorporate More Diverse Data Sources
2. Real-Time or Streaming Predictions
3. Advanced Model Interpretability
4. Integration with HR Tools
5. Time Series Forecasting
6. Custom Recommendations
7. Inclusion of Ethical and Bias Audits
8. Employee Lifecycle Prediction
9. Transfer Learning Across Organizations
