# Hire - 0- Dynamics : Employees Performance Prediction and Classification, Attribution Classification

## Dataset Overview

The dataset used for this project is an HR dataset from Kaggle containing employee-level information such as department, training scores, previous year ratings, awards won, length of service and more. It is from HR analytics systems and is structured to assist with performance and attrition forecasting.

## Key Fields:

avg_training_score, previous_year_rating, no_of_trainings, awards_won, KPIs_met_more_than_80, length_of_service, education, gender, region, age, recruitment_channel and department

### Dataset Usefulness:

Understand employee performance drivers

Predict potential attrition risks

Make proactive HR policy decisions

### Project Objectives

This project is focused on predicting:

Performance Score (regression)

Attrition Risk (classification)

Performance Classification (classification - high vs. low performer)

### Real-world Use

Optimize training programs

Identify employees at attrition risk

Improve internal talent management

### Process Overview & Experience

Explored the dataset with EDA

Cleaned and imputed missing values

Created a performance score as a derived metric

Engineered features like training efficiency and awards per year

Built multiple models iteratively

Tuned hyperparameters via GridSearchCV/RandomSearchCV

### Pivot Moments

Removed performance leakage features from classification

Switched from default models to SMOTE + pipelines due to class imbalance

## Exploratory Data Analysis (EDA)

Target Variables: 
**performance_score** (derived attribute) - Performace Prediction (regression)
**attrition** (derived attribute) -  Attrition classification (classification)
**KPIs_met_more_than_80** - Performance classification (classification)

Observations: ~7000

Features: ~13-17

### Feature Distribution

KPIs_met_more_than_80 is highly imbalanced (80% met KPIs)

awards_won is mostly 0

### Target Distribution

attrition: ~10% attrition cases

performance_classification: balanced high/low after transformation

## Correlation

Strong: previous_year_rating vs. KPIs_met_more_than_80

Low correlation among many categorical variables

## Feature Engineering

### Feature Creation:

training_efficiency = avg_training_score / (no_of_trainings + 1)

experience_rating_ratio = previous_year_rating / (length_of_service + 1)

### Encoding:

One-Hot Encoding for department, region, education

Frequency Encoding for region

Binary/label encoding avoided to maintain model interpretability

## Model Fitting

Train/Test Split: 80/20 stratified

No data leakage ensured:

performance_score not included in classification

## Models Used

Regression: Linear Regression, Random Forest Regressor, SVR

Classification: Logistic Regression, Random Forest Classifier, Decision Tree, SVM

## Hyperparameter Tuning

GridSearchCV for depth, C, estimators, kernels

RandomSearchCV for fast exploration in neural networks

## Validation / Metrics

### Metrics Used:

Regression: R², MSE, MAE

Classification: Accuracy, Precision, Recall, F1 Score, ROC-AUC Curve

### Confusion Matrix Insight

Used heatmaps for clear visual confusion matrix comparisons

Highlighted model precision-recall tradeoffs

### Model Weaknesses

SVM lower recall than Random Forest

### Prediction Samples

Real example: employee with 7 years service, low training score predicted as attrition

Synthesized: new employee with high awards + KPI met = low attrition risk

## Overfitting & Underfitting

Regularization (L2 for LR)

Pruning for DTs

## Production Considerations

Deployability: Logistic Regression / Random Forest due to speed & interpretability

Monitoring: Retrain monthly to capture organization shifts

Precautions: Model shouldn't be used in isolation without HR input

## Further Enhancements

Add more employee engagement metrics

Include time-series trends

Create explainability dashboards (e.g., SHAP values)

Integrate with HR dashboards or BI tools
