# Hire-o-Dynamics : Employee Performance & Attrition Analysis

## Abstract

This project tackles three pivotal challenges in HR analytics: forecasting employee performance, classifying performance tiers, and identifying employees at high risk of attrition. Leveraging a robust, engineered feature set derived from employee engagement and performance metrics, we apply a suite of supervised machine learning algorithms—including Logistic Regression, Random Forest, Decision Trees, and Support Vector Machines (SVM). Model performance is rigorously evaluated using accuracy, precision, recall, and F1-score, while confusion matrices offer a deeper view into the quality of predictions.

## Dataset

- **Source**: [Kaggle - Employee’s Performance for HR Analytics](https://www.kaggle.com/datasets/sanjanchaudhari/employees-performance-for-hr-analytics/data)
- **Total Records**: ~17,000 rows (employees)
- **Attributes**: Includes employee details with total 18 variables, 13 existing and 5 derived: 
   1. **Demographics**: `department`, `region`, `education`, `gender`
   2. **Performance Metrics**: `no_of_trainings`, `avg_training_score`, `previous_year_rating`, `KPIs_met_more_than_80`, `awards_won`, `length_of_service`
   3. **Derived Metrics**: `training_efficiency`, `experience_rating_ratio`, `awards_per_year`, `performance_score`, `attrition`
 
## Practical Use Case

These models support HR decisions like personalized development, promotions and retention strategies.

## Prediction Objectives

- **Performance Prediction (Regression):** Estimate a numerical score to represent overall employee performance.

- **Performance Classification:** Predict whether an employee's performance is satisfactory or not.

- **Attrition Classification:** Predict the likelihood of an employee leaving based on performance-related criteria.


## Methodology

This project is structured around three core tasks in Human Resource analytics—performance prediction, performance classification, and attrition classification. Each task follows a consistent, methodical approach combining data preprocessing pipelines, SMOTE balancing and grid search for hyperparameter tuning. Below are the detailed steps:

1. **Data preprocessing:**
    - **Missing Value Handling:**
        - Numerical columns (e.g., previous_year_rating, length_of_service) are imputed using median strategies.
        - Categorical variables with missing values (e.g., education) are filled with placeholders like "Unknown" or the most frequent category.
    - **Categorical Encoding:**
        - LabelEncoder is applied to binary categorical fields like gender.
        - OneHotEncoder is used within pipelines to handle multi-class categorical variables like department, region, and recruitment channel.
    - **Scaling:**
        - Numerical features are standardized using StandardScaler to ensure models like SVM and Logistic Regression are not biased due to scale differences.
    - **Handling Imbalanced Data:**
        - SMOTE (Synthetic Minority Over-sampling Technique) is applied to generate synthetic samples for minority classes in classification tasks (especially for 
          attrition).
          
2. **Feature Engineering:** To improve model performance and extract more meaning from raw attributes:
     - **Derived Features:**
        - `training_efficiency` = `avg_training_score` / (`no_of_trainings` + 1)
        - `experience_rating_ratio` = `previous_year_rating` / (`length_of_service` + 1)
        - `awards_per_year` = `awards_won` / (`length_of_service` + 1)
     - **Custom Target Creation for Attrition:**
        - Employees were flagged as at-risk ('attrition` = 1) if they met multiple criteria such as low KPI, low training scores, long service, low past ratings, and no 
          awards.
           
3. **Modeling Techniques:** Different models were applied based on the problem type:
     - **Performance Prediction (Regression)**
       - **Algorithms Used:**
         1. Linear Regression
         2. Random Forest Regressor
         3. Support Vector Regressor (SVR)
       - **Target:** A custom performance_score combining training score and KPI metrics.
       - **Evaluation Metrics:**
         1. R² Score
         2. Mean Squared Error (MSE)
         3. Mean Absolute Error (MAE)
         4. Custom accuracy metric: 1 - mean absolute % error
     - **Performance Classification**
       - **Goal:** Classify whether employees met more than 80% of their KPIs.
       - **Algorithms Used:**
         1. Decision Tree Classifier (tuned via GridSearchCV)
         2. Random Forest Classifier
         3. Support Vector Classifier (SVM)
       - **Evaluation Metrics:**
         1. Accuracy, Precision, Recall, F1-Score
         2. Confusion Matrix for error type analysis
     - **Attrition Classification**
       - **Goal:** Predict if an employee is at high risk of leaving.
       - **Label:** Binary (0 = low risk, 1 = high risk) based on a custom rule-based logic.
       - **Algorithms Used:** Logistic Regression (with L2 regularization)
       - **Evaluation Metrics:**
         1. Precision, Recall, F1-Score
         2. Confusion Matrix for identifying false positives/negatives
    
4. **Model Evaluation:** All models were evaluated using:
   - Train-Test Splits (typically 80:20 stratified for classification).
   - Cross-Validation using 5-fold CV for robust performance estimation
   - Hyperparameter Tuning: GridSearchCV was used across models to optimize parameters like max depth, number of estimators 
      (Random Forest), regularization strength (Logistic), and kernel type (SVM).
   



## Distributions

- **Histogram of Numerical Features**
  - Distribution plots for various numeric columns like age, avg_training_score, previous_year_rating, etc.
  ![Image](https://github.com/user-attachments/assets/a55c2304-6102-4341-87b3-45c345707b5f)

- **Pie Chart showing Department-wise Employee Distribution**
  - Visual breakdown of the percentage of employees across departments.
  ![image](https://github.com/user-attachments/assets/c7f50f4c-dcef-4740-a65c-4b269ea8dd65)

- **Histogram showing Age Distribution of Employees**
  - Histogram showing the frequency of different age groups in the organization.
  ![image](https://github.com/user-attachments/assets/fd3becec-862d-4d52-a4be-9352c2cbd181)

- **Bar Graph of Average Training Score by Department**
  - Bar plot comparing the average training score across various departments.
   ![image](https://github.com/user-attachments/assets/575fcd4e-651b-4c5a-b154-129cd4c27182)

- **Bar Chart visualizing Percentage of Employees Meeting KPIs (>80%) by Department**
  - Bar chart visualizing the proportion of high KPI achievers across departments.
  ![image](https://github.com/user-attachments/assets/84e32600-e6af-4dbf-a9b8-ac1de47c1929)

- **Pie Chart representing Education Level Distribution**
  - Visual representation of the distribution of educational qualifications among employees.
  ![image](https://github.com/user-attachments/assets/beaaf0d8-af66-4731-97c4-1eb8d1c97a6a)

 - **Correlation Heatmap of Numeric Features**
   - Heatmap showing relationships between key numerical attributes such as age, length of service, and training score.
   - Previous_year_rating and performance_score are positively correlated.
   - avg_training_score correlates with performance as expected.
   ![image](https://github.com/user-attachments/assets/7ba00ebd-7d11-45a4-a402-193520d103fb)

## Feature Importance (Random Forest)
 - KPIs_met_more_than_80, avg_training_score, previous_year_rating are top predictors

## Feature Engineering
 - attrition = (KPIs_met_more_than_80 == 0) & (avg_training_score < 60) & (length_of_service > 5) & (previous_year_rating <= 
   2) & (awards_won == 0)
 - performance_score = (avg_training_score / 100 ) + (KPIs_met_more_than_80 * 0.5)
 - training_efficiency = avg training score / (no of trainings + 1)
 - experience_rating_ratio = previous_year_rating / (length_of_service + 1)
 - awards_per_year = awards / service length

## Encoding:
 - Used OneHotEncoding for nominal features, Frequency Encoding for region and StandardScaler for numerical features

## Model Fitting
 - **Train-Test Split:** 80:20 using train_test_split with stratification for classification
 - **Data Leakage Prevention:** Feature engineering and leakage-prone variables (performance_score in classification) were excluded where needed

## Models Used
 1. Linear Regression (for performance prediction)
 2. Logistic Regression
 3. Decision Tree Classifier
 4. Random Forest Classifier
 5. Support Vector Machine (SVM)
 6. Hyperparameter Tuning
 7. Used GridSearchCV for Logistic
 8. Decision Tree and Random Forest
 9. Used dropout
 10. batch norm
 11. Adam optimizer in Neural Network tuning

## Metrics

**Regression:** R², MSE, MAE

**Classification:** Accuracy, Precision, Recall, F1-Score

Used SMOTE for class imbalance

All models evaluated using heatmap confusion matrices

## Model Evaluation Metrics

### Performance Prediction
![Image](https://github.com/user-attachments/assets/87bfd331-914c-46ae-81d4-9ef45371cc6c)
![Image](https://github.com/user-attachments/assets/f8c2c19d-a450-47a3-a19a-4b50fb54e3ec)
![Image](https://github.com/user-attachments/assets/8daa7ce0-8edf-4f19-9c60-6fc5b4d6ab06)

   - **Model Performance Comparison:**

| Model                   | R2 Score |   MSE    |   MAE    | Accuracy (%) |
|------------------------|----------|----------|----------|---------------|
| Random Forest Regressor| 0.389813 | 0.048664 | 0.190620 | 76.537546     |
| Linear Regression      | 0.373556 | 0.049961 | 0.198158 | 75.609741     |
| SVR                    | 0.373350 | 0.049977 | 0.185287 | 77.193927     |

### Performance Classification
![Image](https://github.com/user-attachments/assets/ad64df9b-cb57-41a9-a895-59d78a47dc14)
![Image](https://github.com/user-attachments/assets/309c155b-84d0-4147-b66b-33a2b244788c)
![Image](https://github.com/user-attachments/assets/b16969bc-59a3-45be-9866-d2cd3cb26f0d)
![Image](https://github.com/user-attachments/assets/8ffc382f-9f06-4ab4-9c42-ca1f25ae05a6)

   - **Model Performance Comparison:**

| Model               | Accuracy | Precision | Recall | F1 Score |
|---------------------|----------|-----------|--------|----------|
| Logistic Regression | 69.43%   | 60.62%    | 42.24% | 49.79%   |
| Random Forest       | 70.44%   | 63.92%    | 40.40% | 49.51%   |
| Decision Tree       | 70.44%   | 65.24%    | 37.68% | 47.77%   |
| SVM                 | 68.77%   | 57.42%    | 50.16% | 53.54%   |

![image](https://github.com/user-attachments/assets/6e15ad2d-218e-47e0-9ec7-8c812a3c585e)

### Attrition Classification
![Image](https://github.com/user-attachments/assets/c50c8898-980b-4f6c-9967-b304018aa1fc)
![Image](https://github.com/user-attachments/assets/1dc32151-caf9-4aa2-aac7-25ca42a6740b)
![Image](https://github.com/user-attachments/assets/a0556e43-3881-4fbf-aefc-85e642aa816e)
![Image](https://github.com/user-attachments/assets/3f809bc8-e396-4242-91fe-db5a673fc1c9)

   - **Model Performance Comparison:**

| Model               | Accuracy | Precision | Recall | F1 Score |
|---------------------|----------|-----------|--------|----------|
| Logistic Regression | 71.99%   | 11.64%    | 82.24% | 20.39%   |
| Random Forest       | 84.30%   | 17.57%    | 70.39% | 28.12%   |
| Decision Tree       | 87.51%   | 18.34%    | 53.95% | 27.38%   |
| SVM                 | 84.59%   | 14.94%    | 53.95% | 23.40%   |

![image](https://github.com/user-attachments/assets/f28f719d-40fd-4f46-8f02-a86ee8127b96)

## Production & Use

1. Recommend integrating prediction API into HR dashboards
2. Alert HR if an employee has high attrition probability and low predicted performance
3. Risk: Needs retraining on recent data periodically to remain effective

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

## Conclusion

Across the three modeling objectives—performance classification, attrition classification, and performance prediction—different models proved optimal depending on the evaluation criteria. For performance classification, Random Forest and Decision Tree delivered the highest accuracy (70.44%), while Decision Tree was best for precision-driven tasks, and SVM excelled in recall and F1-score, making it suitable for catching true underperformers. In the attrition classification task, models like Logistic Regression and Random Forest often show strong interpretability and generalization, with Random Forest typically offering a balance between accuracy and recall for identifying potential employee exits. For the performance prediction (regression) task, Support Vector Regression (SVR) was used, and performance was best interpreted using metrics like RMSE, MAE, and R², where lower RMSE and higher R² values indicate better predictions. Ultimately, Random Forest consistently shows reliable, balanced performance across tasks, while SVM and Decision Tree offer specialized strengths in recall and precision, respectively.
