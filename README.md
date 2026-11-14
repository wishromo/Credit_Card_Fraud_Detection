# Credit Card Fraud Detection Using Machine Learning

## Project Overview

This project develops and evaluates machine learning models to identify potentially fraudulent credit card transactions using an anonymized transaction dataset containing 284,807 records.

The objective is to detect fraudulent activity while reducing false positives, using a structured machine learning workflow from exploratory data analysis to final model evaluation.

## Dataset

- Total transactions: 284,807
- Fraudulent transactions: 492
- Normal transactions: 284,315
- Features: Time, Amount, V1–V28
- Target: Class (0 = Normal, 1 = Fraud)
- Fraud rate: Approximately 0.173%

The dataset is highly imbalanced, making precision, recall, F1-score, and Average Precision especially important.

## Methodology

### 1. Exploratory Data Analysis (EDA)

- Examined data structure, missing values, and duplicate records.
- Analysed class imbalance.
- Investigated transaction amount distributions and outliers.
- Explored transaction timing patterns.
- Calculated feature correlations and visualised distributions.

### 2. Feature Engineering and Preprocessing

- Applied logarithmic transformation to transaction amounts.
- Investigated time-based transformations.
- Used StandardScaler and ColumnTransformer.
- Implemented reusable scikit-learn pipelines.
- Separated training, validation, and testing data to reduce data leakage.

### 3. Model Development

Implemented and evaluated:

- Logistic Regression with class weighting
- Random Forest Classifier
- Histogram-based Gradient Boosting Classifier

### 4. Model Optimization

- Stratified cross-validation
- RandomizedSearchCV hyperparameter tuning
- Decision threshold tuning
- Precision-recall analysis
- Feature importance and permutation importance

## Final Model

**Selected Model:** HistGradientBoostingClassifier

**Decision Threshold:** 0.40

The final model was fitted on the combined training and validation data and evaluated using a held-out test dataset.

## Final Test Results

| Metric | Result |
|---|---:|
| Precision | 87.21% |
| Recall | 76.53% |
| F1-score | 81.52% |
| Average Precision | 80.76% |
| True Positives | 75 |
| True Negatives | 56,853 |
| False Positives | 11 |
| False Negatives | 23 |

The model correctly identified 75 of 98 fraudulent transactions in the held-out test set, with 11 false positive alerts.

## Key Findings

- Fraud detection requires metrics beyond accuracy because of severe class imbalance.
- Decision threshold tuning substantially affected precision and recall.
- Gradient Boosting achieved the highest validation F1-score among the tested model and threshold combinations.
- Different feature importance methods produced different rankings.
- The final model achieved promising held-out test results.

## Tools and Technologies

Python, pandas, NumPy, Matplotlib, Seaborn, scikit-learn, Google Colab, and joblib.

## Limitations

- The dataset contains anonymized features whose original business meanings are unavailable.
- Random train/test splitting does not fully simulate future transaction detection.
- The model was evaluated on a historical dataset and has not been deployed or monitored in production.
- Decision thresholds would require business-cost analysis and further validation before real-world use.

## Project Deliverables

- Jupyter notebook containing the ML workflow
- Final confusion matrix visualization
- Saved trained model
- Project documentation

## Conclusion

This project demonstrates an end-to-end machine learning workflow for an imbalanced fraud detection problem, including EDA, preprocessing, model development, optimization, explainability, and held-out evaluation.

The final Gradient Boosting model achieved an F1-score of **81.52%** and Average Precision of **80.76%** on the held-out test dataset.
