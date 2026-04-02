# Credit Card Fraud Detection

Binary classification model that detects fraudulent credit card transactions 
on a dataset with extreme class imbalance (0.17% fraud).

## Results
| Model | F1 Score | AUC-ROC |
|---|---|---|
| XGBoost (Class Weights) | 0.8586 | 0.9682 |
| Random Forest (SMOTE) | 0.8410 | 0.9731 |
| Random Forest (Class Weights) | 0.8391 | 0.9529 |
| XGBoost (SMOTE) | 0.8018 | 0.9792 |
| Logistic Regression (CW) | 0.1144 | 0.9722 |
| Logistic Regression (SMOTE) | 0.1092 | 0.9699 |

## Key Findings
- Transaction amount has virtually no correlation with fraud (0.006) — 
  fraudsters deliberately blend in with normal spending patterns
- Fraud transactions are more active during nighttime hours when legitimate 
  spending drops — suggesting fraud ignores cardholder timezone
- V14 is the most influential feature per SHAP analysis, despite V17 having 
  the strongest linear correlation — XGBoost captured non-linear patterns 
  that correlation missed
- Class weights outperformed SMOTE for tree-based models — XGBoost's 
  boosting mechanism already handles rare examples effectively

## Dataset
[Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) 
— Kaggle (ULB Machine Learning Group)

## Tech Stack
Python, pandas, NumPy, scikit-learn, XGBoost, imbalanced-learn, 
matplotlib, seaborn, SHAP

## Project Structure
- `Fraud_Detection.ipynb` — Full notebook with EDA, preprocessing, 
   modelling, evaluation, and SHAP explainability
