# Credit Card Fraud Detection

XGBoost classification model on 284,807 real-world transactions with extreme class imbalance (0.17% fraud rate).

**Results:** F1 Score 0.86 · AUC-ROC 0.97 · Detected 82/98 fraudulent transactions

---

## The Problem

A naive model predicting "legitimate" for every transaction gets 99.83% accuracy — and catches zero fraud. Standard accuracy is useless here. The real challenge is building something that actually finds the 0.17%.

---

## Approach

- Handled class imbalance via XGBoost class weighting and SMOTE
- Evaluated 6 model configurations across Logistic Regression, Random Forest, and Gradient Boosting
- Used stratified splits to preserve class distribution across train/test sets
- Applied SHAP values for full model explainability — identifying which features drive fraud predictions

---

## Results

| Metric | Score |
|--------|-------|
| F1 Score | 0.86 |
| AUC-ROC | 0.97 |
| Fraud Detected | 82 / 98 |

---

## Stack

`Python` `XGBoost` `Scikit-learn` `imbalanced-learn` `SHAP` `Pandas` `Matplotlib`

---

## Run it

```bash
git clone https://github.com/hxxisq/credit-card-fraud-detection.git
cd credit-card-fraud-detection
pip install -r requirements.txt
jupyter notebook Fraud_Detection_Notebook.ipynb
```

> Requires Kaggle API credentials to download the dataset. See [setup guide](https://www.kaggle.com/docs/api).
