# Credit Card Fraud Detection - Machine Learning Classification

A comprehensive machine learning solution for detecting fraudulent credit card transactions using advanced classification techniques on highly imbalanced data. This project demonstrates practical approaches to handling extreme class imbalance and achieving high-performance fraud detection.

---

## Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Key Challenge: Class Imbalance](#key-challenge-class-imbalance)
- [Solution Approach](#solution-approach)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Model Performance](#model-performance)
- [Technologies Used](#technologies-used)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Results & Insights](#results--insights)
- [Explainability](#explainability)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

Fraudulent credit card transactions pose significant financial risks to institutions and consumers. This project builds a robust machine learning classification system capable of identifying fraudulent transactions with high precision and recall, even in the presence of extreme class imbalance.

**Dataset:** 284,807 real-world credit card transactions from Kaggle  
**Fraudulent Cases:** Only 492 (0.17% of total)  
**Goal:** Maximize fraud detection while minimizing false positives

---

## Problem Statement

### The Challenge
Credit card companies receive millions of transactions daily, but fraudulent transactions represent a tiny fraction of all activity. This creates a significant **class imbalance problem**:

- **Legitimate transactions:** 284,315 (99.83%)
- **Fraudulent transactions:** 492 (0.17%)

### The Naive Baseline Problem
A naive model that predicts "legitimate" for every transaction achieves:
- **Accuracy:** 99.83%
- **Fraud Detection Rate:** 0% (catches zero fraud)

This renders traditional accuracy metrics **useless** for this problem.

### The Goal
Build a model that:
1. **Identifies fraud accurately** (high recall for fraud class)
2. **Minimizes false alarms** (high precision)
3. **Balances both metrics** using F1 Score and AUC-ROC
4. **Explains predictions** for business stakeholders

---

## Key Challenge: Class Imbalance

This project directly addresses the extreme class imbalance problem through multiple strategies:

### 1. **Alternative Evaluation Metrics**
Instead of accuracy, we use:
- **F1 Score:** Harmonic mean of precision and recall (0-1 scale)
- **AUC-ROC:** Area under the Receiver Operating Characteristic curve
- **Confusion Matrix:** Detailed breakdown of TP, TN, FP, FN
- **Precision & Recall:** Balanced view of model performance

### 2. **Imbalance Handling Techniques**
- **SMOTE (Synthetic Minority Over-sampling):** Generate synthetic fraudulent samples
- **Class Weights:** Penalize misclassification of minority class
- **Stratified Splitting:** Maintain class distribution in train/test sets

### 3. **Model Selection**
Compare multiple algorithms to find best approach for imbalanced data

---

## Solution Approach

### **Phase 1: Exploratory Data Analysis (EDA)**
- Load and inspect dataset structure
- Analyze feature distributions
- Identify high-signal features through correlation analysis
- Compare fraud vs. legitimate transaction patterns
- Detect missing values and data quality issues

### **Phase 2: Data Preprocessing**
- **Feature Scaling:** StandardScaler for `Amount` and `Time` features
- **Train/Test Split:** Stratified split to maintain class distribution (80/20)
- **Feature Selection:** Retain original features (already PCA-transformed by Kaggle)
- **Handling Imbalance:** Prepare data for SMOTE and class-weight strategies

### **Phase 3: Model Development**
Build and compare 6 model configurations:

| Model | Imbalance Strategy | Algorithm |
|-------|-------------------|-----------|
| M1 | SMOTE | Logistic Regression |
| M2 | Class Weights | Logistic Regression |
| M3 | SMOTE | Random Forest |
| M4 | Class Weights | Random Forest |
| M5 | SMOTE | Gradient Boosting |
| M6 | Class Weights | Gradient Boosting |

### **Phase 4: Evaluation & Selection**
- Compare models using F1 Score and AUC-ROC
- Analyze confusion matrices
- Generate ROC curves
- Select best-performing model

### **Phase 5: Explainability**
- Use SHAP values for feature importance
- Explain global feature contributions
- Analyze individual prediction explanations
- Provide business insights

---

## Dataset

### Source
Kaggle - Credit Card Fraud Detection Dataset (MLG-ULB)

### Dataset Characteristics

| Property | Value |
|----------|-------|
| **Total Transactions** | 284,807 |
| **Features** | 31 (+ 1 target) |
| **Fraudulent Cases** | 492 (0.17%) |
| **Legitimate Cases** | 284,315 (99.83%) |
| **Missing Values** | None |
| **Data Types** | Float64 & Int64 |

### Features Description

| Feature Category | Details |
|-----------------|---------|
| **Time** | Seconds elapsed since first transaction (not normalized) |
| **Amount** | Transaction amount in currency units (not normalized) |
| **V1-V28** | PCA-transformed features (original features not disclosed for privacy) |
| **Class** | Binary target: 0 = Legitimate, 1 = Fraudulent |

### Data Quality Notes
- All features are numeric (no categorical variables)
- No missing values requiring imputation
- Features V1-V28 are already PCA-transformed
- Time and Amount require explicit scaling during preprocessing

---

## Project Structure

```
.
├── README.md                              # This file
├── Fraud_Detection_Notebook.ipynb         # Main analysis & modeling notebook
└── requirements.txt                       # Python dependencies
```

---

## Model Performance

### Expected Results (Typical Ranges)

| Metric | Logistic Regression | Random Forest | Gradient Boosting |
|--------|-------------------|----------------|------------------|
| **F1 Score** | 0.70 - 0.78 | 0.75 - 0.85 | 0.78 - 0.88 |
| **AUC-ROC** | 0.85 - 0.92 | 0.92 - 0.97 | 0.93 - 0.98 |
| **Precision** | 0.80 - 0.90 | 0.85 - 0.95 | 0.85 - 0.95 |
| **Recall** | 0.65 - 0.75 | 0.75 - 0.80 | 0.78 - 0.85 |

**Note:** Exact results depend on random seed, train/test split, and hyperparameter tuning.

### Key Observations
- SMOTE and class weights both effectively address imbalance
- Tree-based models (RF, GB) typically outperform logistic regression
- Ensemble methods (Gradient Boosting) show best AUC-ROC scores
- Trade-off exists between precision and recall — model selection depends on business needs

---

## Technologies Used

**Core Libraries:**
- **Python 3.x** - Programming language
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **Scikit-learn** - Machine learning algorithms and evaluation
- **Imbalanced-learn (imblearn)** - SMOTE and imbalance handling

**Visualization & Explainability:**
- **Matplotlib** - Data visualization
- **Seaborn** - Statistical visualizations
- **SHAP** - Model explainability and feature importance

**Development Environment:**
- **Jupyter Notebook** - Interactive analysis and modeling
- **Kagglehub** - Dataset download from Kaggle

---

## Getting Started

### Prerequisites

Ensure you have the following installed:
- Python 3.8 or higher
- Jupyter Notebook or JupyterLab
- pip (Python package manager)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/credit-card-fraud-detection.git
   cd credit-card-fraud-detection
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate    # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Kaggle Dataset Setup

1. **Create a Kaggle account** (if you don't have one)
   - Visit [kaggle.com](https://www.kaggle.com)
   - Sign up or log in

2. **Set up Kaggle API credentials:**
   ```bash
   # Create .kaggle directory
   mkdir -p ~/.kaggle
   
   # Download API token from Kaggle account settings
   # Place it in ~/.kaggle/kaggle.json
   chmod 600 ~/.kaggle/kaggle.json
   ```

3. **Download the dataset:**
   The notebook automatically downloads the dataset using kagglehub. If using this locally:
   ```python
   import kagglehub
   path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
   ```

### Dependencies (requirements.txt)

```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
imbalanced-learn>=0.9.0
matplotlib>=3.4.0
seaborn>=0.11.0
shap>=0.41.0
kagglehub>=0.1.0
jupyter>=1.0.0
```

---

## 📖 Usage

1. **Launch Jupyter Notebook:**
   ```bash
   jupyter notebook
   ```

2. **Open the analysis notebook:**
   - Navigate to `Fraud_Detection_Notebook.ipynb`
   - Run cells sequentially to reproduce the full analysis

3. **Key sections to explore:**
   - **EDA:** Understand data characteristics and fraud patterns
   - **Preprocessing:** See how imbalance is handled
   - **Modeling:** Compare different algorithms and strategies
   - **Evaluation:** Review model performance metrics
   - **Explainability:** Understand why models make specific predictions

### Notebook Structure

```
1. Environment Setup & Imports
2. Dataset Download & Loading
3. Exploratory Data Analysis (EDA)
   - Dataset overview
   - Missing values & data types
   - Class distribution analysis
   - Feature distributions
4. Data Preprocessing
   - Train/test split
   - Feature scaling
   - SMOTE application
5. Model Development
   - Train 6 models
   - Cross-validation
   - Hyperparameter tuning
6. Model Evaluation
   - Confusion matrices
   - ROC curves
   - F1 & AUC-ROC comparison
7. Feature Explainability
   - SHAP values
   - Feature importance rankings
8. Business Recommendations
```

---

## Results & Insights

### Key Findings

1. **Class Imbalance Impact:**
   - Naive baseline accuracy (99.83%) is misleading
   - F1 Score and AUC-ROC are essential metrics

2. **Best Performing Model:**
   - Gradient Boosting typically achieves best results
   - Combination of SMOTE + Gradient Boosting optimal

3. **Feature Importance:**
   - Most important features identified via SHAP
   - Interpretable ranking of fraud indicators

4. **Trade-offs:**
   - Higher recall = catch more fraud (but more false positives)
   - Higher precision = fewer false alarms (but miss some fraud)
   - Business needs determine optimal threshold

### Actionable Insights

- **Real-time Monitoring:** Deploy model to flag high-risk transactions
- **Customer Contact:** Proactively reach out for verification
- **Pattern Detection:** Focus investigations on top fraud indicators
- **Threshold Tuning:** Adjust decision boundary based on cost of false positives vs. false negatives

---

## Explainability with SHAP

This project uses SHAP (SHapley Additive exPlanations) values to explain model predictions:

### Global Feature Importance
- Identify which features contribute most to fraud detection
- Understand model behavior across all predictions

### Individual Predictions
- Explain why a specific transaction was flagged as fraud
- Show contributing features for each decision
- Build customer trust through transparency

### Business Value
- Compliance requirements for fraud detection systems
- Customer service explanations for declined transactions
- Model debugging and improvement insights

---

## Machine Learning Concepts Demonstrated

This project showcases:
- Handling extreme class imbalance
- SMOTE for synthetic oversampling
- Class weights in model training
- Appropriate metrics for imbalanced data
- Ensemble methods (Random Forest, Gradient Boosting)
- Hyperparameter tuning
- Cross-validation on imbalanced data
- Model explainability with SHAP
- ROC curves and threshold optimization

---

## Next Steps

- **Hyperparameter Optimization:** Use GridSearchCV or Bayesian optimization
- **Feature Engineering:** Create new features from existing ones
- **Ensemble Methods:** Combine multiple models
- **Threshold Tuning:** Optimize decision boundary based on business metrics
- **Production Deployment:** Create API and monitoring system
- **Continuous Learning:** Implement feedback loops for model updates

---

## Contact & Support

For questions, suggestions, or improvements:
- Open an [issue](https://github.com/yourusername/credit-card-fraud-detection/issues)
- Submit a [pull request](https://github.com/yourusername/credit-card-fraud-detection/pulls)
- Contact: [your-email@example.com]

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- **Dataset:** Kaggle Credit Card Fraud Detection Dataset (MLG-ULB/Andrea Dal Pozzolo)
- **Inspiration:** Industry best practices in fraud detection systems
- **Libraries:** Built with scikit-learn, SHAP, imbalanced-learn communities
- **References:** Research papers on handling imbalanced data in classification

---

## References & Resources

### Class Imbalance Handling
- [Imbalanced-learn Documentation](https://imbalanced-learn.org/)
- [SMOTE: Synthetic Minority Over-sampling Technique](https://jmlr.cstp.org/papers/v16/chawla05a.html)
- [Handling Imbalanced Data - Scikit-learn](https://scikit-learn.org/stable/modules/ensemble.html#class-weight)

### Model Explainability
- [SHAP: A Unified Approach to Interpreting Model Predictions](https://arxiv.org/abs/1705.07874)
- [SHAP Documentation](https://shap.readthedocs.io/)

### Evaluation Metrics
- [F1 Score & Classification Metrics](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html)
- [ROC Curves & AUC](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html)

### Fraud Detection
- [Credit Card Fraud Detection - Kaggle Notebook](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

---

**Last Updated:** 2026 | **Python Version:** 3.8+
