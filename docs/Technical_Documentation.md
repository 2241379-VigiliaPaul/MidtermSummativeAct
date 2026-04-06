# Customer Churn Prediction Using Machine Learning Classification Algorithms

## A Comparative Analysis of Naive Bayes, Support Vector Machine, and Random Forest with Decision Tree Rule Extraction

---

**Technical Documentation**  
**Midterm Summative Assessment - Machine Learning**  
**April 2026**

---

## Abstract

This paper presents a comprehensive machine learning approach to predict customer churn in a telecommunications dataset using four classification algorithms: Decision Tree (rule-based analysis), Naive Bayes, Support Vector Machine (SVM), and Random Forest (chosen algorithm). The study implements a rigorous data splitting protocol with 10-fold cross-validation and evaluates models using seven standardized performance metrics: Confusion Matrix, Accuracy, Precision, Recall, F1-score, Specificity, and ROC-AUC. The dataset comprises 7,043 customer records with 19 feature variables after preprocessing. Results indicate that Random Forest achieves the highest overall performance with balanced accuracy and generalization capability, while SVM demonstrates strong recall performance and Naive Bayes offers computational efficiency. The Decision Tree provides interpretable business rules for stakeholder communication. All models were validated on a held-out 10% unseen dataset to assess generalization performance.

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Methodology](#2-methodology)
3. [Experimental Setup](#3-experimental-setup)
4. [Results and Analysis](#4-results-and-analysis)
5. [Discussion](#5-discussion)
6. [Conclusion and Recommendations](#6-conclusion-and-recommendations)
7. [Technical Implementation](#7-technical-implementation)
8. [References](#8-references)

---

## 1. Introduction

### 1.1 Problem Statement

Customer churn prediction is a critical business intelligence application in the telecommunications industry. The ability to identify customers at risk of leaving enables proactive retention strategies, reducing customer acquisition costs and maintaining revenue stability. This study addresses the binary classification problem of predicting whether a customer will churn (leave the service) based on demographic, account, and service usage features.

### 1.2 Research Objectives

1. Implement four classification algorithms on a real-world telecommunications dataset
2. Apply standardized evaluation metrics across all models
3. Compare algorithm performance using 10-fold cross-validation
4. Validate models on unseen data to assess generalization
5. Extract interpretable business rules from the Decision Tree model
6. Recommend the most appropriate algorithm for production deployment

### 1.3 Dataset Overview

**Source:** Telco Customer Churn Dataset  
**Size:** 7,043 records × 20 columns (after preprocessing)  
**Target Variable:** Churn (Binary: 0 = No Churn, 1 = Churn)  
**Class Distribution:** 73.46% No Churn (5,174), 26.54% Churn (1,869)

**Feature Categories:**
- **Demographics:** Gender, Senior Citizen, Partner, Dependents
- **Account Information:** Tenure, Contract Type, Paperless Billing, Payment Method
- **Services:** Phone Service, Multiple Lines, Internet Service, Online Security, Online Backup, Device Protection, Tech Support, Streaming TV, Streaming Movies
- **Charges:** Monthly Charges, Total Charges

---

## 2. Methodology

### 2.1 Data Preprocessing Pipeline

The data preprocessing workflow was implemented to ensure data quality and consistency:

```
Raw Data
    ↓
1. Handle Missing Values
   - TotalCharges: Convert to numeric, fill NaN with 0
    ↓
2. Feature Encoding
   - Binary categorical: Label encoding (0/1)
   - Multi-class categorical: Integer encoding
    ↓
3. Data Validation
   - Remove duplicate records (22 removed)
   - Verify data types
    ↓
Cleaned Dataset (7,043 × 20)
```

### 2.2 Algorithm Selection

| Algorithm | Type | Key Characteristics |
|-----------|------|---------------------|
| **Decision Tree** | Rule-based | Interpretable rules, max_depth=5, standalone analysis |
| **Naive Bayes** | Probabilistic | Fast training, assumes feature independence |
| **Support Vector Machine** | Kernel-based | Handles non-linear boundaries, requires feature scaling |
| **Random Forest** | Ensemble | Chosen algorithm, robust to overfitting, n_estimators=100 |

### 2.3 Feature Engineering

**Scaling for SVM:**
StandardScaler applied to normalize feature magnitudes:
```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

**Class Imbalance Handling:**
- SVM: `class_weight='balanced'` parameter
- All other models: Natural class distribution

---

## 3. Experimental Setup

### 3.1 Data Splitting Protocol

A strict hierarchical splitting protocol was implemented to ensure reproducible evaluation:

```
Original Dataset (100% = 7,043 samples)
    ├── Main Data (90% = 6,338 samples)
    │   ├── Training Set (80% of 90% = 72% total) → 5,070 samples
    │   └── Test Set (20% of 90% = 18% total) → 1,268 samples
    └── Unseen Validation Set (10% = 705 samples)
```

**Splitting Strategy:**
- Stratified sampling to maintain class distribution
- Random state = 42 for reproducibility
- Unseen set held out until final validation

### 3.2 Cross-Validation Configuration

**Method:** 10-Fold Stratified Cross-Validation  
**Implementation:** `StratifiedKFold(n_splits=10, shuffle=True, random_state=42)`  
**Prediction Method:** `cross_val_predict` for consistent fold alignment

### 3.3 Evaluation Metrics

Seven standardized metrics computed for each algorithm:

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **Confusion Matrix** | [[TN, FP], [FN, TP]] | Detailed prediction breakdown |
| **Accuracy** | (TP + TN) / Total | Overall correctness |
| **Precision** | TP / (TP + FP) | Reliability of positive predictions |
| **Recall** | TP / (TP + FN) | Coverage of actual positives |
| **F1-Score** | 2 × (Precision × Recall) / (Precision + Recall) | Harmonic mean of precision and recall |
| **Specificity** | TN / (TN + FP) | Coverage of actual negatives |
| **ROC-AUC** | Area under ROC curve | Discrimination ability across thresholds |

### 3.4 Model Parameters

**Decision Tree:**
```python
DecisionTreeClassifier(random_state=42, max_depth=5)
```

**Naive Bayes:**
```python
GaussianNB()  # Default parameters
```

**Support Vector Machine:**
```python
SVC(random_state=42, probability=True, class_weight='balanced')
```

**Random Forest:**
```python
RandomForestClassifier(n_estimators=100, random_state=42)
```

---

## 4. Results and Analysis

### 4.1 10-Fold Cross-Validation Results

#### Table 1: Performance Metrics Comparison (3 Classification Algorithms)

| Model | Accuracy | Precision | Recall | F1-Score | Specificity | ROC-AUC |
|-------|----------|-----------|--------|----------|-------------|---------|
| Naive Bayes | 0.7231 | 0.4867 | 0.8045 | 0.6065 | 0.6937 | 0.8088 |
| Random Forest | 0.7868 | 0.6280 | 0.4828 | 0.5459 | 0.8967 | 0.8233 |
| SVM | 0.7450 | 0.5129 | 0.7703 | 0.6158 | 0.7358 | 0.8236 |

*Note: Decision Tree analyzed separately as rule-based classifier (Section 4.4)*

#### Table 2: Confusion Matrices (Cross-Validation)

| Model | True Negative | False Positive | False Negative | True Positive |
|-------|---------------|----------------|----------------|---------------|
| Naive Bayes | 2,584 | 1,141 | 263 | 1,082 |
| Random Forest | 4,175 | 481 | 870 | 812 |
| SVM | 2,741 | 984 | 309 | 1,036 |

### 4.2 Test Set vs Unseen Validation Results

#### Table 3: Generalization Performance

| Model | Test Accuracy | Unseen Accuracy | Generalization Gap |
|-------|---------------|-----------------|------------------|
| Naive Bayes | 0.7185 | 0.7135 | +0.0050 |
| Random Forest | 0.7879 | 0.7957 | -0.0079 |
| SVM | 0.7342 | 0.7348 | -0.0005 |

*Note: Decision Tree validated separately - Test: 0.7863, Unseen: 0.7929, Gap: -0.0066*

### 4.3 ROC Curve Analysis

The ROC curves demonstrate the trade-off between True Positive Rate (Sensitivity) and False Positive Rate (1-Specificity) across classification thresholds:

- **Random Forest:** AUC = 0.8233 - Strong discrimination with conservative predictions
- **SVM:** AUC = 0.8236 - Comparable discrimination with balanced threshold
- **Naive Bayes:** AUC = 0.8088 - Slightly lower discrimination but high recall

### 4.4 Decision Tree Rule Extraction

**Stand-alone Analysis:** Decision Tree provides interpretable business rules for stakeholder communication.

**Key Decision Rules Extracted:**

```
1. Contract Type Analysis:
   - Customers with 1-2 year contracts (≤2.50) → Lower churn probability
   - Month-to-month contracts (>2.50) → Higher churn probability

2. Security Services Impact:
   - No Online Security (≤0.50) + Short tenure → Churn risk
   - Online Security enabled (>0.50) → Retention factor

3. Financial Indicators:
   - Monthly Charges > $103.47 + Low Total Charges → Churn indicator
   - High Total Charges (> $7304.98) → Loyalty signal

4. Service Usage Patterns:
   - DSL Internet + Short tenure + No Tech Support → Churn risk
   - Fiber Optic + Electronic Check payment → High churn probability
```

---

## 5. Discussion

### 5.1 Algorithm Performance Analysis

#### 5.1.1 Random Forest (Chosen Algorithm)

**Strengths:**
- Highest accuracy (78.68%) among all models
- Best specificity (89.67%) - excellent at identifying non-churners
- Robust generalization with minimal gap (-0.79%)
- Ensemble approach reduces overfitting

**Considerations:**
- Lower recall (48.28%) - misses more actual churners
- Moderate precision (62.80%) - some false positives
- Computationally intensive compared to Naive Bayes

**Recommendation:** Best for production deployment where balanced performance is critical.

#### 5.1.2 Support Vector Machine

**Strengths:**
- Strong recall (77.03%) - captures most churners
- Good F1-score (61.58%) - balanced precision-recall
- Excellent ROC-AUC (0.8236) - strong discrimination
- Near-zero generalization gap (-0.05%) - highly stable

**Considerations:**
- Requires feature scaling (StandardScaler)
- Moderate specificity (73.58%) - more false alarms
- Longer training time with probability estimation

**Recommendation:** Excellent for churn detection where capturing churners is prioritized over precision.

#### 5.1.3 Naive Bayes

**Strengths:**
- Highest recall (80.45%) - captures most at-risk customers
- Fastest training and inference
- Simple implementation with no hyperparameter tuning
- Good interpretability through probability outputs

**Considerations:**
- Lowest accuracy (72.31%) - more overall errors
- Lowest precision (48.67%) - many false alarms
- Assumes feature independence (may not hold)

**Recommendation:** Ideal for initial screening and high-volume applications where speed is critical.

#### 5.1.4 Decision Tree

**Strengths:**
- Highly interpretable decision rules for business stakeholders
- Automatic feature selection through splits
- No preprocessing required (handles mixed data types)
- Fast prediction once trained

**Considerations:**
- Prone to overfitting (mitigated by max_depth=5)
- Unstable with small data changes
- Lower performance compared to ensemble methods

**Recommendation:** Best for generating business rules and stakeholder presentations.

### 5.2 Comparative Insights

| Criterion | Best Performer | Rationale |
|-----------|----------------|-----------|
| **Overall Accuracy** | Random Forest | Balanced predictions with ensemble robustness |
| **Churn Detection (Recall)** | Naive Bayes | Highest sensitivity to churn signals |
| **Prediction Reliability (Precision)** | Random Forest | Fewer false alarms |
| **Discrimination (ROC-AUC)** | SVM | Superior threshold-independent performance |
| **Generalization Stability** | SVM | Near-zero gap between test and unseen |
| **Interpretability** | Decision Tree | Clear business rules |
| **Computational Efficiency** | Naive Bayes | Fastest training and inference |

### 5.3 Business Implications

**Churn Prevention Strategy Recommendations:**

1. **High-Value Customer Protection:** Deploy Random Forest for general population to minimize false alarms
2. **At-Risk Customer Screening:** Use Naive Bayes for initial broad screening to catch potential churners
3. **Precision Targeting:** Apply SVM for focused campaigns where precision matters
4. **Business Communication:** Present Decision Tree rules to executives for strategic understanding

---

## 6. Conclusion and Recommendations

### 6.1 Summary of Findings

This study successfully implemented and evaluated four classification algorithms for customer churn prediction. The experimental design followed best practices with stratified sampling, 10-fold cross-validation, and held-out validation on unseen data.

**Key Findings:**

1. **Random Forest** is the chosen algorithm for production deployment, offering the best balance of accuracy (78.68%), specificity (89.67%), and generalization stability.

2. **SVM** demonstrates exceptional recall performance (77.03%) with near-perfect generalization, making it suitable for churn detection scenarios.

3. **Naive Bayes** provides the highest recall (80.45%) and computational efficiency, ideal for rapid screening applications.

4. **Decision Tree** delivers valuable business rules for interpretability, though not competitive in performance metrics.

### 6.2 Technical Contributions

- Standardized code architecture across all models with consistent `compute_metrics()` functions
- Rigorous data splitting protocol (90-10, then 80-20) ensuring reproducible evaluation
- Comprehensive metric suite (7 metrics) enabling thorough algorithm comparison
- Visualization suite (confusion matrix heatmaps, ROC curves, comparison charts) for result communication

### 6.3 Recommendations for Future Work

1. **Feature Engineering:** Explore interaction terms (Contract × Tenure, Service bundles)
2. **Hyperparameter Tuning:** Implement grid search for SVM (C, gamma) and Random Forest (max_depth, min_samples_split)
3. **Ensemble Stacking:** Combine Naive Bayes (recall) + Random Forest (precision) for hybrid approach
4. **Time-Series Analysis:** Incorporate temporal features for dynamic churn prediction
5. **Cost-Sensitive Learning:** Weight misclassification costs based on customer lifetime value

---

## 7. Technical Implementation

### 7.1 Project Structure

```
MidtermSummativeAct/
├── data/processed/
│   └── Cleaned_Telco_Customer.csv    # Preprocessed dataset (7,043 × 20)
├── docs/
│   └── Technical_Documentation.md    # This document
├── models/                           # Serialized trained models
│   ├── decision_tree.pkl
│   ├── naive_bayes.pkl
│   ├── random_forest_model.pkl
│   └── svm.pkl
├── notebooks/
│   └── complete_analysis.ipynb       # Main analysis notebook
├── results/                          # Generated visualizations
│   ├── confusion_matrix_heatmaps.png
│   ├── roc_curves_per_model.png
│   ├── all_models_comparison.png
│   ├── eda_overview.png
│   ├── correlation_matrix.png
│   └── decision_tree_rules.txt
├── src/
│   ├── data_preprocessing.py         # Data cleaning and encoding
│   ├── evaluation.py                 # Metric computation utilities
│   └── models/
│       ├── decision_tree.py          # DT implementation
│       ├── naive_bayes.py            # NB implementation
│       ├── random_forest.py          # RF implementation
│       └── svm.py                    # SVM implementation
├── main.py                           # Standalone execution script
└── requirements.txt                  # Python dependencies
```

### 7.2 Dependencies

```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
joblib>=1.0.0
```

### 7.3 Execution Instructions

**Jupyter Notebook (Recommended):**
```bash
jupyter notebook notebooks/complete_analysis.ipynb
```
Run all cells sequentially for complete analysis pipeline.

**Command Line:**
```bash
python main.py
```
Executes full pipeline and generates all outputs.

### 7.4 Reproducibility Notes

- **Random Seed:** 42 (used consistently across all stochastic operations)
- **Cross-Validation:** StratifiedKFold with shuffle=True
- **Scikit-learn Version:** 1.0+ (ensure version compatibility)

---

## 8. References

1. Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5-32.
2. Cortes, C., & Vapnik, V. (1995). Support-vector networks. *Machine Learning*, 20(3), 273-297.
3. Quinlan, J. R. (1986). Induction of decision trees. *Machine Learning*, 1(1), 81-106.
4. Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning* (2nd ed.). Springer.
5. IBM Telco Customer Churn Dataset. Kaggle Repository.

---

## Appendix A: Detailed Metric Computations

### A.1 Confusion Matrix Components

For binary classification with classes 0 (No Churn) and 1 (Churn):

- **True Negatives (TN):** Correctly predicted non-churners
- **False Positives (FP):** Non-churners incorrectly predicted as churners
- **False Negatives (FN):** Churners incorrectly predicted as non-churners
- **True Positives (TP):** Correctly predicted churners

### A.2 Metric Formulas

```python
Accuracy = (TP + TN) / (TP + TN + FP + FN)

Precision = TP / (TP + FP)

Recall (Sensitivity) = TP / (TP + FN)

Specificity = TN / (TN + FP)

F1-Score = 2 × (Precision × Recall) / (Precision + Recall)

ROC-AUC = Area under the Receiver Operating Characteristic curve
```

### A.3 Cross-Validation Implementation

```python
from sklearn.model_selection import cross_val_predict, StratifiedKFold

splitter = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
y_pred_cv = cross_val_predict(model, X, y, cv=splitter)
y_prob_cv = cross_val_predict(model, X, y, cv=splitter, method='predict_proba')[:, 1]
metrics = compute_metrics(y, y_pred_cv, y_prob_cv)
```

---

## Appendix B: Decision Tree Rules Export

Complete decision rules exported to: `results/decision_tree_rules.txt`

Visualization exported to: `results/decision_tree_visualization.png`

---

*Document Version: 1.0*  
*Last Updated: April 2026*  
*Author: Machine Learning Course - Midterm Assessment*
